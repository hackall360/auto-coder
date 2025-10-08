"""Database migration orchestration utilities and agent implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .repo_context import RepoContextAgent
from .runner import RunReport, RunnerAgent

__all__ = [
    "MigrationRecord",
    "SchemaMigrationPlan",
    "EphemeralDatabaseSpec",
    "MigrationResult",
    "DBMigrationAgent",
]


@dataclass(slots=True)
class MigrationRecord:
    """Representation of a migration file tracked inside a framework."""

    path: str
    name: str
    order_key: str
    created: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "order_key": self.order_key,
            "created": self.created,
        }


@dataclass(slots=True)
class SchemaMigrationPlan:
    """Plan describing how to generate migrations for a framework."""

    framework: str
    migrations_dir: str
    workdir: str
    env: Mapping[str, str]
    command_template: tuple[str, ...]
    apply_template: tuple[str, ...] | None
    existing: tuple[MigrationRecord, ...]
    artifact_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "migrations_dir": self.migrations_dir,
            "workdir": self.workdir,
            "env": dict(self.env),
            "command_template": list(self.command_template),
            "apply_template": list(self.apply_template) if self.apply_template else None,
            "existing": [record.to_dict() for record in self.existing],
            "artifact_paths": list(self.artifact_paths),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EphemeralDatabaseSpec:
    """Lifecycle hooks for provisioning an ephemeral database runtime."""

    name: str
    setup: Sequence[str] | str
    teardown: Sequence[str] | str | None = None
    workdir: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    setup_timeout_ms: int | None = None
    teardown_timeout_ms: int | None = None
    artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "setup": list(self.setup) if isinstance(self.setup, Sequence) and not isinstance(self.setup, str) else self.setup,
            "teardown": list(self.teardown) if isinstance(self.teardown, Sequence) and not isinstance(self.teardown, str) else self.teardown,
            "workdir": self.workdir,
            "env": dict(self.env),
            "setup_timeout_ms": self.setup_timeout_ms,
            "teardown_timeout_ms": self.teardown_timeout_ms,
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MigrationResult:
    """Outcome of running a migration generation workflow."""

    framework: str
    plan: SchemaMigrationPlan
    generated: tuple[str, ...]
    final_state: tuple[MigrationRecord, ...]
    applied: bool
    reports: tuple[RunReport, ...]
    artifacts: tuple[str, ...]
    ephemeral: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "plan": self.plan.to_dict(),
            "generated": list(self.generated),
            "final_state": [record.to_dict() for record in self.final_state],
            "applied": self.applied,
            "reports": [report.to_dict() for report in self.reports],
            "artifacts": list(self.artifacts),
            "ephemeral": self.ephemeral,
            "metadata": dict(self.metadata),
        }


class DBMigrationAgent:
    """Orchestrates schema migration generation across multiple frameworks."""

    _DEFAULT_FRAMEWORKS: Mapping[str, Mapping[str, Any]] = {
        "alembic": {
            "migrations_dir": "migrations",
            "generate": ("alembic", "revision", "--autogenerate", "-m", "{name}"),
            "apply": ("alembic", "upgrade", "head"),
        },
        "prisma": {
            "migrations_dir": "prisma/migrations",
            "generate": ("npx", "prisma", "migrate", "dev", "--name", "{name}"),
            "apply": ("npx", "prisma", "migrate", "deploy"),
        },
        "drizzle": {
            "migrations_dir": "drizzle",
            "generate": ("pnpm", "drizzle-kit", "generate"),
            "apply": None,
        },
    }

    def __init__(
        self,
        repo_context: RepoContextAgent,
        *,
        runner: RunnerAgent | None = None,
        frameworks: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.repo_context = repo_context
        self.runner = runner or RunnerAgent(repo_root=self.repo_context.repo_root)
        payload: dict[str, Mapping[str, Any]] = dict(self._DEFAULT_FRAMEWORKS)
        if frameworks:
            for name, spec in frameworks.items():
                payload[name.lower()] = dict(spec)
        self._frameworks: dict[str, Mapping[str, Any]] = payload
        self._ephemeral_specs: dict[str, EphemeralDatabaseSpec] = {}

    # ------------------------------------------------------------------
    # Framework discovery & planning
    # ------------------------------------------------------------------
    def plan_migration(
        self,
        framework: str,
        *,
        migrations_dir: str | os.PathLike[str] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SchemaMigrationPlan:
        """Construct a migration plan for the requested framework."""

        config = self._get_framework_config(framework)
        framework_name = framework.lower()
        migrations_path = self._resolve_path(
            migrations_dir or config.get("migrations_dir") or "migrations"
        )
        workdir_path = self._resolve_path(workdir or config.get("workdir") or self.repo_context.repo_root)
        command_template = self._coerce_command(config.get("generate"))
        if not command_template:
            raise ValueError(f"Framework '{framework_name}' is missing a generate command")
        apply_template = self._coerce_command(config.get("apply"))

        env_payload: dict[str, str] = {}
        config_env = config.get("env")
        if isinstance(config_env, Mapping):
            env_payload.update({str(k): str(v) for k, v in config_env.items()})
        if env:
            env_payload.update({str(k): str(v) for k, v in env.items()})

        migrations_path.mkdir(parents=True, exist_ok=True)
        records = tuple(self._collect_migrations(migrations_path))
        artifact_set = {str(self._relative_to_repo(migrations_path))}
        for item in config.get("artifacts", ()):
            artifact_set.add(str(item))
        artifact_paths = tuple(sorted(artifact_set))

        metadata: dict[str, Any] = {
            "framework": framework_name,
            "config": {
                key: value
                for key, value in config.items()
                if key not in {"generate", "apply"}
            },
        }

        plan = SchemaMigrationPlan(
            framework=framework_name,
            migrations_dir=str(migrations_path),
            workdir=str(workdir_path),
            env=env_payload,
            command_template=command_template,
            apply_template=apply_template,
            existing=records,
            artifact_paths=artifact_paths,
            metadata=metadata,
        )
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_migration(
        self,
        plan: SchemaMigrationPlan,
        *,
        migration_name: str,
        apply: bool = False,
        extra_args: Sequence[str] | None = None,
        ephemeral: str | EphemeralDatabaseSpec | None = None,
    ) -> MigrationResult:
        """Execute a migration plan and capture the resulting artefacts."""

        if not migration_name:
            raise ValueError("migration_name must be provided")

        migrations_path = Path(plan.migrations_dir)
        before_records = {record.name for record in plan.existing}
        reports: list[RunReport] = []
        artifacts: list[str] = list(plan.artifact_paths)
        ephemeral_name: str | None = None
        ephemeral_spec = self._resolve_ephemeral(ephemeral)
        teardown_error: Exception | None = None
        error: Exception | None = None

        try:
            if ephemeral_spec is not None:
                setup_report = self.runner.run_shell(
                    ephemeral_spec.setup,
                    workdir=ephemeral_spec.workdir or plan.workdir,
                    env=self._merge_env(plan.env, ephemeral_spec.env),
                    timeout_ms=ephemeral_spec.setup_timeout_ms,
                )
                reports.append(setup_report)
                if not setup_report.ok:
                    raise RuntimeError(
                        f"Ephemeral database setup '{ephemeral_spec.name}' failed"
                    )
                ephemeral_name = ephemeral_spec.name
                artifacts.extend(str(item) for item in ephemeral_spec.artifacts)

            command = self._render_command(plan.command_template, migration_name, extra_args)
            generate_report = self.runner.run_shell(
                command,
                workdir=plan.workdir,
                env=plan.env,
            )
            reports.append(generate_report)
            if not generate_report.ok:
                raise RuntimeError("Migration generation command failed")

            if apply and plan.apply_template:
                apply_command = self._render_command(plan.apply_template, migration_name, None)
                apply_report = self.runner.run_shell(
                    apply_command,
                    workdir=plan.workdir,
                    env=plan.env,
                )
                reports.append(apply_report)
                if not apply_report.ok:
                    raise RuntimeError("Migration apply command failed")
            applied = bool(apply and plan.apply_template)

            final_records = tuple(self._collect_migrations(migrations_path))
            generated = tuple(
                sorted(
                    str(self._relative_to_repo(Path(plan.migrations_dir) / name))
                    for name in {record.name for record in final_records} - before_records
                )
            )
            result = MigrationResult(
                framework=plan.framework,
                plan=plan,
                generated=generated,
                final_state=final_records,
                applied=applied,
                reports=tuple(reports),
                artifacts=tuple(dict.fromkeys(artifacts)),
                ephemeral=ephemeral_name,
            )
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            if ephemeral_spec is not None and ephemeral_spec.teardown is not None:
                teardown_report = self.runner.run_shell(
                    ephemeral_spec.teardown,
                    workdir=ephemeral_spec.workdir or plan.workdir,
                    env=self._merge_env(plan.env, ephemeral_spec.env),
                    timeout_ms=ephemeral_spec.teardown_timeout_ms,
                )
                reports.append(teardown_report)
                if not teardown_report.ok and teardown_error is None and error is None:
                    teardown_error = RuntimeError(
                        f"Ephemeral database teardown '{ephemeral_spec.name}' failed"
                    )
            if teardown_error is not None and error is None:
                raise teardown_error

    # ------------------------------------------------------------------
    # Ephemeral database helpers
    # ------------------------------------------------------------------
    def register_ephemeral_database(self, spec: EphemeralDatabaseSpec) -> None:
        """Register an ephemeral database specification for later reuse."""

        self._ephemeral_specs[spec.name] = spec

    def get_ephemeral_specs(self) -> tuple[EphemeralDatabaseSpec, ...]:
        return tuple(self._ephemeral_specs.values())

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_result(self, result: MigrationResult) -> str:
        """Create a concise textual summary for a migration result."""

        generated = ", ".join(result.generated) if result.generated else "no new files"
        applied = "applied" if result.applied else "not applied"
        ephemeral = (
            f" (ephemeral runtime: {result.ephemeral})" if result.ephemeral else ""
        )
        return (
            f"Framework '{result.framework}' generated {generated} and {applied} migrations"
            f"{ephemeral}."
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _get_framework_config(self, framework: str) -> Mapping[str, Any]:
        key = framework.lower()
        if key not in self._frameworks:
            raise ValueError(f"Unsupported migration framework '{framework}'")
        return self._frameworks[key]

    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.repo_context.repo_root) / path
        return path

    def _relative_to_repo(self, value: Path) -> Path:
        try:
            return value.relative_to(self.repo_context.repo_root)
        except ValueError:
            return value

    def _collect_migrations(self, path: Path) -> list[MigrationRecord]:
        records: list[MigrationRecord] = []
        if not path.exists():
            return records
        for entry in sorted(path.iterdir()):
            if entry.is_file():
                stat = entry.stat()
                relative = self._relative_to_repo(entry)
                records.append(
                    MigrationRecord(
                        path=str(relative),
                        name=entry.name,
                        order_key=self._build_order_key(entry.name),
                        created=stat.st_mtime,
                    )
                )
        records.sort(key=lambda item: (item.order_key, item.created, item.name))
        return records

    @staticmethod
    def _build_order_key(name: str) -> str:
        prefix = name.split("_", 1)[0]
        return prefix.zfill(4)

    @staticmethod
    def _coerce_command(command: Any) -> tuple[str, ...] | None:
        if command is None:
            return None
        if isinstance(command, (list, tuple)):
            return tuple(str(part) for part in command)
        if isinstance(command, str):
            return (command,)
        raise TypeError("Command must be a sequence or string")

    @staticmethod
    def _merge_env(base: Mapping[str, str], extra: Mapping[str, str]) -> dict[str, str]:
        payload = dict(base)
        payload.update({str(k): str(v) for k, v in extra.items()})
        return payload

    def _render_command(
        self,
        template: tuple[str, ...],
        migration_name: str,
        extra_args: Sequence[str] | None,
    ) -> Sequence[str] | str:
        if len(template) == 1:
            command = template[0].format(name=migration_name)
            if extra_args:
                command = " ".join([command, *map(str, extra_args)])
            return command
        rendered = [part.format(name=migration_name) for part in template]
        if extra_args:
            rendered.extend(str(arg) for arg in extra_args)
        return rendered

    def _resolve_ephemeral(
        self, value: str | EphemeralDatabaseSpec | None
    ) -> EphemeralDatabaseSpec | None:
        if value is None:
            return None
        if isinstance(value, EphemeralDatabaseSpec):
            return value
        spec = self._ephemeral_specs.get(value)
        if spec is None:
            raise KeyError(f"Unknown ephemeral database spec '{value}'")
        return spec
