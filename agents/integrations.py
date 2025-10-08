"""Agents for working with CI/CD integrations, container builds, and releases."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence
from string import Template

from .repo_context import RepoContextAgent
from .runner import RunReport, RunnerAgent

__all__ = [
    "CIJobPlan",
    "PipelineUpdateResult",
    "ReleaseMetadata",
    "IntegrationsAgent",
]


@dataclass(slots=True)
class CIJobPlan:
    """Declarative description of a CI pipeline job to be rendered."""

    provider: str
    name: str
    path: str
    template: str
    variables: Mapping[str, Any] = field(default_factory=dict)
    rendered: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "name": self.name,
            "path": self.path,
            "template": self.template,
            "variables": dict(self.variables),
        }
        if self.rendered is not None:
            payload["rendered"] = self.rendered
        return payload

    def with_rendered(self, content: str) -> "CIJobPlan":
        """Return a copy of the plan with ``rendered`` populated."""

        return CIJobPlan(
            provider=self.provider,
            name=self.name,
            path=self.path,
            template=self.template,
            variables=dict(self.variables),
            rendered=content,
        )


@dataclass(slots=True)
class PipelineUpdateResult:
    """Summary of a pipeline write operation."""

    provider: str
    path: str
    rendered: str
    changed: bool
    digest: str
    previous_digest: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "path": self.path,
            "rendered": self.rendered,
            "changed": self.changed,
            "digest": self.digest,
            "previous_digest": self.previous_digest,
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(slots=True)
class ReleaseMetadata:
    """Structured representation of a release announcement."""

    version: str
    tag: str
    notes: str | None
    artifacts: tuple[str, ...]
    branch: str | None
    commit: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tag": self.tag,
            "notes": self.notes,
            "artifacts": list(self.artifacts),
            "branch": self.branch,
            "commit": self.commit,
            "metadata": dict(self.metadata),
        }


class IntegrationsAgent:
    """High-level helper for CI detection, pipeline authoring, and releases."""

    def __init__(
        self,
        *,
        repo_context: RepoContextAgent,
        runner: RunnerAgent | None = None,
    ) -> None:
        self.repo_context = repo_context
        self.runner = runner or RunnerAgent(repo_root=repo_context.repo_root)

    # ------------------------------------------------------------------
    # CI discovery & planning
    # ------------------------------------------------------------------
    def detect_pipelines(self) -> dict[str, tuple[str, ...]]:
        """Detect existing CI systems using the repository context agent."""

        return self.repo_context.detect_ci_systems()

    def render_pipeline(self, plan: CIJobPlan) -> CIJobPlan:
        """Render a pipeline template with the provided variables."""

        template = Template(plan.template)
        rendered = template.safe_substitute(plan.variables)
        return plan.with_rendered(rendered)

    def apply_pipeline(self, plan: CIJobPlan) -> PipelineUpdateResult:
        """Persist a rendered pipeline to disk."""

        if plan.rendered is None:
            raise ValueError("CIJobPlan must be rendered before applying")

        previous = self.repo_context.read_file(plan.path)
        previous_digest = hashlib.sha256(previous.encode("utf-8")).hexdigest() if previous else None
        digest = hashlib.sha256(plan.rendered.encode("utf-8")).hexdigest()
        changed = self.repo_context.update_file_if_changed(plan.path, plan.rendered)
        metadata: dict[str, Any] = {
            "previous_length": len(previous) if previous is not None else 0,
            "new_length": len(plan.rendered),
        }
        return PipelineUpdateResult(
            provider=plan.provider,
            path=plan.path,
            rendered=plan.rendered,
            changed=changed,
            digest=digest,
            previous_digest=previous_digest,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Container orchestration
    # ------------------------------------------------------------------
    def run_container_build(
        self,
        image: str,
        *,
        context: str = ".",
        dockerfile: str | None = None,
        build_args: Mapping[str, Any] | None = None,
        extra_args: Sequence[str] | None = None,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunReport:
        """Execute a container build and return the resulting run report."""

        command: list[str] = ["docker", "build", "-t", image]
        if dockerfile:
            command.extend(["-f", dockerfile])
        if build_args:
            for key, value in build_args.items():
                command.extend(["--build-arg", f"{key}={value}"])
        if extra_args:
            command.extend(str(arg) for arg in extra_args)
        command.append(context)
        return self.runner.run_shell(
            command,
            workdir=workdir,
            env=env,
            combine_output=True,
        )

    # ------------------------------------------------------------------
    # Release helpers
    # ------------------------------------------------------------------
    def prepare_release_metadata(
        self,
        version: str,
        *,
        tag: str | None = None,
        notes: str | None = None,
        artifacts: Sequence[str] | None = None,
        commit: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReleaseMetadata:
        """Prepare a standardized release metadata payload."""

        branch: str | None = None
        try:
            branch = self.repo_context.current_branch()
        except Exception:  # pragma: no cover - git may not be initialized
            branch = None
        tag_value = tag or f"v{version}"
        artifacts_tuple = tuple(str(item) for item in artifacts) if artifacts else ()
        metadata_payload: Mapping[str, Any] = metadata or {}
        return ReleaseMetadata(
            version=version,
            tag=tag_value,
            notes=notes,
            artifacts=artifacts_tuple,
            branch=branch,
            commit=commit,
            metadata=dict(metadata_payload),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def orchestrate_pipeline(
        self,
        plan: CIJobPlan,
        *,
        apply: bool = True,
    ) -> PipelineUpdateResult:
        """Render and optionally apply a pipeline plan."""

        rendered_plan = self.render_pipeline(plan)
        if not apply:
            return PipelineUpdateResult(
                provider=rendered_plan.provider,
                path=rendered_plan.path,
                rendered=rendered_plan.rendered or "",
                changed=False,
                digest=hashlib.sha256((rendered_plan.rendered or "").encode("utf-8")).hexdigest(),
                previous_digest=None,
                metadata={"applied": False},
            )
        return self.apply_pipeline(rendered_plan)
