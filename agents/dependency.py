"""Dependency resolution helpers leveraging :class:`RunnerAgent`."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .runner import RunReport, RunnerAgent

__all__ = [
    "DependencyCacheDirective",
    "DependencyResolution",
    "DependencyBuildAgent",
    "LockfileDiffSummary",
]


_JSON_KEY_RE = re.compile(r'"(?P<name>[^"\\]+)"\s*:\s*"(?P<version>[^"\\]+)"')
_REQUIREMENT_RE = re.compile(r"(?P<name>[A-Za-z0-9_.\-]+)==(?P<version>[A-Za-z0-9_.\-]+)")
_POETRY_NAME_RE = re.compile(r"name\s*=\s*\"(?P<name>[^\"]+)\"")
_POETRY_VERSION_RE = re.compile(r"version\s*=\s*\"(?P<version>[^\"]+)\"")


@dataclass(slots=True)
class LockfileDiffSummary:
    """Summary of dependency changes extracted from a lockfile diff."""

    path: str
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    raw_diff: str | None = None

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.updated)

    def describe(self) -> str:
        parts: list[str] = []
        if self.added:
            parts.append(f"added {', '.join(self.added)}")
        if self.removed:
            parts.append(f"removed {', '.join(self.removed)}")
        if self.updated:
            parts.append(f"updated {', '.join(self.updated)}")
        if not parts:
            return f"{self.path}: no substantive changes"
        return f"{self.path}: " + "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "added": list(self.added),
            "removed": list(self.removed),
            "updated": list(self.updated),
            "raw_diff": self.raw_diff,
        }

    @classmethod
    def from_contents(
        cls,
        path: str | os.PathLike[str],
        before: str | None,
        after: str | None,
    ) -> "LockfileDiffSummary" | None:
        before_lines = (before.splitlines() if before is not None else [])
        after_lines = (after.splitlines() if after is not None else [])
        if before_lines == after_lines:
            return None
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=str(path),
                tofile=str(path),
                lineterm="",
            )
        )
        diff_text = "\n".join(diff_lines)
        return cls.from_unified_diff(path=str(path), diff_text=diff_text)

    @classmethod
    def from_unified_diff(cls, path: str, diff_text: str) -> "LockfileDiffSummary":
        added: dict[str, str | None] = {}
        removed: dict[str, str | None] = {}
        pending_poetry_name: str | None = None
        pending_poetry_removed_name: str | None = None
        for line in diff_text.splitlines():
            if not line or line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            prefix = line[0]
            if prefix not in "+-":
                continue
            content = line[1:].strip()
            parsed_name, parsed_version = cls._parse_dependency_line(content)
            if parsed_name:
                target = added if prefix == "+" else removed
                target.setdefault(parsed_name, parsed_version)
                pending_poetry_name = parsed_name if prefix == "+" else pending_poetry_name
                pending_poetry_removed_name = (
                    parsed_name if prefix == "-" else pending_poetry_removed_name
                )
                continue
            if prefix == "+":
                version_match = _POETRY_VERSION_RE.search(content)
                if version_match and pending_poetry_name:
                    added[pending_poetry_name] = version_match.group("version")
                    pending_poetry_name = None
            elif prefix == "-":
                version_match = _POETRY_VERSION_RE.search(content)
                if version_match and pending_poetry_removed_name:
                    removed[pending_poetry_removed_name] = version_match.group("version")
                    pending_poetry_removed_name = None
        updated: list[str] = []
        for name in sorted(set(added) & set(removed)):
            new_version = added.pop(name)
            old_version = removed.pop(name)
            updated.append(_format_update(name, old_version, new_version))
        added_entries = [_format_entry(name, version) for name, version in sorted(added.items())]
        removed_entries = [_format_entry(name, version) for name, version in sorted(removed.items())]
        updated_entries = sorted(updated)
        return cls(
            path=path,
            added=tuple(added_entries),
            removed=tuple(removed_entries),
            updated=tuple(updated_entries),
            raw_diff=diff_text,
        )

    @staticmethod
    def _parse_dependency_line(content: str) -> tuple[str | None, str | None]:
        match = _JSON_KEY_RE.search(content)
        if match:
            return match.group("name"), match.group("version")
        if content and ":" in content:
            token = content.split(":", 1)[0].strip('"')
            if token:
                if "@" in token:
                    name, version = token.rsplit("@", 1)
                    if name:
                        return name, version or None
                return token, None
        match = _REQUIREMENT_RE.search(content)
        if match:
            return match.group("name"), match.group("version")
        match = _POETRY_NAME_RE.search(content)
        if match:
            return match.group("name"), None
        return None, None


def _format_entry(name: str, version: str | None) -> str:
    if version:
        return f"{name}@{version}"
    return name


def _format_update(name: str, old: str | None, new: str | None) -> str:
    old_display = old or "?"
    new_display = new or "?"
    return f"{name} {old_display} → {new_display}"


@dataclass(slots=True)
class DependencyCacheDirective:
    """Hint to persist package-manager specific caches between runs."""

    manager: str
    paths: tuple[str, ...]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manager": self.manager,
            "paths": list(self.paths),
            "description": self.description,
        }

    @property
    def hint(self) -> str:
        if not self.paths:
            return self.description or ""
        joined = ", ".join(self.paths)
        if self.description:
            return f"{self.description}: {joined}"
        return joined


@dataclass(slots=True)
class DependencyResolution:
    """Outcome of executing a dependency management command."""

    manager: str
    command: tuple[str, ...]
    report: RunReport
    lockfile_summaries: tuple[LockfileDiffSummary, ...] = ()
    cache_directive: DependencyCacheDirective | None = None

    @property
    def ok(self) -> bool:
        return bool(self.report.ok)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "manager": self.manager,
            "command": list(self.command),
            "status": self.report.status,
            "ok": self.ok,
            "exit_code": self.report.exit_code,
            "stdout": self.report.stdout,
            "stderr": self.report.stderr,
            "lockfiles": [summary.to_dict() for summary in self.lockfile_summaries],
        }
        if self.cache_directive is not None:
            payload["cache_directive"] = self.cache_directive.to_dict()
        return payload

    def describe(self) -> str:
        status = "succeeded" if self.ok else "failed"
        command_display = " ".join(self.command) if self.command else self.manager
        lines = [f"{self.manager} command '{command_display}' {status}."]
        for summary in self.lockfile_summaries:
            lines.append(summary.describe())
        if self.cache_directive:
            lines.append(f"Cache hint: {self.cache_directive.hint}")
        if not self.ok and self.report.stderr:
            lines.append(self.report.stderr.strip())
        return "\n".join(lines)


class DependencyBuildAgent:
    """Utility agent orchestrating dependency related build actions."""

    _DEFAULT_COMMANDS: Mapping[str, tuple[str, ...]] = {
        "npm": ("npm", "install"),
        "pnpm": ("pnpm", "install"),
        "yarn": ("yarn", "install"),
        "pip": ("pip", "install"),
        "pipenv": ("pipenv", "install"),
        "poetry": ("poetry", "install"),
        "cargo": ("cargo", "fetch"),
    }

    _DEFAULT_LOCKFILES: Mapping[str, tuple[str, ...]] = {
        "npm": ("package-lock.json", "package.json"),
        "pnpm": ("pnpm-lock.yaml", "package.json"),
        "yarn": ("yarn.lock", "package.json"),
        "pip": ("requirements.txt", "requirements-dev.txt"),
        "pipenv": ("Pipfile.lock", "Pipfile"),
        "poetry": ("poetry.lock", "pyproject.toml"),
        "cargo": ("Cargo.lock", "Cargo.toml"),
    }

    _CACHE_HINTS: Mapping[str, tuple[str, ...]] = {
        "npm": ("node_modules", "~/.npm"),
        "pnpm": ("node_modules", "~/.pnpm-store"),
        "yarn": ("node_modules", "~/.cache/yarn"),
        "pip": (".venv", "~/.cache/pip"),
        "pipenv": (".venv", "~/.cache/pipenv"),
        "poetry": (".venv", "~/.cache/pypoetry"),
        "cargo": ("target", "~/.cargo"),
    }

    def __init__(
        self,
        *,
        runner: RunnerAgent | None = None,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if runner is None:
            self.runner = RunnerAgent(repo_root=repo_root)
            root_hint = self.runner.repo_root
        else:
            self.runner = runner
            if repo_root is not None:
                root_hint = repo_root
            else:
                root_hint = getattr(runner, "repo_root", os.getcwd())
        self.repo_root = Path(root_hint).resolve()

    def run_task(
        self,
        *,
        manager: str,
        command: Sequence[str] | str | None = None,
        packages: Sequence[str] | None = None,
        lockfiles: Sequence[str | os.PathLike[str]] | None = None,
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        status_callback: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
    ) -> DependencyResolution:
        manager_key = manager.lower()
        resolved_command = self._resolve_command(manager_key, command, packages)
        resolved_lockfiles = tuple(self._resolve_lockfiles(manager_key, lockfiles))
        cwd = self._resolve_workdir(workdir)
        before_snapshots = self._snapshot_lockfiles(cwd, resolved_lockfiles)
        self._emit_status(
            status_callback,
            f"Running {manager_key} command",
            kind="dependency_start",
            payload={"command": list(resolved_command), "lockfiles": list(resolved_lockfiles)},
        )
        report = self.runner.run_shell(
            resolved_command,
            workdir=cwd,
            env=env,
            combine_output=True,
        )
        after_snapshots = self._snapshot_lockfiles(cwd, resolved_lockfiles)
        summaries: list[LockfileDiffSummary] = []
        for path in resolved_lockfiles:
            summary = LockfileDiffSummary.from_contents(
                path,
                before_snapshots.get(path),
                after_snapshots.get(path),
            )
            if summary and summary.has_changes:
                summaries.append(summary)
        cache_directive = self._cache_directive(manager_key)
        resolution = DependencyResolution(
            manager=manager_key,
            command=tuple(resolved_command),
            report=report,
            lockfile_summaries=tuple(summaries),
            cache_directive=cache_directive,
        )
        self._emit_status(
            status_callback,
            f"{manager_key} command completed",
            kind="dependency_complete",
            payload=resolution.to_dict(),
        )
        return resolution

    def format_resolution(self, resolution: DependencyResolution) -> str:
        return resolution.describe()

    def _resolve_command(
        self,
        manager: str,
        command: Sequence[str] | str | None,
        packages: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if command is not None:
            if isinstance(command, str):
                return tuple(command.split())
            return tuple(str(part) for part in command)
        base = self._DEFAULT_COMMANDS.get(manager)
        if base is None:
            base = (manager, "install")
        parts: list[str] = [*base]
        if packages:
            parts.extend(str(item) for item in packages)
        return tuple(parts)

    def _resolve_lockfiles(
        self,
        manager: str,
        lockfiles: Sequence[str | os.PathLike[str]] | None,
    ) -> Iterable[str]:
        if lockfiles:
            for item in lockfiles:
                yield str(item)
            return
        defaults = self._DEFAULT_LOCKFILES.get(manager, ())
        for item in defaults:
            yield item

    def _resolve_workdir(self, workdir: str | os.PathLike[str] | None) -> str:
        if workdir is None:
            return str(self.repo_root)
        path = Path(workdir)
        if not path.is_absolute():
            path = self.repo_root / path
        return str(path.resolve())

    def _snapshot_lockfiles(
        self,
        workdir: str,
        lockfiles: Sequence[str],
    ) -> dict[str, str]:
        snapshots: dict[str, str] = {}
        base = Path(workdir)
        for relative in lockfiles:
            path = base / relative
            try:
                snapshots[relative] = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                snapshots[relative] = ""
        return snapshots

    def _cache_directive(self, manager: str) -> DependencyCacheDirective | None:
        paths = self._CACHE_HINTS.get(manager)
        if not paths:
            return None
        description = "Persist dependency caches"
        return DependencyCacheDirective(manager=manager, paths=paths, description=description)

    @staticmethod
    def _emit_status(
        callback: Callable[[str, str, Mapping[str, Any] | None], None] | None,
        message: str,
        *,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return
        callback(message, kind, payload)
