"""Execution helper responsible for spawning commands and tracking runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from internal.tools import process as process_tools
from internal.tools import shell as shell_tools

__all__ = [
    "RunReport",
    "RunnerAgent",
]


@dataclass(slots=True)
class RunReport:
    """Structured report describing a single command execution."""

    identifier: int
    runner: str
    command: Sequence[str] | str
    command_display: str
    workdir: str
    env: Mapping[str, str]
    status: str
    ok: bool
    exit_code: int | None
    pid: int | None
    stdout: str | None
    stderr: str | None
    combined_output: str | None
    error: str | None
    start_time: float
    end_time: float
    duration: float | None
    artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into a JSON-friendly payload."""

        return {
            "id": self.identifier,
            "runner": self.runner,
            "command": self.command_display,
            "workdir": self.workdir,
            "env": dict(self.env),
            "status": self.status,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output": self.combined_output,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
            "raw": dict(self.raw),
        }

    @property
    def logs(self) -> str:
        """Return a best-effort textual log for quick inspection."""

        if self.combined_output:
            return self.combined_output
        stdout = self.stdout or ""
        stderr = self.stderr or ""
        if stdout and stderr:
            return f"{stdout}\n{stderr}"
        return stdout or stderr


@dataclass(slots=True)
class _RuntimeState:
    setup: Callable[["RunnerAgent", Mapping[str, Any]], Any] | None
    teardown: Callable[["RunnerAgent", Any], None] | None
    prepared: bool = False
    payload: Any = None


class RunnerAgent:
    """High-level helper managing command execution and runtime lifecycle."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str] | None = None,
        default_workdir: str | os.PathLike[str] | None = None,
        default_env: Mapping[str, str] | None = None,
        artifact_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self.default_workdir = (
            Path(default_workdir).resolve()
            if default_workdir is not None
            else self.repo_root
        )
        self._base_env: dict[str, str] = dict(default_env) if default_env else {}
        self._artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else self.repo_root / ".runner_artifacts"
        )
        self._reports: list[RunReport] = []
        self._sequence: int = 0
        self._dependency_cache: set[str] = set()
        self._runtimes: dict[str, _RuntimeState] = {}

    # ------------------------------------------------------------------
    # Public API - command execution
    # ------------------------------------------------------------------
    def run_shell(
        self,
        command: Sequence[str] | str,
        *,
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_ms: int | None = None,
        runtime: str | None = None,
        stdin: str | None = None,
        success_codes: Sequence[int] | None = None,
        fail_on_stderr: bool = False,
        combine_output: bool = False,
        artifacts: Iterable[str | os.PathLike[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunReport:
        """Execute a shell command and capture a structured report."""

        resolved_workdir = self._resolve_workdir(workdir)
        env_payload = self._merge_env(env)
        start_time = time.time()

        result = shell_tools.shell(
            command,
            workdir=resolved_workdir,
            timeout_ms=timeout_ms,
            runtime=runtime,
            env=env_payload,
            stdin=stdin,
            success_codes=list(success_codes) if success_codes else None,
            fail_on_stderr=fail_on_stderr,
            combine_output=combine_output,
        )

        end_time = time.time()
        exit_code = result.get("code")
        status = result.get("status", "error")
        ok = status == "success"
        error = None if ok else result.get("message")
        if not ok and error is None and exit_code is not None:
            error = f"Command exited with code {exit_code}"

        report = self._create_report(
            runner="shell",
            command=command,
            workdir=resolved_workdir,
            env_payload=env_payload,
            status=status,
            ok=ok,
            exit_code=exit_code,
            pid=None,
            stdout=None if combine_output else result.get("stdout"),
            stderr=None if combine_output else result.get("stderr"),
            combined_output=result.get("output") if combine_output else None,
            error=error,
            start_time=start_time,
            end_time=end_time,
            artifacts=artifacts,
            metadata=metadata,
            raw=result,
        )
        return report

    def run_process(
        self,
        command: Sequence[str] | str,
        *,
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        shell: bool | None = None,
        detached: bool = False,
        timeout_ms: int | None = None,
        wait: bool = True,
        artifacts: Iterable[str | os.PathLike[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunReport:
        """Spawn a background process using the unified process tool."""

        resolved_workdir = self._resolve_workdir(workdir)
        env_payload = self._merge_env(env)
        start_time = time.time()

        result = process_tools.process(
            operation="run",
            command=list(command) if isinstance(command, Sequence) and not isinstance(command, str) else command,
            workdir=resolved_workdir,
            env=env_payload,
            shell=shell,
            detached=detached,
        )

        pid = result.get("pid") if result.get("status") == "success" else None
        exit_code: int | None = None
        error = None
        status = result.get("status", "error")
        ok = status == "success"
        wait_result: Mapping[str, Any] | None = None

        if not ok:
            error = result.get("message")
        elif wait and pid is not None:
            wait_result = process_tools.process(
                operation="wait",
                pid=pid,
                timeout_ms=timeout_ms,
            )
            status = wait_result.get("status", status)
            if wait_result.get("status") == "success":
                exit_code = wait_result.get("exitcode")
                ok = ok and (exit_code == 0)
                if not ok and exit_code is not None:
                    error = f"Process exited with code {exit_code}"
            else:
                ok = False
                error = wait_result.get("message") or "Failed to wait for process"

        end_time = time.time()

        report = self._create_report(
            runner="process",
            command=command,
            workdir=resolved_workdir,
            env_payload=env_payload,
            status=status,
            ok=ok,
            exit_code=exit_code,
            pid=pid,
            stdout=None,
            stderr=None,
            combined_output=None,
            error=error,
            start_time=start_time,
            end_time=end_time,
            artifacts=artifacts,
            metadata=metadata,
            raw={"run": result, "wait": wait_result},
        )
        return report

    def get_reports(self) -> tuple[RunReport, ...]:
        """Return an immutable snapshot of the recorded run reports."""

        return tuple(self._reports)

    def clear_reports(self) -> None:
        """Forget all stored run reports."""

        self._reports.clear()

    # ------------------------------------------------------------------
    # Dependency helpers
    # ------------------------------------------------------------------
    def install_dependencies(
        self,
        packages: Sequence[str],
        *,
        installer: Sequence[str] | str = ("pip", "install"),
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime: str | None = None,
        force: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunReport | None:
        """Ensure the provided dependencies are installed once per session."""

        wanted = list(dict.fromkeys(packages))
        if not force:
            wanted = [pkg for pkg in wanted if pkg not in self._dependency_cache]
        if not wanted:
            return None

        if isinstance(installer, Sequence) and not isinstance(installer, str):
            command: Sequence[str] | str = list(installer) + wanted
        else:
            command = str(installer) + " " + " ".join(wanted)

        report = self.run_shell(
            command,
            workdir=workdir,
            env=env,
            runtime=runtime,
            metadata=metadata or {"dependencies": wanted},
        )
        if report.ok:
            self._dependency_cache.update(wanted)
        return report

    def mark_dependency_installed(self, name: str) -> None:
        """Record that a dependency is already satisfied."""

        self._dependency_cache.add(name)

    def clear_dependency_cache(self) -> None:
        """Reset the dependency installation cache."""

        self._dependency_cache.clear()

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------
    def register_runtime(
        self,
        name: str,
        *,
        setup: Callable[["RunnerAgent", Mapping[str, Any]], Any] | None = None,
        teardown: Callable[["RunnerAgent", Any], None] | None = None,
        replace: bool = False,
    ) -> None:
        """Register lifecycle hooks for a named runtime environment."""

        if not replace and name in self._runtimes:
            raise ValueError(f"Runtime '{name}' is already registered")
        self._runtimes[name] = _RuntimeState(setup=setup, teardown=teardown)

    def prepare_runtime(
        self,
        name: str,
        *,
        context: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> Any:
        """Execute the setup hook for ``name`` and cache its state."""

        state = self._runtimes.get(name)
        if state is None:
            raise KeyError(f"Runtime '{name}' is not registered")
        if state.prepared and not force:
            return state.payload
        payload = None
        if state.setup is not None:
            payload = state.setup(self, context or {})
        state.prepared = True
        state.payload = payload
        self._runtimes[name] = state
        return payload

    def teardown_runtime(self, name: str) -> None:
        """Run the teardown hook for ``name`` if it has been prepared."""

        state = self._runtimes.get(name)
        if state is None:
            raise KeyError(f"Runtime '{name}' is not registered")
        if state.prepared and state.teardown is not None:
            state.teardown(self, state.payload)
        state.prepared = False
        state.payload = None
        self._runtimes[name] = state

    def reset_runtimes(self) -> None:
        """Tear down all prepared runtimes."""

        for name in list(self._runtimes):
            state = self._runtimes[name]
            if state.prepared and state.teardown is not None:
                state.teardown(self, state.payload)
            state.prepared = False
            state.payload = None
            self._runtimes[name] = state

    # ------------------------------------------------------------------
    # Artifact utilities
    # ------------------------------------------------------------------
    @property
    def artifact_root(self) -> Path:
        """Directory used for storing generated artifacts."""

        self._artifact_root.mkdir(parents=True, exist_ok=True)
        return self._artifact_root

    def create_artifact_path(self, name: str) -> Path:
        """Return a path inside :attr:`artifact_root` for ``name``."""

        path = self.artifact_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_workdir(self, workdir: str | os.PathLike[str] | None) -> str:
        if workdir is None:
            resolved = self.default_workdir
        else:
            candidate = Path(workdir)
            if not candidate.is_absolute():
                candidate = self.repo_root / candidate
            resolved = candidate.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def _merge_env(self, env: Mapping[str, str] | None) -> dict[str, str] | None:
        if not self._base_env and not env:
            return None
        payload: dict[str, str] = dict(self._base_env)
        if env:
            payload.update(env)
        return payload

    def _create_report(
        self,
        *,
        runner: str,
        command: Sequence[str] | str,
        workdir: str,
        env_payload: Mapping[str, str] | None,
        status: str,
        ok: bool,
        exit_code: int | None,
        pid: int | None,
        stdout: str | None,
        stderr: str | None,
        combined_output: str | None,
        error: str | None,
        start_time: float,
        end_time: float,
        artifacts: Iterable[str | os.PathLike[str]] | None,
        metadata: Mapping[str, Any] | None,
        raw: Mapping[str, Any],
    ) -> RunReport:
        self._sequence += 1
        duration = end_time - start_time if end_time >= start_time else None
        command_display = self._format_command(command)
        normalized_artifacts = self._normalize_artifacts(artifacts)

        report = RunReport(
            identifier=self._sequence,
            runner=runner,
            command=self._snapshot_command(command),
            command_display=command_display,
            workdir=workdir,
            env=dict(env_payload or {}),
            status=status,
            ok=ok,
            exit_code=exit_code,
            pid=pid,
            stdout=stdout,
            stderr=stderr,
            combined_output=combined_output,
            error=error,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            artifacts=normalized_artifacts,
            metadata=dict(metadata) if metadata else {},
            raw=dict(raw),
        )
        self._reports.append(report)
        return report

    def _snapshot_command(self, command: Sequence[str] | str) -> Sequence[str] | str:
        if isinstance(command, str):
            return command
        return tuple(str(part) for part in command)

    def _format_command(self, command: Sequence[str] | str) -> str:
        if isinstance(command, str):
            return command
        if not command:
            return ""
        return shlex.join(str(part) for part in command)

    def _normalize_artifacts(
        self, artifacts: Iterable[str | os.PathLike[str]] | None
    ) -> tuple[str, ...]:
        if not artifacts:
            return ()
        normalized: list[str] = []
        for item in artifacts:
            path = Path(item)
            if not path.is_absolute():
                path = (self.repo_root / path).resolve()
            try:
                relative = path.relative_to(self.repo_root)
                normalized.append(str(relative))
            except ValueError:
                normalized.append(str(path))
        return tuple(normalized)

