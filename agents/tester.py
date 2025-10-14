"""Testing critic agent responsible for lightweight gating checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from internal.tools import process as process_tools
from internal.tools import shell as shell_tools

__all__ = [
    "TestSuiteConfig",
    "TestSuiteResult",
    "CriticStatusEvent",
    "CriticAnalysis",
    "TestCriticReport",
    "TestCriticAgent",
]


@dataclass(slots=True)
class TestSuiteConfig:
    """Configuration describing a single fast test or lint command."""

    name: str
    command: Sequence[str] | str
    description: str | None = None
    runner: str = "shell"  # "shell" | "process"
    workdir: str | None = None
    env: Mapping[str, str] | None = None
    timeout_ms: int | None = None
    success_codes: Sequence[int] | None = None
    shell_runtime: str | None = None
    combine_output: bool = False
    fail_on_stderr: bool = False
    allow_failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TestSuiteConfig":
        data = dict(payload)
        name = str(data.pop("name"))
        command = data.pop("command")
        return cls(name=name, command=command, **data)


@dataclass(slots=True)
class TestSuiteResult:
    """Result payload produced after executing a configured suite."""

    name: str
    command_display: str
    runner: str
    exit_code: int
    ok: bool
    stdout: str
    stderr: str
    duration: float | None
    started_at: float | None
    ended_at: float | None
    allow_failure: bool
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runner": self.runner,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "allow_failure": self.allow_failure,
            "duration": self.duration,
            "command": self.command_display,
            "error": self.error,
            "metadata": dict(self.metadata),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(slots=True)
class CriticStatusEvent:
    """Structured event emitted during critic execution."""

    message: str
    kind: str = "critic"
    suite: str | None = None
    payload: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"message": self.message, "kind": self.kind}
        if self.suite is not None:
            data["suite"] = self.suite
        if self.payload is not None:
            data["payload"] = dict(self.payload)
        return data


@dataclass(slots=True)
class CriticAnalysis:
    """Heuristic analysis of aggregated test outcomes."""

    status: str
    summary: str
    failing_cases: list[dict[str, Any]]
    suggested_tests: list[str]
    patch_hints: list[str]


@dataclass(slots=True)
class TestCriticReport:
    """Aggregate report returned by :class:`TestCriticAgent`."""

    suites: tuple[TestSuiteResult, ...]
    analysis: CriticAnalysis
    status_events: tuple[CriticStatusEvent, ...]

    @property
    def is_blocking(self) -> bool:
        return self.analysis.status == "fail"

    def to_status_payload(self) -> dict[str, Any]:
        return {
            "status": self.analysis.status,
            "summary": self.analysis.summary,
            "failing_cases": [dict(case) for case in self.analysis.failing_cases],
            "suggested_tests": list(self.analysis.suggested_tests),
            "patch_hints": list(self.analysis.patch_hints),
            "suites": [result.to_dict() for result in self.suites],
        }

    def build_block_message(self) -> str:
        lines: list[str] = [self.analysis.summary]
        blocking = [case for case in self.analysis.failing_cases if case.get("blocking", True)]
        if blocking:
            lines.append("Failing suites:")
            for case in blocking:
                name = case.get("name", "unknown")
                exit_code = case.get("exit_code")
                highlights = case.get("highlights") or []
                lines.append(f" - {name} (exit code {exit_code})")
                for highlight in highlights:
                    lines.append(f"    {highlight}")
        if self.analysis.patch_hints:
            lines.append("Suggested fixes:")
            for hint in self.analysis.patch_hints:
                lines.append(f" - {hint}")
        if self.analysis.suggested_tests:
            lines.append("Suggested additional edge tests:")
            for test in self.analysis.suggested_tests:
                lines.append(f" - {test}")
        return "\n".join(lines)


class TestCriticAgent:
    """Agent responsible for executing and analysing fast verification suites."""

    def __init__(
        self,
        suites: Iterable[TestSuiteConfig | Mapping[str, Any]] | None = None,
        *,
        repo_root: str | None = None,
        status_callback: Callable[[CriticStatusEvent], None] | None = None,
    ) -> None:
        self.repo_root = os.path.abspath(repo_root or os.getcwd())
        self._default_suites: list[TestSuiteConfig] = []
        if suites:
            for cfg in suites:
                self.register_suite(cfg)
        self._status_callback = status_callback
        self._status_events: list[CriticStatusEvent] = []
        self._last_report: TestCriticReport | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_suite(self, config: TestSuiteConfig | Mapping[str, Any]) -> TestSuiteConfig:
        suite = self._coerce_config(config)
        self._default_suites.append(suite)
        return suite

    def clear_suites(self) -> None:
        self._default_suites.clear()

    def set_status_callback(self, callback: Callable[[CriticStatusEvent], None] | None) -> None:
        self._status_callback = callback

    @property
    def has_status_callback(self) -> bool:
        return self._status_callback is not None

    @property
    def last_report(self) -> TestCriticReport | None:
        return self._last_report

    def run_and_report(
        self,
        suites: Iterable[TestSuiteConfig | Mapping[str, Any]] | None = None,
    ) -> TestCriticReport:
        self._status_events.clear()
        configs = [self._coerce_config(cfg) for cfg in (suites or self._default_suites)]
        if not configs:
            analysis = CriticAnalysis(
                status="pass",
                summary="No test suites configured for critic.",
                failing_cases=[],
                suggested_tests=[],
                patch_hints=[],
            )
            report = TestCriticReport(
                suites=tuple(),
                analysis=analysis,
                status_events=tuple(self._status_events),
            )
            self._last_report = report
            return report

        results: list[TestSuiteResult] = []
        for cfg in configs:
            results.append(self._run_suite(cfg))
        analysis = self._analyse(results)
        report = TestCriticReport(
            suites=tuple(results),
            analysis=analysis,
            status_events=tuple(self._status_events),
        )
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _run_suite(self, config: TestSuiteConfig) -> TestSuiteResult:
        self._log_status(
            f"Running {config.name}",
            suite=config.name,
            payload={"command": self._command_display(config.command), "runner": config.runner},
        )
        if config.runner not in {"shell", "process"}:
            raise ValueError(f"Unsupported runner '{config.runner}' for {config.name}")
        if config.runner == "shell":
            result = self._run_shell_suite(config)
        else:
            result = self._run_process_suite(config)
        payload = {
            "exit_code": result.exit_code,
            "ok": result.ok,
            "allow_failure": result.allow_failure,
            "duration": result.duration,
        }
        self._log_status(
            f"Completed {config.name}",
            suite=config.name,
            payload=payload,
            kind="critic_success" if result.ok or result.allow_failure else "critic_failure",
        )
        return result

    def _run_shell_suite(self, config: TestSuiteConfig) -> TestSuiteResult:
        start = time.time()
        response = shell_tools.shell(
            config.command,
            workdir=config.workdir or self.repo_root,
            timeout_ms=config.timeout_ms,
            env=dict(config.env) if config.env else None,
            success_codes=list(config.success_codes) if config.success_codes else None,
            runtime=config.shell_runtime,
            combine_output=config.combine_output,
            fail_on_stderr=config.fail_on_stderr,
        )
        ended = time.time()
        stdout = response.get("stdout") or response.get("output") or ""
        stderr = response.get("stderr") or ""
        code = response.get("code")
        if code is None:
            code = 0 if response.get("status") == "success" else -1
        ok = response.get("status") == "success"
        error = None
        if not ok:
            error = response.get("message") or response.get("stderr") or response.get("output")
        return TestSuiteResult(
            name=config.name,
            command_display=self._command_display(config.command),
            runner="shell",
            exit_code=int(code),
            ok=bool(ok),
            stdout=stdout or "",
            stderr=stderr or "",
            duration=ended - start,
            started_at=start,
            ended_at=ended,
            allow_failure=config.allow_failure,
            error=error,
            metadata=dict(config.metadata),
        )

    def _run_process_suite(self, config: TestSuiteConfig) -> TestSuiteResult:
        start = time.time()
        workdir = config.workdir or self.repo_root
        command_display = self._command_display(config.command)
        command_str = command_display if isinstance(config.command, str) else command_display
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = Path(tmpdir) / "stdout.log"
            stderr_path = Path(tmpdir) / "stderr.log"
            redir_cmd = f"{command_str} 1> {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}"
            run_cmd = ["bash", "-lc", redir_cmd]
            run_resp = process_tools.process(
                "run",
                command=run_cmd,
                workdir=workdir,
                env=dict(config.env) if config.env else None,
            )
            if run_resp.get("status") != "success" or "pid" not in run_resp:
                ended = time.time()
                error = run_resp.get("message", "Failed to launch process")
                return TestSuiteResult(
                    name=config.name,
                    command_display=command_display,
                    runner="process",
                    exit_code=-1,
                    ok=False,
                    stdout=self._safe_read(stdout_path),
                    stderr=self._safe_read(stderr_path) or error or "",
                    duration=ended - start,
                    started_at=start,
                    ended_at=ended,
                    allow_failure=config.allow_failure,
                    error=error,
                    metadata=dict(config.metadata),
                )
            pid = int(run_resp["pid"])
            wait_resp = process_tools.process("wait", pid=pid, timeout_ms=config.timeout_ms)
            ended = time.time()
            exit_code = wait_resp.get("exitcode") if wait_resp.get("status") == "success" else None
            if exit_code is None:
                exit_code = -1
            ok = exit_code == 0 or (
                config.success_codes is not None and exit_code in config.success_codes
            )
            error = None
            if wait_resp.get("status") != "success":
                error = wait_resp.get("message", "Process execution failed")
                # ensure the process is not left running
                process_tools.process("terminate", pid=pid)
                process_tools.process("kill", pid=pid)
            stdout = self._safe_read(stdout_path)
            stderr = self._safe_read(stderr_path)
            return TestSuiteResult(
                name=config.name,
                command_display=command_display,
                runner="process",
                exit_code=int(exit_code),
                ok=bool(ok),
                stdout=stdout,
                stderr=stderr,
                duration=ended - start,
                started_at=start,
                ended_at=ended,
                allow_failure=config.allow_failure,
                error=error,
                metadata=dict(config.metadata),
            )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def _analyse(self, results: Sequence[TestSuiteResult]) -> CriticAnalysis:
        failing_cases: list[dict[str, Any]] = []
        suggested_tests: list[str] = []
        patch_hints: list[str] = []
        tolerated: list[TestSuiteResult] = []
        blocking: list[TestSuiteResult] = []
        for result in results:
            if result.ok:
                continue
            highlights = self._extract_highlights(result)
            case_payload = {
                "name": result.name,
                "exit_code": result.exit_code,
                "highlights": highlights,
                "blocking": not result.allow_failure,
                "command": result.command_display,
            }
            failing_cases.append(case_payload)
            if result.allow_failure:
                tolerated.append(result)
            else:
                blocking.append(result)
                suggested_tests.extend(self._suggest_edge_tests(result))
                patch_hints.extend(self._suggest_patch_hints(result))
        if blocking:
            status = "fail"
            summary = f"{len(blocking)} blocking suite(s) failed: {', '.join(r.name for r in blocking)}."
        elif tolerated:
            status = "warn"
            summary = (
                f"{len(tolerated)} suite(s) failed but were tolerated: "
                f"{', '.join(r.name for r in tolerated)}."
            )
        else:
            status = "pass"
            summary = "All critic suites passed."
        # Deduplicate suggestions while preserving order
        suggested_tests = self._dedupe(suggested_tests)
        patch_hints = self._dedupe(patch_hints)
        return CriticAnalysis(
            status=status,
            summary=summary,
            failing_cases=failing_cases,
            suggested_tests=suggested_tests,
            patch_hints=patch_hints,
        )

    def _extract_highlights(self, result: TestSuiteResult, limit: int = 10) -> list[str]:
        keywords = (
            "fail",
            "error",
            "traceback",
            "exception",
            "assert",
            "warning",
        )
        highlights: list[str] = []
        for stream in (result.stderr, result.stdout):
            for line in stream.splitlines():
                lowered = line.lower()
                if any(keyword in lowered for keyword in keywords):
                    cleaned = line.strip()
                    if cleaned and cleaned not in highlights:
                        highlights.append(cleaned)
                if len(highlights) >= limit:
                    break
            if len(highlights) >= limit:
                break
        if not highlights:
            tail = (result.stderr or result.stdout).splitlines()
            highlights = tail[-limit:]
        return highlights[:limit]

    def _suggest_edge_tests(self, result: TestSuiteResult) -> list[str]:
        combined = f"{result.stdout}\n{result.stderr}".lower()
        suggestions: list[str] = []
        exception_hints: Mapping[str, str] = {
            "keyerror": "Add a test that accesses a missing key to ensure graceful handling.",
            "valueerror": "Add coverage for invalid or boundary input values that trigger ValueError.",
            "typeerror": "Add tests passing unexpected types to assert defensive checks.",
            "indexerror": "Add boundary tests around sequence indexing operations.",
            "zerodivisionerror": "Add tests covering zero divisors or denominators.",
            "assertionerror": "Add a regression test reproducing the failing assertion.",
        }
        for keyword, hint in exception_hints.items():
            if keyword in combined:
                suggestions.append(hint)
        if "timeout" in combined:
            suggestions.append("Add a stress test exercising long-running operations to detect timeouts earlier.")
        if "permission" in combined:
            suggestions.append("Add tests simulating restricted permission scenarios.")
        if "flake8" in result.command_display or "lint" in result.name.lower():
            suggestions.append("Add targeted lint tests for files touched in this change.")
        if "pytest" in result.command_display or "failed" in combined:
            suggestions.append("Add a regression test mirroring the failing scenario.")
        return suggestions

    def _suggest_patch_hints(self, result: TestSuiteResult) -> list[str]:
        combined = f"{result.stdout}\n{result.stderr}".lower()
        hints: list[str] = []
        if "importerror" in combined:
            hints.append("Ensure new modules are added to __init__ exports or installed dependencies.")
        if "mypy" in result.command_display or "type" in combined:
            hints.append("Review typing annotations and update interfaces for stricter checks.")
        if "lint" in result.name.lower() or "flake8" in result.command_display:
            hints.append("Resolve style violations reported by the linter output above.")
        if "traceback" in combined or "exception" in combined:
            hints.append("Investigate stack trace and guard against the triggering input path.")
        if "assert" in combined:
            hints.append("Review assumptions made in assertions and harden the corresponding logic.")
        return hints

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _coerce_config(self, config: TestSuiteConfig | Mapping[str, Any]) -> TestSuiteConfig:
        if isinstance(config, TestSuiteConfig):
            return config
        if isinstance(config, Mapping):
            return TestSuiteConfig.from_mapping(config)
        raise TypeError(f"Unsupported suite configuration type: {type(config)!r}")

    def _command_display(self, command: Sequence[str] | str) -> str:
        if isinstance(command, str):
            return command
        return shlex.join(str(part) for part in command)

    def _dedupe(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    def _safe_read(self, path: Path) -> str:
        try:
            if path.exists():
                return path.read_text()
        except Exception:
            return ""
        return ""

    def _log_status(
        self,
        message: str,
        *,
        suite: str | None = None,
        kind: str = "critic",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        event = CriticStatusEvent(message=message, kind=kind, suite=suite, payload=payload)
        self._status_events.append(event)
        if self._status_callback:
            try:
                self._status_callback(event)
            except Exception:
                # Guard against downstream callback issues to keep critic responsive
                pass
