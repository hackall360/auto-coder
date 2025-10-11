from __future__ import annotations

import sys
import types
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

lmstudio_stub = types.ModuleType("lmstudio")


class _ToolFunctionDef:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.parameters = kwargs.get("parameters")
        self.required = kwargs.get("required")
        self.handler = kwargs.get("handler")


lmstudio_stub.ToolFunctionDef = _ToolFunctionDef


class _Chat:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


lmstudio_stub.Chat = _Chat
sys.modules.setdefault("lmstudio", lmstudio_stub)

psutil_stub = types.ModuleType("psutil")

class _PsProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def kill(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: Any | None = None) -> int:
        return 0

    def children(self, recursive: bool = False):
        return []

psutil_stub.Process = _PsProcess
psutil_stub.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
psutil_stub.AccessDenied = type("AccessDenied", (Exception,), {})
psutil_stub.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
psutil_stub.process_iter = lambda *args, **kwargs: []
sys.modules.setdefault("psutil", psutil_stub)

tooling_stub = types.ModuleType("tooling")


@dataclass
class ToolSpec:
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)


def resolve_tools(*args: Any, **kwargs: Any) -> list[ToolSpec]:
    return []


tooling_stub.ToolSpec = ToolSpec
tooling_stub.ToolRegistry = ToolRegistry
tooling_stub.resolve_tools = resolve_tools
sys.modules.setdefault("tooling", tooling_stub)

REPO_ROOT = Path(__file__).resolve().parents[1]
agents_pkg = types.ModuleType("agents")
agents_pkg.__path__ = [str(REPO_ROOT / "agents")]
sys.modules["agents"] = agents_pkg

def _load_agent_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(agents_pkg, name.split(".")[-1], module)
    return module

runner_module = _load_agent_module("agents.runner", "agents/runner.py")
dependency_module = _load_agent_module("agents.dependency", "agents/dependency.py")
db_module = _load_agent_module("agents.db_migration", "agents/db_migration.py")
eval_module = _load_agent_module("agents.eval", "agents/eval.py")
repo_module = _load_agent_module("agents.repo_context", "agents/repo_context.py")
security_module = _load_agent_module("agents.security", "agents/security.py")
manager_module = _load_agent_module("agents.manager", "agents/manager.py")

ManagerAgent = manager_module.ManagerAgent
ManagerStatusUpdate = manager_module.ManagerStatusUpdate
RunReport = runner_module.RunReport
SecurityAgent = security_module.SecurityAgent
SecurityScanFinding = security_module.SecurityScanFinding
SecurityScanReport = security_module.SecurityScanReport
SecurityScanResult = security_module.SecurityScanResult
SecurityToolchain = security_module.SecurityToolchain
from internal.structures import StructuredResponse


def _build_report(
    command: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    raw: Mapping[str, Any] | None = None,
    artifacts: Sequence[str] | None = None,
) -> RunReport:
    return RunReport(
        identifier=1,
        runner="shell",
        command=tuple(command),
        command_display=" ".join(command),
        workdir=".",
        env={},
        status="success",
        ok=True,
        exit_code=0,
        pid=100,
        stdout="",
        stderr="",
        combined_output="",
        error=None,
        start_time=0.0,
        end_time=0.1,
        duration=0.1,
        artifacts=tuple(artifacts or ()),
        metadata=dict(metadata or {}),
        raw=dict(raw or {}),
    )


class DummyRunner:
    def __init__(self, reports: Sequence[RunReport]):
        self._reports = list(reports)
        self.calls: list[tuple[Sequence[str], dict[str, Any]]] = []
        self.default_workdir = "."

    def run_shell(self, command: Sequence[str], **kwargs: Any) -> RunReport:
        if not self._reports:
            raise AssertionError("No more reports configured for DummyRunner")
        self.calls.append((tuple(command), dict(kwargs)))
        return self._reports.pop(0)


def test_security_agent_collects_findings_and_blocks_on_threshold() -> None:
    report = _build_report(
        ("semgrep", "--json"),
        metadata={
            "tool": "semgrep",
            "category": "static_analysis",
            "findings": [
                {
                    "tool": "semgrep",
                    "severity": "HIGH",
                    "message": "Sensitive pattern detected",
                    "location": "app.py:10",
                    "rule_id": "SG001",
                },
            ],
        },
    )
    runner = DummyRunner([report])
    agent = SecurityAgent(runner=runner)

    result = agent.run_scans(toolchains=("static",), severity_threshold="medium")

    assert runner.calls, "SecurityAgent should invoke the configured runner"
    assert result.blocked is True
    assert result.highest_severity == "high"
    assert "HIGH" in (result.summary or "")
    assert result.reports[0].findings[0].location == "app.py:10"
    assert result.reports[0].command[0] == "semgrep"


def test_security_agent_respects_cache_directives() -> None:
    finding = SecurityScanFinding(
        tool="custom",
        message="Issue",
        severity="low",
    )
    custom_chain = SecurityToolchain(
        name="custom",
        category="custom_scan",
        command=("custom", "scan"),
        parser=lambda report: (finding,),
        cache_paths=("~/.cache/custom",),
        cache_description="Custom cache",
    )
    runner = DummyRunner([_build_report(("custom", "scan"))])
    agent = SecurityAgent(runner=runner, toolchains={"custom": custom_chain})

    result = agent.run_scans(toolchains=("custom",), severity_threshold="critical")

    report = result.reports[0]
    assert report.cache_directive is not None
    assert "Custom cache" in report.cache_directive.hint
    assert result.blocked is False


class DummySession:
    def __init__(self) -> None:
        self.rounds: list[Any] = []
        self._on_round_start = None
        self._on_round_end = None

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end

    def act(self, user_message: str | None = None, *, metadata: Mapping[str, Any] | None = None, **_: Any):
        index = len(self.rounds)
        round_record = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "content": "done",
                            "parsed": {"ok": True},
                        }
                    }
                ]
            },
            content="done",
            parsed={"ok": True},
            schema=None,
            structured=False,
        )
        return "done", round_record


@dataclass
class StubSecurityAgent:
    result: SecurityScanResult
    DEFAULT_SEQUENCE: tuple[str, ...] = ("stub",)

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_scans(self, **kwargs: Any) -> SecurityScanResult:
        self.calls.append(dict(kwargs))
        return self.result

    def format_result(self, result: SecurityScanResult) -> str:
        return result.summary or ""


def test_manager_security_task_blocks_workflow() -> None:
    finding = SecurityScanFinding(tool="semgrep", message="Critical", severity="high")
    run_report = _build_report(("semgrep", "--json"))
    scan_report = SecurityScanReport(
        toolchain="static",
        category="static_analysis",
        command=("semgrep", "--json"),
        report=run_report,
        findings=(finding,),
    )
    scan_result = SecurityScanResult(
        reports=(scan_report,),
        threshold="high",
        blocked=True,
        summary="Security block summary",
    )
    security_agent = StubSecurityAgent(result=scan_result)

    def plan_builder(_: str, **__: Any) -> Sequence[Mapping[str, Any]]:
        return [
            {
                "name": "security-task",
                "kind": "security",
                "security": {"toolchains": ["static"], "severity_threshold": "high"},
                "budget": {"limit": 1.0, "unit": "rounds"},
            }
        ]

    session = DummySession()
    captured: list[ManagerStatusUpdate] = []
    manager = ManagerAgent(
        session=session,
        plan_builder=plan_builder,
        security_agent=security_agent,
        status_callback=captured.append,
    )

    result = manager.run("audit the project")

    assert result.response_text == "Security block summary"
    assert result.structured_response is not None
    assert result.structured_response.parsed["blocked"] is True
    assert manager._gate_blocked is True
    assert security_agent.calls and security_agent.calls[0]["toolchains"] == ["static"]

    kinds = {update.kind for update in captured}
    assert "security_start" in kinds
    assert "security_blocked" in kinds
    assert captured[-1].kind == "error"
    assert captured[-1].message.lower().startswith("workflow blocked by security")

    # The manager should not schedule any assistant rounds for security tasks.
    assert result.rounds == []
