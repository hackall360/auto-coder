from __future__ import annotations

import importlib.machinery
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

if "lmstudio" not in sys.modules:
    class _DummyChat:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages: list[Any] = []

        @classmethod
        def from_history(cls, history: Mapping[str, Any] | None) -> "_DummyChat":
            instance = cls()
            if isinstance(history, Mapping):
                instance.messages = list(history.get("messages", []))
            return instance

        def append(self, message: Any) -> None:
            self.messages.append(message)

    class _DummyModel:
        def respond(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            return {"choices": [{"message": {"content": ""}}]}

        def respond_stream(self, *args: Any, **kwargs: Any):  # pragma: no cover - streaming unused
            return iter(())

    def _dummy_llm(*args: Any, **kwargs: Any) -> _DummyModel:
        return _DummyModel()

    class ToolFunctionDef:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple stub
            self.args = args
            self.kwargs = kwargs
            self.name = kwargs.get("name")
    sys.modules["lmstudio"] = types.SimpleNamespace(
        Chat=_DummyChat,
        llm=_dummy_llm,
        ToolFunctionDef=ToolFunctionDef,
    )

if "psutil" not in sys.modules:
    class _DummyPsutil(types.SimpleNamespace):
        class NoSuchProcess(Exception):
            pass

        class TimeoutExpired(Exception):
            pass

        def Process(self, pid: int):  # pragma: no cover - simple stub
            raise self.NoSuchProcess()

        def process_iter(self, attrs: Any):  # pragma: no cover - process listing unused
            return []

    sys.modules["psutil"] = _DummyPsutil()

if "tooling" not in sys.modules:
    class ToolSpec:
        def __init__(self, func: Any, name: str | None = None) -> None:
            self.func = func
            self.name = name or getattr(func, "__name__", "tool")

    class ToolRegistry:
        def __init__(self) -> None:
            self._tools: dict[str, ToolSpec] = {}

        def register(self, tool: Any, replace: bool = False) -> ToolSpec:
            spec = tool if isinstance(tool, ToolSpec) else ToolSpec(tool)
            if not replace and spec.name in self._tools:
                raise ValueError(f"Tool '{spec.name}' already registered")
            self._tools[spec.name] = spec
            return spec

    def resolve_tools(*args: Any, **kwargs: Any) -> list[ToolSpec]:  # pragma: no cover - unused
        return []

    sys.modules["tooling"] = types.SimpleNamespace(
        ToolSpec=ToolSpec,
        ToolRegistry=ToolRegistry,
        resolve_tools=resolve_tools,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if "agents" not in sys.modules:
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = [str(PROJECT_ROOT / "agents")]
    agents_pkg.__spec__ = importlib.machinery.ModuleSpec("agents", loader=None, is_package=True)
    sys.modules["agents"] = agents_pkg

if "chat" not in sys.modules:
    class _StubChatSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - not exercised in tests
            self.model = None
            self.tools = []
            self.system_prompt = None
            self.chat = types.SimpleNamespace(messages=[])

        @classmethod
        def create(cls, **kwargs: Any) -> "_StubChatSession":
            return cls()

        def set_tools(self, *args: Any, **kwargs: Any) -> None:
            return

        def append_user_input(self, content: str) -> None:
            return

        def append_tool_response(self, *args: Any, **kwargs: Any) -> None:
            return

    CallbackMap = Mapping[str, Callable[..., Any]]

    sys.modules["chat"] = types.SimpleNamespace(
        CallbackMap=CallbackMap,
        ChatSession=_StubChatSession,
    )

from agents.dependency import (
    DependencyBuildAgent,
    DependencyCacheDirective,
    DependencyResolution,
    LockfileDiffSummary,
)
from agents.manager import ManagerAgent
from agents.runner import RunReport
from internal.structures import StructuredResponse


def _build_run_report(command: tuple[str, ...], workdir: str) -> RunReport:
    command_display = " ".join(command)
    return RunReport(
        identifier=1,
        runner="test",
        command=command,
        command_display=command_display,
        workdir=workdir,
        env={},
        status="success",
        ok=True,
        exit_code=0,
        pid=123,
        stdout="ok",
        stderr="",
        combined_output="ok",
        error=None,
        start_time=0.0,
        end_time=0.1,
        duration=0.1,
        artifacts=(),
        metadata={},
        raw={},
    )


def test_lockfile_diff_summary_identifies_changes() -> None:
    before = "\n".join(
        [
            "{",
            "  \"dependencies\": {",
            "    \"left-pad\": \"1.0.0\",",
            "    \"lodash\": \"4.17.21\"",
            "  }",
            "}",
        ]
    )
    after = "\n".join(
        [
            "{",
            "  \"dependencies\": {",
            "    \"left-pad\": \"1.2.0\",",
            "    \"chalk\": \"5.0.0\"",
            "  }",
            "}",
        ]
    )

    summary = LockfileDiffSummary.from_contents("package-lock.json", before, after)
    assert summary is not None
    assert summary.has_changes
    assert summary.added == ("chalk@5.0.0",)
    assert summary.removed == ("lodash@4.17.21",)
    assert summary.updated == ("left-pad 1.0.0 → 1.2.0",)
    description = summary.describe()
    assert "package-lock.json" in description
    assert "chalk@5.0.0" in description
    assert "left-pad" in description


def test_dependency_agent_runs_and_summarises_lockfile(tmp_path: Path) -> None:
    lockfile = tmp_path / "package-lock.json"
    before = "\n".join(
        [
            "{",
            "  \"dependencies\": {",
            "    \"left-pad\": \"1.0.0\",",
            "    \"lodash\": \"4.17.21\"",
            "  }",
            "}",
        ]
    )
    after = "\n".join(
        [
            "{",
            "  \"dependencies\": {",
            "    \"left-pad\": \"1.2.0\",",
            "    \"chalk\": \"5.0.0\"",
            "  }",
            "}",
        ]
    )
    lockfile.write_text(before)

    class FakeRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
            self.repo_root = str(tmp_path)

        def run_shell(self, command: tuple[str, ...], **kwargs: Any) -> RunReport:
            self.calls.append((tuple(command), kwargs))
            lockfile.write_text(after)
            workdir = kwargs.get("workdir", str(tmp_path))
            return _build_run_report(tuple(command), workdir)

    runner = FakeRunner()
    agent = DependencyBuildAgent(runner=runner, repo_root=tmp_path)
    status_events: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def capture(message: str, kind: str, payload: Mapping[str, Any] | None) -> None:
        status_events.append((message, kind, payload))

    resolution = agent.run_task(
        manager="npm",
        lockfiles=[lockfile.name],
        status_callback=capture,
    )

    assert runner.calls, "runner should be invoked"
    assert resolution.ok
    assert resolution.lockfile_summaries
    summary = resolution.lockfile_summaries[0]
    assert summary.added == ("chalk@5.0.0",)
    assert summary.removed == ("lodash@4.17.21",)
    assert summary.updated == ("left-pad 1.0.0 → 1.2.0",)
    assert resolution.cache_directive is not None
    assert "node_modules" in resolution.cache_directive.paths
    formatted = agent.format_resolution(resolution)
    assert "npm command" in formatted
    assert any(event[1] == "dependency_start" for event in status_events)
    assert any(event[1] == "dependency_complete" for event in status_events)


def test_manager_delegates_dependency_tasks(tmp_path: Path) -> None:
    class DummySession:
        def __init__(self) -> None:
            self.rounds: list[Any] = []

        def add_round_hooks(self, **_: Any) -> None:  # pragma: no cover - interface compliance
            return

        def act(self, *args: Any, **kwargs: Any) -> tuple[str, StructuredResponse]:  # pragma: no cover
            raise AssertionError("Dependency tasks should not call session.act")

    @dataclass
    class StubDependencyAgent:
        resolution: DependencyResolution

        def run_task(self, **_: Any) -> DependencyResolution:
            return self.resolution

        def format_resolution(self, result: DependencyResolution) -> str:
            return result.describe()

    lock_summary = LockfileDiffSummary(
        path="package-lock.json",
        added=("chalk@5.0.0",),
        removed=(),
        updated=(),
        raw_diff="",
    )
    resolution = DependencyResolution(
        manager="npm",
        command=("npm", "install"),
        report=_build_run_report(("npm", "install"), str(tmp_path)),
        lockfile_summaries=(lock_summary,),
        cache_directive=DependencyCacheDirective(manager="npm", paths=("node_modules",), description="cache"),
    )
    stub_agent = StubDependencyAgent(resolution)
    captured: list[Any] = []
    manager = ManagerAgent(
        session=DummySession(),
        status_callback=captured.append,
        dependency_agent=stub_agent,  # type: ignore[arg-type]
    )
    task = {
        "name": "deps",
        "metadata": {"kind": "dependency", "manager": "npm"},
    }

    text, structured = manager._execute_task(task, user_message="update dependencies")

    assert text
    assert isinstance(structured, StructuredResponse)
    assert structured.parsed is not None
    assert structured.parsed["manager"] == "npm"
    assert any(update.kind == "dependency_summary" for update in captured)
    assert manager._last_response_text == text
