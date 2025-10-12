from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Mapping

import pytest


class _StubModel:
    def respond(self, *_, **__):
        return {"choices": [{"message": {"content": "stub"}}]}

    def respond_stream(self, *_, **__):
        yield {"choices": [{"delta": {"content": "stub"}}]}


class _StubChat:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[Mapping[str, Any]] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    @classmethod
    def from_history(cls, history: Any) -> "_StubChat":
        instance = cls()
        if isinstance(history, Mapping):
            instance.messages = list(history.get("messages", []))
        elif isinstance(history, str):
            instance.messages = [{"role": "user", "content": history}]
        return instance

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})


class _StubToolFunctionDef:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        implementation: Any = None,
        **_: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters or {})
        if implementation is None:
            self.implementation = lambda *args, **kwargs: None
        else:
            self.implementation = implementation


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_StubToolFunctionDef,
    ),
)

_psutil_stub = types.ModuleType("psutil")


class _StubProcess:
    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid or 0


_psutil_stub.Process = _StubProcess
sys.modules.setdefault("psutil", _psutil_stub)

from core import (  # noqa: E402  (import after stubbing lmstudio)
    AutoCoderCore,
    load_core_configuration,
)
from memory import get_shared_memory_facade  # noqa: E402


class StubRepoContext:
    def __init__(self, repo_root: str, **_: Any) -> None:
        self.repo_root = Path(repo_root)
        self.stopped = False

    def stop_background_refresh(self) -> None:
        self.stopped = True


class StubResearchAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class StubRunnerAgent:
    def __init__(self, *, repo_root: str | Path | None = None, **kwargs: Any) -> None:
        self.repo_root = Path(repo_root) if repo_root else None
        self.kwargs = kwargs


class StubDependencyAgent:
    def __init__(self, *, runner: Any = None, repo_root: Any = None) -> None:
        self.runner = runner
        self.repo_root = repo_root


class StubDocAgent:
    def __init__(self, repo_context: StubRepoContext, *, research_agent: Any = None, artifact_dir: Any = None) -> None:
        self.repo_context = repo_context
        self.research_agent = research_agent
        self.artifact_dir = artifact_dir

    def attach_research_agent(self, agent: Any) -> None:
        self.research_agent = agent


class StubDBMigrationAgent:
    def __init__(self, repo_context: StubRepoContext, *, runner: Any = None, **_: Any) -> None:
        self.repo_context = repo_context
        self.runner = runner


class StubSecurityAgent:
    def __init__(self, *, runner: Any = None, **_: Any) -> None:
        self.runner = runner


class StubIntegrationsAgent:
    def __init__(self, *, repo_context: StubRepoContext, runner: Any = None, **_: Any) -> None:
        self.repo_context = repo_context
        self.runner = runner


class StubEvalAgent:
    def __init__(self, *, session: Any = None, session_factory: Any = None, **_: Any) -> None:
        self.session = session
        self.session_factory = session_factory


class StubTestCriticAgent:
    def __init__(self, *, repo_root: str | None = None, **_: Any) -> None:
        self.repo_root = repo_root
        self.callback = None

    def set_status_callback(self, callback: Any) -> None:
        self.callback = callback


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.RepoContextAgent", StubRepoContext)
    monkeypatch.setattr("core.ResearchAgent", StubResearchAgent)
    monkeypatch.setattr("core.RunnerAgent", StubRunnerAgent)
    monkeypatch.setattr("core.DependencyBuildAgent", StubDependencyAgent)
    monkeypatch.setattr("core.DocAgent", StubDocAgent)
    monkeypatch.setattr("core.DBMigrationAgent", StubDBMigrationAgent)
    monkeypatch.setattr("core.SecurityAgent", StubSecurityAgent)
    monkeypatch.setattr("core.IntegrationsAgent", StubIntegrationsAgent)
    monkeypatch.setattr("core.EvalAgent", StubEvalAgent)
    monkeypatch.setattr("core.TestCriticAgent", StubTestCriticAgent)


def test_core_builds_manager_with_specialists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    replaced_specs: list[Any] = []

    def fake_register(registry, servers, replace=False):  # noqa: ANN001
        specs: list[Any] = []
        for server in servers:
            label = getattr(getattr(server, "config", server), "label", "stub")
            spec = types.SimpleNamespace(name=label, tool_type="mcp")
            specs.append(spec)
        return specs

    monkeypatch.setattr("core.register_mcp_servers", fake_register)

    class SessionRecorder:
        def __init__(self) -> None:
            self.tool_registry = None
            self.tools: list[Any] = []
            self.round_hooks: list[tuple[Any, Any]] = []
            self.rounds: list[Any] = []

        def replace_mcp_tools(self, new_specs):  # noqa: ANN001
            replaced_specs.extend(list(new_specs))
            self.tools.extend(list(new_specs))
            return list(new_specs)

        def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
            self.round_hooks.append((on_round_start, on_round_end))

    def fake_create_session(self) -> SessionRecorder:
        session = SessionRecorder()
        session.tool_registry = self.tool_registry
        return session

    monkeypatch.setattr(AutoCoderCore, "_create_session", fake_create_session)

    config = load_core_configuration(
        overrides={
            "paths": {"repo_root": str(tmp_path)},
            "agents": {
                "db_migration": True,
                "security": True,
                "integrations": True,
                "eval": True,
            },
            "mcp": {
                "servers": {
                    "demo": {"type": "remote", "url": "https://example.invalid"}
                }
            },
        }
    )

    core = AutoCoderCore(config=config)
    manager = core.build_manager()

    assert isinstance(manager.repo_context, StubRepoContext)
    assert manager.test_critic is core._test_critic_agent
    assert manager.research_agent is core._research_agent
    assert manager._dependency_agent is core._dependency_agent
    assert manager._db_migration_agent is core._db_migration_agent
    assert manager._security_agent is core._security_agent
    assert manager._integrations_agent is core._integrations_agent
    assert manager._doc_agent is core._doc_agent
    assert manager._doc_agent.research_agent is manager.research_agent
    assert core._eval_agent is not None

    assert replaced_specs and replaced_specs[0].name == "demo"
    assert get_shared_memory_facade() is core.memory_facade

    core.shutdown()


def test_core_shutdown_tears_down(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_core_configuration(overrides={"paths": {"repo_root": str(tmp_path)}})
    core = AutoCoderCore(config=config)

    repo_context = StubRepoContext(str(tmp_path))
    core._repo_context_agent = repo_context

    class StubRegistry:
        def __init__(self) -> None:
            self.shutdown_called = False

        def shutdown_all(self) -> None:
            self.shutdown_called = True

    registry = StubRegistry()
    core._mcp_registry = registry  # type: ignore[assignment]

    core.shutdown()

    assert repo_context.stopped is True
    assert registry.shutdown_called is True
    assert get_shared_memory_facade() is not core.memory_facade
