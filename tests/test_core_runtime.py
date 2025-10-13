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


class StubVariedResearchAgent:
    def __init__(
        self,
        base_agent: Any,
        *,
        mode_defaults: Mapping[str, Any] | None = None,
        profiles: Mapping[str, Any] | None = None,
        default_mode: str = "balanced",
    ) -> None:
        self.base_agent = base_agent
        self.kwargs = {
            "mode_defaults": mode_defaults,
            "profiles": profiles,
            "default_mode": default_mode,
        }


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
    monkeypatch.setattr("core.VariedResearchAgent", StubVariedResearchAgent)
    monkeypatch.setattr("core.RunnerAgent", StubRunnerAgent)
    monkeypatch.setattr("core.DependencyBuildAgent", StubDependencyAgent)
    monkeypatch.setattr("core.DocAgent", StubDocAgent)
    monkeypatch.setattr("core.DBMigrationAgent", StubDBMigrationAgent)
    monkeypatch.setattr("core.SecurityAgent", StubSecurityAgent)
    monkeypatch.setattr("core.IntegrationsAgent", StubIntegrationsAgent)
    monkeypatch.setattr("core.EvalAgent", StubEvalAgent)
    monkeypatch.setattr("core.TestCriticAgent", StubTestCriticAgent)


def test_research_agent_uses_research_settings(tmp_path: Path) -> None:
    env = {
        "AUTO_CODER_REPO_ROOT": str(tmp_path),
        "AUTO_CODER_RESEARCH_CACHE_TOP_K": "13",
        "AUTO_CODER_RESEARCH_USER_AGENT_POOL": "env-one, env-two",
        "AUTO_CODER_RESEARCH_INCOGNITO_CONTEXTS": "true",
    }

    config = load_core_configuration(
        env=env,
        overrides={
            "paths": {"repo_root": str(tmp_path)},
            "models": {"allow_external_browsing": True},
            "research": {
                "cache_size": 21,
                "max_quote_chars": 1024,
                "web": {
                    "proxy": "http://proxy.local",
                    "anonymous_browsing": True,
                },
            },
        },
    )

    core = AutoCoderCore(config=config)
    agent = core._get_research_agent()

    assert isinstance(agent, StubResearchAgent)
    assert agent.kwargs["cache_size"] == 21
    assert agent.kwargs["cache_top_k"] == 13
    assert agent.kwargs["max_quote_chars"] == 1024
    assert agent.kwargs["anonymous_browsing"] is True
    assert agent.kwargs["proxy"] == "http://proxy.local"
    assert agent.kwargs["user_agent_pool"] == ("env-one", "env-two")
    assert agent.kwargs["incognito_contexts"] is True

    core.shutdown()


def test_varied_research_agent_enabled(tmp_path: Path) -> None:
    env = {"AUTO_CODER_REPO_ROOT": str(tmp_path)}

    config = load_core_configuration(
        env=env,
        overrides={
            "paths": {"repo_root": str(tmp_path)},
            "research": {
                "enable_varied_agent": True,
                "default_mode": "deep",
                "mode_defaults": {"deep": {"top_k": 15}},
                "profiles": {"custom": {"top_k": 9, "max_search_results": 30}},
            },
        },
    )

    core = AutoCoderCore(config=config)
    agent = core._get_research_agent()

    assert isinstance(agent, StubVariedResearchAgent)
    assert isinstance(agent.base_agent, StubResearchAgent)
    assert agent.kwargs["default_mode"] == "deep"
    assert agent.kwargs["mode_defaults"]["deep"]["top_k"] == 15
    assert "custom" in agent.kwargs["profiles"]

    core.shutdown()


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


def test_load_configuration_manager_settings(tmp_path: Path) -> None:
    config = load_core_configuration(
        overrides={
            "paths": {"repo_root": str(tmp_path)},
            "manager": {
                "plan_retries": "3",
                "task_retry_limit": 2,
                "specialist_blueprints": [
                    {
                        "name": "custom-docs",
                        "kind": "documentation",
                        "agent": "docs",
                        "keywords": "docs,readme",
                        "budget": {"limit": "2", "unit": "rounds"},
                        "research": {"required": "true", "audience": "writers"},
                    }
                ],
            },
        }
    )

    manager_settings = config.manager
    assert manager_settings.plan_retries == 3
    assert manager_settings.task_retry_limit == 2
    assert manager_settings.specialist_blueprints is not None
    assert len(manager_settings.specialist_blueprints) == 1
    blueprint = manager_settings.specialist_blueprints[0]
    assert blueprint["name"] == "custom-docs"
    assert blueprint["agent"] == "docs"
    assert blueprint["keywords"] == ("docs", "readme")
    assert blueprint["budget"]["limit"] == pytest.approx(2.0)
    assert blueprint["research"]["required"] is True
    assert blueprint["research"]["audience"] == "writers"


def test_core_manager_uses_configured_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class StubManager:
        def __init__(
            self,
            *,
            session,
            plan_retries,
            task_retry_limit,
            specialist_blueprints=None,
            repo_context=None,
            test_critic=None,
            research_agent=None,
            dependency_agent=None,
            db_migration_agent=None,
            eval_agent=None,
            security_agent=None,
            doc_agent=None,
            memory_router=None,
            memory_facade=None,
            mcp_registry=None,
            status_callback=None,
            **_: Any,
        ) -> None:
            captured.update(
                {
                    "plan_retries": plan_retries,
                    "task_retry_limit": task_retry_limit,
                    "specialist_blueprints": tuple(specialist_blueprints or ()),
                    "status_callback": status_callback,
                }
            )
            self.session = session
            self.repo_context = repo_context
            self.test_critic = test_critic
            self.research_agent = research_agent
            self._dependency_agent = dependency_agent
            self._db_migration_agent = db_migration_agent
            self.eval_agent = eval_agent
            self._security_agent = security_agent
            self._doc_agent = doc_agent
            self._integrations_agent = None
            self.memory_router = memory_router
            self.memory_facade = memory_facade
            self.mcp_registry = mcp_registry
            self._attached = {
                "research": None,
                "repo": None,
                "dependency": None,
                "critic": None,
                "eval": None,
            }

        def attach_research_agent(self, agent: Any) -> None:
            self._attached["research"] = agent

        def attach_repo_context(self, agent: Any) -> None:
            self._attached["repo"] = agent

        def attach_dependency_agent(self, agent: Any) -> None:
            self._attached["dependency"] = agent

        def attach_test_critic(self, agent: Any) -> None:
            self._attached["critic"] = agent

        def attach_eval_agent(self, agent: Any) -> None:
            self._attached["eval"] = agent

    monkeypatch.setattr("core.ManagerAgent", StubManager)

    config = load_core_configuration(
        overrides={
            "paths": {"repo_root": str(tmp_path)},
            "manager": {
                "plan_retries": 4,
                "task_retry_limit": "7",
                "specialist_blueprints": [
                    {
                        "name": "regression-suite",
                        "kind": "eval",
                        "agent": "evaluation",
                        "keywords": ["regression", "benchmark"],
                        "budget": {"limit": 5, "unit": "rounds"},
                    }
                ],
            },
            "agents": {"eval": True},
        }
    )

    core = AutoCoderCore(config=config)
    manager = core.build_manager()

    assert captured["plan_retries"] == 4
    assert captured["task_retry_limit"] == 7
    assert captured["specialist_blueprints"]
    custom_blueprint = captured["specialist_blueprints"][0]
    assert custom_blueprint["name"] == "regression-suite"
    assert custom_blueprint["budget"]["limit"] == pytest.approx(5.0)
    assert custom_blueprint["keywords"] == ("regression", "benchmark")
    assert manager._attached["eval"] is core._eval_agent

    core.shutdown()
