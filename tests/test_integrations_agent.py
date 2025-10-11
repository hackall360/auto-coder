from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

if "lmstudio" not in sys.modules:
    class _DummyChat:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages: list[Any] = []

        @classmethod
        def from_history(cls, history: Mapping[str, Any] | None = None) -> "_DummyChat":
            instance = cls()
            if isinstance(history, Mapping):
                instance.messages.extend(history.get("messages", []))
            return instance

        def append(self, message: Any) -> None:  # pragma: no cover - not used in tests
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
            self.name = kwargs.get("name")

    sys.modules["lmstudio"] = types.SimpleNamespace(
        Chat=_DummyChat,
        llm=_dummy_llm,
        ToolFunctionDef=ToolFunctionDef,
    )

if "chat" not in sys.modules:
    CallbackMap = Mapping[str, Any]
    sys.modules["chat"] = types.SimpleNamespace(CallbackMap=CallbackMap)

if "psutil" not in sys.modules:
    class _DummyPsutil(types.SimpleNamespace):
        class NoSuchProcess(Exception):
            pass

        class TimeoutExpired(Exception):
            pass

        def Process(self, pid: int):  # pragma: no cover - process inspection unused
            raise self.NoSuchProcess()

        def process_iter(self, attrs: Any):  # pragma: no cover - not used
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

        def register(self, tool: Any, *args: Any, **kwargs: Any) -> ToolSpec:
            spec = tool if isinstance(tool, ToolSpec) else ToolSpec(tool, kwargs.get("name"))
            self._tools[spec.name] = spec
            return spec

        def list(self) -> list[ToolSpec]:  # pragma: no cover - unused
            return list(self._tools.values())

        def clear(self) -> None:  # pragma: no cover - unused
            self._tools.clear()

    def register_default_toolset(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - unused
        return None

    def list_default_toolsets() -> list[str]:  # pragma: no cover - unused
        return []

    def clear_default_toolsets() -> None:  # pragma: no cover - unused
        return None

    sys.modules["tooling"] = types.SimpleNamespace(
        ToolSpec=ToolSpec,
        ToolRegistry=ToolRegistry,
        register_default_toolset=register_default_toolset,
        list_default_toolsets=list_default_toolsets,
        clear_default_toolsets=clear_default_toolsets,
    )

if "session" not in sys.modules:
    class _DummyAgentSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.rounds: list[Any] = []

        def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:  # pragma: no cover
            return

    class _DummyHook:  # pragma: no cover - documentation only
        pass

    class _DummyAgentRound:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.metadata = kwargs.get("metadata")

        def to_dict(self) -> dict[str, Any]:  # pragma: no cover - unused in tests
            return {"metadata": self.metadata}

    sys.modules["session"] = types.SimpleNamespace(
        AgentSession=_DummyAgentSession,
        Hook=_DummyHook,
        AgentRound=_DummyAgentRound,
    )

from agents.integrations import CIJobPlan, IntegrationsAgent
from agents.repo_context import RepoContextAgent
from agents.runner import RunReport


class StubRepoContext:
    """Minimal stub implementing the subset of RepoContextAgent APIs we exercise."""

    def __init__(self, root: Path) -> None:
        self.repo_root = str(root)

    def detect_ci_systems(self) -> dict[str, tuple[str, ...]]:
        return {}

    def read_file(self, path: str, *, encoding: str = "utf-8") -> str | None:
        file_path = Path(self.repo_root) / path
        if file_path.exists():
            return file_path.read_text(encoding=encoding)
        return None

    def update_file_if_changed(self, path: str, content: str, *, encoding: str = "utf-8") -> bool:
        file_path = Path(self.repo_root) / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and file_path.read_text(encoding=encoding) == content:
            return False
        file_path.write_text(content, encoding=encoding)
        return True

    def current_branch(self) -> str:
        return "main"


class DummyRunner:
    """Simple runner stub capturing invocations for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[str] | str, Mapping[str, Any]]] = []

    def run_shell(self, command: Sequence[str] | str, **kwargs: Any) -> RunReport:
        self.calls.append((command, kwargs))
        command_display = " ".join(command) if isinstance(command, (list, tuple)) else str(command)
        return RunReport(
            identifier=len(self.calls),
            runner="shell",
            command=command,
            command_display=command_display,
            workdir=str(kwargs.get("workdir") or Path(self.calls[-1][1].get("workdir", "."))),
            env=kwargs.get("env") or {},
            status="success",
            ok=True,
            exit_code=0,
            pid=1234,
            stdout="",
            stderr="",
            combined_output=kwargs.get("combine_output") and "" or None,
            error=None,
            start_time=0.0,
            end_time=0.0,
            duration=0.0,
            artifacts=tuple(kwargs.get("artifacts") or ()),
            metadata=kwargs.get("metadata") or {},
            raw={},
        )


def test_repo_context_detects_ci_systems(tmp_path) -> None:
    agent = RepoContextAgent.__new__(RepoContextAgent)
    agent.repo_root = str(tmp_path)
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "ci.yaml").write_text("name: CI\n")
    (tmp_path / ".gitlab-ci.yml").write_text("stages: []\n")

    providers = RepoContextAgent.detect_ci_systems(agent)

    assert providers["github_actions"] == (".github/workflows/ci.yaml",)
    assert providers["gitlab_ci"] == (".gitlab-ci.yml",)


def test_integrations_agent_renders_and_applies_pipeline(tmp_path) -> None:
    repo = StubRepoContext(tmp_path)
    agent = IntegrationsAgent(repo_context=repo, runner=DummyRunner())
    plan = CIJobPlan(
        provider="github_actions",
        name="ci",
        path=".github/workflows/ci.yml",
        template="""name: ${job}\non:\n  push:\n    branches: [${branch}]\n""",
        variables={"job": "CI", "branch": "main"},
    )

    rendered = agent.render_pipeline(plan)

    assert rendered.rendered is not None
    assert "${" not in rendered.rendered

    result = agent.apply_pipeline(rendered)

    assert result.changed is True
    assert (tmp_path / ".github" / "workflows" / "ci.yml").read_text() == rendered.rendered

    second = agent.apply_pipeline(rendered)
    assert second.changed is False


def test_integrations_agent_runs_container_build(tmp_path) -> None:
    repo = StubRepoContext(tmp_path)
    runner = DummyRunner()
    agent = IntegrationsAgent(repo_context=repo, runner=runner)

    report = agent.run_container_build(
        "example:latest",
        context=".",
        dockerfile="Dockerfile",
        build_args={"BUILD": "1"},
        extra_args=["--no-cache"],
        workdir=str(tmp_path),
        env={"CI": "true"},
    )

    assert runner.calls, "run_shell should have been invoked"
    command, kwargs = runner.calls[0]
    assert command[:4] == ["docker", "build", "-t", "example:latest"]
    assert "--no-cache" in command
    assert command[-1] == "."
    assert kwargs["combine_output"] is True
    assert kwargs["env"] == {"CI": "true"}
    assert report.ok


def test_integrations_agent_prepares_release_metadata(tmp_path) -> None:
    repo = StubRepoContext(tmp_path)
    agent = IntegrationsAgent(repo_context=repo, runner=DummyRunner())

    release = agent.prepare_release_metadata("1.2.3", notes="Ready", artifacts=["dist/app.whl"])

    assert release.tag == "v1.2.3"
    assert release.branch == "main"
    assert release.to_dict()["artifacts"] == ["dist/app.whl"]
