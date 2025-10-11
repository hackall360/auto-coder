import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from mcp_tooling import (
    CommandMCPServerConfig,
    CommandServerLifecycle,
    MCPServerRegistry,
    register_mcp_servers,
)
LMSTUDIO_STUB = ModuleType("lmstudio")


class _StubToolFunctionDef:
    def __init__(self, name, description="", parameters=None, implementation=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.implementation = implementation


class _StubChat:
    def __init__(self, system_prompt: str | None = None):  # pragma: no cover - import hook
        self.system_prompt = system_prompt


def _stub_llm(name=None, **kwargs):  # pragma: no cover - import hook
    class _FakeModel:
        def respond(self, *args, **kwargs):
            return None

        def respond_stream(self, *args, **kwargs):
            return []

        def act(self, *args, **kwargs):
            return {}

    return _FakeModel()


LMSTUDIO_STUB.ToolFunctionDef = _StubToolFunctionDef
LMSTUDIO_STUB.Chat = _StubChat
LMSTUDIO_STUB.llm = _stub_llm
sys.modules.setdefault("lmstudio", LMSTUDIO_STUB)

PSUTIL_STUB = ModuleType("psutil")


class _StubPsutilProcess:
    def __init__(self, pid):  # pragma: no cover - import hook
        self.pid = pid

    def kill(self):  # pragma: no cover - import hook
        return None

    def terminate(self):  # pragma: no cover - import hook
        return None

    def wait(self, timeout=None):  # pragma: no cover - import hook
        return 0

    def is_running(self):  # pragma: no cover - import hook
        return False

    def as_dict(self, attrs=None):  # pragma: no cover - import hook
        return {attr: None for attr in (attrs or [])}

    def children(self, recursive=False):  # pragma: no cover - import hook
        return []


def _stub_process_iter(attrs=None):  # pragma: no cover - import hook
    return iter(())


PSUTIL_STUB.Process = _StubPsutilProcess
PSUTIL_STUB.NoSuchProcess = RuntimeError
PSUTIL_STUB.TimeoutExpired = RuntimeError
PSUTIL_STUB.process_iter = _stub_process_iter
sys.modules.setdefault("psutil", PSUTIL_STUB)

from tooling import ToolRegistry, ToolSpec


class _FakeStream:
    def __init__(self, *lines: str) -> None:
        self._lines = list(lines)
        self.closed = False

    def readline(self) -> str:
        if self._lines:
            return self._lines.pop(0)
        # Give the lifecycle loop time to observe the ready flag before finishing.
        time.sleep(0.01)
        return ""

    def close(self) -> None:  # pragma: no cover - exercised via lifecycle shutdown
        self.closed = True


class _FakeProcess:
    def __init__(self, command, **kwargs) -> None:  # noqa: D401 - pytest helper
        self.command = list(command)
        self.kwargs = kwargs
        self.stdout = _FakeStream("READY\n") if kwargs.get("stdout") is not None else None
        self.stderr = _FakeStream("") if kwargs.get("stderr") is not None else None
        self.pid = 4321
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self.signal_sent = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):  # pragma: no cover - defensive
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = 0
        return 0

    def kill(self):
        self.kill_called = True
        self.returncode = -9
        return -9

    def send_signal(self, sig):
        self.signal_sent = sig
        self.returncode = 0
        return 0


@pytest.fixture()
def fake_popen(monkeypatch):
    calls: list[_FakeProcess] = []

    def _launcher(command, **kwargs):
        proc = _FakeProcess(command, **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr("subprocess.Popen", _launcher)
    return SimpleNamespace(calls=calls)


@pytest.fixture()
def agent_builder_cls(monkeypatch):
    stub = LMSTUDIO_STUB
    monkeypatch.setitem(sys.modules, "lmstudio", stub)
    for module_name in ["chat", "session", "agents"]:
        sys.modules.pop(module_name, None)
    import agents

    return agents.AgentBuilder


def test_command_lifecycle_start_and_cleanup(fake_popen):
    config = CommandMCPServerConfig(
        label="demo",
        command=("python", "-m", "demo"),
        ready_pattern="READY",
        ready_timeout=1.0,
    )
    lifecycle = CommandServerLifecycle(config)
    process = lifecycle.start()

    assert fake_popen.calls, "Command launcher should be invoked"
    assert process.pid == 4321
    assert lifecycle.pid == 4321

    lifecycle.shutdown()
    assert fake_popen.calls[0].terminate_called is True
    assert lifecycle.process is None


def test_registry_specs_emit_mcp_payload():
    registry = MCPServerRegistry(
        {
            "remote-search": {
                "type": "remote",
                "url": "https://mcp.example.com/search",
                "allowed_tools": ["search"],
                "headers": {"Authorization": "Bearer demo"},
            }
        }
    )
    tool_registry = ToolRegistry()
    specs = register_mcp_servers(tool_registry, registry.build_specs())
    assert len(specs) == 1
    payload = specs[0].to_payload()
    assert payload["type"] == "mcp"
    assert payload["server_label"] == "remote-search"
    assert payload["server_url"] == "https://mcp.example.com/search"
    assert payload["allowed_tools"] == ["search"]
    assert payload["headers"]["Authorization"] == "Bearer demo"


def test_agent_builder_injects_mcp_specs(monkeypatch, agent_builder_cls):
    registry = ToolRegistry()

    def echo(value: int) -> int:
        """Return the provided integer."""

        return value

    function_spec = registry.register(
        echo,
        name="echo",
        description="Return the provided integer.",
        parameters={"value": int},
    )

    mcp_spec = ToolSpec(
        name="demo-mcp",
        description="Demo MCP",
        parameters={"label": "demo-mcp"},
        implementation=None,
        source={"label": "demo-mcp"},
        tool_type="mcp",
    )

    captured = {}

    def fake_register_mcp_servers(registry_arg, configs, *, replace):
        captured["registry"] = registry_arg
        captured["configs"] = list(configs)
        captured["replace"] = replace
        return [mcp_spec]

    def fake_create_agent(**kwargs):
        captured["kwargs"] = kwargs
        return "session"

    import agents
    import mcp_tooling as mcp_module

    monkeypatch.setattr(mcp_module, "register_mcp_servers", fake_register_mcp_servers)
    monkeypatch.setattr(agents, "create_agent", fake_create_agent)

    AgentBuilder = agent_builder_cls

    builder = AgentBuilder(system_prompt="hi").using_registry(registry)
    session = (
        builder.with_tools(function_spec)
        .with_mcp_servers({"label": "demo-mcp"})
        .build()
    )

    assert session == "session"
    assert captured["registry"] is registry
    assert captured["configs"] == [{"label": "demo-mcp"}]
    assert captured["replace"] is True
    tools = captured["kwargs"]["tools"]
    assert tools == [function_spec, mcp_spec]
