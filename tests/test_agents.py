import sys
import types

import pytest


# Seed a minimal lmstudio stub so that importing ``agents`` succeeds before
# pytest fixtures have a chance to run.
_bootstrap_stub = types.ModuleType("lmstudio")


class _BootstrapChat:
    def __init__(self, *args, **kwargs):  # pragma: no cover - bootstrap only
        self.messages = []

    @classmethod
    def from_history(cls, history):  # pragma: no cover - bootstrap only
        chat = cls()
        chat.messages = history if isinstance(history, list) else []
        return chat


_bootstrap_stub.Chat = _BootstrapChat


class _BootstrapToolFunctionDef:
    def __init__(self, name, description="", parameters=None, implementation=None):  # pragma: no cover - bootstrap only
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.implementation = implementation


_bootstrap_stub.ToolFunctionDef = _BootstrapToolFunctionDef


def _bootstrap_llm(*args, **kwargs):  # pragma: no cover - bootstrap only
    raise RuntimeError("lmstudio stub should be provided by tests")


_bootstrap_stub.llm = _bootstrap_llm
sys.modules.setdefault("lmstudio", _bootstrap_stub)

_psutil_stub = types.ModuleType("psutil")


class _BootstrapProcess:  # pragma: no cover - bootstrap only
    def __init__(self, pid=None):
        self.pid = pid or 0


_psutil_stub.Process = _BootstrapProcess
sys.modules.setdefault("psutil", _psutil_stub)

from agents import (  # noqa: E402  # import after bootstrap stub injection
    AgentBuilder,
    clear_default_toolsets,
    create_agent,
    list_default_toolsets,
    register_default_toolset,
)
from tooling import ToolRegistry


class _FakeTool:
    """Test helper used to emulate a tool callable."""

    def __init__(self, name):
        self.name = name
        self.__name__ = name

    def __call__(self, value: int) -> int:
        """Increment a value for testing."""

        return value + 1


class _FakeModel:
    def __init__(self):
        self.act_calls = []
        self.respond_calls = []
        self.invalid_tool_error: Exception | None = None

    def respond(self, chat_input, **kwargs):
        self.respond_calls.append((chat_input, kwargs))
        return types.SimpleNamespace(message=types.SimpleNamespace(content="respond"))

    def respond_stream(self, chat_input, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def act(self, chat_input, tools, **kwargs):
        self.act_calls.append((chat_input, tools, kwargs))
        tool_name = "demo"
        if tools:
            tool_name = tools[0]["function"]["name"]
        if self.invalid_tool_error is not None:
            handler = kwargs.get("handle_invalid_tool_request")
            if handler is None:
                raise self.invalid_tool_error
            handler(self.invalid_tool_error, {"id": "call-1", "name": tool_name})
        tool_call = {"id": "call-1", "name": tool_name, "arguments": {}}
        tool_result = {
            "role": "tool",
            "content": "ok",
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        }
        if cb := kwargs.get("on_tool_call"):
            cb(tool_call)
        if cb := kwargs.get("on_tool_result"):
            cb(tool_result)
        if cb := kwargs.get("on_message"):
            cb({"role": "assistant", "content": "done"})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "parsed": {"status": "ok"},
                    }
                }
            ],
            "tool_calls": [tool_call],
            "tool_results": [tool_result],
        }


@pytest.fixture(autouse=True)
def lmstudio_stub(monkeypatch):
    clear_default_toolsets()

    stub = types.ModuleType("lmstudio")

    class Chat:
        def __init__(self, system_prompt: str | None = None):
            self.system_prompt = system_prompt
            self.messages = []
            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})

        @classmethod
        def from_history(cls, history):
            chat = cls(None)
            if isinstance(history, dict):
                chat.messages.extend(history.get("messages", []))
            elif isinstance(history, (list, tuple)):
                chat.messages.extend(history)
            elif isinstance(history, str):
                chat.messages.append({"role": "system", "content": history})
            return chat

        def add_user_message(self, content: str) -> None:
            self.messages.append({"role": "user", "content": content})

        def add_assistant_message(self, content: str) -> None:
            self.messages.append({"role": "assistant", "content": content})

        def append(self, message):
            self.messages.append(message)

        def add_tool_message(self, content: str, *, name=None, tool_call_id=None):
            message = {"role": "tool", "content": content}
            if name:
                message["name"] = name
            if tool_call_id:
                message["tool_call_id"] = tool_call_id
            self.messages.append(message)

    class ToolFunctionDef:
        def __init__(self, name, description="", parameters=None, implementation=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
            self.implementation = implementation

    def llm(name=None, **kwargs):
        return _FakeModel()

    stub.Chat = Chat
    stub.ToolFunctionDef = ToolFunctionDef
    stub.llm = llm

    monkeypatch.setitem(sys.modules, "lmstudio", stub)
    monkeypatch.setattr("chat.lms", stub, raising=False)
    monkeypatch.setattr("tooling.ToolFunctionDef", stub.ToolFunctionDef, raising=False)
    yield
    monkeypatch.delitem(sys.modules, "lmstudio", raising=False)


def test_register_default_toolset_and_create_agent():
    tool = _FakeTool("increment")
    register_default_toolset("math", [tool])

    agent = create_agent(system_prompt="Assistant", toolsets=["math"])
    assert list_default_toolsets() == ["math"]

    text, result = agent.act("hi", schema={"type": "object", "properties": {"status": {"type": "string"}}})
    assert text == "done"
    assert result.raw_response["choices"][0]["message"]["parsed"] == {"status": "ok"}

    model = agent.chat_session.model
    assert model.act_calls, "Expected model.act to be called"
    _, tools, kwargs = model.act_calls[-1]
    assert tools[0]["function"]["name"] == "increment"
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


def test_agent_builder_with_custom_registry():
    tool = _FakeTool("increment")
    register_default_toolset("math", [tool])

    registry = ToolRegistry()
    builder = AgentBuilder(system_prompt="Builder")
    agent = (
        builder.using_registry(registry)
        .with_toolsets("math")
        .with_tool_names("increment")
        .build()
    )

    agent.act("ping", response_format={"type": "json_object"})
    model = agent.chat_session.model
    assert model.act_calls, "Builder agent should trigger act()"
    _, tools, kwargs = model.act_calls[-1]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "increment"
    assert "response_format" in kwargs


def test_invalid_tool_error_path_invokes_handler():
    tool = _FakeTool("increment")
    register_default_toolset("math", [tool])
    agent = create_agent(toolsets=["math"])
    model = agent.chat_session.model
    model.invalid_tool_error = RuntimeError("bad tool")

    captured = {}

    def handler(exc, request):
        captured["error"] = exc
        captured["request"] = request
        return "fallback"

    agent.act("ping", handle_invalid_tool_request=handler)
    assert captured["error"].args[0] == "bad tool"
    assert captured["request"]["name"] == "increment"
