import importlib
import sys
import types
from typing import Any

import pytest

from internal.schemas import SchemaError


def _create_lmstudio_stub():
    stub = types.ModuleType("lmstudio")

    class ToolFunctionDef:
        def __init__(self, name: str, description: str = "", parameters=None, implementation=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
            self.implementation = implementation

    class Chat:
        def __init__(self, system_prompt: str | None = None):
            self.system_prompt = system_prompt
            self.messages: list[dict[str, str]] = []
            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})

        @classmethod
        def from_history(cls, history):
            chat = cls(None)
            if isinstance(history, str):
                chat.messages.append({"role": "system", "content": history})
            elif isinstance(history, dict):
                chat.messages.extend(history.get("messages", []))
            else:
                chat.messages.extend(history)
            return chat

        def add_user_message(self, content: str) -> None:
            self.messages.append({"role": "user", "content": content})

        def add_assistant_message(self, content: str) -> None:
            self.messages.append({"role": "assistant", "content": content})

        def append(self, message):
            self.messages.append(message)

        def add_tool_message(self, content: str, *, name: str | None = None, tool_call_id: str | None = None):
            message = {"role": "tool", "content": content}
            if name is not None:
                message["name"] = name
            if tool_call_id is not None:
                message["tool_call_id"] = tool_call_id
            self.messages.append(message)

    class FakeStream:
        def __init__(self, fragments):
            self._fragments = fragments
            self._index = 0

        def __iter__(self):
            return iter(self._fragments)

        def result(self):
            return self._fragments[-1] if self._fragments else None

        def wait_for_result(self):
            return self.result()

    class FakeModel:
        def __init__(self):
            self.respond_calls = []
            self.respond_stream_calls = []
            self.act_calls = []
            self.next_act_response: Any | None = None

        def respond(self, chat_input, **kwargs):
            self.respond_calls.append((chat_input, kwargs))
            if callback := kwargs.get("on_message"):
                callback({"role": "assistant", "content": "respond"})
            return types.SimpleNamespace(message=types.SimpleNamespace(content="respond"))

        def respond_stream(self, chat_input, **kwargs):
            self.respond_stream_calls.append((chat_input, kwargs))
            fragments = [types.SimpleNamespace(content="fragment1"), types.SimpleNamespace(content="fragment2")]
            return FakeStream(fragments)

        def act(self, chat_input, tools, **kwargs):
            self.act_calls.append((chat_input, tools, kwargs))
            tool_call = {"id": "call-1", "name": "demo", "arguments": {"value": 1}}
            if tool_call_cb := kwargs.get("on_tool_call"):
                tool_call_cb(tool_call)
            tool_result = {
                "role": "tool",
                "content": "tool output",
                "name": "demo",
                "tool_call_id": "call-1",
            }
            if tool_result_cb := kwargs.get("on_tool_result"):
                tool_result_cb(tool_result)
            if callback := kwargs.get("on_message"):
                callback({"role": "assistant", "content": "act result"})
            if self.next_act_response is not None:
                return self.next_act_response
            response: dict[str, Any] = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "act result",
                            "parsed": {"value": 1},
                        }
                    }
                ],
                "tool_calls": [tool_call],
                "tool_results": [tool_result],
            }
            return response

    def llm(name=None, **kwargs):
        return FakeModel()

    stub.ToolFunctionDef = ToolFunctionDef
    stub.Chat = Chat
    stub.llm = llm
    stub.FakeModel = FakeModel
    return stub


@pytest.fixture()
def lmstudio_env(monkeypatch):
    stub = _create_lmstudio_stub()
    monkeypatch.setitem(sys.modules, "lmstudio", stub)

    psutil_stub = types.ModuleType("psutil")

    class _FakePsutilProcess:
        def __init__(self, pid):
            self.pid = pid

        def kill(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def is_running(self):
            return False

        def as_dict(self, attrs=None):
            return {attr: None for attr in (attrs or [])}

        def children(self, recursive=False):
            return []

    def _process_iter(attrs=None):
        return iter(())

    psutil_stub.Process = _FakePsutilProcess
    psutil_stub.NoSuchProcess = RuntimeError
    psutil_stub.TimeoutExpired = RuntimeError
    psutil_stub.process_iter = _process_iter
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    for module_name in [
        "tooling",
        "chat",
        "session",
        "internal.tools.shell",
        "internal.tools.planner",
        "internal.tools.patch",
        "internal.tools.file",
        "internal.tools.process",
        "internal.tools.git",
    ]:
        sys.modules.pop(module_name, None)

    tooling = importlib.import_module("tooling")
    chat = importlib.import_module("chat")
    session = importlib.import_module("session")
    return types.SimpleNamespace(stub=stub, tooling=tooling, chat=chat, session=session)


def test_get_tools_by_name_deduplicates(lmstudio_env):
    tooling = lmstudio_env.tooling
    all_tools = tooling.get_all_tools()
    names = [all_tools[0].name, all_tools[0].name]
    resolved = tooling.get_tools(names)
    assert [tool.name for tool in resolved] == [all_tools[0].name]


def test_resolve_tools_combines_sources(lmstudio_env):
    tooling = lmstudio_env.tooling
    first, second = tooling.get_all_tools()[:2]
    resolved = tooling.resolve_tools(tools=[first], tool_names=[second.name, first.name])
    assert resolved == [first, second]


def test_act_requires_tools(lmstudio_env):
    chat = lmstudio_env.chat
    with pytest.raises(ValueError):
        chat.act("hello", model=lmstudio_env.stub.FakeModel())


def test_act_invokes_model_with_resolved_tools(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    model = lmstudio_env.stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    text, result = chat_mod.act("hi", tools=[tool], model=model)
    assert text == "act result"
    assert result.content == "act result"
    assert result.parsed is None
    assert model.act_calls[0][1] == [tool]


def test_chat_session_act_updates_history_and_tools(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    stub = lmstudio_env.stub
    chat_instance = stub.Chat("system")
    model = stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    session = chat_mod.ChatSession(chat=chat_instance, model=model, tools=[tool])
    text, structured = session.act("do something")
    assert text == "act result"
    assert chat_instance.messages[-1]["content"] == "act result"
    assert session.tools == [tool]
    assert structured.parsed is None
    assert model.act_calls[0][1] == [tool]


def test_chat_session_act_requires_tools(lmstudio_env):
    chat_mod = lmstudio_env.chat
    stub = lmstudio_env.stub
    session = chat_mod.ChatSession(chat=stub.Chat("system"), model=stub.FakeModel())
    with pytest.raises(ValueError):
        session.act("hello")


def test_chat_session_create_tracks_system_prompt(lmstudio_env):
    chat_mod = lmstudio_env.chat
    stub = lmstudio_env.stub
    model = stub.FakeModel()
    session = chat_mod.ChatSession.create(system_prompt="base", model=model)
    assert session.system_prompt == "base"
    assert session.chat.messages[0]["content"] == "base"


def test_chat_session_append_tool_response_records_metadata(lmstudio_env):
    chat_mod = lmstudio_env.chat
    stub = lmstudio_env.stub
    session = chat_mod.ChatSession(chat=stub.Chat("system"), model=stub.FakeModel())
    session.append_tool_response("done", name="demo", tool_call_id="call-1")
    assert session.chat.messages[-1] == {
        "role": "tool",
        "content": "done",
        "name": "demo",
        "tool_call_id": "call-1",
    }


def test_agent_session_records_round_metadata(lmstudio_env):
    session_mod = lmstudio_env.session
    tooling = lmstudio_env.tooling
    model = lmstudio_env.stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    agent = session_mod.AgentSession(system_prompt="sys", model=model, tools=[tool])
    text, result = agent.act("hello")
    assert text == "act result"
    assert result.raw_response["tool_calls"][0]["name"] == "demo"
    recorded_round = agent.last_round()
    assert recorded_round is not None
    assert recorded_round.user_message == "hello"
    assert recorded_round.tool_history["results"][0]["role"] == "tool"
    assert agent.transcript[-1]["content"] == "act result"


def test_agent_session_hooks_fire(lmstudio_env):
    session_mod = lmstudio_env.session
    tooling = lmstudio_env.tooling
    model = lmstudio_env.stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    events: list[tuple[str, Any]] = []

    def on_message(message):
        events.append(("message", message))

    def on_tool_call(call):
        events.append(("tool_call", call))

    def on_tool_result(result):
        events.append(("tool_result", result))

    def on_round_start(ctx):
        events.append(("start", ctx["index"]))

    def on_round_end(round_record):
        events.append(("end", round_record.index))

    agent = session_mod.AgentSession(
        system_prompt="sys",
        model=model,
        tools=[tool],
        on_message=on_message,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_round_start=on_round_start,
        on_round_end=on_round_end,
    )
    agent.act("hi there")
    assert ("start", 0) in events
    assert any(event[0] == "message" for event in events)
    assert any(event[0] == "tool_call" for event in events)
    assert any(event[0] == "tool_result" for event in events)
    assert ("end", 0) in events


def test_act_accepts_schema_and_returns_structured(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    model = lmstudio_env.stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    text, result = chat_mod.act(
        "hi",
        tools=[tool],
        model=model,
        schema=schema,
        schema_name="value_schema",
        strict_schema=False,
    )
    assert text == "act result"
    response_format = model.act_calls[0][2]["response_format"]
    assert response_format["json_schema"]["schema"] == schema
    assert response_format["json_schema"]["name"] == "value_schema"
    assert response_format["json_schema"]["strict"] is False
    assert result.parsed == {"value": 1}
    assert result.schema == schema


def test_chat_session_act_supports_response_format_mapping(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    stub = lmstudio_env.stub
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    session = chat_mod.ChatSession.create(
        system_prompt="sys",
        model=stub.FakeModel(),
        tools=[tooling.get_all_tools()[0]],
    )
    response_format = {"schema": schema, "name": "custom", "strict": False}
    _, structured = session.act("hello", response_format=response_format)
    assert structured.parsed == {"value": 1}
    assert structured.schema == schema
    model_call = session.model.act_calls[0]
    assert model_call[2]["response_format"]["json_schema"]["name"] == "custom"
    assert model_call[2]["response_format"]["json_schema"]["strict"] is False


def test_act_raises_schema_error_with_invalid_json(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    model = lmstudio_env.stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    model.next_act_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "not json",
                }
            }
        ],
        "tool_calls": [],
        "tool_results": [],
    }
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    with pytest.raises(SchemaError) as excinfo:
        chat_mod.act("hi", tools=[tool], model=model, schema=schema)
    assert "not json" in str(excinfo.value)
