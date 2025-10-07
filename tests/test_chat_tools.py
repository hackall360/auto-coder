import importlib
import sys
import types

import pytest


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
            if callback := kwargs.get("on_message"):
                callback({"role": "assistant", "content": "act result"})
            return types.SimpleNamespace(message=types.SimpleNamespace(content="act result"))

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
    return types.SimpleNamespace(stub=stub, tooling=tooling, chat=chat)


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
    assert result.message.content == "act result"
    assert model.act_calls[0][1] == [tool]


def test_chat_session_act_updates_history_and_tools(lmstudio_env):
    chat_mod = lmstudio_env.chat
    tooling = lmstudio_env.tooling
    stub = lmstudio_env.stub
    chat_instance = stub.Chat("system")
    model = stub.FakeModel()
    tool = tooling.get_all_tools()[0]
    session = chat_mod.ChatSession(chat=chat_instance, model=model, tools=[tool])
    text, _ = session.act("do something")
    assert text == "act result"
    assert chat_instance.messages[-1]["content"] == "act result"
    assert session.tools == [tool]
    assert model.act_calls[0][1] == [tool]


def test_chat_session_act_requires_tools(lmstudio_env):
    chat_mod = lmstudio_env.chat
    stub = lmstudio_env.stub
    session = chat_mod.ChatSession(chat=stub.Chat("system"), model=stub.FakeModel())
    with pytest.raises(ValueError):
        session.act("hello")
