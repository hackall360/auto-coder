from __future__ import annotations

import sys
import types
from typing import Any, Mapping

import pytest

class _StubModel:
    def respond(self, *_, **__):
        return {"choices": [{"message": {"content": "stub"}}]}

    def respond_stream(self, *_, **__):
        return iter(())


class _StubChat:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[Mapping[str, Any]] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    @classmethod
    def from_history(cls, history: Any) -> "_StubChat":
        instance = cls()
        if isinstance(history, Mapping):
            messages = history.get("messages", [])
            if isinstance(messages, list):
                instance.messages = list(messages)
        elif isinstance(history, str):
            instance.messages = [{"role": "user", "content": history}]
        return instance

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def append(self, message: Mapping[str, Any]) -> None:
        self.messages.append(dict(message))


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=types.SimpleNamespace,
    ),
)


class _StubProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def kill(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def is_running(self) -> bool:
        return False

    def as_dict(self, attrs: list[str] | None = None) -> dict[str, Any]:
        del attrs
        return {"pid": self.pid}

    def children(self, recursive: bool = False) -> list["_StubProcess"]:
        del recursive
        return []


sys.modules.setdefault(
    "psutil",
    types.SimpleNamespace(
        Process=_StubProcess,
        NoSuchProcess=RuntimeError,
        TimeoutExpired=TimeoutError,
        process_iter=lambda *_, **__: [],
    ),
)

from agents.manager import ManagerAgent
from memory import InMemoryMemoryStore, MemoryFacade, MemoryRouter


class _ReplayableChat:
    def __init__(self) -> None:
        self.messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def append(self, message: Mapping[str, Any]) -> None:
        self.messages.append(dict(message))


class _ReplayableChatSession:
    def __init__(self) -> None:
        self.chat = _ReplayableChat()


class ReplayableSession:
    def __init__(self) -> None:
        self.chat_session = _ReplayableChatSession()
        self.rounds: list[Any] = []

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        del on_round_start, on_round_end


@pytest.fixture()
def memory_facade() -> MemoryFacade:
    short_store = InMemoryMemoryStore()
    long_store = InMemoryMemoryStore()
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: short_store,
            MemoryRouter.LONG_TERM: long_store,
        }
    )
    facade = MemoryFacade(router, combined_scope=MemoryRouter.SHORT_TERM)
    return facade


def test_manager_replays_conversation_history(memory_facade: MemoryFacade) -> None:
    session_id = "replay-session"

    memory_facade.add(
        "Prior question",
        scope=MemoryRouter.SHORT_TERM,
        tags=("conversation", "user"),
        attributes={"role": "user", "round_index": 0},
        session_id=session_id,
    )
    memory_facade.add(
        "Prior answer",
        scope=MemoryRouter.SHORT_TERM,
        tags=("conversation", "assistant"),
        attributes={"role": "assistant", "round_index": 0},
        session_id=session_id,
    )
    memory_facade.add(
        "Follow-up question",
        scope=MemoryRouter.LONG_TERM,
        tags=("conversation", "user"),
        attributes={"role": "user", "round_index": 1},
        session_id=session_id,
    )
    memory_facade.add(
        "Follow-up answer",
        scope=MemoryRouter.LONG_TERM,
        tags=("conversation", "assistant"),
        attributes={"role": "assistant", "round_index": 1},
        session_id=session_id,
    )

    session = ReplayableSession()

    ManagerAgent(
        session=session,
        memory_facade=memory_facade,
        memory_router=memory_facade.router,
        session_id=session_id,
    )

    messages = session.chat_session.chat.messages
    assert messages, "Expected chat history to contain seeded messages"

    conversation = [
        (message["role"], message["content"])
        for message in messages
        if message.get("role") != "system"
    ]

    assert conversation == [
        ("user", "Prior question"),
        ("assistant", "Prior answer"),
        ("user", "Follow-up question"),
        ("assistant", "Follow-up answer"),
    ]
