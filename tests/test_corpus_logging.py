import json
import os
import sys
import types
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping


class _StubModel:
    def respond(self, *_, **__):
        return {"choices": [{"message": {"content": "stub"}}]}


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


class _StubToolFunctionDef:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        implementation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters or {})
        self.implementation = implementation
        self.extra = dict(kwargs)


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_StubToolFunctionDef,
    ),
)

_psutil_stub = types.ModuleType("psutil")
_psutil_stub.Process = lambda pid=None: types.SimpleNamespace(pid=pid or 0)
sys.modules.setdefault("psutil", _psutil_stub)

import pytest

from agents.manager import ManagerAgent
from agents.research import ResearchAgent
from corpus import CorpusManager, set_shared_corpus_manager
from internal.structures import StructuredResponse
from internal.tools import file as file_tools
from memory import InMemoryMemoryStore, MemoryFacade, MemoryQuery, MemoryRouter


@pytest.fixture()
def corpus_environment(tmp_path):
    short_store = InMemoryMemoryStore()
    long_store = InMemoryMemoryStore()
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: short_store,
            MemoryRouter.LONG_TERM: long_store,
            MemoryRouter.COMBINED: short_store,
        }
    )
    facade = MemoryFacade(router, combined_scope=MemoryRouter.SHORT_TERM)
    corpus_manager = CorpusManager(facade)
    set_shared_corpus_manager(corpus_manager)
    yield corpus_manager, router, facade, long_store
    set_shared_corpus_manager(None)


class DummyRag:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def search(self, query: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        self.calls.append((query, kwargs))
        return [
            {
                "path": "https://example.com/article",
                "text": "Example body of text",
                "score": 0.9,
                "offset": 0,
            }
        ]

    def _search_ddg(self, *_: Any, **__: Any) -> list[Mapping[str, Any]]:
        return []


def test_research_agent_logs_corpus_event(corpus_environment):
    corpus_manager, router, facade, long_store = corpus_environment
    agent = ResearchAgent(
        rag_factory=lambda **_: DummyRag(),
        cache_size=2,
        cache_top_k=2,
        max_quote_chars=120,
        corpus_manager=corpus_manager,
    )

    assert hasattr(agent, "_normalise_query"), f"ResearchAgent came from {agent.__class__.__module__}"

    result = agent.search("Example query", top_k=1)
    assert result.snippets

    records = router.long_term.fetch(MemoryQuery(limit=10))
    event_records = [
        record
        for record in records
        if record.metadata.attributes.get("event_type") == "web_search"
    ]
    assert event_records, "expected web_search event to be recorded"
    event = event_records[-1]
    assert event.metadata.attributes.get("category") == "research"
    payload = event.metadata.attributes.get("payload", {})
    assert payload.get("query") == "Example query"


def test_file_write_logs_corpus_event(corpus_environment):
    corpus_manager, router, facade, long_store = corpus_environment
    target = os.path.join(os.getcwd(), "test_generated.txt")
    try:
        file_tools.write_file(target, "hello world")
        records = router.long_term.fetch(MemoryQuery(limit=10))
        file_events = [
            record
            for record in records
            if record.metadata.attributes.get("event_type") == "file_write"
        ]
        assert file_events, "expected file_write event"
        payload = file_events[-1].metadata.attributes.get("payload", {})
        assert payload.get("status") == "written"
        assert payload.get("size") == len("hello world")
    finally:
        if os.path.exists(target):
            os.remove(target)


def test_corpus_manager_applies_deduplication(tmp_path: Path) -> None:
    short_store = InMemoryMemoryStore()
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: short_store,
            MemoryRouter.LONG_TERM: short_store,
            MemoryRouter.COMBINED: short_store,
        }
    )
    facade = MemoryFacade(router)
    manager = CorpusManager(facade, dedup_threshold=0.9)

    manager.record_event(
        source="tester",
        payload={"message": "repeat me"},
        event_type="custom_event",
        session_id="session-1",
    )
    skipped = manager.record_event(
        source="tester",
        payload={"message": "repeat me"},
        event_type="custom_event",
        session_id="session-1",
    )

    records = router.long_term.fetch(MemoryQuery(limit=10))
    assert len(records) == 1
    assert skipped is None


def test_corpus_manager_writes_to_storage(tmp_path: Path) -> None:
    storage = tmp_path / "events.jsonl"
    manager = CorpusManager(
        None,
        enabled=True,
        storage_path=storage,
    )

    result = manager.record_event(
        source="tester",
        payload={"value": 42},
        event_type="metric",
        session_id="session-2",
    )

    assert result is None  # No facade was configured
    contents = storage.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "expected corpus manager to write JSONL entries"
    record = json.loads(contents[-1])
    assert record["event_type"] == "metric"
    assert record["payload"]["value"] == 42


@dataclass
class DummyRound:
    index: int
    user_message: str | None
    response_text: str
    result: StructuredResponse
    transcript: list[Any]
    messages: list[Any]
    tool_history: dict[str, list[Any]]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "user_message": self.user_message,
            "response_text": self.response_text,
            "result": self.result,
            "transcript": list(self.transcript),
            "messages": list(self.messages),
            "tool_history": {key: list(values) for key, values in self.tool_history.items()},
            "metadata": dict(self.metadata or {}),
        }


class DummySession:
    def __init__(self) -> None:
        self.rounds: list[DummyRound] = []
        self._round_start_hooks: list[Any] = []
        self._round_end_hooks: list[Any] = []
        self.next_response_text = "done"

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        if on_round_start is not None:
            self._round_start_hooks.append(on_round_start)
        if on_round_end is not None:
            self._round_end_hooks.append(on_round_end)

    def act(self, user_message: str | None = None, *, metadata: Mapping[str, Any] | None = None, **_: Any):
        index = len(self.rounds)
        for hook in list(self._round_start_hooks):
            hook({"index": index, "user_message": user_message, "session": self, "metadata": dict(metadata or {})})
        structured = StructuredResponse(
            raw_response={"choices": [{"message": {"content": self.next_response_text}}]},
            content=self.next_response_text,
            parsed={"ok": True},
            schema=None,
            structured=False,
        )
        round_record = DummyRound(
            index=index,
            user_message=user_message,
            response_text=self.next_response_text,
            result=structured,
            transcript=[],
            messages=[],
            tool_history={"calls": [], "results": []},
            metadata=dict(metadata or {}),
        )
        self.rounds.append(round_record)
        for hook in list(self._round_end_hooks):
            hook(round_record)
        return self.next_response_text, structured


def test_manager_run_records_assistant_reply(corpus_environment):
    corpus_manager, router, facade, long_store = corpus_environment
    session = DummySession()
    manager = ManagerAgent(session=session, memory_facade=facade, corpus_manager=corpus_manager)

    manager.run("summarise this")

    records = router.long_term.fetch(MemoryQuery(limit=20))
    reply_events = [
        record
        for record in records
        if record.metadata.attributes.get("event_type") == "assistant_reply"
    ]
    assert reply_events, "expected assistant_reply event recorded"
    reply = reply_events[-1]
    assert reply.metadata.attributes.get("category") == "assistant_output"
    payload = reply.metadata.attributes.get("payload", {})
    assert payload.get("status") == "completed"
    assert "response_text" in payload
