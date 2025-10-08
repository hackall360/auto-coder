from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any, Mapping

import sys

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
        implementation: Any = None,
        **_: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters or {})
        self.implementation = implementation


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_StubToolFunctionDef,
    ),
)

from agents.manager import ManagerAgent, ManagerStatusUpdate
from internal.structures import StructuredResponse
from session import AgentRound


class DummySession:
    """Lightweight stub mimicking :class:`AgentSession` for manager tests."""

    def __init__(self) -> None:
        self.rounds: list[AgentRound] = []
        self._on_round_start = None
        self._on_round_end = None
        self.next_response_text = "done"

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end

    def act(self, user_message: str | None = None, *, metadata: Mapping[str, Any] | None = None, **_: Any):
        index = len(self.rounds)
        if self._on_round_start is not None:
            self._on_round_start(
                {
                    "index": index,
                    "user_message": user_message,
                    "session": self,
                    "metadata": dict(metadata) if metadata else None,
                }
            )
        structured = StructuredResponse(
            raw_response={"choices": [{"message": {"content": self.next_response_text, "parsed": {"ok": True}}}]},
            content=self.next_response_text,
            parsed={"ok": True},
            schema=None,
            structured=False,
        )
        round_record = AgentRound(
            index=index,
            user_message=user_message,
            response_text=self.next_response_text,
            result=structured,
            transcript=[],
            messages=[],
            tool_history={"calls": [], "results": []},
            metadata=dict(metadata) if metadata else None,
        )
        self.rounds.append(round_record)
        if self._on_round_end is not None:
            self._on_round_end(round_record)
        return self.next_response_text, structured


@pytest.mark.parametrize("message", ["solve this", ""])
def test_manager_tracks_budget_progress(message: str) -> None:
    session = DummySession()
    captured: list[ManagerStatusUpdate] = []
    manager = ManagerAgent(session=session, status_callback=captured.append)

    result = manager.run(message)

    assert session.next_response_text in result.response_text
    assert result.plan, "planner should produce at least one task"
    first_task = result.plan[0]
    assert "name" in first_task
    assert "task-1" in result.budgets
    budget = result.budgets["task-1"]
    assert pytest.approx(budget.consumed) == 1.0
    assert budget.remaining in (0.0, None)
    assert result.rounds
    last_round = result.rounds[-1]
    assert last_round.metadata is not None
    assert last_round.metadata["budget"]["consumed"] == pytest.approx(1.0)
    assert any(update.kind == "progress" for update in captured)
    assert result.structured_response is not None
    parsed = result.structured_response.parsed
    assert isinstance(parsed, Mapping)
    assert "task-1" in parsed.get("tasks", {})


@dataclass
class StubSnippet:
    url: str
    title: str
    quote: str
    citation: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "quote": self.quote,
            "citation": self.citation,
        }


class StubDocResult:
    def __init__(self, payload: Mapping[str, Any], evidence: tuple[StubSnippet, ...]):
        self._payload = dict(payload)
        self.evidence = evidence

    def to_dict(self) -> Mapping[str, Any]:
        return dict(self._payload)


class StubDocAgent:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def draft_updates(self, **kwargs: Any) -> StubDocResult:
        self.calls.append(kwargs)
        snippet = StubSnippet(
            url="https://example.com/docs",
            title="Example",
            quote="Sample evidence",
            citation="[1]",
        )
        return StubDocResult({"doc": "Doc summary"}, (snippet,))

    @staticmethod
    def to_structured_response(result: StubDocResult) -> StructuredResponse:
        payload = result.to_dict()
        return StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Doc summary",
                            "parsed": payload,
                        }
                    }
                ]
            },
            content="Doc summary",
            parsed=payload,
            schema={"name": "DocStub"},
            structured=True,
        )


@dataclass
class StubSecurityResult:
    summary: str | None = None
    blocked: bool = False

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "reports": [],
            "threshold": None,
            "blocked": self.blocked,
            "highest_severity": None,
            "summary": self.summary,
            "artifacts": [],
        }


class StubSecurityAgent:
    DEFAULT_SEQUENCE = ("dependencies",)

    def __init__(self, result: StubSecurityResult) -> None:
        self._result = result

    def run_scans(self, **_: Any) -> StubSecurityResult:
        return self._result

    @staticmethod
    def format_result(result: StubSecurityResult) -> str:
        return result.summary or "Security summary"


def multi_agent_plan(message: str, *, metadata: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
    del metadata
    return [
        {
            "name": "documentation-updates",
            "description": "Draft docs",
            "prompt": message,
            "kind": "documentation",
            "budget": {"limit": 2.0, "unit": "rounds"},
            "metadata": {"kind": "documentation"},
            "research": {"required": False},
        },
        {
            "name": "respond",
            "description": "Answer the question",
            "prompt": message,
            "budget": {"limit": 1.0, "unit": "rounds"},
        },
    ]


def blocking_security_plan(message: str, *, metadata: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
    del metadata
    return [
        {
            "name": "security-audit",
            "description": "Run security checks",
            "prompt": message,
            "kind": "security",
            "budget": {"limit": 1.0, "unit": "rounds"},
            "metadata": {"kind": "security"},
        },
        {
            "name": "respond",
            "description": "Fallback response",
            "prompt": message,
            "budget": {"limit": 1.0, "unit": "rounds"},
        },
    ]


def test_manager_merges_specialist_outputs() -> None:
    session = DummySession()
    captured: list[ManagerStatusUpdate] = []
    doc_agent = StubDocAgent()
    manager = ManagerAgent(
        session=session,
        status_callback=captured.append,
        plan_builder=multi_agent_plan,
        doc_agent=doc_agent,
    )

    result = manager.run("please update the docs")

    assert result.plan and [task["name"] for task in result.plan] == [
        "documentation-updates",
        "respond",
    ]
    assert result.response_text.startswith("documentation-updates:")
    assert "Doc summary" in result.response_text
    assert session.next_response_text in result.response_text
    assert result.structured_response is not None
    parsed_tasks = result.structured_response.parsed.get("tasks", {})
    assert "documentation-updates" in parsed_tasks
    assert parsed_tasks["documentation-updates"]["structured_output"]["parsed"] == {
        "doc": "Doc summary"
    }
    assert result.budgets["documentation-updates"].consumed == pytest.approx(1.0)
    assert result.budgets["respond"].consumed == pytest.approx(1.0)
    task_output_updates = {update.task for update in captured if update.kind == "task_output"}
    assert {"documentation-updates", "respond"}.issubset(task_output_updates)
    evidence = result.evidence.get("documentation-updates")
    assert evidence is not None and evidence[0].url == "https://example.com/docs"


def test_manager_blocks_when_security_fails() -> None:
    session = DummySession()
    captured: list[ManagerStatusUpdate] = []
    security_result = StubSecurityResult(blocked=True, summary="Security blocked")
    manager = ManagerAgent(
        session=session,
        status_callback=captured.append,
        plan_builder=blocking_security_plan,
        security_agent=StubSecurityAgent(security_result),
    )

    result = manager.run("run security scan")

    assert result.plan and [task["name"] for task in result.plan] == [
        "security-audit",
        "respond",
    ]
    assert result.response_text == "Security blocked"
    assert result.structured_response is not None
    assert result.structured_response.schema == {"name": "SecurityScanResult"}
    assert result.budgets["security-audit"].consumed == pytest.approx(1.0)
    assert result.budgets["respond"].consumed == pytest.approx(0.0)
    assert any(update.kind == "security_blocked" for update in captured)
    task_output_updates = [update for update in captured if update.kind == "task_output"]
    assert any(update.task == "security-audit" for update in task_output_updates)
    assert all(update.task != "respond" for update in task_output_updates)
    assert result.rounds == []
