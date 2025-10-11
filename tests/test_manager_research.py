from __future__ import annotations

from typing import Any, Mapping

import pytest

from agents.manager import ManagerAgent
from agents.research import ResearchResult, ResearchSnippet
from internal.structures import StructuredResponse
from session import AgentRound


class DummySession:
    def __init__(self) -> None:
        self.rounds: list[AgentRound] = []
        self._on_round_start = None
        self._on_round_end = None
        self.next_response_text = "done"
        self.last_metadata: Mapping[str, Any] | None = None

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end

    def act(
        self,
        user_message: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ):
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
        record_metadata = dict(metadata) if metadata else None
        round_record = AgentRound(
            index=index,
            user_message=user_message,
            response_text=self.next_response_text,
            result=structured,
            transcript=[],
            metadata=record_metadata,
            messages=[],
            tool_history={"calls": [], "results": []},
        )
        self.last_metadata = round_record.metadata
        self.rounds.append(round_record)
        if self._on_round_end is not None:
            self._on_round_end(round_record)
        return self.next_response_text, structured


class StubResearchAgent:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int, str | None]] = []
        snippet = ResearchSnippet(
            url="https://example.com/doc",
            title="Example",
            quote="Example quote",
            citation="[1](https://example.com/doc)",
        )
        self._result = ResearchResult(query="", snippets=(snippet,))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_search_results: int = 20,
        allow_rewrite: bool = True,
        audience: str | None = None,
        force_refresh: bool = False,
        alpha: float = 0.6,
    ) -> ResearchResult:
        self.queries.append((query, top_k, audience))
        return ResearchResult(query=query, snippets=self._result.snippets)


def test_manager_routes_research_into_metadata() -> None:
    session = DummySession()
    research = StubResearchAgent()

    def plan_builder(_: str, metadata: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
        return [
            {
                "name": "task-1",
                "prompt": "do work",
                "research": {"queries": ["python testing"], "audience": "coder"},
            }
        ]

    manager = ManagerAgent(
        session=session,
        plan_builder=plan_builder,
        research_agent=research,
        external_browsing_default=False,
    )

    result = manager.run("do work", metadata={"external_browsing": True})

    assert research.queries[0][0] == "python testing"
    assert "external_evidence" in session.last_metadata
    evidence = session.last_metadata["external_evidence"]
    assert evidence["coder"][0]["url"] == "https://example.com/doc"
    assert result.evidence["coder"][0].url == "https://example.com/doc"


def test_manager_respects_external_browsing_toggle() -> None:
    session = DummySession()
    research = StubResearchAgent()
    manager = ManagerAgent(session=session, research_agent=research, external_browsing_default=False)

    manager.run("task", metadata={"external_browsing": True})
    manager.request_research("first")

    manager.run("task", metadata={"external_browsing": False})
    with pytest.raises(RuntimeError):
        manager.request_research("second")
