from __future__ import annotations

from typing import Any, Mapping, Sequence

from agents.research import ResearchAgent


class DummyPlaywrightClient:
    def is_available(self) -> bool:
        return False


class DummyRag:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int, int, bool, float]] = []
        self.ddg_calls: list[tuple[str, int]] = []
        self._playwright_client = DummyPlaywrightClient()

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        max_search_results: int = 20,
        allow_rewrite: bool = True,
        alpha: float = 0.6,
    ) -> list[Mapping[str, Any]]:
        self.search_calls.append((query, top_k, max_search_results, allow_rewrite, alpha))
        return [
            {"path": "https://example.com/page#section", "text": "Line one\nLine two", "score": 0.9},
            {"path": "https://example.com/page", "text": "Duplicate", "score": 0.8},
        ]

    def _search_ddg(self, query: str, max_results: int = 10) -> Sequence[Mapping[str, str]]:
        self.ddg_calls.append((query, max_results))
        return [
            {"url": "https://example.com/page", "title": "Example Title", "snippet": "Line one snippet"},
        ]


def _build_agent(dummy: DummyRag) -> ResearchAgent:
    return ResearchAgent(rag_factory=lambda **_: dummy, cache_size=4, cache_top_k=4, max_quote_chars=120)


def test_research_agent_caches_and_deduplicates_results() -> None:
    dummy = DummyRag()
    agent = _build_agent(dummy)

    first = agent.search("Test Query", top_k=3, allow_rewrite=False)
    second = agent.search("  Test   Query  ", top_k=1, allow_rewrite=False)

    assert dummy.search_calls == [("Test Query", 4, 20, False, 0.6)]
    assert dummy.ddg_calls == [("Test Query", 20)]
    assert len(first.snippets) == 1
    assert len(second.snippets) == 1
    snippet = second.snippets[0]
    assert snippet.url == "https://example.com/page#section"
    assert snippet.citation == "[1](https://example.com/page#section)"
    assert "\n" not in snippet.quote
    assert snippet.quote.startswith("Line one")


def test_research_agent_respects_force_refresh_and_sanitizes_title() -> None:
    dummy = DummyRag()
    agent = _build_agent(dummy)

    result = agent.search("Another", top_k=2, allow_rewrite=False)
    assert result.snippets[0].title == "Example Title"

    refreshed = agent.search("Another", top_k=2, allow_rewrite=False, force_refresh=True)
    assert len(dummy.search_calls) == 2
    assert refreshed.snippets[0].quote == result.snippets[0].quote
