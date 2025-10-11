from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys
import types


class _DummyChat:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.messages: list[Any] = []

    @classmethod
    def from_history(cls, history: Mapping[str, Any] | None) -> "_DummyChat":
        instance = cls()
        instance.history = history
        return instance


class _ToolFunctionDef:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _ToolSpec:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, spec: Any | None = None) -> None:
        self._tools[name] = spec

    def clear(self) -> None:
        self._tools.clear()

    def list(self) -> list[str]:
        return list(self._tools)


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *args, **kwargs: object(),
        Chat=_DummyChat,
        ToolFunctionDef=_ToolFunctionDef,
    ),
)

sys.modules.setdefault("psutil", types.SimpleNamespace())
sys.modules.setdefault(
    "tooling",
    types.SimpleNamespace(
        ToolRegistry=_ToolRegistry,
        ToolSpec=_ToolSpec,
        resolve_tools=lambda *args, **kwargs: [],
    ),
)

from agents.doc import DocAgent, DocumentationResult, DocumentationSummary, ReadmeDraft
from agents.manager import ManagerAgent, ManagerStatusUpdate
from agents.repo_context import DiffBundle, DiffFileStat, FileSummary, RepoSearchResult
from agents.research import ResearchResult, ResearchSnippet
from internal.structures import StructuredResponse


class FakeRepoContext:
    """Minimal stub providing the hooks used by :class:`DocAgent`."""

    def __init__(
        self,
        repo_root: Path,
        summaries: Mapping[str, FileSummary],
        search: Mapping[str, Sequence[RepoSearchResult]],
        diff_bundle: DiffBundle | None = None,
    ) -> None:
        self.repo_root = str(repo_root)
        self._summaries = dict(summaries)
        self._search = {key: list(value) for key, value in search.items()}
        self._diff_bundle = diff_bundle or DiffBundle(
            patch="",
            stats=(),
            staged=False,
            include_untracked=False,
            provenance={},
        )

    def focused_diffs(self, *, include_untracked: bool = False, staged: bool = False) -> DiffBundle:
        return self._diff_bundle

    def summarize_ast(self, path: str) -> FileSummary:
        return self._summaries[path]

    def summarize_file(self, path: str) -> FileSummary:
        return self._summaries[path]

    def focused_files(self, query: str, *, top_k: int = 5) -> list[RepoSearchResult]:
        return list(self._search.get(query, [])[:top_k])


class FakeResearchAgent:
    """Captures incoming queries and returns seeded snippets."""

    def __init__(self, responses: Mapping[str, ResearchResult]) -> None:
        self._responses = dict(responses)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 5, **_: Any) -> ResearchResult:
        self.queries.append((query, top_k))
        return self._responses.get(query, ResearchResult(query=query, snippets=()))


def _make_file_summary(path: str, summary: str = "" ) -> FileSummary:
    return FileSummary(
        path=path,
        summary=summary or f"Summary for {path}.",
        language="python" if path.endswith(".py") else "markdown",
        line_count=42,
        size=1024,
        metadata={},
        provenance={},
    )


def test_doc_agent_generates_drafts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Project\n", encoding="utf-8")
    (repo_root / "CHANGELOG.md").write_text("## Unreleased\n", encoding="utf-8")

    summary = _make_file_summary("src/app.py", "Introduces CLI entry point and helpers.")
    diff = DiffBundle(
        patch="--- a/src/app.py\n+++ b/src/app.py\n",
        stats=(DiffFileStat(path="src/app.py", additions=10, deletions=2, status="modified"),),
        staged=False,
        include_untracked=False,
        provenance={"source": "test"},
    )
    search_map = {
        "cli": [RepoSearchResult(path="src/app.py", offset=0, score=0.9, text="def main(): ...", provenance={})]
    }
    repo_context = FakeRepoContext(repo_root, {summary.path: summary}, search_map, diff)

    snippet = ResearchSnippet(
        url="https://example.com/cli",
        title="CLI Overview",
        quote="Document the new CLI commands.",
        citation="[1]",
        score=0.8,
        metadata={},
    )
    research_agent = FakeResearchAgent({
        "cli release": ResearchResult(query="cli release", snippets=(snippet,)),
    })

    agent = DocAgent(repo_context=repo_context, research_agent=research_agent)
    result = agent.draft_updates(
        highlights=["Introduce command line tooling"],
        version="1.2.0",
        walkthrough_topics=["cli"],
        research_queries=["cli release"],
        summary_paths=["src/app.py"],
        include_readme=True,
        include_changelog=True,
    )

    assert result.summary.modules and result.summary.modules[0].path == "src/app.py"
    assert result.readme is not None
    assert "Introduce command line tooling" in result.readme.content
    assert result.changelog is not None and result.changelog.version == "1.2.0"
    assert result.walkthrough and result.walkthrough[0].path == "src/app.py"
    assert result.evidence and result.evidence[0].url == snippet.url
    assert result.diff_bundle is not None and "README.md" in result.diff_bundle.patch
    assert result.artifacts
    for artifact in result.artifacts:
        assert (repo_root / artifact).exists()
    assert ("cli release", 5) in research_agent.queries

    formatted = agent.format_result(result)
    assert "Documentation Update" in formatted

    structured = agent.to_structured_response(result)
    assert structured.structured is True
    assert structured.parsed["summary"]["modules"]


def test_doc_agent_auto_highlights(tmp_path: Path) -> None:
    repo_root = tmp_path / "auto"
    repo_root.mkdir()
    summary = _make_file_summary("src/api.py", "Adds REST handlers and validation.")
    repo_context = FakeRepoContext(repo_root, {summary.path: summary}, {})
    agent = DocAgent(repo_context=repo_context)

    result = agent.draft_updates(
        highlights=None,
        include_changelog=False,
        walkthrough_topics=None,
        summary_paths=["src/api.py"],
    )

    assert result.readme is not None
    assert result.readme.highlights, "auto-generated highlights should populate"
    assert "src/api.py" in result.readme.content
    assert result.diff_bundle is not None


class DummySession:
    """Lightweight stub mimicking :class:`AgentSession` for manager tests."""

    def __init__(self) -> None:
        self.rounds: list[Any] = []

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end

    def act(self, *_: Any, **__: Any) -> tuple[str, StructuredResponse]:
        raise AssertionError("Doc tasks should not call session.act")


@dataclass
class StubDocAgent:
    result: DocumentationResult

    def __post_init__(self) -> None:
        self.called_with: dict[str, Any] | None = None

    def draft_updates(self, **kwargs: Any) -> DocumentationResult:
        self.called_with = dict(kwargs)
        return self.result

    def to_structured_response(self, result: DocumentationResult) -> StructuredResponse:
        payload = result.to_dict()
        return StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "doc summary",
                            "parsed": payload,
                        }
                    }
                ]
            },
            content="doc summary",
            parsed=payload,
            schema={"name": "DocumentationResult"},
            structured=True,
        )

    def format_result(self, _: DocumentationResult) -> str:
        return "doc summary"

    def attach_research_agent(self, *_: Any, **__: Any) -> None:
        return None


def test_manager_routes_documentation_tasks() -> None:
    summary = DocumentationSummary(overview="Overview", modules=(), metadata={})
    doc_result = DocumentationResult(
        summary=summary,
        readme=ReadmeDraft(content="", section="## What's New"),
        changelog=None,
        walkthrough=(),
        artifacts=(".doc_artifacts/readme.md",),
    )
    stub_agent = StubDocAgent(result=doc_result)

    status_updates: list[ManagerStatusUpdate] = []

    manager = ManagerAgent(
        session=DummySession(),
        repo_context=None,
        doc_agent=stub_agent,
        plan_builder=lambda _: [
            {
                "name": "docs",
                "prompt": "docs",
                "metadata": {"kind": "documentation"},
                "documentation": {"highlights": ["Ship docs"]},
            }
        ],
        status_callback=status_updates.append,
    )

    result = manager.run("prepare documentation")

    assert result.response_text == "doc summary"
    assert stub_agent.called_with is not None
    assert stub_agent.called_with["highlights"] == ("Ship docs",)
    assert any(update.kind == "documentation_summary" for update in status_updates)
