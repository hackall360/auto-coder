"""Documentation drafting helpers for README, changelog, and walkthrough updates."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from internal.structures import StructuredResponse

from .repo_context import DiffBundle, DiffFileStat, FileSummary, RepoContextAgent, RepoSearchResult
from .research import ResearchAgent, ResearchResult, ResearchSnippet, VariedResearchAgent

__all__ = [
    "DocumentationSummary",
    "ReadmeDraft",
    "ChangelogDraft",
    "WalkthroughSection",
    "DocumentationResult",
    "DocAgent",
]


@dataclass(slots=True)
class DocumentationSummary:
    """Aggregate overview constructed from relevant repository modules."""

    overview: str
    modules: tuple[FileSummary, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overview": self.overview,
            "modules": [summary.to_dict() for summary in self.modules],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ReadmeDraft:
    """README snippet drafted by :class:`DocAgent`."""

    content: str
    section: str
    highlights: tuple[str, ...] = ()
    artifact_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": self.content,
            "section": self.section,
            "highlights": list(self.highlights),
        }
        if self.artifact_path:
            payload["artifact_path"] = self.artifact_path
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class ChangelogDraft:
    """Structured changelog entry generated for a release."""

    version: str | None
    content: str
    highlights: tuple[str, ...]
    artifact_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "content": self.content,
            "highlights": list(self.highlights),
        }
        if self.artifact_path:
            payload["artifact_path"] = self.artifact_path
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class WalkthroughSection:
    """Walkthrough segment describing an important module or workflow."""

    title: str
    body: str
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": self.title,
            "body": self.body,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class DocumentationResult:
    """Container bundling the full documentation draft output."""

    summary: DocumentationSummary
    readme: ReadmeDraft | None
    changelog: ChangelogDraft | None
    walkthrough: tuple[WalkthroughSection, ...]
    evidence: tuple[ResearchSnippet, ...] = ()
    diff_bundle: DiffBundle | None = None
    artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": self.summary.to_dict(),
            "walkthrough": [section.to_dict() for section in self.walkthrough],
            "evidence": [snippet.to_dict() for snippet in self.evidence],
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
        }
        if self.readme is not None:
            payload["readme"] = self.readme.to_dict()
        if self.changelog is not None:
            payload["changelog"] = self.changelog.to_dict()
        if self.diff_bundle is not None:
            payload["diff_bundle"] = self.diff_bundle.to_dict()
        return payload


class DocAgent:
    """Generate documentation updates grounded in repository state."""

    _DEFAULT_ARTIFACT_DIR = ".doc_artifacts"

    def __init__(
        self,
        repo_context: RepoContextAgent,
        *,
        research_agent: ResearchAgent | VariedResearchAgent | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.repo_context = repo_context
        self.research_agent = research_agent
        self._artifact_dir = self._resolve_artifact_dir(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach_research_agent(self, agent: ResearchAgent | VariedResearchAgent | None) -> None:
        """Attach or detach the research agent used for evidence gathering."""

        self.research_agent = agent

    def draft_updates(
        self,
        *,
        highlights: Sequence[str] | None = None,
        version: str | None = None,
        walkthrough_topics: Sequence[str] | None = None,
        summary_paths: Sequence[str] | None = None,
        summary_queries: Sequence[str] | None = None,
        research_queries: Sequence[str] | None = None,
        include_readme: bool = True,
        include_changelog: bool = True,
        top_k: int = 5,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentationResult:
        """Draft documentation snippets derived from repository context."""

        code_diff = self._safe_diff_bundle()
        module_paths = self._collect_paths(code_diff, summary_paths, summary_queries, top_k=top_k)
        module_summaries = self._summarize_modules(module_paths)
        overview = self._build_overview(module_summaries)
        summary_meta: dict[str, Any] = {
            "path_count": len(module_paths),
            "queries": list(self._to_sequence(summary_queries)),
        }
        summary = DocumentationSummary(
            overview=overview,
            modules=tuple(module_summaries),
            metadata=summary_meta,
        )

        research_snippets = self._gather_research(research_queries, top_k=top_k)

        auto_highlights = self._generate_highlights(summary.modules)
        requested_highlights = tuple(self._normalise_highlights(highlights) or auto_highlights)
        if not requested_highlights:
            requested_highlights = auto_highlights

        readme_draft: ReadmeDraft | None = None
        changelog_draft: ChangelogDraft | None = None
        artifacts: list[str] = []

        doc_targets: MutableMapping[str, str] = {}

        if include_readme:
            readme_content = self._compose_readme(summary, requested_highlights, research_snippets)
            artifact_path = self._write_artifact("README_update.md", readme_content)
            readme_draft = ReadmeDraft(
                content=readme_content,
                section="## What's New",
                highlights=requested_highlights,
                artifact_path=artifact_path,
            )
            artifacts.append(artifact_path)
            doc_targets["README.md"] = readme_content

        if include_changelog:
            changelog_content = self._compose_changelog(summary, version, requested_highlights, research_snippets)
            filename = "CHANGELOG_update.md" if version is None else f"CHANGELOG_{self._slugify(version)}.md"
            artifact_path = self._write_artifact(filename, changelog_content)
            changelog_draft = ChangelogDraft(
                version=version,
                content=changelog_content,
                highlights=requested_highlights,
                artifact_path=artifact_path,
                metadata={"version": version},
            )
            artifacts.append(artifact_path)
            doc_targets["CHANGELOG.md"] = changelog_content

        walkthrough_sections = self._compose_walkthrough(walkthrough_topics, top_k=top_k)

        doc_diff = self._build_doc_diff(doc_targets)
        diff_bundle = doc_diff or code_diff

        result_metadata: dict[str, Any] = {
            "version": version,
        }
        if code_diff is not None:
            result_metadata["code_diff"] = code_diff.to_dict()
        if metadata:
            result_metadata.update(dict(metadata))

        result = DocumentationResult(
            summary=summary,
            readme=readme_draft,
            changelog=changelog_draft,
            walkthrough=tuple(walkthrough_sections),
            evidence=research_snippets,
            diff_bundle=diff_bundle,
            artifacts=tuple(dict.fromkeys(artifacts)),
            metadata=result_metadata,
        )
        return result

    def format_result(self, result: DocumentationResult) -> str:
        """Render a human-readable summary of the documentation drafts."""

        lines: list[str] = ["# Documentation Update", ""]
        lines.append("## Overview")
        lines.append(result.summary.overview or "No repository changes detected.")
        lines.append("")
        if result.readme is not None:
            lines.append("## README Draft")
            lines.append(result.readme.content.strip())
            lines.append("")
        if result.changelog is not None:
            version_label = result.changelog.version or "Unreleased"
            lines.append(f"## Changelog ({version_label})")
            lines.append(result.changelog.content.strip())
            lines.append("")
        if result.walkthrough:
            lines.append("## Walkthrough")
            for section in result.walkthrough:
                lines.append(f"### {section.title}")
                lines.append(section.body.strip())
                lines.append("")
        if result.evidence:
            lines.append("## References")
            for snippet in result.evidence:
                lines.append(f"- [{snippet.title}]({snippet.url}): {snippet.quote}")
        return "\n".join(line.rstrip() for line in lines).strip()

    def to_structured_response(self, result: DocumentationResult) -> StructuredResponse:
        """Convert the draft into a :class:`StructuredResponse`."""

        payload = result.to_dict()
        content = self.format_result(result)
        return StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "parsed": payload,
                        }
                    }
                ]
            },
            content=content,
            parsed=payload,
            schema={"name": "DocumentationResult"},
            structured=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_artifact_dir(self, artifact_dir: str | Path | None) -> Path:
        base = Path(self.repo_context.repo_root)
        if artifact_dir is None:
            return base / self._DEFAULT_ARTIFACT_DIR
        artifact_path = Path(artifact_dir)
        if not artifact_path.is_absolute():
            artifact_path = (base / artifact_path).resolve()
        return artifact_path

    def _write_artifact(self, filename: str, content: str) -> str:
        path = self._artifact_dir / filename
        path.write_text(content, encoding="utf-8")
        try:
            relative = path.relative_to(self.repo_context.repo_root)
            return str(relative)
        except ValueError:
            return str(path)

    def _build_doc_diff(self, targets: Mapping[str, str]) -> DiffBundle | None:
        if not targets:
            return None
        patch_parts: list[str] = []
        stats: list[DiffFileStat] = []
        for rel_path, snippet in targets.items():
            existing = self._read_existing(rel_path)
            proposed = self._merge_snippet(existing, snippet)
            diff_lines = list(
                difflib.unified_diff(
                    existing.splitlines(),
                    proposed.splitlines(),
                    fromfile=rel_path,
                    tofile=rel_path,
                    lineterm="",
                )
            )
            if not diff_lines:
                continue
            patch_parts.append("\n".join(diff_lines))
            additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
            stats.append(
                DiffFileStat(
                    path=rel_path,
                    additions=additions,
                    deletions=deletions,
                    status="proposed",
                )
            )
        if not patch_parts:
            return None
        patch = "\n".join(patch_parts) + "\n"
        provenance: dict[str, Any] = {"source": "doc-agent"}
        return DiffBundle(
            patch=patch,
            stats=tuple(stats),
            staged=False,
            include_untracked=False,
            provenance=provenance,
        )

    def _merge_snippet(self, existing: str, snippet: str) -> str:
        merged = existing.rstrip()
        addition = snippet.strip()
        if not merged:
            result = addition
        else:
            result = f"{merged}\n\n{addition}"
        return result.rstrip() + "\n"

    def _read_existing(self, rel_path: str) -> str:
        path = Path(self.repo_context.repo_root) / rel_path
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _safe_diff_bundle(self) -> DiffBundle | None:
        if not hasattr(self.repo_context, "focused_diffs"):
            return None
        try:
            return self.repo_context.focused_diffs(include_untracked=True)
        except Exception:
            return None

    def _collect_paths(
        self,
        diff_bundle: DiffBundle | None,
        summary_paths: Sequence[str] | None,
        summary_queries: Sequence[str] | None,
        *,
        top_k: int,
    ) -> list[str]:
        paths: list[str] = []
        if diff_bundle is not None:
            for stat in diff_bundle.stats:
                if stat.path not in paths:
                    paths.append(stat.path)
        for path in self._to_sequence(summary_paths):
            if path not in paths:
                paths.append(path)
        for query in self._to_sequence(summary_queries):
            if not hasattr(self.repo_context, "focused_files"):
                continue
            try:
                matches: Iterable[RepoSearchResult] = self.repo_context.focused_files(query, top_k=top_k)
            except Exception:
                continue
            for match in matches:
                if match.path not in paths:
                    paths.append(match.path)
        return paths

    def _summarize_modules(self, paths: Iterable[str]) -> list[FileSummary]:
        summaries: list[FileSummary] = []
        for path in paths:
            try:
                summary = self._summarize_path(path)
            except FileNotFoundError:
                continue
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _summarize_path(self, path: str) -> FileSummary | None:
        if path.endswith(".py") and hasattr(self.repo_context, "summarize_ast"):
            try:
                return self.repo_context.summarize_ast(path)
            except Exception:
                pass
        if hasattr(self.repo_context, "summarize_file"):
            try:
                return self.repo_context.summarize_file(path)
            except Exception:
                return None
        return None

    def _build_overview(self, summaries: Sequence[FileSummary]) -> str:
        if not summaries:
            return "No code changes detected in the current diff."
        lines = ["Key modules impacted:"]
        for summary in summaries:
            language = summary.language or "text"
            description = self._first_sentence(summary.summary)
            lines.append(f"- {summary.path} ({language}, {summary.line_count} lines): {description}")
        return "\n".join(lines)

    def _generate_highlights(self, summaries: Sequence[FileSummary]) -> tuple[str, ...]:
        highlights: list[str] = []
        for summary in summaries:
            description = self._first_sentence(summary.summary)
            highlights.append(f"Update {summary.path}: {description}")
        return tuple(highlights)

    def _compose_readme(
        self,
        summary: DocumentationSummary,
        highlights: Sequence[str],
        evidence: Sequence[ResearchSnippet],
    ) -> str:
        lines: list[str] = ["## What's New", ""]
        if highlights:
            for item in highlights:
                lines.append(f"- {item}")
        else:
            lines.append("- Repository updates captured in this release.")
        if summary.modules:
            lines.append("")
            lines.append("### Key Modules")
            lines.append("")
            for module in summary.modules:
                lines.append(
                    f"- **{module.path}** ({module.language or 'text'}): {self._first_sentence(module.summary)}"
                )
        if evidence:
            lines.append("")
            lines.append("### References")
            lines.append("")
            for snippet in evidence:
                lines.append(f"- {snippet.title}: {snippet.quote} ({snippet.url})")
        return "\n".join(lines).strip() + "\n"

    def _compose_changelog(
        self,
        summary: DocumentationSummary,
        version: str | None,
        highlights: Sequence[str],
        evidence: Sequence[ResearchSnippet],
    ) -> str:
        header = version or "Unreleased"
        lines: list[str] = [f"## {header}", ""]
        if highlights:
            lines.append("### Highlights")
            lines.append("")
            for item in highlights:
                lines.append(f"- {item}")
            lines.append("")
        if summary.modules:
            lines.append("### Modules")
            lines.append("")
            for module in summary.modules:
                lines.append(f"- {module.path}: {self._first_sentence(module.summary)}")
            lines.append("")
        if evidence:
            lines.append("### References")
            lines.append("")
            for snippet in evidence:
                lines.append(f"- {snippet.citation}: {snippet.title}")
        return "\n".join(line.rstrip() for line in lines if line is not None).strip() + "\n"

    def _compose_walkthrough(
        self,
        walkthrough_topics: Sequence[str] | None,
        *,
        top_k: int,
    ) -> list[WalkthroughSection]:
        sections: list[WalkthroughSection] = []
        seen_paths: set[str] = set()
        for topic in self._to_sequence(walkthrough_topics):
            matches: Iterable[RepoSearchResult]
            if not hasattr(self.repo_context, "focused_files"):
                continue
            try:
                matches = self.repo_context.focused_files(topic, top_k=top_k)
            except Exception:
                continue
            for match in matches:
                if match.path in seen_paths:
                    continue
                summary = self._summarize_path(match.path)
                if summary is None:
                    continue
                title = self._title_from_path(summary.path)
                body_lines = [f"**Path:** {summary.path}"]
                description = summary.summary.strip() or "See source for details."
                body_lines.append("")
                body_lines.append(description)
                metadata: dict[str, Any] = {
                    "topic": topic,
                    "score": match.score,
                }
                sections.append(
                    WalkthroughSection(
                        title=title,
                        body="\n".join(body_lines).strip(),
                        path=summary.path,
                        metadata=metadata,
                    )
                )
                seen_paths.add(summary.path)
        return sections

    def _gather_research(
        self,
        queries: Sequence[str] | None,
        *,
        top_k: int,
    ) -> tuple[ResearchSnippet, ...]:
        if not queries or self.research_agent is None:
            return ()
        snippets: list[ResearchSnippet] = []
        for query in self._to_sequence(queries):
            result: ResearchResult = self.research_agent.search(query, top_k=top_k)
            snippets.extend(result.snippets)
        return tuple(snippets)

    @staticmethod
    def _first_sentence(text: str) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return "Updates pending documentation."
        match = re.search(r"([.!?])\s", clean)
        if match:
            return clean[: match.end(1)].strip()
        return clean

    @staticmethod
    def _title_from_path(path: str) -> str:
        name = Path(path).stem.replace("_", " ").title()
        return name or path

    @staticmethod
    def _normalise_highlights(highlights: Sequence[str] | None) -> tuple[str, ...]:
        if highlights is None:
            return ()
        items: list[str] = []
        for item in highlights:
            text = str(item).strip()
            if text:
                items.append(text)
        return tuple(items)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
        return slug.lower() or "release"

    @staticmethod
    def _to_sequence(value: Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

