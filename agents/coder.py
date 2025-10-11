"""Autonomous coding agent that collaborates with the manager workflow.

The :class:`CoderAgent` is responsible for consuming implementation tasks
dispatched by :class:`~agents.manager.ManagerAgent`, consulting repository
context payloads, requesting additional guidance from LM Studio, and finally
applying changes to the working tree.  File mutations are performed via the
`internal.tools.file` and `internal.tools.patch` suites to ensure consistent
behaviour with other automation helpers.  Every applied diff is tracked so the
manager can surface reviewable artifacts or feed follow-up agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from internal.structures import StructuredResponse
from internal.tools import file as file_tools
from internal.tools import patch as patch_tools
from session import AgentRound, AgentSession
from tooling import ToolRegistry, ToolSpec

from .repo_context import DiffBundle, DiffFileStat, RepoSearchResult, RepoSymbolResult

__all__ = [
    "MinimalPatch",
    "AppliedDiff",
    "ChangeSummary",
    "CoderResult",
    "CoderAgent",
]


def _as_sequence(value: Any | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


_PATCH_PATH_PATTERN = re.compile(r"^\+\+\+\s+(?:b/)?(?P<path>[^\s]+)$", re.MULTILINE)


@dataclass(slots=True)
class MinimalPatch:
    """Container describing a unified diff patch returned by the LLM."""

    patch: str
    path: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"patch": self.patch}
        if self.path:
            payload["path"] = self.path
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class AppliedDiff:
    """Record of an applied diff along with summary statistics."""

    patch: MinimalPatch
    apply_result: Mapping[str, Any]
    summary: Mapping[str, Any] | None = None
    tool: str = "patch.apply"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "patch": self.patch.to_dict(),
            "tool": self.tool,
            "result": dict(self.apply_result),
        }
        if self.summary:
            payload["summary"] = dict(self.summary)
        return payload

    @property
    def changed_paths(self) -> tuple[str, ...]:
        if isinstance(self.summary, Mapping):
            files = self.summary.get("files")
            if isinstance(files, Mapping):
                return tuple(str(path) for path in files.keys())
        if self.patch.path:
            return (self.patch.path,)
        return ()


@dataclass(slots=True)
class ChangeSummary:
    """Structured summary describing a change for the manager."""

    path: str
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {"path": self.path, "summary": self.summary}
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True)
class CoderResult:
    """Return payload produced by :class:`CoderAgent.run_task`."""

    rationale: str
    patches: tuple[MinimalPatch, ...]
    applied_diffs: tuple[AppliedDiff, ...]
    change_summaries: tuple[ChangeSummary, ...]
    dependency_hints: tuple[str, ...]
    response_text: str
    structured_response: StructuredResponse
    round_record: AgentRound | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "patches": [patch.to_dict() for patch in self.patches],
            "applied_diffs": [diff.to_dict() for diff in self.applied_diffs],
            "change_summaries": [summary.to_dict() for summary in self.change_summaries],
            "dependency_hints": list(self.dependency_hints),
            "response_text": self.response_text,
            "round_index": self.round_record.index if self.round_record else None,
        }


class CoderAgent:
    """Implementation-focused agent that produces reviewable code changes."""

    CODER_RESPONSE_SCHEMA: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "patch": {"type": "string"},
                        "description": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["patch"],
                },
                "default": [],
            },
            "change_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "summary": {"type": "string"},
                        "details": {"type": "object"},
                    },
                },
                "default": [],
            },
            "dependency_hints": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["rationale", "patches"],
        "additionalProperties": True,
    }

    def __init__(
        self,
        *,
        session: AgentSession | None = None,
        session_factory: Callable[[], AgentSession] | None = None,
        repo_root: str | None = None,
        tool_registry: ToolRegistry | None = None,
        extra_tools: Iterable[ToolSpec | Any] | None = None,
    ) -> None:
        if session is None:
            if session_factory is None:
                raise ValueError("CoderAgent requires a session or session_factory")
            session = session_factory()
        self.session = session
        self.repo_root = os.path.abspath(repo_root) if repo_root else os.getcwd()
        self._tool_registry = tool_registry or ToolRegistry()
        self._default_tools: list[ToolSpec] = []
        self._register_default_tools(extra_tools)
        self._applied_diffs: list[AppliedDiff] = []
        self._last_result: CoderResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def applied_diffs(self) -> tuple[AppliedDiff, ...]:
        return tuple(self._applied_diffs)

    @property
    def last_result(self) -> CoderResult | None:
        return self._last_result

    def run_task(
        self,
        task: str,
        *,
        context_payloads: Sequence[Any] | None = None,
        guidance_hints: Sequence[Mapping[str, Any] | str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CoderResult:
        if not task or not task.strip():
            raise ValueError("CoderAgent.run_task requires a non-empty task description")

        context_items = _as_sequence(context_payloads)
        hints = _as_sequence(guidance_hints)
        prompt = self._compose_prompt(task, context_items, hints)

        text, structured = self.session.act(
            prompt,
            tools=self._default_tools,
            schema=self.CODER_RESPONSE_SCHEMA,
            metadata=self._build_metadata(metadata, context_items, hints),
            handle_invalid_tool_request=self._handle_invalid_tool_request,
        )

        parsed = structured.parsed or {}
        normalized_patches = [self._normalize_patch(entry) for entry in parsed.get("patches", [])]
        normalized_changes = [self._normalize_change_summary(entry) for entry in parsed.get("change_summaries", [])]
        dependency_hints = tuple(str(item).strip() for item in parsed.get("dependency_hints", []) if str(item).strip())
        rationale = str(parsed.get("rationale", structured.content)).strip()

        applied: list[AppliedDiff] = []
        for patch in normalized_patches:
            if not patch.patch.strip():
                continue
            summary = self._summarize_patch(patch.patch)
            result = patch_tools.patch(operation="apply", patch_text=patch.patch, root=self.repo_root)
            status = result.get("status")
            if status != "success":
                message = result.get("message", "failed to apply patch")
                raise RuntimeError(f"Failed to apply patch for {patch.path or 'unknown path'}: {message}")
            applied_diff = AppliedDiff(patch=patch, apply_result=result, summary=summary)
            applied.append(applied_diff)
            self._applied_diffs.append(applied_diff)

        merged_summaries = self._merge_change_summaries(normalized_changes, applied)

        round_record = None
        last_round_getter = getattr(self.session, "last_round", None)
        if callable(last_round_getter):
            round_record = last_round_getter()

        result = CoderResult(
            rationale=rationale,
            patches=tuple(normalized_patches),
            applied_diffs=tuple(applied),
            change_summaries=tuple(merged_summaries),
            dependency_hints=dependency_hints,
            response_text=text,
            structured_response=structured,
            round_record=round_record,
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_default_tools(self, extra_tools: Iterable[ToolSpec | Any] | None) -> None:
        baseline_tools: list[Any] = [file_tools.file_tool, patch_tools.patch_tool]
        if extra_tools:
            baseline_tools.extend(extra_tools)
        specs: list[ToolSpec] = []
        for tool in baseline_tools:
            spec = self._tool_registry.register(tool, replace=True)
            specs.append(spec)
        self._default_tools = specs

    def _build_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        context_payloads: Sequence[Any],
        hints: Sequence[Any],
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(metadata, Mapping):
            payload.update(metadata)
        payload["context_count"] = len(context_payloads)
        payload["hint_count"] = len(hints)
        return payload

    def _compose_prompt(
        self,
        task: str,
        context_payloads: Sequence[Any],
        hints: Sequence[Any],
    ) -> str:
        sections: list[str] = []
        sections.append("You are a senior software engineer applying focused edits. Respond using the provided schema.")
        sections.append("## Task\n" + task.strip())
        if context_payloads:
            rendered = [self._render_context_item(item) for item in context_payloads]
            context_block = "\n\n".join(rendered)
            sections.append("## Repository context\n" + context_block)
        if hints:
            hint_lines = ["- " + self._render_hint(hint) for hint in hints]
            sections.append("## Guidance hints\n" + "\n".join(hint_lines))
        sections.append(
            "## Output requirements\n"
            "Return minimal reviewable patches alongside a concise rationale."
        )
        return "\n\n".join(section for section in sections if section).strip()

    def _render_context_item(self, item: Any) -> str:
        if isinstance(item, RepoSearchResult):
            return (
                f"[search] {item.path}:{item.offset} (score={item.score:.2f})\n"
                f"{item.text.strip()}"
            )
        if isinstance(item, RepoSymbolResult):
            spans = ", ".join(f"{start}-{end}" for start, end in item.spans)
            return (
                f"[symbol] {item.path}:{item.offset} spans[{spans}]\n"
                f"{item.text.strip()}"
            )
        if isinstance(item, DiffBundle):
            stats = ", ".join(
                f"{stat.path} (+{stat.additions}/-{stat.deletions})"
                for stat in item.stats
            )
            header = f"[diff:{'staged' if item.staged else 'unstaged'}] {stats or 'no changes'}"
            if item.patch.strip():
                return header + "\n" + item.patch.strip()
            return header
        if isinstance(item, DiffFileStat):
            return f"[diffstat] {item.path}: +{item.additions}/-{item.deletions} ({item.status or 'modified'})"
        if isinstance(item, Mapping):
            return json.dumps(item, indent=2, sort_keys=True)
        return str(item)

    def _render_hint(self, hint: Any) -> str:
        if isinstance(hint, Mapping):
            path = hint.get("path")
            range_info = hint.get("range") or hint.get("span")
            message = hint.get("message") or hint.get("hint")
            parts = []
            if path:
                parts.append(str(path))
            if range_info:
                parts.append(f"{range_info}")
            if message:
                parts.append(str(message))
            if parts:
                return " | ".join(parts)
        return str(hint)

    def _normalize_patch(self, entry: Any) -> MinimalPatch:
        if isinstance(entry, MinimalPatch):
            return entry
        if isinstance(entry, str):
            return MinimalPatch(patch=entry)
        if isinstance(entry, Mapping):
            patch_text = str(entry.get("patch") or entry.get("diff") or "")
            description = entry.get("description")
            metadata = entry.get("metadata")
            metadata_payload: Mapping[str, Any] = {}
            if isinstance(metadata, Mapping):
                metadata_payload = dict(metadata)
            path_value = entry.get("path") or entry.get("file")
            path = str(path_value) if path_value else None
            patch = MinimalPatch(
                patch=patch_text,
                path=path,
                description=str(description) if isinstance(description, str) else None,
                metadata=metadata_payload,
            )
            if not patch.path:
                paths = self._extract_paths_from_patch(patch.patch)
                if paths:
                    patch.path = paths[0]
            return patch
        raise TypeError(f"Unsupported patch entry type: {type(entry)!r}")

    def _extract_paths_from_patch(self, patch_text: str) -> tuple[str, ...]:
        matches = [m.group("path") for m in _PATCH_PATH_PATTERN.finditer(patch_text)]
        cleaned = []
        for match in matches:
            if match == "/dev/null":
                continue
            cleaned.append(match)
        seen: set[str] = set()
        ordered: list[str] = []
        for path in cleaned:
            normalized = path.lstrip("b/") if path.startswith("b/") else path
            if normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        return tuple(ordered)

    def _normalize_change_summary(self, entry: Any) -> ChangeSummary:
        if isinstance(entry, ChangeSummary):
            return entry
        if isinstance(entry, Mapping):
            path = str(entry.get("path") or entry.get("file") or "*")
            summary = str(entry.get("summary") or entry.get("description") or "").strip()
            details = entry.get("details")
            if not summary:
                summary = "Change applied"
            details_payload: Mapping[str, Any] = {}
            if isinstance(details, Mapping):
                details_payload = dict(details)
            return ChangeSummary(path=path, summary=summary, details=details_payload)
        if isinstance(entry, str):
            return ChangeSummary(path="*", summary=entry.strip() or "Change applied")
        return ChangeSummary(path="*", summary=str(entry))

    def _summarize_patch(self, patch_text: str) -> Mapping[str, Any] | None:
        result = patch_tools.patch(operation="summary", patch_text=patch_text)
        if not isinstance(result, MutableMapping):
            return None
        if result.get("status") != "success":
            return None
        return result

    def _merge_change_summaries(
        self,
        provided: Sequence[ChangeSummary],
        applied: Sequence[AppliedDiff],
    ) -> list[ChangeSummary]:
        summaries: dict[str, ChangeSummary] = {summary.path: summary for summary in provided}
        for diff in applied:
            if not isinstance(diff.summary, Mapping):
                continue
            files = diff.summary.get("files")
            totals = diff.summary.get("totals")
            if isinstance(files, Mapping):
                for path, stats in files.items():
                    details = {"additions": stats.get("additions", 0), "deletions": stats.get("deletions", 0)}
                    summary_text = self._format_stats(details, totals)
                    if path in summaries:
                        merged_details = dict(summaries[path].details)
                        merged_details.update(details)
                        summaries[path] = ChangeSummary(path=path, summary=summaries[path].summary, details=merged_details)
                    else:
                        summaries[path] = ChangeSummary(path=path, summary=summary_text, details=details)
            elif isinstance(totals, Mapping):
                summary_text = self._format_stats(totals, totals)
                target_path = diff.patch.path or "*"
                if target_path in summaries:
                    merged_details = dict(summaries[target_path].details)
                    merged_details.setdefault("additions", totals.get("additions", 0))
                    merged_details.setdefault("deletions", totals.get("deletions", 0))
                    summaries[target_path] = ChangeSummary(
                        path=target_path,
                        summary=summaries[target_path].summary,
                        details=merged_details,
                    )
                else:
                    summaries[target_path] = ChangeSummary(
                        path=target_path,
                        summary=summary_text,
                        details={
                            "additions": totals.get("additions", 0),
                            "deletions": totals.get("deletions", 0),
                        },
                    )
        return list(summaries.values())

    def _format_stats(self, stats: Mapping[str, Any], totals: Mapping[str, Any] | None) -> str:
        adds = stats.get("additions", 0)
        dels = stats.get("deletions", 0)
        files_changed = None
        if totals and isinstance(totals, Mapping):
            files_changed = totals.get("files_changed")
        if files_changed is None and stats is not totals and totals:
            files_changed = totals.get("files_changed") if isinstance(totals, Mapping) else None
        parts = [f"+{adds}", f"-{dels}"]
        if files_changed is not None:
            parts.append(f"files:{files_changed}")
        return " / ".join(parts)

    def _handle_invalid_tool_request(self, error: Exception, call: Mapping[str, Any]) -> None:
        raise RuntimeError(
            f"CoderAgent encountered an invalid tool request: {call.get('name')}"  # noqa: EM102
        ) from error

