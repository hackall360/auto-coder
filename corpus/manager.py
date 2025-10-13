"""Structured corpus event logging backed by the shared memory facade."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from memory import MemoryFacade, MemoryRecord, MemoryRouter

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CorpusEvent:
    """Normalized representation of an interaction captured for the corpus."""

    source: str
    payload: Any
    event_type: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    agent_id: str | None = None

    def enrich_tags(self, extra: Iterable[str]) -> None:
        merged = list(self.tags)
        merged.extend(str(tag) for tag in extra if tag)
        self.tags = tuple(dict.fromkeys(merged))  # Preserve order without duplicates


_DEFAULT_CATEGORY_MAP: Mapping[str, str] = {
    "web_search": "research",
    "web_result": "research",
    "research_snippet": "research",
    "repo_search": "repo_context",
    "repo_symbol_search": "repo_context",
    "repo_summary": "repo_context",
    "repo_read": "repo_context",
    "repo_update": "repo_context",
    "file_read": "repo_context",
    "file_write": "repo_context",
    "file_patch": "repo_context",
    "repo_diff": "repo_context",
    "command_execution": "runtime",
    "process_run": "runtime",
    "code_generation": "assistant_output",
    "assistant_reply": "assistant_output",
    "tool_result": "runtime",
    "test_execution": "qa",
    "security_report": "security",
}


class CorpusManager:
    """Persist structured event payloads into memory-backed corpus storage."""

    def __init__(
        self,
        facade: MemoryFacade | None,
        *,
        scope: str | None = None,
        category_map: Mapping[str, str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._facade = facade
        self._scope = scope or MemoryRouter.LONG_TERM
        merged = dict(_DEFAULT_CATEGORY_MAP)
        if category_map:
            merged.update({str(key): str(value) for key, value in category_map.items()})
        self._category_map = merged
        self._enabled = bool(enabled)
        self._modifiers: list[Callable[[CorpusEvent], None]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled and self._facade is not None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def register_modifier(self, modifier: Callable[[CorpusEvent], None]) -> None:
        """Install a callback that can adjust events before persistence."""

        if modifier not in self._modifiers:
            self._modifiers.append(modifier)

    def infer_category(self, event_type: str | None, default: str = "general") -> str:
        if not event_type:
            return default
        normalized = str(event_type).strip().lower()
        if normalized in self._category_map:
            return self._category_map[normalized]
        head, _, tail = normalized.partition(":")
        if head and head in self._category_map:
            return self._category_map[head]
        if tail and tail in self._category_map:
            return self._category_map[tail]
        return default

    def record_event(
        self,
        *,
        source: str,
        payload: Any,
        event_type: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord | None:
        """Persist an event into the configured corpus scope."""

        if not self.enabled:
            return None
        event = CorpusEvent(
            source=str(source),
            payload=payload,
            event_type=event_type,
            category=category,
            tags=tuple(str(tag) for tag in (tags or ())),
            metadata=dict(metadata or {}),
            session_id=session_id,
            agent_id=agent_id,
        )
        if not event.category:
            event.category = self.infer_category(event.event_type)
        if event.category:
            event.enrich_tags((event.category,))
        event.enrich_tags((event.source,))
        for modifier in list(self._modifiers):
            try:
                modifier(event)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.debug("Corpus modifier %s raised", modifier, exc_info=True)

        attributes: MutableMapping[str, Any] = {
            "source": event.source,
            "event_type": event.event_type,
            "category": event.category,
        }
        attributes.update({str(key): value for key, value in event.metadata.items()})
        attributes["payload"] = event.payload
        attributes["recorded_at"] = datetime.now(timezone.utc).isoformat()

        content = self._stringify_payload(event.payload)
        try:
            return self._facade.add(
                content,
                scope=self._scope,
                tags=event.tags,
                attributes=attributes,
                source="corpus",
                session_id=event.session_id,
                agent_id=event.agent_id,
            )
        except Exception:
            LOGGER.exception("Failed to persist corpus event from %s", event.source)
            return None

    @staticmethod
    def _stringify_payload(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            return payload.decode("utf-8", errors="replace")
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(payload)


_SHARED_CORPUS_MANAGER: CorpusManager | None = None


def set_shared_corpus_manager(manager: CorpusManager | None) -> None:
    global _SHARED_CORPUS_MANAGER
    _SHARED_CORPUS_MANAGER = manager


def get_shared_corpus_manager() -> CorpusManager | None:
    return _SHARED_CORPUS_MANAGER


def record_event(
    source: str,
    payload: Any,
    *,
    event_type: str | None = None,
    category: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> MemoryRecord | None:
    manager = get_shared_corpus_manager()
    if manager is None:
        return None
    return manager.record_event(
        source=source,
        payload=payload,
        event_type=event_type,
        category=category,
        tags=tags,
        metadata=metadata,
        session_id=session_id,
        agent_id=agent_id,
    )
