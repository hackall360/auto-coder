"""Structured corpus event logging backed by the shared memory facade."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
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
        storage_path: str | Path | None = None,
        dedup_threshold: float | None = None,
    ) -> None:
        self._facade = facade
        self._scope = scope or MemoryRouter.LONG_TERM
        merged = dict(_DEFAULT_CATEGORY_MAP)
        if category_map:
            merged.update({str(key): str(value) for key, value in category_map.items()})
        self._category_map = merged
        self._enabled = bool(enabled)
        self._modifiers: list[Callable[[CorpusEvent], None]] = []
        self._storage_path = self._normalise_path(storage_path)
        self._dedup_threshold = self._normalise_threshold(dedup_threshold)
        self._dedup_cache: dict[tuple[str | None, str | None, str | None], deque[str]] = {}
        self._dedup_cache_limit = 32

    @property
    def enabled(self) -> bool:
        if not self._enabled:
            return False
        return self._facade is not None or self._storage_path is not None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def storage_path(self) -> Path | None:
        return self._storage_path

    @property
    def dedup_threshold(self) -> float | None:
        return self._dedup_threshold

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
        payload_text = self._stringify_payload(event.payload)
        if self._dedup_threshold is not None and self._is_duplicate(event, payload_text):
            LOGGER.debug(
                "Skipping corpus event from %s (%s) due to deduplication threshold %.2f",
                event.source,
                event.event_type,
                self._dedup_threshold,
            )
            return None
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

        record: MemoryRecord | None = None
        try:
            if self._facade is not None:
                record = self._facade.add(
                    payload_text,
                    scope=self._scope,
                    tags=event.tags,
                    attributes=attributes,
                    source="corpus",
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                )
        except Exception:
            LOGGER.exception("Failed to persist corpus event from %s", event.source)

        self._remember_event(event, payload_text)
        self._write_to_storage(event, attributes, record)
        return record

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

    @staticmethod
    def _normalise_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _normalise_threshold(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None
        if threshold <= 0:
            return None
        if threshold > 1:
            threshold = 1.0
        return threshold

    def _dedup_key(self, event: CorpusEvent) -> tuple[str | None, str | None, str | None]:
        return event.category, event.event_type, event.session_id

    def _is_duplicate(self, event: CorpusEvent, content: str) -> bool:
        key = self._dedup_key(event)
        bucket = self._dedup_cache.get(key)
        if not bucket:
            return False
        for existing in bucket:
            if not existing:
                continue
            score = SequenceMatcher(None, existing, content).ratio()
            if score >= (self._dedup_threshold or 1.0):
                return True
        return False

    def _remember_event(self, event: CorpusEvent, content: str) -> None:
        if self._dedup_threshold is None:
            return
        key = self._dedup_key(event)
        bucket = self._dedup_cache.setdefault(key, deque(maxlen=self._dedup_cache_limit))
        bucket.append(content)

    def _write_to_storage(
        self,
        event: CorpusEvent,
        attributes: Mapping[str, Any],
        record: MemoryRecord | None,
    ) -> None:
        if self._storage_path is None:
            return

        payload: dict[str, Any] = {
            "recorded_at": attributes.get("recorded_at"),
            "source": event.source,
            "event_type": event.event_type,
            "category": event.category,
            "tags": list(event.tags),
            "metadata": dict(event.metadata),
            "payload": event.payload,
            "session_id": event.session_id,
            "agent_id": event.agent_id,
        }
        if record is not None:
            payload["memory_record_id"] = record.record_id

        try:
            with self._storage_path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
                handle.write("\n")
        except Exception:
            LOGGER.debug(
                "Failed to persist corpus event to %s", self._storage_path,
                exc_info=True,
            )


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
