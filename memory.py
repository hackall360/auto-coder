"""Utilities for configuring and working with agent memory stores.

This module defines a small abstraction layer that allows the rest of the
application to request different flavours of memory (for example, "short" or
"long" term storage) without coupling callers to a particular backend.  It also
includes helpers for loading configuration from environment variables or an
optional ``config.json`` file so that operators can adjust memory behaviour
without modifying code.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import struct
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Coroutine

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
    from redis.exceptions import WatchError  # type: ignore
    try:
        from redis.commands.search.query import Query  # type: ignore
    except Exception:  # pragma: no cover - redis search optional
        Query = None  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None  # type: ignore
    WatchError = RuntimeError  # type: ignore
    Query = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from psycopg import sql  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
    from psycopg_pool import AsyncConnectionPool  # type: ignore
except Exception:  # pragma: no cover - psycopg optional
    sql = None  # type: ignore
    dict_row = None  # type: ignore
    AsyncConnectionPool = None  # type: ignore

LOGGER = logging.getLogger(__name__)


EmbeddingFunction = Callable[[str], Sequence[float]]


def _embedder_cache_key(model_name: Optional[str]) -> str:
    value = (model_name or "__default__").strip()
    return value or "__default__"


_EMBEDDER_CACHE: Dict[str, Tuple[Optional[EmbeddingFunction], Optional[str]]] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


def register_embedding_provider(
    provider: EmbeddingFunction,
    *,
    model_name: Optional[str] = None,
) -> None:
    """Register a custom embedding provider used during memory ingestion."""

    key = _embedder_cache_key(model_name)
    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE[key] = (provider, model_name)


def clear_embedding_providers() -> None:
    """Clear all cached embedding providers."""

    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def _coerce_embedding_sequence(payload: Any) -> List[float]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in ("embedding", "vector", "data", "values"):
            if key in payload:
                return _coerce_embedding_sequence(payload[key])
        return []
    if hasattr(payload, "embedding"):
        return _coerce_embedding_sequence(getattr(payload, "embedding"))
    if hasattr(payload, "vector"):
        return _coerce_embedding_sequence(getattr(payload, "vector"))
    if hasattr(payload, "tolist") and callable(payload.tolist):  # pragma: no cover - numpy arrays
        return _coerce_embedding_sequence(payload.tolist())
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return _coerce_embedding_sequence(decoded)
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        try:
            return [float(value) for value in payload]
        except (TypeError, ValueError):
            return []
    if isinstance(payload, (bytes, bytearray)):
        return []
    try:
        return [float(payload)]
    except (TypeError, ValueError):
        return []


def _load_embedding_function(
    model_name: Optional[str],
) -> Tuple[Optional[EmbeddingFunction], Optional[str]]:
    key = _embedder_cache_key(model_name)
    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None:
            return cached

    resolved_name = model_name or os.getenv("MEMORY_EMBEDDING_MODEL")

    try:  # pragma: no cover - optional dependency
        import lmstudio as lms  # type: ignore
    except Exception:  # pragma: no cover - executed when lmstudio missing
        LOGGER.debug("LM Studio embeddings are unavailable; continuing without automatic embeddings")
        result = (None, resolved_name)
        with _EMBEDDER_CACHE_LOCK:
            _EMBEDDER_CACHE[key] = result
        return result

    try:
        handle = lms.embedding_model(resolved_name) if resolved_name else lms.embedding_model()
    except Exception:  # pragma: no cover - runtime configuration issues
        LOGGER.warning("Failed to load LM Studio embedding model '%s'", resolved_name, exc_info=True)
        result = (None, resolved_name)
        with _EMBEDDER_CACHE_LOCK:
            _EMBEDDER_CACHE[key] = result
        return result

    def _invoke(text: str) -> Sequence[float]:
        try:
            raw = handle.embed(text)
        except Exception:  # pragma: no cover - runtime errors
            LOGGER.debug("Embedding model raised an exception", exc_info=True)
            return []
        if asyncio.iscoroutine(raw):  # pragma: no cover - defensive guard
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                raw = asyncio.run(raw)
            else:
                if loop.is_running():
                    LOGGER.warning(
                        "Embedding model returned coroutine while an event loop is active; skipping embedding computation",
                    )
                    return []
                raw = loop.run_until_complete(raw)
        return _coerce_embedding_sequence(raw)

    def provider(text: str) -> Sequence[float]:
        return list(_invoke(text))

    resolved_handle_name = (
        getattr(handle, "model_key", None)
        or getattr(handle, "model", None)
        or getattr(handle, "name", None)
        or resolved_name
    )

    result = (provider, resolved_handle_name)
    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE[key] = result
    return result


def resolve_embedding_provider(
    model_name: Optional[str] = None,
) -> Tuple[Optional[EmbeddingFunction], Optional[str]]:
    """Return a cached embedding provider and associated model name."""

    return _load_embedding_function(model_name)


def _ensure_embedding(
    record: "MemoryRecord",
    embedder: Optional[EmbeddingFunction],
    model_name: Optional[str],
) -> "MemoryRecord":
    metadata = record.metadata
    existing = _coerce_embedding_sequence(record.embedding) if record.embedding else []
    if embedder is None:
        if existing and metadata.embedding_model is None and model_name:
            metadata = replace(metadata, embedding_model=model_name)
        if existing:
            return replace(record, embedding=existing, metadata=metadata)
        return record

    content = (record.content or "").strip()
    computed: List[float] = []
    if content:
        try:
            computed = _coerce_embedding_sequence(embedder(content))
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Embedding provider raised an exception", exc_info=True)
            computed = []
    if not computed:
        computed = existing
    if computed:
        if metadata.embedding_model is None and model_name:
            metadata = replace(metadata, embedding_model=model_name)
        return replace(record, embedding=computed, metadata=metadata)
    if metadata.embedding_model is None and model_name:
        metadata = replace(metadata, embedding_model=model_name)
    return replace(record, embedding=None, metadata=metadata)


# ---------------------------------------------------------------------------
# Dataclasses describing memory records and queries
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _datetime_to_str(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _datetime_from_str(value: Optional[str]) -> datetime:
    if not value:
        return _utcnow()
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError):
            return _utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _metadata_to_dict(metadata: "MemoryMetadata") -> Dict[str, Any]:
    return {
        "source": metadata.source,
        "created_at": _datetime_to_str(metadata.created_at),
        "updated_at": _datetime_to_str(metadata.updated_at),
        "ttl_seconds": metadata.ttl_seconds,
        "tags": list(metadata.tags),
        "importance": metadata.importance,
        "embedding_model": metadata.embedding_model,
        "attributes": dict(metadata.attributes),
    }


def _metadata_from_dict(payload: Mapping[str, Any]) -> "MemoryMetadata":
    tags_raw = payload.get("tags")
    if isinstance(tags_raw, str):
        try:
            tags_list = list(json.loads(tags_raw))
        except json.JSONDecodeError:
            tags_list = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
    else:
        tags_list = list(tags_raw or [])

    attributes_raw = payload.get("attributes", {})
    if isinstance(attributes_raw, str):
        try:
            attributes = dict(json.loads(attributes_raw))
        except json.JSONDecodeError:
            attributes = {"raw": attributes_raw}
    else:
        attributes = dict(attributes_raw)

    ttl_raw = payload.get("ttl_seconds")
    ttl_seconds: Optional[int]
    if ttl_raw in {None, "", "null"}:
        ttl_seconds = None
    else:
        try:
            ttl_seconds = int(ttl_raw)
        except (TypeError, ValueError):
            ttl_seconds = None

    importance_raw = payload.get("importance")
    if importance_raw in {None, "", "null"}:
        importance = None
    else:
        try:
            importance = float(importance_raw)
        except (TypeError, ValueError):
            importance = None

    embedding_model = payload.get("embedding_model")
    if embedding_model == "":
        embedding_model = None

    created_at = _datetime_from_str(str(payload.get("created_at"))) if payload.get("created_at") else _utcnow()
    updated_at = _datetime_from_str(str(payload.get("updated_at"))) if payload.get("updated_at") else created_at

    return MemoryMetadata(
        source=str(payload.get("source", "unknown")),
        created_at=created_at,
        updated_at=updated_at,
        ttl_seconds=ttl_seconds,
        tags=tuple(str(tag) for tag in tags_list),
        importance=importance,
        embedding_model=embedding_model,
        attributes=attributes,
    )


def _pack_embedding(values: Sequence[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"{len(values)}f", *values)


@dataclass
class MemoryMetadata:
    """Metadata describing a memory record.

    Attributes
    ----------
    source:
        A free-form description of the origin of the memory.
    created_at / updated_at:
        Timestamps in UTC that track the lifecycle of the record.
    ttl_seconds:
        Optional time-to-live. ``None`` indicates that the record should not
        expire automatically.
    tags:
        Arbitrary labels that can be used for downstream filtering.
    importance:
        Optional importance score that downstream ranking systems may use.
    embedding_model:
        Identifier for the embedding model that produced the embedding vector
        stored in :class:`MemoryRecord`.
    attributes:
        Additional key/value metadata for downstream filtering.
    """

    source: str = "unknown"
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    ttl_seconds: Optional[int] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    importance: Optional[float] = None
    embedding_model: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp to the current UTC time."""

        self.updated_at = _utcnow()

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Return ``True`` if the TTL has elapsed."""

        if self.ttl_seconds is None:
            return False
        now = now or _utcnow()
        return now >= self.created_at + timedelta(seconds=self.ttl_seconds)

    def matches(self, filters: Mapping[str, Any]) -> bool:
        """Check whether the metadata satisfies the provided filters.

        ``filters`` can contain attribute names that refer to standard metadata
        fields (``source``, ``tags`` and so on) or keys inside ``attributes``.
        Tag comparisons accept either a single value or an iterable of values.
        """

        if not filters:
            return True

        for key, value in filters.items():
            if key == "tags":
                requested: Iterable[str]
                if isinstance(value, str):
                    requested = (value,)
                else:
                    try:
                        requested = tuple(value)  # type: ignore[arg-type]
                    except TypeError:
                        requested = (str(value),)
                if not set(requested).issubset(set(self.tags)):
                    return False
                continue

            current = getattr(self, key, self.attributes.get(key))
            if isinstance(value, (tuple, list, set, frozenset)):
                if current not in value:
                    return False
            elif current != value:
                return False
        return True


@dataclass
class MemoryRecord:
    """A single memory entry stored in a :class:`MemoryStore`."""

    content: str
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    embedding: Optional[Sequence[float]] = None
    score: Optional[float] = None
    record_id: str = field(default_factory=lambda: os.urandom(16).hex())

    def with_score(self, score: Optional[float]) -> "MemoryRecord":
        """Return a shallow copy of the record with an updated score."""

        return replace(self, score=score)


@dataclass
class _RedisRecordState:
    record: MemoryRecord
    session_id: Optional[str]
    agent_id: Optional[str]
    tags: Tuple[str, ...]
    deleted: bool = False


@dataclass
class MemoryQuery:
    """Parameters describing how to fetch memories from a store."""

    text: Optional[str] = None
    embedding: Optional[Sequence[float]] = None
    limit: int = 10
    min_score: Optional[float] = None
    metadata_filters: Mapping[str, Any] = field(default_factory=dict)
    offset: int = 0


# ---------------------------------------------------------------------------
# Memory store interfaces and implementations
# ---------------------------------------------------------------------------


class MemoryStore(ABC):
    """Abstract base class for memory store implementations."""

    backend_name: str = "abstract"

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Add a record to the store and return the stored value."""

    @abstractmethod
    def update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        """Update an existing record and return the new value."""

    @abstractmethod
    def delete(
        self,
        record_id: str,
        *,
        hard: bool = False,
        undo_window: Optional[int] = None,
    ) -> None:
        """Remove a record from the store."""

    @abstractmethod
    def get(self, record_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        """Return a record by identifier."""

    @abstractmethod
    def fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        """Return records that match the query."""

    @abstractmethod
    def compact(self) -> None:
        """Run maintenance tasks such as removing expired records."""


class InMemoryMemoryStore(MemoryStore):
    """Simple in-memory store used for defaults and testing."""

    backend_name = "memory"

    def __init__(
        self,
        *,
        default_ttl: Optional[int] = None,
        compaction_threshold: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self._records: Dict[str, MemoryRecord] = {}
        self._deleted_records: Dict[str, Tuple[MemoryRecord, Optional[datetime]]] = {}
        self._default_ttl = default_ttl
        self._compaction_threshold = compaction_threshold or 0
        self._compaction_counter = 0
        self._embedding_model = embedding_model
        self._embedder_fn, self._embedder_model_name = _load_embedding_function(embedding_model)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record = self._prepare_record(record)
        self._deleted_records.pop(record.record_id, None)
        self._records[record.record_id] = record
        self._maybe_compact()
        return record

    def update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        if record_id not in self._records:
            raise KeyError(f"Memory record '{record_id}' does not exist")

        record = self._records[record_id]
        if content is not None:
            record = replace(record, content=content)
        if metadata is not None:
            record = replace(record, metadata=metadata)
        if embedding is not None:
            record = replace(record, embedding=list(embedding))
        if score is not None:
            record = replace(record, score=score)

        record = self._prepare_record(record)
        self._deleted_records.pop(record_id, None)
        self._records[record_id] = record
        self._maybe_compact()
        return record

    def delete(self, record_id: str, *, hard: bool = False, undo_window: Optional[int] = None) -> None:
        record = self._records.pop(record_id, None)
        if record is None:
            if hard:
                self._deleted_records.pop(record_id, None)
            return
        if hard or (undo_window is not None and undo_window <= 0):
            self._deleted_records.pop(record_id, None)
            return

        expiry: Optional[datetime] = None
        if undo_window is not None:
            expiry = _utcnow() + timedelta(seconds=max(0, undo_window))
        provenance = dict(record.metadata.attributes.get("deletion", {}))
        provenance["mode"] = "soft"
        provenance["soft_deleted_at"] = _utcnow().isoformat()
        if expiry is not None:
            provenance["undo_expires_at"] = expiry.isoformat()
            provenance["undo_window_seconds"] = int(max(0, undo_window or 0))
        attributes = dict(record.metadata.attributes)
        attributes["deletion"] = provenance
        record = replace(record, metadata=replace(record.metadata, attributes=attributes))
        self._deleted_records[record_id] = (record, expiry)

    def fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        now = _utcnow()
        results: List[Tuple[float, MemoryRecord]] = []
        expired: List[str] = []

        for key, record in list(self._records.items()):
            if record.metadata.is_expired(now):
                expired.append(key)
                continue

            if not record.metadata.matches(query.metadata_filters):
                continue

            score = self._score_record(record, query)
            if query.min_score is not None and (score is None or score < query.min_score):
                continue

            results.append((score or 0.0, record.with_score(score)))

        for key in expired:
            self._records.pop(key, None)

        results.sort(key=lambda item: item[0], reverse=True)
        limited = [record for _, record in results[: max(0, query.limit)]]
        return limited

    def compact(self) -> None:
        now = _utcnow()
        expired = [key for key, record in self._records.items() if record.metadata.is_expired(now)]
        for key in expired:
            self._records.pop(key, None)
        for record_id, (_, expiry) in list(self._deleted_records.items()):
            if expiry is not None and expiry <= now:
                self._deleted_records.pop(record_id, None)
        self._compaction_counter = 0

    def get(self, record_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        if record_id in self._records:
            return self._records[record_id]
        if include_deleted and record_id in self._deleted_records:
            record, expiry = self._deleted_records[record_id]
            if expiry is None or expiry > _utcnow():
                return record
        raise KeyError(f"Memory record '{record_id}' does not exist")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _prepare_record(self, record: MemoryRecord) -> MemoryRecord:
        record = _ensure_embedding(
            record,
            self._embedder_fn,
            self._embedder_model_name or self._embedding_model,
        )
        attributes = dict(record.metadata.attributes)
        attributes.pop("deletion", None)
        metadata = replace(
            record.metadata,
            tags=tuple(record.metadata.tags),
            attributes=attributes,
        )
        if metadata.ttl_seconds is None and self._default_ttl is not None:
            metadata.ttl_seconds = self._default_ttl
        if metadata.embedding_model is None:
            metadata.embedding_model = self._embedding_model
        metadata.touch()

        embedding = list(record.embedding) if record.embedding is not None else None
        return replace(record, metadata=metadata, embedding=embedding)

    def _maybe_compact(self) -> None:
        if not self._compaction_threshold:
            return
        self._compaction_counter += 1
        if self._compaction_counter >= self._compaction_threshold:
            self.compact()

    @staticmethod
    def _score_record(record: MemoryRecord, query: MemoryQuery) -> Optional[float]:
        text_score: Optional[float] = None
        embedding_score: Optional[float] = None

        if query.text:
            if not record.content:
                text_score = 0.0
            else:
                text_lower = query.text.lower()
                content_lower = record.content.lower()
                if text_lower in content_lower:
                    text_score = len(text_lower) / max(len(content_lower), 1)
                else:
                    text_score = 0.0

        if query.embedding and record.embedding:
            embedding_score = InMemoryMemoryStore._cosine_similarity(query.embedding, record.embedding)

        if text_score is None and embedding_score is None:
            return record.score
        if text_score is None:
            return embedding_score
        if embedding_score is None:
            return text_score
        return (text_score + embedding_score) / 2.0

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if len(vec_a) != len(vec_b):
            return 0.0
        numerator = sum(a * b for a, b in zip(vec_a, vec_b))
        denom_a = math.sqrt(sum(a * a for a in vec_a))
        denom_b = math.sqrt(sum(b * b for b in vec_b))
        if denom_a == 0.0 or denom_b == 0.0:
            return 0.0
        return numerator / (denom_a * denom_b)


class ShortTermMemoryStore(MemoryStore):
    """Redis-backed short term memory store leveraging hashes and sorted sets."""

    backend_name = "redis"

    def __init__(
        self,
        config: "StoreConfig",
        *,
        redis_client: Optional["redis.Redis"] = None,
    ) -> None:
        if redis is None:  # pragma: no cover - requires redis dependency at runtime
            raise RuntimeError(
                "Redis backend requested but redis-py is not installed. Install 'redis' to enable the short-term store."
            )

        self.config = config
        settings = config.redis or RedisSettings()
        self._client = redis_client or self._create_client(settings)

        self._default_ttl = self._coerce_int(config.ttl_seconds)
        self._embedding_model = config.embedding_model
        self._embedder_fn, self._embedder_model_name = _load_embedding_function(self._embedding_model)

        options = dict(config.options)
        namespace = options.get("namespace") or f"memory:{config.scope}"
        self._namespace = namespace.rstrip(":")
        self._session_ttl = self._coerce_int(options.get("session_ttl"))
        if self._session_ttl is None:
            self._session_ttl = self._default_ttl
        self._vector_index_name = options.get("vector_index") or options.get("redis_vector_index")
        self._supports_vector = bool(self._vector_index_name and Query is not None and hasattr(self._client, "ft"))

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record = self._prepare_record(record)
        return self._upsert(record)

    def update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        state = self._load_state(record_id, include_deleted=True)
        if state is None:
            raise KeyError(f"Memory record '{record_id}' does not exist")

        record = state.record
        if content is not None:
            record = replace(record, content=content)
        if metadata is not None:
            record = replace(record, metadata=metadata)
        if embedding is not None:
            record = replace(record, embedding=list(embedding))
        if score is not None:
            record = replace(record, score=score)

        record = self._prepare_record(record)
        return self._upsert(record)

    def delete(self, record_id: str, *, hard: bool = False, undo_window: Optional[int] = None) -> None:
        if hard:
            state = self._load_state(record_id, include_deleted=True)
            if state is None:
                return
            self._purge_record(record_id, state)
            return
        self.soft_delete(record_id, undo_window=undo_window)

    def get(self, record_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        state = self._load_state(record_id, include_deleted=include_deleted)
        if state is None:
            raise KeyError(f"Memory record '{record_id}' does not exist")
        return state.record

    def fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        if query.embedding:
            vector_hits = self._vector_search(query.embedding, max(query.limit, 1))
            if vector_hits is not None:
                ordered_ids = [hit[0] for hit in vector_hits]
                states = self._load_states(ordered_ids)
                results: List[Tuple[float, MemoryRecord]] = []
                for state in states:
                    if not state.record.metadata.matches(query.metadata_filters):
                        continue
                    score = InMemoryMemoryStore._score_record(state.record, query)
                    if query.min_score is not None and (score is None or score < query.min_score):
                        continue
                    results.append((score or 0.0, state.record.with_score(score)))
                results.sort(key=lambda item: item[0], reverse=True)
                return [record for _, record in results[: query.limit]]

        filters = query.metadata_filters
        session_filter = filters.get("session_id") or filters.get("session")
        agent_filter = filters.get("agent_id") or filters.get("agent")
        tags_filter = filters.get("tags")

        ordered_ids, _ = self._candidate_record_ids(session_filter, agent_filter, tags_filter)
        if not ordered_ids and not (session_filter or agent_filter or tags_filter):
            ordered_ids, _ = self._candidate_record_ids(None, None, None)

        states = self._load_states(ordered_ids)
        results: List[Tuple[float, MemoryRecord]] = []
        for state in states:
            if not state.record.metadata.matches(filters):
                continue
            score = InMemoryMemoryStore._score_record(state.record, query)
            if query.min_score is not None and (score is None or score < query.min_score):
                continue
            results.append((score or 0.0, state.record.with_score(score)))

        results.sort(key=lambda item: item[0], reverse=True)
        limited = [record for _, record in results[: max(0, query.limit)]]
        return limited

    def compact(self) -> None:
        if redis is None:  # pragma: no cover - defensive
            return
        for key in self._client.scan_iter(match=f"{self._namespace}:record:*", count=500):
            record_id = self._record_id_from_key(key)
            state = self._load_state(record_id, include_deleted=True)
            if state is None:
                continue
            if state.deleted or state.record.metadata.is_expired():
                self._purge_record(record_id, state)

    # ------------------------------------------------------------------
    # Extended operations for Redis-backed store
    # ------------------------------------------------------------------

    def soft_delete(self, record_id: str, *, undo_window: Optional[int] = None) -> None:
        if redis is None:  # pragma: no cover
            return
        record_key = self._record_key(record_id)
        while True:
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(record_key)
                    raw = pipe.hgetall(record_key)
                    if not raw:
                        pipe.unwatch()
                        return
                    data = self._decode_hash(raw)
                    state = self._state_from_data(record_id, data)
                    if state is None:
                        pipe.unwatch()
                        return
                    metadata_dict = _metadata_to_dict(state.record.metadata)
                    attributes = metadata_dict.setdefault("attributes", {})
                    deletion_meta = dict(attributes.get("deletion", {}))
                    now = _utcnow()
                    deletion_meta["mode"] = "soft"
                    deletion_meta["soft_deleted_at"] = now.isoformat()
                    expiry: Optional[datetime] = None
                    if undo_window is not None:
                        expiry = now + timedelta(seconds=max(0, undo_window))
                        deletion_meta["undo_expires_at"] = expiry.isoformat()
                        deletion_meta["undo_window_seconds"] = int(max(0, undo_window))
                    attributes["deletion"] = deletion_meta
                    metadata_dict["attributes"] = attributes
                    pipe.multi()
                    mapping: Dict[str, Any] = {
                        "deleted": "1",
                        "deleted_at": now.isoformat(),
                        "metadata": json.dumps(metadata_dict),
                    }
                    if undo_window is not None:
                        mapping["undo_expires_at"] = deletion_meta.get("undo_expires_at")
                    pipe.hset(record_key, mapping=mapping)
                    if state.session_id:
                        pipe.zrem(self._session_key(state.session_id), record_id)
                    if state.agent_id:
                        pipe.srem(self._agent_key(state.agent_id), record_id)
                    for tag in state.tags:
                        pipe.srem(self._tag_key(tag), record_id)
                    pipe.execute()
                    break
            except WatchError:  # pragma: no cover - retry on contention
                continue

    def refresh_ttl(self, record_id: str, ttl_seconds: Optional[int] = None) -> None:
        state = self._load_state(record_id, include_deleted=True)
        if state is None:
            return
        ttl = self._coerce_int(ttl_seconds) or state.record.metadata.ttl_seconds or self._default_ttl
        if ttl is None:
            return
        session_ttl = self._session_ttl or ttl
        with self._client.pipeline() as pipe:
            pipe.expire(self._record_key(record_id), ttl)
            if state.session_id:
                pipe.expire(self._session_key(state.session_id), session_ttl)
            if state.agent_id:
                pipe.expire(self._agent_key(state.agent_id), ttl)
            for tag in state.tags:
                pipe.expire(self._tag_key(tag), ttl)
            pipe.execute()

    def promote_to_long_term(
        self,
        record_id: str,
        long_term_store: "LongTermMemoryStore",
        *,
        strategy: str = "move",
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecord:
        state = self._load_state(record_id, include_deleted=False)
        if state is None:
            raise KeyError(f"Memory record '{record_id}' does not exist")

        record = state.record
        metadata_payload = _metadata_to_dict(record.metadata)
        attributes = metadata_payload.setdefault("attributes", {})
        provenance_payload = dict(attributes.get("provenance", {}))
        provenance_payload.setdefault("source_backend", self.backend_name)
        provenance_payload.setdefault("source_record_id", record_id)
        provenance_payload.setdefault("strategy", strategy)
        provenance_payload.setdefault("promoted_at", _utcnow().isoformat())
        if provenance:
            provenance_payload.update({str(key): value for key, value in provenance.items()})
        attributes["provenance"] = provenance_payload
        metadata_payload["attributes"] = attributes
        metadata = _metadata_from_dict(metadata_payload)
        enriched = replace(record, metadata=metadata)

        stored = long_term_store.add(enriched)
        audit_operation = "promote" if strategy.lower() == "move" else "copy_to_long_term"
        actor = provenance_payload.get("actor")
        try:
            long_term_store.record_audit_event(
                stored.record_id,
                audit_operation,
                metadata=_metadata_to_dict(stored.metadata),
                content=stored.content,
                embedding=stored.embedding,
                score=stored.score,
                changed_by=str(actor) if actor is not None else None,
            )
        except AttributeError:
            LOGGER.debug("Long-term store does not expose audit logging; skipping promote audit")

        if strategy.lower() == "move":
            self.delete(record_id, hard=True)
        return stored

    def copy_to_long_term(
        self,
        record_id: str,
        long_term_store: "LongTermMemoryStore",
        *,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecord:
        return self.promote_to_long_term(
            record_id,
            long_term_store,
            strategy="copy",
            provenance=provenance,
        )

    def list_entries(
        self,
        *,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        include_deleted: bool = False,
    ) -> List[MemoryRecord]:
        ordered_ids, _ = self._candidate_record_ids(session_id, agent_id, tags)
        states = self._load_states(ordered_ids, include_deleted=include_deleted)
        states.sort(key=lambda item: item.record.metadata.updated_at, reverse=True)
        if limit is not None:
            states = states[:limit]
        return [state.record for state in states]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_record(self, record: MemoryRecord) -> MemoryRecord:
        record = _ensure_embedding(
            record,
            self._embedder_fn,
            self._embedder_model_name or self._embedding_model,
        )
        metadata = replace(
            record.metadata,
            tags=tuple(record.metadata.tags),
            attributes=dict(record.metadata.attributes),
        )
        if metadata.ttl_seconds is None and self._default_ttl is not None:
            metadata.ttl_seconds = self._default_ttl
        if metadata.embedding_model is None:
            metadata.embedding_model = self._embedding_model
        metadata.touch()
        embedding = list(record.embedding) if record.embedding is not None else None
        return replace(record, metadata=metadata, embedding=embedding)

    def _upsert(self, record: MemoryRecord) -> MemoryRecord:
        record_key = self._record_key(record.record_id)
        session_id = self._extract_session_id(record.metadata)
        agent_id = self._extract_agent_id(record.metadata)
        tags = tuple(str(tag) for tag in record.metadata.tags)
        mapping = self._serialize_record(record, session_id, agent_id)
        ttl_seconds = record.metadata.ttl_seconds or self._default_ttl
        session_ttl = self._session_ttl or ttl_seconds

        while True:
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(record_key)
                    raw = pipe.hgetall(record_key)
                    previous = None
                    if raw:
                        previous = self._state_from_data(record.record_id, self._decode_hash(raw))
                    pipe.multi()
                    pipe.hset(record_key, mapping=mapping)
                    if record.embedding is None:
                        pipe.hdel(record_key, "embedding", "embedding_blob", "embedding_dim")
                    pipe.hdel(record_key, "deleted_at", "undo_expires_at")
                    if ttl_seconds:
                        pipe.expire(record_key, ttl_seconds)
                    if session_id:
                        session_key = self._session_key(session_id)
                        pipe.zadd(session_key, {record.record_id: record.metadata.updated_at.timestamp()})
                        if session_ttl:
                            pipe.expire(session_key, session_ttl)
                    if agent_id:
                        agent_key = self._agent_key(agent_id)
                        pipe.sadd(agent_key, record.record_id)
                        if ttl_seconds:
                            pipe.expire(agent_key, ttl_seconds)
                    for tag in tags:
                        tag_key = self._tag_key(tag)
                        pipe.sadd(tag_key, record.record_id)
                        if ttl_seconds:
                            pipe.expire(tag_key, ttl_seconds)
                    if previous is not None:
                        self._remove_stale_indexes(pipe, record.record_id, previous, session_id, agent_id, tags)
                    pipe.execute()
                    break
            except WatchError:  # pragma: no cover - retry on contention
                continue

        return record

    def _remove_stale_indexes(
        self,
        pipe: "redis.client.Pipeline",
        record_id: str,
        previous: _RedisRecordState,
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Tuple[str, ...],
    ) -> None:
        if previous.session_id and previous.session_id != session_id:
            pipe.zrem(self._session_key(previous.session_id), record_id)
        if previous.agent_id and previous.agent_id != agent_id:
            pipe.srem(self._agent_key(previous.agent_id), record_id)
        previous_tags = set(previous.tags)
        for tag in previous_tags - set(tags):
            pipe.srem(self._tag_key(tag), record_id)

    def _serialize_record(
        self,
        record: MemoryRecord,
        session_id: Optional[str],
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        metadata_dict = _metadata_to_dict(record.metadata)
        payload: Dict[str, Any] = {
            "record_id": record.record_id,
            "content": record.content,
            "metadata": json.dumps(metadata_dict),
            "score": "" if record.score is None else json.dumps(record.score),
            "session_id": session_id or "",
            "agent_id": agent_id or "",
            "tags": json.dumps(list(record.metadata.tags)),
            "deleted": "0",
        }
        if record.embedding is not None:
            payload["embedding"] = json.dumps(list(record.embedding))
            payload["embedding_dim"] = str(len(record.embedding))
            payload["embedding_blob"] = _pack_embedding(record.embedding)
        return payload

    def _vector_search(self, embedding: Sequence[float], limit: int) -> Optional[List[Tuple[str, Optional[float]]]]:
        if not self._supports_vector or not embedding:
            return None
        try:  # pragma: no cover - requires redis search module
            search = self._client.ft(self._vector_index_name)
        except Exception:
            LOGGER.debug("Redis vector index '%s' is unavailable", self._vector_index_name, exc_info=True)
            return None
        if Query is None:
            return None
        try:
            blob = _pack_embedding(embedding)
            query = (
                Query(f"*=>[KNN {max(limit, 1)} @embedding_blob $vec AS score]")
                .sort_by("score")
                .paging(0, max(limit, 1))
                .return_fields("__key", "score")
                .dialect(2)
            )
            results = search.search(query, query_params={"vec": blob})
        except Exception:
            LOGGER.debug("Vector search failed on index '%s'", self._vector_index_name, exc_info=True)
            return None
        hits: List[Tuple[str, Optional[float]]] = []
        docs = getattr(results, "docs", None)
        if not docs:
            return []
        for doc in docs:
            record_id = getattr(doc, "id", None) or getattr(doc, "__key", None)
            if record_id is None:
                continue
            record_id = self._record_id_from_key(record_id)
            score_value = getattr(doc, "score", None)
            try:
                score_float = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score_float = None
            hits.append((record_id, score_float))
        return hits

    def _candidate_record_ids(
        self,
        session_id: Optional[str],
        agent_id: Optional[str],
        tags: Optional[Iterable[str]],
    ) -> Tuple[List[str], Set[str]]:
        ordered: List[str] = []
        candidates: Optional[Set[str]] = None

        if session_id:
            session_key = self._session_key(str(session_id))
            ordered_bytes = self._client.zrevrange(session_key, 0, -1)
            ordered = [self._ensure_str(item) for item in ordered_bytes]
            candidates = set(ordered)

        if agent_id:
            agent_members = {self._ensure_str(item) for item in self._client.smembers(self._agent_key(str(agent_id)))}
            candidates = agent_members if candidates is None else candidates & agent_members

        if tags:
            tag_values: List[str]
            if isinstance(tags, str):
                tag_values = [tags]
            else:
                try:
                    tag_values = [str(tag) for tag in tags]
                except TypeError:
                    tag_values = [str(tags)]
            for tag in tag_values:
                tag_members = {self._ensure_str(item) for item in self._client.smembers(self._tag_key(tag))}
                candidates = tag_members if candidates is None else candidates & tag_members

        if candidates is None:
            candidates = set()
            ordered = []
            for key in self._client.scan_iter(match=f"{self._namespace}:record:*", count=500):
                record_id = self._record_id_from_key(key)
                candidates.add(record_id)
                ordered.append(record_id)
        elif not ordered:
            ordered = sorted(candidates)
        else:
            ordered = [record_id for record_id in ordered if record_id in candidates]

        return ordered, candidates

    def _load_state(self, record_id: str, *, include_deleted: bool = False) -> Optional[_RedisRecordState]:
        raw = self._client.hgetall(self._record_key(record_id))
        if not raw:
            return None
        state = self._state_from_data(record_id, self._decode_hash(raw))
        if state is None:
            return None
        if state.record.metadata.is_expired():
            self._purge_record(record_id, state)
            return None
        if state.deleted and not include_deleted:
            return None
        return state

    def _load_states(
        self,
        record_ids: Sequence[str],
        *,
        include_deleted: bool = False,
    ) -> List[_RedisRecordState]:
        ids = list(record_ids)
        if not ids:
            return []
        pipe = self._client.pipeline()
        for record_id in ids:
            pipe.hgetall(self._record_key(record_id))
        raw_results = pipe.execute()
        now = _utcnow()
        states: List[_RedisRecordState] = []
        for record_id, raw in zip(ids, raw_results):
            if not raw:
                continue
            state = self._state_from_data(record_id, self._decode_hash(raw))
            if state is None:
                continue
            if state.record.metadata.is_expired(now):
                self._purge_record(record_id, state)
                continue
            if state.deleted and not include_deleted:
                continue
            states.append(state)
        return states

    def _state_from_data(self, record_id: str, data: Mapping[str, Any]) -> Optional[_RedisRecordState]:
        if not data:
            return None
        metadata_raw = data.get("metadata")
        try:
            metadata_payload = json.loads(metadata_raw) if metadata_raw else {}
        except (TypeError, json.JSONDecodeError):
            metadata_payload = {}
        metadata = _metadata_from_dict(metadata_payload)
        if metadata.embedding_model is None:
            metadata.embedding_model = self._embedding_model
        attributes = dict(metadata.attributes)
        deletion_info = dict(attributes.get("deletion", {}))
        if str(data.get("deleted", "0")) == "1":
            deletion_info.setdefault("mode", "soft")
            deleted_at_raw = data.get("deleted_at")
            if deleted_at_raw:
                deletion_info.setdefault("soft_deleted_at", str(deleted_at_raw))
            undo_expires = data.get("undo_expires_at")
            if undo_expires:
                deletion_info["undo_expires_at"] = str(undo_expires)
        else:
            deletion_info.pop("mode", None)
        if deletion_info:
            attributes["deletion"] = deletion_info
            metadata = replace(metadata, attributes=attributes)
        embedding_json = data.get("embedding")
        embedding: Optional[List[float]] = None
        if embedding_json not in (None, "", "null"):
            try:
                embedding = list(json.loads(embedding_json))
            except json.JSONDecodeError:
                embedding = None
        score_raw = data.get("score")
        score: Optional[float]
        if score_raw in (None, "", "null"):
            score = None
        else:
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = None
        record = MemoryRecord(
            content=str(data.get("content", "")),
            metadata=metadata,
            embedding=embedding,
            score=score,
            record_id=record_id,
        )
        deleted = str(data.get("deleted", "0")) == "1"
        session_id = data.get("session_id") or metadata.attributes.get("session_id") or metadata.attributes.get("session")
        agent_id = data.get("agent_id") or metadata.attributes.get("agent_id") or metadata.attributes.get("agent")
        tags = tuple(str(tag) for tag in metadata.tags)
        return _RedisRecordState(
            record=record,
            session_id=str(session_id) if session_id else None,
            agent_id=str(agent_id) if agent_id else None,
            tags=tags,
            deleted=deleted,
        )

    def _purge_record(self, record_id: str, state: _RedisRecordState) -> None:
        with self._client.pipeline() as pipe:
            pipe.delete(self._record_key(record_id))
            if state.session_id:
                pipe.zrem(self._session_key(state.session_id), record_id)
            if state.agent_id:
                pipe.srem(self._agent_key(state.agent_id), record_id)
            for tag in state.tags:
                pipe.srem(self._tag_key(tag), record_id)
            pipe.execute()

    def _decode_hash(self, raw: Mapping[Any, Any]) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for key, value in raw.items():
            field = self._ensure_str(key)
            if isinstance(value, (bytes, bytearray)):
                if field == "embedding_blob":
                    decoded[field] = bytes(value)
                else:
                    decoded[field] = value.decode()
            else:
                decoded[field] = value
        return decoded

    def _record_key(self, record_id: str) -> str:
        return f"{self._namespace}:record:{record_id}"

    def _session_key(self, session_id: str) -> str:
        return f"{self._namespace}:session:{session_id}"

    def _agent_key(self, agent_id: str) -> str:
        return f"{self._namespace}:agent:{agent_id}"

    def _tag_key(self, tag: str) -> str:
        return f"{self._namespace}:tag:{tag}"

    @staticmethod
    def _ensure_str(value: Any) -> str:
        if isinstance(value, (bytes, bytearray)):
            return value.decode()
        return str(value)

    def _record_id_from_key(self, key: Any) -> str:
        identifier = self._ensure_str(key)
        if ":" in identifier:
            return identifier.rsplit(":", 1)[-1]
        return identifier

    @staticmethod
    def _extract_session_id(metadata: MemoryMetadata) -> Optional[str]:
        for candidate in ("session_id", "session", "conversation_id"):
            value = metadata.attributes.get(candidate)
            if value:
                return str(value)
        return None

    @staticmethod
    def _extract_agent_id(metadata: MemoryMetadata) -> Optional[str]:
        for candidate in ("agent_id", "agent", "assistant_id"):
            value = metadata.attributes.get(candidate)
            if value:
                return str(value)
        return None

    @staticmethod
    def _coerce_int(value: Optional[Any]) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _create_client(settings: RedisSettings) -> "redis.Redis":
        kwargs = dict(settings.options)
        if settings.url:
            return redis.Redis.from_url(settings.url, decode_responses=False, **kwargs)
        if settings.username is not None:
            kwargs["username"] = settings.username
        if settings.password is not None:
            kwargs["password"] = settings.password
        kwargs.setdefault("host", settings.host)
        kwargs.setdefault("port", settings.port)
        kwargs.setdefault("db", settings.db)
        if settings.ssl is not None:
            kwargs["ssl"] = settings.ssl
        return redis.Redis(**kwargs)


class RedisMemoryStore(ShortTermMemoryStore):
    """Backward compatible alias for the Redis-backed store."""


class LongTermMemoryStore(MemoryStore):
    """PostgreSQL-backed implementation designed for durable storage."""

    backend_name = "postgres"

    def __init__(self, config: "StoreConfig") -> None:
        if AsyncConnectionPool is None or sql is None or dict_row is None:
            raise RuntimeError(
                "psycopg3 with connection pooling is required for the PostgreSQL memory store"
            )

        self.config = config
        self._settings = config.postgres or PostgresSettings()
        self._embedding_model = config.embedding_model
        self._embedder_fn, self._embedder_model_name = _load_embedding_function(self._embedding_model)
        self._embedding_dimensions = self._option_int("embedding_dimensions", 1536)
        self._pool = AsyncConnectionPool(
            conninfo=self._build_conninfo(self._settings),
            min_size=self._option_int("pool_min_size", 1),
            max_size=self._option_int("pool_max_size", 10),
            timeout=self._option_float("pool_timeout", 30.0),
            kwargs=self._build_conn_kwargs(self._settings),
            open=False,
        )
        self._run(self._initialize_pool())

    # ------------------------------------------------------------------
    # Helper methods for configuration and connection management
    # ------------------------------------------------------------------

    def _option_int(self, key: str, default: int) -> int:
        try:
            return int(self.config.options.get(key, default))
        except (TypeError, ValueError, AttributeError):
            return default

    def _option_float(self, key: str, default: float) -> float:
        try:
            raw = self.config.options.get(key, default)
        except AttributeError:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_conn_kwargs(settings: "PostgresSettings") -> Dict[str, Any]:
        kwargs = dict(settings.options)
        if settings.user:
            kwargs.setdefault("user", settings.user)
        if settings.password:
            kwargs.setdefault("password", settings.password)
        if settings.host:
            kwargs.setdefault("host", settings.host)
        if settings.port:
            kwargs.setdefault("port", settings.port)
        if settings.database:
            kwargs.setdefault("dbname", settings.database)
        if settings.sslmode:
            kwargs.setdefault("sslmode", settings.sslmode)
        return kwargs

    @staticmethod
    def _build_conninfo(settings: "PostgresSettings") -> str:
        if settings.dsn:
            return settings.dsn

        parts: List[str] = []
        if settings.host:
            parts.append(f"host={settings.host}")
        if settings.port:
            parts.append(f"port={settings.port}")
        if settings.database:
            parts.append(f"dbname={settings.database}")
        if settings.user:
            parts.append(f"user={settings.user}")
        if settings.password:
            parts.append(f"password={settings.password}")
        if settings.sslmode:
            parts.append(f"sslmode={settings.sslmode}")
        for key, value in settings.options.items():
            parts.append(f"{key}={value}")
        return " ".join(parts)

    async def _initialize_pool(self) -> None:
        await self._pool.open(wait=True)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record = _ensure_embedding(
            record,
            self._embedder_fn,
            self._embedder_model_name or self._embedding_model,
        )
        return self._run(self._add(record))

    async def _add(self, record: MemoryRecord) -> MemoryRecord:
        metadata = self._prepare_metadata(record.metadata)
        now = _utcnow()
        created_at = metadata.created_at or now
        updated_at = metadata.updated_at or now
        metadata.created_at = created_at
        metadata.updated_at = updated_at
        expires_at = self._compute_expiry(metadata, created_at)
        embedding_value = self._format_embedding(record.embedding)
        project_id = self._extract_project(metadata)
        context_id = self._extract_context(metadata)
        tags = self._normalize_tags(metadata.tags)

        score_value = float(record.score) if record.score is not None else None
        importance_value = float(metadata.importance) if metadata.importance is not None else None

        payload = {
            "id": record.record_id,
            "project_id": project_id,
            "context_id": context_id,
            "scope": self.config.scope,
            "content": record.content,
            "metadata": json.dumps(_metadata_to_dict(metadata)),
            "embedding": embedding_value,
            "embedding_model": metadata.embedding_model,
            "score": score_value,
            "importance": importance_value,
            "ttl_seconds": metadata.ttl_seconds,
            "expires_at": expires_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    insert_sql = sql.SQL(
                        """
                        INSERT INTO memory_entries (
                            id, project_id, context_id, scope, content, metadata,
                            embedding, embedding_model, score, importance,
                            ttl_seconds, expires_at, created_at, updated_at
                        )
                        VALUES (
                            %(id)s, %(project_id)s, %(context_id)s, %(scope)s, %(content)s,
                            %(metadata)s::jsonb, {embedding}, %(embedding_model)s,
                            %(score)s, %(importance)s, %(ttl_seconds)s, %(expires_at)s,
                            %(created_at)s, %(updated_at)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            project_id = EXCLUDED.project_id,
                            context_id = EXCLUDED.context_id,
                            scope = EXCLUDED.scope,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            embedding_model = EXCLUDED.embedding_model,
                            score = EXCLUDED.score,
                            importance = EXCLUDED.importance,
                            ttl_seconds = EXCLUDED.ttl_seconds,
                            expires_at = EXCLUDED.expires_at,
                            updated_at = EXCLUDED.updated_at,
                            is_deleted = FALSE,
                            deleted_at = NULL,
                            version = memory_entries.version + 1
                        RETURNING version
                        """
                    ).format(
                        embedding=sql.SQL("%(embedding)s::vector") if embedding_value is not None else sql.SQL("NULL")
                    )
                    await cur.execute(insert_sql, payload)
                    version_row = await cur.fetchone()
                    version = version_row[0] if version_row else 1
                    await self._upsert_tags(conn, record.record_id, tags)
                    previous_version = version - 1 if version > 1 else None
                    history_operation = "insert" if version == 1 else "update"
                    await self._record_history(
                        conn,
                        record_id=record.record_id,
                        operation=history_operation,
                        version=version,
                        previous_version=previous_version,
                        content=record.content,
                        metadata=_metadata_to_dict(metadata),
                        embedding=record.embedding,
                        score=score_value,
                        changed_by=self._resolve_actor(metadata),
                    )

        return await self._load_record(record.record_id)

    def update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        return self._run(
            self._update(
                record_id,
                content=content,
                metadata=metadata,
                embedding=embedding,
                score=score,
            )
        )

    async def _update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                current = await self._fetch_entry(conn, record_id, for_update=True)
                if current is None:
                    raise KeyError(f"Memory record '{record_id}' does not exist")

                current_metadata = self._record_to_metadata(current)
                new_metadata = self._prepare_metadata(metadata or current_metadata)
                if metadata is None:
                    new_metadata.created_at = current_metadata.created_at

                new_content = content if content is not None else current["content"]
                new_embedding = list(embedding) if embedding is not None else current.get("embedding")
                if new_embedding is not None and not isinstance(new_embedding, list):
                    new_embedding = list(new_embedding)
                new_score = score if score is not None else current.get("score")
                if new_score is not None:
                    try:
                        new_score = float(new_score)
                    except (TypeError, ValueError):
                        new_score = None
                candidate_record = MemoryRecord(
                    content=new_content,
                    metadata=new_metadata,
                    embedding=new_embedding,
                    score=new_score,
                    record_id=record_id,
                )
                candidate_record = _ensure_embedding(
                    candidate_record,
                    self._embedder_fn,
                    self._embedder_model_name or self._embedding_model,
                )
                new_metadata = candidate_record.metadata
                new_embedding = candidate_record.embedding
                new_score = candidate_record.score
                expires_at = self._compute_expiry(new_metadata, new_metadata.created_at)
                tags = self._normalize_tags(new_metadata.tags)
                actor = self._resolve_actor(new_metadata)
                version = int(current["version"]) + 1
                importance_value = (
                    float(new_metadata.importance) if new_metadata.importance is not None else None
                )

                payload = {
                    "id": record_id,
                    "project_id": self._extract_project(new_metadata),
                    "context_id": self._extract_context(new_metadata),
                    "scope": self.config.scope,
                    "content": new_content,
                    "metadata": json.dumps(_metadata_to_dict(new_metadata)),
                    "embedding": self._format_embedding(new_embedding),
                    "embedding_model": new_metadata.embedding_model,
                    "score": new_score,
                    "importance": importance_value,
                    "ttl_seconds": new_metadata.ttl_seconds,
                    "expires_at": expires_at,
                    "updated_at": new_metadata.updated_at,
                    "version": version,
                }

                async with conn.cursor() as cur:
                    update_sql = sql.SQL(
                        """
                        UPDATE memory_entries
                        SET project_id = %(project_id)s,
                            context_id = %(context_id)s,
                            scope = %(scope)s,
                            content = %(content)s,
                            metadata = %(metadata)s::jsonb,
                            embedding = {embedding},
                            embedding_model = %(embedding_model)s,
                            score = %(score)s,
                            importance = %(importance)s,
                            ttl_seconds = %(ttl_seconds)s,
                            expires_at = %(expires_at)s,
                            updated_at = %(updated_at)s,
                            version = %(version)s,
                            is_deleted = FALSE,
                            deleted_at = NULL
                        WHERE id = %(id)s
                        """
                    ).format(
                        embedding=sql.SQL("%(embedding)s::vector") if payload["embedding"] is not None else sql.SQL("embedding")
                    )
                    await cur.execute(update_sql, payload)

                await self._sync_tags(conn, record_id, tags)
                await self._record_history(
                    conn,
                    record_id=record_id,
                    operation="update",
                    version=version,
                    previous_version=current["version"],
                    content=new_content,
                    metadata=_metadata_to_dict(new_metadata),
                    embedding=new_embedding,
                    score=new_score,
                    changed_by=actor,
                )

        return await self._load_record(record_id)

    def delete(
        self,
        record_id: str,
        *,
        hard: bool = False,
        undo_window: Optional[int] = None,
    ) -> None:
        self._run(self._delete(record_id, hard=hard, undo_window=undo_window))

    def get(self, record_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        return self._run(self._get(record_id, include_deleted=include_deleted))

    def record_audit_event(
        self,
        record_id: str,
        operation: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
        changed_by: Optional[str] = None,
    ) -> None:
        self._run(
            self._record_audit_event_async(
                record_id,
                operation,
                content=content,
                metadata=metadata,
                embedding=embedding,
                score=score,
                changed_by=changed_by,
            )
        )

    async def _delete(self, record_id: str, *, hard: bool, undo_window: Optional[int]) -> None:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                current = await self._fetch_entry(conn, record_id, for_update=True)
                if current is None:
                    raise KeyError(f"Memory record '{record_id}' does not exist")

                now = _utcnow()
                metadata = self._record_to_metadata(current)
                actor = self._resolve_actor(metadata)
                metadata_dict = _metadata_to_dict(metadata)
                attributes = metadata_dict.setdefault("attributes", {})
                deletion_meta = dict(attributes.get("deletion", {}))
                deletion_meta["deleted_at"] = now.isoformat()
                deletion_meta["mode"] = "hard" if hard else "soft"
                if not hard and undo_window is not None:
                    expiry = now + timedelta(seconds=max(0, undo_window))
                    deletion_meta["undo_expires_at"] = expiry.isoformat()
                    deletion_meta["undo_window_seconds"] = int(max(0, undo_window))
                attributes["deletion"] = deletion_meta
                metadata_dict["attributes"] = attributes

                async with conn.cursor() as cur:
                    if hard:
                        await cur.execute(
                            """
                            UPDATE memory_entry_history
                            SET embedding = NULL,
                                content = NULL
                            WHERE entry_id = %(id)s
                            """,
                            {"id": record_id},
                        )
                        await cur.execute(
                            """
                            UPDATE memory_entries
                            SET is_deleted = TRUE,
                                deleted_at = %(now)s,
                                updated_at = %(now)s,
                                content = NULL,
                                embedding = NULL,
                                score = NULL,
                                metadata = %(metadata)s::jsonb
                            WHERE id = %(id)s
                            """,
                            {"id": record_id, "now": now, "metadata": json.dumps(metadata_dict)},
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE memory_entries
                            SET is_deleted = TRUE,
                                deleted_at = %(now)s,
                                updated_at = %(now)s,
                                metadata = %(metadata)s::jsonb
                            WHERE id = %(id)s
                            """,
                            {"id": record_id, "now": now, "metadata": json.dumps(metadata_dict)},
                        )
                    await cur.execute(
                        """
                        UPDATE memory_tags
                        SET is_deleted = TRUE,
                            deleted_at = %(now)s,
                            updated_at = %(now)s
                        WHERE entry_id = %(id)s AND is_deleted = FALSE
                        """,
                        {"id": record_id, "now": now},
                    )

                await self._record_history(
                    conn,
                    record_id=record_id,
                    operation="hard_delete" if hard else "soft_delete",
                    version=current["version"],
                    previous_version=current["version"],
                    content=None if hard else current.get("content"),
                    metadata=metadata_dict,
                    embedding=None if hard else current.get("embedding"),
                    score=None if hard else current.get("score"),
                    changed_by=actor,
                )

    def fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        return self._run(self._fetch(query))

    async def _fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        limit = max(1, query.limit)
        offset = max(0, query.offset)
        filters = dict(query.metadata_filters)

        limit_override = filters.pop("page_size", filters.pop("limit", None))
        if limit_override is not None:
            try:
                limit = max(1, min(limit, int(limit_override)))
            except (TypeError, ValueError):
                pass

        offset_override = filters.pop("offset", None)
        if offset_override is not None:
            try:
                offset = max(0, int(offset_override))
            except (TypeError, ValueError):
                pass

        page = filters.pop("page", None)
        if page is not None:
            try:
                page_value = max(0, int(page))
                offset = page_value * limit
            except (TypeError, ValueError):
                pass

        params: Dict[str, Any] = {
            "scope": self.config.scope,
            "limit": limit,
            "offset": offset,
        }

        where_clauses = [
            "e.is_deleted = FALSE",
            "(e.expires_at IS NULL OR e.expires_at > NOW())",
            "e.scope = %(scope)s",
        ]

        if query.text:
            params["text"] = f"%{query.text}%"
            where_clauses.append("e.content ILIKE %(text)s")

        project = filters.pop("project", filters.pop("project_id", None))
        if project is not None:
            params["project_id"] = str(project)
            where_clauses.append("e.project_id = %(project_id)s")

        context = filters.pop("context", filters.pop("context_id", None))
        if context is not None:
            params["context_id"] = str(context)
            where_clauses.append("e.context_id = %(context_id)s")

        tags_filter = filters.pop("tags", None)
        if tags_filter is not None:
            if isinstance(tags_filter, str):
                tag_values = (tags_filter,)
            else:
                tag_values = tags_filter
            normalized_tags = self._normalize_tags(tag_values)
            if normalized_tags:
                params["tags_filter"] = list(normalized_tags)
                params["tags_required"] = len(normalized_tags)
                where_clauses.append(
                    "(SELECT COUNT(DISTINCT mt.tag) FROM memory_tags mt "
                    "WHERE mt.entry_id = e.id AND mt.is_deleted = FALSE AND mt.tag = ANY(%(tags_filter)s)) >= %(tags_required)s"
                )

        attribute_conditions: List[str] = []
        for index, (key, value) in enumerate(filters.items()):
            if value is None:
                continue
            param_key = f"attr_key_{index}"
            params[param_key] = str(key)
            if isinstance(value, (list, tuple, set, frozenset)):
                param_val = f"attr_val_{index}"
                params[param_val] = [str(item) for item in value]
                attribute_conditions.append(
                    f"(e.metadata -> 'attributes' ->> %({param_key})s) = ANY(%({param_val})s)"
                )
            else:
                param_val = f"attr_val_{index}"
                params[param_val] = str(value)
                attribute_conditions.append(
                    f"(e.metadata -> 'attributes' ->> %({param_key})s) = %({param_val})s"
                )

        where_clauses.extend(attribute_conditions)

        similarity_expr = "NULL AS similarity"
        embedding_value = self._format_embedding(query.embedding)
        if embedding_value is not None:
            params["embedding"] = embedding_value
            similarity_expr = (
                "CASE WHEN e.embedding IS NULL THEN NULL "
                "ELSE 1.0 / (1.0 + (e.embedding <-> %(embedding)s::vector)) END AS similarity"
            )

        base_query = [
            "WITH base AS (",
            "    SELECT",
            "        e.id,",
            "        e.project_id,",
            "        e.context_id,",
            "        e.scope,",
            "        e.content,",
            "        e.metadata,",
            "        e.embedding::float4[] AS embedding,",
            "        e.embedding_model,",
            "        e.score,",
            "        e.importance,",
            "        e.ttl_seconds,",
            "        e.expires_at,",
            "        e.is_deleted,",
            "        e.created_at,",
            "        e.updated_at,",
            "        e.deleted_at,",
            "        e.version,",
            f"        {similarity_expr}",
            "    FROM memory_entries e",
            "    WHERE " + " AND ".join(where_clauses),
            ")",
            "SELECT",
            "    b.id,",
            "    b.project_id,",
            "    b.context_id,",
            "    b.scope,",
            "    b.content,",
            "    b.metadata,",
            "    b.embedding,",
            "    b.embedding_model,",
            "    b.score,",
            "    b.importance,",
            "    b.ttl_seconds,",
            "    b.expires_at,",
            "    b.is_deleted,",
            "    b.created_at,",
            "    b.updated_at,",
            "    b.deleted_at,",
            "    b.version,",
            "    b.similarity,",
            "    COALESCE(tags.tags, ARRAY[]::text[]) AS tags",
            "FROM base b",
            "LEFT JOIN LATERAL (",
            "    SELECT array_agg(mt.tag ORDER BY mt.tag) AS tags",
            "    FROM memory_tags mt",
            "    WHERE mt.entry_id = b.id AND mt.is_deleted = FALSE",
            ") tags ON TRUE",
        ]

        if query.min_score is not None and embedding_value is not None:
            params["min_score"] = float(query.min_score)
            base_query.append("WHERE b.similarity IS NULL OR b.similarity >= %(min_score)s")

        order_by_parts = []
        if embedding_value is not None:
            order_by_parts.append("b.similarity DESC NULLS LAST")
        order_by_parts.append("b.updated_at DESC")

        base_query.append("ORDER BY " + ", ".join(order_by_parts))
        base_query.append("LIMIT %(limit)s OFFSET %(offset)s")

        sql_query = "\n".join(base_query)

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql_query, params)
                rows = await cur.fetchall()

        results: List[MemoryRecord] = []
        for row in rows:
            record = self._row_to_record(row)
            if embedding_value is not None and row.get("similarity") is not None:
                record = record.with_score(float(row.get("similarity")))
            elif query.min_score is not None and (record.score or 0.0) < query.min_score:
                continue
            results.append(record)
        return results

    def compact(self) -> None:
        self._run(self._compact())

    async def _compact(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE memory_entries
                    SET is_deleted = TRUE,
                        deleted_at = NOW(),
                        updated_at = NOW()
                    WHERE is_deleted = FALSE
                        AND expires_at IS NOT NULL
                        AND expires_at <= NOW()
                    """
                )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        if loop.is_running():
            raise RuntimeError(
                "LongTermMemoryStore operations must be executed from a thread without an active event loop"
            )
        return loop.run_until_complete(coroutine)

    @staticmethod
    def _normalize_tags(tags: Optional[Iterable[str]]) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for tag in tags or ():
            clean = str(tag).strip()
            if clean and clean not in seen:
                seen[clean] = None
        return tuple(seen.keys())

    def _format_embedding(self, embedding: Optional[Sequence[float]]) -> Optional[str]:
        if embedding is None:
            return None
        values = [float(value) for value in embedding]
        if not values:
            return None
        if self._embedding_dimensions and values and len(values) != self._embedding_dimensions:
            LOGGER.debug(
                "Embedding length %s does not match configured dimension %s", len(values), self._embedding_dimensions
            )
        formatted = ",".join(f"{value:.8f}" for value in values)
        return f"[{formatted}]"

    @staticmethod
    def _compute_expiry(metadata: MemoryMetadata, created_at: datetime) -> Optional[datetime]:
        if metadata.ttl_seconds is None:
            return None
        return created_at + timedelta(seconds=int(metadata.ttl_seconds))

    def _extract_project(self, metadata: MemoryMetadata) -> str:
        attributes = metadata.attributes
        project = (
            attributes.get("project")
            or attributes.get("project_id")
            or attributes.get("workspace")
            or metadata.source
            or self.config.scope
        )
        return str(project)

    def _extract_context(self, metadata: MemoryMetadata) -> Optional[str]:
        attributes = metadata.attributes
        context = (
            attributes.get("context")
            or attributes.get("context_id")
            or attributes.get("session_id")
            or attributes.get("thread")
        )
        return str(context) if context is not None else None

    def _resolve_actor(self, metadata: MemoryMetadata) -> Optional[str]:
        actor = metadata.attributes.get("updated_by") or metadata.attributes.get("author")
        return str(actor) if actor is not None else None

    def _prepare_metadata(self, metadata: MemoryMetadata) -> MemoryMetadata:
        created_at = metadata.created_at or _utcnow()
        cleaned = MemoryMetadata(
            source=metadata.source,
            created_at=created_at,
            updated_at=_utcnow(),
            ttl_seconds=metadata.ttl_seconds,
            tags=self._normalize_tags(metadata.tags),
            importance=metadata.importance,
            embedding_model=metadata.embedding_model,
            attributes=dict(metadata.attributes),
        )
        return cleaned

    async def _get(self, record_id: str, include_deleted: bool) -> MemoryRecord:
        async with self._pool.connection() as conn:
            row = await self._fetch_entry(conn, record_id)
        if row is None or (not include_deleted and row.get("is_deleted")):
            raise KeyError(f"Memory record '{record_id}' does not exist")
        return self._row_to_record(row)

    async def _load_record(self, record_id: str) -> MemoryRecord:
        async with self._pool.connection() as conn:
            row = await self._fetch_entry(conn, record_id)
        if row is None:
            raise KeyError(f"Memory record '{record_id}' was not found after persistence")
        return self._row_to_record(row)

    async def _fetch_entry(
        self, conn: Any, record_id: str, *, for_update: bool = False
    ) -> Optional[Dict[str, Any]]:
        query = [
            "SELECT",
            "    e.id,",
            "    e.project_id,",
            "    e.context_id,",
            "    e.scope,",
            "    e.content,",
            "    e.metadata,",
            "    e.embedding::float4[] AS embedding,",
            "    e.embedding_model,",
            "    e.score,",
            "    e.importance,",
            "    e.ttl_seconds,",
            "    e.expires_at,",
            "    e.is_deleted,",
            "    e.created_at,",
            "    e.updated_at,",
            "    e.deleted_at,",
            "    e.version,",
            "    COALESCE(tags.tags, ARRAY[]::text[]) AS tags",
            "FROM memory_entries e",
            "LEFT JOIN LATERAL (",
            "    SELECT array_agg(mt.tag ORDER BY mt.tag) AS tags",
            "    FROM memory_tags mt",
            "    WHERE mt.entry_id = e.id AND mt.is_deleted = FALSE",
            ") tags ON TRUE",
            "WHERE e.id = %(id)s",
        ]
        if for_update:
            query.append("FOR UPDATE")

        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("\n".join(query), {"id": record_id})
            return await cur.fetchone()

    async def _upsert_tags(self, conn: Any, record_id: str, tags: Tuple[str, ...]) -> None:
        if not tags:
            return
        async with conn.cursor() as cur:
            values = [(record_id, tag) for tag in tags]
            await cur.executemany(
                """
                INSERT INTO memory_tags (entry_id, tag, is_deleted)
                VALUES (%s, %s, FALSE)
                ON CONFLICT (entry_id, tag)
                DO UPDATE SET
                    is_deleted = FALSE,
                    updated_at = NOW(),
                    deleted_at = NULL
                """,
                values,
            )

    async def _sync_tags(self, conn: Any, record_id: str, tags: Tuple[str, ...]) -> None:
        await self._upsert_tags(conn, record_id, tags)
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT tag FROM memory_tags WHERE entry_id = %(id)s AND is_deleted = FALSE",
                {"id": record_id},
            )
            existing = {row["tag"] for row in await cur.fetchall()}
        to_remove = [tag for tag in existing if tag not in tags]
        if to_remove:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE memory_tags
                    SET is_deleted = TRUE,
                        deleted_at = NOW(),
                        updated_at = NOW()
                    WHERE entry_id = %(id)s AND tag = ANY(%(tags)s) AND is_deleted = FALSE
                    """,
                    {"id": record_id, "tags": to_remove},
                )

    async def _record_history(
        self,
        conn: Any,
        *,
        record_id: str,
        operation: str,
        version: int,
        previous_version: Optional[int],
        content: Optional[str],
        metadata: Mapping[str, Any],
        embedding: Optional[Sequence[float]],
        score: Optional[float],
        changed_by: Optional[str],
    ) -> None:
        async with conn.cursor() as cur:
            score_value = float(score) if score is not None else None
            await cur.execute(
                """
                INSERT INTO memory_entry_history (
                    entry_id, previous_version, version, operation,
                    content, metadata, embedding, score, changed_by
                )
                VALUES (
                    %(entry_id)s, %(previous_version)s, %(version)s, %(operation)s,
                    %(content)s, %(metadata)s::jsonb, {embedding}, %(score)s, %(changed_by)s
                )
                """.format(
                    embedding=sql.SQL("%(embedding)s::vector") if embedding is not None else sql.SQL("NULL")
                ),
                {
                    "entry_id": record_id,
                    "previous_version": previous_version,
                    "version": version,
                    "operation": operation,
                    "content": content,
                    "metadata": json.dumps(metadata),
                    "embedding": self._format_embedding(embedding),
                    "score": score_value,
                    "changed_by": changed_by,
                },
            )

    async def _record_audit_event_async(
        self,
        record_id: str,
        operation: str,
        *,
        content: Optional[str],
        metadata: Optional[Mapping[str, Any]],
        embedding: Optional[Sequence[float]],
        score: Optional[float],
        changed_by: Optional[str],
    ) -> None:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                current = await self._fetch_entry(conn, record_id, for_update=True)
                if current is None:
                    raise KeyError(f"Memory record '{record_id}' does not exist")
                if isinstance(metadata, MemoryMetadata):
                    metadata_payload: Mapping[str, Any] = _metadata_to_dict(metadata)
                elif metadata is None:
                    metadata_payload = _metadata_to_dict(self._record_to_metadata(current))
                else:
                    metadata_payload = dict(metadata)
                await self._record_history(
                    conn,
                    record_id=record_id,
                    operation=operation,
                    version=current["version"],
                    previous_version=current["version"],
                    content=content if content is not None else current.get("content"),
                    metadata=metadata_payload,
                    embedding=embedding if embedding is not None else current.get("embedding"),
                    score=score if score is not None else current.get("score"),
                    changed_by=changed_by,
                )

    @staticmethod
    def _record_to_metadata(row: Mapping[str, Any]) -> MemoryMetadata:
        raw_metadata = row.get("metadata")
        if isinstance(raw_metadata, str):
            try:
                metadata_payload = json.loads(raw_metadata)
            except json.JSONDecodeError:
                metadata_payload = {}
        else:
            metadata_payload = dict(raw_metadata or {})
        metadata_payload.setdefault("attributes", {})
        tags_value = row.get("tags") or []
        if isinstance(tags_value, str):
            try:
                tags_list = list(json.loads(tags_value))
            except json.JSONDecodeError:
                tags_list = [tag.strip() for tag in tags_value.strip("{}{}").split(",") if tag.strip()]
        elif isinstance(tags_value, (list, tuple, set)):
            tags_list = [str(tag) for tag in tags_value]
        else:
            tags_list = []
        metadata_payload.setdefault("tags", tags_list)
        attributes = metadata_payload.get("attributes", {})
        project = row.get("project_id")
        if project is not None and "project" not in attributes and "project_id" not in attributes:
            attributes["project"] = project
        context = row.get("context_id")
        if context is not None and "context" not in attributes and "context_id" not in attributes:
            attributes["context"] = context
        metadata = _metadata_from_dict(metadata_payload)
        metadata.created_at = row.get("created_at", metadata.created_at)
        metadata.updated_at = row.get("updated_at", metadata.updated_at)
        return metadata

    def _row_to_record(self, row: Mapping[str, Any]) -> MemoryRecord:
        metadata = self._record_to_metadata(row)
        embedding = row.get("embedding")
        if embedding is not None:
            embedding = [float(value) for value in embedding]
        similarity = row.get("similarity")
        score = float(similarity) if similarity is not None else row.get("score")
        record = MemoryRecord(
            content=row.get("content", ""),
            metadata=metadata,
            embedding=embedding,
            score=score,
            record_id=str(row.get("id")),
        )
        return record


class PostgresMemoryStore(LongTermMemoryStore):
    """Backward compatible alias for :class:`LongTermMemoryStore`."""


class CompositeMemoryStore(MemoryStore):
    """A store that fans operations out to multiple underlying stores."""

    backend_name = "composite"

    def __init__(self, stores: Sequence[MemoryStore]) -> None:
        if not stores:
            raise ValueError("CompositeMemoryStore requires at least one store")
        self._stores = list(stores)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        result = record
        for store in self._stores:
            result = store.add(result)
        return result

    def update(
        self,
        record_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        embedding: Optional[Sequence[float]] = None,
        score: Optional[float] = None,
    ) -> MemoryRecord:
        result: Optional[MemoryRecord] = None
        for store in self._stores:
            try:
                result = store.update(
                    record_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                    score=score,
                )
            except KeyError:
                continue
        if result is None:
            raise KeyError(f"Memory record '{record_id}' does not exist in any store")
        return result

    def delete(
        self,
        record_id: str,
        *,
        hard: bool = False,
        undo_window: Optional[int] = None,
    ) -> None:
        for store in self._stores:
            store.delete(record_id, hard=hard, undo_window=undo_window)

    def get(self, record_id: str, *, include_deleted: bool = False) -> MemoryRecord:
        last_error: Optional[Exception] = None
        for store in self._stores:
            try:
                return store.get(record_id, include_deleted=include_deleted)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise KeyError(f"Memory record '{record_id}' does not exist in any store")

    def fetch(self, query: MemoryQuery) -> List[MemoryRecord]:
        results: List[MemoryRecord] = []
        for store in self._stores:
            results.extend(store.fetch(query))
        # Deduplicate by record_id, keeping the highest score
        dedup: Dict[str, MemoryRecord] = {}
        for record in results:
            existing = dedup.get(record.record_id)
            if existing is None or (record.score or 0.0) > (existing.score or 0.0):
                dedup[record.record_id] = record
        combined = list(dedup.values())
        combined.sort(key=lambda rec: rec.score or 0.0, reverse=True)
        return combined[: query.limit]

    def compact(self) -> None:
        for store in self._stores:
            store.compact()


# ---------------------------------------------------------------------------
# Configuration dataclasses and helpers
# ---------------------------------------------------------------------------


@dataclass
class RedisSettings:
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PostgresSettings:
    dsn: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    sslmode: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoreConfig:
    scope: str
    backend: str = "memory"
    ttl_seconds: Optional[int] = None
    compaction_threshold: Optional[int] = None
    embedding_model: Optional[str] = None
    redis: Optional[RedisSettings] = None
    postgres: Optional[PostgresSettings] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "StoreConfig":
        return StoreConfig(
            scope=self.scope,
            backend=self.backend,
            ttl_seconds=self.ttl_seconds,
            compaction_threshold=self.compaction_threshold,
            embedding_model=self.embedding_model,
            redis=self.redis,
            postgres=self.postgres,
            options=dict(self.options),
        )


@dataclass
class MemoryConfiguration:
    short_term: StoreConfig
    long_term: StoreConfig
    combined: StoreConfig

    def for_scope(self, scope: str) -> StoreConfig:
        scope = scope.lower()
        if scope in {"short", "short_term", "short-term"}:
            return self.short_term
        if scope in {"long", "long_term", "long-term"}:
            return self.long_term
        if scope in {"combined", "aggregate", "both"}:
            return self.combined
        raise KeyError(f"Unknown memory scope '{scope}'")


def load_config_json(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``config.json`` if present.

    Parameters
    ----------
    config_path:
        Optional path override.  When not supplied the loader will check the
        ``MEMORY_CONFIG_PATH`` environment variable, then look for ``config.json``
        in the project root.
    """

    if config_path is None:
        env_path = os.getenv("MEMORY_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        else:
            candidates = [
                Path.cwd() / "config.json",
                Path(__file__).resolve().parent / "config.json",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    config_path = candidate
                    break
            else:
                config_path = candidates[0]

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        LOGGER.debug("No config file found at %s", config_path)
        return {}
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration errors
        LOGGER.warning("Could not parse config file %s: %s", config_path, exc)
        return {}


def load_memory_configuration(config_path: Optional[Path] = None) -> MemoryConfiguration:
    """Build a :class:`MemoryConfiguration` by merging config file and env vars."""

    config_data = load_config_json(config_path)
    memory_section: Mapping[str, Any] = config_data.get("memory", {})

    short_term = _store_config_from_dict("short_term", memory_section.get("short_term", {}))
    long_term = _store_config_from_dict("long_term", memory_section.get("long_term", {}))
    combined = _store_config_from_dict("combined", memory_section.get("combined", {}))

    short_term = _apply_env_overrides(short_term)
    long_term = _apply_env_overrides(long_term)
    combined = _apply_env_overrides(combined)

    if combined.backend in {"composite", "combined", "router"}:
        combined.backend = "composite"
        combined.options.setdefault("scopes", ["short_term", "long_term"])

    return MemoryConfiguration(short_term=short_term, long_term=long_term, combined=combined)


def _store_config_from_dict(scope: str, raw: Mapping[str, Any]) -> StoreConfig:
    backend = str(raw.get("backend", "memory")).lower()
    ttl_seconds = raw.get("ttl_seconds")
    compaction_threshold = raw.get("compaction_threshold")
    embedding_model = raw.get("embedding_model")

    redis_settings: Optional[RedisSettings] = None
    if "redis" in raw and isinstance(raw["redis"], Mapping):
        redis_settings = _redis_settings_from_dict(raw["redis"])

    postgres_settings: Optional[PostgresSettings] = None
    if "postgres" in raw and isinstance(raw["postgres"], Mapping):
        postgres_settings = _postgres_settings_from_dict(raw["postgres"])

    known_keys = {"backend", "ttl_seconds", "compaction_threshold", "embedding_model", "redis", "postgres"}
    options = {key: value for key, value in raw.items() if key not in known_keys}

    return StoreConfig(
        scope=scope,
        backend=backend,
        ttl_seconds=ttl_seconds,
        compaction_threshold=compaction_threshold,
        embedding_model=embedding_model,
        redis=redis_settings,
        postgres=postgres_settings,
        options=options,
    )


def _redis_settings_from_dict(raw: Mapping[str, Any]) -> RedisSettings:
    return RedisSettings(
        url=raw.get("url"),
        host=raw.get("host", "localhost"),
        port=int(raw.get("port", 6379)),
        db=int(raw.get("db", 0)),
        username=raw.get("username"),
        password=raw.get("password"),
        ssl=raw.get("ssl"),
        options={key: value for key, value in raw.items() if key not in {"url", "host", "port", "db", "username", "password", "ssl"}},
    )


def _postgres_settings_from_dict(raw: Mapping[str, Any]) -> PostgresSettings:
    return PostgresSettings(
        dsn=raw.get("dsn"),
        host=raw.get("host", "localhost"),
        port=int(raw.get("port", 5432)),
        database=raw.get("database"),
        user=raw.get("user"),
        password=raw.get("password"),
        sslmode=raw.get("sslmode"),
        options={
            key: value
            for key, value in raw.items()
            if key not in {"dsn", "host", "port", "database", "user", "password", "sslmode"}
        },
    )


def _apply_env_overrides(config: StoreConfig) -> StoreConfig:
    scope_token = config.scope.upper()

    backend = os.getenv(f"MEMORY_{scope_token}_BACKEND")
    if backend:
        config.backend = backend.lower()

    ttl = _as_int(os.getenv(f"MEMORY_{scope_token}_TTL_SECONDS"))
    if ttl is not None:
        config.ttl_seconds = ttl

    compaction = _as_int(os.getenv(f"MEMORY_{scope_token}_COMPACTION_THRESHOLD"))
    if compaction is not None:
        config.compaction_threshold = compaction

    embedding = os.getenv(f"MEMORY_{scope_token}_EMBEDDING_MODEL") or os.getenv("MEMORY_EMBEDDING_MODEL")
    if embedding:
        config.embedding_model = embedding

    if config.backend == "redis":
        config.redis = _apply_redis_env(scope_token, config.redis)
    elif config.backend == "postgres":
        config.postgres = _apply_postgres_env(scope_token, config.postgres)

    return config


def _apply_redis_env(scope_token: str, current: Optional[RedisSettings]) -> RedisSettings:
    prefix = f"MEMORY_{scope_token}_REDIS_"
    general_prefix = "MEMORY_REDIS_"
    values: Dict[str, Optional[str]] = {
        "url": os.getenv(prefix + "URL") or os.getenv(general_prefix + "URL"),
        "host": os.getenv(prefix + "HOST") or os.getenv(general_prefix + "HOST"),
        "port": os.getenv(prefix + "PORT") or os.getenv(general_prefix + "PORT"),
        "db": os.getenv(prefix + "DB") or os.getenv(general_prefix + "DB"),
        "username": os.getenv(prefix + "USERNAME") or os.getenv(general_prefix + "USERNAME"),
        "password": os.getenv(prefix + "PASSWORD") or os.getenv(general_prefix + "PASSWORD"),
        "ssl": os.getenv(prefix + "SSL") or os.getenv(general_prefix + "SSL"),
    }

    settings = current or RedisSettings()
    if values["url"]:
        settings.url = values["url"]
    if values["host"]:
        settings.host = values["host"] or settings.host
    if values["port"]:
        settings.port = int(values["port"])
    if values["db"]:
        settings.db = int(values["db"])
    if values["username"]:
        settings.username = values["username"]
    if values["password"]:
        settings.password = values["password"]
    ssl_value = _as_bool(values["ssl"])
    if ssl_value is not None:
        settings.ssl = ssl_value
    return settings


def _apply_postgres_env(scope_token: str, current: Optional[PostgresSettings]) -> PostgresSettings:
    prefix = f"MEMORY_{scope_token}_POSTGRES_"
    general_prefix = "MEMORY_POSTGRES_"
    values: Dict[str, Optional[str]] = {
        "dsn": os.getenv(prefix + "DSN") or os.getenv(general_prefix + "DSN"),
        "host": os.getenv(prefix + "HOST") or os.getenv(general_prefix + "HOST"),
        "port": os.getenv(prefix + "PORT") or os.getenv(general_prefix + "PORT"),
        "database": os.getenv(prefix + "DATABASE") or os.getenv(general_prefix + "DATABASE"),
        "user": os.getenv(prefix + "USER") or os.getenv(general_prefix + "USER"),
        "password": os.getenv(prefix + "PASSWORD") or os.getenv(general_prefix + "PASSWORD"),
        "sslmode": os.getenv(prefix + "SSLMODE") or os.getenv(general_prefix + "SSLMODE"),
    }

    settings = current or PostgresSettings()
    if values["dsn"]:
        settings.dsn = values["dsn"]
    if values["host"]:
        settings.host = values["host"] or settings.host
    if values["port"]:
        settings.port = int(values["port"])
    if values["database"]:
        settings.database = values["database"]
    if values["user"]:
        settings.user = values["user"]
    if values["password"]:
        settings.password = values["password"]
    if values["sslmode"]:
        settings.sslmode = values["sslmode"]
    return settings


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class MemoryRouter:
    """Resolve logical memory scopes into configured store instances."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    COMBINED = "combined"

    def __init__(self, stores: Mapping[str, MemoryStore]):
        if not stores:
            raise ValueError("MemoryRouter requires at least one store")
        self._stores: Dict[str, MemoryStore] = dict(stores)

    def get_store(self, scope: str) -> MemoryStore:
        scope_key = scope.lower().replace("-", "_")
        if scope_key not in self._stores:
            raise KeyError(f"Memory scope '{scope}' is not registered")
        return self._stores[scope_key]

    @property
    def short_term(self) -> MemoryStore:
        return self.get_store(self.SHORT_TERM)

    @property
    def long_term(self) -> MemoryStore:
        return self.get_store(self.LONG_TERM)

    @property
    def combined(self) -> MemoryStore:
        return self.get_store(self.COMBINED)

    def register(self, scope: str, store: MemoryStore) -> None:
        self._stores[scope.lower().replace("-", "_")] = store

    def scopes(self) -> Tuple[str, ...]:
        return tuple(self._stores.keys())

    def promote_to_long_term(
        self,
        record_id: str,
        *,
        strategy: str = "move",
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecord:
        short_store = self.short_term
        long_store = self.long_term
        if hasattr(short_store, "promote_to_long_term"):
            return short_store.promote_to_long_term(  # type: ignore[attr-defined]
                record_id,
                long_store,
                strategy=strategy,
                provenance=provenance,
            )

        record = short_store.get(record_id)
        metadata_payload = _metadata_to_dict(record.metadata)
        attributes = metadata_payload.setdefault("attributes", {})
        provenance_payload = dict(attributes.get("provenance", {}))
        provenance_payload.setdefault("source_backend", getattr(short_store, "backend_name", "unknown"))
        provenance_payload.setdefault("source_record_id", record_id)
        provenance_payload.setdefault("strategy", strategy)
        provenance_payload.setdefault("promoted_at", _utcnow().isoformat())
        if provenance:
            provenance_payload.update({str(key): value for key, value in provenance.items()})
        attributes["provenance"] = provenance_payload
        metadata_payload["attributes"] = attributes
        metadata = _metadata_from_dict(metadata_payload)
        enriched = replace(record, metadata=metadata)

        stored = long_store.add(enriched)
        actor = provenance_payload.get("actor")
        try:
            long_store.record_audit_event(
                stored.record_id,
                "promote" if strategy.lower() == "move" else "copy_to_long_term",
                metadata=_metadata_to_dict(stored.metadata),
                content=stored.content,
                embedding=stored.embedding,
                score=stored.score,
                changed_by=str(actor) if actor is not None else None,
            )
        except AttributeError:
            LOGGER.debug("Long-term store does not expose audit logging; skipping router promote audit")

        if strategy.lower() == "move":
            short_store.delete(record_id, hard=True)
        return stored

    def copy_to_long_term(
        self,
        record_id: str,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecord:
        return self.promote_to_long_term(record_id, strategy="copy", provenance=provenance)

    def compact_with_summarizer(
        self,
        summarizer: Callable[[List[Mapping[str, Any]]], Sequence[Mapping[str, Any]]],
        *,
        batch_size: int = 25,
        stale_after: timedelta = timedelta(hours=6),
        summarizer_name: Optional[str] = None,
    ) -> List[MemoryRecord]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        short_store = self.short_term
        long_store = self.long_term
        if not hasattr(short_store, "list_entries"):
            LOGGER.debug("Short-term store does not support listing entries; skipping compaction")
            return []

        try:
            candidates = short_store.list_entries(limit=None, include_deleted=False)  # type: ignore[attr-defined]
        except Exception:
            LOGGER.exception("Failed to enumerate short-term memories for compaction")
            return []

        now = _utcnow()
        stale_candidates = [
            record for record in candidates if now - record.metadata.updated_at >= stale_after
        ]
        if not stale_candidates:
            return []

        created: List[MemoryRecord] = []
        tool_name = summarizer_name or getattr(summarizer, "__name__", "memory_summarizer")
        for index in range(0, len(stale_candidates), batch_size):
            batch = stale_candidates[index : index + batch_size]
            payload = [self._summarizer_payload(record) for record in batch]
            try:
                summary_response = summarizer(payload)
            except Exception:
                LOGGER.exception("Summarizer tool '%s' failed during compaction", tool_name)
                continue

            if isinstance(summary_response, Mapping):
                summaries: Sequence[Mapping[str, Any]] = [summary_response]
            else:
                summaries = list(summary_response or [])

            summary_records, discard_ids = self._build_compaction_records(
                summaries,
                summarizer_name=tool_name,
            )

            for summary_record in summary_records:
                short_store.add(summary_record)
                persisted = long_store.add(summary_record)
                provenance = summary_record.metadata.attributes.get("provenance", {})
                actor = provenance.get("actor")
                try:
                    long_store.record_audit_event(
                        persisted.record_id,
                        "compaction_summary",
                        metadata=_metadata_to_dict(persisted.metadata),
                        content=persisted.content,
                        embedding=persisted.embedding,
                        score=persisted.score,
                        changed_by=str(actor) if actor is not None else None,
                    )
                except AttributeError:
                    LOGGER.debug("Long-term store does not expose audit logging; skipping compaction audit")
                created.append(persisted)

            if discard_ids:
                for record in batch:
                    if record.record_id not in discard_ids:
                        continue
                    short_store.delete(record.record_id, hard=True)
                    try:
                        long_store.delete(record.record_id, hard=True)
                    except Exception:
                        LOGGER.debug("Failed to hard delete long-term record '%s' during compaction", record.record_id)

        return created

    @staticmethod
    def _summarizer_payload(record: MemoryRecord) -> Dict[str, Any]:
        return {
            "record_id": record.record_id,
            "content": record.content,
            "metadata": _metadata_to_dict(record.metadata),
            "score": record.score,
            "embedding": list(record.embedding) if record.embedding is not None else None,
        }

    @staticmethod
    def _build_compaction_records(
        summaries: Sequence[Mapping[str, Any]],
        *,
        summarizer_name: str,
    ) -> Tuple[List[MemoryRecord], Set[str]]:
        summary_records: List[MemoryRecord] = []
        discarded: Set[str] = set()
        now = _utcnow()
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            source_ids_raw = summary.get("source_ids")
            if isinstance(source_ids_raw, (list, tuple, set, frozenset)):
                source_ids = [str(value) for value in source_ids_raw]
            elif source_ids_raw is None:
                source_ids = []
            else:
                source_ids = [str(source_ids_raw)]
            discarded.update(source_ids)
            discarded_ids = summary.get("discarded_ids")
            if isinstance(discarded_ids, (list, tuple, set, frozenset)):
                discarded.update(str(value) for value in discarded_ids)
            elif discarded_ids is not None:
                discarded.add(str(discarded_ids))
            if summary.get("discard"):
                continue
            content = summary.get("content") or summary.get("summary")
            if not content:
                continue
            metadata_payload = dict(summary.get("metadata") or {})
            attributes = metadata_payload.setdefault("attributes", {})
            provenance = dict(attributes.get("provenance", {}))
            provenance.setdefault("summarizer", summarizer_name)
            provenance.setdefault("source_ids", source_ids)
            provenance.setdefault("compacted_at", now.isoformat())
            provenance.setdefault("strategy", "llm_compaction")
            attributes["provenance"] = provenance
            tags = summary.get("tags")
            if tags is not None and "tags" not in metadata_payload:
                if isinstance(tags, (list, tuple, set, frozenset)):
                    metadata_payload["tags"] = list(tags)
                else:
                    metadata_payload["tags"] = [tags]
            importance = summary.get("importance")
            if importance is not None and "importance" not in metadata_payload:
                metadata_payload["importance"] = importance
            metadata_payload.setdefault("source", "compaction")
            metadata = _metadata_from_dict(metadata_payload)
            metadata.touch()
            embedding = summary.get("embedding")
            if isinstance(embedding, (list, tuple)):
                embedding_value: Optional[Sequence[float]] = [float(value) for value in embedding]  # type: ignore[assignment]
            else:
                embedding_value = None
            score_value = summary.get("score")
            try:
                score_numeric = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score_numeric = None
            summary_records.append(
                MemoryRecord(
                    content=str(content),
                    metadata=metadata,
                    embedding=embedding_value,
                    score=score_numeric,
                )
            )
        return summary_records, discarded


def build_memory_store(config: StoreConfig) -> MemoryStore:
    """Instantiate a concrete :class:`MemoryStore` from configuration."""

    backend = config.backend.lower()
    if backend in {"memory", "in_memory", "local"}:
        return InMemoryMemoryStore(
            default_ttl=config.ttl_seconds,
            compaction_threshold=config.compaction_threshold,
            embedding_model=config.embedding_model,
        )
    if backend == "redis":
        if config.redis is None:
            LOGGER.warning("Redis backend selected for %s but no connection details provided; using defaults", config.scope)
            config.redis = RedisSettings()
        if redis is None:
            LOGGER.warning(
                "Redis backend selected for %s but redis-py is not installed; falling back to in-memory store",
                config.scope,
            )
            return InMemoryMemoryStore(
                default_ttl=config.ttl_seconds,
                compaction_threshold=config.compaction_threshold,
                embedding_model=config.embedding_model,
            )
        return RedisMemoryStore(config)
    if backend in {"postgres", "postgresql"}:
        if config.postgres is None:
            LOGGER.warning(
                "PostgreSQL backend selected for %s but no connection details provided; using defaults",
                config.scope,
            )
            config.postgres = PostgresSettings()
        config.backend = "postgres"
        return PostgresMemoryStore(config)
    if backend == "composite":
        raise ValueError(
            "Composite stores should be constructed by the router because they depend on other scopes"
        )

    LOGGER.warning("Unknown backend '%s' for scope '%s'; falling back to in-memory store", backend, config.scope)
    return InMemoryMemoryStore(
        default_ttl=config.ttl_seconds,
        compaction_threshold=config.compaction_threshold,
        embedding_model=config.embedding_model,
    )


def build_memory_router(config: Optional[MemoryConfiguration] = None) -> MemoryRouter:
    """Build a :class:`MemoryRouter` with stores configured for each scope."""

    config = config or load_memory_configuration()

    short_store = build_memory_store(config.short_term.copy())
    long_store = build_memory_store(config.long_term.copy())

    stores: Dict[str, MemoryStore] = {
        MemoryRouter.SHORT_TERM: short_store,
        MemoryRouter.LONG_TERM: long_store,
    }

    combined_backend = config.combined.backend.lower()
    if combined_backend == "composite":
        scopes = config.combined.options.get("scopes", [MemoryRouter.SHORT_TERM, MemoryRouter.LONG_TERM])
        resolved: List[MemoryStore] = []
        for scope in scopes:
            scope_key = scope.lower().replace("-", "_")
            store = stores.get(scope_key)
            if store is None:
                LOGGER.warning("Composite memory scope requested unknown store '%s'", scope)
                continue
            resolved.append(store)
        if not resolved:
            LOGGER.warning("Composite memory scope '%s' has no valid stores; defaulting to short-term store", config.combined.scope)
            resolved = [short_store]
        stores[MemoryRouter.COMBINED] = CompositeMemoryStore(resolved)
    else:
        stores[MemoryRouter.COMBINED] = build_memory_store(config.combined.copy())

        return MemoryRouter(stores)


class MemoryFacade:
    """High-level helper exposing convenience operations across memory scopes."""

    def __init__(
        self,
        router: MemoryRouter,
        *,
        default_scope: str = MemoryRouter.SHORT_TERM,
        combined_scope: str = MemoryRouter.COMBINED,
    ) -> None:
        self.router = router
        self.default_scope = default_scope
        self.combined_scope = combined_scope

    # ------------------------------------------------------------------
    # Public API helpers
    # ------------------------------------------------------------------

    def search(
        self,
        text: str,
        *,
        scope: str | None = None,
        limit: int = 10,
        min_score: float | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> List[MemoryRecord]:
        """Return records matching ``text`` from the requested ``scope``."""

        store = self._get_store(scope or self.combined_scope)
        query = MemoryQuery(
            text=text,
            embedding=list(embedding) if embedding is not None else None,
            limit=max(0, int(limit)),
            min_score=self._coerce_float(min_score),
            metadata_filters=dict(metadata_filters or {}),
        )
        return store.fetch(query)

    def add(
        self,
        content: str,
        *,
        scope: str | None = None,
        tags: Sequence[str] | None = None,
        importance: float | None = None,
        attributes: Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
        embedding: Sequence[float] | None = None,
        score: float | None = None,
        source: str = "conversation",
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord:
        """Persist ``content`` into ``scope`` returning the stored record."""

        store = self._get_store(scope or self.default_scope)
        metadata = self._build_metadata(
            source=source,
            tags=tags,
            importance=importance,
            ttl_seconds=ttl_seconds,
            attributes=attributes,
            session_id=session_id,
            agent_id=agent_id,
        )
        record = MemoryRecord(
            content=content,
            metadata=metadata,
            embedding=list(embedding) if embedding is not None else None,
            score=self._coerce_float(score),
        )
        return store.add(record)

    def update(
        self,
        record_id: str,
        *,
        scope: str | None = None,
        content: str | None = None,
        tags: Sequence[str] | None = None,
        importance: float | None = None,
        attributes: Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
        embedding: Sequence[float] | None = None,
        score: float | None = None,
    ) -> MemoryRecord:
        """Update an existing record and return the stored value."""

        candidate_scopes = self._resolve_update_scopes(scope)
        last_error: Exception | None = None
        for candidate in candidate_scopes:
            store = self._get_store(candidate)
            try:
                current = store.get(record_id)
            except KeyError as exc:
                last_error = exc
                continue

            metadata = current.metadata
            if tags is not None:
                metadata = replace(metadata, tags=tuple(str(tag) for tag in tags))
            if importance is not None:
                metadata = replace(metadata, importance=self._coerce_float(importance))
            if ttl_seconds is not None:
                metadata = replace(metadata, ttl_seconds=self._coerce_int(ttl_seconds))
            if attributes:
                merged = dict(metadata.attributes)
                merged.update(self._sanitize_attributes(attributes))
                metadata = replace(metadata, attributes=merged)
            metadata.touch()

            return store.update(
                record_id,
                content=content,
                metadata=metadata,
                embedding=list(embedding) if embedding is not None else None,
                score=self._coerce_float(score),
            )

        if last_error is not None:
            raise last_error
        raise KeyError(f"Memory record '{record_id}' does not exist")

    def promote(
        self,
        record_id: str,
        *,
        strategy: str = "move",
        provenance: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        """Promote a short-term record into long-term storage."""

        payload = {str(key): value for key, value in (provenance or {}).items()}
        return self.router.promote_to_long_term(record_id, strategy=strategy, provenance=payload)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(record: MemoryRecord) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of ``record``."""

        metadata_payload = _metadata_to_dict(record.metadata)
        return {
            "record_id": record.record_id,
            "content": record.content,
            "metadata": metadata_payload,
            "score": record.score,
            "embedding": list(record.embedding) if record.embedding is not None else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self, scope: str) -> MemoryStore:
        normalized = scope.lower().replace("-", "_")
        return self.router.get_store(normalized)

    def _build_metadata(
        self,
        *,
        source: str,
        tags: Sequence[str] | None,
        importance: float | None,
        ttl_seconds: int | None,
        attributes: Mapping[str, Any] | None,
        session_id: str | None,
        agent_id: str | None,
    ) -> MemoryMetadata:
        attributes_payload = self._sanitize_attributes(attributes)
        if session_id is not None:
            attributes_payload.setdefault("session_id", session_id)
        if agent_id is not None:
            attributes_payload.setdefault("agent_id", agent_id)
        metadata = MemoryMetadata(
            source=str(source or "unknown"),
            tags=tuple(str(tag) for tag in (tags or ())),
            importance=self._coerce_float(importance),
            ttl_seconds=self._coerce_int(ttl_seconds),
            attributes=attributes_payload,
        )
        metadata.touch()
        return metadata

    def _sanitize_attributes(self, attributes: Mapping[str, Any] | None) -> Dict[str, Any]:
        if not attributes:
            return {}
        sanitized: Dict[str, Any] = {}
        for key, value in attributes.items():
            sanitized[str(key)] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): self._sanitize_value(item) for key, item in value.items()}
        return str(value)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_update_scopes(self, scope: str | None) -> Tuple[str, ...]:
        if scope is None:
            return (self.default_scope, MemoryRouter.LONG_TERM)
        normalized = scope.lower().replace("-", "_")
        if normalized == "auto":
            return (self.default_scope, MemoryRouter.LONG_TERM)
        return (normalized,)


class ConversationMemoryHooks:
    """Lifecycle hooks that persist conversation turns via :class:`MemoryFacade`."""

    def __init__(
        self,
        facade: MemoryFacade,
        *,
        session_id: str,
        agent_label: str = "manager",
        short_scope: str = MemoryRouter.SHORT_TERM,
        promotion_threshold: float = 0.85,
    ) -> None:
        self._facade = facade
        self._session_id = session_id
        self._agent_label = agent_label
        self._short_scope = short_scope
        self._promotion_threshold = promotion_threshold
        self._user_records: Dict[int, MemoryRecord] = {}
        self._assistant_records: Dict[int, MemoryRecord] = {}

    # ------------------------------------------------------------------
    # Hooks exposed to :class:`AgentSession`
    # ------------------------------------------------------------------

    def on_round_start(self, payload: Mapping[str, Any]) -> None:
        user_message = payload.get("user_message")
        if not user_message:
            return
        round_index = int(payload.get("index", 0))
        metadata = payload.get("metadata")
        task = None
        if isinstance(metadata, Mapping):
            task = metadata.get("task")
        record = self._facade.add(
            str(user_message),
            scope=self._short_scope,
            tags=("conversation", "user"),
            attributes={
                "role": "user",
                "round_index": round_index,
                "task": task,
            },
            session_id=self._session_id,
            agent_id=self._agent_label,
            source="conversation",
        )
        self._user_records[round_index] = record

    def on_round_end(self, round_record: "AgentRound") -> None:
        round_index = round_record.index
        metadata_payload = dict(round_record.metadata or {})
        task = metadata_payload.get("task")
        record = self._facade.add(
            round_record.response_text,
            scope=self._short_scope,
            tags=("conversation", "assistant"),
            attributes={
                "role": "assistant",
                "round_index": round_index,
                "task": task,
                "metadata": metadata_payload,
            },
            session_id=self._session_id,
            agent_id=self._agent_label,
            source="conversation",
        )
        self._assistant_records[round_index] = record

        memory_meta: Dict[str, Any] = {}
        existing_memory = metadata_payload.get("memory")
        if isinstance(existing_memory, Mapping):
            memory_meta.update(existing_memory)
        user_record = self._user_records.get(round_index)
        if user_record is not None:
            memory_meta.setdefault("user_record_id", user_record.record_id)
        memory_meta["assistant_record_id"] = record.record_id

        promotion_reason = self._promotion_reason(metadata_payload)
        if promotion_reason is not None:
            promoted = self._facade.promote(
                record.record_id,
                strategy="copy",
                provenance={
                    "reason": promotion_reason,
                    "trigger_round": round_index,
                    "task": task,
                },
            )
            memory_meta["long_term_record_id"] = promoted.record_id
            self._facade.update(
                record.record_id,
                attributes={
                    "promotion": {
                        "promoted": True,
                        "long_term_record_id": promoted.record_id,
                        "reason": promotion_reason,
                    }
                },
            )

        metadata_payload["memory"] = memory_meta
        round_record.metadata = metadata_payload

    # ------------------------------------------------------------------
    # Promotion heuristics
    # ------------------------------------------------------------------

    def _promotion_reason(self, metadata: Mapping[str, Any]) -> Optional[str]:
        if not metadata:
            return None
        tags = metadata.get("tags")
        if isinstance(tags, (list, tuple, set, frozenset)):
            for tag in tags:
                if str(tag).lower() in {"important", "high_importance", "milestone"}:
                    return "tagged_high_importance"

        importance = metadata.get("importance") or metadata.get("response_importance")
        try:
            if importance is not None and float(importance) >= self._promotion_threshold:
                return "high_importance_score"
        except (TypeError, ValueError):
            pass

        plan_state = metadata.get("plan_status") or metadata.get("plan")
        if isinstance(plan_state, str) and plan_state.lower() in {"complete", "completed", "done"}:
            return "plan_completed"

        if metadata.get("promote_to_long_term"):
            return "explicit_promotion_request"

        return None


_SHARED_MEMORY_ROUTER: Optional[MemoryRouter] = None
_SHARED_MEMORY_FACADE: Optional[MemoryFacade] = None


def get_shared_memory_router() -> MemoryRouter:
    """Return the shared :class:`MemoryRouter`, constructing one on demand."""

    global _SHARED_MEMORY_ROUTER
    if _SHARED_MEMORY_ROUTER is None:
        _SHARED_MEMORY_ROUTER = build_memory_router()
    return _SHARED_MEMORY_ROUTER


def set_shared_memory_router(router: Optional[MemoryRouter]) -> None:
    """Override the shared router used by tooling and agents."""

    global _SHARED_MEMORY_ROUTER, _SHARED_MEMORY_FACADE
    _SHARED_MEMORY_ROUTER = router
    if router is None:
        _SHARED_MEMORY_FACADE = None


def get_shared_memory_facade() -> MemoryFacade:
    """Return a singleton :class:`MemoryFacade` backed by the shared router."""

    global _SHARED_MEMORY_FACADE
    if _SHARED_MEMORY_FACADE is None:
        _SHARED_MEMORY_FACADE = MemoryFacade(get_shared_memory_router())
    return _SHARED_MEMORY_FACADE


def set_shared_memory_facade(facade: Optional[MemoryFacade]) -> None:
    """Replace the globally shared :class:`MemoryFacade`."""

    global _SHARED_MEMORY_FACADE, _SHARED_MEMORY_ROUTER
    _SHARED_MEMORY_FACADE = facade
    if facade is not None:
        _SHARED_MEMORY_ROUTER = facade.router


__all__ = [
    "register_embedding_provider",
    "clear_embedding_providers",
    "resolve_embedding_provider",
    "MemoryMetadata",
    "MemoryRecord",
    "MemoryQuery",
    "MemoryStore",
    "InMemoryMemoryStore",
    "RedisMemoryStore",
    "PostgresMemoryStore",
    "CompositeMemoryStore",
    "RedisSettings",
    "PostgresSettings",
    "StoreConfig",
    "MemoryConfiguration",
    "MemoryRouter",
    "MemoryFacade",
    "ConversationMemoryHooks",
    "get_shared_memory_router",
    "set_shared_memory_router",
    "get_shared_memory_facade",
    "set_shared_memory_facade",
    "load_config_json",
    "load_memory_configuration",
    "build_memory_store",
    "build_memory_router",
]
