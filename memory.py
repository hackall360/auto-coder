"""Utilities for configuring and working with agent memory stores.

This module defines a small abstraction layer that allows the rest of the
application to request different flavours of memory (for example, "short" or
"long" term storage) without coupling callers to a particular backend.  It also
includes helpers for loading configuration from environment variables or an
optional ``config.json`` file so that operators can adjust memory behaviour
without modifying code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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

LOGGER = logging.getLogger(__name__)


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
    def delete(self, record_id: str) -> None:
        """Remove a record from the store."""

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
        self._default_ttl = default_ttl
        self._compaction_threshold = compaction_threshold or 0
        self._compaction_counter = 0
        self._embedding_model = embedding_model

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record = self._prepare_record(record)
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

        record.metadata.touch()
        if record.metadata.ttl_seconds is None and self._default_ttl is not None:
            record.metadata.ttl_seconds = self._default_ttl
        if record.metadata.embedding_model is None:
            record.metadata.embedding_model = self._embedding_model

        self._records[record_id] = record
        self._maybe_compact()
        return record

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

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
        self._compaction_counter = 0

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _prepare_record(self, record: MemoryRecord) -> MemoryRecord:
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

    def delete(self, record_id: str) -> None:
        self.soft_delete(record_id)

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

    def soft_delete(self, record_id: str) -> None:
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
                    pipe.multi()
                    pipe.hset(record_key, mapping={"deleted": "1"})
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


class PostgresMemoryStore(InMemoryMemoryStore):
    """Placeholder PostgreSQL-backed store sharing behaviour with the in-memory store."""

    backend_name = "postgres"

    def __init__(self, config: "StoreConfig") -> None:
        super().__init__(
            default_ttl=config.ttl_seconds,
            compaction_threshold=config.compaction_threshold,
            embedding_model=config.embedding_model,
        )
        self.config = config


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

    def delete(self, record_id: str) -> None:
        for store in self._stores:
            store.delete(record_id)

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


__all__ = [
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
    "load_config_json",
    "load_memory_configuration",
    "build_memory_store",
    "build_memory_router",
]
