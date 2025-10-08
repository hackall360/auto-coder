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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


class RedisMemoryStore(InMemoryMemoryStore):
    """Placeholder Redis-backed store.

    The implementation currently falls back to the in-memory behaviour while we
    design persistence.  Configuration metadata is still captured so that the
    store can later be swapped with a real Redis-backed implementation without
    changing callers.
    """

    backend_name = "redis"

    def __init__(self, config: "StoreConfig") -> None:
        super().__init__(
            default_ttl=config.ttl_seconds,
            compaction_threshold=config.compaction_threshold,
            embedding_model=config.embedding_model,
        )
        self.config = config


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
