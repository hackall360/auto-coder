"""Hybrid memory retrieval utilities combining vector and keyword search."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .RAG import _HybridRanker, _Tokenizer, _DEFAULT_STOPWORDS
from memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
    MemoryMetadata,
    resolve_embedding_provider,
)

LOGGER = logging.getLogger(__name__)


def _coerce_embedding(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, Mapping):
        for key in ("embedding", "vector", "values", "data"):
            if key in values:
                return _coerce_embedding(values[key])
        return []
    if hasattr(values, "embedding"):
        return _coerce_embedding(getattr(values, "embedding"))
    if hasattr(values, "tolist") and callable(values.tolist):  # pragma: no cover - numpy arrays
        return _coerce_embedding(values.tolist())
    if isinstance(values, str):
        try:
            import json

            return _coerce_embedding(json.loads(values))
        except Exception:  # pragma: no cover - defensive parsing
            return []
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        try:
            return [float(item) for item in values]
        except (TypeError, ValueError):
            return []
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    denom_a = math.sqrt(sum(a * a for a in vec_a))
    denom_b = math.sqrt(sum(b * b for b in vec_b))
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numerator / (denom_a * denom_b)


@dataclass(slots=True)
class MemoryRetrievalConfig:
    """Tunable parameters that control hybrid memory retrieval."""

    short_term_top_k: int = 6
    long_term_top_k: int = 12
    combined_top_k: int = 8
    max_results: int = 20
    embedding_weight: float = 0.6
    keyword_weight: float = 0.4
    hybrid_alpha: float = 0.6
    freshness_weight: float = 0.1
    freshness_halflife_hours: float = 24.0
    short_term_weight: float = 1.1
    long_term_weight: float = 1.0
    combined_weight: float = 1.0
    include_combined: bool = False

    @classmethod
    def from_env(cls, prefix: str = "MEMORY_RAG_") -> "MemoryRetrievalConfig":
        mapping: Dict[str, Tuple[str, Any]] = {
            "short_term_top_k": ("SHORT_TERM_TOP_K", int),
            "long_term_top_k": ("LONG_TERM_TOP_K", int),
            "combined_top_k": ("COMBINED_TOP_K", int),
            "max_results": ("MAX_RESULTS", int),
            "embedding_weight": ("EMBEDDING_WEIGHT", float),
            "keyword_weight": ("KEYWORD_WEIGHT", float),
            "hybrid_alpha": ("HYBRID_ALPHA", float),
            "freshness_weight": ("FRESHNESS_WEIGHT", float),
            "freshness_halflife_hours": ("FRESHNESS_HALFLIFE_HOURS", float),
            "short_term_weight": ("SHORT_TERM_WEIGHT", float),
            "long_term_weight": ("LONG_TERM_WEIGHT", float),
            "combined_weight": ("COMBINED_WEIGHT", float),
        }
        overrides: Dict[str, Any] = {}
        for field_name, (suffix, caster) in mapping.items():
            value = os.getenv(f"{prefix}{suffix}")
            if value is None:
                continue
            try:
                overrides[field_name] = caster(value)
            except (TypeError, ValueError):
                LOGGER.debug("Ignoring invalid value for %s%s", prefix, suffix, exc_info=True)
        include_combined = os.getenv(f"{prefix}INCLUDE_COMBINED")
        if include_combined is not None:
            overrides["include_combined"] = include_combined.strip().lower() in {"1", "true", "yes", "on"}
        return cls(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_term_top_k": self.short_term_top_k,
            "long_term_top_k": self.long_term_top_k,
            "combined_top_k": self.combined_top_k,
            "max_results": self.max_results,
            "embedding_weight": self.embedding_weight,
            "keyword_weight": self.keyword_weight,
            "hybrid_alpha": self.hybrid_alpha,
            "freshness_weight": self.freshness_weight,
            "freshness_halflife_hours": self.freshness_halflife_hours,
            "short_term_weight": self.short_term_weight,
            "long_term_weight": self.long_term_weight,
            "combined_weight": self.combined_weight,
            "include_combined": self.include_combined,
        }


@dataclass(slots=True)
class MemoryHit:
    """Container for ranked memory results returned by :class:`MemoryRAG`."""

    record: MemoryRecord
    scope: str
    backend: str
    score: float
    embedding_score: float
    keyword_score: float
    embedding_raw: float
    keyword_raw: float
    freshness_boost: float

    @property
    def content(self) -> str:
        return self.record.content

    @property
    def metadata(self) -> MemoryMetadata:
        return self.record.metadata

    @property
    def provenance(self) -> Dict[str, Any]:
        metadata = self.record.metadata
        return {
            "record_id": self.record.record_id,
            "scope": self.scope,
            "backend": self.backend,
            "source": metadata.source,
            "updated_at": metadata.updated_at.astimezone(timezone.utc).isoformat(),
            "tags": list(metadata.tags),
            "embedding_model": metadata.embedding_model,
            "embedding_score": self.embedding_score,
            "keyword_score": self.keyword_score,
            "embedding_raw": self.embedding_raw,
            "keyword_raw": self.keyword_raw,
            "freshness_boost": self.freshness_boost,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "record_id": self.record.record_id,
            "metadata": self.record.metadata,
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class _StoreCandidate:
    scope: str
    backend: str
    record: MemoryRecord


class MemoryRAG:
    """Compose multiple memory stores into a hybrid retrieval pipeline."""

    def __init__(
        self,
        *,
        short_term: Optional[MemoryStore] = None,
        long_term: Optional[MemoryStore] = None,
        combined: Optional[MemoryStore] = None,
        config: Optional[MemoryRetrievalConfig] = None,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        stores: Dict[str, MemoryStore] = {}
        if short_term is not None:
            stores["short_term"] = short_term
        if long_term is not None:
            stores["long_term"] = long_term
        if combined is not None:
            stores["combined"] = combined
        if not stores:
            raise ValueError("MemoryRAG requires at least one memory store")
        self._stores = stores
        self.config = config or MemoryRetrievalConfig.from_env()
        self._tokenizer = _Tokenizer(_DEFAULT_STOPWORDS)
        self._embedder, resolved_name = self._resolve_embedder(embedder, embedding_model)
        self._embedding_model_name = resolved_name or embedding_model
        total_weight = float(self.config.embedding_weight + self.config.keyword_weight)
        if total_weight <= 0:
            self._embedding_factor = 0.0
            self._keyword_factor = 0.0
            self._has_weight_signal = False
        else:
            self._embedding_factor = float(self.config.embedding_weight) / total_weight
            self._keyword_factor = float(self.config.keyword_weight) / total_weight
            self._has_weight_signal = True

    def query_memory(
        self,
        text: str,
        *,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
        stream: bool = False,
        embedding: Optional[Sequence[float]] = None,
    ) -> Iterable[MemoryHit]:
        """Return ranked memory chunks for ``text`` enriched with provenance."""

        limit_value = max(1, int(limit or self.config.max_results))
        query_embedding = self._normalize_embedding(embedding)
        if query_embedding is None:
            query_embedding = self._embed_query(text)
        per_store_limits = self._per_store_limits(limit_value)
        filters = dict(metadata_filters or {})
        candidates: List[_StoreCandidate] = []
        for scope, store in self._stores.items():
            store_limit = per_store_limits.get(scope, 0)
            if store_limit <= 0:
                continue
            try:
                records = store.fetch(
                    MemoryQuery(
                        text=text,
                        embedding=query_embedding,
                        limit=store_limit,
                        metadata_filters=dict(filters),
                    )
                )
            except Exception:  # pragma: no cover - store-specific runtime errors
                LOGGER.warning("Memory store '%s' failed to fetch candidates", scope, exc_info=True)
                continue
            backend_name = getattr(store, "backend_name", store.__class__.__name__)
            for record in records:
                candidates.append(_StoreCandidate(scope=scope, backend=backend_name, record=record))

        if not candidates:
            return []

        query_tokens = self._tokenizer.tokenize(text)
        doc_tokens = [self._tokenizer.tokenize(candidate.record.content) for candidate in candidates]
        keyword_scores: Dict[int, float] = {}
        if query_tokens and doc_tokens:
            ranker = _HybridRanker(doc_tokens)
            keyword_scores = dict(ranker.topk(query_tokens, len(doc_tokens), alpha=self.config.hybrid_alpha))

        embedding_scores: Dict[int, float] = {}
        if query_embedding is not None:
            for idx, candidate in enumerate(candidates):
                if candidate.record.embedding:
                    embedding_scores[idx] = _cosine_similarity(query_embedding, candidate.record.embedding)
        for idx, candidate in enumerate(candidates):
            if idx in embedding_scores:
                continue
            if candidate.record.score is None:
                continue
            try:
                embedding_scores[idx] = float(candidate.record.score)
            except (TypeError, ValueError):
                continue

        max_keyword = max(keyword_scores.values(), default=0.0)
        embed_values = list(embedding_scores.values())
        embed_min = min(embed_values, default=0.0)
        embed_max = max(embed_values, default=0.0)
        embed_range = embed_max - embed_min

        def normalize_keyword(raw: float) -> float:
            if max_keyword <= 0.0:
                return 0.0
            return _clamp01(raw / max_keyword)

        def normalize_embedding(raw: float) -> float:
            if not embed_values:
                return 0.0
            if embed_range <= 1e-6:
                if embed_min < 0.0 or embed_max > 1.0:
                    return _clamp01((raw + 1.0) / 2.0)
                return _clamp01(raw)
            return _clamp01((raw - embed_min) / embed_range)

        now = datetime.now(timezone.utc)
        hits: List[MemoryHit] = []
        for idx, candidate in enumerate(candidates):
            keyword_raw = keyword_scores.get(idx, 0.0)
            embedding_raw = embedding_scores.get(idx, 0.0)
            keyword_signal = normalize_keyword(keyword_raw)
            embedding_signal = normalize_embedding(embedding_raw)
            combined = 0.0
            if self._has_weight_signal:
                combined = (embedding_signal * self._embedding_factor) + (
                    keyword_signal * self._keyword_factor
                )
            combined *= self._store_weight(candidate.scope)
            freshness = self._freshness_boost(candidate.record.metadata, candidate.scope, now)
            total_score = combined + freshness
            record_with_score = candidate.record.with_score(total_score)
            hits.append(
                MemoryHit(
                    record=record_with_score,
                    scope=candidate.scope,
                    backend=candidate.backend,
                    score=total_score,
                    embedding_score=embedding_signal,
                    keyword_score=keyword_signal,
                    embedding_raw=embedding_raw,
                    keyword_raw=keyword_raw,
                    freshness_boost=freshness,
                )
            )

        dedup: Dict[str, MemoryHit] = {}
        for hit in hits:
            key = hit.record.record_id
            existing = dedup.get(key)
            if existing is None or hit.score > existing.score:
                dedup[key] = hit
        ranked = list(dedup.values())
        ranked.sort(key=lambda item: item.score, reverse=True)
        ranked = ranked[:limit_value]

        if stream:
            return (hit for hit in ranked)
        return ranked

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _resolve_embedder(
        self,
        embedder: Optional[Any],
        embedding_model: Optional[str],
    ) -> Tuple[Optional[Any], Optional[str]]:
        if callable(embedder):
            return embedder, embedding_model
        inferred = embedding_model or self._infer_model_name()
        provider, resolved_name = resolve_embedding_provider(inferred)
        return provider, resolved_name or inferred

    def _infer_model_name(self) -> Optional[str]:
        for scope in ("short_term", "long_term", "combined"):
            store = self._stores.get(scope)
            if store is None:
                continue
            config = getattr(store, "config", None)
            if config is not None:
                name = getattr(config, "embedding_model", None)
                if name:
                    return name
            name = getattr(store, "_embedding_model", None)
            if name:
                return name
        return None

    def _normalize_embedding(self, embedding: Optional[Sequence[float]]) -> Optional[List[float]]:
        if embedding is None:
            return None
        coerced = _coerce_embedding(embedding)
        return coerced or None

    def _embed_query(self, text: str) -> Optional[List[float]]:
        if self._embedder is None:
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            values = self._embedder(cleaned)
        except Exception:  # pragma: no cover - runtime embedding errors
            LOGGER.debug("Query embedding failed", exc_info=True)
            return None
        coerced = _coerce_embedding(values)
        return coerced or None

    def _per_store_limits(self, limit: int) -> Dict[str, int]:
        limits: Dict[str, int] = {}
        if "short_term" in self._stores:
            limits["short_term"] = min(self.config.short_term_top_k, limit)
        if "long_term" in self._stores:
            limits["long_term"] = min(self.config.long_term_top_k, limit)
        if self.config.include_combined and "combined" in self._stores:
            limits["combined"] = min(self.config.combined_top_k, limit)
        return limits

    def _store_weight(self, scope: str) -> float:
        if scope == "short_term":
            return self.config.short_term_weight
        if scope == "long_term":
            return self.config.long_term_weight
        if scope == "combined":
            return self.config.combined_weight
        return 1.0

    def _freshness_boost(
        self,
        metadata: MemoryMetadata,
        scope: str,
        now: datetime,
    ) -> float:
        if self.config.freshness_weight <= 0.0 or self.config.freshness_halflife_hours <= 0.0:
            return 0.0
        updated_at = metadata.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = max(0.0, (now - updated_at).total_seconds() / 3600.0)
        decay = math.pow(0.5, age / self.config.freshness_halflife_hours)
        base = self.config.freshness_weight * decay
        return base * self._store_weight(scope)


__all__ = ["MemoryRetrievalConfig", "MemoryHit", "MemoryRAG"]
