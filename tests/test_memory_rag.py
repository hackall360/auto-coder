from datetime import datetime, timedelta, timezone

import pytest

from memory import (
    MemoryMetadata,
    MemoryRecord,
    InMemoryMemoryStore,
    clear_embedding_providers,
    register_embedding_provider,
)
from internal.memory_rag import MemoryHit, MemoryRAG, MemoryRetrievalConfig


@pytest.fixture(autouse=True)
def reset_embedding_providers() -> None:
    clear_embedding_providers()
    yield
    clear_embedding_providers()


def _fake_embedder(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if "alpha" in lowered else 0.0, 1.0 if "beta" in lowered else 0.0]


def test_embeddings_computed_on_add() -> None:
    calls: list[str] = []

    def embedder(text: str) -> list[float]:
        calls.append(text)
        return _fake_embedder(text)

    register_embedding_provider(embedder, model_name="unit-test")
    store = InMemoryMemoryStore(embedding_model="unit-test")
    assert getattr(store, "_embedder_fn") is embedder
    record = MemoryRecord(content="Alpha memo", metadata=MemoryMetadata(source="unit"))

    stored = store.add(record)

    assert stored.embedding == [1.0, 0.0]
    assert stored.metadata.embedding_model == "unit-test"

    updated = store.update(
        stored.record_id,
        content="Beta memo",
    )
    assert calls == ["Alpha memo", "Beta memo"]
    assert updated.content == "Beta memo"
    assert updated.embedding == [0.0, 1.0]


def test_memory_rag_ranking_and_streaming() -> None:
    register_embedding_provider(_fake_embedder, model_name="unit-test")
    short_store = InMemoryMemoryStore(embedding_model="unit-test")
    long_store = InMemoryMemoryStore(embedding_model="unit-test")

    short_record = MemoryRecord(
        content="Alpha project details",
        metadata=MemoryMetadata(source="short", tags=("alpha",), updated_at=datetime.now(timezone.utc)),
    )
    long_recent = MemoryRecord(
        content="Beta project notes",
        metadata=MemoryMetadata(source="long", tags=("beta",), updated_at=datetime.now(timezone.utc)),
    )
    long_alpha = MemoryRecord(
        content="Alpha research summary",
        metadata=MemoryMetadata(
            source="long",
            tags=("alpha",),
            updated_at=datetime.now(timezone.utc) - timedelta(hours=3),
        ),
    )

    short_store.add(short_record)
    long_store.add(long_recent)
    long_store.add(long_alpha)

    rag = MemoryRAG(
        short_term=short_store,
        long_term=long_store,
        config=MemoryRetrievalConfig(short_term_weight=1.5, long_term_weight=1.0, freshness_weight=0.0),
    )

    hits = list(rag.query_memory("alpha insights", limit=5))

    assert hits, "expected at least one retrieval result"
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].scope == "short_term"
    assert hits[0].record.content == "Alpha project details"
    assert hits[0].record.score == pytest.approx(hits[0].score)
    assert hits[0].provenance["backend"] == short_store.backend_name

    streamed = list(rag.query_memory("alpha insights", limit=5, stream=True))
    assert [hit.record.record_id for hit in streamed] == [hit.record.record_id for hit in hits]
