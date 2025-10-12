"""Integration tests for Redis and PostgreSQL memory stores."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator

import pytest

from memory import (
    CompositeMemoryStore,
    InMemoryMemoryStore,
    MemoryConfiguration,
    MemoryMetadata,
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
    PostgresSettings,
    RedisSettings,
    StoreConfig,
    build_memory_router,
    build_memory_store,
    clear_embedding_providers,
    register_embedding_provider,
)
from internal.memory_rag import MemoryRAG, MemoryRetrievalConfig
from scripts import setup_memory


@pytest.fixture(autouse=True)
def _reset_embedding_cache() -> Iterator[None]:
    clear_embedding_providers()
    yield
    clear_embedding_providers()


def test_build_memory_router_with_composite_combined_scope() -> None:
    config = MemoryConfiguration(
        short_term=StoreConfig(scope="short_term", backend="memory"),
        long_term=StoreConfig(scope="long_term", backend="memory"),
        combined=StoreConfig(
            scope="combined",
            backend="composite",
            options={"scopes": ["short_term", "long_term"]},
        ),
    )

    router = build_memory_router(config)

    combined_store = router.combined
    assert isinstance(combined_store, CompositeMemoryStore)

    record = MemoryRecord(
        content="Composite record",
        metadata=MemoryMetadata(source="unit-test", attributes={"session_id": "combo"}),
    )

    stored = combined_store.add(record)

    assert router.short_term.get(stored.record_id).content == "Composite record"
    assert router.long_term.get(stored.record_id).content == "Composite record"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def redis_service() -> Iterator[Dict[str, object]]:
    redis_mod = pytest.importorskip("redis")
    url = os.getenv("TEST_REDIS_URL")
    if url:
        client = redis_mod.Redis.from_url(url)
        try:
            client.ping()
        except redis_mod.RedisError as exc:  # type: ignore[attr-defined]
            pytest.skip(f"Redis service not reachable: {exc}")
        yield {"url": url, "client": client}
        client.flushdb()
        return

    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server binary missing; set TEST_REDIS_URL to run Redis integration tests")

    port = _find_free_port()
    proc = subprocess.Popen(
        [
            binary,
            "--save",
            "",
            "--appendonly",
            "no",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = redis_mod.Redis(host="127.0.0.1", port=port, db=0)
    for _ in range(50):
        try:
            client.ping()
            break
        except redis_mod.RedisError:  # type: ignore[attr-defined]
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.skip("Redis server failed to start within the timeout window")

    yield {"url": f"redis://127.0.0.1:{port}/0", "client": client, "process": proc}

    client.flushdb()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def redis_store(redis_service: Dict[str, object]) -> Iterator[MemoryStore]:
    url = str(redis_service["url"])
    config = StoreConfig(
        scope="short_term",
        backend="redis",
        redis=RedisSettings(url=url),
        options={
            "namespace": "autocoder:test",
            "vector_index": "autocoder_test_idx",
            "vector_dimensions": 2,
        },
    )
    result = setup_memory.ensure_redis_ready(config)
    if not result.ok:
        pytest.skip(result.hint or result.detail)

    store = build_memory_store(config.copy())
    try:
        yield store  # type: ignore[misc]
    finally:
        try:
            client = redis_service["client"]
            if hasattr(client, "flushdb"):
                client.flushdb()
        except Exception:
            pass


@pytest.fixture(scope="session")
def postgres_config() -> Iterator[StoreConfig]:
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set TEST_POSTGRES_DSN to run PostgreSQL integration tests")

    psycopg_mod = pytest.importorskip("psycopg")

    try:
        conn = psycopg_mod.connect(dsn)
        conn.close()
    except Exception as exc:  # pragma: no cover - configuration error
        pytest.skip(f"Unable to connect to PostgreSQL using TEST_POSTGRES_DSN: {exc}")

    config = StoreConfig(scope="long_term", backend="postgres", postgres=PostgresSettings(dsn=dsn))
    migrations_dir = Path(__file__).resolve().parents[1] / "internal" / "db" / "migrations"
    result = setup_memory.ensure_postgres_ready(config, migrations_path=migrations_dir)
    if not result.ok:
        pytest.skip(result.hint or result.detail)

    yield config

    with psycopg_mod.connect(dsn, autocommit=True) as conn:
        for table in ("memory_links", "memory_tags", "memory_entry_history", "memory_entries"):
            conn.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")


@pytest.fixture
def postgres_store(postgres_config: StoreConfig) -> Iterator[MemoryStore]:
    store = build_memory_store(postgres_config.copy())
    try:
        yield store  # type: ignore[misc]
    finally:
        pool = getattr(store, "_pool", None)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass


@pytest.mark.integration
def test_redis_store_lifecycle(redis_store: MemoryStore) -> None:
    metadata = MemoryMetadata(source="redis", tags=("alpha",), attributes={"session_id": "redis-session"})
    record = MemoryRecord(content="Alpha memo", metadata=metadata, embedding=[1.0, 0.0])

    stored = redis_store.add(record)
    fetched = redis_store.get(stored.record_id)
    assert fetched.content == "Alpha memo"

    updated = redis_store.update(stored.record_id, content="Updated alpha memo")
    assert updated.content == "Updated alpha memo"

    results = redis_store.fetch(MemoryQuery(text="alpha", limit=5, metadata_filters={"tags": ("alpha",)}))
    assert any(item.record_id == stored.record_id for item in results)

    long_term = InMemoryMemoryStore()
    promoted = redis_store.promote_to_long_term(stored.record_id, long_term, strategy="copy")
    assert long_term.get(promoted.record_id).content == "Updated alpha memo"

    redis_store.delete(stored.record_id, hard=True)
    with pytest.raises(KeyError):
        redis_store.get(stored.record_id)

    redis_store.compact()


@pytest.mark.integration
def test_redis_store_list_sessions(redis_store: MemoryStore) -> None:
    store = redis_store
    if not hasattr(store, "list_sessions"):
        pytest.skip("Redis store does not expose list_sessions")

    alpha_meta = MemoryMetadata(source="redis", attributes={"session_id": "alpha"})
    store.add(MemoryRecord(content="Alpha start", metadata=alpha_meta))
    time.sleep(0.01)
    beta_meta = MemoryMetadata(source="redis", attributes={"session_id": "beta"})
    store.add(MemoryRecord(content="Beta intro", metadata=beta_meta))
    time.sleep(0.01)
    store.add(
        MemoryRecord(
            content="Alpha follow-up",
            metadata=MemoryMetadata(source="redis", attributes={"session_id": "alpha"}),
        )
    )

    sessions = store.list_sessions(preview_limit=2)  # type: ignore[attr-defined]
    assert [entry["session_id"] for entry in sessions] == ["alpha", "beta"]
    assert isinstance(sessions[0]["last_activity_at"], datetime)
    assert sessions[0]["last_activity_at"] >= sessions[1]["last_activity_at"]
    assert len(sessions[0]["preview"]) == 2
    assert sessions[0]["preview"][0].content == "Alpha follow-up"
    assert sessions[0]["preview"][1].content == "Alpha start"
    assert len(sessions[1]["preview"]) == 1
    assert sessions[1]["preview"][0].content == "Beta intro"

    limited = store.list_sessions(limit=1, preview_limit=1)  # type: ignore[attr-defined]
    assert len(limited) == 1
    assert limited[0]["session_id"] == "alpha"


@pytest.mark.integration
def test_postgres_store_lifecycle(postgres_store: MemoryStore) -> None:
    metadata = MemoryMetadata(source="postgres", tags=("beta",), attributes={"project": "proj-1"})
    record = MemoryRecord(content="Beta memo", metadata=metadata)

    stored = postgres_store.add(record)
    fetched = postgres_store.get(stored.record_id)
    assert fetched.content == "Beta memo"

    postgres_store.update(stored.record_id, content="Beta memo updated")
    results = postgres_store.fetch(MemoryQuery(text="updated", limit=5))
    assert any(item.record_id == stored.record_id for item in results)

    postgres_store.delete(stored.record_id)
    with pytest.raises(KeyError):
        postgres_store.get(stored.record_id)

    postgres_store.compact()


@pytest.mark.integration
def test_postgres_store_list_sessions(postgres_store: MemoryStore) -> None:
    store = postgres_store
    if not hasattr(store, "list_sessions"):
        pytest.skip("PostgreSQL store does not expose list_sessions")

    alpha_meta = MemoryMetadata(source="postgres", attributes={"session_id": "pg-alpha"})
    store.add(MemoryRecord(content="PG alpha start", metadata=alpha_meta))
    time.sleep(0.01)
    beta_meta = MemoryMetadata(source="postgres", attributes={"session_id": "pg-beta"})
    store.add(MemoryRecord(content="PG beta intro", metadata=beta_meta))
    time.sleep(0.01)
    store.add(
        MemoryRecord(
            content="PG alpha follow-up",
            metadata=MemoryMetadata(source="postgres", attributes={"session_id": "pg-alpha"}),
        )
    )

    sessions = store.list_sessions(preview_limit=2)  # type: ignore[attr-defined]
    assert [entry["session_id"] for entry in sessions][:2] == ["pg-alpha", "pg-beta"]
    assert isinstance(sessions[0]["last_activity_at"], datetime)
    assert sessions[0]["last_activity_at"] >= sessions[1]["last_activity_at"]
    assert len(sessions[0]["preview"]) == 2
    assert sessions[0]["preview"][0].content == "PG alpha follow-up"
    assert sessions[0]["preview"][1].content == "PG alpha start"
    assert len(sessions[1]["preview"]) == 1
    assert sessions[1]["preview"][0].content == "PG beta intro"

    limited = store.list_sessions(limit=1, preview_limit=1)  # type: ignore[attr-defined]
    assert len(limited) == 1
    assert limited[0]["session_id"] == "pg-alpha"


@pytest.mark.integration
def test_rag_across_backends(
    redis_store: MemoryStore,
    postgres_store: MemoryStore,
) -> None:
    def embed(text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if "alpha" in lowered else 0.0, 1.0 if "beta" in lowered else 0.0]

    register_embedding_provider(embed, model_name="integration-test")

    short_metadata = MemoryMetadata(source="redis", tags=("alpha",), attributes={"session_id": "rag"})
    short_record = MemoryRecord(
        content="Alpha project overview",
        metadata=short_metadata,
        embedding=[1.0, 0.0],
    )
    redis_store.add(short_record)

    long_metadata = MemoryMetadata(source="postgres", tags=("beta",), attributes={"project": "rag"})
    long_record = MemoryRecord(
        content="Beta roadmap and milestones",
        metadata=long_metadata,
        score=0.9,
    )
    postgres_store.add(long_record)

    rag = MemoryRAG(
        short_term=redis_store,
        long_term=postgres_store,
        config=MemoryRetrievalConfig(short_term_weight=1.4, long_term_weight=1.0, include_combined=False),
    )

    alpha_hits = list(rag.query_memory("alpha milestones", limit=5))
    assert any(hit.scope == "short_term" for hit in alpha_hits)

    beta_hits = list(rag.query_memory("beta milestones", limit=5))
    assert any(hit.scope == "long_term" for hit in beta_hits)
