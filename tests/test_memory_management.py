from datetime import datetime, timedelta, timezone

import pytest

from memory import (
    InMemoryMemoryStore,
    MemoryMetadata,
    MemoryRecord,
    MemoryRouter,
)


def test_inmemory_soft_and_hard_delete_behaviour() -> None:
    store = InMemoryMemoryStore()
    record = store.add(MemoryRecord(content="retain", metadata=MemoryMetadata(source="unit")))

    store.delete(record.record_id, undo_window=30)

    with pytest.raises(KeyError):
        store.get(record.record_id)

    soft = store.get(record.record_id, include_deleted=True)
    deletion_meta = soft.metadata.attributes.get("deletion", {})
    assert deletion_meta.get("mode") == "soft"
    assert "undo_window_seconds" in deletion_meta

    store.delete(record.record_id, hard=True)

    with pytest.raises(KeyError):
        store.get(record.record_id, include_deleted=True)


def test_memory_router_promote_and_copy_workflows() -> None:
    short_store = InMemoryMemoryStore()
    long_store = InMemoryMemoryStore()
    router = MemoryRouter({
        MemoryRouter.SHORT_TERM: short_store,
        MemoryRouter.LONG_TERM: long_store,
    })

    promoted_record = short_store.add(
        MemoryRecord(content="Short memo", metadata=MemoryMetadata(source="short"))
    )

    promoted = router.promote_to_long_term(
        promoted_record.record_id,
        strategy="move",
        provenance={"actor": "tester"},
    )

    stored_long = long_store.get(promoted.record_id)
    assert stored_long.content == "Short memo"
    assert stored_long.metadata.attributes["provenance"]["source_backend"] == short_store.backend_name

    with pytest.raises(KeyError):
        short_store.get(promoted_record.record_id)

    copy_record = short_store.add(
        MemoryRecord(content="Another memo", metadata=MemoryMetadata(source="short"))
    )

    copied = router.copy_to_long_term(copy_record.record_id)
    assert long_store.get(copied.record_id).content == "Another memo"
    assert short_store.get(copy_record.record_id).content == "Another memo"


def test_compaction_with_summarizer_creates_summary_and_discards_sources() -> None:
    class ListableInMemoryStore(InMemoryMemoryStore):
        def list_entries(self, *, limit=None, include_deleted=False, **_: object):  # type: ignore[override]
            records = list(self._records.values())
            if include_deleted:
                records.extend(record for record, _ in self._deleted_records.values())
            if limit is not None:
                records = records[:limit]
            return records

    short_store = ListableInMemoryStore()
    long_store = InMemoryMemoryStore()
    router = MemoryRouter({
        MemoryRouter.SHORT_TERM: short_store,
        MemoryRouter.LONG_TERM: long_store,
    })

    stale_metadata = MemoryMetadata(
        source="short",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        updated_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    record = short_store.add(MemoryRecord(content="Alpha context", metadata=stale_metadata))
    long_store.add(record)

    def summarizer(batch: list[dict[str, object]]) -> list[dict[str, object]]:
        assert batch and batch[0]["record_id"] == record.record_id
        return [
            {
                "source_ids": [batch[0]["record_id"]],
                "content": f"Summary: {batch[0]['content']}",
                "metadata": {"attributes": {"actor": "compactor"}},
                "tags": ["summary"],
                "importance": 0.9,
            }
        ]

    summaries = router.compact_with_summarizer(
        summarizer,
        batch_size=1,
        stale_after=timedelta(seconds=0),
        summarizer_name="unit-test-summarizer",
    )

    assert summaries, "Expected summarizer to produce at least one summary"
    summary = summaries[0]
    provenance = summary.metadata.attributes.get("provenance", {})
    assert provenance.get("summarizer") == "unit-test-summarizer"
    assert "Summary:" in summary.content

    with pytest.raises(KeyError):
        short_store.get(record.record_id)
    with pytest.raises(KeyError):
        long_store.get(record.record_id)

    stored_short = short_store.get(summary.record_id)
    stored_long = long_store.get(summary.record_id)
    assert stored_short.content == stored_long.content == summary.content
