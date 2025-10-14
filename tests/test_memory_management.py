from datetime import datetime, timedelta, timezone

import pytest

from memory import (
    ConversationMemoryHooks,
    InMemoryMemoryStore,
    MemoryFacade,
    MemoryMetadata,
    MemoryRecord,
    MemoryRouter,
    set_shared_memory_facade,
)
from internal.tools.memory import (
    memory_add_tool,
    memory_promote_tool,
    memory_search_tool,
    memory_update_tool,
)


class ListableInMemoryStore(InMemoryMemoryStore):
    def list_entries(self, *, limit=None, include_deleted=False, **_: object):  # type: ignore[override]
        records = list(self._records.values())
        if include_deleted:
            records.extend(record for record, _ in self._deleted_records.values())
        if limit is not None:
            records = records[:limit]
        return records


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


def test_memory_facade_basic_workflow() -> None:
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: ListableInMemoryStore(),
            MemoryRouter.LONG_TERM: ListableInMemoryStore(),
        }
    )
    facade = MemoryFacade(router, combined_scope=MemoryRouter.SHORT_TERM)

    record = facade.add(
        "Hello memory",
        tags=("greeting",),
        attributes={"session_id": "session-1", "flag": True},
        importance=0.5,
        session_id="session-1",
        agent_id="tester",
    )

    results = facade.search("Hello", scope=MemoryRouter.SHORT_TERM)
    assert any(found.record_id == record.record_id for found in results)

    updated = facade.update(
        record.record_id,
        tags=("greeting", "updated"),
        importance=0.9,
        attributes={"note": "updated"},
    )
    assert pytest.approx(updated.metadata.importance or 0.0, rel=1e-6) == 0.9
    assert "note" in updated.metadata.attributes

    promoted = facade.promote(record.record_id, strategy="copy", provenance={"actor": "unit"})
    long_term_record = router.long_term.get(promoted.record_id)
    assert long_term_record.content == record.content


def test_conversation_memory_hooks_log_rounds_and_promote() -> None:
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: ListableInMemoryStore(),
            MemoryRouter.LONG_TERM: ListableInMemoryStore(),
        }
    )
    facade = MemoryFacade(router, combined_scope=MemoryRouter.SHORT_TERM)
    hooks = ConversationMemoryHooks(
        facade,
        session_id="session-xyz",
        agent_label="manager",
        promotion_threshold=0.8,
    )

    hooks.on_round_start({"index": 0, "user_message": "Remember this", "metadata": {"task": "task-1"}})

    class DummyRound:
        def __init__(self) -> None:
            self.index = 0
            self.user_message = "Remember this"
            self.response_text = "Important insight"
            self.result = {"content": "Important insight"}
            self.transcript = []
            self.messages = []
            self.tool_history = {}
            self.metadata = {"task": "task-1", "importance": 0.95}

    round_record = DummyRound()

    hooks.on_round_end(round_record)

    short_records = router.short_term.list_entries()
    roles = {entry.metadata.attributes.get("role") for entry in short_records}
    assert {"user", "assistant"}.issubset(roles)

    memory_meta = round_record.metadata.get("memory", {}) if round_record.metadata else {}
    assert "assistant_record_id" in memory_meta
    assert "long_term_record_id" in memory_meta

    long_records = router.long_term.list_entries()
    assert any(record.content == "Important insight" for record in long_records)


def test_memory_tool_callables_use_shared_facade() -> None:
    router = MemoryRouter(
        {
            MemoryRouter.SHORT_TERM: ListableInMemoryStore(),
            MemoryRouter.LONG_TERM: ListableInMemoryStore(),
        }
    )
    facade = MemoryFacade(router, combined_scope=MemoryRouter.SHORT_TERM)
    set_shared_memory_facade(facade)

    added = memory_add_tool("Tool memo", tags=("tool",))
    results = memory_search_tool("Tool memo", scope=MemoryRouter.SHORT_TERM)
    assert any(result["record_id"] == added["record_id"] for result in results)

    updated = memory_update_tool(added["record_id"], tags=("tool", "updated"), importance=0.9)
    assert set(updated["metadata"]["tags"]) == {"tool", "updated"}

    promoted = memory_promote_tool(added["record_id"], strategy="copy")
    stored = router.long_term.get(promoted["record_id"])
    assert stored.content == "Tool memo"

    set_shared_memory_facade(None)
