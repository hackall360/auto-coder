"""Tooling surface for interacting with the shared memory subsystem."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from memory import (
    MemoryFacade,
    MemoryRouter,
    MemoryRecord,
    get_shared_memory_facade,
)


def _facade(facade: MemoryFacade | None = None) -> MemoryFacade:
    return facade or get_shared_memory_facade()


def memory_search_tool(
    query: str,
    *,
    scope: str = MemoryRouter.COMBINED,
    limit: int = 10,
    min_score: float | None = None,
    metadata_filters: Mapping[str, Any] | None = None,
    embedding: Sequence[float] | None = None,
    facade: MemoryFacade | None = None,
) -> list[dict[str, Any]]:
    """Search agent memory for entries relevant to ``query``."""

    records = _facade(facade).search(
        query,
        scope=scope,
        limit=limit,
        min_score=min_score,
        metadata_filters=metadata_filters,
        embedding=embedding,
    )
    return [MemoryFacade.to_dict(record) for record in records]


def memory_add_tool(
    content: str,
    *,
    scope: str = MemoryRouter.SHORT_TERM,
    tags: Sequence[str] | None = None,
    importance: float | None = None,
    attributes: Mapping[str, Any] | None = None,
    ttl_seconds: int | None = None,
    embedding: Sequence[float] | None = None,
    facade: MemoryFacade | None = None,
) -> dict[str, Any]:
    """Add ``content`` into ``scope`` and return the stored record."""

    record = _facade(facade).add(
        content,
        scope=scope,
        tags=tags,
        importance=importance,
        attributes=attributes,
        ttl_seconds=ttl_seconds,
        embedding=embedding,
    )
    return MemoryFacade.to_dict(record)


def memory_update_tool(
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
    facade: MemoryFacade | None = None,
) -> dict[str, Any]:
    """Update an existing memory record and return the new value."""

    record = _facade(facade).update(
        record_id,
        scope=scope,
        content=content,
        tags=tags,
        importance=importance,
        attributes=attributes,
        ttl_seconds=ttl_seconds,
        embedding=embedding,
        score=score,
    )
    return MemoryFacade.to_dict(record)


def memory_promote_tool(
    record_id: str,
    *,
    strategy: str = "move",
    provenance: Mapping[str, Any] | None = None,
    facade: MemoryFacade | None = None,
) -> dict[str, Any]:
    """Promote a short-term memory into long-term storage."""

    record = _facade(facade).promote(record_id, strategy=strategy, provenance=provenance)
    return MemoryFacade.to_dict(record)


def memory_list_sessions_tool(
    *,
    scope: str = MemoryRouter.SHORT_TERM,
    limit: int | None = None,
    preview_limit: int = 0,
    include_deleted: bool = False,
    facade: MemoryFacade | None = None,
) -> list[dict[str, Any]]:
    """Return the active memory sessions for ``scope``."""

    entries = _facade(facade).list_sessions(
        scope=scope,
        limit=limit,
        preview_limit=preview_limit,
        include_deleted=include_deleted,
    )

    serialized: list[dict[str, Any]] = []
    for entry in entries:
        last_activity = entry.get("last_activity_at")
        if isinstance(last_activity, datetime):
            last_activity_iso = last_activity.astimezone().isoformat()
        elif last_activity:
            last_activity_iso = str(last_activity)
        else:
            last_activity_iso = None
        preview = [
            MemoryFacade.to_dict(record)
            for record in entry.get("preview", ())
            if isinstance(record, MemoryRecord)
        ]
        serialized.append(
            {
                "session_id": entry.get("session_id"),
                "scope": entry.get("scope", scope),
                "last_activity_at": last_activity_iso,
                "preview": preview,
            }
        )
    return serialized


__all__ = [
    "memory_search_tool",
    "memory_add_tool",
    "memory_update_tool",
    "memory_promote_tool",
    "memory_list_sessions_tool",
]

