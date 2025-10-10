"""Tooling surface for interacting with the shared memory subsystem."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memory import (
    MemoryFacade,
    MemoryRouter,
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


__all__ = [
    "memory_search_tool",
    "memory_add_tool",
    "memory_update_tool",
    "memory_promote_tool",
]

