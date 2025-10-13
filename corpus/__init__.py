"""Utilities for recording interaction events into the shared corpus."""

from .manager import (
    CorpusEvent,
    CorpusManager,
    get_shared_corpus_manager,
    record_event,
    set_shared_corpus_manager,
)

__all__ = [
    "CorpusEvent",
    "CorpusManager",
    "get_shared_corpus_manager",
    "record_event",
    "set_shared_corpus_manager",
]
