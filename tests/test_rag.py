import pytest

from internal.RAG import DocumentChunk, _RagIndex


def _build_index() -> _RagIndex:
    index = _RagIndex(kind="doc")
    chunks = [
        DocumentChunk("doc1.txt", 0, "alpha beta gamma", "doc"),
        DocumentChunk("doc2.txt", 0, "beta delta epsilon", "doc"),
    ]
    index.ingest_chunks(chunks)
    return index


def _candidate_dict(chunk: DocumentChunk) -> dict:
    return {
        "path": chunk.source_path,
        "offset": chunk.offset,
        "text": chunk.text,
        "kind": chunk.kind,
    }


def test_rerank_scores_consistent_with_explicit_calculation():
    index = _build_index()
    query = "alpha beta"
    alpha = 0.5
    candidates = [_candidate_dict(ch) for ch in index.chunks]

    reranked = index.rerank(query, [dict(c) for c in candidates], alpha=alpha)

    q_tokens = index.tokenizer.tokenize(query)
    idxs = [
        index.chunks.index(
            DocumentChunk(c["path"], c["offset"], c["text"], c.get("kind", index.kind))
        )
        for c in candidates
    ]
    bm_scores = {i: index.ranker.bm25.score(q_tokens, i) for i in idxs}
    tf_scores = {
        i: index.ranker.tfidf.cosine(index.ranker.tfidf.embed_query(q_tokens), i)
        for i in idxs
    }
    expected_scores = {
        i: alpha * bm_scores[i] + (1 - alpha) * tf_scores[i]
        for i in idxs
    }

    assert reranked == sorted(reranked, key=lambda x: x["score"], reverse=True)
    for entry in reranked:
        idx = index.chunks.index(
            DocumentChunk(entry["path"], entry["offset"], entry["text"], entry.get("kind", index.kind))
        )
        assert entry["score"] == pytest.approx(expected_scores[idx])


def test_rerank_handles_empty_query_scores_zero():
    index = _build_index()
    candidates = [_candidate_dict(ch) for ch in index.chunks]

    reranked = index.rerank("", [dict(c) for c in candidates], alpha=0.5)

    for entry in reranked:
        assert entry["score"] == pytest.approx(0.0)
