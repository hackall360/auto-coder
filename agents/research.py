"""High-level research helper built on top of :class:`internal.RAG.WebRAG`."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import re
import threading
import urllib.parse
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from internal.RAG import WebRAG

__all__ = [
    "ResearchSnippet",
    "ResearchResult",
    "ResearchAgent",
]


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class ResearchSnippet:
    """Structured excerpt extracted from a web source."""

    url: str
    title: str
    quote: str
    citation: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "title": self.title,
            "quote": self.quote,
            "citation": self.citation,
        }
        if self.score is not None:
            payload["score"] = self.score
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class ResearchResult:
    """Container bundling the snippets for a single research query."""

    query: str
    snippets: tuple[ResearchSnippet, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "snippets": [snippet.to_dict() for snippet in self.snippets],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def limit(self, top_k: int) -> "ResearchResult":
        if top_k <= 0:
            return ResearchResult(query=self.query, snippets=(), metadata=self.metadata)
        if top_k >= len(self.snippets):
            return self
        return ResearchResult(query=self.query, snippets=self.snippets[:top_k], metadata=self.metadata)


class ResearchAgent:
    """Wrapper around :class:`WebRAG` providing caching and sanitisation."""

    def __init__(
        self,
        *,
        rag_factory: Callable[..., WebRAG] | None = None,
        cache_size: int = 8,
        cache_top_k: int = 8,
        max_quote_chars: int = 320,
        anonymous_browsing: bool | None = None,
        **rag_kwargs: Any,
    ) -> None:
        self._rag_factory = rag_factory or WebRAG
        self._rag_kwargs: dict[str, Any] = dict(rag_kwargs)
        if anonymous_browsing is not None:
            self._rag_kwargs["anonymous_browsing"] = anonymous_browsing
        self._rag: WebRAG | None = None
        self._cache_size = max(1, cache_size)
        self._cache_top_k = max(1, cache_top_k)
        self._max_quote_chars = max(80, max_quote_chars)
        self._cache: "OrderedDict[str, ResearchResult]" = OrderedDict()
        self._lock = threading.Lock()
        self._url_cache: dict[str, MutableMapping[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Invalidate cached query and source lookups."""

        with self._lock:
            self._cache.clear()
            self._url_cache.clear()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_search_results: int = 20,
        allow_rewrite: bool = True,
        audience: str | None = None,
        force_refresh: bool = False,
        alpha: float = 0.6,
    ) -> ResearchResult:
        """Return cached or freshly gathered research snippets for ``query``."""

        cleaned_query = self._normalise_query(query)
        if not cleaned_query:
            return ResearchResult(query=query, snippets=())

        cache_key = cleaned_query.lower()
        with self._lock:
            cached = None if force_refresh else self._cache.get(cache_key)
            if cached is not None and top_k <= len(cached.snippets):
                self._cache.move_to_end(cache_key)
                return cached.limit(top_k)

        rag = self._ensure_rag()
        fetch_top_k = max(top_k, self._cache_top_k)
        raw_results = rag.search(
            cleaned_query,
            top_k=fetch_top_k,
            max_search_results=max_search_results,
            allow_rewrite=allow_rewrite,
            alpha=alpha,
        )
        source_meta = self._collect_source_metadata(rag, cleaned_query, max_search_results)

        snippets: list[ResearchSnippet] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = str(item.get("path") or "").strip()
            if not url:
                continue
            normalised = self._normalise_url(url)
            if normalised in seen_urls:
                continue
            seen_urls.add(normalised)
            cached_source = self._url_cache.get(normalised)
            if cached_source is None:
                title, snippet_hint = self._resolve_source_details(url, source_meta)
                quote = self._build_quote(item.get("text", ""), snippet_hint)
                if not quote:
                    continue
                cached_source = {
                    "title": title,
                    "quote": quote,
                    "source_snippet": snippet_hint,
                }
                self._url_cache[normalised] = cached_source
            else:
                quote = str(cached_source.get("quote") or "")
                if not quote:
                    continue
            citation = self._format_citation(len(snippets) + 1, url)
            title = str(cached_source.get("title") or self._derive_title(url))
            score = item.get("score")
            snippet_metadata: dict[str, Any] = {"source_snippet": cached_source.get("source_snippet")}
            if audience is not None:
                snippet_metadata["audience"] = audience
            if score is not None:
                snippet_metadata["score"] = score
            snippet = ResearchSnippet(
                url=url,
                title=title,
                quote=quote,
                citation=citation,
                score=float(score) if score is not None else None,
                metadata=snippet_metadata,
            )
            snippets.append(snippet)
            if len(snippets) >= fetch_top_k:
                break

        result = ResearchResult(query=cleaned_query, snippets=tuple(snippets))
        with self._lock:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result.limit(top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_rag(self) -> WebRAG:
        if self._rag is None:
            self._rag = self._rag_factory(**self._rag_kwargs)
        return self._rag

    def _normalise_query(self, query: str) -> str:
        return _WHITESPACE_RE.sub(" ", query.strip())

    def _normalise_url(self, url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path.rstrip("/")
        normalised = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
        return normalised or url

    def _collect_source_metadata(
        self,
        rag: WebRAG,
        query: str,
        max_results: int,
    ) -> dict[str, Mapping[str, str]]:
        meta: dict[str, Mapping[str, str]] = {}
        results: Sequence[Mapping[str, str]] = []
        client = getattr(rag, "_playwright_client", None)
        if client is not None and callable(getattr(client, "is_available", None)):
            try:
                if client.is_available():
                    results = client.collect_search_results(query, max_results=max_results)  # type: ignore[attr-defined]
            except Exception:
                results = []
        if not results:
            try:
                results = rag._search_ddg(query, max_results=max_results)
            except Exception:
                results = []

        for entry in results:
            url = str(entry.get("url") or entry.get("href") or "").strip()
            if not url:
                continue
            normalised = self._normalise_url(url)
            if normalised in meta:
                continue
            title = entry.get("title") or entry.get("body") or ""
            snippet = entry.get("snippet") or entry.get("body") or ""
            meta[normalised] = {
                "title": self._sanitize(title, max_chars=160) or self._derive_title(url),
                "snippet": self._sanitize(snippet, max_chars=self._max_quote_chars),
            }
        return meta

    def _resolve_source_details(
        self,
        url: str,
        meta: Mapping[str, Mapping[str, str]],
    ) -> tuple[str, str]:
        normalised = self._normalise_url(url)
        if normalised in meta:
            entry = meta[normalised]
            return entry.get("title", self._derive_title(url)), entry.get("snippet", "")
        return self._derive_title(url), ""

    def _derive_title(self, url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.netloc or parsed.path or url
        host = host.split("@")[-1]
        if parsed.path and parsed.path not in {"", "/"}:
            parts = [segment for segment in parsed.path.split("/") if segment]
            if parts:
                return self._sanitize(parts[-1], max_chars=80)
        return self._sanitize(host, max_chars=80)

    def _build_quote(self, text: str, fallback: str) -> str:
        primary = self._sanitize(text, max_chars=self._max_quote_chars)
        if primary:
            return primary
        return self._sanitize(fallback, max_chars=self._max_quote_chars)

    def _sanitize(self, text: str, *, max_chars: int) -> str:
        cleaned = _CONTROL_RE.sub("", str(text))
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            return ""
        if len(cleaned) > max_chars:
            truncated = cleaned[: max_chars - 1].rstrip(" ,;:\n")
            return f"{truncated}…"
        return cleaned

    def _format_citation(self, index: int, url: str) -> str:
        return f"[{index}]({url})"

