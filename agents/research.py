"""High-level research helper built on top of :class:`internal.RAG.WebRAG`."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import re
import threading
import urllib.parse
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from corpus import CorpusManager, get_shared_corpus_manager
from internal.RAG import WebRAG

__all__ = [
    "ResearchSnippet",
    "ResearchResult",
    "ResearchAgent",
    "VariedResearchAgent",
]


LOGGER = logging.getLogger(__name__)


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
        corpus_manager: CorpusManager | None = None,
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
        self._corpus_manager: CorpusManager | None = corpus_manager or get_shared_corpus_manager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def corpus_manager(self) -> CorpusManager | None:
        return self._corpus_manager

    def attach_corpus_manager(self, manager: CorpusManager | None) -> None:
        self._corpus_manager = manager

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
        limited = result.limit(top_k)
        self._record_corpus_event(
            "web_search",
            {
                "query": query,
                "sanitized_query": cleaned_query,
                "top_k": top_k,
                "max_search_results": max_search_results,
                "snippets": [snippet.to_dict() for snippet in limited.snippets],
            },
        )
        return limited

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

    def _record_corpus_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        manager = self._corpus_manager
        if manager is None:
            return
        try:
            manager.record_event(
                source="research.agent",
                payload=dict(payload),
                event_type=event_type,
                tags=("research",),
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("Failed to record research corpus event '%s'", event_type, exc_info=True)

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


class VariedResearchAgent:
    """Compose :class:`ResearchAgent` with profile-based search presets."""

    _DEFAULT_PROFILES: Mapping[str, Mapping[str, Any]] = {
        "skim": {"top_k": 2, "max_search_results": 6, "allow_rewrite": False, "alpha": 0.3},
        "survey": {"top_k": 4, "max_search_results": 12, "allow_rewrite": False, "alpha": 0.45},
        "balanced": {"top_k": 6, "max_search_results": 18, "allow_rewrite": True, "alpha": 0.55},
        "insight": {"top_k": 8, "max_search_results": 24, "allow_rewrite": True, "alpha": 0.65},
        "investigative": {"top_k": 10, "max_search_results": 32, "allow_rewrite": True, "alpha": 0.7},
        "deep_dive": {"top_k": 12, "max_search_results": 40, "allow_rewrite": True, "alpha": 0.8},
        "forensic": {"top_k": 16, "max_search_results": 60, "allow_rewrite": True, "alpha": 0.85},
    }

    _DEFAULT_MODES: Mapping[str, Mapping[str, Any]] = {
        "light": {"profile": "survey"},
        "balanced": {"profile": "balanced"},
        "deep": {"profile": "deep_dive"},
    }

    def __init__(
        self,
        base_agent: ResearchAgent,
        *,
        profiles: Mapping[str, Mapping[str, Any]] | None = None,
        mode_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        default_mode: str = "balanced",
        corpus_manager: CorpusManager | None = None,
    ) -> None:
        self._base_agent = base_agent
        self._profiles = self._build_profiles(profiles)
        self._mode_defaults = self._build_mode_defaults(mode_defaults)
        self._default_mode = self._normalise_mode(default_mode)
        candidate_manager = corpus_manager or getattr(base_agent, "corpus_manager", None) or get_shared_corpus_manager()
        if hasattr(base_agent, "attach_corpus_manager"):
            base_agent.attach_corpus_manager(candidate_manager)
        self._corpus_manager: CorpusManager | None = candidate_manager

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def base_agent(self) -> ResearchAgent:
        return self._base_agent

    @property
    def profiles(self) -> Mapping[str, Mapping[str, Any]]:
        return self._profiles

    @property
    def modes(self) -> Mapping[str, Mapping[str, Any]]:
        return self._mode_defaults

    @property
    def corpus_manager(self) -> CorpusManager | None:
        return self._corpus_manager

    def attach_corpus_manager(self, manager: CorpusManager | None) -> None:
        self._corpus_manager = manager
        if hasattr(self._base_agent, "attach_corpus_manager"):
            self._base_agent.attach_corpus_manager(manager)

    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        profile: str
        | Mapping[str, Any]
        | Sequence[str | Mapping[str, Any]]
        | None = None,
        top_k: int | None = None,
        max_search_results: int | None = None,
        allow_rewrite: bool | None = None,
        alpha: float | None = None,
        audience: str | None = None,
        force_refresh: bool = False,
    ) -> ResearchResult:
        resolved_mode = self._resolve_mode(mode)
        parameters = dict(self._mode_defaults[resolved_mode])

        profile_overrides = self._resolve_profile_overrides(parameters.pop("profile", None))
        parameters.update(profile_overrides)

        if profile is not None:
            parameters.update(self._resolve_profile_overrides(profile))

        if top_k is not None:
            parameters["top_k"] = self._coerce_positive_int(top_k, fallback=parameters.get("top_k", 5))
        if max_search_results is not None:
            parameters["max_search_results"] = self._coerce_positive_int(
                max_search_results,
                fallback=parameters.get("max_search_results", 20),
            )
        if allow_rewrite is not None:
            parameters["allow_rewrite"] = bool(allow_rewrite)
        if alpha is not None:
            parameters["alpha"] = self._coerce_alpha(alpha, fallback=parameters.get("alpha", 0.6))

        # Ensure "top_k" does not exceed "max_search_results" to avoid invalid calls.
        max_results = parameters.get("max_search_results")
        top_results = parameters.get("top_k")
        if isinstance(max_results, int) and isinstance(top_results, int) and top_results > max_results:
            parameters["top_k"] = max_results

        result = self._base_agent.search(
            query,
            top_k=int(parameters.get("top_k", 5)),
            max_search_results=int(parameters.get("max_search_results", 20)),
            allow_rewrite=bool(parameters.get("allow_rewrite", True)),
            audience=audience,
            force_refresh=force_refresh,
            alpha=float(parameters.get("alpha", 0.6)),
        )
        self._record_corpus_event(
            "web_search",
            {
                "query": query,
                "mode": resolved_mode,
                "profile": profile,
                "parameters": parameters,
                "audience": audience,
                "result": result.to_dict() if hasattr(result, "to_dict") else None,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_profiles(
        self, overrides: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, Mapping[str, Any]]:
        profiles: dict[str, Mapping[str, Any]] = {}
        merged = dict(self._DEFAULT_PROFILES)
        if overrides:
            for name, payload in overrides.items():
                if not isinstance(payload, Mapping):
                    continue
                merged[str(name).strip().lower()] = dict(payload)
        for name, payload in merged.items():
            normalised = self._normalise_parameters(payload)
            if normalised:
                profiles[name.lower()] = normalised
        return profiles

    def _build_mode_defaults(
        self, overrides: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, Mapping[str, Any]]:
        modes: dict[str, Mapping[str, Any]] = {}
        merged = dict(self._DEFAULT_MODES)
        if overrides:
            for name, payload in overrides.items():
                if not isinstance(payload, Mapping):
                    continue
                merged[str(name).strip().lower()] = dict(payload)
        for name, payload in merged.items():
            normalised = dict(self._normalise_parameters(payload))
            profile_name = payload.get("profile")
            if isinstance(profile_name, str) and profile_name.strip():
                normalised["profile"] = profile_name.strip().lower()
            if normalised:
                modes[name.lower()] = normalised
        if not modes:
            modes["balanced"] = dict(self._normalise_parameters({"profile": "balanced"}))
        return modes

    def _normalise_mode(self, mode: str) -> str:
        candidate = str(mode or "balanced").strip().lower()
        if candidate in self._mode_defaults:
            return candidate
        return "balanced" if "balanced" in self._mode_defaults else next(iter(self._mode_defaults))

    def _record_corpus_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        manager = self._corpus_manager
        if manager is None:
            return
        try:
            manager.record_event(
                source="research.varied",
                payload=dict(payload),
                event_type=event_type,
                tags=("research", "varied"),
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("Failed to record varied research corpus event '%s'", event_type, exc_info=True)

    def _resolve_mode(self, mode: str | None) -> str:
        if mode is None:
            return self._default_mode
        candidate = str(mode).strip().lower()
        if candidate in self._mode_defaults:
            return candidate
        return self._default_mode

    def _resolve_profile_overrides(
        self,
        profile: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        payloads: list[Mapping[str, Any]] = []
        if isinstance(profile, str):
            lookup = self._profiles.get(profile.strip().lower())
            if lookup:
                payloads.append(lookup)
        elif isinstance(profile, Mapping):
            payloads.append(profile)
        else:
            for item in profile:
                if isinstance(item, str):
                    lookup = self._profiles.get(item.strip().lower())
                    if lookup:
                        payloads.append(lookup)
                elif isinstance(item, Mapping):
                    payloads.append(item)
        merged: dict[str, Any] = {}
        for payload in payloads:
            merged.update(self._normalise_parameters(payload))
        return merged

    @staticmethod
    def _coerce_positive_int(value: Any, *, fallback: int) -> int:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return fallback
        if integer <= 0:
            return fallback
        return integer

    @staticmethod
    def _coerce_alpha(value: Any, *, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        if number < 0:
            return fallback
        if number > 1:
            return 1.0
        return number

    def _normalise_parameters(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if "top_k" in payload:
            params["top_k"] = self._coerce_positive_int(payload.get("top_k"), fallback=5)
        if "max_search_results" in payload:
            params["max_search_results"] = self._coerce_positive_int(
                payload.get("max_search_results"),
                fallback=max(params.get("top_k", 5), 20),
            )
        if "allow_rewrite" in payload:
            params["allow_rewrite"] = bool(payload.get("allow_rewrite"))
        if "alpha" in payload:
            params["alpha"] = self._coerce_alpha(payload.get("alpha"), fallback=0.6)
        for key in ("top_k", "max_search_results"):
            if key not in params and key in payload:
                params[key] = payload[key]
        for key, value in payload.items():
            if key not in params and key != "profile":
                params[key] = value
        return params


