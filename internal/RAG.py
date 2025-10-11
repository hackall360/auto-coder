import os
import re
import math
import time
import json
import html
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # Will raise at call time if WebRAG.fetch is used

try:  # pragma: no cover - optional dependency
    from .web_playwright import PlaywrightWebClient
except Exception:  # pragma: no cover
    PlaywrightWebClient = None  # type: ignore


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_DEFAULT_STOPWORDS = {
    "the","a","an","and","or","for","to","of","in","on","it","is","are","was","were","be","with","as","by","at","from","that","this","these","those","we","you","your","our","their","them","they","i"
}


@dataclass
class DocumentChunk:
    source_path: str
    offset: int
    text: str
    kind: str  # e.g., "code", "doc", "web"


class _Tokenizer:
    def __init__(self, stopwords: Optional[set[str]] = None) -> None:
        self.stopwords = stopwords or set()

    def tokenize(self, text: str) -> List[str]:
        toks = [t.lower() for t in _WORD_RE.findall(text)]
        if not self.stopwords:
            return toks
        return [t for t in toks if t not in self.stopwords and len(t) > 1]


class _BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.N if self.N else 0.0
        # Build df and idf
        df: Dict[str, int] = {}
        for d in docs:
            seen = set(d)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.idf: Dict[str, float] = {t: math.log((self.N - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
        # Term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for d in docs:
            counts: Dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)

    def score(self, q: List[str], idx: int) -> float:
        tf = self.tf[idx]
        dlen = len(self.docs[idx]) or 1
        score = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dlen / (self.avgdl or 1.0))
            score += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return score

    def topk(self, q: List[str], k: int) -> List[Tuple[int, float]]:
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class _TfIdf:
    def __init__(self, docs: List[List[str]]) -> None:
        self.docs = docs
        self.N = len(docs)
        # Vocabulary and df
        self.df: Dict[str, int] = {}
        for d in docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf: Dict[str, float] = {t: math.log((self.N + 1) / (c + 1)) + 1.0 for t, c in self.df.items()}
        # Precompute tf-idf vectors with L2 normalization
        self.vecs: List[Dict[str, float]] = []
        for d in docs:
            tf: Dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            vec: Dict[str, float] = {t: (tf[t] / len(d)) * self.idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vecs.append({t: v / norm for t, v in vec.items()})

    def embed_query(self, q: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = {}
        for t in q:
            if t in self.idf:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            return {}
        vec: Dict[str, float] = {t: (tf[t] / len(q)) * self.idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def cosine(self, qvec: Dict[str, float], idx: int) -> float:
        dvec = self.vecs[idx]
        return sum(qvec.get(t, 0.0) * v for t, v in dvec.items())

    def topk(self, q: List[str], k: int) -> List[Tuple[int, float]]:
        qvec = self.embed_query(q)
        scores = [(i, self.cosine(qvec, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class _HybridRanker:
    def __init__(self, docs: List[List[str]]) -> None:
        self.bm25 = _BM25(docs)
        self.tfidf = _TfIdf(docs)

    def topk(self, q: List[str], k: int, alpha: float = 0.6) -> List[Tuple[int, float]]:
        bm = dict(self.bm25.topk(q, max(k * 3, k)))
        tf = dict(self.tfidf.topk(q, max(k * 3, k)))
        keys = set(bm.keys()) | set(tf.keys())
        scored = []
        for i in keys:
            s = alpha * bm.get(i, 0.0) + (1 - alpha) * tf.get(i, 0.0)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class _Chunker:
    def __init__(self, kind: str, max_chars: int = 1200, overlap: int = 200) -> None:
        self.kind = kind
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str, source_path: str) -> List[DocumentChunk]:
        out: List[DocumentChunk] = []
        if self.kind == "doc":
            parts = re.split(r"\n\s*\n", text)
            buf = ""
            off = 0
            for p in parts:
                if not p.strip():
                    continue
                if len(buf) + len(p) + 1 <= self.max_chars:
                    if buf:
                        buf += "\n\n" + p
                    else:
                        buf = p
                else:
                    if buf:
                        out.append(DocumentChunk(source_path, off, buf, self.kind))
                        off += max(0, len(buf) - self.overlap)
                    buf = p
            if buf:
                out.append(DocumentChunk(source_path, off, buf, self.kind))
        else:
            lines = text.splitlines()
            cur: List[str] = []
            off = 0
            for ln in lines:
                cur.append(ln)
                if sum(len(x) + 1 for x in cur) >= self.max_chars:
                    chunk_txt = "\n".join(cur)
                    out.append(DocumentChunk(source_path, off, chunk_txt, self.kind))
                    off += max(0, len(chunk_txt) - self.overlap)
                    cur = cur[-max(1, self.overlap // max(1, (len(chunk_txt) // max(1, len(cur))))) :]
            if cur:
                out.append(DocumentChunk(source_path, off, "\n".join(cur), self.kind))
        return out


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _is_textual(filename: str) -> bool:
    name = filename.lower()
    text_exts = (
        ".txt",".md",".rst",".py",".js",".ts",".tsx",".jsx",".java",".kt",".go",".rb",".php",".c",".h",".cpp",".hpp",".cs",".swift",".scala",".json",".yml",".yaml",".toml",".ini",".cfg",".csv",".tsv"
    )
    return name.endswith(text_exts)


class _RagIndex:
    def __init__(self, kind: str = "doc") -> None:
        self.kind = kind
        self.tokenizer = _Tokenizer(_DEFAULT_STOPWORDS)
        self.chunks: List[DocumentChunk] = []
        self.docs_tokens: List[List[str]] = []
        self.ranker: Optional[_HybridRanker] = None

    def ingest_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        for ch in chunks:
            self.chunks.append(ch)
            toks = self.tokenizer.tokenize(ch.text)
            self.docs_tokens.append(toks)
        self.ranker = _HybridRanker(self.docs_tokens) if self.docs_tokens else None

    def ingest_directory(self, root: str, include_exts: Optional[Sequence[str]] = None, exclude_dirs: Optional[Sequence[str]] = None, max_files: Optional[int] = None) -> int:
        count = 0
        ex_dirs = {os.path.normcase(d) for d in (exclude_dirs or [])}
        inc_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (include_exts or [])}
        chunker = _Chunker(self.kind)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if os.path.normcase(os.path.join(dirpath, d)) not in ex_dirs]
            for name in filenames:
                if max_files is not None and count >= max_files:
                    break
                path = os.path.join(dirpath, name)
                if include_exts:
                    if not any(name.lower().endswith(ext) for ext in inc_exts):
                        continue
                else:
                    if not _is_textual(name):
                        continue
                txt = _read_text(path)
                if not txt:
                    continue
                chs = chunker.chunk(txt, path)
                self.ingest_chunks(chs)
                count += 1
        return count

    def search(self, query: str, top_k: int = 10, alpha: float = 0.6) -> List[Dict[str, Any]]:
        if not self.ranker:
            return []
        q = self.tokenizer.tokenize(query)
        results: List[Dict[str, Any]] = []
        for idx, score in self.ranker.topk(q, top_k, alpha=alpha):
            ch = self.chunks[idx]
            results.append({
                "score": float(score),
                "path": ch.source_path,
                "offset": ch.offset,
                "text": ch.text,
                "kind": ch.kind,
            })
        return results

    def rerank(self, query: str, candidates: List[Dict[str, Any]], alpha: float = 0.5) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        q = self.tokenizer.tokenize(query)
        # Light re-ranking: mix BM25 and cosine on candidate subset
        idxs = [self.chunks.index(DocumentChunk(c["path"], c["offset"], c["text"], c.get("kind", self.kind))) for c in candidates]
        bm_scores = {i: self.ranker.bm25.score(q, i) for i in idxs} if self.ranker else {}
        tf_scores = {i: self.ranker.tfidf.cosine(self.ranker.tfidf.embed_query(q), i) for i in idxs} if self.ranker else {}
        for c in candidates:
            try:
                i = self.chunks.index(DocumentChunk(c["path"], c["offset"], c["text"], c.get("kind", self.kind)))
                c["score"] = float(alpha * bm_scores.get(i, 0.0) + (1 - alpha) * tf_scores.get(i, 0.0))
            except Exception:
                pass
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates


class CodebaseRAG:
    def __init__(self, root: str, include_exts: Optional[Sequence[str]] = None, exclude_dirs: Optional[Sequence[str]] = None) -> None:
        self.index = _RagIndex(kind="code")
        self.root = root
        self.include_exts = include_exts or (".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs")
        self.exclude_dirs = list(exclude_dirs or (".git", "node_modules", "dist", "build", "venv", "__pycache__"))

    def build(self, max_files: Optional[int] = None) -> int:
        return self.index.ingest_directory(self.root, include_exts=self.include_exts, exclude_dirs=[os.path.join(self.root, d) for d in self.exclude_dirs], max_files=max_files)

    def query(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.index.search(text, top_k=top_k)

    def rerank(self, text: str, results: List[Dict[str, Any]], alpha: float = 0.5) -> List[Dict[str, Any]]:
        return self.index.rerank(text, results, alpha=alpha)


class DirectoryRAG:
    def __init__(self, root: str, include_exts: Optional[Sequence[str]] = None, exclude_dirs: Optional[Sequence[str]] = None) -> None:
        self.index = _RagIndex(kind="doc")
        self.root = root
        self.include_exts = include_exts  # If None, uses textual heuristic
        self.exclude_dirs = list(exclude_dirs or (".git", "node_modules", "dist", "build", "venv", "__pycache__"))

    def build(self, max_files: Optional[int] = None) -> int:
        return self.index.ingest_directory(self.root, include_exts=self.include_exts, exclude_dirs=[os.path.join(self.root, d) for d in self.exclude_dirs], max_files=max_files)

    def query(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.index.search(text, top_k=top_k)

    def rerank(self, text: str, results: List[Dict[str, Any]], alpha: float = 0.5) -> List[Dict[str, Any]]:
        return self.index.rerank(text, results, alpha=alpha)


class SystemRAG:
    def __init__(self, roots: Optional[Sequence[str]] = None, include_exts: Optional[Sequence[str]] = None, exclude_dirs: Optional[Sequence[str]] = None) -> None:
        self.index = _RagIndex(kind="doc")
        if roots:
            self.roots = list(roots)
        else:
            self.roots = [os.path.expanduser("~")]
        self.include_exts = include_exts
        self.exclude_dirs = list(exclude_dirs or (".git", "node_modules", "dist", "build", "venv", "__pycache__"))

    def build(self, max_files_per_root: Optional[int] = 200) -> int:
        total = 0
        for r in self.roots:
            total += self.index.ingest_directory(r, include_exts=self.include_exts, exclude_dirs=[os.path.join(r, d) for d in self.exclude_dirs], max_files=max_files_per_root)
        return total

    def query(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.index.search(text, top_k=top_k)

    def rerank(self, text: str, results: List[Dict[str, Any]], alpha: float = 0.5) -> List[Dict[str, Any]]:
        return self.index.rerank(text, results, alpha=alpha)


class WebRAG:
    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout: int = 15,
        *,
        user_agent_pool: Optional[Sequence[str]] = None,
        proxy: Optional[str | Mapping[str, str]] = None,
        incognito_contexts: Optional[bool] = None,
        anonymous_browsing: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        env_anon = os.getenv("AUTO_CODER_WEB_ANONYMIZE")
        if anonymous_browsing is None and env_anon is not None:
            anonymous_browsing = env_anon.strip().lower() in {"1", "true", "yes", "on"}
        self.anonymous_browsing = bool(anonymous_browsing)

        env_proxy = os.getenv("AUTO_CODER_WEB_PROXY")
        if proxy is None and env_proxy:
            proxy = env_proxy

        env_user_agents = os.getenv("AUTO_CODER_WEB_USER_AGENTS")
        if user_agent_pool is None and env_user_agents:
            user_agent_pool = self._parse_user_agent_pool(env_user_agents)

        if incognito_contexts is None:
            env_incognito = os.getenv("AUTO_CODER_WEB_INCOGNITO")
            if env_incognito is not None:
                incognito_contexts = env_incognito.strip().lower() in {"1", "true", "yes", "on"}
            elif self.anonymous_browsing:
                incognito_contexts = True
        self.incognito_contexts = bool(incognito_contexts if incognito_contexts is not None else True)

        pool: List[str] = []
        if user_agent_pool:
            pool.extend([ua.strip() for ua in user_agent_pool if ua and ua.strip()])

        default_agent = "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com)"
        self._rng = random.Random(random_seed)
        self._user_agent_pool = pool
        self.user_agent = user_agent or (pool[0] if pool else default_agent)
        if not self._user_agent_pool and self.anonymous_browsing:
            self._user_agent_pool = [self.user_agent]

        self.timeout = timeout
        self.session = requests.Session() if requests else None
        self._requests_proxies, self._playwright_proxy = self._normalize_proxy(proxy)
        if self.session and self._requests_proxies:
            self.session.proxies.update(self._requests_proxies)
        self.index = _RagIndex(kind="web")
        self._lock = threading.Lock()
        self._playwright_client = None
        if PlaywrightWebClient is not None:
            try:
                self._playwright_client = PlaywrightWebClient(
                    timeout_ms=self.timeout * 1000,
                    user_agent=self.user_agent,
                    user_agent_pool=self._user_agent_pool,
                    proxy=self._playwright_proxy,
                    incognito_contexts=self.incognito_contexts,
                    random_seed=random_seed,
                )
            except Exception:
                self._playwright_client = None

        try:
            from duckduckgo_search import DDGS  # type: ignore
            self._ddgs_cls = DDGS
        except Exception:
            self._ddgs_cls = None

    @staticmethod
    def _parse_user_agent_pool(raw: str) -> List[str]:
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(ua).strip() for ua in parsed if str(ua).strip()]
        except Exception:
            pass
        # Fall back to comma or newline separated text
        parts = re.split(r"[,\n]", raw)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_proxy(proxy: Optional[str | Mapping[str, str]]) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        if proxy is None:
            return None, None
        if isinstance(proxy, str):
            proxy = proxy.strip()
            if not proxy:
                return None, None
            return {"http": proxy, "https": proxy}, {"server": proxy}
        cleaned: Dict[str, str] = {}
        playwright_proxy: Dict[str, str] = {}
        for key, value in proxy.items():
            if value is None:
                continue
            val = str(value).strip()
            if not val:
                continue
            lowered = str(key).lower()
            if lowered in {"http", "https"}:
                cleaned[lowered] = val
            elif lowered == "server":
                playwright_proxy["server"] = val
        if "server" not in playwright_proxy:
            server = cleaned.get("https") or cleaned.get("http")
            if server:
                playwright_proxy["server"] = server
        return (cleaned or None), (playwright_proxy or None)

    def _choose_user_agent(self) -> str:
        if self._user_agent_pool:
            return self._rng.choice(self._user_agent_pool)
        return self.user_agent

    def _playwright_available(self) -> bool:
        return bool(self._playwright_client and self._playwright_client.is_available())

    def _fetch_with_playwright(self, url: str) -> Optional[str]:
        if not self._playwright_available():
            return None
        try:
            return self._playwright_client.render_page_text(url)  # type: ignore[union-attr]
        except Exception:
            return None

    def _search_with_playwright(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not self._playwright_available():
            return []
        try:
            return self._playwright_client.collect_search_results(query, max_results=max_results)  # type: ignore[union-attr]
        except Exception:
            return []

    def _headers(self) -> Dict[str, str]:
        ua = self._choose_user_agent()
        return {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _search_ddg(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self._ddgs_cls is not None:
            try:
                with self._ddgs_cls() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        if not isinstance(r, dict):
                            continue
                        url = r.get("href") or r.get("url")
                        title = r.get("title") or r.get("body") or ""
                        snippet = r.get("body") or r.get("snippet") or ""
                        if url:
                            out.append({"url": url, "title": title, "snippet": snippet})
                return out
            except Exception:
                pass
        # Fallback to HTML endpoint
        if not self.session:
            return out
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return out
            html_text = resp.text
            # Very lightweight parsing to extract links and titles
            # DuckDuckGo HTML page: results contained in <a class="result__a" href="...">Title</a>
            for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.IGNORECASE | re.DOTALL):
                url = html.unescape(m.group(1))
                title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2)))
                out.append({"url": url, "title": title, "snippet": ""})
                if len(out) >= max_results:
                    break
        except Exception:
            return out
        return out

    def _fetch_text(self, url: str) -> Optional[str]:
        if not self.session:
            return None
        try:
            r = self.session.get(url, headers=self._headers(), timeout=self.timeout)
            if r.status_code != 200 or not r.text:
                return None
            text = r.text
            # strip scripts/styles and tags
            text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:500000]
        except Exception:
            return None

    def _generate_query_variants(self, query: str) -> List[str]:
        toks = [t for t in _WORD_RE.findall(query) if t.lower() not in _DEFAULT_STOPWORDS]
        base = " ".join(toks) or query
        variants = [query]
        # Heuristic refinements
        variants.append(base)
        variants.append(base + " documentation")
        variants.append(base + " tutorial")
        variants.append(base + " best practices")
        variants.append(base + " site:stackoverflow.com")
        variants.append(base + " site:github.com")
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for v in variants:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq

    def build_from_results(self, results: List[Dict[str, str]], max_pages: int = 10) -> int:
        chunker = _Chunker("web")
        added = 0
        for r in results[:max_pages]:
            url = r.get("url")
            if not url:
                continue
            txt = self._fetch_with_playwright(url)
            if not txt:
                txt = self._fetch_text(url)
            if not txt:
                continue
            chunks = chunker.chunk(txt, url)
            self.index.ingest_chunks(chunks)
            added += 1
        return added

    def search(self, query: str, top_k: int = 10, max_search_results: int = 20, allow_rewrite: bool = True, alpha: float = 0.6) -> List[Dict[str, Any]]:
        queries = self._generate_query_variants(query) if allow_rewrite else [query]
        seen_urls: set[str] = set()
        results: List[Dict[str, str]] = []
        for q in queries:
            remaining = max_search_results - len(results)
            if remaining <= 0:
                break
            rs: List[Dict[str, str]] = []
            if self._playwright_available():
                try:
                    rs = self._playwright_client.collect_search_results(q, max_results=remaining)  # type: ignore[union-attr]
                except Exception:
                    rs = []
                    self._playwright_client = None
            if not rs:
                rs = self._search_ddg(q, max_results=remaining)
            for r in rs:
                u = r.get("url")
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)
                results.append(r)
            if len(results) >= max_search_results:
                break
        self.index = _RagIndex(kind="web")
        self.build_from_results(results, max_pages=min(len(results), 20))
        return self.index.search(query, top_k=top_k, alpha=alpha)
