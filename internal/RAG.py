import os
import re
import math
import time
import json
import html
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # Will raise at call time if WebRAG.fetch is used


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
            cur: deque[str] = deque()
            cur_lens: deque[int] = deque()
            off = 0
            current_len = 0
            for ln in lines:
                line_len = len(ln) + 1
                cur.append(ln)
                cur_lens.append(line_len)
                current_len += line_len
                if current_len >= self.max_chars:
                    chunk_txt = "\n".join(cur)
                    out.append(DocumentChunk(source_path, off, chunk_txt, self.kind))
                    off += max(0, len(chunk_txt) - self.overlap)
                    if self.overlap > 0:
                        keep = max(
                            1,
                            self.overlap
                            // max(1, (len(chunk_txt) // max(1, len(cur)))),
                        )
                    else:
                        keep = 0
                    if keep < len(cur):
                        drop = len(cur) - keep
                        for _ in range(drop):
                            current_len -= cur_lens.popleft()
                            cur.popleft()
                    else:
                        if keep == 0:
                            cur.clear()
                            cur_lens.clear()
                            current_len = 0
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
        self._chunk_index: Dict[Tuple[str, int], int] = {}

    def ingest_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        for ch in chunks:
            self.chunks.append(ch)
            toks = self.tokenizer.tokenize(ch.text)
            self.docs_tokens.append(toks)
        self._chunk_index = {
            (chunk.source_path, chunk.offset): idx
            for idx, chunk in enumerate(self.chunks)
        }
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
        qvec = self.ranker.tfidf.embed_query(q) if self.ranker else {}
        # Light re-ranking: mix BM25 and cosine on candidate subset
        candidate_indices = []
        for c in candidates:
            key = (c.get("path"), c.get("offset"))
            if key[0] is None or key[1] is None:
                candidate_indices.append(None)
                continue
            candidate_indices.append(self._chunk_index.get((key[0], key[1])))

        valid_indices = {idx for idx in candidate_indices if idx is not None}
        bm_scores = {i: self.ranker.bm25.score(q, i) for i in valid_indices} if self.ranker else {}
        if self.ranker:
            if qvec:
                tf_scores = {i: self.ranker.tfidf.cosine(qvec, i) for i in valid_indices}
            else:
                tf_scores = {i: 0.0 for i in valid_indices}
        else:
            tf_scores = {}
        for c, idx in zip(candidates, candidate_indices):
            if idx is None:
                continue
            c["score"] = float(alpha * bm_scores.get(idx, 0.0) + (1 - alpha) * tf_scores.get(idx, 0.0))
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
    def __init__(self, user_agent: Optional[str] = None, timeout: int = 15) -> None:
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com)"
        self.timeout = timeout
        self.session = requests.Session() if requests else None
        self.index = _RagIndex(kind="web")
        self._lock = threading.Lock()

        try:
            from duckduckgo_search import DDGS  # type: ignore
            self._ddgs_cls = DDGS
        except Exception:
            self._ddgs_cls = None

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

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
            raise RuntimeError("requests is required for WebRAG")
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
            rs = self._search_ddg(q, max_results=max_search_results)
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
