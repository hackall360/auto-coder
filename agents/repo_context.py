"""Repository context agent providing code search, summaries, and git helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

from internal.RAG import CodebaseRAG
from internal.tools import git as git_tools

__all__ = [
    "RepoSearchResult",
    "RepoSymbolResult",
    "FileSummary",
    "DiffFileStat",
    "DiffBundle",
    "RepoContextAgent",
]


@dataclass(slots=True)
class RepoSearchResult:
    """Result returned from a repository search query."""

    path: str
    offset: int
    score: float
    text: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "offset": self.offset,
            "score": self.score,
            "text": self.text,
            "provenance": dict(self.provenance),
        }


@dataclass(slots=True)
class RepoSymbolResult:
    """Match highlighting symbol locations inside a search snippet."""

    path: str
    offset: int
    score: float
    text: str
    spans: tuple[tuple[int, int], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "offset": self.offset,
            "score": self.score,
            "text": self.text,
            "spans": [tuple(span) for span in self.spans],
            "provenance": dict(self.provenance),
        }


@dataclass(slots=True)
class FileSummary:
    """Summary information for a file or module."""

    path: str
    summary: str
    language: str | None
    line_count: int
    size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "summary": self.summary,
            "language": self.language,
            "line_count": self.line_count,
            "size": self.size,
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }
        return payload


@dataclass(slots=True)
class DiffFileStat:
    """Per-file statistics for a diff bundle."""

    path: str
    additions: int
    deletions: int
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "additions": self.additions,
            "deletions": self.deletions,
            "status": self.status,
        }


@dataclass(slots=True)
class DiffBundle:
    """Collection of staged or unstaged changes with provenance information."""

    patch: str
    stats: tuple[DiffFileStat, ...]
    staged: bool
    include_untracked: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch": self.patch,
            "stats": [stat.to_dict() for stat in self.stats],
            "staged": self.staged,
            "include_untracked": self.include_untracked,
            "provenance": dict(self.provenance),
        }


class RepoContextAgent:
    """Encapsulates repository-aware context retrieval and git utilities."""

    def __init__(
        self,
        repo_root: str,
        *,
        include_exts: Sequence[str] | None = None,
        exclude_dirs: Sequence[str] | None = None,
        auto_refresh: bool = True,
        refresh_interval: float = 900.0,
    ) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self._include_exts = tuple(include_exts) if include_exts is not None else None
        self._exclude_dirs = tuple(exclude_dirs) if exclude_dirs is not None else None
        self._rag_lock = threading.Lock()
        self._rag = CodebaseRAG(
            self.repo_root,
            include_exts=self._include_exts,
            exclude_dirs=self._exclude_dirs,
        )
        self._refresh_interval = max(refresh_interval, 60.0)
        self._refresh_lock = threading.Lock()
        self._refresh_thread: threading.Thread | None = None
        self._auto_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_refresh: float = 0.0
        self._last_refresh_count: int = 0
        self._last_refresh_error: Exception | None = None
        # Build index synchronously for the initial snapshot
        self.refresh_index(blocking=True)
        if auto_refresh:
            self.start_background_refresh()

    # ------------------------------------------------------------------
    # Refresh lifecycle
    # ------------------------------------------------------------------
    def start_background_refresh(self) -> None:
        """Start a periodic background refresh loop if not already running."""

        if self._auto_thread and self._auto_thread.is_alive():
            return
        self._stop_event.clear()

        def loop() -> None:
            while not self._stop_event.wait(self._refresh_interval):
                try:
                    self.refresh_index(blocking=True)
                except Exception:
                    # Errors are stored in refresh_index; continue loop
                    continue

        thread = threading.Thread(target=loop, name="RepoContextAutoRefresh", daemon=True)
        self._auto_thread = thread
        thread.start()

    def stop_background_refresh(self) -> None:
        """Stop the periodic refresh loop."""

        self._stop_event.set()
        if self._auto_thread and self._auto_thread.is_alive():
            self._auto_thread.join(timeout=1.0)
        self._auto_thread = None

    def refresh_index(self, *, blocking: bool = False, max_files: int | None = None) -> None:
        """Trigger a rebuild of the CodebaseRAG index."""

        def worker() -> None:
            try:
                rag = CodebaseRAG(
                    self.repo_root,
                    include_exts=self._include_exts,
                    exclude_dirs=self._exclude_dirs,
                )
                count = rag.build(max_files=max_files)
                with self._rag_lock:
                    self._rag = rag
                with self._refresh_lock:
                    self._last_refresh = time.time()
                    self._last_refresh_count = count
                    self._last_refresh_error = None
            except Exception as exc:  # pragma: no cover - safeguard
                with self._refresh_lock:
                    self._last_refresh_error = exc
            finally:
                with self._refresh_lock:
                    self._refresh_thread = None

        with self._refresh_lock:
            if self._refresh_thread and self._refresh_thread.is_alive():
                thread = self._refresh_thread
            else:
                thread = threading.Thread(target=worker, name="RepoContextRefresh", daemon=True)
                self._refresh_thread = thread
                thread.start()
        if blocking:
            thread.join()
            if self._last_refresh_error is not None:
                raise self._last_refresh_error

    # ------------------------------------------------------------------
    # Search APIs
    # ------------------------------------------------------------------
    def search(self, query: str, *, top_k: int = 10) -> list[RepoSearchResult]:
        """Execute a semantic search against the repository index."""

        with self._rag_lock:
            rag = self._rag
        raw_results = rag.query(query, top_k=top_k)
        results: list[RepoSearchResult] = []
        for item in raw_results:
            provenance = {
                "path": item.get("path"),
                "offset": item.get("offset"),
                "score": item.get("score"),
            }
            results.append(
                RepoSearchResult(
                    path=self._rel_path(str(item.get("path"))),
                    offset=int(item.get("offset", 0)),
                    score=float(item.get("score", 0.0)),
                    text=str(item.get("text", "")),
                    provenance=provenance,
                )
            )
        return results

    def symbol_search(self, symbol: str, *, top_k: int = 5) -> list[RepoSymbolResult]:
        """Locate symbol occurrences by combining semantic and lexical search."""

        pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        candidates = self.search(symbol, top_k=top_k * 4)
        matches: list[RepoSymbolResult] = []
        for candidate in candidates:
            snippet = candidate.text
            span_list: list[tuple[int, int]] = []
            for match in pattern.finditer(snippet):
                start = candidate.offset + match.start()
                end = candidate.offset + match.end()
                span_list.append((start, end))
            if not span_list:
                continue
            provenance = dict(candidate.provenance)
            provenance["symbol"] = symbol
            matches.append(
                RepoSymbolResult(
                    path=candidate.path,
                    offset=candidate.offset,
                    score=candidate.score,
                    text=snippet,
                    spans=tuple(span_list),
                    provenance=provenance,
                )
            )
            if len(matches) >= top_k:
                break
        return matches

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def summarize_file(self, path: str, *, max_lines: int = 200) -> FileSummary:
        """Return a simple textual summary for a file."""

        abs_path = self._abs_path(path)
        with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        trimmed = self._trim_lines(lines, max_lines=max_lines)
        language = self._infer_language(abs_path)
        metadata: dict[str, Any] = {
            "head": trimmed[:10],
            "tail": trimmed[-10:] if len(trimmed) > 10 else trimmed,
        }
        summary = "".join(trimmed)
        provenance = {
            "path": self._rel_path(abs_path),
            "line_count": len(lines),
            "captured": len(trimmed),
        }
        return FileSummary(
            path=self._rel_path(abs_path),
            summary=summary,
            language=language,
            line_count=len(lines),
            size=os.path.getsize(abs_path),
            metadata=metadata,
            provenance=provenance,
        )

    def summarize_ast(self, path: str) -> FileSummary:
        """Produce a structure-oriented summary leveraging Python's AST when applicable."""

        abs_path = self._abs_path(path)
        rel_path = self._rel_path(abs_path)
        language = self._infer_language(abs_path)
        if not abs_path.endswith(".py"):
            # Fallback to textual summary for non-Python files
            return self.summarize_file(path)

        with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        try:
            module = ast.parse(source, filename=abs_path)
        except SyntaxError:
            return self.summarize_file(path)

        metadata: dict[str, Any] = {
            "functions": [],
            "classes": [],
        }
        lines = source.splitlines()

        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                metadata["functions"].append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, "end_lineno", node.lineno),
                        "doc": doc.strip(),
                        "async": isinstance(node, ast.AsyncFunctionDef),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                members: list[dict[str, Any]] = []
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members.append(
                            {
                                "name": sub.name,
                                "lineno": sub.lineno,
                                "end_lineno": getattr(sub, "end_lineno", sub.lineno),
                                "async": isinstance(sub, ast.AsyncFunctionDef),
                            }
                        )
                metadata["classes"].append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, "end_lineno", node.lineno),
                        "methods": members,
                        "doc": (ast.get_docstring(node) or "").strip(),
                    }
                )

        summary_lines: list[str] = []
        for cls in metadata["classes"]:
            summary_lines.append(
                f"class {cls['name']} (lines {cls['lineno']}-{cls['end_lineno']}):"
            )
            if cls["doc"]:
                summary_lines.append(f"  doc: {cls['doc']}")
            for method in cls["methods"]:
                prefix = "async def" if method["async"] else "def"
                summary_lines.append(
                    f"  {prefix} {method['name']} (lines {method['lineno']}-{method['end_lineno']})"
                )
        for fn in metadata["functions"]:
            prefix = "async def" if fn["async"] else "def"
            summary_lines.append(f"{prefix} {fn['name']} (lines {fn['lineno']}-{fn['end_lineno']})")
            if fn["doc"]:
                summary_lines.append(f"  doc: {fn['doc']}")
        summary = "\n".join(summary_lines)

        provenance = {
            "path": rel_path,
            "line_count": len(lines),
            "node_count": len(module.body),
        }
        return FileSummary(
            path=rel_path,
            summary=summary,
            language=language,
            line_count=len(lines),
            size=len(source.encode("utf-8")),
            metadata=metadata,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------
    def current_branch(self) -> str | None:
        """Return the currently checked-out git branch."""

        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if result["status"] == "success":
            return result["output"].strip()
        return None

    def checkout(self, branch: str, *, create: bool = False) -> Mapping[str, Any]:
        """Checkout an existing branch or create a new one."""

        if create:
            return self._invoke_git(lambda: git_tools.git_checkout_new_branch(branch))
        return self._invoke_git(lambda: git_tools.git_checkout(branch))

    def stage(self, paths: Iterable[str] | None = None) -> Mapping[str, Any]:
        """Stage changes for commit."""

        if paths:
            payload: MutableMapping[str, Any] = {"status": "success", "staged": []}
            for path in paths:
                result = self._invoke_git(lambda: git_tools.git_add(path))
                if result.get("status") != "success":
                    payload = result
                    break
                payload["staged"].append(self._rel_path(path))
            return dict(payload)
        return self._invoke_git(lambda: git_tools.git_add(None))

    def diff_bundle(self, *, staged: bool = False, include_untracked: bool = False, context: int = 3) -> DiffBundle:
        """Collect patch text and file statistics for the current diff."""

        diff_cmd = ["git", "diff", f"-U{context}"]
        if staged:
            diff_cmd.insert(2, "--staged")
        patch_proc = subprocess.run(diff_cmd, cwd=self.repo_root, capture_output=True, text=True)
        patch_text = patch_proc.stdout

        numstat_cmd = ["git", "diff", "--numstat"]
        if staged:
            numstat_cmd.insert(2, "--staged")
        numstat_proc = subprocess.run(numstat_cmd, cwd=self.repo_root, capture_output=True, text=True)
        stats: list[DiffFileStat] = []
        for line in numstat_proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            adds, dels, path = parts[0], parts[1], parts[2]
            try:
                additions = int(adds) if adds.isdigit() else 0
                deletions = int(dels) if dels.isdigit() else 0
            except ValueError:
                additions, deletions = 0, 0
            stats.append(
                DiffFileStat(
                    path=path,
                    additions=additions,
                    deletions=deletions,
                    status=None,
                )
            )

        if include_untracked:
            untracked = self._list_untracked()
            for path in untracked:
                stats.append(DiffFileStat(path=path, additions=0, deletions=0, status="untracked"))

        provenance = {
            "staged": staged,
            "include_untracked": include_untracked,
        }
        return DiffBundle(
            patch=patch_text,
            stats=tuple(stats),
            staged=staged,
            include_untracked=include_untracked,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Focus helpers for manager/agents
    # ------------------------------------------------------------------
    def focused_files(self, query: str, *, top_k: int = 5) -> list[RepoSearchResult]:
        """Convenience alias for manager requests."""

        return self.search(query, top_k=top_k)

    def focused_diffs(self, *, staged: bool = False, include_untracked: bool = False) -> DiffBundle:
        """Return a diff bundle suitable for manager consumption."""

        return self.diff_bundle(staged=staged, include_untracked=include_untracked)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rel_path(self, path: str) -> str:
        return os.path.relpath(os.path.abspath(path), start=self.repo_root).replace("\\", "/")

    def _abs_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_root, path)

    @staticmethod
    def _trim_lines(lines: Sequence[str], *, max_lines: int) -> list[str]:
        if len(lines) <= max_lines:
            return list(lines)
        head = list(lines[: max_lines // 2])
        tail = list(lines[-max_lines // 2 :])
        return head + ["...\n"] + tail

    @staticmethod
    def _infer_language(path: str) -> str | None:
        _, ext = os.path.splitext(path)
        if not ext:
            return None
        return ext.lstrip(".")

    def _list_untracked(self) -> list[str]:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.repo_root, capture_output=True, text=True
        )
        untracked: list[str] = []
        for line in status_proc.stdout.splitlines():
            if line.startswith("??"):
                untracked.append(line[3:])
        return untracked

    def _invoke_git(self, func: Any) -> Mapping[str, Any]:
        with self._within_repo():
            result = func()
        return result if isinstance(result, Mapping) else {"status": "error", "message": "Invalid git response"}

    def _run_git(self, args: Sequence[str]) -> Mapping[str, Any]:
        proc = subprocess.run(["git", *args], cwd=self.repo_root, capture_output=True, text=True)
        if proc.returncode == 0:
            return {"status": "success", "output": proc.stdout}
        return {"status": "error", "message": proc.stderr, "returncode": proc.returncode}

    @contextmanager
    def _within_repo(self) -> Iterator[None]:
        current = os.getcwd()
        try:
            os.chdir(self.repo_root)
            yield
        finally:
            os.chdir(current)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.stop_background_refresh()
        except Exception:
            pass
