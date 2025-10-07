import os
import io
import shutil
import difflib
from typing import Optional, List, Dict, Tuple
from lmstudio import ToolFunctionDef


BASELINE_DIR = os.path.join("internal", ".patch_baselines")


def _ensure_baseline_dir():
    os.makedirs(BASELINE_DIR, exist_ok=True)


def _walk_files(root: str) -> List[str]:
    files: List[str] = []
    for r, _dirs, fs in os.walk(root):
        for name in fs:
            p = os.path.join(r, name)
            files.append(os.path.normpath(p))
    return files


def _read_text_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.readlines()


def _file_rel(path: str, base: str) -> str:
    return os.path.relpath(path, start=base).replace('\\', '/')


def _unified_diff(from_path: str, to_path: str, a_label: str, b_label: str, context: int = 3) -> List[str]:
    a_lines = _read_text_lines(from_path) if os.path.exists(from_path) else []
    b_lines = _read_text_lines(to_path) if os.path.exists(to_path) else []
    return list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=a_label,
            tofile=b_label,
            n=context,
        )
    )


def _dir_diff(from_dir: str, to_dir: str, context: int = 3) -> Tuple[str, Dict[str, Dict[str, int]]]:
    from_files = set(_walk_files(from_dir)) if from_dir and os.path.exists(from_dir) else set()
    to_files = set(_walk_files(to_dir)) if to_dir and os.path.exists(to_dir) else set()
    all_rel = sorted({_file_rel(p, from_dir) for p in from_files} | {_file_rel(p, to_dir) for p in to_files})
    patches: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}
    for rel in all_rel:
        a = os.path.join(from_dir, rel) if from_dir else None
        b = os.path.join(to_dir, rel) if to_dir else None
        a_exists = a and os.path.exists(a)
        b_exists = b and os.path.exists(b)
        a_label = f"a/{rel}" if a_exists else f"a/{rel}"
        b_label = f"b/{rel}" if b_exists else f"b/{rel}"
        if a_exists or b_exists:
            diff_lines = _unified_diff(a if a_exists else "", b if b_exists else "", a_label, b_label, context)
            if diff_lines:
                patches.extend(diff_lines)
                # Compute additions and deletions for this file
                adds = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
                dels = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))
                summary[rel] = {"additions": adds, "deletions": dels}
    return ("".join(patches), summary)


def _apply_unified_patch(patch_text: str, root: Optional[str] = None, reverse: bool = False) -> Dict[str, any]:
    root_dir = root or os.getcwd()
    lines = patch_text.splitlines()
    i = 0
    results: List[Dict[str, str]] = []

    def apply_hunks(filepath: str, hunks: List[List[str]], is_new: bool, is_delete: bool):
        abs_path = os.path.join(root_dir, filepath)
        try:
            if is_delete:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                results.append({"op": "delete", "path": filepath})
                return
            original = []
            if os.path.exists(abs_path):
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    original = f.read().splitlines()
            new_content = original[:]
            for h in hunks:
                # Build sequence of context/removal lines for matching
                seq = [l[1:] for l in h if l and (l[0] == ' ' or (l[0] == '-' and not reverse) or (l[0] == '+' and reverse))]
                if not seq:
                    additions = [l[1:] for l in h if (l and ((l[0] == '+' and not reverse) or (l[0] == '-' and reverse)))]
                    new_content = new_content + additions
                    continue
                # in-order search
                idx = 0
                start_idx = None
                j = 0
                while j < len(new_content) and idx < len(seq):
                    if new_content[j] == seq[idx]:
                        if start_idx is None:
                            start_idx = j
                        idx += 1
                    j += 1
                if idx < len(seq) or start_idx is None:
                    continue
                end_idx = j
                rebuilt: List[str] = []
                for l in h:
                    if not l:
                        continue
                    tag, body = l[0], l[1:]
                    if tag == ' ':
                        rebuilt.append(body)
                    elif (tag == '+' and not reverse) or (tag == '-' and reverse):
                        rebuilt.append(body)
                    elif (tag == '-' and not reverse) or (tag == '+' and reverse):
                        # removed
                        pass
                new_content = new_content[:start_idx] + rebuilt + new_content[end_idx:]
            os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_content))
                if new_content and not new_content[-1].endswith("\n"):
                    f.write("\n")
            results.append({"op": "add" if is_new and not os.path.exists(abs_path) else "update", "path": filepath})
        except Exception as e:
            results.append({"op": "error", "path": filepath, "message": str(e)})

    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            # Parse header until we find --- and +++
            i += 1
            a_path = None
            b_path = None
            while i < len(lines) and not lines[i].startswith('--- '):
                i += 1
            if i < len(lines) and lines[i].startswith('--- '):
                a_line = lines[i]
                i += 1
            else:
                a_line = ''
            if i < len(lines) and lines[i].startswith('+++ '):
                b_line = lines[i]
                i += 1
            else:
                b_line = ''
            if a_line.startswith('--- '):
                a_path = a_line[4:].strip()
            if b_line.startswith('+++ '):
                b_path = b_line[4:].strip()
            # Normalize paths
            def strip_prefix(p: Optional[str]) -> Optional[str]:
                if not p:
                    return None
                if p == '/dev/null':
                    return None
                if p.startswith('a/') or p.startswith('b/'):
                    return p[2:]
                return p
            a_rel = strip_prefix(a_path)
            b_rel = strip_prefix(b_path)
            is_new = a_rel is None and b_rel is not None
            is_delete = b_rel is None and a_rel is not None
            target = (b_rel if not reverse else a_rel) or (a_rel or b_rel)
            hunks: List[List[str]] = []
            # Collect hunks for this file
            while i < len(lines) and not lines[i].startswith('diff --git '):
                if lines[i].startswith('@@'):
                    # start new hunk
                    if hunks and hunks[-1]:
                        # already have a hunk; continue building
                        pass
                    cur: List[str] = []
                    i += 1
                    # gather until next hunk header or next file
                    while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff --git '):
                        cur.append(lines[i])
                        i += 1
                    hunks.append(cur)
                    continue
                else:
                    i += 1
            if target:
                apply_hunks(target, hunks, is_new, is_delete)
            continue
        elif line.startswith('--- ') and i + 1 < len(lines) and lines[i + 1].startswith('+++ '):
            # Minimal support for patch without diff --git lines
            a_path = lines[i][4:].strip()
            b_path = lines[i + 1][4:].strip()
            i += 2
            def strip_prefix2(p: str) -> Optional[str]:
                if p == '/dev/null':
                    return None
                if p.startswith('a/') or p.startswith('b/'):
                    return p[2:]
                return p
            a_rel = strip_prefix2(a_path)
            b_rel = strip_prefix2(b_path)
            is_new = a_rel is None and b_rel is not None
            is_delete = b_rel is None and a_rel is not None
            target = (b_rel if not reverse else a_rel) or (a_rel or b_rel)
            hunks: List[List[str]] = []
            while i < len(lines) and lines[i].startswith('@@'):
                i += 1
                cur: List[str] = []
                while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('--- '):
                    cur.append(lines[i])
                    i += 1
                hunks.append(cur)
            if target:
                apply_hunks(target, hunks, is_new, is_delete)
            continue
        else:
            # Skip unrelated lines
            i += 1

    return {"status": "success", "results": results}


def patch(
    operation: str,
    from_path: Optional[str] = None,
    to_path: Optional[str] = None,
    patch_text: Optional[str] = None,
    context: Optional[int] = None,
    root: Optional[str] = None,
    reverse: Optional[bool] = None,
    snapshot_name: Optional[str] = None,
):
    """Patch and diff suite similar to git.

    Operations:
    - diff: from_path, to_path, [context]
    - generate: alias for diff
    - apply: patch_text, [root, reverse]
    - summary: patch_text -> totals by file and overall
    - snapshot: snapshot_name, from_path (root)
    - list_snapshots: no params
    - diff_snapshot: snapshot_name, to_path (root), [context]
    """
    try:
        op = (operation or "").strip().lower()
        if op in ("diff", "generate"):
            if not from_path or not to_path:
                return {"status": "error", "message": "'from_path' and 'to_path' are required"}
            patch_str, summary = _dir_diff(from_path, to_path, int(context or 3))
            totals = {
                "files_changed": len(summary),
                "additions": sum(v["additions"] for v in summary.values()),
                "deletions": sum(v["deletions"] for v in summary.values()),
            }
            return {"status": "success", "patch": patch_str, "summary": summary, "totals": totals}

        if op in ("apply", "apply_patch"):
            if not patch_text:
                return {"status": "error", "message": "'patch_text' is required for apply"}
            res = _apply_unified_patch(patch_text, root=root, reverse=bool(reverse))
            return res

        if op == "summary":
            if not patch_text:
                return {"status": "error", "message": "'patch_text' is required for summary"}
            adds = dels = 0
            files: Dict[str, Dict[str, int]] = {}
            cur_file = None
            for l in patch_text.splitlines():
                if l.startswith('diff --git '):
                    cur_file = None
                elif l.startswith('--- '):
                    # will see +++ next; determine path
                    pass
                elif l.startswith('+++ '):
                    p = l[4:].strip()
                    if p.startswith('b/'):
                        p = p[2:]
                    cur_file = p
                    files.setdefault(cur_file, {"additions": 0, "deletions": 0})
                elif l.startswith('+') and not l.startswith('+++'):
                    adds += 1
                    if cur_file:
                        files.setdefault(cur_file, {"additions": 0, "deletions": 0})["additions"] += 1
                elif l.startswith('-') and not l.startswith('---'):
                    dels += 1
                    if cur_file:
                        files.setdefault(cur_file, {"additions": 0, "deletions": 0})["deletions"] += 1
            return {"status": "success", "totals": {"additions": adds, "deletions": dels, "files_changed": len(files)}, "files": files}

        if op == "snapshot":
            if not snapshot_name or not from_path:
                return {"status": "error", "message": "'snapshot_name' and 'from_path' are required"}
            _ensure_baseline_dir()
            dst = os.path.join(BASELINE_DIR, snapshot_name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(from_path, dst)
            return {"status": "success", "snapshot_path": dst}

        if op == "list_snapshots":
            _ensure_baseline_dir()
            items = []
            for name in sorted(os.listdir(BASELINE_DIR)):
                p = os.path.join(BASELINE_DIR, name)
                if os.path.isdir(p):
                    items.append({"name": name, "path": p})
            return {"status": "success", "snapshots": items}

        if op == "diff_snapshot":
            if not snapshot_name or not to_path:
                return {"status": "error", "message": "'snapshot_name' and 'to_path' are required"}
            base = os.path.join(BASELINE_DIR, snapshot_name)
            if not os.path.isdir(base):
                return {"status": "error", "message": "snapshot not found"}
            patch_str, summary = _dir_diff(base, to_path, int(context or 3))
            totals = {
                "files_changed": len(summary),
                "additions": sum(v["additions"] for v in summary.values()),
                "deletions": sum(v["deletions"] for v in summary.values()),
            }
            return {"status": "success", "patch": patch_str, "summary": summary, "totals": totals}

        return {"status": "error", "message": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


patch_tool = ToolFunctionDef(
    name="patch",
    description=(
        "Patch/diff suite: diff directories, apply unified patches, summarize changes, manage snapshots."
    ),
    parameters={
        "operation": str,
        "from_path": str,
        "to_path": str,
        "patch_text": str,
        "context": int,
        "root": str,
        "reverse": bool,
        "snapshot_name": str,
    },
    implementation=patch,
)
