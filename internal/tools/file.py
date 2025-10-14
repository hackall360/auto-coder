import os
import re
import time
import json
import shutil
import platform
import subprocess
import difflib
import hashlib
import base64
import zipfile
import tempfile
from collections import deque
from typing import Any, Dict, Optional, List

from lmstudio import ToolFunctionDef

from corpus import record_event as record_corpus_event


def _record_file_event(event_type: str, path: str, payload: Dict[str, Any]) -> None:
    try:
        record_corpus_event(
            source="tool.file",
            payload=payload,
            event_type=event_type,
            tags=("tool", "file"),
        )
    except Exception:
        pass


def create_file(path: str, content: str = ""):
    try:
        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        _record_file_event(
            "file_write",
            path,
            {"status": "created", "size": len(content)},
        )
        return {"status": "success", "message": f"File created at {path}"}
    except Exception as e:
        _record_file_event(
            "file_write",
            path,
            {"status": "error", "message": str(e)},
        )
        return {"status": "error", "message": str(e)}


def read_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        _record_file_event(
            "file_read",
            path,
            {"status": "success", "size": len(content)},
        )
        return {"status": "success", "content": content}
    except Exception as e:
        _record_file_event(
            "file_read",
            path,
            {"status": "error", "message": str(e)},
        )
        return {"status": "error", "message": str(e)}


def write_file(path: str, content: str):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        _record_file_event(
            "file_write",
            path,
            {"status": "written", "size": len(content)},
        )
        return {"status": "success", "message": f"Content written to {path}"}
    except Exception as e:
        _record_file_event(
            "file_write",
            path,
            {"status": "error", "message": str(e)},
        )
        return {"status": "error", "message": str(e)}


def append_to_file(path: str, content: str):
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        _record_file_event(
            "file_write",
            path,
            {"status": "appended", "size": len(content)},
        )
        return {"status": "success", "message": f"Content appended to {path}"}
    except Exception as e:
        _record_file_event(
            "file_write",
            path,
            {"status": "error", "message": str(e)},
        )
        return {"status": "error", "message": str(e)}


def patch_file(path: str, patch_type: str, **kwargs):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        modified_lines = original_lines.copy()
        patch_info = {"patch_type": patch_type}

        if patch_type == "line_replace":
            line_numbers = kwargs.get("line_numbers", [])
            new_content = kwargs.get("new_content", "")
            for line_num in line_numbers:
                if 0 <= line_num < len(modified_lines):
                    modified_lines[line_num] = new_content + "\n"
                else:
                    return {"status": "error", "message": f"Line number {line_num} is out of range"}
            patch_info["line_numbers"] = line_numbers
            patch_info["new_content"] = new_content

        elif patch_type == "line_insert":
            line_number = kwargs.get("line_number", 0)
            new_content = kwargs.get("new_content", "")
            if 0 <= line_number <= len(modified_lines):
                modified_lines.insert(line_number, new_content + "\n")
            else:
                return {"status": "error", "message": f"Line number {line_number} is out of range"}
            patch_info["line_number"] = line_number
            patch_info["new_content"] = new_content

        elif patch_type == "line_delete":
            line_numbers = kwargs.get("line_numbers", [])
            for line_num in sorted(line_numbers, reverse=True):
                if 0 <= line_num < len(modified_lines):
                    modified_lines.pop(line_num)
                else:
                    return {"status": "error", "message": f"Line number {line_num} is out of range"}
            patch_info["line_numbers"] = line_numbers

        elif patch_type == "content_replace":
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            original_content = "".join(original_lines)
            modified_content = original_content.replace(old_text, new_text)
            modified_lines = modified_content.splitlines(keepends=True)
            if not modified_content.endswith('\n'):
                modified_lines.append('\n')
            patch_info["old_text"] = old_text
            patch_info["new_text"] = new_text

        elif patch_type == "content_append":
            new_text = kwargs.get("new_text", "")
            modified_lines.append(new_text + "\n")
            patch_info["new_text"] = new_text

        elif patch_type == "content_prepend":
            new_text = kwargs.get("new_text", "")
            modified_lines.insert(0, new_text + "\n")
            patch_info["new_text"] = new_text

        else:
            return {"status": "error", "message": f"Unsupported patch type: {patch_type}"}

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)

        diff = list(difflib.unified_diff(original_lines, modified_lines, fromfile="original", tofile="modified"))
        _record_file_event(
            "file_patch",
            path,
            {
                "status": "success",
                "patch_type": patch_type,
                "lines_changed": len(diff),
            },
        )
        return {"status": "success", "message": f"File patched using {patch_type} method", "patch_info": patch_info, "diff": diff}
    except Exception as e:
        _record_file_event(
            "file_patch",
            path,
            {"status": "error", "patch_type": patch_type, "message": str(e)},
        )
        return {"status": "error", "message": str(e)}


def move_file(src_path: str, dst_path: str):
    try:
        shutil.move(src_path, dst_path)
        return {"status": "success", "message": f"File moved from {src_path} to {dst_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def copy_file(src_path: str, dst_path: str):
    try:
        shutil.copy2(src_path, dst_path)
        return {"status": "success", "message": f"File copied from {src_path} to {dst_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_file(path: str):
    try:
        os.remove(path)
        return {"status": "success", "message": f"File deleted at {path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def rename_file(old_path: str, new_path: str):
    try:
        os.rename(old_path, new_path)
        return {"status": "success", "message": f"File renamed from {old_path} to {new_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def open_file(path: str):
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
            return {"status": "success", "message": f"File opened at {path}"}
        opener = "open" if system == "Darwin" else (shutil.which("xdg-open") and "xdg-open")
        if opener:
            result = subprocess.run([opener, path], capture_output=True, text=True)
            if result.returncode == 0:
                return {"status": "success", "message": f"File opened at {path}"}
            return {"status": "error", "message": result.stderr or result.stdout}
        return {"status": "error", "message": "No suitable opener found (xdg-open/open)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def open_folder(dirpath: str):
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(dirpath)  # type: ignore[attr-defined]
            return {"status": "success", "message": f"Folder opened at {dirpath}"}
        opener = "open" if system == "Darwin" else (shutil.which("xdg-open") and "xdg-open")
        if opener:
            result = subprocess.run([opener, dirpath], capture_output=True, text=True)
            if result.returncode == 0:
                return {"status": "success", "message": f"Folder opened at {dirpath}"}
            return {"status": "error", "message": result.stderr or result.stdout}
        return {"status": "error", "message": "No suitable opener found (xdg-open/open)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_directory(dirpath: str, recursive: bool = False):
    try:
        entries = []
        if recursive:
            for root, dirs, files in os.walk(dirpath):
                for name in dirs:
                    p = os.path.join(root, name)
                    try:
                        st = os.stat(p)
                        entries.append({
                            "path": p,
                            "type": "dir",
                            "size": st.st_size,
                            "mtime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
                        })
                    except OSError:
                        entries.append({"path": p, "type": "dir", "error": "stat failed"})
                for name in files:
                    p = os.path.join(root, name)
                    try:
                        st = os.stat(p)
                        entries.append({
                            "path": p,
                            "type": "file",
                            "size": st.st_size,
                            "mtime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
                        })
                    except OSError:
                        entries.append({"path": p, "type": "file", "error": "stat failed"})
        else:
            for name in os.listdir(dirpath):
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                    entries.append({
                        "path": p,
                        "type": "dir" if os.path.isdir(p) else "file",
                        "size": st.st_size,
                        "mtime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
                    })
                except OSError:
                    entries.append({"path": p, "type": "unknown", "error": "stat failed"})
        return {"status": "success", "entries": entries}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def make_directory(dirpath: str, exist_ok: bool = True):
    try:
        os.makedirs(dirpath, exist_ok=exist_ok)
        return {"status": "success", "message": f"Directory created at {dirpath}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def remove_directory(dirpath: str):
    try:
        shutil.rmtree(dirpath)
        return {"status": "success", "message": f"Directory removed at {dirpath}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def file_exists(path: str):
    try:
        if os.path.exists(path):
            return {
                "status": "success",
                "exists": True,
                "is_file": os.path.isfile(path),
                "is_dir": os.path.isdir(path),
            }
        return {"status": "success", "exists": False}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def path_info(path: str):
    try:
        st = os.stat(path)
        info = {
            "path": path,
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
            "size": st.st_size,
            "mtime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
            "ctime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_ctime)),
        }
        return {"status": "success", "info": info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_in_files(dirpath: str, pattern: str, fileglob: str = "*", recursive: bool = True, casesensitive: bool = False, maxresults: int = 200):
    try:
        flags = 0 if casesensitive else re.IGNORECASE
        rx = re.compile(pattern, flags)
        matches = []
        if recursive:
            walker = os.walk(dirpath)
        else:
            walker = [(dirpath, [], os.listdir(dirpath))]
        import fnmatch
        for root, _dirs, files in walker:
            for name in files:
                if not fnmatch.fnmatch(name, fileglob):
                    continue
                p = os.path.join(root, name)
                try:
                    with open(p, 'r', encoding='utf-8', errors='replace') as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                matches.append({"path": p, "line": i, "text": line.rstrip('\n')[:1000]})
                                if len(matches) >= maxresults:
                                    return {"status": "success", "matches": matches, "truncated": True}
                except Exception:
                    continue
        return {"status": "success", "matches": matches, "truncated": False}
    except re.error as e:
        return {"status": "error", "message": f"Invalid regex: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def file(
    operation: str,
    path: str | None = None,
    content: str | None = None,
    srcpath: str | None = None,
    dstpath: str | None = None,
    oldpath: str | None = None,
    newpath: str | None = None,
    dirpath: str | None = None,
    exist_ok: bool | None = True,
    recursive: bool | None = False,
    pattern: str | None = None,
    fileglob: str | None = "*",
    casesensitive: bool | None = False,
    maxresults: int | None = 200,
    patchtype: str | None = None,
    kwargs: dict | None = None,
    url: str | None = None,
    destinationpath: str | None = None,
    data: dict | None = None,
    indent: int | None = None,
    encoding: str | None = None,
    errors: str | None = None,
    numlines: int | None = None,
    algorithm: str | None = None,
    bytesb64: str | None = None,
    paths: List[str] | None = None,
    dirs_exist_ok: bool | None = None,
    regex: bool | None = None,
    replacement: str | None = None,
    count: int | None = None,
    dryrun: bool | None = None,
    startline: int | None = None,
    endline: int | None = None,
):
    """Unified cross‑platform filesystem tool.

    Required parameters depend on `operation`:
    - create: path, [content]
    - read: path
    - write: path, content
    - append: path, content
    - move: srcpath, dstpath
    - copy: srcpath, dstpath
    - delete: path
    - rename: oldpath, newpath
    - open: path
    - openfolder: dirpath
    - list: dirpath, [recursive]
    - mkdir: dirpath, [exist_ok]
    - rmdir: dirpath
    - exists: path
    - info: path
    - search: dirpath, pattern, [fileglob, recursive, casesensitive, maxresults]
    - patch: path, patchtype, [kwargs]
    - readjson: path
    - writejson: path, data, [indent]
    - download: url, path|destinationpath
    - head: path, [numlines, encoding]
    - tail: path, [numlines, encoding]
    - readbytes: path (returns base64)
    - writebytes: path, bytesb64 (base64)
    - checksum: path, [algorithm]
    - zip: path (archive), paths (list), [dirs_exist_ok]
    - unzip: path (archive), dstpath
    - touch: path
    - symlink: srcpath (target), dstpath (link)
    - hardlink: srcpath (target), dstpath (link)
    - copytree: srcpath, dstpath, [dirs_exist_ok]
    - movetree: srcpath, dstpath, [dirs_exist_ok]
    - replace: path, pattern, replacement, [regex, count, dryrun]
    - readlines: path, startline, endline, [encoding]
    """
    try:
        op = (operation or "").strip().lower()
        if op == "create":
            if not path:
                return {"status": "error", "message": "'path' is required for create"}
            return create_file(path, content or "")
        if op == "read":
            if not path:
                return {"status": "error", "message": "'path' is required for read"}
            enc = encoding or 'utf-8'
            try:
                with open(path, 'r', encoding=enc, errors=errors or 'strict') as f:
                    content_val = f.read()
                return {"status": "success", "content": content_val}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if op == "write":
            if not path:
                return {"status": "error", "message": "'path' is required for write"}
            if content is None:
                return {"status": "error", "message": "'content' is required for write"}
            try:
                parent = os.path.dirname(path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with open(path, 'w', encoding=encoding or 'utf-8', errors=errors or 'strict') as f:
                    f.write(content)
                return {"status": "success", "message": f"Content written to {path}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if op == "append":
            if not path:
                return {"status": "error", "message": "'path' is required for append"}
            if content is None:
                return {"status": "error", "message": "'content' is required for append"}
            try:
                with open(path, 'a', encoding=encoding or 'utf-8', errors=errors or 'strict') as f:
                    f.write(content)
                return {"status": "success", "message": f"Content appended to {path}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if op == "move":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' and 'dstpath' are required for move"}
            return move_file(srcpath, dstpath)
        if op == "copy":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' and 'dstpath' are required for copy"}
            return copy_file(srcpath, dstpath)
        if op == "delete":
            if not path:
                return {"status": "error", "message": "'path' is required for delete"}
            return delete_file(path)
        if op == "rename":
            if not oldpath or not newpath:
                return {"status": "error", "message": "'oldpath' and 'newpath' are required for rename"}
            return rename_file(oldpath, newpath)
        if op == "open":
            if not path:
                return {"status": "error", "message": "'path' is required for open"}
            return open_file(path)
        if op == "openfolder":
            if not dirpath:
                return {"status": "error", "message": "'dirpath' is required for openfolder"}
            return open_folder(dirpath)
        if op == "list":
            if not dirpath:
                return {"status": "error", "message": "'dirpath' is required for list"}
            return list_directory(dirpath, bool(recursive))
        if op == "mkdir":
            if not dirpath:
                return {"status": "error", "message": "'dirpath' is required for mkdir"}
            return make_directory(dirpath, True if exist_ok is None else bool(exist_ok))
        if op == "rmdir":
            if not dirpath:
                return {"status": "error", "message": "'dirpath' is required for rmdir"}
            return remove_directory(dirpath)
        if op == "exists":
            if not path:
                return {"status": "error", "message": "'path' is required for exists"}
            return file_exists(path)
        if op == "info":
            if not path:
                return {"status": "error", "message": "'path' is required for info"}
            return path_info(path)
        if op == "search":
            if not dirpath or not pattern:
                return {"status": "error", "message": "'dirpath' and 'pattern' are required for search"}
            return search_in_files(
                dirpath,
                pattern,
                fileglob or "*",
                bool(recursive),
                bool(casesensitive),
                int(maxresults or 200),
            )
        if op == "patch":
            if not path or not patchtype:
                return {"status": "error", "message": "'path' and 'patchtype' are required for patch"}
            return patch_file(path, patchtype, **(kwargs or {}))

        if op == "readjson":
            if not path:
                return {"status": "error", "message": "'path' is required for readjson"}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data_obj = json.load(f)
                return {"status": "success", "data": data_obj}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "writejson":
            if not path:
                return {"status": "error", "message": "'path' is required for writejson"}
            if data is None:
                return {"status": "error", "message": "'data' is required for writejson"}
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=int(indent or 2))
                return {"status": "success", "message": f"JSON written to {path}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "download":
            dest = path or destinationpath
            if not url or not dest:
                return {"status": "error", "message": "'url' and 'path' (or 'destinationpath') are required for download"}
            try:
                import urllib.request
                parent = os.path.dirname(dest)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with urllib.request.urlopen(url) as resp, open(dest, 'wb') as f:
                    shutil.copyfileobj(resp, f)
                return {"status": "success", "message": f"Downloaded {url} to {dest}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "head":
            if not path:
                return {"status": "error", "message": "'path' is required for head"}
            n = int(numlines or 10)
            try:
                with open(path, 'r', encoding=encoding or 'utf-8', errors=errors or 'replace') as f:
                    lines = [next(f) for _ in range(n)]
                return {"status": "success", "lines": lines}
            except StopIteration:
                # fewer lines than requested
                return {"status": "success", "lines": lines}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "tail":
            if not path:
                return {"status": "error", "message": "'path' is required for tail"}
            n = int(numlines or 10)
            try:
                dq = deque(maxlen=n)
                with open(path, 'r', encoding=encoding or 'utf-8', errors=errors or 'replace') as f:
                    for line in f:
                        dq.append(line)
                return {"status": "success", "lines": list(dq)}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "readbytes":
            if not path:
                return {"status": "error", "message": "'path' is required for readbytes"}
            try:
                with open(path, 'rb') as f:
                    b = f.read()
                return {"status": "success", "bytesb64": base64.b64encode(b).decode('ascii')}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "writebytes":
            if not path or bytesb64 is None:
                return {"status": "error", "message": "'path' and 'bytesb64' are required for writebytes"}
            try:
                parent = os.path.dirname(path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(bytesb64))
                return {"status": "success", "message": f"Wrote binary data to {path}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "checksum":
            if not path:
                return {"status": "error", "message": "'path' is required for checksum"}
            algo = (algorithm or 'sha256').lower()
            try:
                h = hashlib.new(algo)
            except Exception:
                return {"status": "error", "message": f"Unsupported algorithm: {algorithm}"}
            try:
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(65536), b''):
                        h.update(chunk)
                return {"status": "success", "algorithm": algo, "hexdigest": h.hexdigest()}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "zip":
            if not path or not paths:
                return {"status": "error", "message": "'path' (archive) and 'paths' (list) are required for zip"}
            try:
                parent = os.path.dirname(path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in paths:
                        if os.path.isdir(p):
                            for root, _dirs, files in os.walk(p):
                                for name in files:
                                    fp = os.path.join(root, name)
                                    arcname = os.path.relpath(fp, start=os.path.dirname(p))
                                    zf.write(fp, arcname)
                        else:
                            zf.write(p, os.path.basename(p))
                return {"status": "success", "message": f"Created archive {path}", "archive": path}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "unzip":
            if not path or not dstpath:
                return {"status": "error", "message": "'path' (archive) and 'dstpath' are required for unzip"}
            try:
                os.makedirs(dstpath, exist_ok=True)
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(dstpath)
                return {"status": "success", "message": f"Extracted {path} to {dstpath}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "touch":
            if not path:
                return {"status": "error", "message": "'path' is required for touch"}
            try:
                open(path, 'a').close()
                now = None
                os.utime(path, times=now)
                return {"status": "success", "message": f"Touched {path}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "symlink":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' (target) and 'dstpath' (link) are required for symlink"}
            try:
                os.symlink(srcpath, dstpath)
                return {"status": "success", "message": f"Symlink created {dstpath} -> {srcpath}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "hardlink":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' (target) and 'dstpath' (link) are required for hardlink"}
            try:
                os.link(srcpath, dstpath)
                return {"status": "success", "message": f"Hardlink created {dstpath} -> {srcpath}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "copytree":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' and 'dstpath' are required for copytree"}
            try:
                shutil.copytree(srcpath, dstpath, dirs_exist_ok=bool(dirs_exist_ok))
                return {"status": "success", "message": f"Directory copied from {srcpath} to {dstpath}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "movetree":
            if not srcpath or not dstpath:
                return {"status": "error", "message": "'srcpath' and 'dstpath' are required for movetree"}
            try:
                if bool(dirs_exist_ok) and os.path.isdir(dstpath):
                    # move into existing dir
                    base = os.path.basename(os.path.normpath(srcpath))
                    dst = os.path.join(dstpath, base)
                    shutil.move(srcpath, dst)
                else:
                    shutil.move(srcpath, dstpath)
                return {"status": "success", "message": f"Directory moved from {srcpath} to {dstpath}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "replace":
            if not path or pattern is None or replacement is None:
                return {"status": "error", "message": "'path', 'pattern', and 'replacement' are required for replace"}
            try:
                with open(path, 'r', encoding=encoding or 'utf-8', errors=errors or 'replace') as f:
                    text = f.read()
                if bool(regex):
                    new_text, n = re.subn(pattern, replacement, text, count=int(count or 0))
                else:
                    n = text.count(pattern)
                    new_text = text.replace(pattern, replacement, int(count or 0) or text.count(pattern))
                if not bool(dryrun):
                    with open(path, 'w', encoding=encoding or 'utf-8') as f:
                        f.write(new_text)
                return {"status": "success", "replacements": n, "dryrun": bool(dryrun)}
            except re.error as e:
                return {"status": "error", "message": f"Invalid regex: {e}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if op == "readlines":
            if not path:
                return {"status": "error", "message": "'path' is required for readlines"}
            try:
                s = int(startline or 1)
                e = int(endline) if endline is not None else None
                lines = []
                with open(path, 'r', encoding=encoding or 'utf-8', errors=errors or 'replace') as f:
                    for i, line in enumerate(f, 1):
                        if i < s:
                            continue
                        if e is not None and i > e:
                            break
                        lines.append(line)
                return {"status": "success", "lines": lines}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


file_tool = ToolFunctionDef(
    name="file",
    description=(
        "Unified filesystem tool. Choose an 'operation' and pass the required "
        "parameters for that operation; other parameters are optional."
    ),
    parameters={
        "operation": str,
        "path": str,
        "content": str,
        "srcpath": str,
        "dstpath": str,
        "oldpath": str,
        "newpath": str,
        "dirpath": str,
        "exist_ok": bool,
        "recursive": bool,
        "pattern": str,
        "fileglob": str,
        "casesensitive": bool,
        "maxresults": int,
        "patchtype": str,
        "kwargs": dict,
        "url": str,
        "destinationpath": str,
        "data": dict,
        "indent": int,
        "encoding": str,
        "errors": str,
        "numlines": int,
        "algorithm": str,
        "bytesb64": str,
        "paths": list,
        "dirs_exist_ok": bool,
        "regex": bool,
        "replacement": str,
        "count": int,
        "dryrun": bool,
        "startline": int,
        "endline": int,
    },
    implementation=file,
)
