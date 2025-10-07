import subprocess
from lmstudio import ToolFunctionDef


def git_init(repo_path: str | None = None):
    try:
        cmd = ["git", "init"]
        if repo_path and repo_path.endswith(".git"):
            cmd.append("--bare")
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": "Git repository initialized successfully"}
        return {"status": "error", "message": f"Failed to initialize Git repository: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_clone(repo_url: str, local_path: str | None = None):
    try:
        cmd = ["git", "clone", repo_url]
        if local_path:
            cmd.append(local_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Repository cloned successfully from {repo_url}"}
        return {"status": "error", "message": f"Failed to clone repository: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_add(file_path: str | None = None):
    try:
        cmd = ["git", "add", file_path or "."]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": "Files staged for commit successfully"}
        return {"status": "error", "message": f"Failed to stage files: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_commit(message: str, author: str | None = None):
    try:
        cmd = ["git", "commit", "-m", message]
        if author:
            cmd.extend(["--author", author])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Changes committed successfully with message: {message}"}
        return {"status": "error", "message": f"Failed to commit changes: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_push(remote_name: str = "origin", branch_name: str = "main"):
    try:
        cmd = ["git", "push", remote_name, branch_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Changes pushed successfully to {remote_name}/{branch_name}"}
        return {"status": "error", "message": f"Failed to push changes: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_pull(remote_name: str = "origin", branch_name: str = "main"):
    try:
        cmd = ["git", "pull", remote_name, branch_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Changes pulled successfully from {remote_name}/{branch_name}"}
        return {"status": "error", "message": f"Failed to pull changes: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_status():
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout}
        return {"status": "error", "message": f"Failed to get status: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_log(limit: int = 10):
    try:
        cmd = ["git", "log", "--oneline"]
        if limit:
            cmd.extend(["-n", str(limit)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout}
        return {"status": "error", "message": f"Failed to get commit history: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_branch():
    try:
        result = subprocess.run(["git", "branch"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout}
        return {"status": "error", "message": f"Failed to get branches: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_remote():
    try:
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout}
        return {"status": "error", "message": f"Failed to get remotes: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_checkout(branch_name: str):
    try:
        result = subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Switched to branch {branch_name} successfully"}
        return {"status": "error", "message": f"Failed to switch to branch {branch_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_checkout_new_branch(branch_name: str):
    try:
        result = subprocess.run(["git", "checkout", "-b", branch_name], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Created and switched to new branch {branch_name} successfully"}
        return {"status": "error", "message": f"Failed to create and switch to new branch {branch_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_merge(branch_name: str):
    try:
        result = subprocess.run(["git", "merge", branch_name], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Merged changes from branch {branch_name} into current branch successfully"}
        return {"status": "error", "message": f"Failed to merge changes from branch {branch_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_fetch(remote_name: str = "origin"):
    try:
        result = subprocess.run(["git", "fetch", remote_name], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Fetched all branches from {remote_name} successfully"}
        return {"status": "error", "message": f"Failed to fetch from {remote_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_rebase(branch_name: str):
    try:
        result = subprocess.run(["git", "rebase", branch_name], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Rebased current branch onto {branch_name} successfully"}
        return {"status": "error", "message": f"Failed to rebase current branch onto {branch_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_tag(tag_name: str, message: str | None = None):
    try:
        cmd = ["git", "tag", tag_name]
        if message:
            cmd.extend(["-m", message])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Tag {tag_name} created successfully"}
        return {"status": "error", "message": f"Failed to create tag {tag_name}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_config_get(key: str):
    try:
        result = subprocess.run(["git", "config", "--get", key], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "value": result.stdout.strip()}
        return {"status": "error", "message": f"Failed to get configuration value for {key}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_config_set(key: str, value: str):
    try:
        result = subprocess.run(["git", "config", key, value], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": f"Git configuration {key} set to {value} successfully"}
        return {"status": "error", "message": f"Failed to set Git configuration {key} to {value}: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git(
    operation: str,
    repo_path: str | None = None,
    url: str | None = None,
    repourl: str | None = None,
    localpath: str | None = None,
    filepath: str | None = None,
    message: str | None = None,
    author: str | None = None,
    remotename: str | None = None,
    remote_name: str | None = None,
    branchname: str | None = None,
    branch_name: str | None = None,
    limit: int | None = None,
    tagname: str | None = None,
    tag_name: str | None = None,
    key: str | None = None,
    value: str | None = None,
    # Extended/common options
    force: bool | None = None,
    tags_flag: bool | None = None,
    set_upstream: bool | None = None,
    rebase_flag: bool | None = None,
    all_flag: bool | None = None,
    porcelain: bool | None = None,
    short: bool | None = None,
    since: str | None = None,
    until: str | None = None,
    author_filter: str | None = None,
    grep_filter: str | None = None,
    format: str | None = None,
    pathspec: str | None = None,
    ref: str | None = None,
    mode: str | None = None,
    commitish: str | None = None,
    stash_ref: str | None = None,
    include_untracked: bool | None = None,
    name: str | None = None,
    url2: str | None = None,
    oldbranch: str | None = None,
    newbranch: str | None = None,
    depth: int | None = None,
    branch: str | None = None,
    recurse_submodules: bool | None = None,
    extra: dict | None = None,
):
    """Unified Git tool that wraps common operations.

    Supported operations and required params:
    - init: [repo_path]
    - clone: url|repourl, [localpath, branch, depth, recurse_submodules]
    - add: [filepath], [all_flag]
    - commit: message, [author, all_flag]
    - push: [remotename|remote_name], [branchname|branch_name], [set_upstream, force, tags_flag]
    - pull: [remotename|remote_name], [branchname|branch_name], [rebase_flag]
    - status: [porcelain, short]
    - log: [limit, since, until, author_filter, grep_filter, format, pathspec]
    - branch: (list branches)
    - branch_create: branchname|branch_name, [start_point]
    - branch_delete: branchname|branch_name, [force]
    - branch_rename: oldbranch, newbranch
    - remote: (list remotes)
    - remote_add: remotename|remote_name, url
    - remote_remove: remotename|remote_name
    - remote_set_url: remotename|remote_name, url2
    - checkout: branchname|branch_name
    - checkout_new: branchname|branch_name
    - merge: branchname|branch_name
    - fetch: [remotename|remote_name]
    - rebase: branchname|branch_name
    - tag: tagname|tag_name, [message]
    - config_get: key
    - config_set: key, value
    - diff: [ref], [pathspec]
    - reset: [mode=soft|mixed|hard], [commitish]
    - stash_push: [message], [include_untracked], [pathspec]
    - stash_list: (no params)
    - stash_pop: [stash_ref]
    - stash_apply: [stash_ref]
    - show: [commitish]
    - cherry_pick: commitish
    - revert: commitish
    """
    try:
        op = (operation or "").strip().lower()

        if op == "init":
            return git_init(repo_path)

        if op == "clone":
            repo_url = url or repourl
            if not repo_url:
                return {"status": "error", "message": "'url' (or 'repourl') is required for clone"}
            cmd = ["git", "clone"]
            if branch:
                cmd += ["-b", branch]
            if depth:
                cmd += ["--depth", str(int(depth))]
            if recurse_submodules:
                cmd += ["--recurse-submodules"]
            cmd += [repo_url]
            if localpath:
                cmd.append(localpath)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Repository cloned from {repo_url}"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "add":
            if all_flag:
                r = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
            else:
                r = subprocess.run(["git", "add", filepath or "."], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": "Files staged"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "commit":
            if not message:
                return {"status": "error", "message": "'message' is required for commit"}
            cmd = ["git", "commit", "-m", message]
            if author:
                cmd += ["--author", author]
            if all_flag:
                cmd += ["-a"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": "Committed"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "push":
            rn = remotename or remote_name or "origin"
            bn = branchname or branch_name or "main"
            cmd = ["git", "push"]
            if set_upstream:
                cmd.append("-u")
            if force:
                cmd.append("--force")
            if tags_flag:
                cmd.append("--tags")
            cmd += [rn, bn]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Pushed to {rn}/{bn}"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "pull":
            rn = remotename or remote_name or "origin"
            bn = branchname or branch_name or "main"
            cmd = ["git", "pull"]
            if rebase_flag:
                cmd.append("--rebase")
            cmd += [rn, bn]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Pulled from {rn}/{bn}"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "status":
            cmd = ["git", "status"]
            if porcelain:
                cmd.append("--porcelain")
            if short:
                cmd.append("--short")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "log":
            cmd = ["git", "log", "--oneline"]
            if limit:
                cmd += ["-n", str(limit)]
            if since:
                cmd += ["--since", since]
            if until:
                cmd += ["--until", until]
            if author_filter:
                cmd += ["--author", author_filter]
            if grep_filter:
                cmd += ["--grep", grep_filter]
            if format:
                cmd = ["git", "log", f"--pretty={format}"]
            if pathspec:
                cmd += ["--", pathspec]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "branch":
            r = subprocess.run(["git", "branch"], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "branch_create":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for branch_create"}
            cmd = ["git", "branch", bn]
            if ref:
                cmd.append(ref)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Branch {bn} created"}
            return {"status": "error", "message": r.stderr}

        if op == "branch_delete":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for branch_delete"}
            cmd = ["git", "branch", "-D" if force else "-d", bn]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Branch {bn} deleted"}
            return {"status": "error", "message": r.stderr}

        if op == "branch_rename":
            if not oldbranch or not newbranch:
                return {"status": "error", "message": "'oldbranch' and 'newbranch' are required for branch_rename"}
            r = subprocess.run(["git", "branch", "-m", oldbranch, newbranch], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Branch {oldbranch} -> {newbranch}"}
            return {"status": "error", "message": r.stderr}

        if op == "remote":
            r = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "remote_add":
            rn = remotename or remote_name
            if not rn or not url:
                return {"status": "error", "message": "'remotename' and 'url' are required for remote_add"}
            r = subprocess.run(["git", "remote", "add", rn, url], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Remote {rn} added"}
            return {"status": "error", "message": r.stderr}

        if op == "remote_remove":
            rn = remotename or remote_name
            if not rn:
                return {"status": "error", "message": "'remotename' is required for remote_remove"}
            r = subprocess.run(["git", "remote", "remove", rn], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Remote {rn} removed"}
            return {"status": "error", "message": r.stderr}

        if op == "remote_set_url":
            rn = remotename or remote_name
            if not rn or not url2:
                return {"status": "error", "message": "'remotename' and 'url2' are required for remote_set_url"}
            r = subprocess.run(["git", "remote", "set-url", rn, url2], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Remote {rn} url updated"}
            return {"status": "error", "message": r.stderr}

        if op == "checkout":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for checkout"}
            return git_checkout(bn)

        if op == "checkout_new":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for checkout_new"}
            return git_checkout_new_branch(bn)

        if op == "merge":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for merge"}
            return git_merge(bn)

        if op == "fetch":
            rn = remotename or remote_name or "origin"
            r = subprocess.run(["git", "fetch", rn], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Fetched from {rn}"}
            return {"status": "error", "message": r.stderr or r.stdout}

        if op == "rebase":
            bn = branchname or branch_name
            if not bn:
                return {"status": "error", "message": "'branchname' is required for rebase"}
            r = subprocess.run(["git", "rebase", bn], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Rebased onto {bn}"}
            return {"status": "error", "message": r.stderr}

        if op == "tag":
            tn = tagname or tag_name
            if not tn:
                return {"status": "error", "message": "'tagname' is required for tag"}
            cmd = ["git", "tag", tn]
            if message:
                cmd += ["-m", message]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Tag {tn} created"}
            return {"status": "error", "message": r.stderr}

        if op == "config_get":
            if not key:
                return {"status": "error", "message": "'key' is required for config_get"}
            r = subprocess.run(["git", "config", "--get", key], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "value": r.stdout.strip()}
            return {"status": "error", "message": r.stderr}

        if op == "config_set":
            if not key or value is None:
                return {"status": "error", "message": "'key' and 'value' are required for config_set"}
            r = subprocess.run(["git", "config", key, value], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Config {key} set"}
            return {"status": "error", "message": r.stderr}

        if op == "diff":
            cmd = ["git", "diff"]
            if ref:
                cmd.append(ref)
            if pathspec:
                cmd += ["--", pathspec]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "reset":
            m = (mode or 'mixed').lower()
            if m not in ('soft', 'mixed', 'hard'):
                return {"status": "error", "message": "'mode' must be soft|mixed|hard"}
            commit = commitish or 'HEAD'
            r = subprocess.run(["git", "reset", f"--{m}", commit], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": f"Reset {m} to {commit}"}
            return {"status": "error", "message": r.stderr}

        if op == "stash_push":
            cmd = ["git", "stash", "push"]
            if message:
                cmd += ["-m", message]
            if include_untracked:
                cmd.append("-u")
            if pathspec:
                cmd += ["--", pathspec]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": r.stdout.strip() or "Stashed"}
            return {"status": "error", "message": r.stderr}

        if op == "stash_list":
            r = subprocess.run(["git", "stash", "list"], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "stash_pop":
            cmd = ["git", "stash", "pop"]
            if stash_ref:
                cmd.append(stash_ref)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": r.stdout.strip() or "Stash popped"}
            return {"status": "error", "message": r.stderr}

        if op == "stash_apply":
            cmd = ["git", "stash", "apply"]
            if stash_ref:
                cmd.append(stash_ref)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": r.stdout.strip() or "Stash applied"}
            return {"status": "error", "message": r.stderr}

        if op == "show":
            c = commitish or 'HEAD'
            r = subprocess.run(["git", "show", c], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "output": r.stdout}
            return {"status": "error", "message": r.stderr}

        if op == "cherry_pick":
            if not commitish:
                return {"status": "error", "message": "'commitish' is required for cherry_pick"}
            r = subprocess.run(["git", "cherry-pick", commitish], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": "Cherry-picked"}
            return {"status": "error", "message": r.stderr}

        if op == "revert":
            if not commitish:
                return {"status": "error", "message": "'commitish' is required for revert"}
            r = subprocess.run(["git", "revert", commitish, "--no-edit"], capture_output=True, text=True)
            if r.returncode == 0:
                return {"status": "success", "message": "Reverted"}
            return {"status": "error", "message": r.stderr}

        return {"status": "error", "message": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


git_tool = ToolFunctionDef(
    name="git",
    description=(
        "Unified git tool. Choose an 'operation' (init, clone, add, commit, push, pull, status, "
        "log, branch, remote, checkout, checkout_new, merge, fetch, rebase, tag, config_get, config_set) "
        "and pass the required parameters."
    ),
    parameters={
        "operation": str,
        "repo_path": str,
        "url": str,
        "repourl": str,
        "localpath": str,
        "filepath": str,
        "message": str,
        "author": str,
        "remotename": str,
        "remote_name": str,
        "branchname": str,
        "branch_name": str,
        "limit": int,
        "tagname": str,
        "tag_name": str,
        "key": str,
        "value": str,
        "force": bool,
        "tags_flag": bool,
        "set_upstream": bool,
        "rebase_flag": bool,
        "all_flag": bool,
        "porcelain": bool,
        "short": bool,
        "since": str,
        "until": str,
        "author_filter": str,
        "grep_filter": str,
        "format": str,
        "pathspec": str,
        "ref": str,
        "mode": str,
        "commitish": str,
        "stash_ref": str,
        "include_untracked": bool,
        "name": str,
        "url2": str,
        "oldbranch": str,
        "newbranch": str,
        "depth": int,
        "branch": str,
        "recurse_submodules": bool,
        "extra": dict,
    },
    implementation=git,
)
