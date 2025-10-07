import os
import subprocess
import psutil
from typing import List, Union, Optional, Dict
from lmstudio import ToolFunctionDef


def process(
    operation: str,
    command: Union[str, List[str], None] = None,
    pid: Optional[int] = None,
    workdir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    shell: Optional[bool] = None,
    detached: Optional[bool] = None,
    timeout_ms: Optional[int] = None,
    name_substr: Optional[str] = None,
    cmd_substr: Optional[str] = None,
    recursive: Optional[bool] = None,
):
    """Unified process management tool.

    Operations:
    - run|execute: command (str or argv list), [workdir, env, shell, detached]. Returns pid.
    - kill: pid (int). Force-kills process.
    - terminate: pid (int). Graceful terminate.
    - wait: pid (int), [timeout_ms]. Waits for exit and returns exit code.
    - poll: pid (int). Returns running status and info.
    - info: pid (int). Returns details (name, status, cmdline, create_time).
    - children: pid (int), [recursive]. Returns child PIDs.
    - list: [name_substr, cmd_substr]. Lists processes with optional filters.
    """
    try:
        op = (operation or "").strip().lower()

        if op in ("run", "execute"):
            if command is None:
                return {"status": "error", "message": "'command' is required for run/execute"}
            use_shell = bool(shell) if shell is not None else isinstance(command, str)
            creationflags = 0
            if bool(detached) and os.name == 'nt':
                # CREATE_NEW_CONSOLE | DETACHED_PROCESS
                creationflags = 0x00000010 | 0x00000008
            p = subprocess.Popen(
                command,
                shell=use_shell,
                cwd=workdir,
                env=(None if env is None else {**os.environ, **env}),
                creationflags=creationflags,
            )
            return {"status": "success", "pid": p.pid, "message": f"Program executed with PID {p.pid}"}

        if op == "kill":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for kill"}
            try:
                proc = psutil.Process(pid)
                proc.kill()
                return {"status": "success", "message": f"Process with PID {pid} terminated"}
            except psutil.NoSuchProcess:
                return {"status": "error", "message": f"No such process with PID {pid}"}

        if op == "terminate":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for terminate"}
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                return {"status": "success", "message": f"Terminate signal sent to PID {pid}"}
            except psutil.NoSuchProcess:
                return {"status": "error", "message": f"No such process with PID {pid}"}

        if op == "wait":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for wait"}
            try:
                proc = psutil.Process(pid)
                timeout = (timeout_ms / 1000.0) if timeout_ms else None
                code = proc.wait(timeout=timeout)
                return {"status": "success", "exitcode": code}
            except psutil.TimeoutExpired:
                return {"status": "error", "message": "Timeout waiting for process"}
            except psutil.NoSuchProcess:
                return {"status": "error", "message": f"No such process with PID {pid}"}

        if op == "poll":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for poll"}
            try:
                proc = psutil.Process(pid)
                info = proc.as_dict(attrs=['pid', 'name', 'status', 'create_time'])
                return {"status": "success", "running": proc.is_running(), "info": info}
            except psutil.NoSuchProcess:
                return {"status": "success", "running": False}

        if op == "info":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for info"}
            try:
                proc = psutil.Process(pid)
                info = proc.as_dict(attrs=['pid', 'ppid', 'name', 'exe', 'cmdline', 'status', 'username', 'create_time', 'cpu_percent', 'memory_info'])
                return {"status": "success", "info": info}
            except psutil.NoSuchProcess:
                return {"status": "error", "message": f"No such process with PID {pid}"}

        if op == "children":
            if pid is None:
                return {"status": "error", "message": "'pid' is required for children"}
            try:
                proc = psutil.Process(pid)
                kids = proc.children(recursive=bool(recursive))
                return {"status": "success", "children": [p.pid for p in kids]}
            except psutil.NoSuchProcess:
                return {"status": "error", "message": f"No such process with PID {pid}"}

        if op == "list":
            procs = []
            for pr in psutil.process_iter(['pid', 'name', 'cmdline']):
                n = (pr.info.get('name') or '')
                cl = ' '.join(pr.info.get('cmdline') or [])
                if name_substr and (name_substr.lower() not in n.lower()):
                    continue
                if cmd_substr and (cmd_substr.lower() not in cl.lower()):
                    continue
                procs.append({"pid": pr.info['pid'], "name": n, "cmdline": cl})
            return {"status": "success", "processes": procs}

        return {"status": "error", "message": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


process_tool = ToolFunctionDef(
    name="process",
    description=(
        "Unified process tool. run/execute, kill/terminate, wait, poll, info, children, list."
    ),
    parameters={
        "operation": str,
        "command": list,
        "pid": int,
        "workdir": str,
        "env": dict,
        "shell": bool,
        "detached": bool,
        "timeout_ms": int,
        "name_substr": str,
        "cmd_substr": str,
        "recursive": bool,
    },
    implementation=process,
)
