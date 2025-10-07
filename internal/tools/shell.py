import os
import platform
import shutil
import subprocess
from typing import List, Optional, Union, Dict
from lmstudio import ToolFunctionDef


def shell(
    command: Union[str, List[str]],
    workdir: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    with_escalated_permissions: bool = False,
    justification: Optional[str] = None,
    runtime: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdin: Optional[str] = None,
    success_codes: Optional[List[int]] = None,
    encoding: Optional[str] = None,
    fail_on_stderr: bool = False,
    combine_output: bool = False,
):
    """Runs a command using the selected shell/runtime.

    - `command`: string or argv array.
    - `runtime`: one of ['native', 'cmd', 'powershell', 'pwsh', 'wsl', 'sh', 'bash'].
      On Windows, exposes 'cmd', 'powershell'/'pwsh', and 'wsl'.
      If omitted, uses 'native' execution (argv → direct exec; string → OS shell).
    - `workdir`, `timeout_ms`: optional.
    - `with_escalated_permissions`: acknowledged but not supported.
    """
    try:
        if with_escalated_permissions:
            return {
                "status": "error",
                "message": "Escalated permissions are not supported in this environment.",
            }

        sysname = platform.system()
        timeout = (timeout_ms / 1000.0) if timeout_ms else None
        rt = (runtime or "native").lower()

        def run_direct(argv: List[str]):
            return subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=(None if env is None else {**os.environ, **env}),
                input=stdin,
                encoding=encoding,
            )

        def run_shell_str(cmd: str):
            return subprocess.run(
                cmd,
                cwd=workdir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=(None if env is None else {**os.environ, **env}),
                input=stdin,
                encoding=encoding,
            )

        # Native default behavior
        if rt == "native":
            if isinstance(command, list):
                result = run_direct(command)
            else:
                result = run_shell_str(command)
            ok = (result.returncode == 0) or (success_codes and result.returncode in success_codes)
            if fail_on_stderr and result.stderr:
                ok = False
            out = result.stdout if not combine_output else (result.stdout or "") + (result.stderr or "")
            return {
                "status": "success" if ok else "error",
                "code": result.returncode,
                "stdout": None if combine_output else result.stdout,
                "stderr": None if combine_output else result.stderr,
                "output": out if combine_output else None,
                "runtime": rt,
            }

        # Windows-specific runtimes
        if sysname == "Windows" and rt in ("cmd", "powershell", "pwsh", "wsl"):
            if rt == "cmd":
                # Prefer string form for cmd.exe
                cmdline = command if isinstance(command, str) else " ".join(map(str, command))
                result = run_direct(["cmd.exe", "/C", cmdline])
            elif rt in ("powershell", "pwsh"):
                exe = shutil.which("pwsh") if rt == "pwsh" else (shutil.which("powershell") or "powershell")
                cmdline = command if isinstance(command, str) else " ".join(map(str, command))
                result = run_direct([exe, "-NoProfile", "-Command", cmdline])
            else:  # wsl
                wsl = shutil.which("wsl") or shutil.which("wsl.exe") or "wsl.exe"
                if isinstance(command, list):
                    result = run_direct([wsl, "-e", *command])
                else:
                    # Run via sh -lc inside WSL
                    result = run_direct([wsl, "sh", "-lc", command])
            ok = (result.returncode == 0) or (success_codes and result.returncode in success_codes)
            if fail_on_stderr and result.stderr:
                ok = False
            out = result.stdout if not combine_output else (result.stdout or "") + (result.stderr or "")
            return {
                "status": "success" if ok else "error",
                "code": result.returncode,
                "stdout": None if combine_output else result.stdout,
                "stderr": None if combine_output else result.stderr,
                "output": out if combine_output else None,
                "runtime": rt,
            }

        # POSIX shells
        if rt in ("sh", "bash"):
            exe = shutil.which(rt) or ("/bin/" + rt)
            if isinstance(command, list):
                result = run_direct([exe, "-c", " ".join(map(str, command))])
            else:
                # login-like shell command execution
                result = run_direct([exe, "-lc", command]) if rt == "bash" else run_direct([exe, "-c", command])
            ok = (result.returncode == 0) or (success_codes and result.returncode in success_codes)
            if fail_on_stderr and result.stderr:
                ok = False
            out = result.stdout if not combine_output else (result.stdout or "") + (result.stderr or "")
            return {
                "status": "success" if ok else "error",
                "code": result.returncode,
                "stdout": None if combine_output else result.stdout,
                "stderr": None if combine_output else result.stderr,
                "output": out if combine_output else None,
                "runtime": rt,
            }

        # Fallback: treat as native
        if isinstance(command, list):
            result = run_direct(command)
        else:
            result = run_shell_str(command)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "runtime": rt,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "error",
            "message": f"Command timed out after {timeout_ms} ms" if timeout_ms else "Command timed out",
            "partial_stdout": e.stdout,
            "partial_stderr": e.stderr,
            "runtime": runtime or "native",
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "runtime": runtime or "native"}


shell_tool = ToolFunctionDef(
    name="shell",
    description="Runs a shell command and returns its output. Supports runtimes like native/cmd/powershell/pwsh/wsl/sh/bash.",
    parameters={
        "command": list,
        "workdir": str,
        "timeout_ms": int,
        "with_escalated_permissions": bool,
        "justification": str,
        "runtime": str,
        "env": dict,
        "stdin": str,
        "success_codes": list,
        "encoding": str,
        "fail_on_stderr": bool,
        "combine_output": bool,
    },
    implementation=shell,
)
