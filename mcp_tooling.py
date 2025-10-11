"""Helpers for configuring and launching MCP servers.

This module defines a small collection of dataclasses that capture
configuration for different flavours of MCP servers.  A registry class
is provided to validate and normalise raw configuration payloads loaded
from ``config.json``.  For command based servers a small lifecycle helper
is included to manage subprocess creation, readiness detection and
shutdown behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import atexit
import contextlib
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence
import urllib.request


class MCPConfigurationError(RuntimeError):
    """Raised when a configuration entry is invalid."""


@dataclass(slots=True)
class MCPServerConfig:
    """Base dataclass for MCP server configuration entries."""

    label: str
    allowed_tools: Optional[tuple[str, ...]] = None
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        label = (self.label or "").strip()
        if not label:
            raise MCPConfigurationError("MCP server entries must provide a non-empty label")
        object.__setattr__(self, "label", label)
        if self.allowed_tools is not None:
            coerced = tuple(tool.strip() for tool in self.allowed_tools if tool and tool.strip())
            object.__setattr__(self, "allowed_tools", coerced or None)

    @property
    def kind(self) -> str:
        return "base"

    def descriptor(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "type": self.kind,
        }
        if self.allowed_tools is not None:
            payload["allowed_tools"] = list(self.allowed_tools)
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class LocalMCPServerConfig(MCPServerConfig):
    """Configuration for locally running MCP servers."""

    url: str | None = None
    headers: Optional[Mapping[str, str]] = None

    @property
    def kind(self) -> str:
        return "local"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.url:
            raise MCPConfigurationError(
                f"Local MCP server '{self.label}' must provide a 'url' pointing to the local endpoint",
            )
        object.__setattr__(self, "url", self.url.strip())

    def descriptor(self) -> dict[str, Any]:
        payload = super().descriptor()
        payload.update({
            "url": self.url,
            "headers": dict(self.headers) if self.headers else {},
        })
        return payload


@dataclass(slots=True)
class RemoteMCPServerConfig(MCPServerConfig):
    """Configuration for remote MCP servers accessible over the network."""

    url: str
    headers: Optional[Mapping[str, str]] = None
    verify_tls: bool = True

    @property
    def kind(self) -> str:
        return "remote"

    def __post_init__(self) -> None:
        super().__post_init__()
        url = (self.url or "").strip()
        if not url:
            raise MCPConfigurationError(
                f"Remote MCP server '{self.label}' must provide a non-empty 'url' value",
            )
        object.__setattr__(self, "url", url)

    def descriptor(self) -> dict[str, Any]:
        payload = super().descriptor()
        payload.update({
            "url": self.url,
            "headers": dict(self.headers) if self.headers else {},
            "verify_tls": bool(self.verify_tls),
        })
        return payload


@dataclass(slots=True)
class CommandMCPServerConfig(MCPServerConfig):
    """Configuration for MCP servers launched via a local command."""

    command: tuple[str, ...]
    env: Optional[Mapping[str, str]] = None
    cwd: Optional[str | Path] = None
    ready_pattern: Optional[str] = None
    ready_timeout: float = 30.0
    ready_probe_url: Optional[str] = None
    shutdown_command: Optional[tuple[str, ...]] = None
    shutdown_signal: Optional[int] = None
    capture_output: bool = True

    @property
    def kind(self) -> str:
        return "command"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.command:
            raise MCPConfigurationError(
                f"Command based MCP server '{self.label}' must declare a launch command",
            )
        normalised_command = []
        for part in self.command:
            if isinstance(part, str):
                stripped = part.strip()
                if stripped:
                    normalised_command.append(stripped)
        if not normalised_command:
            raise MCPConfigurationError(
                f"Command based MCP server '{self.label}' provided an empty command sequence",
            )
        object.__setattr__(self, "command", tuple(normalised_command))
        if self.shutdown_command:
            shutdown = []
            for part in self.shutdown_command:
                if isinstance(part, str) and part.strip():
                    shutdown.append(part.strip())
            object.__setattr__(self, "shutdown_command", tuple(shutdown) or None)
        if self.ready_timeout <= 0:
            raise MCPConfigurationError(
                f"Command based MCP server '{self.label}' requires a positive ready_timeout value",
            )
        if self.ready_probe_url:
            object.__setattr__(self, "ready_probe_url", self.ready_probe_url.strip())

    def descriptor(self) -> dict[str, Any]:
        payload = super().descriptor()
        payload.update(
            {
                "command": list(self.command),
                "env": dict(self.env) if self.env else {},
                "cwd": str(self.cwd) if self.cwd else None,
                "ready_pattern": self.ready_pattern,
                "ready_timeout": self.ready_timeout,
                "ready_probe_url": self.ready_probe_url,
                "shutdown_command": list(self.shutdown_command) if self.shutdown_command else None,
                "shutdown_signal": self.shutdown_signal,
            }
        )
        return payload


def load_mcp_config(config_path: Optional[Path | str] = None) -> Mapping[str, Any]:
    """Load the ``mcp_servers`` configuration section from ``config.json``.

    The loader defers importing :mod:`memory` until runtime to avoid
    circular imports.  When ``config_path`` is ``None`` the environment
    variable ``MCP_CONFIG_PATH`` is honoured before falling back to the
    defaults used by :func:`memory.load_config_json`.
    """

    resolved_path: Optional[Path] = None
    if config_path is not None:
        resolved_path = Path(config_path)
    else:
        env_path = os.getenv("MCP_CONFIG_PATH")
        if env_path:
            resolved_path = Path(env_path)
    from importlib import import_module

    memory_module = import_module("memory")
    config_data = memory_module.load_config_json(resolved_path)
    servers = config_data.get("mcp_servers", {})
    if not isinstance(servers, Mapping):
        raise MCPConfigurationError("The 'mcp_servers' section must be a mapping of label to config")
    return servers


class CommandServerLifecycle:
    """Lifecycle helper for command-based MCP servers."""

    def __init__(self, config: CommandMCPServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._stdout_log: list[str] = []
        self._stderr_log: list[str] = []
        self._log_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pattern = re.compile(config.ready_pattern) if config.ready_pattern else None

    def start(self) -> subprocess.Popen[str]:
        if self.process is not None:
            return self.process
        env = os.environ.copy()
        if self.config.env:
            env.update({str(k): str(v) for k, v in self.config.env.items()})
        stdout_setting = subprocess.PIPE if self.config.capture_output else None
        stderr_setting = subprocess.PIPE if self.config.capture_output else None
        process = subprocess.Popen(
            list(self.config.command),
            cwd=str(self.config.cwd) if self.config.cwd else None,
            env=env,
            stdout=stdout_setting,
            stderr=stderr_setting,
            text=True,
            bufsize=1,
        )
        self.process = process
        if stdout_setting is not None and process.stdout is not None:
            self._threads.append(
                threading.Thread(
                    target=self._pump_stream,
                    args=(process.stdout, self._stdout_log),
                    daemon=True,
                ),
            )
        if stderr_setting is not None and process.stderr is not None:
            self._threads.append(
                threading.Thread(
                    target=self._pump_stream,
                    args=(process.stderr, self._stderr_log),
                    daemon=True,
                ),
            )
        for thread in self._threads:
            thread.start()
        atexit.register(self.shutdown)
        self._wait_until_ready()
        return process

    def _pump_stream(
        self,
        stream: Any,
        log_target: list[str],
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                text_line = line.rstrip("\n")
                with self._log_lock:
                    log_target.append(text_line)
                if self._pattern and self._pattern.search(text_line):
                    self._ready_event.set()
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def _probe_ready(self) -> bool:
        if not self.config.ready_probe_url:
            return False
        try:
            with urllib.request.urlopen(self.config.ready_probe_url, timeout=5) as response:
                return 200 <= getattr(response, "status", 200) < 400
        except Exception:
            return False

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.config.ready_timeout
        if self._pattern is None and not self.config.ready_probe_url:
            # Best effort: ensure the process stays alive for a short period.
            time.sleep(min(0.5, self.config.ready_timeout))
            if self.process.poll() is not None:
                raise MCPConfigurationError(
                    f"MCP server '{self.config.label}' exited early with code {self.process.returncode}",
                )
            return
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise MCPConfigurationError(
                    f"MCP server '{self.config.label}' exited early with code {self.process.returncode}",
                )
            remaining = max(0.0, deadline - time.monotonic())
            if self._pattern and self._ready_event.wait(timeout=min(0.2, remaining)):
                return
            if self.config.ready_probe_url and self._probe_ready():
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for MCP server '{self.config.label}' to become ready",
        )

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def stdout_log(self) -> list[str]:
        with self._log_lock:
            return list(self._stdout_log)

    @property
    def stderr_log(self) -> list[str]:
        with self._log_lock:
            return list(self._stderr_log)

    def shutdown(self) -> None:
        if self.process is None:
            return
        process = self.process
        if process.poll() is not None:
            return
        if self.config.shutdown_command:
            try:
                subprocess.run(
                    list(self.config.shutdown_command),
                    cwd=str(self.config.cwd) if self.config.cwd else None,
                    env=os.environ.copy(),
                    timeout=10,
                )
            except Exception:
                pass
        if process.poll() is not None:
            return
        if self.config.shutdown_signal is not None:
            with contextlib.suppress(Exception):
                process.send_signal(self.config.shutdown_signal)
        else:
            with contextlib.suppress(Exception):
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
        self.process = None

    def __enter__(self) -> "CommandServerLifecycle":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


@dataclass(slots=True)
class MCPServerSpec:
    """Normalized representation produced by :class:`MCPServerRegistry`."""

    config: MCPServerConfig
    descriptor: dict[str, Any] = field(default_factory=dict)
    lifecycle: Optional[CommandServerLifecycle] = None


class MCPServerRegistry:
    """Validate and normalise MCP server configuration entries."""

    def __init__(self, entries: Optional[Mapping[str, Any]] = None) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        if entries:
            self.update(entries)

    @staticmethod
    def from_loaded_config(config_path: Optional[Path | str] = None) -> "MCPServerRegistry":
        servers = load_mcp_config(config_path)
        return MCPServerRegistry(servers)

    def update(self, entries: Mapping[str, Any]) -> None:
        for label, payload in entries.items():
            config = self._normalise_entry(label, payload)
            self._configs[config.label] = config

    def _normalise_entry(self, label: str, payload: Any) -> MCPServerConfig:
        if not isinstance(payload, Mapping):
            raise MCPConfigurationError(
                f"Configuration for MCP server '{label}' must be a mapping of options",
            )
        data = dict(payload)
        data.setdefault("label", label)
        allowed_raw = data.get("allowed_tools")
        allowed: Optional[tuple[str, ...]] = None
        if isinstance(allowed_raw, str):
            allowed = (allowed_raw,)
        elif isinstance(allowed_raw, Sequence) and not isinstance(allowed_raw, (str, bytes, bytearray)):
            allowed = tuple(str(item) for item in allowed_raw)
        elif isinstance(allowed_raw, MutableMapping):
            allowed = tuple(str(item) for item in allowed_raw.values())
        kind = (data.get("type") or data.get("kind") or data.get("mode") or "remote").lower()
        description = data.get("description")
        metadata = data.get("metadata")

        if kind == "command" or "command" in data:
            command_raw = data.get("command")
            if isinstance(command_raw, str):
                command = tuple(part for part in command_raw.split() if part)
            elif isinstance(command_raw, Sequence):
                command = tuple(str(part) for part in command_raw)
            else:
                command = tuple()
            shutdown_raw = data.get("shutdown_command")
            if isinstance(shutdown_raw, str):
                shutdown = tuple(part for part in shutdown_raw.split() if part)
            elif isinstance(shutdown_raw, Sequence):
                shutdown = tuple(str(part) for part in shutdown_raw)
            else:
                shutdown = None
            return CommandMCPServerConfig(
                label=data["label"],
                allowed_tools=allowed,
                description=description,
                metadata=metadata,
                command=command,
                env=data.get("env"),
                cwd=data.get("cwd"),
                ready_pattern=data.get("ready_pattern"),
                ready_timeout=float(data.get("ready_timeout", 30.0)),
                ready_probe_url=data.get("ready_probe_url"),
                shutdown_command=shutdown,
                shutdown_signal=data.get("shutdown_signal"),
                capture_output=bool(data.get("capture_output", True)),
            )
        if kind == "local":
            url = data.get("url")
            headers = data.get("headers")
            return LocalMCPServerConfig(
                label=data["label"],
                allowed_tools=allowed,
                description=description,
                metadata=metadata,
                url=str(url) if url else None,
                headers=headers,
            )
        if kind == "remote":
            url = data.get("url")
            headers = data.get("headers")
            verify_tls = data.get("verify_tls", True)
            return RemoteMCPServerConfig(
                label=data["label"],
                allowed_tools=allowed,
                description=description,
                metadata=metadata,
                url=str(url) if url else "",
                headers=headers,
                verify_tls=bool(verify_tls),
            )
        raise MCPConfigurationError(f"Unsupported MCP server type '{kind}' for entry '{label}'")

    def get(self, label: str) -> MCPServerConfig:
        return self._configs[label]

    def all(self) -> list[MCPServerConfig]:
        return list(self._configs.values())

    def build_specs(self, *, auto_start: bool = False) -> list[MCPServerSpec]:
        specs: list[MCPServerSpec] = []
        for config in self._configs.values():
            lifecycle: Optional[CommandServerLifecycle] = None
            descriptor = config.descriptor()
            if isinstance(config, CommandMCPServerConfig):
                lifecycle = CommandServerLifecycle(config)
                descriptor["pid"] = None
                if auto_start:
                    lifecycle.start()
                    descriptor["pid"] = lifecycle.pid
            spec = MCPServerSpec(
                config=config,
                descriptor=descriptor,
                lifecycle=lifecycle,
            )
            specs.append(spec)
        return specs

    def start_all(self) -> list[CommandServerLifecycle]:
        lifecycles: list[CommandServerLifecycle] = []
        for config in self._configs.values():
            if isinstance(config, CommandMCPServerConfig):
                lifecycle = CommandServerLifecycle(config)
                lifecycle.start()
                lifecycles.append(lifecycle)
        return lifecycles

    def labels(self) -> Iterable[str]:
        return list(self._configs.keys())
