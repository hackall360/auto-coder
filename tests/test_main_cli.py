from __future__ import annotations

from types import SimpleNamespace

import builtins
import sys
import types
from typing import Any

import pytest


class _StubModel:
    def respond(self, *_, **__):
        return {"choices": [{"message": {"content": "stub"}}]}

    def respond_stream(self, *_, **__):
        yield {"choices": [{"delta": {"content": "stub"}}]}


class _StubChat:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    @classmethod
    def from_history(cls, history: Any) -> "_StubChat":
        instance = cls()
        if isinstance(history, dict):
            instance.messages = list(history.get("messages", []))
        elif isinstance(history, str):
            instance.messages = [{"role": "user", "content": history}]
        return instance

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})


class _StubToolFunctionDef:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        implementation: Any | None = None,
        parameters: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters or {})
        if implementation is None:
            self.implementation = lambda *args, **kwargs: None
        else:
            self.implementation = implementation


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_StubToolFunctionDef,
    ),
)

_psutil_stub = types.ModuleType("psutil")


class _StubProcess:
    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid or 0


_psutil_stub.Process = _StubProcess
sys.modules.setdefault("psutil", _psutil_stub)

import main


class StubManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, message: str) -> SimpleNamespace:
        self.calls.append(message)
        return SimpleNamespace(
            status_updates=[],
            response_text="ok",
            plan=[],
            budgets={},
            rounds=[],
            structured_response=None,
        )


def test_main_constructs_core_with_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["quit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    monkeypatch.setattr(builtins, "print", lambda *_, **__: None)

    stub_manager = StubManager()
    instances: list[StubCore] = []

    class StubCore:
        def __init__(self, *, config_path=None, overrides=None, env=None):  # noqa: ANN001
            self.config_path = config_path
            self.overrides = overrides
            self.env = env
            self.entered = False
            self.exited = False
            self.shutdown_called = False
            self.status_callback = None
            instances.append(self)

        def __enter__(self) -> "StubCore":
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            self.exited = True
            self.shutdown()

        def build_manager(self, *, status_callback=None):  # noqa: ANN001
            self.status_callback = status_callback
            return stub_manager

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(main, "AutoCoderCore", StubCore)

    exit_code = main.main(
        [
            "--config",
            "/tmp/config.json",
            "--mcp-config",
            "/tmp/mcp.json",
            "--default-model",
            "gpt-mini",
            "--reasoning-model",
            "gpt-pro",
            "--research-model",
            "gpt-research",
            "--allow-browsing",
            "--enable-agent",
            "db_migration",
            "--disable-agent",
            "research",
            "--repo-include-ext",
            "py",
            "--repo-include-ext",
            "md,txt",
            "--repo-exclude-dir",
            "node_modules",
            "--repo-no-auto-refresh",
            "--repo-refresh-interval",
            "120",
            "--memory-config",
            "/tmp/memory.json",
            "--no-shared-memory",
            "--mcp-auto-start",
        ]
    )

    assert exit_code == 0
    assert instances, "core should have been instantiated"
    instance = instances[0]
    assert instance.config_path == "/tmp/config.json"
    assert instance.entered and instance.exited
    assert instance.shutdown_called
    assert callable(instance.status_callback)
    assert stub_manager.calls == []

    overrides = instance.overrides or {}
    assert overrides.get("models", {}) == {
        "default_model": "gpt-mini",
        "reasoning_model": "gpt-pro",
        "research_model": "gpt-research",
        "allow_external_browsing": True,
    }
    assert overrides.get("agents", {}) == {
        "db_migration": True,
        "research": False,
    }
    assert overrides.get("repo_context", {}) == {
        "include_exts": ["py", "md", "txt"],
        "exclude_dirs": ["node_modules"],
        "auto_refresh": False,
        "refresh_interval": 120.0,
    }
    assert overrides.get("memory", {}) == {
        "config_path": "/tmp/memory.json",
        "share_globally": False,
    }
    assert overrides.get("mcp", {}) == {
        "config_path": "/tmp/mcp.json",
        "auto_start": True,
    }


def test_interactive_loop_uses_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Any] = []

    class DummyRuntime:
        def __enter__(self) -> "DummyRuntime":
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            events.append("exit")

    class DummyManager:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, message: str) -> SimpleNamespace:
            self.calls.append(message)
            events.append(("run", message))
            return SimpleNamespace(
                status_updates=[],
                response_text="ack",
                plan=[],
                budgets={},
                rounds=[],
                structured_response=None,
            )

    inputs = iter(["hello", "quit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    printed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: printed.append((args, kwargs)))

    manager_holder: list[DummyManager] = []

    def factory() -> DummyManager:
        events.append("factory")
        manager = DummyManager()
        manager_holder.append(manager)
        return manager

    runtime = DummyRuntime()

    exit_code = main._interactive_loop(factory, runtime=runtime)

    assert exit_code == 0
    assert events == ["enter", "factory", ("run", "hello"), "exit"]
    assert manager_holder and manager_holder[0].calls == ["hello"]
    # Ensure interactive banner and response were printed
    assert any("Auto-Coder manager ready" in args[0] for args, _ in printed)
    assert any(args[0].startswith("Manager:") for args, _ in printed)
