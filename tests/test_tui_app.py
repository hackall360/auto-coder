from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, Awaitable, Callable, Mapping

import pytest


# ---------------------------------------------------------------------------
# Minimal dependency shims loaded before importing the TUI module.
# ---------------------------------------------------------------------------


class _StubModel:
    def respond(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "stub"}}]}

    def respond_stream(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"choices": [{"delta": {"content": "stub"}}]}]


class _StubChat:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    @classmethod
    def from_history(cls, history: Any) -> _StubChat:
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
    def __init__(self, *, name: str, description: str, **_: Any) -> None:
        self.name = name
        self.description = description


sys.modules.setdefault(
    "lmstudio",
    SimpleNamespace(
        llm=lambda *args, **kwargs: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_StubToolFunctionDef,
    ),
)
sys.modules.setdefault("psutil", SimpleNamespace(Process=lambda pid=None: SimpleNamespace(pid=pid or 0)))

# Stub the ``agents`` package to provide the small surface area required by TUI.
agents_pkg = ModuleType("agents")
agents_pkg.__path__ = []  # type: ignore[attr-defined]


@dataclass(slots=True)
class _ManagerStatusUpdate:
    message: str
    kind: str = "info"
    task: str | None = None
    payload: Mapping[str, Any] | None = None
    timestamp: float = 0.0


@dataclass(slots=True)
class _TaskBudget:
    name: str
    consumed: float = 0.0
    limit: float | None = None

    def as_dict(self) -> dict[str, Any]:
        remaining = None if self.limit is None else max(self.limit - self.consumed, 0)
        return {
            "name": self.name,
            "consumed": self.consumed,
            "limit": self.limit,
            "remaining": remaining,
        }


@dataclass(slots=True)
class _ManagerResult:
    response_text: str
    structured_response: Any | None
    rounds: list[Any]
    plan: list[dict[str, Any]]
    budgets: dict[str, Any]
    status_updates: list[_ManagerStatusUpdate]


manager_module = ModuleType("agents.manager")
manager_module.ManagerAgent = type("ManagerAgent", (), {})
manager_module.ManagerResult = _ManagerResult
manager_module.ManagerStatusUpdate = _ManagerStatusUpdate
manager_module.TaskBudget = _TaskBudget
sys.modules.setdefault("agents", agents_pkg)
sys.modules.setdefault("agents.manager", manager_module)

# Provide a lightweight ``tooling`` module.
tooling_module = ModuleType("tooling")


def _resolve_tools(*_: Any, **__: Any) -> list[Any]:
    return []


class _ToolRegistry:
    def register(self, candidate: Any) -> None:  # pragma: no cover - no-op stub
        return None

    def all(self) -> list[Any]:  # pragma: no cover - unused
        return []

    def get(self, name: str) -> Any:  # pragma: no cover - unused
        raise KeyError(name)


tooling_module.ToolFunctionDef = _StubToolFunctionDef
tooling_module.ToolSpec = SimpleNamespace
tooling_module.ToolRegistry = _ToolRegistry
tooling_module.resolve_tools = _resolve_tools
sys.modules.setdefault("tooling", tooling_module)

# Minimal ``core`` module used only for CLI argument defaults.
core_module = ModuleType("core")
core_module.AgentToggleSettings = type("AgentToggleSettings", (), {"__annotations__": {"stub": bool}})
core_module.AutoCoderCore = type("AutoCoderCore", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("core", core_module)

from TUI import (  # noqa: E402  # pylint: disable=wrong-import-position
    AutoCoderApp,
    BudgetMeterWidget,
    ManagerStatusMessage,
    ManagerResultMessage,
    PromptInput,
    PromptSubmitted,
    PlanTrackerWidget,
    TranscriptWidget,
    _build_parser,
)


class _StubManager:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.next_result: _ManagerResult | None = None

    def run(self, prompt: str) -> _ManagerResult:
        self.calls.append(prompt)
        if self.next_result is None:
            raise AssertionError("Stub manager result not configured")
        return self.next_result


def _table_rows(table: Any) -> list[tuple[str, ...]]:
    columns = [list(column._cells) for column in table.columns]  # type: ignore[attr-defined]
    return list(zip(*columns)) if columns else []


def _log_lines(widget: TranscriptWidget) -> list[str]:
    return [strip.text for strip in widget.lines]


def _run_app(
    monkeypatch: pytest.MonkeyPatch,
    body: Callable[[AutoCoderApp, Any, _StubManager], Awaitable[None]],
) -> None:
    manager = _StubManager()

    async def _fake_initialise(self: AutoCoderApp) -> None:
        self._core = SimpleNamespace()
        self._manager = manager
        prompt = self.query_one("#prompt", PromptInput)
        prompt.is_busy = False

    monkeypatch.setattr(AutoCoderApp, "_initialise_runtime", _fake_initialise, raising=False)

    async def _fake_run_worker(
        self: AutoCoderApp,
        coro: Awaitable[Any],
        *,
        name: str | None = None,
        exit_on_error: bool = False,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Any], None] | None = None,
        on_cancel: Callable[[Any], None] | None = None,
    ) -> asyncio.Task[Any]:
        async def task_runner() -> Any:
            try:
                result = await coro
            except asyncio.CancelledError:
                if on_cancel is not None:
                    on_cancel(SimpleNamespace(result=None, error=None))
                raise
            except Exception as exc:  # pragma: no cover - defensive
                if on_error is not None:
                    on_error(SimpleNamespace(result=None, error=exc))
                if exit_on_error:
                    raise
            else:
                if on_success is not None:
                    on_success(SimpleNamespace(result=result, error=None))
                return result

        return asyncio.create_task(task_runner(), name=name or "test-worker")

    monkeypatch.setattr(AutoCoderApp, "run_worker", _fake_run_worker, raising=False)

    async def _test_handle_prompt(self: AutoCoderApp, value: str) -> None:
        prompt = value.strip()
        if not prompt:
            return
        transcript = self.query_one("#transcript", TranscriptWidget)
        prompt_widget = self.query_one("#prompt", PromptInput)
        prompt_widget.is_busy = True
        transcript.add_user_message(prompt)
        self._pending_prompt = prompt
        result = manager.run(prompt)
        self._pending_prompt = None
        prompt_widget.is_busy = False
        self.post_message(ManagerResultMessage(prompt, result))

    monkeypatch.setattr(AutoCoderApp, "handle_prompt_submitted", _test_handle_prompt, raising=False)

    async def runner() -> None:
        app = AutoCoderApp()
        async with app.run_test() as pilot:  # type: ignore[assignment]
            await pilot.pause()
            await body(app, pilot, manager)

    asyncio.run(runner())


def test_status_updates_render_widgets(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario(app: AutoCoderApp, pilot: Any, _: _StubManager) -> None:
        transcript = app.query_one("#transcript", TranscriptWidget)
        assert any("Initialising Auto-Coder runtime" in line for line in _log_lines(transcript))

        planning_update = _ManagerStatusUpdate(
            message="Planning tasks",
            kind="planning",
            payload={"tasks": ["alpha", "beta"]},
            timestamp=1.0,
        )
        progress_update = _ManagerStatusUpdate(
            message="Working",
            kind="round_start",
            task="alpha",
            payload={"budget": {"name": "alpha", "consumed": 1, "limit": 5, "remaining": 4}},
            timestamp=2.0,
        )

        app.post_message(ManagerStatusMessage(planning_update))
        await pilot.pause()
        app.post_message(ManagerStatusMessage(progress_update))
        await pilot.pause()

        plan_rows = _table_rows(app.query_one("#plan", PlanTrackerWidget).renderable)
        assert ("alpha", "in progress", "1/5") in plan_rows
        assert ("beta", "planned", "") in plan_rows

        budget_rows = _table_rows(app.query_one("#budgets", BudgetMeterWidget).renderable)
        assert ("alpha", "1", "5", "4") in budget_rows

    _run_app(monkeypatch, scenario)


def test_manager_completion_updates_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario(app: AutoCoderApp, pilot: Any, manager: _StubManager) -> None:
        status_update = _ManagerStatusUpdate(
            message="alpha finished",
            kind="success",
            task="alpha",
            timestamp=3.0,
        )
        manager.next_result = _ManagerResult(
            response_text="All done",
            structured_response=None,
            rounds=[],
            plan=[{"name": "alpha", "description": "Sample"}],
            budgets={"alpha": _TaskBudget(name="alpha", consumed=2.0, limit=5.0)},
            status_updates=[status_update],
        )

        app.post_message(PromptSubmitted("build feature"))
        await pilot.pause()
        await pilot.pause()

        assert manager.calls == ["build feature"]

        transcript_lines = _log_lines(app.query_one("#transcript", TranscriptWidget))
        assert any("Auto-Coder" in line and "All done" in line for line in transcript_lines)

        plan_rows = _table_rows(app.query_one("#plan", PlanTrackerWidget).renderable)
        assert ("alpha", "completed", "Sample") in plan_rows

        budget_rows = _table_rows(app.query_one("#budgets", BudgetMeterWidget).renderable)
        assert ("alpha", "2.0", "5.0", "3.0") in budget_rows

        prompt = app.query_one("#prompt", PromptInput)
        assert not prompt.disabled
        assert prompt.placeholder == "Type a prompt and press Enter…"

    _run_app(monkeypatch, scenario)


def test_cli_help_output(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0

    help_output = capsys.readouterr().out
    assert "Launch the Auto-Coder Textual UI" in help_output
    assert "--config" in help_output
