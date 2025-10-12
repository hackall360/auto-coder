"""Textual-based terminal UI for interacting with Auto-Coder."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rich.table import Table

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.worker import Worker

from agents.manager import ManagerAgent, ManagerResult, ManagerStatusUpdate
from core import AgentToggleSettings, AutoCoderCore
from mcp_tooling import MCPConfigurationError
from main import _build_overrides


class ManagerStatusMessage(Message):
    """Message emitted when the manager publishes a status update."""

    def __init__(self, update: ManagerStatusUpdate) -> None:
        super().__init__()
        self.update = update


class ManagerResultMessage(Message):
    """Message emitted after the manager finishes handling a prompt."""

    def __init__(self, prompt: str, result: ManagerResult) -> None:
        super().__init__()
        self.prompt = prompt
        self.result = result


class PromptSubmitted(Message):
    """Message emitted when the user submits input through the prompt widget."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class TranscriptWidget(RichLog):
    """Stream conversation entries between the user and Auto-Coder."""

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, id=widget_id)
        self.border_title = "Transcript"

    def add_user_message(self, message: str) -> None:
        self.write(f"[bold cyan]You:[/bold cyan] {message}")

    def add_manager_message(self, message: str) -> None:
        self.write(f"[bold green]Auto-Coder:[/bold green] {message}")

    def add_system_message(self, message: str) -> None:
        self.write(f"[bold yellow]System:[/bold yellow] {message}")


@dataclass(slots=True)
class _PlanEntry:
    description: str = ""
    status: str = "pending"
    progress: str | None = None


class PlanTrackerWidget(Static):
    """Render a simple table outlining the active execution plan."""

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__(id=widget_id or "plan")
        self._entries: "OrderedDict[str, _PlanEntry]" = OrderedDict()
        self.border_title = "Plan"

    def reset_from_plan(self, plan: Iterable[Mapping[str, Any]]) -> None:
        self._entries.clear()
        for item in plan:
            name = str(item.get("name") or f"task-{len(self._entries) + 1}")
            description = str(item.get("description") or item.get("summary") or "")
            status = str(item.get("status") or "pending")
            self._entries[name] = _PlanEntry(description=description, status=status)
        self._refresh()

    def handle_status(self, update: ManagerStatusUpdate) -> None:
        task = update.task
        if update.kind == "planning":
            payload = update.payload or {}
            tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
            if isinstance(tasks, Iterable) and not isinstance(tasks, (str, bytes)):
                for name in tasks:
                    key = str(name)
                    self._entries.setdefault(key, _PlanEntry(status="planned"))
                self._refresh()
            return

        if not task:
            return

        entry = self._entries.setdefault(task, _PlanEntry())
        if update.kind in {"round_start", "progress"}:
            entry.status = "in progress"
            payload = update.payload or {}
            if isinstance(payload, Mapping):
                budget = payload.get("budget")
                if isinstance(budget, Mapping):
                    consumed = budget.get("consumed")
                    limit = budget.get("limit")
                    if consumed is not None and limit not in (None, 0):
                        entry.progress = f"{consumed}/{limit}"
                    elif consumed is not None:
                        entry.progress = f"{consumed}"
        elif update.kind == "success":
            entry.status = "completed"
            entry.progress = None
        elif update.kind == "error":
            entry.status = "error"
            entry.progress = None
        self._refresh()

    def _refresh(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column("Task", overflow="fold")
        table.add_column("Status", overflow="fold")
        table.add_column("Notes", overflow="fold")
        for name, entry in self._entries.items():
            notes = entry.progress or entry.description
            table.add_row(name, entry.status, notes)
        if not self._entries:
            table.add_row("(none)", "pending", "Awaiting plan")
        self.update(table)


class BudgetMeterWidget(Static):
    """Display the budget consumption for each task."""

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__(id=widget_id or "budgets")
        self._budgets: dict[str, Mapping[str, Any]] = {}
        self.border_title = "Budgets"
        self._refresh()

    def handle_status(self, update: ManagerStatusUpdate) -> None:
        payload = update.payload or {}
        budget = payload.get("budget") if isinstance(payload, Mapping) else None
        if not budget or not isinstance(budget, Mapping):
            return
        task = update.task or str(budget.get("name") or "task")
        self._budgets[task] = budget
        self._refresh()

    def apply_result(self, budgets: Mapping[str, Any]) -> None:
        for name, budget in budgets.items():
            if hasattr(budget, "as_dict"):
                self._budgets[name] = budget.as_dict()
            elif isinstance(budget, Mapping):
                self._budgets[name] = dict(budget)
        self._refresh()

    def _refresh(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column("Task", overflow="fold")
        table.add_column("Consumed", justify="right")
        table.add_column("Limit", justify="right")
        table.add_column("Remaining", justify="right")
        if not self._budgets:
            table.add_row("(none)", "-", "-", "-")
        else:
            for name, data in self._budgets.items():
                consumed = data.get("consumed")
                limit = data.get("limit")
                remaining = data.get("remaining")
                table.add_row(
                    name,
                    f"{consumed}" if consumed is not None else "-",
                    f"{limit}" if limit is not None else "-",
                    f"{remaining}" if remaining is not None else "-",
                )
        self.update(table)


class StatusFeedWidget(RichLog):
    """Live feed of status updates from the manager."""

    _KIND_STYLES = {
        "info": "white",
        "success": "green",
        "error": "red",
        "warning": "yellow",
        "planning": "blue",
        "progress": "cyan",
        "round_start": "magenta",
        "budget": "yellow",
    }

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, id=widget_id or "status")
        self.border_title = "Status"

    def add_update(self, update: ManagerStatusUpdate) -> None:
        style = self._KIND_STYLES.get(update.kind, "white")
        task = f" [{update.task}]" if update.task else ""
        self.write(f"[{style}]{update.kind.upper()}{task}: {update.message}[/{style}]")

    def add_message(self, message: str, *, style: str = "yellow") -> None:
        self.write(f"[{style}]{message}[/{style}]")


class PromptInput(Input):
    """Input widget dedicated to capturing user prompts."""

    is_busy = reactive(False)

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__(placeholder="Type a prompt and press Enter…", id=widget_id or "prompt")
        self.disabled = True

    def watch_is_busy(self, busy: bool) -> None:
        self.disabled = busy
        if busy:
            self.placeholder = "Processing request…"
        else:
            self.placeholder = "Type a prompt and press Enter…"

    def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore[override]
        event.stop()
        value = event.value
        self.value = ""
        self.post_message(PromptSubmitted(value))


class AutoCoderApp(App[None]):
    """Textual application hosting the Auto-Coder conversational interface."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #content {
        height: 1fr;
        layout: horizontal;
        padding: 1 2;
        gap: 2;
    }

    #left, #right {
        layout: vertical;
        width: 1fr;
        gap: 1;
    }

    #transcript {
        height: 1fr;
        min-height: 16;
    }

    #plan {
        min-height: 10;
    }

    #budgets {
        min-height: 8;
    }

    #status {
        height: 1fr;
        min-height: 12;
    }

    #prompt-container {
        layout: horizontal;
        padding: 1 2;
    }

    #prompt {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "graceful_quit", "Quit"),
        Binding("escape", "cancel_request", "Cancel"),
    ]

    def __init__(
        self,
        *,
        config_path: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._config_path = config_path
        self._overrides = dict(overrides or {})
        self._core: AutoCoderCore | None = None
        self._manager: ManagerAgent | None = None
        self._worker: Worker[ManagerResult] | None = None
        self._pending_prompt: str | None = None
        self._seen_status_timestamps: set[float] = set()
        self._shutting_down = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="content"):
            with Vertical(id="left"):
                yield TranscriptWidget(widget_id="transcript")
                yield PlanTrackerWidget()
            with Vertical(id="right"):
                yield BudgetMeterWidget()
                yield StatusFeedWidget()
        with Container(id="prompt-container"):
            yield PromptInput()
        yield Footer()

    async def on_mount(self) -> None:
        transcript = self.query_one(TranscriptWidget)
        status_feed = self.query_one(StatusFeedWidget)
        prompt = self.query_one(PromptInput)
        transcript.add_system_message("Initialising Auto-Coder runtime…")
        status_feed.add_message("Starting AutoCoderCore…")
        try:
            await self._initialise_runtime()
        except MCPConfigurationError as exc:
            status_feed.add_message(
                f"Failed to initialise MCP integration: {exc}", style="red"
            )
            transcript.add_system_message(
                "A configuration error occurred while setting up MCP."
            )
            if hasattr(self, "notify"):
                try:
                    self.notify(
                        f"MCP configuration error: {exc}",
                        severity="error",
                    )
                except Exception:  # pragma: no cover - safety net
                    pass
            await asyncio.sleep(0)
            self.exit(result=2)
            return
        except Exception as exc:  # pragma: no cover - defensive
            status_feed.add_message(f"Failed to initialise Auto-Coder: {exc}", style="red")
            transcript.add_system_message("Shutting down due to initialisation failure.")
            await asyncio.sleep(0)
            self.exit(result=1)
            return
        status_feed.add_message("Auto-Coder manager ready", style="green")
        transcript.add_system_message("Manager ready. Type /quit to exit or /cancel to cancel a request.")
        prompt.is_busy = False
        self.set_focus(prompt)

    async def on_shutdown(self, event: events.Shutdown) -> None:  # type: ignore[override]
        await self._shutdown_runtime()

    async def action_graceful_quit(self) -> None:
        await self._shutdown_runtime()
        self.exit()

    def action_cancel_request(self) -> None:
        self._cancel_active_worker()

    async def _initialise_runtime(self) -> None:
        prompt = self.query_one(PromptInput)
        prompt.is_busy = True

        def builder() -> tuple[AutoCoderCore, ManagerAgent]:
            core = AutoCoderCore(config_path=self._config_path, overrides=self._overrides)
            manager = core.build_manager(status_callback=self._handle_status_callback)
            return core, manager

        core, manager = await asyncio.to_thread(builder)
        self._core = core
        self._manager = manager
        prompt.is_busy = False

    def _handle_status_callback(self, update: ManagerStatusUpdate) -> None:
        self.call_from_thread(self.post_message, ManagerStatusMessage(update))

    def _cancel_active_worker(self) -> None:
        if self._worker and not self._worker.is_finished:
            self._worker.cancel()
            status_feed = self.query_one(StatusFeedWidget)
            status_feed.add_message("Cancelling active request…", style="yellow")

    async def _shutdown_runtime(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        prompt = self.query_one(PromptInput)
        prompt.is_busy = True
        if self._worker and not self._worker.is_finished:
            self._worker.cancel()
            with contextlib.suppress(Exception):
                await self._worker.wait()
        if self._core is not None:
            with contextlib.suppress(Exception):
                self._core.shutdown()
            self._core = None
        self._manager = None

    async def handle_prompt_submitted(self, value: str) -> None:
        prompt = value.strip()
        prompt_widget = self.query_one(PromptInput)
        status_feed = self.query_one(StatusFeedWidget)
        transcript = self.query_one(TranscriptWidget)

        if not prompt:
            return
        if prompt.lower() in {"/quit", "quit", "exit"}:
            await self.action_graceful_quit()
            return
        if prompt.lower() in {"/cancel", "cancel"}:
            self._cancel_active_worker()
            return

        if self._manager is None:
            status_feed.add_message("Manager not ready yet", style="red")
            return

        if self._worker and not self._worker.is_finished:
            status_feed.add_message("Request already running. Press Esc or type /cancel to abort.", style="yellow")
            return

        self._pending_prompt = prompt
        prompt_widget.is_busy = True
        transcript.add_user_message(prompt)
        status_feed.add_message("Submitting request…", style="cyan")

        async def runner() -> ManagerResult:
            assert self._manager is not None
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._manager.run, prompt)

        self._worker = self.run_worker(
            runner(),
            name="manager-run",
            exit_on_error=False,
            on_success=self._on_worker_success,
            on_error=self._on_worker_error,
            on_cancel=self._on_worker_cancel,
        )

    def on_prompt_submitted(self, message: PromptSubmitted) -> None:
        message.stop()
        asyncio.create_task(self.handle_prompt_submitted(message.value))

    def _on_worker_success(self, worker: Worker[ManagerResult]) -> None:
        result = worker.result
        prompt = self._pending_prompt or ""
        self._pending_prompt = None
        self._worker = None
        self.query_one(PromptInput).is_busy = False
        self.post_message(ManagerResultMessage(prompt, result))

    def _on_worker_error(self, worker: Worker[ManagerResult]) -> None:
        self._pending_prompt = None
        self._worker = None
        self.query_one(PromptInput).is_busy = False
        status_feed = self.query_one(StatusFeedWidget)
        status_feed.add_message(f"Manager error: {worker.error}", style="red")

    def _on_worker_cancel(self, worker: Worker[ManagerResult]) -> None:
        self._pending_prompt = None
        self._worker = None
        self.query_one(PromptInput).is_busy = False
        status_feed = self.query_one(StatusFeedWidget)
        status_feed.add_message("Request cancelled.", style="yellow")

    def on_manager_status_message(self, message: ManagerStatusMessage) -> None:
        update = message.update
        if update.timestamp in self._seen_status_timestamps:
            return
        self._seen_status_timestamps.add(update.timestamp)
        self.query_one(StatusFeedWidget).add_update(update)
        self.query_one(PlanTrackerWidget).handle_status(update)
        self.query_one(BudgetMeterWidget).handle_status(update)

    def on_manager_result_message(self, message: ManagerResultMessage) -> None:
        result = message.result
        transcript = self.query_one(TranscriptWidget)
        plan = self.query_one(PlanTrackerWidget)
        budgets = self.query_one(BudgetMeterWidget)
        status_feed = self.query_one(StatusFeedWidget)

        if result.plan:
            plan.reset_from_plan(result.plan)
        for update in result.status_updates:
            if update.timestamp in self._seen_status_timestamps:
                continue
            self._seen_status_timestamps.add(update.timestamp)
            status_feed.add_update(update)
            plan.handle_status(update)
            budgets.handle_status(update)
        budgets.apply_result(result.budgets)
        transcript.add_manager_message(result.response_text)

    async def on_key(self, event: events.Key) -> None:  # type: ignore[override]
        if event.key == "enter" and not isinstance(self.focused, PromptInput):
            self.set_focus(self.query_one(PromptInput))
        await super().on_key(event)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Auto-Coder Textual UI")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to config.json providing memory and MCP settings",
    )
    parser.add_argument(
        "--mcp-config",
        dest="mcp_config_path",
        help="Optional override for the MCP server configuration file",
    )
    parser.add_argument(
        "--default-model",
        dest="default_model",
        help="Override the default LLM model used by Auto-Coder",
    )
    parser.add_argument(
        "--reasoning-model",
        dest="reasoning_model",
        help="Override the reasoning model used for complex planning",
    )
    parser.add_argument(
        "--research-model",
        dest="research_model",
        help="Override the research model for web lookups",
    )
    parser.add_argument(
        "--allow-browsing",
        dest="allow_browsing",
        action="store_true",
        help="Enable external browsing tools for the research agent",
    )
    parser.add_argument(
        "--disable-browsing",
        dest="allow_browsing",
        action="store_false",
        help="Disable external browsing tools for the research agent",
    )
    parser.set_defaults(allow_browsing=None)

    agent_choices = sorted(AgentToggleSettings.__annotations__.keys())
    parser.add_argument(
        "--enable-agent",
        dest="enable_agent",
        choices=agent_choices,
        action="append",
        help="Explicitly enable a specialist agent",
    )
    parser.add_argument(
        "--disable-agent",
        dest="disable_agent",
        choices=agent_choices,
        action="append",
        help="Explicitly disable a specialist agent",
    )

    parser.add_argument(
        "--repo-include-ext",
        dest="repo_include_ext",
        action="append",
        help="File extensions to include when indexing the repository context",
    )
    parser.add_argument(
        "--repo-exclude-dir",
        dest="repo_exclude_dir",
        action="append",
        help="Directories to exclude from the repository context index",
    )
    parser.add_argument(
        "--repo-auto-refresh",
        dest="repo_auto_refresh",
        action="store_true",
        help="Enable background refresh of the repository semantic index",
    )
    parser.add_argument(
        "--repo-no-auto-refresh",
        dest="repo_auto_refresh",
        action="store_false",
        help="Disable background refresh of the repository semantic index",
    )
    parser.set_defaults(repo_auto_refresh=None)
    parser.add_argument(
        "--repo-refresh-interval",
        dest="repo_refresh_interval",
        type=float,
        help="Seconds between repository context refreshes",
    )

    parser.add_argument(
        "--memory-config",
        dest="memory_config_path",
        help="Override the memory configuration file path",
    )
    parser.add_argument(
        "--shared-memory",
        dest="share_memory",
        action="store_true",
        help="Share the constructed memory facade globally",
    )
    parser.add_argument(
        "--no-shared-memory",
        dest="share_memory",
        action="store_false",
        help="Avoid sharing the constructed memory facade globally",
    )
    parser.set_defaults(share_memory=None)

    parser.add_argument(
        "--mcp-auto-start",
        dest="mcp_auto_start",
        action="store_true",
        help="Automatically start configured MCP servers",
    )
    parser.add_argument(
        "--no-mcp-auto-start",
        dest="mcp_auto_start",
        action="store_false",
        help="Skip automatic MCP server startup",
    )
    parser.set_defaults(mcp_auto_start=None)

    return parser


def run_tui(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    overrides = _build_overrides(args)

    app = AutoCoderApp(config_path=args.config_path, overrides=overrides)
    result = app.run()
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(run_tui())
