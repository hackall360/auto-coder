"""Generate a static screenshot of the Auto-Coder Text UI layout."""

from __future__ import annotations

from pathlib import Path

from rich.table import Table

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

ASSET_PATH = Path(__file__).resolve().parents[1] / "docs" / "assets" / "text-ui-demo.png"


class TranscriptWidget(RichLog):
    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, id="transcript")
        self.border_title = "Transcript"


class PlanWidget(Static):
    def __init__(self) -> None:
        super().__init__(id="plan")
        self.border_title = "Plan"

    def load_sample(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column("Task", overflow="fold")
        table.add_column("Status", overflow="fold")
        table.add_column("Notes", overflow="fold")
        table.add_row("outline", "completed", "Summarise requested change")
        table.add_row("tests", "in progress", "1/3")
        table.add_row("docs", "pending", "Awaiting manager")
        self.update(table)


class BudgetWidget(Static):
    def __init__(self) -> None:
        super().__init__(id="budgets")
        self.border_title = "Budgets"

    def load_sample(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column("Task", overflow="fold")
        table.add_column("Consumed", justify="right")
        table.add_column("Limit", justify="right")
        table.add_column("Remaining", justify="right")
        table.add_row("outline", "1", "1", "0")
        table.add_row("tests", "1", "3", "2")
        table.add_row("docs", "0", "2", "2")
        self.update(table)


class StatusWidget(RichLog):
    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, id="status")
        self.border_title = "Status"

    def load_sample(self) -> None:
        messages = [
            ("blue", "PLANNING [outline]: Drafting execution plan"),
            ("magenta", "ROUND_START [tests]: Evaluating repository"),
            ("cyan", "PROGRESS [tests]: Running pytest -q"),
            ("green", "SUCCESS [outline]: Outline approved"),
        ]
        for style, message in messages:
            self.write(f"[{style}]{message}[/{style}]")


class PromptInput(Input):
    is_busy = False

    def __init__(self) -> None:
        super().__init__(id="prompt", placeholder="Type a prompt and press Enter…")


class DemoTextUI(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #content {
        height: 1fr;
        layout: horizontal;
        padding: 1 2;
    }

    #left, #right {
        layout: vertical;
        width: 1fr;
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
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="content"):
            with Vertical(id="left"):
                yield TranscriptWidget()
                yield PlanWidget()
            with Vertical(id="right"):
                yield BudgetWidget()
                yield StatusWidget()
        with Container(id="prompt-container"):
            yield PromptInput()
        yield Footer()

    async def on_mount(self) -> None:
        transcript = self.query_one(TranscriptWidget)
        plan = self.query_one(PlanWidget)
        budgets = self.query_one(BudgetWidget)
        status = self.query_one(StatusWidget)
        prompt = self.query_one(PromptInput)

        transcript.write("[bold yellow]System:[/bold yellow] Manager ready. Type /quit to exit.")
        transcript.write("[bold cyan]You:[/bold cyan] Document the Text UI workflow.")
        transcript.write("[bold green]Auto-Coder:[/bold green] Starting plan execution…")

        plan.load_sample()
        budgets.load_sample()
        status.load_sample()

        prompt.value = "/help"

        # Allow the UI to render for a few frames before capturing a screenshot.
        self.set_timer(0.3, self._capture_and_quit)

    async def _capture_and_quit(self) -> None:
        ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.save_screenshot(str(ASSET_PATH))
        await self.action_quit()


def main() -> None:
    app = DemoTextUI()
    app.run(headless=True)


if __name__ == "__main__":
    main()
