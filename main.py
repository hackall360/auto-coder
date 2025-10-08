"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from agents import AgentBuilder
from agents.manager import ManagerAgent, ManagerResult, ManagerStatusUpdate


def _build_manager() -> ManagerAgent:
    builder = AgentBuilder()
    session = builder.build()

    def status_printer(update: ManagerStatusUpdate) -> None:
        prefix = update.kind.upper()
        task_label = f" [{update.task}]" if update.task else ""
        print(f"[{prefix}{task_label}] {update.message}")

    return ManagerAgent(session=session, status_callback=status_printer)


def _interactive_loop(manager_factory: Callable[[], ManagerAgent]) -> int:
    manager = manager_factory()
    print("Auto-Coder manager ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print()
            break
        if not user_input.strip():
            continue
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        result = manager.run(user_input)
        _render_result(result)
    return 0


def _render_result(result: ManagerResult) -> None:
    for update in result.status_updates:
        if update.kind in {"info", "success", "progress", "round_start", "budget", "planning"}:
            continue  # already surfaced live via callback
        prefix = update.kind.upper()
        task_label = f" [{update.task}]" if update.task else ""
        print(f"[{prefix}{task_label}] {update.message}")
    print(f"Manager: {result.response_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the Auto-Coder manager agent")
    _ = parser.parse_args(argv)
    return _interactive_loop(_build_manager)


if __name__ == "__main__":
    sys.exit(main())
