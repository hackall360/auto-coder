"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
from typing import Any, Callable

from agents.manager import ManagerAgent, ManagerResult, ManagerStatusUpdate
from cli import apply_common_flags, build_overrides
from core import AutoCoderCore
from mcp_tooling import MCPConfigurationError


def _status_printer(update: ManagerStatusUpdate) -> None:
    prefix = update.kind.upper()
    task_label = f" [{update.task}]" if update.task else ""
    print(f"[{prefix}{task_label}] {update.message}")


def _interactive_loop(
    manager_or_factory: ManagerAgent | Callable[[], ManagerAgent],
    *,
    runtime: AutoCoderCore | None = None,
) -> int:
    context: contextlib.AbstractContextManager[Any]
    if runtime is None:
        context = contextlib.nullcontext()
    else:
        context = runtime

    with context:
        manager = manager_or_factory() if callable(manager_or_factory) else manager_or_factory
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
    apply_common_flags(parser)

    args = parser.parse_args(argv)

    overrides = build_overrides(args)

    try:
        core = AutoCoderCore(config_path=args.config_path, overrides=overrides)
    except MCPConfigurationError as exc:
        print(f"Failed to initialise MCP integration: {exc}", file=sys.stderr)
        return 2

    manager_factory = functools.partial(core.build_manager, status_callback=_status_printer)
    return _interactive_loop(manager_factory, runtime=core)


if __name__ == "__main__":
    sys.exit(main())
