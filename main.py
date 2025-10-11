"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Sequence

from agents import AgentBuilder
from agents.manager import ManagerAgent, ManagerResult, ManagerStatusUpdate
from memory import MemoryFacade, build_memory_router, set_shared_memory_facade
from mcp_tooling import MCPConfigurationError, MCPServerRegistry


def _build_manager(
    *,
    mcp_servers: Sequence[Any] | None = None,
    mcp_registry: MCPServerRegistry | None = None,
) -> ManagerAgent:
    router = build_memory_router()
    facade = MemoryFacade(router)
    set_shared_memory_facade(facade)

    builder = AgentBuilder()
    builder.with_toolsets("memory")
    if mcp_servers:
        builder.with_mcp_servers(*mcp_servers)
    session = builder.build()

    def status_printer(update: ManagerStatusUpdate) -> None:
        prefix = update.kind.upper()
        task_label = f" [{update.task}]" if update.task else ""
        print(f"[{prefix}{task_label}] {update.message}")

    return ManagerAgent(
        session=session,
        status_callback=status_printer,
        memory_router=router,
        memory_facade=facade,
        mcp_registry=mcp_registry,
    )


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
    args = parser.parse_args(argv)

    config_override = args.mcp_config_path or args.config_path

    try:
        mcp_registry = MCPServerRegistry.from_loaded_config(config_override)
    except MCPConfigurationError as exc:
        print(f"Failed to load MCP configuration: {exc}", file=sys.stderr)
        return 2

    try:
        mcp_specs = mcp_registry.build_specs(auto_start=True)
    except (MCPConfigurationError, TimeoutError, OSError) as exc:
        print(f"Failed to start MCP servers: {exc}", file=sys.stderr)
        return 2

    try:
        return _interactive_loop(
            lambda: _build_manager(mcp_servers=mcp_specs, mcp_registry=mcp_registry)
        )
    finally:
        mcp_registry.shutdown_all()


if __name__ == "__main__":
    sys.exit(main())
