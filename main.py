"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
from typing import Any, Callable, Iterable

from agents.manager import ManagerAgent, ManagerResult, ManagerStatusUpdate
from core import AgentToggleSettings, AutoCoderCore
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


def _split_multi(values: Iterable[str] | None) -> list[str]:
    items: list[str] = []
    if not values:
        return items
    for raw in values:
        segments = [segment.strip() for segment in str(raw).split(",")]
        items.extend(segment for segment in segments if segment)
    return items


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    def section(name: str) -> dict[str, Any]:
        return overrides.setdefault(name, {})  # type: ignore[return-value]

    if args.default_model:
        section("models")["default_model"] = args.default_model
    if args.reasoning_model:
        section("models")["reasoning_model"] = args.reasoning_model
    if args.research_model:
        section("models")["research_model"] = args.research_model
    if args.allow_browsing is not None:
        section("models")["allow_external_browsing"] = bool(args.allow_browsing)

    include_exts = _split_multi(getattr(args, "repo_include_ext", None))
    if include_exts:
        section("repo_context")["include_exts"] = include_exts
    exclude_dirs = _split_multi(getattr(args, "repo_exclude_dir", None))
    if exclude_dirs:
        section("repo_context")["exclude_dirs"] = exclude_dirs
    if args.repo_auto_refresh is not None:
        section("repo_context")["auto_refresh"] = bool(args.repo_auto_refresh)
    if args.repo_refresh_interval is not None:
        section("repo_context")["refresh_interval"] = float(args.repo_refresh_interval)

    enabled = getattr(args, "enable_agent", None) or []
    disabled = getattr(args, "disable_agent", None) or []
    if enabled or disabled:
        agent_section = section("agents")
        for name in enabled:
            agent_section[name] = True
        for name in disabled:
            agent_section[name] = False

    if args.memory_config_path:
        section("memory")["config_path"] = args.memory_config_path
    if args.share_memory is not None:
        section("memory")["share_globally"] = bool(args.share_memory)

    if args.mcp_config_path:
        section("mcp")["config_path"] = args.mcp_config_path
    if args.mcp_auto_start is not None:
        section("mcp")["auto_start"] = bool(args.mcp_auto_start)

    return overrides


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

    args = parser.parse_args(argv)

    overrides = _build_overrides(args)

    try:
        core = AutoCoderCore(config_path=args.config_path, overrides=overrides)
    except MCPConfigurationError as exc:
        print(f"Failed to initialise MCP integration: {exc}", file=sys.stderr)
        return 2

    manager_factory = functools.partial(core.build_manager, status_callback=_status_printer)
    return _interactive_loop(manager_factory, runtime=core)


if __name__ == "__main__":
    sys.exit(main())
