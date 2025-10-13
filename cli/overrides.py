"""Shared helpers for CLI argument parsing and overrides construction."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

from core import AgentToggleSettings

__all__ = ["apply_common_flags", "build_overrides"]


def _split_multi(values: Iterable[str] | None) -> list[str]:
    """Split comma-delimited CLI arguments while preserving order."""

    items: list[str] = []
    if not values:
        return items
    for raw in values:
        segments = [segment.strip() for segment in str(raw).split(",")]
        items.extend(segment for segment in segments if segment)
    return items


def _parse_key_value(values: Iterable[str] | None) -> dict[str, str]:
    """Parse ``key=value`` pairs supplied via repeated CLI flags."""

    mapping: dict[str, str] = {}
    if not values:
        return mapping
    for raw in values:
        text = str(raw).strip()
        if not text or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            mapping[key] = value
    return mapping


def _parse_log_level(value: str) -> str:
    """Validate and normalise a logging level supplied via the CLI."""

    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("Log level cannot be empty")
    if text.isdigit():
        return str(int(text))
    try:
        mapping = logging.getLevelNamesMapping()
        valid = {name.upper() for name in mapping}
    except AttributeError:  # pragma: no cover - Python < 3.11 fallback
        valid = {
            name.upper()
            for name in logging._nameToLevel  # type: ignore[attr-defined]
            if isinstance(name, str)
        }
    upper = text.upper()
    if upper not in valid:
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}")
    return upper


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build a nested overrides dictionary from parsed CLI arguments."""

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

    if getattr(args, "corpus_enabled", None) is not None:
        section("corpus")["enabled"] = bool(args.corpus_enabled)
    if getattr(args, "corpus_storage_path", None):
        section("corpus")["storage_path"] = args.corpus_storage_path
    if getattr(args, "corpus_dedup_threshold", None) is not None:
        section("corpus")["dedup_threshold"] = float(args.corpus_dedup_threshold)
    corpus_categories = _parse_key_value(getattr(args, "corpus_category", None))
    if corpus_categories:
        section("corpus")["default_categories"] = corpus_categories

    if args.mcp_config_path:
        section("mcp")["config_path"] = args.mcp_config_path
    if args.mcp_auto_start is not None:
        section("mcp")["auto_start"] = bool(args.mcp_auto_start)

    return overrides


def apply_common_flags(parser: argparse.ArgumentParser) -> None:
    """Attach shared CLI options used across Auto-Coder entry points."""

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
        "--enable-corpus",
        dest="corpus_enabled",
        action="store_true",
        help="Enable structured corpus capture",
    )
    parser.add_argument(
        "--disable-corpus",
        dest="corpus_enabled",
        action="store_false",
        help="Disable structured corpus capture",
    )
    parser.set_defaults(corpus_enabled=None)
    parser.add_argument(
        "--corpus-path",
        dest="corpus_storage_path",
        help="Path to a JSONL file where corpus events should be written",
    )
    parser.add_argument(
        "--corpus-dedup-threshold",
        dest="corpus_dedup_threshold",
        type=float,
        help="Similarity threshold (0-1) used to deduplicate corpus events",
    )
    parser.add_argument(
        "--corpus-category",
        dest="corpus_category",
        action="append",
        metavar="EVENT=CATEGORY",
        help="Override default category mapping for a corpus event type",
    )

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

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        dest="log_level",
        action="store_const",
        const="DEBUG",
        help="Increase logging verbosity to DEBUG",
    )
    verbosity.add_argument(
        "--quiet",
        dest="log_level",
        action="store_const",
        const="WARNING",
        help="Reduce logging verbosity to WARNING",
    )
    verbosity.add_argument(
        "--log-level",
        dest="log_level",
        metavar="LEVEL",
        type=_parse_log_level,
        help="Explicit logging level (e.g. debug, info, warning)",
    )
    parser.set_defaults(log_level=None)

