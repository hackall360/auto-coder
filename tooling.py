from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from lmstudio import ToolFunctionDef

# Explicitly import and expose tools from internal/tools
from internal.tools.shell import shell_tool
from internal.tools.planner import planner_tool
from internal.tools.patch import patch_tool
from internal.tools.file import file_tool
from internal.tools.process import process_tool
from internal.tools.git import git_tool


# Build a registry of all tools for easy access
def _collect_tools():
    glb = globals()
    tools = [v for k, v in glb.items() if k.endswith('_tool') and isinstance(v, ToolFunctionDef)]
    # Sort by tool name for consistency
    tools.sort(key=lambda t: getattr(t, 'name', ''))
    return tools


TOOLS: list[ToolFunctionDef] = _collect_tools()
TOOL_MAP: dict[str, ToolFunctionDef] = {t.name: t for t in TOOLS}


def get_all_tools() -> list[ToolFunctionDef]:
    """Return a shallow copy of every registered :class:`ToolFunctionDef`."""

    return list(TOOLS)


def get_tool(name: str) -> ToolFunctionDef:
    """Return a tool definition by name.

    Raises
    ------
    KeyError
        If ``name`` does not correspond to a registered tool.
    """

    return TOOL_MAP[name]


def get_tools(names: Sequence[str] | None = None) -> list[ToolFunctionDef]:
    """Return tool definitions matching ``names``.

    When ``names`` is ``None`` the full registry is returned. Duplicate names
    are ignored while preserving the requested order.
    """

    if names is None:
        return get_all_tools()
    seen: set[str] = set()
    resolved: list[ToolFunctionDef] = []
    for name in names:
        if name in seen:
            continue
        resolved.append(get_tool(name))
        seen.add(name)
    return resolved


def resolve_tools(
    *,
    tools: Iterable[ToolFunctionDef] | None = None,
    tool_names: Sequence[str] | None = None,
) -> list[ToolFunctionDef]:
    """Resolve explicit tool definitions and names into a merged tool list."""

    resolved: list[ToolFunctionDef] = []
    if tools is not None:
        for tool in tools:
            if tool not in resolved:
                resolved.append(tool)
    if tool_names is not None:
        for tool in get_tools(tool_names):
            if tool not in resolved:
                resolved.append(tool)
    return resolved


__all__ = [
    "get_all_tools",
    "get_tool",
    "get_tools",
    "resolve_tools",
    "TOOLS",
    "TOOL_MAP",
] + [t.name + "_tool" for t in TOOLS if hasattr(t, 'name')]
