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
    """Returns a list of all ToolFunctionDef objects available in the project."""
    return list(TOOLS)


__all__ = [
    "get_all_tools",
    "TOOLS",
    "TOOL_MAP",
] + [t.name + "_tool" for t in TOOLS if hasattr(t, 'name')]
