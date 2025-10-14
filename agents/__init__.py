"""Convenience helpers for constructing :class:`AgentSession` instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, MutableSequence, Optional, Sequence

from chat import CallbackMap
from session import AgentSession, Hook
from tooling import ToolRegistry, ToolSpec
from internal.tools.memory import (
    memory_add_tool,
    memory_promote_tool,
    memory_search_tool,
    memory_update_tool,
)

__all__ = [
    "AgentBuilder",
    "create_agent",
    "register_default_toolset",
    "list_default_toolsets",
    "clear_default_toolsets",
]

from .manager import (
    ManagerAgent,
    ManagerResult,
    ManagerStatusUpdate,
    TaskBudget,
)
from .eval import (
    EvalAgent,
    PromptComparison,
    PromptEvalResult,
    RegressionSummary,
)
from .coder import (
    AppliedDiff,
    ChangeSummary,
    CoderAgent,
    CoderResult,
    MinimalPatch,
)
from .dependency import (
    DependencyBuildAgent,
    DependencyCacheDirective,
    DependencyResolution,
    LockfileDiffSummary,
)
from .db_migration import (
    DBMigrationAgent,
    EphemeralDatabaseSpec,
    MigrationRecord,
    MigrationResult,
    SchemaMigrationPlan,
)
from .doc import (
    DocAgent,
    DocumentationResult,
    DocumentationSummary,
    ReadmeDraft,
    ChangelogDraft,
    WalkthroughSection,
)
from .integrations import (
    CIJobPlan,
    PipelineUpdateResult,
    ReleaseMetadata,
    IntegrationsAgent,
)

from .repo_context import (
    DiffBundle,
    DiffFileStat,
    FileSummary,
    RepoContextAgent,
    RepoSearchResult,
    RepoSymbolResult,
)
from .tester import (
    CriticAnalysis,
    CriticStatusEvent,
    TestCriticAgent,
    TestCriticReport,
    TestSuiteConfig,
    TestSuiteResult,
)
from .runner import (
    RunReport,
    RunnerAgent,
)
from .security import (
    SecurityAgent,
    SecurityCacheDirective,
    SecurityScanFinding,
    SecurityScanReport,
    SecurityScanResult,
    SecurityToolchain,
)
from .research import (
    ResearchAgent,
    ResearchResult,
    ResearchSnippet,
)

__all__.extend([
    "ManagerAgent",
    "ManagerResult",
    "ManagerStatusUpdate",
    "TaskBudget",
])

__all__.extend(
    [
        "EvalAgent",
        "PromptEvalResult",
        "PromptComparison",
        "RegressionSummary",
    ]
)

__all__.extend(
    [
        "DependencyBuildAgent",
        "DependencyCacheDirective",
        "DependencyResolution",
        "LockfileDiffSummary",
    ]
)

__all__.extend(
    [
        "DBMigrationAgent",
        "SchemaMigrationPlan",
        "MigrationResult",
        "MigrationRecord",
        "EphemeralDatabaseSpec",
    ]
)

__all__.extend(
    [
        "DocAgent",
        "DocumentationResult",
        "DocumentationSummary",
        "ReadmeDraft",
        "ChangelogDraft",
        "WalkthroughSection",
    ]
)

__all__.extend(
    [
        "IntegrationsAgent",
        "CIJobPlan",
        "PipelineUpdateResult",
        "ReleaseMetadata",
    ]
)

__all__.extend(
    [
        "CoderAgent",
        "CoderResult",
        "MinimalPatch",
        "AppliedDiff",
        "ChangeSummary",
    ]
)

__all__.extend(
    [
        "RepoContextAgent",
        "RepoSearchResult",
        "RepoSymbolResult",
        "FileSummary",
        "DiffFileStat",
        "DiffBundle",
    ]
)

__all__.extend(
    [
        "TestCriticAgent",
        "TestCriticReport",
        "TestSuiteConfig",
        "TestSuiteResult",
        "CriticStatusEvent",
        "CriticAnalysis",
    ]
)

__all__.extend(
    [
        "RunnerAgent",
        "RunReport",
    ]
)

__all__.extend(
    [
        "ResearchAgent",
        "ResearchResult",
        "ResearchSnippet",
    ]
)

__all__.extend(
    [
        "SecurityAgent",
        "SecurityCacheDirective",
        "SecurityScanFinding",
        "SecurityScanReport",
        "SecurityScanResult",
        "SecurityToolchain",
    ]
)


_DEFAULT_TOOL_REGISTRY = ToolRegistry()
_DEFAULT_TOOLSETS: dict[str, tuple[ToolSpec, ...]] = {}


def _to_sequence(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def clear_default_toolsets() -> None:
    """Remove all globally registered default toolsets."""

    _DEFAULT_TOOLSETS.clear()


def list_default_toolsets() -> list[str]:
    """Return the names of configured default toolsets."""

    return sorted(_DEFAULT_TOOLSETS)


def register_default_toolset(
    name: str,
    tools: Iterable[ToolSpec | Any],
    *,
    replace: bool = False,
) -> tuple[ToolSpec, ...]:
    """Register a reusable toolset that can be referenced by ``create_agent``.

    Parameters
    ----------
    name:
        Unique identifier for the toolset.
    tools:
        Iterable of callables, :class:`ToolSpec` instances, or LM Studio
        ``ToolFunctionDef`` objects representing the tools that should be
        included in the set.
    replace:
        When ``True``, an existing toolset with the same ``name`` will be
        overwritten. The default behaviour raises :class:`ValueError` to guard
        against accidental replacement.
    """

    if not replace and name in _DEFAULT_TOOLSETS:
        raise ValueError(f"Toolset '{name}' is already registered")

    specs: list[ToolSpec] = []
    for tool in tools:
        spec = _DEFAULT_TOOL_REGISTRY.register(tool, replace=True)
        specs.append(spec)
    payload = tuple(specs)
    _DEFAULT_TOOLSETS[name] = payload
    return payload


register_default_toolset(
    "memory",
    (
        memory_search_tool,
        memory_add_tool,
        memory_update_tool,
        memory_promote_tool,
    ),
    replace=True,
)


def _ensure_registered(registry: ToolRegistry, tool: ToolSpec | Any) -> ToolSpec:
    if registry is _DEFAULT_TOOL_REGISTRY:
        return _DEFAULT_TOOL_REGISTRY.register(tool, replace=True)
    return registry.register(tool, replace=True)


def _resolve_tools(
    *,
    registry: ToolRegistry,
    toolsets: Iterable[str] | None = None,
    tools: Iterable[ToolSpec | Any] | None = None,
    tool_names: Sequence[str] | None = None,
) -> list[ToolSpec]:
    resolved: list[ToolSpec] = []
    seen: set[str] = set()

    for set_name in _to_sequence(toolsets):
        if set_name not in _DEFAULT_TOOLSETS:
            raise KeyError(f"Unknown toolset '{set_name}'")
        for spec in _DEFAULT_TOOLSETS[set_name]:
            registered = _ensure_registered(registry, spec)
            if registered.name not in seen:
                resolved.append(registered)
                seen.add(registered.name)

    if tools is not None:
        for tool in tools:
            registered = _ensure_registered(registry, tool)
            if registered.name not in seen:
                resolved.append(registered)
                seen.add(registered.name)

    if tool_names:
        for spec in registry.resolve(tool_names=tool_names):
            if spec.name not in seen:
                resolved.append(spec)
                seen.add(spec.name)

    return resolved


def create_agent(
    *,
    system_prompt: str | None = None,
    history: Any | None = None,
    model: Any | None = None,
    model_name: str | None = None,
    toolsets: Iterable[str] | None = None,
    tools: Iterable[ToolSpec | Any] | None = None,
    tool_names: Sequence[str] | None = None,
    registry: ToolRegistry | None = None,
    callbacks: Optional[CallbackMap] = None,
    on_message: Hook | None = None,
    on_tool_call: Hook | None = None,
    on_tool_result: Hook | None = None,
    on_round_start: Hook | None = None,
    on_round_end: Hook | None = None,
) -> AgentSession:
    """Instantiate an :class:`AgentSession` configured with helper defaults."""

    registry = registry or _DEFAULT_TOOL_REGISTRY
    resolved_tools = _resolve_tools(
        registry=registry,
        toolsets=toolsets,
        tools=tools,
        tool_names=tool_names,
    )

    return AgentSession(
        system_prompt=system_prompt,
        history=history,
        model=model,
        model_name=model_name,
        tools=resolved_tools,
        tool_names=None,
        callbacks=callbacks,
        on_message=on_message,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_round_start=on_round_start,
        on_round_end=on_round_end,
        tool_registry=registry,
    )


@dataclass
class AgentBuilder:
    """Fluent builder for assembling agent sessions."""

    system_prompt: str | None = None
    history: Any | None = None
    model: Any | None = None
    model_name: str | None = None
    callbacks: Optional[CallbackMap] = None
    on_message: Hook | None = None
    on_tool_call: Hook | None = None
    on_tool_result: Hook | None = None
    on_round_start: Hook | None = None
    on_round_end: Hook | None = None
    registry: ToolRegistry | None = None
    _toolsets: MutableSequence[str] = field(default_factory=list)
    _tools: MutableSequence[ToolSpec | Any] = field(default_factory=list)
    _tool_names: MutableSequence[str] = field(default_factory=list)
    _mcp_servers: MutableSequence[Any] = field(default_factory=list)

    def with_toolsets(self, *names: str) -> "AgentBuilder":
        self._toolsets.extend(names)
        return self

    def with_tools(self, *tools: ToolSpec | Any) -> "AgentBuilder":
        self._tools.extend(tools)
        return self

    def with_tool_names(self, *names: str) -> "AgentBuilder":
        self._tool_names.extend(names)
        return self

    def with_mcp_servers(self, *configs: Any) -> "AgentBuilder":
        """Schedule MCP server configurations to be registered at build time."""

        if configs:
            self._mcp_servers.extend(configs)
        return self

    def with_model(self, *, model: Any | None = None, model_name: str | None = None) -> "AgentBuilder":
        if model is not None:
            self.model = model
        if model_name is not None:
            self.model_name = model_name
        return self

    def using_registry(self, registry: ToolRegistry) -> "AgentBuilder":
        self.registry = registry
        return self

    def build(self) -> AgentSession:
        registry = self.registry or _DEFAULT_TOOL_REGISTRY
        tools: list[ToolSpec | Any] = list(self._tools)
        if self._mcp_servers:
            from mcp_tooling import register_mcp_servers

            mcp_specs = register_mcp_servers(registry, self._mcp_servers, replace=True)
            tools.extend(mcp_specs)

        return create_agent(
            system_prompt=self.system_prompt,
            history=self.history,
            model=self.model,
            model_name=self.model_name,
            toolsets=self._toolsets,
            tools=tools,
            tool_names=self._tool_names,
            registry=registry,
            callbacks=self.callbacks,
            on_message=self.on_message,
            on_tool_call=self.on_tool_call,
            on_tool_result=self.on_tool_result,
            on_round_start=self.on_round_start,
            on_round_end=self.on_round_end,
        )
