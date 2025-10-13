"""Core runtime orchestration utilities for Auto-Coder."""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json

from agents import AgentBuilder
from agents.dependency import DependencyBuildAgent
from agents.doc import DocAgent
from agents.db_migration import DBMigrationAgent
from agents.eval import EvalAgent
from agents.integrations import IntegrationsAgent
from agents.manager import ManagerAgent
from agents.repo_context import RepoContextAgent
from agents.research import ResearchAgent, VariedResearchAgent
from agents.runner import RunnerAgent
from agents.security import SecurityAgent
from agents.tester import TestCriticAgent
from session import AgentSession
from tooling import ToolRegistry
from memory import (
    MemoryFacade,
    MemoryRouter,
    build_memory_router,
    load_config_json,
    load_memory_configuration,
    set_shared_memory_facade,
)
from mcp_tooling import MCPServerRegistry, MCPConfigurationError, register_mcp_servers

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PathSettings:
    """Filesystem locations used by the runtime.

    Attributes
    ----------
    repo_root:
        Absolute path to the repository that should be inspected by context-aware
        agents. Defaults to the current working directory when not configured.
    workspace_root:
        Optional directory for ephemeral artifacts (for example, generated
        documentation or cached command outputs). When omitted the repository
        root is reused.
    artifact_root:
        Directory used by helper agents such as :class:`RunnerAgent` for
        persisted artifacts. Defaults to ``workspace_root / ".autocoder"`` when
        neither configured nor provided via environment variables.
    """

    repo_root: Path
    workspace_root: Path | None = None
    artifact_root: Path | None = None


@dataclass(slots=True)
class ModelSettings:
    """Model selection and agent-wide behavioural flags."""

    default_model: str | None = None
    reasoning_model: str | None = None
    research_model: str | None = None
    allow_external_browsing: bool = False


@dataclass(slots=True)
class ResearchSettings:
    """Options controlling the behaviour of the :class:`ResearchAgent`."""

    cache_size: int = 8
    cache_top_k: int = 8
    max_quote_chars: int = 320
    web: Mapping[str, Any] | None = None
    enable_varied_agent: bool = False
    default_mode: str = "balanced"
    mode_defaults: Mapping[str, Mapping[str, Any]] | None = None
    profiles: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(slots=True)
class RepoContextSettings:
    """Filters applied when building the repository semantic index."""

    include_exts: tuple[str, ...] | None = None
    exclude_dirs: tuple[str, ...] | None = None
    auto_refresh: bool = True
    refresh_interval: float = 900.0


@dataclass(slots=True)
class AgentToggleSettings:
    """Enable/disable specialist agents without code changes."""

    repo_context: bool = True
    research: bool = True
    documentation: bool = True
    dependency: bool = True
    runner: bool = True
    db_migration: bool = False
    security: bool = False
    integrations: bool = False
    eval: bool = False
    test_critic: bool = True


@dataclass(slots=True)
class MemorySettings:
    """Overrides controlling memory configuration resolution."""

    config_path: Path | None = None
    default_scope: str = MemoryRouter.SHORT_TERM
    combined_scope: str = MemoryRouter.COMBINED
    share_globally: bool = True


@dataclass(slots=True)
class MCPSettings:
    """Configuration for MCP server discovery and lifecycle management."""

    config_path: Path | None = None
    servers: Mapping[str, Any] | None = None
    auto_start: bool = False


@dataclass(slots=True)
class ManagerSettings:
    """Controls for the top-level manager agent."""

    plan_retries: int = 1
    task_retry_limit: int = 0
    specialist_blueprints: tuple[Mapping[str, Any], ...] | None = None


@dataclass(slots=True)
class AutoCoderConfig:
    """Aggregate configuration consumed by :class:`AutoCoderCore`."""

    paths: PathSettings
    models: ModelSettings
    research: ResearchSettings
    repo_context: RepoContextSettings
    agents: AgentToggleSettings
    memory: MemorySettings
    mcp: MCPSettings
    manager: ManagerSettings


def _as_mapping(payload: Any | None) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _coerce_path(value: Any | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _coerce_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_int(value: Any | None, *, default: int = 0, minimum: int | None = None) -> int:
    try:
        if value is None:
            raise ValueError
        integer = int(value)
    except (TypeError, ValueError):
        integer = default
    if minimum is not None and integer < minimum:
        return minimum
    return integer


def _coerce_sequence(value: Any | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip()) or None
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(",")]
        return tuple(part for part in parts if part) or None
    return None


def _coerce_mapping_payload(value: Any | None, *, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse %s as mapping", context, exc_info=True)
            return None
        if isinstance(parsed, Mapping):
            return {str(key): val for key, val in parsed.items()}
        LOGGER.warning("Expected mapping for %s but received %s", context, type(parsed).__name__)
    return None


def _coerce_mapping_tree(value: Any | None, *, context: str) -> dict[str, Mapping[str, Any]] | None:
    root = _coerce_mapping_payload(value, context=context)
    if not root:
        return None
    normalised: dict[str, Mapping[str, Any]] = {}
    for key, candidate in root.items():
        name = str(key).strip().lower()
        if not name:
            continue
        if isinstance(candidate, Mapping):
            normalised[name] = dict(candidate)
            continue
        if isinstance(candidate, str):
            nested = _coerce_mapping_payload(candidate, context=f"{context}.{name}")
            if nested is not None:
                normalised[name] = nested
    return normalised or None


def _first_non_empty(*candidates: Any | None) -> Any | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        return candidate
    return None


def _resolve_config_path(
    config_path: Path | str | None,
    env: Mapping[str, str],
) -> Path | None:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = env.get("AUTO_CODER_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return None


def _merge_sections(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_sections(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_blueprint_keywords(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return _coerce_sequence(raw) or ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return _coerce_sequence(tuple(raw)) or ()
    return ()


def _normalise_blueprint_budget(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    payload: dict[str, Any] = {}
    if "limit" in raw:
        try:
            limit_value = float(raw.get("limit")) if raw.get("limit") is not None else None
        except (TypeError, ValueError):
            limit_value = None
        if limit_value is not None:
            payload["limit"] = limit_value
    if "unit" in raw and raw.get("unit") is not None:
        unit_value = str(raw.get("unit")).strip()
        if unit_value:
            payload["unit"] = unit_value
    for key in ("consumed", "remaining", "progress"):
        if key in payload:
            payload.pop(key, None)
    return payload


def _normalise_blueprint_research(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    payload: dict[str, Any] = {}
    required_flag = _coerce_bool(raw.get("required"))
    if required_flag is not None:
        payload["required"] = required_flag
    audience_value = raw.get("audience")
    if audience_value is not None:
        text = str(audience_value).strip()
        if text:
            payload["audience"] = text
    return payload


def _validate_specialist_blueprints(raw: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    validated: list[Mapping[str, Any]] = []
    for index, blueprint in enumerate(raw):
        if not isinstance(blueprint, Mapping):
            LOGGER.warning("Skipping specialist blueprint %s: expected mapping, got %s", index, type(blueprint).__name__)
            continue
        name = str(blueprint.get("name") or blueprint.get("kind") or "").strip()
        kind = str(blueprint.get("kind") or "").strip()
        agent = str(blueprint.get("agent") or kind or "").strip()
        if not name or not kind or not agent:
            LOGGER.warning(
                "Skipping specialist blueprint %s: missing required fields (name=%r, kind=%r, agent=%r)",
                index,
                blueprint.get("name"),
                blueprint.get("kind"),
                blueprint.get("agent"),
            )
            continue
        description = str(blueprint.get("description") or "").strip()
        keywords = _normalise_blueprint_keywords(blueprint.get("keywords"))
        budget = _normalise_blueprint_budget(blueprint.get("budget"))
        research = _normalise_blueprint_research(blueprint.get("research"))
        metadata = blueprint.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_payload = dict(metadata)
        else:
            metadata_payload = None
        validated_blueprint = dict(blueprint)
        validated_blueprint.update(
            {
                "name": name,
                "kind": kind,
                "agent": agent,
            }
        )
        if description:
            validated_blueprint["description"] = description
        elif "description" in validated_blueprint:
            validated_blueprint["description"] = description
        if keywords:
            validated_blueprint["keywords"] = keywords
        elif "keywords" in validated_blueprint:
            validated_blueprint["keywords"] = ()
        if budget:
            validated_blueprint["budget"] = budget
        elif "budget" in validated_blueprint:
            validated_blueprint["budget"] = {}
        if research:
            validated_blueprint["research"] = research
        elif "research" in validated_blueprint:
            validated_blueprint["research"] = {}
        if metadata_payload is not None:
            validated_blueprint["metadata"] = metadata_payload
        elif "metadata" in validated_blueprint:
            validated_blueprint.pop("metadata", None)
        validated.append(validated_blueprint)
    return tuple(validated) or None


def load_core_configuration(
    config_path: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AutoCoderConfig:
    """Load :class:`AutoCoderConfig` by merging config file, env, and overrides."""

    env_map = dict(env or os.environ)
    resolved_path = _resolve_config_path(config_path, env_map)
    config_payload = load_config_json(resolved_path) if resolved_path else load_config_json()
    core_section = _as_mapping(config_payload.get("core"))
    override_section = _as_mapping(overrides)
    if "core" in override_section:
        override_section = _as_mapping(override_section.get("core"))
    core_section = _merge_sections(core_section, override_section)

    paths_config = _as_mapping(core_section.get("paths"))
    models_config = _as_mapping(core_section.get("models"))
    research_config = _as_mapping(core_section.get("research"))
    repo_config = _as_mapping(core_section.get("repo_context"))
    agent_config = _as_mapping(core_section.get("agents"))
    memory_config = _as_mapping(core_section.get("memory"))
    mcp_config = _as_mapping(core_section.get("mcp"))
    manager_config = _as_mapping(core_section.get("manager"))

    repo_root = _coerce_path(
        _first_non_empty(
            paths_config.get("repo_root"),
            env_map.get("AUTO_CODER_REPO_ROOT"),
        )
    ) or Path.cwd()

    workspace_root = _coerce_path(
        _first_non_empty(
            paths_config.get("workspace_root"),
            env_map.get("AUTO_CODER_WORKSPACE_ROOT"),
        )
    )
    artifact_root = _coerce_path(
        _first_non_empty(
            paths_config.get("artifact_root"),
            env_map.get("AUTO_CODER_ARTIFACT_ROOT"),
        )
    )
    if workspace_root is None:
        workspace_root = repo_root
    if artifact_root is None:
        artifact_root = workspace_root / ".autocoder"

    paths = PathSettings(
        repo_root=repo_root,
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )

    research_override = _as_mapping(override_section.get("research"))

    default_model = _first_non_empty(
        override_section.get("default_model"),
        env_map.get("AUTO_CODER_MODEL"),
        models_config.get("default_model"),
    )
    reasoning_model = _first_non_empty(
        override_section.get("reasoning_model"),
        env_map.get("AUTO_CODER_REASONING_MODEL"),
        models_config.get("reasoning_model"),
    )
    research_model = _first_non_empty(
        override_section.get("research_model"),
        env_map.get("AUTO_CODER_RESEARCH_MODEL"),
        models_config.get("research_model"),
    )
    allow_browsing = _first_non_empty(
        override_section.get("allow_external_browsing"),
        env_map.get("AUTO_CODER_ALLOW_BROWSING"),
        models_config.get("allow_external_browsing"),
    )
    allow_browsing_flag = _coerce_bool(allow_browsing)
    models = ModelSettings(
        default_model=str(default_model) if default_model is not None else None,
        reasoning_model=str(reasoning_model) if reasoning_model is not None else None,
        research_model=str(research_model) if research_model is not None else None,
        allow_external_browsing=bool(allow_browsing_flag) if allow_browsing_flag is not None else False,
    )

    cache_size = _coerce_int(
        _first_non_empty(
            research_override.get("cache_size"),
            env_map.get("AUTO_CODER_RESEARCH_CACHE_SIZE"),
            research_config.get("cache_size"),
        ),
        default=8,
        minimum=1,
    )
    cache_top_k = _coerce_int(
        _first_non_empty(
            research_override.get("cache_top_k"),
            env_map.get("AUTO_CODER_RESEARCH_CACHE_TOP_K"),
            research_config.get("cache_top_k"),
        ),
        default=8,
        minimum=1,
    )
    max_quote_chars = _coerce_int(
        _first_non_empty(
            research_override.get("max_quote_chars"),
            env_map.get("AUTO_CODER_RESEARCH_MAX_QUOTE_CHARS"),
            research_config.get("max_quote_chars"),
        ),
        default=320,
        minimum=80,
    )

    web_config = _as_mapping(research_config.get("web"))
    web_override = _as_mapping(research_override.get("web"))

    proxy_candidate = _first_non_empty(
        web_override.get("proxy"),
        env_map.get("AUTO_CODER_RESEARCH_PROXY"),
        web_config.get("proxy"),
    )
    proxy_value = str(proxy_candidate).strip() if proxy_candidate is not None else None
    if proxy_value == "":
        proxy_value = None

    user_agents_raw = _first_non_empty(
        web_override.get("user_agent_pool"),
        env_map.get("AUTO_CODER_RESEARCH_USER_AGENT_POOL"),
        web_config.get("user_agent_pool"),
    )
    user_agent_pool = _coerce_sequence(user_agents_raw)

    incognito_raw = _first_non_empty(
        web_override.get("incognito_contexts"),
        env_map.get("AUTO_CODER_RESEARCH_INCOGNITO_CONTEXTS"),
        web_config.get("incognito_contexts"),
    )
    incognito_flag = _coerce_bool(incognito_raw)

    anonymous_raw = _first_non_empty(
        web_override.get("anonymous_browsing"),
        env_map.get("AUTO_CODER_RESEARCH_ANONYMOUS_BROWSING"),
        web_config.get("anonymous_browsing"),
    )
    anonymous_flag = _coerce_bool(anonymous_raw)

    web_settings: dict[str, Any] = {}
    if proxy_value:
        web_settings["proxy"] = proxy_value
    if user_agent_pool:
        web_settings["user_agent_pool"] = tuple(user_agent_pool)
    if incognito_flag is not None:
        web_settings["incognito_contexts"] = bool(incognito_flag)
    if anonymous_flag is not None:
        web_settings["anonymous_browsing"] = bool(anonymous_flag)

    varied_raw = _first_non_empty(
        research_override.get("enable_varied_agent"),
        env_map.get("AUTO_CODER_RESEARCH_ENABLE_VARIED_AGENT"),
        research_config.get("enable_varied_agent"),
    )
    varied_flag = _coerce_bool(varied_raw)
    enable_varied_agent = bool(varied_flag) if varied_flag is not None else False

    default_mode_raw = _first_non_empty(
        research_override.get("default_mode"),
        env_map.get("AUTO_CODER_RESEARCH_DEFAULT_MODE"),
        research_config.get("default_mode"),
    )
    default_mode = str(default_mode_raw).strip().lower() if default_mode_raw is not None else "balanced"
    if not default_mode:
        default_mode = "balanced"

    mode_defaults = _coerce_mapping_tree(
        _first_non_empty(
            research_override.get("mode_defaults"),
            env_map.get("AUTO_CODER_RESEARCH_MODE_DEFAULTS"),
            research_config.get("mode_defaults"),
        ),
        context="research.mode_defaults",
    )
    profiles = _coerce_mapping_tree(
        _first_non_empty(
            research_override.get("profiles"),
            env_map.get("AUTO_CODER_RESEARCH_PROFILES"),
            research_config.get("profiles"),
        ),
        context="research.profiles",
    )

    research = ResearchSettings(
        cache_size=cache_size,
        cache_top_k=cache_top_k,
        max_quote_chars=max_quote_chars,
        web=web_settings or None,
        enable_varied_agent=enable_varied_agent,
        default_mode=default_mode,
        mode_defaults=mode_defaults,
        profiles=profiles,
    )

    include_exts = _coerce_sequence(
        _first_non_empty(
            repo_config.get("include_exts"),
            env_map.get("AUTO_CODER_REPO_INCLUDE_EXTS"),
        )
    )
    exclude_dirs = _coerce_sequence(
        _first_non_empty(
            repo_config.get("exclude_dirs"),
            env_map.get("AUTO_CODER_REPO_EXCLUDE_DIRS"),
        )
    )
    auto_refresh = _coerce_bool(
        _first_non_empty(
            repo_config.get("auto_refresh"),
            env_map.get("AUTO_CODER_REPO_AUTO_REFRESH"),
        )
    )
    refresh_interval_candidate = _first_non_empty(
        repo_config.get("refresh_interval"),
        env_map.get("AUTO_CODER_REPO_REFRESH_INTERVAL"),
    )
    try:
        refresh_interval = float(refresh_interval_candidate) if refresh_interval_candidate is not None else 900.0
    except (TypeError, ValueError):
        refresh_interval = 900.0
    repo_context = RepoContextSettings(
        include_exts=include_exts,
        exclude_dirs=exclude_dirs,
        auto_refresh=bool(auto_refresh) if auto_refresh is not None else True,
        refresh_interval=max(60.0, refresh_interval),
    )

    def _toggle(name: str, default: bool) -> bool:
        raw = _first_non_empty(
            agent_config.get(name),
            env_map.get(f"AUTO_CODER_ENABLE_{name.upper()}"),
        )
        coerced = _coerce_bool(raw)
        if coerced is None:
            raw_disable = env_map.get(f"AUTO_CODER_DISABLE_{name.upper()}")
            disable_flag = _coerce_bool(raw_disable)
            if disable_flag is True:
                return False
            return default
        return coerced

    agents = AgentToggleSettings(
        repo_context=_toggle("repo_context", True),
        research=_toggle("research", True),
        documentation=_toggle("documentation", True),
        dependency=_toggle("dependency", True),
        runner=_toggle("runner", True),
        db_migration=_toggle("db_migration", False),
        security=_toggle("security", False),
        integrations=_toggle("integrations", False),
        eval=_toggle("eval", False),
        test_critic=_toggle("test_critic", True),
    )

    memory_path = _coerce_path(
        _first_non_empty(
            memory_config.get("config_path"),
            env_map.get("AUTO_CODER_MEMORY_CONFIG"),
        )
    )
    default_scope = str(
        _first_non_empty(
            memory_config.get("default_scope"),
            env_map.get("AUTO_CODER_MEMORY_DEFAULT_SCOPE"),
            MemoryRouter.SHORT_TERM,
        )
    ).lower()
    combined_scope = str(
        _first_non_empty(
            memory_config.get("combined_scope"),
            env_map.get("AUTO_CODER_MEMORY_COMBINED_SCOPE"),
            MemoryRouter.COMBINED,
        )
    ).lower()
    share_globally = _coerce_bool(
        _first_non_empty(
            memory_config.get("share_globally"),
            env_map.get("AUTO_CODER_MEMORY_SHARE"),
        )
    )
    memory = MemorySettings(
        config_path=memory_path,
        default_scope=default_scope,
        combined_scope=combined_scope,
        share_globally=True if share_globally is None else bool(share_globally),
    )

    mcp_path = _coerce_path(
        _first_non_empty(
            mcp_config.get("config_path"),
            env_map.get("AUTO_CODER_MCP_CONFIG"),
        )
    )
    servers_raw = mcp_config.get("servers") if isinstance(mcp_config.get("servers"), Mapping) else None
    auto_start_flag = _coerce_bool(
        _first_non_empty(
            mcp_config.get("auto_start"),
            env_map.get("AUTO_CODER_MCP_AUTO_START"),
        )
    )
    mcp = MCPSettings(
        config_path=mcp_path,
        servers=servers_raw,
        auto_start=bool(auto_start_flag) if auto_start_flag is not None else False,
    )

    plan_retries = _coerce_int(manager_config.get("plan_retries"), default=1, minimum=0)
    task_retry_limit = _coerce_int(manager_config.get("task_retry_limit"), default=0, minimum=0)
    blueprints = _validate_specialist_blueprints(manager_config.get("specialist_blueprints"))
    manager = ManagerSettings(
        plan_retries=plan_retries,
        task_retry_limit=task_retry_limit,
        specialist_blueprints=blueprints,
    )

    return AutoCoderConfig(
        paths=paths,
        models=models,
        research=research,
        repo_context=repo_context,
        agents=agents,
        memory=memory,
        mcp=mcp,
        manager=manager,
    )


class AutoCoderCore:
    """Central orchestrator responsible for wiring high-level agents together."""

    def __init__(
        self,
        config: AutoCoderConfig | None = None,
        *,
        config_path: Path | str | None = None,
        overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or load_core_configuration(
            config_path,
            env=env,
            overrides=overrides,
        )
        self.tool_registry = ToolRegistry()
        self._memory_config = load_memory_configuration(self.config.memory.config_path)
        self.memory_router = build_memory_router(self._memory_config)
        self.memory_facade = MemoryFacade(
            self.memory_router,
            default_scope=self.config.memory.default_scope,
            combined_scope=self.config.memory.combined_scope,
        )
        self._shared_memory_installed = False
        if self.config.memory.share_globally:
            set_shared_memory_facade(self.memory_facade)
            self._shared_memory_installed = True

        self._mcp_registry: MCPServerRegistry | None = None
        self._mcp_specs: tuple[Any, ...] = ()
        self._setup_mcp_registry()

        self._repo_context_agent: RepoContextAgent | None = None
        self._research_agent: ResearchAgent | VariedResearchAgent | None = None
        self._doc_agent: DocAgent | None = None
        self._runner_agent: RunnerAgent | None = None
        self._dependency_agent: DependencyBuildAgent | None = None
        self._db_migration_agent: DBMigrationAgent | None = None
        self._security_agent: SecurityAgent | None = None
        self._integrations_agent: IntegrationsAgent | None = None
        self._eval_agent: EvalAgent | None = None
        self._test_critic_agent: TestCriticAgent | None = None

    # ------------------------------------------------------------------
    # MCP setup
    # ------------------------------------------------------------------
    def _setup_mcp_registry(self) -> None:
        settings = self.config.mcp
        registry: MCPServerRegistry | None = None
        if settings.servers:
            registry = MCPServerRegistry(settings.servers)
        else:
            path_hint = settings.config_path
            try:
                registry = MCPServerRegistry.from_loaded_config(path_hint)
            except MCPConfigurationError:
                LOGGER.debug("MCP configuration not available; continuing without MCP integration", exc_info=True)
                registry = None
            except FileNotFoundError:
                LOGGER.debug("MCP configuration file %s not found", path_hint)
                registry = None
        if registry:
            try:
                specs = registry.build_specs(auto_start=settings.auto_start)
            except (MCPConfigurationError, TimeoutError, OSError):
                LOGGER.warning("Failed to initialise MCP servers; continuing without MCP tools", exc_info=True)
                specs = []
            self._mcp_registry = registry
            self._mcp_specs = tuple(specs)
        else:
            self._mcp_registry = None
            self._mcp_specs = ()

    # ------------------------------------------------------------------
    # Agent factories
    # ------------------------------------------------------------------
    def _get_repo_context(self) -> RepoContextAgent | None:
        if not self.config.agents.repo_context:
            return None
        if self._repo_context_agent is None:
            try:
                repo_root = str(self.config.paths.repo_root)
                include_exts = self.config.repo_context.include_exts
                exclude_dirs = self.config.repo_context.exclude_dirs
                agent = RepoContextAgent(
                    repo_root,
                    include_exts=include_exts,
                    exclude_dirs=exclude_dirs,
                    auto_refresh=self.config.repo_context.auto_refresh,
                    refresh_interval=self.config.repo_context.refresh_interval,
                )
                self._repo_context_agent = agent
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise RepoContextAgent; repository features disabled", exc_info=True)
                self._repo_context_agent = None
        return self._repo_context_agent

    def _get_research_agent(self) -> ResearchAgent | VariedResearchAgent | None:
        if not self.config.agents.research:
            return None
        if self._research_agent is None:
            try:
                settings = self.config.research
                web_kwargs = dict(settings.web or {})
                anonymous_override = web_kwargs.pop("anonymous_browsing", None)
                if anonymous_override is None:
                    anonymous_flag = not self.config.models.allow_external_browsing
                else:
                    anonymous_flag = bool(anonymous_override)
                base_agent = ResearchAgent(
                    cache_size=settings.cache_size,
                    cache_top_k=settings.cache_top_k,
                    max_quote_chars=settings.max_quote_chars,
                    anonymous_browsing=anonymous_flag,
                    **web_kwargs,
                )
                if settings.enable_varied_agent:
                    mode_defaults = (
                        {name: dict(payload) for name, payload in settings.mode_defaults.items()}
                        if settings.mode_defaults
                        else None
                    )
                    profiles = (
                        {name: dict(payload) for name, payload in settings.profiles.items()}
                        if settings.profiles
                        else None
                    )
                    self._research_agent = VariedResearchAgent(
                        base_agent,
                        mode_defaults=mode_defaults,
                        profiles=profiles,
                        default_mode=settings.default_mode,
                    )
                else:
                    self._research_agent = base_agent
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise ResearchAgent; disabling research features", exc_info=True)
                self._research_agent = None
        return self._research_agent

    def _get_runner(self) -> RunnerAgent | None:
        if not self.config.agents.runner:
            return None
        if self._runner_agent is None:
            try:
                self._runner_agent = RunnerAgent(
                    repo_root=str(self.config.paths.repo_root),
                    artifact_root=str(self.config.paths.artifact_root),
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise RunnerAgent; disabling command execution helpers", exc_info=True)
                self._runner_agent = None
        return self._runner_agent

    def _get_dependency_agent(self) -> DependencyBuildAgent | None:
        if not self.config.agents.dependency:
            return None
        if self._dependency_agent is None:
            runner = self._get_runner()
            try:
                self._dependency_agent = DependencyBuildAgent(
                    runner=runner,
                    repo_root=self.config.paths.repo_root,
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise DependencyBuildAgent; disabling dependency workflows", exc_info=True)
                self._dependency_agent = None
        return self._dependency_agent

    def _get_doc_agent(self) -> DocAgent | None:
        if not self.config.agents.documentation:
            return None
        repo_context = self._get_repo_context()
        if repo_context is None:
            return None
        research = self._get_research_agent()
        if self._doc_agent is None:
            try:
                self._doc_agent = DocAgent(
                    repo_context,
                    research_agent=research,
                    artifact_dir=self.config.paths.workspace_root / ".doc_artifacts",
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise DocAgent; documentation support disabled", exc_info=True)
                self._doc_agent = None
        else:
            self._doc_agent.attach_research_agent(research)
        return self._doc_agent

    def _get_db_migration_agent(self) -> DBMigrationAgent | None:
        if not self.config.agents.db_migration:
            return None
        repo_context = self._get_repo_context()
        if repo_context is None:
            return None
        runner = self._get_runner()
        if self._db_migration_agent is None:
            try:
                self._db_migration_agent = DBMigrationAgent(
                    repo_context,
                    runner=runner,
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise DBMigrationAgent; migration workflows disabled", exc_info=True)
                self._db_migration_agent = None
        return self._db_migration_agent

    def _get_security_agent(self) -> SecurityAgent | None:
        if not self.config.agents.security:
            return None
        runner = self._get_runner()
        if self._security_agent is None:
            try:
                self._security_agent = SecurityAgent(runner=runner)
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise SecurityAgent; security scanning disabled", exc_info=True)
                self._security_agent = None
        return self._security_agent

    def _get_integrations_agent(self) -> IntegrationsAgent | None:
        if not self.config.agents.integrations:
            return None
        repo_context = self._get_repo_context()
        if repo_context is None:
            return None
        runner = self._get_runner()
        if self._integrations_agent is None:
            try:
                self._integrations_agent = IntegrationsAgent(
                    repo_context=repo_context,
                    runner=runner,
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise IntegrationsAgent; integrations support disabled", exc_info=True)
                self._integrations_agent = None
        return self._integrations_agent

    def _get_eval_agent(self) -> EvalAgent | None:
        if not self.config.agents.eval:
            return None
        if self._eval_agent is None:
            try:
                self._eval_agent = EvalAgent(
                    session_factory=self._create_session,
                )
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise EvalAgent; evaluation disabled", exc_info=True)
                self._eval_agent = None
        return self._eval_agent

    def _get_test_critic(self) -> TestCriticAgent | None:
        if not self.config.agents.test_critic:
            return None
        if self._test_critic_agent is None:
            try:
                self._test_critic_agent = TestCriticAgent(repo_root=str(self.config.paths.repo_root))
            except Exception:  # pragma: no cover - safeguard
                LOGGER.warning("Failed to initialise TestCriticAgent; gating disabled", exc_info=True)
                self._test_critic_agent = None
        return self._test_critic_agent

    def _create_session(self) -> AgentSession:
        builder = AgentBuilder()
        builder.using_registry(self.tool_registry)
        builder.with_toolsets("memory")
        if self.config.models.default_model:
            builder.with_model(model_name=self.config.models.default_model)
        if self._mcp_specs:
            builder.with_mcp_servers(*self._mcp_specs)
        return builder.build()

    # ------------------------------------------------------------------
    # Manager construction
    # ------------------------------------------------------------------
    def build_manager(
        self,
        *,
        status_callback: Callable[[Any], None] | None = None,
    ) -> ManagerAgent:
        session = self._create_session()
        repo_context = self._get_repo_context()
        research_agent = self._get_research_agent()
        doc_agent = self._get_doc_agent()
        dependency_agent = self._get_dependency_agent()
        db_agent = self._get_db_migration_agent()
        security_agent = self._get_security_agent()
        integrations_agent = self._get_integrations_agent()
        eval_agent = self._get_eval_agent()
        test_critic = self._get_test_critic()

        manager = ManagerAgent(
            session=session,
            status_callback=status_callback,
            plan_retries=self.config.manager.plan_retries,
            task_retry_limit=self.config.manager.task_retry_limit,
            specialist_blueprints=self.config.manager.specialist_blueprints,
            repo_context=repo_context,
            test_critic=test_critic,
            research_agent=research_agent,
            dependency_agent=dependency_agent,
            db_migration_agent=db_agent,
            eval_agent=eval_agent,
            security_agent=security_agent,
            doc_agent=doc_agent,
            external_browsing_default=self.config.models.allow_external_browsing,
            memory_router=self.memory_router,
            memory_facade=self.memory_facade,
            mcp_registry=self._mcp_registry,
        )

        if integrations_agent is not None:
            manager._integrations_agent = integrations_agent
        if research_agent is not None:
            manager.attach_research_agent(research_agent)
        if repo_context is not None:
            manager.attach_repo_context(repo_context)
        if dependency_agent is not None:
            manager.attach_dependency_agent(dependency_agent)
        if test_critic is not None:
            manager.attach_test_critic(test_critic)
        if eval_agent is not None:
            manager.attach_eval_agent(eval_agent)

        if self._mcp_specs:
            specs = register_mcp_servers(self.tool_registry, self._mcp_specs, replace=True)
            with contextlib.suppress(Exception):
                session.replace_mcp_tools(specs)

        return manager

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._repo_context_agent is not None:
            with contextlib.suppress(Exception):
                self._repo_context_agent.stop_background_refresh()
            self._repo_context_agent = None
        if self._mcp_registry is not None:
            with contextlib.suppress(Exception):
                self._mcp_registry.shutdown_all()
        if self._shared_memory_installed:
            set_shared_memory_facade(None)
            self._shared_memory_installed = False

    def __enter__(self) -> "AutoCoderCore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
