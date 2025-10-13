"""High-level manager agent orchestrating DAG-driven task workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING
from uuid import uuid4

from internal.DAG import DAG
from internal.structures import StructuredResponse
from session import AgentRound, AgentSession
from tooling import ToolRegistry, ToolSpec
from memory import (
    ConversationMemoryHooks,
    MemoryFacade,
    MemoryRouter,
    MemoryRecord,
    get_shared_memory_facade,
    get_shared_memory_router,
    set_shared_memory_facade,
)

from .dependency import DependencyBuildAgent
from .doc import DocAgent
from .db_migration import DBMigrationAgent
from .eval import EvalAgent, RegressionSummary
from .integrations import CIJobPlan, IntegrationsAgent
from .repo_context import (
    DiffBundle,
    RepoContextAgent,
    RepoSearchResult,
    RepoSymbolResult,
)
from .security import SecurityAgent, SecurityScanResult
from .research import VariedResearchAgent

if TYPE_CHECKING:
    from mcp_tooling import MCPServerRegistry
    from .research import ResearchAgent, ResearchResult, ResearchSnippet
    from .tester import CriticStatusEvent, TestCriticAgent, TestCriticReport
    from .eval import PromptComparison

__all__ = [
    "TaskBudget",
    "ManagerStatusUpdate",
    "ManagerResult",
    "ManagerAgent",
]


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskBudget:
    """Track quota consumption for an individual task."""

    name: str
    limit: float | None = 1.0
    unit: str = "rounds"
    consumed: float = 0.0

    def consume(self, amount: float = 1.0) -> None:
        if amount <= 0:
            return
        self.consumed += float(amount)
        if self.limit is not None and self.consumed > self.limit:
            self.consumed = self.limit

    @property
    def remaining(self) -> float | None:
        if self.limit is None:
            return None
        return max(self.limit - self.consumed, 0.0)

    @property
    def progress(self) -> float | None:
        if self.limit in (None, 0):
            return None
        return min(self.consumed / self.limit, 1.0)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "consumed": self.consumed,
            "unit": self.unit,
        }
        if self.limit is not None:
            payload["limit"] = self.limit
            payload["remaining"] = self.remaining
            payload["progress"] = self.progress
        else:
            payload["limit"] = None
            payload["remaining"] = None
            payload["progress"] = None
        return payload

    def copy(self) -> "TaskBudget":
        return TaskBudget(name=self.name, limit=self.limit, unit=self.unit, consumed=self.consumed)


@dataclass(slots=True)
class ManagerStatusUpdate:
    """Message surfaced to users describing manager workflow progress."""

    message: str
    kind: str = "info"
    task: str | None = None
    payload: Mapping[str, Any] | None = None
    timestamp: float = field(default_factory=lambda: time.time())

    def as_dict(self) -> dict[str, Any]:
        data = {
            "message": self.message,
            "kind": self.kind,
            "task": self.task,
            "timestamp": self.timestamp,
        }
        if self.payload is not None:
            data["payload"] = dict(self.payload)
        return data


@dataclass(slots=True)
class ManagerResult:
    """Aggregate output returned after the manager completes a workflow."""

    response_text: str
    structured_response: StructuredResponse | None
    rounds: list[AgentRound]
    plan: list[dict[str, Any]]
    budgets: dict[str, TaskBudget]
    status_updates: list[ManagerStatusUpdate]
    evidence: Mapping[str, tuple["ResearchSnippet", ...]] = field(default_factory=dict)
    evaluations: tuple[RegressionSummary, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "structured_response": self.structured_response,
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "plan": [dict(step) for step in self.plan],
            "budgets": {name: budget.as_dict() for name, budget in self.budgets.items()},
            "status_updates": [update.as_dict() for update in self.status_updates],
            "evidence": {
                key: [
                    snippet.to_dict() if hasattr(snippet, "to_dict") else snippet
                    for snippet in value
                ]
                for key, value in self.evidence.items()
            },
            "evaluations": [summary.to_dict() for summary in self.evaluations],
        }


class ManagerAgent:
    """Top-level orchestrator for user-facing conversations."""

    def __init__(
        self,
        *,
        session: AgentSession | None = None,
        session_factory: Callable[[], AgentSession] | None = None,
        status_callback: Callable[[ManagerStatusUpdate], None] | None = None,
        plan_builder: Callable[[str], Sequence[Mapping[str, Any]]] | None = None,
        plan_retries: int = 1,
        task_retry_limit: int = 0,
        specialist_blueprints: Sequence[Mapping[str, Any]] | None = None,
        repo_context: RepoContextAgent | None = None,
        test_critic: "TestCriticAgent" | None = None,
        research_agent: "ResearchAgent" | VariedResearchAgent | None = None,
        external_browsing_default: bool = False,
        dependency_agent: DependencyBuildAgent | None = None,
        db_migration_agent: DBMigrationAgent | None = None,
        eval_agent: EvalAgent | None = None,
        security_agent: SecurityAgent | None = None,
        doc_agent: DocAgent | None = None,
        memory_router: MemoryRouter | None = None,
        memory_facade: MemoryFacade | None = None,
        session_id: str | None = None,
        mcp_registry: "MCPServerRegistry" | None = None,
    ) -> None:
        if session is None:
            if session_factory is None:
                raise ValueError("ManagerAgent requires a session or session_factory")
            session = session_factory()
        self.session = session
        self.tool_registry: ToolRegistry | None = getattr(session, "tool_registry", None)
        self.mcp_registry = mcp_registry
        self.session_id = session_id or uuid4().hex
        self._status_callback = status_callback
        self._plan_builder = plan_builder or self._default_plan_builder
        self.plan_retries = max(0, plan_retries)
        self.task_retry_limit = max(0, task_retry_limit)
        self._specialist_blueprints = self._resolve_specialist_blueprints(specialist_blueprints)

        self._active_plan: list[dict[str, Any]] = []
        self._budgets: dict[str, TaskBudget] = {}
        self._status_log: list[ManagerStatusUpdate] = []
        self._task_outputs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._current_task: str | None = None
        self._current_round_task: str | None = None
        self._last_response_text: str = ""
        self._last_structured: StructuredResponse | None = None
        self._initial_round_index: int = len(self.session.rounds)
        self._runtime_metadata: dict[str, Any] = {}
        self.repo_context = repo_context
        self.test_critic: "TestCriticAgent" | None = None
        self._gate_report: "TestCriticReport" | None = None
        self._gate_blocked: bool = False
        self.research_agent: "ResearchAgent" | VariedResearchAgent | None = research_agent
        self._external_browsing_default = bool(external_browsing_default)
        self._external_browsing_enabled = self._external_browsing_default
        self._shared_evidence: dict[str, list["ResearchSnippet"]] = {}
        self._dependency_agent: DependencyBuildAgent | None = dependency_agent
        self._db_migration_agent: DBMigrationAgent | None = db_migration_agent
        self.eval_agent: EvalAgent | None = None
        self._security_agent: SecurityAgent | None = security_agent
        self._security_report: SecurityScanResult | None = None
        self._doc_agent: DocAgent | None = doc_agent
        self._integrations_agent: IntegrationsAgent | None = None
        self._gate_source: str | None = None
        self._last_eval_summary: RegressionSummary | None = None
        self._completed_evaluations: list[RegressionSummary] = []

        resolved_router = memory_router
        resolved_facade = memory_facade

        if resolved_facade is None:
            if resolved_router is None:
                try:
                    resolved_facade = get_shared_memory_facade()
                except Exception:  # pragma: no cover - defensive fallback
                    resolved_router = get_shared_memory_router()
                    resolved_facade = MemoryFacade(resolved_router)
                else:
                    resolved_router = resolved_facade.router
            else:
                resolved_facade = MemoryFacade(resolved_router)
        else:
            if resolved_router is None:
                resolved_router = resolved_facade.router
            set_shared_memory_facade(resolved_facade)

        self.memory_router = resolved_router
        self.memory_facade = resolved_facade
        self._memory_hooks: ConversationMemoryHooks | None = None
        if self.memory_facade is not None:
            set_shared_memory_facade(self.memory_facade)
            self._memory_hooks = ConversationMemoryHooks(
                self.memory_facade,
                session_id=self.session_id,
                agent_label="manager",
            )
            self._seed_conversation_history_from_memory()
        if test_critic is not None:
            self.attach_test_critic(test_critic)
        if eval_agent is not None:
            self.attach_eval_agent(eval_agent)
        if doc_agent is not None and self.research_agent is not None:
            doc_agent.attach_research_agent(self.research_agent)

        self.session.add_round_hooks(
            on_round_start=self._handle_round_start,
            on_round_end=self._handle_round_end,
        )
        if self._memory_hooks is not None:
            self.session.add_round_hooks(
                on_round_start=self._memory_hooks.on_round_start,
                on_round_end=self._memory_hooks.on_round_end,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, user_message: str, *, metadata: Mapping[str, Any] | None = None) -> ManagerResult:
        """Execute the manager workflow for a new user input."""

        self._reset_runtime_state()
        self._runtime_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        requested_browsing = None
        for key in ("external_browsing", "allow_external_browsing", "enable_external_browsing"):
            if key in self._runtime_metadata:
                requested_browsing = self._runtime_metadata[key]
                break
        if requested_browsing is not None:
            self._external_browsing_enabled = bool(requested_browsing)
        if self.research_agent is not None:
            self._publish_status(
                "External browsing "
                + ("enabled" if self._external_browsing_enabled else "disabled"),
                kind="browsing",
                payload={"enabled": self._external_browsing_enabled},
            )
        self._publish_status(
            f"Received user request ({len(user_message)} characters)",
            kind="info",
            payload={"user_message": user_message, "metadata": dict(self._runtime_metadata)},
        )

        dag = self._build_workflow()
        dag.set_constant("user_input", user_message)
        dag.run(targets=["execute"])

        if self._gate_blocked:
            response_text = self._last_response_text or "Workflow blocked by a gating task."
            structured_response = self._last_structured
            payload: Mapping[str, Any] | None = None
            message = "Workflow blocked"
            if self._gate_source == "critic" and self._gate_report is not None:
                critic_payload = self._gate_report.to_status_payload()
                payload = {"critic": critic_payload}
                message = "Workflow blocked by critic failures"
                structured_response = None
            elif self._gate_source == "security" and self._security_report is not None:
                payload = {"security": self._security_report.to_dict()}
                message = "Workflow blocked by security findings"
            self._publish_status(message, kind="error", payload=payload)
        else:
            response_text = self._last_response_text
            structured_response = self._last_structured
            self._publish_status("Workflow completed", kind="success")

        new_rounds = self.session.rounds[self._initial_round_index :]
        return ManagerResult(
            response_text=response_text,
            structured_response=structured_response,
            rounds=new_rounds,
            plan=[dict(step) for step in self._active_plan],
            budgets={name: budget.copy() for name, budget in self._budgets.items()},
            status_updates=list(self._status_log),
            evidence=self._evidence_snapshot(),
            evaluations=tuple(self._completed_evaluations),
        )

    # ------------------------------------------------------------------
    # Repository context helpers
    # ------------------------------------------------------------------
    def attach_repo_context(self, repo_context: RepoContextAgent) -> None:
        """Attach or replace the repo context agent used for focused queries."""

        self.repo_context = repo_context

    def attach_test_critic(self, critic: "TestCriticAgent" | None) -> None:
        """Attach or detach the fast-test critic responsible for gating completion."""

        if critic is None:
            if self.test_critic is not None:
                try:
                    self.test_critic.set_status_callback(None)
                except Exception:
                    pass
            self.test_critic = None
            return

        critic.set_status_callback(self._relay_critic_status)
        self.test_critic = critic

    def attach_eval_agent(self, agent: EvalAgent | None) -> None:
        """Attach or detach the evaluation agent used for regression checks."""

        self.eval_agent = agent

    def attach_dependency_agent(self, agent: DependencyBuildAgent | None) -> None:
        """Attach or detach the dependency build helper."""

        self._dependency_agent = agent

    def request_focused_files(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return serialized search results for the given query."""

        if not self.repo_context:
            raise RuntimeError("Repo context agent is not configured")
        results: list[RepoSearchResult] = self.repo_context.focused_files(query, top_k=top_k)
        return [result.to_dict() for result in results]

    def request_symbol_locations(self, symbol: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return symbol match snippets serialized with provenance."""

        if not self.repo_context:
            raise RuntimeError("Repo context agent is not configured")
        matches: list[RepoSymbolResult] = self.repo_context.symbol_search(symbol, top_k=top_k)
        return [match.to_dict() for match in matches]

    def request_diff_bundle(
        self,
        *,
        staged: bool = False,
        include_untracked: bool = False,
    ) -> dict[str, Any]:
        """Return a serialized diff bundle for the current working tree."""

        if not self.repo_context:
            raise RuntimeError("Repo context agent is not configured")
        bundle: DiffBundle = self.repo_context.focused_diffs(
            staged=staged,
            include_untracked=include_untracked,
        )
        return bundle.to_dict()

    # ------------------------------------------------------------------
    # Conversation replay helpers
    # ------------------------------------------------------------------

    def _seed_conversation_history_from_memory(self) -> None:
        if not self.session_id or self.memory_facade is None:
            return

        chat_session = getattr(self.session, "chat_session", None)
        if chat_session is None:
            return
        chat = getattr(chat_session, "chat", None)
        if chat is None or not hasattr(chat, "append"):
            return

        existing = getattr(chat, "messages", None)
        if isinstance(existing, list):
            if any(self._is_conversation_message(message) for message in existing):
                return

        try:
            records = self.memory_facade.iter_session_messages(self.session_id)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "Failed to replay conversation history for session '%s'", self.session_id
            )
            return

        if not records:
            return

        for message in self._records_to_chat_messages(records):
            chat.append(message)

    @staticmethod
    def _is_conversation_message(message: Any) -> bool:
        if not isinstance(message, Mapping):
            return False
        role = message.get("role")
        if not isinstance(role, str):
            return False
        return role.lower() in {"user", "assistant", "tool"}

    def _records_to_chat_messages(self, records: Sequence[MemoryRecord]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for record in records:
            message = self._memory_record_to_chat_message(record)
            if message:
                messages.append(message)
        return messages

    def _memory_record_to_chat_message(self, record: MemoryRecord) -> dict[str, Any]:
        attributes = record.metadata.attributes if isinstance(record.metadata.attributes, Mapping) else {}
        role = self._infer_memory_role(record, attributes)

        message: dict[str, Any] = {"role": role, "content": record.content}

        if role == "tool" and isinstance(attributes, Mapping):
            tool_call_id = attributes.get("tool_call_id")
            if isinstance(tool_call_id, str):
                message["tool_call_id"] = tool_call_id
            tool_name = attributes.get("name")
            if isinstance(tool_name, str):
                message["name"] = tool_name

        return message

    @staticmethod
    def _infer_memory_role(
        record: MemoryRecord, attributes: Mapping[str, Any] | None
    ) -> str:
        if isinstance(attributes, Mapping):
            role = attributes.get("role")
            if isinstance(role, str):
                normalized = role.lower()
                if normalized in {"system", "user", "assistant", "tool"}:
                    return normalized

        tags = {str(tag).lower() for tag in record.metadata.tags}
        for candidate in ("system", "user", "assistant", "tool"):
            if candidate in tags:
                return candidate

        return "assistant"

    # ------------------------------------------------------------------
    # Research helpers
    # ------------------------------------------------------------------
    def attach_research_agent(self, agent: "ResearchAgent" | VariedResearchAgent | None) -> None:
        """Attach or detach the research agent used for external browsing."""

        self.research_agent = agent
        if self._doc_agent is not None:
            self._doc_agent.attach_research_agent(agent)

    def set_external_browsing_default(self, enabled: bool) -> None:
        """Configure the default browsing behaviour for subsequent runs."""

        self._external_browsing_default = bool(enabled)
        self._external_browsing_enabled = bool(enabled)

    def set_external_browsing(self, enabled: bool) -> None:
        """Manually override the browsing toggle for the current workflow."""

        self._external_browsing_enabled = bool(enabled)

    def request_research(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_search_results: int | None = None,
        allow_rewrite: bool | None = None,
        audience: str | None = None,
        force_refresh: bool = False,
        mode: str | None = None,
        profile: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return structured snippets for ``query`` and cache the evidence."""

        if not query or not str(query).strip():
            return []
        if self.research_agent is None:
            raise RuntimeError("Research agent is not configured")
        if not self._external_browsing_enabled:
            raise RuntimeError("External browsing is disabled for this request")

        sanitized_top_k: int | None = None
        if top_k is not None:
            try:
                sanitized_top_k = int(top_k)
            except (TypeError, ValueError):
                sanitized_top_k = 5
            if sanitized_top_k <= 0:
                sanitized_top_k = 5
        sanitized_max_results: int | None = None
        if max_search_results is not None:
            try:
                sanitized_max_results = int(max_search_results)
            except (TypeError, ValueError):
                sanitized_max_results = 20
            if sanitized_max_results <= 0:
                sanitized_max_results = 20
        sanitized_allow_rewrite: bool | None = None
        if allow_rewrite is not None:
            sanitized_allow_rewrite = bool(allow_rewrite)
        sanitized_alpha: float | None = None
        if alpha is not None:
            try:
                sanitized_alpha = float(alpha)
            except (TypeError, ValueError):
                sanitized_alpha = None
            else:
                if sanitized_alpha < 0:
                    sanitized_alpha = 0.0
                elif sanitized_alpha > 1:
                    sanitized_alpha = 1.0
        audience_value = None
        if audience is not None and str(audience).strip():
            audience_value = str(audience).strip()
        agent = self.research_agent
        assert agent is not None  # For type checkers; guarded above.
        search_kwargs: dict[str, Any] = {
            "audience": audience_value,
            "force_refresh": force_refresh,
        }
        if sanitized_top_k is not None:
            search_kwargs["top_k"] = sanitized_top_k
        if sanitized_max_results is not None:
            search_kwargs["max_search_results"] = sanitized_max_results
        if sanitized_allow_rewrite is not None:
            search_kwargs["allow_rewrite"] = sanitized_allow_rewrite
        if sanitized_alpha is not None:
            search_kwargs["alpha"] = sanitized_alpha
        if isinstance(agent, VariedResearchAgent):
            if mode is not None:
                search_kwargs["mode"] = mode
            if profile is not None:
                search_kwargs["profile"] = profile
        result = agent.search(
            query,
            **search_kwargs,
        )
        self._record_research_evidence(audience_value, result)
        return [snippet.to_dict() for snippet in result.snippets]

    def get_research_evidence(self, audience: str | None = None) -> list[dict[str, Any]]:
        """Return cached research evidence scoped to ``audience`` if provided."""

        snapshot = self._evidence_snapshot()
        if audience is None:
            snippets: list[Any] = []
            for items in snapshot.values():
                snippets.extend(items)
        else:
            key = audience.lower()
            snippets = list(snapshot.get(key, ()))
        return [snippet.to_dict() if hasattr(snippet, "to_dict") else snippet for snippet in snippets]

    # ------------------------------------------------------------------
    # MCP server integration helpers
    # ------------------------------------------------------------------
    def refresh_mcp_tools(
        self,
        *,
        registry: "MCPServerRegistry" | None = None,
        tool_registry: ToolRegistry | None = None,
        auto_start: bool = False,
    ) -> list[ToolSpec]:
        """Refresh MCP tool metadata using live descriptors from ``registry``."""

        target_registry: "MCPServerRegistry" | None = registry or self.mcp_registry
        if target_registry is None:
            raise ValueError("No MCP server registry is available for refresh")

        active_tool_registry = tool_registry or self.tool_registry
        if active_tool_registry is None:
            raise ValueError("The manager session is not associated with a tool registry")

        server_specs = target_registry.build_specs(auto_start=auto_start)
        from mcp_tooling import register_mcp_servers

        updated_specs = register_mcp_servers(active_tool_registry, server_specs, replace=True)
        return self.session.replace_mcp_tools(updated_specs)

    def _record_research_evidence(
        self,
        audience: str | None,
        result: "ResearchResult",
    ) -> None:
        key = (audience or "general").lower()
        bucket = self._shared_evidence.setdefault(key, [])
        seen = {snippet.url for snippet in bucket}
        for snippet in result.snippets:
            if snippet.url in seen:
                continue
            bucket.append(snippet)
            seen.add(snippet.url)

    def _build_evidence_payload(self) -> dict[str, list[dict[str, Any]]]:
        snapshot = self._evidence_snapshot()
        return {
            key: [snippet.to_dict() if hasattr(snippet, "to_dict") else snippet for snippet in snippets]
            for key, snippets in snapshot.items()
        }

    def _evidence_snapshot(self) -> dict[str, tuple["ResearchSnippet", ...]]:
        return {key: tuple(values) for key, values in self._shared_evidence.items()}

    @staticmethod
    def _coerce_queries(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        queries: list[str] = []
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                queries.append(text)
        return queries

    @staticmethod
    def _ensure_sequence(value: Any | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        normalized: list[str] = []
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(normalized)

    # ------------------------------------------------------------------
    # DAG lifecycle
    # ------------------------------------------------------------------
    def _build_workflow(self) -> DAG:
        dag = DAG()
        dag.add_node("user_input", value="", is_constant=True)
        dag.add_node(
            "plan",
            func=self._dag_plan,
            deps=["user_input"],
            retries=self.plan_retries,
            metadata={"stage": "planning"},
        )
        dag.add_node(
            "prepare",
            func=self._dag_prepare,
            deps=["plan"],
            metadata={"stage": "budgeting"},
        )
        dag.add_node(
            "execute",
            func=self._dag_execute,
            deps=["prepare", "user_input"],
            retries=self.task_retry_limit,
            metadata={"stage": "execution"},
        )
        return dag

    def _dag_plan(self, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
        user_message = str(inputs.get("user_input", ""))
        planned = self._call_plan_builder(user_message)
        tasks: list[dict[str, Any]] = []
        for index, item in enumerate(planned, start=1):
            payload = dict(item)
            payload.setdefault("name", f"task-{index}")
            payload.setdefault("prompt", user_message)
            tasks.append(payload)
        self._active_plan = [dict(task) for task in tasks]
        self._publish_status(
            f"Planner produced {len(tasks)} task(s)",
            kind="planning",
            payload={"tasks": [task["name"] for task in tasks]},
        )
        return tasks

    def _dag_prepare(self, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
        tasks: Iterable[Mapping[str, Any]] = inputs.get("plan", [])
        self._allocate_budgets(tasks)
        return [dict(task) for task in tasks]

    def _dag_execute(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        tasks: Sequence[Mapping[str, Any]] = inputs.get("prepare", [])
        user_message = str(inputs.get("user_input", ""))
        response_text = ""
        structured: StructuredResponse | None = None
        for task in tasks:
            response_text, structured = self._execute_task(task, user_message=user_message)
            if self._gate_blocked:
                break

        if not self._gate_blocked:
            response_text, structured = self._synthesise_task_outputs(
                default_text=response_text,
                default_structured=structured,
            )

        if self.test_critic is not None and not self._gate_blocked:
            report = self.test_critic.run_and_report()
            self._gate_report = report
            if not self.test_critic.has_status_callback:
                for event in report.status_events:
                    self._relay_critic_status(event)
            summary_payload = {"critic": report.to_status_payload()}
            if report.is_blocking:
                self._gate_blocked = True
                self._last_response_text = report.build_block_message()
                self._last_structured = None
                self._gate_source = self._gate_source or "critic"
                response_text = self._last_response_text
                structured = None
                self._publish_status(
                    "Test critic reported blocking failures",
                    kind="critic_failure",
                    payload=summary_payload,
                )
            else:
                self._publish_status(
                    "Test critic checks passed",
                    kind="critic_success",
                    payload=summary_payload,
                )

        return {"response_text": response_text, "structured_response": structured}

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    _SPECIALIST_BLUEPRINTS: tuple[Mapping[str, Any], ...] = (
        {
            "name": "dependency-resolution",
            "kind": "dependency",
            "agent": "dependency",
            "description": "Resolve or install project dependencies",
            "keywords": ("dependency", "dependencies", "install", "package", "pip", "npm"),
            "budget": {"limit": 1.0, "unit": "rounds"},
            "research": {"required": False},
        },
        {
            "name": "database-migration",
            "kind": "db_migration",
            "agent": "db_migration",
            "description": "Prepare and run database migrations",
            "keywords": ("migration", "migrate", "schema"),
            "budget": {"limit": 1.0, "unit": "rounds"},
            "research": {"required": False},
        },
        {
            "name": "security-audit",
            "kind": "security",
            "agent": "security",
            "description": "Execute security scans and report findings",
            "keywords": ("security", "vulnerability", "scan", "audit"),
            "budget": {"limit": 1.0, "unit": "rounds"},
            "research": {"required": False},
        },
        {
            "name": "documentation-updates",
            "kind": "documentation",
            "agent": "documentation",
            "description": "Draft documentation or release notes",
            "keywords": ("document", "docs", "changelog", "readme", "release notes"),
            "budget": {"limit": 2.0, "unit": "rounds"},
            "research": {"required": True, "audience": "docs"},
        },
        {
            "name": "research-discovery",
            "kind": "research_discovery",
            "agent": "research",
            "description": "Explore external sources using tuned research depth modes",
            "keywords": ("research", "discover", "investigate", "analysis", "insight"),
            "budget": {"limit": 1.5, "unit": "rounds"},
            "research": {
                "required": True,
                "mode": "balanced",
                "profiles": (
                    "skim",
                    "survey",
                    "balanced",
                    "insight",
                    "investigative",
                    "deep_dive",
                    "forensic",
                ),
            },
            "metadata": {
                "research_modes": ("light", "balanced", "deep"),
                "research_profiles": (
                    "skim",
                    "survey",
                    "balanced",
                    "insight",
                    "investigative",
                    "deep_dive",
                    "forensic",
                ),
            },
        },
        {
            "name": "ci-integration",
            "kind": "integrations",
            "agent": "integrations",
            "description": "Update CI/CD integrations",
            "keywords": ("pipeline", "ci", "integration", "deploy", "release"),
            "budget": {"limit": 1.0, "unit": "rounds"},
            "research": {"required": False},
        },
        {
            "name": "regression-evaluation",
            "kind": "eval",
            "agent": "evaluation",
            "description": "Run regression or evaluation suites",
            "keywords": ("evaluate", "evaluation", "benchmark", "regression"),
            "budget": {"limit": 1.0, "unit": "rounds"},
            "research": {"required": False},
        },
    )

    @classmethod
    def _resolve_specialist_blueprints(
        cls, override: Sequence[Mapping[str, Any]] | None
    ) -> tuple[Mapping[str, Any], ...]:
        if override is None:
            return tuple(cls._SPECIALIST_BLUEPRINTS)
        resolved: list[Mapping[str, Any]] = []
        for blueprint in override:
            if not isinstance(blueprint, Mapping):
                continue
            resolved.append(cls._normalise_blueprint(blueprint))
        return tuple(resolved) or tuple(cls._SPECIALIST_BLUEPRINTS)

    @classmethod
    def _normalise_blueprint(cls, blueprint: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(blueprint)
        name = str(payload.get("name") or payload.get("kind") or "task").strip()
        kind = str(payload.get("kind") or "").strip()
        agent = str(payload.get("agent") or kind or "session").strip()
        if not name:
            name = kind or "task"
        if not agent:
            agent = kind or "session"
        description = str(payload.get("description") or "").strip()
        keywords = cls._ensure_sequence(payload.get("keywords"))
        budget = cls._normalise_blueprint_budget(payload.get("budget"))
        research = cls._normalise_blueprint_research(payload.get("research"))
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_payload = dict(metadata)
        else:
            metadata_payload = None

        normalised = dict(payload)
        normalised.update({"name": name, "kind": kind, "agent": agent})
        normalised["description"] = description
        if keywords is not None:
            normalised["keywords"] = keywords
        if budget:
            normalised["budget"] = budget
        elif "budget" in normalised:
            normalised["budget"] = {}
        if research:
            normalised["research"] = research
        elif "research" in normalised:
            normalised["research"] = {}
        if metadata_payload is not None:
            normalised["metadata"] = metadata_payload
        elif "metadata" in normalised:
            normalised.pop("metadata", None)
        return normalised

    @staticmethod
    def _normalise_blueprint_budget(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        budget: dict[str, Any] = {}
        if "limit" in payload:
            try:
                limit_value = float(payload["limit"])
            except (TypeError, ValueError):
                limit_value = None
            if limit_value is not None:
                budget["limit"] = limit_value
        if "unit" in payload and payload["unit"] is not None:
            unit_value = str(payload["unit"]).strip()
            if unit_value:
                budget["unit"] = unit_value
        return budget

    @staticmethod
    def _normalise_blueprint_research(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        research: dict[str, Any] = {}
        if "required" in payload:
            required_flag = payload.get("required")
            if isinstance(required_flag, bool):
                research["required"] = required_flag
            else:
                text = str(required_flag).strip().lower()
                if text in {"1", "true", "yes", "on"}:
                    research["required"] = True
                elif text in {"0", "false", "no", "off"}:
                    research["required"] = False
        if "audience" in payload and payload["audience"] is not None:
            audience_value = str(payload["audience"]).strip()
            if audience_value:
                research["audience"] = audience_value
        if "mode" in payload and payload["mode"] is not None:
            mode_value = str(payload["mode"]).strip()
            if mode_value:
                research["mode"] = mode_value
        if "profile" in payload and payload["profile"] is not None:
            research["profile"] = payload["profile"]
        if "profiles" in payload and payload["profiles"] is not None:
            profiles_value = payload["profiles"]
            if isinstance(profiles_value, (list, tuple, set)):
                normalised = tuple(
                    str(item).strip() for item in profiles_value if isinstance(item, (str, bytes)) and str(item).strip()
                )
            elif isinstance(profiles_value, str):
                normalised = tuple(part.strip() for part in profiles_value.split(",") if part.strip())
            else:
                normalised = ()
            if normalised:
                research["profiles"] = normalised
        return research

    def _default_plan_builder(
        self,
        user_message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        cleaned = user_message.strip()
        lowered = cleaned.lower()
        metadata = metadata or {}
        requested = self._ensure_sequence(metadata.get("requested_tasks"))
        tasks: list[dict[str, Any]] = []

        def _matches(blueprint: Mapping[str, Any]) -> bool:
            if requested and blueprint["kind"] not in requested:
                return False
            for keyword in blueprint.get("keywords", ()):  # pragma: no cover - defensive
                if keyword and keyword in lowered:
                    return True
            return bool(requested)

        for blueprint in self._specialist_blueprints:
            if _matches(blueprint):
                tasks.append(self._build_task_from_blueprint(blueprint, user_message))

        if not tasks:
            description = "Respond to the user's request"
            if not cleaned:
                description = "Prompt the user for additional details"
            tasks.append(
                {
                    "name": "task-1",
                    "description": description,
                    "prompt": user_message,
                    "budget": {"limit": 1.0, "unit": "rounds"},
                    "metadata": {"kind": "general", "agent": "session"},
                    "research": {"required": False},
                }
            )

        return tasks

    def _build_task_from_blueprint(
        self,
        blueprint: Mapping[str, Any],
        user_message: str,
    ) -> dict[str, Any]:
        name = str(blueprint.get("name") or blueprint.get("kind") or "task")
        kind_value = str(blueprint.get("kind") or "").strip()
        agent_value = str(blueprint.get("agent") or kind_value or "session").strip()
        budget_payload = dict(blueprint.get("budget", {}))
        if "limit" in budget_payload:
            try:
                budget_payload["limit"] = float(budget_payload["limit"])
            except (TypeError, ValueError):
                budget_payload.pop("limit", None)
        if "unit" in budget_payload and budget_payload["unit"] is not None:
            budget_payload["unit"] = str(budget_payload["unit"]).strip()
            if not budget_payload["unit"]:
                budget_payload.pop("unit", None)
        research_payload = dict(blueprint.get("research", {}))
        if "required" in research_payload:
            flag = research_payload["required"]
            if isinstance(flag, bool):
                research_payload["required"] = flag
            else:
                text = str(flag).strip().lower()
                if text in {"1", "true", "yes", "on"}:
                    research_payload["required"] = True
                elif text in {"0", "false", "no", "off"}:
                    research_payload["required"] = False
                else:
                    research_payload.pop("required", None)
        if "audience" in research_payload and research_payload["audience"] is not None:
            research_payload["audience"] = str(research_payload["audience"]).strip()
            if not research_payload["audience"]:
                research_payload.pop("audience", None)
        metadata_payload: dict[str, Any] = {}
        if kind_value:
            metadata_payload["kind"] = kind_value
        if agent_value:
            metadata_payload["agent"] = agent_value
        if research_payload:
            metadata_payload["research"] = dict(research_payload)
        extra_metadata = blueprint.get("metadata")
        if isinstance(extra_metadata, Mapping):
            metadata_payload.update(extra_metadata)
        return {
            "name": name,
            "description": blueprint.get("description", ""),
            "prompt": user_message,
            "kind": kind_value or None,
            "budget": budget_payload,
            "metadata": metadata_payload,
            "research": research_payload,
        }

    def _call_plan_builder(self, user_message: str) -> Sequence[Mapping[str, Any]]:
        builder = self._plan_builder
        runtime_metadata = dict(self._runtime_metadata)
        try:
            return builder(user_message, metadata=runtime_metadata)
        except TypeError:
            return builder(user_message)

    def _allocate_budgets(self, tasks: Iterable[Mapping[str, Any]]) -> None:
        self._budgets.clear()
        for index, task in enumerate(tasks, start=1):
            name = str(task.get("name") or f"task-{index}")
            budget_info = task.get("budget") or {}
            limit_value = budget_info.get("limit", 1.0)
            unit = budget_info.get("unit", "rounds")
            limit: float | None
            if limit_value is None:
                limit = None
            else:
                try:
                    limit = float(limit_value)
                except (TypeError, ValueError):
                    limit = 1.0
            budget = TaskBudget(name=name, limit=limit, unit=str(unit))
            self._budgets[name] = budget
            self._publish_status(
                f"Allocated budget for {name}",
                kind="budget",
                task=name,
                payload={"budget": budget.as_dict()},
            )

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _synthesise_task_outputs(
        self,
        *,
        default_text: str,
        default_structured: StructuredResponse | None,
    ) -> tuple[str, StructuredResponse | None]:
        if not self._task_outputs:
            self._last_response_text = default_text
            self._last_structured = default_structured
            return default_text, default_structured

        summary_lines: list[str] = []
        tasks_payload: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for name, record in self._task_outputs.items():
            text = str(record.get("text") or "").strip()
            if text:
                summary_lines.append(f"{name}: {text}")
            kind = record.get("kind") or "general"
            entry: dict[str, Any] = {"kind": kind, "text": text}
            structured = record.get("structured")
            if structured is not None:
                structured_payload: dict[str, Any] = {
                    "content": structured.content,
                    "structured": bool(structured.structured),
                }
                if structured.parsed is not None:
                    structured_payload["parsed"] = structured.parsed
                if structured.schema is not None:
                    structured_payload["schema"] = structured.schema
                entry["structured_output"] = structured_payload
            tasks_payload[name] = entry

        aggregated_text = "\n\n".join(summary_lines) if summary_lines else default_text
        parsed_payload = {"tasks": tasks_payload}
        aggregated = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": aggregated_text,
                            "parsed": parsed_payload,
                        }
                    }
                ]
            },
            content=aggregated_text,
            parsed=parsed_payload,
            schema={"name": "ManagerTaskOutputs"},
            structured=True,
        )

        self._last_response_text = aggregated_text
        self._last_structured = aggregated
        return aggregated_text, aggregated

    def _execute_task(
        self,
        task: Mapping[str, Any],
        *,
        user_message: str,
    ) -> tuple[str, StructuredResponse | None]:
        name = str(task.get("name") or "task")
        self._current_task = name
        prompt = str(task.get("prompt", user_message))
        budget = self._budgets.get(name)
        metadata: MutableMapping[str, Any] = dict(task.get("metadata", {}))
        metadata.setdefault("task", name)
        if budget:
            metadata["budget"] = budget.as_dict()

        task_kind_raw = metadata.get("kind") or task.get("kind")
        task_kind = str(task_kind_raw).lower() if isinstance(task_kind_raw, str) else None
        if task_kind == "dependency":
            metadata.pop("kind", None)
            return self._run_dependency_task(name, task, metadata, budget=budget)
        if task_kind == "db_migration":
            metadata.pop("kind", None)
            return self._run_db_migration_task(name, task, metadata, budget=budget)
        if task_kind == "eval":
            metadata.pop("kind", None)
            return self._run_eval_task(name, task, metadata, budget=budget)
        if task_kind == "security":
            metadata.pop("kind", None)
            return self._run_security_task(name, task, metadata, budget=budget)
        if task_kind == "documentation":
            metadata.pop("kind", None)
            return self._run_documentation_task(name, task, metadata, budget=budget)
        if task_kind == "integrations":
            metadata.pop("kind", None)
            return self._run_integrations_task(name, task, metadata, budget=budget)

        research_spec = task.get("research")
        if not isinstance(research_spec, Mapping):
            research_spec = metadata.get("research") if isinstance(metadata, Mapping) else None

        audience_hint = None
        raw_queries: Any = None
        mode_hint: Any | None = None
        profile_hint: Any | None = None
        alpha_hint: Any | None = None
        max_results_hint: Any | None = None
        top_k_hint: Any | None = None
        allow_rewrite_hint: Any | None = None

        if isinstance(research_spec, Mapping):
            raw_queries = research_spec.get("queries")
            if raw_queries is None and research_spec.get("query") is not None:
                raw_queries = research_spec.get("query")
            audience_hint = research_spec.get("audience")
            required_research = bool(research_spec.get("required"))
            mode_hint = research_spec.get("mode")
            profile_hint = research_spec.get("profile")
            alpha_hint = research_spec.get("alpha")
            max_results_hint = research_spec.get("max_search_results")
            top_k_hint = research_spec.get("top_k")
            allow_rewrite_hint = research_spec.get("allow_rewrite")
        else:
            raw_queries = task.get("research_queries")
            audience_hint = task.get("research_audience")
            required_research = False
            mode_hint = task.get("research_mode")
            profile_hint = task.get("research_profile")
            alpha_hint = task.get("research_alpha")
            max_results_hint = task.get("research_max_results")
            top_k_hint = task.get("research_top_k")
            allow_rewrite_hint = task.get("research_allow_rewrite")
        queries = self._coerce_queries(raw_queries)
        if required_research and not queries and prompt.strip():
            queries = [prompt.strip()]
        audience_value = str(audience_hint).strip() if audience_hint is not None else None
        if queries:
            top_k = None
            if top_k_hint is not None:
                try:
                    top_k = int(top_k_hint)
                except (TypeError, ValueError):
                    top_k = None
            elif task.get("research_top_k") is not None:
                try:
                    top_k = int(task.get("research_top_k"))
                except (TypeError, ValueError):
                    top_k = None
            if isinstance(top_k, int) and top_k <= 0:
                top_k = None
            if max_results_hint is None and task.get("research_max_results") is not None:
                max_results_hint = task.get("research_max_results")
            if allow_rewrite_hint is None and task.get("research_allow_rewrite") is not None:
                allow_rewrite_hint = task.get("research_allow_rewrite")
            if mode_hint is None and task.get("research_mode") is not None:
                mode_hint = task.get("research_mode")
            if profile_hint is None and task.get("research_profile") is not None:
                profile_hint = task.get("research_profile")
            if alpha_hint is None and task.get("research_alpha") is not None:
                alpha_hint = task.get("research_alpha")
            metadata["requested_research"] = list(queries)
            for query_text in queries:
                try:
                    self.request_research(
                        query_text,
                        top_k=top_k,
                        max_search_results=max_results_hint,
                        allow_rewrite=allow_rewrite_hint,
                        audience=audience_value,
                        mode=mode_hint,
                        profile=profile_hint,
                        alpha=alpha_hint,
                    )
                except Exception as exc:
                    self._publish_status(
                        f"Research lookup failed for {name}: {exc}",
                        kind="warning",
                        task=name,
                        payload={"query": query_text, "error": str(exc)},
                    )

        metadata["external_browsing_enabled"] = self._external_browsing_enabled
        if self._shared_evidence:
            metadata["external_evidence"] = self._build_evidence_payload()

        act_kwargs: dict[str, Any] = dict(task.get("act_kwargs", {}))
        try:
            text, structured = self.session.act(
                prompt,
                tools=task.get("tools"),
                tool_names=task.get("tool_names"),
                config=task.get("config"),
                callbacks=task.get("callbacks"),
                metadata=metadata,
                **act_kwargs,
            )
        except Exception as exc:  # pragma: no cover - safeguard for downstream errors
            return self._handle_task_failure(name, exc)
        else:
            self._mark_task_complete(name)
            self._last_response_text = text
            self._last_structured = structured
            self._register_task_output(name, text, structured, kind=task_kind or "general")
            return text, structured
        finally:
            self._current_task = None

    def _register_task_output(
        self,
        task_name: str,
        text: str,
        structured: StructuredResponse | None,
        *,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_kind = kind or "general"
        payload: dict[str, Any] = {"kind": normalized_kind, "text": text}
        if structured is not None:
            if structured.parsed is not None:
                payload["parsed"] = structured.parsed
            if structured.schema is not None:
                payload["schema"] = structured.schema
        if metadata:
            payload["metadata"] = dict(metadata)
        self._task_outputs[task_name] = {
            "text": text,
            "structured": structured,
            "kind": normalized_kind,
        }
        self._publish_status(
            f"Recorded output for {task_name}",
            kind="task_output",
            task=task_name,
            payload=payload,
        )

    def _ingest_task_evidence(
        self,
        task_name: str,
        evidence: Iterable[Any] | None,
    ) -> None:
        if not evidence:
            return
        bucket = self._shared_evidence.setdefault(task_name, [])
        seen = {getattr(snippet, "url", None) for snippet in bucket if getattr(snippet, "url", None)}
        for snippet in evidence:
            url = getattr(snippet, "url", None)
            if url and url in seen:
                continue
            bucket.append(snippet)
            if url:
                seen.add(url)

    def _run_dependency_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        agent = self._ensure_dependency_agent()
        spec: dict[str, Any] = {}
        dependency_spec = task.get("dependency")
        if isinstance(dependency_spec, Mapping):
            spec.update(dependency_spec)
        for key in ("manager", "command", "packages", "lockfiles", "workdir", "env"):
            if key in task and key not in spec:
                spec[key] = task[key]
            if key in metadata and key not in spec:
                spec[key] = metadata[key]

        manager_value = spec.get("manager")
        if manager_value is None:
            return self._handle_task_failure(
                name,
                ValueError("Dependency task missing 'manager' value"),
            )

        def status_callback(message: str, kind: str, payload: Mapping[str, Any] | None) -> None:
            self._publish_status(message, kind=kind, task=name, payload=payload)

        try:
            resolution = agent.run_task(
                manager=str(manager_value),
                command=spec.get("command"),
                packages=spec.get("packages"),
                lockfiles=spec.get("lockfiles"),
                workdir=spec.get("workdir"),
                env=spec.get("env"),
                status_callback=status_callback,
            )
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        summary_text = agent.format_resolution(resolution)
        structured_payload = resolution.to_dict()
        structured = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": summary_text,
                            "parsed": structured_payload,
                        }
                    }
                ]
            },
            content=summary_text,
            parsed=structured_payload,
            schema={"name": "DependencyResolution"},
            structured=True,
        )
        if budget is not None:
            budget.consume()
        self._publish_status(
            f"Dependency task '{name}' completed",
            kind="dependency_summary",
            task=name,
            payload=structured_payload,
        )
        self._mark_task_complete(name)
        self._last_response_text = summary_text
        self._last_structured = structured
        self._register_task_output(name, summary_text, structured, kind="dependency")
        return summary_text, structured

    def _run_db_migration_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        try:
            agent = self._ensure_db_migration_agent()
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        spec: dict[str, Any] = {}
        migration_spec = task.get("db_migration") or task.get("migration")
        if isinstance(migration_spec, Mapping):
            spec.update(migration_spec)
        for key in (
            "framework",
            "migrations_dir",
            "workdir",
            "env",
            "apply",
            "migration_name",
            "extra_args",
            "ephemeral",
            "ephemeral_db",
        ):
            if key in task and key not in spec:
                spec[key] = task[key]
            if key in metadata and key not in spec:
                spec[key] = metadata[key]

        framework_value = spec.get("framework")
        if framework_value is None:
            return self._handle_task_failure(
                name,
                ValueError("DB migration task missing 'framework' value"),
            )

        try:
            env_payload = spec.get("env") if isinstance(spec.get("env"), Mapping) else None
            plan = agent.plan_migration(
                str(framework_value),
                migrations_dir=spec.get("migrations_dir"),
                workdir=spec.get("workdir"),
                env=env_payload,
            )
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        self._publish_status(
            f"Prepared migration plan for framework '{plan.framework}'",
            kind="db_migration_plan",
            task=name,
            payload={"plan": plan.to_dict()},
        )

        migration_name = spec.get("migration_name") or name
        apply_flag = bool(spec.get("apply", False))
        extra_args = spec.get("extra_args")
        if isinstance(extra_args, (str, bytes)):
            extra_args = [extra_args]
        elif extra_args is not None and not isinstance(extra_args, Sequence):
            extra_args = [extra_args]
        if extra_args is not None:
            extra_args = [str(arg) for arg in extra_args]
        ephemeral_key = spec.get("ephemeral") or spec.get("ephemeral_db")

        try:
            result = agent.run_migration(
                plan,
                migration_name=str(migration_name),
                apply=apply_flag,
                extra_args=extra_args,
                ephemeral=ephemeral_key,
            )
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        if budget is not None:
            budget.consume()

        payload = result.to_dict()
        self._publish_status(
            f"Database migration '{name}' completed",
            kind="db_migration_result",
            task=name,
            payload=payload,
        )

        summary_text = agent.format_result(result)
        structured = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": summary_text,
                            "parsed": payload,
                        }
                    }
                ]
            },
            content=summary_text,
            parsed=payload,
            schema={"name": "MigrationResult"},
            structured=True,
        )
        self._mark_task_complete(name)
        self._last_response_text = summary_text
        self._last_structured = structured
        self._register_task_output(name, summary_text, structured, kind="db_migration")
        return summary_text, structured

    def _run_eval_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        try:
            agent = self._ensure_eval_agent()
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        spec_payload: dict[str, Any]
        evaluation_spec = task.get("evaluation") or task.get("eval")
        if isinstance(evaluation_spec, Mapping):
            spec_payload = dict(evaluation_spec)
        else:
            spec_payload = {
                key: task[key]
                for key in ("comparisons", "pairs", "cases", "gate", "scoring", "metadata")
                if key in task
            }
        spec_payload.setdefault("name", name)

        metadata_payload = dict(metadata)
        try:
            summary = agent.run(spec_payload, metadata=metadata_payload)
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        self._last_eval_summary = summary
        self._completed_evaluations.append(summary)

        structured = agent.to_structured_response(summary)
        text = structured.content

        status_kind = "evaluation_success"
        if summary.is_blocking:
            status_kind = "evaluation_failure"
        self._publish_status(
            f"Evaluation '{name}' completed",
            kind=status_kind,
            task=name,
            payload={"summary": summary.to_dict()},
        )

        if budget is not None:
            budget.consume()
        self._mark_task_complete(name)

        if summary.is_blocking:
            self._gate_blocked = True
            self._last_response_text = text
            self._last_structured = structured
        else:
            self._last_response_text = text
            self._last_structured = structured
        self._register_task_output(name, text, structured, kind="eval")
        return text, structured

    def _run_security_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        try:
            agent = self._ensure_security_agent()
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        spec: dict[str, Any] = {}
        security_spec = task.get("security")
        if isinstance(security_spec, Mapping):
            spec.update(security_spec)
        for key in (
            "toolchains",
            "severity_threshold",
            "threshold",
            "stop_on_failure",
            "workdir",
            "env",
        ):
            if key in task and key not in spec:
                spec[key] = task[key]
            if key in metadata and key not in spec:
                spec[key] = metadata[key]

        sequence_value = spec.get("toolchains")
        if isinstance(sequence_value, (str, bytes)):
            requested_sequence = [str(sequence_value)]
        elif isinstance(sequence_value, Sequence):
            requested_sequence = [str(item) for item in sequence_value]
        else:
            requested_sequence = None

        severity_threshold = spec.get("severity_threshold") or spec.get("threshold")
        stop_on_failure = bool(spec.get("stop_on_failure", True))
        workdir_value = spec.get("workdir")
        env_value = spec.get("env")
        env_spec = env_value if isinstance(env_value, Mapping) else None

        start_payload: dict[str, Any] = {
            "toolchains": list(requested_sequence or agent.DEFAULT_SEQUENCE),
            "severity_threshold": severity_threshold,
            "stop_on_failure": stop_on_failure,
        }
        self._publish_status(
            f"Running security scans for '{name}'",
            kind="security_start",
            task=name,
            payload=start_payload,
        )

        try:
            result = agent.run_scans(
                toolchains=requested_sequence,
                severity_threshold=severity_threshold,
                stop_on_failure=stop_on_failure,
                workdir=workdir_value,
                env=env_spec,
            )
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        summary_text = result.summary or agent.format_result(result)
        structured_payload = result.to_dict()
        structured = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": summary_text,
                            "parsed": structured_payload,
                        }
                    }
                ]
            },
            content=summary_text,
            parsed=structured_payload,
            schema={"name": "SecurityScanResult"},
            structured=True,
        )

        if budget is not None:
            budget.consume()

        self._security_report = result
        status_kind = "security_blocked" if result.blocked else "security_summary"
        message = (
            f"Security scans for '{name}' blocked the workflow"
            if result.blocked
            else f"Security scans for '{name}' completed"
        )
        self._publish_status(message, kind=status_kind, task=name, payload=structured_payload)

        self._last_response_text = summary_text
        self._last_structured = structured
        if result.blocked:
            self._gate_blocked = True
            self._gate_source = "security"
        self._mark_task_complete(name)
        self._register_task_output(name, summary_text, structured, kind="security")
        return summary_text, structured

    def _run_documentation_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        try:
            agent = self._ensure_doc_agent()
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        spec: dict[str, Any] = {}
        doc_spec = task.get("documentation") or task.get("docs")
        if isinstance(doc_spec, Mapping):
            spec.update(doc_spec)

        keys_to_copy = (
            "highlights",
            "version",
            "walkthrough",
            "walkthrough_topics",
            "summary_paths",
            "summary_queries",
            "research_queries",
            "research",
            "include_readme",
            "include_changelog",
            "top_k",
            "metadata",
        )
        for key in keys_to_copy:
            if key in task and key not in spec:
                spec[key] = task[key]
            if key in metadata and key not in spec:
                spec[key] = metadata[key]

        highlights_value = spec.get("highlights")
        version_value = spec.get("version")
        walkthrough_value = spec.get("walkthrough_topics", spec.get("walkthrough"))
        summary_paths_value = spec.get("summary_paths")
        summary_queries_value = spec.get("summary_queries", spec.get("queries"))

        research_value = spec.get("research_queries", spec.get("research"))
        if research_value is None:
            research_spec = task.get("research") or metadata.get("research")
            if isinstance(research_spec, Mapping):
                research_value = research_spec.get("queries") or research_spec.get("query")
            elif research_spec is not None:
                research_value = research_spec

        metadata_value = spec.get("metadata") if isinstance(spec.get("metadata"), Mapping) else None

        include_readme = bool(spec.get("include_readme", True))
        include_changelog = bool(spec.get("include_changelog", True))
        top_k_value = spec.get("top_k", 5)
        try:
            top_k = int(top_k_value)
        except (TypeError, ValueError):
            top_k = 5
        if top_k <= 0:
            top_k = 5

        try:
            result = agent.draft_updates(
                highlights=self._ensure_sequence(highlights_value),
                version=str(version_value) if version_value is not None else None,
                walkthrough_topics=self._ensure_sequence(walkthrough_value),
                summary_paths=self._ensure_sequence(summary_paths_value),
                summary_queries=self._ensure_sequence(summary_queries_value),
                research_queries=self._ensure_sequence(research_value),
                include_readme=include_readme,
                include_changelog=include_changelog,
                top_k=top_k,
                metadata=metadata_value,
            )
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        payload = result.to_dict()
        structured = agent.to_structured_response(result)
        text = structured.content

        if budget is not None:
            budget.consume()
        self._publish_status(
            f"Documentation draft '{name}' prepared",
            kind="documentation_summary",
            task=name,
            payload=payload,
        )
        self._mark_task_complete(name)
        self._last_response_text = text
        self._last_structured = structured
        self._ingest_task_evidence(name, getattr(result, "evidence", None))
        self._register_task_output(name, text, structured, kind="documentation")
        return text, structured

    def _run_integrations_task(
        self,
        name: str,
        task: Mapping[str, Any],
        metadata: MutableMapping[str, Any],
        *,
        budget: TaskBudget | None,
    ) -> tuple[str, StructuredResponse | None]:
        try:
            agent = self._ensure_integrations_agent()
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        spec: dict[str, Any] = {}
        integrations_spec = task.get("integrations") or task.get("integration")
        if isinstance(integrations_spec, Mapping):
            spec.update(integrations_spec)
        metadata_spec = metadata.get("integrations") if isinstance(metadata, Mapping) else None
        if isinstance(metadata_spec, Mapping):
            spec.update(metadata_spec)
        for key in ("detect", "pipeline", "build", "release"):
            if key in task and key not in spec:
                spec[key] = task[key]
            if key in metadata and key not in spec:
                spec[key] = metadata[key]

        outputs: dict[str, Any] = {}
        summary_lines: list[str] = []

        try:
            detect_flag = bool(spec.get("detect")) or bool(spec.get("discover"))
            if detect_flag:
                detected = agent.detect_pipelines()
                outputs["detected"] = detected
                if detected:
                    provider_list = ", ".join(sorted(detected))
                    message = f"Detected CI providers: {provider_list}"
                else:
                    message = "No CI providers detected"
                self._publish_status(
                    message,
                    kind="integrations_detect",
                    task=name,
                    payload={"providers": detected},
                )
                summary_lines.append(message)

            pipeline_spec = spec.get("pipeline")
            if isinstance(pipeline_spec, Mapping):
                template_value = pipeline_spec.get("template")
                path_value = pipeline_spec.get("path")
                if not path_value:
                    raise ValueError("Integrations pipeline spec requires a 'path'")
                if template_value is None:
                    raise ValueError("Integrations pipeline spec requires a 'template'")
                variables_value = pipeline_spec.get("variables") or {}
                if not isinstance(variables_value, Mapping):
                    raise ValueError("Integrations pipeline 'variables' must be a mapping")
                plan = CIJobPlan(
                    provider=str(pipeline_spec.get("provider", "github_actions")),
                    name=str(pipeline_spec.get("name", name)),
                    path=str(path_value),
                    template=str(template_value),
                    variables=dict(variables_value),
                )
                apply_flag = bool(pipeline_spec.get("apply", True))
                pipeline_result = agent.orchestrate_pipeline(plan, apply=apply_flag)
                pipeline_payload = pipeline_result.to_dict()
                outputs["pipeline"] = pipeline_payload
                message = f"Pipeline '{plan.name}' processed for {plan.provider}"
                self._publish_status(
                    message,
                    kind="integrations_pipeline",
                    task=name,
                    payload=pipeline_payload,
                )
                summary_lines.append(message)

            build_spec = spec.get("build")
            if isinstance(build_spec, Mapping):
                image_value = build_spec.get("image")
                if not image_value:
                    raise ValueError("Integrations build spec requires an 'image'")
                extra_args = build_spec.get("extra_args")
                if isinstance(extra_args, (str, bytes)):
                    extra_args = [extra_args]
                report = agent.run_container_build(
                    str(image_value),
                    context=str(build_spec.get("context", ".")),
                    dockerfile=str(build_spec.get("dockerfile")) if build_spec.get("dockerfile") else None,
                    build_args=build_spec.get("build_args") if isinstance(build_spec.get("build_args"), Mapping) else None,
                    extra_args=list(extra_args) if extra_args is not None else None,
                    workdir=str(build_spec.get("workdir")) if build_spec.get("workdir") else None,
                    env=build_spec.get("env") if isinstance(build_spec.get("env"), Mapping) else None,
                )
                build_payload = report.to_dict()
                outputs["build"] = build_payload
                message = f"Container image '{image_value}' build triggered"
                self._publish_status(
                    message,
                    kind="integrations_build",
                    task=name,
                    payload=build_payload,
                )
                summary_lines.append(message)

            release_spec = spec.get("release")
            if release_spec is None and "version" in spec:
                release_spec = spec
            if isinstance(release_spec, Mapping) and release_spec.get("version"):
                metadata_value = release_spec.get("metadata")
                if metadata_value is not None and not isinstance(metadata_value, Mapping):
                    raise ValueError("Integrations release 'metadata' must be a mapping if provided")
                release = agent.prepare_release_metadata(
                    str(release_spec.get("version")),
                    tag=release_spec.get("tag"),
                    notes=release_spec.get("notes"),
                    artifacts=release_spec.get("artifacts"),
                    commit=release_spec.get("commit"),
                    metadata=metadata_value,
                )
                release_payload = release.to_dict()
                outputs["release"] = release_payload
                message = f"Prepared release metadata for version {release.version}"
                self._publish_status(
                    message,
                    kind="integrations_release",
                    task=name,
                    payload=release_payload,
                )
                summary_lines.append(message)
        except Exception as exc:
            return self._handle_task_failure(name, exc)

        if budget is not None:
            budget.consume()

        if not summary_lines:
            summary_lines.append("Integrations task completed with no actions performed")

        summary_text = "\n".join(summary_lines)
        structured = StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": summary_text,
                            "parsed": outputs,
                        }
                    }
                ]
            },
            content=summary_text,
            parsed=outputs,
            schema={"name": "IntegrationsResult"},
            structured=True,
        )
        self._mark_task_complete(name)
        self._last_response_text = summary_text
        self._last_structured = structured
        self._register_task_output(name, summary_text, structured, kind="integrations")
        return summary_text, structured

    def _handle_task_failure(
        self,
        task_name: str,
        exc: Exception,
    ) -> tuple[str, StructuredResponse | None]:
        message = f"Task '{task_name}' failed: {exc}"
        self._publish_status(message, kind="error", task=task_name, payload={"error": str(exc)})
        fallback = f"Unable to complete {task_name}: {exc}"
        self._last_response_text = fallback
        self._last_structured = None
        return fallback, None

    def _mark_task_complete(self, task_name: str) -> None:
        budget = self._budgets.get(task_name)
        payload = {"budget": budget.as_dict()} if budget else None
        self._publish_status(
            f"Task '{task_name}' completed",
            kind="success",
            task=task_name,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # AgentSession hooks
    # ------------------------------------------------------------------
    def _handle_round_start(self, payload: Mapping[str, Any]) -> None:
        metadata = payload.get("metadata") or {}
        task_name = metadata.get("task") if isinstance(metadata, Mapping) else None
        if task_name is None:
            task_name = self._current_task
        self._current_round_task = task_name
        if task_name:
            self._publish_status(
                f"Starting round {payload.get('index', 0)} for {task_name}",
                kind="round_start",
                task=task_name,
                payload={"round_index": payload.get("index"), "metadata": metadata},
            )

    def _handle_round_end(self, round_record: AgentRound) -> None:
        task_name = self._current_round_task or self._current_task
        budget = self._budgets.get(task_name) if task_name else None
        if budget:
            budget.consume()
            metadata: dict[str, Any] = dict(round_record.metadata or {})
            metadata["task"] = task_name
            metadata["budget"] = budget.as_dict()
            metadata["progress"] = budget.progress
            if self._runtime_metadata:
                metadata.setdefault("run_metadata", dict(self._runtime_metadata))
            round_record.metadata = metadata
            self._publish_status(
                f"{task_name} progress: {budget.consumed} {budget.unit}",
                kind="progress",
                task=task_name,
                payload={
                    "budget": budget.as_dict(),
                    "round_index": round_record.index,
                    "run_metadata": dict(self._runtime_metadata) if self._runtime_metadata else None,
                },
            )
        self._current_round_task = None
        self._runtime_metadata = {}

    # ------------------------------------------------------------------
    # Critic gating helpers
    # ------------------------------------------------------------------
    def _relay_critic_status(self, event: "CriticStatusEvent") -> None:
        if event.payload is None:
            payload = None
        elif isinstance(event.payload, Mapping):
            payload = dict(event.payload)
        else:
            payload = {"data": event.payload}
        self._publish_status(
            event.message,
            kind=event.kind,
            task=event.suite,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _reset_runtime_state(self) -> None:
        self._status_log.clear()
        self._initial_round_index = len(self.session.rounds)
        self._last_response_text = ""
        self._last_structured = None
        self._current_task = None
        self._current_round_task = None
        self._gate_report = None
        self._security_report = None
        self._gate_blocked = False
        self._gate_source = None
        self._external_browsing_enabled = self._external_browsing_default
        self._shared_evidence.clear()
        self._task_outputs.clear()
        self._last_eval_summary = None
        self._completed_evaluations.clear()

    def _publish_status(
        self,
        message: str,
        *,
        kind: str = "info",
        task: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        update = ManagerStatusUpdate(message=message, kind=kind, task=task, payload=payload)
        self._status_log.append(update)
        if self._status_callback is not None:
            self._status_callback(update)

    def _ensure_dependency_agent(self) -> DependencyBuildAgent:
        if self._dependency_agent is None:
            self._dependency_agent = DependencyBuildAgent()
        return self._dependency_agent

    def _ensure_db_migration_agent(self) -> DBMigrationAgent:
        if self._db_migration_agent is None:
            if self.repo_context is None:
                raise RuntimeError("DB migration tasks require a RepoContextAgent")
            self._db_migration_agent = DBMigrationAgent(repo_context=self.repo_context)
        return self._db_migration_agent

    def _ensure_eval_agent(self) -> EvalAgent:
        if self.eval_agent is None:
            raise RuntimeError("Evaluation tasks require an EvalAgent to be attached")
        return self.eval_agent

    def _ensure_security_agent(self) -> SecurityAgent:
        if self._security_agent is None:
            self._security_agent = SecurityAgent()
        return self._security_agent

    def _ensure_doc_agent(self) -> DocAgent:
        if self._doc_agent is None:
            if self.repo_context is None:
                raise RuntimeError("Documentation tasks require a RepoContextAgent")
            self._doc_agent = DocAgent(
                repo_context=self.repo_context,
                research_agent=self.research_agent,
            )
        return self._doc_agent

    def _ensure_integrations_agent(self) -> IntegrationsAgent:
        if self.repo_context is None:
            raise RuntimeError("Integration tasks require a RepoContextAgent")
        if self._integrations_agent is None:
            self._integrations_agent = IntegrationsAgent(repo_context=self.repo_context)
        return self._integrations_agent
