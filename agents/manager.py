"""High-level manager agent orchestrating DAG-driven task workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from internal.DAG import DAG
from internal.structures import StructuredResponse
from session import AgentRound, AgentSession

from .dependency import DependencyBuildAgent
from .db_migration import DBMigrationAgent
from .eval import EvalAgent, RegressionSummary
from .repo_context import (
    DiffBundle,
    RepoContextAgent,
    RepoSearchResult,
    RepoSymbolResult,
)

if TYPE_CHECKING:
    from .research import ResearchAgent, ResearchResult, ResearchSnippet
    from .tester import CriticStatusEvent, TestCriticAgent, TestCriticReport
    from .eval import PromptComparison

__all__ = [
    "TaskBudget",
    "ManagerStatusUpdate",
    "ManagerResult",
    "ManagerAgent",
]


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
        repo_context: RepoContextAgent | None = None,
        test_critic: "TestCriticAgent" | None = None,
        research_agent: "ResearchAgent" | None = None,
        external_browsing_default: bool = False,
        dependency_agent: DependencyBuildAgent | None = None,
        db_migration_agent: DBMigrationAgent | None = None,
        eval_agent: EvalAgent | None = None,
    ) -> None:
        if session is None:
            if session_factory is None:
                raise ValueError("ManagerAgent requires a session or session_factory")
            session = session_factory()
        self.session = session
        self._status_callback = status_callback
        self._plan_builder = plan_builder or self._default_plan_builder
        self.plan_retries = max(0, plan_retries)
        self.task_retry_limit = max(0, task_retry_limit)

        self._active_plan: list[dict[str, Any]] = []
        self._budgets: dict[str, TaskBudget] = {}
        self._status_log: list[ManagerStatusUpdate] = []
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
        self.research_agent: "ResearchAgent" | None = research_agent
        self._external_browsing_default = bool(external_browsing_default)
        self._external_browsing_enabled = self._external_browsing_default
        self._shared_evidence: dict[str, list["ResearchSnippet"]] = {}
        self._dependency_agent: DependencyBuildAgent | None = dependency_agent
        self._db_migration_agent: DBMigrationAgent | None = db_migration_agent
        self.eval_agent: EvalAgent | None = None
        self._last_eval_summary: RegressionSummary | None = None
        self._completed_evaluations: list[RegressionSummary] = []
        if test_critic is not None:
            self.attach_test_critic(test_critic)
        if eval_agent is not None:
            self.attach_eval_agent(eval_agent)

        self.session.add_round_hooks(
            on_round_start=self._handle_round_start,
            on_round_end=self._handle_round_end,
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
            response_text = self._last_response_text or "Test critic blocked completion due to failing suites."
            structured_response = None
            payload = self._gate_report.to_status_payload() if self._gate_report else None
            summary_payload = {"critic": payload} if payload is not None else None
            self._publish_status(
                "Workflow blocked by critic failures",
                kind="error",
                payload=summary_payload,
            )
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
    # Research helpers
    # ------------------------------------------------------------------
    def attach_research_agent(self, agent: "ResearchAgent" | None) -> None:
        """Attach or detach the research agent used for external browsing."""

        self.research_agent = agent

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
        top_k: int = 5,
        max_search_results: int = 20,
        allow_rewrite: bool = True,
        audience: str | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return structured snippets for ``query`` and cache the evidence."""

        if not query or not str(query).strip():
            return []
        if self.research_agent is None:
            raise RuntimeError("Research agent is not configured")
        if not self._external_browsing_enabled:
            raise RuntimeError("External browsing is disabled for this request")

        try:
            sanitized_top_k = int(top_k)
        except (TypeError, ValueError):
            sanitized_top_k = 5
        if sanitized_top_k <= 0:
            sanitized_top_k = 5
        audience_value = None
        if audience is not None and str(audience).strip():
            audience_value = str(audience).strip()
        result = self.research_agent.search(
            query,
            top_k=sanitized_top_k,
            max_search_results=max_search_results,
            allow_rewrite=allow_rewrite,
            audience=audience_value,
            force_refresh=force_refresh,
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

        if self.test_critic is not None:
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
    def _default_plan_builder(self, user_message: str) -> Sequence[Mapping[str, Any]]:
        cleaned = user_message.strip()
        description = "Respond to the user's request"
        if not cleaned:
            description = "Prompt the user for additional details"
        return [
            {
                "name": "task-1",
                "description": description,
                "prompt": user_message,
                "budget": {"limit": 1.0, "unit": "rounds"},
            }
        ]

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

        research_spec = task.get("research")
        audience_hint = None
        raw_queries: Any = None
        if isinstance(research_spec, Mapping):
            raw_queries = research_spec.get("queries")
            if raw_queries is None and research_spec.get("query") is not None:
                raw_queries = research_spec.get("query")
            audience_hint = research_spec.get("audience")
        else:
            raw_queries = task.get("research_queries")
            audience_hint = task.get("research_audience")
        queries = self._coerce_queries(raw_queries)
        audience_value = str(audience_hint).strip() if audience_hint is not None else None
        if queries:
            try:
                top_k_raw = task.get("research_top_k", 5)
                top_k = int(top_k_raw)
            except (TypeError, ValueError):
                top_k = 5
            if top_k <= 0:
                top_k = 5
            metadata["requested_research"] = list(queries)
            for query_text in queries:
                try:
                    self.request_research(
                        query_text,
                        top_k=top_k,
                        audience=audience_value,
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
            return text, structured
        finally:
            self._current_task = None

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
        return text, structured

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
        self._gate_blocked = False
        self._external_browsing_enabled = self._external_browsing_default
        self._shared_evidence.clear()
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
