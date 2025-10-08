"""High-level manager agent orchestrating DAG-driven task workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from internal.DAG import DAG
from internal.structures import StructuredResponse
from session import AgentRound, AgentSession

from .repo_context import (
    DiffBundle,
    RepoContextAgent,
    RepoSearchResult,
    RepoSymbolResult,
)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "structured_response": self.structured_response,
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "plan": [dict(step) for step in self.plan],
            "budgets": {name: budget.as_dict() for name, budget in self.budgets.items()},
            "status_updates": [update.as_dict() for update in self.status_updates],
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
        self._publish_status(
            f"Received user request ({len(user_message)} characters)",
            kind="info",
            payload={"user_message": user_message, "metadata": dict(self._runtime_metadata)},
        )

        dag = self._build_workflow()
        dag.set_constant("user_input", user_message)
        dag.run(targets=["execute"])

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
        )

    # ------------------------------------------------------------------
    # Repository context helpers
    # ------------------------------------------------------------------
    def attach_repo_context(self, repo_context: RepoContextAgent) -> None:
        """Attach or replace the repo context agent used for focused queries."""

        self.repo_context = repo_context

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
    # State helpers
    # ------------------------------------------------------------------
    def _reset_runtime_state(self) -> None:
        self._status_log.clear()
        self._initial_round_index = len(self.session.rounds)
        self._last_response_text = ""
        self._last_structured = None
        self._current_task = None
        self._current_round_task = None

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
