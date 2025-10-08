from __future__ import annotations

from typing import Any, Mapping

import pytest

from agents.manager import ManagerAgent, ManagerStatusUpdate
from internal.structures import StructuredResponse
from session import AgentRound


class DummySession:
    """Lightweight stub mimicking :class:`AgentSession` for manager tests."""

    def __init__(self) -> None:
        self.rounds: list[AgentRound] = []
        self._on_round_start = None
        self._on_round_end = None
        self.next_response_text = "done"

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end

    def act(self, user_message: str | None = None, *, metadata: Mapping[str, Any] | None = None, **_: Any):
        index = len(self.rounds)
        if self._on_round_start is not None:
            self._on_round_start(
                {
                    "index": index,
                    "user_message": user_message,
                    "session": self,
                    "metadata": dict(metadata) if metadata else None,
                }
            )
        structured = StructuredResponse(
            raw_response={"choices": [{"message": {"content": self.next_response_text, "parsed": {"ok": True}}}]},
            content=self.next_response_text,
            parsed={"ok": True},
            schema=None,
            structured=False,
        )
        round_record = AgentRound(
            index=index,
            user_message=user_message,
            response_text=self.next_response_text,
            result=structured,
            transcript=[],
            messages=[],
            tool_history={"calls": [], "results": []},
            metadata=dict(metadata) if metadata else None,
        )
        self.rounds.append(round_record)
        if self._on_round_end is not None:
            self._on_round_end(round_record)
        return self.next_response_text, structured


@pytest.mark.parametrize("message", ["solve this", ""])
def test_manager_tracks_budget_progress(message: str) -> None:
    session = DummySession()
    captured: list[ManagerStatusUpdate] = []
    manager = ManagerAgent(session=session, status_callback=captured.append)

    result = manager.run(message)

    assert result.response_text == session.next_response_text
    assert result.plan, "planner should produce at least one task"
    first_task = result.plan[0]
    assert "name" in first_task
    assert "task-1" in result.budgets
    budget = result.budgets["task-1"]
    assert pytest.approx(budget.consumed) == 1.0
    assert budget.remaining in (0.0, None)
    assert result.rounds
    last_round = result.rounds[-1]
    assert last_round.metadata is not None
    assert last_round.metadata["budget"]["consumed"] == pytest.approx(1.0)
    assert any(update.kind == "progress" for update in captured)
