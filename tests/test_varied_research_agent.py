import sys
import types
from typing import Any, Mapping

import pytest

class _StubModel:
    def respond(self, *_, **__):
        return {"choices": [{"message": {"content": "stub"}}]}

    def respond_stream(self, *_, **__):
        yield {"choices": [{"delta": {"content": "stub"}}]}


class _StubChat:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.messages: list[Mapping[str, Any]] = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    @classmethod
    def from_history(cls, history: Any) -> "_StubChat":
        instance = cls()
        if isinstance(history, Mapping):
            instance.messages = list(history.get("messages", []))
        elif isinstance(history, str):
            instance.messages = [{"role": "user", "content": history}]
        return instance

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})


class _ToolFunctionDef:
    def __init__(
        self,
        name: str,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        implementation: Any = None,
        **_: Any,
    ) -> None:
        self.name = name
        self.description = description or ""
        self.parameters = dict(parameters or {})
        self.implementation = implementation


sys.modules.setdefault(
    "lmstudio",
    types.SimpleNamespace(
        llm=lambda *_, **__: _StubModel(),
        Chat=_StubChat,
        ToolFunctionDef=_ToolFunctionDef,
    ),
)

sys.modules.setdefault("psutil", types.SimpleNamespace(Process=lambda pid=None: types.SimpleNamespace(pid=pid)))

from agents.research import ResearchResult, VariedResearchAgent


class RecordingResearchAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_search_results: int = 20,
        allow_rewrite: bool = True,
        audience: str | None = None,
        force_refresh: bool = False,
        alpha: float = 0.6,
    ) -> ResearchResult:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "max_search_results": max_search_results,
                "allow_rewrite": allow_rewrite,
                "alpha": alpha,
                "audience": audience,
            }
        )
        return ResearchResult(query=query, snippets=())


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("light", {"top_k": 4, "max_search_results": 12, "allow_rewrite": False, "alpha": pytest.approx(0.45)}),
        ("balanced", {"top_k": 6, "max_search_results": 18, "allow_rewrite": True, "alpha": pytest.approx(0.55)}),
        ("deep", {"top_k": 12, "max_search_results": 40, "allow_rewrite": True, "alpha": pytest.approx(0.8)}),
    ],
)
def test_varied_agent_modes_apply_defaults(mode: str, expected: dict[str, object]) -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(base)

    agent.search("topic", mode=mode)

    call = base.calls[-1]
    assert call["top_k"] == expected["top_k"]
    assert call["max_search_results"] == expected["max_search_results"]
    assert call["allow_rewrite"] is expected["allow_rewrite"]
    assert call["alpha"] == expected["alpha"]


def test_varied_agent_supports_custom_mode_overrides() -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(
        base,
        mode_defaults={"deep": {"top_k": 20, "max_search_results": 50, "allow_rewrite": False, "alpha": 0.9}},
    )

    agent.search("topic", mode="deep")

    call = base.calls[-1]
    assert call["top_k"] == 20
    assert call["max_search_results"] == 50
    assert call["allow_rewrite"] is False
    assert call["alpha"] == pytest.approx(0.9)


def test_varied_agent_profile_chain_merges_overrides() -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(base)

    agent.search(
        "topic",
        mode="balanced",
        profile=["investigative", {"max_search_results": 70, "alpha": 0.5}],
    )

    call = base.calls[-1]
    assert call["top_k"] == 10  # investigative profile
    assert call["max_search_results"] == 70
    assert call["allow_rewrite"] is True
    assert call["alpha"] == pytest.approx(0.5)


def test_varied_agent_caps_top_k_to_max_results() -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(base)

    agent.search(
        "topic",
        mode="light",
        profile={"top_k": 25, "max_search_results": 9},
    )

    call = base.calls[-1]
    assert call["top_k"] == 9
    assert call["max_search_results"] == 9


def test_varied_agent_falls_back_to_default_mode() -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(base, default_mode="deep")

    agent.search("topic", mode="unknown")

    call = base.calls[-1]
    assert call["top_k"] == 12
    assert call["max_search_results"] == 40
    assert call["allow_rewrite"] is True


def test_varied_agent_exposes_multiple_profiles() -> None:
    base = RecordingResearchAgent()
    agent = VariedResearchAgent(base)

    assert len(agent.profiles) >= 5
    for key in ("skim", "survey", "balanced", "insight", "deep_dive"):
        assert key in agent.profiles
