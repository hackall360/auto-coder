from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Mapping
import types
import sys

import pytest


_lmstudio_stub = types.ModuleType("lmstudio")
_psutil_stub = types.ModuleType("psutil")


class _ToolFunctionDef:
    def __init__(self, name, description="", parameters=None, implementation=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.implementation = implementation


_lmstudio_stub.ToolFunctionDef = _ToolFunctionDef
_lmstudio_stub.Chat = type("Chat", (), {})


def _llm_stub(*args, **kwargs):  # pragma: no cover - safety fallback
    raise RuntimeError("lmstudio stub should not be invoked in unit tests")


_lmstudio_stub.llm = _llm_stub
sys.modules.setdefault("lmstudio", _lmstudio_stub)
_psutil_stub.Process = type("Process", (), {"__init__": lambda self, pid=None: setattr(self, "pid", pid or 0)})
sys.modules.setdefault("psutil", _psutil_stub)


from agents.coder import CoderAgent
from agents.repo_context import RepoSearchResult
from internal.structures import StructuredResponse
from session import AgentRound


@dataclass
class _SessionState:
    parsed: Mapping[str, Any]
    text: str = "complete"


class DummySession:
    """Lightweight stub mimicking :class:`AgentSession` for coder tests."""

    def __init__(self) -> None:
        self.rounds: list[AgentRound] = []
        self.state = _SessionState(parsed={"rationale": "", "patches": []})
        self.last_prompt: str | None = None
        self.last_metadata: Mapping[str, Any] | None = None

    def act(self, user_message: str | None = None, **kwargs: Any):
        self.last_prompt = user_message
        self.last_metadata = kwargs.get("metadata")
        parsed = self.state.parsed
        structured = StructuredResponse(
            raw_response={
                "choices": [
                    {"message": {"content": self.state.text, "parsed": parsed}},
                ]
            },
            content=self.state.text,
            parsed=parsed,
            schema=kwargs.get("schema"),
            structured=True,
        )
        round_record = AgentRound(
            index=len(self.rounds),
            user_message=user_message,
            response_text=self.state.text,
            result=structured,
            transcript=[],
            messages=[],
            tool_history={"calls": [], "results": []},
            metadata=kwargs.get("metadata"),
        )
        self.rounds.append(round_record)
        return self.state.text, structured

    def last_round(self) -> AgentRound | None:
        return self.rounds[-1] if self.rounds else None


def _build_patch(original: str, updated: str, path: Path) -> str:
    return textwrap.dedent(
        f"""
        diff --git a/{path.name} b/{path.name}
        --- a/{path.name}
        +++ b/{path.name}
        @@ -1 +1 @@
        -{original}
        +{updated}
        """
    ).strip() + "\n"


def test_coder_agent_applies_patch_and_tracks_diff(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello\n", encoding="utf-8")

    session = DummySession()
    patch_text = _build_patch("hello", "hello world", target)
    session.state = _SessionState(
        parsed={
            "rationale": "Updated greeting for clarity.",
            "patches": [{"path": "hello.txt", "patch": patch_text}],
            "change_summaries": [
                {"path": "hello.txt", "summary": "Switch to friendly greeting"}
            ],
            "dependency_hints": ["run pytest"],
        }
    )

    agent = CoderAgent(session=session, repo_root=str(tmp_path))
    result = agent.run_task("Improve the salutation")

    assert target.read_text(encoding="utf-8") == "hello world\n"
    assert result.rationale == "Updated greeting for clarity."
    assert result.applied_diffs and result.applied_diffs[0].changed_paths == ("hello.txt",)
    assert result.change_summaries[0].summary == "Switch to friendly greeting"
    assert result.dependency_hints == ("run pytest",)
    assert agent.applied_diffs == result.applied_diffs
    payload = result.to_dict()
    assert payload["patches"][0]["path"] == "hello.txt"
    assert payload["change_summaries"][0]["summary"] == "Switch to friendly greeting"


def test_prompt_includes_context_and_hints(tmp_path: Path) -> None:
    target = tmp_path / "noop.txt"
    target.write_text("keep\n", encoding="utf-8")

    session = DummySession()
    session.state = _SessionState(parsed={"rationale": "No changes", "patches": []})

    context = RepoSearchResult(path="src/app.py", offset=10, score=0.9, text="def foo(): pass")
    agent = CoderAgent(session=session, repo_root=str(tmp_path))

    agent.run_task(
        "Analyze the helper function",
        context_payloads=[context],
        guidance_hints=[{"path": "src/app.py", "range": "L5-L12", "message": "Check logic"}],
    )

    assert session.last_prompt is not None
    assert "src/app.py" in session.last_prompt
    assert "L5-L12" in session.last_prompt
    assert session.last_metadata["context_count"] == 1
    assert session.last_metadata["hint_count"] == 1


def test_change_summary_falls_back_to_patch_stats(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("1\n", encoding="utf-8")

    session = DummySession()
    patch_text = _build_patch("1", "2", target)
    session.state = _SessionState(
        parsed={
            "rationale": "Increment constant",
            "patches": [{"patch": patch_text}],
            "change_summaries": [],
            "dependency_hints": [],
        }
    )

    agent = CoderAgent(session=session, repo_root=str(tmp_path))
    result = agent.run_task("Increment the value")

    assert any("+" in summary.summary for summary in result.change_summaries)
    assert target.read_text(encoding="utf-8").startswith("2")
