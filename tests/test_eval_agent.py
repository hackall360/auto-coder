from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

if "lmstudio" not in sys.modules:
    stub = types.ModuleType("lmstudio")

    class _StubChat:  # minimal stub to satisfy chat imports
        pass

    stub.Chat = _StubChat

    class _StubToolFunctionDef:  # pragma: no cover - structural stub only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    stub.ToolFunctionDef = _StubToolFunctionDef

    def _stub_llm(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - test stub only
        raise RuntimeError("lmstudio.llm is not available in the test environment")

    stub.llm = _stub_llm
    sys.modules["lmstudio"] = stub

if "agents" not in sys.modules:
    package_stub = types.ModuleType("agents")
    package_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "agents")]
    sys.modules["agents"] = package_stub

if "session" not in sys.modules:
    session_stub = types.ModuleType("session")

    class _StubAgentSession:
        def act(self, *_: Any, **__: Any) -> tuple[str, Any]:  # pragma: no cover - stub only
            raise NotImplementedError

    session_stub.AgentSession = _StubAgentSession
    sys.modules["session"] = session_stub

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")

    class _StubProcess:  # pragma: no cover - structural stub only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("psutil is unavailable in the test environment")

    class _StubNoSuchProcess(Exception):
        pass

    psutil_stub.Process = _StubProcess
    psutil_stub.NoSuchProcess = _StubNoSuchProcess
    sys.modules["psutil"] = psutil_stub

from agents.eval import EvalAgent, RegressionSummary
from internal.structures import StructuredResponse


def _structured(text: str) -> StructuredResponse:
    return StructuredResponse(
        raw_response={"choices": [{"message": {"role": "assistant", "content": text}}]},
        content=text,
        parsed=None,
        schema=None,
        structured=False,
    )


class StubSession:
    def __init__(self, responses: Mapping[str, tuple[str, StructuredResponse]]):
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    def act(self, prompt: str, *, metadata: Mapping[str, Any] | None = None, **_: Any) -> tuple[str, StructuredResponse]:
        self.calls.append({"prompt": prompt, "metadata": metadata})
        if prompt not in self.responses:
            raise KeyError(f"No stubbed response for {prompt}")
        return self.responses[prompt]


def _install_timer(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    sequence = iter(values)

    def fake_perf_counter() -> float:
        return next(sequence)

    monkeypatch.setattr("agents.eval.time.perf_counter", fake_perf_counter)


def test_eval_agent_aggregates_and_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "prompt-a": ("baseline-1", _structured("baseline-1")),
        "prompt-b": ("ok", _structured("ok")),
        "prompt-c": ("baseline-2", _structured("baseline-2")),
        "prompt-d": ("nope", _structured("nope")),
    }
    session = StubSession(responses)
    agent = EvalAgent(session=session)

    _install_timer(monkeypatch, [0.0, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.7])

    spec = {
        "name": "regression-check",
        "comparisons": [
            {
                "name": "case-1",
                "baseline": "prompt-a",
                "candidate": "prompt-b",
                "scoring": [{"name": "exact_match", "expect": "ok"}],
            },
            {
                "name": "case-2",
                "baseline": "prompt-c",
                "candidate": "prompt-d",
                "scoring": [{"name": "exact_match", "expect": "expected"}],
            },
        ],
        "gate": {"allow_failures": 0},
    }

    summary = agent.run(spec)

    assert isinstance(summary, RegressionSummary)
    assert summary.total == 2
    assert summary.metrics["failed"] == 1
    assert summary.is_blocking
    assert summary.analysis is not None
    assert summary.analysis.status == "fail"
    assert summary.comparisons[0].verdict == "pass"
    assert summary.comparisons[1].verdict == "fail"
    assert len(summary.status_events) == 2
    assert len(session.calls) == 4
    first_metadata = session.calls[0]["metadata"]
    assert first_metadata is not None
    assert first_metadata["evaluation"]["role"] == "baseline"


def test_eval_agent_allows_grace_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "prompt-a": ("base", _structured("base")),
        "prompt-b": ("bad", _structured("bad")),
    }
    session = StubSession(responses)
    agent = EvalAgent(session=session)

    _install_timer(monkeypatch, [0.0, 0.1, 0.1, 0.2])

    spec = {
        "comparisons": [
            {
                "name": "single",
                "baseline": "prompt-a",
                "candidate": "prompt-b",
                "scoring": [{"name": "exact_match", "expect": "expected"}],
            }
        ],
        "gate": {"allow_failures": 1},
    }

    summary = agent.run(spec)
    assert summary.total == 1
    assert summary.failed == 1
    assert not summary.is_blocking
    assert summary.analysis is not None
    assert summary.analysis.status == "pass"


def test_eval_agent_loads_yaml_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_mod = pytest.importorskip("yaml")

    spec_data = {
        "pairs": [
            {
                "name": "yaml-case",
                "control": "prompt-a",
                "treatment": "prompt-b",
                "scoring": {"name": "exact_match", "expect": "baseline"},
            }
        ],
        "gate": False,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml_mod.safe_dump(spec_data), encoding="utf-8")

    responses = {
        "prompt-a": ("alpha", _structured("alpha")),
        "prompt-b": ("alpha", _structured("alpha")),
    }
    session = StubSession(responses)
    agent = EvalAgent(session=session)

    _install_timer(monkeypatch, [0.0, 0.1, 0.1, 0.2])

    summary = agent.run(str(spec_path))
    assert summary.total == 1
    assert summary.failed == 0
    assert not summary.is_blocking
    assert summary.comparisons[0].verdict == "pass"
