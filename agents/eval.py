"""Evaluation helpers for prompt regressions and structured summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # YAML is optional and only needed when users provide YAML specs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from internal.structures import StructuredResponse
from session import AgentSession
from .tester import CriticAnalysis, CriticStatusEvent

__all__ = [
    "PromptEvalResult",
    "PromptComparison",
    "RegressionSummary",
    "EvalAgent",
]


@dataclass(slots=True)
class PromptEvalResult:
    """Outcome returned after issuing a single prompt to an :class:`AgentSession`."""

    prompt: str
    response_text: str
    structured: StructuredResponse | None
    latency: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "response_text": self.response_text,
            "latency": self.latency,
        }
        if self.structured is not None:
            payload["structured"] = {
                "content": self.structured.content,
                "parsed": self.structured.parsed,
                "schema": self.structured.schema,
                "structured": self.structured.structured,
            }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class PromptComparison:
    """Container capturing paired baseline and candidate prompt executions."""

    name: str
    baseline: PromptEvalResult
    candidate: PromptEvalResult
    score: float | None = None
    verdict: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }
        if self.score is not None:
            payload["score"] = self.score
        if self.verdict is not None:
            payload["verdict"] = self.verdict
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class RegressionSummary:
    """Summary of all prompt comparisons produced by :class:`EvalAgent`."""

    name: str | None
    comparisons: tuple[PromptComparison, ...]
    metrics: Mapping[str, Any]
    analysis: CriticAnalysis | None = None
    status_events: tuple[CriticStatusEvent, ...] = ()
    gate_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "metrics": dict(self.metrics),
        }
        if self.analysis is not None:
            payload["analysis"] = {
                "status": self.analysis.status,
                "summary": self.analysis.summary,
                "failing_cases": [dict(case) for case in self.analysis.failing_cases],
                "suggested_tests": list(self.analysis.suggested_tests),
                "patch_hints": list(self.analysis.patch_hints),
            }
        if self.status_events:
            payload["status_events"] = [event.to_dict() for event in self.status_events]
        if self.gate_config:
            payload["gate"] = dict(self.gate_config)
        return payload

    @property
    def failed(self) -> int:
        return sum(1 for comparison in self.comparisons if comparison.verdict == "fail")

    @property
    def passed(self) -> int:
        return sum(1 for comparison in self.comparisons if comparison.verdict == "pass")

    @property
    def total(self) -> int:
        return len(self.comparisons)

    @property
    def is_blocking(self) -> bool:
        if self.analysis is not None:
            return self.analysis.status == "fail"
        enabled = bool(self.gate_config.get("enabled"))
        if not enabled:
            return False
        try:
            allow_failures = int(self.gate_config.get("allow_failures", 0))
        except (TypeError, ValueError):
            allow_failures = 0
        return self.failed > allow_failures


class EvalAgent:
    """Run paired prompt comparisons and compute regression summaries."""

    _DEFAULT_SCORERS: Mapping[str, Callable[[PromptComparison, Mapping[str, Any]], Mapping[str, Any] | None]]

    def __init__(
        self,
        *,
        session: AgentSession | None = None,
        session_factory: Callable[[], AgentSession] | None = None,
        scorers: Mapping[str, Callable[[PromptComparison, Mapping[str, Any]], Mapping[str, Any] | None]] | None = None,
        status_callback: Callable[[CriticStatusEvent], None] | None = None,
    ) -> None:
        if session is None and session_factory is None:
            raise ValueError("EvalAgent requires either a session or a session_factory")

        self._session = session
        self._session_factory = session_factory
        self._status_callback = status_callback

        self._scorers: dict[str, Callable[[PromptComparison, Mapping[str, Any]], Mapping[str, Any] | None]] = {
            "latency": self._score_latency,
            "exact_match": self._score_exact_match,
        }
        if scorers:
            self._scorers.update(scorers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        spec: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RegressionSummary:
        """Execute an evaluation spec and return the aggregated summary."""

        spec_payload = self._load_spec(spec)
        name = str(spec_payload.get("name") or "").strip() or None
        comparisons_spec = self._extract_comparisons(spec_payload)
        if not comparisons_spec:
            raise ValueError("Evaluation spec must define at least one comparison")

        gate_config = self._normalise_gate_config(spec_payload.get("gate"))
        global_scoring = self._normalise_scoring(spec_payload.get("scoring"))

        comparisons: list[PromptComparison] = []
        status_events: list[CriticStatusEvent] = []

        for item in comparisons_spec:
            comparison = self._run_comparison(
                item,
                metadata=metadata,
                global_scoring=global_scoring,
            )
            comparisons.append(comparison)
            event = CriticStatusEvent(
                message=f"Comparison '{comparison.name}' completed",
                kind="evaluation",
                suite=comparison.name,
                payload={
                    "verdict": comparison.verdict,
                    "score": comparison.score,
                    "latency": {
                        "baseline": comparison.baseline.latency,
                        "candidate": comparison.candidate.latency,
                    },
                },
            )
            status_events.append(event)
            if self._status_callback is not None:
                self._status_callback(event)

        metrics = self._aggregate_metrics(comparisons)
        analysis = self._build_analysis(name, comparisons, metrics, gate_config)

        return RegressionSummary(
            name=name,
            comparisons=tuple(comparisons),
            metrics=metrics,
            analysis=analysis,
            status_events=tuple(status_events),
            gate_config=gate_config,
        )

    def format_summary(self, summary: RegressionSummary) -> str:
        """Return a readable textual overview of the evaluation results."""

        lines: list[str] = []
        header = "Evaluation results"
        if summary.name:
            header = f"Evaluation '{summary.name}' results"
        lines.append(header)
        total = summary.total
        passed = summary.passed
        failed = summary.failed
        unknown = total - passed - failed
        lines.append(f"- Passed: {passed}/{total}")
        if failed:
            lines.append(f"- Failed: {failed}")
        if unknown:
            lines.append(f"- Pending: {unknown}")
        baseline_avg = summary.metrics.get("baseline_latency_avg")
        candidate_avg = summary.metrics.get("candidate_latency_avg")
        delta_avg = summary.metrics.get("latency_delta_avg")
        if baseline_avg is not None and candidate_avg is not None:
            lines.append(
                f"- Avg latency (baseline/candidate): {baseline_avg:.3f}s / {candidate_avg:.3f}s"
            )
        if delta_avg is not None:
            lines.append(f"- Avg latency delta: {delta_avg:+.3f}s")

        for comparison in summary.comparisons:
            verdict = comparison.verdict or "unknown"
            ratio = None
            base_latency = comparison.baseline.latency
            cand_latency = comparison.candidate.latency
            if base_latency > 0:
                ratio = cand_latency / base_latency
            ratio_display = f" (ratio {ratio:.2f}x)" if ratio is not None else ""
            lines.append(
                f"  • {comparison.name}: {verdict}"
                f" [{base_latency:.3f}s → {cand_latency:.3f}s{ratio_display}]"
            )

        if summary.analysis is not None:
            lines.append("")
            lines.append(summary.analysis.summary)

        return "\n".join(lines)

    def to_structured_response(self, summary: RegressionSummary) -> StructuredResponse:
        """Convert a regression summary into a :class:`StructuredResponse`."""

        payload = summary.to_dict()
        text = self.format_summary(summary)
        return StructuredResponse(
            raw_response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "parsed": payload,
                        }
                    }
                ]
            },
            content=text,
            parsed=payload,
            schema={"name": "RegressionSummary"},
            structured=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_spec(
        self,
        spec: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if isinstance(spec, Mapping):
            return dict(spec)

        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
            return {"comparisons": list(spec)}

        if not isinstance(spec, (str, bytes, bytearray)):
            raise TypeError("Evaluation spec must be a mapping, sequence, or path to a spec file")

        path = Path(str(spec))
        if not path.exists():
            raise FileNotFoundError(f"Evaluation spec file not found: {path}")
        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._coerce_spec(json.loads(content))
        if suffix in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover - depends on optional dependency
                raise RuntimeError("PyYAML is required to load YAML evaluation specs")
            return self._coerce_spec(yaml.safe_load(content))  # type: ignore[arg-type]

        # Attempt JSON first, then YAML as a fallback for extension-less files.
        try:
            return self._coerce_spec(json.loads(content))
        except json.JSONDecodeError:
            if yaml is None:  # pragma: no cover - depends on optional dependency
                raise
            return self._coerce_spec(yaml.safe_load(content))  # type: ignore[arg-type]

    @staticmethod
    def _coerce_spec(payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            return dict(payload)
        raise ValueError("Evaluation spec file must contain a mapping at the top level")

    @staticmethod
    def _extract_comparisons(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        comparisons = spec.get("comparisons") or spec.get("pairs") or spec.get("cases")
        if comparisons is None:
            return []
        if not isinstance(comparisons, Sequence):
            raise ValueError("Evaluation spec 'comparisons' must be a sequence")
        extracted: list[Mapping[str, Any]] = []
        for item in comparisons:
            if not isinstance(item, Mapping):
                raise ValueError("Comparison entries must be mappings")
            extracted.append(dict(item))
        return extracted

    @staticmethod
    def _normalise_gate_config(value: Any) -> Mapping[str, Any]:
        if value is None:
            return {"enabled": False, "allow_failures": 0}
        if value is True:
            return {"enabled": True, "allow_failures": 0}
        if value is False:
            return {"enabled": False, "allow_failures": 0}
        if not isinstance(value, Mapping):
            raise ValueError("Evaluation 'gate' configuration must be a mapping or boolean")
        payload: dict[str, Any] = {"enabled": True, "allow_failures": 0}
        payload.update(value)
        allow_failures = payload.get("allow_failures", 0)
        try:
            payload["allow_failures"] = max(0, int(allow_failures))
        except (TypeError, ValueError):
            payload["allow_failures"] = 0
        return payload

    def _normalise_scoring(
        self, value: Any
    ) -> list[Mapping[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            return [dict(value)]
        if isinstance(value, Sequence):
            configs: list[Mapping[str, Any]] = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError("Scoring configuration entries must be mappings")
                configs.append(dict(item))
            return configs
        raise ValueError("Scoring configuration must be a mapping or sequence of mappings")

    def _run_comparison(
        self,
        spec: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None,
        global_scoring: Sequence[Mapping[str, Any]],
    ) -> PromptComparison:
        name = str(spec.get("name") or spec.get("id") or "comparison").strip()
        baseline_prompt = self._resolve_prompt(spec, keys=("baseline", "control", "reference"))
        candidate_prompt = self._resolve_prompt(spec, keys=("candidate", "treatment", "prompt"))
        comparison_metadata: dict[str, Any] = dict(spec.get("metadata", {}))

        baseline_result = self._run_prompt(
            baseline_prompt,
            role="baseline",
            comparison=name,
            metadata=metadata,
            comparison_metadata=comparison_metadata,
        )
        candidate_result = self._run_prompt(
            candidate_prompt,
            role="candidate",
            comparison=name,
            metadata=metadata,
            comparison_metadata=comparison_metadata,
        )

        comparison = PromptComparison(
            name=name,
            baseline=baseline_result,
            candidate=candidate_result,
            metadata=comparison_metadata,
        )

        scoring_configs = list(global_scoring)
        scoring_configs.extend(self._normalise_scoring(spec.get("scoring")))
        self._apply_scorers(comparison, scoring_configs)
        return comparison

    def _run_prompt(
        self,
        prompt: str,
        *,
        role: str,
        comparison: str,
        metadata: Mapping[str, Any] | None,
        comparison_metadata: Mapping[str, Any],
    ) -> PromptEvalResult:
        session = self._create_session()
        meta_payload: dict[str, Any] = {
            "evaluation": {
                "comparison": comparison,
                "role": role,
                "metadata": dict(comparison_metadata),
            }
        }
        if metadata:
            meta_payload["run_metadata"] = dict(metadata)

        start = time.perf_counter()
        text, structured = session.act(prompt, metadata=meta_payload)
        end = time.perf_counter()
        latency = max(0.0, float(end - start))
        result_metadata = {
            "role": role,
            "comparison": comparison,
        }
        if comparison_metadata:
            result_metadata["comparison_metadata"] = dict(comparison_metadata)
        if metadata:
            result_metadata["run_metadata"] = dict(metadata)
        return PromptEvalResult(
            prompt=prompt,
            response_text=text,
            structured=structured,
            latency=latency,
            metadata=result_metadata,
        )

    def _create_session(self) -> AgentSession:
        if self._session_factory is not None:
            return self._session_factory()
        assert self._session is not None
        return self._session

    def _apply_scorers(
        self,
        comparison: PromptComparison,
        configs: Iterable[Mapping[str, Any]],
    ) -> None:
        verdict = comparison.verdict
        score = comparison.score
        metadata_bucket: MutableMapping[str, Any] = dict(comparison.metadata or {})
        for config in configs:
            name = str(config.get("name") or config.get("type") or "").strip().lower()
            if not name:
                continue
            scorer = self._scorers.get(name)
            if scorer is None:
                raise ValueError(f"Unknown scoring hook: {name}")
            outcome = scorer(comparison, config)
            if not outcome:
                continue
            if "score" in outcome and outcome["score"] is not None:
                try:
                    score = float(outcome["score"])
                except (TypeError, ValueError):
                    pass
            new_verdict = outcome.get("verdict")
            if isinstance(new_verdict, str):
                normalised = new_verdict.lower()
                if normalised == "fail" or verdict != "fail":
                    verdict = normalised
            details = outcome.get("details")
            if isinstance(details, Mapping):
                key = config.get("key") or name
                metadata_bucket[str(key)] = dict(details)
        comparison.score = score
        comparison.verdict = verdict
        if metadata_bucket:
            comparison.metadata = metadata_bucket

    def _aggregate_metrics(self, comparisons: Sequence[PromptComparison]) -> Mapping[str, Any]:
        total = len(comparisons)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "unknown": 0}
        baseline_latencies = [comparison.baseline.latency for comparison in comparisons]
        candidate_latencies = [comparison.candidate.latency for comparison in comparisons]
        deltas = [cand - base for cand, base in zip(candidate_latencies, baseline_latencies)]
        passed = sum(1 for comparison in comparisons if comparison.verdict == "pass")
        failed = sum(1 for comparison in comparisons if comparison.verdict == "fail")
        unknown = total - passed - failed
        metrics: dict[str, Any] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "baseline_latency_avg": sum(baseline_latencies) / total,
            "candidate_latency_avg": sum(candidate_latencies) / total,
            "latency_delta_avg": sum(deltas) / total,
        }
        return metrics

    def _build_analysis(
        self,
        name: str | None,
        comparisons: Sequence[PromptComparison],
        metrics: Mapping[str, Any],
        gate_config: Mapping[str, Any],
    ) -> CriticAnalysis | None:
        if not gate_config.get("enabled", False):
            return None
        allow_failures = int(gate_config.get("allow_failures", 0))
        failures = [comparison for comparison in comparisons if comparison.verdict == "fail"]
        status = "pass"
        summary = f"{metrics.get('passed', 0)}/{metrics.get('total', 0)} comparisons passed"
        if len(failures) > allow_failures:
            status = "fail"
            summary = f"Evaluation '{name}' failed" if name else "Evaluation failed"
        failing_cases: list[dict[str, Any]] = []
        for comparison in failures:
            details = {
                "name": comparison.name,
                "verdict": comparison.verdict,
                "score": comparison.score,
                "latency": {
                    "baseline": comparison.baseline.latency,
                    "candidate": comparison.candidate.latency,
                },
            }
            metadata = comparison.metadata
            if metadata:
                details["metadata"] = dict(metadata)
            failing_cases.append(details)

        return CriticAnalysis(
            status=status,
            summary=summary,
            failing_cases=failing_cases,
            suggested_tests=[],
            patch_hints=[],
        )

    @staticmethod
    def _resolve_prompt(spec: Mapping[str, Any], *, keys: Sequence[str]) -> str:
        for key in keys:
            if key in spec:
                text = spec.get(key)
                if text is None:
                    continue
                prompt = str(text)
                if prompt.strip():
                    return prompt
        raise ValueError(f"Comparison spec missing required prompt (expected one of {keys})")

    # ------------------------------------------------------------------
    # Built-in scoring hooks
    # ------------------------------------------------------------------
    def _score_latency(
        self,
        comparison: PromptComparison,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        baseline_latency = comparison.baseline.latency
        candidate_latency = comparison.candidate.latency
        ratio = None
        if baseline_latency > 0:
            ratio = candidate_latency / baseline_latency
        delta = candidate_latency - baseline_latency

        verdict: str | None = None
        max_ratio = config.get("max_ratio")
        if max_ratio is not None and ratio is not None:
            try:
                threshold = float(max_ratio)
            except (TypeError, ValueError):
                threshold = None
            else:
                if ratio > threshold:
                    verdict = "fail"

        max_delta = config.get("max_delta")
        if max_delta is not None:
            try:
                delta_threshold = float(max_delta)
            except (TypeError, ValueError):
                delta_threshold = None
            else:
                if delta > delta_threshold:
                    verdict = "fail"

        if verdict is None and config.get("enforce"):
            verdict = "pass"

        return {
            "score": ratio,
            "verdict": verdict,
            "details": {
                "latency_ratio": ratio,
                "latency_delta": delta,
                "baseline_latency": baseline_latency,
                "candidate_latency": candidate_latency,
            },
        }

    def _score_exact_match(
        self,
        comparison: PromptComparison,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        target = config.get("expect")
        candidate_text = comparison.candidate.response_text.strip()
        if target is None or str(target).lower() == "baseline":
            expected = comparison.baseline.response_text.strip()
        else:
            expected = str(target).strip()
        success = candidate_text == expected
        verdict = "pass" if success else "fail"
        return {
            "score": 1.0 if success else 0.0,
            "verdict": verdict,
            "details": {
                "expected": expected,
                "observed": candidate_text,
            },
        }

