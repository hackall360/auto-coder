"""Security scanning orchestration utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from .runner import RunReport, RunnerAgent

__all__ = [
    "SecurityCacheDirective",
    "SecurityScanFinding",
    "SecurityScanReport",
    "SecurityScanResult",
    "SecurityToolchain",
    "SecurityAgent",
]

_SEVERITY_ORDER = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "moderate": 3,
    "low": 2,
    "info": 1,
    "informational": 1,
    "none": 0,
    "unknown": 0,
}


def _normalize_severity(value: Any) -> str:
    if value is None:
        return "info"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 4:
            return "critical"
        if numeric >= 3:
            return "high"
        if numeric >= 2:
            return "medium"
        if numeric >= 1:
            return "low"
        return "info"
    text = str(value).strip().lower()
    if not text:
        return "info"
    if text in _SEVERITY_ORDER:
        if text == "informational":
            return "info"
        if text == "moderate":
            return "medium"
        return text
    return text


@dataclass(slots=True)
class SecurityCacheDirective:
    """Hint describing cache paths that should be persisted between scans."""

    tool: str
    paths: tuple[str, ...]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "paths": list(self.paths),
            "description": self.description,
        }

    @property
    def hint(self) -> str:
        if not self.paths:
            return self.description or ""
        joined = ", ".join(self.paths)
        if self.description:
            return f"{self.description}: {joined}"
        return joined


@dataclass(slots=True)
class SecurityScanFinding:
    """Individual issue discovered by a security scan."""

    tool: str
    message: str
    severity: str
    category: str | None = None
    location: str | None = None
    rule_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_severity(self) -> str:
        return _normalize_severity(self.severity)

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_ORDER.get(self.normalized_severity, 0)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "message": self.message,
            "severity": self.normalized_severity,
        }
        if self.category is not None:
            payload["category"] = self.category
        if self.location is not None:
            payload["location"] = self.location
        if self.rule_id is not None:
            payload["rule_id"] = self.rule_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def describe(self) -> str:
        severity = self.normalized_severity.upper()
        location = f" ({self.location})" if self.location else ""
        rule = f" [{self.rule_id}]" if self.rule_id else ""
        return f"{severity}{rule}{location}: {self.message}"


@dataclass(slots=True)
class SecurityScanReport:
    """Structured representation of a single toolchain run."""

    toolchain: str
    category: str
    command: tuple[str, ...]
    report: RunReport
    findings: tuple[SecurityScanFinding, ...] = ()
    cache_directive: SecurityCacheDirective | None = None

    @property
    def ok(self) -> bool:
        return bool(self.report.ok)

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(self.report.artifacts or ())

    @property
    def highest_severity(self) -> str | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda finding: finding.severity_rank).normalized_severity

    @property
    def severity_counts(self) -> Mapping[str, int]:
        counter: Counter[str] = Counter(
            finding.normalized_severity for finding in self.findings
        )
        return dict(counter)

    def describe(self, *, limit: int = 3) -> str:
        title = f"{self.toolchain} ({self.category})"
        if not self.findings:
            return f"{title}: no findings"
        counts = ", ".join(
            f"{severity.upper()}×{count}" for severity, count in sorted(self.severity_counts.items())
        )
        lines = [f"{title}: {len(self.findings)} findings ({counts})"]
        for finding in self.findings[:limit]:
            lines.append(f" - {finding.describe()}")
        if len(self.findings) > limit:
            lines.append(f" - … {len(self.findings) - limit} more findings")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "toolchain": self.toolchain,
            "category": self.category,
            "command": list(self.command),
            "status": self.report.status,
            "ok": self.ok,
            "exit_code": self.report.exit_code,
            "highest_severity": self.highest_severity,
            "severity_counts": dict(self.severity_counts),
            "findings": [finding.to_dict() for finding in self.findings],
            "artifacts": list(self.artifacts),
        }
        if self.cache_directive is not None:
            payload["cache_directive"] = self.cache_directive.to_dict()
        return payload


@dataclass(slots=True)
class SecurityScanResult:
    """Aggregate outcome for a set of security scans."""

    reports: tuple[SecurityScanReport, ...]
    threshold: str | None = None
    blocked: bool = False
    summary: str | None = None

    @property
    def highest_severity(self) -> str | None:
        best: SecurityScanFinding | None = None
        for report in self.reports:
            if not report.findings:
                continue
            candidate = max(report.findings, key=lambda finding: finding.severity_rank)
            if best is None or candidate.severity_rank > best.severity_rank:
                best = candidate
        return best.normalized_severity if best is not None else None

    @property
    def artifacts(self) -> tuple[str, ...]:
        collected: list[str] = []
        for report in self.reports:
            for path in report.artifacts:
                if path not in collected:
                    collected.append(path)
        return tuple(collected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [report.to_dict() for report in self.reports],
            "threshold": self.threshold,
            "blocked": self.blocked,
            "highest_severity": self.highest_severity,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
        }


@dataclass(slots=True)
class SecurityToolchain:
    """Configuration describing how to invoke and parse a scan tool."""

    name: str
    category: str
    command: tuple[str, ...]
    parser: Callable[[RunReport], Iterable[SecurityScanFinding]] | None = None
    workdir: str | os.PathLike[str] | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    cache_paths: tuple[str, ...] = ()
    cache_description: str | None = None

    def with_overrides(self, **overrides: Any) -> "SecurityToolchain":
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "category": self.category,
            "command": self.command,
            "parser": self.parser,
            "workdir": self.workdir,
            "env": dict(self.env),
            "artifacts": self.artifacts,
            "cache_paths": self.cache_paths,
            "cache_description": self.cache_description,
        }
        payload.update(overrides)
        return SecurityToolchain(**payload)


class SecurityAgent:
    """Coordinate multiple security scanning toolchains via :class:`RunnerAgent`."""

    DEFAULT_SEQUENCE: tuple[str, ...] = ("dependencies", "static", "secrets", "sbom")

    def __init__(
        self,
        *,
        runner: RunnerAgent | None = None,
        toolchains: Mapping[str, SecurityToolchain | Mapping[str, Any]] | None = None,
    ) -> None:
        self.runner = runner or RunnerAgent()
        self._toolchains: dict[str, SecurityToolchain] = self._build_default_toolchains()
        if toolchains:
            for name, spec in toolchains.items():
                self.configure_toolchain(name, spec)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def configure_toolchain(
        self,
        name: str,
        spec: SecurityToolchain | Mapping[str, Any],
    ) -> None:
        if isinstance(spec, SecurityToolchain):
            toolchain = spec
        else:
            payload = dict(spec)
            payload.setdefault("name", name)
            payload.setdefault("category", name)
            command = payload.get("command")
            if not command:
                raise ValueError(f"Security toolchain '{name}' missing command configuration")
            if isinstance(command, (str, bytes)):
                payload["command"] = tuple(str(command).split())
            elif isinstance(command, Sequence):
                payload["command"] = tuple(str(part) for part in command)
            else:
                raise TypeError("Command must be a string or sequence of strings")
            parser = payload.get("parser")
            if parser is not None and not callable(parser):
                raise TypeError("Toolchain parser must be callable")
            env = payload.get("env")
            if env is not None and not isinstance(env, Mapping):
                raise TypeError("Toolchain env must be a mapping")
            cache_paths = payload.get("cache_paths")
            if isinstance(cache_paths, (str, bytes, os.PathLike)):
                payload["cache_paths"] = (str(cache_paths),)
            elif cache_paths is not None:
                payload["cache_paths"] = tuple(str(path) for path in cache_paths)
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, (str, bytes, os.PathLike)):
                payload["artifacts"] = (str(artifacts),)
            elif artifacts is not None:
                payload["artifacts"] = tuple(str(path) for path in artifacts)
            toolchain = SecurityToolchain(**payload)
        self._toolchains[name] = toolchain

    def get_toolchain(self, name: str) -> SecurityToolchain:
        try:
            return self._toolchains[name]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Security toolchain '{name}' is not configured") from exc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_scans(
        self,
        *,
        toolchains: Sequence[str] | None = None,
        severity_threshold: str | None = "high",
        stop_on_failure: bool = True,
        workdir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SecurityScanResult:
        sequence = list(toolchains) if toolchains else list(self.DEFAULT_SEQUENCE)
        reports: list[SecurityScanReport] = []
        blocked = False
        normalized_threshold = _normalize_severity(severity_threshold) if severity_threshold else None
        threshold_rank = (
            _SEVERITY_ORDER.get(normalized_threshold, 0)
            if normalized_threshold is not None
            else None
        )

        for name in sequence:
            if name not in self._toolchains:
                continue
            toolchain = self._toolchains[name]
            report = self._run_toolchain(toolchain, workdir=workdir, env=env)
            reports.append(report)
            highest = report.highest_severity
            if highest is None or threshold_rank is None:
                continue
            if _SEVERITY_ORDER.get(highest, 0) >= threshold_rank:
                blocked = True
                if stop_on_failure:
                    break

        summary = self._build_summary(reports, normalized_threshold, blocked)
        result = SecurityScanResult(
            reports=tuple(reports),
            threshold=normalized_threshold,
            blocked=blocked,
            summary=summary,
        )
        return result

    def format_result(self, result: SecurityScanResult) -> str:
        return result.summary or self._build_summary(list(result.reports), result.threshold, result.blocked)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_toolchain(
        self,
        toolchain: SecurityToolchain,
        *,
        workdir: str | os.PathLike[str] | None,
        env: Mapping[str, str] | None,
    ) -> SecurityScanReport:
        effective_workdir = Path(workdir or toolchain.workdir or self.runner.default_workdir)
        if toolchain.workdir is not None and workdir is not None:
            effective_workdir = Path(workdir)
        env_payload: dict[str, str] = {}
        if isinstance(toolchain.env, Mapping):
            env_payload.update({str(key): str(value) for key, value in toolchain.env.items()})
        if env:
            env_payload.update({str(key): str(value) for key, value in env.items()})
        report = self.runner.run_shell(
            toolchain.command,
            workdir=str(effective_workdir),
            env=env_payload,
            artifacts=toolchain.artifacts,
            metadata={
                "toolchain": toolchain.name,
                "category": toolchain.category,
            },
        )
        findings = tuple(self._parse_findings(report, toolchain=toolchain))
        cache_directive = None
        if toolchain.cache_paths:
            cache_directive = SecurityCacheDirective(
                tool=toolchain.name,
                paths=tuple(str(Path(path)) for path in toolchain.cache_paths),
                description=toolchain.cache_description,
            )
        return SecurityScanReport(
            toolchain=toolchain.name,
            category=toolchain.category,
            command=toolchain.command,
            report=report,
            findings=findings,
            cache_directive=cache_directive,
        )

    def _parse_findings(
        self,
        report: RunReport,
        *,
        toolchain: SecurityToolchain,
    ) -> Iterable[SecurityScanFinding]:
        parser = toolchain.parser or self._default_parser
        parsed = parser(report)
        if parsed is None:
            return ()
        if isinstance(parsed, SecurityScanFinding):
            return (parsed,)
        if not isinstance(parsed, Iterable):  # pragma: no cover - defensive guard
            raise TypeError("Security toolchain parser must return an iterable of findings")
        return tuple(parsed)

    @staticmethod
    def _default_parser(report: RunReport) -> Iterable[SecurityScanFinding]:
        sources: list[Mapping[str, Any]] = []
        if isinstance(report.metadata, Mapping):
            sources.append(report.metadata)
        if isinstance(report.raw, Mapping):
            sources.append(report.raw)
        metadata_source = report.metadata if isinstance(report.metadata, Mapping) else {}
        collected: list[SecurityScanFinding] = []
        for source in sources:
            findings = source.get("findings") if isinstance(source, Mapping) else None
            if not findings:
                continue
            for item in findings:
                if not isinstance(item, Mapping):
                    continue
                tool_name = item.get("tool") or source.get("tool") or metadata_source.get("toolchain")
                category = item.get("category") or source.get("category") or metadata_source.get("category")
                message = item.get("message") or item.get("detail") or ""
                finding = SecurityScanFinding(
                    tool=str(tool_name or "unknown"),
                    message=str(message),
                    severity=_normalize_severity(item.get("severity")),
                    category=str(category) if category is not None else None,
                    location=item.get("location"),
                    rule_id=item.get("rule") or item.get("rule_id"),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {"tool", "message", "detail", "severity", "category", "location", "rule", "rule_id"}
                    },
                )
                collected.append(finding)
            if collected:
                break
        return collected

    def _build_summary(
        self,
        reports: Sequence[SecurityScanReport],
        threshold: str | None,
        blocked: bool,
    ) -> str:
        if not reports:
            return "No security toolchains executed."
        lines: list[str] = []
        for report in reports:
            lines.append(report.describe())
        highest = None
        for report in reports:
            if report.highest_severity is None:
                continue
            if highest is None or _SEVERITY_ORDER.get(report.highest_severity, 0) > _SEVERITY_ORDER.get(highest, 0):
                highest = report.highest_severity
        threshold_display = threshold.upper() if threshold else "NONE"
        highest_display = highest.upper() if highest else "NONE"
        conclusion = f"Highest severity detected: {highest_display}."
        if threshold:
            conclusion += f" Threshold for blocking: {threshold_display}."
        if blocked:
            conclusion += " Blocking workflow.";
        lines.append(conclusion)
        return "\n".join(lines)

    def _build_default_toolchains(self) -> dict[str, SecurityToolchain]:
        toolchains: dict[str, SecurityToolchain] = {}
        toolchains["dependencies"] = SecurityToolchain(
            name="dependencies",
            category="dependency_scan",
            command=(
                "trivy",
                "fs",
                "--security-checks",
                "vuln",
                "--format",
                "json",
                ".",
            ),
            cache_paths=("~/.cache/trivy",),
            cache_description="Trivy vulnerability database",
        )
        toolchains["static"] = SecurityToolchain(
            name="static",
            category="static_analysis",
            command=("semgrep", "--config", "auto", "--json"),
        )
        toolchains["secrets"] = SecurityToolchain(
            name="secrets",
            category="secret_scanning",
            command=("gitleaks", "detect", "--report-format", "json", "--no-git"),
        )
        toolchains["sbom"] = SecurityToolchain(
            name="sbom",
            category="sbom_export",
            command=(
                "trivy",
                "sbom",
                "--format",
                "cyclonedx",
                "--output",
                "sbom.json",
                ".",
            ),
            artifacts=("sbom.json",),
        )
        return toolchains
