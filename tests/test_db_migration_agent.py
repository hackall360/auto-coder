from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Mapping

import pytest

# Seed a lightweight lmstudio stub before importing the agent package.
_lmstudio_stub = types.ModuleType("lmstudio")


class _StubChat:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - bootstrap only
        self.messages: list[Any] = []

    @classmethod
    def from_history(cls, history: Any) -> "_StubChat":  # pragma: no cover - bootstrap only
        chat = cls()
        if isinstance(history, list):
            chat.messages = list(history)
        return chat


class _StubToolDef:
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
        implementation: Any | None = None,
    ) -> None:  # pragma: no cover - bootstrap only
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.implementation = implementation


def _stub_llm(*_: Any, **__: Any) -> None:  # pragma: no cover - bootstrap only
    raise RuntimeError("lmstudio stub should be overridden by fixtures")


_lmstudio_stub.Chat = _StubChat
_lmstudio_stub.ToolFunctionDef = _StubToolDef
_lmstudio_stub.llm = _stub_llm
sys.modules.setdefault("lmstudio", _lmstudio_stub)

_psutil_stub = types.ModuleType("psutil")


class _StubProcess:
    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - bootstrap only
        self.pid = 0

    def kill(self) -> None:  # pragma: no cover - bootstrap only
        raise RuntimeError("psutil stub")

    def terminate(self) -> None:  # pragma: no cover - bootstrap only
        raise RuntimeError("psutil stub")

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover - bootstrap only
        raise RuntimeError("psutil stub")

    def as_dict(self, attrs: list[str] | None = None) -> dict[str, Any]:  # pragma: no cover - bootstrap only
        return {}

    def is_running(self) -> bool:  # pragma: no cover - bootstrap only
        return False

    def children(self, recursive: bool = False) -> list["_StubProcess"]:  # pragma: no cover - bootstrap only
        return []


class _PsutilNoSuchProcess(RuntimeError):  # pragma: no cover - bootstrap only
    pass


class _PsutilTimeout(RuntimeError):  # pragma: no cover - bootstrap only
    pass


def _process_iter(*_: Any, **__: Any):  # pragma: no cover - bootstrap only
    return []


_psutil_stub.Process = _StubProcess
_psutil_stub.NoSuchProcess = _PsutilNoSuchProcess
_psutil_stub.TimeoutExpired = _PsutilTimeout
_psutil_stub.process_iter = _process_iter
sys.modules.setdefault("psutil", _psutil_stub)

from agents.db_migration import DBMigrationAgent, EphemeralDatabaseSpec
from agents.manager import ManagerAgent, TaskBudget
from agents.repo_context import RepoContextAgent
from agents.runner import RunnerAgent


class SessionStub:
    def __init__(self) -> None:
        self.rounds = []

    def add_round_hooks(self, *, on_round_start=None, on_round_end=None) -> None:  # noqa: D401 - test stub
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end


@pytest.fixture()
def dummy_framework() -> Mapping[str, Any]:
    generate_script = (
        "import pathlib, sys; "
        "root = pathlib.Path('migrations'); "
        "root.mkdir(exist_ok=True); "
        "(root / (sys.argv[1] + '.sql')).write_text('generated')"
    )
    apply_script = (
        "import pathlib, sys; "
        "pathlib.Path('applied.log').write_text(sys.argv[1])"
    )
    return {
        "migrations_dir": "migrations",
        "generate": ("python", "-c", generate_script, "{name}"),
        "apply": ("python", "-c", apply_script, "{name}"),
    }


@pytest.fixture()
def repo_context(tmp_path: Path) -> RepoContextAgent:
    rc = RepoContextAgent(str(tmp_path), auto_refresh=False)
    yield rc
    rc.stop_background_refresh()


def build_agent(repo_context: RepoContextAgent, framework: Mapping[str, Any]) -> DBMigrationAgent:
    runner = RunnerAgent(repo_root=repo_context.repo_root)
    return DBMigrationAgent(repo_context, runner=runner, frameworks={"dummy": framework})


def test_plan_discovers_existing_migrations(repo_context: RepoContextAgent, dummy_framework: Mapping[str, Any]) -> None:
    migrations = Path(repo_context.repo_root) / "migrations"
    migrations.mkdir(parents=True, exist_ok=True)
    (migrations / "001_initial.sql").write_text("create table")
    (migrations / "010_second.sql").write_text("alter table")

    agent = build_agent(repo_context, dummy_framework)
    plan = agent.plan_migration("dummy")

    assert plan.framework == "dummy"
    assert [record.name for record in plan.existing] == [
        "001_initial.sql",
        "010_second.sql",
    ]
    assert plan.command_template[0] == "python"


def test_run_migration_generates_and_applies(repo_context: RepoContextAgent, dummy_framework: Mapping[str, Any]) -> None:
    migrations = Path(repo_context.repo_root) / "migrations"
    migrations.mkdir(parents=True, exist_ok=True)
    (migrations / "000_base.sql").write_text("baseline")

    agent = build_agent(repo_context, dummy_framework)
    plan = agent.plan_migration("dummy")
    result = agent.run_migration(plan, migration_name="feature", apply=True)

    assert (migrations / "feature.sql").exists()
    assert result.generated == ("migrations/feature.sql",)
    assert result.applied is True
    assert Path(repo_context.repo_root, "applied.log").read_text() == "feature"
    summary = agent.format_result(result)
    assert "feature.sql" in summary


def test_ephemeral_cleanup_on_failure(repo_context: RepoContextAgent) -> None:
    failing_framework = {
        "migrations_dir": "migrations",
        "generate": ("python", "-c", "import sys; sys.exit(1)"),
    }
    agent = build_agent(repo_context, failing_framework)
    agent.register_ephemeral_database(
        EphemeralDatabaseSpec(
            name="compose",
            setup=(
                "python",
                "-c",
                "import pathlib; pathlib.Path('db.lock').write_text('lock')",
            ),
            teardown=(
                "python",
                "-c",
                "import pathlib; path = pathlib.Path('db.lock'); "
                "path.exists() and path.unlink()",
            ),
        )
    )
    plan = agent.plan_migration("dummy")

    with pytest.raises(RuntimeError):
        agent.run_migration(plan, migration_name="broken", ephemeral="compose")

    assert not Path(repo_context.repo_root, "db.lock").exists()


def test_manager_db_migration_dispatch(repo_context: RepoContextAgent, dummy_framework: Mapping[str, Any]) -> None:
    agent = build_agent(repo_context, dummy_framework)
    budget = TaskBudget(name="migration", limit=2)
    captured = []
    manager = ManagerAgent(
        session=SessionStub(),
        status_callback=captured.append,
        repo_context=repo_context,
        db_migration_agent=agent,
    )

    task = {"framework": "dummy", "apply": True, "migration_name": "manager"}
    summary, structured = manager._run_db_migration_task("migration", task, {}, budget=budget)

    assert "manager.sql" in summary
    assert structured is not None
    assert budget.consumed == pytest.approx(1.0)
    assert any(update.kind == "db_migration_plan" for update in captured)
    assert any(update.kind == "db_migration_result" for update in captured)
    assert Path(repo_context.repo_root, "migrations", "manager.sql").exists()
