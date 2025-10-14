import importlib
import sys
import types
from typing import Any, Dict, List


def _configure_planner(tmp_path, monkeypatch):
    stub = types.ModuleType("lmstudio")

    class ToolFunctionDef:
        def __init__(self, name: str, description: str = "", parameters=None, implementation=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}
            self.implementation = implementation

    stub.ToolFunctionDef = ToolFunctionDef
    monkeypatch.setitem(sys.modules, "lmstudio", stub)

    planner_module = importlib.import_module("internal.tools.planner")
    planner_dir = tmp_path / ".planner"
    state_path = planner_dir / "plans.json"
    monkeypatch.setattr(planner_module, "PLANNER_DIR", str(planner_dir), raising=False)
    monkeypatch.setattr(planner_module, "STATE_PATH", str(state_path), raising=False)
    return planner_module


def _extract_titles(plan_payload: Dict[str, Any]) -> List[str]:
    root_ids = plan_payload.get("root", [])
    tasks = plan_payload.get("tasks", {})
    return [tasks[tid]["title"] for tid in root_ids if tid in tasks]


def test_update_plan_uses_provided_step_titles(tmp_path, monkeypatch):
    planner_module = _configure_planner(tmp_path, monkeypatch)
    created_plan = planner_module.planner("plan_create", plan_name="Regression Plan")
    assert created_plan["status"] == "success"
    plan_id = created_plan["plan"]["id"]

    steps = [
        {"title": "Document requirements", "status": "completed"},
        {"step": "Implement feature"},
        "Ship release",
    ]

    response = planner_module.planner(
        "update_plan",
        plan_id=plan_id,
        explanation="Regression coverage for update_plan",
        plan=steps,
    )

    assert response["status"] == "success"
    created_titles = [task["title"] for task in response["created"]]
    assert created_titles == ["Document requirements", "Implement feature", "Ship release"]

    plan_response = planner_module.planner("plan_get", plan_id=plan_id)
    assert plan_response["status"] == "success"
    active_plan = plan_response["plan"]
    assert _extract_titles(active_plan) == created_titles
