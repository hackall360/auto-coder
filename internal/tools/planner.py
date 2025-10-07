import os
import json
import time
import uuid
from typing import Optional, List, Dict, Any
from lmstudio import ToolFunctionDef


PLANNER_DIR = os.path.join("internal", ".planner")
STATE_PATH = os.path.join(PLANNER_DIR, "plans.json")


def _now_ts() -> float:
    return time.time()


def _ensure_dirs():
    os.makedirs(PLANNER_DIR, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(STATE_PATH):
        state = {"plans": [], "active_plan_id": None}
        _save_state(state)
        return state
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"plans": [], "active_plan_id": None}


def _save_state(state: Dict[str, Any]):
    _ensure_dirs()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _find_plan(state: Dict[str, Any], plan_id: Optional[str] = None, plan_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if plan_id:
        for p in state["plans"]:
            if p["id"] == plan_id:
                return p
    if plan_name:
        for p in state["plans"]:
            if p.get("name") == plan_name:
                return p
    if not plan_id and not plan_name and state.get("active_plan_id"):
        return _find_plan(state, plan_id=state["active_plan_id"])
    return None


def _create_plan(name: str) -> Dict[str, Any]:
    pid = str(uuid.uuid4())
    ts = _now_ts()
    return {
        "id": pid,
        "name": name or f"Plan {pid[:8]}",
        "status": "active",
        "created_at": ts,
        "updated_at": ts,
        "root": [],  # list of task ids at root
        "tasks": {},  # id -> task
    }


def _new_task(title: str, description: Optional[str] = None, parent_id: Optional[str] = None) -> Dict[str, Any]:
    tid = str(uuid.uuid4())
    ts = _now_ts()
    return {
        "id": tid,
        "title": title or f"Task {tid[:8]}",
        "description": description or "",
        "status": "pending",  # pending|in_progress|completed|blocked|canceled
        "priority": "normal",  # low|normal|high|urgent
        "due_date": None,
        "tags": [],
        "parent_id": parent_id,
        "children": [],
        "dependencies": [],
        "order": 0,
        "assignee": None,
        "progress": 0,
        "estimate": None,
        "notes": [],
        "metadata": {},
        "created_at": ts,
        "updated_at": ts,
    }


def planner(
    operation: str,
    # Plan addressing
    plan_id: Optional[str] = None,
    plan_name: Optional[str] = None,
    # Task addressing
    task_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    # Task data
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    tags_add: Optional[List[str]] = None,
    tags_remove: Optional[List[str]] = None,
    assignee: Optional[str] = None,
    progress: Optional[int] = None,
    estimate: Optional[float] = None,
    note: Optional[str] = None,
    # Structure
    new_parent_id: Optional[str] = None,
    order: Optional[int] = None,
    dependency_id: Optional[str] = None,
    # Search/filters
    query: Optional[str] = None,
    status_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    assignee_filter: Optional[str] = None,
    # Bulk/compat
    explanation: Optional[str] = None,
    plan: Optional[List[Dict[str, Any]]] = None,
):
    """Planner tool with branching tasks and full plan management.

    Operations:
    - plan_create: plan_name
    - plan_rename: plan_id|plan_name, title
    - plan_list: (no params)
    - plan_set_active: plan_id|plan_name
    - plan_get: plan_id|plan_name
    - plan_archive|plan_unarchive: plan_id|plan_name

    - task_add: title[, description, parent_id]
    - task_update: task_id[, title, description, status, priority, due_date, assignee, progress, estimate]
    - task_set_status: task_id, status
    - task_complete: task_id
    - task_block|task_unblock: task_id
    - task_delete: task_id
    - task_move: task_id, new_parent_id[, order]
    - task_reorder: task_id, order
    - task_add_dependency: task_id, dependency_id
    - task_remove_dependency: task_id, dependency_id
    - task_add_note: task_id, note
    - task_tags: task_id, [tags_add], [tags_remove]
    - task_get: task_id
    - task_list: [query, status_filter, tag_filter, assignee_filter]
    - update_plan: explanation, plan (list of step dicts)
    """
    try:
        op = (operation or "").strip().lower()
        state = _load_state()

        # Plan operations
        if op == "plan_create":
            if not plan_name:
                return {"status": "error", "message": "'plan_name' is required"}
            plan = _create_plan(plan_name)
            state["plans"].append(plan)
            state["active_plan_id"] = plan["id"]
            _save_state(state)
            return {"status": "success", "plan": plan}

        if op == "plan_rename":
            if not title:
                return {"status": "error", "message": "'title' (new plan name) is required"}
            plan = _find_plan(state, plan_id, plan_name)
            if not plan:
                return {"status": "error", "message": "Plan not found"}
            plan["name"] = title
            plan["updated_at"] = _now_ts()
            _save_state(state)
            return {"status": "success", "plan": plan}

        if op == "plan_list":
            return {"status": "success", "plans": [{k: p[k] for k in ("id", "name", "status", "created_at", "updated_at")} for p in state["plans"]], "active_plan_id": state.get("active_plan_id")}

        if op == "plan_set_active":
            plan = _find_plan(state, plan_id, plan_name)
            if not plan:
                return {"status": "error", "message": "Plan not found"}
            state["active_plan_id"] = plan["id"]
            _save_state(state)
            return {"status": "success", "active_plan_id": plan["id"], "plan": plan}

        if op == "plan_get":
            plan = _find_plan(state, plan_id, plan_name)
            if not plan:
                return {"status": "error", "message": "Plan not found"}
            return {"status": "success", "plan": plan}

        if op in ("plan_archive", "plan_unarchive"):
            plan = _find_plan(state, plan_id, plan_name)
            if not plan:
                return {"status": "error", "message": "Plan not found"}
            plan["status"] = "archived" if op == "plan_archive" else "active"
            plan["updated_at"] = _now_ts()
            _save_state(state)
            return {"status": "success", "plan": plan}

        # Must have an active plan for task ops
        plan = _find_plan(state, plan_id, plan_name)
        if not plan:
            # If none exists, create a default plan when performing task operations
            default_name = plan_name or "Default Plan"
            plan = _create_plan(default_name)
            state["plans"].append(plan)
            state["active_plan_id"] = plan["id"]
            _save_state(state)

        tasks = plan["tasks"]

        def get_task(tid: str) -> Optional[Dict[str, Any]]:
            return tasks.get(tid)

        def touch_plan():
            plan["updated_at"] = _now_ts()

        # Bulk plan import/update (compat with old update_plan)
        if op == "update_plan":
            created: List[Dict[str, Any]] = []
            if explanation:
                plan.setdefault("log", []).append({"ts": _now_ts(), "explanation": explanation})
            for step in (plan or []):
                title_val = None
                status_val = None
                if isinstance(step, dict):
                    title_val = step.get("step") or step.get("title") or ""
                    status_val = step.get("status")
                else:
                    title_val = str(step)
                if not title_val:
                    continue
                t = _new_task(title_val, None, None)
                plan["tasks"][t["id"]] = t
                plan["root"].append(t["id"])
                if status_val:
                    t["status"] = status_val
                t["updated_at"] = _now_ts()
                created.append(t)
            plan["updated_at"] = _now_ts()
            _save_state(state)
            return {"status": "success", "created": created}

        # Task creation
        if op == "task_add":
            if not title:
                return {"status": "error", "message": "'title' is required for task_add"}
            t = _new_task(title, description, parent_id)
            tasks[t["id"]] = t
            if parent_id and parent_id in tasks:
                tasks[parent_id]["children"].append(t["id"])
            else:
                plan["root"].append(t["id"])
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_update":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' not found"}
            t = tasks[task_id]
            for k, v in (
                ("title", title),
                ("description", description),
                ("status", status),
                ("priority", priority),
                ("due_date", due_date),
                ("assignee", assignee),
                ("progress", progress),
                ("estimate", estimate),
            ):
                if v is not None:
                    t[k] = v
            if tags_add:
                t["tags"] = sorted(set(t.get("tags", [])) | set(tags_add))
            if tags_remove:
                t["tags"] = [x for x in t.get("tags", []) if x not in set(tags_remove)]
            if note:
                t.setdefault("notes", []).append({"ts": _now_ts(), "note": note})
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_set_status":
            if not task_id or task_id not in tasks or not status:
                return {"status": "error", "message": "'task_id' and 'status' are required"}
            t = tasks[task_id]
            t["status"] = status
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_complete":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            t = tasks[task_id]
            t["status"] = "completed"
            t["progress"] = 100
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op in ("task_block", "task_unblock"):
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            t = tasks[task_id]
            t["status"] = "blocked" if op == "task_block" else "pending"
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_delete":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            # remove recursively
            def _remove_recursive(tid: str):
                tt = tasks.get(tid)
                if not tt:
                    return
                for cid in list(tt.get("children", [])):
                    _remove_recursive(cid)
                tasks.pop(tid, None)
            # detach from parent or root
            pt = tasks.get(tasks.get(task_id, {}).get("parent_id"))
            if pt:
                pt["children"] = [c for c in pt.get("children", []) if c != task_id]
            else:
                plan["root"] = [c for c in plan.get("root", []) if c != task_id]
            _remove_recursive(task_id)
            touch_plan()
            _save_state(state)
            return {"status": "success", "deleted": task_id}

        if op == "task_move":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            newp = new_parent_id
            # detach from current
            cur = tasks[task_id]
            oldp = cur.get("parent_id")
            if oldp and oldp in tasks:
                tasks[oldp]["children"] = [c for c in tasks[oldp]["children"] if c != task_id]
            else:
                plan["root"] = [c for c in plan["root"] if c != task_id]
            # attach to new parent or root
            if newp:
                if newp not in tasks:
                    return {"status": "error", "message": "new_parent_id not found"}
                tasks[newp]["children"].append(task_id)
                cur["parent_id"] = newp
            else:
                plan["root"].append(task_id)
                cur["parent_id"] = None
            # reordering if requested
            if order is not None:
                sibs = tasks[newp]["children"] if newp else plan["root"]
                if task_id in sibs:
                    sibs.remove(task_id)
                idx = max(0, min(int(order), len(sibs)))
                sibs.insert(idx, task_id)
            cur["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": cur}

        if op == "task_reorder":
            if not task_id or task_id not in tasks or order is None:
                return {"status": "error", "message": "'task_id' and 'order' are required"}
            t = tasks[task_id]
            parent = tasks.get(t.get("parent_id"))
            sibs = parent["children"] if parent else plan["root"]
            if task_id in sibs:
                sibs.remove(task_id)
            idx = max(0, min(int(order), len(sibs)))
            sibs.insert(idx, task_id)
            t["order"] = idx
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_add_dependency":
            if not task_id or not dependency_id or task_id not in tasks or dependency_id not in tasks:
                return {"status": "error", "message": "'task_id' and 'dependency_id' must exist"}
            t = tasks[task_id]
            if dependency_id not in t["dependencies"]:
                t["dependencies"].append(dependency_id)
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_remove_dependency":
            if not task_id or not dependency_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' and 'dependency_id' are required"}
            t = tasks[task_id]
            t["dependencies"] = [d for d in t.get("dependencies", []) if d != dependency_id]
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_add_note":
            if not task_id or task_id not in tasks or not note:
                return {"status": "error", "message": "'task_id' and 'note' are required"}
            t = tasks[task_id]
            t.setdefault("notes", []).append({"ts": _now_ts(), "note": note})
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_tags":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            t = tasks[task_id]
            if tags_add:
                t["tags"] = sorted(set(t.get("tags", [])) | set(tags_add))
            if tags_remove:
                t["tags"] = [x for x in t.get("tags", []) if x not in set(tags_remove)]
            t["updated_at"] = _now_ts()
            touch_plan()
            _save_state(state)
            return {"status": "success", "task": t}

        if op == "task_get":
            if not task_id or task_id not in tasks:
                return {"status": "error", "message": "'task_id' is required"}
            return {"status": "success", "task": tasks[task_id]}

        if op == "task_list":
            q = (query or "").lower()
            out = []
            for t in tasks.values():
                if q and q not in t["title"].lower() and q not in (t.get("description") or "").lower():
                    continue
                if status_filter and t.get("status") != status_filter:
                    continue
                if tag_filter and tag_filter not in t.get("tags", []):
                    continue
                if assignee_filter and t.get("assignee") != assignee_filter:
                    continue
                out.append(t)
            # sort by order then created_at
            out.sort(key=lambda x: (x.get("order", 0), x.get("created_at", 0)))
            return {"status": "success", "tasks": out}

        return {"status": "error", "message": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


planner_tool = ToolFunctionDef(
    name="planner",
    description=(
        "Full planner with branching tasks: create plans, add/update/move tasks, set statuses, dependencies, notes, tags."
    ),
    parameters={
        "operation": str,
        "plan_id": str,
        "plan_name": str,
        "task_id": str,
        "parent_id": str,
        "title": str,
        "description": str,
        "status": str,
        "priority": str,
        "due_date": str,
        "tags_add": list,
        "tags_remove": list,
        "assignee": str,
        "progress": int,
        "estimate": float,
        "note": str,
        "new_parent_id": str,
        "order": int,
        "dependency_id": str,
        "query": str,
        "status_filter": str,
        "tag_filter": str,
        "assignee_filter": str,
        "explanation": str,
        "plan": list,
    },
    implementation=planner,
)
