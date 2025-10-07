import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import inspect
import threading
import queue


@dataclass
class DAGNode:
    """A single node in a DAG.

    - name: unique identifier for the node
    - func: a callable that produces the node's value. It can return any value.
      The callable may declare optional parameters:
        - `context` or `ctx`: receives a DAGContext object
        - `inputs` or `_inputs`: receives a dict of upstream results keyed by node name
      In all cases, any `params` are also supplied as keyword args if supported.
    - deps: names of upstream nodes this node depends on
    - params: constant keyword arguments provided to the callable
    - value: if provided together with `is_constant=True`, the node acts as a constant
    - is_constant: when True, the node's value is the given `value` and func is ignored
    - retries: number of times to retry the callable on failure
    - retry_delay: seconds to wait between retries
    - timeout: optional timeout in seconds for this node. If exceeded, the attempt fails
    - metadata: free-form metadata for higher-level orchestration
    """

    name: str
    func: Optional[Callable[..., Any]] = None
    deps: Set[str] = field(default_factory=set)
    params: Dict[str, Any] = field(default_factory=dict)
    value: Any = None
    is_constant: bool = False
    retries: int = 0
    retry_delay: float = 0.0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRunResult:
    status: str  # pending|running|success|failed|skipped|canceled
    attempts: int = 0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    traceback: Optional[str] = None
    skipped_reason: Optional[str] = None


class DAGContext:
    """Execution-time context passed to node functions when requested.

    Attributes:
    - dag: the DAG instance
    - node: the DAGNode being executed
    - inputs: dict of upstream results {dep_name: value}
    - params: the node's params (shallow copy)
    - attempt: 1-based attempt counter for retries
    """

    def __init__(self, dag: "DAG", node: DAGNode, inputs: Dict[str, Any], params: Dict[str, Any], attempt: int) -> None:
        self.dag = dag
        self.node = node
        self.inputs = dict(inputs)
        self.params = dict(params)
        self.attempt = attempt


class DAG:
    """A lightweight, dependency-aware task graph executor.

    Usage example:
        dag = DAG()
        dag.add_node("a", value=2, is_constant=True)
        dag.add_node("b", value=3, is_constant=True)

        def add(inputs, factor: int = 1):
            return (inputs["a"] + inputs["b"]) * factor

        dag.add_node("c", func=add, deps=["a", "b"], params={"factor": 10})
        results = dag.run(targets=["c"])  # {"c": 50}

    Notes:
    - Node callables may accept `context` (or `ctx`) to receive a DAGContext.
    - Node callables may accept `inputs` (or `_inputs`) to receive upstream results.
    - `params` are passed as keyword args when supported by the callable.
    - When `timeout` expires, the attempt fails but the underlying thread may continue;
      any late result is ignored. Retries may be applied if configured.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, DAGNode] = {}
        self._succ: Dict[str, Set[str]] = {}  # adjacency list: node -> set(children)
        self._pred: Dict[str, Set[str]] = {}  # reverse adjacency: node -> set(parents)

    # ---------------------- Graph construction ----------------------
    def add_node(
        self,
        name: str,
        func: Optional[Callable[..., Any]] = None,
        *,
        deps: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        value: Any = None,
        is_constant: bool = False,
        retries: int = 0,
        retry_delay: float = 0.0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        node = DAGNode(
            name=name,
            func=func,
            deps=set(deps or []),
            params=dict(params or {}),
            value=value,
            is_constant=is_constant,
            retries=max(0, int(retries or 0)),
            retry_delay=float(retry_delay or 0.0),
            timeout=timeout,
            metadata=dict(metadata or {}),
        )
        # Register
        self._nodes[name] = node
        self._succ.setdefault(name, set())
        self._pred.setdefault(name, set())
        # Create edges from deps -> name
        for d in node.deps:
            if d not in self._nodes:
                # allow forward-declared deps by touching maps
                self._succ.setdefault(d, set())
                self._pred.setdefault(d, set())
            self._succ.setdefault(d, set()).add(name)
            self._pred[name].add(d)

    def add_edge(self, upstream: str, downstream: str) -> None:
        """Add a dependency edge `upstream -> downstream`.
        Nodes are implicitly created as empty holders if not present yet.
        """
        if downstream not in self._pred:
            self._pred[downstream] = set()
        if upstream not in self._succ:
            self._succ[upstream] = set()
        self._succ[upstream].add(downstream)
        self._pred[downstream].add(upstream)
        # Ensure node dict entries exist
        self._nodes.setdefault(upstream, DAGNode(name=upstream))
        self._nodes.setdefault(downstream, DAGNode(name=downstream))

    def set_params(self, name: str, params: Dict[str, Any]) -> None:
        n = self._require_node(name)
        n.params = dict(params)

    def set_constant(self, name: str, value: Any) -> None:
        n = self._require_node(name)
        n.value = value
        n.is_constant = True
        n.func = None

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    # ---------------------- Graph analysis ----------------------
    def _require_node(self, name: str) -> DAGNode:
        if name not in self._nodes:
            raise KeyError(f"Unknown node '{name}'")
        return self._nodes[name]

    def parents(self, name: str) -> Set[str]:
        return set(self._pred.get(name, set()))

    def children(self, name: str) -> Set[str]:
        return set(self._succ.get(name, set()))

    def leaves(self) -> List[str]:
        return [n for n in self._nodes if not self._succ.get(n)]

    def roots(self) -> List[str]:
        return [n for n in self._nodes if not self._pred.get(n)]

    def _subgraph_needed_for(self, targets: Optional[Sequence[str]]) -> Set[str]:
        if not targets:
            return set(self._nodes.keys())
        needed: Set[str] = set()
        stack: List[str] = list(targets)
        while stack:
            cur = stack.pop()
            if cur in needed:
                continue
            needed.add(cur)
            for p in self._pred.get(cur, ()):  # pull parents recursively
                if p not in needed:
                    stack.append(p)
        return needed

    def validate_acyclic(self) -> None:
        """Raises ValueError if a cycle is found."""
        self._ensure_no_cycles()

    def topological_order(self, targets: Optional[Sequence[str]] = None) -> List[str]:
        """Return a topological ordering for the subgraph needed for `targets`.
        If targets is None, order the full graph.
        """
        needed = self._subgraph_needed_for(targets)
        indeg: Dict[str, int] = {n: 0 for n in needed}
        for n in needed:
            indeg[n] = len(self._pred.get(n, set()) & needed)
        q = [n for n, d in indeg.items() if d == 0]
        order: List[str] = []
        i = 0
        while i < len(q):
            n = q[i]
            i += 1
            order.append(n)
            for c in self._succ.get(n, set()):
                if c not in needed:
                    continue
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(needed):
            cycle = self._find_cycle(needed)
            raise ValueError(f"Cycle detected: {' -> '.join(cycle)}")
        return order

    def _find_cycle(self, within: Optional[Set[str]] = None) -> List[str]:
        """Return one cycle path if exists, else []."""
        color: Dict[str, int] = {}  # 0 white, 1 gray, 2 black
        parent: Dict[str, Optional[str]] = {}
        nodes = within or set(self._nodes.keys())

        def dfs(u: str) -> Optional[Tuple[str, str]]:
            color[u] = 1
            for v in self._succ.get(u, set()):
                if v not in nodes:
                    continue
                if color.get(v, 0) == 0:
                    parent[v] = u
                    res = dfs(v)
                    if res:
                        return res
                elif color.get(v) == 1:
                    return (u, v)  # back edge
            color[u] = 2
            return None

        for n in nodes:
            if color.get(n, 0) == 0:
                parent[n] = None
                be = dfs(n)
                if be:
                    u, v = be
                    # reconstruct cycle v -> ... -> u -> v
                    path = [v]
                    cur = u
                    while cur != v and cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    path.append(v)
                    path.reverse()
                    return path
        return []

    def _ensure_no_cycles(self, targets: Optional[Sequence[str]] = None) -> None:
        needed = self._subgraph_needed_for(targets)
        if not needed:
            return
        order = self.topological_order(targets)
        # If topological_order returned without error, graph is acyclic for these nodes
        _ = order

    # ---------------------- Execution ----------------------
    def run(
        self,
        *,
        targets: Optional[Sequence[str]] = None,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
        propagate_skip: bool = True,
        return_scope: str = "targets",  # one of: targets|all|leaves
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the DAG and return results.

        Params:
        - targets: subset of nodes to compute. Their dependencies are included.
        - max_workers: degree of parallelism (threads). Defaults to min(32, cpu+4).
        - fail_fast: if True, stop scheduling new work after first failure.
        - propagate_skip: if True, downstream of failed/skipped nodes are skipped.
        - return_scope: which results to return: 'targets' | 'all' | 'leaves'.
        - progress: optional callback called on notable events with a dict payload.

        Returns dict mapping node name to computed result based on `return_scope`.
        """
        if max_workers is None or max_workers <= 0:
            try:
                import os

                cpu = os.cpu_count() or 4
            except Exception:
                cpu = 4
            max_workers = min(32, cpu + 4)

        needed = self._subgraph_needed_for(targets)
        if not needed:
            return {}
        # Validate acyclic for the subgraph
        self._ensure_no_cycles(list(needed))

        # Build indegrees and initial ready set
        indeg: Dict[str, int] = {n: 0 for n in needed}
        for n in needed:
            indeg[n] = len(self._pred.get(n, set()) & needed)
        ready: List[str] = [n for n in needed if indeg[n] == 0]

        # Execution state
        states: Dict[str, NodeRunResult] = {n: NodeRunResult(status="pending") for n in needed}
        results: Dict[str, Any] = {}
        failures: Set[str] = set()
        skipped: Set[str] = set()
        # Legacy placeholder from previous executor; no longer used
        canceled = False

        def notify(evt: str, **kw: Any) -> None:
            if progress:
                payload = {"event": evt, **kw}
                try:
                    progress(payload)
                except Exception:
                    # Progress callback errors are ignored to avoid interfering with execution
                    pass

        # Threaded execution engine avoiding concurrent.futures to sidestep local logging.py shadowing stdlib
        lock = threading.Lock()
        ready_q: "queue.Queue[str]" = queue.Queue()
        total = len(needed)
        completed = 0
        all_done = threading.Event()

        # Seed ready queue
        for n in ready:
            ready_q.put(n)

        def mark_done_and_release(node_name: str) -> None:
            nonlocal completed
            # Release children: decrement indeg and enqueue children that reach zero
            for c in self._succ.get(node_name, set()):
                if c in needed:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        ready_q.put(c)
            completed += 1
            notify("progress", completed=completed, total=total)
            if completed >= total:
                all_done.set()

        def cascade_skip(start: str, reason: str) -> None:
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt in self._succ.get(cur, set()):
                    if nxt not in needed:
                        continue
                    st = states.get(nxt)
                    if st and st.status in ("success", "failed", "skipped", "running"):
                        continue
                    states[nxt] = NodeRunResult(status="skipped", skipped_reason=reason)
                    skipped.add(nxt)
                    notify("node_skipped", node=nxt, reason=reason)
                    # Consider the node "done" and propagate further
                    mark_done_and_release(nxt)
                    stack.append(nxt)

        def run_callable(n: DAGNode, inputs_map: Dict[str, Any]) -> Tuple[str, NodeRunResult, Any]:
            node_name = n.name
            st = states[node_name] = NodeRunResult(status="running", started_at=time.time(), attempts=0)
            notify("node_start", node=node_name)

            # Constant node
            if n.is_constant:
                st.attempts = 1
                val = n.value
                st.ended_at = time.time()
                st.duration = (st.ended_at - st.started_at) if st.started_at else None
                st.status = "success"
                return node_name, st, val

            if n.func is None:
                st.attempts = 1
                st.status = "failed"
                st.error = "No func provided for non-constant node"
                st.ended_at = time.time()
                st.duration = (st.ended_at - st.started_at) if st.started_at else None
                return node_name, st, None

            # Build call kwargs via introspection
            want_ctx = False
            want_inputs = False
            call_kwargs: Dict[str, Any] = {}
            try:
                sig = inspect.signature(n.func)
                params = sig.parameters
                if any(k in params for k in ("context", "ctx")):
                    want_ctx = True
                if any(k in params for k in ("inputs", "_inputs")):
                    want_inputs = True
                accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                for k, v in n.params.items():
                    if accepts_kwargs or k in params:
                        call_kwargs[k] = v
            except Exception:
                for k, v in n.params.items():
                    call_kwargs[k] = v

            attempts = 0
            last_exc: Optional[BaseException] = None
            last_tb: Optional[str] = None

            while True:
                attempts += 1
                st.attempts = attempts
                ctx = DAGContext(self, n, inputs_map, n.params, attempts)
                try:
                    if want_ctx and want_inputs:
                        kwargs = dict(call_kwargs)
                        kwargs.update({"context": ctx, "inputs": inputs_map})
                        val = n.func(**kwargs)  # type: ignore[misc]
                    elif want_ctx:
                        kwargs = dict(call_kwargs)
                        kwargs.update({"context": ctx})
                        val = n.func(**kwargs)  # type: ignore[misc]
                    elif want_inputs:
                        kwargs = dict(call_kwargs)
                        kwargs.update({"inputs": inputs_map})
                        val = n.func(**kwargs)  # type: ignore[misc]
                    else:
                        val = n.func(**call_kwargs)  # type: ignore[misc]
                    st.status = "success"
                    st.ended_at = time.time()
                    st.duration = (st.ended_at - st.started_at) if st.started_at else None
                    return node_name, st, val
                except Exception as e:
                    last_exc = e
                    last_tb = traceback.format_exc()
                    if attempts <= (n.retries or 0):
                        if n.retry_delay:
                            time.sleep(n.retry_delay)
                        continue
                    st.status = "failed"
                    st.exception = last_exc
                    st.traceback = last_tb
                    st.error = f"{type(e).__name__}: {e}"
                    st.ended_at = time.time()
                    st.duration = (st.ended_at - st.started_at) if st.started_at else None
                    return node_name, st, None

        def run_with_timeout(n: DAGNode, inputs_map: Dict[str, Any]) -> Tuple[str, NodeRunResult, Any]:
            if not n.timeout or n.timeout <= 0:
                return run_callable(n, inputs_map)

            res_box: Dict[str, Any] = {}
            done_event = threading.Event()

            def target():
                try:
                    res_box["triple"] = run_callable(n, inputs_map)
                finally:
                    done_event.set()

            t = threading.Thread(target=target, daemon=True)
            t.start()
            finished = done_event.wait(timeout=n.timeout)
            if finished:
                return res_box.get("triple")  # type: ignore[return-value]
            # Timed out
            st = NodeRunResult(status="failed", error="timeout", attempts=1, started_at=None, ended_at=time.time())
            st.duration = None
            return n.name, st, None

        def worker() -> None:
            nonlocal canceled
            while True:
                try:
                    node_name = ready_q.get(timeout=0.1)
                except queue.Empty:
                    if all_done.is_set() or (fail_fast and failures and ready_q.empty()):
                        break
                    else:
                        continue
                if node_name is None:  # sentinel
                    ready_q.task_done()
                    break
                with lock:
                    if canceled:
                        ready_q.task_done()
                        continue
                    # Check upstream failures/skips
                    if propagate_skip:
                        if any(p in failures for p in self._pred.get(node_name, set())) or any(
                            p in skipped for p in self._pred.get(node_name, set())
                        ):
                            states[node_name] = NodeRunResult(status="skipped", skipped_reason="upstream")
                            skipped.add(node_name)
                            notify("node_skipped", node=node_name, reason="upstream")
                            mark_done_and_release(node_name)
                            ready_q.task_done()
                            continue

                # Build inputs map (outside lock but inputs are immutable once set)
                inputs_map = {p: results[p] for p in self._pred.get(node_name, set()) if p in results}
                n = self._nodes[node_name]
                name, st, val = run_with_timeout(n, inputs_map)

                with lock:
                    states[node_name] = st
                    if st.status == "success":
                        results[node_name] = val
                        notify("node_success", node=node_name, duration=st.duration)
                        mark_done_and_release(node_name)
                    elif st.status == "skipped":
                        skipped.add(node_name)
                        notify("node_skipped", node=node_name, reason=st.skipped_reason or "unknown")
                        mark_done_and_release(node_name)
                    else:
                        failures.add(node_name)
                        notify("node_failed", node=node_name, error=st.error)
                        mark_done_and_release(node_name)
                        if propagate_skip:
                            cascade_skip(node_name, reason="upstream_failed")
                        if fail_fast:
                            canceled = True

                ready_q.task_done()

        # Start workers
        workers = [threading.Thread(target=worker, daemon=True) for _ in range(max_workers)]
        for t in workers:
            t.start()

        # Wait for all work to finish
        all_done.wait()

        # Join workers
        for _ in workers:
            ready_q.put(None)  # sentinel
        for t in workers:
            t.join(timeout=1.0)

        # Decide which results to return
        if return_scope == "all":
            out_nodes = list(needed)
        elif return_scope == "leaves":
            out_nodes = [n for n in needed if not (self._succ.get(n, set()) & needed)]
        else:  # "targets"
            out_nodes = list(targets or [n for n in needed if not (self._succ.get(n, set()) & needed)])

        return {n: results.get(n) for n in out_nodes if n in results}

    # ---------------------- Utilities ----------------------
    def to_dot(self, targets: Optional[Sequence[str]] = None) -> str:
        """Return a Graphviz DOT representation of the (sub)graph."""
        needed = self._subgraph_needed_for(targets)
        lines = ["digraph DAG {"]
        for n in needed:
            label = n
            node = self._nodes.get(n)
            if node and node.is_constant:
                label = f"{n}\\nconst"
            lines.append(f'  "{n}" [label="{label}"];')
        for u in needed:
            for v in self._succ.get(u, set()):
                if v in needed:
                    lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines)



__all__ = [
    "DAG",
    "DAGNode",
    "DAGContext",
    "NodeRunResult",
]
