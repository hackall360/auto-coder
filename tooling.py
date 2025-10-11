from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from lmstudio import ToolFunctionDef


ToolCallable = Callable[..., Any]


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


def _is_tool_function_def(obj: Any) -> bool:
    return isinstance(obj, ToolFunctionDef)


def _normalize_parameters(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    if isinstance(params, MutableMapping):
        return dict(params)
    return {str(key): value for key, value in dict(params).items()}


def _callable_signature_parameters(func: ToolCallable) -> dict[str, Any]:
    signature = inspect.signature(func)
    parameters: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError(
                f"Tool callable {func.__name__}() uses variadic parameters which are"
                " not supported."
            )
        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            raise ValueError(
                "Tool callables must provide type annotations for every argument "
                f"(missing annotation for parameter '{parameter.name}' in {func.__name__}())."
            )
        parameters[parameter.name] = annotation
    return parameters


def _callable_description(func: ToolCallable, explicit: str | None) -> str:
    if explicit is not None:
        explicit = explicit.strip()
        if explicit:
            return explicit
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(
            f"Tool callable {func.__name__}() must have a docstring describing its behaviour."
        )
    return doc.strip()


@dataclass(slots=True)
class ToolSpec:
    """Normalized representation of a tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: ToolCallable | None
    source: Any
    tool_type: str = "function"
    payload_overrides: dict[str, Any] | None = None

    def _merge_payload(self, base: Mapping[str, Any]) -> dict[str, Any]:
        def _merge_dict(
            lhs: Mapping[str, Any], rhs: Mapping[str, Any]
        ) -> dict[str, Any]:
            merged: dict[str, Any] = dict(lhs)
            for key, value in rhs.items():
                if (
                    key in merged
                    and isinstance(merged[key], Mapping)
                    and isinstance(value, Mapping)
                ):
                    merged[key] = _merge_dict(merged[key], value)
                else:
                    merged[key] = value
            return merged

        if not self.payload_overrides:
            return dict(base)
        if not isinstance(self.payload_overrides, Mapping):
            raise TypeError("payload_overrides must be a mapping if provided")
        return _merge_dict(base, self.payload_overrides)

    def to_payload(self) -> dict[str, Any]:
        """Return the structure expected by :meth:`model.act`."""

        if self.tool_type == "function":
            if not callable(self.implementation):
                raise ValueError(
                    f"Tool '{self.name}' declares type 'function' but is missing a callable implementation"
                )
            base_payload: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": dict(self.parameters),
                    "implementation": self.implementation,
                },
            }
            return self._merge_payload(base_payload)

        if self.tool_type == "mcp":
            descriptor = dict(self.parameters)
            label = (descriptor.get("label") or self.name).strip()
            if not label:
                raise ValueError("MCP tool specifications require a non-empty label")
            base_payload: dict[str, Any] = {
                "type": "mcp",
                "server_label": label,
            }
            description = self.description or descriptor.get("description")
            if description:
                base_payload["description"] = description
            url = descriptor.get("server_url") or descriptor.get("url")
            if url:
                base_payload["server_url"] = url
            allowed = descriptor.get("allowed_tools")
            if allowed:
                if isinstance(allowed, str):
                    allowed_list = [allowed]
                else:
                    allowed_list = list(allowed)
                base_payload["allowed_tools"] = allowed_list
            headers = descriptor.get("headers")
            if headers:
                base_payload["headers"] = dict(headers)
            metadata = descriptor.get("metadata")
            if metadata:
                base_payload["metadata"] = dict(metadata)
            server_type = descriptor.get("server_type") or descriptor.get("type")
            if server_type:
                base_payload["server_type"] = server_type
            optional_keys = {
                "verify_tls",
                "command",
                "env",
                "cwd",
                "ready_pattern",
                "ready_timeout",
                "ready_probe_url",
                "shutdown_command",
                "shutdown_signal",
                "capture_output",
            }
            for key in optional_keys:
                if key in descriptor and descriptor[key] is not None:
                    value = descriptor[key]
                    if key in {"env", "command", "shutdown_command"} and isinstance(
                        value, (tuple, list)
                    ):
                        value = list(value)
                    base_payload[key] = value
            remaining = {
                key: value
                for key, value in descriptor.items()
                if key
                not in (
                    "label",
                    "description",
                    "url",
                    "server_url",
                    "allowed_tools",
                    "headers",
                    "metadata",
                    "server_type",
                    "type",
                )
                and key not in optional_keys
            }
            for key, value in remaining.items():
                if key not in base_payload and value is not None:
                    base_payload[key] = value
            return self._merge_payload(base_payload)

        raise ValueError(f"Unsupported tool type '{self.tool_type}' for tool '{self.name}'")


class ToolRegistry:
    """Registry for collecting and validating tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._sources: dict[str, Any] = {}

    def register(
        self,
        tool: ToolSpec | ToolFunctionDef | ToolCallable,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> ToolSpec:
        """Register ``tool`` ensuring it is safe for exposure to the model."""

        spec = self._ensure_spec(
            tool,
            name=name,
            description=description,
            parameters=parameters,
        )
        if not replace and spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec
        self._sources[spec.name] = spec.source
        return spec

    def register_mcp_tool(
        self,
        label: str,
        payload: Mapping[str, Any],
        *,
        description: str | None = None,
        replace: bool = False,
        payload_overrides: Mapping[str, Any] | None = None,
    ) -> ToolSpec:
        descriptor = dict(payload)
        resolved_label = (descriptor.get("label") or label or "").strip()
        if not resolved_label:
            raise ValueError("MCP tools must provide a non-empty label")
        descriptor["label"] = resolved_label
        allowed_raw = descriptor.get("allowed_tools")
        if allowed_raw is not None:
            if isinstance(allowed_raw, str):
                descriptor["allowed_tools"] = [allowed_raw]
            elif isinstance(allowed_raw, Sequence):
                descriptor["allowed_tools"] = [str(value) for value in allowed_raw]
            else:  # pragma: no cover - defensive fallback
                descriptor["allowed_tools"] = [str(allowed_raw)]
        headers = descriptor.get("headers")
        if isinstance(headers, Mapping):
            descriptor["headers"] = dict(headers)
        metadata = descriptor.get("metadata")
        if isinstance(metadata, Mapping):
            descriptor["metadata"] = dict(metadata)
        spec_description = (
            description
            if description is not None
            else str(descriptor.get("description") or "")
        )
        overrides_dict = dict(payload_overrides) if payload_overrides else None
        spec = ToolSpec(
            name=resolved_label,
            description=spec_description,
            parameters=descriptor,
            implementation=None,
            source=dict(payload),
            tool_type="mcp",
            payload_overrides=overrides_dict,
        )
        return self.register(spec, replace=replace)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._sources.pop(name, None)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def resolve(
        self,
        *,
        tools: Iterable[ToolSpec | ToolFunctionDef | ToolCallable] | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> list[ToolSpec]:
        """Resolve tool references into normalized :class:`ToolSpec` instances."""

        resolved: list[ToolSpec] = []
        seen: set[str] = set()

        if tools is not None:
            for tool in tools:
                spec = self._ensure_spec(tool)
                if spec.name not in seen:
                    resolved.append(spec)
                    seen.add(spec.name)

        if tool_names is not None:
            for name in tool_names:
                spec = self.get(name)
                if spec.name not in seen:
                    resolved.append(spec)
                    seen.add(spec.name)

        return resolved

    def _ensure_spec(
        self,
        tool: ToolSpec | ToolFunctionDef | ToolCallable,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            if tool.tool_type == "function" and not callable(tool.implementation):
                raise ValueError(
                    f"Tool '{tool.name}' declares type 'function' but is missing a callable implementation"
                )
            return tool
        if _is_tool_function_def(tool):
            spec_name = name or _safe_getattr(tool, "name")
            spec_description = description or _safe_getattr(tool, "description", "")
            if not spec_name:
                raise ValueError("ToolFunctionDef instances must declare a name")
            if not spec_description:
                raise ValueError(
                    f"Tool '{spec_name}' is missing a description; provide one to register it."
                )
            impl = _safe_getattr(tool, "implementation")
            if not callable(impl):
                raise ValueError(
                    f"Tool '{spec_name}' is missing an implementation callable."
                )
            params = parameters or _normalize_parameters(_safe_getattr(tool, "parameters", {}))
            return ToolSpec(
                name=spec_name,
                description=str(spec_description),
                parameters=params,
                implementation=impl,
                source=tool,
                tool_type="function",
            )
        if callable(tool):
            spec_name = name or getattr(tool, "__name__", None)
            if not spec_name:
                raise ValueError("Tool callables must have a resolvable __name__. Provide name=...")
            spec_description = _callable_description(tool, description)
            params = _normalize_parameters(parameters) or _callable_signature_parameters(tool)
            return ToolSpec(
                name=spec_name,
                description=spec_description,
                parameters=params,
                implementation=tool,
                source=tool,
                tool_type="function",
            )
        raise TypeError(f"Unsupported tool definition type: {type(tool)!r}")


def _iter_tool_modules() -> list[ModuleType]:
    package = importlib.import_module("internal.tools")
    search_paths = []
    if hasattr(package, "__path__"):
        search_paths.extend(str(Path(path)) for path in package.__path__)
    elif getattr(package, "__file__", None):
        search_paths.append(str(Path(package.__file__).parent))
    modules: list[ModuleType] = []
    for module_info in pkgutil.iter_modules(search_paths):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        modules.append(module)
    return modules


def _load_default_tools(registry: ToolRegistry) -> None:
    for module in _iter_tool_modules():
        for attr_name in dir(module):
            if not attr_name.endswith("_tool"):
                continue
            candidate = getattr(module, attr_name)
            if _is_tool_function_def(candidate) or callable(candidate):
                try:
                    registry.register(candidate)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Failed to register tool '{attr_name}' from {module.__name__}: {exc}"
                    ) from exc


_REGISTRY = ToolRegistry()
_load_default_tools(_REGISTRY)


class _ToolSequence(Sequence[ToolSpec]):
    def __getitem__(self, index: int) -> ToolSpec:
        return _REGISTRY.all()[index]

    def __len__(self) -> int:
        return len(_REGISTRY.all())

    def __iter__(self):  # type: ignore[override]
        return iter(_REGISTRY.all())


class _ToolMapping(Mapping[str, ToolSpec]):
    def __getitem__(self, key: str) -> ToolSpec:
        return _REGISTRY.get(key)

    def __len__(self) -> int:
        return len(_REGISTRY.all())

    def __iter__(self):  # type: ignore[override]
        return (spec.name for spec in _REGISTRY.all())


TOOLS: Sequence[ToolSpec] = _ToolSequence()
TOOL_MAP: Mapping[str, ToolSpec] = _ToolMapping()


def get_all_tools() -> list[ToolSpec]:
    """Return a shallow copy of every registered tool specification."""

    return _REGISTRY.all()


def get_tool(name: str) -> ToolSpec:
    """Return the normalized tool specification by ``name``."""

    return _REGISTRY.get(name)


def get_tools(names: Sequence[str] | None = None) -> list[ToolSpec]:
    """Return registered tools matching ``names`` (or all when ``None``)."""

    if names is None:
        return get_all_tools()
    resolved: list[ToolSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        resolved.append(get_tool(name))
        seen.add(name)
    return resolved


def resolve_tools(
    *,
    tools: Iterable[ToolSpec | ToolFunctionDef | ToolCallable] | None = None,
    tool_names: Sequence[str] | None = None,
) -> list[ToolSpec]:
    """Resolve explicit tool definitions and names into normalized specifications."""

    return _REGISTRY.resolve(tools=tools, tool_names=tool_names)


def register_tool(
    tool: ToolSpec | ToolFunctionDef | ToolCallable,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> ToolSpec:
    """Register ``tool`` into the default registry."""

    return _REGISTRY.register(
        tool,
        name=name,
        description=description,
        parameters=parameters,
        replace=replace,
    )


def register_mcp_tool(
    label: str,
    payload: Mapping[str, Any],
    *,
    description: str | None = None,
    replace: bool = False,
    payload_overrides: Mapping[str, Any] | None = None,
) -> ToolSpec:
    """Register an MCP tool definition into the default registry."""

    return _REGISTRY.register_mcp_tool(
        label,
        payload,
        description=description,
        replace=replace,
        payload_overrides=payload_overrides,
    )


def unregister_tool(name: str) -> None:
    """Remove ``name`` from the default registry if present."""

    _REGISTRY.unregister(name)


__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "TOOLS",
    "TOOL_MAP",
    "get_all_tools",
    "get_tool",
    "get_tools",
    "resolve_tools",
    "register_tool",
    "register_mcp_tool",
    "unregister_tool",
]


# Re-export individual tool definitions for compatibility.
for spec in get_all_tools():
    if isinstance(spec.source, ToolFunctionDef):
        globals()[f"{spec.name}_tool"] = spec.source
        __all__.append(f"{spec.name}_tool")
