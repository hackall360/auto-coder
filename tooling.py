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
    implementation: ToolCallable
    source: Any

    def to_payload(self) -> dict[str, Any]:
        """Return the structure expected by :meth:`model.act`."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
                "implementation": self.implementation,
            },
        }


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
    "unregister_tool",
]


# Re-export individual tool definitions for compatibility.
for spec in get_all_tools():
    if isinstance(spec.source, ToolFunctionDef):
        globals()[f"{spec.name}_tool"] = spec.source
        __all__.append(f"{spec.name}_tool")
