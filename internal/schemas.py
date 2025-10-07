"""Utilities for working with LM Studio structured output schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, Union
from typing import runtime_checkable

SchemaDict: TypeAlias = Mapping[str, Any]
SchemaLike = Union[SchemaDict, "ModelSchema", type["ModelSchema"], str]


class SchemaError(ValueError):
    """Raised when schema values cannot be normalized."""


@runtime_checkable
class ModelSchema(Protocol):
    """Protocol for classes that provide a JSON schema definition."""

    @classmethod
    def model_json_schema(cls) -> SchemaDict:
        ...


try:  # pragma: no cover - optional dependency path
    import msgspec
    from msgspec.json import schema as _msgspec_schema

    class BaseModel(msgspec.Struct, kw_only=True, omit_defaults=True):
        """msgspec-based helper mirroring LM Studio SDK BaseModel."""

        @classmethod
        def model_json_schema(cls) -> SchemaDict:
            return _ensure_str_dict(_msgspec_schema(cls))

except ImportError:  # pragma: no cover - fallback path
    class BaseModel:
        """Fallback base class requiring subclasses to implement model_json_schema."""

        @classmethod
        def model_json_schema(cls) -> SchemaDict:
            raise SchemaError(
                "msgspec is not installed; either install it or override model_json_schema()."
            )


def _ensure_str_dict(data: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise SchemaError("Schema keys must be strings.")
        if isinstance(value, Mapping):
            value = _ensure_str_dict(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            new_list = []
            for item in value:
                if isinstance(item, Mapping):
                    new_list.append(_ensure_str_dict(item))
                else:
                    new_list.append(item)
            value = new_list
        result[key] = value
    return result


def _has_json_type(schema: Mapping[str, Any]) -> bool:
    if any(key in schema for key in ("type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const")):
        return True
    return False


def _derive_schema_name(source: SchemaLike, schema: Mapping[str, Any]) -> str:
    name_candidates = (
        schema.get("name"),
        schema.get("title"),
        schema.get("$id"),
    )
    for candidate in name_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    if isinstance(source, type):
        return source.__name__
    if isinstance(source, ModelSchema):
        return source.__class__.__name__
    if isinstance(source, str):
        digest = hashlib.sha1(source.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
        return f"schema_{digest[:8]}"
    digest = hashlib.sha1(repr(schema).encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
    return f"schema_{digest[:8]}"


def normalize_schema(schema_like: SchemaLike) -> dict[str, Any]:
    schema_dict: Mapping[str, Any]
    if isinstance(schema_like, Mapping):
        schema_dict = schema_like
    elif isinstance(schema_like, str):
        try:
            parsed = json.loads(schema_like)
        except json.JSONDecodeError as exc:
            raise SchemaError("Schema string must contain valid JSON.") from exc
        if not isinstance(parsed, Mapping):
            raise SchemaError("Parsed schema must be a JSON object.")
        schema_dict = parsed
    elif isinstance(schema_like, type) and hasattr(schema_like, "model_json_schema"):
        schema_dict = schema_like.model_json_schema()  # type: ignore[misc]
    elif hasattr(schema_like, "model_json_schema"):
        schema_dict = schema_like.model_json_schema()  # type: ignore[misc]
    else:
        raise SchemaError("Unsupported schema type provided.")

    plain_schema = _ensure_str_dict(schema_dict)
    if not _has_json_type(plain_schema):
        raise SchemaError("Schema must include JSON schema type information.")
    return plain_schema


def build_response_format(schema: SchemaLike, *, name: str | None = None, strict: bool = True) -> dict[str, Any]:
    normalized = normalize_schema(schema)
    schema_name = name or _derive_schema_name(schema, normalized)
    json_schema_payload: dict[str, Any] = {
        "name": schema_name,
        "schema": normalized,
    }
    json_schema_payload["strict"] = bool(strict)
    return {
        "type": "json_schema",
        "json_schema": json_schema_payload,
    }


def parse_structured_content(content: Any) -> dict[str, Any]:
    if content is None:
        raise SchemaError("Structured content is empty.")
    if isinstance(content, Mapping):
        return _ensure_str_dict(content)
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            raise SchemaError("Structured content is empty.")
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SchemaError("Structured content must be valid JSON.") from exc
        if not isinstance(parsed, Mapping):
            raise SchemaError("Structured content must decode to a JSON object.")
        return _ensure_str_dict(parsed)
    raise SchemaError("Structured content must be a JSON string or mapping.")


__all__ = [
    "BaseModel",
    "ModelSchema",
    "SchemaError",
    "SchemaDict",
    "SchemaLike",
    "build_response_format",
    "normalize_schema",
    "parse_structured_content",
]
