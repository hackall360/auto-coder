"""Agent option dataclasses targeting the LM Studio Python SDK.

The helpers defined here are thin, typed containers that describe the most
common agent configuration knobs exposed by the LM Studio SDK (tested with
versions 0.3.3 and newer). Default values mirror the defaults shown in the LM
Studio application presets: ``temperature=0.7`` for balanced creativity,
``max_rounds=6`` to keep conversations bounded, and structured outputs are
opt-in with ``strict`` JSON schema validation enabled when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict
from typing import NotRequired

from .schemas import SchemaLike, build_response_format


class AgentConfigPayload(TypedDict, total=False):
    """Typed dictionary describing configuration passed to the SDK."""

    model: str
    temperature: NotRequired[float]
    system_prompt: NotRequired[str]
    max_rounds: NotRequired[int]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    response_format: NotRequired[dict[str, Any]]


@dataclass(slots=True)
class StructuredResponseSettings:
    """Options controlling LM Studio's structured output helpers."""

    schema: SchemaLike
    name: str | None = None
    strict: bool = True

    def as_response_format(self) -> dict[str, Any]:
        """Return the response_format payload accepted by LM Studio."""

        return build_response_format(self.schema, name=self.name, strict=self.strict)


@dataclass(slots=True)
class AgentOptions:
    """Container describing common LM Studio agent configuration options."""

    model: str
    temperature: float = 0.7
    system_prompt: str | None = None
    max_rounds: int = 6
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    structured_schema: SchemaLike | None = None
    structured_name: str | None = None
    structured_strict: bool = True
    response_format: bool = False

    def build_payload(self) -> AgentConfigPayload:
        """Assemble a dictionary compatible with LM Studio's chat helpers."""

        payload: AgentConfigPayload = {"model": self.model}

        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.system_prompt is not None:
            payload["system_prompt"] = self.system_prompt
        if self.max_rounds is not None:
            payload["max_rounds"] = int(self.max_rounds)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = float(self.frequency_penalty)
        if self.presence_penalty is not None:
            payload["presence_penalty"] = float(self.presence_penalty)

        if self.structured_schema is not None:
            structured = StructuredResponseSettings(
                schema=self.structured_schema,
                name=self.structured_name,
                strict=self.structured_strict,
            )
            payload["response_format"] = structured.as_response_format()
        elif self.response_format:
            raise ValueError(
                "response_format flag requires structured_schema to be provided."
            )

        return payload


__all__ = [
    "AgentConfigPayload",
    "AgentOptions",
    "StructuredResponseSettings",
]
