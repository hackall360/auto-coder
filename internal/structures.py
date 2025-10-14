"""Result containers and adapters for OpenAI-compatible responses."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .schemas import parse_structured_content, SchemaError


def _ensure_mapping(value: Mapping[str, Any] | None, *, error_message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(error_message)
    return value


def _coalesce_content(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        text = payload.get("text")
        if isinstance(text, str):
            return text
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        pieces: list[str] = []
        for item in payload:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
            elif isinstance(item, str):
                pieces.append(item)
        if pieces:
            return "".join(pieces)
    return None


def _extract_choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response payload is missing choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("Choice payload must be a mapping.")
    return first


def _extract_text(choice: Mapping[str, Any], *, fallback: str | None) -> str:
    message = choice.get("message")
    if isinstance(message, Mapping):
        content = _coalesce_content(message.get("content"))
        if content is not None:
            return content
    text = choice.get("text")
    if isinstance(text, str):
        return text
    if fallback is not None:
        return fallback
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        content = _coalesce_content(delta.get("content"))
        if content is not None:
            return content
        text = delta.get("text")
        if isinstance(text, str):
            return text
    raise ValueError("Response payload is missing textual content.")


def _extract_structured_payload(
    choice: Mapping[str, Any], *, fallback_text: str | None
) -> Any:
    message = choice.get("message")
    if isinstance(message, Mapping):
        if "parsed" in message:
            parsed = message.get("parsed")
            if parsed is not None:
                return parsed
        content = message.get("content")
        structured = _coalesce_content(content)
        if structured is not None:
            return structured
    if fallback_text is not None:
        return fallback_text
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        if "parsed" in delta and delta["parsed"] is not None:
            return delta["parsed"]
        structured = _coalesce_content(delta.get("content"))
        if structured is not None:
            return structured
    return None


@dataclass(slots=True)
class StructuredResponse:
    """Representation of a prediction result.

    Attributes
    ----------
    raw_response:
        The original OpenAI-compatible response payload.
    content:
        The textual representation of the prediction.
    parsed:
        Structured data parsed from the response when a schema is requested.
    schema:
        Schema descriptor (if provided) used to guide parsing.
    structured:
        Flag indicating whether structured output was requested.
    """

    raw_response: Mapping[str, Any]
    content: str
    parsed: Mapping[str, Any] | None
    schema: Any | None
    structured: bool

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
        *,
        schema: Any | None = None,
        structured: bool | None = None,
        fallback_text: str | None = None,
    ) -> "StructuredResponse":
        if not isinstance(response, Mapping):
            raise ValueError("Response must be a mapping.")
        choice = _extract_choice(response)
        expect_structured = bool(structured) if structured is not None else schema is not None
        content = _extract_text(choice, fallback=fallback_text)
        parsed_data: Mapping[str, Any] | None = None
        if expect_structured:
            structured_payload = _extract_structured_payload(choice, fallback_text=content)
            if structured_payload is None:
                raise SchemaError("Structured content is empty.")
            parsed_data = parse_structured_content(structured_payload)
        return cls(
            raw_response=response,
            content=content,
            parsed=parsed_data,
            schema=schema,
            structured=expect_structured,
        )


class PredictionResultAdapter:
    """Helper for assembling streaming prediction results."""

    def __init__(self, *, schema: Any | None = None, structured: bool | None = None) -> None:
        self._schema = schema
        self._expect_structured = bool(structured) if structured is not None else schema is not None
        self._chunks: list[Mapping[str, Any]] = []
        self._content_parts: list[str] = []
        self._raw_response: Mapping[str, Any] | None = None
        self._result: StructuredResponse | None = None

    def add_chunk(self, chunk: Mapping[str, Any]) -> None:
        """Record a streamed chunk and accumulate its textual content."""

        chunk = _ensure_mapping(chunk, error_message="Stream chunks must be mappings.")
        self._chunks.append(chunk)
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            content = _coalesce_content(delta.get("content"))
            if isinstance(content, str) and content:
                self._content_parts.append(content)
            elif isinstance(delta.get("text"), str):
                self._content_parts.append(delta["text"])

    def result(self, final_response: Mapping[str, Any] | None = None) -> StructuredResponse:
        """Return the final prediction result, parsing structured data on demand."""

        if final_response is not None:
            self._raw_response = final_response
        if self._result is not None:
            return self._result
        if self._raw_response is None:
            if self._chunks:
                text = "".join(self._content_parts)
                self._raw_response = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": text,
                            }
                        }
                    ]
                }
            else:
                raise ValueError("No response data has been provided.")
        fallback_text = "".join(self._content_parts) if self._content_parts else None
        self._result = StructuredResponse.from_response(
            self._raw_response,
            schema=self._schema,
            structured=self._expect_structured,
            fallback_text=fallback_text,
        )
        return self._result


__all__ = ["PredictionResultAdapter", "StructuredResponse"]
