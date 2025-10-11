"""Utilities for interacting with LM Studio chat models.

This module provides convenience wrappers around the `lmstudio` Python SDK,
loosely following the quick-start snippets from
``docs/LMStudio/developer/python/llm-prediction/chat-completion.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import lmstudio as lms

from internal.schemas import SchemaError, SchemaLike, build_response_format
from internal.structures import StructuredResponse
from tooling import ToolSpec, resolve_tools

ChatInput = Union[str, lms.Chat, Mapping[str, Any]]
CallbackMap = Mapping[str, Callable[..., Any]]
ToolList = Sequence[ToolSpec]

__all__ = [
    "ChatInput",
    "CallbackMap",
    "ChatSession",
    "ResponseStream",
    "act",
    "get_model",
    "respond",
    "respond_stream",
]


def get_model(name: str | None = None, /, **llm_kwargs: Any):
    """Return an LM Studio model handle using the high-level ``lms.llm`` helper.

    Parameters
    ----------
    name:
        Optional identifier for the model to load. When omitted, the default
        model configured in LM Studio is loaded, matching the
        ``model = lms.llm()`` pattern from the quick-start examples.
    llm_kwargs:
        Additional keyword arguments forwarded to :func:`lmstudio.llm`.

    Returns
    -------
    The model handle yielded by :func:`lmstudio.llm`.
    """

    return lms.llm(name, **llm_kwargs)


def _prepare_input(prompt_or_chat: ChatInput) -> ChatInput:
    if isinstance(prompt_or_chat, (str, lms.Chat)):
        return prompt_or_chat
    if isinstance(prompt_or_chat, Mapping):
        return prompt_or_chat
    raise TypeError(
        "Expected a prompt string, lms.Chat instance, or chat history mapping; "
        f"received {type(prompt_or_chat)!r}."
    )


def _call_model(
    model: Any,
    prompt_or_chat: ChatInput,
    *,
    config: Optional[Mapping[str, Any]] = None,
    callbacks: Optional[CallbackMap] = None,
    streaming: bool = False,
) -> Any:
    kwargs: Dict[str, Any] = {}
    if config is not None:
        kwargs["config"] = config
    if callbacks:
        kwargs.update(callbacks)
    chat_input = _prepare_input(prompt_or_chat)
    if streaming:
        return model.respond_stream(chat_input, **kwargs)
    return model.respond(chat_input, **kwargs)


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    for attr in ("content", "text"):
        if hasattr(result, attr):
            return getattr(result, attr)
    if isinstance(result, Mapping):
        message = result.get("message")
        if isinstance(message, Mapping):
            for attr in ("content", "text"):
                value = message.get(attr)
                if isinstance(value, str):
                    return value
        content = result.get("content")
        if isinstance(content, str):
            return content
        choices = result.get("choices")
        if isinstance(choices, Sequence) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                choice_message = first_choice.get("message")
                if isinstance(choice_message, Mapping):
                    for attr in ("content", "text"):
                        value = choice_message.get(attr)
                        if isinstance(value, str):
                            return value
                for attr in ("content", "text"):
                    value = first_choice.get(attr)
                    if isinstance(value, str):
                        return value
    message = getattr(result, "message", None)
    if message is not None:
        for attr in ("content", "text"):
            if hasattr(message, attr):
                return getattr(message, attr)
    return ""


def _coerce_response_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    for attr in ("raw_response", "response"):
        candidate = getattr(result, attr, None)
        if isinstance(candidate, Mapping):
            return candidate
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(result, method_name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return candidate
    try:
        mapping = vars(result)
    except TypeError:  # pragma: no cover - non-object fallback
        mapping = None
    if isinstance(mapping, Mapping):
        return mapping
    raise TypeError("Model.act response must be a mapping-compatible object.")


def _resolve_response_format(
    *,
    schema: SchemaLike | None = None,
    response_format: Any | None = None,
    schema_name: str | None = None,
    strict: bool = True,
) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None, bool]:
    if schema is not None and response_format is not None:
        raise ValueError("Specify either schema or response_format, not both.")

    if schema is None and response_format is None:
        return None, None, False

    payload: dict[str, Any] | None = None
    normalized_schema: Mapping[str, Any] | None = None

    candidate = schema if schema is not None else response_format

    if hasattr(candidate, "as_response_format") and callable(
        getattr(candidate, "as_response_format")
    ):
        payload = candidate.as_response_format()  # type: ignore[assignment]
        if isinstance(payload, Mapping):
            json_schema = payload.get("json_schema")
            if isinstance(json_schema, Mapping):
                normalized_schema = json_schema.get("schema")  # type: ignore[assignment]
                strict_flag = json_schema.get("strict")
                if isinstance(strict_flag, bool):
                    strict = strict_flag
        payload = dict(payload)  # shallow copy for mutation safety
    elif isinstance(candidate, Mapping) and {
        "type",
        "json_schema",
    }.issubset(candidate.keys()):
        payload = dict(candidate)
        json_schema = candidate.get("json_schema")
        if isinstance(json_schema, Mapping):
            normalized_schema = json_schema.get("schema")  # type: ignore[assignment]
            strict_flag = json_schema.get("strict")
            if isinstance(strict_flag, bool):
                strict = strict_flag
    else:
        schema_like: Any = candidate
        if isinstance(candidate, Mapping) and "schema" in candidate:
            schema_like = candidate["schema"]
            if schema_name is None and isinstance(candidate.get("name"), str):
                schema_name = candidate["name"]
            if "strict" in candidate:
                strict = bool(candidate["strict"])
        payload = build_response_format(schema_like, name=schema_name, strict=strict)
        json_schema = payload.get("json_schema")
        if isinstance(json_schema, Mapping):
            normalized_schema = json_schema.get("schema")  # type: ignore[assignment]

    if payload is None:
        return None, None, False

    return payload, normalized_schema, True


def respond(
    prompt_or_chat: ChatInput,
    *,
    model: Any | None = None,
    model_name: str | None = None,
    config: Optional[Mapping[str, Any]] = None,
    callbacks: Optional[CallbackMap] = None,
) -> tuple[str, Any]:
    """Generate a single response using :meth:`model.respond`.

    The function accepts the same chat inputs showcased in the "Quick Example"
    and "Generate a response" snippets from the LM Studio documentation: a
    plain string, an :class:`lmstudio.Chat` instance, or a chat-history mapping.
    The underlying :meth:`model.respond` call returns the SDK response object;
    this function extracts the assistant text for convenience and returns the
    pair ``(text, result)``.
    """

    model = model or get_model(model_name)
    result = _call_model(
        model,
        prompt_or_chat,
        config=config,
        callbacks=callbacks,
        streaming=False,
    )
    return _extract_text(result), result


def _prepare_tools(
    tools: Iterable[Any] | None = None,
    tool_names: Sequence[str] | None = None,
    *,
    default: ToolList | None = None,
) -> list[ToolSpec]:
    """Resolve tool arguments into a deduplicated list."""

    resolved: list[ToolSpec] = []
    seen: set[str] = set()

    if default:
        for spec in default:
            if spec.name not in seen:
                resolved.append(spec)
                seen.add(spec.name)

    if tools is not None or tool_names is not None:
        merged = resolve_tools(tools=tools, tool_names=tool_names)
        for spec in merged:
            if spec.name not in seen:
                resolved.append(spec)
                seen.add(spec.name)

    return resolved


def act(
    prompt_or_chat: ChatInput,
    *,
    tools: Iterable[Any] | None = None,
    tool_names: Sequence[str] | None = None,
    model: Any | None = None,
    model_name: str | None = None,
    config: Optional[Mapping[str, Any]] = None,
    callbacks: Optional[CallbackMap] = None,
    schema: SchemaLike | None = None,
    response_format: Any | None = None,
    schema_name: str | None = None,
    strict_schema: bool = True,
    handle_invalid_tool_request: Any | None = None,
    **act_kwargs: Any,
) -> tuple[str, StructuredResponse]:
    """Execute :meth:`model.act` with resolved tool definitions.

    This mirrors the tool-calling workflow documented in
    ``docs/LMStudio/developer/python/agent/act.md`` by resolving tool
    definitions from ``tooling.py`` and forwarding any configuration or
    callback arguments to the underlying SDK call. The return value follows
    :func:`respond`, yielding a tuple of the assistant's final text and the raw
    SDK result.
    """

    resolved_tools = _prepare_tools(tools, tool_names)
    if not resolved_tools:
        raise ValueError(
            "No tools were provided to act(); specify tool definitions or names."
        )
    model = model or get_model(model_name)
    kwargs: Dict[str, Any] = {}
    if config is not None:
        kwargs["config"] = config
    if callbacks:
        kwargs.update(callbacks)
    kwargs.update(act_kwargs)
    if handle_invalid_tool_request is not None:
        kwargs["handle_invalid_tool_request"] = handle_invalid_tool_request
    response_payload, normalized_schema, expect_structured = _resolve_response_format(
        schema=schema,
        response_format=response_format,
        schema_name=schema_name,
        strict=strict_schema,
    )
    if response_payload is not None:
        kwargs["response_format"] = response_payload
    chat_input = _prepare_input(prompt_or_chat)
    tool_payloads = [spec.to_payload() for spec in resolved_tools]
    result = model.act(chat_input, tool_payloads, **kwargs)
    raw_payload = _coerce_response_mapping(result)
    fallback_text = _extract_text(raw_payload)
    try:
        structured = StructuredResponse.from_response(
            raw_payload,
            schema=normalized_schema,
            structured=expect_structured,
            fallback_text=fallback_text,
        )
    except SchemaError as exc:
        message = "Model response did not match the expected structured schema"
        if fallback_text:
            message = f"{message}: {fallback_text!r}"
        raise SchemaError(message) from exc
    return structured.content, structured


class ResponseStream(Iterator[Any]):
    """Iterator wrapper for streamed responses.

    Instances yield the fragments produced by :meth:`model.respond_stream` while
    exposing :meth:`result` and :meth:`wait_for_result` helpers analogous to the
    SDK's streaming example in the LM Studio documentation.
    """

    def __init__(self, stream: Any):
        self._stream = stream
        self._iterator = iter(stream)
        self._fragments: list[Any] = []

    def __iter__(self) -> "ResponseStream":
        return self

    def __next__(self) -> Any:
        fragment = next(self._iterator)
        self._fragments.append(fragment)
        return fragment

    def result(self) -> Any:
        if hasattr(self._stream, "result"):
            return self._stream.result()
        raise AttributeError("Underlying stream does not expose result().")

    def wait_for_result(self) -> Any:
        if hasattr(self._stream, "wait_for_result"):
            return self._stream.wait_for_result()
        for _ in self:
            pass
        return self._fragments[-1] if self._fragments else None

    @property
    def text(self) -> str:
        contents: list[str] = []
        for fragment in self._fragments:
            for attr in ("content", "text"):
                if hasattr(fragment, attr):
                    contents.append(getattr(fragment, attr))
                    break
            else:
                contents.append(str(fragment))
        return "".join(contents)


def respond_stream(
    prompt_or_chat: ChatInput,
    *,
    model: Any | None = None,
    model_name: str | None = None,
    config: Optional[Mapping[str, Any]] = None,
    callbacks: Optional[CallbackMap] = None,
) -> ResponseStream:
    """Stream a response in the style of the documentation's streaming samples.

    The returned :class:`ResponseStream` object can be iterated to receive
    fragments, with :meth:`ResponseStream.result` and
    :meth:`ResponseStream.wait_for_result` available for accessing the final
    prediction metadata once generation completes.
    """

    model = model or get_model(model_name)
    stream = _call_model(
        model,
        prompt_or_chat,
        config=config,
        callbacks=callbacks,
        streaming=True,
    )
    return ResponseStream(stream)


@dataclass
class ChatSession:
    """Utility for managing a multi-turn conversation with LM Studio models.

    The helper mirrors the "Multi-turn Chat" walkthrough by reusing an
    :class:`lmstudio.Chat` instance between turns while offering convenience
    methods for sending user messages and streaming responses.
    """

    chat: lms.Chat
    model: Any
    tools: list[ToolSpec] = field(default_factory=list)
    system_prompt: str | None = None

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str | None = None,
        history: ChatInput | None = None,
        model: Any | None = None,
        model_name: str | None = None,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> "ChatSession":
        system = system_prompt
        if history is not None:
            if isinstance(history, lms.Chat):
                chat = history
            elif isinstance(history, str):
                chat = lms.Chat.from_history(history)
            else:
                chat = lms.Chat.from_history(history)
            if system is None and getattr(chat, "messages", None):
                first_message = chat.messages[0]
                if isinstance(first_message, Mapping) and first_message.get("role") == "system":
                    system = first_message.get("content")
        else:
            if system_prompt is not None:
                chat = lms.Chat(system_prompt)
            else:
                chat = lms.Chat.from_history({"messages": []})
        model = model or get_model(model_name)
        resolved_tools = _prepare_tools(tools, tool_names)
        return cls(chat=chat, model=model, tools=resolved_tools, system_prompt=system)

    def add_user_message(self, content: str) -> None:
        self.chat.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        self.chat.add_assistant_message(content)

    def append_user_input(self, content: str) -> None:
        """Alias for :meth:`add_user_message` for parity with agent helpers."""

        self.add_user_message(content)

    def append_tool_response(
        self,
        content: str,
        *,
        name: str | None = None,
        tool_call_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a tool response message to the chat history.

        Parameters
        ----------
        content:
            Tool output text to record in the chat transcript.
        name:
            Optional tool name associated with the response.
        tool_call_id:
            Identifier correlating the response with a prior tool invocation.
        payload:
            Optional mapping merged into the message before appending. This is
            convenient when callers want to preserve additional metadata (e.g.,
            structured outputs) in the chat history.
        """

        message: Dict[str, Any] = {"role": "tool", "content": content}
        if name is not None:
            message["name"] = name
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if payload:
            for key, value in payload.items():
                if key not in message:
                    message[key] = value
        self.chat.append(message)

    def send(
        self,
        user_message: str | None = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        callbacks: Optional[CallbackMap] = None,
    ) -> tuple[str, Any]:
        if user_message:
            self.add_user_message(user_message)
        merged_callbacks = dict(callbacks or {})
        merged_callbacks.setdefault("on_message", self.chat.append)
        return respond(
            self.chat,
            model=self.model,
            config=config,
            callbacks=merged_callbacks,
        )

    def send_stream(
        self,
        user_message: str | None = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        callbacks: Optional[CallbackMap] = None,
    ) -> ResponseStream:
        if user_message:
            self.add_user_message(user_message)
        merged_callbacks = dict(callbacks or {})
        merged_callbacks.setdefault("on_message", self.chat.append)
        return respond_stream(
            self.chat,
            model=self.model,
            config=config,
            callbacks=merged_callbacks,
        )

    def set_tools(
        self,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        """Replace the active tool list for subsequent :meth:`act` calls."""

        self.tools = _prepare_tools(tools, tool_names)

    def act(
        self,
        user_message: str | None = None,
        *,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
        config: Optional[Mapping[str, Any]] = None,
        callbacks: Optional[CallbackMap] = None,
        schema: SchemaLike | None = None,
        response_format: Any | None = None,
        schema_name: str | None = None,
        strict_schema: bool = True,
        handle_invalid_tool_request: Any | None = None,
        **act_kwargs: Any,
    ) -> tuple[str, Any]:
        if user_message:
            self.add_user_message(user_message)
        resolved_tools = _prepare_tools(tools, tool_names, default=self.tools)
        if not resolved_tools:
            raise ValueError(
                "ChatSession.act() requires at least one tool; call set_tools() or"
                " provide tools/tool_names explicitly."
            )
        merged_callbacks = dict(callbacks or {})
        merged_callbacks.setdefault("on_message", self.chat.append)
        text, result = act(
            self.chat,
            tools=resolved_tools,
            model=self.model,
            config=config,
            callbacks=merged_callbacks,
            schema=schema,
            response_format=response_format,
            schema_name=schema_name,
            strict_schema=strict_schema,
            handle_invalid_tool_request=handle_invalid_tool_request,
            **act_kwargs,
        )
        # Cache the resolved tools for future rounds when overrides are provided.
        self.tools = list(resolved_tools)
        return text, result
