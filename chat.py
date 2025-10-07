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

from tooling import resolve_tools

ChatInput = Union[str, lms.Chat, Mapping[str, Any]]
CallbackMap = Mapping[str, Callable[..., Any]]
ToolList = Sequence[Any]

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
    message = getattr(result, "message", None)
    if message is not None:
        for attr in ("content", "text"):
            if hasattr(message, attr):
                return getattr(message, attr)
    return ""


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
) -> list[Any]:
    """Resolve tool arguments into a deduplicated list."""

    resolved: list[Any] = []
    if default:
        resolved.extend(default)
    if tools is not None or tool_names is not None:
        merged = resolve_tools(tools=tools, tool_names=tool_names)
        for tool in merged:
            if tool not in resolved:
                resolved.append(tool)
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
    **act_kwargs: Any,
) -> tuple[str, Any]:
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
    chat_input = _prepare_input(prompt_or_chat)
    result = model.act(chat_input, resolved_tools, **kwargs)
    return _extract_text(result), result


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
    tools: list[Any] = field(default_factory=list)

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
        if history is not None:
            if isinstance(history, lms.Chat):
                chat = history
            elif isinstance(history, str):
                chat = lms.Chat.from_history(history)
            else:
                chat = lms.Chat.from_history(history)
        else:
            if system_prompt is not None:
                chat = lms.Chat(system_prompt)
            else:
                chat = lms.Chat.from_history({"messages": []})
        model = model or get_model(model_name)
        resolved_tools = _prepare_tools(tools, tool_names)
        return cls(chat=chat, model=model, tools=resolved_tools)

    def add_user_message(self, content: str) -> None:
        self.chat.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        self.chat.add_assistant_message(content)

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
            **act_kwargs,
        )
        # Cache the resolved tools for future rounds when overrides are provided.
        self.tools = list(resolved_tools)
        return text, result
