"""High-level session orchestration for LM Studio agent workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from chat import CallbackMap, ChatSession
from internal.structures import StructuredResponse
from tooling import ToolSpec

__all__ = ["AgentRound", "AgentSession"]


Hook = Callable[..., None]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _coerce_mapping(obj: Any) -> Mapping[str, Any] | None:
    if isinstance(obj, Mapping):
        return obj
    return None


def _extract_sequence(obj: Any, names: Sequence[str]) -> list[Any]:
    if isinstance(obj, StructuredResponse):
        return _extract_sequence(obj.raw_response, names)
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return _as_list(obj[name])
        if hasattr(obj, name):
            return _as_list(getattr(obj, name))
    return []


@dataclass(slots=True)
class AgentRound:
    """Container for per-round metadata captured by :class:`AgentSession`."""

    index: int
    user_message: str | None
    response_text: str
    result: StructuredResponse
    transcript: list[Any]
    messages: list[Any] = field(default_factory=list)
    tool_history: dict[str, list[Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user_message": self.user_message,
            "response_text": self.response_text,
            "result": self.result,
            "transcript": list(self.transcript),
            "messages": list(self.messages),
            "tool_history": {
                key: list(values) for key, values in self.tool_history.items()
            },
        }


class AgentSession:
    """Wrap LM Studio ``model.act`` calls with convenient state tracking."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        history: Any | None = None,
        model: Any | None = None,
        model_name: str | None = None,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
        callbacks: Optional[CallbackMap] = None,
        on_message: Hook | None = None,
        on_tool_call: Hook | None = None,
        on_tool_result: Hook | None = None,
        on_round_start: Hook | None = None,
        on_round_end: Hook | None = None,
    ) -> None:
        self.chat_session = ChatSession.create(
            system_prompt=system_prompt,
            history=history,
            model=model,
            model_name=model_name,
            tools=tools,
            tool_names=tool_names,
        )
        self._base_callbacks: dict[str, Callable[..., Any]] = dict(callbacks or {})
        self._current_messages: list[Any] = []
        self._current_tool_calls: list[Any] = []
        self._current_tool_results: list[Any] = []
        self._on_message = on_message
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._on_round_start = on_round_start
        self._on_round_end = on_round_end
        self.rounds: list[AgentRound] = []

    @property
    def model(self) -> Any:
        return self.chat_session.model

    @property
    def tools(self) -> list[ToolSpec]:
        return self.chat_session.tools

    @property
    def system_prompt(self) -> str | None:
        return self.chat_session.system_prompt

    @property
    def transcript(self) -> list[Any]:
        messages = getattr(self.chat_session.chat, "messages", [])
        return [message.copy() if isinstance(message, dict) else message for message in messages]

    def set_tools(
        self,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        self.chat_session.set_tools(tools, tool_names)

    def append_user_input(self, content: str) -> None:
        self.chat_session.append_user_input(content)

    def append_tool_response(
        self,
        content: str,
        *,
        name: str | None = None,
        tool_call_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.chat_session.append_tool_response(
            content,
            name=name,
            tool_call_id=tool_call_id,
            payload=payload,
        )

    def _compose_callbacks(self, callbacks: Optional[CallbackMap]) -> dict[str, Callable[..., Any]]:
        merged: dict[str, Callable[..., Any]] = dict(self._base_callbacks)
        if callbacks:
            merged.update(callbacks)

        downstream_on_message = merged.get("on_message")
        downstream_tool_call = merged.get("on_tool_call")
        downstream_tool_result = merged.get("on_tool_result")

        def handle_message(message: Any, *args: Any, **kwargs: Any) -> None:
            appended_via_downstream = False
            if downstream_on_message and downstream_on_message is not handle_message:
                downstream_on_message(message, *args, **kwargs)
                appended_via_downstream = (
                    getattr(downstream_on_message, "__self__", None)
                    is self.chat_session.chat
                    and getattr(downstream_on_message, "__func__", None)
                    is getattr(self.chat_session.chat.append, "__func__", None)
                )
            if not appended_via_downstream:
                self.chat_session.chat.append(message)
            self._current_messages.append(message)
            if self._on_message:
                self._on_message(message)

        def handle_tool_call(call: Any, *args: Any, **kwargs: Any) -> None:
            if downstream_tool_call and downstream_tool_call is not handle_tool_call:
                downstream_tool_call(call, *args, **kwargs)
            self._current_tool_calls.append(call)
            if self._on_tool_call:
                self._on_tool_call(call)

        def handle_tool_result(result: Any, *args: Any, **kwargs: Any) -> None:
            if downstream_tool_result and downstream_tool_result is not handle_tool_result:
                downstream_tool_result(result, *args, **kwargs)
            self._current_tool_results.append(result)
            self._append_tool_result_to_chat(result)
            if self._on_tool_result:
                self._on_tool_result(result)

        merged["on_message"] = handle_message
        merged["on_tool_call"] = handle_tool_call
        merged["on_tool_result"] = handle_tool_result
        return merged

    def _append_tool_result_to_chat(self, result: Any) -> None:
        mapping = _coerce_mapping(result)
        if mapping is not None:
            if mapping.get("role") == "tool":
                self.chat_session.chat.append(dict(mapping))
                return
            content = mapping.get("content") or mapping.get("output") or mapping.get("result")
            if content is not None:
                payload = mapping.copy()
                payload.pop("content", None)
                payload.pop("output", None)
                payload.pop("result", None)
                name = payload.pop("name", None)
                tool_call_id = payload.pop("tool_call_id", None) or payload.pop("id", None)
                self.append_tool_response(
                    str(content),
                    name=name,
                    tool_call_id=tool_call_id,
                    payload=payload if payload else None,
                )
                return
        if isinstance(result, str):
            self.append_tool_response(result)

    def _finalize_round(
        self,
        *,
        index: int,
        user_message: str | None,
        response_text: str,
        result: StructuredResponse,
    ) -> AgentRound:
        transcript_snapshot = self.transcript
        tool_calls = list(self._current_tool_calls)
        tool_results = list(self._current_tool_results)

        result_calls = _extract_sequence(result, ("tool_calls", "toolInvocations", "tool_invocations"))
        result_results = _extract_sequence(result, ("tool_results", "toolOutputs", "tool_outputs"))

        if result_calls:
            tool_calls.extend(item for item in result_calls if item not in tool_calls)
        if result_results:
            tool_results.extend(item for item in result_results if item not in tool_results)

        round_record = AgentRound(
            index=index,
            user_message=user_message,
            response_text=response_text,
            result=result,
            transcript=transcript_snapshot,
            messages=list(self._current_messages),
            tool_history={
                "calls": tool_calls,
                "results": tool_results,
            },
        )
        return round_record

    def act(
        self,
        user_message: str | None = None,
        *,
        tools: Iterable[Any] | None = None,
        tool_names: Sequence[str] | None = None,
        config: Optional[Mapping[str, Any]] = None,
        callbacks: Optional[CallbackMap] = None,
        handle_invalid_tool_request: Any | None = None,
        **act_kwargs: Any,
    ) -> tuple[str, StructuredResponse]:
        self._current_messages = []
        self._current_tool_calls = []
        self._current_tool_results = []

        round_index = len(self.rounds)
        if self._on_round_start:
            self._on_round_start({
                "index": round_index,
                "user_message": user_message,
                "session": self,
            })

        composed_callbacks = self._compose_callbacks(callbacks)
        text, result = self.chat_session.act(
            user_message,
            tools=tools,
            tool_names=tool_names,
            config=config,
            callbacks=composed_callbacks,
            handle_invalid_tool_request=handle_invalid_tool_request,
            **act_kwargs,
        )

        round_record = self._finalize_round(
            index=round_index,
            user_message=user_message,
            response_text=text,
            result=result,
        )
        self.rounds.append(round_record)

        if self._on_round_end:
            self._on_round_end(round_record)

        return text, result

    def last_round(self) -> AgentRound | None:
        return self.rounds[-1] if self.rounds else None

    @property
    def tool_history(self) -> list[dict[str, list[Any]]]:
        return [round.tool_history for round in self.rounds]
