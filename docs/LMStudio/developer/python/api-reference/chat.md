---
title: "`Chat`"
sidebar_title: "`Chat`"
description: "`Chat` - API reference for representing a chat conversation with an LLM"
index: 5
---

`Chat` is a mutable container that tracks the ordered messages shared between the user, the
assistant, the system, and tool executions. It is the preferred structure for providing context to
[`LLM.respond()`](./respond.md) and agent workflows such as
[`LLM.act()`](../../agent/act.md), and it is accepted anywhere the SDK expects a chat history (for
example [`LLM.apply_prompt_template()`](./llm-namespace.md#preview-a-prompt-template)).

Import the class from the top-level SDK namespace:

```python
from lmstudio import Chat
```

## Creating a chat history

### Start from scratch

```python
chat = Chat()
```

Create an empty conversation. You can optionally seed the conversation with a system prompt:

```python
chat = Chat(initial_prompt="You are a concise assistant.")
```

Passing `initial_prompt` is equivalent to calling [`add_system_prompt`](#add_system_prompt) after
construction.

### Clone or restore an existing history

Use `Chat.from_history()` when you already have serialized messages:

```python
saved_history = {
    "messages": [
        {"role": "system", "content": [{"text": "Stay upbeat."}]},
        {"role": "user", "content": [{"text": "Tell me a joke."}]},
    ]
}

chat = Chat.from_history(saved_history)
```

`from_history()` accepts:

- Another `Chat` instance (creates a deep copy)
- A `ChatHistoryData` struct or its plain-`dict` form
- A single `str`, which becomes a one-line user message

## Mutating the conversation

The public helpers on `Chat` correspond to the roles that can appear in prediction transcripts.
They accept flexible input types that are validated by the SDK before they are serialized for the
server. The most common helpers are summarised below.

| Method | Signature | Description |
| --- | --- | --- |
| [`add_system_prompt`](#add_system_prompt) | `add_system_prompt(prompt)` | Add a system message that sets behaviour for future turns. |
| [`add_user_message`](#add_user_message) | `add_user_message(content, *, images=())` | Append user text and optional files or images. |
| [`add_assistant_response`](#add_assistant_response) | `add_assistant_response(response, tool_call_requests=())` | Record an assistant reply and any tool call requests returned by the model. |
| [`add_tool_results`](#add_tool_results) | `add_tool_results(results)` | Append tool outputs returned to the assistant. |
| [`append`](#append) | `append(message)` | Copy an already-formatted message (dict or struct) into the chat. |
| [`copy`](#copy) | `copy()` | Produce a deep copy of the chat. |

### `add_system_prompt`

```python
chat.add_system_prompt("Answer in fewer than 20 words.")
```

Accepts either a string, a `TextData` instance, or a dictionary containing a `text` field.
Consecutive system prompts are rejected to prevent
accidental duplication.

### `add_user_message`

```python
from lmstudio import prepare_image

handle = prepare_image("diagram.png")
chat.add_user_message(
    "Summarize the attachment",
    images=[handle],
)
```

`content` may be a string, a `TextData` object, a `FileHandle`, or an iterable combining those
values. Additional file handles can be supplied through the `images` keyword (other file types will
become available as the server adds support). When multiple user calls occur back to back the SDK
merges them into a single multi-part message so the serialized transcript always alternates roles.

### `add_assistant_response`

```python
response = chat.add_assistant_response("Here is what I found:")
chat.add_assistant_response(
    {"text": "Calling the lookup tool."},
    tool_call_requests=[{"type": "toolCallRequest", "toolCallId": "lookup"}],
)
```

`response` accepts the same shapes as user content (string, `TextData`, or `FileHandle`). Tool call
requests are optional; when provided they can be dictionaries with `toolCallId`/`content` fields or
`ToolCallRequest` structs. The helper ensures two assistant messages never appear consecutively in
the history.

### `add_tool_results` and `add_tool_result`

Record the results of a tool invocation so the model can read them on the next prediction round:

```python
chat.add_tool_results([
    {
        "toolCallId": "lookup",
        "content": [{"text": "Temperature is 21°C."}],
    }
])
```

For a single result you can call `add_tool_result(result)` instead of wrapping it in a list. Both
helpers accept dictionaries that match the wire schema or pre-built `ToolCallResultData` structs.

### `add_entry` and `append`

`add_entry(role, content)` routes to the specialised helpers above based on the role string, while
`append(message)` copies an already-normalized message object (`AnyChatMessage` or `dict`). These
are convenient when you receive structured events from streaming APIs like
[`PredictionStream`](./llm-namespace.md#streaming-predictions).

### `copy`

`chat.copy()` (and the equivalent `copy.copy(chat)` or `copy.deepcopy(chat)`) returns a fully
independent conversation that can be mutated without affecting the original object. This is useful
when branching conversations or retrying with a different prediction configuration.

## Using `Chat` with predictions

```python
import lmstudio as lms

chat = Chat(initial_prompt="You are a travel assistant.")
chat.add_user_message("Plan a 3 day trip to Tokyo")

with lms.Client() as client:
    model = client.llm.model("lmstudio-community/llama-3.2-3b-instruct")
    result = model.respond(chat)
    chat.append(result.message)  # Keep the assistant turn in the transcript
```

`Chat` instances can be reused for subsequent turns, or copied before modifying them to preserve the
previous state. When passing `Chat` to [`LLM.respond()`](./respond.md) or
[`LLM.act()`](../../agent/act.md), the SDK automatically serializes the messages into the format
expected by the LM Studio server. You can also hand the same object to
[`LLM.apply_prompt_template()`](./llm-namespace.md#preview-a-prompt-template) to preview how the
conversation will be rendered prior to sending a prediction.

## See also

- [`LLM.respond()`](./respond.md) – generate chat completions from an LLM handle.
- [Chat completion guide](../../llm-prediction/chat-completion.md) – end-to-end walkthrough of
  multi-turn interactions.
- [`PredictionStream`](./llm-namespace.md#streaming-predictions) – consume streaming events while
  updating a `Chat` incrementally.
- [Prompt template helpers](../../more/apply-prompt-template.md) – format conversations outside of a
  prediction call.
