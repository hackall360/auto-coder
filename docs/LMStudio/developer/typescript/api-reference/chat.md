---
title: "`Chat`"
sidebar_title: "`Chat`"
description: "`Chat` - API reference for representing a chat conversation with an LLM"
index: 5
---

`Chat` is a mutable helper for building and replaying multi-turn conversations. It owns an ordered
collection of `ChatMessage` objects and understands every shape that the LM Studio TypeScript SDK
accepts as a chat history. Use it to construct inputs for
[`.respond()`](./respond.md), agent workflows such as
[`LLM.act()`](../../agent/act.md), or utilities like
[`LLM.applyPromptTemplate()`](./llm-namespace.md#preview-a-prompt-template).

```ts
import { Chat } from "@lmstudio/sdk";
```

## Creating a chat history

### Start from an empty transcript

```ts
const chat = Chat.empty();
chat.append("system", "You are a concise assistant.");
```

`Chat.empty()` returns a mutable instance. All mutators (`append`, `pop`, `withAppended`, …) operate on
that instance until you branch it.

### Bootstrap from existing data

`Chat.from(initializer)` creates a mutable copy from any `ChatLike` value:

```ts
const chat = Chat.from([
  { role: "system", content: "Stay upbeat." },
  { role: "user", content: "Tell me a joke." },
]);
```

Valid initializers include:

- A plain string (treated as a single user message)
- Arrays of `{ role, content }` objects or richer `ChatMessageInput` structures
- A `Chat` or `ChatHistoryData` instance (creates a deep copy)
- A single `ChatMessage` or `ChatMessageData` object

This makes it easy to restore serialized transcripts or fork an in-memory conversation before
continuing.

### Functional branching with `withAppended`

Prefer `withAppended(...)` when you want to stage changes without mutating the original:

```ts
const nextAttempt = chat.withAppended("user", "Rephrase that in fewer than 20 words.");
```

`withAppended` returns a brand-new `Chat` while leaving `chat` untouched—perfect for retries or A/B
experiments.

## Mutation helpers

| Helper | Signature (simplified) | Description |
| --- | --- | --- |
| `append` | `append(role: ChatMessageRoleData, content: string, opts?: ChatAppendOpts)`<br/>`append(message: ChatMessageLike)` | Push a new turn into the transcript. The `ChatAppendOpts` bag currently accepts `images?: FileHandle[]`. |
| `withAppended` | `withAppended(role, content, opts?)`<br/>`withAppended(message)` | Return a cloned chat that already includes the supplied message. |
| `pop` | `pop(): ChatMessage` | Remove and return the most recent message; throws if the chat is empty. |
| `at` | `at(index: number): ChatMessage` | Random-access getter. Negative indexes count from the end. |
| `getMessagesArray` | `getMessagesArray(): ChatMessage[]` | Materialize the entire history as `ChatMessage` objects. |
| `map` / `flatMap` | `map(mapper)` / `flatMap(mapper)` | Transform the transcript while preserving message metadata. |
| `getSystemPrompt` | `getSystemPrompt(): string` | Return the first system message as plain text (empty string if none). |
| `replaceSystemPrompt` | `replaceSystemPrompt(text: string): void` | Overwrite the leading system message, creating one if necessary. |
| `hasFiles` | `hasFiles(): boolean` | Quick check for any attached `FileHandle`s. |
| `getAllFiles` | `getAllFiles(client: LMStudioClient): FileHandle[]` | Collect every file handle present in the history. |
| `consumeFiles` / `consumeFilesAsync` | `consumeFiles(client, predicate)` | Remove matching files while returning them for custom processing. |

### Appending rich content

Combine text and prepared files in a single turn. The SDK ensures role ordering stays valid (assistant
messages never follow assistant messages, etc.).

```ts
const image = await client.files.prepareImage("./diagram.png");

chat.append("user", "Summarize the attachment", {
  images: [image],
});
```

You can also copy pre-composed messages, for example when replaying streamed fragments:

```ts
import type { ChatMessage } from "@lmstudio/sdk";

function appendFromStream(chat: Chat, message: ChatMessage) {
  chat.append(message);
}
```

### Inspecting and pruning history

Use iteration helpers to walk the transcript, whether you need summary statistics or to filter out
sensitive turns before logging:

```ts
const userTurns = chat
  .map((message) => message)
  .filter((message) => message.isUserMessage())
  .map((message) => message.getText());

chat.filterInPlace((message) => !message.isSystemPrompt());
```

`filterInPlace` keeps only the messages that satisfy the predicate, which is handy when trimming
branches during long-running sessions. Call `pop()` to roll back the last turn.

### Working with file attachments

The file helpers require an `LMStudioClient` instance because the handles originate from the server.

```ts
const removed = chat.consumeFiles(client, (file) => file.mimeType?.startsWith("image/"));
console.info("Stripped", removed.length, "inline images");
```

The asynchronous variant `consumeFilesAsync` lets you await transformations (for example, resizing
images) before deciding whether to remove them.

## Using `Chat` with predictions

```ts
import { Chat, LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
const model = await client.llm.model();

const chat = Chat.empty();
chat.append("system", "You are a travel planner.");
chat.append("user", "Plan a 3 day trip to Tokyo.");

const prediction = model.respond(chat, {
  onMessage: (message) => chat.append(message),
});

for await (const fragment of prediction) {
  process.stdout.write(fragment.content);
}

const result = await prediction.result();
console.info("Stop reason:", result.stats.stopReason);
```

A single `Chat` instance can be reused across turns. Use `withAppended(...)` or `Chat.from(chat)` to
branch if you need to retry with different [`LLMPredictionConfig`](./llm-prediction-config-input.md)
settings.

## See also

- [Chat completion guide](../../llm-prediction/chat-completion.md) for an end-to-end walkthrough.
- [Working with Chats](../../llm-prediction/working-with-chats.md) for additional patterns and
  initializer shapes.
- [`LLM.respond()`](./respond.md) – generate assistant replies from a loaded model.
- [Apply prompt templates](../../more/apply-prompt-template.md) to inspect how a conversation will be
  rendered.
