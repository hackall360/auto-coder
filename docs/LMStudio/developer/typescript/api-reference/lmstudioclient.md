---
title: "`LMStudioClient`"
sidebar_title: "`LMStudioClient`"
description: "LMStudioClient - API reference for the `LMStudioClient` class"
index: 1
---

`LMStudioClient` is the entry point into the LM Studio TypeScript SDK. A single instance maintains the
connection to the desktop runtime, brokers authentication, and exposes the typed namespaces that carry
out language, embedding, file, diagnostics, and system operations. You typically create one client per
process and reuse it for the lifetime of your application.

## Creating a client

The default constructor attempts to connect to a locally running LM Studio desktop application. It will
probe the standard websocket ports until it finds a server, so most applications can simply do the
following:

```ts
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
```

The class also implements `Symbol.asyncDispose`, which means you can rely on top-level `await using` in
Node 22+ or Bun to automatically close the underlying websocket when the scope exits:

```ts
await using client = new LMStudioClient();
// use the client
```

### Custom connection and authentication

Pass an `LMStudioClientConstructorOpts` object to the constructor when you need to point at a remote
host, change logging behavior, or coordinate credentials across multiple processes.

```ts
const client = new LMStudioClient({
  baseUrl: "wss://studio.internal.example.com:9000",
  logger: myStructuredLogger,
  verboseErrorMessages: true,
  clientIdentifier: "batch-service",
  clientPasskey: process.env.LMSTUDIO_PASSKEY,
});
```

- `baseUrl` lets you override the auto-discovery logic and target a specific websocket endpoint.
- `logger` accepts any object with `info`, `warn`, `error`, and `debug` methods. The SDK defaults to
  `console` but you can inject your own structured logger.
- `verboseErrorMessages` instructs the SDK to include server-side stack traces in thrown errors—useful
  while developing.
- `clientIdentifier` and `clientPasskey` control authentication. Reuse the same pair when you want two
  client instances to share server-side resources (such as cached model weights).

## Primary namespaces

Once constructed, the client exposes cohesive namespaces for the different feature areas:

| Namespace | Description |
| --- | --- |
| [`client.llm`](./llm-namespace.md) | Load, run, and unload language models. Handles chat, completion, and agent workflows. |
| [`client.embedding`](../embedding/index.md) | Access embedding models for text vectorization and similarity search. |
| [`client.files`](../llm-prediction/image-input.md) | Prepare binary assets (images, audio, documents) before sending them to a model. |
| [`client.system`](./system-namespace.md) | Inspect the desktop host, list downloaded artifacts, watch for disconnects, and manage the embedded HTTP server. |
| `client.diagnostics` | Surface performance and health details while debugging workloads. |
| `client.repository` | Authenticate with LM Studio Hub, list remote artifacts, and manage entitlements. |
| `client.plugins` *(experimental)* | Discover and interact with LM Studio plugins while the plugin system stabilizes. |

Every namespace exposes strongly typed methods documented in the corresponding reference pages. The
client itself only acts as a connection manager and registry.

## Common workflow

Most applications follow the same pattern: create a client, load the model you need, issue one or more
predictions, and then unload the model when you are finished.

```ts
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();

// 1. Load a model (downloads must already exist on disk).
const model = await client.llm.model("llama-3.2-1b-instruct");

// 2. Issue a chat-style response.
const response = await model.respond([
  { role: "system", content: "You are a concise assistant." },
  { role: "user", content: "Summarize the latest release notes." },
]);

console.log(response.content);

// 3. Unload the model when it is no longer needed.
await model.unload();
```

When you need to perform a single request, it is common to combine the workflow with `await using` so
the client detaches automatically:

```ts
await using client = new LMStudioClient();
const model = await client.llm.model("llama-3.2-1b-instruct");
const { content } = await model.complete("Write a haiku about TypeScript.");
console.log(content);
await model.unload();
```

See the rest of the API reference for deeper dives into each namespace and their helper methods.
