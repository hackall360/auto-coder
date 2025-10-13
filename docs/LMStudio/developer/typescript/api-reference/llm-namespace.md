---
title: "`client.llm`"
sidebar_title: "`client.llm` namespace"
description: "`client.llm` - API reference for the llm namespace in an `LMStudioClient` instance"
index: 6
---

The `llm` namespace surfaces lifecycle operations for large language models. It lives on every
[`LMStudioClient`](./lmstudioclient.md) instance and returns strongly typed `LLM` handles whose
methods cover prediction, tokenization, and prompt templating.

```ts
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
const llama = await client.llm.model("lmstudio-community/llama-3.2-3b-instruct");
const reply = await llama.respond("Hello there!");
console.info(reply.content);
```

Once you have a handle, reuse it for subsequent requests to avoid reloading weights. The sections
below document the namespace-level helpers that fetch, load, and enumerate those handles.

## Core operations

| Method | Description |
| --- | --- |
| [`model(modelKey?, options?)`](#retrieve-a-handle) | Get a handle to any loaded model, optionally loading it on demand. |
| [`load(modelKey, options?)`](#load-additional-instances) | Force LM Studio to spin up a brand-new instance even if one already exists. |
| [`listLoaded()`](#inspect-loaded-models) | Enumerate every LLM currently in memory. |
| [`unload(identifier)`](#unload-a-model) | Free GPU/CPU resources tied to a specific instance identifier. |
| [`createDynamicHandle(query)`](#dynamic-handles) | Obtain a handle that tracks "whichever model matches this query". |
| [`createDynamicHandle(identifier)`](#dynamic-handles) | Short-hand for `createDynamicHandle({ identifier })`. |
| [`createDynamicHandleFromInstanceReference(instanceRef)`](#dynamic-handles) | Build a handle from the server's immutable instance reference (advanced). |

Every function returns a `Promise` and mirrors the options used by the CLI: auto-load models, control
idle eviction (TTL), monitor load progress, and cancel work with `AbortController`.

## Retrieve a handle

```ts
const model = await client.llm.model();
```

- Calling `.model()` with no arguments returns the first loaded model. The promise rejects if no LLMs
  are in memory.
- Pass a `modelKey` (for example `"lmstudio-community/phi-3.5-mini-instruct"`) to target a specific
  download. LM Studio loads it on demand when necessary.
- The optional `options` argument matches `BaseLoadModelOpts<LLMLoadModelConfig>`:

  ```ts
  const llama = await client.llm.model("lmstudio-community/llama-3.2-3b-instruct", {
    identifier: "analysis-llama",      // Custom instance name
    ttl: 3600,                         // Idle seconds before auto-unload
    signal: abortController.signal,    // Cancel loading if needed
    onProgress: (progress) => {
      console.info(`Loading… ${Math.round(progress * 100)}%`);
    },
    config: {
      contextLength: 8192,
      gpu: { /* see LLMLoadModelConfig docs */ },
      flashAttention: true,
    },
  });
  ```

  See [`LLMLoadModelConfig`](./llm-load-model-config.md) for the available load-time parameters.

Subsequent calls to `client.llm.model("…")` reuse the same instance. To change load configuration,
explicitly `unload` the current instance or create an additional one with [`client.llm.load`](#load-additional-instances).

## Load additional instances

```ts
const streaming = await client.llm.load("lmstudio-community/llama-3.2-3b-instruct", {
  identifier: "streaming-copy",
  ttl: 600,
});
const analysis = await client.llm.load("lmstudio-community/llama-3.2-3b-instruct", {
  identifier: "analysis-copy",
  config: { contextLength: 16384 },
});
```

`load()` always returns a fresh `LLM` handle even if another instance for the same model key already
exists. Use this when you need separate runtime settings (for example, different context lengths or
GPU splits). Options match those accepted by `.model()`.

## Inspect loaded models

```ts
const handles = await client.llm.listLoaded();
for (const handle of handles) {
  const info = await handle.getModelInfo();
  console.info(handle.identifier, info.displayName, info.maxContextLength);
}
```

`listLoaded()` resolves to an array of `LLM` handles. Each handle exposes metadata (`identifier`,
`path`, `instanceReference`) and the full suite of prediction helpers.

To inspect loaded models without instantiating handles, call `handle.getModelInfo()` or query by
identifier with [`client.llm.createDynamicHandle`](#dynamic-handles).

## Unload a model

```ts
await client.llm.unload("analysis-copy");
```

Pass the instance identifier (the same value returned by `handle.identifier` or listed in the LM
Studio UI). Unloading frees resources immediately instead of waiting for the idle TTL.

You can also unload via the handle itself: `await handle.unload()`.

## Dynamic handles

Dynamic handles are lightweight references that stay valid even if the underlying model is reloaded.
They are useful when you need to hold onto a capability rather than a specific runtime instance.

```ts
const dynamic = client.llm.createDynamicHandle({
  path: "lmstudio-community/llama-3.2-3b-instruct",
});

const prediction = dynamic.respond("Summarize the release notes.");
for await (const fragment of prediction) {
  process.stdout.write(fragment.content);
}
```

- Queries accept either a `string` identifier or a `ModelQuery` object (`domain`, `identifier`, `path`,
  `vision`).
- The handle stays constructed even if the matching model unloads; subsequent method calls fail until a
  new instance satisfies the query.
- Advanced integrations that cache the server-issued `instanceReference` can resurrect a handle via
  `createDynamicHandleFromInstanceReference(instanceReference)`.

## Preview a prompt template

Prompt templating lives on the `LLM` handle returned by the methods above:

```ts
import { Chat } from "@lmstudio/sdk";

const formatted = await llama.applyPromptTemplate(Chat.from("Hello"));
console.info(formatted);
```

Use this to debug prompt rendering without triggering a prediction. The helper accepts anything that
`Chat.from(...)` understands. See [Apply prompt templates](../../more/apply-prompt-template.md) for
usage patterns and option details.

## Asynchronous environments

All namespace helpers are `async`. When using the SDK in ESM modules or top-level `await` contexts, no
additional scaffolding is required. In CommonJS projects create an async function to wrap your
workflow.

## See also

- [`.model()` API reference](./model.md) for handle semantics and streaming behaviour.
- [Load and manage models](../model-management/loading.md) for higher-level lifecycle patterns.
- [List loaded models](../model-management/list-loaded.md) and
  [list downloaded models](../model-management/list-downloaded.md) for complementary utilities.
- [Chat completion guide](../../llm-prediction/chat-completion.md) and
  [text completion guide](../../llm-prediction/completion.md) to explore prediction flows once you
  have an `LLM` handle.
