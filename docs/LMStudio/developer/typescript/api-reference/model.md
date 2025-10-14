---
title: "`.model()`"
sidebar_title: "`.model()`"
description: ".model() - API reference for obtaining a model handle from an `LMStudioClient` instance"
index: 2
---

The `.model()` method returns a handle to a language model managed by LM Studio. It can attach to
an already loaded model or trigger a just-in-time (JIT) load when necessary. Once you have a handle
you can reuse it to run multiple predictions without paying the cost of reloading weights.

## Signatures

```ts
// Promise-based signature
const model = await client.llm.model(
  key?: string,
  options?: {
    config?: LLMLoadModelConfigInput;
    ttl?: number;
  }
);
```

- The method lives on both `client.llm` (LLMs) and `client.embedding` (embedding models).
- Type definitions such as `LLMLoadModelConfigInput` ship with `@lmstudio/sdk`.

## Parameters

| Name | Type | Description |
| --- | --- | --- |
| `key` | `string \| undefined` | Identifier or friendly name of the model to load. If omitted, `.model()` returns the first LLM that is already in memory. |
| `options.config` | [`LLMLoadModelConfigInput`](../llm-load-model-config.md) \| `undefined` | Load-time configuration applied only if this call ends up loading the model (context length, GPU ratio, engine overrides, etc.). |
| `options.ttl` | `number \| undefined` | Idle time in **seconds** before LM Studio automatically unloads the model. Only applied when the call performs a load. |

## Default model selection

Calling `.model()` with no arguments simply gives you a handle to whichever model is currently
loaded in LM Studio. This is the fastest way to use the GUI-selected model from TypeScript code.

```ts
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
const model = await client.llm.model();
const result = await model.respond("Summarize the plot of Dune in one paragraph.");
console.info(result.content);
```

This call **never** initiates a load. If no LLMs are resident, the SDK rejects the promise with an
error prompting you to load a model first (either via the LM Studio UI or by supplying a `key`).

## Select a specific model

Provide a `key` to deterministically choose the model you need. If the model is not already loaded,
LM Studio loads it on demand and applies any configuration you pass during that first load.

```ts
const llama = await client.llm.model("lmstudio-community/llama-3.2-3b-instruct", {
  config: {
    contextLength: 8192,
    gpu: { ratio: 0.5 },
  },
  ttl: 3600, // keep the model in memory for up to one idle hour
});
```

Subsequent `client.llm.model("lmstudio-community/llama-3.2-3b-instruct")` calls reuse the cached
instance regardless of new `config` or `ttl` values; adjust load settings by explicitly creating a
fresh instance with [`client.llm.load`](../model-management/loading.md#load-a-new-instance).

## Streaming vs. non-streaming handles

A model handle works for both streaming and aggregate response patterns:

```ts
const prediction = model.respond("List three renewable energy sources.");

// Streamed handling (tokens arrive as they are generated)
for await (const fragment of prediction) {
  process.stdout.write(fragment.content);
}

// Aggregate handling (wait for the full response)
const full = await prediction.result();
console.info("Stop reason:", full.stats.stopReason);
```

- `model.respond(...)`, `model.complete(...)`, and similar methods return a `Prediction` object that
  implements `AsyncIterable` for streaming.
- Calling `.result()` (or simply `await model.respond(...)`) resolves to the fully collected
  response. Use the approach that matches your UX without needing different model handles.

## Caching, JIT loading, and TTL behaviour

- **JIT loading** – When you provide a `key`, LM Studio loads the model on demand if necessary.
  Load-time parameters (`config`, `ttl`) only apply during that first load.
- **Handle reuse** – `.model(key)` always returns the single cached instance for that identifier.
  Spawn additional copies with [`client.llm.load`](../model-management/loading.md).
- **Idle eviction** – Each loaded model inherits the workspace default idle timeout (60 minutes by
  default). Override it per call with the `ttl` option to keep a model resident longer or reclaim
  memory sooner.

## See also

- [Model management: loading and unloading](../model-management/loading.md)
- [Prediction parameters](../../llm-prediction/parameters.md) for setting inference defaults
- [Get the active model's load configuration](../model-info/get-load-config.md)
