---
title: Predicting with LLMs
sidebar_title: Overview
description: APIs to predict with LLMs managed by LM Studio.
index: 1
---

LM Studio provides a unified TypeScript interface for generating text and structured
responses from local models. This page introduces the SDK surface you will use to
create predictions, explains how LM Studio picks the right inference engine, and
points you to deeper guides for specific workflows.

## SDK entry points

All LLM interactions begin with the [`LMStudioClient`](../index.md). The client
exposes an `llm` namespace that lets you discover, configure, and invoke local
models.

```lms_code_snippet
  title: "llm.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";
        const client = new LMStudioClient();

        // Create a handle to an LLM by identifier or friendly name
        const model = await client.llm.model("llama-3.2-1b-instruct");

        // Issue a single prediction
        const prediction = await model.respond("Summarize the plot of The Hobbit.");
        console.log(prediction.content);
```

A model handle is reusable—you can call `respond`, `complete`, or start a streaming
prediction without reloading model weights. You can also set defaults such as
`maxTokens`, `temperature`, and `stop` sequences when you acquire the model handle,
then override them per request. See [prediction parameters](./parameters.md) for the
full list of configurable options.

## Engine selection

LM Studio bundles multiple optimized runtimes (including MLX on Apple Silicon and
llama.cpp on every platform). When you request a model, the SDK inspects the model's
manifest and your current hardware to automatically select the engine that offers the
best balance of performance and compatibility. In most cases you do not need to make
any changes—the selection logic handles device-specific quirks such as GPU/CPU
availability and low-memory fallbacks.

If you have a preferred runtime, you can override the automatic selection when you
create the model handle:

```ts
const model = await client.llm.model("llama-3.2-1b-instruct", {
  engine: "mlx", // or "llamacpp"
});
```

To audit the decision that LM Studio made for a model already in memory, call
`model.getLoadConfig()`; the resolved engine is returned alongside other load
parameters.

## Next steps

- Learn how to generate text completions with [`model.complete`](./completion.md).
- Build conversational experiences with [`model.respond`](./working-with-chats.md).
- Enforce structured outputs for downstream automation using the
  [structured response guide](./structured-response.md).
- Explore advanced topics such as [speculative decoding](./speculative-decoding.md),
  [image inputs](./image-input.md), and [cancelling predictions](./cancelling-predictions.md).

Once you are comfortable with these building blocks you can combine them with the SDK's
agent APIs, embeddings support, and model management tooling to deliver fully local AI
experiences.
