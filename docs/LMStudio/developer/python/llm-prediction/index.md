---
title: Predicting with LLMs
sidebar_title: Overview
description: APIs to predict with LLMs managed by LM Studio.
index: 1
---

LM Studio runs local LLMs through runtime engines such as [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and, on
Apple Silicon, Apple's [`MLX`](https://github.com/ml-explore/mlx). When you request a model through the SDK, the server
looks at the model's compatibility type and automatically launches the appropriate engine, so your code only needs to
specify the model key (for example `"llama-3.2-1b-instruct"` or `"qwen2-vl-2b-instruct"`).

### Pick an API surface that matches your application

After importing the SDK (`import lmstudio as lms`), you can choose among three equivalent entry points that share the
same LLM handle API:

- `lms.llm(...)` — a convenience helper that reuses the default client, perfect for scripts and notebooks.
- `lms.Client().llm.model(...)` — the synchronous scoped resource API that gives you deterministic control over
  connections and loaded models.
- `lms.AsyncClient().llm.model(...)` — the asynchronous variant for structured-concurrency applications (requires SDK
  1.5.0 or newer).

All three return an `LLMHandle` that can send single responses, stream tokens, or be configured before issuing
predictions.

### Learn how to issue predictions

Once you have a handle, dive into the rest of this section for task-specific guidance:

- [Chat Completions](./chat-completion) for multi-turn conversations.
- [Text Completions](./completion) for single-prompt continuation workloads.
- [Working with Chats](./working-with-chats) to manage reusable conversation state.
- [Parameters](./parameters) to customize sampling, load-time options, and idle TTL.
- [Structured Response](./structured-response) to coerce JSON or grammar-bound outputs.
- [Image Input](./image-input) for multimodal prompts.
- [Speculative Decoding](./speculative-decoding) to trade extra compute for lower latency.
- [Cancelling Predictions](./cancelling-predictions) to stop in-flight requests from any API surface.

Every page uses the same model-loading patterns shown above, so you can freely mix and match synchronous, asynchronous,
and streaming examples to suit your workflow.
