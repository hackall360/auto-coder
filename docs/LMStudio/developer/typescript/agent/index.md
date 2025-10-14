---
title: Agent workflows
sidebar_title: Overview
description: Learn how to turn LLMs into autonomous agents that call tools and react to their outputs.
index: 1
---

The agent APIs in `lmstudio-js` let you orchestrate multi-step workflows where a model chooses tools, interprets their
results, and produces final answers. Agents are ideal when you need the model to go beyond single responses and instead
loop through planning, execution, and reflection until a task is complete.

## Core agent APIs

- **`model.act(...)`** – Runs an interactive loop where the LLM can automatically call tools, receive their output, and decide
  whether to invoke more tools or return a final message. Use this when you want the model to stay in control of the flow.
- **`tool(...)` helper** – Defines strongly typed tools (powered by [Zod](https://zod.dev)) that the LLM can call. Tools can be
  synchronous or async and can interface with files, services, or any local capability.
- **`onMessage` / `onPredictionFragment` callbacks** – Stream intermediate thinking, monitor tool invocations, and build rich
  UIs around the agent loop.

A minimal agent looks like this:

```lms_code_snippet
  title: "agent.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient, tool } from "@lmstudio/sdk";
        import { z } from "zod";

        const client = new LMStudioClient();
        const calculator = tool({
          name: "multiply",
          description: "Multiply two numbers together.",
          parameters: { a: z.number(), b: z.number() },
          implementation: ({ a, b }) => a * b,
        });

        const model = await client.llm.model("qwen2.5-7b-instruct");
        await model.act("What is 1234 × 5678?", [calculator], {
          onMessage: (message) => console.log(message.toString()),
        });
```

## Typical use cases

- **Local automation** – Let an LLM trigger scripts, CLI commands, and file operations on your machine.
- **Research and analysis** – Combine web search, retrieval, and computation tools so the model can gather and reason over data.
- **Developer copilots** – Build agents that edit files, run tests, or scaffold projects completely offline.
- **Interactive assistants** – Create conversational experiences that respond in natural language while quietly calling tools.

## Learn more

Dive deeper into the agent capabilities:

- [Call tools automatically with `.act()`](./act)
- [Define reusable tools and schemas](./tools)
- Explore the [LLM prediction docs](../llm-prediction) for prompt construction and parameter tuning tips

When you are ready to combine agents with other features (such as embeddings or model management), return to the
[TypeScript SDK guide](../index.md) for cross-cutting concepts and examples.
