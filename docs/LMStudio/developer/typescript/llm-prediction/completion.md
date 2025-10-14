---
title: Text Completions
description: "Provide a string input for the model to complete"
---

Use `llm.complete(...)` to generate text completions from a loaded language model. Text completions mean sending an non-formatted string to the model with the expectation that the model will complete the text.

This is different from multi-turn chat conversations. For more information on chat completions, see [Chat Completions](./chat-completion).

## 1. Instantiate a Model

First, you need to load a model to generate completions from. This can be done using the `model` method on the `llm` handle.

```lms_code_snippet
  title: "index.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";

        const client = new LMStudioClient();
        const model = await client.llm.model("qwen2.5-7b-instruct");
```

## 2. Generate a Completion

Once you have a loaded model, you can generate completions by passing a string to the `complete` method on the `llm` handle.

```lms_code_snippet
  variants:
    Streaming:
      language: typescript
      code: |
        const completion = model.complete("My name is", {
          maxTokens: 100,
        });

        for await (const { content } of completion) {
          process.stdout.write(content);
        }

        console.info(); // Write a new line for cosmetic purposes

    "Non-streaming":
      language: typescript
      code: |
        const completion = await model.complete("My name is", {
          maxTokens: 100,
        });

        console.info(completion.content);
```

## 3. Print Prediction Stats

You can also print prediction metadata, such as the model used for generation, number of generated tokens, time to first token, and stop reason.

```lms_code_snippet
  title: "index.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        console.info("Model used:", completion.modelInfo.displayName);
        console.info("Predicted tokens:", completion.stats.predictedTokensCount);
        console.info("Time to first token (seconds):", completion.stats.timeToFirstTokenSec);
        console.info("Stop reason:", completion.stats.stopReason);
```

## Example: Get an LLM to Simulate a Terminal

Here's an example of how you might use the `complete` method to simulate a terminal.

```lms_code_snippet
  title: "terminal-sim.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";
        import { createInterface } from "node:readline/promises";

        const rl = createInterface({ input: process.stdin, output: process.stdout });
        const client = new LMStudioClient();
        const model = await client.llm.model();
        let history = "";

        while (true) {
          const command = await rl.question("$ ");
          history += "$ " + command + "\n";

          const prediction = model.complete(history, { stopStrings: ["$"] });
          for await (const { content } of prediction) {
            process.stdout.write(content);
          }
          process.stdout.write("\n");

          const { content } = await prediction.result();
          history += content;
        }
```

## Advanced Usage

### Prediction metadata

Prediction responses are returned as `PredictionResult` objects that expose rich metadata about the
inference request. You can inspect the model that produced the output, the configuration that was
used, and a detailed breakdown of timing statistics such as stop reason, time to first token, and
tokens per second. Refer to the TypeScript SDK reference for the full list of available fields.

### Progress callbacks

Long prompts can spend noticeable time in the "prompt processing" phase before the first token
streams back. The TypeScript SDK supports two callbacks that help you react to that lifecycle:

* `onPromptProcessingProgress(progress)` fires with a number between `0` and `1` while the prompt is
  being embedded. You can use it to update progress bars or log statements.
* `onFirstToken(info)` runs exactly once when the first token is emitted. The `info` object includes
  properties such as `elapsedMs`, allowing you to measure time-to-first-token latency.

The example below shows how to connect both callbacks while still streaming tokens with `for await`.

```lms_code_snippet
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";

        const client = new LMStudioClient();
        const model = await client.llm.model("qwen2.5-7b-instruct");

        const prediction = model.complete("My name is", {
          onPromptProcessingProgress: (progress) => {
            const percent = Math.round(progress * 100);
            process.stdout.write(`\rProcessing prompt… ${percent}%`);
          },
          onFirstToken: ({ elapsedMs }) => {
            process.stdout.write(`\nFirst token after ${elapsedMs} ms\n`);
          },
        });

        for await (const { content } of prediction) {
          process.stdout.write(content);
        }

        await prediction.result();
        process.stdout.write("\n");
```

### Prediction configuration

You can also specify the same prediction configuration options as you could in the in-app chat
window sidebar. Please consult your specific SDK to see exact syntax.
