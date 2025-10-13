---
title: Chat Completions
sidebar_title: Chat
description: APIs for a multi-turn chat conversations with an LLM
index: 2
---

Use `llm.respond(...)` to generate completions for a chat conversation.

## Quick Example: Generate a Chat Response

The following snippet shows how to stream the AI's response to quick chat prompt.

```lms_code_snippet
  title: "index.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";
        const client = new LMStudioClient();

        const model = await client.llm.model();

        for await (const fragment of model.respond("What is the meaning of life?")) {
          process.stdout.write(fragment.content);
        }
```

## Obtain a Model

First, you need to get a model handle. This can be done using the `model` method in the `llm` namespace. For example, here is how to use Qwen2.5 7B Instruct.

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

There are other ways to get a model handle. See [Managing Models in Memory](./../manage-models/loading) for more info.

## Manage Chat Context

The input to the model is referred to as the "context". Conceptually, the model receives a multi-turn conversation as input, and it is asked to predict the assistant's response in that conversation.

```lms_code_snippet
  variants:
    "Using an array of messages":
      language: typescript
      code: |
        import { Chat } from "@lmstudio/sdk";

        // Create a chat object from an array of messages.
        const chat = Chat.from([
          { role: "system", content: "You are a resident AI philosopher." },
          { role: "user", content: "What is the meaning of life?" },
        ]);
    "Constructing a Chat object":
      language: typescript
      code: |
        import { Chat } from "@lmstudio/sdk";

        // Create an empty chat object.
        const chat = Chat.empty();

        // Build the chat context by appending messages.
        chat.append("system", "You are a resident AI philosopher.");
        chat.append("user", "What is the meaning of life?");
```

See [Working with Chats](./working-with-chats) for more information on managing chat context.

<!-- , and [`Chat`](./../api-reference/chat) for API reference for the `Chat` class. -->

## Generate a response

You can ask the LLM to predict the next response in the chat context using the `respond()` method.

```lms_code_snippet
  variants:
    Streaming:
      language: typescript
      code: |
        // The `chat` object is created in the previous step.
        const prediction = model.respond(chat);

        for await (const { content } of prediction) {
          process.stdout.write(content);
        }

        console.info(); // Write a new line to prevent text from being overwritten by your shell.

    "Non-streaming":
      language: typescript
      code: |
        // The `chat` object is created in the previous step.
        const result = await model.respond(chat);

        console.info(result.content);
```

## Customize Inferencing Parameters

You can pass in inferencing parameters as the second parameter to `.respond()`.

```lms_code_snippet
  variants:
    Streaming:
      language: typescript
      code: |
        const prediction = model.respond(chat, {
          temperature: 0.6,
          maxTokens: 50,
        });

    "Non-streaming":
      language: typescript
      code: |
        const result = await model.respond(chat, {
          temperature: 0.6,
          maxTokens: 50,
        });
```

See [Configuring the Model](./parameters) for more information on what can be configured.

## Print prediction stats

You can also print prediction metadata, such as the model used for generation, number of generated
tokens, time to first token, and stop reason.

```lms_code_snippet
  variants:
    Streaming:
      language: typescript
      code: |
        // If you have already iterated through the prediction fragments,
        // doing this will not result in extra waiting.
        const result = await prediction.result();

        console.info("Model used:", result.modelInfo.displayName);
        console.info("Predicted tokens:", result.stats.predictedTokensCount);
        console.info("Time to first token (seconds):", result.stats.timeToFirstTokenSec);
        console.info("Stop reason:", result.stats.stopReason);
    "Non-streaming":
      language: typescript
      code: |
        // `result` is the response from the model.
        console.info("Model used:", result.modelInfo.displayName);
        console.info("Predicted tokens:", result.stats.predictedTokensCount);
        console.info("Time to first token (seconds):", result.stats.timeToFirstTokenSec);
        console.info("Stop reason:", result.stats.stopReason);
```

## Example: Multi-turn Chat

The example below turns the single request flow into a conversational REPL. It keeps the full
history in a `Chat` instance, streams each assistant reply as it is generated, and handles common
runtime concerns such as clean shutdown and error reporting.

```lms_code_snippet
  variants:
    TypeScript:
      language: typescript
      code: |
        import { Chat, LMStudioClient } from "@lmstudio/sdk";
        import { createInterface } from "readline/promises";

        async function main() {
          const rl = createInterface({ input: process.stdin, output: process.stdout });
          const client = new LMStudioClient();
          const model = await client.llm.model();
          const chat = Chat.empty();

          try {
            while (true) {
              const input = await rl.question("You: ");
              if (!input.trim()) {
                console.info("(send a message or press Ctrl+C to exit)");
                continue;
              }

              chat.append("user", input);

              try {
                const prediction = model.respond(chat, {
                  onMessage: (message) => chat.append(message),
                });

                process.stdout.write("Bot: ");
                for await (const { content } of prediction) {
                  process.stdout.write(content);
                }
                process.stdout.write("\n");
              } catch (error) {
                console.error("Prediction failed:", error);
              }
            }
          } catch (error) {
            console.error("Chat loop ended:", error);
          } finally {
            rl.close();
          }
        }

        main().catch((error) => {
          console.error("Fatal error:", error);
          process.exitCode = 1;
        });
```

## Track Generation Progress in TypeScript

Longer prompts may require time to process before tokens appear. The TypeScript SDK exposes several
hooks that surface progress signals so you can build responsive UX while waiting for the model.

```lms_code_snippet
  variants:
    TypeScript:
      language: typescript
      code: |
        const prediction = model.respond(chat, {
          onPromptProcessingProgress: (progress) => {
            const percentage = Math.round(progress * 100);
            process.stdout.write(`\rProcessing prompt… ${percentage}%`);
          },
          onFirstToken: ({ elapsedMs }) => {
            console.info(`\nFirst token streamed after ${elapsedMs} ms`);
          },
          onMessageFragment: ({ content }) => {
            process.stdout.write(content);
          },
          onMessage: (message) => {
            console.info("\nFull assistant message received");
            chat.append(message);
          },
        });

        await prediction.result();
```

* `onPromptProcessingProgress(progress)` receives a `number` between `0` and `1` while the prompt
  is being embedded. Use it to update progress bars or log statements.
* `onFirstToken(info)` fires exactly once when the first token is emitted. This is useful for
  measuring "time to first token" metrics.
* `onMessageFragment(fragment)` streams incremental chunks of assistant output; it complements the
  `for await` loop shown earlier when you want callback-style handling.
* `onMessage(message)` runs after the response is complete. Appending the message to your `Chat`
  instance keeps the conversation context synchronized across iterations.
