---
title: Overview
description: Learn the fundamentals of building LM Studio Python agents, from core concepts to first steps and deeper resources.
index: 1
---

Agents let language models take actions on your behalf by orchestrating multi-round conversations with tools that run on your
local machine. This page introduces the mental model for Python agents in LM Studio and points you to the most relevant guides
and API references for building your first agent-powered workflow.

## Key concepts

- **Agent round** – Each call to [`llm.act()`](../api-reference/act) can span multiple rounds. A round covers the model's
  response, any tool requests it emits, and the follow-up prompt once tool outputs are returned.
- **Tools** – Regular Python functions or [`ToolFunctionDef`](../api-reference/llm-namespace) instances that describe the
  actions your agent can take. See [Tool Definition](./tools) for patterns and best practices.
- **Chat history** – A [`Chat`](../api-reference/chat) object (or similar structure) that accumulates user, assistant, and tool
  messages so the model understands the ongoing task.
- **Callbacks** – Optional hooks such as `on_message`, `on_prediction_fragment`, and `handle_invalid_tool_request` that let you
  stream responses, monitor progress, or intercept failures during an agent session. Dive into [The `.act()` call](./act) for
  a comprehensive list.

## Basic agent workflow

1. **Select a capable model** for tool use (for example, `qwen2.5-7b-instruct`).
2. **Define tools** that expose the operations you want the model to perform.
3. **Prepare a chat history** describing the task and any context the agent needs.
4. **Call `llm.act()`** with the chat history, tool definitions, and optional callbacks to manage the interaction loop.
5. **Handle the result** once the agent returns a final assistant message or terminates after reporting an error.

### Quick start example

```lms_code_snippet
  variants:
    "Python (convenience API)":
      language: python
      code: |
        import lmstudio as lms

        def search_docs(query: str) -> str:
            """Return a short answer pulled from your local document index."""
            # Replace with your own retrieval logic
            return f"Pretend we looked up: {query!r}"

        model = lms.llm("qwen2.5-7b-instruct")
        chat = lms.Chat("You are an engineering assistant that answers using the provided tools.")

        chat.add_user_message("Find the latency benchmarks for the latest release.")

        model.act(
          chat,
          [search_docs],
          on_message=chat.append,
          on_prediction_fragment=lambda fragment, round_index=0: print(fragment.content, end=""),
        )
```

The SDK automatically serializes the tool schema, routes tool requests to your Python functions, and feeds their outputs back to
the model until it produces a final response.

## Choosing the right API surface

- Use `lms.llm()` for a convenience client in notebooks or scripts. Prefer [`Client().llm`](../api-reference/lmstudioclient)
  when you need deterministic resource cleanup or want to orchestrate multiple agents simultaneously.
- Pick the synchronous API for simpler applications, or [`AsyncClient().llm`](../api-reference/lmstudioclient) when building
  structured-concurrency workloads that benefit from async/await.

## Continue learning

- Deep dive into [The `.act()` call](./act) to understand execution rounds, callbacks, and error handling.
- Explore [Tool Definition](./tools) to craft robust tool schemas and wrap existing functions.
- Reference the [API docs for agent primitives](../api-reference/act) and the broader [`lmstudio` namespace](../api-reference/llm-namespace)
  when you need detailed parameter descriptions.
- Follow the [getting started tutorial](../getting-started/index) for a step-by-step walkthrough of the Python SDK, including
  setting up your environment and running first predictions.
