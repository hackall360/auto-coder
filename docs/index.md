---
title: LMStudioAutoCoder Documentation
description: Reference materials for LM Studio usage and the LMF2 model family.
---

Welcome! This repository bundles two complementary sets of documentation:

## LM Studio
Guides and references for installing, configuring, and extending the LM Studio application. Start with the [LM Studio documentation hub](LMStudio/index) to browse app guides, developer SDKs, and configuration references.

## LMF2 Model Family
Model cards, weights, and usage guidance for the Liquid LMF2 series that powers parts of this project. Review the [LMF2 index](LMF2/index) to find model-specific details such as context length, licensing, and download locations.

Use the section that matches your task, and feel free to cross-reference between them when you need both application behaviour and underlying model characteristics.

## Quick-start: Structured Agents with Tools

```python
from agents import AgentBuilder, register_default_toolset


def get_weather(city: str) -> str:
    """Return a canned weather string for demo purposes."""

    return f"The weather in {city} is sunny."


# Register a reusable toolset that can be referenced by name.
register_default_toolset(
    "demo-tools",
    [
        get_weather,
    ],
)

# Build an agent with a system prompt and structured output requirements.
agent = (
    AgentBuilder(system_prompt="You are a helpful assistant.")
    .with_toolsets("demo-tools")
    .build()
)

response_text, structured = agent.act(
    "What's the weather in Paris?",
    schema={
        "type": "object",
        "properties": {"status": {"type": "string"}, "city": {"type": "string"}},
        "required": ["status", "city"],
    },
)

print(response_text)
print(structured.parsed)
```

### Configuration reference

| Setting | Location | Description |
| --- | --- | --- |
| `register_default_toolset(name, tools)` | `agents.py` | Attach reusable tool collections to a name for quick reuse across sessions. |
| `AgentBuilder.with_toolsets(*names)` | `agents.py` | Include one or more registered toolsets when constructing a session. |
| `AgentBuilder.with_tools(*tools)` | `agents.py` | Add ad-hoc tools in addition to registered sets. |
| `AgentSession.act(..., schema=..., response_format=...)` | `session.py` | Enforce structured outputs via JSON Schema or LM Studio response formats. |
| `AgentSession.act(..., handle_invalid_tool_request=...)` | `session.py` | Provide graceful fallbacks when the model requests unavailable tools. |

## Web search retrieval with Playwright

The `internal.web_playwright.PlaywrightWebClient` module encapsulates Playwright usage for the `WebRAG` helper. Install the dependency and the matching browser binaries before running code that performs web searches:

```bash
pip install -r requirements.txt
playwright install
```

`WebRAG` will default to the Playwright-powered search and rendering pipeline when available, and automatically fall back to the legacy `requests` workflow if the dependency is missing.
