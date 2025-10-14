# Plan: Tool-Enabled Chat Integrations

## Background

The current `chat.py` module wraps the LM Studio Python SDK's high-level chat helpers (`respond`, `respond_stream`, and `ChatSession`) but does not expose any utilities for automatic tool use. LM Studio's documentation describes the `.act()` API for multi-round tool execution and the `ToolFunctionDef` structure for defining tools. Our repository already defines several tools (`shell`, `planner`, `patch`, `file`, `process`, `git`) under `internal/tools`, and `tooling.py` gathers them into `TOOLS` and `TOOL_MAP`. However, there is no helper to surface these tools through the chat helpers, nor a convenience wrapper that mirrors the documented `.act()` workflow.

## Goals

1. Extend the chat utilities with functions that forward to `model.act(...)`, mirroring the LM Studio docs for automatic tool use.
2. Allow callers to select tools by name or pass custom tool definitions, defaulting to the project's registered tools.
3. Update `ChatSession` so a conversation can opt into tool usage while maintaining the existing respond/respond_stream behaviour.
4. Ensure `tooling.py` exposes ergonomic helpers to fetch tool definitions by name and report all registered tools.
5. Provide automated tests covering tool resolution and the chat/tool integration using lightweight stubs for the LM Studio SDK.

## References

- LM Studio Docs — Python Agent `.act()` API: `docs/LMStudio/developer/python/agent/act.md`
- LM Studio Docs — Defining tools: `docs/LMStudio/developer/python/agent/tools.md`
- Existing chat helpers: `chat.py`
- Tool registry: `tooling.py` and `internal/tools/*.py`

## Implementation Steps

1. **Tooling enhancements**
   - Add helper functions to `tooling.py` to look up tools by name (`get_tool`, `get_tools`).
   - Provide a reusable `resolve_tools` utility that accepts tool names and/or explicit definitions and returns a deduplicated, ordered list of `ToolFunctionDef` instances.
   - Update `__all__` to export the new helpers and ensure the registry still loads all tools from `internal/tools`.

2. **Chat helper updates**
   - Introduce a `_resolve_tools` helper inside `chat.py` that delegates to `tooling.resolve_tools` while handling default fallbacks.
   - Implement a new top-level `act(...)` function that:
     - Resolves the requested tools.
     - Prepares chat input (string, `lms.Chat`, or mapping) consistently with existing helpers.
     - Calls `model.act` with configuration/callback kwargs and returns `(assistant_text, raw_result)` similar to `respond`.
     - Raises a clear error when no tools are supplied or resolved.
   - Optionally expose an iterator-friendly wrapper if needed (e.g., future streaming), but keep scope to parity with docs for now.

3. **ChatSession integration**
   - Expand the `ChatSession` dataclass with an optional `tools` attribute storing the active tool list.
   - Extend `ChatSession.create` to accept `tools` or `tool_names`, resolving and storing them for reuse across turns.
   - Add methods to update tools (`set_tools`) and to execute `.act()` rounds (`act`), reusing the new top-level helper.
   - Ensure callbacks default to appending messages to the chat history and that user messages are queued before tool execution.

4. **Testing**
   - Create lightweight LM Studio stubs (`ToolFunctionDef`, `Chat`, `llm`) inside the tests to avoid depending on the real SDK.
   - Write tests validating:
     - `tooling.resolve_tools` and the new lookup helpers correctly return registered tools and handle invalid names.
     - `chat.act` forwards tool lists and merges kwargs/callbacks, returning extracted text from stubbed responses.
     - `ChatSession.act` uses stored tools, appends user messages, and captures assistant replies via the callback stub.
   - Use `importlib.reload` to ensure modules pick up the stubs during tests.

5. **Documentation & developer notes**
   - Keep this plan document in `docs/plans/tool-enabled-chat.md` for future contributors.
   - Update docstrings or inline comments where necessary to reference the new tool-usage functionality.

## Risk & Mitigations

- **SDK availability**: Since the real `lmstudio` package may not be present in all environments, rely on stubs within tests and avoid importing `lmstudio` at test collection time before stubbing.
- **Tool side effects**: The helpers will resolve tool definitions without executing them during tests; ensure tests do not call tool implementations directly.
- **Backward compatibility**: Existing respond/stream behaviour should remain unchanged; keep default behaviour intact unless tools are explicitly requested.

## Future Considerations

- Provide higher-level orchestration utilities for multi-round inspection (progress callbacks, tool result introspection).
- Surface richer metadata objects for `.act()` outcomes, including per-round summaries and tool execution logs.
- Integrate configuration presets for different tool subsets once more tools are added.
