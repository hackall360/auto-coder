# Core Runtime Components

## Command-Line Interface (`main.py`)

- Bootstraps an `AgentBuilder` and constructs a `ManagerAgent` with a status
  callback that prints real-time updates. [`main.py`](../../main.py)
- Provides an interactive loop that accepts user prompts until `exit`/`quit`
  are entered, forwarding each message to the manager and rendering any
  additional status updates before printing the final response. [`main.py`](../../main.py)

## Chat Abstractions (`chat.py`)

- Wraps LM Studio's SDK (`lmstudio.llm`) via `get_model`, returning a handle to
  the configured model. [`chat.py`](../../chat.py)
- Normalises prompt inputs through `_prepare_input` so callers can supply plain
  strings, `lms.Chat` histories, or mapping payloads. [`chat.py`](../../chat.py)
- Provides `_call_model`, `_extract_text`, and `_coerce_response_mapping` to
  standardise the various return types produced by the SDK. [`chat.py`](../../chat.py)
- Implements `ChatSession` to manage conversation history, append tool
  responses, and issue `respond`/`respond_stream` calls with optional
  callbacks/tool definitions. [`chat.py`](../../chat.py)
- Supports structured outputs through `_resolve_response_format` and
  `StructuredResponse`, enabling schema-guided replies. [`chat.py`](../../chat.py)

## Session Management (`session.py`)

- `AgentSession` builds on `ChatSession`, capturing each round in an
  `AgentRound` object with transcript, messages, tool history, and metadata.
  [`session.py`](../../session.py)
- Exposes hooks (`on_message`, `on_tool_call`, `on_tool_result`, `on_round_start`,
  `on_round_end`) that external orchestrators can use to observe the workflow.
  [`session.py`](../../session.py)
- Provides convenience helpers to mutate tools, append user input, and append
  tool responses, while keeping callbacks in sync. [`session.py`](../../session.py)

## Tool Registry (`tooling.py`)

- Defines `ToolSpec`, a dataclass capturing tool metadata, callable
  implementation, parameters, and payload overrides for LM Studio's tool
  interface. The spec now differentiates between standard function tools and
  MCP integrations via the `tool_type` attribute, allowing non-callable
  metadata-only tools to be registered safely. [`tooling.py`](../../tooling.py)
- Offers `ToolRegistry` for registering callables, resolving tool references by
  name, preventing duplicates, and ensuring each tool exposes type annotations
  and documentation. Registry helpers deduplicate by tool name and support
  pre-built `ToolSpec` instances alongside callables. [`tooling.py`](../../tooling.py)
- Provides `register_mcp_tool` for registering MCP servers from JSON payloads,
  merging header/metadata overrides while co-existing with standard tools.
  [`tooling.py`](../../tooling.py)
- Includes discovery helpers (`discover_module_tools`, `discover_package_tools`)
  to automatically register tools from modules or packages. [`tooling.py`](../../tooling.py)

## Placeholder Modules

- `core.py`, `memory.py`, and `TUI.py` currently contain module headers only,
  indicating planned areas for core orchestration, long-term memory, and a
  terminal UI. [`core.py`](../../core.py), [`memory.py`](../../memory.py), [`TUI.py`](../../TUI.py)
