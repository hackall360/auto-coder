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

## MCP Server Integration (`mcp_tooling.py`)

- `MCPServerRegistry` validates `mcp_servers` entries loaded from `config.json`
  (or the path pointed to by the `MCP_CONFIG_PATH` environment variable),
  coercing headers, metadata, and allowed tool lists before constructing
  `MCPServerSpec` objects. [`mcp_tooling.py`](../../mcp_tooling.py)
- `CommandServerLifecycle` starts command-based MCP servers, waits for readiness
  via stdout patterns or probe URLs, and issues shutdown commands, signals, or
  process kills on teardown. [`mcp_tooling.py`](../../mcp_tooling.py)
- `register_mcp_servers` mirrors `ToolRegistry.register_mcp_tool`, ensuring each
  MCP descriptor serialises to a `{"type": "mcp", ...}` payload that can live
  alongside callable tools inside `model.act`. [`mcp_tooling.py`](../../mcp_tooling.py)
- `AgentBuilder.with_mcp_servers()` wires descriptors into the same code path as
  `with_tools`, allowing sessions to mix local callables with remote/local MCP
  endpoints without manual registry plumbing. [`agents/__init__.py`](../../agents/__init__.py)
- Example configuration snippets:

  - Local server:

    ```json
    {
      "mcp_servers": {
        "filesystem": {
          "type": "local",
          "url": "http://127.0.0.1:3030",
          "allowed_tools": ["fs.read"]
        }
      }
    }
    ```

  - Remote server:

    ```json
    {
      "mcp_servers": {
        "knowledge-base": {
          "type": "remote",
          "url": "https://mcp.example.com/api",
          "verify_tls": false,
          "allowed_tools": ["search", "summarize"]
        }
      }
    }
    ```

  - Command-launched server:

    ```json
    {
      "mcp_servers": {
        "git-helper": {
          "type": "command",
          "command": ["python", "-m", "git_mcp"],
          "ready_pattern": "Server ready",
          "ready_timeout": 15,
          "ready_probe_url": "http://127.0.0.1:4040/health",
          "shutdown_command": ["python", "-m", "git_mcp", "--shutdown"],
          "shutdown_signal": 15,
          "env": {"MCP_API_KEY": "${GIT_MCP_TOKEN}"},
          "capture_output": true
        }
      }
    }
    ```

  Optional keys such as `headers`, `metadata`, `cwd`, and `allowed_tools` work
  uniformly across server types, ensuring consistent payloads during tool
  resolution and `model.act` execution.

## Placeholder Modules

- `core.py`, `memory.py`, and `TUI.py` currently contain module headers only,
  indicating planned areas for core orchestration, long-term memory, and a
  terminal UI. [`core.py`](../../core.py), [`memory.py`](../../memory.py), [`TUI.py`](../../TUI.py)
