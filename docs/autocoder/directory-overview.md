# Directory Overview

The repository is intentionally organized around agent capabilities and shared
runtime infrastructure. The table below summarises the top-level directories
and notable standalone modules.

| Path | Description |
| --- | --- |
| `agents/` | Collection of domain-specific agent implementations that orchestrate coding, documentation, testing, security review, research, and more. Each module exports the agent class plus result/summary data structures. |
| `chat.py` | High-level helpers around the LM Studio Python SDK to send prompts, stream responses, and normalise outputs into structured responses. |
| `session.py` | Defines `AgentSession` and `AgentRound`, wrapping the chat helpers with tool management, callback hooks, and transcript tracking. |
| `tooling.py` | Implements the tool registry, normalisation logic, and helpers for discovering/validating callables that can be exposed to models. |
| `mcp_tooling.py` | Normalises Model Context Protocol server configs (`mcp_servers`) and honours the `MCP_CONFIG_PATH` override for command, local, and remote MCP integrations. |
| `main.py` | Command-line entry point that builds the manager agent, renders status updates, and runs an interactive REPL loop. |
| `TUI.py`, `core.py`, `memory.py` | Reserved placeholders for future terminal UI, application core, and memory subsystems respectively. |
| `internal/` | Shared libraries used across agents (retrieval, speech, schemas, tool wrappers, web automation, etc.). |
| `tests/` | Pytest-based regression suite covering agents, toolchains, schema helpers, and speech/retrieval integrations. |
| `docs/` | Existing documentation space housing LM Studio quick-start notes, plans, and this Auto-Coder reference. |
| `requirements.txt` | Python package dependencies required to run the agents and tests. |
| `TODO.md` | High-level backlog or future improvements list. |

Refer to the other documents in this directory for deeper dives into each area.
