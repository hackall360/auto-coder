# Auto-Coder

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status: Active" />
<img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=for-the-badge" alt="Made with ❤️ for Builders" />

</div>

> **Auto-Coder** is a multi-agent development companion that coordinates code generation, documentation, testing, and research tasks through LM Studio powered large language models.

---

## Table of Contents

- [Overview](#overview)
- [Feature Highlights](#feature-highlights)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Launch the Interactive Manager](#launch-the-interactive-manager)
- [Text UI](#text-ui)
- [Development Workflow](#development-workflow)
  - [Running Tests](#running-tests)
  - [Playwright Browser Support](#playwright-browser-support)
- [MCP Server Integration](#mcp-server-integration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Community & Support](#community--support)

---

## Overview

Auto-Coder orchestrates a suite of specialised agents—coding, documentation, dependency management, research, testing, and more—to automate the day-to-day workflows of shipping software. Each agent collaborates through a shared session layer backed by LM Studio's chat runtime, enabling the system to draft plans, execute tool calls, apply diffs, and validate outcomes with minimal human guidance.

The repository includes:

- A production-ready manager agent with live status streaming.
- Tooling abstractions that expose safe file editing, shell execution, and repository inspection capabilities to language models.
- Rich documentation for Auto-Coder itself, LM Studio usage patterns, and the LMF2 model family that powers key features.

## Feature Highlights

- **LM Studio Native** – Uses the `lmstudio` Python SDK for chat completions, streaming responses, and schema-constrained outputs.
- **Composable Tooling** – The `ToolRegistry` makes it trivial to register Python callables as LM Studio tools, complete with discovery helpers for module or package scans.
- **Deep Agent Catalogue** – Manager, coder, researcher, tester, security, documentation, dependency, and database migration agents ship out of the box, each emitting structured results for downstream orchestration.
- **Repository Context & RAG** – Built-in repository search, diff summarisation, and web retrieval pipelines keep models grounded in real project state.
- **Extensible Internals** – Internal libraries such as TTS/STT wrappers, DAG scheduling, and process runners are factored for reuse across custom workloads.

## Architecture at a Glance

| Layer | Responsibilities | Key Modules |
| --- | --- | --- |
| Session & Chat | Normalise prompts, stream responses, and manage message history while enforcing optional schemas. | [`chat.py`](./chat.py), [`session.py`](./session.py) |
| Orchestration | Build multi-agent workflows, surface live status updates, and coordinate retries or plan execution. | [`main.py`](./main.py), [`agents/manager.py`](./agents/manager.py) |
| Specialised Agents | Apply diffs, generate docs, run tests, manage dependencies, perform research, and more. | [`agents/`](./agents) |
| Tooling | Register reusable tools, expose safe file/process utilities, and integrate with git/patch flows. | [`tooling.py`](./tooling.py), [`internal/tools`](./internal/tools) |
| Retrieval & Utilities | Provide repository-aware RAG, structured responses, and speech utilities for future integrations. | [`internal/RAG.py`](./internal/RAG.py), [`internal/schemas.py`](./internal/schemas.py), [`internal/TTS.py`](./internal/TTS.py) |

Dive deeper with the [Auto-Coder documentation set](./docs/autocoder/README.md) for module-by-module references.

## Getting Started

### Prerequisites

- Python 3.10 or newer.
- [LM Studio](https://lmstudio.ai/) installed locally with at least one model downloaded.
- (Optional) Node.js 18+ if you plan to work with the bundled Playwright integrations.

### Installation

```bash
# Clone the repository
git clone https://github.com/hackall360/auto-coder.git
cd auto-coder

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Quick start

```bash
python main.py --config path/to/config.json --default-model anthropic/claude-3-sonnet
```

Run the command from the repository root to launch the Textual-powered interface. The entry point honours every shared flag defined in [`cli/overrides.py`](./cli/overrides.py), so you can pass model overrides, repository indexing preferences, memory settings, and MCP options directly on the command line. Existing automation that shells into `python TUI.py` continues to work and boots the same UI when you prefer the module-level target.

To capture structured corpus events during a session, enable the new corpus pipeline on the CLI:

```bash
python main.py --config path/to/config.json --enable-corpus --corpus-path ~/.autocoder/corpus/events.jsonl
```

Complementary flags let you disable capture (`--disable-corpus`), tune the similarity filter (`--corpus-dedup-threshold 0.7`), or override event categories (`--corpus-category web_search=research`).

### Logging controls

Logging defaults to structured JSON at the `INFO` level. Use the new verbosity flags to adjust output without editing environment variables:

- `--verbose` raises the root logger to `DEBUG`.
- `--quiet` drops it to `WARNING`.
- `--log-level LEVEL` accepts any standard logging level name (or numeric value) and takes precedence over the `AUTO_CODER_LOG_LEVEL` environment variable.

Handler-specific environment overrides such as `AUTO_CODER_CONSOLE_LEVEL`, `AUTO_CODER_FILE_LEVEL`, and `AUTO_CODER_LOG_FILE` continue to work alongside the CLI flags, so you can still direct logs to files or adjust per-handler verbosity when required.

## Text UI

![Auto-Coder Text UI overview](docs/assets/text-ui-demo.png)

Auto-Coder now launches its Textual terminal interface by default. The UI layers a live plan tracker, transcript, and status feeds on top of the manager runtime so you can monitor each agent while a request runs.

### Installation Requirements

- Ensure the core dependencies are installed via `pip install -r requirements.txt`. If you only need the Text UI, install `textual>=0.56.4` and `rich>=13.7` alongside the base prerequisites listed above.
- The UI renders best in terminals with **true colour** and **Unicode** support (iTerm2, Windows Terminal, Kitty, Alacritty, and most modern Linux terminals).
- Optional: configure LM Studio and any MCP servers referenced by your `config.json` when you want the UI to orchestrate live agent runs.

### Launch Command

```bash
python main.py --config path/to/config.json --repo-refresh-interval 120
```

`main.py` directly boots the Textual UI and accepts the full suite of shared flags (model overrides, repository indexing controls, memory settings, MCP startup options, and more). If you prefer calling the UI module explicitly, `python TUI.py` remains supported and recognises the exact same arguments.

### Feature Highlights

- **Transcript panel** retains the full conversation between you and Auto-Coder, including system events.
- **Plan tracker** surfaces the manager's execution plan and updates each task as agents make progress.
- **Budget meter** visualises consumption for per-task round budgets so you can see when a workflow is close to its limits.
- **Status feed** streams structured status updates (planning, progress, successes, and errors) in real time.
- **Prompt input** mirrors the CLI experience with `/cancel` and `/quit` helpers for graceful shutdowns.

### Troubleshooting & Fallbacks

- **Terminal quirks** – If colours look incorrect, export `TERM=xterm-256color` or switch to a terminal emulator with true-colour support.
- **Dependency errors** – Install Textual with `pip install textual>=0.56.4 rich>=13.7` or reinstall using the full `requirements.txt` to pick up Rich and Textual dependencies.
- **Conflicting keybindings** – Some terminals intercept `Ctrl+C`; press `Esc` followed by `/quit` to exit safely.
  

## Development Workflow

- **Repository Context** – Populate the `agents/repo_context.py` helpers with up-to-date repository information for best results.
- **Tool Registration** – Attach your own toolsets with `AgentBuilder.with_tools()` or `register_default_toolset()`.
- **Configuration** – Use the `core.research` section in `config.json` or the `AUTO_CODER_RESEARCH_*` environment variables to adjust WebRAG proxies, caching, and anonymous browsing defaults.

### Customising research and browsing

Override the defaults for the built-in research agent by extending the
`core.research` section of your `config.json`. Cache-related knobs control how
many queries and snippets remain in memory, while the nested `web` mapping is
forwarded directly to the underlying web retriever:

```json
{
  "core": {
    "research": {
      "cache_size": 16,
      "cache_top_k": 12,
      "max_quote_chars": 480,
      "web": {
        "proxy": "http://127.0.0.1:8080",
        "user_agent_pool": ["Mozilla/5.0", "Brave/1.64"],
        "incognito_contexts": true,
        "anonymous_browsing": false
      }
    }
  }
}
```

Environment variables such as `AUTO_CODER_RESEARCH_USER_AGENT_POOL` (comma
separated) or `AUTO_CODER_RESEARCH_PROXY` provide quick overrides without
editing the file. If `anonymous_browsing` is omitted, Auto-Coder assumes the
inverse of `core.models.allow_external_browsing`, matching previous releases.

### Customising manager planning

Override the manager's planning behaviour by extending the `core.manager`
section of your `config.json`. The example below increases plan retries, allows
tasks to retry twice, and injects a bespoke documentation blueprint:

```json
{
  "core": {
    "manager": {
      "plan_retries": 2,
      "task_retry_limit": 2,
      "specialist_blueprints": [
        {
          "name": "release-notes",
          "kind": "documentation",
          "agent": "documentation",
          "keywords": ["release", "changelog"],
          "budget": {"limit": 2, "unit": "rounds"},
          "research": {"required": true, "audience": "docs"}
        }
      ]
    }
  }
}
```

The override is optional—Auto-Coder keeps its default blueprint catalogue and
single-attempt planning unless this section is provided.

### Capturing corpus events

Corpus capture is disabled by default so development sessions remain ephemeral. Enable it by extending the `core.corpus` section of your configuration or by supplying the new CLI flags described earlier:

```json
{
  "core": {
    "corpus": {
      "enabled": true,
      "storage_path": "~/autocoder/corpus/events.jsonl",
      "dedup_threshold": 0.7,
      "default_categories": {
        "web_search": "research",
        "file_write": "repo_activity"
      }
    }
  }
}
```

Auto-Coder will instantiate a shared `CorpusManager`, persist events to long-term memory, and append JSONL entries to the configured `storage_path`. Environment variables (`AUTO_CODER_CORPUS_ENABLED`, `AUTO_CODER_CORPUS_PATH`, `AUTO_CODER_CORPUS_DEDUP_THRESHOLD`, `AUTO_CODER_CORPUS_DEFAULT_CATEGORIES`) provide zero-touch overrides. Adjust the deduplication threshold closer to `1.0` to suppress near-identical payloads, or omit it entirely to capture every event.

### Running Tests

```bash
pytest
```

The automated suite exercises agent builders, schema utilities, speech interfaces, and integration helpers. Feel free to augment the suite when extending functionality.

### Playwright Browser Support

Some retrieval flows leverage Playwright for deterministic rendering. After installing Python dependencies, add the browser binaries:

```bash
playwright install
```

This unlocks the Playwright-backed search pipeline in `internal/web_playwright.py` and `internal/RAG.py`.

## MCP Server Integration

Auto-Coder can treat [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) servers as first-class tools alongside local Python callables. The `mcp_tooling` module normalises configuration, launches command-based servers, and registers remote/local descriptors with the default `ToolRegistry` so that every entry serialises to `{"type": "mcp", ...}` during `model.act` calls.

- Define MCP servers inside the `mcp_servers` section of your `config.json`. Override the config path with the `MCP_CONFIG_PATH` environment variable when running custom deployments.
- Use `AgentBuilder.with_mcp_servers()` to inject MCP descriptors (raw mappings, `MCPServerConfig`, or `MCPServerSpec` instances) before calling `.build()`. The builder defers to `register_mcp_servers()` under the hood, ensuring deduplication and seamless mixing with callable tools.
- `MCPServerRegistry` and `CommandServerLifecycle` manage validation and command lifecycle (stdout readiness patterns, probe URLs, graceful shutdown signals, etc.).

### Configuration Examples

**Local HTTP server**

```json
{
  "mcp_servers": {
    "filesystem": {
      "type": "local",
      "url": "http://127.0.0.1:3030",
      "allowed_tools": ["fs.read", "fs.write"],
      "headers": {"Authorization": "Bearer local-token"}
    }
  }
}
```

**Remote HTTPS server**

```json
{
  "mcp_servers": {
    "knowledge-base": {
      "type": "remote",
      "url": "https://mcp.example.com/api",
      "verify_tls": false,
      "allowed_tools": ["search", "summarize"],
      "metadata": {"tier": "beta"}
    }
  }
}
```

**Command-launched server**

```json
{
  "mcp_servers": {
    "git-helper": {
      "type": "command",
      "command": ["python", "-m", "git_mcp"],
      "env": {"MCP_API_KEY": "${GIT_MCP_TOKEN}"},
      "cwd": "/srv/mcp/git",
      "ready_pattern": "Server ready",
      "ready_timeout": 15,
      "ready_probe_url": "http://127.0.0.1:4040/health",
      "shutdown_command": ["python", "-m", "git_mcp", "--shutdown"],
      "shutdown_signal": 15,
      "capture_output": true
    }
  }
}
```

Each descriptor supports optional `allowed_tools`, `headers`, and `metadata` fields in addition to lifecycle controls such as `ready_pattern`, `ready_timeout`, and graceful shutdown commands or signals. When combined with callable tools, the registry keeps payload ordering intact so mixed tool sets continue to serialise without regression.

## Documentation

The `docs/` directory houses three complementary knowledge bases:

- [`docs/autocoder`](./docs/autocoder/README.md) – Deep dive into this repository's layout, runtime architecture, agent catalogue, and tests.
- [`docs/LMStudio`](./docs/LMStudio/index.md) – Tutorials, SDK guides, and API references for LM Studio.
- [`docs/LMF2`](./docs/LMF2/index.md) – Model cards and usage notes for the Liquid LMF2 family.

A curated overview is available in [`docs/README.md`](./docs/README.md).

## Contributing

1. Fork the repository and create a feature branch.
2. Install dependencies and run the test suite with `pytest`.
3. Format or lint your changes as appropriate for the touched modules.
4. Submit a pull request describing the motivation, design, and testing strategy.

When proposing agent or tooling updates, include documentation changes so the knowledge base stays authoritative.

## Community & Support

- File issues or feature requests through the repository's issue tracker.
- Share reproducible steps and logs when reporting bugs to accelerate triage.
- Contributions of new agents, tool integrations, or docs are welcome—let us know what you're building!

Happy building! 🚀

## License

This project is licensed under the [Mozilla Public License 2.0](./LICENSE).
