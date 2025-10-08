# Auto-Coder

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+" height="36" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status: Active" height="36" />
<img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red" alt="Made with ❤️ for Builders" height="36" />

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
- [Development Workflow](#development-workflow)
  - [Running Tests](#running-tests)
  - [Playwright Browser Support](#playwright-browser-support)
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

### Launch the Interactive Manager

```bash
python main.py
```

You will enter an interactive loop; type your request and watch the manager agent coordinate the supporting agents. Use `exit` or `quit` to leave the session.

## Development Workflow

- **Repository Context** – Populate the `agents/repo_context.py` helpers with up-to-date repository information for best results.
- **Tool Registration** – Attach your own toolsets with `AgentBuilder.with_tools()` or `register_default_toolset()`.
- **Configuration** – Set environment variables referenced by `internal.RAG.WebRAG` or other modules to customise behaviour (proxies, anonymous browsing, etc.).

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
