# Core Runtime (`core.py`)

The core runtime is implemented in [`core.py`](../../core.py) and provides a
single orchestration class, `AutoCoderCore`, that wires together every
subsystem needed to run the Auto-Coder agents. It replaces the earlier
placeholder module and is now the preferred entry point for CLIs, TUIs, and
embedding Auto-Coder inside other applications.

## Configuration model

`AutoCoderCore` accepts an explicit [`AutoCoderConfig`](../../core.py) instance
or can load configuration automatically via `load_core_configuration()`. The
loader merges three sources in the following order:

1. `config.json` (or the file referenced by `AUTO_CODER_CONFIG_PATH`) under the
   `{"core": { ... }}` section.
2. Environment variables.
3. Runtime overrides supplied to the constructor.

The configuration is broken down into a handful of nested sections:

| Section | Purpose | Representative keys | Environment overrides |
| --- | --- | --- | --- |
| `paths` | Locate the repository and scratch space. | `repo_root`, `workspace_root`, `artifact_root` | `AUTO_CODER_REPO_ROOT`, `AUTO_CODER_WORKSPACE_ROOT`, `AUTO_CODER_ARTIFACT_ROOT` |
| `models` | Select default/reasoning/research models and browsing defaults. | `default_model`, `reasoning_model`, `research_model`, `allow_external_browsing` | `AUTO_CODER_MODEL`, `AUTO_CODER_REASONING_MODEL`, `AUTO_CODER_RESEARCH_MODEL`, `AUTO_CODER_ALLOW_BROWSING` |
| `research` | Tune caching and WebRAG networking defaults. | `cache_size`, `cache_top_k`, `max_quote_chars`, `web.proxy`, `web.user_agent_pool`, `web.incognito_contexts`, `web.anonymous_browsing` | `AUTO_CODER_RESEARCH_CACHE_SIZE`, `AUTO_CODER_RESEARCH_CACHE_TOP_K`, `AUTO_CODER_RESEARCH_MAX_QUOTE_CHARS`, `AUTO_CODER_RESEARCH_PROXY`, `AUTO_CODER_RESEARCH_USER_AGENT_POOL`, `AUTO_CODER_RESEARCH_INCOGNITO_CONTEXTS`, `AUTO_CODER_RESEARCH_ANONYMOUS_BROWSING` |
| `repo_context` | Control semantic indexing. | `include_exts`, `exclude_dirs`, `auto_refresh`, `refresh_interval` | `AUTO_CODER_REPO_INCLUDE_EXTS`, `AUTO_CODER_REPO_EXCLUDE_DIRS`, `AUTO_CODER_REPO_AUTO_REFRESH`, `AUTO_CODER_REPO_REFRESH_INTERVAL` |
| `agents` | Enable/disable specialist helpers. | Flags for `repo_context`, `research`, `documentation`, `dependency`, `runner`, `db_migration`, `security`, `integrations`, `eval`, `test_critic` | `AUTO_CODER_ENABLE_<NAME>`, `AUTO_CODER_DISABLE_<NAME>` |
| `manager` | Configure planning retries and specialist overrides. | `plan_retries`, `task_retry_limit`, `specialist_blueprints` | – |
| `memory` | Configure vector-memory backends. | `config_path`, `default_scope`, `combined_scope`, `share_globally` | `AUTO_CODER_MEMORY_CONFIG`, `AUTO_CODER_MEMORY_DEFAULT_SCOPE`, `AUTO_CODER_MEMORY_COMBINED_SCOPE`, `AUTO_CODER_MEMORY_SHARE` |
| `corpus` | Control structured corpus capture. | `enabled`, `storage_path`, `dedup_threshold`, `default_categories` | `AUTO_CODER_CORPUS_ENABLED`, `AUTO_CODER_CORPUS_PATH`, `AUTO_CODER_CORPUS_DEDUP_THRESHOLD`, `AUTO_CODER_CORPUS_DEFAULT_CATEGORIES` |
| `mcp` | Discover Model Context Protocol servers. | `config_path`, `servers`, `auto_start` | `AUTO_CODER_MCP_CONFIG`, `AUTO_CODER_MCP_AUTO_START` |

Additional helper variables include `AUTO_CODER_CONFIG_PATH` (alternate core
config file) and `AUTO_CODER_WORKSPACE_ROOT`/`AUTO_CODER_ARTIFACT_ROOT` when the
workspace layout should deviate from the repository root.

### Research agent tuning

Research-centric workflows can be customised through the `core.research`
section. The numeric knobs control the cache eviction policy and the length of
quotes that `ResearchAgent` keeps when summarising search results. The nested
`web` mapping is passed directly to [`WebRAG`](../../internal/RAG.py)
and accepts networking controls such as proxies, user-agent pools, and
incognito defaults. Values originating from environment variables may be
expressed as comma-separated lists (`AUTO_CODER_RESEARCH_USER_AGENT_POOL`) or
common boolean aliases (`true`, `false`, `yes`, `no`).

```json
{
  "core": {
    "research": {
      "cache_size": 32,
      "cache_top_k": 12,
      "max_quote_chars": 480,
      "web": {
        "proxy": "socks5h://127.0.0.1:9050",
        "user_agent_pool": ["Mozilla/5.0", "curl/8.3.0"],
        "incognito_contexts": true,
        "anonymous_browsing": false
      }
    }
  }
}
```

If `anonymous_browsing` is omitted, the runtime falls back to the inverse of the
`allow_external_browsing` flag in `core.models`, preserving the previous
behaviour.

### Manager overrides

The optional `core.manager` section controls the planning behaviour exposed by
`AutoCoderCore.build_manager()`. Set `plan_retries` to retry plan generation in
the DAG, `task_retry_limit` to cap how often individual tasks are retried, and
`specialist_blueprints` to supply custom task templates that augment or replace
the built-in specialist detection heuristics. Each blueprint is a mapping with
`name`, `kind`, `agent`, optional `keywords`, and optional `budget`/`research`
metadata:

```json
{
  "core": {
    "manager": {
      "plan_retries": 3,
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

When omitted, Auto-Coder falls back to the built-in blueprint catalogue and a
single planning attempt per workflow.

### Corpus capture

The optional `core.corpus` section toggles structured corpus logging. Capture is disabled by default, keeping local runs ephemeral. Setting `enabled` to `true` instantiates a shared [`CorpusManager`](../../corpus/manager.py) that persists events to long-term memory and, when `storage_path` is provided, appends JSONL records to that path. Use `dedup_threshold` to suppress near-duplicate payloads (values close to `1.0` keep only unique events) and `default_categories` to override the built-in event-type category map. All values can be overridden via the CLI flags (`--enable-corpus`, `--corpus-path`, `--corpus-dedup-threshold`, `--corpus-category`) or the corresponding environment variables listed above.

## Orchestration pipeline

When instantiated, `AutoCoderCore` performs the following orchestration steps:

1. **Tool registry** – Creates a shared [`ToolRegistry`](../../tooling.py) and
   exposes it to every session built through the core.
2. **Memory routing** – Loads the memory configuration using
   [`memory.load_memory_configuration`](../../memory.py), builds a
   [`MemoryRouter`](../../memory.py), and mounts a `MemoryFacade`. When
   `share_globally` is true, the facade is registered as the global shared
   memory instance so individual agents can access it without additional
   plumbing.
3. **MCP registry** – Hydrates [`MCPServerRegistry`](../../mcp_tooling.py) from
   inline descriptors or from a JSON file, optionally starting command-based
   servers when `auto_start` is enabled. The resulting specs are registered in
   the tool registry and applied to any sessions the core creates.
4. **Agent wiring** – Lazily constructs specialist agents (`RepoContextAgent`,
   `ResearchAgent`, `DocAgent`, `RunnerAgent`, `DependencyBuildAgent`,
   `DBMigrationAgent`, `SecurityAgent`, `IntegrationsAgent`, `EvalAgent`, and
   `TestCriticAgent`) based on the toggles in the configuration. Each agent
   shares the session, memory, and tool infrastructure initialised by the core.
5. **Manager construction** – `build_manager()` assembles a fully-wired
   [`ManagerAgent`](../../agents/manager.py) with callbacks, context attachments,
   and MCP tools already installed. The returned manager is ready to receive
   prompts immediately.
6. **Lifecycle management** – The core exposes `shutdown()` and implements the
   context-manager protocol to tear down background repo refreshers, MCP
   processes, and shared memory state cleanly.

## Front-end integrations

Existing front ends, such as [`main.py`](../../main.py) (Textual UI) and custom
automation harnesses, use the core runtime to obtain a manager agent while
handling user interaction themselves. The Textual entry point instantiates
`AutoCoderCore`, calls `build_manager(status_callback=...)`, and feeds events
into the UI's message loop. Alternative surfaces—web services, batch scripts,
notebooks—can follow the same pattern to reuse the orchestration logic without
duplicating the setup.

## Usage examples

Create and tear down the runtime manually:

```python
from core import AutoCoderCore

core = AutoCoderCore()
manager = core.build_manager()
try:
    response = manager.run("Add a unit test for the new parser")
    print(response)
finally:
    core.shutdown()
```

Or rely on the context-manager shorthand:

```python
from core import AutoCoderCore

with AutoCoderCore() as core:
    manager = core.build_manager()
    print(manager.run("Summarise the latest commit"))
```

To override settings from a script, supply overrides or a custom environment:

```python
from core import AutoCoderCore

overrides = {
    "core": {
        "models": {"default_model": "anthropic/claude-3-sonnet"},
        "mcp": {"auto_start": True},
    }
}

with AutoCoderCore(overrides=overrides) as core:
    manager = core.build_manager()
    print(manager.run("Scan the repo for TODO comments"))
```

These snippets apply equally to shell entry points, notebook cells, or
long-running daemons that wish to embed Auto-Coder.
