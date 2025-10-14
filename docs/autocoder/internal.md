# Internal Libraries

The `internal/` package provides reusable building blocks that power multiple
agents. Key areas include task planning, retrieval-augmented generation,
schemas, speech interfaces, tool wrappers, and browser automation.

## DAG Execution (`internal/DAG.py`)

- Supplies a general-purpose directed acyclic graph executor with retry,
  timeout, and context propagation support via `DAG`, `DAGNode`, and helper
  classes.
- Enables the manager workflow to model multi-step plans with dependencies and
  collect structured results from each node.

## Retrieval-Augmented Generation (`internal/RAG.py`)

- Implements both codebase (`CodebaseRAG`) and web (`WebRAG`) retrieval
  strategies, including tokenisation, chunk ranking, optional Playwright-backed
  browsing, and web fetch fallbacks.
- Provides `DocumentChunk` structures and stopword-aware token scoring to power
  semantic search in repository and online sources.

## Speech Interfaces (`internal/STT.py`, `internal/TTS.py`)

- Wrap speech-to-text (STT) and text-to-speech (TTS) pipelines with optional
  streaming, batching, and caching primitives.
- Designed to integrate with LM Studio tool definitions so conversational agents
  can transcribe audio or generate spoken responses when needed.

## Agent Configuration Helpers (`internal/agents.py`)

- Defines typed containers for LM Studio agent settings (`AgentConfigPayload`,
  `StructuredResponseSettings`) and utilities to build response format payloads
  from schemas.

## Schema Utilities (`internal/schemas.py`)

- Offers helpers to construct JSON schema payloads, validate structured
  responses, and convert between multiple schema representations.
- Exposes `SchemaError` and `SchemaLike` abstractions shared by chat/session
  modules.

## Structured Response Wrapper (`internal/structures.py`)

- Provides `StructuredResponse`, a thin wrapper around raw model responses that
  preserves the original payload while exposing parsed/validated content.

## Tool Wrappers (`internal/tools/`)

- Bundles reusable LM Studio tools for file manipulation, git operations, patch
  application, planning assistance, process management, and shell execution.
- Each module exposes callables adhering to the `ToolFunctionDef` protocol so
  they can be registered with `ToolRegistry` without additional adapters.

## Browser Automation (`internal/web_playwright.py`)

- Supplies a Playwright-backed web client for agents requiring headless browser
  automation (used by RAG web retrieval when available).

These utilities are intentionally separated from the agent implementations to
promote reuse and make it easier to test shared behaviour in isolation.
