# Testing Strategy

Automated tests live under `tests/` and are executed with `pytest`. The suite is
organised by agent or subsystem so failures quickly pinpoint the impacted area.

## Agent Coverage

- `test_manager_agent.py`, `test_manager_research.py`, and `test_agents.py`
  validate manager orchestration, round tracking, and agent builder utilities.
- `test_coder_agent.py`, `test_doc_agent.py`, `test_dependency_agent.py`,
  `test_db_migration_agent.py`, `test_security_agent.py`,
  `test_integrations_agent.py`, and `test_research_agent.py` exercise the domain
  agents and their interactions with shared tools.
- `test_eval_agent.py` covers regression/evaluation workflows and structured
  summary generation.

## Tooling & Infrastructure

- `test_chat_tools.py`, `test_structures.py`, and `test_schemas.py` ensure the
  chat/session abstractions, structured response wrappers, and schema utilities
  behave as expected.
- `test_rag_web.py` focuses on the RAG web retriever, including fallback
  behaviour when optional dependencies are unavailable.
- `test_web_playwright.py` validates the Playwright integration and its
  resilience to environment quirks.

## Speech Interfaces

- `test_stt.py` and `test_tts.py` cover speech-to-text and text-to-speech helper
  modules, verifying streaming and configuration pathways.

## Running the Suite

1. Install dependencies from `requirements.txt` (consider using a virtual
   environment).
2. Execute `pytest` from the repository root. Individual files can be targeted
   via `pytest tests/test_coder_agent.py` for quicker iteration.
