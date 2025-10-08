# Agents

This page catalogues every agent class shipped with Auto-Coder. Each section
covers the agent's responsibilities, inputs/outputs, and any notable
collaborators.

## Manager (`agents/manager.py`)

- Orchestrates end-to-end workflows by delegating tasks to specialised agents
  (coder, documentation, dependency, research, testing, etc.).
- Tracks per-task budgets via `TaskBudget`, emits human-friendly progress
  updates through `ManagerStatusUpdate`, and aggregates results in
  `ManagerResult`.
- Builds plans (optionally via injected plan builders) and coordinates retries,
  research evidence, evaluations, and repo context usage.

## Coding (`agents/coder.py`)

- Applies code changes using LM Studio guidance, repository context artefacts,
  and the `internal.tools.file`/`internal.tools.patch` helpers.
- Produces `CoderResult` summaries with applied diffs, change summaries, and
  structured responses, allowing the manager to present diffs or chain follow-up
  actions.
- Leverages a `ToolRegistry` to expose safe editing primitives to the model.

## Documentation (`agents/doc.py`)

- Generates README, changelog, and walkthrough updates using repo context and
  optional research snippets.
- Outputs rich structures such as `DocumentationSummary`, `ReadmeDraft`,
  `ChangelogDraft`, and `WalkthroughSection` for downstream rendering.

## Dependency Build (`agents/dependency.py`)

- Uses `RunnerAgent` to execute dependency installation commands, capturing
  diffs to lockfiles and offering caching hints through
  `DependencyCacheDirective` objects.
- Summarises the overall dependency resolution in `DependencyResolution`,
  including environment notes and follow-up actions.

## Database Migration (`agents/db_migration.py`)

- Coordinates schema migration plans, leverages `RunnerAgent` for command
  execution, and uses `RepoContextAgent` for migration file discovery.
- Tracks executed migrations in `MigrationRecord`, `SchemaMigrationPlan`, and
  `MigrationResult`, with optional ephemeral database specifications.

## Evaluation (`agents/eval.py`)

- Runs prompt regressions or scenario evaluations, optionally parsing YAML specs
  for evaluation cases.
- Produces `PromptComparison`, `PromptEvalResult`, and `RegressionSummary`
  structures to inform the manager about behavioural drift.

## Integrations (`agents/integrations.py`)

- Manages CI/CD pipeline updates, release metadata, and external integration
  chores.
- Emits `PipelineUpdateResult` and `CIJobPlan` records describing planned or
  executed automation steps.

## Repository Context (`agents/repo_context.py`)

- Performs repository searches, symbol extraction, and diff summarisation to
  provide context for other agents.
- Wraps git helpers and the `internal.RAG.CodebaseRAG` retriever to surface
  semantic matches.
- Returns `RepoSearchResult`, `RepoSymbolResult`, `DiffBundle`, and `DiffFileStat`
  records for downstream consumers.

## Research (`agents/research.py`)

- Implements higher-level browsing and summarisation on top of
  `internal.RAG.WebRAG`.
- Delivers `ResearchSnippet` collections grouped into `ResearchResult`
  structures for evidence gathering.

## Runner (`agents/runner.py`)

- Encapsulates shell/process execution with timeout handling, working directory
  resolution, and environment capture.
- Produces `RunReport` entries with stdout/stderr, exit codes, and timing
  metadata for reuse by dependency, testing, and security agents.

## Security (`agents/security.py`)

- Drives security scanning toolchains, capturing findings in
  `SecurityScanFinding` entries and aggregated `SecurityScanReport` results.
- Supports caching through `SecurityCacheDirective` hints and wraps execution
  via `RunnerAgent`.

## Testing Critic (`agents/tester.py`)

- Provides quick validation by running targeted test commands or lint checks via
  `RunnerAgent`.
- Emits `TestSuiteResult`, `CriticStatusEvent`, and `CriticAnalysis` objects to
  convey confidence levels and remediation guidance.

## Research & Testing Collaborators

- Multiple agents (coder, doc, manager) accept injected `ResearchAgent` or
  `TestCriticAgent` instances, enabling richer multi-agent workflows without
  tight coupling between modules.
