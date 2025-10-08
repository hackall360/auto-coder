# Auto-Coder Documentation

Welcome to the Auto-Coder documentation set. This hub explains how the repository is organised, how the runtime fits together, and where each specialised agent lives. Use it as your starting point when onboarding, planning refactors, or updating supporting materials.

## 🧭 Quick Index

| Topic | Description |
| --- | --- |
| [Directory Overview](./directory-overview.md) | High-level tour of the repository layout and responsibilities of each top-level folder. |
| [Core Runtime Components](./runtime.md) | Walkthrough of the CLI entry point, chat/session abstractions, tool registry, and other runtime helpers. |
| [Agents](./agents.md) | Responsibilities, inputs/outputs, and collaborators for every agent implementation in `agents/`. |
| [Internal Libraries](./internal.md) | Shared building blocks in the `internal/` namespace plus bundled tool implementations. |
| [Testing Strategy](./testing.md) | Coverage summary for the automated test suite and the behaviours it exercises. |

## 🔄 Keeping Docs Fresh

- Update the relevant page whenever you introduce, rename, or remove a module. That keeps the mental model aligned with the code.
- When you add a new agent, capture its remit in [`agents.md`](./agents.md) and reference any novel data structures.
- Use relative links so these files render correctly on GitHub and within static doc tooling.

## 📎 Related Resources

- The repository-wide onboarding guide lives in the [root README](../../README.md).
- Broader documentation for LM Studio and the LMF2 model family is indexed in the [docs hub](../README.md).

Your future self (and teammates) will thank you for keeping this knowledge base accurate. 🙌
