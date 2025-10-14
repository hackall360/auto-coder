# Auto-Coder Documentation

Welcome to the Auto-Coder documentation set. This hub explains how the repository is organised, how the runtime fits together, and where each specialised agent lives. Use it as your starting point when onboarding, planning refactors, or updating supporting materials.

## 🧭 Quick Index

| Topic | Description |
| --- | --- |
| [Codebase Reference](./codebase-reference.md) | Auto-generated inventory of modules, public classes, and top-level functions across the project. |
| [Directory Overview](./directory-overview.md) | High-level tour of the repository layout and responsibilities of each top-level folder. |
| [Core Runtime Components](./runtime.md) | Walkthrough of the CLI entry point, chat/session abstractions, tool registry, and other runtime helpers. |
| [Agents](./agents.md) | Responsibilities, inputs/outputs, and collaborators for every agent implementation in `agents/`. |
| [Internal Libraries](./internal.md) | Shared building blocks in the `internal/` namespace plus bundled tool implementations. |
| [Memory Backends](./memory.md) | Redis/PostgreSQL setup instructions, environment variables, and troubleshooting tips. |
| [Testing Strategy](./testing.md) | Coverage summary for the automated test suite and the behaviours it exercises. |
| [Text UI Overview](#%F0%9F%96%A5%EF%B8%8F-text-ui-overview) | Launching and operating the Textual-powered terminal interface. |

## 🔄 Keeping Docs Fresh

- Update the relevant page whenever you introduce, rename, or remove a module. That keeps the mental model aligned with the code.
- When you add a new agent, capture its remit in [`agents.md`](./agents.md) and reference any novel data structures.
- Use relative links so these files render correctly on GitHub and within static doc tooling.

## 📎 Related Resources

- The repository-wide onboarding guide lives in the [root README](../../README.md).
- Broader documentation for LM Studio and the LMF2 model family is indexed in the [docs hub](../README.md).

Your future self (and teammates) will thank you for keeping this knowledge base accurate. 🙌

## 🖥️ Text UI Overview

![Auto-Coder Text UI](../assets/text-ui-demo.png)

The Text UI is now the default front end for Auto-Coder. It wraps the manager runtime with a Rich/Textual layout so you can observe execution plans, status feeds, and resource budgets without leaving the terminal.

### Installation Checklist

- Install the project dependencies with `pip install -r requirements.txt`. For lean environments, ensure `textual>=0.56.4` and `rich>=13.7` are available alongside Python 3.10+.
- Use a terminal emulator with 24-bit colour and Unicode rendering for the best experience. Setting `TERM=xterm-256color` resolves most palette issues.
- (Optional) Populate `config.json` with LM Studio settings, memory overrides, and MCP server definitions so the UI can bootstrap the full manager stack.

### Launching the Interface

```bash
python main.py --config config.json --allow-browsing --repo-refresh-interval 300
```

Invoke the command from the repository root to launch the Textual interface. Every shared flag defined in [`cli/overrides.py`](../../cli/overrides.py)—model overrides, repository index tuning, memory configuration, and MCP lifecycle controls—works here. When you need to target the module directly (for example, custom wrappers that import `TUI.py`), `python TUI.py` remains available and accepts the same arguments.

### Feature Tour

- **Transcript** – Streams user prompts, agent responses, and system notices with syntax highlighting.
- **Plan tracker** – Displays the manager plan and updates rows as tasks move from pending to in-progress, completed, or errored states.
- **Budget meter** – Reports consumed, remaining, and total round budgets for each task.
- **Status feed** – Renders structured status updates so you can follow planning, progress, and error events in real time.
- **Prompt bar** – Accepts prompts, `/cancel`, and `/quit` commands without leaving the UI.

### Troubleshooting

- **Terminal compatibility** – If borders look jagged or colours are muted, switch to a Nerd Font or powerline-friendly font and confirm the terminal advertises true-colour support.
- **Dependency import errors** – Install Textual and Rich individually (`pip install textual rich`) or rerun the main requirements install.
- **Keyboard shortcuts** – `Ctrl+C` triggers a graceful exit; `Esc` + `/cancel` cancels the active request when the manager is busy.
