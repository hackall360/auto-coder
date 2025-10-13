"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import sys

from TUI import run_tui


def main(argv: list[str] | None = None) -> int:
    """Launch the Textual TUI entrypoint."""

    return run_tui(argv)


if __name__ == "__main__":
    sys.exit(main())
