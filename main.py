"""Command-line entry point bootstrapping the manager agent."""

from __future__ import annotations

import sys

from TUI import run_tui
from logging_config import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Launch the Textual TUI entrypoint."""

    configure_logging()
    return run_tui(argv)


if __name__ == "__main__":
    sys.exit(main())
