"""Compatibility shim that re-exports Python's standard :mod:`logging` package."""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path


def _load_stdlib_logging() -> type[object]:
    """Load the standard library ``logging`` module.

    The repository provides a ``logging.py`` file which shadows Python's built-in
    :mod:`logging` package when the project root is on ``sys.path``.  The Textual
    test harness (and several other dependencies) expect the full logging API to
    be available.  To avoid repeating a large stub we import the canonical module
    directly from the stdlib path and mirror its namespace.
    """

    stdlib_path = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "_stdlib_logging",
        stdlib_path,
        submodule_search_locations=[str(stdlib_path.parent)],
    )
    if spec is None or spec.loader is None:
        msg = f"Failed to resolve stdlib logging module from {stdlib_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_stdlib_logging", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_stdlib_logging = _load_stdlib_logging()

# Mirror the public namespace from the real logging module.
globals().update(vars(_stdlib_logging))

# Ensure ``sys.modules['logging']`` refers to this compatibility shim so that
# subsequent ``import logging`` statements reuse the populated namespace.
sys.modules.setdefault("logging", sys.modules[__name__])

