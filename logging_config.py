"""Centralized logging configuration utilities for Auto-Coder entrypoints."""

from __future__ import annotations

import json
import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Mapping, MutableMapping

__all__ = ["configure_logging"]

_DEFAULT_ENV_VAR = "AUTO_CODER_LOGGING_CONFIG"
_LAST_FINGERPRINT: str | None = None


def _normalise_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *config* with dictionaries sorted for hashing."""

    def _convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _convert(value[key]) for key in sorted(value.keys(), key=str)}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return _convert(dict(config))  # type: ignore[return-value]


def _fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for *config* used to avoid reconfiguration."""

    normalised = _normalise_mapping(config)
    try:
        return json.dumps(normalised, sort_keys=True, default=str)
    except TypeError:
        return repr(normalised)


def _coerce_level(default: str = "INFO") -> str:
    level = os.getenv("AUTO_CODER_LOG_LEVEL", default)
    if not level:
        return default
    return str(level).upper()


def _build_default_config() -> dict[str, Any]:
    """Return the default logging configuration used across entrypoints."""

    level = _coerce_level()
    console_level = os.getenv("AUTO_CODER_CONSOLE_LEVEL", level)
    file_level = os.getenv("AUTO_CODER_FILE_LEVEL", level)
    log_file = os.getenv("AUTO_CODER_LOG_FILE")

    handlers: dict[str, MutableMapping[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": str(console_level).upper(),
            "formatter": "structured",
            "stream": "ext://sys.stderr",
        }
    }
    handler_names: list[str] = ["console"]

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": str(file_level).upper(),
            "formatter": "structured",
            "filename": str(path),
            "encoding": "utf-8",
        }
        handler_names.append("file")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": (
                    "{" "\"timestamp\":\"%(asctime)s\"," "\"level\":\"%(levelname)s\"," "\"name\":\"%(name)s\"," "\"message\":\"%(message)s\"}"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": handler_names,
        },
    }


def _load_config_from_path(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        msg = f"Failed to parse logging configuration from {path}: {exc}"
        raise ValueError(msg) from exc


def _load_config_from_payload(payload: str) -> Mapping[str, Any]:
    path = Path(payload)
    if path.exists():
        return _load_config_from_path(path)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        msg = "Logging configuration overrides must be valid JSON or file paths"
        raise ValueError(msg) from exc


def _resolve_configuration(
    config: Mapping[str, Any] | None,
    config_path: str | os.PathLike[str] | None,
    env_var: str,
) -> Mapping[str, Any]:
    if config is not None:
        return config
    if config_path is not None:
        return _load_config_from_payload(str(config_path))
    env_override = os.getenv(env_var)
    if env_override:
        return _load_config_from_payload(env_override)
    return _build_default_config()


def configure_logging(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    env_var: str = _DEFAULT_ENV_VAR,
    force: bool = False,
) -> None:
    """Install the Auto-Coder logging configuration.

    Parameters
    ----------
    config:
        Explicit configuration mapping compatible with :func:`logging.config.dictConfig`.
    config_path:
        Path to a JSON file or a JSON string containing the configuration.  When
        provided it takes precedence over ``env_var``.
    env_var:
        Name of the environment variable providing overrides (defaults to
        ``AUTO_CODER_LOGGING_CONFIG``).
    force:
        When ``True`` the configuration is always applied, even if it matches the
        previous invocation.
    """

    global _LAST_FINGERPRINT

    resolved = _resolve_configuration(config, config_path, env_var)
    fingerprint = _fingerprint(resolved)

    if not force and _LAST_FINGERPRINT == fingerprint:
        return

    dictConfig(resolved)
    _LAST_FINGERPRINT = fingerprint

    logging.captureWarnings(True)
