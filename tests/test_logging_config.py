from __future__ import annotations

import json
import logging
from logging import LogRecord

import pytest

from logging_config import configure_logging


def _reset_root_logger() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def _reset_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_root_logger()
    monkeypatch.delenv("AUTO_CODER_LOGGING_CONFIG", raising=False)
    monkeypatch.delenv("AUTO_CODER_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AUTO_CODER_CONSOLE_LEVEL", raising=False)
    monkeypatch.delenv("AUTO_CODER_FILE_LEVEL", raising=False)
    monkeypatch.delenv("AUTO_CODER_LOG_FILE", raising=False)


def test_configure_logging_installs_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_CODER_LOG_LEVEL", "WARNING")

    configure_logging(force=True)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1

    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.WARNING
    formatter = handler.formatter
    assert formatter is not None

    record = LogRecord(
        name="auto-coder.tests",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg="example",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "\"level\":\"WARNING\"" in rendered
    assert "\"message\":\"example\"" in rendered


def test_configure_logging_is_idempotent() -> None:
    configure_logging(force=True)

    root = logging.getLogger()
    first_handlers = tuple(root.handlers)

    configure_logging()

    assert tuple(root.handlers) == first_handlers
    for left, right in zip(root.handlers, first_handlers):
        assert left is right


def test_configure_logging_prefers_explicit_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_CODER_LOG_LEVEL", "ERROR")

    configure_logging(force=True, level="debug")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    handler = root.handlers[0]
    assert handler.level == logging.DEBUG


def test_configure_logging_respects_json_override(monkeypatch: pytest.MonkeyPatch) -> None:
    override = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"format": "%(message)s"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": "ERROR", "handlers": ["console"]},
    }
    monkeypatch.setenv("AUTO_CODER_LOGGING_CONFIG", json.dumps(override))

    configure_logging(force=True)

    root = logging.getLogger()
    assert root.level == logging.ERROR
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.ERROR
