from __future__ import annotations

import argparse
import importlib
import sys
import types

import pytest


@pytest.fixture()
def overrides_module(monkeypatch):
    stub_core = types.ModuleType("core")

    class AgentToggleSettings:
        repo_context: bool = True
        research: bool = True
        documentation: bool = True
        dependency: bool = True
        runner: bool = True
        db_migration: bool = False
        security: bool = False
        integrations: bool = False
        eval: bool = False
        test_critic: bool = True

    stub_core.AgentToggleSettings = AgentToggleSettings
    monkeypatch.setitem(sys.modules, "core", stub_core)
    if "cli.overrides" in sys.modules:
        del sys.modules["cli.overrides"]
    module = importlib.import_module("cli.overrides")
    return module


def parse_args(module, argv: list[str]):
    parser = argparse.ArgumentParser(prog="auto-coder")
    module.apply_common_flags(parser)
    return parser.parse_args(argv)


def test_build_overrides_constructs_expected_structure(overrides_module):
    args = parse_args(
        overrides_module,
        [
            "--config",
            "/tmp/config.json",
            "--mcp-config",
            "/tmp/mcp.json",
            "--default-model",
            "gpt-mini",
            "--reasoning-model",
            "gpt-pro",
            "--research-model",
            "gpt-research",
            "--allow-browsing",
            "--enable-agent",
            "db_migration",
            "--disable-agent",
            "research",
            "--repo-include-ext",
            "py",
            "--repo-include-ext",
            "md,txt",
            "--repo-exclude-dir",
            "node_modules",
            "--repo-no-auto-refresh",
            "--repo-refresh-interval",
            "120",
            "--memory-config",
            "/tmp/memory.json",
            "--no-shared-memory",
            "--enable-corpus",
            "--corpus-path",
            "/tmp/corpus.jsonl",
            "--corpus-dedup-threshold",
            "0.7",
            "--corpus-category",
            "web_search=discovery",
            "--corpus-category",
            "file_write=repo_activity",
            "--mcp-auto-start",
        ],
    )

    overrides = overrides_module.build_overrides(args)

    assert overrides == {
        "models": {
            "default_model": "gpt-mini",
            "reasoning_model": "gpt-pro",
            "research_model": "gpt-research",
            "allow_external_browsing": True,
        },
        "agents": {
            "db_migration": True,
            "research": False,
        },
        "repo_context": {
            "include_exts": ["py", "md", "txt"],
            "exclude_dirs": ["node_modules"],
            "auto_refresh": False,
            "refresh_interval": 120.0,
        },
        "memory": {
            "config_path": "/tmp/memory.json",
            "share_globally": False,
        },
        "corpus": {
            "enabled": True,
            "storage_path": "/tmp/corpus.jsonl",
            "dedup_threshold": 0.7,
            "default_categories": {
                "web_search": "discovery",
                "file_write": "repo_activity",
            },
        },
        "mcp": {
            "config_path": "/tmp/mcp.json",
            "auto_start": True,
        },
    }


def test_build_overrides_ignores_missing_values(overrides_module):
    args = argparse.Namespace(
        default_model=None,
        reasoning_model=None,
        research_model=None,
        allow_browsing=None,
        repo_include_ext=None,
        repo_exclude_dir=None,
        repo_auto_refresh=None,
        repo_refresh_interval=None,
        enable_agent=None,
        disable_agent=None,
        memory_config_path=None,
        share_memory=None,
        corpus_enabled=None,
        corpus_storage_path=None,
        corpus_dedup_threshold=None,
        corpus_category=None,
        mcp_config_path=None,
        mcp_auto_start=None,
    )

    assert overrides_module.build_overrides(args) == {}


def test_apply_common_flags_supports_tristate_toggles(overrides_module):
    parser = argparse.ArgumentParser(prog="auto-coder")
    overrides_module.apply_common_flags(parser)

    defaults = parser.parse_args([])
    assert defaults.allow_browsing is None
    assert defaults.repo_auto_refresh is None
    assert defaults.share_memory is None
    assert defaults.corpus_enabled is None
    assert defaults.mcp_auto_start is None
    assert defaults.log_level is None

    assert parser.parse_args(["--allow-browsing"]).allow_browsing is True
    assert parser.parse_args(["--disable-browsing"]).allow_browsing is False

    assert parser.parse_args(["--repo-auto-refresh"]).repo_auto_refresh is True
    assert parser.parse_args(["--repo-no-auto-refresh"]).repo_auto_refresh is False

    assert parser.parse_args(["--shared-memory"]).share_memory is True
    assert parser.parse_args(["--no-shared-memory"]).share_memory is False

    assert parser.parse_args(["--enable-corpus"]).corpus_enabled is True
    assert parser.parse_args(["--disable-corpus"]).corpus_enabled is False

    assert parser.parse_args(["--mcp-auto-start"]).mcp_auto_start is True
    assert parser.parse_args(["--no-mcp-auto-start"]).mcp_auto_start is False


def test_apply_common_flags_handles_logging_levels(overrides_module):
    parser = argparse.ArgumentParser(prog="auto-coder")
    overrides_module.apply_common_flags(parser)

    assert parser.parse_args(["--verbose"]).log_level == "DEBUG"
    assert parser.parse_args(["--quiet"]).log_level == "WARNING"
    assert parser.parse_args(["--log-level", "info"]).log_level == "INFO"
    assert parser.parse_args(["--log-level", "10"]).log_level == "10"

    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "invalid"])
