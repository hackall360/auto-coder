from __future__ import annotations

import importlib
import sys
import types


def import_main(monkeypatch, run_tui_impl):
    stub_module = types.SimpleNamespace(run_tui=run_tui_impl)
    monkeypatch.setitem(sys.modules, "TUI", stub_module)
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


def test_main_forwards_arguments(monkeypatch):
    captured = {}

    def fake_run_tui(argv):
        captured["argv"] = argv
        return 17

    main_module = import_main(monkeypatch, fake_run_tui)

    exit_code = main_module.main(["--example", "value"])

    assert exit_code == 17
    assert captured["argv"] == ["--example", "value"]


def test_main_uses_none_when_no_arguments(monkeypatch):
    captured = {}

    def fake_run_tui(argv):
        captured["argv"] = argv
        return 0

    main_module = import_main(monkeypatch, fake_run_tui)

    exit_code = main_module.main()

    assert exit_code == 0
    assert captured["argv"] is None
