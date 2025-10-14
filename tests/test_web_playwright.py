from internal import web_playwright as module


class _FakePage:
    def __init__(self, goto_log):
        self._goto_log = goto_log

    def goto(self, url, wait_until=None):  # noqa: D401 - signature match
        self._goto_log.append((url, wait_until))

    def wait_for_selector(self, *args, **kwargs):  # noqa: D401 - behaviour mocked
        return None

    def query_selector_all(self, selector):  # noqa: D401 - behaviour mocked
        return []

    def wait_for_timeout(self, ms):  # noqa: D401 - behaviour mocked
        return None

    def evaluate(self, script):  # noqa: D401 - behaviour mocked
        return "Rendered body text"


class _FakeContext:
    def __init__(self, options, timeout_log, goto_log, close_log):
        self.options = options
        self._timeout_log = timeout_log
        self._goto_log = goto_log
        self._close_log = close_log

    def set_default_timeout(self, timeout):
        self._timeout_log.append(timeout)

    def new_page(self):
        return _FakePage(self._goto_log)

    def close(self):
        self._close_log.append(True)


class _FakeBrowser:
    def __init__(self, context_log, timeout_log, goto_log, close_log):
        self._context_log = context_log
        self._timeout_log = timeout_log
        self._goto_log = goto_log
        self._close_log = close_log

    def new_context(self, **kwargs):
        options = dict(kwargs)
        self._context_log.append(options)
        return _FakeContext(options, self._timeout_log, self._goto_log, self._close_log)

    def close(self):
        return None


class _FakeLauncher:
    def __init__(self, launch_log, context_log, timeout_log, goto_log, close_log):
        self._launch_log = launch_log
        self._context_log = context_log
        self._timeout_log = timeout_log
        self._goto_log = goto_log
        self._close_log = close_log

    def launch(self, **kwargs):
        self._launch_log.append(dict(kwargs))
        return _FakeBrowser(self._context_log, self._timeout_log, self._goto_log, self._close_log)


class _FakePlaywright:
    def __init__(self, launcher):
        self.chromium = launcher


def _build_sync_playwright(launcher):
    class _Manager:
        def __enter__(self):
            return _FakePlaywright(launcher)

        def __exit__(self, exc_type, exc, tb):
            return False

    def _sync_playwright():
        return _Manager()

    return _sync_playwright


def test_playwright_client_applies_proxy_and_incognito(monkeypatch):
    launch_log: list[dict] = []
    context_log: list[dict] = []
    timeout_log: list[int] = []
    goto_log: list[tuple[str, str | None]] = []
    close_log: list[bool] = []

    launcher = _FakeLauncher(launch_log, context_log, timeout_log, goto_log, close_log)
    monkeypatch.setattr(module, "sync_playwright", _build_sync_playwright(launcher))

    client = module.PlaywrightWebClient(
        timeout_ms=3210,
        user_agent_pool=["UA-A", "UA-B"],
        proxy={"server": "http://proxy.local:8888"},
        random_seed=0,
    )

    client.collect_search_results("example query", max_results=1)
    text = client.render_page_text("https://example.com")

    assert text == "Rendered body text"
    assert launch_log and launch_log[0]["proxy"] == {"server": "http://proxy.local:8888"}
    assert any(entry.get("user_agent") in {"UA-A", "UA-B"} for entry in context_log)
    assert timeout_log == [3210, 3210]
    assert len(close_log) == 2, "Expected a fresh incognito context per call"
    assert goto_log and goto_log[0][0].startswith("https://duckduckgo.com")
