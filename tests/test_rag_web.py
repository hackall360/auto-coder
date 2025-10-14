from internal import RAG as rag_module


class _RecordPlaywrightClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.collect_calls = []
        self.render_calls = []

    def is_available(self) -> bool:
        return True

    def collect_search_results(self, query: str, max_results: int = 10):
        self.collect_calls.append((query, max_results))
        return [
            {"url": "https://example.com/page", "title": "Example", "snippet": "Snippet"},
        ]

    def render_page_text(self, url: str):
        self.render_calls.append(url)
        return "This page discusses Test query usage."


class _UnavailablePlaywrightClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return False


def test_web_rag_search_uses_playwright(monkeypatch):
    monkeypatch.setattr(rag_module, "PlaywrightWebClient", _RecordPlaywrightClient)

    def _unexpected_ddg(self, query, max_results):
        raise AssertionError("Fallback DuckDuckGo search should not run when Playwright succeeds")

    monkeypatch.setattr(rag_module.WebRAG, "_search_ddg", _unexpected_ddg)

    web = rag_module.WebRAG()
    results = web.search("Test query", top_k=1, max_search_results=5, allow_rewrite=False)

    assert results, "Expected search results from Playwright ingestion"
    client = web._playwright_client
    assert isinstance(client, _RecordPlaywrightClient)
    assert client.collect_calls == [("Test query", 5)]
    assert client.render_calls == ["https://example.com/page"]


def test_web_rag_search_falls_back_without_playwright(monkeypatch):
    monkeypatch.setattr(rag_module, "PlaywrightWebClient", _UnavailablePlaywrightClient)

    ddg_calls: list[tuple[str, int]] = []

    def _fake_ddg(self, query, max_results):
        ddg_calls.append((query, max_results))
        return [
            {"url": "https://fallback.example", "title": "Fallback", "snippet": ""},
        ]

    def _fake_fetch(self, url):
        return "Fallback content for Fallback test"

    monkeypatch.setattr(rag_module.WebRAG, "_search_ddg", _fake_ddg)
    monkeypatch.setattr(rag_module.WebRAG, "_fetch_text", _fake_fetch)

    web = rag_module.WebRAG()
    results = web.search("Fallback test", top_k=1, max_search_results=3, allow_rewrite=False)

    assert ddg_calls == [("Fallback test", 3)]
    assert results, "Expected fallback results when Playwright is unavailable"
    assert results[0]["path"] == "https://fallback.example"


def test_web_rag_configuration_passed_to_playwright(monkeypatch):
    captured_clients: list[_RecordPlaywrightClient] = []

    class _CapturePlaywrightClient(_RecordPlaywrightClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured_clients.append(self)

    monkeypatch.setattr(rag_module, "PlaywrightWebClient", _CapturePlaywrightClient)

    web = rag_module.WebRAG(
        user_agent_pool=["AgentOne", "AgentTwo"],
        proxy="http://proxy.example:8080",
        incognito_contexts=True,
        random_seed=123,
    )

    assert captured_clients, "Expected Playwright client to be constructed"
    client = captured_clients[0]
    assert client.kwargs["user_agent_pool"] == ["AgentOne", "AgentTwo"]
    assert client.kwargs["proxy"] == {"server": "http://proxy.example:8080"}
    assert client.kwargs["incognito_contexts"] is True
    assert client.kwargs["random_seed"] == 123

    headers = web._headers()
    assert headers["User-Agent"] in {"AgentOne", "AgentTwo"}
    if web.session is not None:
        assert web.session.proxies.get("http") == "http://proxy.example:8080"
