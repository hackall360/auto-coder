"""Utilities for interacting with the web via Playwright.

This module centralises the logic required to perform search queries and to
render web pages before extracting readable text.  The functionality is kept in
one place so that higher-level modules can depend on a thin abstraction without
having to manage browser lifecycles directly.
"""
from __future__ import annotations

import contextlib
import re
import urllib.parse
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:  # pragma: no cover - import guarded for environments without Playwright
    from playwright.sync_api import (  # type: ignore
        Browser,
        BrowserContext,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        sync_playwright,
    )
except Exception:  # pragma: no cover - dependency may be optional at runtime
    Browser = BrowserContext = Page = object  # type: ignore[assignment]
    PlaywrightTimeoutError = Exception
    sync_playwright = None


_CLEAN_RE = re.compile(r"\s+")


class PlaywrightWebClient:
    """Small helper around the Playwright sync API.

    Parameters
    ----------
    browser : str
        The browser family to launch (``"chromium"`` by default).
    headless : bool
        Whether to launch the browser in headless mode.
    timeout_ms : int
        Default timeout applied to navigation and selector waits.
    user_agent : str | None
        Optional custom user-agent string for the created context.
    """

    def __init__(
        self,
        browser: str = "chromium",
        headless: bool = True,
        timeout_ms: int = 15_000,
        user_agent: Optional[str] = None,
        *,
        user_agent_pool: Optional[Sequence[str]] = None,
        proxy: Optional[Dict[str, Any]] = None,
        incognito_contexts: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        self.browser = browser
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.user_agent = user_agent
        self.user_agent_pool = [ua.strip() for ua in user_agent_pool or [] if ua and ua.strip()]
        self.proxy = dict(proxy) if proxy else None
        self.incognito_contexts = incognito_contexts
        self._rng = random.Random(random_seed)

    def _choose_user_agent(self) -> Optional[str]:
        if self.user_agent_pool:
            return self._rng.choice(self.user_agent_pool)
        return self.user_agent

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return True when Playwright can be imported and started."""

        return sync_playwright is not None

    @contextlib.contextmanager
    def _new_context(self) -> Iterator[BrowserContext]:
        if not self.is_available():
            raise RuntimeError("Playwright is not available")

        with sync_playwright() as p:  # type: ignore[misc]
            browser_launcher = getattr(p, self.browser, None)
            if browser_launcher is None:
                browser_launcher = p.chromium
            launch_kwargs: Dict[str, Any] = {"headless": self.headless}
            if self.proxy:
                launch_kwargs["proxy"] = dict(self.proxy)
            browser: Browser = browser_launcher.launch(**launch_kwargs)
            context_options: Dict[str, Any] = {}
            user_agent = self._choose_user_agent()
            if user_agent:
                context_options["user_agent"] = user_agent
            context: BrowserContext = browser.new_context(**context_options)
            context.set_default_timeout(self.timeout_ms)
            try:
                yield context
            finally:
                try:
                    context.close()
                finally:
                    browser.close()

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def collect_search_results(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Query DuckDuckGo and return structured search results.

        The Playwright driven fetch performs the same role as the previous
        ``requests``-based HTML scraping, but renders the page to allow the
        search engine to serve dynamic results when necessary.
        """

        if max_results <= 0:
            return []
        if not self.is_available():
            raise RuntimeError("Playwright is not available")

        encoded_query = urllib.parse.quote_plus(query)
        results: List[Dict[str, str]] = []
        try:
            with self._new_context() as context:
                page = context.new_page()
                page.goto(f"https://duckduckgo.com/?q={encoded_query}", wait_until="domcontentloaded")
                try:
                    page.wait_for_selector("a.result__a", timeout=self.timeout_ms)
                except PlaywrightTimeoutError:
                    pass
                anchors = page.query_selector_all("a.result__a")
                for anchor in anchors:
                    url = anchor.get_attribute("href") or ""
                    if not url:
                        continue
                    title = (anchor.inner_text() or "").strip()
                    snippet = ""
                    try:
                        snippet = anchor.evaluate(
                            "el => {\n"
                            "  const article = el.closest('article');\n"
                            "  if (!article) return '';\n"
                            "  const snippet = article.querySelector('.result__snippet, .result__snippet.js-result-snippet');\n"
                            "  return snippet ? snippet.innerText : '';\n"
                            "}"
                        )
                    except Exception:
                        snippet = ""
                    results.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet.strip(),
                    })
                    if len(results) >= max_results:
                        break
        except Exception:
            return []
        return results

    # ------------------------------------------------------------------
    # Page rendering helpers
    # ------------------------------------------------------------------
    def render_page_text(self, url: str) -> Optional[str]:
        """Return readable text for the provided URL."""

        if not self.is_available():
            raise RuntimeError("Playwright is not available")

        try:
            with self._new_context() as context:
                page: Page = context.new_page()
                page.goto(url, wait_until="domcontentloaded")
                # Give the page a tiny bit of time to settle.
                page.wait_for_timeout(300)
                raw_text = page.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:
            return None

        if not isinstance(raw_text, str):
            return None
        cleaned = _CLEAN_RE.sub(" ", raw_text)
        return cleaned.strip()[:500_000]


__all__ = ["PlaywrightWebClient"]
