"""News collector failure-handling tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from core.enums import EventType
from core.persistence import SQLitePersistence
from news.collector import NewsCollector
from news.rss_sources import SourceConfig


def test_news_collector_skips_failing_source(tmp_path: Path) -> None:
    """Collector should continue when one source returns an HTTP error."""

    async def scenario() -> None:
        persistence = SQLitePersistence(tmp_path / "news.sqlite3")
        collector = NewsCollector(persistence)

        async def fake_fetch_html(source: SourceConfig, client: httpx.AsyncClient):  # type: ignore[override]
            raise httpx.HTTPStatusError(
                "403 forbidden",
                request=httpx.Request("GET", source.url),
                response=httpx.Response(403, request=httpx.Request("GET", source.url)),
            )

        collector._fetch_html = fake_fetch_html  # type: ignore[method-assign]

        import news.collector as collector_module

        original_default_sources = collector_module.default_sources
        collector_module.default_sources = lambda: [  # type: ignore[assignment]
            SourceConfig("Bitget Notice", "https://www.bitget.com/support/", "html", EventType.EXCHANGE_NOTICE)
        ]
        try:
            items = await collector.fetch_all()
            assert items == []
        finally:
            collector_module.default_sources = original_default_sources  # type: ignore[assignment]

    asyncio.run(scenario())
