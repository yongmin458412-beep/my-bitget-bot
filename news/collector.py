"""Periodic news collection from RSS and HTML sources."""

from __future__ import annotations

import asyncio
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup

from core.logger import get_logger
from core.persistence import SQLitePersistence

from .economic_calendar import EconomicCalendarAdapter
from .parser import ParsedNewsItem, normalize_entry
from .rss_sources import SourceConfig, default_sources


class NewsCollector:
    """Collect crypto news, exchange notices, and macro events."""

    def __init__(self, persistence: SQLitePersistence) -> None:
        self.persistence = persistence
        self.calendar_adapter = EconomicCalendarAdapter()
        self.logger = get_logger(__name__)

    async def fetch_all(self) -> list[ParsedNewsItem]:
        """Fetch every configured source and return new items only."""

        items: list[ParsedNewsItem] = []
        async with httpx.AsyncClient(timeout=15) as client:
            for source in default_sources():
                try:
                    if source.kind == "rss":
                        items.extend(await self._fetch_rss(source))
                    else:
                        items.extend(await self._fetch_html(source, client))
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "News source skipped after fetch error",
                        extra={"extra_data": {"source": source.name, "url": source.url, "error": str(exc)}},
                    )
        return self._filter_new(items)

    async def _fetch_rss(self, source: SourceConfig) -> list[ParsedNewsItem]:
        """Fetch and parse an RSS source."""

        feed = await asyncio.to_thread(feedparser.parse, source.url)
        entries = [dict(entry) for entry in feed.entries[:20]]
        if source.event_type.value == "economic":
            return self.calendar_adapter.parse_entries(source, entries)
        return [
            normalize_entry(
                source=source.name,
                title=str(entry.get("title", "")),
                url=str(entry.get("link", source.url)),
                published_at=entry.get("published"),
                content=str(entry.get("summary", entry.get("description", ""))),
                event_type=source.event_type,
            )
            for entry in entries
        ]

    async def _fetch_html(self, source: SourceConfig, client: httpx.AsyncClient) -> list[ParsedNewsItem]:
        """Fetch and parse a simple HTML listing page."""

        response = await client.get(source.url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        items: list[ParsedNewsItem] = []
        for anchor in soup.select("a")[:50]:
            title = anchor.get_text(" ", strip=True)
            href = anchor.get("href") or source.url
            if not title or len(title) < 12:
                continue
            if href.startswith("/"):
                href = f"https://www.bitget.com{href}"
            items.append(
                normalize_entry(
                    source=source.name,
                    title=title,
                    url=href,
                    published_at=None,
                    content=title,
                    event_type=source.event_type,
                )
            )
        return items[:20]

    def _filter_new(self, items: list[ParsedNewsItem]) -> list[ParsedNewsItem]:
        """Remove already persisted items."""

        fresh: list[ParsedNewsItem] = []
        for item in items:
            row = self.persistence.fetchone(
                "SELECT news_hash FROM news_items WHERE news_hash = ?",
                (item.news_hash,),
            )
            if row is None:
                fresh.append(item)
        return fresh
