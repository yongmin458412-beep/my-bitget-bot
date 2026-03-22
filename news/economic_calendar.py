"""Macro event adapter using official Fed and BLS feeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parser import ParsedNewsItem, normalize_entry
from .rss_sources import SourceConfig


@dataclass(slots=True)
class CalendarEvent:
    """Normalized macro event."""

    title: str
    source: str
    published_at: str
    url: str
    content: str


class EconomicCalendarAdapter:
    """Transform official macro feeds into normalized news items."""

    IMPORTANT_KEYWORDS = ("fomc", "federal open market committee", "cpi", "inflation", "nfp", "employment")

    def parse_entries(self, source: SourceConfig, entries: list[dict[str, Any]]) -> list[ParsedNewsItem]:
        """Filter and normalize important macro items."""

        items: list[ParsedNewsItem] = []
        for entry in entries:
            title = str(entry.get("title", ""))
            content = str(entry.get("summary", entry.get("description", title)))
            merged = f"{title} {content}".lower()
            if not any(keyword in merged for keyword in self.IMPORTANT_KEYWORDS):
                continue
            items.append(
                normalize_entry(
                    source=source.name,
                    title=title,
                    url=str(entry.get("link", source.url)),
                    published_at=entry.get("published"),
                    content=content,
                    event_type=source.event_type,
                )
            )
        return items

