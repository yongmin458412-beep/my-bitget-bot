"""Tests for selective AI usage in the news analyzer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai.schemas import NewsAnalysis
from core.enums import EventType
from core.persistence import SQLitePersistence
from core.settings import AppSettings
from news.analyzer import NewsAnalyzer
from news.parser import ParsedNewsItem


class DummySummarizer:
    """Track whether OpenAI-backed analysis would have been called."""

    def __init__(self, responses: list[NewsAnalysis | None]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def analyze_news(self, **_: object) -> NewsAnalysis | None:
        """Return the next prepared response."""

        self.calls += 1
        if not self.responses:
            return None
        return self.responses.pop(0)


def _item(*, title: str, published_at: datetime, event_type: EventType) -> ParsedNewsItem:
    """Create a normalized parsed-news item."""

    return ParsedNewsItem(
        news_hash=f"hash-{title}",
        source="unit-test",
        title=title,
        url="https://example.com",
        published_at=published_at.isoformat(timespec="seconds"),
        content=title,
        related_assets=["BTC"],
        event_type=event_type.value,
    )


def _analysis(summary: str) -> NewsAnalysis:
    """Create a deterministic AI response."""

    return NewsAnalysis(
        summary_ko=summary,
        impacted_assets=["BTC"],
        impact_level="high",
        direction_bias="neutral",
        validity_window_minutes=120,
        confidence=0.9,
        event_type=EventType.ECONOMIC.value,
        should_block_new_entries=True,
        notes="ai",
    )


def test_news_analyzer_skips_ai_for_expired_items(tmp_path: Path) -> None:
    """Already-expired events should not spend AI tokens."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.news.ai_analysis_mode = "important_only"
        persistence = SQLitePersistence(tmp_path / "news.sqlite3")
        summarizer = DummySummarizer([_analysis("unused")])
        analyzer = NewsAnalyzer(summarizer, persistence, settings)

        item = _item(
            title="CPI old release",
            published_at=datetime.now(tz=UTC) - timedelta(hours=5),
            event_type=EventType.ECONOMIC,
        )
        result = await analyzer.analyze_item(item)

        assert summarizer.calls == 0
        assert result.summary_ko == "CPI old release"

    asyncio.run(scenario())


def test_news_analyzer_uses_ai_for_current_high_impact_event(tmp_path: Path) -> None:
    """Fresh, important events should still use AI when available."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.news.ai_analysis_mode = "important_only"
        persistence = SQLitePersistence(tmp_path / "news.sqlite3")
        summarizer = DummySummarizer([_analysis("AI 요약")])
        analyzer = NewsAnalyzer(summarizer, persistence, settings)

        item = _item(
            title="CPI surprise today",
            published_at=datetime.now(tz=UTC) - timedelta(minutes=5),
            event_type=EventType.ECONOMIC,
        )
        result = await analyzer.analyze_item(item)

        assert summarizer.calls == 1
        assert result.summary_ko == "AI 요약"

    asyncio.run(scenario())


def test_news_analyzer_pauses_ai_after_failure(tmp_path: Path) -> None:
    """After an AI failure, follow-up items should respect the cooldown window."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.news.ai_analysis_mode = "important_only"
        settings.news.ai_cooldown_minutes_after_failure = 180
        persistence = SQLitePersistence(tmp_path / "news.sqlite3")
        summarizer = DummySummarizer([None, _analysis("should-not-run")])
        analyzer = NewsAnalyzer(summarizer, persistence, settings)

        first = _item(
            title="FOMC today",
            published_at=datetime.now(tz=UTC) - timedelta(minutes=1),
            event_type=EventType.ECONOMIC,
        )
        second = _item(
            title="CPI follow-up",
            published_at=datetime.now(tz=UTC) - timedelta(minutes=1),
            event_type=EventType.ECONOMIC,
        )

        await analyzer.analyze_item(first)
        await analyzer.analyze_item(second)

        assert summarizer.calls == 1

    asyncio.run(scenario())
