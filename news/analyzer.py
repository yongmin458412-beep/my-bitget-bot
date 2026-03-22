"""Analyze news items via OpenAI with deterministic fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai.schemas import NewsAnalysis
from ai.summarizer import AISummarizer
from core.enums import EventType
from core.persistence import SQLitePersistence
from core.settings import AppSettings

from .parser import ParsedNewsItem


class NewsAnalyzer:
    """Analyze and persist news impact results."""

    def __init__(self, summarizer: AISummarizer, persistence: SQLitePersistence, settings: AppSettings) -> None:
        self.summarizer = summarizer
        self.persistence = persistence
        self.settings = settings
        self._ai_paused_until: datetime | None = None

    async def analyze_item(self, item: ParsedNewsItem) -> NewsAnalysis:
        """Analyze a single item with AI or fallback heuristics."""

        analysis = self._fallback(item)
        if self._should_use_ai(item, analysis) and not self._ai_temporarily_paused():
            ai_analysis = await self.summarizer.analyze_news(
                title=item.title,
                body=item.content,
                source=item.source,
                published_at=item.published_at,
                related_assets=item.related_assets,
            )
            if ai_analysis is None:
                self._pause_ai_temporarily()
            else:
                analysis = ai_analysis

        self.persistence.save_news_item(
            {
                "news_hash": item.news_hash,
                "source": item.source,
                "title": item.title,
                "url": item.url,
                "published_at": item.published_at,
                "content": item.content,
                "related_assets": item.related_assets,
                "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
        )
        self.persistence.save_ai_news_analysis(
            {
                "news_hash": item.news_hash,
                "summary_ko": analysis.summary_ko,
                "impacted_assets": analysis.impacted_assets,
                "impact_level": analysis.impact_level,
                "direction_bias": analysis.direction_bias,
                "validity_window_minutes": analysis.validity_window_minutes,
                "confidence": analysis.confidence,
                "event_type": analysis.event_type,
                "should_block_new_entries": analysis.should_block_new_entries,
                "notes": analysis.notes,
                "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
        )
        return analysis

    def _ai_temporarily_paused(self) -> bool:
        """Return whether OpenAI usage is paused after a recent failure."""

        return self._ai_paused_until is not None and self._ai_paused_until > datetime.now(tz=UTC)

    def _pause_ai_temporarily(self) -> None:
        """Pause AI analysis for a while after failures like quota exhaustion."""

        cooldown = max(15, int(self.settings.news.ai_cooldown_minutes_after_failure))
        self._ai_paused_until = datetime.now(tz=UTC) + timedelta(minutes=cooldown)

    def _should_use_ai(self, item: ParsedNewsItem, heuristic: NewsAnalysis) -> bool:
        """Only spend tokens on items important enough to refine."""

        if not self._is_currently_relevant(item, heuristic):
            return False
        mode = str(self.settings.news.ai_analysis_mode).lower()
        if mode == "disabled":
            return False
        if mode == "always":
            return True
        if heuristic.impact_level == "high" or heuristic.should_block_new_entries:
            return True
        if item.event_type in {EventType.ECONOMIC.value, EventType.EXCHANGE_NOTICE.value}:
            return True
        important_assets = {"BTC", "ETH", "TOTAL"}
        return any(asset.upper() in important_assets for asset in item.related_assets) and heuristic.impact_level != "low"

    @staticmethod
    def _is_currently_relevant(item: ParsedNewsItem, heuristic: NewsAnalysis) -> bool:
        """Ignore already-expired events for AI spending and trade blocking."""

        try:
            published_at = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        expires_at = published_at + timedelta(minutes=max(1, int(heuristic.validity_window_minutes or 0)))
        return expires_at > datetime.now(tz=UTC)

    @staticmethod
    def _fallback(item: ParsedNewsItem) -> NewsAnalysis:
        """Fallback heuristic when OpenAI is unavailable."""

        merged = f"{item.title} {item.content}".lower()
        high = any(keyword in merged for keyword in ["hack", "lawsuit", "fomc", "cpi", "liquidation", "delist"])
        direction = "uncertain"
        if any(keyword in merged for keyword in ["approval", "launch", "listing", "inflow"]):
            direction = "bullish"
        elif any(keyword in merged for keyword in ["outflow", "ban", "hack", "delist", "sec"]):
            direction = "bearish"
        return NewsAnalysis(
            summary_ko=item.title,
            impacted_assets=item.related_assets,
            impact_level="high" if high else "medium",
            direction_bias=direction,
            validity_window_minutes=120 if high else 45,
            confidence=0.45,
            event_type=item.event_type,
            should_block_new_entries=high,
            notes="OpenAI 미사용 fallback 분석",
        )
