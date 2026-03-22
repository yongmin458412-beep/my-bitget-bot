"""High-level AI helper functions."""

from __future__ import annotations

import json
from typing import Any

from core.utils import hash_text

from .client import OpenAIResponsesClient
from .prompts import journal_prompt, news_system_prompt, news_user_prompt, why_prompt
from .schemas import JournalSummary, NewsAnalysis, WhyExplanation


class AISummarizer:
    """Convenience layer around the structured OpenAI client."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self.client = client

    async def analyze_news(
        self,
        *,
        title: str,
        body: str,
        source: str,
        published_at: str,
        related_assets: list[str],
    ) -> NewsAnalysis | None:
        """Analyze a news item."""

        cache_key = hash_text(f"{title}|{published_at}|{source}")
        return await self.client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt=news_system_prompt(),
            user_prompt=news_user_prompt(title, body, source, published_at, related_assets),
            cache_key=cache_key,
        )

    async def explain_why(self, symbol: str, payload: dict[str, Any]) -> WhyExplanation | None:
        """Explain a trade decision in Korean."""

        return await self.client.structured_response(
            schema_name="why_explanation",
            schema_model=WhyExplanation,
            system_prompt="너는 규칙형 자동매매 시스템의 설명 보조 엔진이다. 판단을 바꾸지 말고 쉽게 설명하라.",
            user_prompt=why_prompt(symbol, json.dumps(payload, ensure_ascii=False, indent=2)),
            cache_key=None,
            max_output_tokens=500,
        )

    async def summarize_journal(self, rows: list[dict[str, Any]]) -> JournalSummary | None:
        """Summarize recent journal entries."""

        return await self.client.structured_response(
            schema_name="journal_summary",
            schema_model=JournalSummary,
            system_prompt="너는 자동매매 리뷰어다. 성과를 과장하지 말고 교훈을 짧고 명확하게 정리하라.",
            user_prompt=journal_prompt(json.dumps(rows, ensure_ascii=False, indent=2)),
            cache_key=None,
            max_output_tokens=600,
        )

