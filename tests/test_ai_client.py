"""Tests for OpenAI structured response fallback behavior."""

from __future__ import annotations

import asyncio

import httpx
from openai import RateLimitError

from ai.client import OpenAIResponsesClient
from ai.schemas import NewsAnalysis
from core.settings import AppSettings


def test_structured_response_returns_none_on_rate_limit() -> None:
    """Quota and rate-limit failures should degrade to fallback instead of crashing loops."""

    async def scenario() -> None:
        client = OpenAIResponsesClient(AppSettings())

        class FakeResponses:
            async def create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise RateLimitError(
                    "quota exceeded",
                    response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
                    body={"error": {"message": "quota exceeded"}},
                )

        class FakeOpenAI:
            responses = FakeResponses()

        client.client = FakeOpenAI()  # type: ignore[assignment]
        result = await client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt="system",
            user_prompt="user",
        )
        assert result is None

    asyncio.run(scenario())
