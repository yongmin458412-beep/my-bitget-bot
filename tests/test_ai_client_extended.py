"""Extended AI client tests: cache hit/miss, connection errors, no-key behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIConnectionError, APIStatusError

from ai.client import OpenAIResponsesClient
from ai.schemas import NewsAnalysis
from core.persistence import SQLitePersistence
from core.settings import AppSettings


def _make_client(tmp_path: Path, *, api_key: str = "sk-test") -> tuple[OpenAIResponsesClient, SQLitePersistence]:
    settings = AppSettings()
    settings.secrets.openai_api_key = api_key
    settings.secrets.openai_model = "gpt-test"
    persistence = SQLitePersistence(tmp_path / "ai.sqlite3")
    client = OpenAIResponsesClient(settings, persistence)
    return client, persistence


def _fake_response_payload() -> dict:
    return {
        "summary_ko": "테스트 요약",
        "impacted_assets": ["BTC"],
        "impact_level": "high",
        "direction_bias": "bearish",
        "validity_window_minutes": 60,
        "confidence": 0.85,
        "event_type": "economic",
        "should_block_new_entries": True,
        "notes": "test",
    }


def _mock_api_success(client: OpenAIResponsesClient, payload: dict) -> None:
    import json

    mock_response = MagicMock()
    mock_response.output_text = json.dumps(payload)

    fake_responses = MagicMock()
    fake_responses.create = AsyncMock(return_value=mock_response)

    fake_openai = MagicMock()
    fake_openai.responses = fake_responses
    client.client = fake_openai


def test_client_returns_none_when_no_api_key() -> None:
    """If no API key configured, all calls should return None gracefully."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.secrets.openai_api_key = ""
        client = OpenAIResponsesClient(settings)

        result = await client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt="s",
            user_prompt="u",
        )
        assert result is None

    asyncio.run(scenario())


def test_client_returns_cached_result_without_api_call(tmp_path: Path) -> None:
    """Cache hit should return model instance without calling OpenAI."""

    async def scenario() -> None:
        client, persistence = _make_client(tmp_path)
        payload = _fake_response_payload()

        # Pre-populate cache
        persistence.upsert_runtime_value("ai_cache:test-key", payload, "cached")

        # Mock API — should NOT be called
        api_called = False

        async def mock_create(*a, **kw):
            nonlocal api_called
            api_called = True
            raise AssertionError("API should not be called on cache hit")

        fake_responses = MagicMock()
        fake_responses.create = mock_create
        fake_openai = MagicMock()
        fake_openai.responses = fake_responses
        client.client = fake_openai

        result = await client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt="s",
            user_prompt="u",
            cache_key="test-key",
        )

        assert result is not None
        assert isinstance(result, NewsAnalysis)
        assert result.summary_ko == "테스트 요약"
        assert not api_called

    asyncio.run(scenario())


def test_client_populates_cache_on_success(tmp_path: Path) -> None:
    """Successful API call should store result in cache for next time."""

    async def scenario() -> None:
        client, persistence = _make_client(tmp_path)
        payload = _fake_response_payload()
        _mock_api_success(client, payload)

        result = await client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt="s",
            user_prompt="u",
            cache_key="populate-test",
        )

        assert result is not None

        # Check cache was written
        cached = persistence.get_runtime_value("ai_cache:populate-test")
        assert cached is not None
        assert cached["summary_ko"] == "테스트 요약"

    asyncio.run(scenario())


def test_client_retries_on_connection_error(tmp_path: Path) -> None:
    """APIConnectionError should trigger retries (at least 2 attempts)."""

    async def scenario() -> None:
        import pytest
        client, _ = _make_client(tmp_path)

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        call_count = 0

        async def always_fail(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise APIConnectionError(request=request)

        fake_responses = MagicMock()
        fake_responses.create = always_fail
        fake_openai = MagicMock()
        fake_openai.responses = fake_responses
        client.client = fake_openai

        # APIConnectionError reraises after 3 attempts — confirm retry happened
        with pytest.raises(APIConnectionError):
            await client.structured_response(
                schema_name="news_analysis",
                schema_model=NewsAnalysis,
                system_prompt="s",
                user_prompt="u",
            )

        assert call_count == 3  # tenacity retried 3 times

    asyncio.run(scenario())


def test_client_does_not_retry_rate_limit(tmp_path: Path) -> None:
    """RateLimitError should NOT be retried — return None immediately."""

    async def scenario() -> None:
        from openai import RateLimitError
        client, _ = _make_client(tmp_path)

        call_count = 0

        async def rate_limited(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise RateLimitError(
                "quota exceeded",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
                body={"error": {"message": "quota exceeded"}},
            )

        fake_responses = MagicMock()
        fake_responses.create = rate_limited
        fake_openai = MagicMock()
        fake_openai.responses = fake_responses
        client.client = fake_openai

        result = await client.structured_response(
            schema_name="news_analysis",
            schema_model=NewsAnalysis,
            system_prompt="s",
            user_prompt="u",
        )

        assert result is None
        assert call_count == 1  # No retries

    asyncio.run(scenario())
