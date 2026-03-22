"""OpenAI Responses API wrapper with structured outputs and caching."""

from __future__ import annotations

import json
from typing import Any, Type

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError, RateLimitError
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from core.logger import get_logger
from core.persistence import SQLitePersistence
from core.settings import AppSettings


class OpenAIResponsesClient:
    """Minimal async Responses API wrapper."""

    def __init__(self, settings: AppSettings, persistence: SQLitePersistence | None = None) -> None:
        self.settings = settings
        self.persistence = persistence
        self.logger = get_logger(__name__)
        self.client = AsyncOpenAI(api_key=settings.secrets.openai_api_key) if settings.secrets.openai_api_key else None

    @staticmethod
    def _should_retry_exception(exc: BaseException) -> bool:
        """Retry only for transient transport or server-side API failures."""

        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code >= 500
        return False

    async def structured_response(
        self,
        *,
        schema_name: str,
        schema_model: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        cache_key: str | None = None,
        max_output_tokens: int = 600,
    ) -> BaseModel | None:
        """Call the Responses API and parse a typed schema."""

        if cache_key and self.persistence:
            cached = self.persistence.get_runtime_value(f"ai_cache:{cache_key}")
            if cached:
                return schema_model.model_validate(cached)

        if self.client is None:
            return None

        if hasattr(schema_model, "json_schema_strict"):
            schema = schema_model.json_schema_strict()  # type: ignore[assignment]
        else:
            schema = schema_model.model_json_schema()
            schema["additionalProperties"] = False

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception(self._should_retry_exception),
                reraise=True,
            ):
                with attempt:
                    response = await self.client.responses.create(
                        model=self.settings.secrets.openai_model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": schema,
                                "strict": True,
                            }
                        },
                        max_output_tokens=max_output_tokens,
                    )
                    output_text = getattr(response, "output_text", None)
                    if not output_text:
                        chunks: list[str] = []
                        for item in getattr(response, "output", []):
                            for content in getattr(item, "content", []):
                                if getattr(content, "type", "") == "output_text":
                                    chunks.append(content.text)
                        output_text = "".join(chunks)
                    payload = json.loads(output_text)
                    parsed = schema_model.model_validate(payload)
                    if cache_key and self.persistence:
                        self.persistence.upsert_runtime_value(f"ai_cache:{cache_key}", parsed.model_dump(mode="json"), "cached")
                    return parsed
        except (BadRequestError, RateLimitError) as exc:
            self.logger.warning(
                "Structured OpenAI response skipped after non-retryable API error",
                extra={"extra_data": {"schema_name": schema_name, "error_type": type(exc).__name__, "error": str(exc)}},
            )
            return None
        except APIStatusError as exc:
            self.logger.warning(
                "Structured OpenAI response skipped after API status error",
                extra={
                    "extra_data": {
                        "schema_name": schema_name,
                        "status_code": exc.status_code,
                        "error": str(exc),
                    }
                },
            )
            return None
        return None
