"""Anthropic Claude API 래퍼 — 에스컬레이션 전용."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Type

from pydantic import BaseModel

from core.logger import get_logger
from core.settings import AppSettings


class AnthropicClient:
    """Claude API wrapper for escalation scenarios."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self.client: Any = None
        api_key = getattr(settings.secrets, "anthropic_api_key", "")
        if api_key:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=api_key)
            except ImportError:
                self.logger.warning("anthropic 패키지 미설치 — 에스컬레이션 비활성화")

    @property
    def available(self) -> bool:
        return self.client is not None

    async def structured_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        max_tokens: int = 600,
        timeout_seconds: float = 3.0,
    ) -> BaseModel | None:
        """Claude 호출 + JSON 파싱. 타임아웃 시 None 반환."""
        if not self.client:
            return None
        try:
            model = getattr(self.settings, "_ai_anthropic_model", None) or "claude-sonnet-4-20250514"
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=timeout_seconds,
            )
            text = response.content[0].text if response.content else ""
            # JSON 추출
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return schema_model.model_validate(data)
            return None
        except asyncio.TimeoutError:
            self.logger.warning("Anthropic 타임아웃", extra={"extra_data": {"timeout": timeout_seconds}})
            return None
        except Exception as exc:
            self.logger.warning("Anthropic 호출 실패", extra={"extra_data": {"error": str(exc)[:200]}})
            return None
