"""AI 트레이드 필터 — 규칙 통과 후 AI 확인/거부 (보조 역할).

핵심 원칙:
  - AI는 최종 결정자가 아닌 "보조 필터"
  - 규칙 기반 신호가 먼저 통과한 후, AI가 확인/거부
  - 비용 효율: GPT-4o-mini 우선, 에스컬레이션 시 Claude Sonnet
  - 실패/타임아웃 → 규칙만으로 진행 (fallback)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field as PydanticField

from ai.anthropic_client import AnthropicClient
from ai.client import OpenAIResponsesClient
from ai.cost_tracker import AICostTracker
from core.enums import RegimeType, Side
from core.logger import get_logger


# ── AI 응답 스키마 ──────────────────────────────────────────────────────────


class TradeFilterAnalysis(BaseModel):
    """AI 트레이드 필터 응답."""

    approved: bool = True
    confidence: float = PydanticField(default=0.7, ge=0.0, le=1.0)
    reasoning_ko: str = ""
    risk_flags: list[str] = PydanticField(default_factory=list)


# ── 결과 ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AIFilterVerdict:
    approved: bool
    confidence: float
    provider_used: str  # "openai" | "anthropic" | "skipped" | "fallback"
    reason: str
    cost_usd: float
    latency_ms: float


# ── 필터 본체 ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a crypto futures trading risk analyst.
Analyze the market data and respond ONLY with a JSON object matching this schema:
{"approved": bool, "confidence": 0.0-1.0, "reasoning_ko": "한국어 1-2문장", "risk_flags": ["flag1"]}
confidence < 0.6 means you are unsure → set approved=false.
Be conservative: reject if uncertain."""


class AITradeFilter:
    """규칙 통과 후 AI 확인 필터."""

    def __init__(
        self,
        openai_client: OpenAIResponsesClient,
        anthropic_client: AnthropicClient | None,
        cost_tracker: AICostTracker,
        config: Any,
    ) -> None:
        self.openai = openai_client
        self.anthropic = anthropic_client
        self.cost_tracker = cost_tracker
        self.config = config
        self.logger = get_logger(__name__)

    def _should_escalate(
        self,
        *,
        position_size_usd: float,
        signal_confidence: float,
        regime: RegimeType,
        account_equity: float,
    ) -> bool:
        """에스컬레이션 조건: 대형 포지션 / 낮은 신뢰 / 고변동성."""
        if not self.anthropic or not self.anthropic.available:
            return False
        pct = (position_size_usd / account_equity * 100) if account_equity > 0 else 0
        if pct >= getattr(self.config, "escalation_position_size_pct", 3.0):
            return True
        if signal_confidence < getattr(self.config, "escalation_confidence_threshold", 0.7):
            return True
        if regime in {RegimeType.EVENT_RISK, RegimeType.EXPANSION}:
            return True
        return False

    def _build_prompt(self, signal: Any, context: Any, regime: Any, position_size_usd: float) -> str:
        """AI에게 보낼 시장 데이터 프롬프트."""
        side_str = signal.side.value if hasattr(signal.side, "value") else str(signal.side)
        return (
            f"Symbol: {signal.symbol}\n"
            f"Signal: {side_str.upper()}\n"
            f"Entry: {signal.entry_price}, SL: {signal.stop_price}\n"
            f"Strategy: {signal.strategy.value}\n"
            f"Score: {signal.score:.3f}, Confidence: {signal.confidence:.3f}\n"
            f"Regime: {regime.regime.value if hasattr(regime, 'regime') else str(regime)}\n"
            f"Trend Quality: {context.trend_quality_score:.3f}\n"
            f"Position Size: ${position_size_usd:,.0f}\n"
            f"News Blocked: {context.blocked_by_news}\n"
        )

    async def evaluate(
        self,
        *,
        signal: Any,
        context: Any,
        regime: Any,
        position_size_usd: float,
        account_equity: float = 0.0,
    ) -> AIFilterVerdict:
        """AI 필터 평가. 실패 시 approved=True (fallback)."""

        # 비용 한도 체크
        est_cost = getattr(self.config, "estimated_cost_per_openai_call", 0.0006)
        if not self.cost_tracker.can_call(est_cost):
            return AIFilterVerdict(True, 1.0, "skipped", "cost_limit_reached", 0.0, 0.0)

        user_prompt = self._build_prompt(signal, context, regime, position_size_usd)
        timeout = getattr(self.config, "timeout_seconds", 3.0)
        min_conf = getattr(self.config, "min_confidence", 0.6)

        start = time.monotonic()
        try:
            # 에스컬레이션 판단
            escalate = self._should_escalate(
                position_size_usd=position_size_usd,
                signal_confidence=signal.confidence,
                regime=regime.regime if hasattr(regime, "regime") else regime,
                account_equity=account_equity,
            )

            result: TradeFilterAnalysis | None = None
            provider = "openai"
            cost = est_cost

            if escalate and self.anthropic and self.anthropic.available:
                provider = "anthropic"
                cost = getattr(self.config, "estimated_cost_per_anthropic_call", 0.003)
                result = await asyncio.wait_for(
                    self.anthropic.structured_response(
                        system_prompt=_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        schema_model=TradeFilterAnalysis,
                        timeout_seconds=timeout,
                    ),
                    timeout=timeout + 1,
                )
            elif self.openai and self.openai.client:
                result = await asyncio.wait_for(
                    self.openai.structured_response(
                        schema_name="TradeFilterAnalysis",
                        schema_model=TradeFilterAnalysis,
                        system_prompt=_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        max_output_tokens=400,
                    ),
                    timeout=timeout,
                )

            latency = (time.monotonic() - start) * 1000
            self.cost_tracker.record_call(provider, cost, purpose="trade_filter")

            if result is None:
                return AIFilterVerdict(True, 1.0, "fallback", "ai_no_response", cost, latency)

            approved = result.approved and result.confidence >= min_conf
            return AIFilterVerdict(
                approved=approved,
                confidence=result.confidence,
                provider_used=provider,
                reason=result.reasoning_ko or ("low_confidence" if not approved else "approved"),
                cost_usd=cost,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            self.logger.warning("AI 필터 타임아웃 — 규칙만으로 진행", extra={"extra_data": {"timeout": timeout}})
            return AIFilterVerdict(True, 1.0, "fallback", "timeout", 0.0, latency)
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            self.logger.warning("AI 필터 예외 — 규칙만으로 진행", extra={"extra_data": {"error": str(exc)[:200]}})
            return AIFilterVerdict(True, 1.0, "fallback", f"error:{type(exc).__name__}", 0.0, latency)
