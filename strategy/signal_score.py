"""Signal scoring and regime weighting."""

from __future__ import annotations

from dataclasses import dataclass

from core.enums import RegimeType, StrategyName

from .base import StrategySignal


@dataclass(slots=True)
class ScoreBreakdown:
    """Breakdown of the final signal score."""

    base_score: float
    regime_multiplier: float
    cost_penalty: float
    final_score: float


class SignalScorer:
    """Apply regime-aware score adjustments."""

    def __init__(
        self,
        trend_weight: float = 1.2,
        raid_weight: float = 1.15,
        breakout_weight: float = 1.1,
        momentum_weight: float = 1.12,
    ) -> None:
        self.trend_weight = trend_weight
        self.raid_weight = raid_weight
        self.breakout_weight = breakout_weight
        self.momentum_weight = momentum_weight

    def score(self, signal: StrategySignal, regime: RegimeType) -> ScoreBreakdown:
        """Return the final score after regime weighting and cost penalties."""

        multiplier = 1.0
        if signal.strategy == StrategyName.BREAK_RETEST and regime == RegimeType.TRENDING:
            multiplier = self.trend_weight
        elif signal.strategy == StrategyName.LIQUIDITY_RAID and regime in {RegimeType.RANGING, RegimeType.EVENT_RISK}:
            multiplier = self.raid_weight
        elif signal.strategy == StrategyName.FAIR_VALUE_GAP and regime in {RegimeType.TRENDING, RegimeType.EXPANSION}:
            multiplier = self.breakout_weight
        elif signal.strategy == StrategyName.CHOCH and regime in {RegimeType.TRENDING, RegimeType.EXPANSION}:
            multiplier = self.momentum_weight
        elif signal.strategy == StrategyName.ORDER_BLOCK:
            multiplier = self.raid_weight
        elif regime == RegimeType.DEAD_MARKET:
            multiplier = 0.8

        cost_penalty = signal.fees_r + signal.slippage_r
        final_score = max(0.0, min(1.0, signal.score * multiplier - cost_penalty * 0.15))
        return ScoreBreakdown(
            base_score=signal.score,
            regime_multiplier=multiplier,
            cost_penalty=cost_penalty,
            final_score=final_score,
        )
