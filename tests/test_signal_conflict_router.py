"""Signal dedupe and conflict routing tests."""

from __future__ import annotations

from core.enums import RegimeType, Side, StrategyName
from core.settings import StrategyRouterConfig
from market.market_regime import RegimeSnapshot
from strategy.base import StrategySignal
from strategy.conflict_router import SUPPORTED_STRATEGY_GROUPS, choose_primary_signal


def _signal(strategy: StrategyName, side: Side, *, score: float, entry: float = 100.0, stop: float = 99.0, tp1: float = 101.2) -> StrategySignal:
    """Build a compact synthetic signal."""

    return StrategySignal(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        strategy=strategy,
        side=side,
        entry_price=entry,
        stop_price=stop,
        tp1_price=tp1,
        tp2_price=tp1 + 1.0,
        score=score,
        confidence=0.7,
        expected_r=2.0,
        fees_r=0.01,
        slippage_r=0.01,
        tags=["breakout", "trend"] if side == Side.LONG else ["reclaim", "range"],
        target_plan=[
            {"price": tp1, "reason": "previous_high" if side == Side.LONG else "previous_low", "priority": 1, "inside_range_middle": False},
            {"price": tp1 + 1.0 if side == Side.LONG else tp1 - 1.0, "reason": "session_high", "priority": 2, "inside_range_middle": False},
        ],
    )


def _regime(regime: RegimeType) -> RegimeSnapshot:
    """Build a minimal regime snapshot for routing tests."""

    return RegimeSnapshot(
        regime=regime,
        adx_value=25.0,
        atr_percentile=0.6,
        volume_percentile=0.6,
        trend_quality_score=0.5,
        above_vwap=True,
        allowed_strategies=[],
        notes=[],
    )


def test_vincent_and_momentum_same_story_are_deduped() -> None:
    """Vincent and Momentum signals on the same structure should collapse to one primary signal."""

    router = StrategyRouterConfig()
    vincent = _signal(StrategyName.BREAK_RETEST, Side.LONG, score=0.83)
    momentum = _signal(StrategyName.MOMENTUM_PULLBACK, Side.LONG, score=0.79)

    decision = choose_primary_signal([vincent, momentum], regime=_regime(RegimeType.TRENDING), router=router)

    assert decision.chosen is not None
    assert decision.chosen.strategy == StrategyName.BREAK_RETEST
    assert any(item.strategy == StrategyName.MOMENTUM_PULLBACK for item in decision.rejected)


def test_opposite_reclaim_and_session_breakout_no_trade_in_ranging() -> None:
    """In ranging regime, opposite breakout/reclaim signals should result in no trade."""

    router = StrategyRouterConfig()
    reclaim = _signal(StrategyName.LIQUIDITY_RAID, Side.LONG, score=0.82, entry=100.0, stop=99.2, tp1=101.1)
    breakout = _signal(StrategyName.SESSION_BREAKOUT, Side.SHORT, score=0.88, entry=100.1, stop=101.0, tp1=99.0)

    decision = choose_primary_signal([reclaim, breakout], regime=_regime(RegimeType.RANGING), router=router)

    assert decision.chosen is None
    assert decision.decision == "no_trade_opposite_conflict"


def test_same_direction_duplicate_signals_merge() -> None:
    """Same-direction overlapping signals should merge down to one chosen signal."""

    router = StrategyRouterConfig()
    left = _signal(StrategyName.BREAK_RETEST, Side.LONG, score=0.81, entry=100.0, stop=99.0, tp1=101.3)
    right = _signal(StrategyName.SESSION_BREAKOUT, Side.LONG, score=0.78, entry=100.05, stop=99.02, tp1=101.28)

    decision = choose_primary_signal([left, right], regime=_regime(RegimeType.EXPANSION), router=router)

    assert decision.chosen is not None
    assert decision.chosen.side == Side.LONG
    assert len(decision.rejected) == 1


def test_only_supported_strategy_groups_are_registered() -> None:
    """The router should expose only the final four supported strategy groups."""

    assert SUPPORTED_STRATEGY_GROUPS == (
        StrategyName.BREAK_RETEST,
        StrategyName.LIQUIDITY_RAID,
        StrategyName.SESSION_BREAKOUT,
        StrategyName.MOMENTUM_PULLBACK,
    )
