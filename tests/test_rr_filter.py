"""Structural RR filter tests."""

from __future__ import annotations

import pandas as pd

from core.enums import RegimeType, Side
from strategy.base import SignalContext
from strategy.rr_filter import compute_structural_rr, evaluate_trade_viability


def _context() -> SignalContext:
    index = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100 + i for i in range(20)],
            "high": [100.5 + i for i in range(20)],
            "low": [99.5 + i for i in range(20)],
            "close": [100.2 + i for i in range(20)],
            "volume": [1000 + i for i in range(20)],
            "quote_volume": [100000 + i * 10 for i in range(20)],
        },
        index=index,
    )
    return SignalContext(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame, "5m": frame, "15m": frame, "1H": frame},
        levels={},
        level_tags={},
        regime=RegimeType.TREND,
        regime_notes=[],
        ticker={"last_price": 100.0, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.5, "avg_win_r": 1.8, "avg_loss_r": 1.0},
    )


def test_compute_structural_rr_handles_long_and_short() -> None:
    """RR calculation should be symmetric for long and short examples."""

    assert compute_structural_rr(100.0, 99.0, 102.0, Side.LONG) == 2.0
    assert compute_structural_rr(100.0, 101.0, 98.0, Side.SHORT) == 2.0


def test_evaluate_trade_viability_rejects_insufficient_targets() -> None:
    """A setup with no structural targets should be rejected."""

    result = evaluate_trade_viability(
        _context(),
        99.0,
        [],
        0.02,
        0.02,
        entry_price=100.0,
        side=Side.LONG,
        strategy_name="break_retest",
        min_rr_to_tp1_break_retest=1.3,
        preferred_rr_to_tp2_break_retest=1.8,
        min_rr_to_tp1_liquidity_raid=1.5,
        preferred_rr_to_tp2_liquidity_raid=2.0,
        min_rr_to_tp1_session_breakout=1.35,
        preferred_rr_to_tp2_session_breakout=2.0,
        min_rr_to_tp1_momentum_pullback=1.2,
        preferred_rr_to_tp2_momentum_pullback=1.7,
    )
    assert result.approved is False
    assert result.reject_reason == "insufficient_structural_targets"


def test_evaluate_trade_viability_rejects_low_rr() -> None:
    """A structure that cannot clear the minimum RR filter should be rejected."""

    result = evaluate_trade_viability(
        _context(),
        99.0,
        [{"price": 100.6, "reason": "nearby_high", "priority": 1, "inside_range_middle": False}],
        0.02,
        0.02,
        entry_price=100.0,
        side=Side.LONG,
        strategy_name="break_retest",
        min_rr_to_tp1_break_retest=1.3,
        preferred_rr_to_tp2_break_retest=1.8,
        min_rr_to_tp1_liquidity_raid=1.5,
        preferred_rr_to_tp2_liquidity_raid=2.0,
        min_rr_to_tp1_session_breakout=1.35,
        preferred_rr_to_tp2_session_breakout=2.0,
        min_rr_to_tp1_momentum_pullback=1.2,
        preferred_rr_to_tp2_momentum_pullback=1.7,
    )
    assert result.approved is False
    assert result.reject_reason == "min_rr_to_tp1_failed"


def test_evaluate_trade_viability_uses_session_breakout_thresholds() -> None:
    """Session breakout should use its own RR gate instead of Vincent defaults."""

    result = evaluate_trade_viability(
        _context(),
        99.0,
        [{"price": 101.32, "reason": "session_high", "priority": 1, "inside_range_middle": False}],
        0.0,
        0.0,
        entry_price=100.0,
        side=Side.LONG,
        strategy_name="session_breakout",
        min_rr_to_tp1_break_retest=1.3,
        preferred_rr_to_tp2_break_retest=1.8,
        min_rr_to_tp1_liquidity_raid=1.5,
        preferred_rr_to_tp2_liquidity_raid=2.0,
        min_rr_to_tp1_session_breakout=1.35,
        preferred_rr_to_tp2_session_breakout=2.0,
        min_rr_to_tp1_momentum_pullback=1.2,
        preferred_rr_to_tp2_momentum_pullback=1.7,
    )
    assert result.approved is False
    assert result.reject_reason == "min_rr_to_tp1_failed"


def test_evaluate_trade_viability_uses_momentum_thresholds() -> None:
    """Momentum pullback should allow slightly lower TP1 RR than Vincent."""

    result = evaluate_trade_viability(
        _context(),
        99.0,
        [{"price": 101.25, "reason": "recent_swing_high", "priority": 1, "inside_range_middle": False}],
        0.0,
        0.0,
        entry_price=100.0,
        side=Side.LONG,
        strategy_name="momentum_pullback",
        min_rr_to_tp1_break_retest=1.3,
        preferred_rr_to_tp2_break_retest=1.8,
        min_rr_to_tp1_liquidity_raid=1.5,
        preferred_rr_to_tp2_liquidity_raid=2.0,
        min_rr_to_tp1_session_breakout=1.35,
        preferred_rr_to_tp2_session_breakout=2.0,
        min_rr_to_tp1_momentum_pullback=1.2,
        preferred_rr_to_tp2_momentum_pullback=1.7,
    )
    assert result.approved is True
