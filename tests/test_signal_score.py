"""Signal score tests."""

from __future__ import annotations

from core.enums import RegimeType, Side, StrategyName
from strategy.base import StrategySignal
from strategy.signal_score import SignalScorer


def test_break_retest_gets_trend_weight() -> None:
    """Break-retest should be boosted in trend regime."""

    signal = StrategySignal(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        strategy=StrategyName.BREAK_RETEST,
        side=Side.LONG,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=101.0,
        tp2_price=102.0,
        score=0.6,
        confidence=0.6,
        expected_r=2.0,
        fees_r=0.01,
        slippage_r=0.01,
    )
    scorer = SignalScorer(trend_weight=1.2, raid_weight=1.15)
    breakdown = scorer.score(signal, RegimeType.TREND)
    assert breakdown.final_score > signal.score

