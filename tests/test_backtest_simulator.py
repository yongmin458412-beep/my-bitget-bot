"""Backtest simulator tests for structural multi-target logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from core.enums import ProductType, Side, StrategyName
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from backtest.simulator import TradeSimulator
from risk.risk_engine import RiskEngine
from strategy.base import StrategySignal


def _sample_contract() -> ContractConfig:
    """Create a minimal contract config."""

    return ContractConfig(
        symbol="BTCUSDT",
        product_type=ProductType.USDT_FUTURES,
        base_coin="BTC",
        quote_coin="USDT",
        margin_coin="USDT",
        min_order_size=0.001,
        size_step=0.001,
        price_step=0.1,
    )


def _sample_signal() -> StrategySignal:
    """Create a structural trade signal."""

    return StrategySignal(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        strategy=StrategyName.BREAK_RETEST,
        side=Side.LONG,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=102.0,
        tp2_price=104.0,
        tp3_price=106.0,
        score=0.8,
        confidence=0.7,
        expected_r=4.0,
        fees_r=0.01,
        slippage_r=0.01,
    )


def test_trade_simulator_handles_tp1_tp2_tp3_flow() -> None:
    """Backtest simulator should realize partials before the runner exit."""

    settings = AppSettings()
    approved = RiskEngine(settings).approve(
        signal=_sample_signal(),
        contract=_sample_contract(),
        account_equity=10_000.0,
        open_positions={},
        runtime_metrics={
            "daily_loss_r": 0.0,
            "consecutive_losses": 0,
            "daily_order_count": 0,
            "paused": False,
            "account_drawdown_pct": 0.0,
            "unrealized_loss_pct": 0.0,
        },
        atr_value=1.0,
        news_blocked=False,
        funding_blocked=False,
    )
    assert approved is not None

    simulator = TradeSimulator(
        fee_bps=settings.ev.estimated_maker_fee_bps,
        slippage_bps=settings.ev.slippage_bps,
        tp1_partial_close_pct=settings.risk.tp1_partial_close_pct,
        tp2_partial_close_pct=settings.risk.tp2_partial_close_pct,
        move_sl_to_be=settings.risk.move_sl_to_be_after_tp1,
        be_offset_r=settings.risk.be_offset_r,
        max_hold_minutes=settings.risk.max_position_hold_minutes,
    )
    simulator.open_trade(approved, datetime(2026, 3, 18, 0, 0, tzinfo=UTC))

    simulator.on_bar(pd.Series({"high": 102.5, "low": 100.2, "close": 102.0}, name=pd.Timestamp("2026-03-18T00:03:00Z")))
    assert simulator.active_trade is not None
    assert simulator.active_trade.tp1_done is True

    simulator.on_bar(pd.Series({"high": 104.5, "low": 102.3, "close": 104.0}, name=pd.Timestamp("2026-03-18T00:06:00Z")))
    assert simulator.active_trade is not None
    assert simulator.active_trade.tp2_done is True

    simulator.on_bar(pd.Series({"high": 106.5, "low": 104.5, "close": 106.0}, name=pd.Timestamp("2026-03-18T00:09:00Z")))
    assert simulator.active_trade is None
    assert len(simulator.closed_trades) == 1
    assert simulator.closed_trades[0].exit_reason == "tp3"
    assert simulator.closed_trades[0].pnl_usdt > 0
