"""SLTP manager tests for structural multi-target exits."""

from __future__ import annotations

import pytest

from core.enums import ProductType, Side, StrategyName
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from execution.sltp_manager import SLTPManager
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
    """Create a structural signal with three targets."""

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
        stop_reason="retest_low_atr_buffer",
        target_plan=[
            {"price": 102.0, "reason": "previous_high", "priority": 1},
            {"price": 104.0, "reason": "session_high", "priority": 2},
            {"price": 106.0, "reason": "range_opposite_side", "priority": 3},
        ],
        rr_to_tp1=2.0,
        rr_to_tp2=4.0,
        rr_to_best_target=6.0,
    )


def test_sltp_manager_runs_tp1_tp2_tp3_sequence() -> None:
    """25/50/75/100% milestones should be handled as 50/25/15/잔량 청산."""

    settings = AppSettings()
    engine = RiskEngine(settings)
    approved = engine.approve(
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

    manager = SLTPManager(settings)
    manager.register(approved)

    final_target = approved.target_plan[0]["price"]
    milestone_25 = approved.signal.entry_price + (final_target - approved.signal.entry_price) * 0.25
    milestone_50 = approved.signal.entry_price + (final_target - approved.signal.entry_price) * 0.50
    milestone_75 = approved.signal.entry_price + (final_target - approved.signal.entry_price) * 0.75

    actions_tp1 = manager.evaluate_price("BTCUSDT", milestone_25 + 0.01)
    assert [item["action"] for item in actions_tp1] == ["partial_tp1", "move_stop"]
    assert actions_tp1[0]["quantity"] == approved.quantity * settings.risk.tp1_partial_close_pct
    assert actions_tp1[0].get("pending_fill") is True
    assert actions_tp1[1]["new_stop_price"] > approved.signal.entry_price
    # Simulate fill confirmation before next evaluation
    manager.confirm_partial_fill("BTCUSDT", actions_tp1[0]["quantity"])

    actions_tp2 = manager.evaluate_price("BTCUSDT", milestone_50 + 0.1)
    assert [item["action"] for item in actions_tp2] == ["partial_tp2"]
    assert actions_tp2[0]["quantity"] == approved.quantity * settings.risk.tp2_partial_close_pct
    manager.confirm_partial_fill("BTCUSDT", actions_tp2[0]["quantity"])

    actions_tp3 = manager.evaluate_price("BTCUSDT", milestone_75 + 0.1)
    assert [item["action"] for item in actions_tp3] == ["partial_tp3"]
    assert actions_tp3[0]["quantity"] == pytest.approx(approved.quantity * settings.risk.tp3_partial_close_pct)
    manager.confirm_partial_fill("BTCUSDT", actions_tp3[0]["quantity"])

    actions_final = manager.evaluate_price("BTCUSDT", final_target + 0.1)
    assert [item["action"] for item in actions_final] == ["final_target_exit"]
    remaining_after_scale_out = approved.quantity * (
        1
        - settings.risk.tp1_partial_close_pct
        - settings.risk.tp2_partial_close_pct
        - settings.risk.tp3_partial_close_pct
    )
    assert actions_final[0]["quantity"] == pytest.approx(remaining_after_scale_out)
