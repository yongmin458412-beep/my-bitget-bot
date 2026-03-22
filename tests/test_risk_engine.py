"""Risk engine tests."""

from __future__ import annotations

from core.enums import ProductType, Side, StrategyName
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from risk.risk_engine import RiskEngine
from strategy.base import StrategySignal


def sample_signal() -> StrategySignal:
    """Create a sample signal."""

    return StrategySignal(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        strategy=StrategyName.BREAK_RETEST,
        side=Side.LONG,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=101.0,
        tp2_price=102.0,
        score=0.8,
        confidence=0.7,
        expected_r=2.0,
        fees_r=0.01,
        slippage_r=0.01,
    )


def sample_contract() -> ContractConfig:
    """Create a sample contract."""

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


def sample_capped_contract() -> ContractConfig:
    """Create a contract with an exchange-side leverage cap."""

    return ContractConfig(
        symbol="BTCUSDT",
        product_type=ProductType.USDT_FUTURES,
        base_coin="BTC",
        quote_coin="USDT",
        margin_coin="USDT",
        min_order_size=0.001,
        size_step=0.001,
        price_step=0.1,
        min_leverage=1.0,
        max_leverage=25.0,
    )


def test_risk_engine_approves_valid_trade() -> None:
    """Risk engine should approve a valid setup."""

    engine = RiskEngine(AppSettings())
    approved = engine.approve(
        signal=sample_signal(),
        contract=sample_contract(),
        account_equity=1000.0,
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
    assert approved.quantity > 0


def test_risk_engine_blocks_when_paused() -> None:
    """Risk engine should block while paused."""

    engine = RiskEngine(AppSettings())
    approved = engine.approve(
        signal=sample_signal(),
        contract=sample_contract(),
        account_equity=1000.0,
        open_positions={},
        runtime_metrics={
            "daily_loss_r": 0.0,
            "consecutive_losses": 0,
            "daily_order_count": 0,
            "paused": True,
            "account_drawdown_pct": 0.0,
            "unrealized_loss_pct": 0.0,
        },
        atr_value=1.0,
        news_blocked=False,
        funding_blocked=False,
    )
    assert approved is None


def test_risk_engine_uses_nearest_target_as_execution_anchor() -> None:
    """Execution TP milestones should be based on the nearest structural target."""

    signal = sample_signal()
    signal.target_plan = [
        {"price": 101.2, "reason": "previous_high", "priority": 1},
        {"price": 103.5, "reason": "session_high", "priority": 2},
    ]
    engine = RiskEngine(AppSettings())
    approved = engine.approve(
        signal=signal,
        contract=sample_contract(),
        account_equity=1000.0,
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
    assert approved.target_plan[0]["price"] == 101.2
    assert approved.tp1_price == 100.3
    assert approved.tp2_price == 100.6
    assert approved.tp3_price == 100.9


def test_risk_engine_clamps_leverage_to_contract_max() -> None:
    """Configured strategy leverage should not exceed the contract's max leverage."""

    engine = RiskEngine(AppSettings())
    approved = engine.approve(
        signal=sample_signal(),
        contract=sample_capped_contract(),
        account_equity=1000.0,
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
    assert approved.leverage == 20.0

    signal = sample_signal()
    signal.strategy = StrategyName.MOMENTUM_PULLBACK
    approved = engine.approve(
        signal=signal,
        contract=sample_capped_contract(),
        account_equity=1000.0,
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
    assert approved.leverage == 25.0
    assert any("configured_leverage=30.0" in note for note in approved.notes)
    assert any("applied_leverage=25.0" in note for note in approved.notes)
