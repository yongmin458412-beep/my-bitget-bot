"""Order router tests."""

from __future__ import annotations

from core.enums import ProductType, Side, StrategyName
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from execution.router import OrderRouter
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
    """Create a minimal strategy signal."""

    return StrategySignal(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        strategy=StrategyName.BREAK_RETEST,
        side=Side.LONG,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=101.5,
        tp2_price=103.0,
        tp3_price=105.0,
        score=0.8,
        confidence=0.7,
        expected_r=3.0,
        fees_r=0.01,
        slippage_r=0.01,
    )


def test_router_keeps_tp_local_when_exchange_plan_orders_disabled() -> None:
    """Preset TP/SL should stay empty unless exchange plan orders are enabled."""

    settings = AppSettings()
    settings.execution.use_exchange_plan_orders = False
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

    decision = OrderRouter(settings).build_entry_order(
        approved_trade=approved,
        contract=_sample_contract(),
        best_bid=99.9,
        best_ask=100.1,
        spread_bps=4.0,
        volatility_score=0.2,
    )

    assert decision.order.preset_take_profit_price is None
    assert decision.order.preset_stop_loss_price is None

