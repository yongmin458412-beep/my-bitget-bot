"""Tests for order-manager fallback behavior."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.enums import OrderStatus, OrderType, ProductType, TradingMode
from core.persistence import SQLitePersistence
from core.settings import AppSettings
from core.state_store import StateStore
from exchange.bitget_models import ContractConfig, OrderResult
from execution.order_manager import OrderManager


class DummyRest:
    """Minimal REST surface for stale-order replacement tests."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_order(
        self,
        product_type: ProductType,
        symbol: str,
        margin_coin: str,
        *,
        client_order_id: str,
    ) -> None:
        """Record a cancel request."""

        self.cancelled.append(client_order_id)

    async def dry_run_validate(self, *_: object, **__: object) -> list[str]:
        """Pretend validation succeeds."""

        return []


class DummyExchange:
    """Minimal exchange double for OrderManager tests."""

    def __init__(self) -> None:
        self.mode = TradingMode.DEMO
        self.rest = DummyRest()
        self.placed_orders: list[object] = []

    async def place_order(self, order: object) -> OrderResult:
        """Record the replacement market order."""

        self.placed_orders.append(order)
        return OrderResult(
            client_order_id="entryfb-BTCUSDT-test",
            exchange_order_id="ex-1",
            status=OrderStatus.NEW,
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            raw={},
        )


def _sample_contract() -> ContractConfig:
    """Build a minimal contract definition."""

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


def test_replace_stale_limit_with_market_replaces_entry_order(tmp_path: Path) -> None:
    """Old maker orders should be canceled and retried as market orders."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.execution.maker_timeout_seconds = 1
        persistence = SQLitePersistence(tmp_path / "orders.sqlite3")
        state_store = StateStore(tmp_path / "runtime.json", persistence)
        exchange = DummyExchange()
        manager = OrderManager(settings, exchange, persistence, state_store)

        stale_payload = {
            "client_order_id": "entry-old",
            "exchange_order_id": "old-ex",
            "signal_id": "signal-123456",
            "symbol": "BTCUSDT",
            "mode": "DEMO",
            "side": "long",
            "order_type": OrderType.LIMIT.value,
            "status": "open",
            "price": 100.0,
            "quantity": 0.01,
            "stop_price": 98.0,
            "tp1_price": 102.0,
            "tp2_price": 104.0,
            "tp3_price": 106.0,
            "strategy": "break_retest",
            "created_at": (datetime.now(tz=UTC) - timedelta(seconds=5)).isoformat(timespec="seconds"),
            "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        state_store.update_order("entry-old", stale_payload)

        result = await manager.replace_stale_limit_with_market(
            contract=_sample_contract(),
            order_payload=stale_payload,
        )

        assert result is not None
        assert exchange.rest.cancelled == ["entry-old"]
        assert len(exchange.placed_orders) == 1
        placed_order = exchange.placed_orders[0]
        assert getattr(placed_order, "order_type") == OrderType.MARKET
        assert "entry-old" not in state_store.state.open_orders
        replacement = state_store.state.open_orders[result.client_order_id]
        assert replacement["reason"] == "maker_timeout_fallback"
        assert replacement["order_type"] == OrderType.MARKET.value

    asyncio.run(scenario())
