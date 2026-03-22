"""Extended OrderManager tests: orphan detection, cancel failure handling."""

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


class DummyRestRaisesOnCancel:
    """REST stub that fails on cancel."""

    async def cancel_order(self, *_a, **_kw) -> None:
        raise RuntimeError("exchange unavailable")

    async def dry_run_validate(self, *_a, **_kw) -> list[str]:
        return []


class DummyExchangeRaisesOnCancel:
    mode = TradingMode.DEMO
    rest: DummyRestRaisesOnCancel

    def __init__(self) -> None:
        self.rest = DummyRestRaisesOnCancel()
        self.placed_orders: list = []
        self._pending_orders: list = []

    async def place_order(self, order) -> OrderResult:
        self.placed_orders.append(order)
        return OrderResult(
            client_order_id="fb-order",
            exchange_order_id="ex-fb",
            status=OrderStatus.NEW,
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            raw={},
        )

    async def get_pending_orders(self) -> list:
        return self._pending_orders


class DummyRestOk:
    cancelled: list[str]

    def __init__(self) -> None:
        self.cancelled = []

    async def cancel_order(self, *_a, client_order_id: str, **_kw) -> None:
        self.cancelled.append(client_order_id)

    async def dry_run_validate(self, *_a, **_kw) -> list[str]:
        return []


class DummyExchangeOk:
    mode = TradingMode.DEMO

    def __init__(self) -> None:
        self.rest = DummyRestOk()
        self.placed_orders: list = []
        self._pending_orders: list = []

    async def place_order(self, order) -> OrderResult:
        self.placed_orders.append(order)
        return OrderResult(
            client_order_id="fb-ok",
            exchange_order_id="ex-ok",
            status=OrderStatus.NEW,
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            raw={},
        )

    async def get_pending_orders(self) -> list:
        return self._pending_orders


def _contract() -> ContractConfig:
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


def test_cancel_if_stale_keeps_state_on_cancel_failure(tmp_path: Path) -> None:
    """If cancel raises, the order should remain in local state (not silently removed)."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.execution.maker_timeout_seconds = 1
        persistence = SQLitePersistence(tmp_path / "orders.sqlite3")
        state_store = StateStore(tmp_path / "runtime.json", persistence)
        exchange = DummyExchangeRaisesOnCancel()
        manager = OrderManager(settings, exchange, persistence, state_store)

        from execution.order_manager import PendingOrder
        manager._pending_by_signal["sig-001"] = PendingOrder(
            client_order_id="order-stale",
            symbol="BTCUSDT",
            created_at=datetime.now(tz=UTC) - timedelta(seconds=10),
            route_notes=[],
        )
        state_store.update_order("order-stale", {"symbol": "BTCUSDT"})

        result = await manager.cancel_if_stale(
            contract=_contract(),
            signal_id="sig-001",
            margin_coin="USDT",
            max_age_seconds=1,
        )

        # Should return False (cancel failed)
        assert result is False
        # Order should STILL be in state
        assert "order-stale" in state_store.state.open_orders
        assert "sig-001" in manager._pending_by_signal

    asyncio.run(scenario())


def test_cancel_if_stale_cleans_up_on_success(tmp_path: Path) -> None:
    """Successful cancel should remove order from state and signal tracking."""

    async def scenario() -> None:
        settings = AppSettings()
        settings.execution.maker_timeout_seconds = 1
        persistence = SQLitePersistence(tmp_path / "orders.sqlite3")
        state_store = StateStore(tmp_path / "runtime.json", persistence)
        exchange = DummyExchangeOk()
        manager = OrderManager(settings, exchange, persistence, state_store)

        from execution.order_manager import PendingOrder
        manager._pending_by_signal["sig-002"] = PendingOrder(
            client_order_id="order-ok",
            symbol="BTCUSDT",
            created_at=datetime.now(tz=UTC) - timedelta(seconds=10),
            route_notes=[],
        )
        state_store.update_order("order-ok", {"symbol": "BTCUSDT"})

        result = await manager.cancel_if_stale(
            contract=_contract(),
            signal_id="sig-002",
            margin_coin="USDT",
            max_age_seconds=1,
        )

        assert result is True
        assert "order-ok" not in state_store.state.open_orders
        assert "sig-002" not in manager._pending_by_signal

    asyncio.run(scenario())


def test_sync_pending_orders_detects_orphans(tmp_path: Path) -> None:
    """Orders in local state but not on exchange should be cleaned up."""

    async def scenario() -> None:
        settings = AppSettings()
        persistence = SQLitePersistence(tmp_path / "orders.sqlite3")
        state_store = StateStore(tmp_path / "runtime.json", persistence)
        exchange = DummyExchangeOk()
        manager = OrderManager(settings, exchange, persistence, state_store)

        # Put two orders in local state
        state_store.update_order("local-only-1", {"symbol": "BTCUSDT", "signal_id": "sig-local-1"})
        state_store.update_order("local-only-2", {"symbol": "ETHUSDT", "signal_id": ""})

        # Exchange returns zero pending orders
        exchange._pending_orders = []

        await manager.sync_pending_orders()

        # Both orphans should be removed
        assert "local-only-1" not in state_store.state.open_orders
        assert "local-only-2" not in state_store.state.open_orders

    asyncio.run(scenario())


def test_sync_pending_orders_preserves_live_orders(tmp_path: Path) -> None:
    """Orders still on exchange should be kept and updated in local state."""

    async def scenario() -> None:
        settings = AppSettings()
        persistence = SQLitePersistence(tmp_path / "orders.sqlite3")
        state_store = StateStore(tmp_path / "runtime.json", persistence)
        exchange = DummyExchangeOk()
        manager = OrderManager(settings, exchange, persistence, state_store)

        # Exchange returns one live order
        exchange._pending_orders = [{
            "clientOid": "live-order-1",
            "orderId": "ex-live-1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "orderType": "limit",
            "state": "open",
            "price": "100.0",
            "size": "0.01",
            "baseVolume": "0.0",
            "priceAvg": None,
            "reduceOnly": "no",
        }]

        await manager.sync_pending_orders()

        # Live order should be in state
        assert "live-order-1" in state_store.state.open_orders

    asyncio.run(scenario())
