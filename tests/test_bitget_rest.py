"""Tests for Bitget REST client order handling."""

from __future__ import annotations

import asyncio
from typing import Any

from core.enums import OrderType, ProductType
from core.settings import AppSettings, EnvSecrets
from exchange.bitget_models import OrderRequest
from exchange.bitget_rest import BitgetAPIError, BitgetRestClient


def build_client() -> BitgetRestClient:
    """Create a small test client."""

    settings = AppSettings(secrets=EnvSecrets(bitget_demo_api_key="k", bitget_demo_api_secret="s", bitget_demo_api_passphrase="p"))
    return BitgetRestClient(
        settings,
        api_key="k",
        api_secret="s",
        passphrase="p",
        demo=True,
    )


def test_place_order_retries_without_trade_side_on_40774() -> None:
    """One-way accounts should retry without tradeSide when Bitget rejects hedge payloads."""

    async def scenario() -> None:
        client = build_client()
        calls: list[dict[str, Any]] = []

        async def fake_request(
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            payload: dict[str, Any] | None = None,
            auth: bool = False,
        ) -> dict[str, Any]:
            calls.append({"method": method, "path": path, "payload": dict(payload or {})})
            if len(calls) == 1:
                raise BitgetAPIError(
                    "Bitget HTTP 400 40774: one-way position mode",
                    code="40774",
                    status_code=400,
                    payload={"code": "40774", "msg": "one-way position mode"},
                )
            return {"code": "00000", "msg": "success", "data": {"orderId": "1", "clientOid": "abc"}}

        client._request = fake_request  # type: ignore[method-assign]
        order = OrderRequest(
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            margin_coin="USDT",
            side="buy",
            trade_side="open",
            order_type=OrderType.MARKET,
            quantity=0.001,
        )

        result = await client.place_order(order)

        assert result.exchange_order_id == "1"
        assert len(calls) == 2
        assert calls[0]["payload"]["tradeSide"] == "open"
        assert "tradeSide" not in calls[1]["payload"]
        await client.aclose()

    asyncio.run(scenario())


def test_place_order_does_not_retry_unrelated_bitget_error() -> None:
    """Non-position-mode errors should propagate unchanged."""

    async def scenario() -> None:
        client = build_client()

        async def fake_request(
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            payload: dict[str, Any] | None = None,
            auth: bool = False,
        ) -> dict[str, Any]:
            raise BitgetAPIError(
                "Bitget HTTP 400 40762: size error",
                code="40762",
                status_code=400,
                payload={"code": "40762", "msg": "size error"},
            )

        client._request = fake_request  # type: ignore[method-assign]
        order = OrderRequest(
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            margin_coin="USDT",
            side="buy",
            trade_side="open",
            order_type=OrderType.MARKET,
            quantity=0.001,
        )

        try:
            await client.place_order(order)
        except BitgetAPIError as exc:
            assert exc.code == "40762"
        else:
            raise AssertionError("BitgetAPIError was expected")
        await client.aclose()

    asyncio.run(scenario())


def test_get_pending_orders_handles_missing_entrusted_list() -> None:
    """Pending-order parsing should tolerate null entrusted lists."""

    async def scenario() -> None:
        client = build_client()

        async def fake_request(
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            payload: dict[str, Any] | None = None,
            auth: bool = False,
        ) -> dict[str, Any]:
            return {"code": "00000", "msg": "success", "data": {"entrustedList": None}}

        client._request = fake_request  # type: ignore[method-assign]
        rows = await client.get_pending_orders(ProductType.USDT_FUTURES)
        assert rows == []
        await client.aclose()

    asyncio.run(scenario())
