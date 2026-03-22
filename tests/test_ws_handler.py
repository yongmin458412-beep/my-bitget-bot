"""Tests for BitgetWebSocketClient handler safety and login timeout."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.settings import AppSettings
from exchange.bitget_ws import BitgetWebSocketClient


def _make_client() -> BitgetWebSocketClient:
    settings = AppSettings()
    return BitgetWebSocketClient(
        settings,
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_pass",
        demo=True,
    )


def test_handler_exception_does_not_propagate() -> None:
    """Exception in message handler must not crash the WS loop."""

    async def scenario() -> None:
        client = _make_client()

        call_count = 0

        async def bad_handler(payload: dict) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("simulated handler crash")

        client._subscriptions["test"] = MagicMock(
            private=False,
            args=[{"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"}],
            handler=bad_handler,
        )

        message = json.dumps({
            "arg": {"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"},
            "data": [{"last": "50000"}],
        })

        # Should not raise
        await client._handle_message(message, private=False)
        assert call_count == 1  # Handler was called

    asyncio.run(scenario())


def test_subscribe_and_pong_messages_ignored() -> None:
    """Subscribe-ack and pong frames should be silently discarded."""

    async def scenario() -> None:
        client = _make_client()

        called = False

        async def handler(payload: dict) -> None:
            nonlocal called
            called = True

        client._subscriptions["x"] = MagicMock(
            private=False,
            args=[{"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"}],
            handler=handler,
        )

        for event in ("subscribe", "pong"):
            await client._handle_message(json.dumps({"event": event}), private=False)

        assert not called

    asyncio.run(scenario())


def test_login_timeout_raises() -> None:
    """_login should raise RuntimeError if exchange takes >10s to respond."""

    async def scenario() -> None:
        client = _make_client()

        async def slow_recv():
            await asyncio.sleep(15)
            return json.dumps({"event": "login", "code": "0"})

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = slow_recv

        with pytest.raises(RuntimeError, match="login timeout"):
            await asyncio.wait_for(client._login(mock_ws), timeout=12)

    asyncio.run(scenario())


def test_login_error_response_raises() -> None:
    """_login should raise RuntimeError when exchange returns event=error."""

    async def scenario() -> None:
        client = _make_client()

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "event": "error",
            "code": "30001",
            "msg": "invalid sign",
        }))

        with pytest.raises(RuntimeError, match="login failed"):
            await client._login(mock_ws)

    asyncio.run(scenario())


def test_multiple_handlers_only_first_matching_called() -> None:
    """Only the first matching subscription handler should be dispatched."""

    async def scenario() -> None:
        client = _make_client()

        counts = {"a": 0, "b": 0}

        async def handler_a(payload: dict) -> None:
            counts["a"] += 1

        async def handler_b(payload: dict) -> None:
            counts["b"] += 1

        # Two subscriptions for same channel
        client._subscriptions["sub_a"] = MagicMock(
            private=False,
            args=[{"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"}],
            handler=handler_a,
        )
        client._subscriptions["sub_b"] = MagicMock(
            private=False,
            args=[{"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"}],
            handler=handler_b,
        )

        message = json.dumps({
            "arg": {"instType": "USDT-FUTURES", "channel": "ticker", "instId": "BTCUSDT"},
            "data": [{"last": "50000"}],
        })

        await client._handle_message(message, private=False)

        # First match wins; second should not be called
        total = counts["a"] + counts["b"]
        assert total == 1

    asyncio.run(scenario())
