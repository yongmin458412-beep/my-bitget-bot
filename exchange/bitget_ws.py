"""Resilient Bitget WebSocket client with auto-reconnect."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from core.enums import ProductType
from core.logger import get_logger
from core.settings import AppSettings
from core.utils import chunks


MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class Subscription:
    """A registered subscription."""

    private: bool
    args: list[dict[str, Any]] = field(default_factory=list)
    handler: MessageHandler | None = None


class BitgetWebSocketClient:
    """Manage public/private Bitget WebSocket connections with reconnect."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        demo: bool = True,
    ) -> None:
        self.settings = settings
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo = demo
        self.logger = get_logger(__name__)
        self._public_task: asyncio.Task[None] | None = None
        self._private_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._subscriptions: dict[str, Subscription] = {}

    def _public_url(self) -> str:
        """Return environment-specific public WS URL."""

        if self.demo:
            return "wss://wspap.bitget.com/v2/ws/public"
        return self.settings.exchange.ws_public_url

    def _private_url(self) -> str:
        """Return environment-specific private WS URL."""

        if self.demo:
            return "wss://wspap.bitget.com/v2/ws/private"
        return self.settings.exchange.ws_private_url

    async def start(self) -> None:
        """Start background reader loops."""

        self._stop_event.clear()
        if self._public_task is None:
            self._public_task = asyncio.create_task(self._run_loop(private=False))
        if any(subscription.private for subscription in self._subscriptions.values()) and self._private_task is None:
            self._private_task = asyncio.create_task(self._run_loop(private=True))

    async def stop(self) -> None:
        """Stop both websocket loops."""

        self._stop_event.set()
        tasks = [task for task in [self._public_task, self._private_task] if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._public_task = None
        self._private_task = None

    async def subscribe(
        self,
        *,
        private: bool,
        key: str,
        args: list[dict[str, Any]],
        handler: MessageHandler,
    ) -> None:
        """Register a subscription for reconnect-safe reuse."""

        self._subscriptions[key] = Subscription(private=private, args=args, handler=handler)
        await self.start()
        if private and self._private_task is None:
            self._private_task = asyncio.create_task(self._run_loop(private=True))

    async def subscribe_tickers(
        self,
        product_type: ProductType,
        symbols: list[str],
        handler: MessageHandler,
    ) -> None:
        """Subscribe to ticker updates for the given symbols."""

        args = [
            {"instType": product_type.value, "channel": "ticker", "instId": symbol.upper()}
            for symbol in symbols
        ]
        await self.subscribe(
            private=False,
            key=f"ticker:{product_type.value}:{','.join(symbols)}",
            args=args,
            handler=handler,
        )

    async def subscribe_orderbooks(
        self,
        product_type: ProductType,
        symbols: list[str],
        handler: MessageHandler,
        depth_channel: str = "books5",
    ) -> None:
        """Subscribe to top-of-book depth updates."""

        args = [
            {"instType": product_type.value, "channel": depth_channel, "instId": symbol.upper()}
            for symbol in symbols
        ]
        await self.subscribe(
            private=False,
            key=f"depth:{product_type.value}:{depth_channel}:{','.join(symbols)}",
            args=args,
            handler=handler,
        )

    async def subscribe_candles(
        self,
        product_type: ProductType,
        symbol: str,
        timeframe: str,
        handler: MessageHandler,
    ) -> None:
        """Subscribe to candle updates."""

        channel = f"candle{timeframe}"
        await self.subscribe(
            private=False,
            key=f"candle:{product_type.value}:{symbol}:{channel}",
            args=[{"instType": product_type.value, "channel": channel, "instId": symbol.upper()}],
            handler=handler,
        )

    async def subscribe_private_channel(
        self,
        product_type: ProductType,
        channel: str,
        handler: MessageHandler,
    ) -> None:
        """Subscribe to a private channel such as orders or positions."""

        await self.subscribe(
            private=True,
            key=f"private:{product_type.value}:{channel}",
            args=[{"instType": product_type.value, "channel": channel, "instId": "default"}],
            handler=handler,
        )

    async def _run_loop(self, *, private: bool) -> None:
        """Maintain a live websocket connection and reconnect on failures."""

        url = self._private_url() if private else self._public_url()
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    if private:
                        await self._login(ws)
                    await self._resubscribe(ws, private=private)
                    backoff = 1.0
                    async for raw_message in ws:
                        await self._handle_message(raw_message, private=private)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Bitget WS reconnect", extra={"extra_data": {"private": private, "error": str(exc)}})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _login(self, ws: ClientConnection) -> None:
        """Authenticate the private websocket session."""

        timestamp = str(int(time.time() * 1000))
        prehash = f"{timestamp}GET/user/verify"
        sign = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        payload = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": sign,
                }
            ],
        }
        await ws.send(json.dumps(payload))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
        except TimeoutError:
            raise RuntimeError("Bitget WS login timeout (10s)")
        response = json.loads(raw)
        if response.get("event") == "error":
            raise RuntimeError(f"Bitget WS login failed: {response}")

    async def _resubscribe(self, ws: ClientConnection, *, private: bool) -> None:
        """Re-send subscriptions after reconnect."""

        subscriptions = [item for item in self._subscriptions.values() if item.private == private]
        if not subscriptions:
            return
        args: list[dict[str, Any]] = []
        for subscription in subscriptions:
            args.extend(subscription.args)
        for batch in chunks(args, 10):
            await ws.send(json.dumps({"op": "subscribe", "args": batch}))
            await asyncio.sleep(0.05)

    async def _handle_message(self, raw_message: str, *, private: bool) -> None:
        """Dispatch websocket messages to the matching handler."""

        payload = json.loads(raw_message)
        if payload.get("event") in {"subscribe", "pong"}:
            return
        arg = payload.get("arg") or {}
        inst_type = arg.get("instType", "")
        channel = arg.get("channel", "")
        inst_id = arg.get("instId", "")
        for subscription in self._subscriptions.values():
            if subscription.private != private:
                continue
            for sub_arg in subscription.args:
                if (
                    sub_arg.get("instType") == inst_type
                    and sub_arg.get("channel") == channel
                    and sub_arg.get("instId") == inst_id
                ):
                    if subscription.handler:
                        try:
                            await subscription.handler(payload)
                        except Exception as exc:  # noqa: BLE001
                            self.logger.error(
                                "WS handler error",
                                extra={"extra_data": {"channel": channel, "instId": inst_id, "error": str(exc)}},
                                exc_info=True,
                            )
                    return

