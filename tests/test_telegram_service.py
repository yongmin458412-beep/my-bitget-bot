"""Tests for Telegram bot service lifecycle and diagnostics."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from core.settings import AppSettings, EnvSecrets, TelegramConfig
from telegram_bot.bot import TelegramBotService
from telegram_bot.commands import TelegramCommandRouter


class DummyProvider:
    """Minimal provider used by Telegram service tests."""

    async def get_status_payload(self) -> dict[str, Any]:
        return {}

    async def get_positions_payload(self) -> list[dict[str, Any]]:
        return []

    async def get_balance_payload(self) -> dict[str, Any]:
        return {}

    async def get_pnl_payload(self) -> dict[str, Any]:
        return {}

    async def get_watchlist_payload(self) -> list[str]:
        return []

    async def get_recent_signals_payload(self) -> list[dict[str, Any]]:
        return []

    async def get_today_payload(self) -> dict[str, Any]:
        return {}

    async def get_journal_payload(self) -> list[dict[str, Any]]:
        return []

    async def get_why_payload(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol}

    async def pause_trading(self) -> str:
        return "paused"

    async def resume_trading(self) -> str:
        return "resumed"

    async def switch_mode(self) -> str:
        return "DEMO"

    async def get_risk_payload(self) -> dict[str, Any]:
        return {}

    async def get_settings_payload(self) -> dict[str, Any]:
        return {}

    async def close_symbol(self, symbol: str) -> str:
        return symbol

    async def close_all(self) -> str:
        return "all"

    async def get_news_payload(self) -> list[dict[str, Any]]:
        return []

    async def get_events_payload(self) -> list[dict[str, Any]]:
        return []

    async def reload_settings(self) -> str:
        return "reloaded"

    async def demo_roundtrip(self, symbol: str | None = None) -> str:
        return f"demo:{symbol}"


class FakeResponse:
    """Small fake HTTP response for service tests."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("GET", "https://example.com")
        self.response = httpx.Response(status_code, json=payload, request=self.request)

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "http error",
                request=self.request,
                response=self.response,
            )


class FakeAsyncClient:
    """Minimal async HTTP client stub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.get_queue: list[FakeResponse] = []
        self.post_queue: list[FakeResponse] = []
        self.closed = False

    async def get(self, url: str, params: dict[str, Any] | None = None) -> FakeResponse:
        self.calls.append(("GET", url, params))
        if self.get_queue:
            return self.get_queue.pop(0)
        return FakeResponse({"ok": True, "result": []})

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> FakeResponse:
        self.calls.append(("POST", url, json or data))
        if self.post_queue:
            return self.post_queue.pop(0)
        return FakeResponse({"ok": True, "result": True})

    async def aclose(self) -> None:
        self.closed = True


def build_service(*, admin_ids: list[int] | None = None) -> tuple[TelegramBotService, FakeAsyncClient]:
    """Create a Telegram service with fake HTTP transport."""

    settings = AppSettings(
        telegram=TelegramConfig(enabled=True, admin_ids=admin_ids or [123456789]),
        secrets=EnvSecrets(telegram_bot_token="token", telegram_chat_id="123456789"),
    )
    service = TelegramBotService(settings, TelegramCommandRouter(DummyProvider()))
    fake_client = FakeAsyncClient()
    service._client = fake_client  # type: ignore[assignment]
    return service, fake_client


def test_start_is_idempotent_and_prepares_polling() -> None:
    """Repeated starts should not spawn duplicate polling tasks."""

    async def scenario() -> None:
        service, fake_client = build_service()
        fake_client.get_queue.append(FakeResponse({"ok": True, "result": {"url": ""}}))

        async def fake_poll_loop() -> None:
            await service._stop_event.wait()

        service._poll_loop = fake_poll_loop  # type: ignore[method-assign]

        await service.start()
        await service.start()
        await service.stop()

        delete_webhook_calls = [call for call in fake_client.calls if call[1].endswith("/deleteWebhook")]
        assert len(delete_webhook_calls) == 1
        get_webhook_calls = [call for call in fake_client.calls if call[1].endswith("/getWebhookInfo")]
        assert len(get_webhook_calls) == 1

    asyncio.run(scenario())


def test_poll_conflict_alert_is_throttled() -> None:
    """Conflict warnings should notify admins at most once per throttle window."""

    async def scenario() -> None:
        service, fake_client = build_service(admin_ids=[987654321])
        fake_client.get_queue.extend(
            [
                FakeResponse({"ok": True, "result": {"url": ""}}),
                FakeResponse({"ok": True, "result": {"url": ""}}),
            ]
        )
        await service._handle_poll_conflict("terminated by other getUpdates request")
        await service._handle_poll_conflict("terminated by other getUpdates request")

        send_message_calls = [call for call in fake_client.calls if call[1].endswith("/sendMessage")]
        assert len(send_message_calls) == 1

    asyncio.run(scenario())
