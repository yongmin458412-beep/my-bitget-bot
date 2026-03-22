"""Tests for Telegram command routing."""

from __future__ import annotations

import asyncio
from typing import Any

from telegram_bot.commands import TelegramCommandRouter


class DummyProvider:
    """Minimal provider for command-router tests."""

    async def get_status_payload(self) -> dict[str, Any]:
        return {"settings": {}, "runtime": {}, "balance": {}}

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


def test_help_includes_demo_roundtrip() -> None:
    """The help command should expose the new demo command."""

    router = TelegramCommandRouter(DummyProvider())
    response = asyncio.run(router.dispatch("/help"))
    assert "/demo_roundtrip [SYMBOL]" in response


def test_demo_roundtrip_dispatches_symbol() -> None:
    """The router should pass the normalized symbol to the provider."""

    router = TelegramCommandRouter(DummyProvider())
    response = asyncio.run(router.dispatch("/demo_roundtrip btcusdt"))
    assert response == "demo:BTCUSDT"
