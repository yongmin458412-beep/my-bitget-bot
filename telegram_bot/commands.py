"""Telegram command routing."""

from __future__ import annotations

from typing import Any, Protocol

from .formatters import format_daily_summary, format_news_alert, format_positions, format_status, format_why_response


def _format_bot_status(payload: dict[str, Any]) -> str:
    """Format detailed bot status information in Korean."""

    mode = payload.get("mode", "DEMO")
    bot_status = payload.get("bot_status", "UNKNOWN")
    paused = payload.get("paused", False)
    active_positions = payload.get("active_positions", [])
    pending_orders = payload.get("open_orders", {})
    active_strategies = payload.get("enabled_strategies", {})
    risk_flags = payload.get("risk_flags", [])

    status_emoji = "🟢" if bot_status == "RUNNING" and not paused else "🟡" if paused else "🔴"
    mode_emoji = "🔴" if mode == "LIVE" else "⚪"

    lines = [
        f"{status_emoji} 봇 상태: {bot_status}{'(일시정지)' if paused else ''}",
        f"{mode_emoji} 모드: {mode}",
        f"",
        f"📍 현재 상태:",
        f"  • 열린 포지션: {len(active_positions)}개 ({', '.join([p.get('symbol', '?') for p in active_positions[:5]]) or 'None'})",
        f"  • 미체결 주문: {len(pending_orders)}개 ({', '.join(list(pending_orders.keys())[:5]) or 'None'})",
        f"  • 활성 전략: {', '.join(k for k, v in active_strategies.items() if v) or '없음'}",
        f"",
    ]

    if risk_flags:
        lines.append(f"⚠️ 리스크 플래그: {', '.join(risk_flags)}")
        lines.append("")

    lines.append(f"마지막 이벤트: {payload.get('last_event_title', '-')}")

    return "\n".join(lines)


class CommandProvider(Protocol):
    """Control surface used by the Telegram bot."""

    async def get_status_payload(self) -> dict[str, Any]: ...
    async def get_positions_payload(self) -> list[dict[str, Any]]: ...
    async def get_balance_payload(self) -> dict[str, Any]: ...
    async def get_pnl_payload(self) -> dict[str, Any]: ...
    async def get_watchlist_payload(self) -> list[str]: ...
    async def get_recent_signals_payload(self) -> list[dict[str, Any]]: ...
    async def get_today_payload(self) -> dict[str, Any]: ...
    async def get_journal_payload(self) -> list[dict[str, Any]]: ...
    async def get_why_payload(self, symbol: str) -> dict[str, Any]: ...
    async def pause_trading(self) -> str: ...
    async def resume_trading(self) -> str: ...
    async def switch_mode(self) -> str: ...
    async def get_risk_payload(self) -> dict[str, Any]: ...
    async def get_settings_payload(self) -> dict[str, Any]: ...
    async def close_symbol(self, symbol: str) -> str: ...
    async def close_all(self) -> str: ...
    async def get_news_payload(self) -> list[dict[str, Any]]: ...
    async def get_events_payload(self) -> list[dict[str, Any]]: ...
    async def reload_settings(self) -> str: ...
    async def demo_roundtrip(self, symbol: str | None = None) -> str: ...
    async def ai_scan(self) -> str: ...
    async def get_bot_status_detailed(self) -> dict[str, Any]: ...


class TelegramCommandRouter:
    """Map Telegram commands to provider calls."""

    def __init__(self, provider: CommandProvider) -> None:
        self.provider = provider

    async def dispatch(self, text: str) -> str:
        """Handle a Telegram text command."""

        command, _, raw_args = text.partition(" ")
        args = raw_args.strip()

        if command == "/start":
            return "Bitget 선물 자동매매 봇입니다. /help 로 명령어를 확인하세요."
        if command == "/help":
            return (
                "/status /positions /balance /pnl /watchlist /signals /today /journal /why SYMBOL "
                "/pause /resume /mode /risk /settings /close SYMBOL /closeall /news /events /reload "
                "/demo_roundtrip [SYMBOL] /ai_scan /bot_status"
            )
        if command == "/status":
            payload = await self.provider.get_status_payload()
            return format_status(payload["settings"], payload["runtime"], payload["balance"])
        if command == "/positions":
            return format_positions(await self.provider.get_positions_payload())
        if command == "/balance":
            balance = await self.provider.get_balance_payload()
            return f"잔고 {balance.get('balance', 0):,.2f} USDT / 사용증거금 {balance.get('used_margin', 0):,.2f} / 미실현 {balance.get('unrealized_pnl', 0):,.2f}"
        if command == "/pnl":
            pnl = await self.provider.get_pnl_payload()
            return f"실현 {pnl.get('realized_pnl', 0):,.2f} USDT / 미실현 {pnl.get('unrealized_pnl', 0):,.2f} / ROI {pnl.get('roi_pct', 0):.2f}%"
        if command == "/watchlist":
            symbols = await self.provider.get_watchlist_payload()
            return "활성 유니버스\n- " + "\n- ".join(symbols[:50])
        if command == "/signals":
            signals = await self.provider.get_recent_signals_payload()
            if not signals:
                return "최근 시그널이 없습니다."
            return "\n".join(
                f"- {signal.get('symbol')} {signal.get('side')} {signal.get('strategy')} EV {signal.get('expected_value', 0):.2f}"
                for signal in signals[:10]
            )
        if command == "/today":
            return format_daily_summary(await self.provider.get_today_payload())
        if command == "/journal":
            rows = await self.provider.get_journal_payload()
            return "\n".join(
                f"- {row.get('created_at')} {row.get('symbol')} {row.get('side')} pnl {row.get('realized_pnl_usdt', 0)}"
                for row in rows[:15]
            ) or "거래 일지가 없습니다."
        if command == "/why":
            if not args:
                return "/why SYMBOL 형식으로 입력하세요."
            payload = await self.provider.get_why_payload(args.upper())
            return format_why_response(payload)
        if command == "/pause":
            return await self.provider.pause_trading()
        if command == "/resume":
            return await self.provider.resume_trading()
        if command == "/mode":
            return await self.provider.switch_mode()
        if command == "/risk":
            return str(await self.provider.get_risk_payload())
        if command == "/settings":
            return str(await self.provider.get_settings_payload())
        if command == "/close":
            if not args:
                return "/close SYMBOL 형식으로 입력하세요."
            return await self.provider.close_symbol(args.upper())
        if command == "/closeall":
            return await self.provider.close_all()
        if command == "/news":
            rows = await self.provider.get_news_payload()
            if not rows:
                return "최근 뉴스가 없습니다."
            return "\n\n".join(format_news_alert(row) for row in rows[:5])
        if command == "/events":
            rows = await self.provider.get_events_payload()
            return "\n".join(f"- {row.get('title')}" for row in rows[:10]) or "이벤트가 없습니다."
        if command == "/reload":
            return await self.provider.reload_settings()
        if command == "/demo_roundtrip":
            return await self.provider.demo_roundtrip(args.upper() if args else None)
        if command == "/ai_scan":
            return await self.provider.ai_scan()
        if command == "/bot_status":
            payload = await self.provider.get_bot_status_detailed()
            return _format_bot_status(payload)
        return "알 수 없는 명령입니다. /help 를 확인하세요."
