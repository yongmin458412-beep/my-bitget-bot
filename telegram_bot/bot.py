"""Telegram long-polling service using the Bot API directly."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from core.logger import get_logger
from core.settings import AppSettings
from core.utils import chunk_text

from .commands import TelegramCommandRouter
from .formatters import format_single_position
from .keyboards import default_admin_keyboard


class TelegramBotService:
    """Optional Telegram service for notifications and bot control."""

    def __init__(self, settings: AppSettings, router: TelegramCommandRouter) -> None:
        self.settings = settings
        self.router = router
        self.logger = get_logger(__name__)
        self._client = httpx.AsyncClient(timeout=30)
        self._stop_event = asyncio.Event()
        self._offset = 0
        self._poll_task: asyncio.Task[None] | None = None
        self._last_conflict_alert_at = 0.0

    @property
    def enabled(self) -> bool:
        """Whether Telegram is configured."""

        return bool(self.settings.telegram.enabled and self.settings.secrets.telegram_bot_token)

    @property
    def base_url(self) -> str:
        """Telegram Bot API base URL."""

        token = self.settings.secrets.telegram_bot_token
        return f"https://api.telegram.org/bot{token}"

    async def close(self) -> None:
        """Close HTTP resources."""

        await self.stop()
        await self._client.aclose()

    async def start(self) -> None:
        """Start long polling if enabled."""

        if not self.enabled:
            self.logger.info("Telegram disabled or token missing")
            return
        if self._poll_task and not self._poll_task.done():
            self.logger.info("Telegram polling already running")
            return
        self._stop_event.clear()
        await self._prepare_polling()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="telegram_poll_loop")

    async def verify_bot(self) -> dict[str, Any] | None:
        """Verify that the bot token is valid."""

        if not self.enabled:
            return None
        try:
            response = await self._client.get(f"{self.base_url}/getMe")
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                self.logger.warning(
                    "Telegram getMe returned not ok",
                    extra={"extra_data": {"payload": payload}},
                )
                return None
            return payload.get("result", {})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Telegram verification failed", extra={"extra_data": {"error": str(exc)}})
            return None

    async def stop(self) -> None:
        """Stop long polling."""

        self._stop_event.set()
        task = self._poll_task
        self._poll_task = None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def get_webhook_info(self) -> dict[str, Any] | None:
        """Return webhook configuration for diagnostics."""

        if not self.enabled:
            return None
        try:
            response = await self._client.get(f"{self.base_url}/getWebhookInfo")
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                self.logger.warning(
                    "Telegram getWebhookInfo returned not ok",
                    extra={"extra_data": {"payload": payload}},
                )
                return None
            result = payload.get("result", {})
            return result if isinstance(result, dict) else None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Telegram webhook info lookup failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return None

    async def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        """Disable webhook delivery so long polling can own update intake."""

        if not self.enabled:
            return False
        try:
            response = await self._client.post(
                f"{self.base_url}/deleteWebhook",
                json={"drop_pending_updates": drop_pending_updates},
            )
            response.raise_for_status()
            payload = response.json()
            ok = bool(payload.get("ok"))
            if not ok:
                self.logger.warning(
                    "Telegram deleteWebhook returned not ok",
                    extra={"extra_data": {"payload": payload}},
                )
            return ok
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Telegram deleteWebhook failed",
                extra={"extra_data": {"error": str(exc), "drop_pending_updates": drop_pending_updates}},
            )
            return False

    async def _prepare_polling(self) -> None:
        """Ensure the bot is in long-polling mode before starting updates."""

        webhook = await self.get_webhook_info()
        webhook_url = (webhook or {}).get("url", "")
        deleted = await self.delete_webhook(drop_pending_updates=False)
        self.logger.info(
            "Telegram polling prepared",
            extra={
                "extra_data": {
                    "webhook_url": webhook_url,
                    "delete_webhook_ok": deleted,
                }
            },
        )

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        keyboard: dict[str, Any] | None = None,
    ) -> None:
        """Send a single Telegram message."""

        if not self.enabled:
            return
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if keyboard is not None:
            payload["reply_markup"] = keyboard
        try:
            response = await self._client.post(f"{self.base_url}/sendMessage", json=payload)
            response.raise_for_status()
            response_payload = response.json()
            if not response_payload.get("ok"):
                self.logger.warning(
                    "Telegram sendMessage returned not ok",
                    extra={"extra_data": {"chat_id": str(chat_id), "payload": response_payload}},
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Telegram sendMessage failed",
                extra={"extra_data": {"chat_id": str(chat_id), "error": str(exc)}},
            )

    async def send_chunks(self, chat_id: int | str, text: str) -> None:
        """Send long text in multiple messages."""

        for chunk in chunk_text(text):
            await self.send_message(chat_id, chunk)

    async def send_photo(
        self,
        chat_id: int | str,
        photo_path: str | Path,
        *,
        caption: str = "",
        keyboard: dict[str, Any] | None = None,
    ) -> None:
        """Send a local image file to Telegram, optionally with an inline keyboard."""

        if not self.enabled:
            return
        path = Path(photo_path)
        if not path.exists():
            self.logger.warning(
                "Telegram sendPhoto skipped because file is missing",
                extra={"extra_data": {"chat_id": str(chat_id), "photo_path": str(path)}},
            )
            return
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
        if keyboard is not None:
            data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
        try:
            with path.open("rb") as handle:
                response = await self._client.post(
                    f"{self.base_url}/sendPhoto",
                    data=data,
                    files={"photo": (path.name, handle, "image/png")},
                )
            response.raise_for_status()
            response_payload = response.json()
            if not response_payload.get("ok"):
                self.logger.warning(
                    "Telegram sendPhoto returned not ok",
                    extra={"extra_data": {"chat_id": str(chat_id), "payload": response_payload}},
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Telegram sendPhoto failed",
                extra={"extra_data": {"chat_id": str(chat_id), "error": str(exc), "photo_path": str(path)}},
            )

    async def broadcast_admins(self, text: str) -> None:
        """Broadcast a message to all admins or the configured chat."""

        admin_ids = self.settings.telegram.admin_ids or []
        if not admin_ids and self.settings.secrets.telegram_chat_id:
            try:
                admin_ids = [int(self.settings.secrets.telegram_chat_id)]
            except ValueError:
                self.logger.warning(
                    "Invalid TELEGRAM_CHAT_ID",
                    extra={"extra_data": {"telegram_chat_id": self.settings.secrets.telegram_chat_id}},
                )
                admin_ids = []
        if not admin_ids:
            self.logger.info("Telegram recipients unavailable")
            return
        for admin_id in admin_ids:
            await self.send_chunks(admin_id, text)

    async def broadcast_admins_photo(self, photo_path: str | Path, *, caption: str = "") -> None:
        """Broadcast a local image file to every configured admin/chat."""

        admin_ids = self.settings.telegram.admin_ids or []
        if not admin_ids and self.settings.secrets.telegram_chat_id:
            try:
                admin_ids = [int(self.settings.secrets.telegram_chat_id)]
            except ValueError:
                self.logger.warning(
                    "Invalid TELEGRAM_CHAT_ID",
                    extra={"extra_data": {"telegram_chat_id": self.settings.secrets.telegram_chat_id}},
                )
                admin_ids = []
        if not admin_ids:
            self.logger.info("Telegram photo recipients unavailable")
            return
        for admin_id in admin_ids:
            await self.send_photo(admin_id, photo_path, caption=caption)

    async def _poll_loop(self) -> None:
        """Long-poll update loop."""

        _consecutive_errors = 0
        _backoff_base = self.settings.telegram.poll_seconds
        _backoff_max = 60.0

        while not self._stop_event.is_set():
            try:
                response = await self._client.get(
                    f"{self.base_url}/getUpdates",
                    params={"timeout": 25, "offset": self._offset},
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    description = str(data.get("description", "unknown error"))
                    self.logger.warning(
                        "Telegram getUpdates returned not ok",
                        extra={"extra_data": {"description": description, "offset": self._offset}},
                    )
                    if "conflict" in description.lower():
                        await self._handle_poll_conflict(description)
                    _consecutive_errors += 1
                    await asyncio.sleep(min(_backoff_base * (2 ** min(_consecutive_errors - 1, 5)), _backoff_max))
                    continue
                _consecutive_errors = 0
                for update in data.get("result", []):
                    self._offset = int(update["update_id"]) + 1
                    await self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                description = ""
                with contextlib.suppress(Exception):
                    payload = exc.response.json()
                    description = str(payload.get("description", ""))
                if exc.response.status_code == 409:
                    await self._handle_poll_conflict(description or str(exc))
                else:
                    self.logger.warning(
                        "Telegram polling HTTP error",
                        extra={
                            "extra_data": {
                                "status_code": exc.response.status_code,
                                "description": description,
                                "offset": self._offset,
                            }
                        },
                    )
                _consecutive_errors += 1
                await asyncio.sleep(min(_backoff_base * (2 ** min(_consecutive_errors - 1, 5)), _backoff_max))
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc) or type(exc).__name__
                self.logger.warning(
                    "Telegram polling error",
                    extra={"extra_data": {"error": error_msg, "error_type": type(exc).__name__}},
                )
                _consecutive_errors += 1
                await asyncio.sleep(min(_backoff_base * (2 ** min(_consecutive_errors - 1, 5)), _backoff_max))

    async def _handle_poll_conflict(self, description: str) -> None:
        """Warn operators that another update consumer is using the same bot token."""

        webhook = await self.get_webhook_info()
        self.logger.warning(
            "Telegram polling conflict detected",
            extra={
                "extra_data": {
                    "description": description,
                    "offset": self._offset,
                    "webhook_url": (webhook or {}).get("url", ""),
                }
            },
        )
        now = time.monotonic()
        if now - self._last_conflict_alert_at < 300:
            return
        self._last_conflict_alert_at = now
        await self.broadcast_admins(
            "텔레그램 명령 수신 충돌\n"
            "- 같은 봇 토큰을 다른 세션이 사용 중입니다.\n"
            "- 같은 PC의 중복 실행, 다른 서버/앱 polling, 외부 운영 세션을 확인하세요."
        )

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Process a Telegram update."""

        callback_query = update.get("callback_query")
        if callback_query:
            message = callback_query.get("message", {})
            chat_id = message.get("chat", {}).get("id")
            from_user = callback_query.get("from", {})
            text = callback_query.get("data", "")
            await self._handle_text(chat_id, from_user, text)
            await self._client.post(
                f"{self.base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_query["id"]},
            )
            return

        message = update.get("message", {})
        text = message.get("text", "")
        if not text:
            return
        chat_id = message.get("chat", {}).get("id")
        from_user = message.get("from", {})
        await self._handle_text(chat_id, from_user, text)

    async def _handle_text(self, chat_id: int | str, from_user: dict[str, Any], text: str) -> None:
        """Authorize and dispatch a command."""

        user_id = int(from_user.get("id", 0))
        if self.settings.telegram.admin_ids and user_id not in self.settings.telegram.admin_ids:
            await self.send_message(chat_id, "관리자만 사용할 수 있습니다.")
            return

        cmd = text.strip()

        # ─────────────────────────────────────────────────────────
        # 💼 포지션: 심볼별 개별 메시지 + [차트 보기] [청산] 버튼
        # ─────────────────────────────────────────────────────────
        if cmd in ("/positions", "positions"):
            positions = await self.router.provider.get_positions_payload()
            if not positions:
                await self.send_message(chat_id, "📊 포지션\n\n⚪ 없음(관망)")
                return
            for pos in positions:
                sym = str(pos.get("symbol", "?"))
                block = format_single_position(pos)
                kb = {
                    "inline_keyboard": [[
                        {"text": "📷 차트 보기", "callback_data": f"/chart_{sym}"},
                        {"text": "🚨 청산", "callback_data": f"/close_{sym}"},
                    ]]
                }
                await self.send_message(chat_id, f"📊 {sym}\n\n{block}", keyboard=kb)
            return

        # ─────────────────────────────────────────────────────────
        # 📷 차트 보기: 실시간 차트 사진 + [청산] 버튼
        # ─────────────────────────────────────────────────────────
        if cmd.startswith("/chart_"):
            sym = cmd[len("/chart_"):]
            try:
                chart_path = await self.router.provider.get_position_chart(sym)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("get_position_chart failed", extra={"extra_data": {"symbol": sym, "error": str(exc)}})
                chart_path = None
            if chart_path:
                close_kb = {
                    "inline_keyboard": [[
                        {"text": f"🚨 {sym} 청산", "callback_data": f"/close_{sym}"}
                    ]]
                }
                await self.send_photo(
                    chat_id,
                    chart_path,
                    caption=f"📷 {sym} 실시간 차트",
                    keyboard=close_kb,
                )
            else:
                await self.send_message(chat_id, f"⚪ {sym} 포지션 없음 또는 차트 생성 실패")
            return

        # ─────────────────────────────────────────────────────────
        # 🚨 개별 청산: /close_BTCUSDT 형태 (close_all 제외)
        # ─────────────────────────────────────────────────────────
        if cmd.startswith("/close_") and cmd != "/closeall":
            sym = cmd[len("/close_"):]
            result = await self.router.provider.close_symbol(sym)
            await self.send_message(chat_id, result)
            return

        # ─────────────────────────────────────────────────────────
        # 기존 커맨드 라우팅
        # ─────────────────────────────────────────────────────────
        response = await self.router.dispatch(cmd)
        keyboard = default_admin_keyboard() if cmd == "/start" else None
        await self.send_chunks(chat_id, response)
        if keyboard is not None:
            await self.send_message(chat_id, "빠른 메뉴", keyboard=keyboard)
