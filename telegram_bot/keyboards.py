"""Inline keyboard helpers."""

from __future__ import annotations


def default_admin_keyboard() -> dict[str, list[list[dict[str, str]]]]:
    """Default quick-action keyboard."""

    return {
        "inline_keyboard": [
            [
                {"text": "📊 상태", "callback_data": "/status"},
                {"text": "💼 포지션", "callback_data": "/positions"},
                {"text": "💰 잔고", "callback_data": "/balance"},
            ],
            [
                {"text": "📈 오늘 성과", "callback_data": "/today"},
                {"text": "📰 뉴스", "callback_data": "/news"},
                {"text": "📋 시그널", "callback_data": "/signals"},
            ],
            [
                {"text": "🤖 AI스캔", "callback_data": "/ai_scan"},
                {"text": "🔍 봇상태", "callback_data": "/bot_status"},
            ],
            [
                {"text": "⏸ 일시정지", "callback_data": "/pause"},
                {"text": "▶️ 재개", "callback_data": "/resume"},
                {"text": "🔄 리로드", "callback_data": "/reload"},
            ],
            [
                {"text": "🚨 모두청산", "callback_data": "/closeall"},
            ],
        ]
    }

