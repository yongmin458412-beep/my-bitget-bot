"""Telegram formatter tests."""

from __future__ import annotations

from telegram_bot.formatters import (
    format_entry_fill_alert,
    format_positions,
    format_signal_alert,
    format_status,
    format_why_response,
)


def test_format_status_contains_mode_and_balance() -> None:
    """Status formatter should include core fields."""

    message = format_status(
        {"mode": "DEMO"},
        {"bot_status": "running", "paused": False, "active_universe": ["BTCUSDT"], "risk_flags": [], "last_event": {"title": "ok"}},
        {"balance": 1000.0, "used_margin": 20.0, "realized_pnl": 10.0, "unrealized_pnl": -3.0},
    )
    assert "DEMO" in message
    assert "잔고" in message


def test_format_positions_handles_empty() -> None:
    """Position formatter should handle no positions."""

    assert "없습니다" in format_positions([])


def test_format_positions_is_compact_and_usdt_focused() -> None:
    """Positions formatter should show return %, pnl, and notionals."""

    message = format_positions(
        [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "side_display": "롱",
                "entry_notional_usdt": 1803.16,
                "current_notional_usdt": 1817.42,
                "position_return_pct": 0.79,
                "unrealized_pnl": 14.25,
                "stop_price": 69524.3397,
            }
        ]
    )
    assert "BTCUSDT 롱" in message
    assert "+0.79%" in message
    assert "+14.25 USDT" in message
    assert "진입 1,803.16 USDT" in message
    assert "현재 1,817.42 USDT" in message


def test_format_signal_alert_includes_structural_fields() -> None:
    """Signal alert should expose stop/target reasons and RR."""

    message = format_signal_alert(
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "entry_price": 100.0,
            "stop_price": 99.0,
            "tp1_price": 101.5,
            "tp2_price": 103.0,
            "tp3_price": 105.0,
            "stop_reason": "retest_low_atr_buffer",
            "target_reasons": ["previous_high", "session_high", "range_opposite_side"],
            "rr_to_tp1": 1.5,
            "rr_to_tp2": 3.0,
            "strategy": "break_retest",
            "display_name": "브레이크 앤 리테스트",
            "chosen_strategy": "break_retest",
            "market_regime": "trend",
            "conflict_resolution_decision": "single_candidate",
            "expected_r": 3.0,
            "fees_r": 0.01,
            "slippage_r": 0.02,
            "mode": "DEMO",
            "tags": ["break", "retest"],
        },
        account_summary={"balance": 1000.0, "used_margin": 10.0, "unrealized_pnl": 0.0},
    )
    assert "retest_low_atr_buffer" in message
    assert "previous_high" in message
    assert "RR(TP1/TP2)" in message
    assert "TP1/TP2/TP3" in message
    assert "선택 전략" in message
    assert "현재 레짐" in message
    assert "충돌 해결" in message


def test_format_why_response_includes_rule_and_raw_fields() -> None:
    """Why response should be human-readable and retain raw rule hints."""

    message = format_why_response(
        {
            "symbol": "BTCUSDT",
            "summary_ko": "손절은 구조 무효화 지점에 두고 목표가는 다음 유동성으로 잡았습니다.",
            "bullet_points": ["손절 사유: retest_low_atr_buffer"],
            "rule_summary": {
                "stop_commentary": "리테스트 저점이 깨지면 가설이 무효입니다.",
                "target_commentary": "직전 고점과 세션 고점을 목표로 삼았습니다.",
                "rr_commentary": "TP1 RR 1.4, TP2 RR 2.1 이라 통과했습니다.",
                "target_reasons": ["previous_high", "session_high"],
            },
            "signal": {"stop_reason": "retest_low_atr_buffer", "rr_to_tp1": 1.4, "rr_to_tp2": 2.1},
            "trade": {},
        }
    )
    assert "손절 선택" in message
    assert "원본 stop_reason" in message
    assert "previous_high" in message


def test_format_entry_fill_alert_includes_reason_lines_and_leverage() -> None:
    """Filled-entry alert should explain the setup in Korean."""

    message = format_entry_fill_alert(
        {
            "symbol": "XRPUSDT",
            "side": "short",
            "avg_fill_price": 1.4563,
            "filled_quantity": 2192.0,
            "leverage": 20.0,
            "display_name": "유동성 리클레임",
            "market_regime": "range",
            "stop_price": 1.4618,
            "tp1_price": 1.4517,
            "tp2_price": 1.4469,
            "tp3_price": 1.4422,
            "final_target_price": 1.4385,
            "entry_notional_usdt": 3191.77,
            "remaining_notional_usdt": 3191.77,
            "rationale": {
                "entry_reason_title": "직전 고점 스윕 후 재진입",
                "entry_reason_lines": [
                    "직전 고점을 먼저 위로 스윕했습니다.",
                    "이후 다시 레벨 아래로 복귀하며 숏 reclaim이 확인됐습니다.",
                    "5분 구조 전환과 확인 캔들이 붙어 진입했습니다.",
                ],
            },
        }
    )
    assert "왜 진입했나" in message
    assert "직전 고점을 먼저 위로 스윕했습니다." in message
    assert "레버리지: 20.0배" in message
    assert "선택 전략" in message
    assert "진입한 USDT: 3,191.77" in message
    assert "남은 USDT: 3,191.77" in message
