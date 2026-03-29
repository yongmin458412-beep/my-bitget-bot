"""Korean Telegram message formatters."""

from __future__ import annotations

import math
from typing import Any


def _fmt_price(price: float) -> str:
    """가격 크기에 따라 자동으로 소수점 자릿수 결정 (SHIB, PEPE 등 마이크로 가격 지원)."""
    if price <= 0:
        return "0"
    magnitude = math.floor(math.log10(abs(price)))
    if magnitude >= 3:      # 1000+: BTC 등
        return f"{price:,.2f}"
    elif magnitude >= 0:    # 1~999: 일반
        return f"{price:,.4f}"
    elif magnitude >= -4:   # 0.0001~1: DOGE 등
        return f"{price:,.6f}"
    else:                   # 0.000001 이하: SHIB, PEPE 등
        decimals = abs(magnitude) + 3
        return f"{price:.{decimals}f}"


def _calc_margin(payload: dict[str, Any]) -> float:
    """투입 마진 = 명목금액 / 레버리지."""
    notional = float(payload.get("entry_notional_usdt", 0) or 0)
    leverage = float(payload.get("leverage", 1) or 1) or 1.0
    return notional / leverage


_TERM_MAP = {
    "break_retest": "브레이크 앤 리테스트",
    "liquidity_raid": "유동성 리클레임",
    "fair_value_gap": "공정가치 갭",
    "order_block": "오더블록",
    "choch": "시장구조 변화",
    "session_breakout": "세션 레인지 돌파",
    "momentum_pullback": "모멘텀 눌림/플래그 재개",
    "trend": "추세장",
    "range": "횡보장",
    "expansion": "확장장",
    "event_risk": "이벤트 위험 구간",
    "dead_market": "죽은 장",
}


def _ko_term(value: Any) -> str:
    """Render internal tokens in Korean where possible."""

    raw = str(value or "").strip()
    if not raw:
        return "-"
    return _TERM_MAP.get(raw, raw.replace("_", " "))


def format_signal_alert(signal: dict[str, Any], account_summary: dict[str, Any] | None = None) -> str:
    """Format a signal alert."""

    account_summary = account_summary or {}
    target_reasons = [str(item) for item in signal.get("target_reasons", []) if str(item).strip()]
    return (
        f"시그널 감지\n"
        f"- 심볼: {signal.get('symbol')}\n"
        f"- 방향: {signal.get('side')}\n"
        f"- 선택 전략: {_ko_term(signal.get('display_name') or signal.get('chosen_strategy') or signal.get('strategy'))}\n"
        f"- 현재 레짐: {_ko_term(signal.get('market_regime'))}\n"
        f"- 진입가: {signal.get('entry_price')}\n"
        f"- 손절가: {signal.get('stop_price')}\n"
        f"- TP1/TP2/TP3: {signal.get('tp1_price')} / {signal.get('tp2_price')} / {signal.get('tp3_price')}\n"
        f"- 손절 사유: {signal.get('stop_reason') or '-'}\n"
        f"- 목표 사유: {', '.join(target_reasons[:3]) or '-'}\n"
        f"- RR(TP1/TP2): {signal.get('rr_to_tp1', 0):.2f} / {signal.get('rr_to_tp2', 0):.2f}\n"
        f"- 충돌 해결: {signal.get('conflict_resolution_decision') or '-'}\n"
        f"- 태그: {', '.join(signal.get('tags', [])[:3])}\n"
        f"- 예상 R: {signal.get('expected_r', 0):.2f}\n"
        f"- 예상 비용 R: 수수료 {signal.get('fees_r', 0):.3f} / 슬리피지 {signal.get('slippage_r', 0):.3f}\n"
        f"- 활성 전략: {signal.get('strategy')}\n"
        f"- 모드: {signal.get('mode', '-')}\n"
        f"- 계좌: 잔고 {account_summary.get('balance', 0):,.2f} USDT / 사용증거금 {account_summary.get('used_margin', 0):,.2f} / 미실현 {account_summary.get('unrealized_pnl', 0):,.2f}"
    )


def format_entry_fill_alert(payload: dict[str, Any]) -> str:
    """Format a filled-entry alert with concrete Korean entry reasons."""

    rationale = payload.get("rationale") or {}
    lines = [str(item).strip() for item in rationale.get("entry_reason_lines", []) if str(item).strip()]
    reasons = "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines[:3], start=1)) or "1. 최근 구조 기준 진입 조건이 확인됐습니다."
    side = "롱" if str(payload.get("side")).lower() == "long" else "숏"
    return (
        f"진입 완료\n"
        f"- 심볼: {payload.get('symbol')}\n"
        f"- 방향: {side}\n"
        f"- 체결가: {payload.get('avg_fill_price', payload.get('price'))}\n"
        f"- 수량: {payload.get('filled_quantity', payload.get('quantity'))}\n"
        f"- 레버리지: {float(payload.get('leverage', 0) or 0):.1f}배\n"
        f"- 선택 전략: {_ko_term(payload.get('display_name') or payload.get('chosen_strategy') or payload.get('strategy'))}\n"
        f"- 현재 레짐: {_ko_term(payload.get('market_regime') or payload.get('regime'))}\n"
        f"- 왜 진입했나: {rationale.get('entry_reason_title') or '-'}\n"
        f"{reasons}\n"
        f"- 투입 마진: {_calc_margin(payload):,.2f} USDT\n"
        f"- 남은 잔고: {float(payload.get('account_balance_after', 0) or 0):,.2f} USDT"
    )


def format_status(settings_snapshot: dict[str, Any], runtime_status: dict[str, Any], balance: dict[str, Any]) -> str:
    """Format /status response."""

    return (
        f"봇 상태\n"
        f"- 모드: {settings_snapshot.get('mode')}\n"
        f"- 런타임: {runtime_status.get('bot_status')}\n"
        f"- 일시정지: {runtime_status.get('paused')}\n"
        f"- 활성 유니버스: {len(runtime_status.get('active_universe', []))}개\n"
        f"- 리스크 플래그: {', '.join(runtime_status.get('risk_flags', [])) or '없음'}\n"
        f"- 잔고: {balance.get('balance', 0):,.2f} USDT\n"
        f"- 사용 증거금: {balance.get('used_margin', 0):,.2f} USDT\n"
        f"- 실현/미실현 손익: {balance.get('realized_pnl', 0):,.2f} / {balance.get('unrealized_pnl', 0):,.2f} USDT\n"
        f"- 마지막 이벤트: {runtime_status.get('last_event', {}).get('title', '-')}"
    )


def format_positions(positions: list[dict[str, Any]]) -> str:
    """Format open positions list."""

    if not positions:
        return "현재 열린 포지션이 없습니다."
    lines = [f"📊 현재 포지션 ({len(positions)}개)"]
    for position in positions:
        side = position.get("side_display") or ("롱" if str(position.get("side")).lower() == "long" else "숏")
        pnl = float(position.get("unrealized_pnl", 0) or 0)
        ret = float(position.get("position_return_pct", 0) or 0)
        tgt = float(position.get("target_pnl_pct", 0) or 0)
        final_tp = float(position.get("final_tp_price", 0) or 0)
        entry = float(position.get("entry_price", 0) or 0)
        sl = float(position.get("stop_price", 0) or 0)
        symbol = position.get("symbol", "?")
        lev = float(position.get("leverage", 1) or 1)
        entry_notional = float(position.get("entry_notional_usdt", 0) or 0)
        if entry_notional <= 0:
            qty = float(position.get("quantity", 0) or 0)
            entry_notional = entry * qty
        target_usdt = entry_notional * tgt / 100.0
        tgt_lev = tgt * lev
        # 손절 시 손실 USDT 계산
        if sl > 0 and entry > 0:
            sl_dist_pct = abs(entry - sl) / entry * 100.0
            loss_usdt = entry_notional * sl_dist_pct / 100.0
            loss_lev_pct = sl_dist_pct * lev
            loss_line = f"  손절시: -{sl_dist_pct:.2f}% × {lev:.0f}배 = -{loss_lev_pct:.2f}% → -{loss_usdt:,.2f} USDT\n"
        else:
            loss_line = ""
        # 수익/손실로 색상, 방향으로 화살표
        pnl_color = "🟢" if ret >= 0 else "🔴"
        direction_arrow = "📈" if side == "롱" else "📉"
        tp_label = _fmt_price(final_tp) if final_tp > 0 else "0"
        tp_line = f"  목표: {tgt:+.2f}% × {lev:.0f}배 = {tgt_lev:+.2f}% → +{target_usdt:,.2f} USDT (TP {tp_label})\n"
        entry_sl_line = (
            f"  진입가: {_fmt_price(entry)} | SL: {_fmt_price(sl)}" if sl > 0
            else f"  진입가: {_fmt_price(entry)}"
        )
        lines.append(
            f"\n{pnl_color} {direction_arrow} {symbol} {side}\n"
            f"  현재 수익률: {ret:+.2f}% ({pnl:+,.2f} USDT)\n"
            f"{tp_line}"
            f"{loss_line}"
            f"{entry_sl_line}"
        )
    return "\n".join(lines)


def format_single_position(position: dict[str, Any]) -> str:
    """Format a single position block (for per-symbol Telegram messages)."""

    side = position.get("side_display") or ("롱" if str(position.get("side")).lower() == "long" else "숏")
    pnl = float(position.get("unrealized_pnl", 0) or 0)
    ret = float(position.get("position_return_pct", 0) or 0)
    tgt = float(position.get("target_pnl_pct", 0) or 0)
    final_tp = float(position.get("final_tp_price", 0) or 0)
    entry = float(position.get("entry_price", 0) or 0)
    sl = float(position.get("stop_price", 0) or 0)
    lev = float(position.get("leverage", 1) or 1)
    strategy = str(position.get("strategy", "") or "-")
    entry_notional = float(position.get("entry_notional_usdt", 0) or 0)
    if entry_notional <= 0:
        qty = float(position.get("quantity", 0) or 0)
        entry_notional = entry * qty
    tgt_lev = tgt * lev
    target_usdt = entry_notional * tgt / 100.0
    pnl_color = "🟢" if ret >= 0 else "🔴"
    direction_arrow = "📈" if side == "롱" else "📉"
    lines = [
        f"{pnl_color} {direction_arrow} {side} | 전략: {strategy} | {lev:.0f}배",
        f"수익률: {ret:+.2f}% ({pnl:+,.2f} USDT)",
        f"목표: {tgt:+.2f}% × {lev:.0f}배 = {tgt_lev:+.2f}% → +{target_usdt:,.2f} USDT"
        + (f" (TP {_fmt_price(final_tp)})" if final_tp > 0 else ""),
        f"진입가: {_fmt_price(entry)}" + (f" | SL: {_fmt_price(sl)}" if sl > 0 else ""),
    ]
    return "\n".join(lines)


def format_news_alert(item: dict[str, Any]) -> str:
    """Format high-impact news alert."""

    return (
        f"고영향 이벤트 감지\n"
        f"- 제목: {item.get('title')}\n"
        f"- 영향도: {item.get('impact_level')}\n"
        f"- 방향 바이어스: {item.get('direction_bias')}\n"
        f"- 차단 여부: {'예' if item.get('should_block_new_entries') else '아니오'}\n"
        f"- 요약: {item.get('summary_ko')}"
    )


def format_daily_summary(summary: dict[str, Any]) -> str:
    """Format daily or weekly summary."""

    return (
        f"{summary.get('title', '요약')}\n"
        f"- 거래 수: {summary.get('trade_count', 0)}\n"
        f"- 승률: {summary.get('win_rate', 0) * 100:.1f}%\n"
        f"- 실현손익: {summary.get('realized_pnl', 0):,.2f} USDT\n"
        f"- PF: {summary.get('profit_factor', 0):.2f}\n"
        f"- 기대값: {summary.get('expectancy', 0):,.2f} USDT"
    )


def format_why_response(payload: dict[str, Any]) -> str:
    """Format a /why response with human-readable summary and raw rule details."""

    if not payload:
        return "설명할 최근 기록이 없습니다."
    rule_summary = payload.get("rule_summary") or {}
    bullet_points = payload.get("bullet_points") or rule_summary.get("bullet_points") or []
    raw_signal = payload.get("signal") or {}
    raw_trade = payload.get("trade") or {}
    return (
        f"왜 이런 계획이었는지\n"
        f"- 심볼: {payload.get('symbol') or raw_signal.get('symbol') or raw_trade.get('symbol')}\n"
        f"- 요약: {payload.get('summary_ko') or rule_summary.get('summary_ko') or '최근 기록이 부족합니다.'}\n"
        f"- 손절 선택: {rule_summary.get('stop_commentary', '-')}\n"
        f"- 목표 선택: {rule_summary.get('target_commentary', '-')}\n"
        f"- RR 판단: {rule_summary.get('rr_commentary', payload.get('risk_commentary', '-'))}\n"
        f"- 핵심 포인트: {' / '.join(str(item) for item in bullet_points[:4]) or '-'}\n"
        f"- 원본 stop_reason: {raw_trade.get('stop_reason') or raw_signal.get('stop_reason') or '-'}\n"
        f"- 원본 RR(TP1/TP2): {raw_trade.get('rr_to_tp1') or raw_signal.get('rr_to_tp1') or 0} / {raw_trade.get('rr_to_tp2') or raw_signal.get('rr_to_tp2') or 0}\n"
        f"- 원본 target_reasons: {', '.join(rule_summary.get('target_reasons', [])[:3]) or '-'}"
    )
