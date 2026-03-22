"""Positions page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    enqueue_control_command,
    latest_rows,
    load_control_commands,
    load_runtime_state,
    load_settings,
)


def _pnl_pct(entry: float, current: float, side: str) -> float:
    """현재 수익률 (%)."""
    if entry <= 0:
        return 0.0
    if side == "long":
        return (current - entry) / entry * 100
    return (entry - current) / entry * 100


def _target_pct(entry: float, tp_price: float, side: str) -> float:
    """목표 수익률 (%)."""
    if entry <= 0 or tp_price <= 0:
        return 0.0
    if side == "long":
        return (tp_price - entry) / entry * 100
    return (entry - tp_price) / entry * 100


st.title("Positions")

runtime = load_runtime_state()
settings = load_settings()
open_positions: dict = runtime.get("open_positions", {})

# ── 모두 청산 버튼 ─────────────────────────────────────────────────────────
col_chk, col_btn = st.columns([3, 1])
with col_chk:
    confirmed = st.checkbox("모두 청산 전 확인 (체크 후 버튼 활성화)", key="pos_close_all_confirm")
with col_btn:
    if st.button(
        "모두 청산",
        type="primary",
        disabled=not confirmed or len(open_positions) == 0,
        use_container_width=True,
    ):
        command = enqueue_control_command(
            "close_all_positions",
            payload={
                "source": "streamlit_positions",
                "mode": settings.get("mode", "DEMO"),
                "requested_symbols": list(open_positions.keys()),
            },
        )
        st.warning(
            f"모두 청산 요청이 큐에 등록됐습니다. 보통 5초 안에 처리됩니다.\n"
            f"- 명령 ID: `{command['id']}`"
        )

# 처리 상태 표시
commands = load_control_commands()
latest_cmd = (commands.get("history") or [None])[-1]
if latest_cmd:
    st.caption(
        f"마지막 처리: {latest_cmd.get('action')} → {latest_cmd.get('status')} "
        f"({latest_cmd.get('result', '')})"
    )

st.divider()

# ── 포지션 목록 ────────────────────────────────────────────────────────────
if not open_positions:
    positions_db = latest_rows("positions", 50)
    if positions_db:
        st.subheader("최근 청산 포지션 (DB)")
        st.dataframe(positions_db, use_container_width=True)
    else:
        st.info("현재 열린 포지션이 없습니다.")
else:
    st.subheader(f"열린 포지션 ({len(open_positions)}개)")

    for symbol, pos in open_positions.items():
        side = str(pos.get("side", "long")).lower()
        entry = float(pos.get("entry_price") or 0)
        mark = float(pos.get("mark_price") or 0)
        stop = float(pos.get("stop_price") or 0)
        tp1 = float(pos.get("tp1_price") or 0)
        tp2 = float(pos.get("tp2_price") or 0)
        tp3 = float(pos.get("tp3_price") or 0)
        qty = float(pos.get("remaining_quantity") or pos.get("quantity") or 0)
        strategy = pos.get("strategy", "-")

        cur_pct = _pnl_pct(entry, mark, side)

        # 최종 목표: tp3 > tp2 > tp1 순으로 유효한 것
        final_tp = tp3 if tp3 > 0 else (tp2 if tp2 > 0 else tp1)
        tgt_pct = _target_pct(entry, final_tp, side)

        # TP 달성 현황
        tp1_done = pos.get("tp1_done", False)
        tp2_done = pos.get("tp2_done", False)
        tp3_done = pos.get("tp3_done", False)

        side_emoji = "🟢" if side == "long" else "🔴"
        pnl_color = "🟩" if cur_pct >= 0 else "🟥"

        with st.expander(
            f"{side_emoji} **{symbol}** | {side.upper()} | 현재 {cur_pct:+.2f}% → 목표 {tgt_pct:+.2f}%",
            expanded=True,
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("진입가", f"{entry:,.4f}")
            m2.metric("현재가 (Mark)", f"{mark:,.4f}", f"{cur_pct:+.2f}%")
            m3.metric(
                "목표가 (최종 TP)",
                f"{final_tp:,.4f}" if final_tp > 0 else "-",
                f"{tgt_pct:+.2f}%" if final_tp > 0 else None,
            )
            m4.metric("손절가", f"{stop:,.4f}" if stop > 0 else "-")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("잔여 수량", f"{qty:.4f}")
            c2.metric("전략", strategy or "-")
            c3.metric(
                "현재 수익률",
                f"{cur_pct:+.2f}%",
                delta_color="normal",
            )
            c4.metric(
                "목표 수익률",
                f"{tgt_pct:+.2f}%" if final_tp > 0 else "-",
                delta_color="off",
            )

            # TP 진행 현황
            tp_cols = st.columns(3)
            tp_cols[0].metric("TP1", f"{tp1:,.4f}" if tp1 > 0 else "-", "✅ 완료" if tp1_done else "대기")
            tp_cols[1].metric("TP2", f"{tp2:,.4f}" if tp2 > 0 else "-", "✅ 완료" if tp2_done else "대기")
            tp_cols[2].metric("TP3", f"{tp3:,.4f}" if tp3 > 0 else "-", "✅ 완료" if tp3_done else ("대기" if tp3 > 0 else "-"))
