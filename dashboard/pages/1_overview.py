"""Overview page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    latest_rows,
    load_runtime_state,
    load_settings,
)


st.title("Overview")
settings = load_settings()
runtime = load_runtime_state()
daily_stats = latest_rows("daily_stats", 1)
latest_stat = daily_stats[0] if daily_stats else {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 모드", settings.get("mode", "DEMO"))
col2.metric("봇 상태", runtime.get("bot_status", "unknown"))
col3.metric("오늘 손익", f"{latest_stat.get('realized_pnl', 0):,.2f} USDT")
col4.metric("리스크 플래그", len(runtime.get("risk_flags", [])))

st.subheader("활성 전략")
enabled = settings.get("strategy", {}).get("enabled", {})
st.write({key: value for key, value in enabled.items()})

st.subheader("활성 유니버스")
st.write(runtime.get("active_universe", []))

st.subheader("최근 이벤트")
events = latest_rows("bot_events", 10)
st.dataframe(events, use_container_width=True)
