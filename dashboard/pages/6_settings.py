"""Settings page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import load_settings, save_settings_patch


st.title("Settings")
settings = load_settings()

mode = st.selectbox("모드", ["DEMO", "LIVE"], index=0 if settings.get("mode") == "DEMO" else 1)
risk_per_trade = st.slider(
    "트레이드당 리스크",
    min_value=0.0025,
    max_value=0.01,
    value=float(settings.get("risk", {}).get("risk_per_trade", 0.005)),
    step=0.0005,
    format="%.4f",
)
max_positions = st.number_input(
    "최대 동시 포지션",
    min_value=1,
    max_value=20,
    value=int(settings.get("risk", {}).get("max_concurrent_positions", 5)),
)
min_ev = st.slider(
    "최소 EV",
    min_value=0.0,
    max_value=1.0,
    value=float(settings.get("ev", {}).get("min_ev", 0.15)),
    step=0.01,
)
telegram_enabled = st.checkbox("텔레그램 사용", value=bool(settings.get("telegram", {}).get("enabled", True)))
news_enabled = st.checkbox("뉴스 필터 사용", value=bool(settings.get("news", {}).get("enabled", True)))
live_confirmed = st.checkbox(
    "LIVE 최종 확인",
    value=bool(settings.get("exchange", {}).get("live_streamlit_confirmed", False)),
    help="LIVE 모드 전환 전 최종 확인 스위치입니다.",
)

if st.button("설정 저장", type="primary"):
    patch = {
        "mode": mode,
        "risk": {
            "risk_per_trade": risk_per_trade,
            "max_concurrent_positions": int(max_positions),
        },
        "ev": {"min_ev": min_ev},
        "telegram": {"enabled": telegram_enabled},
        "news": {"enabled": news_enabled},
        "exchange": {"live_streamlit_confirmed": live_confirmed},
    }
    save_settings_patch(patch)
    st.success("bot_settings.json 이 업데이트되었습니다. /reload 또는 봇 자동 감지를 기다리세요.")
