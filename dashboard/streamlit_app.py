"""Shared helpers and landing page for Streamlit."""

from __future__ import annotations

import streamlit as st

from dashboard.common import enqueue_control_command, inspect_bot_runtime, load_control_commands, load_runtime_state, load_settings


st.set_page_config(page_title="Bitget Futures Control Panel", layout="wide")
st.title("Bitget 선물 자동매매 제어판")
st.caption("DEMO 기본 모드, LIVE 이중 확인, 뉴스/리스크/전략 분리 구조")

settings = load_settings()
runtime = load_runtime_state()
col1, col2, col3, col4 = st.columns(4)
col1.metric("모드", settings.get("mode", "DEMO"))
col2.metric("런타임", runtime.get("bot_status", "unknown"))
col3.metric("활성 유니버스", len(runtime.get("active_universe", [])))
col4.metric("열린 포지션", len(runtime.get("open_positions", {})))

st.markdown(
    """
    왼쪽 사이드바에서 페이지를 선택하세요.
    - Overview: 전체 상태
    - Positions: 현재 포지션
    - Symbols: 감시/활성 심볼과 차트
    - News: 최근 뉴스와 AI 분석
    - Journal: 거래 일지
    - Settings: bot_settings.json 수정
    """
)

st.subheader("빠른 제어")
runtime_probe = inspect_bot_runtime()
probe_col1, probe_col2, probe_col3, probe_col4 = st.columns(4)
probe_col1.metric("프로세스", "실행중" if runtime_probe.get("is_running") else "중지")
probe_col2.metric("진단 상태", runtime_probe.get("status", "unknown"))
probe_col3.metric(
    "마지막 헬스체크 경과",
    f"{runtime_probe['health_age_seconds']:.1f}초" if runtime_probe.get("health_age_seconds") is not None else "-",
)
probe_col4.metric(
    "상태 파일 갱신 경과",
    f"{runtime_probe['state_file_age_seconds']:.1f}초" if runtime_probe.get("state_file_age_seconds") is not None else "-",
)

button_col1, button_col2, button_col3 = st.columns([1, 1, 1.2])
with button_col1:
    if st.button("지금 작동중인지 확인", use_container_width=True):
        probe = inspect_bot_runtime()
        detail = (
            f"프로세스: {'실행중' if probe.get('is_running') else '중지'}\n"
            f"봇 상태: {probe.get('bot_status', 'unknown')}\n"
            f"마지막 헬스체크: {probe.get('last_healthcheck_at') or '-'}\n"
            f"마지막 이벤트: {(probe.get('last_event') or {}).get('title', '-')}"
        )
        if probe.get("is_running") and probe.get("is_fresh"):
            st.success(f"봇이 현재 작동 중으로 보입니다.\n{detail}")
        elif probe.get("is_running"):
            st.warning(f"프로세스는 살아 있지만 상태 갱신이 지연되고 있습니다.\n{detail}")
        else:
            st.error(f"봇 프로세스를 찾지 못했습니다.\n{detail}")
with button_col2:
    close_confirmed = st.checkbox("모두청산 확인", key="dashboard_close_all_confirm")
with button_col3:
    if st.button(
        "모두청산 요청",
        type="primary",
        disabled=not close_confirmed or len(runtime.get("open_positions", {})) == 0,
        use_container_width=True,
    ):
        command = enqueue_control_command(
            "close_all_positions",
            payload={
                "source": "streamlit",
                "mode": settings.get("mode", "DEMO"),
                "requested_symbols": list(runtime.get("open_positions", {}).keys()),
            },
        )
        st.warning(
            "모두청산 요청을 큐에 등록했습니다. 봇 주문 동기화 루프가 처리하며, 보통 5초 안팎으로 반영됩니다.\n"
            f"- 명령 ID: {command['id']}"
        )

commands = load_control_commands()
pending_commands = commands.get("pending", [])
history_commands = commands.get("history", [])
latest_command = history_commands[-1] if history_commands else None
st.caption(
    f"제어 명령 대기 {len(pending_commands)}건"
    + (f" / 마지막 처리 결과: {latest_command.get('status')} - {latest_command.get('result')}" if latest_command else "")
)
