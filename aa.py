
# =========================================================
#  Bitget AI Wonyoti Agent (Final Integrated) - 유지보수/확장판
#  - Streamlit: 제어판/차트/포지션/일지/AI 시야/백테스트/내보내기
#  - Telegram: 실시간 보고/조회/일지 요약 + (채널/그룹 분리 지원) + /menu
#  - AutoTrade: 데모(IS_SANDBOX=True) 기본
#
#  ⚠️ 주의: 트레이딩은 손실 위험이 큽니다. (특히 레버리지)
#
#  requirements.txt 추천(있으면 사용, 없어도 동작하도록 optional import 처리):
#  - streamlit
#  - ccxt
#  - openai
#  - requests
#  - pandas
#  - numpy
#  - ta
#  - streamlit-autorefresh
#  - orjson
#  - tenacity
#  - diskcache
#  - pandas_ta
#  - scipy
#  - feedparser
#  - cachetools
#  - openpyxl              # Excel 내보내기
#  - gspread               # Google Sheets (선택)
#  - google-auth           # Google Sheets (선택)
#  - deep-translator       # 한글화(선택, 없으면 AI/룰 기반)
#  - loguru                # 로그(선택)
# =========================================================

import os
import re
import json
import time
import uuid
import math
import threading
import traceback
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests
import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except Exception:
    # 구버전/환경 차이 대응(스레드 컨텍스트 미지원이어도 봇이 죽지 않게)
    def add_script_run_ctx(_th):  # type: ignore
        return None

import ccxt
from openai import OpenAI

# ---- optional pip ----
try:
    import ta  # pip: ta
except Exception:
    ta = None

try:
    from streamlit_autorefresh import st_autorefresh  # pip: streamlit-autorefresh
except Exception:
    st_autorefresh = None

try:
    import orjson
except Exception:
    orjson = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter
except Exception:
    retry = None
    stop_after_attempt = None
    wait_exponential_jitter = None

try:
    from diskcache import Cache
except Exception:
    Cache = None

try:
    import pandas_ta as pta
except Exception:
    pta = None

try:
    from scipy.signal import argrelextrema
except Exception:
    argrelextrema = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None

try:
    import openpyxl  # noqa: F401  # pip: openpyxl
except Exception:
    openpyxl = None

try:
    import gspread  # pip: gspread
    from google.oauth2.service_account import Credentials as GoogleCredentials  # pip: google-auth
except Exception:
    gspread = None
    GoogleCredentials = None

try:
    from deep_translator import GoogleTranslator  # pip: deep-translator
except Exception:
    GoogleTranslator = None

try:
    from loguru import logger  # pip: loguru
except Exception:
    logger = None


# =========================================================
# ✅ 빌드/버전 토큰(운영 디버깅용)
# - Streamlit은 rerun 시에도 daemon thread가 남을 수 있어, "지금 어떤 코드가 돌아가고 있는지"
#   확인하기 쉽게 토큰을 만든다.
# =========================================================
def _code_version_token() -> str:
    try:
        p = str(__file__ or "").strip()
        if not p:
            return "unknown"
        mtime = int(os.path.getmtime(p))
        return f"{os.path.basename(p)}@{mtime}"
    except Exception:
        return "unknown"


CODE_VERSION = _code_version_token()


# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Final Integrated)")

IS_SANDBOX = True  # ✅ 데모/모의투자 (실전 전환은 파일 하단 안내 참고)

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"

DETAIL_DIR = "trade_details"
DAILY_REPORT_DIR = "daily_reports"
os.makedirs(DETAIL_DIR, exist_ok=True)
os.makedirs(DAILY_REPORT_DIR, exist_ok=True)

_cache = Cache("cache") if Cache else None  # 선택(디스크 캐시)

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

# OpenAI 호출 타임아웃(초) - 스레드 멈춤 방지
OPENAI_TIMEOUT_SEC = 20

# HTTP 요청 타임아웃(초)
HTTP_TIMEOUT_SEC = 12

# 외부 시황 갱신 hard-timeout(초) - 네트워크/번역 지연으로 봇 스레드가 멈춘 것처럼 보이는 문제 완화
EXTERNAL_CONTEXT_TIMEOUT_SEC = max(10, HTTP_TIMEOUT_SEC + 4)

_THREAD_POOL = ThreadPoolExecutor(max_workers=4)


# =========================================================
# ✅ 1) 시간 유틸 (KST)
# =========================================================
KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")


def _parse_time_kst(s: str) -> Optional[datetime]:
    try:
        # "YYYY-MM-DD HH:MM:SS"
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except Exception:
        return None


def _dt_to_epoch(dt: datetime) -> float:
    try:
        return dt.timestamp()
    except Exception:
        return time.time()


def _epoch_to_kst_str(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return now_kst_str()


# =========================================================
# ✅ 2) JSON 안전 저장/로드 (원자적)
# =========================================================
def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        if orjson:
            with open(tmp, "wb") as f:
                f.write(orjson.dumps(data))
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # 파일 I/O 에러가 봇을 죽이면 안 됨
        pass


def read_json_safe(path: str, default=None):
    try:
        if orjson:
            with open(path, "rb") as f:
                return orjson.loads(f.read())
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_json_dumps(x: Any, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        try:
            s = str(x)
        except Exception:
            s = ""
    if len(s) > limit:
        return s[:limit] + "..."
    return s


# =========================================================
# ✅ 2.5) 상세일지 저장/조회
# =========================================================
def save_trade_detail(trade_id: str, payload: Dict[str, Any]) -> None:
    try:
        write_json_atomic(os.path.join(DETAIL_DIR, f"{trade_id}.json"), payload)
    except Exception:
        pass


def load_trade_detail(trade_id: str) -> Optional[Dict[str, Any]]:
    try:
        return read_json_safe(os.path.join(DETAIL_DIR, f"{trade_id}.json"), None)
    except Exception:
        return None


def list_recent_trade_ids(limit: int = 10) -> List[str]:
    try:
        files = [f for f in os.listdir(DETAIL_DIR) if f.endswith(".json")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(DETAIL_DIR, x)), reverse=True)
        return [os.path.splitext(f)[0] for f in files[:limit]]
    except Exception:
        return []


# =========================================================
# ✅ 2.6) Streamlit/pyarrow 호환: DataFrame 안전 변환
# - object 타입에 숫자/문자/딕트 혼재 시 Arrow 변환이 터질 수 있음
#   (사용자 로그: "Expected bytes, got a 'int' object")
# =========================================================
def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    try:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].apply(
                    lambda v: safe_json_dumps(v, limit=400)
                    if isinstance(v, (dict, list))
                    else ("" if v is None else str(v))
                )
        return out
    except Exception:
        try:
            return df.astype(str)
        except Exception:
            return pd.DataFrame()


# =========================================================
# ✅ 2.7) Streamlit DataFrame 표시 호환(버전 차이 대응)
# - 일부 Streamlit 버전에서 st.dataframe(width="stretch") / hide_index / use_container_width 호환 문제
# - UI 기능이 "작동 안 함"처럼 보이는 런타임 오류를 줄인다.
# =========================================================
def st_dataframe_safe(data, **kwargs):
    """
    Streamlit 버전 차이로 인한 파라미터 TypeError를 흡수하면서 최대한 표시.
    - 최신 Streamlit(2025+): `use_container_width`가 deprecate → `width="stretch"` 우선 사용
    - 구버전 Streamlit: `width` 미지원이면 `use_container_width=True/False`로 폴백
    """
    try:
        # ✅ 최신 Streamlit 권장: width="stretch"/"content"
        # - 호출자가 use_container_width를 줬다면(레거시), 가능한 경우 width로 변환해 경고를 없앤다.
        if "use_container_width" in kwargs and "width" not in kwargs:
            try:
                kwargs["width"] = "stretch" if bool(kwargs.get("use_container_width")) else "content"
            except Exception:
                kwargs["width"] = "stretch"
            kwargs.pop("use_container_width", None)
        kwargs.setdefault("width", "stretch")
        return st.dataframe(data, **kwargs)
    except TypeError:
        # 구버전 Streamlit: width 미지원 → use_container_width로 폴백
        try:
            w = kwargs.pop("width", None)
            if "use_container_width" not in kwargs:
                if w == "content":
                    kwargs["use_container_width"] = False
                else:
                    kwargs["use_container_width"] = True
            return st.dataframe(data, **kwargs)
        except TypeError:
            # 지원하지 않는 kwargs 제거 후 재시도
            for k in ["use_container_width", "hide_index", "column_config", "column_order", "width"]:
                kwargs.pop(k, None)
            try:
                return st.dataframe(data, **kwargs)
            except Exception:
                return st.dataframe(data)
        try:
            return st.dataframe(data)
        except Exception:
            return st.dataframe(data)
    except Exception:
        return st.dataframe(data)


# =========================================================
# ✅ 3) MODE_RULES (기존 유지)
# =========================================================
MODE_RULES = {
    "안전모드": {"min_conf": 85, "entry_pct_min": 2, "entry_pct_max": 8, "lev_min": 2, "lev_max": 8},
    "공격모드": {"min_conf": 80, "entry_pct_min": 8, "entry_pct_max": 25, "lev_min": 2, "lev_max": 10},
    "하이리스크/하이리턴": {"min_conf": 85, "entry_pct_min": 15, "entry_pct_max": 40, "lev_min": 8, "lev_max": 25},
}


# =========================================================
# ✅ 4) 설정 관리 (load/save)
# =========================================================
def default_settings() -> Dict[str, Any]:
    return {
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",
        "timeframe": "5m",
        "order_usdt": 100.0,

        # Telegram (기본 유지)
        "tg_enable_reports": True,  # 이벤트 알림(진입/청산 등)
        "tg_send_entry_reason": False,

        # ✅ 주기 리포트/시야 리포트
        "tg_enable_periodic_report": True,
        "report_interval_min": 15,
        "tg_enable_hourly_vision_report": True,
        "vision_report_interval_min": 60,

        # ✅ 텔레그램 라우팅: channel/group (secrets로 설정 권장)
        "tg_route_events_to": "channel",  # "channel"|"group"|"both"
        "tg_route_queries_to": "group",   # "group"|"channel"|"both"

        # 지표 파라미터
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 지표 ON/OFF
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True, "use_ma": True,
        "use_macd": True, "use_stoch": True, "use_mfi": True, "use_willr": True, "use_adx": True,

        # 방어/전략
        "use_trailing_stop": True,
        "use_dca": True, "dca_trigger": -20.0, "dca_max_count": 1, "dca_add_pct": 50.0,
        "use_switching": True, "switch_trigger": -12.0,  # (옵션만 유지: 기존 코드도 로직 미구현)
        "no_trade_weekend": False,

        # 연속손실 보호
        "loss_pause_enable": True, "loss_pause_after": 3, "loss_pause_minutes": 30,

        # AI 추천
        "ai_reco_show": True,
        "ai_reco_apply": False,
        "ai_reco_refresh_sec": 20,
        "ai_easy_korean": True,

        # 🌍 외부 시황 통합
        "use_external_context": True,
        "macro_blackout_minutes": 30,
        "external_refresh_sec": 60,
        "news_enable": True,
        "news_refresh_sec": 300,
        "news_max_headlines": 12,
        # 뉴스 한글화는 외부 네트워크(번역기) 사용 시 느려질 수 있어 시간 예산을 둔다.
        # - 예산 초과 시 남은 헤드라인은 룰 기반 보정(_translate_ko_rule)만 적용
        "news_translate_budget_sec": 10,
        "external_koreanize_enable": True,
        "external_ai_translate_enable": False,  # 외부시황 번역에 AI 사용(비용↑, 기본 OFF)

        # ✅ 매일 아침 BTC 경제뉴스 5개 브리핑
        "daily_btc_brief_enable": True,
        "daily_btc_brief_hour_kst": 9,
        "daily_btc_brief_minute_kst": 0,
        "daily_btc_brief_max_items": 5,
        "daily_btc_brief_ai_summarize": True,  # OpenAI 키 있을 때만 동작

        # ✅ 스타일(스캘핑/스윙) 자동 선택/전환
        # - regime_mode: Telegram /mode로도 변경 가능(auto|scalping|swing)
        # - regime_switch_control: 시간락 없이 흔들림 방지(confirm2/hysteresis/off)
        "regime_mode": "auto",                 # "auto"|"scalping"|"swing"
        "regime_switch_control": "confirm2",   # "confirm2"|"hysteresis"|"off"
        "regime_hysteresis_step": 0.55,
        "regime_hysteresis_enter_swing": 0.75,
        "regime_hysteresis_enter_scalp": 0.25,
        "style_auto_enable": True,
        "style_lock_minutes": 20,  # 전환 최소 유지 시간
        "scalp_max_hold_minutes": 25,          # 스캘핑 포지션 최대 보유(넘으면 스윙 전환 검토)
        "scalp_to_swing_min_roi": -12.0,       # 너무 큰 손실이면 전환 대신 정리 유도(기본)
        "scalp_to_swing_require_long_align": True,  # 장기추세까지 맞아야 스윙 전환
        "scalp_disable_dca": True,             # 스캘핑은 기본 추매 금지
        "scalp_tp_roi_min": 0.8,
        "scalp_tp_roi_max": 6.0,
        "scalp_sl_roi_min": 0.8,
        "scalp_sl_roi_max": 5.0,
        "scalp_entry_pct_mult": 0.65,
        "scalp_lev_cap": 8,

        "swing_tp_roi_min": 3.0,
        "swing_tp_roi_max": 50.0,
        "swing_sl_roi_min": 1.5,
        "swing_sl_roi_max": 30.0,
        "swing_entry_pct_mult": 1.0,
        "swing_lev_cap": 25,

        # ✅ 스윙: 부분익절/순환매도(옵션)
        "swing_partial_tp_enable": True,
        # TP(목표익절)의 비율로 단계 실행(예: TP의 35% 도달 시 1차 부분익절)
        "swing_partial_tp1_at_tp_frac": 0.35, "swing_partial_tp1_close_pct": 33,
        "swing_partial_tp2_at_tp_frac": 0.60, "swing_partial_tp2_close_pct": 33,
        "swing_partial_tp3_at_tp_frac": 0.85, "swing_partial_tp3_close_pct": 34,

        "swing_recycle_enable": False,
        "swing_recycle_cooldown_min": 20,
        "swing_recycle_max_count": 2,
        "swing_recycle_reentry_roi": 0.8,

        # ✅ 외부 시황 위험 시 신규진입 감산(완전 금지 X)
        "entry_risk_reduce_enable": True,
        "entry_risk_reduce_factor": 0.65,

        # ✅ 지지/저항(SR) 기반 손절/익절
        "use_sr_stop": True,
        "sr_timeframe": "15m",
        "sr_lookback": 220,
        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,
        "sr_rr_min": 1.5,
        "sr_levels_cache_sec": 60,

        # ✅ 추세 필터 정책(기능 유지/확장)
        "trend_filter_enabled": True,
        "trend_filter_timeframe": "1h",
        "trend_filter_cache_sec": 60,
        # "STRICT"=기존처럼 역추세 금지, "ALLOW_SCALP"=역추세 허용하되 스캘핑 강제, "OFF"=미사용
        "trend_filter_policy": "ALLOW_SCALP",

        # ✅ 내보내기(일별 엑셀/구글시트)
        "export_daily_enable": True,
        "export_excel_enable": True,
        "export_gsheet_enable": False,  # secrets 설정 필요
        "export_gsheet_spreadsheet_id": "",  # 비워두면 secrets의 GSHEET_ID 사용
    }


def load_settings() -> Dict[str, Any]:
    cfg = default_settings()
    if os.path.exists(SETTINGS_FILE):
        saved = read_json_safe(SETTINGS_FILE, {})
        if isinstance(saved, dict):
            cfg.update(saved)
    # 이전 키 호환
    if "openai_key" in cfg and not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = cfg["openai_key"]
    # 누락 키 보정
    base = default_settings()
    for k, v in base.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def save_settings(cfg: Dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, cfg)


config = load_settings()


# =========================================================
# ✅ 5) 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime() -> Dict[str, Any]:
    return {
        "date": today_kst_str(),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {},
        # ✅ 일별 브리핑/내보내기/상태 보존
        "daily_btc_brief": {},
        "last_export_date": "",
        "open_targets": {},  # sym -> active_targets snapshot
        # ✅ Telegram /scan 강제 스캔 요청
        "force_scan": {},
    }


def load_runtime() -> Dict[str, Any]:
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()
    if rt.get("date") != today_kst_str():
        # 날짜 바뀌면 일일 상태 초기화(기존 유지)
        rt = default_runtime()
    base = default_runtime()
    for k, v in base.items():
        if k not in rt:
            rt[k] = v
    return rt


def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ 6) 매매일지 CSV (기존 유지 + 표시용 이모티콘/내보내기 확장)
# =========================================================
def _read_csv_header_cols(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = (f.readline() or "").strip()
        if header.startswith("\ufeff"):
            header = header.lstrip("\ufeff")
        cols = [c.strip() for c in header.split(",") if c.strip()]
        return cols
    except Exception:
        return []


def log_trade(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_amount: float,
    pnl_percent: float,
    reason: str,
    one_line: str = "",
    review: str = "",
    trade_id: str = "",
) -> None:
    # ⚠️ CSV 컬럼 호환성 유지: 기존 컬럼 유지하면서 안전하게 append
    base_cols = ["Time", "Coin", "Side", "Entry", "Exit", "PnL_USDT", "PnL_Percent", "Reason", "OneLine", "Review", "TradeID"]
    try:
        row_dict = {
            "Time": now_kst_str(),
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Reason": reason,
            "OneLine": one_line,
            "Review": review,
            "TradeID": trade_id,
        }

        if not os.path.exists(LOG_FILE):
            pd.DataFrame([row_dict], columns=base_cols).to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            existing_cols = _read_csv_header_cols(LOG_FILE)
            cols = existing_cols if existing_cols else base_cols
            # 기존 파일 헤더와 컬럼 순서 맞춤(누락값은 공백)
            out = {c: row_dict.get(c, "") for c in cols}
            pd.DataFrame([out], columns=cols).to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    # ✅ Google Sheets 매매일지(요구사항: TRADE 이벤트) - CSV와 동일한 정보를 payload로 남김
    try:
        if gsheet_is_enabled():
            gsheet_log_trade(
                stage="JOURNAL",
                symbol=str(coin or ""),
                trade_id=str(trade_id or ""),
                message=str(reason or "")[:160],
                payload={
                    "time_kst": row_dict.get("Time"),
                    "coin": row_dict.get("Coin"),
                    "side": row_dict.get("Side"),
                    "entry": row_dict.get("Entry"),
                    "exit": row_dict.get("Exit"),
                    "pnl_usdt": row_dict.get("PnL_USDT"),
                    "pnl_pct": row_dict.get("PnL_Percent"),
                    "reason": row_dict.get("Reason"),
                    "one_line": row_dict.get("OneLine"),
                    "review": str(row_dict.get("Review", ""))[:800],
                    "trade_id": row_dict.get("TradeID"),
                },
            )
    except Exception:
        pass


def read_trade_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def reset_trade_log() -> None:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception:
        pass


def get_past_mistakes_text(max_items: int = 5) -> str:
    df = read_trade_log()
    if df.empty or "PnL_Percent" not in df.columns:
        return "과거 매매 기록 없음."
    try:
        worst = df.sort_values("PnL_Percent", ascending=True).head(max_items)
        lines = []
        for _, r in worst.iterrows():
            lines.append(
                f"- {r.get('Coin','?')} {r.get('Side','?')} {float(r.get('PnL_Percent',0)):.2f}% 손실 | 이유: {str(r.get('Reason',''))[:40]}"
            )
        return "\n".join(lines) if lines else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# =========================================================
# ✅ 6.5) 일별 내보내기(엑셀/구글시트)
# =========================================================
def _day_df_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df is None or df.empty or "Time" not in df.columns:
        return pd.DataFrame()
    try:
        # Time이 "YYYY-MM-DD HH:MM:SS"
        return df[df["Time"].astype(str).str.startswith(str(date_str))].copy()
    except Exception:
        return pd.DataFrame()


def _trade_day_summary(df_day: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "date": today_kst_str(),
        "trades": 0,
        "win_rate_pct": 0.0,
        "total_pnl_usdt": 0.0,
        "avg_pnl_pct": 0.0,
        "max_dd_pct": 0.0,
        "profit_factor": 0.0,
    }
    if df_day is None or df_day.empty:
        return out
    try:
        pnl_pct = pd.to_numeric(df_day.get("PnL_Percent", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        pnl_usdt = pd.to_numeric(df_day.get("PnL_USDT", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        out["trades"] = int(len(df_day))
        wins = (pnl_pct > 0).sum()
        out["win_rate_pct"] = float(wins / max(1, len(df_day)) * 100.0)
        out["total_pnl_usdt"] = float(pnl_usdt.sum())
        out["avg_pnl_pct"] = float(pnl_pct.mean())
        # 간이 MDD: 누적 PnL% 기준(정확한 equity curve는 아님)
        eq = pnl_pct.cumsum()
        dd = (eq - eq.cummax()).min() if len(eq) else 0.0
        out["max_dd_pct"] = float(dd)
        gains = pnl_usdt[pnl_usdt > 0].sum()
        losses = (-pnl_usdt[pnl_usdt < 0]).sum()
        out["profit_factor"] = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        return out
    except Exception:
        return out


def export_trade_log_daily(date_str: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    - daily_reports/ 아래 날짜별 파일 생성
    - openpyxl 있으면 xlsx, 없으면 csv로 fallback
    - Google Sheets는 설정/시크릿 있을 때만
    """
    df = read_trade_log()
    df_day = _day_df_filter(df, date_str)
    summary = _trade_day_summary(df_day)
    summary["date"] = date_str
    out = {"ok": True, "date": date_str, "rows": int(len(df_day)), "excel_path": "", "csv_path": "", "gsheet": ""}

    try:
        # 표시용 이모티콘 컬럼 추가(파일 내보내기에도 반영)
        if df_day is not None and not df_day.empty and "PnL_Percent" in df_day.columns:
            pnl_pct = pd.to_numeric(df_day["PnL_Percent"], errors="coerce")
            df_day = df_day.copy()
            df_day.insert(
                0,
                "상태",
                pnl_pct.apply(lambda v: "🟢 수익" if pd.notna(v) and float(v) > 0 else ("🔴 손실" if pd.notna(v) and float(v) < 0 else "⚪ 보합")),
            )
    except Exception:
        pass

    if not cfg.get("export_daily_enable", True):
        out["ok"] = False
        out["error"] = "export_daily_enable=OFF"
        return out

    # Excel
    if cfg.get("export_excel_enable", True):
        try:
            xlsx_path = os.path.join(DAILY_REPORT_DIR, f"trade_log_{date_str}.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
                df_day.to_excel(writer, sheet_name="trades", index=False)
            out["excel_path"] = xlsx_path
        except Exception as e:
            out["excel_path"] = ""
            out["excel_error"] = str(e)

    # CSV fallback(항상 생성해두면 편함)
    try:
        csv_path = os.path.join(DAILY_REPORT_DIR, f"trade_log_{date_str}.csv")
        df_day.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out["csv_path"] = csv_path
    except Exception:
        pass

    # Google Sheets (optional)
    # - 요구사항: GSHEET_ENABLED == "true" 일 때만 동작
    if cfg.get("export_gsheet_enable", False) and str(st.secrets.get("GSHEET_ENABLED", "")).strip().lower() == "true":
        try:
            res = export_trade_log_to_gsheet(date_str, df_day, summary, cfg)
            out["gsheet"] = res.get("msg", "")
            if not res.get("ok", False):
                out["gsheet_error"] = res.get("error", "")
        except Exception as e:
            out["gsheet_error"] = str(e)
    elif cfg.get("export_gsheet_enable", False):
        out["gsheet_error"] = "GSHEET_ENABLED != 'true'"

    return out


def _get_gsheet_client_from_secrets() -> Optional[Any]:
    """
    Streamlit secrets 예시:
    - [gcp_service_account] (dict 형태)
    - 혹은 GOOGLE_SERVICE_ACCOUNT_JSON (JSON 문자열)
    """
    if gspread is None or GoogleCredentials is None:
        return None
    try:
        # ✅ 요구사항: GSHEET_SERVICE_ACCOUNT_JSON (멀티라인 포함) 지원
        info = None
        try:
            info = _gsheet_service_account_info()  # type: ignore[name-defined]
        except Exception:
            info = None
        if not info:
            return None
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def export_trade_log_to_gsheet(date_str: str, df_day: pd.DataFrame, summary: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if gspread is None:
        return {"ok": False, "error": "gspread 미설치(requirements.txt에 gspread/google-auth 추가)", "msg": ""}
    if str(st.secrets.get("GSHEET_ENABLED", "")).strip().lower() != "true":
        return {"ok": False, "error": "GSHEET_ENABLED != 'true'", "msg": ""}
    client = _get_gsheet_client_from_secrets()
    if client is None:
        return {"ok": False, "error": "Google 서비스 계정 secrets 없음", "msg": ""}

    sid = (cfg.get("export_gsheet_spreadsheet_id") or "").strip() or str(st.secrets.get("GSHEET_SPREADSHEET_ID") or "").strip() or str(st.secrets.get("GSHEET_ID") or "").strip()
    if not sid:
        return {"ok": False, "error": "GSHEET_SPREADSHEET_ID 미설정(secrets 또는 설정)", "msg": ""}

    try:
        sh = client.open_by_key(sid)
        # 날짜별 워크시트 생성/갱신
        title = str(date_str)
        try:
            ws = sh.worksheet(title)
        except Exception:
            ws = sh.add_worksheet(title=title, rows=2000, cols=30)

        # summary 먼저
        ws.clear()
        sum_rows = [["key", "value"]] + [[k, str(v)] for k, v in summary.items()]
        ws.update("A1", sum_rows)

        # trades 테이블
        start_row = len(sum_rows) + 2
        if df_day is not None and not df_day.empty:
            df2 = df_day.copy()
            df2 = df2.fillna("")
            values = [df2.columns.tolist()] + df2.astype(str).values.tolist()
            ws.update(f"A{start_row}", values)
        return {"ok": True, "msg": f"Google Sheets 업데이트 완료({title})"}
    except Exception as e:
        return {"ok": False, "error": str(e), "msg": ""}


# =========================================================
# ✅ 7) Secrets (Bitget / Telegram / OpenAI)
# =========================================================
try:
    # 일부 환경/스레드에서 st.secrets 접근이 예외가 날 수 있어 fallback 스냅샷 유지
    _SECRETS_SNAPSHOT = dict(st.secrets)
except Exception:
    _SECRETS_SNAPSHOT = {}


def _sget(key: str, default: Any = "") -> Any:
    try:
        return st.secrets.get(key, default)
    except Exception:
        try:
            return _SECRETS_SNAPSHOT.get(key, default)
        except Exception:
            return default


def _sget_str(key: str, default: str = "") -> str:
    try:
        v = _sget(key, default)
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return str(default).strip()


def _parse_id_set(csv_like: str) -> set:
    s = str(csv_like or "").strip()
    if not s:
        return set()
    out = set()
    for p in re.split(r"[,\s]+", s):
        p = p.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _boolish(v: Any) -> bool:
    return str(v or "").strip().lower() in ["true", "1", "yes", "y", "on"]


# ✅ Bitget Secrets (요구사항 규격)
api_key = _sget_str("BITGET_API_KEY") or _sget_str("API_KEY")
api_secret = _sget_str("BITGET_API_SECRET") or _sget_str("API_SECRET")
api_password = _sget_str("BITGET_API_PASSPHRASE") or _sget_str("API_PASSWORD")

# ✅ Telegram Secrets (요구사항 규격)
tg_token = _sget_str("TG_TOKEN")
tg_target_chat_id = _sget_str("TG_TARGET_CHAT_ID") or _sget_str("TG_CHAT_ID")

# (확장) TG_CHANNEL_ID / TG_GROUP_ID가 있으면 자동 감지해 라우팅
tg_channel_id = _sget_str("TG_CHANNEL_ID") or _sget_str("TG_CHAT_ID_CHANNEL") or _sget_str("TG_CHAT_ID_CHANNEL_ID")
tg_group_id = _sget_str("TG_GROUP_ID") or _sget_str("TG_CHAT_ID_GROUP") or _sget_str("TG_CHAT_ID_GROUP_ID")

tg_id_default = tg_target_chat_id
if tg_channel_id or tg_group_id:
    tg_id_channel = tg_channel_id or tg_target_chat_id
    tg_id_group = tg_group_id or tg_target_chat_id
else:
    tg_id_channel = tg_target_chat_id
    tg_id_group = tg_target_chat_id

TG_ADMIN_IDS = _parse_id_set(_sget_str("TG_ADMIN_USER_IDS"))

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. Secrets에 BITGET_API_KEY/BITGET_API_SECRET/BITGET_API_PASSPHRASE 설정하세요.")
    st.stop()


_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}
_OPENAI_CLIENT_LOCK = threading.RLock()


# =========================================================
# ✅ OpenAI Health/Suspension (쿼터/레이트리밋 대응)
# - 429(insufficient_quota) 같은 오류가 반복되면 스캔/스레드가 "계속 오류"처럼 보일 수 있어
#   일정 시간 OpenAI 호출을 자동 중지(suspend)해서 스팸/부하를 줄인다.
# - 키를 바꾸면(suffix/len 변화) 자동으로 suspend를 해제한다.
# =========================================================
_OPENAI_HEALTH_LOCK = threading.RLock()
_OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
_OPENAI_SUSPENDED_REASON = ""
_OPENAI_SUSPENDED_KEY_FPR = ""
_OPENAI_LAST_ERROR_SUMMARY = ""
_OPENAI_LAST_ERROR_EPOCH = 0.0


def _openai_key_fingerprint(key: str) -> str:
    try:
        k = str(key or "")
        if not k:
            return ""
        suf = k[-4:] if len(k) >= 4 else k
        return f"len{len(k)}..{suf}"
    except Exception:
        return ""


def _openai_err_kind(err: BaseException) -> str:
    """
    OpenAI 오류를 대략 분류(라이브러리 버전 차이/에러 형태 차이를 흡수).
    """
    try:
        name = str(type(err).__name__ or "").lower()
    except Exception:
        name = ""
    try:
        s = str(err or "").lower()
    except Exception:
        s = ""

    # quota/결제 부족
    if "insufficient_quota" in s or "exceeded your current quota" in s or "plan and billing" in s:
        return "insufficient_quota"
    # 잘못된 키
    if "invalid_api_key" in s or "incorrect api key" in s or "api key" in s and "invalid" in s:
        return "invalid_api_key"
    # rate limit
    if "ratelimit" in name or ("rate limit" in s and "insufficient_quota" not in s):
        return "rate_limit"
    # timeout
    if "timeout" in s or "timed out" in s:
        return "timeout"
    return "other"


def openai_health_info(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    returns:
      - available: bool
      - status: OK|NO_KEY|SUSPENDED
      - message: human readable(KO)
      - until_kst: str (when suspended)
    """
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    if not key:
        return {"available": False, "status": "NO_KEY", "message": "OpenAI 키 없음", "until_kst": ""}

    fpr = _openai_key_fingerprint(key)
    now = time.time()
    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        # 키가 바뀌면 suspend 해제
        if _OPENAI_SUSPENDED_KEY_FPR and _OPENAI_SUSPENDED_KEY_FPR != fpr:
            _OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
            _OPENAI_SUSPENDED_REASON = ""
            _OPENAI_SUSPENDED_KEY_FPR = ""

        if now < float(_OPENAI_SUSPENDED_UNTIL_EPOCH or 0.0) and _OPENAI_SUSPENDED_KEY_FPR == fpr:
            until_kst = _epoch_to_kst_str(float(_OPENAI_SUSPENDED_UNTIL_EPOCH))
            reason = str(_OPENAI_SUSPENDED_REASON or "").strip() or "일시 중지"
            return {"available": False, "status": "SUSPENDED", "message": f"OpenAI 일시중지: {reason}", "until_kst": until_kst}

    return {"available": True, "status": "OK", "message": "OpenAI OK", "until_kst": ""}


def openai_suspend(cfg: Optional[Dict[str, Any]], reason: str, duration_sec: int, err: Optional[BaseException] = None) -> None:
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    fpr = _openai_key_fingerprint(key)
    until = time.time() + float(max(5, int(duration_sec)))
    msg_err = ""
    try:
        msg_err = str(err)[:240] if err is not None else ""
    except Exception:
        msg_err = ""

    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        global _OPENAI_LAST_ERROR_SUMMARY, _OPENAI_LAST_ERROR_EPOCH
        _OPENAI_SUSPENDED_UNTIL_EPOCH = float(until)
        _OPENAI_SUSPENDED_REASON = str(reason or "").strip()[:120]
        _OPENAI_SUSPENDED_KEY_FPR = str(fpr or "")
        _OPENAI_LAST_ERROR_SUMMARY = msg_err
        _OPENAI_LAST_ERROR_EPOCH = time.time()

    try:
        gsheet_log_event(
            "OPENAI_SUSPEND",
            message=str(reason or "suspend"),
            payload={"until_kst": _epoch_to_kst_str(float(until)), "duration_sec": int(duration_sec), "err": msg_err},
        )
    except Exception:
        pass


def openai_handle_failure(err: BaseException, cfg: Optional[Dict[str, Any]], where: str = "") -> str:
    """
    OpenAI 실패를 분류하고, 필요 시 suspend 설정.
    returns: kind string
    """
    kind = _openai_err_kind(err)
    # quota 부족은 모델을 바꿔도 해결되지 않으므로 길게 suspend
    if kind == "insufficient_quota":
        openai_suspend(cfg, reason="insufficient_quota(API 결제/크레딧: ChatGPT와 별개)", duration_sec=6 * 60 * 60, err=err)
    elif kind == "invalid_api_key":
        openai_suspend(cfg, reason="invalid_api_key(키 오류)", duration_sec=10 * 60, err=err)
    elif kind == "rate_limit":
        openai_suspend(cfg, reason="rate_limit(잠시 대기)", duration_sec=120, err=err)
    elif kind == "timeout":
        openai_suspend(cfg, reason="timeout(잠시 대기)", duration_sec=60, err=err)
    else:
        # 기타 오류도 짧게 suspend 해서 스팸/부하 완화
        openai_suspend(cfg, reason="openai_error(잠시 대기)", duration_sec=45, err=err)
    return kind


def openai_clear_suspension(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    수동 테스트/운영자가 결제/쿼터를 복구한 직후 즉시 재시도할 수 있게 suspend를 해제.
    - 자동매매/스캔 루프에서는 사용하지 않는 것이 안전.
    """
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    fpr = _openai_key_fingerprint(key)
    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        if not _OPENAI_SUSPENDED_KEY_FPR:
            return
        if fpr and _OPENAI_SUSPENDED_KEY_FPR != fpr:
            # 다른 키면 이미 openai_health_info()에서 자동 해제되지만, 안전하게 클리어
            pass
        _OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
        _OPENAI_SUSPENDED_REASON = ""
        _OPENAI_SUSPENDED_KEY_FPR = ""
    try:
        gsheet_log_event("OPENAI_UNSUSPEND", message="manual_clear", payload={"code": CODE_VERSION})
    except Exception:
        pass


def get_openai_client(cfg: Dict[str, Any]) -> Optional[OpenAI]:
    # ✅ secrets 규격(요구사항): OPENAI_API_KEY
    # - 일부 환경에서 st.secrets.get 호환 이슈를 피하기 위해 _sget_str 사용
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    if not key:
        return None
    # suspend 상태면 호출하지 않음(스팸/부하 방지)
    try:
        h = openai_health_info(cfg)
        if not bool(h.get("available", False)):
            return None
    except Exception:
        pass
    with _OPENAI_CLIENT_LOCK:
        if key in _OPENAI_CLIENT_CACHE:
            return _OPENAI_CLIENT_CACHE[key]
        try:
            c = OpenAI(api_key=key)
            _OPENAI_CLIENT_CACHE[key] = c
            return c
        except Exception:
            return None


def _call_with_timeout(fn, timeout_sec: int):
    # 스레드가 멈추는 걸 방지하기 위해 OpenAI 같은 외부 호출에 hard-timeout을 건다.
    fut = _THREAD_POOL.submit(fn)
    return fut.result(timeout=timeout_sec)


def openai_chat_create_with_fallback(
    client: OpenAI,
    models: List[str],
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_sec: int = OPENAI_TIMEOUT_SEC,
) -> Tuple[str, Any]:
    """
    OpenAI 호출 모델 fallback:
    - 일부 계정/환경에서 특정 모델이 없을 수 있어(예: gpt-4o 미지원) 순차 시도
    - 성공 시 (model_used, response) 반환
    """
    last_err: Optional[BaseException] = None
    tried: List[str] = []
    for m in models:
        m2 = str(m or "").strip()
        if not m2:
            continue
        tried.append(m2)
        try:
            def _do(use_response_format: bool = True):
                kwargs: Dict[str, Any] = {
                    "model": m2,
                    "messages": messages,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                }
                if response_format is not None and use_response_format:
                    kwargs["response_format"] = response_format
                return client.chat.completions.create(**kwargs)

            resp = _call_with_timeout(_do, timeout_sec)
            return m2, resp
        except FuturesTimeoutError as e:
            last_err = e
            continue
        except TypeError as e:
            # 일부 openai 라이브러리/환경에서 response_format 파라미터가 지원되지 않을 수 있음
            # (예: "got an unexpected keyword argument 'response_format'")
            msg = str(e or "")
            if response_format is not None and ("response_format" in msg):
                try:
                    resp = _call_with_timeout(lambda: _do(use_response_format=False), timeout_sec)
                    return m2, resp
                except Exception as e2:
                    last_err = e2
                    continue
            last_err = e
            continue
        except Exception as e:
            # 모델 자체가 response_format을 지원하지 않는 경우도 있어, 1회는 response_format 없이 재시도
            msg = str(e or "")
            if response_format is not None and ("response_format" in msg.lower()):
                try:
                    resp = _call_with_timeout(lambda: _do(use_response_format=False), timeout_sec)
                    return m2, resp
                except Exception as e2:
                    last_err = e2
                    continue
            # quota/키오류 등은 모델 바꿔도 해결되지 않으므로 즉시 중단
            kind = ""
            try:
                kind = _openai_err_kind(e)
            except Exception:
                kind = ""
            if kind in ["insufficient_quota", "invalid_api_key"]:
                raise e
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"OpenAI call failed (models_tried={tried})")


# =========================================================
# ✅ 7.5) Google Sheets Logger (TRADE/EVENT/SCAN) - 요구사항 필수
# - GSHEET_ENABLED == "true" 일 때만 동작
# - 네트워크 오류가 나도 봇이 죽지 않게 retry/예외처리
# - append_row 방식으로 누적 기록
# =========================================================
GSHEET_HEADER = ["time_kst", "type", "stage", "symbol", "tf", "signal", "score", "trade_id", "message", "payload_json"]

# ✅ SCAN은 빈도가 매우 높을 수 있으니, TRADE/EVENT를 우선 처리(요구사항)
_GSHEET_QUEUE_HIGH = deque()  # TRADE/EVENT
_GSHEET_QUEUE_SCAN = deque()  # SCAN
_GSHEET_QUEUE_LOCK = threading.RLock()
_GSHEET_CACHE_LOCK = threading.RLock()
_GSHEET_CACHE: Dict[str, Any] = {
    "ws": None,
    "header_ok": False,
    "last_init_epoch": 0.0,
    "last_err": "",
    "last_tb": "",
    "service_account_email": "",
    "worksheet": "",
    "spreadsheet_id": "",
    "last_append_epoch": 0.0,
    "last_append_kst": "",
    "last_append_type": "",
    "last_append_stage": "",
}

_GSHEET_NOTIFY_LOCK = threading.RLock()
_GSHEET_LAST_NOTIFY_EPOCH = 0.0
_GSHEET_LAST_NOTIFY_MSG = ""

# ✅ SCAN 로그는 빈도가 매우 높아 Google Sheets API rate-limit(429)을 유발할 수 있음
# - UI(monitor_state.json)에는 전체 SCAN 과정을 남기되,
# - 시트에는 stage/심볼별로 일정 간격(throttle) 샘플링해서 누적한다.
_GSHEET_SCAN_THROTTLE_LOCK = threading.RLock()
_GSHEET_SCAN_LAST: Dict[str, float] = {}
_GSHEET_SCAN_THROTTLE_SEC = 8.0
_GSHEET_SCAN_THROTTLE_MAX_KEYS = 1500
_GSHEET_SCAN_ALWAYS_STAGES = {
    "ai_result",
    "trade_opened",
    "trade_skipped",
    "in_position",
    "ai_error",
    "fetch_short_fail",
    "fetch_long_fail",
    "support_resistance_fail",
}


def gsheet_is_enabled() -> bool:
    # secrets 우선 (요구사항)
    return _boolish(_sget_str("GSHEET_ENABLED"))


def _gsheet_get_settings() -> Dict[str, str]:
    sid = _sget_str("GSHEET_SPREADSHEET_ID") or _sget_str("GSHEET_ID")
    ws_name = _sget_str("GSHEET_WORKSHEET") or "BOT_LOG"
    sa_json = _sget_str("GSHEET_SERVICE_ACCOUNT_JSON") or _sget_str("GOOGLE_SERVICE_ACCOUNT_JSON")
    return {"spreadsheet_id": sid, "worksheet": ws_name, "service_account_json": sa_json}


def _gsheet_service_account_info() -> Optional[Dict[str, Any]]:
    """
    secrets 형태 다양성 흡수:
    - GSHEET_SERVICE_ACCOUNT_JSON: JSON 문자열(요구사항)
    - 혹시 dict로 넣은 경우도 방어적으로 지원
    - (호환) [gcp_service_account] dict
    """
    try:
        raw = _sget("GSHEET_SERVICE_ACCOUNT_JSON", None)
        if raw is None or raw == "":
            raw = _sget("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        if (raw is None or raw == "") and ("gcp_service_account" in st.secrets) and isinstance(st.secrets.get("gcp_service_account"), dict):
            return dict(st.secrets.get("gcp_service_account") or {})
        if isinstance(raw, dict):
            return dict(raw)
        s = str(raw or "").strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception as je:
            # 스트림릿 secrets(TOML)에서 "\n" escape가 실제 개행으로 풀리면 JSON이 깨질 수 있음
            # → 사용자에게 원인을 알려주기 위해 last_err를 남긴다.
            try:
                _GSHEET_CACHE["last_err"] = f"GSHEET_SERVICE_ACCOUNT_JSON 파싱 실패: {je}"
            except Exception:
                pass
            # ✅ Streamlit secrets(TOML) 멀티라인 문자열에서 private_key의 개행이 실제 개행으로 들어오면 JSON이 깨짐
            #    → private_key 값 내부의 개행만 \\n으로 이스케이프해서 1회 복구 시도
            try:
                if s.startswith("{") and '"private_key"' in s and ("\n" in s or "\r" in s):
                    m = re.search(r'"private_key"\s*:\s*"(.*?)"', s, flags=re.S)
                    if m:
                        pk = m.group(1)
                        if "\n" in pk or "\r" in pk:
                            pk_fixed = pk.replace("\r\n", "\n").replace("\n", "\\n")
                            s2 = s[: m.start(1)] + pk_fixed + s[m.end(1) :]
                            v2 = json.loads(s2)
                            if isinstance(v2, dict):
                                try:
                                    _GSHEET_CACHE["last_err"] = ""
                                except Exception:
                                    pass
                                return dict(v2)
            except Exception:
                pass
            try:
                import ast as _ast

                # 혹시 dict가 str()로 변환돼 들어온 경우(단, 안전한 literal_eval만 사용)
                if s.startswith("{") and ("'client_email'" in s or "'private_key'" in s):
                    v = _ast.literal_eval(s)
                    if isinstance(v, dict):
                        return dict(v)
            except Exception:
                pass
            return None
    except Exception:
        return None


def _gsheet_service_account_email() -> str:
    try:
        info = _gsheet_service_account_info() or {}
        return str(info.get("client_email") or "").strip()
    except Exception:
        return ""


def _gsheet_exception_detail(err: BaseException, limit: int = 900) -> str:
    """
    gspread/google 오류는 str(err)가 비어있거나(특히 SpreadsheetNotFound/RetryError),
    APIError의 response 본문에만 정보가 있는 경우가 있어 최대한 detail을 뽑아낸다.
    """
    try:
        name = str(type(err).__name__ or "Exception")
    except Exception:
        name = "Exception"
    try:
        msg = str(err or "").strip()
    except Exception:
        msg = ""
    detail = msg if msg else name

    # gspread.exceptions.APIError / googleapiclient.errors.HttpError 등 response/body가 있는 케이스
    try:
        resp = getattr(err, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            text = getattr(resp, "text", None)
            if code:
                detail = f"{detail} | http={code}"
            if text:
                t = str(text).strip()
                if t:
                    t = t.replace("\n", " ")[:400]
                    detail = f"{detail} | body={t}"
    except Exception:
        pass

    # 일부 에러는 args[0]에 dict로 내려오는 경우가 있음
    try:
        if not msg and getattr(err, "args", None):
            a0 = err.args[0]
            if isinstance(a0, dict):
                s = safe_json_dumps(a0, limit=420).replace("\n", " ")
                if s:
                    detail = f"{name} | {s}"
    except Exception:
        pass

    try:
        if len(detail) > int(limit):
            detail = detail[: int(limit)] + "..."
    except Exception:
        pass
    return detail


def _gsheet_notify_connect_issue(where: str, msg: str, min_interval_sec: float = 300.0) -> None:
    """
    Google Sheets 연결/권한 문제를 관리자 DM으로 안내(과다 스팸 방지).
    """
    try:
        if not TG_ADMIN_IDS:
            return
        now = time.time()
        with _GSHEET_NOTIFY_LOCK:
            global _GSHEET_LAST_NOTIFY_EPOCH, _GSHEET_LAST_NOTIFY_MSG
            if (now - float(_GSHEET_LAST_NOTIFY_EPOCH or 0.0)) < float(min_interval_sec):
                return
            if msg and msg == _GSHEET_LAST_NOTIFY_MSG and (now - float(_GSHEET_LAST_NOTIFY_EPOCH or 0.0)) < float(min_interval_sec) * 2:
                return
            _GSHEET_LAST_NOTIFY_EPOCH = now
            _GSHEET_LAST_NOTIFY_MSG = msg
        stg = _gsheet_get_settings()
        email = _gsheet_service_account_email()
        hint = ""
        if email and stg.get("spreadsheet_id"):
            hint = f"\n- 서비스계정 이메일: {email}\n- 공유: 시트에 위 이메일을 '편집자'로 공유해야 합니다."
        tb_txt = ""
        try:
            tb_txt = str(_GSHEET_CACHE.get("last_tb", "") or "")
        except Exception:
            tb_txt = ""
        notify_admin_error(
            where,
            RuntimeError(msg),
            context={"spreadsheet_id": stg.get("spreadsheet_id", ""), "worksheet": stg.get("worksheet", ""), "service_account_email": email, "code": CODE_VERSION},
            tb=tb_txt,
            min_interval_sec=min_interval_sec,
        )
        if hint:
            tg_send(hint, target="admin")
    except Exception:
        pass


def _gsheet_connect_ws() -> Optional[Any]:
    if not gsheet_is_enabled():
        return None
    if gspread is None or GoogleCredentials is None:
        _GSHEET_CACHE["last_err"] = "gspread/google-auth 미설치(requirements.txt 확인)"
        return None

    stg = _gsheet_get_settings()
    sid = stg.get("spreadsheet_id", "").strip()
    ws_name = stg.get("worksheet", "BOT_LOG").strip() or "BOT_LOG"
    info = _gsheet_service_account_info()
    if not sid:
        _GSHEET_CACHE["last_err"] = "GSHEET_SPREADSHEET_ID 누락"
        return None
    if not info:
        cur = str(_GSHEET_CACHE.get("last_err", "") or "").strip()
        _GSHEET_CACHE["last_err"] = cur or "GSHEET_SERVICE_ACCOUNT_JSON 누락/파싱 실패"
        return None

    try:
        try:
            _GSHEET_CACHE["service_account_email"] = str((info or {}).get("client_email") or "").strip()
        except Exception:
            _GSHEET_CACHE["service_account_email"] = ""
        _GSHEET_CACHE["worksheet"] = ws_name
        _GSHEET_CACHE["spreadsheet_id"] = sid
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sid)
        try:
            ws = sh.worksheet(ws_name)
        except Exception:
            # 없으면 생성
            ws = sh.add_worksheet(title=ws_name, rows=5000, cols=len(GSHEET_HEADER) + 5)
        return ws
    except Exception as e:
        detail = _gsheet_exception_detail(e, limit=900)
        low = detail.lower()
        # 가장 흔한 케이스: 시트 미공유/ID 오타 → SpreadsheetNotFound(메시지 비어있을 수 있음)
        if "spreadsheetnotfound" in low:
            detail = "SpreadsheetNotFound (시트를 서비스계정 이메일에 '편집자'로 공유 + GSHEET_SPREADSHEET_ID 확인)"
        elif "permission" in low or "forbidden" in low:
            detail = f"권한 문제(Forbidden): 시트 공유/드라이브 권한/스코프 확인 | {detail}".strip()
        elif ("api" in low and "enable" in low) or "has not been used" in low:
            detail = f"API 활성화 필요: Google Sheets/Drive API | {detail}".strip()
        _GSHEET_CACHE["last_err"] = f"GSHEET 연결 실패: {detail}".strip()
        try:
            _GSHEET_CACHE["last_tb"] = traceback.format_exc()
        except Exception:
            _GSHEET_CACHE["last_tb"] = ""
        return None


def gsheet_status_snapshot() -> Dict[str, Any]:
    try:
        stg = _gsheet_get_settings()
        with _GSHEET_QUEUE_LOCK:
            qh = len(_GSHEET_QUEUE_HIGH)
            qs = len(_GSHEET_QUEUE_SCAN)
        with _GSHEET_CACHE_LOCK:
            last_init = float(_GSHEET_CACHE.get("last_init_epoch", 0) or 0)
            return {
                "enabled": bool(gsheet_is_enabled()),
                "spreadsheet_id": stg.get("spreadsheet_id", ""),
                "worksheet": stg.get("worksheet", ""),
                "service_account_email": _gsheet_service_account_email(),
                "connected": bool(_GSHEET_CACHE.get("ws", None) is not None),
                "header_ok": bool(_GSHEET_CACHE.get("header_ok", False)),
                "queue_high": qh,
                "queue_scan": qs,
                "last_init_kst": _epoch_to_kst_str(last_init) if last_init else "",
                "last_append_kst": str(_GSHEET_CACHE.get("last_append_kst", "") or ""),
                "last_append_type": str(_GSHEET_CACHE.get("last_append_type", "") or ""),
                "last_append_stage": str(_GSHEET_CACHE.get("last_append_stage", "") or ""),
                "last_err": str(_GSHEET_CACHE.get("last_err", "") or ""),
            }
    except Exception:
        return {"enabled": bool(gsheet_is_enabled()), "last_err": str(_GSHEET_CACHE.get("last_err", "") if isinstance(_GSHEET_CACHE, dict) else "")}


def gsheet_test_append_row(timeout_sec: int = 20) -> Dict[str, Any]:
    """
    수동 진단용:
    - 연결 + 헤더 + append_row를 즉시 수행해서 권한/설정 문제를 바로 확인.
    """
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gspread is None or GoogleCredentials is None:
        return {"ok": False, "error": "gspread/google-auth 미설치(requirements.txt 확인)"}

    def _do():
        ws = _gsheet_connect_ws()
        if ws is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "unknown_error")
            raise RuntimeError(err)
        _gsheet_ensure_header(ws)
        rec = {
            "time_kst": now_kst_str(),
            "type": "EVENT",
            "stage": "GSHEET_TEST",
            "symbol": "",
            "tf": "",
            "signal": "",
            "score": "",
            "trade_id": "",
            "message": f"manual_test code={CODE_VERSION}",
            "payload_json": {"code": CODE_VERSION},
        }
        row = [
            rec["time_kst"],
            rec["type"],
            rec["stage"],
            rec["symbol"],
            rec["tf"],
            rec["signal"],
            rec["score"],
            rec["trade_id"],
            rec["message"],
            safe_json_dumps(rec["payload_json"], limit=1800),
        ]
        ws.append_row(row, value_input_option="USER_ENTERED")
        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_append_epoch"] = time.time()
            _GSHEET_CACHE["last_append_kst"] = now_kst_str()
            _GSHEET_CACHE["last_append_type"] = "EVENT"
            _GSHEET_CACHE["last_append_stage"] = "GSHEET_TEST"
        return True

    try:
        _call_with_timeout(_do, timeout_sec)
        return {"ok": True}
    except Exception as e:
        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_err"] = f"GSHEET 테스트 실패: {e}"
        _gsheet_notify_connect_issue("GSHEET_TEST", f"GSHEET 테스트 실패: {e}", min_interval_sec=120.0)
        return {"ok": False, "error": str(e)}


def _gsheet_ensure_header(ws: Any) -> None:
    try:
        if _GSHEET_CACHE.get("header_ok"):
            return
        first = []
        try:
            first = ws.row_values(1)  # network
        except Exception:
            first = []
        if not first or (first and str(first[0]).strip().lower() != "time_kst"):
            ws.append_row(GSHEET_HEADER, value_input_option="USER_ENTERED")
        _GSHEET_CACHE["header_ok"] = True
    except Exception:
        pass


def gsheet_enqueue(rec: Dict[str, Any]) -> None:
    if not gsheet_is_enabled():
        return
    try:
        rr = dict(rec or {})
        rr.setdefault("time_kst", now_kst_str())
        rr.setdefault("type", "EVENT")
        rr.setdefault("stage", "")
        rr.setdefault("symbol", "")
        rr.setdefault("tf", "")
        rr.setdefault("signal", "")
        rr.setdefault("score", "")
        rr.setdefault("trade_id", "")
        rr.setdefault("message", "")
        payload = rr.get("payload_json", "")
        if not isinstance(payload, str):
            rr["payload_json"] = safe_json_dumps(payload, limit=1800)
        with _GSHEET_QUEUE_LOCK:
            typ = str(rr.get("type", "EVENT")).strip().upper()
            if typ in ["TRADE", "EVENT"]:
                _GSHEET_QUEUE_HIGH.append(rr)
                # 과도 누적 방지(중요 로그는 최대한 유지)
                while len(_GSHEET_QUEUE_HIGH) > 600:
                    _GSHEET_QUEUE_HIGH.popleft()
            else:
                _GSHEET_QUEUE_SCAN.append(rr)
                # 과도 누적 방지(SCAN은 오래된 것부터 버림)
                while len(_GSHEET_QUEUE_SCAN) > 1800:
                    _GSHEET_QUEUE_SCAN.popleft()
    except Exception:
        pass


def gsheet_log_trade(stage: str, symbol: str, trade_id: str = "", message: str = "", payload: Optional[Dict[str, Any]] = None):
    gsheet_enqueue(
        {
            "type": "TRADE",
            "stage": stage,
            "symbol": symbol,
            "trade_id": trade_id,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_log_event(stage: str, message: str = "", payload: Optional[Dict[str, Any]] = None):
    gsheet_enqueue(
        {
            "type": "EVENT",
            "stage": stage,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_log_scan(stage: str, symbol: str, tf: str = "", signal: str = "", score: Any = "", message: str = "", payload: Optional[Dict[str, Any]] = None):
    # ✅ SCAN은 매우 자주 발생하므로(특히 5코인 * 다단계),
    #    시트에는 stage/심볼별로 throttle 샘플링해서 API 에러/레이트리밋을 줄인다.
    try:
        stg = str(stage or "").strip()
        sym = str(symbol or "").strip()
        stg_key = stg.lower()
        # 중요한 단계는 무조건 기록
        if stg_key and (stg_key not in _GSHEET_SCAN_ALWAYS_STAGES) and sym and sym != "*":
            now = time.time()
            k = f"{sym}|{stg_key}"
            with _GSHEET_SCAN_THROTTLE_LOCK:
                last = float(_GSHEET_SCAN_LAST.get(k, 0) or 0)
                if (now - last) < float(_GSHEET_SCAN_THROTTLE_SEC):
                    return
                _GSHEET_SCAN_LAST[k] = now
                if len(_GSHEET_SCAN_LAST) > int(_GSHEET_SCAN_THROTTLE_MAX_KEYS):
                    # 오래된 것부터 정리(메모리 누수 방지)
                    for kk in sorted(_GSHEET_SCAN_LAST, key=_GSHEET_SCAN_LAST.get)[:300]:
                        _GSHEET_SCAN_LAST.pop(kk, None)
    except Exception:
        pass

    gsheet_enqueue(
        {
            "type": "SCAN",
            "stage": stage,
            "symbol": symbol,
            "tf": tf,
            "signal": signal,
            "score": score,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_worker_thread():
    backoff = 1.0
    while True:
        batch: List[Dict[str, Any]] = []
        batch_is_high = False  # TRADE/EVENT batch
        try:
            if not gsheet_is_enabled():
                time.sleep(2.0)
                continue

            # ✅ secrets 변경(시트/워크시트) 감지 시 즉시 재연결
            try:
                stg_now = _gsheet_get_settings()
                with _GSHEET_CACHE_LOCK:
                    sid_now = str(stg_now.get("spreadsheet_id", "") or "").strip()
                    ws_now = str(stg_now.get("worksheet", "") or "").strip()
                    sid_old = str(_GSHEET_CACHE.get("spreadsheet_id", "") or "").strip()
                    ws_old = str(_GSHEET_CACHE.get("worksheet", "") or "").strip()
                    if (sid_now and sid_old and sid_now != sid_old) or (ws_now and ws_old and ws_now != ws_old):
                        _GSHEET_CACHE["ws"] = None
                        _GSHEET_CACHE["header_ok"] = False
                        _GSHEET_CACHE["last_init_epoch"] = 0.0
                        _GSHEET_CACHE["last_err"] = ""
            except Exception:
                pass

            # ✅ batch pop: Google Sheets API 호출 수를 줄이기 위해 묶어서 append (rate limit 완화)
            with _GSHEET_QUEUE_LOCK:
                while _GSHEET_QUEUE_HIGH and len(batch) < 25:
                    batch.append(_GSHEET_QUEUE_HIGH.popleft())
                if batch:
                    batch_is_high = True
                else:
                    while _GSHEET_QUEUE_SCAN and len(batch) < 40:
                        batch.append(_GSHEET_QUEUE_SCAN.popleft())
            if not batch:
                time.sleep(0.35)
                continue

            # 연결 캐시
            ws = None
            with _GSHEET_CACHE_LOCK:
                ws = _GSHEET_CACHE.get("ws", None)
                last_init = float(_GSHEET_CACHE.get("last_init_epoch", 0) or 0)
                # 오래됐으면 재연결 시도(네트워크/세션 이슈 대비)
                if ws is None or (time.time() - last_init) > 60 * 30:
                    ws = _gsheet_connect_ws()
                    _GSHEET_CACHE["ws"] = ws
                    _GSHEET_CACHE["header_ok"] = False
                    _GSHEET_CACHE["last_init_epoch"] = time.time()

            if ws is None:
                # 연결 실패면 재시도 위해 되돌려놓고 backoff
                try:
                    msg = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 연결 실패")
                    _gsheet_notify_connect_issue("GSHEET_CONNECT", msg, min_interval_sec=300.0)
                except Exception:
                    pass
                with _GSHEET_QUEUE_LOCK:
                    if batch_is_high:
                        for r in reversed(batch):
                            _GSHEET_QUEUE_HIGH.appendleft(r)
                    else:
                        for r in reversed(batch):
                            _GSHEET_QUEUE_SCAN.appendleft(r)
                time.sleep(backoff)
                backoff = float(clamp(backoff * 1.4, 1.0, 12.0))
                continue

            _gsheet_ensure_header(ws)

            rows = []
            for rec in batch:
                rows.append(
                    [
                        str(rec.get("time_kst", "")),
                        str(rec.get("type", "")),
                        str(rec.get("stage", "")),
                        str(rec.get("symbol", "")),
                        str(rec.get("tf", "")),
                        str(rec.get("signal", "")),
                        str(rec.get("score", "")),
                        str(rec.get("trade_id", "")),
                        str(rec.get("message", ""))[:500],
                        str(rec.get("payload_json", ""))[:1800],
                    ]
                )

            def _append_batch():
                # gspread 5+ 에서 append_rows 지원(요청 수 절감)
                if hasattr(ws, "append_rows"):
                    return ws.append_rows(rows, value_input_option="USER_ENTERED")  # type: ignore[attr-defined]
                # fallback: 구버전은 append_row 루프
                for row in rows:
                    ws.append_row(row, value_input_option="USER_ENTERED")
                return True

            if retry is not None:
                @_retry_wrapper_append_row  # type: ignore  # defined below
                def _append_retry():
                    return _append_batch()

                _append_retry()
            else:
                _append_batch()

            try:
                last_rec = batch[-1] if batch else {}
                with _GSHEET_CACHE_LOCK:
                    _GSHEET_CACHE["last_append_epoch"] = time.time()
                    _GSHEET_CACHE["last_append_kst"] = now_kst_str()
                    _GSHEET_CACHE["last_append_type"] = str(last_rec.get("type", "") or "")
                    _GSHEET_CACHE["last_append_stage"] = str(last_rec.get("stage", "") or "")
                    _GSHEET_CACHE["last_err"] = ""
                    _GSHEET_CACHE["last_tb"] = ""
            except Exception:
                pass
            backoff = 1.0
        except Exception as e:
            # 실패해도 봇은 살아야 함(오류는 관리자에게 알림)
            try:
                with _GSHEET_CACHE_LOCK:
                    # tenacity RetryError는 메시지가 빈 경우가 많아, root cause를 뽑아서 보여준다.
                    root = e
                    wrapped = ""
                    try:
                        if type(e).__name__ == "RetryError" or hasattr(e, "last_attempt"):
                            la = getattr(e, "last_attempt", None)
                            if la is not None and hasattr(la, "exception"):
                                ex0 = la.exception()
                                if ex0 is not None:
                                    root = ex0
                                    wrapped = type(e).__name__
                    except Exception:
                        root = e
                        wrapped = ""
                    _GSHEET_CACHE["last_err"] = f"GSHEET append 실패: {_gsheet_exception_detail(root, limit=900)}"
                    _GSHEET_CACHE["last_tb"] = traceback.format_exc()
            except Exception:
                pass
            # 실패한 batch는 되돌려서(특히 TRADE/EVENT) 나중에 재시도
            try:
                if batch:
                    with _GSHEET_QUEUE_LOCK:
                        if batch_is_high:
                            for r in reversed(batch):
                                _GSHEET_QUEUE_HIGH.appendleft(r)
                        else:
                            for r in reversed(batch):
                                _GSHEET_QUEUE_SCAN.appendleft(r)
            except Exception:
                pass
            try:
                # 관리자 DM에 root cause를 최대한 전달
                with _GSHEET_CACHE_LOCK:
                    msg2 = str(_GSHEET_CACHE.get("last_err", "") or str(e))
                    tb2 = str(_GSHEET_CACHE.get("last_tb", "") or "")
                notify_admin_error(
                    "GSHEET_THREAD",
                    RuntimeError(msg2),
                    context={"batch_len": len(batch), "batch_is_high": bool(batch_is_high)},
                    tb=tb2,
                    min_interval_sec=120.0,
                )
            except Exception:
                notify_admin_error("GSHEET_THREAD", e, min_interval_sec=120.0)
            time.sleep(backoff)
            backoff = float(clamp(backoff * 1.5, 1.0, 12.0))


# tenacity가 있을 때만 사용하는 데코레이터를 늦게 정의(옵션 의존성)
def _retry_wrapper_append_row(fn):  # noqa: D401
    """append_row retry wrapper (tenacity optional)"""
    if retry is None:
        return fn

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1.0, max=6.0))
    def _inner():
        return fn()

    return _inner


# =========================================================
# ✅ 8) 거래소 연결
# =========================================================
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "timeout": 15000,  # 네트워크 hang 방지
                "options": {"defaultType": "swap"},
            }
        )
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except Exception:
        return None


def create_exchange_client_uncached() -> Optional[Any]:
    """
    ⚠️ ccxt exchange 객체는 스레드-세이프하지 않을 수 있어,
    Streamlit UI(메인 스레드)와 트레이딩 루프(TG_THREAD)를 분리하기 위한 전용 인스턴스.
    - 실패 시 None 반환(호출부에서 cached exchange로 fallback)
    """
    try:
        ex = ccxt.bitget(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "timeout": 15000,
                "options": {"defaultType": "swap"},
            }
        )
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except Exception:
        return None


exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키/권한/네트워크 확인.")
    st.stop()


# =========================================================
# ✅ 9) Bitget 헬퍼
# =========================================================
def safe_fetch_balance(ex) -> Tuple[float, float]:
    try:
        bal = ex.fetch_balance({"type": "swap"})
        free = float(bal["USDT"]["free"])
        total = float(bal["USDT"]["total"])
        return free, total
    except Exception:
        return 0.0, 0.0


def safe_fetch_positions(ex, symbols: List[str]) -> List[Dict[str, Any]]:
    try:
        return ex.fetch_positions(symbols)
    except TypeError:
        try:
            return ex.fetch_positions(symbols=symbols)
        except Exception:
            return []
    except Exception:
        return []


def get_last_price(ex, sym: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(sym)
        return float(t["last"])
    except Exception:
        return None


def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo


def to_precision_qty(ex, sym: str, qty: float) -> float:
    try:
        return float(ex.amount_to_precision(sym, qty))
    except Exception:
        return float(qty)


def set_leverage_safe(ex, sym: str, lev: int) -> None:
    try:
        ex.set_leverage(int(lev), sym)
    except Exception:
        pass


def market_order_safe(ex, sym: str, side: str, qty: float) -> bool:
    try:
        ex.create_market_order(sym, side, qty)
        return True
    except Exception:
        return False


def close_position_market(ex, sym: str, pos_side: str, contracts: float) -> bool:
    if contracts <= 0:
        return False
    if pos_side in ["long", "buy"]:
        return market_order_safe(ex, sym, "sell", contracts)
    return market_order_safe(ex, sym, "buy", contracts)


def position_roi_percent(p: Dict[str, Any]) -> float:
    try:
        if p.get("percentage") is not None:
            return float(p.get("percentage"))
    except Exception:
        pass
    return 0.0


def position_side_normalize(p: Dict[str, Any]) -> str:
    s = (p.get("side") or p.get("positionSide") or "").lower()
    if s in ["long", "buy"]:
        return "long"
    if s in ["short", "sell"]:
        return "short"
    return "long"


def _pos_leverage(p: Dict[str, Any]) -> float:
    try:
        v = p.get("leverage", None)
        if v is None:
            return 1.0
        return float(v)
    except Exception:
        return 1.0


# =========================================================
# ✅ 9.3) (핵심) 추세 계산 캐시
# =========================================================
_TREND_CACHE: Dict[str, Dict[str, Any]] = {}  # {"BTC/USDT:USDT|1h": {"ts":..., "trend":"하락추세"}}


def compute_ma_trend_from_df(df: pd.DataFrame, fast: int = 7, slow: int = 99) -> str:
    try:
        if df is None or df.empty or len(df) < slow + 5:
            return "중립"
        close = df["close"].astype(float)
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        last_close = float(close.iloc[-1])
        f = float(ma_fast.iloc[-1])
        s = float(ma_slow.iloc[-1])
        if f > s and last_close > s:
            return "상승추세"
        if f < s and last_close < s:
            return "하락추세"
        return "횡보/전환"
    except Exception:
        return "중립"


def get_htf_trend_cached(ex, sym: str, tf: str, fast: int, slow: int, cache_sec: int = 60) -> str:
    key = f"{sym}|{tf}"
    now = time.time()
    if key in _TREND_CACHE:
        if (now - float(_TREND_CACHE[key].get("ts", 0))) < cache_sec:
            return str(_TREND_CACHE[key].get("trend", "중립"))
    try:
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=max(220, slow + 50))
        hdf = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        trend = compute_ma_trend_from_df(hdf, fast=fast, slow=slow)
        _TREND_CACHE[key] = {"ts": now, "trend": trend}
        return trend
    except Exception:
        return "중립"


# =========================================================
# ✅ 9.5) SR(지지/저항) 기반 SL/TP 계산
# =========================================================
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def pivot_levels(df: pd.DataFrame, order: int = 6, max_levels: int = 12) -> Tuple[List[float], List[float]]:
    if df is None or df.empty or len(df) < order * 4:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    if argrelextrema is not None:
        hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    else:
        hi_idx, lo_idx = [], []
        for i in range(order, len(df) - order):
            if highs[i] == np.max(highs[i - order:i + order + 1]):
                hi_idx.append(i)
            if lows[i] == np.min(lows[i - order:i + order + 1]):
                lo_idx.append(i)

    resistances = sorted(list(set(np.round(highs[hi_idx], 8))), reverse=True)[:max_levels] if len(highs) else []
    supports = sorted(list(set(np.round(lows[lo_idx], 8))))[:max_levels] if len(lows) else []
    return supports, resistances


def sr_stop_take(
    entry_price: float,
    side: str,
    htf_df: pd.DataFrame,
    atr_period: int = 14,
    pivot_order: int = 6,
    buffer_atr_mult: float = 0.25,
    rr_min: float = 1.5,
) -> Optional[Dict[str, Any]]:
    if htf_df is None or htf_df.empty:
        return None

    atr = calc_atr(htf_df, atr_period)
    supports, resistances = pivot_levels(htf_df, pivot_order)
    buf = atr * buffer_atr_mult if atr > 0 else entry_price * 0.0015

    if side == "buy":
        below = [s for s in supports if s < entry_price]
        sl_price = (max(below) - buf) if below else (entry_price - max(buf, entry_price * 0.003))
        risk = entry_price - sl_price
        if risk <= 0:
            return None
        above = [r for r in resistances if r > entry_price]
        tp_candidate = min(above) if above else None
        tp_by_rr = entry_price + risk * rr_min
        tp_price = tp_candidate if (tp_candidate and tp_candidate > tp_by_rr) else tp_by_rr
    else:
        above = [r for r in resistances if r > entry_price]
        sl_price = (min(above) + buf) if above else (entry_price + max(buf, entry_price * 0.003))
        risk = sl_price - entry_price
        if risk <= 0:
            return None
        below = [s for s in supports if s < entry_price]
        tp_candidate = max(below) if below else None
        tp_by_rr = entry_price - risk * rr_min
        tp_price = tp_candidate if (tp_candidate and tp_candidate < tp_by_rr) else tp_by_rr

    return {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "atr": float(atr),
        "supports": supports,
        "resistances": resistances,
    }


# ✅ SR 레벨 캐시(스캔 과정 표시/안정성/요청 과다 방지)
_SR_CACHE: Dict[str, Dict[str, Any]] = {}


def get_sr_levels_cached(ex, sym: str, tf: str, pivot_order: int = 6, cache_sec: int = 60, limit: int = 220) -> Dict[str, Any]:
    key = f"{sym}|{tf}|{pivot_order}|{limit}"
    now = time.time()
    try:
        if key in _SR_CACHE and (now - float(_SR_CACHE[key].get("ts", 0) or 0)) < float(cache_sec):
            return dict(_SR_CACHE[key])
    except Exception:
        pass
    out = {"ts": now, "tf": tf, "supports": [], "resistances": []}
    try:
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=int(limit))
        hdf = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        supports, resistances = pivot_levels(hdf, order=int(pivot_order))
        out["supports"] = supports
        out["resistances"] = resistances
    except Exception:
        pass
    try:
        _SR_CACHE[key] = dict(out)
    except Exception:
        pass
    return out


# =========================================================
# ✅ 10) TradingView 다크모드 차트 (기존 유지)
# =========================================================
def tv_symbol_from_ccxt(sym: str) -> str:
    base = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    return f"BITGET:{base}{quote}.P"


def render_tradingview(symbol_ccxt: str, interval: str = "5", height: int = 560) -> None:
    tvsym = tv_symbol_from_ccxt(symbol_ccxt)
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tv_chart" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tvsym}",
          "interval": "{interval}",
          "timezone": "Asia/Seoul",
          "theme": "dark",
          "style": "1",
          "locale": "kr",
          "toolbar_bg": "#0e1117",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "withdateranges": true,
          "save_image": false,
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height)


# =========================================================
# ✅ 11) 지표 계산 (기존 유지)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.Series]]:
    status: Dict[str, Any] = {}
    if df is None or df.empty or len(df) < 120:
        return df, status, None
    # ✅ 지표 라이브러리 호환:
    # - 1순위: ta (기존)
    # - 2순위: pandas_ta (ta 미설치 환경에서 기능 복구)
    use_ta = ta is not None
    use_pta = (not use_ta) and (pta is not None)
    if not use_ta and not use_pta:
        status["_ERROR"] = "ta/pandas_ta 모듈 없음(requirements.txt에 ta 또는 pandas_ta 추가 필요)"
        return df, status, None
    status["_backend"] = "ta" if use_ta else "pandas_ta"
    if use_pta:
        status["_INFO"] = "ta 미설치 → pandas_ta로 지표 계산"

    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))
    bb_period = int(cfg.get("bb_period", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    ma_fast = int(cfg.get("ma_fast", 7))
    ma_slow = int(cfg.get("ma_slow", 99))
    stoch_k = int(cfg.get("stoch_k", 14))
    vol_mul = float(cfg.get("vol_mul", 2.0))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]
    orig_cols = set(df.columns)

    if cfg.get("use_rsi", True):
        try:
            if use_ta:
                df["RSI"] = ta.momentum.rsi(close, window=rsi_period)
            else:
                df["RSI"] = pta.rsi(close, length=rsi_period)
        except Exception as e:
            status["_RSI_ERROR"] = str(e)[:160]

    if cfg.get("use_bb", True):
        try:
            if use_ta:
                bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
                df["BB_upper"] = bb.bollinger_hband()
                df["BB_lower"] = bb.bollinger_lband()
                df["BB_mid"] = bb.bollinger_mavg()
            else:
                bb = pta.bbands(close, length=bb_period, std=bb_std)
                if isinstance(bb, pd.DataFrame) and not bb.empty:
                    col_u = next((c for c in bb.columns if str(c).startswith("BBU_")), "")
                    col_l = next((c for c in bb.columns if str(c).startswith("BBL_")), "")
                    col_m = next((c for c in bb.columns if str(c).startswith("BBM_")), "")
                    if col_u:
                        df["BB_upper"] = bb[col_u]
                    if col_l:
                        df["BB_lower"] = bb[col_l]
                    if col_m:
                        df["BB_mid"] = bb[col_m]
        except Exception as e:
            status["_BB_ERROR"] = str(e)[:160]

    if cfg.get("use_ma", True):
        try:
            if use_ta:
                df["MA_fast"] = ta.trend.sma_indicator(close, window=ma_fast)
                df["MA_slow"] = ta.trend.sma_indicator(close, window=ma_slow)
            else:
                df["MA_fast"] = pta.sma(close, length=ma_fast)
                df["MA_slow"] = pta.sma(close, length=ma_slow)
        except Exception as e:
            status["_MA_ERROR"] = str(e)[:160]

    if cfg.get("use_macd", True):
        try:
            if use_ta:
                macd = ta.trend.MACD(close)
                df["MACD"] = macd.macd()
                df["MACD_signal"] = macd.macd_signal()
            else:
                macd = pta.macd(close)
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    col_macd = next((c for c in macd.columns if str(c).startswith("MACD_") and not str(c).startswith("MACDh_") and not str(c).startswith("MACDs_")), "")
                    col_sig = next((c for c in macd.columns if str(c).startswith("MACDs_")), "")
                    if col_macd:
                        df["MACD"] = macd[col_macd]
                    if col_sig:
                        df["MACD_signal"] = macd[col_sig]
        except Exception as e:
            status["_MACD_ERROR"] = str(e)[:160]

    if cfg.get("use_stoch", True):
        try:
            if use_ta:
                df["STO_K"] = ta.momentum.stoch(high, low, close, window=stoch_k, smooth_window=3)
                df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=stoch_k, smooth_window=3)
            else:
                stoch = pta.stoch(high, low, close, k=stoch_k, d=3, smooth_k=3)
                if isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    col_k = next((c for c in stoch.columns if str(c).startswith("STOCHk_")), "")
                    col_d = next((c for c in stoch.columns if str(c).startswith("STOCHd_")), "")
                    if col_k:
                        df["STO_K"] = stoch[col_k]
                    if col_d:
                        df["STO_D"] = stoch[col_d]
        except Exception as e:
            status["_STOCH_ERROR"] = str(e)[:160]

    if cfg.get("use_cci", True):
        try:
            if use_ta:
                df["CCI"] = ta.trend.cci(high, low, close, window=20)
            else:
                df["CCI"] = pta.cci(high, low, close, length=20)
        except Exception as e:
            status["_CCI_ERROR"] = str(e)[:160]

    if cfg.get("use_mfi", True):
        try:
            if use_ta:
                df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)
            else:
                df["MFI"] = pta.mfi(high, low, close, vol, length=14)
        except Exception as e:
            status["_MFI_ERROR"] = str(e)[:160]

    if cfg.get("use_willr", True):
        try:
            if use_ta:
                df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)
            else:
                df["WILLR"] = pta.willr(high, low, close, length=14)
        except Exception as e:
            status["_WILLR_ERROR"] = str(e)[:160]

    if cfg.get("use_adx", True):
        try:
            if use_ta:
                df["ADX"] = ta.trend.adx(high, low, close, window=14)
            else:
                adx = pta.adx(high, low, close, length=14)
                if isinstance(adx, pd.DataFrame) and not adx.empty:
                    col_adx = next((c for c in adx.columns if str(c).startswith("ADX_")), "")
                    if col_adx:
                        df["ADX"] = adx[col_adx]
        except Exception as e:
            status["_ADX_ERROR"] = str(e)[:160]

    if cfg.get("use_vol", True):
        try:
            df["VOL_MA"] = vol.rolling(20).mean()
            df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * vol_mul)).astype(int)
        except Exception as e:
            status["_VOL_ERROR"] = str(e)[:160]

    if pta is not None:
        try:
            df["ATR_ref"] = pta.atr(df["high"], df["low"], df["close"], length=14)
        except Exception:
            pass

    # ✅ 일부 지표가 전부 NaN이면 dropna()가 전체를 비울 수 있으므로, all-NaN 컬럼은 제거
    try:
        new_cols = [c for c in df.columns if c not in orig_cols]
        dropped = []
        for c in new_cols:
            try:
                if df[c].isna().all():
                    df.drop(columns=[c], inplace=True)
                    dropped.append(c)
            except Exception:
                continue
        if dropped:
            status["_DROP_ALL_NAN_COLS"] = dropped[:25]
    except Exception:
        pass

    # dropna는 유지(기존 동작)하되, 전부 비어버리면 close 기준으로라도 복구 시도
    df2 = df.dropna()
    if df2.empty or len(df2) < 5:
        try:
            df2 = df.dropna(subset=["close"])
        except Exception:
            df2 = df2
    if df2.empty or len(df2) < 5:
        return df2, status, None

    last = df2.iloc[-1]
    prev = df2.iloc[-2] if len(df2) >= 2 else last

    used = []

    # RSI
    if cfg.get("use_rsi", True) and "RSI" in df2.columns:
        used.append("RSI")
        rsi_now = float(last.get("RSI", 50))
        if rsi_now < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_now:.1f})"
        elif rsi_now > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_now:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_now:.1f})"

    # BB
    if cfg.get("use_bb", True) and all(c in df2.columns for c in ["BB_upper", "BB_lower"]):
        used.append("볼린저밴드")
        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

    # MA 추세(단기: 현재 timeframe 기준)
    trend = "중립"
    if cfg.get("use_ma", True):
        used.append("이동평균(MA)")
        try:
            if all(c in df2.columns for c in ["MA_fast", "MA_slow"]):
                if last["MA_fast"] > last["MA_slow"] and last["close"] > last["MA_slow"]:
                    trend = "상승추세"
                elif last["MA_fast"] < last["MA_slow"] and last["close"] < last["MA_slow"]:
                    trend = "하락추세"
                else:
                    trend = "횡보/전환"
            else:
                # 최소 기능: close만으로도 추세 산출(표시용)
                trend = compute_ma_trend_from_df(df2, fast=ma_fast, slow=ma_slow)
        except Exception:
            trend = "중립"
        status["추세"] = f"📈 {trend}"

    # MACD
    if cfg.get("use_macd", True) and all(c in df2.columns for c in ["MACD", "MACD_signal"]):
        used.append("MACD")
        status["MACD"] = "📈 상승(골든)" if last["MACD"] > last["MACD_signal"] else "📉 하락(데드)"

    # ADX
    if cfg.get("use_adx", True) and "ADX" in df2.columns:
        used.append("ADX(추세강도)")
        adx = float(last.get("ADX", 0))
        status["ADX"] = "🔥 추세 강함" if adx >= 25 else "💤 추세 약함"

    # Volume
    if cfg.get("use_vol", True) and "VOL_SPIKE" in df2.columns:
        used.append("거래량")
        status["거래량"] = "🔥 거래량 급증" if int(last.get("VOL_SPIKE", 0)) == 1 else "⚪ 보통"

    # RSI 해소
    rsi_prev = float(prev.get("RSI", 50)) if (cfg.get("use_rsi", True) and "RSI" in df2.columns) else 50.0
    rsi_now = float(last.get("RSI", 50)) if (cfg.get("use_rsi", True) and "RSI" in df2.columns) else 50.0
    rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
    rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

    adx_now = float(last.get("ADX", 0)) if (cfg.get("use_adx", True) and "ADX" in df2.columns) else 0.0
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_now >= 18)

    status["_used_indicators"] = used
    status["_rsi_resolve_long"] = bool(rsi_resolve_long)
    status["_rsi_resolve_short"] = bool(rsi_resolve_short)
    status["_pullback_candidate"] = bool(pullback_candidate)

    return df2, status, last


# =========================================================
# ✅ 12) 외부 시황 통합(거시/심리/레짐/뉴스) - 캐시/한글화/안정성 강화
# =========================================================
_ext_cache = TTLCache(maxsize=12, ttl=60) if TTLCache else None
_translate_cache = TTLCache(maxsize=256, ttl=60 * 60 * 24) if TTLCache else None  # 24h


def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT_SEC, attempts: int = 3):
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    try:
        attempts_i = max(1, int(attempts or 1))
    except Exception:
        attempts_i = 1

    if retry is None or stop_after_attempt is None or wait_exponential_jitter is None:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    @retry(stop=stop_after_attempt(attempts_i), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def _do():
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r.json()

    try:
        return _do()
    except Exception:
        return None


def _translate_ko_rule(text: str) -> str:
    """AI/번역기 없이도 최소한 읽히게 만드는 룰 기반 '한글화 보정'."""
    t = str(text or "")
    rep = {
        "Extreme Fear": "극공포",
        "Fear": "공포",
        "Neutral": "중립",
        "Greed": "탐욕",
        "Extreme Greed": "극탐욕",
        "High": "매우 중요",
        "Medium": "중요",
        "Low": "낮음",
        "United States": "미국",
        "Euro Zone": "유로존",
        "Japan": "일본",
        "China": "중국",
        "United Kingdom": "영국",
        "Germany": "독일",
        "France": "프랑스",
        "Korea": "한국",
        "Bitcoin": "비트코인",
        "BTC": "BTC",
        "ETF": "ETF",
        "Inflation": "인플레이션",
        "Interest Rate": "금리",
        "Rate Decision": "금리결정",
        "CPI": "CPI(소비자물가)",
        "PPI": "PPI(생산자물가)",
        "FOMC": "FOMC(연준회의)",
        "Nonfarm Payrolls": "NFP(비농업 고용)",
        "Unemployment Rate": "실업률",
        "Retail Sales": "소매판매",
        "GDP": "GDP",
        "PMI": "PMI",
        "Core": "근원",
        "YoY": "전년대비",
        "MoM": "전월대비",
    }
    for k, v in rep.items():
        t = t.replace(k, v)
    return t


def translate_to_korean(text: str, cfg: Dict[str, Any], use_cache: bool = True) -> str:
    """
    우선순위:
    1) deep-translator(선택) -> 2) OpenAI(설정 ON + 키 존재) -> 3) 룰 기반 보정 -> 4) 원문
    """
    s = str(text or "").strip()
    if not s:
        return ""
    if use_cache and _translate_cache is not None:
        try:
            k = f"ko:{hash(s)}"
            if k in _translate_cache:
                return _translate_cache[k]
        except Exception:
            pass

    out = s

    # deep-translator (네트워크 hang 방지: hard-timeout)
    if GoogleTranslator is not None:
        try:
            def _do_trans():
                return GoogleTranslator(source="auto", target="ko").translate(s)

            out = _call_with_timeout(_do_trans, 4)
        except Exception:
            out = s

    # OpenAI 번역(옵션)
    if out == s and cfg.get("external_ai_translate_enable", False):
        client = get_openai_client(cfg)
        if client is not None:
            try:
                def _do():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "너는 번역기다. 입력 문장을 자연스러운 한국어로만 번역해라. 다른 말 금지."},
                            {"role": "user", "content": s},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    )

                resp = _call_with_timeout(_do, OPENAI_TIMEOUT_SEC)
                out = (resp.choices[0].message.content or "").strip()
                if not out:
                    out = s
            except Exception:
                out = s

    # 룰 기반 보정
    if cfg.get("external_koreanize_enable", True):
        out = _translate_ko_rule(out)

    if use_cache and _translate_cache is not None:
        try:
            _translate_cache[f"ko:{hash(s)}"] = out
        except Exception:
            pass
    return out


def fetch_fear_greed(cfg: Dict[str, Any]):
    # 외부시황은 "참고"용 → 과도한 재시도는 봇 루프를 멈추게 할 수 있어 attempts를 낮춘다.
    data = _http_get_json("https://api.alternative.me/fng/?limit=1&format=json", timeout=6, attempts=2)
    if not data or "data" not in data or not data["data"]:
        return None
    d0 = data["data"][0]
    try:
        v = int(d0.get("value", 0))
        cls = str(d0.get("value_classification", ""))
        cls_ko = translate_to_korean(cls, cfg)
        # 이모티콘
        emo = "😱" if v <= 25 else ("🙂" if v <= 55 else ("😋" if v <= 75 else "🤑"))
        return {"value": v, "classification": cls_ko, "emoji": emo, "timestamp": str(d0.get("timestamp", ""))}
    except Exception:
        return None


def fetch_coingecko_global():
    data = _http_get_json("https://api.coingecko.com/api/v3/global", timeout=8, attempts=2)
    if not data or "data" not in data:
        return None
    g = data["data"]
    mcp = g.get("market_cap_percentage", {}) or {}
    try:
        return {
            "btc_dominance": float(mcp.get("btc", 0.0)),
            "eth_dominance": float(mcp.get("eth", 0.0)),
            "total_mcap_usd": float((g.get("total_market_cap", {}) or {}).get("usd", 0.0)),
            "mcap_change_24h_pct": float(g.get("market_cap_change_percentage_24h_usd", 0.0)),
        }
    except Exception:
        return None


def _country_to_ko(country: str, cfg: Dict[str, Any]) -> str:
    c = str(country or "").strip()
    m = {
        "USD": "미국",
        "US": "미국",
        "EUR": "유로존",
        "EU": "유로존",
        "JPY": "일본",
        "JP": "일본",
        "CNY": "중국",
        "CN": "중국",
        "GBP": "영국",
        "UK": "영국",
        "CHF": "스위스",
        "CAD": "캐나다",
        "AUD": "호주",
        "NZD": "뉴질랜드",
        "KRW": "한국",
        "KR": "한국",
    }
    return m.get(c, translate_to_korean(c, cfg))


def fetch_upcoming_high_impact_events(cfg: Dict[str, Any], within_minutes: int = 30, limit: int = 80):
    data = _http_get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=8, attempts=2)
    if not isinstance(data, list):
        return []
    now = now_kst()
    out = []
    for x in data[:limit]:
        try:
            if str(x.get("impact", "")) != "High":
                continue
            dt_str = str(x.get("date", ""))
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo:
                    dt = dt.astimezone(KST)
                else:
                    dt = dt.replace(tzinfo=KST)
            except Exception:
                continue

            diff_min = (dt - now).total_seconds() / 60.0
            if 0 <= diff_min <= within_minutes:
                title = str(x.get("title", ""))
                title_ko = translate_to_korean(title, cfg)
                country_ko = _country_to_ko(str(x.get("country", "")), cfg)
                out.append(
                    {
                        "time_kst": dt.strftime("%m-%d %H:%M"),
                        "title": f"🚨 {title_ko}",
                        "country": country_ko,
                        "impact": "매우 중요",
                    }
                )
        except Exception:
            continue
    return out


def fetch_news_headlines_rss(cfg: Dict[str, Any], max_items: int = 12):
    if feedparser is None:
        return []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    items = []
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    for url in feeds:
        try:
            # feedparser.parse(url)은 내부 네트워크 fetch가 hang될 수 있음 → requests로 timeout 보장 후 parse
            r = requests.get(url, timeout=8, headers=headers)
            r.raise_for_status()
            d = feedparser.parse(r.content)
            for e in (d.entries or [])[: max_items * 2]:
                title = str(getattr(e, "title", "")).strip()
                if not title:
                    continue
                items.append(title)
        except Exception:
            continue
    uniq, seen = [], set()
    for t in items:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    uniq = uniq[:max_items]
    # 한글화(옵션)
    if cfg.get("external_koreanize_enable", True):
        # ✅ 번역기는 느릴 수 있음(특히 deep-translator) → 시간 예산 초과 시 룰 기반만 적용
        try:
            raw = cfg.get("news_translate_budget_sec", 10)
            budget = float(10.0 if raw is None else raw)
        except Exception:
            budget = 10.0
        budget = max(0.0, budget)
        if budget <= 0:
            uniq = [_translate_ko_rule(t) for t in uniq]
        else:
            t0 = time.time()
            out_titles = []
            for t in uniq:
                if (time.time() - t0) > budget:
                    out_titles.append(_translate_ko_rule(t))
                    continue
                out_titles.append(translate_to_korean(t, cfg))
            uniq = out_titles
    return uniq


def fetch_daily_btc_brief(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    매일 아침: BTC 관련 경제뉴스 5개 선정
    - feedparser 없으면 빈 값 반환
    - OpenAI 키 있으면 요약/한글화 강화(옵션)
    """
    date_str = today_kst_str()
    if _ext_cache is not None and f"daily_btc_brief:{date_str}" in _ext_cache:
        return _ext_cache[f"daily_btc_brief:{date_str}"]

    out = {"date": date_str, "items": [], "asof_kst": now_kst_str(), "source": "rss"}
    if feedparser is None:
        out["source"] = "feedparser_missing"
        return out

    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    keywords = [
        "bitcoin",
        "btc",
        "etf",
        "fed",
        "fomc",
        "cpi",
        "ppi",
        "rate",
        "inflation",
        "macro",
        "economy",
        "jobs",
        "nfp",
        "powell",
        "interest",
        "treasury",
        "yield",
    ]

    raw_titles: List[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    for url in feeds:
        try:
            r = requests.get(url, timeout=8, headers=headers)
            r.raise_for_status()
            d = feedparser.parse(r.content)
            for e in (d.entries or [])[:60]:
                title = str(getattr(e, "title", "")).strip()
                if not title:
                    continue
                low = title.lower()
                if any(k in low for k in keywords):
                    raw_titles.append(title)
        except Exception:
            continue

    # 중복 제거 + 상위 N개
    uniq: List[str] = []
    seen = set()
    for t in raw_titles:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
        if len(uniq) >= int(cfg.get("daily_btc_brief_max_items", 5)):
            break

    if not uniq:
        out["items"] = []
        if _ext_cache is not None:
            _ext_cache[f"daily_btc_brief:{date_str}"] = out
        return out

    # 한글화/요약
    items_ko = [translate_to_korean(t, cfg) for t in uniq]

    if cfg.get("daily_btc_brief_ai_summarize", True):
        client = get_openai_client(cfg)
        if client is not None:
            try:
                payload = {"date": date_str, "titles": items_ko}

                def _do():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "너는 암호화폐 트레이딩용 아침 브리핑 에디터다.\n"
                                    "입력된 제목 리스트에서 '비트코인/거시경제' 관점으로 중요한 5개를 골라,"
                                    "각 항목을 아주 짧고 쉬운 한국어 한줄로 정리해라.\n"
                                    "출력은 반드시 JSON만.\n"
                                    '형식: {"items":[{"emoji":"📰","title":"...","note":"한줄 요약"}], "bias":"중립|보수|공격", "risk":"낮음|보통|높음"}'
                                ),
                            },
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                        max_tokens=700,
                    )

                resp = _call_with_timeout(_do, OPENAI_TIMEOUT_SEC)
                jj = json.loads(resp.choices[0].message.content)
                items = jj.get("items", [])
                if isinstance(items, list) and items:
                    out["items"] = items[: int(cfg.get("daily_btc_brief_max_items", 5))]
                    out["bias"] = str(jj.get("bias", "중립"))
                    out["risk"] = str(jj.get("risk", "보통"))
                    out["source"] = "openai"
                else:
                    out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]
            except Exception:
                out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]
    else:
        out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]

    if _ext_cache is not None:
        _ext_cache[f"daily_btc_brief:{date_str}"] = out
    return out


def build_external_context(cfg: Dict[str, Any], rt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not cfg.get("use_external_context", True):
        return {"enabled": False}

    ttl = int(cfg.get("external_refresh_sec", 60))
    cache_key = f"ext:{today_kst_str()}:{ttl}"
    if _ext_cache is not None and cache_key in _ext_cache:
        return _ext_cache[cache_key]

    blackout = int(cfg.get("macro_blackout_minutes", 30))
    high_events = fetch_upcoming_high_impact_events(cfg, within_minutes=blackout)

    fg = fetch_fear_greed(cfg)
    cg = fetch_coingecko_global()

    headlines: List[str] = []
    if cfg.get("news_enable", True):
        headlines = fetch_news_headlines_rss(cfg, max_items=int(cfg.get("news_max_headlines", 12)))

    daily_brief = {}
    try:
        # 런타임에 저장된 브리핑이 있으면 우선 사용, 없으면 즉시 가져오진 않음(아침 스케줄에서 처리)
        if rt and isinstance(rt.get("daily_btc_brief"), dict) and rt["daily_btc_brief"].get("date") == today_kst_str():
            daily_brief = rt["daily_btc_brief"]
    except Exception:
        daily_brief = {}

    ext = {
        "enabled": True,
        "blackout_minutes": blackout,
        "high_impact_events_soon": high_events,
        "fear_greed": fg,
        "global": cg,
        "headlines": headlines,
        "daily_btc_brief": daily_brief,
        "asof_kst": now_kst_str(),
    }

    if _ext_cache is not None:
        _ext_cache[cache_key] = ext
    return ext


def external_risk_multiplier(ext: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    외부 시황이 위험하면 신규 진입을 "감산/보수"로 조정(완전 금지 X).
    """
    if not cfg.get("entry_risk_reduce_enable", True):
        return 1.0
    mul = 1.0
    try:
        evs = (ext or {}).get("high_impact_events_soon") or []
        if evs:
            mul *= float(cfg.get("entry_risk_reduce_factor", 0.65))
    except Exception:
        pass
    try:
        fg = (ext or {}).get("fear_greed") or {}
        v = int(fg.get("value", -1)) if fg else -1
        if 0 <= v <= 25:  # 극공포
            mul *= 0.85
        elif v >= 75:  # 극탐욕
            mul *= 0.85
    except Exception:
        pass
    try:
        brief = (ext or {}).get("daily_btc_brief") or {}
        risk = str(brief.get("risk", "")).strip()
        if risk == "높음":
            mul *= 0.8
    except Exception:
        pass
    return float(clamp(mul, 0.2, 1.0))


# =========================================================
# ✅ 13) AI 판단 + 리스크 매니저(기존 유지/강화)
# =========================================================
def _atr_price_pct(df: pd.DataFrame, window: int = 14) -> float:
    try:
        if df is None or df.empty or len(df) < window + 5:
            return 0.0
        if ta is not None:
            atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=window)
            v = float(atr.iloc[-1])
        else:
            # ta 미설치 환경에서도 최소 기능 유지(수동 ATR)
            v = float(calc_atr(df, period=window))
        c = float(df["close"].iloc[-1])
        if c <= 0:
            return 0.0
        return (v / c) * 100.0
    except Exception:
        return 0.0


def _swing_stop_price_pct(df: pd.DataFrame, decision: str, lookback: int = 40, buffer_atr_mul: float = 0.25) -> float:
    try:
        if df is None or df.empty or len(df) < lookback + 5:
            return 0.0
        recent = df.tail(lookback)
        last_close = float(df["close"].iloc[-1])
        atr_pct = _atr_price_pct(df, 14)
        buf_pct = atr_pct * buffer_atr_mul

        if decision == "buy":
            swing = float(recent["low"].min())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 - buf_pct / 100.0)
            return max(0.0, ((last_close - stop_price) / last_close) * 100.0)

        if decision == "sell":
            swing = float(recent["high"].max())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 + buf_pct / 100.0)
            return max(0.0, ((stop_price - last_close) / last_close) * 100.0)

        return 0.0
    except Exception:
        return 0.0


def _rr_min_by_mode(mode: str) -> float:
    if mode == "안전모드":
        return 1.8
    if mode == "공격모드":
        return 2.1
    return 2.6


def _rr_min_by_style(style: str) -> float:
    # 스타일별 최소 손익비 가이드
    if style == "스캘핑":
        return 1.2
    if style == "스윙":
        return 1.8
    return 1.5


def _risk_guardrail(out: Dict[str, Any], df: pd.DataFrame, decision: str, mode: str, style: str, external: Dict[str, Any]) -> Dict[str, Any]:
    lev = max(1, int(out.get("leverage", 1)))
    sl_roi = float(out.get("sl_pct", 1.2))
    tp_roi = float(out.get("tp_pct", 3.0))
    rr = float(out.get("rr", 0))

    sl_price_pct_now = sl_roi / max(lev, 1)

    atr_pct = _atr_price_pct(df, 14)
    min_price_stop = max(0.25, atr_pct * 0.9)

    swing_stop = _swing_stop_price_pct(df, decision, lookback=40, buffer_atr_mul=0.25)
    if swing_stop > 0:
        swing_stop = min(swing_stop, max(min_price_stop * 3.0, atr_pct * 3.0))
    recommended_price_stop = max(min_price_stop, swing_stop)

    notes = []

    # 외부시황: 극공포면 SL 여유 약간 추가
    try:
        fg = (external or {}).get("fear_greed") or {}
        v = int(fg.get("value", -1)) if fg else -1
        if 0 <= v <= 25:
            recommended_price_stop = max(recommended_price_stop, min_price_stop * 1.2)
            notes.append("외부시황: 극공포 → 손절 여유 추가")
    except Exception:
        pass

    if sl_price_pct_now < recommended_price_stop:
        sl_price_pct_now = recommended_price_stop
        sl_roi = sl_price_pct_now * lev
        notes.append(f"손절폭(가격기준) 확장({recommended_price_stop:.2f}%)")

    rr_min_mode = _rr_min_by_mode(mode)
    rr_min_style = _rr_min_by_style(style)
    rr_min = max(rr_min_mode, rr_min_style)

    if rr <= 0:
        rr = max(rr_min, tp_roi / max(sl_roi, 0.01))

    if tp_roi < sl_roi * rr_min:
        tp_roi = sl_roi * rr_min
        notes.append(f"손익비 최소 {rr_min:.1f} 확보(익절 상향)")

    rr = max(rr, tp_roi / max(sl_roi, 0.01))

    out["sl_pct"] = float(sl_roi)
    out["tp_pct"] = float(tp_roi)
    out["rr"] = float(rr)
    out["sl_price_pct"] = float(sl_roi / max(lev, 1))
    out["tp_price_pct"] = float(tp_roi / max(lev, 1))
    out["risk_note"] = " / ".join(notes) if notes else "보정 없음"
    return out


def ai_decide_trade(df: pd.DataFrame, status: Dict[str, Any], symbol: str, mode: str, cfg: Dict[str, Any], external: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ 기존 기능 유지: AI가 buy/sell/hold + entry/leverage/sl/tp/rr/근거(JSON)
    ✅ 안정성 강화: timeout + 예외 처리
    """
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가"))
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return {"decision": "hold", "confidence": 0, "reason_easy": msg, "used_indicators": status.get("_used_indicators", [])}
    if df is None or df.empty or status is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "데이터 부족", "used_indicators": status.get("_used_indicators", [])}

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    past_mistakes = get_past_mistakes_text(5)

    # daily brief를 포함한 외부시황(이미 thread에서 build했으면 그걸 쓰게 external 파라미터로 전달)
    ext = external or {}
    daily_brief = (ext.get("daily_btc_brief") or {}) if isinstance(ext, dict) else {}

    features = {
        "symbol": symbol,
        "mode": mode,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)) if "RSI" in df.columns else None,
        "rsi_now": float(last.get("RSI", 50)) if "RSI" in df.columns else None,
        "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
        "trend_short": status.get("추세", ""),  # 단기추세(timeframe)
        "bb": status.get("BB", ""),
        "macd": status.get("MACD", ""),
        "vol": status.get("거래량", ""),
        "rsi_resolve_long": bool(status.get("_rsi_resolve_long", False)),
        "rsi_resolve_short": bool(status.get("_rsi_resolve_short", False)),
        "pullback_candidate": bool(status.get("_pullback_candidate", False)),
        "atr_price_pct": _atr_price_pct(df, 14),
        "external": {
            "fear_greed": ext.get("fear_greed"),
            "high_impact_events_soon": (ext.get("high_impact_events_soon") or [])[:3],
            "global": ext.get("global"),
            "daily_btc_brief": daily_brief,
        },
    }

    fg_txt = ""
    try:
        fg = (ext or {}).get("fear_greed") or {}
        if fg:
            fg_txt = f"- 공포탐욕지수: {fg.get('emoji','')} {int(fg.get('value', 0))} / {fg.get('classification','')}"
    except Exception:
        fg_txt = ""

    ev_txt = ""
    try:
        evs = (ext or {}).get("high_impact_events_soon") or []
        if evs:
            ev_txt = "- 중요 이벤트(임박): " + " | ".join([f"{e.get('country','')} {e.get('title','')}" for e in evs[:3]])
    except Exception:
        ev_txt = ""

    brief_txt = ""
    try:
        items = (daily_brief or {}).get("items") or []
        if items:
            brief_txt = "- 오늘 아침 BTC 브리핑(요약): " + " / ".join([str(i.get("title", ""))[:40] for i in items[:3]])
    except Exception:
        brief_txt = ""

    sys = f"""
너는 '워뇨띠 스타일(눌림목/해소 타이밍) + 손익비' 기반의 자동매매 트레이더 AI다.

[과거 실수(요약)]
{past_mistakes}

[외부 시황(참고)]
{fg_txt}
{ev_txt}
{brief_txt}

[핵심 룰]
1) RSI 과매도/과매수 '상태'에 즉시 진입하지 말고, '해소되는 시점'에서만 진입 후보.
2) 상승추세에서는 롱 우선, 하락추세에서는 숏 우선. (역추세는 더 짧게/보수적으로)
3) 모드 규칙 반드시 준수:
   - 최소 확신도: {rule["min_conf"]}
   - 진입 비중(%): {rule["entry_pct_min"]}~{rule["entry_pct_max"]}
   - 레버리지: {rule["lev_min"]}~{rule["lev_max"]}

[중요]
- sl_pct / tp_pct는 ROI%(레버 반영 수익률)로 출력한다.
- 변동성(atr_price_pct)이 작으면 손절을 너무 타이트하게 잡지 마라.
- 영어 금지. 쉬운 한글.
- 반드시 JSON만 출력.
"""

    user = f"""
시장 데이터(JSON):
{json.dumps(features, ensure_ascii=False)}

JSON 형식:
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0-100,
  "entry_pct": {rule["entry_pct_min"]}-{rule["entry_pct_max"]},
  "leverage": {rule["lev_min"]}-{rule["lev_max"]},
  "sl_pct": 0.3-50.0,
  "tp_pct": 0.5-150.0,
  "rr": 0.5-10.0,
  "used_indicators": ["..."],
  "reason_easy": "쉬운 한글"
}}
"""
    try:
        # 모델 fallback (gpt-4o 미지원 계정/환경 대응)
        models = [
            str(cfg.get("openai_model_trade", "") or "").strip(),
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        # 중복 제거(순서 유지)
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=900,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        out["_openai_model"] = model_used

        out["decision"] = out.get("decision", "hold")
        if out["decision"] not in ["buy", "sell", "hold"]:
            out["decision"] = "hold"

        out["confidence"] = int(clamp(int(out.get("confidence", 0)), 0, 100))

        out["entry_pct"] = float(out.get("entry_pct", rule["entry_pct_min"]))
        out["entry_pct"] = float(clamp(out["entry_pct"], rule["entry_pct_min"], rule["entry_pct_max"]))

        out["leverage"] = int(out.get("leverage", rule["lev_min"]))
        out["leverage"] = int(clamp(out["leverage"], rule["lev_min"], rule["lev_max"]))

        out["sl_pct"] = float(out.get("sl_pct", 1.2))
        out["tp_pct"] = float(out.get("tp_pct", 3.0))
        out["rr"] = float(out.get("rr", max(0.5, out["tp_pct"] / max(out["sl_pct"], 0.01))))

        used = out.get("used_indicators", status.get("_used_indicators", []))
        if not isinstance(used, list):
            used = status.get("_used_indicators", [])
        out["used_indicators"] = used

        out["reason_easy"] = str(out.get("reason_easy", ""))[:500]

        if out["decision"] in ["buy", "sell"] and out["confidence"] < rule["min_conf"]:
            out["decision"] = "hold"

        return out

    except FuturesTimeoutError:
        return {"decision": "hold", "confidence": 0, "reason_easy": "AI 타임아웃(대기 너무 김)", "used_indicators": status.get("_used_indicators", [])}
    except Exception as e:
        openai_handle_failure(e, cfg, where="DECIDE_TRADE")
        notify_admin_error("AI:DECIDE_TRADE", e, context={"symbol": symbol, "mode": mode}, tb=traceback.format_exc(), min_interval_sec=120.0)
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": status.get("_used_indicators", [])}


def ai_decide_style(symbol: str, decision: str, trend_short: str, trend_long: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    룰 기반으로 애매할 때만 AI로 스캘핑/스윙 판단.
    비용/지연 최소화를 위해 기본은 룰 기반.
    """
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return {"style": "스캘핑", "confidence": 55, "reason": f"{msg} → 룰 기반(보수적으로 스캘핑)"}

    payload = {
        "symbol": symbol,
        "decision": decision,
        "trend_short": trend_short,
        "trend_long": trend_long,
    }
    sys = (
        "너는 트레이딩 스타일 분류기다.\n"
        "단기/장기 추세와 방향(decision)을 보고 지금은 '스캘핑'이 유리한지 '스윙'이 유리한지 결정한다.\n"
        "출력은 반드시 JSON만.\n"
        '형식: {"style":"스캘핑"|"스윙","confidence":0-100,"reason":"쉬운 한글"}'
    )
    try:
        models = [
            str(cfg.get("openai_model_style", "") or "").strip(),
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        _model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        style = str(out.get("style", "스캘핑"))
        if style not in ["스캘핑", "스윙"]:
            style = "스캘핑"
        conf = int(clamp(int(out.get("confidence", 55)), 0, 100))
        reason = str(out.get("reason", ""))[:240]
        return {"style": style, "confidence": conf, "reason": reason}
    except Exception as e:
        openai_handle_failure(e, cfg, where="DECIDE_STYLE")
        notify_admin_error("AI:DECIDE_STYLE", e, context={"symbol": symbol}, tb=traceback.format_exc(), min_interval_sec=180.0)
        return {"style": "스캘핑", "confidence": 55, "reason": "스타일 AI 판단 실패 → 스캘핑"}


def decide_style_rule_based(decision: str, trend_short: str, trend_long: str) -> Tuple[str, int, str]:
    """
    ✅ 핵심 요구 반영:
    - 단기/장기 추세가 모두 같은 방향이면 '스윙'
    - 단기만 맞으면 '스캘핑'(역추세 허용 but 짧게)
    """
    ts = str(trend_short or "")
    tl = str(trend_long or "")
    d = str(decision or "")

    def _align(tr: str, dec: str) -> bool:
        if dec == "buy":
            return "상승" in tr
        if dec == "sell":
            return "하락" in tr
        return False

    short_ok = _align(ts, d)
    long_ok = _align(tl, d)

    if short_ok and long_ok:
        return "스윙", 85, "단기+장기 추세가 같은 방향 → 스윙 유리"
    if short_ok and not long_ok:
        return "스캘핑", 82, "단기만 같은 방향(역추세/전환 구간) → 스캘핑 유리"
    if (not short_ok) and long_ok:
        return "스캘핑", 65, "장기만 같은 방향(단기 흔들림) → 보수적으로 스캘핑"
    return "스캘핑", 55, "추세 애매/불일치 → 스캘핑(보수)"


def apply_style_envelope(ai: Dict[str, Any], style: str, cfg: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI 출력은 유지하되, 스타일별 상한/하한으로 보정한다(기능 축소 X, 안전장치).
    """
    out = dict(ai or {})
    try:
        entry_pct = float(out.get("entry_pct", rule["entry_pct_min"]))
        lev = int(out.get("leverage", rule["lev_min"]))
        sl = float(out.get("sl_pct", 1.2))
        tp = float(out.get("tp_pct", 3.0))

        if style == "스캘핑":
            entry_pct = float(clamp(entry_pct * float(cfg.get("scalp_entry_pct_mult", 0.65)), rule["entry_pct_min"], rule["entry_pct_max"]))
            lev = int(min(lev, int(cfg.get("scalp_lev_cap", rule["lev_max"]))))
            sl = float(clamp(sl, float(cfg.get("scalp_sl_roi_min", 0.8)), float(cfg.get("scalp_sl_roi_max", 5.0))))
            tp = float(clamp(tp, float(cfg.get("scalp_tp_roi_min", 0.8)), float(cfg.get("scalp_tp_roi_max", 6.0))))

        elif style == "스윙":
            entry_pct = float(clamp(entry_pct * float(cfg.get("swing_entry_pct_mult", 1.0)), rule["entry_pct_min"], rule["entry_pct_max"]))
            lev = int(min(lev, int(cfg.get("swing_lev_cap", rule["lev_max"]))))
            sl = float(clamp(sl, float(cfg.get("swing_sl_roi_min", 1.5)), float(cfg.get("swing_sl_roi_max", 30.0))))
            tp = float(clamp(tp, float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))

        out["entry_pct"] = entry_pct
        out["leverage"] = lev
        out["sl_pct"] = sl
        out["tp_pct"] = tp
        out["rr"] = float(out.get("rr", tp / max(sl, 0.01)))
    except Exception:
        pass
    return out


# =========================================================
# ✅ 14) AI 회고(후기) (기존 유지 + 안정성)
# =========================================================
def ai_write_review(symbol: str, side: str, pnl_percent: float, reason: str, cfg: Dict[str, Any]) -> Tuple[str, str]:
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        one = "익절" if pnl_percent >= 0 else "손절"
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return (f"{one}({pnl_percent:.2f}%)", f"{msg} - 후기 자동작성 불가")

    sys = "너는 매매 회고를 아주 쉽게 써주는 코치다. 출력은 반드시 JSON만. 영어 금지."
    user = f"""
상황:
- 코인: {symbol}
- 포지션: {side}
- 결과: {pnl_percent:.2f}%
- 청산 이유: {reason}

JSON 형식:
{{
  "one_line": "한줄평(아주 짧게)",
  "review": "후기(손절이면 다음에 개선 / 익절이면 유지할 점)"
}}
    """
    try:
        models = [
            str(cfg.get("openai_model_review", "") or "").strip(),
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        _model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        return str(out.get("one_line", ""))[:120], str(out.get("review", ""))[:800]
    except Exception as e:
        openai_handle_failure(e, cfg, where="WRITE_REVIEW")
        notify_admin_error("AI:WRITE_REVIEW", e, context={"symbol": symbol}, tb=traceback.format_exc(), min_interval_sec=180.0)
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")


# =========================================================
# ✅ 15) 모니터 상태(하트비트) + 이벤트 링버퍼
# =========================================================
def monitor_init():
    mon = read_json_safe(MONITOR_FILE, {"coins": {}, "events": [], "scan_process": []}) or {"coins": {}, "events": [], "scan_process": []}
    mon["_boot_time_kst"] = now_kst_str()
    mon["_last_write"] = 0
    write_json_atomic(MONITOR_FILE, mon)
    return mon


def monitor_write_throttled(mon: Dict[str, Any], min_interval_sec: float = 1.0):
    lastw = float(mon.get("_last_write", 0))
    if time.time() - lastw >= min_interval_sec:
        write_json_atomic(MONITOR_FILE, mon)
        mon["_last_write"] = time.time()


def mon_add_event(mon: Dict[str, Any], ev_type: str, symbol: str = "", message: str = "", extra: Optional[Dict[str, Any]] = None):
    try:
        ev = {"time_kst": now_kst_str(), "type": ev_type, "symbol": symbol, "message": message, "extra": extra or {}}
        mon.setdefault("events", [])
        mon["events"].append(ev)
        mon["events"] = mon["events"][-250:]
        # Google Sheets EVENT 누적(비동기 큐)
        try:
            gsheet_log_event(stage=ev_type, message=f"{symbol} {message}".strip(), payload={"symbol": symbol, **(extra or {})})
        except Exception:
            pass
    except Exception:
        pass


def mon_add_scan(mon: Dict[str, Any], stage: str, symbol: str, tf: str = "", signal: str = "", score: Any = "", message: str = "", extra: Optional[Dict[str, Any]] = None):
    """
    SCAN Process 로그(요구사항):
    - stage: fetch_short/fetch_long/support_resistance/rule_signal/ai_call/ai_result/trade_opened/trade_skipped/in_position ...
    - monitor_state.json에 저장되어 UI/Telegram이 항상 최신을 볼 수 있게 함
    """
    try:
        ts = time.time()
        rec = {
            "time_kst": now_kst_str(),
            "stage": stage,
            "symbol": symbol,
            "tf": tf,
            "signal": signal,
            "score": score,
            "message": message,
            "extra": extra or {},
        }
        mon.setdefault("scan_process", [])
        mon["scan_process"].append(rec)
        mon["scan_process"] = mon["scan_process"][-400:]
        mon["last_scan_epoch"] = ts
        mon["last_scan_kst"] = rec.get("time_kst", "")
        # 코인별 진행상황(요구사항: "어떤 단계로 분석중인지" 직관적으로)
        try:
            sym0 = str(symbol or "").strip()
            if sym0 and sym0 != "*":
                mon.setdefault("coins", {}).setdefault(sym0, {})
                mon["coins"][sym0]["scan_stage"] = stage
                mon["coins"][sym0]["scan_stage_kst"] = rec.get("time_kst", "")
                mon["coins"][sym0]["last_scan_epoch"] = ts
                mon["coins"][sym0]["last_scan_kst"] = rec.get("time_kst", "")
        except Exception:
            pass
        # ✅ 스캔이 길어져도 UI에서 "멈춤 의심"이 과도하게 뜨지 않게, 스캔 단계도 heartbeat로 간주
        mon["last_heartbeat_epoch"] = ts
        mon["last_heartbeat_kst"] = rec.get("time_kst", "")
        # Google Sheets에도 SCAN 누적(비동기 큐)
        try:
            gsheet_log_scan(stage=stage, symbol=symbol, tf=tf, signal=signal, score=score, message=message, payload=extra or {})
        except Exception:
            pass
        # ✅ 블로킹 호출(ex.fetch_ohlcv 등) 직전에 stage/heartbeat가 화면에 보이도록 파일을 주기적으로 flush
        try:
            monitor_write_throttled(mon, min_interval_sec=0.8)
        except Exception:
            pass
    except Exception:
        pass


def mon_recent_events(mon: Dict[str, Any], within_min: int = 15) -> List[Dict[str, Any]]:
    try:
        evs = mon.get("events", []) or []
        now = now_kst()
        out = []
        for e in reversed(evs):
            dt = _parse_time_kst(str(e.get("time_kst", "")))
            if not dt:
                continue
            if (now - dt).total_seconds() <= within_min * 60:
                out.append(e)
            else:
                break
        return list(reversed(out))
    except Exception:
        return []


# =========================================================
# ✅ 16) 텔레그램 유틸 (timeout/retry + 채널/그룹 라우팅)
# =========================================================
def _tg_post(url: str, data: Dict[str, Any]):
    if retry is None:
        return requests.post(url, data=data, timeout=HTTP_TIMEOUT_SEC)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
    def _do():
        r = requests.post(url, data=data, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r

    return _do()


def tg_admin_chat_ids() -> List[str]:
    """
    Telegram Bot API에서 개인 DM의 chat_id는 보통 user_id와 동일합니다.
    - 단, 봇이 해당 사용자에게 DM을 보내려면 사용자가 먼저 봇을 시작(/start)해야 합니다.
    """
    try:
        if not TG_ADMIN_IDS:
            return []
        ids = []
        for x in sorted(list(TG_ADMIN_IDS)):
            try:
                ids.append(str(int(x)))
            except Exception:
                continue
        return ids
    except Exception:
        return []


def tg_send_chat(chat_id: Any, text: str):
    """특정 chat_id(채널/그룹/개인)로 직접 전송."""
    if not tg_token:
        return
    if chat_id is None:
        return
    cid = str(chat_id).strip()
    if not cid:
        return
    try:
        _tg_post(f"https://api.telegram.org/bot{tg_token}/sendMessage", {"chat_id": cid, "text": text})
    except Exception:
        pass


def _tg_chat_id_by_target(target: str, cfg: Dict[str, Any]) -> List[str]:
    target = (target or "default").lower()
    if target == "channel":
        return [tg_id_channel] if tg_id_channel else []
    if target == "group":
        return [tg_id_group] if tg_id_group else []
    if target == "admin":
        ids = tg_admin_chat_ids()
        if ids:
            return ids
        # fallback: 기존 동작(그룹/디폴트)
        if tg_id_group:
            return [tg_id_group]
        return [tg_id_default] if tg_id_default else []
    if target == "both":
        ids = []
        if tg_id_channel:
            ids.append(tg_id_channel)
        if tg_id_group and tg_id_group != tg_id_channel:
            ids.append(tg_id_group)
        return ids
    # default: 이전 동작 유지
    return [tg_id_default] if tg_id_default else []


def tg_send(text: str, target: str = "default", cfg: Optional[Dict[str, Any]] = None):
    if not tg_token:
        return
    # 요구사항: Telegram 상태/라우팅이 전역 config가 아니라 최신 load_settings() 기준으로 일치
    cfg = cfg or load_settings()
    ids = _tg_chat_id_by_target(target, cfg)
    for cid in ids:
        if not cid:
            continue
        try:
            _tg_post(f"https://api.telegram.org/bot{tg_token}/sendMessage", {"chat_id": cid, "text": text})
        except Exception:
            pass


def tg_send_menu(cfg: Optional[Dict[str, Any]] = None):
    if not tg_token:
        return
    cfg = cfg or load_settings()
    kb = {
        "inline_keyboard": [
            [{"text": "📡 상태", "callback_data": "status"}, {"text": "👁️ AI시야", "callback_data": "vision"}],
            [{"text": "📊 포지션", "callback_data": "position"}, {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "📜 일지(최근)", "callback_data": "log"}, {"text": "🧾 일지상세", "callback_data": "log_detail_help"}],
            [{"text": "🔎 강제스캔", "callback_data": "scan"}, {"text": "🎚️ /mode", "callback_data": "mode_help"}],
            [{"text": "📎 시트", "callback_data": "gsheet"}, {"text": "🛑 전량청산", "callback_data": "close_all"}],
        ]
    }
    # ✅ 사용자의 요구: TG_TARGET_CHAT_ID는 채널로(알림/결과),
    #    관리/버튼은 TG_ADMIN_USER_IDS(관리자 DM)로 보내기.
    # - admin ids가 있으면 admin에게, 없으면 group(default)에게.
    to_ids = tg_admin_chat_ids() or ([tg_id_group] if tg_id_group else ([tg_id_default] if tg_id_default else []))
    if not to_ids:
        return
    try:
        for cid in to_ids:
            _tg_post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                {
                    "chat_id": cid,
                "text": "✅ /menu\n/status /positions /scan /mode auto|scalping|swing /log <id> /gsheet\n(일지상세: '일지상세 <ID>')",
                "reply_markup": json.dumps(kb, ensure_ascii=False),
            },
        )
    except Exception:
        pass


def tg_answer_callback(cb_id: str):
    if not tg_token:
        return
    try:
        _tg_post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", {"callback_query_id": cb_id})
    except Exception:
        pass


# =========================================================
# ✅ 16.2) 오류 알림(관리자 DM) - 요구사항
# - "코드에서 나오는 모든 오류"를 TG_ADMIN_USER_IDS로 전송(스팸 방지용 dedup/쿨다운 포함)
# =========================================================
_ERR_NOTIFY_LOCK = threading.RLock()
_ERR_NOTIFY_LAST: Dict[str, float] = {}


def notify_admin_error(where: str, err: BaseException, context: Optional[Dict[str, Any]] = None, tb: str = "", min_interval_sec: float = 60.0):
    """
    안전한 오류 알림:
    - Telegram 전송 실패가 또 다른 예외를 만들지 않게 100% swallow
    - 동일 오류는 min_interval_sec 동안 중복 전송 방지
    """
    try:
        if not tg_token:
            return
        if not TG_ADMIN_IDS:
            return
        where_s = str(where or "unknown")[:120]
        msg_s = str(err)[:300]
        sig = f"{where_s}|{type(err).__name__}|{msg_s}"

        now = time.time()
        with _ERR_NOTIFY_LOCK:
            last = float(_ERR_NOTIFY_LAST.get(sig, 0) or 0)
            if (now - last) < float(min_interval_sec):
                return
            _ERR_NOTIFY_LAST[sig] = now
            # 메모리 누수 방지(최대 300개 유지)
            if len(_ERR_NOTIFY_LAST) > 300:
                # 오래된 것부터 제거
                for k in sorted(_ERR_NOTIFY_LAST, key=_ERR_NOTIFY_LAST.get)[:80]:
                    _ERR_NOTIFY_LAST.pop(k, None)

        tb_text = tb or ""
        if not tb_text:
            try:
                tb_text = traceback.format_exc()
            except Exception:
                tb_text = ""
        tb_short = ""
        if tb_text:
            try:
                tb_lines = tb_text.strip().splitlines()
                tb_short = "\n".join(tb_lines[-8:])
            except Exception:
                tb_short = ""

        ctx_txt = ""
        if context:
            try:
                ctx_txt = safe_json_dumps(context, limit=900)
            except Exception:
                ctx_txt = str(context)[:900]

        text = (
            f"🧨 오류 알림\n"
            f"- where: {where_s}\n"
            f"- time_kst: {now_kst_str()}\n"
            f"- code: {CODE_VERSION}\n"
            f"- error: {type(err).__name__}: {msg_s}\n"
        )
        if ctx_txt:
            text += f"- ctx: {ctx_txt}\n"
        if tb_short:
            text += f"- tb(last):\n{tb_short}\n"

        # Telegram 길이 제한 보호
        if len(text) > 3500:
            text = text[:3500] + "..."

        # 관리자 DM으로만 전송
        tg_send(text, target="admin", cfg=load_settings())
        try:
            # Google Sheets에도 ERROR 이벤트 남김(가능할 때만)
            # ✅ Google Sheets 자체 장애일 때는 무한 루프/스팸을 막기 위해 시트로 ERROR를 다시 쓰지 않는다.
            if not str(where_s).upper().startswith("GSHEET"):
                gsheet_log_event("ERROR", message=f"{where_s}: {type(err).__name__}", payload={"msg": msg_s, "ctx": context or {}})
        except Exception:
            pass
    except Exception:
        pass


# =========================================================
# ✅ 16.3) Global excepthook (best-effort)
# - 잡히지 않은 예외(특히 스레드)도 관리자 DM으로 전달
# =========================================================
def install_global_error_hooks():
    try:
        import sys as _sys
        import threading as _threading

        def _fmt_tb(exc_type, exc, tb_obj) -> str:
            try:
                return "".join(traceback.format_exception(exc_type, exc, tb_obj))
            except Exception:
                try:
                    return traceback.format_exc()
                except Exception:
                    return ""

        # sys.excepthook (메인 스레드 unhandled)
        def _sys_hook(exc_type, exc, tb_obj):  # type: ignore
            try:
                notify_admin_error("SYS_EXCEPTHOOK", exc, tb=_fmt_tb(exc_type, exc, tb_obj), min_interval_sec=10.0)
            except Exception:
                pass
            # 기본 훅도 호출(가능하면)
            try:
                _sys.__excepthook__(exc_type, exc, tb_obj)
            except Exception:
                pass

        _sys.excepthook = _sys_hook

        # threading.excepthook (Python 3.8+)
        if hasattr(_threading, "excepthook"):
            _orig_thread_hook = _threading.excepthook

            def _th_hook(args):  # type: ignore
                try:
                    where = f"THREAD_EXCEPTHOOK:{getattr(args.thread, 'name', '')}"
                    notify_admin_error(where, args.exc_value, tb=_fmt_tb(args.exc_type, args.exc_value, args.exc_traceback), min_interval_sec=10.0)
                except Exception:
                    pass
                try:
                    _orig_thread_hook(args)
                except Exception:
                    pass

            _threading.excepthook = _th_hook

    except Exception:
        pass


# =========================================================
# ✅ 16.5) Telegram Update Long Polling Thread (daemon)
# - 요구사항: getUpdates long polling을 별도 스레드로 수행(트레이딩 루프 멈춤 방지)
# =========================================================
_TG_UPDATES_QUEUE: List[Dict[str, Any]] = []
_TG_UPDATES_LOCK = threading.RLock()


def tg_updates_push(up: Dict[str, Any]) -> None:
    try:
        with _TG_UPDATES_LOCK:
            _TG_UPDATES_QUEUE.append(up)
            if len(_TG_UPDATES_QUEUE) > 400:
                _TG_UPDATES_QUEUE[:] = _TG_UPDATES_QUEUE[-300:]
    except Exception:
        pass


def tg_updates_pop_all(max_items: int = 50) -> List[Dict[str, Any]]:
    try:
        with _TG_UPDATES_LOCK:
            if not _TG_UPDATES_QUEUE:
                return []
            items = _TG_UPDATES_QUEUE[:max_items]
            del _TG_UPDATES_QUEUE[: len(items)]
        return items
    except Exception:
        return []


def telegram_polling_thread():
    """
    Telegram long polling(getUpdates).
    - TG_TOKEN 없으면 비활성
    - 네트워크 오류에도 지속 실행(backoff)
    """
    offset = 0
    backoff = 1.0
    while True:
        if not tg_token:
            time.sleep(2.0)
            continue
        try:
            url = f"https://api.telegram.org/bot{tg_token}/getUpdates"
            params = {"offset": offset + 1, "timeout": 25}
            r = requests.get(url, params=params, timeout=40)
            data = {}
            try:
                data = r.json()
            except Exception:
                data = {"ok": False}

            if data.get("ok"):
                backoff = 1.0
                for up in data.get("result", []) or []:
                    try:
                        offset = max(offset, int(up.get("update_id", offset)))
                    except Exception:
                        pass
                    tg_updates_push(up)
            else:
                time.sleep(0.4)
        except Exception as e:
            # 폴링 오류도 관리자에게 알림(과다 스팸 방지: 120s dedup)
            notify_admin_error("TG_POLL_THREAD", e, context={"offset": offset}, min_interval_sec=120.0)
            time.sleep(backoff)
            backoff = float(clamp(backoff * 1.5, 1.0, 15.0))


def tg_is_admin(user_id: Optional[int]) -> bool:
    try:
        uid = int(user_id or 0)
    except Exception:
        uid = 0
    # TG_ADMIN_USER_IDS가 비어있으면 제한 없이 허용
    if not TG_ADMIN_IDS:
        return True
    return uid in TG_ADMIN_IDS


# =========================================================
# ✅ 17) 자동매매 핵심 스레드 (기능 유지 + 주기보고 + 스타일전환 + 안정성)
# =========================================================
def _fmt_pos_line(sym: str, side: str, lev: Any, roi: float, upnl: float, style: str = "") -> str:
    emo = "🟢" if roi >= 0 else "🔴"
    s_txt = f" | 스타일:{style}" if style else ""
    return f"{emo} {sym} {('롱' if side=='long' else '숏')} x{lev} | ROI {roi:.2f}% | PnL {upnl:.2f} USDT{s_txt}"


def _style_for_entry(
    symbol: str,
    decision: str,
    trend_short: str,
    trend_long: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    style, conf, reason = decide_style_rule_based(decision, trend_short, trend_long)
    # 애매하면 AI로 2차 판단
    if cfg.get("style_auto_enable", True) and conf <= 60:
        ai = ai_decide_style(symbol, decision, trend_short, trend_long, cfg)
        # AI가 스윙이라고 강하게 말하면 반영
        if int(ai.get("confidence", 0)) >= 70:
            style = ai.get("style", style)
            conf = int(ai.get("confidence", conf))
            reason = str(ai.get("reason", reason))
    return {"style": style, "confidence": conf, "reason": reason}


def _trend_align(trend_txt: str, side: str) -> bool:
    if side == "long":
        return "상승" in (trend_txt or "")
    if side == "short":
        return "하락" in (trend_txt or "")
    return False


def _maybe_switch_style_for_open_position(
    ex,
    sym: str,
    pos_side: str,
    tgt: Dict[str, Any],
    cfg: Dict[str, Any],
    mon: Dict[str, Any],
) -> Dict[str, Any]:
    """
    포지션 보유 중 차트 상황이 바뀌면 스타일을 전환(스윙->스캘핑 청산모드, 스캘핑->스윙 전환)
    """
    try:
        fast = int(cfg.get("ma_fast", 7))
        slow = int(cfg.get("ma_slow", 99))

        short_tf = str(cfg.get("timeframe", "5m"))
        long_tf = str(cfg.get("trend_filter_timeframe", "1h"))

        short_trend = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=25)
        long_trend = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))

        cur_style = str(tgt.get("style", "스캘핑"))
        # 추천 스타일(룰 기반)
        dec = "buy" if pos_side == "long" else "sell"
        rec = _style_for_entry(sym, dec, short_trend, long_trend, cfg)
        rec_style = rec.get("style", cur_style)
        # ✅ 레짐(스캘핑/스윙) 강제/자동 선택
        # 요구사항: "시간 기반 최소유지기간(style_lock_minutes) 강제 금지"
        # 대신 confirm2/hysteresis로 흔들림 방지
        regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
        if regime_mode in ["scalping", "scalp", "short"]:
            rec_style = "스캘핑"
        elif regime_mode in ["swing", "long"]:
            rec_style = "스윙"

        switch_ctl = str(cfg.get("regime_switch_control", "confirm2")).lower().strip()  # confirm2|hysteresis|off
        if regime_mode == "auto" and rec_style == cur_style:
            # 연속 확인 로직이 "연속"이 되도록, 동일 스타일이 나오면 pending을 초기화
            try:
                tgt["_pending_style"] = ""
                tgt["_pending_style_count"] = 0
            except Exception:
                pass
        if regime_mode == "auto" and rec_style != cur_style:
            if switch_ctl == "confirm2":
                pending = str(tgt.get("_pending_style", ""))
                cnt = int(tgt.get("_pending_style_count", 0) or 0)
                if pending == rec_style:
                    cnt += 1
                else:
                    pending = rec_style
                    cnt = 1
                tgt["_pending_style"] = pending
                tgt["_pending_style_count"] = cnt
                if cnt < 2:
                    # 2회 연속 동일 레짐일 때만 전환
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt
                # 전환 확정
                tgt["_pending_style"] = ""
                tgt["_pending_style_count"] = 0
            elif switch_ctl == "hysteresis":
                bias = float(tgt.get("_regime_bias", 0.5) or 0.5)  # 0=스캘핑, 1=스윙
                step = float(cfg.get("regime_hysteresis_step", 0.55))
                enter_swing = float(cfg.get("regime_hysteresis_enter_swing", 0.75))
                enter_scalp = float(cfg.get("regime_hysteresis_enter_scalp", 0.25))
                if rec_style == "스윙":
                    bias = min(1.0, bias + step)
                else:
                    bias = max(0.0, bias - step)
                tgt["_regime_bias"] = bias
                # 임계값을 넘을 때만 전환
                if cur_style == "스캘핑" and bias < enter_swing:
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt
                if cur_style == "스윙" and bias > enter_scalp:
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt

        if rec_style != cur_style:
            # 전환 기록
            tgt["style"] = rec_style
            tgt["style_confidence"] = int(rec.get("confidence", 0))
            tgt["style_reason"] = str(rec.get("reason", ""))[:240]
            tgt["style_last_switch_epoch"] = time.time()
            tgt["trend_short_now"] = f"{short_tf} {short_trend}"
            tgt["trend_long_now"] = f"{long_tf} {long_trend}"

            # 전환 시 목표 보정: 스윙->스캘핑이면 "빨리 청산" 모드로 목표 낮춤
            if rec_style == "스캘핑":
                # 기존 TP/SL이 너무 크면 스캘핑 범위로 조임
                tgt["tp"] = float(clamp(float(tgt.get("tp", 3.0)), float(cfg.get("scalp_tp_roi_min", 0.8)), float(cfg.get("scalp_tp_roi_max", 6.0))))
                tgt["sl"] = float(clamp(float(tgt.get("sl", 2.0)), float(cfg.get("scalp_sl_roi_min", 0.8)), float(cfg.get("scalp_sl_roi_max", 5.0))))
                tgt["scalp_exit_mode"] = True
            else:
                tgt["tp"] = float(clamp(float(tgt.get("tp", 6.0)), float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))
                tgt["sl"] = float(clamp(float(tgt.get("sl", 3.0)), float(cfg.get("swing_sl_roi_min", 1.5)), float(cfg.get("swing_sl_roi_max", 30.0))))
                tgt["scalp_exit_mode"] = False

            mon_add_event(mon, "STYLE_SWITCH", sym, f"{cur_style} → {rec_style}", {"reason": tgt.get("style_reason", "")})
            # 사용자 체감용: 스타일 전환 즉시 알림(채널/이벤트 라우팅)
            try:
                tg_send(
                    f"🔄 스타일 전환\n- 코인: {sym}\n- {cur_style} → {rec_style}\n- 단기({short_tf}): {short_trend}\n- 장기({long_tf}): {long_trend}\n- 이유: {tgt.get('style_reason','')}",
                    target=cfg.get("tg_route_events_to", "channel"),
                    cfg=cfg,
                )
            except Exception:
                pass
        else:
            tgt["style_reco"] = rec_style
            tgt["trend_short_now"] = f"{short_tf} {short_trend}"
            tgt["trend_long_now"] = f"{long_tf} {long_trend}"

    except Exception:
        pass
    return tgt


def _should_convert_scalp_to_swing(tgt: Dict[str, Any], roi: float, cfg: Dict[str, Any], long_align: bool) -> bool:
    try:
        if str(tgt.get("style", "")) != "스캘핑":
            return False
        entry_epoch = float(tgt.get("entry_epoch", 0) or 0)
        if not entry_epoch:
            return False
        hold_min = (time.time() - entry_epoch) / 60.0
        if hold_min < float(cfg.get("scalp_max_hold_minutes", 25)):
            return False
        # 너무 큰 손실이면 전환보다 정리가 낫다(기본)
        if roi < float(cfg.get("scalp_to_swing_min_roi", -12.0)):
            return False
        if cfg.get("scalp_to_swing_require_long_align", True) and not long_align:
            return False
        return True
    except Exception:
        return False


def _try_scalp_to_swing_dca(ex, sym: str, side: str, cur_px: float, tgt: Dict[str, Any], rt: Dict[str, Any], cfg: Dict[str, Any], mon: Dict[str, Any]) -> bool:
    """
    스캘핑 포지션이 스윙으로 전환해도 될 때(장기추세 align 등) 1회 추매 + 목표 RR 상향
    """
    try:
        if not cfg.get("use_dca", True):
            return False
        # 추매는 스윙 전환 시점에만 허용(스캘핑 기본 추매X)
        trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
        dca_count = int(trade_state.get("dca_count", 0))
        dca_max = max(0, int(cfg.get("dca_max_count", 1)))
        if dca_count >= max(1, dca_max):
            return False

        free, _ = safe_fetch_balance(ex)
        base_entry = float(tgt.get("entry_usdt", 0.0))
        dca_add_pct = float(cfg.get("dca_add_pct", 50.0))
        add_usdt = base_entry * (dca_add_pct / 100.0)
        if add_usdt > free:
            add_usdt = free * 0.5
        if add_usdt < 5:
            return False

        lev = int(float(tgt.get("lev", MODE_RULES.get(cfg.get("trade_mode", "안전모드"), MODE_RULES["안전모드"])["lev_min"])) or 1)
        set_leverage_safe(ex, sym, lev)
        qty = to_precision_qty(ex, sym, (add_usdt * lev) / max(cur_px, 1e-9))
        if qty <= 0:
            return False
        ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
        if ok:
            trade_state["dca_count"] = dca_count + 1
            save_runtime(rt)
            mon_add_event(mon, "DCA_CONVERT", sym, f"스캘핑→스윙 전환 추매 {add_usdt:.2f} USDT", {"add_usdt": add_usdt})
            try:
                gsheet_log_trade(
                    stage="DCA_CONVERT",
                    symbol=sym,
                    trade_id=str(tgt.get("trade_id", "") or ""),
                    message=f"add_usdt={add_usdt:.2f}",
                    payload={"add_usdt": add_usdt, "qty": qty, "lev": lev},
                )
            except Exception:
                pass
            return True
    except Exception:
        return False
    return False


def _swing_partial_tp_levels(tp_roi: float, cfg: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """
    returns: [(trigger_roi, close_frac, label), ...]
    """
    try:
        steps = [
            (float(cfg.get("swing_partial_tp1_at_tp_frac", 0.35)), float(cfg.get("swing_partial_tp1_close_pct", 33)) / 100.0, "TP1"),
            (float(cfg.get("swing_partial_tp2_at_tp_frac", 0.60)), float(cfg.get("swing_partial_tp2_close_pct", 33)) / 100.0, "TP2"),
            (float(cfg.get("swing_partial_tp3_at_tp_frac", 0.85)), float(cfg.get("swing_partial_tp3_close_pct", 34)) / 100.0, "TP3"),
        ]
        out = []
        for frac, close_frac, label in steps:
            if frac <= 0 or close_frac <= 0:
                continue
            out.append((max(0.1, tp_roi * frac), float(clamp(close_frac, 0.01, 0.95)), label))
        # 트리거 기준 오름차순 정렬
        out.sort(key=lambda x: x[0])
        return out
    except Exception:
        return []


def telegram_thread(ex):
    offset = 0
    mon = monitor_init()

    # runtime에서 open_targets 복구(스레드 재시작에도 목표/스타일 일부 유지)
    rt_boot = load_runtime()
    active_targets: Dict[str, Dict[str, Any]] = {}
    try:
        ot = rt_boot.get("open_targets", {}) or {}
        if isinstance(ot, dict):
            active_targets.update({k: v for k, v in ot.items() if isinstance(v, dict)})
    except Exception:
        pass

    # ✅ 시작 EVENT (Google Sheets/모니터)
    try:
        mon_add_event(mon, "START", "", "봇 시작", {"sandbox": bool(IS_SANDBOX)})
        gsheet_log_event("START", message="bot_started", payload={"sandbox": bool(IS_SANDBOX), "boot_time_kst": mon.get("_boot_time_kst", "")})
    except Exception:
        pass

    # 부팅 메시지(그룹: 메뉴, 채널: 시작 알림)
    cfg_boot = load_settings()
    boot_msg = f"🚀 AI 봇 가동 시작! (모의투자)\n- code: {CODE_VERSION}\n명령: /menu /status /positions /scan /mode /log /gsheet"
    tg_send(boot_msg, target="channel", cfg=cfg_boot)
    # ✅ 요구: TG_TARGET_CHAT_ID는 채널(브로드캐스트), 관리는 관리자 DM으로(중복/스팸 방지)
    if TG_ADMIN_IDS:
        tg_send(boot_msg, target="admin", cfg=cfg_boot)
    elif tg_id_group and tg_id_group != tg_id_channel:
        tg_send(boot_msg, target="group", cfg=cfg_boot)
    tg_send_menu(cfg=cfg_boot)

    # 주기 작업 스케줄러 상태
    next_report_ts = 0.0
    next_heartbeat_ts = 0.0  # 요구사항: 15분(900초) 고정 하트비트
    next_vision_ts = 0.0
    last_daily_brief_date = ""
    last_daily_brief_attempt_epoch = 0.0
    last_export_attempt_epoch = 0.0

    backoff_sec = 1.0

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()
            mode = cfg.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])

            # =========================================================
            # ✅ 루프 하트비트(즉시 기록)
            # - 외부시황/거래소 호출이 느리거나 일시 장애여도 UI에서 "멈춤 의심"이 과도하게 뜨지 않게
            # - (중요) trade heartbeat(텔레그램 15분 리포트)와 별개로, '스레드 생존' 신호다.
            # =========================================================
            try:
                now_str0 = now_kst_str()
                mon["loop_stage"] = "LOOP_START"
                mon["loop_stage_kst"] = now_str0
                mon["last_heartbeat_epoch"] = time.time()
                mon["last_heartbeat_kst"] = now_str0
                mon["auto_trade"] = bool(cfg.get("auto_trade", False))
                mon["trade_mode"] = mode
                mon["pause_until"] = rt.get("pause_until", 0)
                mon["consec_losses"] = rt.get("consec_losses", 0)
                mon["trend_filter_policy"] = cfg.get("trend_filter_policy", "ALLOW_SCALP")
                mon["code_version"] = CODE_VERSION
                monitor_write_throttled(mon, 0.5)
            except Exception:
                pass

            # ✅ 매일 아침 브리핑(한 번만)
            try:
                if cfg.get("daily_btc_brief_enable", True):
                    h = int(cfg.get("daily_btc_brief_hour_kst", 9))
                    m = int(cfg.get("daily_btc_brief_minute_kst", 0))
                    now = now_kst()
                    today = today_kst_str()
                    # 이미 저장되어 있으면 사용
                    if rt.get("daily_btc_brief", {}).get("date") == today:
                        last_daily_brief_date = today
                    # 스케줄 시각 이후, 오늘 브리핑이 없으면 생성
                    if last_daily_brief_date != today and (now.hour > h or (now.hour == h and now.minute >= m)):
                        # 실패 시 무한 루프/멈춤 방지: 재시도는 20분 간격으로만
                        if (time.time() - float(last_daily_brief_attempt_epoch or 0.0)) < 20 * 60:
                            pass
                        else:
                            last_daily_brief_attempt_epoch = time.time()
                            # UI에 "어디서 멈췄는지" 보이도록 stage 먼저 기록
                            try:
                                mon["loop_stage"] = "DAILY_BRIEF"
                                mon["loop_stage_kst"] = now_kst_str()
                                mon["last_heartbeat_epoch"] = time.time()
                                mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                                monitor_write_throttled(mon, 0.2)
                            except Exception:
                                pass

                            brief = {}
                            try:
                                # feedparser/외부 네트워크가 걸려도 스레드가 오래 멈추지 않도록 hard-timeout
                                brief = _call_with_timeout(lambda: fetch_daily_btc_brief(cfg), 35)
                            except FuturesTimeoutError:
                                mon_add_event(mon, "DAILY_BRIEF_TIMEOUT", "", "daily brief timeout", {"timeout_sec": 35})
                                notify_admin_error("DAILY_BRIEF", RuntimeError("timeout"), context={"timeout_sec": 35}, min_interval_sec=300.0)
                                brief = {}
                            except Exception as e:
                                mon_add_event(mon, "DAILY_BRIEF_FAIL", "", f"{type(e).__name__}: {e}"[:140], {})
                                notify_admin_error("DAILY_BRIEF", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                                brief = {}

                            if isinstance(brief, dict) and brief:
                                rt["daily_btc_brief"] = brief
                                save_runtime(rt)
                                last_daily_brief_date = today
                                # 채널로 브리핑 전송
                                try:
                                    if brief.get("items"):
                                        bias = str(brief.get("bias", "중립"))
                                        risk = str(brief.get("risk", "보통"))
                                        lines = [f"🌅 오늘 아침 BTC 브리핑 ({today})", f"- 시황 톤: {bias} | 리스크: {risk}"]
                                        for it in brief["items"][: int(cfg.get("daily_btc_brief_max_items", 5))]:
                                            emo = str(it.get("emoji", "📰"))
                                            title = str(it.get("title", ""))[:90]
                                            note = str(it.get("note", ""))[:90]
                                            if note:
                                                lines.append(f"{emo} {title}\n   └ {note}")
                                            else:
                                                lines.append(f"{emo} {title}")
                                        tg_send("\n".join(lines), target="channel", cfg=cfg)
                                except Exception:
                                    pass
            except Exception:
                pass

            # 외부 시황 갱신(캐시 포함) + daily brief 포함
            try:
                mon["loop_stage"] = "EXTERNAL_CONTEXT"
                mon["loop_stage_kst"] = now_kst_str()
                # stage를 먼저 flush(외부 호출이 길어져도 UI에서 어디서 멈췄는지 보이게)
                try:
                    monitor_write_throttled(mon, 0.2)
                except Exception:
                    pass
                try:
                    ext = _call_with_timeout(lambda: build_external_context(cfg, rt=rt), EXTERNAL_CONTEXT_TIMEOUT_SEC)
                except FuturesTimeoutError:
                    # 타임아웃이면 이전 ext(있으면)로 유지하고, 이벤트로만 남김
                    prev = mon.get("external") if isinstance(mon.get("external"), dict) else {}
                    if prev:
                        ext = dict(prev)
                        ext["asof_kst"] = now_kst_str()
                        ext["timeout"] = True
                    else:
                        ext = {"enabled": False, "error": "external_context_timeout", "asof_kst": now_kst_str(), "timeout": True}
                    mon_add_event(mon, "EXTERNAL_TIMEOUT", "", "external_context timeout", {"timeout_sec": int(EXTERNAL_CONTEXT_TIMEOUT_SEC)})
            except Exception as e:
                ext = {"enabled": False, "error": str(e)[:240], "asof_kst": now_kst_str()}
                notify_admin_error("EXTERNAL_CONTEXT", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                mon_add_event(mon, "EXTERNAL_FAIL", "", f"{e}"[:140], {})
            mon["external"] = ext

            # ✅ 일별 내보내기 자동(새벽 00시대, 전일 기준)
            try:
                if cfg.get("export_daily_enable", True):
                    now0 = now_kst()
                    if now0.hour == 0 and now0.minute < 10:
                        today = today_kst_str()
                        if str(rt.get("last_export_date", "")) != today:
                            # 실패 시 반복 시도로 스레드가 멈추지 않게, 재시도는 10분 간격으로만
                            if (time.time() - float(last_export_attempt_epoch or 0.0)) < 10 * 60:
                                pass
                            else:
                                last_export_attempt_epoch = time.time()
                                # stage 먼저 기록
                                try:
                                    mon["loop_stage"] = "DAILY_EXPORT"
                                    mon["loop_stage_kst"] = now_kst_str()
                                    mon["last_heartbeat_epoch"] = time.time()
                                    mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                                    monitor_write_throttled(mon, 0.2)
                                except Exception:
                                    pass

                                yday = (now0 - timedelta(days=1)).strftime("%Y-%m-%d")
                                res = {}
                                try:
                                    # Excel/CSV/gsheet export가 네트워크/파일 이슈로 오래 걸려도 스레드가 멈추지 않게 hard-timeout
                                    res = _call_with_timeout(lambda: export_trade_log_daily(yday, cfg), 40)
                                except FuturesTimeoutError:
                                    mon_add_event(mon, "EXPORT_TIMEOUT", "", "daily export timeout", {"timeout_sec": 40})
                                    notify_admin_error(
                                        "DAILY_EXPORT", RuntimeError("timeout"), context={"timeout_sec": 40, "yday": yday}, min_interval_sec=300.0
                                    )
                                    res = {"ok": False, "error": "timeout"}
                                except Exception as e:
                                    mon_add_event(mon, "EXPORT_FAIL", "", f"{type(e).__name__}: {e}"[:140], {"yday": yday})
                                    notify_admin_error("DAILY_EXPORT", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                                    res = {"ok": False, "error": str(e)[:240]}
                                # 스팸/정체 방지: 하루 1회만 시도하도록 날짜는 시도 시점에 고정
                                rt["last_export_date"] = today
                                save_runtime(rt)
                                # 채널로 완료 보고(스팸 방지: 하루 1회)
                                try:
                                    if isinstance(res, dict) and res.get("ok"):
                                        msg = (
                                            f"📤 일별 일지 내보내기({yday})\n"
                                            f"- rows: {res.get('rows')}\n"
                                            f"- xlsx: {res.get('excel_path','')}\n"
                                            f"- csv: {res.get('csv_path','')}\n"
                                            f"- gsheet: {res.get('gsheet','')}"
                                        )
                                        tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg)
                                except Exception:
                                    pass
            except Exception:
                pass

            # 하트비트
            mon["last_heartbeat_epoch"] = time.time()
            mon["last_heartbeat_kst"] = now_kst_str()
            mon["auto_trade"] = bool(cfg.get("auto_trade", False))
            mon["trade_mode"] = mode
            mon["pause_until"] = rt.get("pause_until", 0)
            mon["consec_losses"] = rt.get("consec_losses", 0)
            mon["trend_filter_policy"] = cfg.get("trend_filter_policy", "ALLOW_SCALP")

            # ✅ 하트비트(요구사항: 15분=900초마다)
            try:
                if tg_token:
                    if next_heartbeat_ts <= 0:
                        # 부팅 직후 첫 하트비트는 조금 지연(스팸 방지)
                        next_heartbeat_ts = time.time() + 20
                    if time.time() >= next_heartbeat_ts:
                        # stage 먼저 기록(잔고/포지션 조회가 느려도 UI에 표시)
                        try:
                            mon["loop_stage"] = "TG_HEARTBEAT"
                            mon["loop_stage_kst"] = now_kst_str()
                            mon["last_heartbeat_epoch"] = time.time()
                            mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                            monitor_write_throttled(mon, 0.2)
                        except Exception:
                            pass
                        free, total = safe_fetch_balance(ex)
                        realized = float(rt.get("daily_realized_pnl", 0.0) or 0.0)
                        regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                        regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")

                        # 포지션 요약
                        pos_lines = []
                        ps = safe_fetch_positions(ex, TARGET_COINS)
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if act:
                            for p in act[:10]:
                                sym = p.get("symbol", "")
                                side = position_side_normalize(p)
                                roi = float(position_roi_percent(p))
                                upnl = float(p.get("unrealizedPnl") or 0.0)
                                lev = p.get("leverage", "?")
                                style = str((active_targets.get(sym, {}) or {}).get("style", ""))
                                pos_lines.append(_fmt_pos_line(sym, side, lev, roi, upnl, style=style))
                        else:
                            pos_lines.append("⚪ 무포지션(관망)")

                        last_scan_kst = mon.get("last_scan_kst", "-")
                        last_hb_kst = mon.get("last_heartbeat_kst", "-")
                        txt = "\n".join(
                            [
                                "💓 하트비트(15분)",
                                f"- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}",
                                f"- 모드: {mode}",
                                f"- 레짐: {regime_txt}",
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})",
                                f"- 리얼손익(오늘): {realized:.2f} USDT",
                                f"- 포지션:",
                                *[f"  {x}" for x in pos_lines],
                                f"- 마지막 스캔: {last_scan_kst}",
                                f"- 마지막 하트비트: {last_hb_kst}",
                            ]
                        )
                        tg_send(txt, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg)
                        try:
                            mon["last_tg_heartbeat_epoch"] = time.time()
                            mon["last_tg_heartbeat_kst"] = now_kst_str()
                        except Exception:
                            pass
                        try:
                            gsheet_log_event("HEARTBEAT", message=f"regime={regime_txt} pos={len(act)} bal={total:.2f}", payload={"regime": regime_txt, "positions": len(act), "total": total, "free": free})
                        except Exception:
                            pass
                        next_heartbeat_ts = time.time() + 900
            except Exception:
                pass

            # ✅ 주기 리포트(15분 기본)
            try:
                if cfg.get("tg_enable_periodic_report", True):
                    interval = max(3, int(cfg.get("report_interval_min", 15)))
                    # 하트비트(15분)는 별도 고정 스케줄이므로, 동일(15)이면 중복 전송 방지
                    if interval == 15:
                        # heartbeat가 이미 15분 고정으로 전송되므로, 별도 주기 리포트는 스킵
                        next_report_ts = 0.0
                    else:
                        if next_report_ts <= 0:
                            next_report_ts = time.time() + interval * 60
                        if time.time() >= next_report_ts:
                            free, total = safe_fetch_balance(ex)
                            # 포지션 요약
                            pos_lines = []
                            ps = safe_fetch_positions(ex, TARGET_COINS)
                            act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                            if act:
                                for p in act[:8]:
                                    sym = p.get("symbol", "")
                                    side = position_side_normalize(p)
                                    roi = float(position_roi_percent(p))
                                    upnl = float(p.get("unrealizedPnl") or 0.0)
                                    lev = p.get("leverage", "?")
                                    try:
                                        tgt0 = (active_targets.get(sym, {}) or {})
                                        style = str(tgt0.get("style", ""))
                                        tp0 = float(tgt0.get("tp", 0) or 0)
                                        sl0 = float(tgt0.get("sl", 0) or 0)
                                        rr0 = (tp0 / max(abs(sl0), 0.01)) if (tp0 and sl0) else 0.0
                                    except Exception:
                                        style, tp0, sl0, rr0 = "", 0.0, 0.0, 0.0
                                    emo = "🟢" if roi >= 0 else "🔴"
                                    pos_lines.append(
                                        f"{emo} {sym} {('롱' if side=='long' else '숏')} x{lev} | ROI {roi:.2f}% | PnL {upnl:.2f} USDT"
                                        f" | 스타일:{style or '-'} | TP {tp0:.2f}% / SL {sl0:.2f}% / RR {rr0:.2f}"
                                    )
                            else:
                                pos_lines.append("⚪ 무포지션(관망)")

                            # 최근 이벤트(지난 interval)
                            evs = mon_recent_events(mon, within_min=interval)
                            ev_lines = []
                            for e in evs[-12:]:
                                ev_lines.append(f"- {e.get('time_kst','')} {e.get('type','')} {e.get('symbol','')} {str(e.get('message',''))[:60]}")
                            if not ev_lines:
                                ev_lines = ["- (이벤트 없음)"]

                            # 외부 시황 요약
                            fg = (ext or {}).get("fear_greed") or {}
                            fg_line = ""
                            if fg:
                                fg_line = f"{fg.get('emoji','')} 공포탐욕 {fg.get('value','?')} ({fg.get('classification','')})"
                            ev_soon = (ext or {}).get("high_impact_events_soon") or []
                            ev_soon_line = " / ".join([f"{x.get('country','')} {x.get('title','')[:18]}" for x in ev_soon[:2]]) if ev_soon else "없음"
                            regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            last_scan_kst = mon.get("last_scan_kst", "-")
                            last_hb_kst = mon.get("last_heartbeat_kst", "-")
                            realized = float(rt.get("daily_realized_pnl", 0.0) or 0.0)

                            txt = "\n".join(
                                [
                                    f"🕒 {interval}분 상황보고",
                                    f"- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}",
                                    f"- 모드: {mode}",
                                    f"- 레짐: {regime_txt}",
                                    f"- 잔고: {total:.2f} USDT (가용 {free:.2f})",
                                    f"- 리얼손익(오늘): {realized:.2f} USDT",
                                    f"- 보유포지션:",
                                    *[f"  {x}" for x in pos_lines],
                                    f"- 최근 이벤트({interval}분):",
                                    *ev_lines,
                                    f"- 마지막 스캔: {last_scan_kst}",
                                    f"- 마지막 하트비트: {last_hb_kst}",
                                    f"- 외부시황: {fg_line}",
                                    f"- 이벤트 임박: {ev_soon_line}",
                                ]
                            )
                            tgt = cfg.get("tg_route_events_to", "channel")
                            tg_send(txt, target=tgt, cfg=cfg)
                            try:
                                gsheet_log_event(
                                    "PERIODIC_REPORT",
                                    message=f"interval={interval} pos={len(act)}",
                                    payload={"interval_min": interval, "positions": len(act), "total": total, "free": free},
                                )
                            except Exception:
                                pass
                            next_report_ts = time.time() + interval * 60
            except Exception:
                pass

            # ✅ 1시간마다 AI 시야 리포트(채널)
            try:
                if cfg.get("tg_enable_hourly_vision_report", True):
                    interval = max(10, int(cfg.get("vision_report_interval_min", 60)))
                    if next_vision_ts <= 0:
                        next_vision_ts = time.time() + interval * 60
                    if time.time() >= next_vision_ts:
                        mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                        coins = mon_now.get("coins", {}) or {}
                        lines = [
                            "👁️ AI 시야 리포트",
                            f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                            f"- 모드: {mon_now.get('trade_mode','-')}",
                            f"- 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                        ]
                        for sym, cs in list(coins.items())[:12]:
                            style = str(cs.get("style_reco", "")) or str(cs.get("style", ""))
                            style_txt = f"[{style}]" if style else ""
                            lines.append(
                                f"- {sym}: {style_txt} {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                            )
                        tg_send("\n".join(lines), target="channel", cfg=cfg)
                        next_vision_ts = time.time() + interval * 60
            except Exception:
                pass

            # ✅ /scan 강제스캔 요청(runtime_state.json)
            force_scan_req = rt.get("force_scan", {}) if isinstance(rt.get("force_scan", {}), dict) else {}
            force_scan_id = str(force_scan_req.get("id", "") or "")
            force_scan_done = bool(force_scan_req.get("done", False))
            force_scan_only = bool(force_scan_req.get("scan_only", True))
            force_scan_symbols = force_scan_req.get("symbols", [])
            if not isinstance(force_scan_symbols, list):
                force_scan_symbols = []
            try:
                force_scan_ts = float(force_scan_req.get("requested_at_epoch", 0) or 0)
            except Exception:
                force_scan_ts = 0.0
            force_scan_pending = bool(force_scan_id) and (not force_scan_done) and (time.time() - force_scan_ts < 60 * 10)

            # ✅ 스캔 루프는 항상 실행(시야/스캔 갱신)
            # - 신규 진입(주문)만 auto_trade/paused/weekend 정책으로 제어
            if True:
                trade_enabled = bool(cfg.get("auto_trade", False))
                force_scan_syms_set = set(force_scan_symbols or [])
                force_scan_summary_lines: List[str] = []

                # 주말 거래 금지: 신규진입만 제한(스캔/시야는 계속)
                # - weekend_block_now는 entry_allowed_global에 반영

                # 일시정지(연속손실)
                paused_now = cfg.get("loss_pause_enable", True) and time.time() < float(rt.get("pause_until", 0))
                if False and paused_now and trade_enabled and not force_scan_pending:
                    mon["global_state"] = "일시정지 중(연속손실/보호)"
                    monitor_write_throttled(mon, 2.0)
                    time.sleep(1.0)
                else:
                    # 신규 진입 허용 여부(강제스캔 scan_only면 '강제로 추가 호출된 AI'로는 진입 금지)
                    weekend_block_now = cfg.get("no_trade_weekend", False) and (now_kst().weekday() in [5, 6])
                    entry_allowed_global = trade_enabled and (not paused_now) and (not weekend_block_now)

                    # 상태 표시(사용자 체감 개선)
                    if force_scan_pending:
                        mon["global_state"] = "강제 스캔 중(/scan)"
                    elif not trade_enabled:
                        mon["global_state"] = "스캔 중(자동매매 OFF)"
                    elif paused_now:
                        mon["global_state"] = "스캔 중(정지: 연속손실 보호)"
                    elif weekend_block_now:
                        mon["global_state"] = "스캔 중(주말: 신규진입 OFF)"
                    else:
                        mon["global_state"] = "스캔/매매 중"

                    # 1) 포지션 관리
                    open_pos_snapshot = []

                    # ✅ 포지션은 1회 스냅샷으로 사용(API 호출 최소화)
                    pos_by_sym: Dict[str, Dict[str, Any]] = {}
                    try:
                        # stage 먼저 기록(fetch_positions가 느려도 UI에 표시)
                        try:
                            mon["loop_stage"] = "FETCH_POSITIONS"
                            mon["loop_stage_kst"] = now_kst_str()
                            mon["last_heartbeat_epoch"] = time.time()
                            mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                            monitor_write_throttled(mon, 0.2)
                        except Exception:
                            pass
                        ps_all = safe_fetch_positions(ex, TARGET_COINS)
                        for p0 in (ps_all or []):
                            try:
                                sym0 = str(p0.get("symbol") or "").strip()
                                if not sym0:
                                    continue
                                if float(p0.get("contracts") or 0) > 0:
                                    pos_by_sym[sym0] = p0
                            except Exception:
                                continue
                    except Exception:
                        pos_by_sym = {}

                    for sym in (TARGET_COINS if entry_allowed_global else []):
                        p = pos_by_sym.get(sym)
                        if not p:
                            continue
                        side = position_side_normalize(p)
                        contracts = float(p.get("contracts") or 0)
                        entry = float(p.get("entryPrice") or 0)
                        roi = float(position_roi_percent(p))
                        cur_px = get_last_price(ex, sym) or entry
                        lev_live = _pos_leverage(p)
                        upnl = float(p.get("unrealizedPnl") or 0.0)

                        tgt = active_targets.get(
                            sym,
                            {
                                "sl": 2.0,
                                "tp": 5.0,
                                "entry_usdt": 0.0,
                                "entry_pct": 0.0,
                                "lev": p.get("leverage", "?"),
                                "reason": "",
                                "trade_id": "",
                                "sl_price": None,
                                "tp_price": None,
                                "sl_price_pct": None,
                                "style": "스캘핑",
                                "entry_epoch": time.time(),
                                "style_last_switch_epoch": time.time(),
                            },
                        )

                        # ✅ 스타일 자동 전환(포지션 보유 중)
                        tgt = _maybe_switch_style_for_open_position(ex, sym, side, tgt, cfg, mon)
                        style_now = str(tgt.get("style", "스캘핑"))

                        # 저장(스레드 재시작 대비)
                        rt.setdefault("open_targets", {})[sym] = tgt
                        save_runtime(rt)

                        sl = float(tgt.get("sl", 2.0))
                        tp = float(tgt.get("tp", 5.0))
                        trade_id = str(tgt.get("trade_id") or "")

                        # 트레일링(가격폭 기준으로만 조임) - 기존 유지
                        if cfg.get("use_trailing_stop", True):
                            if roi >= (tp * 0.5):
                                lev_now = float(tgt.get("lev", p.get("leverage", 1))) or 1.0
                                base_price_sl = float(tgt.get("sl_price_pct") or max(0.25, float(sl) / max(lev_now, 1)))
                                trail_price_pct = max(0.20, base_price_sl * 0.60)
                                trail_roi = trail_price_pct * lev_now
                                sl = min(sl, max(1.2, float(trail_roi)))

                        # SR 가격 트리거
                        hit_sl_by_price = False
                        hit_tp_by_price = False
                        sl_price = tgt.get("sl_price")
                        tp_price = tgt.get("tp_price")
                        if cfg.get("use_sr_stop", True):
                            if sl_price is not None:
                                if side == "long" and cur_px <= float(sl_price):
                                    hit_sl_by_price = True
                                if side == "short" and cur_px >= float(sl_price):
                                    hit_sl_by_price = True
                            if tp_price is not None:
                                if side == "long" and cur_px >= float(tp_price):
                                    hit_tp_by_price = True
                                if side == "short" and cur_px <= float(tp_price):
                                    hit_tp_by_price = True

                        # ✅ 스윙: 부분익절(순환매도 옵션) - 요구사항 반영
                        if style_now == "스윙" and cfg.get("swing_partial_tp_enable", True) and contracts > 0:
                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                            done = set(trade_state.get("partial_tp_done", []) or [])
                            # TP 기반 트리거
                            levels = _swing_partial_tp_levels(tp, cfg)
                            contracts_left = contracts
                            for trig_roi, close_frac, label in levels:
                                if label in done:
                                    continue
                                if roi >= float(trig_roi) and contracts_left > 0:
                                    close_qty = to_precision_qty(ex, sym, contracts_left * close_frac)
                                    # 너무 작은 수량은 스킵
                                    if close_qty <= 0:
                                        done.add(label)
                                        continue
                                    ok = close_position_market(ex, sym, side, close_qty)
                                    if ok:
                                        done.add(label)
                                        # 순환매도(재진입)용 메모리: 부분익절 수량 누적 + 타임스탬프
                                        try:
                                            trade_state["last_partial_tp_epoch"] = time.time()
                                            trade_state["recycle_qty"] = float(trade_state.get("recycle_qty", 0.0) or 0.0) + float(close_qty)
                                        except Exception:
                                            pass
                                        trade_state["partial_tp_done"] = list(done)
                                        save_runtime(rt)
                                        contracts_left = max(0.0, contracts_left - close_qty)
                                        mon_add_event(mon, "PARTIAL_TP", sym, f"{label} 부분익절({close_frac*100:.0f}%)", {"roi": roi, "qty": close_qty})
                                        try:
                                            gsheet_log_trade(
                                                stage="PARTIAL_TP",
                                                symbol=sym,
                                                trade_id=trade_id,
                                                message=f"{label} close_qty={close_qty}",
                                                payload={"label": label, "roi": roi, "qty": close_qty, "contracts_left": contracts_left},
                                            )
                                        except Exception:
                                            pass
                                        # 텔레그램 채널 보고
                                        tg_send(
                                            f"🧩 부분익절({label})\n- 코인: {sym}\n- 스타일: 스윙\n- ROI: +{roi:.2f}%\n- 청산수량: {close_qty}\n- 남은수량: {contracts_left}\n- 일지ID: {trade_id or '-'}",
                                            target=cfg.get("tg_route_events_to", "channel"),
                                            cfg=cfg,
                                        )
                                        # 상세일지 기록
                                        if trade_id:
                                            d = load_trade_detail(trade_id) or {}
                                            evs = d.get("events", []) or []
                                            evs.append({"time": now_kst_str(), "type": "PARTIAL_TP", "label": label, "roi": roi, "qty": close_qty})
                                            d["events"] = evs
                                            save_trade_detail(trade_id, d)

                        # ✅ 스윙: 순환매도(부분익절 후 재진입/리밸런싱) - 옵션 ON일 때만
                        if style_now == "스윙" and cfg.get("swing_recycle_enable", False) and contracts > 0:
                            try:
                                trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                                rc = int(trade_state.get("recycle_count", 0) or 0)
                                rc_max = int(cfg.get("swing_recycle_max_count", 2))
                                cooldown = int(cfg.get("swing_recycle_cooldown_min", 20)) * 60
                                last_tp_epoch = float(trade_state.get("last_partial_tp_epoch", 0) or 0)
                                qty_avail = float(trade_state.get("recycle_qty", 0.0) or 0.0)
                                reentry_roi = float(cfg.get("swing_recycle_reentry_roi", 0.8))

                                if rc < rc_max and qty_avail > 0 and last_tp_epoch > 0:
                                    if (time.time() - last_tp_epoch) >= cooldown and roi <= reentry_roi:
                                        # 추세가 계속 같은 방향이면 재진입(리밸런싱)
                                        short_tf = str(cfg.get("timeframe", "5m"))
                                        long_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                                        fast = int(cfg.get("ma_fast", 7))
                                        slow = int(cfg.get("ma_slow", 99))
                                        short_tr = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=30)
                                        long_tr = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))

                                        if _trend_align(short_tr, side) and _trend_align(long_tr, side):
                                            lev = int(float(tgt.get("lev", 1)) or 1)
                                            free, _ = safe_fetch_balance(ex)
                                            margin_need = (qty_avail * cur_px) / max(lev, 1)
                                            if margin_need <= free * 0.9:
                                                set_leverage_safe(ex, sym, lev)
                                                qty_re = to_precision_qty(ex, sym, qty_avail)
                                                if qty_re > 0:
                                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty_re)
                                                    if ok:
                                                        trade_state["recycle_count"] = rc + 1
                                                        trade_state["recycle_qty"] = max(0.0, qty_avail - float(qty_re))
                                                        save_runtime(rt)
                                                        mon_add_event(mon, "RECYCLE_REENTRY", sym, f"재진입 {qty_re}", {"roi": roi, "trend": f"{short_tr}/{long_tr}"})
                                                        try:
                                                            gsheet_log_trade(
                                                                stage="RECYCLE_REENTRY",
                                                                symbol=sym,
                                                                trade_id=trade_id,
                                                                message=f"qty={qty_re}",
                                                                payload={"roi": roi, "qty": qty_re, "trend": f"{short_tr}/{long_tr}", "recycle_count": rc + 1},
                                                            )
                                                        except Exception:
                                                            pass
                                                        tg_send(
                                                            f"♻️ 순환매도 재진입\n- 코인: {sym}\n- 스타일: 스윙\n- 재진입수량: {qty_re}\n- 조건: ROI {roi:.2f}% <= {reentry_roi}%\n- 단기({short_tf}): {short_tr}\n- 장기({long_tf}): {long_tr}\n- 일지ID: {trade_id or '-'}",
                                                            target=cfg.get("tg_route_events_to", "channel"),
                                                            cfg=cfg,
                                                        )
                                                        if trade_id:
                                                            d = load_trade_detail(trade_id) or {}
                                                            evs = d.get("events", []) or []
                                                            evs.append({"time": now_kst_str(), "type": "RECYCLE_REENTRY", "roi": roi, "qty": qty_re})
                                                            d["events"] = evs
                                                            save_trade_detail(trade_id, d)
                            except Exception:
                                pass

                        # ✅ 스캘핑 -> 스윙 전환 조건(보유시간/정렬) + 필요시 추매
                        try:
                            short_tf = str(cfg.get("timeframe", "5m"))
                            long_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                            fast = int(cfg.get("ma_fast", 7))
                            slow = int(cfg.get("ma_slow", 99))
                            short_tr = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=25)
                            long_tr = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))
                            long_align = _trend_align(long_tr, side)
                            if _should_convert_scalp_to_swing(tgt, roi, cfg, long_align=long_align):
                                # 전환 + (선택) 1회 추매
                                did_dca = _try_scalp_to_swing_dca(ex, sym, side, cur_px, tgt, rt, cfg, mon)
                                tgt["style"] = "스윙"
                                tgt["style_reason"] = f"스캘핑 장기화({cfg.get('scalp_max_hold_minutes',25)}m+) → 스윙 전환"
                                tgt["style_last_switch_epoch"] = time.time()
                                # 스윙 목표로 확장
                                tgt["tp"] = float(clamp(max(tp, float(cfg.get("swing_tp_roi_min", 3.0))), float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))
                                tgt["sl"] = float(clamp(max(sl, float(cfg.get("swing_sl_roi_min", 1.5))), float(cfg.get("swing_sl_roi_min", 1.5)), float(cfg.get("swing_sl_roi_max", 30.0))))
                                active_targets[sym] = tgt
                                rt.setdefault("open_targets", {})[sym] = tgt
                                save_runtime(rt)
                                mon_add_event(mon, "SCALP_TO_SWING", sym, f"전환 완료(추매:{'Y' if did_dca else 'N'})", {"roi": roi})
                                try:
                                    gsheet_log_trade(
                                        stage="SCALP_TO_SWING",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message=f"did_dca={'Y' if did_dca else 'N'}",
                                        payload={"roi": roi, "did_dca": bool(did_dca)},
                                    )
                                except Exception:
                                    pass
                                tg_send(
                                    f"🔄 스타일 전환\n- 코인: {sym}\n- 스캘핑 → 스윙\n- 이유: {tgt.get('style_reason','')}\n- ROI: {roi:.2f}%\n- (전환추매): {'있음' if did_dca else '없음'}\n- 일지ID: {trade_id or '-'}",
                                    target=cfg.get("tg_route_events_to", "channel"),
                                    cfg=cfg,
                                )
                        except Exception:
                            pass

                        # ✅ DCA: 스캘핑은 기본 금지(요구사항), 스윙에서만 허용
                        if cfg.get("use_dca", True) and not (style_now == "스캘핑" and cfg.get("scalp_disable_dca", True)):
                            dca_trig = float(cfg.get("dca_trigger", -20.0))
                            dca_max = int(cfg.get("dca_max_count", 1))
                            dca_add_pct = float(cfg.get("dca_add_pct", 50.0))

                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                            dca_count = int(trade_state.get("dca_count", 0))

                            if roi <= dca_trig and dca_count < dca_max:
                                free, _ = safe_fetch_balance(ex)
                                base_entry = float(tgt.get("entry_usdt", 0.0))
                                add_usdt = base_entry * (dca_add_pct / 100.0)
                                if add_usdt > free:
                                    add_usdt = free * 0.5

                                if cur_px and add_usdt > 5:
                                    lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                    set_leverage_safe(ex, sym, lev)
                                    qty = to_precision_qty(ex, sym, (add_usdt * lev) / cur_px)
                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                    if ok:
                                        trade_state["dca_count"] = dca_count + 1
                                        save_runtime(rt)
                                        tg_send(
                                            f"💧 물타기(DCA)\n- 코인: {sym}\n- 스타일: {style_now}\n- 추가금: {add_usdt:.2f} USDT\n- 이유: 손실 {roi:.2f}% (기준 {dca_trig}%)\n- 일지ID: {trade_id or '-'}",
                                            target=cfg.get("tg_route_events_to", "channel"),
                                            cfg=cfg,
                                        )
                                        mon_add_event(mon, "DCA", sym, f"DCA {add_usdt:.2f} USDT", {"roi": roi})
                                        try:
                                            gsheet_log_trade(
                                                stage="DCA",
                                                symbol=sym,
                                                trade_id=trade_id,
                                                message=f"add_usdt={add_usdt:.2f}",
                                                payload={"roi": roi, "add_usdt": add_usdt, "qty": qty, "lev": lev, "dca_count": dca_count + 1},
                                            )
                                        except Exception:
                                            pass

                        # 스캘핑 전환 청산 모드: 목표를 더 보수적으로(빨리 끝내기)
                        scalp_exit_mode = bool(tgt.get("scalp_exit_mode", False))
                        if scalp_exit_mode:
                            tp = min(tp, float(cfg.get("scalp_tp_roi_max", 6.0)))
                            sl = min(sl, float(cfg.get("scalp_sl_roi_max", 5.0)))

                        do_stop = hit_sl_by_price or (roi <= -abs(sl))
                        do_take = hit_tp_by_price or (roi >= tp)

                        # 손절
                        if do_stop:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                one, review = ai_write_review(sym, side, roi, "자동 손절", cfg)
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 손절", one_line=one, review=review, trade_id=trade_id)
                                try:
                                    gsheet_log_trade(
                                        stage="EXIT_SL",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message="auto_sl",
                                        payload={"roi": roi, "pnl_usdt": pnl_usdt_snapshot, "entry": entry, "exit": exit_px, "hit_sr": bool(hit_sl_by_price), "style": style_now},
                                    )
                                except Exception:
                                    pass

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update(
                                        {
                                            "exit_time": now_kst_str(),
                                            "exit_price": exit_px,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "pnl_pct": roi,
                                            "result": "SL",
                                            "review": review,
                                        }
                                    )
                                    save_trade_detail(trade_id, d)

                                rt["consec_losses"] = int(rt.get("consec_losses", 0)) + 1
                                if cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(
                                        f"🛑 연속손실 보호\n- 연속손실: {rt['consec_losses']}회\n- {int(cfg.get('loss_pause_minutes',30))}분 자동 정지",
                                        target=cfg.get("tg_route_events_to", "channel"),
                                        cfg=cfg,
                                    )
                                    mon_add_event(mon, "PAUSE", "", "연속손실 자동정지", {"consec": rt["consec_losses"]})
                                    try:
                                        gsheet_log_event("PAUSE", message="loss_pause", payload={"consec_losses": rt["consec_losses"], "minutes": int(cfg.get("loss_pause_minutes", 30))})
                                    except Exception:
                                        pass
                                save_runtime(rt)

                                emo = "🟢" if roi >= 0 else "🔴"
                                tg_send(
                                    f"{emo} 손절\n- 코인: {sym}\n- 스타일: {style_now}\n- 수익률: {roi:.2f}% (손익 {pnl_usdt_snapshot:.2f} USDT)\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 현재잔고: {total_after:.2f} USDT (가용 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 이탈' if hit_sl_by_price else '목표 손절 도달'}\n"
                                    f"- 한줄평: {one}\n- 일지ID: {trade_id or '없음'}",
                                    target=cfg.get("tg_route_events_to", "channel"),
                                    cfg=cfg,
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                rt.setdefault("open_targets", {}).pop(sym, None)
                                save_runtime(rt)

                                mon_add_event(mon, "STOP", sym, f"ROI {roi:.2f}%", {"trade_id": trade_id})
                                monitor_write_throttled(mon, 0.2)

                        # 익절
                        elif do_take:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                one, review = ai_write_review(sym, side, roi, "자동 익절", cfg)
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 익절", one_line=one, review=review, trade_id=trade_id)
                                try:
                                    gsheet_log_trade(
                                        stage="EXIT_TP",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message="auto_tp",
                                        payload={"roi": roi, "pnl_usdt": pnl_usdt_snapshot, "entry": entry, "exit": exit_px, "hit_sr": bool(hit_tp_by_price), "style": style_now},
                                    )
                                except Exception:
                                    pass

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update(
                                        {
                                            "exit_time": now_kst_str(),
                                            "exit_price": exit_px,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "pnl_pct": roi,
                                            "result": "TP",
                                            "review": review,
                                        }
                                    )
                                    save_trade_detail(trade_id, d)

                                rt["consec_losses"] = 0
                                save_runtime(rt)

                                tg_send(
                                    f"🎉 익절\n- 코인: {sym}\n- 스타일: {style_now}\n- 수익률: +{roi:.2f}% (손익 {pnl_usdt_snapshot:.2f} USDT)\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 현재잔고: {total_after:.2f} USDT (가용 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 목표 도달' if hit_tp_by_price else '목표 익절 도달'}\n"
                                    f"- 한줄평: {one}\n- 일지ID: {trade_id or '없음'}",
                                    target=cfg.get("tg_route_events_to", "channel"),
                                    cfg=cfg,
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                rt.setdefault("open_targets", {}).pop(sym, None)
                                save_runtime(rt)

                                mon_add_event(mon, "TAKE", sym, f"ROI +{roi:.2f}%", {"trade_id": trade_id})
                                monitor_write_throttled(mon, 0.2)

                        open_pos_snapshot.append(
                            {
                                "symbol": sym,
                                "side": side,
                                "roi": roi,
                                "upnl": upnl,
                                "lev": lev_live,
                                "style": style_now,
                                "tp": tp,
                                "sl": sl,
                                "trade_id": trade_id,
                            }
                        )

                    # ✅ 자동매매가 OFF/정지/주말이어도 포지션 스냅샷은 UI에 표시
                    if (not open_pos_snapshot) and (not entry_allowed_global) and pos_by_sym:
                        try:
                            for sym in TARGET_COINS:
                                p = pos_by_sym.get(sym)
                                if not p:
                                    continue
                                side = position_side_normalize(p)
                                roi = float(position_roi_percent(p))
                                upnl = float(p.get("unrealizedPnl") or 0.0)
                                lev_live = _pos_leverage(p)
                                tgt0 = (active_targets.get(sym, {}) or {})
                                style_now = str(tgt0.get("style", "") or "")
                                tp = float(tgt0.get("tp", 0.0) or 0.0)
                                sl = float(tgt0.get("sl", 0.0) or 0.0)
                                trade_id = str(tgt0.get("trade_id") or "")
                                open_pos_snapshot.append(
                                    {
                                        "symbol": sym,
                                        "side": side,
                                        "roi": roi,
                                        "upnl": upnl,
                                        "lev": lev_live,
                                        "style": style_now,
                                        "tp": tp,
                                        "sl": sl,
                                        "trade_id": trade_id,
                                    }
                                )
                        except Exception:
                            pass

                    mon["open_positions"] = open_pos_snapshot

                    # 2) 신규 진입 스캔
                    risk_mul = external_risk_multiplier(ext, cfg)
                    mon["entry_risk_multiplier"] = risk_mul
                    free_usdt = 0.0
                    if entry_allowed_global:
                        free_usdt, _ = safe_fetch_balance(ex)
                    active_syms = set(pos_by_sym.keys())

                    scan_cycle_start = time.time()
                    for sym in TARGET_COINS:
                        # 포지션 있으면 스킵
                        if sym in active_syms:
                            mon_add_scan(mon, stage="in_position", symbol=sym, tf=str(cfg.get("timeframe", "")), message="이미 포지션 보유")
                            continue

                        # 쿨다운
                        cd = float(rt.get("cooldowns", {}).get(sym, 0))
                        if time.time() < cd:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "쿨다운(잠깐 쉬는중)"
                            mon_add_scan(mon, stage="trade_skipped", symbol=sym, tf=str(cfg.get("timeframe", "")), message="쿨다운")
                            continue

                        # 데이터 로드(단기: cfg timeframe)
                        try:
                            mon_add_scan(mon, stage="fetch_short", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="OHLCV 로드")
                            ohlcv = ex.fetch_ohlcv(sym, cfg.get("timeframe", "5m"), limit=220)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except Exception as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = f"데이터 실패: {e}"
                            mon_add_scan(mon, stage="fetch_short_fail", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message=str(e)[:140])
                            # 강제스캔 요약에도 반영
                            try:
                                if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                    force_scan_summary_lines.append(f"- {sym}: fetch_short_fail | {str(e)[:80]}")
                            except Exception:
                                pass
                            continue

                        df, stt, last = calc_indicators(df, cfg)
                        mon.setdefault("coins", {}).setdefault(sym, {})
                        cs = mon["coins"][sym]

                        if last is None:
                            cs.update({"last_scan_kst": now_kst_str(), "ai_called": False, "skip_reason": "지표 계산 실패(ta/데이터 부족)"})
                            mon_add_scan(mon, stage="rule_signal", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="지표 계산 실패")
                            try:
                                if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                    force_scan_summary_lines.append(f"- {sym}: indicator_fail(ta/데이터 부족)")
                            except Exception:
                                pass
                            continue

                        # 장기추세(1h) 계산 + 캐시
                        htf_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                        htf_trend = get_htf_trend_cached(
                            ex,
                            sym,
                            htf_tf,
                            fast=int(cfg.get("ma_fast", 7)),
                            slow=int(cfg.get("ma_slow", 99)),
                            cache_sec=int(cfg.get("trend_filter_cache_sec", 60)),
                        )
                        cs["trend_htf"] = f"🧭 {htf_tf} {htf_trend}"
                        mon_add_scan(mon, stage="fetch_long", symbol=sym, tf=htf_tf, signal=htf_trend, message="장기추세 계산")

                        # 모니터 기록(단기/장기 같이)
                        cs.update(
                            {
                                "last_scan_epoch": time.time(),
                                "last_scan_kst": now_kst_str(),
                                "price": float(last["close"]),
                                "trend_short": stt.get("추세", ""),  # 단기추세(timeframe)
                                "trend_long": cs.get("trend_htf", ""),  # 장기추세(1h)
                                "rsi": float(last.get("RSI", 0)) if "RSI" in df.columns else None,
                                "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
                                "bb": stt.get("BB", ""),
                                "macd": stt.get("MACD", ""),
                                "vol": stt.get("거래량", ""),
                                "pullback_candidate": bool(stt.get("_pullback_candidate", False)),
                            }
                        )

                        # ✅ S/R 계산(스캔 과정 표시용) - 캐시 사용
                        try:
                            sr_tf0 = str(cfg.get("sr_timeframe", "15m"))
                            sr_lb0 = int(cfg.get("sr_lookback", 220))
                            sr_cache0 = int(cfg.get("sr_levels_cache_sec", 60))
                            sr_levels = get_sr_levels_cached(
                                ex,
                                sym,
                                sr_tf0,
                                pivot_order=int(cfg.get("sr_pivot_order", 6)),
                                cache_sec=sr_cache0,
                                limit=sr_lb0,
                            )
                            supports = list(sr_levels.get("supports") or [])
                            resistances = list(sr_levels.get("resistances") or [])
                            px0 = float(last["close"])
                            near_sup = max([s for s in supports if s < px0], default=None) if supports else None
                            near_res = min([r for r in resistances if r > px0], default=None) if resistances else None
                            cs["sr_tf"] = sr_tf0
                            cs["sr_support_near"] = near_sup
                            cs["sr_resistance_near"] = near_res
                            mon_add_scan(
                                mon,
                                stage="support_resistance",
                                symbol=sym,
                                tf=sr_tf0,
                                signal="S/R",
                                score="",
                                message=f"sup={near_sup} res={near_res}",
                                extra={"support": near_sup, "resistance": near_res},
                            )
                        except Exception as e:
                            mon_add_scan(mon, stage="support_resistance", symbol=sym, tf=str(cfg.get("sr_timeframe", "")), message=f"SR 실패: {e}"[:140])

                        # AI 호출 필터(기존 유지)
                        call_ai = False
                        if bool(stt.get("_pullback_candidate", False)):
                            call_ai = True
                        elif bool(stt.get("_rsi_resolve_long", False)) or bool(stt.get("_rsi_resolve_short", False)):
                            call_ai = True
                        else:
                            adxv = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            if adxv >= 25:
                                call_ai = True

                        # ✅ /scan 강제스캔: 원래 call_ai=False인 경우에만 AI를 "추가로" 호출(주문은 막기 위해 플래그 보관)
                        forced_ai = False
                        try:
                            if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)) and (not call_ai):
                                call_ai = True
                                forced_ai = True
                        except Exception:
                            forced_ai = False

                        # ✅ rule_signal 단계 기록
                        try:
                            sigs = []
                            if bool(stt.get("_pullback_candidate", False)):
                                sigs.append("pullback")
                            if bool(stt.get("_rsi_resolve_long", False)):
                                sigs.append("rsi_resolve_long")
                            if bool(stt.get("_rsi_resolve_short", False)):
                                sigs.append("rsi_resolve_short")
                            adxv2 = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            mon_add_scan(
                                mon,
                                stage="rule_signal",
                                symbol=sym,
                                tf=str(cfg.get("timeframe", "5m")),
                                signal=",".join(sigs) if sigs else "none",
                                score=adxv2,
                                message=("AI 호출(강제스캔)" if forced_ai else ("AI 호출" if call_ai else "AI 스킵(휩쏘 위험)")),
                                extra={"pullback": bool(stt.get("_pullback_candidate", False)), "adx": adxv2},
                            )
                        except Exception:
                            pass

                        if not call_ai:
                            cs["ai_called"] = False
                            cs["skip_reason"] = "횡보/해소 신호 없음(휩쏘 위험)"
                            monitor_write_throttled(mon, 1.0)
                            mon_add_scan(mon, stage="trade_skipped", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="call_ai=False")
                            continue

                        # AI 판단
                        mon_add_scan(mon, stage="ai_call", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="AI 판단 요청")
                        ai = ai_decide_trade(df, stt, sym, mode, cfg, external=ext)
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))
                        mon_add_scan(mon, stage="ai_result", symbol=sym, tf=str(cfg.get("timeframe", "5m")), signal=str(decision), score=conf, message=str(ai.get("reason_easy", ""))[:80])
                        # 강제스캔 요약 라인(요구사항: /scan 결과는 짧게)
                        try:
                            if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                force_scan_summary_lines.append(f"- {sym}: {str(decision).upper()}({conf}%) | {str(ai.get('reason_easy',''))[:60]}")
                        except Exception:
                            pass

                        cs.update(
                            {
                                "ai_called": True,
                                "ai_decision": decision,
                                "ai_confidence": conf,
                                "ai_entry_pct": float(ai.get("entry_pct", rule["entry_pct_min"])),
                                "ai_leverage": int(ai.get("leverage", rule["lev_min"])),
                                "ai_sl_pct": float(ai.get("sl_pct", 1.2)),
                                "ai_tp_pct": float(ai.get("tp_pct", 3.0)),
                                "ai_rr": float(ai.get("rr", 1.5)),
                                "ai_used": ", ".join(ai.get("used_indicators", [])),
                                "ai_reason_easy": ai.get("reason_easy", ""),
                                "min_conf_required": int(rule["min_conf"]),
                                "skip_reason": "",
                            }
                        )
                        monitor_write_throttled(mon, 1.0)

                        # 진입
                        if decision in ["buy", "sell"] and conf >= int(rule["min_conf"]):
                            # ✅ 강제스캔(scan_only) 또는 auto_trade OFF/정지/주말이면 신규진입 금지
                            if (not entry_allowed_global) or (forced_ai and force_scan_only):
                                try:
                                    why = "entry_disabled"
                                    if forced_ai and force_scan_only:
                                        why = "force_scan(scan_only)"
                                    elif not trade_enabled:
                                        why = "auto_trade=OFF"
                                    elif paused_now:
                                        why = "paused(loss_protect)"
                                    elif cfg.get("no_trade_weekend", False) and (now_kst().weekday() in [5, 6]):
                                        why = "weekend_block"
                                    cs["skip_reason"] = f"신규진입 금지({why})"
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message=f"신규진입 금지({why})",
                                        extra={"forced_ai": forced_ai, "force_scan_only": force_scan_only, "trade_enabled": trade_enabled},
                                    )
                                except Exception:
                                    pass
                                continue
                            px = float(last["close"])

                            # ✅ 스타일 결정 (단기/장기 추세로 스캘핑/스윙)
                            style_info = _style_for_entry(sym, decision, stt.get("추세", ""), htf_trend, cfg)
                            style = style_info.get("style", "스캘핑")
                            cs["style_reco"] = style
                            cs["style_confidence"] = int(style_info.get("confidence", 0))
                            cs["style_reason"] = str(style_info.get("reason", ""))[:240]
                            # ✅ /mode 레짐 강제(auto|scalping|swing)
                            regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                            if regime_mode in ["scalping", "scalp", "short"]:
                                style = "스캘핑"
                                cs["style_reco"] = "스캘핑"
                                cs["style_confidence"] = 100
                                cs["style_reason"] = "레짐 강제: scalping"
                            elif regime_mode in ["swing", "long"]:
                                style = "스윙"
                                cs["style_reco"] = "스윙"
                                cs["style_confidence"] = 100
                                cs["style_reason"] = "레짐 강제: swing"

                            # ✅ 추세 필터 정책(기존 "금지" 기능 유지 + 새로운 "허용-스캘핑" 추가)
                            if cfg.get("trend_filter_enabled", True) and cfg.get("trend_filter_policy", "ALLOW_SCALP") == "STRICT":
                                is_down = ("하락" in str(htf_trend))
                                is_up = ("상승" in str(htf_trend))
                                if is_down and decision == "buy":
                                    cs["skip_reason"] = f"장기추세({htf_tf}) 하락이라 롱 금지(STRICT)"
                                    continue
                                if is_up and decision == "sell":
                                    cs["skip_reason"] = f"장기추세({htf_tf}) 상승이라 숏 금지(STRICT)"
                                    continue
                            elif cfg.get("trend_filter_enabled", True) and cfg.get("trend_filter_policy", "ALLOW_SCALP") == "ALLOW_SCALP" and regime_mode == "auto":
                                # 역추세면 스캘핑 강제
                                is_down = ("하락" in str(htf_trend))
                                is_up = ("상승" in str(htf_trend))
                                if is_down and decision == "buy":
                                    style = "스캘핑"
                                    cs["style_reco"] = "스캘핑"
                                    cs["style_reason"] = f"장기추세({htf_tf}) 하락 → 역추세는 스캘핑만"
                                if is_up and decision == "sell":
                                    style = "스캘핑"
                                    cs["style_reco"] = "스캘핑"
                                    cs["style_reason"] = f"장기추세({htf_tf}) 상승 → 역추세는 스캘핑만"

                            # 스타일별 envelope + 리스크가드레일
                            ai2 = apply_style_envelope(ai, style, cfg, rule)
                            ai2 = _risk_guardrail(ai2, df, decision, mode, style, ext)

                            entry_pct = float(ai2.get("entry_pct", rule["entry_pct_min"]))
                            lev = int(ai2.get("leverage", rule["lev_min"]))
                            slp = float(ai2.get("sl_pct", 1.2))
                            tpp = float(ai2.get("tp_pct", 3.0))

                            # 외부시황 위험 감산
                            entry_usdt = free_usdt * (entry_pct / 100.0) * risk_mul
                            if entry_usdt < 5:
                                cs["skip_reason"] = "잔고 부족(진입금 너무 작음)"
                                continue

                            set_leverage_safe(ex, sym, lev)
                            qty = to_precision_qty(ex, sym, (entry_usdt * lev) / max(px, 1e-9))
                            if qty <= 0:
                                cs["skip_reason"] = "수량 계산 실패"
                                continue

                            ok = market_order_safe(ex, sym, decision, qty)
                            if ok:
                                trade_id = uuid.uuid4().hex[:10]
                                mon_add_scan(
                                    mon,
                                    stage="trade_opened",
                                    symbol=sym,
                                    tf=str(cfg.get("timeframe", "5m")),
                                    signal=str(decision),
                                    score=conf,
                                    message=f"주문 체결, trade_id={trade_id}",
                                    extra={"qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style},
                                )
                                try:
                                    gsheet_log_trade(
                                        stage="ENTRY",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message=f"{decision} style={style} conf={conf}",
                                        payload={"qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style, "tp": tpp, "sl": slp},
                                    )
                                except Exception:
                                    pass

                                # SR 기반 SL/TP 가격도 계산
                                sl_price = None
                                tp_price = None
                                if cfg.get("use_sr_stop", True):
                                    try:
                                        sr_tf = cfg.get("sr_timeframe", "15m")
                                        sr_lb = int(cfg.get("sr_lookback", 220))
                                        htf = ex.fetch_ohlcv(sym, sr_tf, limit=sr_lb)
                                        hdf = pd.DataFrame(htf, columns=["time", "open", "high", "low", "close", "vol"])
                                        hdf["time"] = pd.to_datetime(hdf["time"], unit="ms")
                                        sr = sr_stop_take(
                                            entry_price=px,
                                            side=decision,
                                            htf_df=hdf,
                                            atr_period=int(cfg.get("sr_atr_period", 14)),
                                            pivot_order=int(cfg.get("sr_pivot_order", 6)),
                                            buffer_atr_mult=float(cfg.get("sr_buffer_atr_mult", 0.25)),
                                            rr_min=float(cfg.get("sr_rr_min", 1.5)),
                                        )
                                        if sr:
                                            sl_price = sr["sl_price"]
                                            tp_price = sr["tp_price"]
                                    except Exception:
                                        pass

                                # 목표 저장
                                active_targets[sym] = {
                                    "sl": slp,
                                    "tp": tpp,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "reason": ai2.get("reason_easy", ""),
                                    "trade_id": trade_id,
                                    "sl_price": sl_price,
                                    "tp_price": tp_price,
                                    "sl_price_pct": float(ai2.get("sl_price_pct", slp / max(lev, 1))),
                                    "style": style,
                                    "style_confidence": int(cs.get("style_confidence", 0)),
                                    "style_reason": str(cs.get("style_reason", ""))[:240],
                                    "entry_epoch": time.time(),
                                    "style_last_switch_epoch": time.time(),
                                }

                                rt.setdefault("open_targets", {})[sym] = active_targets[sym]
                                save_runtime(rt)

                                # 상세일지
                                save_trade_detail(
                                    trade_id,
                                    {
                                        "trade_id": trade_id,
                                        "time": now_kst_str(),
                                        "coin": sym,
                                        "decision": decision,
                                        "confidence": conf,
                                        "entry_price": px,
                                        "entry_usdt": entry_usdt,
                                        "entry_pct": entry_pct,
                                        "lev": lev,
                                        "sl_pct_roi": slp,
                                        "tp_pct_roi": tpp,
                                        "sl_price_sr": sl_price,
                                        "tp_price_sr": tp_price,
                                        "used_indicators": ai2.get("used_indicators", []),
                                        "reason_easy": ai2.get("reason_easy", ""),
                                        "raw_status": stt,
                                        "trend_short": stt.get("추세", ""),
                                        "trend_long": f"🧭 {htf_tf} {htf_trend}",
                                        "style": style,
                                        "style_confidence": int(cs.get("style_confidence", 0)),
                                        "style_reason": str(cs.get("style_reason", ""))[:240],
                                        "events": [],
                                        "external_used": {
                                            "fear_greed": (ext or {}).get("fear_greed"),
                                            "high_impact_events_soon": ((ext or {}).get("high_impact_events_soon") or [])[:3],
                                            "asof_kst": (ext or {}).get("asof_kst", ""),
                                            "daily_btc_brief": (ext or {}).get("daily_btc_brief", {}),
                                        },
                                    },
                                )

                                # 쿨다운
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)

                                # 텔레그램 보고
                                if cfg.get("tg_enable_reports", True):
                                    direction = "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"
                                    msg = (
                                        f"🎯 진입\n- 코인: {sym}\n- 스타일: {style}\n- 방향: {direction}\n"
                                        f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%)\n"
                                        f"- 레버리지: x{lev}\n"
                                        f"- 목표익절: +{tpp:.2f}% / 목표손절: -{slp:.2f}%\n"
                                        f"- 단기추세({cfg.get('timeframe','5m')}): {stt.get('추세','-')}\n"
                                        f"- 장기추세({htf_tf}): 🧭 {htf_trend}\n"
                                        f"- 외부리스크 감산: x{risk_mul:.2f}\n"
                                    )
                                    if sl_price is not None and tp_price is not None:
                                        msg += f"- SR기준가: TP {tp_price:.6g} / SL {sl_price:.6g}\n"
                                    msg += f"- 확신도: {conf}% (기준 {rule['min_conf']}%)\n- 일지ID: {trade_id}\n"
                                    if cfg.get("tg_send_entry_reason", False):
                                        # 요구사항: 텔레그램에는 '긴 근거'를 보내지 않고, /log <id>로 조회
                                        msg += (
                                            f"- 근거(짧게): {str(ai2.get('reason_easy',''))[:120]}\n"
                                            f"- 자세한 근거: /log {trade_id}\n"
                                            f"- AI지표: {', '.join(ai2.get('used_indicators', []))}\n"
                                        )
                                    tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg)

                                mon_add_event(mon, "ENTRY", sym, f"{decision} {style} conf{conf}", {"trade_id": trade_id})
                                monitor_write_throttled(mon, 0.2)
                                time.sleep(1.0)

                        else:
                            # AI 결과가 HOLD이거나, 확신도/조건 미달로 진입하지 않음
                            mon_add_scan(
                                mon,
                                stage="trade_skipped",
                                symbol=sym,
                                tf=str(cfg.get("timeframe", "5m")),
                                signal=str(decision),
                                score=conf,
                                message="진입 조건 미달/보류",
                                extra={"decision": decision, "confidence": conf, "min_conf": int(rule.get("min_conf", 0))},
                            )

                        time.sleep(0.4)

                # 스캔 사이클 시간(멈춤 감지/표시용)
                try:
                    if "scan_cycle_start" in locals():
                        mon["scan_cycle_sec"] = float(time.time() - float(scan_cycle_start))
                        mon["last_scan_cycle_kst"] = now_kst_str()
                except Exception:
                    pass

                # ✅ 강제스캔 결과 전송 및 요청 해제(1회)
                if force_scan_pending and force_scan_id:
                    try:
                        lines = [f"🔎 강제스캔 결과: {force_scan_id}", f"- 시각(KST): {now_kst_str()}"]
                        if force_scan_summary_lines:
                            lines += force_scan_summary_lines[:12]
                        else:
                            lines.append("- (수집된 결과 없음)")
                        # ✅ 요구: TG_TARGET_CHAT_ID는 채널(브로드캐스트), 관리/버튼/강제스캔 결과는 관리자 DM으로
                        try:
                            force_by = int(force_scan_req.get("requested_by", 0) or 0)
                        except Exception:
                            force_by = 0
                        if TG_ADMIN_IDS and force_by:
                            tg_send_chat(force_by, "\n".join(lines))
                        elif TG_ADMIN_IDS:
                            tg_send("\n".join(lines), target="admin", cfg=cfg)
                        else:
                            tg_send("\n".join(lines), target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)
                        mon_add_event(mon, "SCAN_DONE", "", f"id={force_scan_id}", {"symbols": list(force_scan_syms_set)[:50], "scan_only": force_scan_only})
                        gsheet_log_event("SCAN_DONE", message=f"id={force_scan_id}", payload={"symbols": list(force_scan_syms_set)[:50], "scan_only": force_scan_only})
                    except Exception:
                        pass
                    try:
                        rt["force_scan"] = {}
                        save_runtime(rt)
                    except Exception:
                        pass

            # 텔레그램 수신 처리(요구사항: long polling 스레드(getUpdates) -> 큐 처리)
            updates = tg_updates_pop_all(max_items=80)
            for up in updates:
                try:
                    # 텍스트 명령
                    if "message" in up and "text" in (up.get("message") or {}):
                        msg0 = up.get("message") or {}
                        txt = str(msg0.get("text") or "").strip()
                        chat_id = ((msg0.get("chat") or {}) if isinstance(msg0.get("chat"), dict) else {}).get("id", None)
                        from0 = msg0.get("from") or {}
                        uid = from0.get("id", None)
                        is_admin = tg_is_admin(uid)

                        def _reply_to_chat(m: str):
                            # /status처럼 누구나 허용되는 응답은 "요청이 온 채팅"으로 답장
                            if chat_id is not None:
                                tg_send_chat(chat_id, m)
                            else:
                                tg_send(m, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)

                        def _reply_admin_dm(m: str):
                            # ✅ 요구: 관리/버튼 결과는 TG_ADMIN_USER_IDS(관리자 DM)로
                            if TG_ADMIN_IDS:
                                if uid is not None:
                                    tg_send_chat(uid, m)
                                else:
                                    tg_send(m, target="admin", cfg=cfg)
                            else:
                                _reply_to_chat(m)

                        def _deny():
                            _reply_to_chat("⛔️ 관리자만 사용할 수 있는 명령입니다.")

                        low = txt.lower().strip()

                        # /menu (관리자) - TG_ADMIN_USER_IDS 설정 시, /status 외에는 관리자만 허용
                        if low.startswith("/menu") or low in ["menu", "메뉴"]:
                            if TG_ADMIN_IDS and not is_admin:
                                _deny()
                            else:
                                tg_send_menu(cfg=cfg)

                        # /status (누구나 허용)
                        elif low.startswith("/status") or txt == "상태":
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt2 = load_runtime()
                            mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                            regime_mode = str(cfg_live.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            h = openai_health_info(cfg_live)
                            ai_txt = "OK" if bool(h.get("available", False)) else str(h.get("message", "OFF"))
                            until = str(h.get("until_kst", "")).strip()
                            if until and (not bool(h.get("available", False))):
                                ai_txt = f"{ai_txt} (~{until} KST)"
                            msg = (
                                f"📡 상태\n- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 레짐: {regime_txt}\n"
                                f"- OpenAI: {ai_txt}\n"
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})\n"
                                f"- 연속손실: {rt2.get('consec_losses',0)}\n"
                                f"- 정지해제: {('정지중' if time.time() < float(rt2.get('pause_until',0)) else '정상')}\n"
                                f"- 마지막 스캔: {mon_now.get('last_scan_kst','-')}\n"
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}\n"
                            )
                            _reply_to_chat(msg)

                        # /gsheet (관리자) - 상태/연결 테스트
                        elif low.startswith("/gsheet") or txt in ["시트", "구글시트", "gsheet"]:
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                do_test = False
                                if len(parts) >= 2:
                                    arg = str(parts[1]).lower().strip()
                                    do_test = arg in ["test", "t", "ping", "check"]
                                if do_test:
                                    res = gsheet_test_append_row(timeout_sec=25)
                                    if res.get("ok"):
                                        _reply_admin_dm("✅ Google Sheets TEST 성공(GSHEET_TEST)")
                                    else:
                                        _reply_admin_dm(f"❌ Google Sheets TEST 실패: {res.get('error','')}")
                                stg = gsheet_status_snapshot()
                                msg = (
                                    "📎 Google Sheets 상태\n"
                                    f"- enabled: {stg.get('enabled')}\n"
                                    f"- connected: {stg.get('connected')}\n"
                                    f"- spreadsheet_id: {stg.get('spreadsheet_id')}\n"
                                    f"- worksheet: {stg.get('worksheet')}\n"
                                    f"- service_account_email: {stg.get('service_account_email')}\n"
                                    f"- queue_high/scan: {stg.get('queue_high')}/{stg.get('queue_scan')}\n"
                                    f"- last_append: {stg.get('last_append_kst')} ({stg.get('last_append_type')}:{stg.get('last_append_stage')})\n"
                                    f"- last_err: {stg.get('last_err')}\n"
                                    "사용법: /gsheet test"
                                )
                                _reply_admin_dm(msg[:3500])

                        # /positions (관리자)
                        elif low.startswith("/positions") or txt == "포지션":
                            if not is_admin:
                                _deny()
                            else:
                                msg = ["📊 포지션"]
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    msg.append("- ⚪ 없음(관망)")
                                else:
                                    for p in act:
                                        sym = p.get("symbol", "")
                                        side = position_side_normalize(p)
                                        roi = float(position_roi_percent(p))
                                        upnl = float(p.get("unrealizedPnl") or 0.0)
                                        lev = p.get("leverage", "?")
                                        style = str((active_targets.get(sym, {}) or {}).get("style", ""))
                                        msg.append(_fmt_pos_line(sym, side, lev, roi, upnl, style=style))
                                _reply_admin_dm("\n".join(msg))

                        # /scan (관리자) - 강제스캔(스캔만, 주문X)
                        elif low.startswith("/scan") or txt == "스캔":
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                sym_arg = parts[1].strip().upper() if len(parts) >= 2 else ""
                                # 심볼 필터(간단): "BTC" 또는 "BTC/USDT:USDT" 형태 지원
                                syms = list(TARGET_COINS)
                                if sym_arg:
                                    if "/" in sym_arg:
                                        syms = [s for s in TARGET_COINS if s.upper().startswith(sym_arg)]
                                    else:
                                        syms = [s for s in TARGET_COINS if s.upper().startswith(f"{sym_arg}/")]
                                if not syms:
                                    _reply_admin_dm("대상 심볼이 없습니다. 예) /scan BTC 또는 /scan BTC/USDT:USDT")
                                else:
                                    rid = uuid.uuid4().hex[:8]
                                    rt2 = load_runtime()
                                    rt2["force_scan"] = {
                                        "id": rid,
                                        "requested_at_epoch": time.time(),
                                        "requested_at_kst": now_kst_str(),
                                        "requested_by": int(uid or 0),
                                        "symbols": syms,
                                        "scan_only": True,  # 안전: 강제스캔은 기본 주문X
                                        "done": False,
                                    }
                                    save_runtime(rt2)
                                    try:
                                        mon_add_event(mon, "SCAN_REQUEST", "", f"force_scan id={rid}", {"symbols": syms, "by": uid})
                                        gsheet_log_event("SCAN_REQUEST", message=f"id={rid}", payload={"symbols": syms, "by": uid})
                                    except Exception:
                                        pass
                                    _reply_admin_dm(f"🔎 강제스캔 요청 완료: {rid}\n- 대상: {', '.join(syms)}\n- 주의: 강제스캔은 '스캔만' 수행(주문X)")

                        # /mode auto|scalping|swing (관리자)
                        elif low.startswith("/mode") or low.startswith("모드"):
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) < 2:
                                    _reply_admin_dm("사용법: /mode auto|scalping|swing")
                                else:
                                    arg = str(parts[1]).lower().strip()
                                    if arg in ["auto", "a"]:
                                        m = "auto"
                                    elif arg in ["scalping", "scalp", "short", "s"]:
                                        m = "scalping"
                                    elif arg in ["swing", "long", "l"]:
                                        m = "swing"
                                    else:
                                        m = ""
                                    if not m:
                                        _reply_admin_dm("사용법: /mode auto|scalping|swing")
                                    else:
                                        cfg2 = load_settings()
                                        cfg2["regime_mode"] = m
                                        save_settings(cfg2)
                                        try:
                                            mon_add_event(mon, "MODE_CHANGE", "", f"regime_mode={m}", {"by": uid})
                                            gsheet_log_event("MODE_CHANGE", message=f"regime_mode={m}", payload={"by": uid})
                                        except Exception:
                                            pass
                                        _reply_admin_dm(f"✅ 레짐 변경: {m}")

                        # /vision (관리자)
                        elif low.startswith("/vision") or txt == "시야":
                            if not is_admin:
                                _deny()
                            else:
                                mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                                coins = mon_now.get("coins", {}) or {}
                                lines = [
                                    "👁️ AI 시야(요약)",
                                    f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                                    f"- 모드: {mon_now.get('trade_mode','-')}",
                                    f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                                ]
                                for sym, cs in list(coins.items())[:10]:
                                    style = str(cs.get("style_reco", "")) or ""
                                    stxt = f"[{style}] " if style else ""
                                    lines.append(
                                        f"- {sym}: {stxt}{str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                        f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                        f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:30]}"
                                    )
                                _reply_admin_dm("\n".join(lines))

                        # /log 또는 /log <id> (관리자)
                        elif low.startswith("/log") or txt == "일지":
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) >= 2 and parts[1].strip():
                                    tid = parts[1].strip()
                                    d = load_trade_detail(tid)
                                    if not d:
                                        _reply_admin_dm("해당 ID를 찾지 못했어요.")
                                    else:
                                        evs = d.get("events", []) or []
                                        ev_short = []
                                        for e in evs[-6:]:
                                            try:
                                                ev_short.append(f"- {e.get('time','')} {e.get('type','')}: {str(e)[:60]}")
                                            except Exception:
                                                continue
                                        msg = (
                                            f"🧾 /log {tid}\n"
                                            f"- 코인: {d.get('coin')}\n"
                                            f"- 스타일: {d.get('style','-')} ({d.get('style_confidence','-')}%)\n"
                                            f"- 방향: {d.get('decision')}\n"
                                            f"- 확신도: {d.get('confidence')}\n"
                                            f"- 진입: {d.get('time','-')} @ {d.get('entry_price')}\n"
                                            f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                            f"- 레버: x{d.get('lev')}\n"
                                            f"- TP/SL(ROI): +{d.get('tp_pct_roi')}% / -{d.get('sl_pct_roi')}%\n"
                                            f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                            f"- 한줄근거: {str(d.get('reason_easy',''))[:800]}\n"
                                        )
                                        if d.get("exit_time"):
                                            msg += (
                                                f"- 청산: {d.get('exit_time')} @ {d.get('exit_price')}\n"
                                                f"- 결과: {d.get('result','-')} | PnL {float(d.get('pnl_usdt',0)):.2f} USDT | ROI {float(d.get('pnl_pct',0)):.2f}%\n"
                                            )
                                        if ev_short:
                                            msg += "최근 이벤트:\n" + "\n".join(ev_short)
                                        # 텔레그램 길이 제한 보호
                                        _reply_admin_dm(msg[:3500])
                                else:
                                    df_log = read_trade_log()
                                    if df_log.empty:
                                        _reply_admin_dm("📜 일지 없음(아직 기록된 매매가 없어요)")
                                    else:
                                        top = df_log.head(8)
                                        msg = ["📜 최근 매매일지(요약)"]
                                        for _, r in top.iterrows():
                                            tid = str(r.get("TradeID", "") or "")
                                            pnl = float(r.get("PnL_Percent", 0) or 0)
                                            emo = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                                            msg.append(
                                                f"- {emo} {r['Time']} {r['Coin']} {r['Side']} {pnl:.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}"
                                            )
                                        _reply_admin_dm("\n".join(msg))

                        # (호환) 일지상세 /detail (관리자)
                        elif txt.startswith("일지상세") or low.startswith("/detail"):
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) < 2:
                                    _reply_admin_dm("사용법: 일지상세 <ID>\n(예: 일지상세 a1b2c3d4e5)")
                                else:
                                    tid = parts[1].strip()
                                    d = load_trade_detail(tid)
                                    if not d:
                                        _reply_admin_dm("해당 ID를 찾지 못했어요.")
                                    else:
                                        _reply_admin_dm(
                                            (
                                                f"🧾 일지상세 {tid}\n"
                                                f"- 코인: {d.get('coin')}\n"
                                                f"- 스타일: {d.get('style','-')} ({d.get('style_confidence','-')}%)\n"
                                                f"- 방향: {d.get('decision')}\n"
                                                f"- 확신도: {d.get('confidence')}\n"
                                                f"- 진입가: {d.get('entry_price')}\n"
                                                f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                                f"- 레버: x{d.get('lev')}\n"
                                                f"- 단기추세: {d.get('trend_short','-')}\n"
                                                f"- 장기추세: {d.get('trend_long','-')}\n"
                                                f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                                f"- 한줄근거: {str(d.get('reason_easy',''))[:200]}\n"
                                                f"- 사용지표: {', '.join(d.get('used_indicators', []))[:200]}\n"
                                            )[:3500]
                                        )

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up.get("callback_query") or {}
                        data = str(cb.get("data", "") or "")
                        cb_id = str(cb.get("id", "") or "")
                        uid = (cb.get("from") or {}).get("id", None)
                        is_admin = tg_is_admin(uid)
                        cb_chat_id = (((cb.get("message") or {}).get("chat") or {}) if isinstance((cb.get("message") or {}).get("chat"), dict) else {}).get("id", None)

                        def _cb_reply(m: str):
                            # ✅ 요구: 버튼 응답은 관리자 DM(TG_ADMIN_USER_IDS) 우선
                            if TG_ADMIN_IDS:
                                if uid is not None:
                                    tg_send_chat(uid, m)
                                else:
                                    tg_send(m, target="admin", cfg=cfg)
                            else:
                                # fallback: 버튼이 있던 채팅으로 답장
                                if cb_chat_id is not None:
                                    tg_send_chat(cb_chat_id, m)
                                else:
                                    tg_send(m, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)

                        if data == "status":
                            # 누구나
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt2 = load_runtime()
                            regime_mode = str(cfg_live.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            _cb_reply(
                                f"📡 상태\n- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 레짐: {regime_txt}\n"
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})\n"
                                f"- 연속손실: {rt2.get('consec_losses',0)}\n"
                            )

                        elif data == "vision":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                                coins = mon_now.get("coins", {}) or {}
                                lines = ["👁️ AI 시야(요약)", f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}"]
                                for sym, cs in list(coins.items())[:10]:
                                    style = str(cs.get("style_reco", "")) or ""
                                    stxt = f"[{style}] " if style else ""
                                    lines.append(
                                        f"- {sym}: {stxt}{str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                        f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                        f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                                    )
                                _cb_reply("\n".join(lines))

                        elif data == "balance":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                free, total = safe_fetch_balance(ex)
                                _cb_reply(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT")

                        elif data == "position":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                msg = ["📊 포지션"]
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    msg.append("- ⚪ 없음(관망)")
                                else:
                                    for p in act:
                                        sym = p.get("symbol", "")
                                        side = position_side_normalize(p)
                                        roi = float(position_roi_percent(p))
                                        upnl = float(p.get("unrealizedPnl") or 0.0)
                                        lev = p.get("leverage", "?")
                                        style = str((active_targets.get(sym, {}) or {}).get("style", ""))
                                        msg.append(_fmt_pos_line(sym, side, lev, roi, upnl, style=style))
                                _cb_reply("\n".join(msg))

                        elif data == "log":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                df_log = read_trade_log()
                                if df_log.empty:
                                    _cb_reply("📜 일지 없음")
                                else:
                                    top = df_log.head(8)
                                    msg = ["📜 최근 매매일지(요약)"]
                                    for _, r in top.iterrows():
                                        tid = str(r.get("TradeID", "") or "")
                                        pnl = float(r.get("PnL_Percent", 0) or 0)
                                        emo = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                                        msg.append(
                                            f"- {emo} {r['Time']} {r['Coin']} {r['Side']} {pnl:.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}"
                                        )
                                    _cb_reply("\n".join(msg))

                        elif data == "log_detail_help":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🧾 일지 조회\n- /log : 최근 요약\n- /log <ID> : 상세\n- (호환) 일지상세 <ID>")

                        elif data == "scan":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                rid = uuid.uuid4().hex[:8]
                                rt2 = load_runtime()
                                rt2["force_scan"] = {
                                    "id": rid,
                                    "requested_at_epoch": time.time(),
                                    "requested_at_kst": now_kst_str(),
                                    "requested_by": int(uid or 0),
                                    "symbols": list(TARGET_COINS),
                                    "scan_only": True,
                                    "done": False,
                                }
                                save_runtime(rt2)
                                try:
                                    mon_add_event(mon, "SCAN_REQUEST", "", f"force_scan id={rid}", {"symbols": list(TARGET_COINS), "by": uid})
                                    gsheet_log_event("SCAN_REQUEST", message=f"id={rid}", payload={"symbols": list(TARGET_COINS), "by": uid})
                                except Exception:
                                    pass
                                _cb_reply(f"🔎 강제스캔 요청 완료: {rid}\n- 주의: 스캔만 수행(주문X)")

                        elif data == "mode_help":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🎚️ /mode 사용법\n- /mode auto\n- /mode scalping\n- /mode swing")

                        elif data == "gsheet":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                stg = gsheet_status_snapshot()
                                msg = (
                                    "📎 Google Sheets 상태\n"
                                    f"- enabled: {stg.get('enabled')}\n"
                                    f"- connected: {stg.get('connected')}\n"
                                    f"- spreadsheet_id: {stg.get('spreadsheet_id')}\n"
                                    f"- worksheet: {stg.get('worksheet')}\n"
                                    f"- service_account_email: {stg.get('service_account_email')}\n"
                                    f"- queue_high/scan: {stg.get('queue_high')}/{stg.get('queue_scan')}\n"
                                    f"- last_append: {stg.get('last_append_kst')} ({stg.get('last_append_type')}:{stg.get('last_append_stage')})\n"
                                    f"- last_err: {stg.get('last_err')}\n"
                                    "테스트: /gsheet test"
                                )
                                _cb_reply(msg[:3500])

                        elif data == "close_all":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🛑 전량 청산 시도")
                                for sym in TARGET_COINS:
                                    ps = safe_fetch_positions(ex, [sym])
                                    act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                    if not act:
                                        continue
                                    p = act[0]
                                    side = position_side_normalize(p)
                                    contracts = float(p.get("contracts") or 0)
                                    close_position_market(ex, sym, side, contracts)
                                _cb_reply("✅ 전량 청산 요청 완료")
                                try:
                                    mon_add_event(mon, "CLOSE_ALL", "", "close_all requested", {"by": uid})
                                    gsheet_log_event("CLOSE_ALL", message="close_all", payload={"by": uid})
                                except Exception:
                                    pass

                        if cb_id:
                            tg_answer_callback(cb_id)

                except Exception as _e:
                    # 업데이트 처리 중 오류도 EVENT로 남김(봇은 계속)
                    try:
                        mon_add_event(mon, "TG_UPDATE_ERROR", "", "TG update 처리 오류", {"err": str(_e)[:240]})
                        gsheet_log_event("TG_UPDATE_ERROR", message=str(_e)[:240])
                    except Exception:
                        pass
                    notify_admin_error("TG_UPDATE_HANDLER", _e, tb=traceback.format_exc(), min_interval_sec=60.0)

            monitor_write_throttled(mon, 2.0)
            backoff_sec = 1.0
            time.sleep(0.8)

        except Exception as e:
            # 스레드가 죽지 않도록 backoff
            try:
                notify_admin_error("TG_THREAD_LOOP", e, tb=traceback.format_exc(), min_interval_sec=45.0)
                err = f"{e}"
                if len(err) > 500:
                    err = err[:500] + "..."
                # ✅ 요구: 오류는 관리자 DM으로(채널 스팸 방지)
                if not TG_ADMIN_IDS:
                    tg_send(f"⚠️ 스레드 오류: {err}", target="channel", cfg=load_settings())
            except Exception:
                pass
            time.sleep(backoff_sec)
            backoff_sec = float(clamp(backoff_sec * 1.6, 1.0, 15.0))


# =========================================================
# ✅ 17.5) Watchdog: 하트비트 멈춤 감시/경고/재시작 시도
# =========================================================
def watchdog_thread():
    warned = False
    while True:
        try:
            mon = read_json_safe(MONITOR_FILE, {}) or {}
            hb = float(mon.get("last_heartbeat_epoch", 0) or 0)
            age = (time.time() - hb) if hb else 9999
            cfg = load_settings()
            if age >= 60 and not warned:
                warned = True
                msg = f"🧯 워치독 경고: 하트비트 {age:.0f}초 정체(스레드 멈춤 의심)"
                tg_send(msg, target="channel", cfg=cfg)
                tg_send(msg, target="admin", cfg=cfg)
            if age < 30:
                warned = False

            # 스레드가 아예 없으면 재시작
            alive = False
            for t in threading.enumerate():
                if t.name == "TG_THREAD" and t.is_alive():
                    alive = True
                    break
            if not alive:
                try:
                    ex2 = create_exchange_client_uncached() or exchange
                    th = threading.Thread(target=telegram_thread, args=(ex2,), daemon=True, name="TG_THREAD")
                    add_script_run_ctx(th)
                    th.start()
                    msg2 = "🧯 워치독: TG_THREAD 재시작 시도"
                    tg_send(msg2, target="channel", cfg=cfg)
                    tg_send(msg2, target="admin", cfg=cfg)
                except Exception:
                    pass

        except Exception:
            pass
        time.sleep(5.0)


# =========================================================
# ✅ 18) 스레드 시작(중복 방지) - TG_THREAD + WATCHDOG
# =========================================================
def ensure_threads_started():
    has_tg = False
    has_wd = False
    has_poll = False
    has_gs = False
    for t in threading.enumerate():
        if t.name == "TG_THREAD":
            has_tg = True
        if t.name == "TG_POLL_THREAD":
            has_poll = True
        if t.name == "GSHEET_THREAD":
            has_gs = True
        if t.name == "WATCHDOG_THREAD":
            has_wd = True
    if not has_poll:
        # Telegram long polling(getUpdates) 전용 스레드 (요구사항)
        thp = threading.Thread(target=telegram_polling_thread, args=(), daemon=True, name="TG_POLL_THREAD")
        add_script_run_ctx(thp)
        thp.start()
    if not has_gs:
        # Google Sheets append_row 전용 워커 (요구사항)
        thg = threading.Thread(target=gsheet_worker_thread, args=(), daemon=True, name="GSHEET_THREAD")
        add_script_run_ctx(thg)
        thg.start()
    if not has_tg:
        ex2 = create_exchange_client_uncached() or exchange
        th = threading.Thread(target=telegram_thread, args=(ex2,), daemon=True, name="TG_THREAD")
        add_script_run_ctx(th)
        th.start()
    if not has_wd:
        wd = threading.Thread(target=watchdog_thread, args=(), daemon=True, name="WATCHDOG_THREAD")
        add_script_run_ctx(wd)
        wd.start()


# 전역 예외 훅 설치(가능한 경우): 스레드/런타임에서 잡히지 않은 오류를 관리자 DM으로
install_global_error_hooks()
ensure_threads_started()


# =========================================================
# ✅ 19) Streamlit UI
# =========================================================
st.sidebar.title("🛠️ 제어판")
st.sidebar.caption("Streamlit=제어/상태 확인용, Telegram=실시간 보고/조회용")

openai_key_secret = _sget_str("OPENAI_API_KEY")
if not openai_key_secret and not config.get("openai_api_key"):
    k = st.sidebar.text_input("OpenAI API Key 입력(선택)", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

with st.sidebar.expander("🧪 디버그: 저장된 설정(bot_settings.json) 확인"):
    st.json(read_json_safe(SETTINGS_FILE, {}))

mode_keys = list(MODE_RULES.keys())
safe_mode = config.get("trade_mode", "안전모드")
if safe_mode not in mode_keys:
    safe_mode = "안전모드"
config["trade_mode"] = st.sidebar.selectbox("매매 모드", mode_keys, index=mode_keys.index(safe_mode))

auto_on = st.sidebar.checkbox("🤖 자동매매 (텔레그램 연동)", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.divider()
config["timeframe"] = st.sidebar.selectbox(
    "타임프레임",
    ["1m", "3m", "5m", "15m", "1h"],
    index=["1m", "3m", "5m", "15m", "1h"].index(config.get("timeframe", "5m")),
)
config["tg_enable_reports"] = st.sidebar.checkbox("📨 텔레그램 이벤트 알림(진입/청산 등)", value=bool(config.get("tg_enable_reports", True)))
config["tg_send_entry_reason"] = st.sidebar.checkbox("📌 텔레그램에 진입근거(긴글)도 보내기", value=bool(config.get("tg_send_entry_reason", False)))

st.sidebar.subheader("⏱️ 주기 리포트")
config["tg_enable_periodic_report"] = st.sidebar.checkbox("15분(기본) 상황보고", value=bool(config.get("tg_enable_periodic_report", True)))
config["report_interval_min"] = st.sidebar.number_input("상황보고 주기(분)", 3, 120, int(config.get("report_interval_min", 15)))
config["tg_enable_hourly_vision_report"] = st.sidebar.checkbox("1시간 AI시야 리포트(채널)", value=bool(config.get("tg_enable_hourly_vision_report", True)))
config["vision_report_interval_min"] = st.sidebar.number_input("AI시야 리포트 주기(분)", 10, 240, int(config.get("vision_report_interval_min", 60)))

st.sidebar.subheader("📡 텔레그램 라우팅")
config["tg_route_events_to"] = st.sidebar.selectbox("이벤트(진입/익절/손절/보고) 전송 대상", ["channel", "group", "both"], index=["channel", "group", "both"].index(config.get("tg_route_events_to", "channel")))
config["tg_route_queries_to"] = st.sidebar.selectbox("조회/버튼 응답 전송 대상", ["group", "channel", "both"], index=["group", "channel", "both"].index(config.get("tg_route_queries_to", "group")))
st.sidebar.caption("※ TG_CHAT_ID_GROUP / TG_CHAT_ID_CHANNEL secrets를 설정하면 채널/그룹 분리가 됩니다.")

st.sidebar.divider()
st.sidebar.subheader("🧭 추세/스타일 정책")
config["trend_filter_enabled"] = st.sidebar.checkbox("장기추세(1h) 정책 사용", value=bool(config.get("trend_filter_enabled", True)))
config["trend_filter_timeframe"] = "1h"
config["trend_filter_policy"] = st.sidebar.selectbox("정책", ["ALLOW_SCALP", "STRICT", "OFF"], index=["ALLOW_SCALP", "STRICT", "OFF"].index(config.get("trend_filter_policy", "ALLOW_SCALP")))
st.sidebar.caption("ALLOW_SCALP: 역추세 허용(스캘핑 강제) / STRICT: 역추세 금지 / OFF: 미사용")

config["regime_mode"] = st.sidebar.selectbox(
    "레짐 모드(/mode)",
    ["auto", "scalping", "swing"],
    index=["auto", "scalping", "swing"].index(str(config.get("regime_mode", "auto")).lower() if str(config.get("regime_mode", "auto")).lower() in ["auto", "scalping", "swing"] else "auto"),
)
config["regime_switch_control"] = st.sidebar.selectbox(
    "레짐 흔들림 방지(시간락 없음)",
    ["confirm2", "hysteresis", "off"],
    index=["confirm2", "hysteresis", "off"].index(str(config.get("regime_switch_control", "confirm2")).lower() if str(config.get("regime_switch_control", "confirm2")).lower() in ["confirm2", "hysteresis", "off"] else "confirm2"),
)
with st.sidebar.expander("히스테리시스 상세(선택)"):
    c_h1, c_h2, c_h3 = st.columns(3)
    config["regime_hysteresis_step"] = c_h1.number_input("step", 0.05, 1.0, float(config.get("regime_hysteresis_step", 0.55)), step=0.05)
    config["regime_hysteresis_enter_swing"] = c_h2.number_input("enter swing", 0.1, 0.99, float(config.get("regime_hysteresis_enter_swing", 0.75)), step=0.05)
    config["regime_hysteresis_enter_scalp"] = c_h3.number_input("enter scalp", 0.01, 0.9, float(config.get("regime_hysteresis_enter_scalp", 0.25)), step=0.05)

config["style_auto_enable"] = st.sidebar.checkbox("스캘핑/스윙 자동 선택/전환", value=bool(config.get("style_auto_enable", True)))
config["style_lock_minutes"] = st.sidebar.number_input("스타일 전환 락(분) [DEPRECATED]", 0, 180, int(config.get("style_lock_minutes", 20)))
st.sidebar.caption("※ 요구사항 반영: 시간 기반 최소유지기간은 사용하지 않습니다(레짐 흔들림 방지=confirm2/hysteresis).")

st.sidebar.subheader("🧩 스윙 분할익절/순환")
config["swing_partial_tp_enable"] = st.sidebar.checkbox("스윙: 1/2/3차 분할익절", value=bool(config.get("swing_partial_tp_enable", True)))
with st.sidebar.expander("분할익절 상세 설정"):
    p1a, p1b = st.columns(2)
    config["swing_partial_tp1_at_tp_frac"] = p1a.number_input("1차: TP비율", 0.05, 0.95, float(config.get("swing_partial_tp1_at_tp_frac", 0.35)), step=0.05)
    config["swing_partial_tp1_close_pct"] = p1b.number_input("1차: 청산%", 1, 90, int(config.get("swing_partial_tp1_close_pct", 33)))
    p2a, p2b = st.columns(2)
    config["swing_partial_tp2_at_tp_frac"] = p2a.number_input("2차: TP비율", 0.05, 0.95, float(config.get("swing_partial_tp2_at_tp_frac", 0.60)), step=0.05)
    config["swing_partial_tp2_close_pct"] = p2b.number_input("2차: 청산%", 1, 90, int(config.get("swing_partial_tp2_close_pct", 33)))
    p3a, p3b = st.columns(2)
    config["swing_partial_tp3_at_tp_frac"] = p3a.number_input("3차: TP비율", 0.05, 0.99, float(config.get("swing_partial_tp3_at_tp_frac", 0.85)), step=0.05)
    config["swing_partial_tp3_close_pct"] = p3b.number_input("3차: 청산%", 1, 95, int(config.get("swing_partial_tp3_close_pct", 34)))

config["swing_recycle_enable"] = st.sidebar.checkbox("스윙: 순환매도(부분익절 후 재진입)", value=bool(config.get("swing_recycle_enable", False)))
with st.sidebar.expander("순환매도 상세 설정"):
    r1, r2, r3 = st.columns(3)
    config["swing_recycle_cooldown_min"] = r1.number_input("쿨다운(분)", 1, 240, int(config.get("swing_recycle_cooldown_min", 20)))
    config["swing_recycle_max_count"] = r2.number_input("최대횟수", 0, 10, int(config.get("swing_recycle_max_count", 2)))
    config["swing_recycle_reentry_roi"] = r3.number_input("재진입ROI(%)", 0.1, 20.0, float(config.get("swing_recycle_reentry_roi", 0.8)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🧱 지지/저항(SR) 손절/익절")
config["use_sr_stop"] = st.sidebar.checkbox("SR 기반 가격 손절/익절 사용", value=bool(config.get("use_sr_stop", True)))
c_sr1, c_sr2 = st.sidebar.columns(2)
config["sr_timeframe"] = c_sr1.selectbox("SR 타임프레임", ["5m", "15m", "1h", "4h"], index=["5m", "15m", "1h", "4h"].index(config.get("sr_timeframe", "15m")))
config["sr_pivot_order"] = c_sr2.number_input("피벗 민감도", 3, 10, int(config.get("sr_pivot_order", 6)))
c_sr_lb1, c_sr_lb2 = st.sidebar.columns(2)
config["sr_lookback"] = c_sr_lb1.number_input("SR Lookback", 120, 800, int(config.get("sr_lookback", 220)), step=10)
config["sr_levels_cache_sec"] = c_sr_lb2.number_input("SR Cache(초)", 5, 600, int(config.get("sr_levels_cache_sec", 60)), step=5)
c_sr3, c_sr4 = st.sidebar.columns(2)
config["sr_atr_period"] = c_sr3.number_input("ATR 기간", 7, 30, int(config.get("sr_atr_period", 14)))
config["sr_buffer_atr_mult"] = c_sr4.number_input("버퍼(ATR배)", 0.05, 2.0, float(config.get("sr_buffer_atr_mult", 0.25)), step=0.05)
config["sr_rr_min"] = st.sidebar.number_input("SR 최소 RR", 1.0, 5.0, float(config.get("sr_rr_min", 1.5)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🛡️ 방어/자금 관리")
config["loss_pause_enable"] = st.sidebar.checkbox("연속손실 보호(자동 정지)", value=bool(config.get("loss_pause_enable", True)))
c1, c2 = st.sidebar.columns(2)
config["loss_pause_after"] = c1.number_input("연속손실 N회", 1, 20, int(config.get("loss_pause_after", 3)))
config["loss_pause_minutes"] = c2.number_input("정지(분)", 1, 240, int(config.get("loss_pause_minutes", 30)))

st.sidebar.divider()
config["use_trailing_stop"] = st.sidebar.checkbox("🚀 트레일링 스탑(수익보호)", value=bool(config.get("use_trailing_stop", True)))
config["use_dca"] = st.sidebar.checkbox("💧 물타기(DCA) (스윙 중심)", value=bool(config.get("use_dca", True)))
c3, c4 = st.sidebar.columns(2)
config["dca_trigger"] = c3.number_input("DCA 발동(%)", -90.0, -1.0, float(config.get("dca_trigger", -20.0)), step=0.5)
config["dca_max_count"] = c4.number_input("최대 횟수", 0, 10, int(config.get("dca_max_count", 1)))
config["dca_add_pct"] = st.sidebar.slider("추가 규모(원진입 대비 %)", 10, 200, int(config.get("dca_add_pct", 50)))

st.sidebar.divider()
st.sidebar.subheader("🪙 외부 시황")
config["use_external_context"] = st.sidebar.checkbox("외부 시황 통합", value=bool(config.get("use_external_context", True)))
config["external_koreanize_enable"] = st.sidebar.checkbox("외부시황 한글화(가능한 범위)", value=bool(config.get("external_koreanize_enable", True)))
config["external_ai_translate_enable"] = st.sidebar.checkbox("외부시황 AI 번역(비용↑)", value=bool(config.get("external_ai_translate_enable", False)))
config["news_translate_budget_sec"] = st.sidebar.number_input(
    "뉴스 번역 시간예산(초, 0=룰기반만)",
    0,
    60,
    int(config.get("news_translate_budget_sec", 10)),
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("🌅 아침 브리핑")
config["daily_btc_brief_enable"] = st.sidebar.checkbox("매일 아침 BTC 경제뉴스 5개", value=bool(config.get("daily_btc_brief_enable", True)))
cc_b1, cc_b2 = st.sidebar.columns(2)
config["daily_btc_brief_hour_kst"] = cc_b1.number_input("시(KST)", 0, 23, int(config.get("daily_btc_brief_hour_kst", 9)))
config["daily_btc_brief_minute_kst"] = cc_b2.number_input("분(KST)", 0, 59, int(config.get("daily_btc_brief_minute_kst", 0)))

st.sidebar.divider()
st.sidebar.subheader("📤 일별 내보내기")
config["export_daily_enable"] = st.sidebar.checkbox("일별 내보내기 활성화", value=bool(config.get("export_daily_enable", True)))
config["export_excel_enable"] = st.sidebar.checkbox("Excel(xlsx) 저장", value=bool(config.get("export_excel_enable", True)))
config["export_gsheet_enable"] = st.sidebar.checkbox("Google Sheets 저장(선택)", value=bool(config.get("export_gsheet_enable", False)))

st.sidebar.divider()
st.sidebar.subheader("📊 보조지표 (10종) ON/OFF")
colA, colB = st.sidebar.columns(2)
config["use_rsi"] = colA.checkbox("RSI", value=bool(config.get("use_rsi", True)))
config["use_bb"] = colB.checkbox("볼린저", value=bool(config.get("use_bb", True)))
config["use_ma"] = colA.checkbox("MA(이평)", value=bool(config.get("use_ma", True)))
config["use_macd"] = colB.checkbox("MACD", value=bool(config.get("use_macd", True)))
config["use_stoch"] = colA.checkbox("스토캐스틱", value=bool(config.get("use_stoch", True)))
config["use_cci"] = colB.checkbox("CCI", value=bool(config.get("use_cci", True)))
config["use_mfi"] = colA.checkbox("MFI", value=bool(config.get("use_mfi", True)))
config["use_willr"] = colB.checkbox("윌리엄%R", value=bool(config.get("use_willr", True)))
config["use_adx"] = colA.checkbox("ADX", value=bool(config.get("use_adx", True)))
config["use_vol"] = colB.checkbox("거래량", value=bool(config.get("use_vol", True)))

st.sidebar.divider()
st.sidebar.subheader("지표 파라미터")
r1, r2, r3 = st.sidebar.columns(3)
config["rsi_period"] = r1.number_input("RSI 기간", 5, 50, int(config.get("rsi_period", 14)))
config["rsi_buy"] = r2.number_input("과매도", 10, 50, int(config.get("rsi_buy", 30)))
config["rsi_sell"] = r3.number_input("과매수", 50, 90, int(config.get("rsi_sell", 70)))

b1, b2 = st.sidebar.columns(2)
config["bb_period"] = b1.number_input("BB 기간", 5, 50, int(config.get("bb_period", 20)))
config["bb_std"] = b2.number_input("BB 승수", 1.0, 5.0, float(config.get("bb_std", 2.0)))

m1, m2 = st.sidebar.columns(2)
config["ma_fast"] = m1.number_input("MA 단기", 3, 50, int(config.get("ma_fast", 7)))
config["ma_slow"] = m2.number_input("MA 장기", 50, 300, int(config.get("ma_slow", 99)))

st.sidebar.divider()
st.sidebar.subheader("🔍 긴급 점검")
if st.sidebar.button("📡 텔레그램 메뉴 전송(/menu)"):
    tg_send_menu(cfg=config)

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    # 운영자가 결제/쿼터를 복구한 직후 즉시 재시도할 수 있게 수동 clear
    openai_clear_suspension(config)
    h = openai_health_info(config)
    client = get_openai_client(config)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        st.sidebar.error(f"❌ OpenAI 사용 불가: {msg}")
        if "insufficient_quota" in msg:
            st.sidebar.caption("OpenAI 결제/크레딧(Quota) 부족입니다. OpenAI 콘솔에서 Billing/크레딧을 확인하세요.")
        elif str(h.get("status")) == "NO_KEY":
            st.sidebar.caption("Streamlit secrets에 OPENAI_API_KEY를 설정하세요.")
    else:
        models_to_try = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
        last_err: Optional[BaseException] = None
        tried = []
        for m in models_to_try:
            tried.append(m)
            try:
                def _do():
                    return client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                        temperature=0.0,
                        max_tokens=16,
                    )

                resp = _call_with_timeout(_do, max(OPENAI_TIMEOUT_SEC, 30))
                out = (resp.choices[0].message.content or "").strip()
                st.sidebar.success(f"✅ 연결 성공({m}): {out}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                # quota/키오류면 더 시도해도 의미 없음
                kind = _openai_err_kind(e)
                openai_handle_failure(e, config, where="UI_OPENAI_TEST")
                if kind in ["insufficient_quota", "invalid_api_key"]:
                    break
                continue
        if last_err is not None:
            st.sidebar.error(f"❌ 실패: {last_err}")
            notify_admin_error("UI:OPENAI_TEST", last_err, context={"models_tried": tried})

# ✅ Google Sheets 연결 테스트(요구사항)
if st.sidebar.button("📎 Google Sheets 연결 테스트"):
    try:
        res = gsheet_test_append_row(timeout_sec=25)
        if res.get("ok"):
            st.sidebar.success("✅ Google Sheets append_row 성공(GSHEET_TEST)")
        else:
            st.sidebar.error(f"❌ Google Sheets 실패: {res.get('error','')}")
        stg = gsheet_status_snapshot()
        email = str(stg.get("service_account_email", "")).strip()
        if email:
            st.sidebar.caption(f"서비스계정 이메일(시트 공유 필요): {email}")
        else:
            st.sidebar.caption("서비스계정 이메일을 읽지 못했어요(GSHEET_SERVICE_ACCOUNT_JSON 확인).")
        with st.sidebar.expander("Google Sheets 상태(디버그)"):
            st.json(stg)
    except Exception as e:
        st.sidebar.error(f"❌ 테스트 오류: {e}")
        notify_admin_error("UI:GSHEET_TEST", e, context={"code": CODE_VERSION})

save_settings(config)

with st.sidebar:
    st.divider()
    st.header("내 지갑 현황")
    free, total = safe_fetch_balance(exchange)
    st.metric("총 자산(USDT)", f"{total:,.2f}")
    st.metric("주문 가능", f"{free:,.2f}")

    st.divider()
    st.subheader("보유 포지션(주요 5개)")
    try:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        if not act:
            st.caption("무포지션(관망)")
        else:
            for p in act[:5]:
                sym = p.get("symbol", "")
                side = position_side_normalize(p)
                roi = float(position_roi_percent(p))
                lev = p.get("leverage", "?")
                upnl = float(p.get("unrealizedPnl") or 0.0)
                emo = "🟢" if roi >= 0 else "🔴"
                st.info(f"**{emo} {sym}** ({'롱' if side=='long' else '숏'} x{lev})\nROI: **{roi:.2f}%** (PnL {upnl:.2f} USDT)")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")


# =========================================================
# ✅ Main UI
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트 (Final Integrated)")
st.caption("Streamlit=제어판/모니터링, Telegram=실시간 보고/조회. (모의투자 IS_SANDBOX=True)")

markets = exchange.markets or {}
if markets:
    symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
    if not symbol_list:
        symbol_list = TARGET_COINS
else:
    symbol_list = TARGET_COINS

symbol = st.selectbox("코인 선택", symbol_list, index=0)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("📉 TradingView 차트 (다크모드)")
    interval_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}
    render_tradingview(symbol, interval=interval_map.get(config.get("timeframe", "5m"), "5"), height=560)

with right:
    st.subheader("🧾 실시간 지표 요약")
    if ta is None and pta is None:
        st.error("ta/pandas_ta 모듈이 없습니다. requirements.txt에 `ta` 또는 `pandas_ta` 추가 후 재배포하세요.")
    else:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df2, stt, last = calc_indicators(df, config)

            # 장기추세(1h)도 같이 표시
            htf_tf = "1h"
            htf_trend = get_htf_trend_cached(
                exchange,
                symbol,
                htf_tf,
                fast=int(config.get("ma_fast", 7)),
                slow=int(config.get("ma_slow", 99)),
                cache_sec=int(config.get("trend_filter_cache_sec", 60)),
            )

            if last is None:
                # 지표가 부족해도 장기추세/스타일은 표시(사용자 체감 개선)
                st.warning("지표 계산 실패(데이터 부족/지표 계산 오류)")
                style_hint = _style_for_entry(symbol, "buy", "", htf_trend, config)
                st.write(
                    {
                        "장기추세(1h)": f"🧭 {htf_trend}",
                        "추천 스타일(롱 관점)": f"{style_hint.get('style','-')} ({style_hint.get('confidence','-')}%)",
                        "상태": stt.get("_ERROR") or stt.get("_INFO") or "-",
                    }
                )
            else:
                st.metric("현재가", f"{float(last['close']):,.4f}")
                # 스타일 추천(현재 차트 기준)
                style_hint = _style_for_entry(symbol, "buy", stt.get("추세", ""), htf_trend, config)
                show = {
                    "단기추세(현재봉)": stt.get("추세", "-"),
                    "장기추세(1h)": f"🧭 {htf_trend}",
                    "추천 스타일(롱 관점)": f"{style_hint.get('style','-')} ({style_hint.get('confidence','-')}%)",
                    "RSI": stt.get("RSI", "-"),
                    "BB": stt.get("BB", "-"),
                    "MACD": stt.get("MACD", "-"),
                    "ADX": stt.get("ADX", "-"),
                    "거래량": stt.get("거래량", "-"),
                    "눌림목후보(해소)": "✅" if stt.get("_pullback_candidate") else "—",
                    "지표엔진": stt.get("_backend", "-"),
                }
                st.write(show)

                if config.get("use_sr_stop", True):
                    try:
                        sr_tf = config.get("sr_timeframe", "15m")
                        sr_lb = int(config.get("sr_lookback", 220))
                        htf = exchange.fetch_ohlcv(symbol, sr_tf, limit=sr_lb)
                        hdf = pd.DataFrame(htf, columns=["time", "open", "high", "low", "close", "vol"])
                        hdf["time"] = pd.to_datetime(hdf["time"], unit="ms")
                        sr = sr_stop_take(
                            entry_price=float(last["close"]),
                            side="buy",
                            htf_df=hdf,
                            atr_period=int(config.get("sr_atr_period", 14)),
                            pivot_order=int(config.get("sr_pivot_order", 6)),
                            buffer_atr_mult=float(config.get("sr_buffer_atr_mult", 0.25)),
                            rr_min=float(config.get("sr_rr_min", 1.5)),
                        )
                        if sr:
                            st.caption(f"SR(참고): 롱 기준 TP {sr['tp_price']:.6g} / SL {sr['sl_price']:.6g}")
                    except Exception as e:
                        notify_admin_error("UI:SR_CALC", e, context={"symbol": symbol, "tf": str(config.get("sr_timeframe", ""))}, min_interval_sec=120.0)

        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            notify_admin_error("UI:INDICATOR_SUMMARY", e, context={"symbol": symbol, "tf": str(config.get("timeframe", ""))})

st.divider()

tabs = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지", "🧪 간이 백테스트"])
t1, t2, t3, t4, t5 = tabs

with t1:
    st.subheader("👁️ 실시간 AI 모니터링(봇 시야)")
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, key="mon_refresh")
    else:
        st.caption("자동 새로고침을 원하면 requirements.txt에 streamlit-autorefresh 추가")

    mon = read_json_safe(MONITOR_FILE, None)
    if not mon:
        st.warning("monitor_state.json이 아직 없습니다. (스레드 시작 확인)")
    else:
        # ✅ Google Sheets 상태(요구사항)
        try:
            st.subheader("📎 Google Sheets 상태")
            st.write(gsheet_status_snapshot())
            st.caption("※ 서비스계정 이메일로 스프레드시트를 '편집자'로 공유해야 append 됩니다.")
        except Exception:
            pass
        # 외부 시황 요약(항상 보이게)
        st.subheader("🌍 외부 시황 요약(한글/이모티콘)")
        ext = (mon.get("external") or {})
        if not ext or not ext.get("enabled", False):
            st.caption("외부 시황 통합 OFF")
        else:
            st.write(
                {
                    "갱신시각(KST)": ext.get("asof_kst"),
                    "중요이벤트(임박)": len(ext.get("high_impact_events_soon") or []),
                    "공포탐욕": (ext.get("fear_greed") or {}),
                    "도미넌스/시총": (ext.get("global") or {}),
                    "아침브리핑": (ext.get("daily_btc_brief") or {}),
                    "진입감산배수": mon.get("entry_risk_multiplier", 1.0),
                }
            )
            evs = ext.get("high_impact_events_soon") or []
            if evs:
                st.warning("⚠️ 중요 이벤트 임박(신규진입 보수적으로)")
                st_dataframe_safe(df_for_display(pd.DataFrame(evs)), hide_index=True)
            hd = ext.get("headlines") or []
            if hd:
                st.caption("뉴스 헤드라인(요약용)")
                st.write(hd[:10])

        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = (time.time() - hb) if hb else 9999
        try:
            scan_cycle_sec = float(mon.get("scan_cycle_sec", 0) or 0)
        except Exception:
            scan_cycle_sec = 0.0
        # 요구사항: heartbeat lag가 scan_interval*4 이상이면 '멈춤 의심'
        stale_thresh = max(60.0, float(scan_cycle_sec) * 4.0) if scan_cycle_sec > 0 else 60.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        c2.metric("모드", mon.get("trade_mode", "-"))
        c3.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < stale_thresh else "🔴 멈춤 의심")
        c4.metric("연속손실", str(mon.get("consec_losses", 0)))

        if age >= stale_thresh:
            st.error(f"⚠️ 봇 스레드가 멈췄거나(크래시) 갱신이 안될 수 있어요. ({stale_thresh:.0f}초 이상)")

        st.caption(
            f"봇 상태: {mon.get('global_state','-')} | stage: {mon.get('loop_stage','-')}@{mon.get('loop_stage_kst','-')} | code: {mon.get('code_version','-')}"
        )
        try:
            h = openai_health_info(load_settings())
            ai_txt = "OK" if bool(h.get("available", False)) else str(h.get("message", "OFF"))
            until = str(h.get("until_kst", "")).strip()
            if until and (not bool(h.get("available", False))):
                ai_txt = f"{ai_txt} (~{until} KST)"
            st.caption(f"OpenAI: {ai_txt}")
        except Exception:
            pass

        # ✅ 포지션/진입 정보(직관적 표시)
        st.subheader("📊 현재 포지션(스타일/목표 포함)")
        ops = mon.get("open_positions") or []
        if ops:
            st.caption(f"현재 포지션 수: {len(ops)}")
            st_dataframe_safe(df_for_display(pd.DataFrame(ops)), hide_index=True)
        else:
            st.caption("⚪ 포지션 없음(관망)")

        # ✅ 최근 이벤트(가독성 강화)
        st.subheader("🧾 최근 이벤트(봇 로그)")
        evs = (mon.get("events") or [])[-30:]
        if evs:
            st_dataframe_safe(df_for_display(pd.DataFrame(evs[::-1])), hide_index=True)
        else:
            st.caption("이벤트 없음")

        # ✅ AI/Scan Process (요구사항: 단계별 스캔 과정 표시)
        st.subheader("🧠 AI/Scan Process (최근)")
        scan_logs = (mon.get("scan_process") or [])
        if scan_logs:
            max_show = st.number_input("표시 개수(N)", 20, 400, 120, step=10)
            try:
                df_scan = pd.DataFrame(scan_logs[-int(max_show):])
                df_scan = df_scan.iloc[::-1].reset_index(drop=True)
                st_dataframe_safe(df_for_display(df_scan), hide_index=True)
            except Exception:
                st_dataframe_safe(df_for_display(pd.DataFrame(scan_logs[-int(max_show):][::-1])), hide_index=True)
        else:
            st.caption("SCAN 로그 없음")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0) or 0)
            scan_age = (time.time() - last_scan) if last_scan else 9999
            rows.append(
                {
                    "코인": sym,
                    "스캔(초전)": f"{scan_age:.1f}",
                    "스캔단계": cs.get("scan_stage", ""),
                    "단계시각": cs.get("scan_stage_kst", ""),
                    "가격": cs.get("price", ""),
                    "단기추세": cs.get("trend_short", ""),
                    "장기추세(1h)": cs.get("trend_long", ""),
                    "추천스타일": cs.get("style_reco", ""),
                    "스타일확신": cs.get("style_confidence", ""),
                    "RSI": cs.get("rsi", ""),
                    "ADX": cs.get("adx", ""),
                    "BB": cs.get("bb", ""),
                    "MACD": cs.get("macd", ""),
                    "눌림목후보": "✅" if cs.get("pullback_candidate") else "—",
                    "AI호출": "✅" if cs.get("ai_called") else "—",
                    "AI결론": str(cs.get("ai_decision", "-")).upper(),
                    "확신도": cs.get("ai_confidence", "-"),
                    "필요확신도": cs.get("min_conf_required", "-"),
                    "진입%": cs.get("ai_entry_pct", "-"),
                    "레버": cs.get("ai_leverage", "-"),
                    "SL%": cs.get("ai_sl_pct", "-"),
                    "TP%": cs.get("ai_tp_pct", "-"),
                    "손익비": cs.get("ai_rr", "-"),
                    "AI지표": cs.get("ai_used", ""),
                    "스킵/근거": (cs.get("skip_reason") or cs.get("ai_reason_easy") or "")[:160],
                }
            )
        if rows:
            st_dataframe_safe(df_for_display(pd.DataFrame(rows)), hide_index=True)
        else:
            st.info("아직 스캔 데이터가 없습니다.")

    st.divider()
    st.subheader("🔍 현재 코인 AI 분석(수동 버튼)")
    if st.button("현재 코인 AI 분석 실행"):
        # 수동 실행은 운영자가 즉시 재시도할 수 있게 suspend를 클리어
        openai_clear_suspension(config)
        if get_openai_client(config) is None:
            h = openai_health_info(config)
            msg = str(h.get("message", "OpenAI 사용 불가")).strip()
            until = str(h.get("until_kst", "")).strip()
            if until:
                msg = f"{msg} (~{until} KST)"
            st.error(msg)
        elif ta is None and pta is None:
            st.error("ta/pandas_ta 모듈 없음")
        else:
            try:
                ext_now = build_external_context(config, rt=load_runtime())
                ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if last is None:
                    st.warning("지표 계산 실패")
                else:
                    ai = ai_decide_trade(df2, stt, symbol, config.get("trade_mode", "안전모드"), config, external=ext_now)
                    # 스타일 힌트
                    htf_trend = get_htf_trend_cached(exchange, symbol, "1h", int(config.get("ma_fast", 7)), int(config.get("ma_slow", 99)), int(config.get("trend_filter_cache_sec", 60)))
                    style_info = _style_for_entry(symbol, ai.get("decision", "hold"), stt.get("추세", ""), htf_trend, config)
                    st.json({"ai": ai, "style": style_info, "htf_trend": htf_trend})
            except Exception as e:
                st.error(f"분석 오류: {e}")
                notify_admin_error("UI:MANUAL_AI_ANALYSIS", e, context={"symbol": symbol, "tf": str(config.get("timeframe", ""))})

with t2:
    st.subheader("⚡ 수동 주문(데모용)")
    st.caption("⚠️ 수동 주문은 실수 방지를 위해 기본은 '설명/테스트' 중심입니다.")
    amt = st.number_input("주문 금액(USDT)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    config["order_usdt"] = float(amt)
    save_settings(config)

    enable_manual = st.checkbox("수동 주문 활성화(주의!)", value=False)
    b1, b2, b3 = st.columns(3)

    if b1.button("🟢 롱 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "buy", qty)
            st.success("롱 진입 성공" if ok else "롱 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b2.button("🔴 숏 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "sell", qty)
            st.success("숏 진입 성공" if ok else "숏 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b3.button("🚫 전량 청산") and enable_manual:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        for p in act:
            sym = p.get("symbol", "")
            side = position_side_normalize(p)
            contracts = float(p.get("contracts") or 0)
            close_position_market(exchange, sym, side, contracts)
        st.success("전량 청산 요청 완료(데모)")

with t3:
    st.subheader("📅 시장정보(외부 시황)")
    try:
        ext = build_external_context(config, rt=load_runtime())
        if not ext.get("enabled"):
            st.info("외부 시황 통합 OFF")
        else:
            st.json(ext)
    except Exception as e:
        st.error(f"시장정보 로딩 오류: {e}")
        notify_admin_error("UI:MARKET_INFO", e, min_interval_sec=120.0)

with t4:
    st.subheader("📜 매매일지 (이모티콘/색상 + 일별 내보내기)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    if c1.button("🔄 새로고침"):
        st.rerun()
    if c2.button("🧹 매매일지 초기화"):
        reset_trade_log()
        st.success("매매일지 초기화 완료")
        st.rerun()
    if c3.button("📤 오늘 일지 내보내기"):
        try:
            res = export_trade_log_daily(today_kst_str(), config)
            if res.get("ok"):
                st.success(f"내보내기 완료: rows={res.get('rows')} | xlsx={res.get('excel_path','')} | csv={res.get('csv_path','')}")
            else:
                st.error(f"내보내기 실패: {res.get('error','')}")
        except Exception as e:
            st.error(f"내보내기 오류: {e}")
            notify_admin_error("UI:EXPORT_TODAY", e, min_interval_sec=120.0)

    df_log = read_trade_log()
    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        # 표시 개선: 이모티콘 + 색상
        df_show = df_log.copy()
        try:
            df_show["PnL_Percent"] = pd.to_numeric(df_show.get("PnL_Percent"), errors="coerce")
            df_show["PnL_USDT"] = pd.to_numeric(df_show.get("PnL_USDT"), errors="coerce")
            df_show.insert(
                0,
                "상태",
                df_show["PnL_Percent"].apply(lambda v: "🟢" if pd.notna(v) and float(v) > 0 else ("🔴" if pd.notna(v) and float(v) < 0 else "⚪")),
            )
        except Exception:
            pass

        show_cols = [c for c in ["상태", "Time", "Coin", "Side", "PnL_Percent", "PnL_USDT", "OneLine", "Reason", "Review", "TradeID"] if c in df_show.columns]

        def _color_pnl(v):
            try:
                x = float(v)
            except Exception:
                return ""
            if x > 0:
                return "background-color: rgba(0, 200, 0, 0.18); color: #00c853;"
            if x < 0:
                return "background-color: rgba(220, 0, 0, 0.18); color: #ff1744;"
            return ""

        try:
            # pandas 2.2+: Styler.applymap deprecate → map 사용
            sty0 = df_show[show_cols].style
            if hasattr(sty0, "map"):
                sty = sty0.map(_color_pnl, subset=["PnL_Percent", "PnL_USDT"])  # type: ignore[attr-defined]
            else:
                sty = sty0.applymap(_color_pnl, subset=["PnL_Percent", "PnL_USDT"])
            st_dataframe_safe(sty, hide_index=True)
        except Exception:
            st_dataframe_safe(df_for_display(df_show[show_cols]), hide_index=True)

        csv_bytes = df_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

    st.divider()
    st.subheader("🧾 상세일지 조회(TradeID)")
    tid = st.text_input("TradeID 입력 (텔레그램 '일지'에 ID가 나옵니다)")
    if st.button("상세일지 열기"):
        if not tid.strip():
            st.warning("TradeID를 입력해줘.")
        else:
            d = load_trade_detail(tid.strip())
            if not d:
                st.error("해당 ID를 찾지 못했어.")
            else:
                st.json(d)

    st.divider()
    st.subheader("📌 runtime_state.json (현재 상태)")
    rt = load_runtime()
    st.json(rt)
    if st.button("🧼 runtime_state 초기화(오늘 기준)"):
        write_json_atomic(RUNTIME_FILE, default_runtime())
        st.success("runtime_state.json 초기화 완료")
        st.rerun()

with t5:
    st.subheader("🧪 간이 백테스트(가벼운 규칙 기반, 버튼 실행형)")
    st.caption("실제 주문이 아니라 과거 OHLCV로 '대략' 성능을 확인합니다. (기본 OFF, 클릭 시 실행)")

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    bt_symbol = bt_col1.selectbox("심볼", symbol_list, index=symbol_list.index(symbol) if symbol in symbol_list else 0)
    bt_tf = bt_col2.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m", "3m", "5m", "15m", "1h"].index(config.get("timeframe", "5m")))
    bt_n = bt_col3.number_input("최근 N봉", 200, 2000, 600, step=50)

    bt_style = st.selectbox("전략 스타일", ["스캘핑", "스윙"], index=0)
    run_bt = st.button("▶️ 백테스트 실행")

    if run_bt:
        if ta is None and pta is None:
            st.error("ta/pandas_ta 모듈 없음")
        else:
            try:
                ohlcv = exchange.fetch_ohlcv(bt_symbol, bt_tf, limit=int(bt_n))
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if df2 is None or df2.empty:
                    st.error("데이터 부족")
                else:
                    # 간이 시뮬: RSI 해소 + MA 추세 기반
                    trades = []
                    in_pos = False
                    side = None
                    entry_px = 0.0
                    peak = 0.0
                    equity = 0.0
                    max_equity = 0.0
                    max_dd = 0.0

                    # 스타일별 목표(대략)
                    tp = 1.8 if bt_style == "스캘핑" else 6.0
                    sl = 1.2 if bt_style == "스캘핑" else 3.0

                    for i in range(2, len(df2)):
                        row = df2.iloc[i]
                        prev = df2.iloc[i - 1]
                        price = float(row["close"])

                        # 간이 신호
                        trend = "횡보/전환"
                        if "MA_fast" in df2.columns and "MA_slow" in df2.columns:
                            if float(row["MA_fast"]) > float(row["MA_slow"]) and price > float(row["MA_slow"]):
                                trend = "상승추세"
                            elif float(row["MA_fast"]) < float(row["MA_slow"]) and price < float(row["MA_slow"]):
                                trend = "하락추세"
                        rsi_prev = float(prev.get("RSI", 50))
                        rsi_now = float(row.get("RSI", 50))
                        rsi_buy = float(config.get("rsi_buy", 30))
                        rsi_sell = float(config.get("rsi_sell", 70))

                        rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
                        rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

                        if not in_pos:
                            if trend == "상승추세" and rsi_resolve_long:
                                in_pos = True
                                side = "long"
                                entry_px = price
                                peak = price
                            elif trend == "하락추세" and rsi_resolve_short:
                                in_pos = True
                                side = "short"
                                entry_px = price
                                peak = price
                        else:
                            # ROI 계산(레버 무시, 단순 퍼센트)
                            if side == "long":
                                roi = ((price - entry_px) / entry_px) * 100.0
                                peak = max(peak, price)
                            else:
                                roi = ((entry_px - price) / entry_px) * 100.0
                                peak = min(peak, price)

                            if roi >= tp or roi <= -sl:
                                trades.append(roi)
                                equity += roi
                                max_equity = max(max_equity, equity)
                                max_dd = min(max_dd, equity - max_equity)
                                in_pos = False
                                side = None
                                entry_px = 0.0

                    if trades:
                        wins = sum(1 for x in trades if x > 0)
                        win_rate = wins / len(trades) * 100.0
                        gains = sum(x for x in trades if x > 0)
                        losses = -sum(x for x in trades if x < 0)
                        pf = gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0
                        total_ret = sum(trades)
                        avg_r = float(np.mean(trades))
                        st.metric("총 수익률(단순합)", f"{total_ret:.2f}%")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("거래수", str(len(trades)))
                        c2.metric("승률", f"{win_rate:.1f}%")
                        c3.metric("PF", f"{pf:.2f}" if pf != float("inf") else "inf")
                        c4.metric("MDD(단순)", f"{max_dd:.2f}%")
                        st.caption(f"평균 R(간이): {avg_r:.2f}")
                        st.write(pd.DataFrame({"trade_roi_pct": trades}).tail(50))
                    else:
                        st.warning("조건에 맞는 거래가 없었습니다.")
            except Exception as e:
                st.error(f"백테스트 오류: {e}")
                notify_admin_error("UI:BACKTEST", e, context={"symbol": bt_symbol, "tf": bt_tf, "n": int(bt_n)}, min_interval_sec=120.0)


st.caption("⚠️ 이 봇은 모의투자(IS_SANDBOX=True)에서 충분히 검증 후 사용하세요.")


# =========================================================
# ✅ [중요] 모의투자 → 실전 전환 방법(자동 전환 절대 없음, 사용자가 직접 변경)
# =========================================================
# 1) 이 파일 상단의 IS_SANDBOX = True 를 False 로 변경
#    - IS_SANDBOX = False
# 2) Bitget 실계정 API 키를 Streamlit Secrets에 설정:
#    - (요구사항) BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
#    - (호환) API_KEY / API_SECRET / API_PASSWORD 도 자동 인식
#    - (권한) 선물(SWAP) 주문/포지션 조회 권한 필요
# 3) Telegram 채널/그룹 분리를 원하면 Secrets에 추가:
#    - (요구사항) TG_TOKEN / TG_TARGET_CHAT_ID
#    - (확장) TG_GROUP_ID / TG_CHANNEL_ID (있으면 자동 라우팅: 채널=알림/하트비트, 그룹=명령)
#      * 채널로 보내려면 봇이 채널 관리자여야 합니다.
# 4) 실전 전에는 반드시:
#    - 주문 수량/레버/SL/TP 로직을 소액으로 점검
#    - 거래소 최소수량/정밀도/슬리피지/수수료 고려
#    - 예기치 못한 버그/네트워크 장애 대비(위험 제한, 손실 감내 범위 설정)
# =========================================================

# =========================================================
# ✅ 검증 체크리스트(요구사항)
# =========================================================
# - Streamlit 실행 시 UI가 정상 표시되는가?
# - TG_TOKEN/TG_TARGET_CHAT_ID 설정 시 메시지가 정상 발송되는가?
#   - 채널 사용 시 봇을 채널 관리자(게시 권한)로 추가해야 함.
# - Telegram 명령이 동작하는가?
#   - /status (누구나)
#   - /positions /scan /mode auto|scalping|swing /log <id> (관리자: TG_ADMIN_USER_IDS 설정 시 제한)
# - GSHEET_ENABLED="true"일 때 Google Sheets에 append_row가 동작하는가?
#   - GSHEET_SERVICE_ACCOUNT_JSON 을 json.loads로 읽음
#   - GSHEET_SPREADSHEET_ID / GSHEET_WORKSHEET 로 워크시트 열고 없으면 생성
#   - 서비스계정 이메일로 스프레드시트 공유 필요
#   - TRADE/EVENT/SCAN 로그가 누적되는가?
# - 레짐 전환에 시간락(style_lock_minutes) 강제가 없는가?
#   - confirm2/hysteresis/off 로만 흔들림 제어
# - 기존 기능이 삭제되지 않았는가? (Streamlit 탭/수동주문/일지/백테스트/외부시황/내보내기 등)
# =========================================================
