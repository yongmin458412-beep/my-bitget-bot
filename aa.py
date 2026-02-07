# =========================================================
#  Bitget AI Wonyoti Agent (Final Integrated) + Dynamic Trend Filter
#  + Google Sheets Daily Log + Emoji/Color + Regime Switch (Scalp<->Swing)
#  + Telegram Channel/Group split + /menu + Daily BTC morning brief + Hourly AI vision
#
#  - Streamlit: 제어판/차트/포지션/일지/AI 시야
#  - Telegram:
#     * 채널: 진입/손절/익절/정기리포트/아침뉴스/시간당 AI시야
#     * 그룹: 버튼/조회/일지상세 등 인터랙션
#  - AutoTrade: 데모(IS_SANDBOX=True) 기반
#
#  ⚠️ 주의: 트레이딩은 손실 위험이 큽니다. (특히 레버리지)
#
#  Optional requirements.txt (있으면 사용 / 없어도 동작)
#   - ta
#   - streamlit-autorefresh
#   - orjson
#   - tenacity
#   - diskcache
#   - pandas_ta
#   - scipy
#   - feedparser
#   - cachetools
#   - loguru
#   - gspread / google-auth / gspread-formatting (구글시트)
# =========================================================

import os
import json
import time
import uuid
import math
import threading
import traceback
import random
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx

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
    from loguru import logger
except Exception:
    import logging
    logger = logging.getLogger("wonyoti")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Google Sheets (optional)
try:
    import gspread
except Exception:
    gspread = None

try:
    from google.oauth2.service_account import Credentials
except Exception:
    Credentials = None

try:
    # pip: gspread-formatting
    from gspread_formatting import (
        CellFormat, Color, format_cell_range, set_frozen, set_column_width,
        TextFormat, NumberFormat
    )
except Exception:
    CellFormat = None


# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Final)")

IS_SANDBOX = True  # ✅ 데모/모의투자

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"

DETAIL_DIR = "trade_details"
os.makedirs(DETAIL_DIR, exist_ok=True)

_cache = Cache("cache") if Cache else None  # 선택
_ext_cache = TTLCache(maxsize=8, ttl=60) if TTLCache else None
_ohlcv_cache = TTLCache(maxsize=256, ttl=10) if TTLCache else None
_translate_cache = TTLCache(maxsize=256, ttl=60 * 60) if TTLCache else None  # 한글화 캐시(1시간)

REQUEST_TIMEOUT = 10
OPENAI_TIMEOUT = 20
TG_TIMEOUT = 10

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]


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


# =========================================================
# ✅ 1.5) 유틸
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_csv_floats(s: str, default: List[float]) -> List[float]:
    try:
        parts = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
        return parts if parts else default
    except Exception:
        return default

def normalize_portions(portions: List[float], max_sum: float = 1.0) -> List[float]:
    clean = [max(0.0, float(x)) for x in portions]
    s = sum(clean) if clean else 0.0
    if s <= 0:
        return clean
    if s > max_sum:
        return [x * (max_sum / s) for x in clean]
    return clean

def tf_to_minutes(tf: str) -> int:
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 5

def df_for_display(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in df.columns:
            df[c] = df[c].apply(lambda x: "" if x is None else str(x))
        return df
    except Exception:
        return pd.DataFrame(rows)

def trend_dir_from_text(t: str) -> str:
    s = str(t or "")
    if "상승" in s:
        return "up"
    if "하락" in s:
        return "down"
    if "횡보" in s or "전환" in s:
        return "flat"
    return "flat"

def side_to_dir(decision: str) -> str:
    return "up" if decision == "buy" else ("down" if decision == "sell" else "flat")

def pnl_emoji(pnl_pct: float) -> str:
    if pnl_pct > 0:
        return "🟢✅"
    if pnl_pct < 0:
        return "🔴❌"
    return "⚪"


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
# ✅ 3) MODE_RULES
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

        # Telegram
        "tg_enable_reports": True,
        "tg_send_entry_reason": False,

        # 채널/그룹 분리
        "tg_split_channel_group": True,
        "tg_enable_periodic_report": True,
        "report_interval_min": 15,
        "tg_enable_hourly_vision": True,
        "tg_hourly_vision_min": 60,

        # 아침 뉴스
        "tg_daily_morning_brief": True,
        "morning_brief_hour": 9,     # KST 09:00
        "morning_brief_minute": 0,
        "morning_brief_top_n": 5,

        # Google Sheets
        "gs_enable": True,
        "gs_daily_tabs": True,
        "gs_daily_summary": True,

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

        # DCA / 스위칭 (기존 기능 유지)
        "use_dca": True, "dca_trigger": -20.0, "dca_max_count": 1, "dca_add_pct": 50.0,
        "use_switching": True, "switch_trigger": -12.0,

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

        # ✅ 지지/저항(SR) 기반 손절/익절
        "use_sr_stop": True,
        "sr_timeframe": "15m",
        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,
        "sr_rr_min": 1.5,

        # ✅ 레짐/스타일 자동 선택
        "auto_style": True,
        "fixed_style": "스캘핑",
        "style_lock_minutes": 30,
        "style_ai_fallback": True,
        "style_ai_min_interval_min": 10,

        # ✅ 추세 필터(스캘핑/스윙 TF) - 기존 유지
        "trend_filter_timeframe": "1h",
        "trend_filter_cache_sec": 60,
        "trend_filter_tf_scalp": "5m",
        "trend_filter_tf_swing": "1h",

        # ✅ 역추세 정책 변경: 금지 X -> 역추세면 스캘핑 강제, 정추세면 스윙
        "countertrend_allowed": True,
        "countertrend_force_scalp": True,
        "countertrend_entry_reduce": 0.65,
        "countertrend_lev_reduce": 1,
        "countertrend_rr_cap": 1.8,  # 역추세면 RR 목표 상한(너무 길게 끌지 않게)

        # ✅ 스윙 분할익절/순환매도
        "swing_partial_tp_enable": True,
        "swing_partial_tp_levels": "0.35,0.60,0.90",
        "swing_partial_tp_sizes": "0.30,0.30,0.40",
        "swing_recycle_enable": False,
        "swing_recycle_trigger_roi": 4.0,
        "swing_recycle_add_pct": 20.0,
        "swing_recycle_cooldown_min": 30,
        "swing_recycle_max_count": 1,

        # ✅ 워치독
        "watchdog_enabled": True,
        "watchdog_timeout_sec": 60,
        "watchdog_check_sec": 15,

        # ✅ 백테스트
        "backtest_default_bars": 800,

        # ✅ 포지션 레짐 전환(진입 후)
        "pos_regime_check_sec": 30,
        "swing_to_scalp_exit_roi": 0.8,      # 스윙 진입 후 레짐이 스캘핑이면 이 정도면 빠르게 익절 청산
        "swing_to_scalp_max_hold_min": 60,   # 레짐 스캘핑인데 오래 끌면 청산 유도
        "scalp_to_swing_upgrade_hold_min": 12,
        "scalp_to_swing_tp_boost": 1.35,
        "scalp_to_swing_sl_boost": 1.10,
    }

def load_settings() -> Dict[str, Any]:
    cfg = default_settings()
    if os.path.exists(SETTINGS_FILE):
        saved = read_json_safe(SETTINGS_FILE, {})
        if isinstance(saved, dict):
            cfg.update(saved)
    if "openai_key" in cfg and not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = cfg["openai_key"]
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
        "events": [],
        "last_report_epoch": 0,
        "last_hourly_vision_epoch": 0,
        "last_morning_brief_date": "",
        "current_style": "스캘핑",
        "style_confidence": 0,
        "style_reason": "",
        "style_since_epoch": 0,
        "style_lock_until": 0,
        "style_last_ai_epoch": 0,
        "last_watchdog_warn": 0,
    }

def load_runtime() -> Dict[str, Any]:
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()
    if rt.get("date") != today_kst_str():
        rt = default_runtime()
    for k, v in default_runtime().items():
        if k not in rt:
            rt[k] = v
    return rt

def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ 6) 매매일지 CSV (기존 유지) + Google Sheets(추가)
# =========================================================
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
            lines.append(f"- {r.get('Coin','?')} {r.get('Side','?')} {float(r.get('PnL_Percent',0)):.2f}% 손실 | 이유: {str(r.get('Reason',''))[:40]}")
        return "\n".join(lines) if lines else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# ---------------------------
# Google Sheets helpers
# ---------------------------
_GS_CACHE: Dict[str, Any] = {}
_GS_LOCK = threading.Lock()

def _gs_get_service_account_info():
    # Streamlit secrets에서 dict 형태 권장
    try:
        if "GSPREAD_SERVICE_ACCOUNT" in st.secrets:
            return dict(st.secrets["GSPREAD_SERVICE_ACCOUNT"])
    except Exception:
        pass
    return None

def gs_enabled(cfg: Dict[str, Any]) -> bool:
    if not cfg.get("gs_enable", True):
        return False
    if gspread is None or Credentials is None:
        return False
    sheet_id = st.secrets.get("GSPREAD_SHEET_ID") or cfg.get("GSPREAD_SHEET_ID")
    return bool(sheet_id)

def gs_client(cfg: Dict[str, Any]):
    if not gs_enabled(cfg):
        return None
    key = "client"
    with _GS_LOCK:
        if key in _GS_CACHE:
            return _GS_CACHE[key]
        info = _gs_get_service_account_info()
        if not info:
            return None
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            gc = gspread.authorize(creds)
            _GS_CACHE[key] = gc
            return gc
        except Exception:
            return None

def gs_open_sheet(cfg: Dict[str, Any]):
    gc = gs_client(cfg)
    if gc is None:
        return None
    sheet_id = st.secrets.get("GSPREAD_SHEET_ID") or cfg.get("GSPREAD_SHEET_ID")
    try:
        return gc.open_by_key(sheet_id)
    except Exception:
        return None

def gs_get_or_create_ws(sh, title: str, rows: int = 2000, cols: int = 20):
    try:
        ws = sh.worksheet(title)
        return ws
    except Exception:
        try:
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
            return ws
        except Exception:
            return None

def gs_apply_formatting(ws):
    # 서식은 optional
    if CellFormat is None or ws is None:
        return
    try:
        set_frozen(ws, rows=1)
        set_column_width(ws, "A", 170)
        set_column_width(ws, "B", 120)
        set_column_width(ws, "C", 80)
        set_column_width(ws, "D", 110)
        set_column_width(ws, "E", 110)
        set_column_width(ws, "F", 110)
        set_column_width(ws, "G", 110)
        set_column_width(ws, "H", 80)
        set_column_width(ws, "I", 300)
    except Exception:
        pass

def gs_ensure_header(ws, headers: List[str]):
    try:
        row1 = ws.row_values(1)
        if row1 and len(row1) >= len(headers):
            return
        ws.update("A1", [headers])
        gs_apply_formatting(ws)
    except Exception:
        pass

def gs_color_row(ws, row_idx: int, pnl_pct: float):
    if CellFormat is None or ws is None:
        return
    try:
        # PnL_Percent 컬럼(G) 기준 색
        if pnl_pct > 0:
            bg = Color(0.15, 0.55, 0.20)
        elif pnl_pct < 0:
            bg = Color(0.65, 0.15, 0.15)
        else:
            bg = Color(0.25, 0.25, 0.25)

        fmt = CellFormat(
            backgroundColor=bg,
            textFormat=TextFormat(bold=False, foregroundColor=Color(1, 1, 1))
        )
        format_cell_range(ws, f"A{row_idx}:I{row_idx}", fmt)
    except Exception:
        pass

def gs_append_trade(cfg: Dict[str, Any], row: Dict[str, Any]):
    if not gs_enabled(cfg):
        return
    sh = gs_open_sheet(cfg)
    if sh is None:
        return

    date = str(row.get("Date", today_kst_str()))
    tab = f"trades_{date}" if cfg.get("gs_daily_tabs", True) else "trades_all"
    ws = gs_get_or_create_ws(sh, tab, rows=5000, cols=20)
    if ws is None:
        return

    headers = ["Time","Coin","Side","Entry","Exit","PnL_USDT","PnL_Percent","Emoji","Reason","OneLine","Review","TradeID","Style","Mode"]
    gs_ensure_header(ws, headers)

    values = [
        str(row.get("Time","")),
        str(row.get("Coin","")),
        str(row.get("Side","")),
        float(row.get("Entry", 0) or 0),
        float(row.get("Exit", 0) or 0),
        float(row.get("PnL_USDT", 0) or 0),
        float(row.get("PnL_Percent", 0) or 0),
        str(row.get("Emoji","")),
        str(row.get("Reason","")),
        str(row.get("OneLine","")),
        str(row.get("Review","")),
        str(row.get("TradeID","")),
        str(row.get("Style","")),
        str(row.get("Mode","")),
    ]
    try:
        ws.append_row(values, value_input_option="USER_ENTERED")
        r = len(ws.get_all_values())
        gs_color_row(ws, r, float(row.get("PnL_Percent", 0) or 0))
    except Exception:
        pass

    if cfg.get("gs_daily_summary", True):
        try:
            gs_upsert_daily_summary(cfg, sh, date)
        except Exception:
            pass

def gs_upsert_daily_summary(cfg: Dict[str, Any], sh, date: str):
    ws = gs_get_or_create_ws(sh, "daily_summary", rows=2000, cols=12)
    if ws is None:
        return
    headers = ["Date","Trades","WinRate%","TotalPnL_USDT","AvgPnL%","BestPnL%","WorstPnL%","Note"]
    gs_ensure_header(ws, headers)

    # CSV 기반 요약 (가장 단순/안전)
    df = read_trade_log()
    if df.empty:
        return
    df["Date"] = df["Time"].astype(str).str[:10]
    ddf = df[df["Date"] == date]
    if ddf.empty:
        return

    trades = len(ddf)
    win = (ddf["PnL_Percent"].astype(float) > 0).sum()
    win_rate = (win / trades) * 100.0 if trades else 0.0
    total_pnl = float(ddf["PnL_USDT"].astype(float).sum())
    avg_pct = float(ddf["PnL_Percent"].astype(float).mean())
    best = float(ddf["PnL_Percent"].astype(float).max())
    worst = float(ddf["PnL_Percent"].astype(float).min())

    # upsert row by date
    try:
        all_vals = ws.get_all_values()
        idx = None
        for i in range(2, len(all_vals)+1):
            if len(all_vals[i-1]) > 0 and all_vals[i-1][0] == date:
                idx = i
                break
        row_vals = [date, trades, round(win_rate,2), round(total_pnl,2), round(avg_pct,2), round(best,2), round(worst,2), ""]
        if idx is None:
            ws.append_row(row_vals, value_input_option="USER_ENTERED")
        else:
            ws.update(f"A{idx}:H{idx}", [row_vals])
    except Exception:
        pass

def log_trade(
    coin: str, side: str, entry_price: float, exit_price: float,
    pnl_amount: float, pnl_percent: float, reason: str,
    one_line: str = "", review: str = "", trade_id: str = "",
    style: str = "", mode: str = ""
) -> None:
    # CSV 기존 유지
    try:
        row = pd.DataFrame([{
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
            "Style": style,
            "Mode": mode,
        }])
        if not os.path.exists(LOG_FILE):
            row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    # Google Sheets 추가
    try:
        cfg = load_settings()
        gs_append_trade(cfg, {
            "Date": today_kst_str(),
            "Time": now_kst_str(),
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Emoji": pnl_emoji(float(pnl_percent)),
            "Reason": reason,
            "OneLine": one_line,
            "Review": review,
            "TradeID": trade_id,
            "Style": style,
            "Mode": mode,
        })
    except Exception:
        pass


# =========================================================
# ✅ 7) Secrets (Bitget / Telegram / OpenAI)
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")

# legacy fallback
tg_chat_legacy = st.secrets.get("TG_CHAT_ID")

# split ids (recommended)
tg_channel_id = None
tg_group_id = None
try:
    if "TG" in st.secrets:
        tg_channel_id = st.secrets["TG"].get("TG_CHANNEL_ID")
        tg_group_id = st.secrets["TG"].get("TG_GROUP_ID")
except Exception:
    pass

# ultimate fallback
TG_CHANNEL = tg_channel_id or tg_chat_legacy
TG_GROUP = tg_group_id or tg_chat_legacy or TG_CHANNEL

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. Secrets에 API_KEY/API_SECRET/API_PASSWORD 설정하세요.")
    st.stop()

_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}
_OPENAI_CLIENT_LOCK = threading.Lock()

def get_openai_client(cfg: Dict[str, Any]) -> Optional[OpenAI]:
    key = st.secrets.get("OPENAI_API_KEY") or cfg.get("openai_api_key", "")
    if not key:
        return None
    with _OPENAI_CLIENT_LOCK:
        if key in _OPENAI_CLIENT_CACHE:
            return _OPENAI_CLIENT_CACHE[key]
        try:
            c = OpenAI(api_key=key)
            _OPENAI_CLIENT_CACHE[key] = c
            return c
        except Exception:
            return None


# =========================================================
# ✅ 8) 거래소 연결
# =========================================================
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_password,
            "enableRateLimit": True,
            "timeout": 10000,
            "options": {"defaultType": "swap"},
        })
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
# ✅ 9) 네트워크 안전 요청 (timeout + retry)
# =========================================================
def _safe_request(method: str, url: str, timeout: int = REQUEST_TIMEOUT, **kwargs):
    if retry is not None:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
        def _do():
            r = requests.request(method, url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        try:
            return _do()
        except Exception:
            return None
    else:
        last = None
        for i in range(3):
            try:
                r = requests.request(method, url, timeout=timeout, **kwargs)
                r.raise_for_status()
                return r
            except Exception as e:
                last = e
                if i < 2:
                    time.sleep(0.6 * (2 ** i) + random.random() * 0.2)
        return None

def _safe_get_json(url: str, timeout: int = 10):
    try:
        r = _safe_request("GET", url, timeout=timeout)
        if not r:
            return None
        return r.json()
    except Exception:
        return None


# =========================================================
# ✅ 9.1) OHLCV 캐시
# =========================================================
def fetch_ohlcv_cached(ex, sym: str, tf: str, limit: int = 220, cache_sec: int = 8):
    key = f"{sym}|{tf}|{limit}"
    now = time.time()
    if _ohlcv_cache is not None and key in _ohlcv_cache:
        item = _ohlcv_cache.get(key, {})
        if now - float(item.get("ts", 0)) <= cache_sec:
            return item.get("data", None)
    try:
        data = ex.fetch_ohlcv(sym, tf, limit=limit)
        if _ohlcv_cache is not None:
            _ohlcv_cache[key] = {"ts": now, "data": data}
        return data
    except Exception:
        return None


# =========================================================
# ✅ 9.2) Bitget 헬퍼
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


# =========================================================
# ✅ 9.3) (핵심) 단기추세/장기추세 계산
# =========================================================
_TREND_CACHE: Dict[str, Dict[str, Any]] = {}

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
    key = f"{sym}|{tf}|{fast}|{slow}"
    now = time.time()
    if key in _TREND_CACHE:
        if (now - float(_TREND_CACHE[key].get("ts", 0))) < cache_sec:
            return str(_TREND_CACHE[key].get("trend", "중립"))
    try:
        ohlcv = fetch_ohlcv_cached(ex, sym, tf, limit=max(220, slow + 50), cache_sec=cache_sec)
        if not ohlcv:
            return "중립"
        hdf = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
        trend = compute_ma_trend_from_df(hdf, fast=fast, slow=slow)
        _TREND_CACHE[key] = {"ts": now, "trend": trend}
        return trend
    except Exception:
        return "중립"


# =========================================================
# ✅ 9.5) SR(지지/저항) 기반 SL/TP 계산 (기존 유지)
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

def sr_stop_take(entry_price: float, side: str, htf_df: pd.DataFrame,
                 atr_period: int = 14, pivot_order: int = 6,
                 buffer_atr_mult: float = 0.25, rr_min: float = 1.5) -> Optional[Dict[str, Any]]:
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

    return {"sl_price": float(sl_price), "tp_price": float(tp_price), "atr": float(atr),
            "supports": supports, "resistances": resistances}


# =========================================================
# ✅ 10) TradingView 다크모드 차트
# =========================================================
def tv_symbol_from_ccxt(sym: str) -> str:
    base = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    return f"BITGET:{base}{quote}.P"

def render_tradingview(symbol_ccxt: str, interval="5", height=560) -> None:
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
# ✅ 11) 지표 계산 (10종 + 상태요약 + “눌림목 해소” 감지)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.Series]]:
    status: Dict[str, Any] = {}
    if df is None or df.empty or len(df) < 120:
        return df, status, None
    if ta is None:
        status["_ERROR"] = "ta 모듈 없음(requirements.txt에 ta 추가 필요)"
        return df, status, None

    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))
    bb_period = int(cfg.get("bb_period", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    ma_fast = int(cfg.get("ma_fast", 7))
    ma_slow = int(cfg.get("ma_slow", 99))
    stoch_k = int(cfg.get("stoch_k", 14))
    vol_mul = float(cfg.get("vol_mul", 2.0))

    close = df["close"]; high = df["high"]; low = df["low"]; vol = df["vol"]

    if cfg.get("use_rsi", True):
        df["RSI"] = ta.momentum.rsi(close, window=rsi_period)

    if cfg.get("use_bb", True):
        bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()

    if cfg.get("use_ma", True):
        df["MA_fast"] = ta.trend.sma_indicator(close, window=ma_fast)
        df["MA_slow"] = ta.trend.sma_indicator(close, window=ma_slow)

    if cfg.get("use_macd", True):
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

    if cfg.get("use_stoch", True):
        df["STO_K"] = ta.momentum.stoch(high, low, close, window=stoch_k, smooth_window=3)
        df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=stoch_k, smooth_window=3)

    if cfg.get("use_cci", True):
        df["CCI"] = ta.trend.cci(high, low, close, window=20)

    if cfg.get("use_mfi", True):
        df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)

    if cfg.get("use_willr", True):
        df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

    if cfg.get("use_adx", True):
        df["ADX"] = ta.trend.adx(high, low, close, window=14)

    if cfg.get("use_vol", True):
        df["VOL_MA"] = vol.rolling(20).mean()
        df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * vol_mul)).astype(int)

    if pta is not None:
        try:
            df["ATR_ref"] = pta.atr(df["high"], df["low"], df["close"], length=14)
        except Exception:
            pass

    df = df.dropna()
    if df.empty or len(df) < 5:
        return df, status, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    used = []

    # RSI
    if cfg.get("use_rsi", True):
        used.append("RSI")
        rsi_now = float(last.get("RSI", 50))
        if rsi_now < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_now:.1f})"
        elif rsi_now > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_now:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_now:.1f})"

    # BB
    if cfg.get("use_bb", True):
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
        if last["MA_fast"] > last["MA_slow"] and last["close"] > last["MA_slow"]:
            trend = "상승추세"
        elif last["MA_fast"] < last["MA_slow"] and last["close"] < last["MA_slow"]:
            trend = "하락추세"
        else:
            trend = "횡보/전환"
        status["추세"] = f"📈 {trend}"

    # MACD
    if cfg.get("use_macd", True):
        used.append("MACD")
        status["MACD"] = "📈 상승(골든)" if last["MACD"] > last["MACD_signal"] else "📉 하락(데드)"

    # ADX
    if cfg.get("use_adx", True):
        used.append("ADX(추세강도)")
        adx = float(last.get("ADX", 0))
        status["ADX"] = "🔥 추세 강함" if adx >= 25 else "💤 추세 약함"

    # Volume
    if cfg.get("use_vol", True):
        used.append("거래량")
        status["거래량"] = "🔥 거래량 급증" if int(last.get("VOL_SPIKE", 0)) == 1 else "⚪ 보통"

    # RSI 해소
    rsi_prev = float(prev.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0
    rsi_now = float(last.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0
    rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
    rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

    adx_now = float(last.get("ADX", 0)) if cfg.get("use_adx", True) else 0.0
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_now >= 18)

    status["_used_indicators"] = used
    status["_rsi_resolve_long"] = bool(rsi_resolve_long)
    status["_rsi_resolve_short"] = bool(rsi_resolve_short)
    status["_pullback_candidate"] = bool(pullback_candidate)

    return df, status, last


# =========================================================
# ✅ 12) 외부 시황 통합(거시/심리/레짐/뉴스) - 캐시 포함
# =========================================================
def fetch_fear_greed():
    data = _safe_get_json("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
    if not data or "data" not in data or not data["data"]:
        return None
    d0 = data["data"][0]
    try:
        return {"value": int(d0.get("value", 0)),
                "classification": str(d0.get("value_classification", "")),
                "timestamp": str(d0.get("timestamp", ""))}
    except Exception:
        return None

def fetch_coingecko_global():
    data = _safe_get_json("https://api.coingecko.com/api/v3/global", timeout=10)
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

def fetch_upcoming_high_impact_events(within_minutes: int = 30, limit: int = 120):
    data = _safe_get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
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
                out.append({
                    "time_kst": dt.strftime("%m-%d %H:%M"),
                    "title": str(x.get("title","")),
                    "country": str(x.get("country","")),
                    "impact": "매우 중요",
                })
        except Exception:
            continue
    return out

def fetch_news_headlines_rss(max_items: int = 12):
    if feedparser is None:
        return []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    items = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in (d.entries or [])[:max_items]:
                title = str(getattr(e, "title", "")).strip()
                if title:
                    items.append(title)
        except Exception:
            continue
    uniq, seen = [], set()
    for t in items:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq[:max_items]

def _openai_chat_json(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    if client is None:
        return None
    if retry is not None:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
        def _do():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=OPENAI_TIMEOUT
            )
        try:
            return _do()
        except Exception:
            return None
    else:
        for i in range(3):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    timeout=OPENAI_TIMEOUT
                )
            except Exception:
                if i < 2:
                    time.sleep(0.7 * (2 ** i))
        return None

def translate_lines_to_korean(cfg: Dict[str, Any], lines: List[str], purpose: str = "general") -> List[str]:
    # 가능한 경우 OpenAI로 한글화(캐시)
    if not lines:
        return []
    if not cfg.get("ai_easy_korean", True):
        return lines
    key = "tr:" + purpose + ":" + "|".join(lines)[:400]
    if _translate_cache is not None and key in _translate_cache:
        return _translate_cache[key]

    client = get_openai_client(cfg)
    if client is None:
        return lines

    sys = "너는 영어 문장을 아주 자연스러운 한국어로 번역한다. 과장 없이 짧고 직관적으로. 반드시 JSON만 출력."
    user = {
        "purpose": purpose,
        "lines": lines[:20]
    }
    try:
        resp = _openai_chat_json(client, "gpt-4o-mini", [
            {"role":"system","content":sys},
            {"role":"user","content":json.dumps(user, ensure_ascii=False)}
        ], temperature=0.2)
        if not resp:
            return lines
        out = json.loads(resp.choices[0].message.content)
        ko = out.get("korean_lines", None)
        if isinstance(ko, list) and ko:
            ko = [str(x) for x in ko][:len(lines)]
            if _translate_cache is not None:
                _translate_cache[key] = ko
            return ko
        return lines
    except Exception:
        return lines

def build_external_context(cfg: dict) -> Dict[str, Any]:
    if not cfg.get("use_external_context", True):
        return {"enabled": False}

    ttl = int(cfg.get("external_refresh_sec", 60))
    if _ext_cache is not None and "ext_raw" in _ext_cache:
        return _ext_cache["ext_raw"]

    if _cache is not None:
        cached = _cache.get("ext_context_raw")
        if cached and isinstance(cached, dict):
            return cached

    blackout = int(cfg.get("macro_blackout_minutes", 30))
    high_events = fetch_upcoming_high_impact_events(within_minutes=blackout)

    fg = fetch_fear_greed()
    cg = fetch_coingecko_global()

    headlines = []
    if cfg.get("news_enable", True):
        headlines = fetch_news_headlines_rss(max_items=int(cfg.get("news_max_headlines", 12)))

    ext = {
        "enabled": True,
        "blackout_minutes": blackout,
        "high_impact_events_soon": high_events,
        "fear_greed": fg,
        "global": cg,
        "headlines": headlines,
        "asof_kst": now_kst_str()
    }

    if _ext_cache is not None:
        _ext_cache["ext_raw"] = ext
    if _cache is not None:
        try:
            _cache.set("ext_context_raw", ext, expire=ttl)
        except Exception:
            pass

    return ext

def build_external_context_korean(cfg: dict) -> Dict[str, Any]:
    # raw를 기반으로 "표시용 한글"만 별도로 만든다 (과도한 호출 방지: 캐시)
    key = "ext_ko"
    if _ext_cache is not None and key in _ext_cache:
        return _ext_cache[key]

    ext = build_external_context(cfg)
    if not ext.get("enabled"):
        return ext

    ko = dict(ext)
    # 이벤트 타이틀 한글화
    evs = ko.get("high_impact_events_soon") or []
    titles = [str(e.get("title","")) for e in evs[:10]]
    if titles:
        ko_titles = translate_lines_to_korean(cfg, titles, purpose="economic_calendar")
        for i in range(min(len(evs), len(ko_titles))):
            evs[i]["title_ko"] = ko_titles[i]
    ko["high_impact_events_soon"] = evs

    # 헤드라인 한글화
    hds = ko.get("headlines") or []
    if hds:
        ko_hds = translate_lines_to_korean(cfg, hds[:12], purpose="crypto_headlines")
        ko["headlines_ko"] = ko_hds
    else:
        ko["headlines_ko"] = []

    if _ext_cache is not None:
        _ext_cache[key] = ko
    return ko

def assess_external_risk(ext: Dict[str, Any]) -> Tuple[str, int, str]:
    score = 0
    notes = []
    try:
        events = (ext or {}).get("high_impact_events_soon") or []
        if events:
            score += 2
            notes.append("중요 이벤트 임박")
    except Exception:
        pass
    try:
        fg = (ext or {}).get("fear_greed") or {}
        v = int(fg.get("value", -1)) if fg else -1
        if 0 <= v <= 20:
            score += 1
            notes.append("극공포")
        elif v >= 80:
            score += 1
            notes.append("극탐욕")
    except Exception:
        pass
    try:
        g = (ext or {}).get("global") or {}
        mcap_change = float(g.get("mcap_change_24h_pct", 0.0))
        if abs(mcap_change) >= 8.0:
            score += 1
            notes.append("시총 급변")
    except Exception:
        pass

    if score >= 3:
        return "high", score, " / ".join(notes)
    if score >= 1:
        return "medium", score, " / ".join(notes)
    return "low", score, "정상"


# =========================================================
# ✅ 13) 리스크/AI + 외부시황 + 스타일
# =========================================================
def _atr_price_pct(df: pd.DataFrame, window: int = 14) -> float:
    try:
        if ta is None or df is None or df.empty or len(df) < window + 5:
            return 0.0
        atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=window)
        v = float(atr.iloc[-1])
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

def _risk_guardrail(out: Dict[str, Any], df: pd.DataFrame, decision: str, mode: str, external: Dict[str, Any]) -> Dict[str, Any]:
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

    # 외부시황: 공포탐욕이 극공포면 SL 여유 약간 추가
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

    rr_min = _rr_min_by_mode(mode)
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


def ai_decide_trade(df: pd.DataFrame, status: Dict[str, Any], symbol: str, mode: str, cfg: Dict[str, Any], trade_style: str = "스캘핑") -> Dict[str, Any]:
    client = get_openai_client(cfg)
    if client is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "OpenAI 키 없음", "used_indicators": status.get("_used_indicators", [])}
    if df is None or df.empty or status is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "데이터 부족", "used_indicators": status.get("_used_indicators", [])}

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    past_mistakes = get_past_mistakes_text(5)

    external = build_external_context(cfg)

    features = {
        "symbol": symbol,
        "mode": mode,
        "trade_style": trade_style,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)) if "RSI" in df.columns else None,
        "rsi_now": float(last.get("RSI", 50)) if "RSI" in df.columns else None,
        "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
        "trend_short": status.get("추세", ""),
        "bb": status.get("BB", ""),
        "macd": status.get("MACD", ""),
        "vol": status.get("거래량", ""),
        "rsi_resolve_long": bool(status.get("_rsi_resolve_long", False)),
        "rsi_resolve_short": bool(status.get("_rsi_resolve_short", False)),
        "pullback_candidate": bool(status.get("_pullback_candidate", False)),
        "atr_price_pct": _atr_price_pct(df, 14),
        "external": external,
    }

    fg_txt = ""
    try:
        fg = (external or {}).get("fear_greed") or {}
        if fg:
            fg_txt = f"- 공포탐욕지수: {int(fg.get('value', 0))} / {fg.get('classification','')}"
    except Exception:
        fg_txt = ""

    ev_txt = ""
    try:
        evs = (external or {}).get("high_impact_events_soon") or []
        if evs:
            ev_txt = "- 중요 이벤트(임박): " + " | ".join([f"{e.get('country','')} {e.get('title','')}" for e in evs[:3]])
    except Exception:
        ev_txt = ""

    sys = f"""
너는 '워뇨띠 스타일(눌림목/해소 타이밍) + 손익비' 기반의 자동매매 트레이더 AI다.
현재 매매 스타일은 "{trade_style}"이다.

[과거 실수(요약)]
{past_mistakes}

[외부 시황(참고)]
{fg_txt}
{ev_txt}

[핵심 룰]
1) RSI 과매도/과매수 '상태'에 즉시 진입하지 말고, '해소되는 시점'에서만 진입 후보.
2) 상승추세에서는 롱 우선, 하락추세에서는 숏 우선. (역추세는 '짧게' 운영)
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
        resp = _openai_chat_json(client, "gpt-4o", [{"role": "system", "content": sys},
                                                    {"role": "user", "content": user}],
                                 temperature=0.2)
        if not resp:
            return {"decision": "hold", "confidence": 0, "reason_easy": "AI 응답 없음", "used_indicators": status.get("_used_indicators", [])}

        out = json.loads(resp.choices[0].message.content)

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

        if out["decision"] in ["buy", "sell"]:
            out = _risk_guardrail(out, df, out["decision"], mode, external)

        out["external_used"] = {
            "fear_greed": (external or {}).get("fear_greed"),
            "high_impact_events_soon": ((external or {}).get("high_impact_events_soon") or [])[:3],
            "asof_kst": (external or {}).get("asof_kst", ""),
        }
        return out

    except Exception as e:
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": status.get("_used_indicators", [])}


# =========================================================
# ✅ 14) AI 회고(후기)
# =========================================================
def ai_write_review(symbol: str, side: str, pnl_percent: float, reason: str, cfg: Dict[str, Any]) -> Tuple[str, str]:
    client = get_openai_client(cfg)
    if client is None:
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "OpenAI 키 없음 - 후기 자동작성 불가")

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
        resp = _openai_chat_json(client, "gpt-4o-mini", [{"role": "system", "content": sys},
                                                         {"role": "user", "content": user}],
                                 temperature=0.3)
        if not resp:
            one = "익절" if pnl_percent >= 0 else "손절"
            return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")
        out = json.loads(resp.choices[0].message.content)
        return str(out.get("one_line", ""))[:120], str(out.get("review", ""))[:800]
    except Exception:
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")


# =========================================================
# ✅ 15) 스타일 자동 선택 (스캘핑 vs 스윙) + 심볼별 추천
# =========================================================
def _adx_last(df: pd.DataFrame) -> float:
    if ta is None or df is None or df.empty or len(df) < 50:
        return 0.0
    try:
        adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        return float(adx.iloc[-1])
    except Exception:
        return 0.0

def decide_style_rule_for_symbol(ex, sym: str, cfg: Dict[str, Any]) -> Tuple[str, int, str, bool]:
    short_tf = cfg.get("timeframe", "5m")
    long_tf = cfg.get("trend_filter_tf_swing", "1h")

    ohlcv_s = fetch_ohlcv_cached(ex, sym, short_tf, limit=220, cache_sec=8)
    ohlcv_l = fetch_ohlcv_cached(ex, sym, long_tf, limit=220, cache_sec=30)
    if not ohlcv_s or not ohlcv_l:
        return "스캘핑", 50, "데이터 부족", True

    ds = pd.DataFrame(ohlcv_s, columns=["time","open","high","low","close","vol"])
    dl = pd.DataFrame(ohlcv_l, columns=["time","open","high","low","close","vol"])

    trend_l = compute_ma_trend_from_df(dl, fast=int(cfg.get("ma_fast", 7)), slow=int(cfg.get("ma_slow", 99)))
    adx_l = _adx_last(dl)
    atr_s = _atr_price_pct(ds, 14)
    atr_l = _atr_price_pct(dl, 14)

    swing_score = 0
    scalp_score = 0
    reasons = []

    if trend_l in ["상승추세", "하락추세"] and adx_l >= 22:
        swing_score += 2
        reasons.append("1h 강한 추세")
    if trend_l == "횡보/전환":
        scalp_score += 1
        reasons.append("1h 횡보")

    if adx_l < 18:
        scalp_score += 1
        reasons.append("추세 약함")

    if atr_s >= 0.45:
        scalp_score += 1
        reasons.append("단기 변동성 높음")

    if atr_l >= 0.7:
        swing_score += 1
        reasons.append("장기 변동성 충분")

    diff = swing_score - scalp_score
    ambiguous = abs(diff) < 2

    if diff >= 2:
        style = "스윙"
        conf = min(95, 70 + diff * 8)
    elif diff <= -2:
        style = "스캘핑"
        conf = min(95, 70 + abs(diff) * 8)
    else:
        style = "스캘핑" if atr_s >= atr_l else "스윙"
        conf = 55

    reason = " / ".join(reasons) if reasons else "룰 기반"
    return style, int(conf), reason, ambiguous

def get_trade_style(ex, cfg: Dict[str, Any], rt: Dict[str, Any]) -> Tuple[str, int, str]:
    if not cfg.get("auto_style", True):
        style = cfg.get("fixed_style", "스캘핑")
        return style, 100, "고정"

    now = time.time()
    cur = rt.get("current_style", "스캘핑")
    lock_until = float(rt.get("style_lock_until", 0))

    if now < lock_until:
        return cur, int(rt.get("style_confidence", 0)), rt.get("style_reason", "잠금 유지")

    # BTC 기준 레짐 판단
    style, conf, reason, _amb = decide_style_rule_for_symbol(ex, TARGET_COINS[0], cfg)

    if style != cur:
        rt["current_style"] = style
        rt["style_confidence"] = conf
        rt["style_reason"] = reason
        rt["style_since_epoch"] = now
        rt["style_lock_until"] = now + int(cfg.get("style_lock_minutes", 30)) * 60
    else:
        rt["style_confidence"] = conf
        rt["style_reason"] = reason

    save_runtime(rt)
    return style, conf, reason

def get_trend_filter_tf(cfg: Dict[str, Any], style: str) -> str:
    if style == "스윙":
        return str(cfg.get("trend_filter_tf_swing", cfg.get("trend_filter_timeframe", "1h")))
    return str(cfg.get("trend_filter_tf_scalp", cfg.get("timeframe", "5m")))

def determine_entry_style(cfg: Dict[str, Any], short_trend_txt: str, htf_trend: str, decision: str) -> Tuple[str, str]:
    """
    - 정추세(단기+장기 둘 다 방향 일치) => 스윙
    - 그 외(역추세/불일치) => 스캘핑
    """
    sd = side_to_dir(decision)
    st = trend_dir_from_text(short_trend_txt)
    ht = trend_dir_from_text(htf_trend)
    if st == sd and ht == sd:
        return "스윙", "단기+장기 추세 일치 → 스윙"
    return "스캘핑", "추세 불일치/역추세 → 스캘핑"

def apply_countertrend_adjustments(cfg: Dict[str, Any], ai: Dict[str, Any], entry_style: str, htf_trend: str, decision: str) -> Dict[str, Any]:
    """
    역추세면: entry_pct 감산 + leverage 감산 + RR 상한
    """
    if not cfg.get("countertrend_allowed", True):
        return ai

    if entry_style != "스캘핑":
        return ai

    # 역추세 판단: 장기추세가 매수/매도 방향과 다르면
    ht = trend_dir_from_text(htf_trend)
    sd = side_to_dir(decision)
    is_counter = (ht in ["up","down"]) and (ht != sd)
    if not is_counter:
        return ai

    ai["entry_pct"] = float(ai.get("entry_pct", 0)) * float(cfg.get("countertrend_entry_reduce", 0.65))
    ai["leverage"] = max(1, int(ai.get("leverage", 1)) - int(cfg.get("countertrend_lev_reduce", 1)))

    rr_cap = float(cfg.get("countertrend_rr_cap", 1.8))
    try:
        if float(ai.get("rr", 0)) > rr_cap:
            # tp를 rr_cap에 맞춰 제한
            sl = float(ai.get("sl_pct", 1.0))
            ai["tp_pct"] = sl * rr_cap
            ai["rr"] = rr_cap
    except Exception:
        pass

    ai["countertrend_note"] = "역추세(장기추세 반대) → 스캘핑/보수 세팅"
    return ai


# =========================================================
# ✅ 16) 외부시황 리스크 조정 (기존 유지)
# =========================================================
def apply_external_risk_adjustment(ai: Dict[str, Any], ext: Dict[str, Any], cfg: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    level, score, note = assess_external_risk(ext)
    entry_pct = float(ai.get("entry_pct", rule["entry_pct_min"]))
    lev = int(ai.get("leverage", rule["lev_min"]))
    min_conf = int(rule["min_conf"])

    if level == "high":
        entry_pct *= float(cfg.get("external_risk_reduce_entry_pct_high", 0.6)) if "external_risk_reduce_entry_pct_high" in cfg else 0.6
        lev = max(rule["lev_min"], lev - int(cfg.get("external_risk_reduce_lev_high", 1)) if "external_risk_reduce_lev_high" in cfg else max(1, lev-1))
        min_conf += int(cfg.get("external_risk_raise_conf_high", 8)) if "external_risk_raise_conf_high" in cfg else 8
    elif level == "medium":
        entry_pct *= float(cfg.get("external_risk_reduce_entry_pct_med", 0.8)) if "external_risk_reduce_entry_pct_med" in cfg else 0.8
        lev = max(rule["lev_min"], lev - int(cfg.get("external_risk_reduce_lev_med", 0)) if "external_risk_reduce_lev_med" in cfg else lev)
        min_conf += int(cfg.get("external_risk_raise_conf_med", 4)) if "external_risk_raise_conf_med" in cfg else 4

    entry_pct = max(0.5, min(entry_pct, rule["entry_pct_max"]))

    ai["entry_pct"] = entry_pct
    ai["leverage"] = lev
    ai["min_conf_adj"] = min_conf
    ai["external_risk_note"] = note
    ai["external_risk_level"] = level
    return ai


# =========================================================
# ✅ 17) 모니터 상태(하트비트)
# =========================================================
def monitor_init():
    mon = read_json_safe(MONITOR_FILE, {"coins": {}}) or {"coins": {}}
    mon["_boot_time_kst"] = now_kst_str()
    mon["_last_write"] = 0
    write_json_atomic(MONITOR_FILE, mon)
    return mon

def monitor_write_throttled(mon: Dict[str, Any], min_interval_sec: float = 1.0):
    lastw = float(mon.get("_last_write", 0))
    if time.time() - lastw >= min_interval_sec:
        write_json_atomic(MONITOR_FILE, mon)
        mon["_last_write"] = time.time()


# =========================================================
# ✅ 18) 텔레그램 유틸 (채널/그룹 분리)
# =========================================================
def _tg_post(url: str, data: Dict[str, Any]):
    r = _safe_request("POST", url, timeout=TG_TIMEOUT, data=data)
    return r

def tg_send(text: str, target: str = "channel"):
    """
    target:
      - "channel": 채널에 전송(진입/손절/익절/리포트/아침뉴스/시간당시야)
      - "group": 그룹에 전송(버튼/조회/상태)
      - "both": 둘 다
    """
    if not tg_token:
        return
    chat_id = None
    ids = []
    if target == "channel":
        chat_id = TG_CHANNEL
        ids = [TG_CHANNEL]
    elif target == "group":
        chat_id = TG_GROUP
        ids = [TG_GROUP]
    else:
        ids = [TG_CHANNEL, TG_GROUP]

    for cid in ids:
        if not cid:
            continue
        try:
            _tg_post(f"https://api.telegram.org/bot{tg_token}/sendMessage", {"chat_id": cid, "text": text})
        except Exception:
            pass

def tg_send_menu():
    if not tg_token or not TG_GROUP:
        return
    kb = {
        "inline_keyboard": [
            [{"text": "📡 상태", "callback_data": "status"},
             {"text": "👁️ AI시야", "callback_data": "vision"}],
            [{"text": "📊 포지션", "callback_data": "position"},
             {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "📜 일지(최근)", "callback_data": "log"},
             {"text": "🧾 일지상세", "callback_data": "log_detail_help"}],
            [{"text": "🛑 전량청산", "callback_data": "close_all"}]
        ]
    }
    try:
        _tg_post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            {"chat_id": TG_GROUP, "text": "✅ /menu\n(일지상세: '일지상세 <ID>')", "reply_markup": json.dumps(kb, ensure_ascii=False)},
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
# ✅ 19) 이벤트 기록 + 리포트
# =========================================================
def record_event(rt: Dict[str, Any], ev_type: str, symbol: str = "", detail: str = ""):
    try:
        ev = {
            "time_kst": now_kst_str(),
            "epoch": time.time(),
            "type": ev_type,
            "symbol": symbol,
            "detail": detail
        }
        events = rt.setdefault("events", [])
        events.append(ev)
        if len(events) > 250:
            rt["events"] = events[-250:]
    except Exception:
        pass

def get_recent_events(rt: Dict[str, Any], minutes: int = 15) -> List[Dict[str, Any]]:
    cut = time.time() - minutes * 60
    evs = rt.get("events", []) or []
    return [e for e in evs if float(e.get("epoch", 0)) >= cut]

def _format_positions_summary(positions: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for p in positions:
        try:
            sym = p.get("symbol", "")
            side = position_side_normalize(p)
            roi = float(position_roi_percent(p))
            lev = p.get("leverage", "?")
            upnl = float(p.get("unrealizedPnl") or 0.0)
            emo = "🟢" if roi > 0 else ("🔴" if roi < 0 else "⚪")
            lines.append(f"{emo} {sym} {('롱' if side=='long' else '숏')} x{lev} | ROI {roi:.2f}% | {upnl:.2f} USDT")
        except Exception:
            continue
    return lines

def send_periodic_report(ex, cfg: Dict[str, Any], rt: Dict[str, Any], mon: Dict[str, Any]):
    if not cfg.get("tg_enable_periodic_report", True):
        return
    interval = max(5, int(cfg.get("report_interval_min", 15)))
    last = float(rt.get("last_report_epoch", 0))
    if time.time() - last < interval * 60:
        return

    free, total = safe_fetch_balance(ex)
    positions = safe_fetch_positions(ex, TARGET_COINS)
    act = [p for p in positions if float(p.get("contracts") or 0) > 0]
    pos_lines = _format_positions_summary(act) if act else ["⚪ 없음(관망)"]

    events = get_recent_events(rt, interval)
    ev_lines = []
    for e in events[-8:]:
        ev_lines.append(f"- {e.get('time_kst','')} | {e.get('type','')} {e.get('symbol','')} | {str(e.get('detail',''))[:60]}")
    if not ev_lines:
        ev_lines = ["- 없음"]

    ext_ko = build_external_context_korean(cfg)
    fg = (ext_ko.get("fear_greed") or {})
    fg_txt = f"{fg.get('value','-')} ({fg.get('classification','-')})" if fg else "-"
    evs = ext_ko.get("high_impact_events_soon") or []
    hd = ext_ko.get("headlines_ko") or ext_ko.get("headlines") or []
    hd_txt = "; ".join(hd[:3]) if hd else "-"

    msg = [
        "🕒 정기 리포트",
        f"✅ 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'} | 모드: {cfg.get('trade_mode','-')}",
        f"🎛️ 레짐(전략): {mon.get('trade_style','-')} ({mon.get('style_confidence','-')}%)",
        f"💰 잔고: {total:.2f} USDT (가용 {free:.2f})",
        "📌 포지션:",
        *[f"- {x}" for x in pos_lines],
        f"🧠 최근 {interval}분 이벤트:",
        *ev_lines,
        f"🌍 공포탐욕: {fg_txt} | 중요이벤트: {len(evs)}건",
        f"📰 헤드라인: {hd_txt}",
        f"❤️ 하트비트: {mon.get('last_heartbeat_kst','-')}",
    ]

    tg_send("\n".join(msg), target="channel")
    rt["last_report_epoch"] = time.time()
    save_runtime(rt)
    mon["last_report_kst"] = now_kst_str()

def send_hourly_vision_report(cfg: Dict[str, Any], rt: Dict[str, Any], mon: Dict[str, Any]):
    if not cfg.get("tg_enable_hourly_vision", True):
        return
    interval_min = max(10, int(cfg.get("tg_hourly_vision_min", 60)))
    last = float(rt.get("last_hourly_vision_epoch", 0))
    if time.time() - last < interval_min * 60:
        return

    coins = (mon.get("coins", {}) or {})
    lines = [
        "👁️ AI 시야 (시간당 요약)",
        f"🎛️ 레짐: {mon.get('trade_style','-')} ({mon.get('style_confidence','-')}%)",
        f"🧭 필터TF: {mon.get('trend_filter_tf','-')}",
        f"✅ 자동매매: {'ON' if mon.get('auto_trade') else 'OFF'} | 모드: {mon.get('trade_mode','-')}",
        "—",
    ]
    shown = 0
    for sym, cs in list(coins.items()):
        if shown >= 10:
            break
        shown += 1
        dec = str(cs.get("ai_decision","-")).upper()
        conf = cs.get("ai_confidence","-")
        sk = (cs.get("skip_reason") or cs.get("ai_reason_easy") or "")
        lines.append(f"- {sym} | {dec}({conf}%) | 단기:{cs.get('trend_short','-')} | 장기:{cs.get('trend_filter','-')} | {str(sk)[:40]}")

    tg_send("\n".join(lines), target="channel")
    rt["last_hourly_vision_epoch"] = time.time()
    save_runtime(rt)

def build_btc_morning_brief(cfg: Dict[str, Any]) -> str:
    ext = build_external_context_korean(cfg)
    raw = ext.get("headlines") or []
    ko = ext.get("headlines_ko") or []
    items = ko if ko else raw
    # BTC 관련 우선 필터
    picked = []
    for t in items:
        s = str(t)
        if ("BTC" in s) or ("Bitcoin" in s) or ("비트코인" in s):
            picked.append(s)
    if len(picked) < int(cfg.get("morning_brief_top_n", 5)):
        for t in items:
            if t not in picked:
                picked.append(str(t))
            if len(picked) >= int(cfg.get("morning_brief_top_n", 5)):
                break
    picked = picked[:int(cfg.get("morning_brief_top_n", 5))]

    client = get_openai_client(cfg)
    if client is None:
        # 키 없으면 그냥 타이틀만
        lines = ["🌅 오늘 아침 BTC 뉴스(원문)"] + [f"{i+1}) {x}" for i, x in enumerate(picked)]
        lines.append("⚠️ 한글 요약은 OpenAI 키가 있어야 가능")
        return "\n".join(lines)

    sys = "너는 암호화폐 뉴스 큐레이터다. 주어진 헤드라인에서 '오늘 BTC 시황에 영향'을 중심으로 5개를 뽑아 아주 쉬운 한국어로 1줄 요약해라. 반드시 JSON만."
    user = {"headlines": picked, "need": int(cfg.get("morning_brief_top_n", 5))}
    try:
        resp = _openai_chat_json(client, "gpt-4o-mini", [
            {"role":"system","content":sys},
            {"role":"user","content":json.dumps(user, ensure_ascii=False)}
        ], temperature=0.2)
        if not resp:
            raise Exception("no resp")
        out = json.loads(resp.choices[0].message.content)
        arr = out.get("top5", []) or out.get("items", [])
        if not isinstance(arr, list) or not arr:
            raise Exception("bad json")
        lines = [
            "🌅 오늘 아침 BTC 핵심 뉴스 5",
            f"🕘 기준: {now_kst_str()}",
            "—"
        ]
        for i, it in enumerate(arr[:int(cfg.get("morning_brief_top_n", 5))]):
            if isinstance(it, dict):
                title = str(it.get("title",""))
                summary = str(it.get("summary",""))
                impact = str(it.get("impact",""))
                lines.append(f"{i+1}) 📰 {title}")
                if summary:
                    lines.append(f"   • {summary}")
                if impact:
                    lines.append(f"   • 🎯 영향: {impact}")
            else:
                lines.append(f"{i+1}) {str(it)}")
        lines.append("—")
        lines.append("✅ 오늘은 위 내용 반영해서 ‘변동성/이벤트 주의’ 여부를 자동으로 리스크에 반영합니다.")
        return "\n".join(lines)
    except Exception:
        lines = ["🌅 오늘 아침 BTC 뉴스(간단)"] + [f"{i+1}) {x}" for i, x in enumerate(picked)]
        return "\n".join(lines)

def send_daily_morning_brief_if_due(cfg: Dict[str, Any], rt: Dict[str, Any]):
    if not cfg.get("tg_daily_morning_brief", True):
        return
    today = today_kst_str()
    if rt.get("last_morning_brief_date") == today:
        return

    h = int(cfg.get("morning_brief_hour", 9))
    m = int(cfg.get("morning_brief_minute", 0))
    now = now_kst()
    due = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if now < due:
        return

    msg = build_btc_morning_brief(cfg)
    tg_send(msg, target="channel")
    rt["last_morning_brief_date"] = today
    record_event(rt, "MORNING_BRIEF", "BTC", "아침뉴스 전송")
    save_runtime(rt)


# =========================================================
# ✅ 20) 자동매매 핵심 스레드 (기능 유지 + 신규 기능 통합)
# =========================================================
def telegram_thread(ex):
    offset = 0
    mon = monitor_init()

    tg_send("🚀 AI 봇 가동 시작! (모의투자)\n그룹에서 /menu 로 메뉴 열기", target="channel")
    tg_send_menu()

    active_targets: Dict[str, Dict[str, Any]] = {}
    backoff_sec = 1

    def _tg_status_text() -> str:
        cfg_live = load_settings()
        free, total = safe_fetch_balance(ex)
        rt2 = load_runtime()
        mon_now = read_json_safe(MONITOR_FILE, {}) or {}
        paused = (time.time() < float(rt2.get("pause_until", 0)))
        return (
            "📡 상태\n"
            f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
            f"- 모드: {cfg_live.get('trade_mode','-')}\n"
            f"- 레짐: {mon_now.get('trade_style','-')} ({mon_now.get('style_confidence','-')}%)\n"
            f"- 잔고: {total:.2f} USDT (가용 {free:.2f})\n"
            f"- 연속손실: {rt2.get('consec_losses',0)}\n"
            f"- 보호정지: {'🛑 정지중' if paused else '✅ 정상'}\n"
            f"- 하트비트: {mon_now.get('last_heartbeat_kst','-')}\n"
        )

    def _tg_vision_text(limit: int = 10) -> str:
        mon_now = read_json_safe(MONITOR_FILE, {}) or {}
        coins = mon_now.get("coins", {}) or {}
        lines = [
            "👁️ AI 시야(요약)",
            f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
            f"- 모드: {mon_now.get('trade_mode','-')}",
            f"- 레짐: {mon_now.get('trade_style','-')} ({mon_now.get('style_confidence','-')}%)",
            f"- 레짐 이유: {str(mon_now.get('style_reason','-'))[:60]}",
            f"- 필터TF: {mon_now.get('trend_filter_tf','-')}",
            f"- 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
            "—"
        ]
        for sym, cs in list(coins.items())[:limit]:
            lines.append(
                f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_filter','-')} "
                f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
            )
        return "\n".join(lines)

    def _tg_log_text(n: int = 8) -> str:
        df_log = read_trade_log()
        if df_log.empty:
            return "📜 일지 없음(아직 기록된 매매가 없어요)"
        top = df_log.head(n)
        msg = ["📜 최근 매매일지(요약)"]
        for _, r in top.iterrows():
            tid = str(r.get("TradeID","") or "")
            pct = float(r.get("PnL_Percent", 0) or 0)
            emo = pnl_emoji(pct)
            msg.append(f"- {emo} {r['Time']} {r['Coin']} {r['Side']} {pct:.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}")
        return "\n".join(msg)

    def _tg_positions_text() -> str:
        msg = ["📊 포지션"]
        has = False
        for sym in TARGET_COINS:
            ps = safe_fetch_positions(ex, [sym])
            act = [p for p in ps if float(p.get("contracts") or 0) > 0]
            if act:
                p = act[0]
                has = True
                side = position_side_normalize(p)
                roi = float(position_roi_percent(p))
                upnl = float(p.get("unrealizedPnl") or 0.0)
                emo = "🟢" if roi > 0 else ("🔴" if roi < 0 else "⚪")
                msg.append(f"- {emo} {sym}: {('롱' if side=='long' else '숏')} (ROI {roi:.2f}%, 손익 {upnl:.2f} USDT)")
        if not has:
            msg.append("- ⚪ 없음(관망)")
        return "\n".join(msg)

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()
            mode = cfg.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])

            # (1) 레짐 결정(시장상황 기반)
            style, style_conf, style_reason = get_trade_style(ex, cfg, rt)
            trend_filter_tf = get_trend_filter_tf(cfg, style)

            # (2) 외부 시황 갱신(캐시 포함)
            ext_raw = build_external_context(cfg)
            ext_ko = build_external_context_korean(cfg)
            mon["external"] = ext_ko
            risk_level, risk_score, risk_note = assess_external_risk(ext_raw)

            # (3) 하트비트
            mon["last_heartbeat_epoch"] = time.time()
            mon["last_heartbeat_kst"] = now_kst_str()
            mon["auto_trade"] = bool(cfg.get("auto_trade", False))
            mon["trade_mode"] = mode
            mon["trade_style"] = style
            mon["style_confidence"] = style_conf
            mon["style_reason"] = style_reason
            mon["trend_filter_tf"] = trend_filter_tf
            mon["external_risk"] = {"level": risk_level, "note": risk_note}
            mon["pause_until"] = rt.get("pause_until", 0)
            mon["consec_losses"] = rt.get("consec_losses", 0)

            free_usdt, total_usdt = safe_fetch_balance(ex)

            # (4) 아침 뉴스 자동 전송(채널)
            send_daily_morning_brief_if_due(cfg, rt)

            # (5) 자동매매 ON일 때만
            if cfg.get("auto_trade", False):
                # 주말 거래 금지
                if cfg.get("no_trade_weekend", False):
                    wd = now_kst().weekday()
                    if wd in [5, 6]:
                        mon["global_state"] = "주말 거래 OFF"
                        monitor_write_throttled(mon, 2.0)
                        time.sleep(2.0)
                        continue

                # 일시정지(연속손실)
                if cfg.get("loss_pause_enable", True) and time.time() < float(rt.get("pause_until", 0)):
                    mon["global_state"] = "일시정지 중(연속손실 보호)"
                    monitor_write_throttled(mon, 2.0)
                    time.sleep(1.0)
                else:
                    mon["global_state"] = "스캔/매매 중"

                    # -------------------------------------------------
                    # 1) 포지션 관리 (기존 + 레짐 전환)
                    # -------------------------------------------------
                    for sym in TARGET_COINS:
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if not act:
                            continue

                        p = act[0]
                        side = position_side_normalize(p)
                        contracts = float(p.get("contracts") or 0)
                        entry = float(p.get("entryPrice") or 0)
                        roi = float(position_roi_percent(p))
                        cur_px = get_last_price(ex, sym) or entry

                        tgt = active_targets.get(sym, {
                            "sl": 2.0, "tp": 5.0,
                            "entry_usdt": 0.0, "entry_pct": 0.0,
                            "lev": p.get("leverage", "?"),
                            "reason": "", "trade_id": "",
                            "sl_price": None, "tp_price": None,
                            "sl_price_pct": None,
                            "style": style,
                            "entry_epoch": time.time(),
                            "regime_state": "normal",
                            "last_regime_check": 0,
                        })
                        sl = float(tgt.get("sl", 2.0))
                        tp = float(tgt.get("tp", 5.0))
                        trade_id = str(tgt.get("trade_id") or "")
                        style_at_entry = tgt.get("style", style)
                        entry_epoch = float(tgt.get("entry_epoch", time.time()) or time.time())

                        # ✅ 포지션 레짐 전환 체크(심볼별 추천)
                        if (time.time() - float(tgt.get("last_regime_check", 0))) >= int(cfg.get("pos_regime_check_sec", 30)):
                            tgt["last_regime_check"] = time.time()
                            sug_style, sug_conf, sug_reason, _amb = decide_style_rule_for_symbol(ex, sym, cfg)

                            # 스윙 -> 스캘핑으로 “다운그레이드”
                            if style_at_entry == "스윙" and sug_style == "스캘핑":
                                tgt["regime_state"] = "swing_to_scalp"
                                # 더 짧게 끝내도록 TP/SL 약간 조임
                                tp = min(tp, max(2.2, tp * 0.65))
                                sl = min(sl, max(1.8, sl * 0.85))
                                tgt["tp"], tgt["sl"] = tp, sl
                                record_event(rt, "REGIME", sym, "스윙→스캘핑(청산 보수)")
                                tg_send(f"🔁 레짐 전환 감지\n- {sym}\n- 스윙→스캘핑\n- 사유: {sug_reason}\n- 운영: 빠른 청산 모드", target="channel")

                            # 스캘핑 -> 스윙으로 “업그레이드”
                            if style_at_entry == "스캘핑" and sug_style == "스윙":
                                hold_min = (time.time() - entry_epoch) / 60.0
                                if hold_min >= float(cfg.get("scalp_to_swing_upgrade_hold_min", 12)):
                                    tgt["regime_state"] = "scalp_to_swing"
                                    tgt["style"] = "스윙"  # 이제부터 스윙 기능 허용
                                    style_at_entry = "스윙"
                                    # 목표 손익비 상향(스윙로직)
                                    tp = max(tp, tp * float(cfg.get("scalp_to_swing_tp_boost", 1.35)))
                                    sl = max(sl, sl * float(cfg.get("scalp_to_swing_sl_boost", 1.10)))
                                    tgt["tp"], tgt["sl"] = tp, sl
                                    record_event(rt, "REGIME", sym, "스캘핑→스윙(목표상향)")
                                    tg_send(f"🔁 레짐 전환 감지\n- {sym}\n- 스캘핑→스윙(유지/확장)\n- 사유: {sug_reason}\n- 변경: TP/SL 상향 + 스윙 기능 허용", target="channel")

                            active_targets[sym] = tgt
                            save_runtime(rt)

                        # ✅ 스윙→스캘핑 상태면 “빠른 청산 조건” 추가
                        if tgt.get("regime_state") == "swing_to_scalp":
                            hold_min = (time.time() - entry_epoch) / 60.0
                            # 어느 정도 수익이면 빠른 청산
                            if roi >= float(cfg.get("swing_to_scalp_exit_roi", 0.8)):
                                ok = close_position_market(ex, sym, side, contracts)
                                if ok:
                                    exit_px = get_last_price(ex, sym) or entry
                                    pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                                    one, review = ai_write_review(sym, side, roi, "레짐전환(스윙→스캘핑) 빠른청산", cfg)
                                    log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "레짐전환 빠른 익절", one, review, trade_id, style=tgt.get("style",""), mode=mode)
                                    tg_send(f"✅ 레짐전환 청산(빠른익절)\n- {sym}\n- ROI: +{roi:.2f}% ({pnl_usdt_snapshot:.2f} USDT)\n- 사유: 스윙→스캘핑 전환\n- ID: {trade_id}", target="channel")
                                    active_targets.pop(sym, None)
                                    rt.setdefault("trades", {}).pop(sym, None)
                                    save_runtime(rt)
                                    continue

                            # 오래 끌면(손익 미미) 청산 유도
                            if hold_min >= float(cfg.get("swing_to_scalp_max_hold_min", 60)) and roi > -0.6:
                                ok = close_position_market(ex, sym, side, contracts)
                                if ok:
                                    exit_px = get_last_price(ex, sym) or entry
                                    pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                                    one, review = ai_write_review(sym, side, roi, "레짐전환(스윙→스캘핑) 시간초과", cfg)
                                    log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "레짐전환 시간초과 청산", one, review, trade_id, style=tgt.get("style",""), mode=mode)
                                    tg_send(f"⏱️ 레짐전환 청산(시간초과)\n- {sym}\n- ROI: {roi:.2f}% ({pnl_usdt_snapshot:.2f} USDT)\n- 사유: 스윙→스캘핑 전환 후 장시간\n- ID: {trade_id}", target="channel")
                                    active_targets.pop(sym, None)
                                    rt.setdefault("trades", {}).pop(sym, None)
                                    save_runtime(rt)
                                    continue

                        # 트레일링(기존 유지)
                        if cfg.get("use_trailing_stop", True):
                            if roi >= (tp * 0.5):
                                lev_now = float(tgt.get("lev", p.get("leverage", 1))) or 1.0
                                base_price_sl = float(tgt.get("sl_price_pct") or max(0.25, float(sl) / max(lev_now, 1)))
                                trail_price_pct = max(0.20, base_price_sl * 0.60)
                                trail_roi = trail_price_pct * lev_now
                                sl = min(sl, max(1.2, float(trail_roi)))
                                tgt["sl"] = sl
                                active_targets[sym] = tgt

                        # SR 가격 트리거 (기존 유지)
                        sl_price = tgt.get("sl_price")
                        tp_price = tgt.get("tp_price")
                        hit_sl_by_price = False
                        hit_tp_by_price = False
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

                        # ✅ 스캘핑이면 기본 DCA/스위칭 OFF (단, 스윙으로 업그레이드된 경우 허용)
                        allow_dca = bool(cfg.get("use_dca", True)) and (tgt.get("style","") == "스윙")
                        allow_switch = bool(cfg.get("use_switching", True)) and (tgt.get("style","") == "스윙")

                        # DCA (기능 유지: 단 스윙만 기본 허용)
                        if allow_dca:
                            dca_trig = float(cfg.get("dca_trigger", -20.0))
                            dca_max = int(cfg.get("dca_max_count", 1))
                            dca_add_pct = float(cfg.get("dca_add_pct", 50.0))

                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0})
                            dca_count = int(trade_state.get("dca_count", 0))

                            if roi <= dca_trig and dca_count < dca_max:
                                free_usdt, _ = safe_fetch_balance(ex)
                                base_entry = float(tgt.get("entry_usdt", 0.0))
                                add_usdt = base_entry * (dca_add_pct / 100.0)
                                if add_usdt > free_usdt:
                                    add_usdt = free_usdt * 0.5

                                if cur_px and add_usdt > 5:
                                    lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                    set_leverage_safe(ex, sym, lev)
                                    qty = to_precision_qty(ex, sym, (add_usdt * lev) / cur_px)
                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                    if ok:
                                        trade_state["dca_count"] = dca_count + 1
                                        save_runtime(rt)
                                        record_event(rt, "DCA", sym, f"+{add_usdt:.2f} USDT")
                                        tg_send(
                                            f"💧 물타기(DCA)\n- {sym}\n- 추가금: {add_usdt:.2f} USDT\n- 이유: 손실 {roi:.2f}% (기준 {dca_trig}%)",
                                            target="channel"
                                        )
                                        mon["last_action"] = {"time_kst": now_kst_str(), "type": "DCA", "symbol": sym, "roi": roi}
                                        monitor_write_throttled(mon, 0.2)

                        # ✅ 스윙 분할익절 (기존 유지)
                        if tgt.get("style","") == "스윙" and cfg.get("swing_partial_tp_enable", True):
                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0})
                            partial_done = trade_state.setdefault("partial_done", [])
                            levels = parse_csv_floats(cfg.get("swing_partial_tp_levels", "0.35,0.60,0.90"), [0.35, 0.60, 0.90])
                            sizes = parse_csv_floats(cfg.get("swing_partial_tp_sizes", "0.30,0.30,0.40"), [0.30, 0.30, 0.40])
                            n = min(len(levels), len(sizes))
                            levels = levels[:n]
                            sizes = normalize_portions(sizes[:n], max_sum=0.95)
                            tp_roi = float(tgt.get("tp", 0))

                            for i in range(n):
                                if i in partial_done:
                                    continue
                                target_roi = tp_roi * levels[i] if levels[i] <= 1.5 else levels[i]
                                if roi >= target_roi and roi < tp_roi:
                                    qty_part = to_precision_qty(ex, sym, contracts * sizes[i])
                                    if qty_part > 0:
                                        ok = close_position_market(ex, sym, side, qty_part)
                                        if ok:
                                            part_pnl = float(p.get("unrealizedPnl") or 0.0) * (qty_part / max(contracts, 1e-9))
                                            log_trade(sym, side, entry, cur_px, part_pnl, roi, "부분익절",
                                                      one_line="부분익절", review="", trade_id=trade_id,
                                                      style=tgt.get("style",""), mode=mode)

                                            d = load_trade_detail(trade_id) or {}
                                            plist = d.setdefault("partials", [])
                                            plist.append({
                                                "time": now_kst_str(),
                                                "roi": roi,
                                                "qty": qty_part,
                                                "price": cur_px,
                                                "pnl_usdt_est": part_pnl,
                                                "level": i + 1,
                                            })
                                            save_trade_detail(trade_id, d)

                                            partial_done.append(i)
                                            trade_state["partial_done"] = partial_done
                                            save_runtime(rt)
                                            record_event(rt, "PARTIAL_TP", sym, f"L{i+1} ROI {roi:.2f}%")

                                            if cfg.get("tg_enable_reports", True):
                                                tg_send(f"🔹 부분익절\n- {sym}\n- 단계: {i+1}\n- ROI: {roi:.2f}%\n- 수량: {qty_part}", target="channel")
                                            mon["last_action"] = {"time_kst": now_kst_str(), "type": "PARTIAL_TP", "symbol": sym, "roi": roi}

                        # ✅ 순환매도 (기존 유지)
                        if tgt.get("style","") == "스윙" and cfg.get("swing_recycle_enable", False):
                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0})
                            if trade_state.get("partial_done"):
                                max_cnt = int(cfg.get("swing_recycle_max_count", 1))
                                cnt = int(trade_state.get("recycle_count", 0))
                                cooldown = int(cfg.get("swing_recycle_cooldown_min", 30)) * 60
                                last_rc = float(trade_state.get("last_recycle_epoch", 0))
                                trigger_roi = float(cfg.get("swing_recycle_trigger_roi", 4.0))
                                if cnt < max_cnt and (time.time() - last_rc) >= cooldown and roi <= trigger_roi:
                                    free_usdt, _ = safe_fetch_balance(ex)
                                    add_usdt = float(tgt.get("entry_usdt", 0.0)) * (float(cfg.get("swing_recycle_add_pct", 20.0)) / 100.0)
                                    if add_usdt > free_usdt:
                                        add_usdt = free_usdt * 0.5
                                    if add_usdt > 5 and cur_px:
                                        lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                        set_leverage_safe(ex, sym, lev)
                                        qty = to_precision_qty(ex, sym, (add_usdt * lev) / cur_px)
                                        ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                        if ok:
                                            trade_state["recycle_count"] = cnt + 1
                                            trade_state["last_recycle_epoch"] = time.time()
                                            save_runtime(rt)
                                            record_event(rt, "RECYCLE", sym, f"+{add_usdt:.2f} USDT")
                                            if cfg.get("tg_enable_reports", True):
                                                tg_send(f"♻️ 순환매도(재진입)\n- {sym}\n- 추가금: {add_usdt:.2f} USDT\n- ROI: {roi:.2f}%", target="channel")

                                            d = load_trade_detail(trade_id) or {}
                                            rlist = d.setdefault("recycles", [])
                                            rlist.append({
                                                "time": now_kst_str(),
                                                "roi": roi,
                                                "add_usdt": add_usdt,
                                                "price": cur_px,
                                            })
                                            save_trade_detail(trade_id, d)

                        # ✅ 스위칭(기능 유지) - 스윙만 허용 (손실이 크고, 반대 방향 추세 강할 때)
                        if allow_switch:
                            switch_trig = float(cfg.get("switch_trigger", -12.0))
                            if roi <= switch_trig:
                                # 간단히 "반대 방향이 더 유리"하면 전량청산(스위칭은 보수적으로)
                                record_event(rt, "SWITCH_CHECK", sym, f"ROI {roi:.2f}%")
                                # 여기서는 무리한 반대진입 자동 실행은 피하고, “청산 + 쿨다운”만 수행(실전 안정성)
                                ok = close_position_market(ex, sym, side, contracts)
                                if ok:
                                    exit_px = get_last_price(ex, sym) or entry
                                    pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                                    one, review = ai_write_review(sym, side, roi, "스위칭 보호(손실 제한)", cfg)
                                    log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "스위칭 보호 청산", one, review, trade_id,
                                              style=tgt.get("style",""), mode=mode)
                                    tg_send(f"🧯 스위칭 보호(청산)\n- {sym}\n- ROI: {roi:.2f}% ({pnl_usdt_snapshot:.2f} USDT)\n- 사유: 손실 {switch_trig}% 이하\n- ID: {trade_id}", target="channel")
                                    active_targets.pop(sym, None)
                                    rt.setdefault("trades", {}).pop(sym, None)
                                    rt.setdefault("cooldowns", {})[sym] = time.time() + 180
                                    save_runtime(rt)
                                    continue

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
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 손절",
                                          one_line=one, review=review, trade_id=trade_id,
                                          style=tgt.get("style",""), mode=mode)

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update({"exit_time": now_kst_str(), "exit_price": exit_px,
                                              "pnl_usdt": pnl_usdt_snapshot, "pnl_pct": roi,
                                              "result": "SL", "review": review})
                                    save_trade_detail(trade_id, d)

                                rt["consec_losses"] = int(rt.get("consec_losses", 0)) + 1
                                record_event(rt, "STOP", sym, f"ROI {roi:.2f}%")
                                if cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(f"🛑 연속손실 보호\n- 연속손실: {rt['consec_losses']}회\n- {int(cfg.get('loss_pause_minutes',30))}분 자동 정지", target="channel")
                                    record_event(rt, "PAUSE", "", f"{int(cfg.get('loss_pause_minutes',30))}분")
                                save_runtime(rt)

                                tg_send(
                                    "🩸 손절(청산)\n"
                                    f"- 코인: {sym}\n"
                                    f"- 결과: {pnl_emoji(roi)} ROI {roi:.2f}% | 손익 {pnl_usdt_snapshot:.2f} USDT\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')} | 레짐: {tgt.get('style','-')}\n"
                                    f"- 잔고: {total_after:.2f} USDT (가용 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 이탈' if hit_sl_by_price else '목표 손절 도달'}\n"
                                    f"- 한줄평: {one}\n"
                                    f"- 일지ID: {trade_id or '없음'}",
                                    target="channel"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "STOP", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                        # 익절
                        elif do_take:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                one, review = ai_write_review(sym, side, roi, "자동 익절", cfg)
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 익절",
                                          one_line=one, review=review, trade_id=trade_id,
                                          style=tgt.get("style",""), mode=mode)

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update({"exit_time": now_kst_str(), "exit_price": exit_px,
                                              "pnl_usdt": pnl_usdt_snapshot, "pnl_pct": roi,
                                              "result": "TP", "review": review})
                                    save_trade_detail(trade_id, d)

                                rt["consec_losses"] = 0
                                record_event(rt, "TAKE", sym, f"ROI {roi:.2f}%")
                                save_runtime(rt)

                                tg_send(
                                    "🎉 익절(청산)\n"
                                    f"- 코인: {sym}\n"
                                    f"- 결과: {pnl_emoji(roi)} ROI +{roi:.2f}% | 손익 {pnl_usdt_snapshot:.2f} USDT\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')} | 레짐: {tgt.get('style','-')}\n"
                                    f"- 잔고: {total_after:.2f} USDT (가용 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 목표 도달' if hit_tp_by_price else '목표 익절 도달'}\n"
                                    f"- 한줄평: {one}\n"
                                    f"- 일지ID: {trade_id or '없음'}",
                                    target="channel"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "TAKE", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                    # -------------------------------------------------
                    # 2) 신규 진입 스캔 (역추세 금지 X -> 역추세면 스캘핑 강제)
                    # -------------------------------------------------
                    for sym in TARGET_COINS:
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if act:
                            continue

                        # 쿨다운
                        cd = float(rt.get("cooldowns", {}).get(sym, 0))
                        if time.time() < cd:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "⏳ 쿨다운(잠깐 쉬는중)"
                            continue

                        # 데이터 로드(단기: cfg timeframe)
                        try:
                            ohlcv = fetch_ohlcv_cached(ex, sym, cfg.get("timeframe", "5m"), limit=220, cache_sec=8)
                            if not ohlcv:
                                raise Exception("ohlcv 없음")
                            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except Exception as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = f"데이터 실패: {e}"
                            continue

                        df, stt, last = calc_indicators(df, cfg)
                        mon.setdefault("coins", {}).setdefault(sym, {})
                        cs = mon["coins"][sym]

                        if last is None:
                            cs.update({"last_scan_kst": now_kst_str(), "ai_called": False,
                                       "skip_reason": "지표 계산 실패(ta/데이터 부족)"})
                            continue

                        # 장기추세(필터용) - 현재 레짐 기준 TF
                        htf_tf = trend_filter_tf
                        htf_trend = get_htf_trend_cached(
                            ex, sym, htf_tf,
                            fast=int(cfg.get("ma_fast", 7)),
                            slow=int(cfg.get("ma_slow", 99)),
                            cache_sec=int(cfg.get("trend_filter_cache_sec", 60)),
                        )
                        cs["trend_filter"] = f"🧭 {htf_tf} {htf_trend}"

                        cs.update({
                            "last_scan_epoch": time.time(),
                            "last_scan_kst": now_kst_str(),
                            "price": float(last["close"]),
                            "trend_short": stt.get("추세", ""),
                            "trend_filter": cs.get("trend_filter", ""),
                            "trend_filter_tf": htf_tf,
                            "rsi": float(last.get("RSI", 0)) if "RSI" in df.columns else None,
                            "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
                            "bb": stt.get("BB", ""),
                            "macd": stt.get("MACD", ""),
                            "vol": stt.get("거래량", ""),
                            "pullback_candidate": bool(stt.get("_pullback_candidate", False)),
                            "style_global": style,
                        })

                        # AI 호출 조건 (기존 유지)
                        call_ai = False
                        if bool(stt.get("_pullback_candidate", False)):
                            call_ai = True
                        elif bool(stt.get("_rsi_resolve_long", False)) or bool(stt.get("_rsi_resolve_short", False)):
                            call_ai = True
                        else:
                            adxv = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            if adxv >= 25:
                                call_ai = True

                        if not call_ai:
                            cs["ai_called"] = False
                            cs["skip_reason"] = "⚪ 해소 신호 약함(휩쏘 위험)"
                            monitor_write_throttled(mon, 1.0)
                            continue

                        # AI 판단 (일단 글로벌 레짐을 넘겨서 판단)
                        ai = ai_decide_trade(df, stt, sym, mode, cfg, trade_style=style)
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))

                        # 외부 시황 리스크 반영(기존 유지)
                        ai = apply_external_risk_adjustment(ai, ext_raw, cfg, rule)
                        min_conf_adj = int(ai.get("min_conf_adj", rule["min_conf"]))

                        # ✅ 역추세 정책: entry_style 결정 (단기+장기 일치면 스윙, 아니면 스캘핑)
                        entry_style = style
                        entry_style_reason = ""
                        if decision in ["buy","sell"]:
                            entry_style, entry_style_reason = determine_entry_style(cfg, stt.get("추세",""), htf_trend, decision)
                            # 역추세면 보수 세팅(진입%/레버/RR 제한)
                            ai = apply_countertrend_adjustments(cfg, ai, entry_style, htf_trend, decision)

                        cs.update({
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
                            "min_conf_required": min_conf_adj,
                            "external_risk": ai.get("external_risk_note", ""),
                            "entry_style": entry_style,
                            "entry_style_reason": entry_style_reason,
                            "skip_reason": ""
                        })
                        monitor_write_throttled(mon, 1.0)

                        # 진입
                        if decision in ["buy", "sell"] and conf >= min_conf_adj:
                            entry_pct = float(ai.get("entry_pct", rule["entry_pct_min"]))
                            lev = int(ai.get("leverage", rule["lev_min"]))
                            slp = float(ai.get("sl_pct", 1.2))
                            tpp = float(ai.get("tp_pct", 3.0))

                            entry_usdt = free_usdt * (entry_pct / 100.0)
                            px = float(last["close"])
                            if entry_usdt < 5:
                                cs["skip_reason"] = "잔고 부족(진입금 너무 작음)"
                                continue

                            set_leverage_safe(ex, sym, lev)
                            qty = to_precision_qty(ex, sym, (entry_usdt * lev) / px)
                            if qty <= 0:
                                cs["skip_reason"] = "수량 계산 실패"
                                continue

                            ok = market_order_safe(ex, sym, decision, qty)
                            if ok:
                                trade_id = uuid.uuid4().hex[:10]

                                # SR 기반 SL/TP 가격도 계산 (기존 유지)
                                sl_price = None
                                tp_price = None
                                if cfg.get("use_sr_stop", True):
                                    try:
                                        sr_tf = cfg.get("sr_timeframe", "15m")
                                        htf = fetch_ohlcv_cached(ex, sym, sr_tf, limit=220, cache_sec=15)
                                        if htf:
                                            hdf = pd.DataFrame(htf, columns=["time","open","high","low","close","vol"])
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

                                active_targets[sym] = {
                                    "sl": slp, "tp": tpp,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "reason": ai.get("reason_easy", ""),
                                    "trade_id": trade_id,
                                    "sl_price": sl_price,
                                    "tp_price": tp_price,
                                    "sl_price_pct": float(ai.get("sl_price_pct", slp / max(lev, 1))),
                                    "style": entry_style,         # ✅ 진입 스타일 확정 저장
                                    "entry_epoch": time.time(),
                                    "regime_state": "normal",
                                    "last_regime_check": 0,
                                }

                                # 상세일지(기존 유지 + 필드 추가)
                                save_trade_detail(trade_id, {
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
                                    "used_indicators": ai.get("used_indicators", []),
                                    "reason_easy": ai.get("reason_easy", ""),
                                    "raw_status": stt,
                                    "trend_short": stt.get("추세", ""),
                                    "trend_filter": cs.get("trend_filter", ""),
                                    "style": entry_style,
                                    "style_reason": entry_style_reason,
                                    "mode": mode,
                                    "external_risk": ai.get("external_risk_note",""),
                                    "countertrend_note": ai.get("countertrend_note",""),
                                })

                                # 쿨다운
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)
                                record_event(rt, "ENTRY", sym, f"{decision} {entry_usdt:.2f} USDT")

                                # 텔레그램 보고(채널)
                                if cfg.get("tg_enable_reports", True):
                                    direction = "🟢 롱" if decision == "buy" else "🔴 숏"
                                    msg = (
                                        "🎯 진입\n"
                                        f"- 코인: {sym}\n"
                                        f"- 방향: {direction}\n"
                                        f"- 레짐: {entry_style} | 이유: {entry_style_reason}\n"
                                        f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%)\n"
                                        f"- 레버리지: x{lev}\n"
                                        f"- 목표익절: +{tpp:.2f}% / 목표손절: -{slp:.2f}%\n"
                                        f"- 단기추세({cfg.get('timeframe','5m')}): {stt.get('추세','-')}\n"
                                        f"- 장기추세({htf_tf}): {htf_trend}\n"
                                        f"- 확신도: {conf}% (기준 {min_conf_adj}%)\n"
                                        f"- 외부리스크: {ai.get('external_risk_note','-')}\n"
                                    )
                                    if sl_price is not None and tp_price is not None:
                                        msg += f"- SR기준가: TP {tp_price:.6g} / SL {sl_price:.6g}\n"
                                    msg += f"- 일지ID: {trade_id}\n"
                                    if cfg.get("tg_send_entry_reason", False):
                                        msg += f"- 근거(쉬운말): {ai.get('reason_easy','')[:220]}\n- AI지표: {', '.join(ai.get('used_indicators', []))}\n"
                                    tg_send(msg, target="channel")

                                mon["last_action"] = {
                                    "time_kst": now_kst_str(),
                                    "type": "ENTRY",
                                    "symbol": sym,
                                    "decision": decision,
                                    "conf": conf,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "tp": tpp,
                                    "sl": slp,
                                    "trade_id": trade_id,
                                    "style": entry_style,
                                }
                                monitor_write_throttled(mon, 0.2)
                                time.sleep(1.0)

                        time.sleep(0.35)

            # -------------------------------------------------
            # 텔레그램 수신 처리(그룹)
            # -------------------------------------------------
            try:
                res = _safe_request(
                    "GET",
                    f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1",
                    timeout=TG_TIMEOUT
                )
                res = res.json() if res else {"ok": False}
            except Exception:
                res = {"ok": False}

            if res.get("ok"):
                for up in res.get("result", []):
                    offset = up.get("update_id", offset)

                    # 텍스트 명령
                    if "message" in up and "text" in up["message"]:
                        txt = up["message"]["text"].strip()

                        if txt in ["/start", "/menu", "menu", "메뉴"]:
                            tg_send_menu()

                        elif txt in ["상태", "/status"]:
                            tg_send(_tg_status_text(), target="group")

                        elif txt in ["시야", "/vision"]:
                            tg_send(_tg_vision_text(), target="group")

                        elif txt in ["일지", "/log"]:
                            tg_send(_tg_log_text(), target="group")

                        elif txt in ["포지션", "/position"]:
                            tg_send(_tg_positions_text(), target="group")

                        elif txt in ["잔고", "/balance"]:
                            free, total = safe_fetch_balance(ex)
                            tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT", target="group")

                        elif txt.startswith("일지상세"):
                            parts = txt.split()
                            if len(parts) < 2:
                                tg_send("사용법: 일지상세 <ID>\n(예: 일지상세 a1b2c3d4e5)", target="group")
                            else:
                                tid = parts[1].strip()
                                d = load_trade_detail(tid)
                                if not d:
                                    tg_send("해당 ID를 찾지 못했어요.", target="group")
                                else:
                                    tg_send(
                                        f"🧾 일지상세 {tid}\n"
                                        f"- 코인: {d.get('coin')}\n"
                                        f"- 방향: {d.get('decision')}\n"
                                        f"- 확신도: {d.get('confidence')}\n"
                                        f"- 진입가: {d.get('entry_price')}\n"
                                        f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                        f"- 레버: x{d.get('lev')}\n"
                                        f"- 레짐: {d.get('style','-')} | {str(d.get('style_reason',''))[:60]}\n"
                                        f"- 단기추세: {d.get('trend_short','-')}\n"
                                        f"- 장기추세: {d.get('trend_filter','-')}\n"
                                        f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                        f"- 한줄근거: {str(d.get('reason_easy',''))[:200]}\n"
                                        f"- 사용지표: {', '.join(d.get('used_indicators', []))[:200]}\n",
                                        target="group"
                                    )

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up["callback_query"]
                        data = cb.get("data", "")
                        cb_id = cb.get("id", "")

                        if data == "status":
                            tg_send(_tg_status_text(), target="group")
                        elif data == "vision":
                            tg_send(_tg_vision_text(), target="group")
                        elif data == "balance":
                            free, total = safe_fetch_balance(ex)
                            tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT", target="group")
                        elif data == "position":
                            tg_send(_tg_positions_text(), target="group")
                        elif data == "log":
                            tg_send(_tg_log_text(), target="group")
                        elif data == "log_detail_help":
                            tg_send("🧾 일지상세 사용법\n- 일지상세 <ID>\n예) 일지상세 a1b2c3d4e5\n(최근 ID는 '일지'에서 확인)", target="group")
                        elif data == "close_all":
                            tg_send("🛑 전량 청산 시도", target="group")
                            for sym in TARGET_COINS:
                                ps = safe_fetch_positions(ex, [sym])
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    continue
                                p = act[0]
                                side = position_side_normalize(p)
                                contracts = float(p.get("contracts") or 0)
                                close_position_market(ex, sym, side, contracts)
                            tg_send("✅ 전량 청산 요청 완료", target="group")

                        tg_answer_callback(cb_id)

            # (6) 15분 정기 리포트(채널)
            send_periodic_report(ex, cfg, rt, mon)
            # (7) 1시간 AI시야 리포트(채널)
            send_hourly_vision_report(cfg, rt, mon)

            monitor_write_throttled(mon, 2.0)
            time.sleep(0.8)
            backoff_sec = 1

        except Exception as e:
            tg_send(f"⚠️ 스레드 오류: {e}", target="channel")
            time.sleep(min(30, backoff_sec))
            backoff_sec = min(30, backoff_sec * 2)


# =========================================================
# ✅ 21) 워치독 스레드(하트비트 감시)
# =========================================================
def watchdog_thread():
    while True:
        try:
            cfg = load_settings()
            if not cfg.get("watchdog_enabled", True):
                time.sleep(10)
                continue

            mon = read_json_safe(MONITOR_FILE, {}) or {}
            hb = float(mon.get("last_heartbeat_epoch", 0) or 0)
            age = time.time() - hb if hb else 9999
            rt = load_runtime()

            if age >= int(cfg.get("watchdog_timeout_sec", 60)):
                last_warn = float(rt.get("last_watchdog_warn", 0))
                if time.time() - last_warn > 60:
                    tg_send(f"⚠️ 하트비트 지연: {age:.0f}초 (봇 스레드 점검)", target="channel")
                    rt["last_watchdog_warn"] = time.time()
                    save_runtime(rt)

                alive = any(t.name == "TG_THREAD" and t.is_alive() for t in threading.enumerate())
                if not alive:
                    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
                    add_script_run_ctx(th)
                    th.start()

            time.sleep(int(cfg.get("watchdog_check_sec", 15)))
        except Exception:
            time.sleep(5)


# =========================================================
# ✅ 22) 스레드 시작(중복 방지)
# =========================================================
def ensure_thread_started():
    for t in threading.enumerate():
        if t.name == "TG_THREAD":
            return
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(th)
    th.start()

def ensure_watchdog_started():
    for t in threading.enumerate():
        if t.name == "WATCHDOG":
            return
    th = threading.Thread(target=watchdog_thread, args=(), daemon=True, name="WATCHDOG")
    add_script_run_ctx(th)
    th.start()

ensure_thread_started()
ensure_watchdog_started()


# =========================================================
# ✅ 23) Streamlit UI (기존 기능 유지 + 구글시트 상태 표시)
# =========================================================
st.sidebar.title("🛠️ 제어판")
st.sidebar.caption("Streamlit=제어/모니터링, Telegram(채널)=알림, Telegram(그룹)=조회/버튼")

openai_key_secret = st.secrets.get("OPENAI_API_KEY", "")
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

auto_on = st.sidebar.checkbox("🤖 자동매매", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.divider()

config["timeframe"] = st.sidebar.selectbox("타임프레임", ["1m","3m","5m","15m","1h"],
                                           index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))

config["tg_enable_reports"] = st.sidebar.checkbox("📨 텔레그램 보고", value=bool(config.get("tg_enable_reports", True)))
config["tg_send_entry_reason"] = st.sidebar.checkbox("📌 진입근거(긴글)도 보내기", value=bool(config.get("tg_send_entry_reason", False)))

config["tg_enable_periodic_report"] = st.sidebar.checkbox("🕒 정기 리포트", value=bool(config.get("tg_enable_periodic_report", True)))
config["report_interval_min"] = st.sidebar.number_input("리포트 주기(분)", 5, 120, int(config.get("report_interval_min", 15)))

config["tg_enable_hourly_vision"] = st.sidebar.checkbox("👁️ 시간당 AI시야(채널)", value=bool(config.get("tg_enable_hourly_vision", True)))
config["tg_hourly_vision_min"] = st.sidebar.number_input("시야 리포트 주기(분)", 10, 180, int(config.get("tg_hourly_vision_min", 60)))

st.sidebar.divider()
st.sidebar.subheader("🌅 아침 BTC 뉴스")
config["tg_daily_morning_brief"] = st.sidebar.checkbox("아침 뉴스 전송(채널)", value=bool(config.get("tg_daily_morning_brief", True)))
c_m1, c_m2, c_m3 = st.sidebar.columns(3)
config["morning_brief_hour"] = c_m1.number_input("시", 0, 23, int(config.get("morning_brief_hour", 9)))
config["morning_brief_minute"] = c_m2.number_input("분", 0, 59, int(config.get("morning_brief_minute", 0)))
config["morning_brief_top_n"] = c_m3.number_input("개수", 3, 10, int(config.get("morning_brief_top_n", 5)))

st.sidebar.divider()
st.sidebar.subheader("🧾 구글시트 일지")
config["gs_enable"] = st.sidebar.checkbox("구글시트 기록 사용", value=bool(config.get("gs_enable", True)))
config["gs_daily_tabs"] = st.sidebar.checkbox("일별 탭 분리", value=bool(config.get("gs_daily_tabs", True)))
config["gs_daily_summary"] = st.sidebar.checkbox("일별 요약 시트", value=bool(config.get("gs_daily_summary", True)))

if config["gs_enable"]:
    ok = gs_enabled(config)
    st.sidebar.caption("✅ 구글시트 연결" if ok else "⚠️ 구글시트 연결 불가(서비스계정/시트ID/패키지 확인)")

st.sidebar.divider()
st.sidebar.subheader("🧭 자동 스타일(스캘핑/스윙)")
config["auto_style"] = st.sidebar.checkbox("자동 레짐 선택", value=bool(config.get("auto_style", True)))
if not config["auto_style"]:
    config["fixed_style"] = st.sidebar.selectbox("고정 레짐", ["스캘핑","스윙"], index=["스캘핑","스윙"].index(config.get("fixed_style","스캘핑")))
config["style_lock_minutes"] = st.sidebar.number_input("최소 유지(분)", 5, 240, int(config.get("style_lock_minutes", 30)))

mon_view = read_json_safe(MONITOR_FILE, {}) or {}
st.sidebar.caption(f"현재 레짐: {mon_view.get('trade_style','-')} ({mon_view.get('style_confidence','-')}%)")
st.sidebar.caption(f"레짐 이유: {str(mon_view.get('style_reason','-'))[:60]}")
st.sidebar.caption(f"장기TF: {mon_view.get('trend_filter_tf','-')}")

st.sidebar.divider()
st.sidebar.subheader("🔁 역추세 정책")
config["countertrend_allowed"] = st.sidebar.checkbox("역추세 허용", value=bool(config.get("countertrend_allowed", True)))
config["countertrend_force_scalp"] = st.sidebar.checkbox("역추세면 스캘핑 강제", value=bool(config.get("countertrend_force_scalp", True)))
config["countertrend_entry_reduce"] = st.sidebar.slider("역추세 진입% 감산", 0.2, 1.0, float(config.get("countertrend_entry_reduce", 0.65)), step=0.05)
config["countertrend_lev_reduce"] = st.sidebar.number_input("역추세 레버 감산", 0, 5, int(config.get("countertrend_lev_reduce", 1)))
config["countertrend_rr_cap"] = st.sidebar.number_input("역추세 RR 상한", 1.0, 4.0, float(config.get("countertrend_rr_cap", 1.8)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🧱 지지/저항(SR) 손절/익절")
config["use_sr_stop"] = st.sidebar.checkbox("SR 기반 가격 손절/익절 사용", value=bool(config.get("use_sr_stop", True)))
c_sr1, c_sr2 = st.sidebar.columns(2)
config["sr_timeframe"] = c_sr1.selectbox("SR 타임프레임", ["5m","15m","1h","4h"],
                                         index=["5m","15m","1h","4h"].index(config.get("sr_timeframe","15m")))
config["sr_pivot_order"] = c_sr2.number_input("피벗 민감도", 3, 10, int(config.get("sr_pivot_order", 6)))
c_sr3, c_sr4 = st.sidebar.columns(2)
config["sr_atr_period"] = c_sr3.number_input("ATR 기간", 7, 30, int(config.get("sr_atr_period", 14)))
config["sr_buffer_atr_mult"] = c_sr4.number_input("버퍼(ATR배)", 0.05, 2.0, float(config.get("sr_buffer_atr_mult", 0.25)), step=0.05)
config["sr_rr_min"] = st.sidebar.number_input("SR 최소 RR", 1.0, 5.0, float(config.get("sr_rr_min", 1.5)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("♻️ 스윙 분할익절/순환")
config["swing_partial_tp_enable"] = st.sidebar.checkbox("스윙 분할익절", value=bool(config.get("swing_partial_tp_enable", True)))
config["swing_partial_tp_levels"] = st.sidebar.text_input("분할익절 구간(목표TP 대비 비율)", value=str(config.get("swing_partial_tp_levels", "0.35,0.60,0.90")))
config["swing_partial_tp_sizes"] = st.sidebar.text_input("청산 비중(합<=1)", value=str(config.get("swing_partial_tp_sizes", "0.30,0.30,0.40")))
config["swing_recycle_enable"] = st.sidebar.checkbox("순환매도(재진입)", value=bool(config.get("swing_recycle_enable", False)))
if config["swing_recycle_enable"]:
    c_rc1, c_rc2 = st.sidebar.columns(2)
    config["swing_recycle_trigger_roi"] = c_rc1.number_input("재진입 트리거 ROI", -5.0, 20.0, float(config.get("swing_recycle_trigger_roi", 4.0)), step=0.5)
    config["swing_recycle_add_pct"] = c_rc2.number_input("재진입 규모(%)", 5.0, 100.0, float(config.get("swing_recycle_add_pct", 20.0)), step=1.0)
    c_rc3, c_rc4 = st.sidebar.columns(2)
    config["swing_recycle_cooldown_min"] = c_rc3.number_input("쿨다운(분)", 5, 240, int(config.get("swing_recycle_cooldown_min", 30)))
    config["swing_recycle_max_count"] = c_rc4.number_input("최대 횟수", 0, 5, int(config.get("swing_recycle_max_count", 1)))

st.sidebar.divider()
st.sidebar.subheader("🛡️ 방어/자금 관리")
config["loss_pause_enable"] = st.sidebar.checkbox("연속손실 보호(자동 정지)", value=bool(config.get("loss_pause_enable", True)))
c1, c2 = st.sidebar.columns(2)
config["loss_pause_after"] = c1.number_input("연속손실 N회", 1, 20, int(config.get("loss_pause_after", 3)))
config["loss_pause_minutes"] = c2.number_input("정지(분)", 1, 240, int(config.get("loss_pause_minutes", 30)))

st.sidebar.divider()
config["use_dca"] = st.sidebar.checkbox("💧 물타기(DCA) (스윙에서 기본 허용)", value=bool(config.get("use_dca", True)))
c3, c4 = st.sidebar.columns(2)
config["dca_trigger"] = c3.number_input("DCA 발동(%)", -90.0, -1.0, float(config.get("dca_trigger", -20.0)), step=0.5)
config["dca_max_count"] = c4.number_input("최대 횟수", 0, 10, int(config.get("dca_max_count", 1)))
config["dca_add_pct"] = st.sidebar.slider("추가 규모(원진입 대비 %)", 10, 200, int(config.get("dca_add_pct", 50)))

config["use_switching"] = st.sidebar.checkbox("🔁 스위칭 보호(스윙에서만 자동)", value=bool(config.get("use_switching", True)))
config["switch_trigger"] = st.sidebar.number_input("스위칭 트리거(%)", -50.0, -1.0, float(config.get("switch_trigger", -12.0)), step=0.5)

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
if st.sidebar.button("📡 /menu 전송(그룹)"):
    tg_send_menu()

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    if get_openai_client(config) is None:
        st.sidebar.error("OpenAI 연결 실패(키/설정 확인)")
    else:
        try:
            resp = get_openai_client(config).chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                max_tokens=10,
                timeout=OPENAI_TIMEOUT
            )
            st.sidebar.success("✅ 연결 성공: " + resp.choices[0].message.content)
        except Exception as e:
            st.sidebar.error(f"❌ 실패: {e}")

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
            for p in act:
                sym = p.get("symbol", "")
                side = position_side_normalize(p)
                roi = float(position_roi_percent(p))
                lev = p.get("leverage", "?")
                upnl = float(p.get("unrealizedPnl") or 0.0)
                st.info(f"**{sym}** ({'🟢롱' if side=='long' else '🔴숏'} x{lev})\n수익률: **{roi:.2f}%** (손익 {upnl:.2f} USDT)")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")


# =========================================================
# ✅ Main UI (기존 유지)
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트 (Final)")
st.caption("Streamlit=제어판/모니터링, Telegram 채널=알림, Telegram 그룹=/menu 조회. (모의투자 IS_SANDBOX=True)")

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
    if ta is None:
        st.error("ta 모듈이 없습니다. requirements.txt에 `ta` 추가 후 재배포하세요.")
    else:
        try:
            ohlcv = fetch_ohlcv_cached(exchange, symbol, config.get("timeframe", "5m"), limit=220, cache_sec=8)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df2, stt, last = calc_indicators(df, config)

            mon_now = read_json_safe(MONITOR_FILE, {}) or {}
            htf_tf = mon_now.get("trend_filter_tf", "1h")
            htf_trend = get_htf_trend_cached(
                exchange, symbol, htf_tf,
                fast=int(config.get("ma_fast", 7)),
                slow=int(config.get("ma_slow", 99)),
                cache_sec=int(config.get("trend_filter_cache_sec", 60)),
            )

            if last is None:
                st.warning("지표 계산 실패(데이터 부족)")
            else:
                st.metric("현재가", f"{float(last['close']):,.4f}")
                show = {
                    "단기추세(현재봉)": stt.get("추세", "-"),
                    f"장기추세({htf_tf})": f"🧭 {htf_trend}",
                    "RSI": stt.get("RSI", "-"),
                    "BB": stt.get("BB", "-"),
                    "MACD": stt.get("MACD", "-"),
                    "ADX": stt.get("ADX", "-"),
                    "거래량": stt.get("거래량", "-"),
                    "눌림목후보(해소)": "✅" if stt.get("_pullback_candidate") else "—",
                }
                st.write(show)

                # 진입 스타일 미리보기
                for dec in ["buy", "sell"]:
                    s_style, s_reason = determine_entry_style(config, stt.get("추세",""), htf_trend, dec)
                    st.caption(f"👉 {('롱' if dec=='buy' else '숏')} 가정 시: {s_style} ({s_reason})")

                if config.get("use_sr_stop", True):
                    try:
                        sr_tf = config.get("sr_timeframe","15m")
                        htf = fetch_ohlcv_cached(exchange, symbol, sr_tf, limit=220, cache_sec=15)
                        hdf = pd.DataFrame(htf, columns=["time","open","high","low","close","vol"])
                        hdf["time"] = pd.to_datetime(hdf["time"], unit="ms")
                        sr = sr_stop_take(
                            entry_price=float(last["close"]),
                            side="buy",
                            htf_df=hdf,
                            atr_period=int(config.get("sr_atr_period",14)),
                            pivot_order=int(config.get("sr_pivot_order",6)),
                            buffer_atr_mult=float(config.get("sr_buffer_atr_mult",0.25)),
                            rr_min=float(config.get("sr_rr_min",1.5)),
                        )
                        if sr:
                            st.caption(f"SR(참고): 롱 기준 TP {sr['tp_price']:.6g} / SL {sr['sl_price']:.6g}")
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")

st.divider()

t1, t2, t3, t4, t5 = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지", "🧪 간이 백테스트"])

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
        st.subheader("🌍 외부 시황 요약(한글)")
        ext = (mon.get("external") or {})
        if not ext or not ext.get("enabled", False):
            st.caption("외부 시황 통합 OFF")
        else:
            st.write({
                "갱신시각(KST)": ext.get("asof_kst"),
                "중요이벤트(임박)": len(ext.get("high_impact_events_soon") or []),
                "공포탐욕": (ext.get("fear_greed") or {}),
                "도미넌스/시총": (ext.get("global") or {}),
            })
            evs = ext.get("high_impact_events_soon") or []
            if evs:
                show_evs = []
                for e in evs[:10]:
                    show_evs.append({
                        "시간(KST)": e.get("time_kst"),
                        "국가": e.get("country"),
                        "중요": e.get("impact"),
                        "이벤트": e.get("title_ko") or e.get("title"),
                    })
                st.warning("⚠️ 중요 이벤트 임박(신규진입 보수적으로)")
                st.dataframe(df_for_display(show_evs), width="stretch", hide_index=True)

            hd = ext.get("headlines_ko") or ext.get("headlines") or []
            if hd:
                st.caption("📰 헤드라인(한글/가능하면)")
                st.write(hd[:10])

        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = (time.time() - hb) if hb else 9999

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        c2.metric("모드", mon.get("trade_mode", "-"))
        c3.metric("레짐", f"{mon.get('trade_style','-')} ({mon.get('style_confidence','-')}%)")
        c4.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < 15 else "🔴 멈춤 의심")
        c5.metric("연속손실", str(mon.get("consec_losses", 0)))

        st.caption(f"레짐 이유: {str(mon.get('style_reason','-'))[:100]}")
        st.caption(f"장기TF: {mon.get('trend_filter_tf','-')}")
        st.caption(f"봇 상태: {mon.get('global_state','-')} | 마지막 액션: {mon.get('last_action',{})}")

        if age >= 60:
            st.error("⚠️ 봇 스레드가 멈췄거나(크래시) 갱신이 안될 수 있어요. (60초 이상)")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0) or 0)
            scan_age = (time.time() - last_scan) if last_scan else 9999
            rows.append({
                "코인": sym,
                "스캔(초전)": f"{scan_age:.1f}",
                "가격": cs.get("price", ""),
                "단기추세": cs.get("trend_short", ""),
                f"장기추세({cs.get('trend_filter_tf','-')})": cs.get("trend_filter", ""),
                "AI결론": str(cs.get("ai_decision", "-")).upper(),
                "확신도": cs.get("ai_confidence", "-"),
                "필요확신도": cs.get("min_conf_required", "-"),
                "진입%": cs.get("ai_entry_pct", "-"),
                "레버": cs.get("ai_leverage", "-"),
                "SL%": cs.get("ai_sl_pct", "-"),
                "TP%": cs.get("ai_tp_pct", "-"),
                "손익비": cs.get("ai_rr", "-"),
                "진입레짐": cs.get("entry_style","-"),
                "레짐이유": cs.get("entry_style_reason","")[:60],
                "스킵/근거": (cs.get("skip_reason") or cs.get("ai_reason_easy") or "")[:120],
            })
        if rows:
            st.dataframe(df_for_display(rows), width="stretch", hide_index=True)
        else:
            st.info("아직 스캔 데이터가 없습니다.")

    st.divider()
    st.subheader("🔍 현재 코인 AI 분석(수동 버튼)")
    if st.button("현재 코인 AI 분석 실행"):
        if get_openai_client(config) is None:
            st.error("OpenAI 키 없음")
        elif ta is None:
            st.error("ta 모듈 없음")
        else:
            try:
                ohlcv = fetch_ohlcv_cached(exchange, symbol, config.get("timeframe", "5m"), limit=220, cache_sec=8)
                df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if last is None:
                    st.warning("지표 계산 실패")
                else:
                    mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                    style_now = mon_now.get("trade_style", "스캘핑")
                    ai = ai_decide_trade(df2, stt, symbol, config.get("trade_mode", "안전모드"), config, trade_style=style_now)
                    st.json(ai)
            except Exception as e:
                st.error(f"분석 오류: {e}")

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
    st.subheader("📅 시장정보(원본 + 한글표시)")
    ext_raw = build_external_context(config)
    ext_ko = build_external_context_korean(config)
    if not ext_raw.get("enabled"):
        st.info("외부 시황 통합 OFF")
    else:
        st.write({"raw": ext_raw, "korean_view": ext_ko})

with t4:
    st.subheader("📜 매매일지 (CSV + 구글시트)")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🔄 새로고침"):
        st.rerun()
    if c2.button("🧹 매매일지 초기화(CSV)"):
        reset_trade_log()
        st.success("매매일지 초기화 완료")
        st.rerun()

    df_log = read_trade_log()
    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        show_cols = [c for c in ["Time","Coin","Side","PnL_Percent","PnL_USDT","Style","Mode","OneLine","Reason","Review","TradeID"] if c in df_log.columns]
        # 이모지 컬럼 추가 표시
        df_show = df_log.copy()
        if "PnL_Percent" in df_show.columns:
            df_show["Emoji"] = df_show["PnL_Percent"].apply(lambda x: pnl_emoji(float(x)))
            show_cols = ["Emoji"] + show_cols
        st.dataframe(df_show[show_cols], width="stretch", hide_index=True)
        csv_bytes = df_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

    st.info("✅ 구글시트 기록은 ‘체결/청산 시점’에 자동 반영됩니다. (연결/공유 필요)")

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
    st.subheader("🧪 간이 백테스트")
    st.caption("⚠️ 실제 주문이 아닌 과거 OHLCV 기반 간이 시뮬레이션입니다. 결과는 참고용입니다.")
    bt_symbol = st.selectbox("심볼", symbol_list, index=0, key="bt_symbol")
    bt_style = st.selectbox("전략 스타일", ["스캘핑", "스윙"], index=0, key="bt_style")
    bt_tf = st.selectbox("타임프레임", ["1m","3m","5m","15m","1h"], index=2, key="bt_tf")
    bt_bars = st.number_input("최근 N봉", 200, 2000, int(config.get("backtest_default_bars", 800)), step=50, key="bt_bars")

    def simple_backtest(df: pd.DataFrame, cfg: Dict[str, Any], style: str) -> Dict[str, Any]:
        if ta is None or df is None or df.empty or len(df) < 120:
            return {"error": "ta 모듈 없음 또는 데이터 부족"}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        rsi = ta.momentum.rsi(close, window=int(cfg.get("rsi_period", 14)))
        ma_fast = ta.trend.sma_indicator(close, window=int(cfg.get("ma_fast", 7)))
        ma_slow = ta.trend.sma_indicator(close, window=int(cfg.get("ma_slow", 99)))
        adx = ta.trend.adx(high, low, close, window=14)
        atr = ta.volatility.average_true_range(high, low, close, window=14)

        df2 = df.copy()
        df2["RSI"] = rsi
        df2["MAF"] = ma_fast
        df2["MAS"] = ma_slow
        df2["ADX"] = adx
        df2["ATR"] = atr
        df2 = df2.dropna()

        if len(df2) < 50:
            return {"error": "지표 계산 후 데이터 부족"}

        rsi_buy = float(cfg.get("rsi_buy", 30))
        rsi_sell = float(cfg.get("rsi_sell", 70))

        in_pos = False
        side = ""
        entry = 0.0
        sl = 0.0
        tp = 0.0
        r_list = []
        equity = 0.0
        peak = 0.0
        mdd = 0.0

        for i in range(2, len(df2)):
            row = df2.iloc[i]
            prev = df2.iloc[i-1]

            trend_up = row["MAF"] > row["MAS"] and row["close"] > row["MAS"]
            trend_dn = row["MAF"] < row["MAS"] and row["close"] < row["MAS"]

            rsi_resolve_long = (prev["RSI"] < rsi_buy) and (row["RSI"] >= rsi_buy)
            rsi_resolve_short = (prev["RSI"] > rsi_sell) and (row["RSI"] <= rsi_sell)

            adx_ok = row["ADX"] >= (22 if style == "스윙" else 18)

            if not in_pos:
                if rsi_resolve_long and (trend_up or style == "스캘핑") and adx_ok:
                    side = "long"
                    entry = row["close"]
                    sl_pct = max(0.25, (row["ATR"] / entry) * (1.2 if style == "스윙" else 0.9) * 100)
                    tp_pct = sl_pct * (2.0 if style == "스윙" else 1.5)
                    sl = entry * (1 - sl_pct / 100)
                    tp = entry * (1 + tp_pct / 100)
                    in_pos = True
                elif rsi_resolve_short and (trend_dn or style == "스캘핑") and adx_ok:
                    side = "short"
                    entry = row["close"]
                    sl_pct = max(0.25, (row["ATR"] / entry) * (1.2 if style == "스윙" else 0.9) * 100)
                    tp_pct = sl_pct * (2.0 if style == "스윙" else 1.5)
                    sl = entry * (1 + sl_pct / 100)
                    tp = entry * (1 - tp_pct / 100)
                    in_pos = True
                continue

            hi = row["high"]
            lo = row["low"]

            exit_price = None
            if side == "long":
                hit_sl = lo <= sl
                hit_tp = hi >= tp
                exit_price = sl if hit_sl else (tp if hit_tp else None)
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp
                exit_price = sl if hit_sl else (tp if hit_tp else None)

            if exit_price is not None:
                risk = abs(entry - sl)
                pnl = (exit_price - entry) if side == "long" else (entry - exit_price)
                r = pnl / max(risk, 1e-9)
                r_list.append(r)
                equity += r
                peak = max(peak, equity)
                mdd = max(mdd, peak - equity)
                in_pos = False

        if not r_list:
            return {"error": "체결된 트레이드가 없습니다"}

        wins = [r for r in r_list if r > 0]
        losses = [r for r in r_list if r <= 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        win_rate = (len(wins) / len(r_list)) * 100.0

        return {
            "trades": len(r_list),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2) if pf != float("inf") else "∞",
            "mdd_r": round(mdd, 2),
            "total_r": round(sum(r_list), 2),
            "avg_r": round(np.mean(r_list), 2),
        }

    if st.button("백테스트 실행"):
        try:
            ohlcv = fetch_ohlcv_cached(exchange, bt_symbol, bt_tf, limit=int(bt_bars), cache_sec=2)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
            result = simple_backtest(df, config, bt_style)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("백테스트 완료")
                st.json(result)
        except Exception as e:
            st.error(f"백테스트 실패: {e}")

st.caption("⚠️ 이 봇은 모의투자(IS_SANDBOX=True)에서 충분히 검증 후 사용하세요.")

# =========================================================
# ✅ 실전 전환 방법 (명시적으로 사용자 변경 필요)
# 1) IS_SANDBOX = False 로 변경
# 2) Streamlit Secrets의 API_KEY/API_SECRET/API_PASSWORD를 실계정 키로 교체
# 3) 실계정 권한/레버리지/주문 최소 수량/위험 관리(손절/익절) 재검증
# 4) 소액으로 테스트 후 단계적으로 증액
# =========================================================
