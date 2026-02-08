# =========================================================
#  Bitget AI Wonyoti Agent (Final Integrated)
#  - Streamlit: 제어판/차트/포지션/일지/AI 시야
#  - Telegram: 실시간 보고/조회/일지 요약
#  - AutoTrade: 데모(IS_SANDBOX=True) 기반
#
#  ⚠️ 주의: 트레이딩은 손실 위험이 큽니다. (특히 레버리지)
# =========================================================

import os
import json
import time
import uuid
import math
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

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

# ===== 추가 PIP(있으면 사용, 없어도 기존 코드 그대로 동작) =====
try:
    import orjson  # 빠른 JSON
except Exception:
    orjson = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter
except Exception:
    retry = None

try:
    from loguru import logger
except Exception:
    logger = None

try:
    from diskcache import Cache
except Exception:
    Cache = None

try:
    import pandas_ta as pta  # 추가 지표/보조 기능
except Exception:
    pta = None

try:
    from scipy.signal import argrelextrema  # 피벗 탐지
except Exception:
    argrelextrema = None

try:
    from pydantic import BaseModel, Field, ValidationError  # AI JSON 안정화
except Exception:
    BaseModel = None
# ---- external context pip ----
try:
    import feedparser  # pip: feedparser
except Exception:
    feedparser = None

try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None

try:
    from tenacity import retry, stop_after_attempt, wait_fixed
except Exception:
    retry = None


# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Final)")

IS_SANDBOX = True  # ✅ 데모/모의투자

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"
BRAIN_DB = "wonyousi_brain.db"  # (선택) 향후 확장

# ===== 추가(상세 일지 저장 폴더) =====
DETAIL_DIR = "trade_details"
os.makedirs(DETAIL_DIR, exist_ok=True)

# ===== 추가(캐시) =====
_cache = Cache("cache") if Cache else None

# 감시 대상 코인
TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

# =========================================================
# ✅ 1) 시간 유틸 (KST, timezone-aware) - DeprecationWarning 제거
# =========================================================
from datetime import datetime, timedelta, timezone
KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")


# =========================================================
# ✅ 2) JSON 안전 저장/로드 (원자적)  (추가: orjson 있으면 자동 사용)
# =========================================================
def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        if orjson:
            b = orjson.dumps(data)
            with open(tmp, "wb") as f:
                f.write(b)
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
# ✅ 2.5) (추가) 상세일지 저장/조회
# =========================================================
def save_trade_detail(trade_id: str, payload: Dict[str, Any]) -> None:
    try:
        path = os.path.join(DETAIL_DIR, f"{trade_id}.json")
        write_json_atomic(path, payload)
    except Exception:
        pass


def load_trade_detail(trade_id: str) -> Optional[Dict[str, Any]]:
    try:
        path = os.path.join(DETAIL_DIR, f"{trade_id}.json")
        return read_json_safe(path, None)
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
# ✅ 3) MODE_RULES (사용자 제공) - 3단계 모드
# =========================================================
MODE_RULES = {
    "안전모드": {
        "min_conf": 85,
        "entry_pct_min": 2,
        "entry_pct_max": 8,
        "lev_min": 2,
        "lev_max": 8,
    },
    "공격모드": {
        "min_conf": 80,
        "entry_pct_min": 8,     # ✅ 공격: 최소 8% ~ 25%
        "entry_pct_max": 25,
        "lev_min": 2,
        "lev_max": 10,          # ✅ 레버는 낮게
    },
    "하이리스크/하이리턴": {
        "min_conf": 85,
        "entry_pct_min": 15,
        "entry_pct_max": 40,
        "lev_min": 8,
        "lev_max": 25,          # ✅ 레버도 높게
    }
}


# =========================================================
# ✅ 4) 설정 관리 (load/save)
# =========================================================
def default_settings() -> Dict[str, Any]:
    return {
        # 공통
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",
        "timeframe": "5m",
        "order_usdt": 100.0,

        # 텔레그램
        "tg_enable_reports": True,

        # ===== 추가: 텔레그램에 진입 근거 길게 보내지 않기(기본 False) =====
        "tg_send_entry_reason": False,

        # 10종 지표 파라미터
        "rsi_period": 14,
        "rsi_buy": 30,
        "rsi_sell": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_fast": 7,
        "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 10종 지표 ON/OFF
        "use_rsi": True,
        "use_bb": True,
        "use_cci": True,
        "use_vol": True,
        "use_ma": True,
        "use_macd": True,
        "use_stoch": True,
        "use_mfi": True,
        "use_willr": True,
        "use_adx": True,

        # 방어/자금/전략 옵션
        # 방어/자금/전략 옵션
        "use_trailing_stop": True,

        "use_dca": True,
        "dca_trigger": -20.0,
        "dca_max_count": 1,
        "dca_add_pct": 50.0,

        # ✅ 스위칭(손실이 커졌는데 반대 시그널 강하면 방향 전환)
        "use_switching": True,
        "switch_trigger": -12.0,          # ROI%가 이 값 이하이면 스위칭 검토
        "switch_entry_pct": 6.0,          # 스위칭 후 재진입은 잔고의 몇 %로 할지
        "switch_cooldown_min": 15,        # 같은 심볼 스위칭 재시도 쿨다운(분)
        "switch_conf_boost": 5,           # 스위칭은 min_conf + boost 이상일 때만

        "no_trade_weekend": False,

        # ✅ 외부 이벤트 블랙아웃(중요 이벤트 임박 시 신규 진입 제한)
        "macro_blackout_minutes": 30,
        "macro_blackout_action": "skip",  # "skip" 또는 "reduce"
        "macro_reduce_entry_mult": 0.5,   # reduce일 때 진입비중 배수
        "macro_reduce_lev_mult": 0.7,     # reduce일 때 레버 배수

        # ===== 추가: 지지/저항(SR) 기반 손절/익절 =====
        "use_sr_stop": True,

        # ✅ 멀티 타임프레임 SR (여러 TF를 합성해서 더 강한 SR로 SL/TP 잡기)
        "sr_timeframes": ["15m", "1h", "4h"],  # 기존 sr_timeframe 단일 대신 합성(우선)
        "sr_timeframe": "15m",                 # (호환용) sr_timeframes 없으면 이걸 사용

        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,
        "sr_rr_min": 1.5,

        "no_trade_weekend": False,

        # 연속손실/일시정지
        "loss_pause_enable": True,
        "loss_pause_after": 3,        # 연속 3번 손실이면
        "loss_pause_minutes": 30,     # 30분 정지

        # AI 추천 글로벌옵션
        "ai_reco_show": True,
        "ai_reco_apply": False,  # ✅ ON이면 AI 추천값을 자동으로 config에 반영
        "ai_reco_refresh_sec": 20,  # 추천 갱신 주기(너무 잦으면 비용/지연)

        # AI 출력 쉬운말(한글)
        "ai_easy_korean": True,

                # 🌍 외부 시황 통합
        "use_external_context": True,
        "macro_blackout_minutes": 30,      # 중요 이벤트 전후 신규진입 줄이기(분)
        "external_refresh_sec": 60,        # 외부시황 갱신 주기
        "news_enable": True,
        "news_refresh_sec": 300,
        "news_max_headlines": 12,


        # ===== 추가: 지지/저항(SR) 기반 손절/익절 =====
        "use_sr_stop": True,
        "sr_timeframe": "15m",
        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,   # 지지/저항 이탈 버퍼
        "sr_rr_min": 1.5,             # SR 기반 TP 계산 시 최소 RR
    }


def load_settings() -> Dict[str, Any]:
    cfg = default_settings()
    if os.path.exists(SETTINGS_FILE):
        saved = read_json_safe(SETTINGS_FILE, {})
        if isinstance(saved, dict):
            cfg.update(saved)
    # 예전 키 이름 호환
    if "openai_key" in cfg and not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = cfg["openai_key"]
    return cfg


def save_settings(cfg: Dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, cfg)


config = load_settings()


# =========================================================
# ✅ 5) 런타임 상태(runtime_state.json) - 사용자 포맷 유지
# =========================================================
def default_runtime() -> Dict[str, Any]:
    return {
        "date": today_kst_str(),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {}  # 심볼별 dca 횟수 등 저장
    }


def load_runtime() -> Dict[str, Any]:
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()
    # 날짜 rollover
    if rt.get("date") != today_kst_str():
        rt = default_runtime()
    # 필드 보정
    for k, v in default_runtime().items():
        if k not in rt:
            rt[k] = v
    return rt


def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ 6) 매매일지 CSV (상세 저장 + 한줄평 + 후기)
# =========================================================
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
    trade_id: str = ""
) -> None:
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
        }])

        if not os.path.exists(LOG_FILE):
            row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
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
            lines.append(f"- {r.get('Coin','?')} {r.get('Side','?')} {float(r.get('PnL_Percent',0)):.2f}% 손실 | 이유: {str(r.get('Reason',''))[:40]}")
        return "\n".join(lines) if lines else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# =========================================================
# ✅ 7) Secrets (Bitget / Telegram / OpenAI)
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. Secrets에 API_KEY/API_SECRET/API_PASSWORD 설정하세요.")
    st.stop()

# ✅ OpenAI 클라이언트(전역) - 스레드에서도 사용
openai_client = None

def init_openai_client():
    global openai_client
    key = st.secrets.get("OPENAI_API_KEY") or load_settings().get("openai_api_key", "")
    if not key:
        openai_client = None
        return None
    try:
        openai_client = OpenAI(api_key=key)
        return openai_client
    except Exception:
        openai_client = None
        return None

# 최초 1회 생성
init_openai_client()


# =========================================================
# ✅ (추가) OpenAI 클라이언트는 '스레드에서도 최신 키'를 쓰도록 유틸로 제공
# =========================================================
_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}


def get_openai_client(cfg: Dict[str, Any]) -> Optional[OpenAI]:
    key = st.secrets.get("OPENAI_API_KEY", cfg.get("openai_api_key", ""))
    if not key:
        return None
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
# ✅ 9) Bitget 헬퍼 (포지션/잔고/수량 정밀)
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
    return max(lo, min(hi, v))


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
    # pos_side: long/short OR buy/sell
    if contracts <= 0:
        return False
    if pos_side in ["long", "buy"]:
        return market_order_safe(ex, sym, "sell", contracts)
    return market_order_safe(ex, sym, "buy", contracts)


def position_roi_percent(p: Dict[str, Any]) -> float:
    # ccxt 포지션 dict에서 ROI% 가져오거나 계산
    try:
        if p.get("percentage") is not None:
            return float(p.get("percentage"))
    except Exception:
        pass
    return 0.0


def position_side_normalize(p: Dict[str, Any]) -> str:
    # bitget/ccxt는 side가 long/short 또는 buy/sell로 올 수 있음
    s = (p.get("side") or p.get("positionSide") or "").lower()
    if s in ["long", "buy"]:
        return "long"
    if s in ["short", "sell"]:
        return "short"
    # fallback
    return "long"


# =========================================================
# ✅ 9.5) (추가) SR(지지/저항) 기반 손절/익절 계산
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
    """
    supports, resistances
    """
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

    return {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "atr": float(atr),
        "supports": supports,
        "resistances": resistances,
    }


# =========================================================
# ✅ 10) TradingView 다크모드 차트
# =========================================================
def tv_symbol_from_ccxt(sym: str) -> str:
    # BTC/USDT:USDT -> BITGET:BTCUSDT.P (가능하면)
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

    # parameters
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

    # RSI
    if cfg.get("use_rsi", True):
        df["RSI"] = ta.momentum.rsi(close, window=rsi_period)

    # Bollinger
    if cfg.get("use_bb", True):
        bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()

    # MA
    if cfg.get("use_ma", True):
        df["MA_fast"] = ta.trend.sma_indicator(close, window=ma_fast)
        df["MA_slow"] = ta.trend.sma_indicator(close, window=ma_slow)

    # MACD
    if cfg.get("use_macd", True):
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

    # Stoch
    if cfg.get("use_stoch", True):
        df["STO_K"] = ta.momentum.stoch(high, low, close, window=stoch_k, smooth_window=3)
        df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=stoch_k, smooth_window=3)

    # CCI
    if cfg.get("use_cci", True):
        df["CCI"] = ta.trend.cci(high, low, close, window=20)

    # MFI
    if cfg.get("use_mfi", True):
        df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)

    # Williams %R
    if cfg.get("use_willr", True):
        df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # ADX
    if cfg.get("use_adx", True):
        df["ADX"] = ta.trend.adx(high, low, close, window=14)

    # Volume spike
    if cfg.get("use_vol", True):
        df["VOL_MA"] = vol.rolling(20).mean()
        df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * vol_mul)).astype(int)

    # ===== (추가) pandas-ta가 있으면 참고용 지표를 더 계산할 수 있음 (기존 기능 변경 X) =====
    # (지표를 더 추가로 계산하되, 기존 판단 로직은 그대로 유지)
    if pta is not None:
        try:
            # 예시: ATR(참고)
            df["ATR_ref"] = pta.atr(df["high"], df["low"], df["close"], length=14)
        except Exception:
            pass

    df = df.dropna()
    if df.empty or len(df) < 5:
        return df, status, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- status text (Korean) ----
    used = []

    # RSI status
    if cfg.get("use_rsi", True):
        used.append("RSI")
        rsi_now = float(last.get("RSI", 50))
        if rsi_now < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_now:.1f})"
        elif rsi_now > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_now:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_now:.1f})"

    # Bollinger
    if cfg.get("use_bb", True):
        used.append("볼린저밴드")
        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

    # MA trend
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

    # volume
    if cfg.get("use_vol", True):
        used.append("거래량")
        status["거래량"] = "🔥 거래량 급증" if int(last.get("VOL_SPIKE", 0)) == 1 else "⚪ 보통"

    # ---- 핵심: “과매도에 바로 진입” 방지 -> “해소 시점(반등/반락 확인)” ----
    rsi_prev = float(prev.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0
    rsi_now = float(last.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0

    rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
    rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

    # 눌림목 후보: 상승추세 + 과매도 해소 + (ADX 너무 약하지 않음)
    adx_now = float(last.get("ADX", 0)) if cfg.get("use_adx", True) else 0.0
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_now >= 18)

    status["_used_indicators"] = used
    status["_rsi_resolve_long"] = bool(rsi_resolve_long)
    status["_rsi_resolve_short"] = bool(rsi_resolve_short)
    status["_pullback_candidate"] = bool(pullback_candidate)

    return df, status, last


# =========================================================
# ✅ 12) AI 판단 + 리스크 매니저(ATR/스윙 기반 SL/TP 자동보정) + 외부시황(공포탐욕/이벤트)
# - 목표: 레버가 높을수록 SL/TP(ROI%)가 자동으로 넓어져 휩쏘 손절 반복을 줄임
# - 외부시황: 공포/탐욕 지수 + 고중요 이벤트(이번주 캘린더에서 High만 일부)
# - 주의: 외부 요청은 캐시로 최소화(기본 60초)
# =========================================================

_EXT_CACHE = {"ts": 0.0, "data": {}}

def _fear_greed_kr(v: int) -> str:
    # Alternative.me 기준 구간
    if v <= 25:
        return "극공포(패닉 구간)"
    if v <= 45:
        return "공포(조심 구간)"
    if v <= 55:
        return "중립(보통 구간)"
    if v <= 75:
        return "탐욕(과열 주의)"
    return "극탐욕(과열/변동성 주의)"


def _fetch_fear_greed() -> Dict[str, Any]:
    """
    공포/탐욕 지수(Alternative.me)
    실패 시 빈 dict 반환
    """
    try:
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        r = requests.get(url, timeout=8)
        j = r.json()
        v = int(j["data"][0]["value"])
        cls = str(j["data"][0].get("value_classification", ""))
        ts = str(j["data"][0].get("timestamp", ""))
        return {
            "value": v,
            "label_kr": _fear_greed_kr(v),
            "label_en": cls,
            "timestamp": ts,
        }
    except Exception:
        return {}


def _fetch_high_impact_events(limit: int = 6) -> List[Dict[str, Any]]:
    """
    ForexFactory 이번주 JSON에서 impact=High만 일부 추려서 제공
    - 시간대/포맷이 환경마다 다를 수 있어 '참고용'으로만 사용
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=8)
        data = r.json()
        out = []
        for x in data:
            if x.get("impact") != "High":
                continue
            out.append({
                "date": x.get("date", ""),
                "time": x.get("time", ""),
                "country": x.get("country", ""),
                "title": x.get("title", ""),
                "impact_kr": "매우 중요",
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def get_external_context_cached(refresh_sec: int = 60) -> Dict[str, Any]:
    """
    외부시황 스냅샷(캐시)
    refresh_sec 지나면 갱신 시도
    """
    now = time.time()
    if (now - float(_EXT_CACHE.get("ts", 0))) < refresh_sec and isinstance(_EXT_CACHE.get("data"), dict):
        return _EXT_CACHE["data"]

    snap = {
        "fng": _fetch_fear_greed(),
        "high_impact_events": _fetch_high_impact_events(limit=6),
        "updated_kst": now_kst_str(),
    }
    _EXT_CACHE["ts"] = now
    _EXT_CACHE["data"] = snap
    return snap


def _atr_price_pct(df: pd.DataFrame, window: int = 14) -> float:
    """ATR을 가격 대비 %로 반환 (예: 0.45%)"""
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
    """
    최근 스윙 저점/고점 기준으로 "가격 손절폭%" 추정
    - buy(롱): 최근 N봉 최저가 아래로 버퍼
    - sell(숏): 최근 N봉 최고가 위로 버퍼
    """
    try:
        if df is None or df.empty or len(df) < lookback + 5:
            return 0.0
        recent = df.tail(lookback)
        last_close = float(df["close"].iloc[-1])
        atr_pct = _atr_price_pct(df, 14)
        buf_pct = atr_pct * buffer_atr_mul  # ATR의 일부를 버퍼로

        if decision == "buy":
            swing = float(recent["low"].min())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 - buf_pct / 100.0)
            dist_pct = ((last_close - stop_price) / last_close) * 100.0
            return max(0.0, dist_pct)

        if decision == "sell":
            swing = float(recent["high"].max())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 + buf_pct / 100.0)
            dist_pct = ((stop_price - last_close) / last_close) * 100.0
            return max(0.0, dist_pct)

        return 0.0
    except Exception:
        return 0.0


def _rr_min_by_mode(mode: str) -> float:
    # 모드별 최소 손익비(“손절 짧게 / 익절 길게” 방향)
    if mode == "안전모드":
        return 1.8
    if mode == "공격모드":
        return 2.1
    return 2.6  # 하이리스크/하이리턴


def _risk_guardrail(out: Dict[str, Any], df: pd.DataFrame, decision: str, mode: str, external: Dict[str, Any]) -> Dict[str, Any]:
    """
    out(sl_pct,tp_pct,leverage,rr)을 '휩쏘에 안 잘릴 정도'로 자동 보정
    - 핵심: SL/TP는 ROI%가 아니라 "가격 변동폭%"을 기준으로 잡고 ROI로 변환
    - 외부시황 반영(완만): 극공포면 보수적으로(손절 약간 넓게, 레버는 AI가 정한 범위 내에서만)
    """
    lev = max(1, int(out.get("leverage", 1)))
    sl_roi = float(out.get("sl_pct", 1.2))
    tp_roi = float(out.get("tp_pct", 3.0))
    rr = float(out.get("rr", 0))

    # 현재 out이 암시하는 가격 손절폭(%) = ROI손절 / 레버
    sl_price_pct_now = sl_roi / max(lev, 1)

    # 변동성 기반 최소 가격 손절폭(휩쏘 방지)
    atr_pct = _atr_price_pct(df, 14)
    min_price_stop = max(0.25, atr_pct * 0.9)  # 5m 기준 최소 0.25% 또는 ATR의 0.9배

    # 스윙 기준 손절폭도 고려
    swing_stop = _swing_stop_price_pct(df, decision, lookback=40, buffer_atr_mul=0.25)
    if swing_stop > 0:
        swing_stop = min(swing_stop, max(min_price_stop * 3.0, atr_pct * 3.0))
    recommended_price_stop = max(min_price_stop, swing_stop)

    notes = []

    # ✅ 외부시황(공포탐욕)로 '약간' 보정: 극공포면 손절폭을 조금 더 여유(휩쏘 방지)
    try:
        fng = (external or {}).get("fng", {}) or {}
        fng_v = int(fng.get("value", -1)) if fng.get("value") is not None else -1
        if 0 <= fng_v <= 25:
            recommended_price_stop = max(recommended_price_stop, min_price_stop * 1.2)
            notes.append("외부시황: 극공포라 휩쏘 대비 손절 여유 추가")
    except Exception:
        pass

    # ✅ 레버가 높은데 가격손절폭이 너무 작으면 SL 확장
    if sl_price_pct_now < recommended_price_stop:
        sl_price_pct_now = recommended_price_stop
        sl_roi = sl_price_pct_now * lev
        notes.append(f"손절폭(가격기준)을 변동성/스윙에 맞게 확장({recommended_price_stop:.2f}%)")

    # ✅ 손익비 최소치 확보: TP가 SL*RRmin 보다 작으면 TP를 올림
    rr_min = _rr_min_by_mode(mode)
    if rr <= 0:
        rr = max(rr_min, tp_roi / max(sl_roi, 0.01))

    if tp_roi < sl_roi * rr_min:
        tp_roi = sl_roi * rr_min
        notes.append(f"손익비 최소 {rr_min:.1f} 확보하도록 익절 상향")

    rr = max(rr, tp_roi / max(sl_roi, 0.01))

    # 결과 저장
    out["sl_pct"] = float(sl_roi)
    out["tp_pct"] = float(tp_roi)
    out["rr"] = float(rr)

    # 디버그/표시용(가격 기준도 같이 기록)
    out["sl_price_pct"] = float(sl_roi / max(lev, 1))
    out["tp_price_pct"] = float(tp_roi / max(lev, 1))
    out["risk_note"] = " / ".join(notes) if notes else "보정 없음"
    return out


def ai_decide_trade(
    df: pd.DataFrame,
    status: Dict[str, Any],
    symbol: str,
    mode: str,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:

    if openai_client is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "OpenAI 키 없음", "used_indicators": status.get("_used_indicators", [])}

    if df is None or df.empty or status is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "데이터 부족", "used_indicators": status.get("_used_indicators", [])}

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    past_mistakes = get_past_mistakes_text(5)

    # ✅ 외부시황 스냅샷(캐시)
    ext_refresh = int(cfg.get("ai_reco_refresh_sec", 20))  # 이미 cfg에 있는 값을 재활용
    external = get_external_context_cached(refresh_sec=max(20, min(ext_refresh * 3, 180)))

    features = {
        "symbol": symbol,
        "mode": mode,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)) if "RSI" in df.columns else None,
        "rsi_now": float(last.get("RSI", 50)) if "RSI" in df.columns else None,
        "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
        "trend": status.get("추세", ""),
        "bb": status.get("BB", ""),
        "macd": status.get("MACD", ""),
        "vol": status.get("거래량", ""),
        "rsi_resolve_long": bool(status.get("_rsi_resolve_long", False)),
        "rsi_resolve_short": bool(status.get("_rsi_resolve_short", False)),
        "pullback_candidate": bool(status.get("_pullback_candidate", False)),
        "atr_price_pct": _atr_price_pct(df, 14),
        "external": external,  # ✅ 여기서 외부시황을 AI에게 전달
    }

    # 외부시황을 시스템 프롬프트에도 명시(“참고해서 판단”하도록)
    fng_txt = ""
    try:
        fng = (external or {}).get("fng", {}) or {}
        if fng:
            fng_txt = f"- 공포탐욕지수: {int(fng.get('value', -1))}점 / {fng.get('label_kr','')}"
    except Exception:
        fng_txt = ""

    ev_txt = ""
    try:
        evs = (external or {}).get("high_impact_events", []) or []
        if evs:
            # 너무 길지 않게 3개만
            top3 = evs[:3]
            ev_txt = "- 중요 이벤트(참고): " + " | ".join([f"{e.get('country','')} {e.get('title','')}" for e in top3])
    except Exception:
        ev_txt = ""

    sys = f"""
너는 '워뇨띠 스타일(눌림목/해소 타이밍) + 손익비' 기반의 자동매매 트레이더 AI다.

[과거 실수(요약)]
{past_mistakes}

[외부 시황(참고)]
{fng_txt}
{ev_txt}

[핵심 룰]
1) RSI 과매도/과매수 "상태"에 즉시 진입하지 말고, '해소되는 시점'에서만 진입 후보.
2) 상승추세에서는 롱 우선, 하락추세에서는 숏 우선. (역추세는 매우 신중)
3) 모드 규칙은 반드시 준수:
   - 최소 확신도: {rule["min_conf"]}
   - 진입 비중(%): {rule["entry_pct_min"]}~{rule["entry_pct_max"]}
   - 레버리지: {rule["lev_min"]}~{rule["lev_max"]}

[중요]
- sl_pct / tp_pct는 "ROI%"(레버 반영 수익률)로 출력한다.
- 변동성(atr_price_pct)이 작으면 손절을 너무 타이트하게 잡지 마라.
- 외부시황이 '극공포'면 휩쏘/변동성 리스크를 고려해 신중(확신/손절/익절 설계)해라.
- 영어 금지. 쉬운 한글(괄호로 뜻 추가).
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
  "reason_easy": "쉬운 한글(괄호로 의미 추가)"
}}
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
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

        # ✅ (핵심) 리스크 매니저로 SL/TP 자동 보정 (+ 외부시황 일부 반영)
        if out["decision"] in ["buy", "sell"]:
            out = _risk_guardrail(out, df, out["decision"], mode, external)

        # ✅ 외부시황 스냅샷도 결과에 같이 포함(디버그/표시용)
        out["external_used"] = {
            "fng": (external or {}).get("fng", {}),
            "high_impact_events": (external or {}).get("high_impact_events", [])[:3],
            "updated_kst": (external or {}).get("updated_kst", ""),
        }

        return out

    except Exception as e:
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": status.get("_used_indicators", [])}


# =========================================================
# ✅ 13) AI 회고(후기) 작성 (청산 시 일지에 저장)
# =========================================================
def ai_write_review(
    symbol: str,
    side: str,
    pnl_percent: float,
    reason: str,
    cfg: Dict[str, Any]
) -> Tuple[str, str]:
    """
    return: (one_line, review_long)
    """
    client = get_openai_client(cfg)
    if client is None:
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "OpenAI 키 없음 - 후기 자동작성 불가")

    sys = """
너는 매매 회고를 아주 쉽게 써주는 코치다.
출력은 반드시 JSON만.
영어 금지. 초보도 이해하도록 쉬운 한글로.
"""

    user = f"""
상황:
- 코인: {symbol}
- 포지션: {side}
- 결과: {pnl_percent:.2f}%
- 청산 이유: {reason}

JSON 형식:
{{
  "one_line": "한줄평(아주 짧게)",
  "review": "후기(손절이면 다음에 어떻게 개선할지 / 익절이면 다음에 무엇을 유지할지)"
}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        out = json.loads(resp.choices[0].message.content)
        one = str(out.get("one_line", ""))[:120]
        rev = str(out.get("review", ""))[:800]
        return one, rev
    except Exception:
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")


# =========================================================
# ✅ 14) 경제 캘린더 (한글)
# =========================================================
def get_forex_events_kr(limit: int = 80) -> pd.DataFrame:
    """
    ForexFactory JSON(이번주) 불러와서 한글로 표기.
    네트워크 제한/실패 시 빈 DF.
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        rows = []
        for x in data[:limit]:
            impact = x.get("impact", "")
            imp_kr = {"High": "매우 중요", "Medium": "중요", "Low": "낮음"}.get(impact, impact)
            rows.append({
                "날짜": x.get("date", ""),
                "시간": x.get("time", ""),
                "국가": x.get("country", ""),
                "지표": x.get("title", ""),
                "중요도": imp_kr,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["날짜", "시간", "국가", "지표", "중요도"])


# =========================================================
# ✅ 15) 모니터 상태 (AI 시야/하트비트)
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
# ✅ X) 외부 시황 통합(거시/심리/레짐/뉴스) - 데모용
# =========================================================
_ext_cache = TTLCache(maxsize=4, ttl=60) if TTLCache else None

def _safe_get_json(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def fetch_fear_greed():
    # Alternative.me Fear & Greed (public)
    data = _safe_get_json("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
    if not data or "data" not in data or not data["data"]:
        return None
    d0 = data["data"][0]
    try:
        return {
            "value": int(d0.get("value", 0)),
            "classification": str(d0.get("value_classification", "")),
            "timestamp": str(d0.get("timestamp", "")),
        }
    except Exception:
        return None

def fetch_coingecko_global():
    # CoinGecko global (public)
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

def fetch_upcoming_high_impact_events(within_minutes: int = 30, limit: int = 80):
    # ForexFactory weekly JSON (네가 이미 쓰는 소스)
    data = _safe_get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
    if not isinstance(data, list):
        return []
    now = now_kst()
    out = []
    for x in data[:limit]:
        try:
            impact = str(x.get("impact", ""))
            if impact != "High":
                continue
            # date가 ISO8601(+offset)로 오는 케이스가 많음
            dt_str = str(x.get("date", ""))
            dt = None
            try:
                dt = datetime.fromisoformat(dt_str)
                # dt가 tz-aware면 KST로 변환
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
    # 너무 많이 말고 “헤드라인만”
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
    # 중복 제거
    uniq = []
    seen = set()
    for t in items:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq[:max_items]

def build_external_context(cfg: dict):
    """
    외부시황을 '요약 가능한 형태'로 묶어서 반환
    (스레드 멈춤 방지 위해 timeout + 실패해도 None/[] 리턴)
    """
    if not cfg.get("use_external_context", True):
        return {"enabled": False}

    # 캐시(스레드가 계속 도는 구조라, 이거 없으면 외부요청 과다로 멈출 수 있음)
    if _ext_cache is not None and "ext" in _ext_cache:
        return _ext_cache["ext"]

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
        "high_impact_events_soon": high_events,  # 리스트(0개면 안전)
        "fear_greed": fg,                        # None 가능
        "global": cg,                            # None 가능
        "headlines": headlines,                  # [] 가능
        "asof_kst": now_kst_str()
    }

    if _ext_cache is not None:
        _ext_cache["ext"] = ext
    return ext

# =========================================================
# ✅ 16) 텔레그램 유틸 (추가: retry 있으면 적용)
# =========================================================
def _tg_post(url: str, data: Dict[str, Any]):
    if retry is None:
        return requests.post(url, data=data, timeout=10)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
    def _do():
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        return r
    return _do()


def tg_send(text: str):
    if not tg_token or not tg_id:
        return
    try:
        _tg_post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            {"chat_id": tg_id, "text": text},
        )
    except Exception:
        pass


def tg_send_menu():
    if not tg_token or not tg_id:
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
            {"chat_id": tg_id, "text": "✅ 메뉴 갱신\n(일지상세: '일지상세 <ID>')", "reply_markup": json.dumps(kb, ensure_ascii=False)},
        )
    except Exception:
        pass


def tg_answer_callback(cb_id: str):
    if not tg_token:
        return
    try:
        _tg_post(
            f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery",
            {"callback_query_id": cb_id},
        )
    except Exception:
        pass


# =========================================================
# ✅ 17) 자동매매 핵심 스레드 (24시간 모니터 + 매매 + 일지 + 시야)
# =========================================================
def telegram_thread(ex):
    offset = 0
    mon = monitor_init()

    tg_send("🚀 AI 봇 가동 시작! (모의투자)\n명령: 상태 / 시야 / 일지 / 일지상세 <ID>")
    tg_send_menu()

    # active_targets: 심볼별 목표/정보 저장
    # (추가: trade_id / SR가격 기반 sl_price,tp_price 저장)
    active_targets: Dict[str, Dict[str, Any]] = {}

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()
            mode = cfg.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
                        # 🌍 외부 시황 갱신 (AI 시야/의사결정에 반영)
            ext = build_external_context(cfg)
            mon["external"] = ext


            # ✅ 하트비트
            mon["last_heartbeat_epoch"] = time.time()
            mon["last_heartbeat_kst"] = now_kst_str()
            mon["auto_trade"] = bool(cfg.get("auto_trade", False))
            mon["trade_mode"] = mode
            mon["pause_until"] = rt.get("pause_until", 0)
            mon["consec_losses"] = rt.get("consec_losses", 0)

            # ✅ 자동매매 ON일 때만 스캔/매매
            if cfg.get("auto_trade", False):
                # 주말 거래 금지 옵션
                if cfg.get("no_trade_weekend", False):
                    wd = now_kst().weekday()  # 0=월 ... 5=토 6=일
                    if wd in [5, 6]:
                        mon["global_state"] = "주말 거래 OFF"
                        monitor_write_throttled(mon, 2.0)
                        time.sleep(2.0)
                        continue

                # 일시정지(연속손실)
                if cfg.get("loss_pause_enable", True) and time.time() < float(rt.get("pause_until", 0)):
                    mon["global_state"] = "일시정지 중(연속손실/보호)"
                    monitor_write_throttled(mon, 2.0)
                    time.sleep(1.0)
                else:
                    mon["global_state"] = "스캔/매매 중"

                    # 1) 포지션 관리 (손절/익절/트레일링/DCA/스위칭)
                    for sym in TARGET_COINS:
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if not act:
                            continue

                        p = act[0]
                        side = position_side_normalize(p)  # long/short
                        contracts = float(p.get("contracts") or 0)
                        entry = float(p.get("entryPrice") or 0)
                        roi = float(position_roi_percent(p))
                        cur_px = get_last_price(ex, sym) or entry

                        # 목표가: active_targets에 없으면 fallback
                        tgt = active_targets.get(sym, {
                            "sl": 2.0,     # 손절(%) 기준
                            "tp": 5.0,     # 익절(%) 기준
                            "entry_usdt": 0.0,
                            "entry_pct": 0.0,
                            "lev": p.get("leverage", "?"),
                            "reason": "",
                            "trade_id": "",
                            "sl_price": None,
                            "tp_price": None,
                        })
                        sl = float(tgt.get("sl", 2.0))
                        tp = float(tgt.get("tp", 5.0))

                        sl_price = tgt.get("sl_price")
                        tp_price = tgt.get("tp_price")
                        trade_id = str(tgt.get("trade_id") or "")

                        # ✅ 트레일링: 절반 익절 도달하면 손절을 당겨서 수익보호(기존 로직 유지)
                        # ✅ 트레일링: "가격 변동폭 기준"으로만 조여서 휩쏘 방지
                        if cfg.get("use_trailing_stop", True):
                            # 목표 절반 도달 시, 손절을 '너무 타이트하지 않게' 올려서 수익 보호
                            if roi >= (tp * 0.5):
                                lev_now = float(tgt.get("lev", p.get("leverage", 1))) or 1.0
                                # entry 때 계산된 가격 손절폭이 있으면 그걸 기준으로, 없으면 현재 SL/레버로 추정
                                base_price_sl = float(tgt.get("sl_price_pct", max(0.25, float(sl) / max(lev_now, 1))))
                                # 트레일링은 원래 손절폭의 60% 정도로만 조임(너무 꽉 조이면 휩쏘)
                                trail_price_pct = max(0.20, base_price_sl * 0.60)
                                trail_roi = trail_price_pct * lev_now
                        
                                # sl은 "허용 손실폭"이므로 더 작아지면 더 타이트해짐 → min으로 조이되, 너무 작게는 금지
                                sl = min(sl, max(1.2, float(trail_roi)))  # 최소 -1.2% ROI 이하로는 안 조임


                        # ✅ (추가) SR 기반 가격 트리거가 있으면 우선 체크
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

                        # ✅ DCA (물타기): 손실이 일정 수준 이하일 때 1회 추가 진입 (기존 유지)
                        if cfg.get("use_dca", True):
                            dca_trig = float(cfg.get("dca_trigger", -20.0))
                            dca_max = int(cfg.get("dca_max_count", 1))
                            dca_add_pct = float(cfg.get("dca_add_pct", 50.0))

                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0})
                            dca_count = int(trade_state.get("dca_count", 0))

                            if roi <= dca_trig and dca_count < dca_max:
                                free, total = safe_fetch_balance(ex)
                                base_entry = float(tgt.get("entry_usdt", 0.0))
                                add_usdt = base_entry * (dca_add_pct / 100.0)
                                if add_usdt > free:
                                    add_usdt = free * 0.5

                                px = cur_px
                                if px and add_usdt > 5:
                                    lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                    set_leverage_safe(ex, sym, lev)
                                    qty = to_precision_qty(ex, sym, (add_usdt * lev) / px)
                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                    if ok:
                                        trade_state["dca_count"] = dca_count + 1
                                        save_runtime(rt)
                                        tg_send(f"💧 물타기(DCA)\n- 코인: {sym}\n- 추가금: {add_usdt:.2f} USDT\n- 이유: 손실 {roi:.2f}% (기준 {dca_trig}%)")
                                        mon["last_action"] = {"time_kst": now_kst_str(), "type": "DCA", "symbol": sym, "roi": roi}
                                        monitor_write_throttled(mon, 0.2)

                        # ===== 손절 조건: (추가) SR 가격 트리거 OR (기존) ROI 손절 =====
                        do_stop = hit_sl_by_price or (roi <= -abs(sl))
                        do_take = hit_tp_by_price or (roi >= tp)

                        # ✅ 손절
                        if do_stop:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                one, review = ai_write_review(sym, side, roi, "자동 손절(지지/저항 이탈 또는 목표 손절)", cfg)
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 손절", one_line=one, review=review, trade_id=trade_id)

                                # (추가) 상세일지 업데이트
                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update({
                                        "exit_time": now_kst_str(),
                                        "exit_price": exit_px,
                                        "pnl_usdt": pnl_usdt_snapshot,
                                        "pnl_pct": roi,
                                        "result": "SL",
                                        "review": review,
                                    })
                                    save_trade_detail(trade_id, d)

                                # 연속손실 증가 및 일시정지 조건
                                rt["consec_losses"] = int(rt.get("consec_losses", 0)) + 1
                                if cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(f"🛑 연속손실 보호\n- 연속손실: {rt['consec_losses']}회\n- {int(cfg.get('loss_pause_minutes',30))}분 자동 정지")
                                save_runtime(rt)

                                # (추가) 텔레그램: USDT 손익/현재잔고까지 표시
                                tg_send(
                                    f"🩸 손절\n"
                                    f"- 코인: {sym}\n"
                                    f"- 수익률: {roi:.2f}% (손익 {pnl_usdt_snapshot:.2f} USDT)\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 현재잔고: {total_after:.2f} USDT (사용가능 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 이탈' if hit_sl_by_price else '목표 손절 도달'}\n"
                                    f"- 한줄평: {one}\n"
                                    f"- 일지ID: {trade_id or '없음'}"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "STOP", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                        # ✅ 익절
                        elif do_take:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                one, review = ai_write_review(sym, side, roi, "자동 익절(지지/저항 목표 또는 목표 익절)", cfg)
                                log_trade(sym, side, entry, exit_px, pnl_usdt_snapshot, roi, "자동 익절", one_line=one, review=review, trade_id=trade_id)

                                # (추가) 상세일지 업데이트
                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update({
                                        "exit_time": now_kst_str(),
                                        "exit_price": exit_px,
                                        "pnl_usdt": pnl_usdt_snapshot,
                                        "pnl_pct": roi,
                                        "result": "TP",
                                        "review": review,
                                    })
                                    save_trade_detail(trade_id, d)

                                rt["consec_losses"] = 0
                                save_runtime(rt)

                                tg_send(
                                    f"🎉 익절\n"
                                    f"- 코인: {sym}\n"
                                    f"- 수익률: +{roi:.2f}% (손익 {pnl_usdt_snapshot:.2f} USDT)\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 현재잔고: {total_after:.2f} USDT (사용가능 {free_after:.2f})\n"
                                    f"- 이유: {'지지/저항 목표 도달' if hit_tp_by_price else '목표 익절 도달'}\n"
                                    f"- 한줄평: {one}\n"
                                    f"- 일지ID: {trade_id or '없음'}"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "TAKE", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                    # 2) 신규 진입 스캔
                    free_usdt, total_usdt = safe_fetch_balance(ex)

                    for sym in TARGET_COINS:
                        # 이미 포지션 있으면 스킵
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if act:
                            continue

                        # 쿨다운
                        cd = float(rt.get("cooldowns", {}).get(sym, 0))
                        if time.time() < cd:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "쿨다운(잠깐 쉬는중)"
                            continue

                        # 데이터 로드
                        try:
                            ohlcv = ex.fetch_ohlcv(sym, cfg.get("timeframe", "5m"), limit=220)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except Exception as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = f"데이터 실패: {e}"
                            continue

                        df, stt, last = calc_indicators(df, cfg)
                        mon.setdefault("coins", {}).setdefault(sym, {})
                        cs = mon["coins"][sym]

                        if last is None:
                            cs.update({
                                "last_scan_kst": now_kst_str(),
                                "ai_called": False,
                                "skip_reason": "지표 계산 실패(ta/데이터 부족)"
                            })
                            continue

                        # 모니터 기록(지표/상태)
                        cs.update({
                            "last_scan_epoch": time.time(),
                            "last_scan_kst": now_kst_str(),
                            "price": float(last["close"]),
                            "trend": stt.get("추세", ""),
                            "rsi": float(last.get("RSI", 0)) if "RSI" in df.columns else None,
                            "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
                            "bb": stt.get("BB", ""),
                            "macd": stt.get("MACD", ""),
                            "vol": stt.get("거래량", ""),
                            "pullback_candidate": bool(stt.get("_pullback_candidate", False)),
                        })

                        # ✅ AI 호출 필터 (기존 유지)
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
                            cs["skip_reason"] = "횡보/해소 신호 없음(휩쏘 위험)"
                            monitor_write_throttled(mon, 1.0)
                            continue

                        # AI 판단
                        ai = ai_decide_trade(df, stt, sym, mode, cfg)
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))

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
                            "min_conf_required": int(rule["min_conf"]),
                            "skip_reason": ""
                        })
                        monitor_write_throttled(mon, 1.0)
                        
                        # ✅ 강제 방향 필터: 하락추세면 롱 금지, 상승추세면 숏 금지 (역추세 방지)
                        trend_txt = (stt.get("추세", "") or "")
                        is_down = ("하락" in trend_txt)
                        is_up = ("상승" in trend_txt)
                        
                        if is_down and decision == "buy":
                            cs["skip_reason"] = "하락추세라 롱 금지(역추세 방지)"
                            continue
                        
                        if is_up and decision == "sell":
                            cs["skip_reason"] = "상승추세라 숏 금지(역추세 방지)"
                            continue

                        # 진입 조건
                        if decision in ["buy", "sell"] and conf >= int(rule["min_conf"]):
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

                                # ===== (추가) SR 기반 SL/TP 가격도 함께 계산해서 저장 =====
                                sl_price = None
                                tp_price = None
                                if cfg.get("use_sr_stop", True):
                                    try:
                                        sr_tf = cfg.get("sr_timeframe", "15m")
                                        htf = ex.fetch_ohlcv(sym, sr_tf, limit=220)
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

                                # 목표 저장(기존 + 추가)
                                active_targets[sym] = {
                                    "sl": slp, "tp": tpp,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "reason": ai.get("reason_easy", ""),
                                    "trade_id": trade_id,
                                    "sl_price": sl_price,
                                    "tp_price": tp_price,
                                }

                                # (추가) 상세일지 저장(진입 근거는 여기로)
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
                                })

                                # 쿨다운 60초
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)

                                # 텔레그램 보고(기존 + 추가: 잔고/일지ID / 근거는 옵션)
                                if cfg.get("tg_enable_reports", True):
                                    direction = "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"
                                    msg = (
                                        f"🎯 진입\n"
                                        f"- 코인: {sym}\n"
                                        f"- 방향: {direction}\n"
                                        f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%)\n"
                                        f"- 레버리지: x{lev}\n"
                                        f"- 목표익절: +{tpp:.2f}% / 목표손절: -{slp:.2f}%\n"
                                    )
                                    if sl_price is not None and tp_price is not None:
                                        msg += f"- SR기준가: TP {tp_price:.6g} / SL {sl_price:.6g}\n"
                                    msg += (
                                        f"- 확신도: {conf}% (기준 {rule['min_conf']}%)\n"
                                        f"- 일지ID: {trade_id}\n"
                                    )
                                    # 근거 길게 전송 여부
                                    if cfg.get("tg_send_entry_reason", False):
                                        msg += (
                                            f"- 근거(쉬운말): {ai.get('reason_easy','')[:220]}\n"
                                            f"- AI가 본 지표: {', '.join(ai.get('used_indicators', []))}\n"
                                        )
                                    tg_send(msg)

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
                                }
                                monitor_write_throttled(mon, 0.2)

                                time.sleep(1.0)

                        time.sleep(0.4)

            # =================================================
            # 텔레그램 수신 처리 (텍스트 명령 / 콜백 버튼)
            # =================================================
            try:
                res = requests.get(
                    f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1",
                    timeout=10
                ).json()
            except Exception:
                res = {"ok": False}

            if res.get("ok"):
                for up in res.get("result", []):
                    offset = up.get("update_id", offset)

                    # 텍스트 명령
                    if "message" in up and "text" in up["message"]:
                        txt = up["message"]["text"].strip()

                        if txt == "상태":
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt = load_runtime()
                            tg_send(
                                f"📡 상태\n"
                                f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 잔고: {total:.2f} USDT (사용가능 {free:.2f})\n"
                                f"- 연속손실: {rt.get('consec_losses',0)}\n"
                                f"- 정지해제: {('정지중' if time.time() < float(rt.get('pause_until',0)) else '정상')}\n"
                            )

                        elif txt == "시야":
                            mon_now = read_json_safe(MONITOR_FILE, {})
                            coins = mon_now.get("coins", {}) or {}
                            lines = [
                                "👁️ AI 시야(요약)",
                                f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                                f"- 모드: {mon_now.get('trade_mode','-')}",
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                            ]
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                    f"/ RSI {cs.get('rsi','-')} / ADX {cs.get('adx','-')} "
                                    f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:30]}"
                                )
                            tg_send("\n".join(lines))

                        elif txt == "일지":
                            df_log = read_trade_log()
                            if df_log.empty:
                                tg_send("📜 일지 없음(아직 기록된 매매가 없어요)")
                            else:
                                top = df_log.head(8)
                                msg = ["📜 최근 매매일지(요약)"]
                                for _, r in top.iterrows():
                                    tid = str(r.get("TradeID","") or "")
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {float(r['PnL_Percent']):.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}")
                                tg_send("\n".join(msg))

                        # ===== (추가) 상세일지 조회 =====
                        elif txt.startswith("일지상세"):
                            parts = txt.split()
                            if len(parts) < 2:
                                tg_send("사용법: 일지상세 <ID>\n(예: 일지상세 a1b2c3d4e5)")
                            else:
                                tid = parts[1].strip()
                                d = load_trade_detail(tid)
                                if not d:
                                    tg_send("해당 ID를 찾지 못했어요.")
                                else:
                                    tg_send(
                                        f"🧾 일지상세 {tid}\n"
                                        f"- 코인: {d.get('coin')}\n"
                                        f"- 방향: {d.get('decision')}\n"
                                        f"- 확신도: {d.get('confidence')}\n"
                                        f"- 진입가: {d.get('entry_price')}\n"
                                        f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                        f"- 레버: x{d.get('lev')}\n"
                                        f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                        f"- 한줄근거: {str(d.get('reason_easy',''))[:200]}\n"
                                        f"- 사용지표: {', '.join(d.get('used_indicators', []))[:200]}\n"
                                    )

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up["callback_query"]
                        data = cb.get("data", "")
                        cb_id = cb.get("id", "")

                        if data == "status":
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt = load_runtime()
                            tg_send(
                                f"📡 상태\n"
                                f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 잔고: {total:.2f} USDT (사용가능 {free:.2f})\n"
                                f"- 연속손실: {rt.get('consec_losses',0)}\n"
                            )

                        elif data == "vision":
                            mon_now = read_json_safe(MONITOR_FILE, {})
                            coins = mon_now.get("coins", {}) or {}
                            lines = [
                                "👁️ AI 시야(요약)",
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                            ]
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                    f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                                )
                            tg_send("\n".join(lines))

                        elif data == "balance":
                            free, total = safe_fetch_balance(ex)
                            tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT")

                        elif data == "position":
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
                                    msg.append(f"- {sym}: {('롱' if side=='long' else '숏')} (수익률 {roi:.2f}%, 손익 {upnl:.2f} USDT)")
                            if not has:
                                msg.append("- 없음(관망)")
                            tg_send("\n".join(msg))

                        elif data == "log":
                            df_log = read_trade_log()
                            if df_log.empty:
                                tg_send("📜 일지 없음")
                            else:
                                top = df_log.head(8)
                                msg = ["📜 최근 매매일지(요약)"]
                                for _, r in top.iterrows():
                                    tid = str(r.get("TradeID","") or "")
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {float(r['PnL_Percent']):.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}")
                                tg_send("\n".join(msg))

                        elif data == "log_detail_help":
                            tg_send("🧾 일지상세 사용법\n- 일지상세 <ID>\n예) 일지상세 a1b2c3d4e5\n(최근 ID는 '일지'에서 확인)")

                        elif data == "close_all":
                            tg_send("🛑 전량 청산 시도")
                            for sym in TARGET_COINS:
                                ps = safe_fetch_positions(ex, [sym])
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    continue
                                p = act[0]
                                side = position_side_normalize(p)
                                contracts = float(p.get("contracts") or 0)
                                close_position_market(ex, sym, side, contracts)
                            tg_send("✅ 전량 청산 요청 완료")

                        tg_answer_callback(cb_id)

            monitor_write_throttled(mon, 2.0)
            time.sleep(0.8)

        except Exception as e:
            tg_send(f"⚠️ 스레드 오류: {e}")
            time.sleep(3.0)


# =========================================================
# ✅ 18) 스레드 시작 (중복 방지)
# =========================================================
def ensure_thread_started():
    for t in threading.enumerate():
        if t.name == "TG_THREAD":
            return
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(th)
    th.start()


ensure_thread_started()


# =========================================================
# ✅ 19) Streamlit UI
# =========================================================
st.sidebar.title("🛠️ 제어판")
st.sidebar.caption("Streamlit은 제어/상태 확인용, Telegram은 실시간 보고/조회용")

# OpenAI 키 입력(선택)
if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)

        # ✅ 추가: 즉시 전역 클라이언트 재생성 (스레드도 바로 사용 가능)
        init_openai_client()

        st.rerun()


with st.sidebar.expander("🧪 디버그: 저장된 설정(bot_settings.json) 확인"):
    st.json(read_json_safe(SETTINGS_FILE, {}))


# 모드 선택 (MODE_RULES 기반)
mode_keys = list(MODE_RULES.keys())
safe_mode = config.get("trade_mode", "안전모드")
if safe_mode not in mode_keys:
    safe_mode = "안전모드"
config["trade_mode"] = st.sidebar.selectbox("매매 모드", mode_keys, index=mode_keys.index(safe_mode))

# 자동매매 ON/OFF
auto_on = st.sidebar.checkbox("🤖 자동매매 (텔레그램 연동)", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

# 기본 옵션
st.sidebar.divider()
config["timeframe"] = st.sidebar.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))
config["tg_enable_reports"] = st.sidebar.checkbox("📨 텔레그램 보고 활성화", value=bool(config.get("tg_enable_reports", True)))
config["use_trailing_stop"] = st.sidebar.checkbox("🚀 트레일링 스탑(수익보호)", value=bool(config.get("use_trailing_stop", True)))

# ===== (추가) 텔레그램 근거 전송 토글 =====
config["tg_send_entry_reason"] = st.sidebar.checkbox("📌 텔레그램에 진입근거(긴글)도 보내기", value=bool(config.get("tg_send_entry_reason", False)))

st.sidebar.divider()
st.sidebar.subheader("🧱 지지/저항(SR) 손절/익절(추가)")
config["use_sr_stop"] = st.sidebar.checkbox("SR 기반 가격 손절/익절 사용", value=bool(config.get("use_sr_stop", True)))
c_sr1, c_sr2 = st.sidebar.columns(2)
config["sr_timeframe"] = c_sr1.selectbox("SR 타임프레임", ["5m","15m","1h","4h"], index=["5m","15m","1h","4h"].index(config.get("sr_timeframe","15m")))
config["sr_pivot_order"] = c_sr2.number_input("피벗 민감도", 3, 10, int(config.get("sr_pivot_order", 6)))
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
config["use_dca"] = st.sidebar.checkbox("💧 물타기(DCA)", value=bool(config.get("use_dca", True)))
c3, c4 = st.sidebar.columns(2)
config["dca_trigger"] = c3.number_input("DCA 발동(%)", -90.0, -1.0, float(config.get("dca_trigger", -20.0)), step=0.5)
config["dca_max_count"] = c4.number_input("최대 횟수", 0, 10, int(config.get("dca_max_count", 1)))
config["dca_add_pct"] = st.sidebar.slider("추가 규모(원진입 대비 %)", 10, 200, int(config.get("dca_add_pct", 50)))

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
if st.sidebar.button("📡 텔레그램 메뉴 전송"):
    tg_send_menu()

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    if get_openai_client(config) is None:
        st.sidebar.error("OpenAI 연결 실패(키/설정 확인)")
    else:
        try:
            resp = get_openai_client(config).chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                max_tokens=10
            )
            st.sidebar.success("✅ 연결 성공: " + resp.choices[0].message.content)
        except Exception as e:
            st.sidebar.error(f"❌ 실패: {e}")

# 설정 저장
save_settings(config)

# =========================================================
# ✅ Sidebar: 잔고/포지션 현황
# =========================================================
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
# ✅ Main UI: 차트/지표/탭
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트 (Final)")
st.caption("Streamlit=제어판/모니터링, Telegram=실시간 보고/조회. (모의투자 IS_SANDBOX=True)")

# 코인 선택
markets = exchange.markets or {}
if markets:
    symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
    if not symbol_list:
        symbol_list = TARGET_COINS
else:
    symbol_list = TARGET_COINS

symbol = st.selectbox("코인 선택", symbol_list, index=0)

# 상단 레이아웃
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
            ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df2, stt, last = calc_indicators(df, config)

            if last is None:
                st.warning("지표 계산 실패(데이터 부족)")
            else:
                st.metric("현재가", f"{float(last['close']):,.4f}")
                show = {
                    "RSI": stt.get("RSI", "-"),
                    "BB": stt.get("BB", "-"),
                    "MACD": stt.get("MACD", "-"),
                    "ADX": stt.get("ADX", "-"),
                    "추세": stt.get("추세", "-"),
                    "거래량": stt.get("거래량", "-"),
                    "눌림목후보(해소)": "✅" if stt.get("_pullback_candidate") else "—",
                }
                st.write(show)

                # ===== (추가) SR 기준 TP/SL 미리보기 =====
                if config.get("use_sr_stop", True):
                    try:
                        sr_tf = config.get("sr_timeframe","15m")
                        htf = exchange.fetch_ohlcv(symbol, sr_tf, limit=220)
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

# 탭
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("👁️ 실시간 AI 모니터링(봇 시야)")
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, key="mon_refresh")  # 2초
    else:
        st.caption("자동 새로고침을 원하면 requirements.txt에 streamlit-autorefresh 추가하세요.")
        st.subheader("🌍 외부 시황 요약")
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
                st.warning("⚠️ 중요 이벤트 임박(신규진입 보수적으로)")
                st.dataframe(pd.DataFrame(evs), width="stretch", hide_index=True)
            hd = ext.get("headlines") or []
            if hd:
                st.caption("뉴스 헤드라인(요약용)")
                st.write(hd[:10])

        st.button("🔄 수동 새로고침")

    mon = read_json_safe(MONITOR_FILE, None)
    if not mon:
        st.warning("monitor_state.json이 아직 없습니다. (스레드 시작 확인)")
    else:
        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = (time.time() - hb) if hb else 9999

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        c2.metric("모드", mon.get("trade_mode", "-"))
        c3.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < 6 else "🔴 멈춤 의심")
        c4.metric("연속손실", str(mon.get("consec_losses", 0)))

        if age >= 6:
            st.error("⚠️ 봇 스레드가 멈췄거나(크래시) 갱신이 안될 수 있어요.")

        st.caption(f"봇 상태: {mon.get('global_state','-')} | 마지막 액션: {mon.get('last_action',{})}")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0) or 0)
            scan_age = (time.time() - last_scan) if last_scan else 9999
            rows.append({
                "코인": sym,
                "스캔(초전)": f"{scan_age:.1f}",
                "가격": cs.get("price", ""),
                "추세": cs.get("trend", ""),
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
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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
                ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if last is None:
                    st.warning("지표 계산 실패")
                else:
                    ai = ai_decide_trade(df2, stt, symbol, config.get("trade_mode", "안전모드"), config)
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
    st.subheader("📅 시장정보 (경제 캘린더)")
    ev = get_forex_events_kr()
    if ev.empty:
        st.info("일정 없음/불러오기 실패(네트워크 제한일 수 있음)")
    else:
        st.dataframe(ev, width="stretch", hide_index=True)

with t4:
    st.subheader("📜 매매일지 (보기 쉽게 + 초기화)")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🔄 새로고침"):
        st.rerun()
    if c2.button("🧹 매매일지 초기화"):
        reset_trade_log()
        st.success("매매일지 초기화 완료")
        st.rerun()

    df_log = read_trade_log()
    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        show_cols = [c for c in ["Time", "Coin", "Side", "PnL_Percent", "PnL_USDT", "OneLine", "Reason", "Review", "TradeID"] if c in df_log.columns]
        st.dataframe(df_log[show_cols], width="stretch", hide_index=True)

        csv_bytes = df_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

    # ===== (추가) 상세일지 조회 UI =====
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

st.caption("⚠️ 이 봇은 모의투자(IS_SANDBOX=True)에서 충분히 검증 후 사용하세요.")
