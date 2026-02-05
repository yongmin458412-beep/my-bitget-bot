# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
import time
import requests
import threading
import os
import json
import uuid
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from openai import OpenAI

# Streamlit thread ctx
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Optional libs (pip 추가했으면 자동 사용됨)
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
import orjson
from diskcache import Cache

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

try:
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False

try:
    import pandas_ta as pta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False

try:
    from scipy.signal import argrelextrema
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================================================
# ✅ 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Ultimate Integration)")

IS_SANDBOX = True  # 실전: False
SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"
DETAIL_DIR = "trade_details"
LOG_DIR = "logs"
CACHE_DIR = "cache"
DB_FILE = "wonyousi_brain.db"

os.makedirs(DETAIL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_cache = Cache(CACHE_DIR)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(os.path.join(LOG_DIR, "app.log"), rotation="1 MB", retention="10 days", level="INFO")

# 감시 코인(기본 5개)
TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT"
]

# =========================================================
# ✅ 모드 규칙(사용자가 준 값)
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
        "entry_pct_min": 8,
        "entry_pct_max": 25,
        "lev_min": 2,
        "lev_max": 10,
    },
    "하이리스크/하이리턴": {
        "min_conf": 85,
        "entry_pct_min": 15,
        "entry_pct_max": 40,
        "lev_min": 8,
        "lev_max": 25,
    }
}


# =========================================================
# ✅ 유틸 (시간/JSON/텔레그램/안전 호출)
# =========================================================
def now_utc():
    return datetime.now(timezone.utc)

def now_kst():
    # timezone-aware KST
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))

def read_json_safe(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception as e:
        logger.warning(f"read_json_safe fail {path}: {e}")
        return default

def write_json_atomic(path: str, obj) -> bool:
    try:
        tmp = f"{path}.tmp.{uuid.uuid4().hex}"
        data = orjson.dumps(obj)
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error(f"write_json_atomic fail {path}: {e}")
        return False

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
def http_get(url, **kwargs):
    r = requests.get(url, timeout=kwargs.pop("timeout", 10), **kwargs)
    r.raise_for_status()
    return r

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
def http_post(url, **kwargs):
    r = requests.post(url, timeout=kwargs.pop("timeout", 10), **kwargs)
    r.raise_for_status()
    return r


# =========================================================
# ✅ 설정 관리
# =========================================================
def default_settings():
    return {
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",

        "leverage": 10,
        "order_usdt": 100.0,

        # 지표
        "use_rsi": True,
        "use_bb": True,
        "use_cci": True,
        "use_vol": True,
        "use_ma": True,
        "use_macd": False,
        "use_stoch": False,
        "use_mfi": False,
        "use_willr": False,
        "use_adx": True,

        "rsi_period": 14,
        "rsi_buy": 30,
        "rsi_sell": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_fast": 7,
        "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 방어/자금관리
        "use_switching": True,
        "use_dca": True,
        "dca_trigger": -20.0,
        "dca_max_count": 1,

        # AI 옵션
        "ai_apply_global": True,   # AI 추천값을 자동 적용(모드 범위 내)
        "rr_min_safe": 1.6,
        "rr_min_aggr": 1.4,
        "rr_min_hr": 1.3,
        "sr_tf": "15m",           # SR 계산 타임프레임
        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,

        # 보고/시야
        "vision_interval_sec": 3,
        "scan_interval_sec": 2,
        "report_interval_sec": 900,

        # 경제캘린더
        "econ_calendar_region": "US",  # US/KR/EU 등
    }

def load_settings():
    cfg = default_settings()
    saved = read_json_safe(SETTINGS_FILE, {})
    if isinstance(saved, dict):
        cfg.update(saved)
    return cfg

def save_settings(cfg: dict):
    ok = write_json_atomic(SETTINGS_FILE, cfg)
    if ok:
        st.toast("✅ 설정 저장 완료", icon="💾")
    else:
        st.error("설정 저장 실패(파일 권한/경로 확인)")

config = load_settings()
if "order_usdt" not in st.session_state:
    st.session_state["order_usdt"] = float(config.get("order_usdt", 100.0))


# =========================================================
# ✅ 런타임 상태(runtime_state.json)
# =========================================================
def runtime_default():
    return {
        "date": now_kst().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0.0,
        "cooldowns": {},
        "trades": {}
    }

def load_runtime():
    rt = runtime_default()
    saved = read_json_safe(RUNTIME_FILE, {})
    if isinstance(saved, dict):
        rt.update(saved)
    # 날짜 넘어가면 자동 초기화(일별)
    if rt.get("date") != now_kst().strftime("%Y-%m-%d"):
        rt = runtime_default()
        write_json_atomic(RUNTIME_FILE, rt)
    return rt

def save_runtime(rt: dict):
    write_json_atomic(RUNTIME_FILE, rt)

def reset_runtime_and_logs():
    # 일지 초기화 버튼용
    if os.path.exists(RUNTIME_FILE):
        os.remove(RUNTIME_FILE)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    # 상세일지 폴더 비우기
    try:
        for f in os.listdir(DETAIL_DIR):
            if f.endswith(".json"):
                os.remove(os.path.join(DETAIL_DIR, f))
    except Exception:
        pass
    write_json_atomic(RUNTIME_FILE, runtime_default())


# =========================================================
# ✅ 일지(로그) 저장: 한줄평 + 상세 JSON
# =========================================================
def save_trade_detail(trade_id: str, payload: dict):
    path = os.path.join(DETAIL_DIR, f"{trade_id}.json")
    write_json_atomic(path, payload)

def load_trade_detail(trade_id: str):
    path = os.path.join(DETAIL_DIR, f"{trade_id}.json")
    return read_json_safe(path, None)

def list_recent_trade_ids(limit: int = 10):
    files = [f for f in os.listdir(DETAIL_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(DETAIL_DIR, x)), reverse=True)
    ids = [os.path.splitext(f)[0] for f in files[:limit]]
    return ids

def log_trade_csv(coin, side, entry_price, exit_price, pnl_usdt, pnl_pct, one_line, trade_id):
    try:
        now = now_kst().strftime("%Y-%m-%d %H:%M:%S")
        row = pd.DataFrame([{
            "Time": now,
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_usdt,
            "PnL_Percent": pnl_pct,
            "OneLine": one_line,
            "TradeID": trade_id
        }])
        if not os.path.exists(LOG_FILE):
            row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"log_trade_csv error: {e}")

def get_past_mistakes():
    try:
        if not os.path.exists(LOG_FILE):
            return "과거 매매 기록 없음."
        df = pd.read_csv(LOG_FILE)
        if df.empty or "PnL_Percent" not in df.columns:
            return "과거 매매 기록 없음."
        worst = df.sort_values(by="PnL_Percent", ascending=True).head(5)
        out = []
        for _, r in worst.iterrows():
            out.append(f"- {r.get('Coin','?')} {r.get('Side','?')} {r.get('PnL_Percent',0)}% (한줄: {str(r.get('OneLine',''))})")
        return "\n".join(out) if out else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# =========================================================
# ✅ SR 기반 손절/익절(지지/저항 이탈 + ATR 버퍼)
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

def pivot_levels(df: pd.DataFrame, order: int = 6, max_levels: int = 10):
    if df is None or df.empty or len(df) < order * 4:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    if _HAS_SCIPY:
        hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    else:
        hi_idx = []
        lo_idx = []
        for i in range(order, len(df) - order):
            if highs[i] == np.max(highs[i - order:i + order + 1]):
                hi_idx.append(i)
            if lows[i] == np.min(lows[i - order:i + order + 1]):
                lo_idx.append(i)
        hi_idx = np.array(hi_idx, dtype=int)
        lo_idx = np.array(lo_idx, dtype=int)

    resistances = sorted(list(set(highs[hi_idx].round(8))), reverse=True)[:max_levels]
    supports = sorted(list(set(lows[lo_idx].round(8))))[:max_levels]
    return supports, resistances

def sr_stop_take(entry_price: float, side: str, htf_df: pd.DataFrame,
                 atr_period: int = 14, pivot_order: int = 6,
                 buffer_atr_mult: float = 0.25, rr_min: float = 1.5):
    if htf_df is None or htf_df.empty:
        return None
    atr = calc_atr(htf_df, atr_period)
    supports, resistances = pivot_levels(htf_df, pivot_order)

    buf = atr * buffer_atr_mult if atr > 0 else (entry_price * 0.0015)  # fallback

    if side == "buy":
        below = [s for s in supports if s < entry_price]
        if not below:
            sl = entry_price - max(buf, entry_price * 0.003)
        else:
            sl = max(below) - buf
        risk = entry_price - sl
        if risk <= 0:
            return None
        above_r = [r for r in resistances if r > entry_price]
        tp_candidate = min(above_r) if above_r else None
        tp_by_rr = entry_price + risk * rr_min
        tp = tp_candidate if (tp_candidate and tp_candidate > tp_by_rr) else tp_by_rr
    else:
        above = [r for r in resistances if r > entry_price]
        if not above:
            sl = entry_price + max(buf, entry_price * 0.003)
        else:
            sl = min(above) + buf
        risk = sl - entry_price
        if risk <= 0:
            return None
        below_s = [s for s in supports if s < entry_price]
        tp_candidate = max(below_s) if below_s else None
        tp_by_rr = entry_price - risk * rr_min
        tp = tp_candidate if (tp_candidate and tp_candidate < tp_by_rr) else tp_by_rr

    return {
        "sl_price": float(sl),
        "tp_price": float(tp),
        "atr": float(atr),
        "supports": supports,
        "resistances": resistances
    }


# =========================================================
# ✅ 지표 계산 (ta 기반 + optional pandas-ta 확장)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: dict):
    """
    df: columns [time, open, high, low, close, vol]
    return df, status(dict), last(row)
    """
    try:
        if df is None or df.empty or len(df) < 60:
            return df, {}, None

        # 기본 지표 계산(ta)
        if _HAS_TA:
            # RSI
            if cfg.get("use_rsi", True):
                df["RSI"] = ta.momentum.rsi(df["close"], window=int(cfg.get("rsi_period", 14)))

            # BB
            if cfg.get("use_bb", True):
                bb = ta.volatility.BollingerBands(
                    df["close"], window=int(cfg.get("bb_period", 20)), window_dev=float(cfg.get("bb_std", 2.0))
                )
                df["BB_upper"] = bb.bollinger_hband()
                df["BB_lower"] = bb.bollinger_lband()
                df["BB_mid"] = bb.bollinger_mavg()

            # MA
            if cfg.get("use_ma", True):
                df["MA_fast"] = ta.trend.sma_indicator(df["close"], window=int(cfg.get("ma_fast", 7)))
                df["MA_slow"] = ta.trend.sma_indicator(df["close"], window=int(cfg.get("ma_slow", 99)))

            # ADX
            if cfg.get("use_adx", True):
                df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

            # MACD
            if cfg.get("use_macd", False):
                macd = ta.trend.MACD(df["close"])
                df["MACD"] = macd.macd()
                df["MACD_signal"] = macd.macd_signal()
        else:
            # ta가 없으면 최소한만 (앱이 안죽게)
            df["RSI"] = np.nan
            df["ADX"] = np.nan

        # pandas-ta 추가(있으면 더 계산)
        if _HAS_PANDAS_TA:
            if cfg.get("use_stoch", False):
                stoch = pta.stoch(df["high"], df["low"], df["close"], k=int(cfg.get("stoch_k", 14)))
                if stoch is not None:
                    for c in stoch.columns:
                        df[c] = stoch[c]
            if cfg.get("use_mfi", False):
                df["MFI"] = pta.mfi(df["high"], df["low"], df["close"], df["vol"])
            if cfg.get("use_willr", False):
                df["WILLR"] = pta.willr(df["high"], df["low"], df["close"])
            if cfg.get("use_cci", True):
                df["CCI"] = pta.cci(df["high"], df["low"], df["close"])

        # NaN drop
        df = df.dropna()
        if df.empty:
            return df, {}, None

        last = df.iloc[-1]
        status = {}

        # RSI 상태
        rsi = safe_float(last.get("RSI", np.nan), np.nan)
        if np.isnan(rsi):
            status["RSI"] = "정보없음"
        else:
            if rsi > cfg.get("rsi_sell", 70):
                status["RSI"] = "🔴 과매수"
            elif rsi < cfg.get("rsi_buy", 30):
                status["RSI"] = "🟢 과매도"
            else:
                status["RSI"] = "⚪ 중립"

        # BB
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            if last["close"] > last["BB_upper"]:
                status["BB"] = "🔴 상단 돌파"
            elif last["close"] < last["BB_lower"]:
                status["BB"] = "🟢 하단 이탈"
            else:
                status["BB"] = "⚪ 밴드 내"

        # ADX
        adx = safe_float(last.get("ADX", np.nan), np.nan)
        if np.isnan(adx):
            status["ADX"] = "정보없음"
        else:
            status["ADX"] = "🔥 추세장" if adx >= 25 else "💤 횡보장"

        # MA
        if "MA_fast" in df.columns and "MA_slow" in df.columns:
            status["MA"] = "📈 상승(단기>장기)" if last["MA_fast"] > last["MA_slow"] else "📉 하락(단기<장기)"

        # MACD
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            status["MACD"] = "📈 골든(상승 신호)" if last["MACD"] > last["MACD_signal"] else "📉 데드(하락 신호)"

        # 거래량
        if cfg.get("use_vol", True):
            v = safe_float(last.get("vol", 0.0))
            status["VOL"] = f"거래량 {v:,.0f}"

        return df, status, last

    except Exception as e:
        logger.error(f"calc_indicators error: {e}")
        return df, {}, None


# =========================================================
# ✅ AI 응답 스키마 (깨진 JSON 방지)
# =========================================================
class AIDecision(BaseModel):
    decision: str = Field("hold", description="buy/sell/hold")
    confidence: int = Field(0, ge=0, le=100)
    percentage: float = Field(10.0, ge=0.5, le=100.0)  # 잔고 대비 %
    leverage: int = Field(5, ge=1, le=50)
    rr_min: float = Field(1.5, ge=0.5, le=10.0)
    buffer_atr_mult: float = Field(0.25, ge=0.0, le=3.0)
    used_indicators: list[str] = Field(default_factory=list)
    one_line: str = Field("", description="한줄평(쉬운말)")
    reason: str = Field("", description="상세 근거(일지 저장용)")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ai_client_from_cfg(cfg: dict):
    key = st.secrets.get("OPENAI_API_KEY", cfg.get("openai_api_key", ""))
    if not key:
        return None, ""
    return OpenAI(api_key=key), key


def pick_rr_by_mode(mode: str, cfg: dict):
    if mode == "안전모드":
        return float(cfg.get("rr_min_safe", 1.6))
    if mode == "공격모드":
        return float(cfg.get("rr_min_aggr", 1.4))
    return float(cfg.get("rr_min_hr", 1.3))


def generate_ai_plan(df: pd.DataFrame, status: dict, cfg: dict, coin: str):
    """
    - 과매도 '진입'이 아니라 과매도 '해소(반등)'를 더 높게 점수 주도록 유도
    - 모드 규칙은 후처리 clamp로 강제(안전/공격/하이리스크)
    - 진입근거는 텔레그램에 길게 안 보내고 일지에 저장
    """
    client, _ = ai_client_from_cfg(cfg)
    if client is None or df is None or df.empty:
        return AIDecision()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    past = get_past_mistakes()
    mode = cfg.get("trade_mode", "안전모드")
    rules = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    rr_target = pick_rr_by_mode(mode, cfg)

    # 지표 요약(짧게)
    rsi_prev = safe_float(prev.get("RSI", np.nan), np.nan)
    rsi_now = safe_float(last.get("RSI", np.nan), np.nan)
    adx = safe_float(last.get("ADX", np.nan), np.nan)
    ma = status.get("MA", "")
    bb = status.get("BB", "")
    macd = status.get("MACD", "")

    system_prompt = f"""
너는 선물 트레이딩 자동매매의 '의사결정 AI'야.
목표:
- 손실은 짧게, 수익은 추세가 맞으면 길게(손익비 확보)
- 과매도/과매수 '진입'이 아니라, 과매도/과매수에서 '돌아서는 타이밍(해소/반등/반락)'을 더 높게 평가
- 노이즈(휩쏘) 손절을 줄이기 위해 확실한 구조 변화에 가산점

규칙:
- 응답은 반드시 JSON 하나로만.
- decision: buy/sell/hold
- confidence: 0~100
- percentage: 잔고 대비 진입비중(%) -> 모드 규칙 범위 내 추천
- leverage: 레버리지 -> 모드 규칙 범위 내 추천
- rr_min: 최소 손익비(권장 {rr_target})
- buffer_atr_mult: 지지/저항 손절 버퍼(ATR 배수)
- used_indicators: 이번 판단에 실제로 참고한 지표 리스트
- one_line: 아주 쉬운 한줄평(한국어)
- reason: 상세 근거(한국어, 길어도 됨. 텔레그램에는 저장만)

[모드: {mode}]
- 최소 확신도: {rules["min_conf"]}
- 진입비중 범위: {rules["entry_pct_min"]}~{rules["entry_pct_max"]}%
- 레버리지 범위: {rules["lev_min"]}~{rules["lev_max"]}

[과거 손실 사례(반복 실수 방지)]
{past}
"""

    user_prompt = f"""
[코인] {coin}
[현재가] {safe_float(last.get("close",0))}
[RSI 흐름] {rsi_prev:.2f} -> {rsi_now:.2f}
[ADX] {adx:.2f}
[상태요약] RSI:{status.get("RSI","")} / BB:{bb} / MA:{ma} / MACD:{macd}

힌트:
- "우상향 추세 + RSI 과매도"는 눌림목일 수 있음.
- 하지만 '과매도 구간' 그 자체로 진입하지 말고, RSI가 바닥에서 돌아서는지(해소/반등) 확인이 있으면 점수↑
- 추세가 약하면(ADX 낮음) 무리한 진입/고배율 금지

JSON으로만 답해.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        # 스키마 검증
        try:
            plan = AIDecision(**data)
        except ValidationError:
            plan = AIDecision()

        # 모드 규칙 clamp
        plan.confidence = int(clamp(plan.confidence, 0, 100))
        plan.percentage = float(clamp(plan.percentage, rules["entry_pct_min"], rules["entry_pct_max"]))
        plan.leverage = int(clamp(plan.leverage, rules["lev_min"], rules["lev_max"]))
        plan.rr_min = float(plan.rr_min) if plan.rr_min else rr_target
        plan.buffer_atr_mult = float(plan.buffer_atr_mult) if plan.buffer_atr_mult is not None else float(cfg.get("sr_buffer_atr_mult", 0.25))

        # 최소 손익비는 모드 기본 이상으로 살짝 강제(너 목표 반영)
        plan.rr_min = max(plan.rr_min, rr_target)

        # decision 정리
        d = (plan.decision or "hold").lower().strip()
        if d not in ("buy", "sell", "hold"):
            d = "hold"
        plan.decision = d

        return plan

    except Exception as e:
        logger.warning(f"AI plan error: {e}")
        return AIDecision()


# =========================================================
# ✅ 거래소 연결
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except Exception as e:
        logger.error(f"init_exchange fail: {e}")
        return None

exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키/권한/네트워크를 확인하세요.")
    st.stop()

# =========================================================
# ✅ 잔고/포지션 안전 조회
# =========================================================
def safe_fetch_balance(ex):
    try:
        bal = ex.fetch_balance({"type": "swap"})
        free = safe_float(bal.get("USDT", {}).get("free", 0))
        total = safe_float(bal.get("USDT", {}).get("total", 0))
        return free, total
    except Exception:
        return 0.0, 0.0

def safe_fetch_positions(ex, symbols):
    try:
        ps = ex.fetch_positions(symbols=symbols)
        return ps if ps else []
    except Exception:
        return []

def position_summary_korean(p):
    # p: ccxt position dict
    sym = str(p.get("symbol", "")).split(":")[0]
    side_raw = str(p.get("side", "")).lower()
    # bitget ccxt에서는 side가 'long'/'short' 또는 'buy'/'sell' 섞일 수 있음
    if side_raw in ("long", "buy"):
        side_k = "🟢 롱(상승에 베팅)"
    else:
        side_k = "🔴 숏(하락에 베팅)"
    roi = safe_float(p.get("percentage", 0.0))
    upnl = safe_float(p.get("unrealizedPnl", 0.0))
    lev = safe_float(p.get("leverage", 0))
    contracts = safe_float(p.get("contracts", 0))
    entry = safe_float(p.get("entryPrice", 0))
    return sym, side_k, roi, upnl, lev, contracts, entry


# =========================================================
# ✅ 트레이딩뷰 차트 (다크모드)
# =========================================================
def tradingview_embed(symbol_ccxt: str, interval: str = "5", height: int = 520):
    """
    TradingView 위젯은 거래소 심볼 표기 문제가 있어.
    가장 안정적인 기본은 BINANCE:BTCUSDT 같은 형태.
    Bitget 심볼로 정확히 맞추고 싶으면 BITGET:BTCUSDT.P 등으로 바꿔야 할 수 있음.
    """
    base = symbol_ccxt.split("/")[0].replace(":", "")
    tv_symbol = f"BINANCE:{base}USDT"

    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "{interval}",
          "timezone": "Asia/Seoul",
          "theme": "dark",
          "style": "1",
          "locale": "kr",
          "toolbar_bg": "#1a1a1a",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_side_toolbar": false,
          "details": true,
          "withdateranges": true,
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height + 20)


# =========================================================
# ✅ 경제캘린더(한글 요약) - 안전한 방식(사이트 구조 바뀌면 빈값)
# =========================================================
def get_econ_calendar_korean(region="US", limit=10):
    """
    고장/차단 방지를 위해: 실패하면 빈 DF 반환.
    (실전 전에 더 안정적인 소스로 갈아타는 걸 추천)
    """
    if not _HAS_BS4:
        return pd.DataFrame(columns=["날짜", "시간", "지표", "중요도", "국가"])

    try:
        # 간단 예시: investing.com은 차단이 잦음 → 여기선 안전 fallback
        # 원하는 경우 실전 전용으로 API/다른 소스로 바꾸자.
        return pd.DataFrame(columns=["날짜", "시간", "지표", "중요도", "국가"])
    except Exception:
        return pd.DataFrame(columns=["날짜", "시간", "지표", "중요도", "국가"])


# =========================================================
# ✅ 텔레그램 send
# =========================================================
def tg_send(text: str, parse_mode=None, reply_markup=None, chat_id=None):
    if not tg_token or not tg_id:
        return
    cid = chat_id or tg_id
    url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
    data = {"chat_id": cid, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    try:
        http_post(url, data=data)
    except Exception as e:
        logger.warning(f"tg_send fail: {e}")


# =========================================================
# ✅ 텔레그램 봇 스레드(자동매매 + 조회/일지)
# =========================================================
def telegram_thread(ex):
    active_trades = {}  # coin -> dict(sl_price,tp_price,trade_id,entry_usdt,entry_pct,lev,...)
    offset = 0
    last_report = time.time()

    menu_kb = {
        "inline_keyboard": [
            [{"text": "📡 상태", "callback_data": "status"},
             {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "📊 포지션", "callback_data": "position"},
             {"text": "🌍 전체스캔", "callback_data": "scan_all"}],
            [{"text": "📜 일지(최근)", "callback_data": "logs"},
             {"text": "🛑 전량청산", "callback_data": "close_all"}],
        ]
    }
    tg_send("🚀 봇 가동 시작! (Streamlit 설정을 기준으로 동작)\n메뉴가 필요하면 아래 버튼을 눌러줘.", reply_markup=menu_kb)

    while True:
        try:
            cfg = load_settings()  # ✅ 반드시 최신 파일로
            rt = load_runtime()

            # pause (연속손실 등으로 멈춤 상태)
            if time.time() < safe_float(rt.get("pause_until", 0.0), 0.0):
                auto_on = False
            else:
                auto_on = bool(cfg.get("auto_trade", False))

            mode = cfg.get("trade_mode", "안전모드")
            rules = MODE_RULES.get(mode, MODE_RULES["안전모드"])

            # ===== AI 시야 파일 업데이트 =====
            free, total = safe_fetch_balance(ex)
            vision = {
                "time": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                "auto_trade": auto_on,
                "trade_mode": mode,
                "min_conf": rules["min_conf"],
                "balance_total": total,
                "balance_free": free,
                "watch": TARGET_COINS,
                "active_trades": {k: {kk: vv for kk, vv in v.items() if kk in ("trade_id","sl_price","tp_price","entry_usdt","entry_pct","lev")} for k, v in active_trades.items()},
            }
            write_json_atomic(MONITOR_FILE, vision)

            # ===== 자동매매 루프 =====
            if auto_on:
                for coin in TARGET_COINS:
                    try:
                        # 포지션 확인
                        positions = safe_fetch_positions(ex, [coin])
                        active_pos = [p for p in positions if safe_float(p.get("contracts", 0)) > 0]

                        # 1) 포지션 관리: SR 기반 손절/익절(가격 기준)
                        if active_pos:
                            p = active_pos[0]
                            sym, side_k, roi, upnl, lev, contracts, entry = position_summary_korean(p)

                            info = active_trades.get(coin, {})
                            sl_price = info.get("sl_price")
                            tp_price = info.get("tp_price")
                            trade_id = info.get("trade_id", "")

                            ticker = ex.fetch_ticker(coin)
                            cur_price = safe_float(ticker.get("last") or ticker.get("close") or ticker.get("mark"), 0.0)

                            # side 판정
                            side_raw = str(p.get("side","")).lower()
                            is_long = side_raw in ("long","buy")
                            hit_sl = False
                            hit_tp = False

                            if is_long:
                                if sl_price is not None and cur_price <= float(sl_price):
                                    hit_sl = True
                                if tp_price is not None and cur_price >= float(tp_price):
                                    hit_tp = True
                            else:
                                if sl_price is not None and cur_price >= float(sl_price):
                                    hit_sl = True
                                if tp_price is not None and cur_price <= float(tp_price):
                                    hit_tp = True

                            if hit_sl or hit_tp:
                                close_side = "sell" if is_long else "buy"
                                ex.create_market_order(coin, close_side, contracts)

                                # 청산 후 다시 조회해서 realized 추정 어렵지만 upnl/roi 기록
                                one_line = "손절(지지/저항 이탈)" if hit_sl else "익절(목표 도달)"
                                log_trade_csv(
                                    coin=coin,
                                    side=("long" if is_long else "short"),
                                    entry_price=entry,
                                    exit_price=cur_price,
                                    pnl_usdt=upnl,
                                    pnl_pct=roi,
                                    one_line=one_line,
                                    trade_id=trade_id or uuid.uuid4().hex[:10]
                                )

                                # 상세 일지 업데이트
                                if trade_id:
                                    detail = load_trade_detail(trade_id) or {}
                                    detail.update({
                                        "exit_time": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                                        "exit_price": cur_price,
                                        "pnl_usdt": upnl,
                                        "pnl_pct": roi,
                                        "result": "SL" if hit_sl else "TP",
                                        "review": ("손절이라면: 다음엔 손절 버퍼/손익비를 조정"
                                                   if hit_sl else
                                                   "익절이라면: 다음엔 추세 유지 시 분할익절/트레일링 고려")
                                    })
                                    save_trade_detail(trade_id, detail)

                                # 텔레그램은 짧고 직관적으로(USDT 포함)
                                entry_usdt = safe_float(info.get("entry_usdt", 0.0))
                                entry_pct = safe_float(info.get("entry_pct", 0.0))
                                tg_send(
                                    f"{'🩸 손절' if hit_sl else '🎉 익절'}: {coin}\n"
                                    f"- 방향: {side_k}\n"
                                    f"- 수익률: {roi:.2f}% (손익 {upnl:.2f} USDT)\n"
                                    f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%) / 레버 x{lev}\n"
                                    f"- 현재잔고: {total:.2f} USDT (가용 {free:.2f})\n"
                                    f"- 상세일지: {trade_id if trade_id else '없음'}"
                                )
                                if coin in active_trades:
                                    del active_trades[coin]

                            continue  # 포지션 있으면 신규진입 분석 생략

                        # 2) 신규 진입 분석
                        ohlcv = ex.fetch_ohlcv(coin, "5m", limit=120)
                        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                        df["time"] = pd.to_datetime(df["time"], unit="ms")
                        df, status, last = calc_indicators(df, cfg)
                        if last is None:
                            continue

                        # 필터(너무 애매한 횡보 줄이기) - 모드가 공격/하이리스크면 완화
                        adx = safe_float(last.get("ADX", 0.0))
                        rsi = safe_float(last.get("RSI", 50.0))
                        if mode == "안전모드":
                            if 30 <= rsi <= 70 and adx < 18:
                                continue
                        else:
                            if 35 <= rsi <= 65 and adx < 15:
                                continue

                        plan = generate_ai_plan(df, status, cfg, coin)
                        req_conf = int(rules["min_conf"])
                        if plan.decision in ("buy","sell") and plan.confidence >= req_conf:

                            # 모드 범위 내로 이미 clamp됨
                            lev = int(plan.leverage)
                            pct = float(plan.percentage)

                            # 자금 계산
                            free2, total2 = safe_fetch_balance(ex)
                            entry_usdt = free2 * (pct / 100.0)

                            # 최소 주문금액(혹시 너무 작으면)
                            if entry_usdt < 5:
                                continue

                            # 레버 적용
                            try:
                                ex.set_leverage(lev, coin)
                            except Exception:
                                pass

                            price = safe_float(last.get("close", 0.0))
                            if price <= 0:
                                continue

                            qty = ex.amount_to_precision(coin, (entry_usdt * lev) / price)
                            if safe_float(qty, 0.0) <= 0:
                                continue

                            # SR 기반 SL/TP 계산(HTF)
                            sr_tf = cfg.get("sr_tf", "15m")
                            htf = ex.fetch_ohlcv(coin, sr_tf, limit=200)
                            htf_df = pd.DataFrame(htf, columns=["time","open","high","low","close","vol"])
                            htf_df["time"] = pd.to_datetime(htf_df["time"], unit="ms")

                            rr_min = float(plan.rr_min)
                            buf_mult = float(plan.buffer_atr_mult)

                            sr = sr_stop_take(
                                entry_price=price,
                                side=plan.decision,
                                htf_df=htf_df,
                                atr_period=int(cfg.get("sr_atr_period", 14)),
                                pivot_order=int(cfg.get("sr_pivot_order", 6)),
                                buffer_atr_mult=buf_mult,
                                rr_min=rr_min
                            )

                            trade_id = uuid.uuid4().hex[:10]

                            # 주문 실행
                            ex.create_market_order(coin, plan.decision, qty)

                            # active_trades 저장
                            active_trades[coin] = {
                                "trade_id": trade_id,
                                "sl_price": sr["sl_price"] if sr else None,
                                "tp_price": sr["tp_price"] if sr else None,
                                "entry_usdt": float(entry_usdt),
                                "entry_pct": float(pct),
                                "lev": int(lev),
                            }

                            # 상세 일지 저장(근거는 텔레그램에 안 보냄)
                            save_trade_detail(trade_id, {
                                "trade_id": trade_id,
                                "time": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                                "coin": coin,
                                "decision": plan.decision,
                                "confidence": plan.confidence,
                                "entry_price": price,
                                "lev": lev,
                                "entry_usdt": float(entry_usdt),
                                "entry_pct": float(pct),
                                "rr_min": rr_min,
                                "buffer_atr_mult": buf_mult,
                                "sl_price": sr["sl_price"] if sr else None,
                                "tp_price": sr["tp_price"] if sr else None,
                                "used_indicators": plan.used_indicators,
                                "one_line": plan.one_line or "진입(근거는 상세일지에 저장)",
                                "reason": plan.reason,
                                "status": status,
                            })

                            # 텔레그램은 짧게 + 숫자 명확히(USDT/잔고%)
                            tg_send(
                                f"🎯 진입: {coin}\n"
                                f"- 방향: {'🟢 롱(상승)' if plan.decision=='buy' else '🔴 숏(하락)'}\n"
                                f"- 확신도: {plan.confidence}% / 모드: {mode}\n"
                                f"- 진입금: {entry_usdt:.2f} USDT (잔고 {pct:.1f}%) / 레버 x{lev}\n"
                                f"- 목표: TP {active_trades[coin]['tp_price']} / SL {active_trades[coin]['sl_price']}\n"
                                f"- 한줄평: {plan.one_line}\n"
                                f"- 상세일지 ID: {trade_id}"
                            )
                            time.sleep(3)

                    except Exception as e:
                        logger.warning(f"auto loop err {coin}: {e}")

                    time.sleep(float(cfg.get("scan_interval_sec", 2)))

            # ===== 정기 리포트 =====
            if time.time() - last_report > float(cfg.get("report_interval_sec", 900)):
                free3, total3 = safe_fetch_balance(ex)
                tg_send(f"💤 생존신고\n- 현재 총자산: {total3:.2f} USDT (가용 {free3:.2f})\n- 모드: {mode} / 자동매매: {'ON' if auto_on else 'OFF'}")
                last_report = time.time()

            # ===== 텔레그램 업데이트 처리 =====
            try:
                res = http_get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1").json()
                if res.get("ok"):
                    for up in res.get("result", []):
                        offset = up["update_id"]

                        # 버튼 콜백
                        if "callback_query" in up:
                            cb = up["callback_query"]
                            data = cb.get("data", "")
                            cid = cb["message"]["chat"]["id"]

                            if data == "status":
                                cfg_live = load_settings()
                                rt_live = load_runtime()
                                free_s, total_s = safe_fetch_balance(ex)
                                md = cfg_live.get("trade_mode","-")
                                au = cfg_live.get("auto_trade", False)
                                tg_send(
                                    f"📡 상태\n"
                                    f"- 자동매매: {'ON' if au else 'OFF'}\n"
                                    f"- 모드: {md}\n"
                                    f"- 잔고: {total_s:.2f} USDT (가용 {free_s:.2f})\n"
                                    f"- 연속손실: {rt_live.get('consec_losses',0)}\n",
                                    chat_id=cid
                                )

                            elif data == "balance":
                                free_b, total_b = safe_fetch_balance(ex)
                                tg_send(
                                    f"💰 잔고\n- 총자산: {total_b:.2f} USDT\n- 가용: {free_b:.2f} USDT",
                                    chat_id=cid
                                )

                            elif data == "position":
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if safe_float(p.get("contracts", 0)) > 0]
                                if not act:
                                    tg_send("📊 현재 무포지션(관망 중)", chat_id=cid)
                                else:
                                    free_p, total_p = safe_fetch_balance(ex)
                                    lines = [f"📊 포지션 ({len(act)}개)\n- 잔고: {total_p:.2f} USDT (가용 {free_p:.2f})"]
                                    for p in act:
                                        sym, side_k, roi, upnl, lev, contracts, entry = position_summary_korean(p)
                                        lines.append(
                                            f"\n[{sym}]\n"
                                            f"- 방향: {side_k}\n"
                                            f"- 수익률: {roi:.2f}% (손익 {upnl:.2f} USDT)\n"
                                            f"- 레버: x{lev} / 수량: {contracts:.4f}\n"
                                            f"- 진입가: {entry}"
                                        )
                                    tg_send("\n".join(lines), chat_id=cid)

                            elif data == "scan_all":
                                # 텔레그램 전체스캔은 비용/시간이 커서 간단히 안내
                                tg_send("🌍 전체스캔: Streamlit의 '전체 코인 스캔'을 추천해. (AI 호출이 많아질 수 있어)", chat_id=cid)

                            elif data == "logs":
                                ids = list_recent_trade_ids(10)
                                if not ids:
                                    tg_send("📭 저장된 매매일지가 아직 없어요.", chat_id=cid)
                                else:
                                    lines = ["📜 최근 매매일지(한줄평)\n(상세: '일지상세 ID'로 조회)"]
                                    for tid in ids:
                                        d = load_trade_detail(tid) or {}
                                        lines.append(f"- {tid} | {d.get('coin','?')} | {d.get('one_line','')}")
                                    tg_send("\n".join(lines), chat_id=cid)

                            elif data == "close_all":
                                tg_send("🛑 전량 청산 시도합니다!", chat_id=cid)
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if safe_float(p.get("contracts",0)) > 0]
                                for p in act:
                                    sym = p.get("symbol")
                                    contracts = safe_float(p.get("contracts",0))
                                    side_raw = str(p.get("side","")).lower()
                                    is_long = side_raw in ("long","buy")
                                    close_side = "sell" if is_long else "buy"
                                    try:
                                        ex.create_market_order(sym, close_side, contracts)
                                    except Exception:
                                        pass

                            # 콜백 응답
                            try:
                                http_post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery",
                                          data={"callback_query_id": cb["id"]})
                            except Exception:
                                pass

                        # 텍스트 명령
                        if "message" in up and "text" in up["message"]:
                            txt = up["message"]["text"].strip()
                            cid = up["message"]["chat"]["id"]

                            if txt == "메뉴":
                                tg_send("✅ 메뉴 갱신", reply_markup=menu_kb, chat_id=cid)

                            elif txt == "상태":
                                cfg_live = load_settings()
                                rt_live = load_runtime()
                                free_s, total_s = safe_fetch_balance(ex)
                                tg_send(
                                    f"📡 상태\n"
                                    f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                    f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                    f"- 잔고: {total_s:.2f} USDT (가용 {free_s:.2f})\n"
                                    f"- 연속손실: {rt_live.get('consec_losses',0)}\n",
                                    chat_id=cid
                                )

                            elif txt == "일지":
                                ids = list_recent_trade_ids(10)
                                if not ids:
                                    tg_send("📭 저장된 매매일지가 아직 없어요.", chat_id=cid)
                                else:
                                    lines = ["📜 최근 매매일지(한줄평)\n(상세: '일지상세 ID')"]
                                    for tid in ids:
                                        d = load_trade_detail(tid) or {}
                                        lines.append(f"- {tid} | {d.get('coin','?')} | {d.get('one_line','')}")
                                    tg_send("\n".join(lines), chat_id=cid)

                            elif txt.startswith("일지상세"):
                                parts = txt.split()
                                if len(parts) < 2:
                                    tg_send("사용법: 일지상세 <ID>", chat_id=cid)
                                else:
                                    tid = parts[1].strip()
                                    d = load_trade_detail(tid)
                                    if not d:
                                        tg_send("해당 ID를 찾지 못했어요.", chat_id=cid)
                                    else:
                                        tg_send(
                                            f"📌 일지상세 {tid}\n"
                                            f"- 코인: {d.get('coin')}\n"
                                            f"- 방향: {d.get('decision')}\n"
                                            f"- 확신도: {d.get('confidence')}\n"
                                            f"- 진입가: {d.get('entry_price')}\n"
                                            f"- 레버: x{d.get('lev')}\n"
                                            f"- 진입금: {d.get('entry_usdt'):.2f} USDT (잔고 {d.get('entry_pct'):.1f}%)\n"
                                            f"- SL/TP: {d.get('sl_price')} / {d.get('tp_price')}\n"
                                            f"- 한줄평: {d.get('one_line')}\n"
                                            f"- 참고지표: {d.get('used_indicators')}\n",
                                            chat_id=cid
                                        )
            except Exception as e:
                logger.warning(f"tg update loop err: {e}")

            time.sleep(float(cfg.get("vision_interval_sec", 3)))

        except Exception as e:
            logger.error(f"telegram_thread fatal: {e}")
            time.sleep(5)


# =========================================================
# ✅ Streamlit UI (제어판 + 차트 + 포지션 + 일지)
# =========================================================
st.sidebar.title("🛠️ 제어판(컨트롤)")
st.sidebar.caption("Streamlit은 제어/확인용, 실시간 보고/조회는 Telegram으로!")

# 디버그(저장된 설정 확인)
with st.sidebar.expander("🧪 디버그: 저장된 설정(bot_settings.json)"):
    st.json(read_json_safe(SETTINGS_FILE, {}))

# 자동매매 스위치/모드 선택
trade_mode = st.sidebar.selectbox(
    "매매 모드",
    list(MODE_RULES.keys()),
    index=list(MODE_RULES.keys()).index(config.get("trade_mode", "안전모드")) if config.get("trade_mode","안전모드") in MODE_RULES else 0
)
auto_on = st.sidebar.checkbox("🤖 자동매매 ON/OFF", value=bool(config.get("auto_trade", False)))
ai_apply_global = st.sidebar.checkbox("🧠 AI 추천값 자동 적용(모드 범위 내)", value=bool(config.get("ai_apply_global", True)))

st.sidebar.divider()

# SR 설정
st.sidebar.subheader("🧱 손절/익절(지지/저항 기반)")
sr_tf = st.sidebar.selectbox("SR 기준 타임프레임", ["5m","15m","1h","4h"], index=["5m","15m","1h","4h"].index(config.get("sr_tf","15m")) if config.get("sr_tf","15m") in ["5m","15m","1h","4h"] else 1)
sr_pivot_order = st.sidebar.slider("피벗 민감도(낮을수록 더 자주)", 3, 10, int(config.get("sr_pivot_order", 6)))
sr_atr_period = st.sidebar.slider("ATR 기간", 7, 30, int(config.get("sr_atr_period", 14)))
sr_buffer_atr = st.sidebar.slider("손절 버퍼(ATR 배수)", 0.05, 1.0, float(config.get("sr_buffer_atr_mult", 0.25)), step=0.05)

st.sidebar.divider()
st.sidebar.subheader("📌 모드별 최소 손익비(RR)")
rr_safe = st.sidebar.slider("안전모드 RR", 1.0, 3.0, float(config.get("rr_min_safe", 1.6)), step=0.1)
rr_aggr = st.sidebar.slider("공격모드 RR", 1.0, 3.0, float(config.get("rr_min_aggr", 1.4)), step=0.1)
rr_hr = st.sidebar.slider("하이리스크 RR", 1.0, 3.0, float(config.get("rr_min_hr", 1.3)), step=0.1)

st.sidebar.divider()

# 지표 ON/OFF
st.sidebar.subheader("📊 보조지표(10종) ON/OFF")
use_rsi = st.sidebar.checkbox("RSI", value=bool(config.get("use_rsi", True)))
use_bb = st.sidebar.checkbox("볼린저밴드", value=bool(config.get("use_bb", True)))
use_ma = st.sidebar.checkbox("이동평균(MA)", value=bool(config.get("use_ma", True)))
use_adx = st.sidebar.checkbox("ADX(추세강도)", value=bool(config.get("use_adx", True)))
use_macd = st.sidebar.checkbox("MACD", value=bool(config.get("use_macd", False)))
use_stoch = st.sidebar.checkbox("스토캐스틱", value=bool(config.get("use_stoch", False)))
use_cci = st.sidebar.checkbox("CCI", value=bool(config.get("use_cci", True)))
use_mfi = st.sidebar.checkbox("MFI", value=bool(config.get("use_mfi", False)))
use_willr = st.sidebar.checkbox("Williams %R", value=bool(config.get("use_willr", False)))
use_vol = st.sidebar.checkbox("거래량", value=bool(config.get("use_vol", True)))

# 저장 반영
new_conf = dict(config)
new_conf.update({
    "trade_mode": trade_mode,
    "auto_trade": auto_on,
    "ai_apply_global": ai_apply_global,

    "sr_tf": sr_tf,
    "sr_pivot_order": sr_pivot_order,
    "sr_atr_period": sr_atr_period,
    "sr_buffer_atr_mult": sr_buffer_atr,

    "rr_min_safe": rr_safe,
    "rr_min_aggr": rr_aggr,
    "rr_min_hr": rr_hr,

    "use_rsi": use_rsi,
    "use_bb": use_bb,
    "use_ma": use_ma,
    "use_adx": use_adx,
    "use_macd": use_macd,
    "use_stoch": use_stoch,
    "use_cci": use_cci,
    "use_mfi": use_mfi,
    "use_willr": use_willr,
    "use_vol": use_vol,
})
if new_conf != config:
    save_settings(new_conf)
    config = new_conf
    st.rerun()

st.sidebar.divider()

# 일지 초기화 버튼
if st.sidebar.button("🧹 매매일지/런타임 상태 초기화(주의)"):
    reset_runtime_and_logs()
    st.sidebar.success("초기화 완료! 새로고침합니다.")
    time.sleep(0.5)
    st.rerun()

# 텔레그램 메뉴 전송
if st.sidebar.button("📡 텔레그램 메뉴 전송"):
    kb = {"inline_keyboard": [[{"text": "📡 상태", "callback_data": "status"},
                              {"text": "💰 잔고", "callback_data": "balance"}],
                             [{"text": "📊 포지션", "callback_data": "position"},
                              {"text": "📜 일지(최근)", "callback_data": "logs"}]]}
    tg_send("✅ <b>메뉴 갱신</b>", parse_mode="HTML", reply_markup=kb)


# =========================================================
# ✅ 텔레그램 스레드 시작(1회)
# =========================================================
found = any(t.name == "TG_THREAD" for t in threading.enumerate())
if not found:
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(th)
    th.start()


# =========================================================
# ✅ 메인 화면(차트 + 지표 + 포지션 + 일지 + AI 시야)
# =========================================================
st.title("📈 워뇨띠 AI 트레이딩 제어판")

# 자동 새로고침(시야/포지션 최신)
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=3000, key="main_refresh")

# 시장 선택
markets = exchange.markets
symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
symbol = st.selectbox("코인 선택", symbol_list, index=0)

colA, colB, colC, colD = st.columns(4)

free, total = safe_fetch_balance(exchange)
colA.metric("총 자산(USDT)", f"{total:,.2f}")
colB.metric("가용(USDT)", f"{free:,.2f}")
colC.metric("자동매매", "ON" if config.get("auto_trade") else "OFF")
colD.metric("모드", config.get("trade_mode","-"))

st.divider()

# 차트 + 지표 계산
df = None
status = {}
last = None
data_loaded = False

try:
    ohlcv = exchange.fetch_ohlcv(symbol, "5m", limit=160)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df, status, last = calc_indicators(df, config)
    data_loaded = last is not None
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("🕯️ TradingView 차트 (Dark)")
    tradingview_embed(symbol, interval="5", height=540)

    if data_loaded:
        st.subheader("📌 지표 요약(쉬운 말)")
        rsi_txt = status.get("RSI","")
        bb_txt = status.get("BB","")
        adx_txt = status.get("ADX","")
        ma_txt = status.get("MA","")
        macd_txt = status.get("MACD","")

        st.write(f"- RSI: {rsi_txt} (과매도/과매수 여부)")
        st.write(f"- 볼린저밴드: {bb_txt} (밴드 위/아래 이탈 여부)")
        st.write(f"- ADX: {adx_txt} (추세가 강한지/약한지)")
        if ma_txt:
            st.write(f"- 이동평균: {ma_txt} (단기/장기 방향)")
        if macd_txt:
            st.write(f"- MACD: {macd_txt} (추세 전환 신호)")

        with st.expander("🔍 지표 상세값(개발자용/원하면 안 봐도 됨)"):
            st.json({k: str(v) for k, v in status.items()})

with right:
    st.subheader("👁️ AI 시야(실시간 모니터링)")
    vision = read_json_safe(MONITOR_FILE, {})
    if vision:
        st.json(vision)
    else:
        st.info("시야 파일이 아직 없어요. 잠시 후 자동 생성됩니다.")

    st.divider()
    st.subheader("📊 보유 포지션(요약)")
    ps = safe_fetch_positions(exchange, TARGET_COINS)
    act = [p for p in ps if safe_float(p.get("contracts", 0)) > 0]

    if not act:
        st.caption("현재 무포지션(관망 중)")
    else:
        for p in act:
            sym, side_k, roi, upnl, lev, contracts, entry = position_summary_korean(p)
            st.info(
                f"**{sym}**\n\n"
                f"- 방향: {side_k}\n"
                f"- 수익률: {roi:.2f}%  (손익 {upnl:.2f} USDT)\n"
                f"- 레버: x{lev} / 수량: {contracts:.4f}\n"
                f"- 진입가: {entry}"
            )

st.divider()

# 탭 구성
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🧠 AI 전략 센터")
    st.write("자동매매 ON/OFF와 모드는 **왼쪽 제어판**에서 조절합니다.")
    if st.button("🔍 현재 코인 AI 분석(진입은 하지 않음)"):
        if not data_loaded:
            st.warning("데이터가 아직 없어요.")
        else:
            plan = generate_ai_plan(df, status, config, symbol)
            st.write(f"결론: **{plan.decision.upper()}** / 확신도 **{plan.confidence}%**")
            st.write(f"추천: 진입비중 {plan.percentage:.1f}% / 레버 x{plan.leverage}")
            st.write(f"한줄평: {plan.one_line}")
            with st.expander("상세 근거(저장되는 내용)"):
                st.write(plan.reason)

    if st.button("🌍 전체 코인 스캔(5개)"):
        st.info("5개 코인을 순차 분석합니다. (AI 호출이 많아질 수 있어요)")
        rows = []
        prog = st.progress(0)
        for i, c in enumerate(TARGET_COINS):
            try:
                o = exchange.fetch_ohlcv(c, "5m", limit=140)
                d = pd.DataFrame(o, columns=["time","open","high","low","close","vol"])
                d["time"] = pd.to_datetime(d["time"], unit="ms")
                d, stt, lst = calc_indicators(d, config)
                if lst is None:
                    raise Exception("지표 계산 실패")
                pl = generate_ai_plan(d, stt, config, c)
                rows.append({
                    "코인": c.split("/")[0],
                    "현재가": f"{safe_float(lst.get('close',0)):.4f}",
                    "결론": pl.decision.upper(),
                    "확신도": pl.confidence,
                    "한줄": pl.one_line[:40]
                })
            except Exception as e:
                rows.append({"코인": c, "결론": "ERROR", "한줄": str(e)[:60]})
            prog.progress((i+1)/len(TARGET_COINS))
        st.dataframe(pd.DataFrame(rows), width="stretch")

with t2:
    st.subheader("✋ 수동주문(선택)")
    st.caption("수동주문 로직은 너의 기존 구조를 해치지 않기 위해 '틀만 유지'했어. 원하면 다음에 강화해줄게.")
    amt = st.number_input("주문 금액(USDT)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    c1, c2, c3 = st.columns(3)
    if c1.button("🟢 롱 진입(수동)"):
        st.warning("수동주문 로직은 다음 단계에서 안전장치 포함해 완성하는 걸 추천!")
    if c2.button("🔴 숏 진입(수동)"):
        st.warning("수동주문 로직은 다음 단계에서 안전장치 포함해 완성하는 걸 추천!")
    if c3.button("🚫 포지션 종료(수동)"):
        st.warning("수동청산 로직은 다음 단계에서 안전장치 포함해 완성하는 걸 추천!")

with t3:
    st.subheader("📅 경제캘린더(한글)")
    ev = get_econ_calendar_korean(config.get("econ_calendar_region","US"), limit=10)
    if ev is None or ev.empty:
        st.info("지금은 안전모드로 '일정 없음' 표시 중이에요. (실전 전 안정적인 소스로 업그레이드 추천)")
    else:
        st.dataframe(ev, width="stretch")

with t4:
    st.subheader("📜 매매일지(보기 쉽게)")
    colx, coly = st.columns([1, 5])
    if colx.button("🔄 새로고침"):
        st.rerun()

    if os.path.exists(LOG_FILE):
        try:
            h = pd.read_csv(LOG_FILE)
            if "Time" in h.columns:
                h = h.sort_values(by="Time", ascending=False)
            st.dataframe(h, width="stretch", hide_index=True)
            csv = h.to_csv(index=False).encode("utf-8-sig")
            coly.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")
        except Exception as e:
            st.error(f"일지 읽기 실패: {e}")
    else:
        st.caption("아직 기록된 매매가 없습니다.")

    st.divider()
    st.subheader("📌 상세일지(TradeID) 빠른조회")
    tid = st.text_input("TradeID 입력(예: 텔레그램에 뜬 ID)")
    if st.button("조회"):
        d = load_trade_detail(tid.strip()) if tid else None
        if not d:
            st.warning("해당 ID를 찾지 못했어요.")
        else:
            st.json(d)


# =========================================================
# ✅ 안내(필수)
# =========================================================
st.caption("⚠️ 이 앱은 모의투자(IS_SANDBOX=True) 기준입니다. 실전 전에는 주문/청산/예외처리를 더 강화하는 걸 권장합니다.")
