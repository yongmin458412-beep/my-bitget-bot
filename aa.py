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
from datetime import datetime, timedelta
from openai import OpenAI
from streamlit.runtime.scriptrunner import add_script_run_ctx

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="Bitget AI Bot (Wonyo-Style)")

IS_SANDBOX = True  # 실전이면 False
SETTINGS_FILE = "bot_settings.json"
RUNTIME_STATE_FILE = "runtime_state.json"
TRADE_LOG_FILE = "trade_log.csv"

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

TV_SYMBOL_MAP = {
    "BTC/USDT:USDT": "BINANCE:BTCUSDT",
    "ETH/USDT:USDT": "BINANCE:ETHUSDT",
    "SOL/USDT:USDT": "BINANCE:SOLUSDT",
    "XRP/USDT:USDT": "BINANCE:XRPUSDT",
    "DOGE/USDT:USDT": "BINANCE:DOGEUSDT",
}

# =========================================================
# 유틸
# =========================================================
def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except:
        return default

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def utc_now():
    return datetime.utcnow()

def read_json(path, default_obj):
    if not os.path.exists(path):
        return default_obj
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default_obj

def write_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except:
        pass

# =========================================================
# 설정 관리
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "telegram_enabled": True,

        # 모드
        "trade_mode": "SAFE",  # SAFE / AGGRESSIVE

        # AI 글로벌옵션 자동 적용
        "use_ai_global": True,

        # 루프 주기
        "manage_interval_sec": 2,
        "entry_scan_interval_sec": 10,
        "report_interval_sec": 900,

        # 차트 UI
        "ui_symbol": TARGET_COINS[0],
        "ui_interval_tf": "5",

        # 가드레일(원금손실 최소화 목적, 원하면 OFF 가능)
        "enable_hard_guardrails": True,
        "hard_max_leverage_safe": 10,
        "hard_max_leverage_aggressive": 20,
        "hard_max_risk_pct_safe": 15.0,
        "hard_max_risk_pct_aggressive": 30.0,

        # 스타일
        "prefer_short_sl": True,         # 손절은 짧게 (단, '과매도 진입' 금지와 함께 작동)
        "prefer_long_tp_trend": True,    # 추세면 익절 길게(트레일링/TP연장)
        "allow_tp_extend": True,
        "tp_extend_mult": 1.7,

        # 멀티타임프레임
        "confirm_tf": "1h",   # 상위 TF
        "base_tf": "5m",

        # 구조/스윙
        "swing_window": 5,     # 스윙 찾는 윈도(클수록 더 큰 구조)
        "structure_lookback": 120,

        # 레이더(=AI 호출 전 필터)
        "radar_use": True,

        # 로그 UI
        "log_rows_ui": 200,
    }
    saved = read_json(SETTINGS_FILE, {})
    if isinstance(saved, dict):
        default.update(saved)
    return default

def save_settings(cfg):
    write_json(SETTINGS_FILE, cfg)

config = load_settings()

# =========================================================
# runtime_state
# =========================================================
def default_runtime_state():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},

        # 포지션 메타(봇 내부)
        "trades": {},

        # 분할 진입 대기(2차 진입)
        "pending_scaleins": {},

        # 텔레그램 offset
        "tg_offset": 0,

        # 상태 메모
        "last_bot_note": "",

        # AI 투명성: 최근 입력/출력
        "last_ai_inputs": {},
        "last_ai_outputs": {},

        # AI 글로벌옵션
        "ai_global": {
            "cooldown_minutes": 10,
            "max_consec_losses": 3,
            "pause_minutes": 30,
            "news_avoid": True,
            "news_block_before_min": 15,
            "news_block_after_min": 15,
        },
    }

def load_runtime_state():
    s = read_json(RUNTIME_STATE_FILE, None)
    if not isinstance(s, dict):
        s = default_runtime_state()
        write_json(RUNTIME_STATE_FILE, s)
    return s

def save_runtime_state(state):
    write_json(RUNTIME_STATE_FILE, state)

def maybe_roll_daily_state(state, equity_now: float):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state["date"] = today
        state["day_start_equity"] = float(equity_now)
        state["daily_realized_pnl"] = 0.0
        state["consec_losses"] = 0
        state["pause_until"] = 0
        state["cooldowns"] = {}
        state["trades"] = {}
        state["pending_scaleins"] = {}
        state["last_bot_note"] = "데일리 리셋"
        save_runtime_state(state)

def is_paused(state):
    return time.time() < safe_float(state.get("pause_until", 0))

def in_cooldown(state, symbol):
    until = safe_float(state.get("cooldowns", {}).get(symbol, 0))
    return time.time() < until

def set_cooldown(state, symbol, minutes: int):
    state.setdefault("cooldowns", {})
    state["cooldowns"][symbol] = int(time.time() + int(minutes) * 60)
    save_runtime_state(state)

# =========================================================
# trade_log.csv
# =========================================================
TRADE_LOG_COLUMNS = [
    "Time", "Mode", "Symbol", "Event", "Side", "Qty",
    "EntryPrice", "ExitPrice", "PnL_USDT", "PnL_Percent",
    "Leverage", "RiskPct",
    "TP_Target", "SL_Target",
    "Regime", "HTF_Trend", "SetupTag", "FailTag",
    "MFE_ROI", "MAE_ROI",
    "Reason", "Review", "OneLine",
    "Snapshot_JSON"
]

def append_trade_log(row: dict):
    df = pd.DataFrame([{c: row.get(c, "") for c in TRADE_LOG_COLUMNS}])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

def read_trade_log(n=200):
    if not os.path.exists(TRADE_LOG_FILE):
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df.head(n)
    except:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

def make_oneline_summary(row: dict):
    t = row.get("Time", "")
    sym = row.get("Symbol", "")
    ev = row.get("Event", "")
    pnlp = row.get("PnL_Percent", "")
    mode = row.get("Mode", "")
    tag = row.get("SetupTag", "")
    short = str(row.get("Review", "") or row.get("Reason", "")).replace("\n", " ")
    short = short[:34] + ("..." if len(short) > 34 else "")
    return f"{t} | {mode} | {sym} | {ev} | {pnlp}% | {tag} | {short}"

def summarize_recent_mistakes():
    df = read_trade_log(120)
    if df.empty:
        return "기록 없음"
    try:
        df["PnL_Percent"] = pd.to_numeric(df["PnL_Percent"], errors="coerce")
        worst = df.sort_values("PnL_Percent", ascending=True).head(5)
        lines = []
        for _, r in worst.iterrows():
            lines.append(f"- {r['Symbol']} {r.get('Side','')} {r['PnL_Percent']:.2f}% ({str(r.get('FailTag',''))})")
        return "\n".join(lines) if lines else "큰 손실 기록 없음"
    except:
        return "손실 요약 실패"

# =========================================================
# Secrets
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key or not api_secret or not api_password:
    st.error("🚨 Bitget API 키가 secrets.toml에 없습니다. (API_KEY/API_SECRET/API_PASSWORD)")
    st.stop()

openai_client = None
if openai_key:
    try:
        openai_client = OpenAI(api_key=openai_key)
    except:
        openai_client = None

# =========================================================
# Exchange
# =========================================================
def create_exchange():
    ex = ccxt.bitget({
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.set_sandbox_mode(IS_SANDBOX)
    ex.load_markets()
    try:
        ex.set_position_mode(hedged=False)
    except:
        pass
    return ex

@st.cache_resource
def init_exchange_ui():
    return create_exchange()

exchange = init_exchange_ui()

# =========================================================
# Indicators (no external libs)
# =========================================================
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14):
    d = close.diff()
    gain = d.where(d > 0, 0.0)
    loss = -d.where(d < 0, 0.0)
    ag = gain.rolling(period).mean()
    al = loss.rolling(period).mean()
    rs = ag / (al.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return mid, upper, lower

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

def true_range(high, low, close):
    prev = close.shift(1)
    return pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()

def adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr_)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr_)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.rolling(period).mean()
    return adx_val, plus_di, minus_di

def stoch(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def mfi(high, low, close, vol, period=14):
    tp = (high + low + close) / 3
    mf = tp * vol
    dir_ = tp.diff()
    pos = mf.where(dir_ > 0, 0.0)
    neg = mf.where(dir_ < 0, 0.0).abs()
    ps = pos.rolling(period).sum()
    ns = neg.rolling(period).sum()
    mfr = ps / (ns.replace(0, np.nan))
    return 100 - (100 / (1 + mfr))

def williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

# =========================================================
# Structure / Regime / Signals
# =========================================================
def rolling_slope(series: pd.Series, n=10):
    if len(series) < n + 2:
        return 0.0
    y = series.tail(n).values
    x = np.arange(len(y))
    try:
        m = np.polyfit(x, y, 1)[0]
        return float(m)
    except:
        return 0.0

def find_swings(df: pd.DataFrame, window: int = 5):
    """
    매우 단순한 스윙 탐지:
    - 스윙고점: high가 주변 window 봉보다 높음
    - 스윙저점: low가 주변 window 봉보다 낮음
    """
    if df is None or df.empty or len(df) < window * 2 + 5:
        return [], []
    highs = df["high"].values
    lows = df["low"].values
    swing_highs = []
    swing_lows = []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows

def classify_structure(df: pd.DataFrame, window: int, lookback: int):
    """
    HH/HL 기반 추세 구조 판단.
    """
    if df is None or df.empty:
        return {"structure": "unknown"}
    sub = df.tail(lookback).copy()
    sh, sl = find_swings(sub, window=window)
    if len(sh) < 2 or len(sl) < 2:
        return {"structure": "unknown", "swing_high": None, "swing_low": None}

    # 최근 2개 스윙고점/저점 비교
    last_sh = sh[-1][1]
    prev_sh = sh[-2][1]
    last_sl = sl[-1][1]
    prev_sl = sl[-2][1]

    structure = "range"
    if last_sh > prev_sh and last_sl > prev_sl:
        structure = "uptrend"
    elif last_sh < prev_sh and last_sl < prev_sl:
        structure = "downtrend"
    else:
        structure = "range"

    return {
        "structure": structure,
        "swing_high": float(last_sh),
        "swing_low": float(last_sl),
        "prev_swing_high": float(prev_sh),
        "prev_swing_low": float(prev_sl),
    }

def classify_regime(adx_val: float, atr_pct: float, ma_slope: float):
    """
    레짐(시장 상태) 분류:
    - TREND: ADX 높고, MA기울기 있음
    - RANGE: ADX 낮음
    - VOLATILE: ATR% 높음
    """
    adx_val = safe_float(adx_val, 0)
    atr_pct = safe_float(atr_pct, 0)
    ma_slope = safe_float(ma_slope, 0)

    if atr_pct >= 2.5:
        return "VOLATILE"
    if adx_val >= 25 and abs(ma_slope) > 0:
        return "TREND"
    if adx_val < 18:
        return "RANGE"
    return "MIXED"

def detect_sweep(df: pd.DataFrame):
    """
    스윕(유동성 사냥) 단순 감지:
    - 직전 저점을 살짝 깨고(저가 < 이전 저가)
    - 종가가 비교적 위에서 마감(롱꼬리)
    """
    if df is None or df.empty or len(df) < 5:
        return False, ""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rng = safe_float(last["high"] - last["low"], 0)
    if rng <= 0:
        return False, ""
    wick_ratio = safe_float((last["close"] - last["low"]) / rng, 0)  # 아래꼬리 회복 비율
    sweep_down = (last["low"] < prev["low"]) and (wick_ratio >= 0.65)
    wick_ratio_up = safe_float((last["high"] - last["close"]) / rng, 0)
    sweep_up = (last["high"] > prev["high"]) and (wick_ratio_up >= 0.65)

    if sweep_down:
        return True, "sweep_down"
    if sweep_up:
        return True, "sweep_up"
    return False, ""

def relative_strength_score(df: pd.DataFrame):
    """
    로테이션용 상대강도 점수(간단 버전):
    - 최근 60봉 수익률
    - ADX
    - 거래량(현재/평균)
    """
    if df is None or df.empty or len(df) < 80:
        return 0.0
    sub = df.tail(80)
    ret = safe_float((sub["close"].iloc[-1] / sub["close"].iloc[-60] - 1) * 100, 0)
    vol_sma = safe_float(sub["vol"].rolling(20).mean().iloc[-1], 1)
    vol_now = safe_float(sub["vol"].iloc[-1], 0)
    vol_ratio = safe_float(vol_now / vol_sma, 1)
    adx_val, _, _ = adx(sub["high"], sub["low"], sub["close"], 14)
    adx_last = safe_float(adx_val.dropna().iloc[-1] if not adx_val.dropna().empty else 0, 0)

    # 점수
    score = ret * 1.2 + adx_last * 0.6 + (vol_ratio - 1) * 5.0
    return float(score)

def calc_indicators(df: pd.DataFrame):
    """
    10종 + 추가(ATR, MA slope 등)
    """
    if df is None or df.empty or len(df) < 250:
        return df, None

    df = df.copy()
    df["RSI"] = rsi(df["close"], 14)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(df["close"], 20, 2.0)
    df["MA_fast"] = df["close"].rolling(7).mean()
    df["MA_slow"] = df["close"].rolling(99).mean()
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["close"], 12, 26, 9)
    df["ADX"], df["PDI"], df["MDI"] = adx(df["high"], df["low"], df["close"], 14)
    df["STO_K"], df["STO_D"] = stoch(df["high"], df["low"], df["close"], 14, 3)
    df["CCI"] = cci(df["high"], df["low"], df["close"], 20)
    df["MFI"] = mfi(df["high"], df["low"], df["close"], df["vol"], 14)
    df["WILLR"] = williams_r(df["high"], df["low"], df["close"], 14)
    df["VOL_SMA"] = df["vol"].rolling(20).mean()
    df["ATR"] = atr(df["high"], df["low"], df["close"], 14)
    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100

    df = df.dropna()
    if df.empty:
        return df, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    bb_pos = "above" if last["close"] > last["BB_upper"] else ("below" if last["close"] < last["BB_lower"] else "inside")
    ma_cross = "golden" if (prev["MA_fast"] <= prev["MA_slow"] and last["MA_fast"] > last["MA_slow"]) else (
        "dead" if (prev["MA_fast"] >= prev["MA_slow"] and last["MA_fast"] < last["MA_slow"]) else "flat"
    )
    macd_cross = "golden" if (prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]) else (
        "dead" if (prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]) else "flat"
    )
    vol_spike = True if (safe_float(last["VOL_SMA"], 0) > 0 and last["vol"] >= last["VOL_SMA"] * 2.0) else False

    # RSI 과매도/과매수 "해소" 감지(핵심)
    rsi_oversold = prev["RSI"] < 30 or last["RSI"] < 30
    rsi_oversold_recovered = (prev["RSI"] < 30) and (last["RSI"] >= 30)
    rsi_overbought = prev["RSI"] > 70 or last["RSI"] > 70
    rsi_overbought_recovered = (prev["RSI"] > 70) and (last["RSI"] <= 70)

    # confirmation 후보들(2단계 확인)
    confirm_bb_mid_reclaim = (prev["close"] <= prev["BB_mid"]) and (last["close"] > last["BB_mid"])
    confirm_macd_hist_turn = (last["MACD_hist"] > prev["MACD_hist"])
    confirm_vol_recover = (last["vol"] >= last["VOL_SMA"])

    sweep_flag, sweep_type = detect_sweep(df)

    status = {
        "RSI_flow": f"{prev['RSI']:.1f}->{last['RSI']:.1f}",
        "BB_pos": bb_pos,
        "MA_cross": ma_cross,
        "MACD_cross": macd_cross,
        "ADX": float(last["ADX"]),
        "STO": f"{last['STO_K']:.1f}/{last['STO_D']:.1f}",
        "CCI": float(last["CCI"]),
        "MFI": float(last["MFI"]),
        "WILLR": float(last["WILLR"]),
        "VOL_SPIKE": bool(vol_spike),
        "ATR_PCT": float(last["ATR_PCT"]),
    }

    values = {
        "close": float(last["close"]),
        "RSI": float(last["RSI"]),
        "BB_mid": float(last["BB_mid"]),
        "BB_upper": float(last["BB_upper"]),
        "BB_lower": float(last["BB_lower"]),
        "MA_fast": float(last["MA_fast"]),
        "MA_slow": float(last["MA_slow"]),
        "MACD": float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
        "MACD_hist": float(last["MACD_hist"]),
        "ADX": float(last["ADX"]),
        "STO_K": float(last["STO_K"]),
        "STO_D": float(last["STO_D"]),
        "CCI": float(last["CCI"]),
        "MFI": float(last["MFI"]),
        "WILLR": float(last["WILLR"]),
        "VOL": float(last["vol"]),
        "VOL_SMA20": float(last["VOL_SMA"]),
        "ATR_PCT": float(last["ATR_PCT"]),
    }

    signals = {
        "rsi_oversold": bool(rsi_oversold),
        "rsi_oversold_recovered": bool(rsi_oversold_recovered),
        "rsi_overbought": bool(rsi_overbought),
        "rsi_overbought_recovered": bool(rsi_overbought_recovered),
        "confirm_bb_mid_reclaim": bool(confirm_bb_mid_reclaim),
        "confirm_macd_hist_turn": bool(confirm_macd_hist_turn),
        "confirm_vol_recover": bool(confirm_vol_recover),
        "sweep": bool(sweep_flag),
        "sweep_type": sweep_type,
    }

    return df, {"last": last, "prev": prev, "status": status, "values": values, "signals": signals}

def build_pack(df5: pd.DataFrame, df1h: pd.DataFrame, cfg: dict):
    """
    - 5m 지표 + 1h 구조/추세 컨펌
    - 레짐/구조 피처 포함
    """
    df5, p5 = calc_indicators(df5)
    if p5 is None:
        return None

    # 1h 컨펌
    df1h, p1 = (df1h, None)
    if df1h is not None and not df1h.empty:
        df1h, p1 = calc_indicators(df1h)

    # 구조
    struct5 = classify_structure(df5, window=int(cfg.get("swing_window", 5)), lookback=int(cfg.get("structure_lookback", 120)))
    struct1 = classify_structure(df1h, window=max(3, int(cfg.get("swing_window", 5)) - 2), lookback=200) if (df1h is not None and not df1h.empty) else {"structure": "unknown"}

    # 레짐
    ma_slope = rolling_slope(df5["MA_slow"], n=12)
    regime = classify_regime(p5["values"]["ADX"], p5["values"]["ATR_PCT"], ma_slope)

    # HTF 추세
    htf_trend = struct1.get("structure", "unknown")

    # 상대강도
    rs_score = relative_strength_score(df5)

    pack = {
        "df5": df5,
        "p5": p5,
        "df1h": df1h,
        "p1": p1,
        "structure_5m": struct5,
        "structure_1h": struct1,
        "regime": regime,
        "htf_trend": htf_trend,
        "rs_score": rs_score,
    }
    return pack

# =========================================================
# TradingView (dark)
# =========================================================
def render_tradingview(symbol_tv: str, interval: str = "5", height: int = 520):
    html = f"""
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "autosize": true,
    "symbol": "{symbol_tv}",
    "interval": "{interval}",
    "timezone": "Asia/Seoul",
    "theme": "dark",
    "style": "1",
    "locale": "kr",
    "toolbar_bg": "#1f1f1f",
    "enable_publishing": false,
    "hide_top_toolbar": false,
    "hide_legend": false,
    "save_image": false,
    "container_id": "tradingview_chart"
  }}
  );
  </script>
</div>
"""
    components.html(html, height=height, scrolling=False)

# =========================================================
# Telegram util + 메뉴
# =========================================================
def tg_send(token, chat_id, text, reply_markup=None):
    if not token or not chat_id:
        return
    try:
        payload = {"chat_id": chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data=payload, timeout=6)
    except:
        pass

def tg_answer(token, callback_query_id):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            data={"callback_query_id": callback_query_id},
            timeout=5
        )
    except:
        pass

def tg_send_document(token, chat_id, filepath, caption=""):
    if not token or not chat_id or not os.path.exists(filepath):
        return
    try:
        with open(filepath, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": f},
                timeout=15
            )
    except:
        pass

TG_MENU = {
    "inline_keyboard": [
        [{"text": "📊 브리핑", "callback_data": "brief"},
         {"text": "🌍 스캔(5)", "callback_data": "scan"}],
        [{"text": "💰 잔고", "callback_data": "balance"},
         {"text": "📌 포지션", "callback_data": "pos"}],
        [{"text": "🧾 로그(한줄)", "callback_data": "log_recent"},
         {"text": "📎 CSV파일", "callback_data": "log_file"}],
        [{"text": "🧠 투명성", "callback_data": "transparency"},
         {"text": "🤖 ON/OFF", "callback_data": "toggle"}],
        [{"text": "🛑 전량청산", "callback_data": "close_all"}],
    ]
}

# =========================================================
# 경제 캘린더 (뉴스회피)
# =========================================================
def fetch_econ_calendar():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=7)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list):
            return pd.DataFrame()

        rows = []
        now = utc_now()
        for ev in data:
            date_s = ev.get("date")
            time_s = ev.get("time") or "00:00"
            if not date_s:
                continue
            try:
                dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M")
            except:
                continue

            if dt < now - timedelta(days=1) or dt > now + timedelta(days=8):
                continue

            impact = (ev.get("impact") or "").lower()
            imp_ko = "높음" if "high" in impact else ("중간" if "medium" in impact else ("낮음" if "low" in impact else ""))

            title = ev.get("title", "")
            currency = ev.get("currency", "")

            rows.append({
                "utc_dt": dt,
                "날짜(UTC)": dt.strftime("%m-%d"),
                "시간(UTC)": time_s,
                "통화": currency,
                "중요도": imp_ko,
                "지표": title,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("utc_dt", ascending=True)
    except:
        return pd.DataFrame()

def is_news_block(ai_global: dict, cal_df: pd.DataFrame):
    if not ai_global.get("news_avoid", True):
        return (False, None)
    if cal_df is None or cal_df.empty:
        return (False, None)

    before = int(ai_global.get("news_block_before_min", 15))
    after = int(ai_global.get("news_block_after_min", 15))

    now = utc_now()
    for _, r in cal_df.iterrows():
        if str(r.get("중요도","")) != "높음":
            continue
        dt = r.get("utc_dt")
        if not isinstance(dt, datetime):
            continue
        if dt - timedelta(minutes=before) <= now <= dt + timedelta(minutes=after):
            return (True, f"{r.get('통화','')} {r.get('지표','')} ({r.get('중요도','')})")
    return (False, None)

# =========================================================
# 포지션/주문 유틸
# =========================================================
def normalize_side(p_side: str):
    s = (p_side or "").lower()
    if s in ["long", "buy"]:
        return "long"
    if s in ["short", "sell"]:
        return "short"
    return s or "long"

def get_active_positions(ex, symbols):
    try:
        ps = ex.fetch_positions(symbols=symbols)
        act = []
        for p in ps:
            if safe_float(p.get("contracts", 0)) > 0:
                p["side"] = normalize_side(p.get("side", ""))
                act.append(p)
        return act
    except:
        return []

def close_position_market(ex, symbol, side, contracts):
    close_side = "sell" if normalize_side(side) == "long" else "buy"
    try:
        ex.create_market_order(symbol, close_side, contracts, params={"reduceOnly": True})
        return True
    except:
        try:
            ex.create_market_order(symbol, close_side, contracts)
            return True
        except:
            return False

# =========================================================
# 레이더(=AI 호출 전 필터): "휩쏘 줄이고 비용 줄임"
# =========================================================
def radar_should_call_ai(symbol: str, pack: dict, mode: str):
    """
    - 과매도 진입 금지
    - 과매도 해소 + confirmation 후보가 있을 때
    - 또는 추세 재개/돌파 시점
    """
    p5 = pack["p5"]
    s = p5["status"]
    sig = p5["signals"]
    regime = pack["regime"]
    htf = pack["htf_trend"]

    # SAFE는 더 엄격
    strict = (mode == "SAFE")

    # RANGE는 신호 품질 낮음(완전 배제 X, but 엄격)
    if regime == "RANGE" and strict:
        # 확실한 반전(해소+확인) 아니면 호출 안 함
        if not (sig["rsi_oversold_recovered"] or sig["rsi_overbought_recovered"]):
            return False

    # 눌림목(상승) 시그널: oversold 회복 + 확인
    pullback_long = (htf == "uptrend") and sig["rsi_oversold_recovered"] and (
        sig["confirm_bb_mid_reclaim"] or sig["confirm_macd_hist_turn"] or sig["confirm_vol_recover"]
    )

    # 되돌림 숏(하락) 시그널: overbought 회복 + 확인
    pullback_short = (htf == "downtrend") and sig["rsi_overbought_recovered"] and (
        sig["confirm_bb_mid_reclaim"] or sig["confirm_macd_hist_turn"] or sig["confirm_vol_recover"]
    )

    # 추세 재개(크로스/ADX)
    trend_resume = (safe_float(s.get("ADX", 0), 0) >= (25 if strict else 22)) and (
        s.get("MA_cross") in ["golden", "dead"] or s.get("MACD_cross") in ["golden", "dead"]
    )

    # 스윕 감지면, 오히려 "바로 진입"이 아니라 "AI 판단"을 붙여서 선별
    if sig["sweep"]:
        return True

    return pullback_long or pullback_short or trend_resume

# =========================================================
# AI 결정
# =========================================================
INDICATOR_LIST = [
    "RSI(흐름)", "볼린저 위치/중단선", "MA_fast/MA_slow", "MACD/Hist", "ADX",
    "Stoch", "CCI", "MFI", "Williams %R", "거래량", "ATR%(변동성)", "구조(HH/HL)", "상위TF 컨펌", "스윕 필터"
]

def ai_decide(symbol: str, pack: dict, state: dict, mode: str, cfg: dict):
    """
    핵심 요구 반영:
    - 워뇨띠 베이스(눌림목/추세/구조)
    - RSI 과매도 '진입' 금지, 과매도 '해소' 진입 우선
    - 손익비/레버/리스크는 상황에 따라 유동(단, 가드레일 옵션)
    - 순환매(상대강도) 고려
    - 성공/실패 케이스 학습(최근 손실 요약)
    """
    p5 = pack["p5"]
    s = p5["status"]
    v = p5["values"]
    sig = p5["signals"]
    struct5 = pack["structure_5m"]
    struct1 = pack["structure_1h"]
    regime = pack["regime"]
    htf = pack["htf_trend"]
    rs_score = pack["rs_score"]

    mistakes = summarize_recent_mistakes()

    # AI 입력(투명성 저장)
    ai_input = {
        "symbol": symbol,
        "base_tf": cfg.get("base_tf", "5m"),
        "confirm_tf": cfg.get("confirm_tf", "1h"),
        "mode": mode,
        "goal": "원금손실 최소화 + 기회 포착 + 회고로 성장",
        "wonyo_core": "과매도 진입 금지, 과매도 해소(복귀) + 확인 후 진입. 추세/구조를 먼저 본다.",
        "indicators_used": INDICATOR_LIST,
        "values_5m": v,
        "status_5m": s,
        "signals_5m": sig,
        "structure_5m": struct5,
        "structure_1h": struct1,
        "regime": regime,
        "htf_trend": htf,
        "relative_strength_score": rs_score,
        "consec_losses": int(state.get("consec_losses", 0)),
        "open_trades_count": len(state.get("trades", {})),
        "prefs": {
            "prefer_short_sl": bool(cfg.get("prefer_short_sl", True)),
            "prefer_long_tp_trend": bool(cfg.get("prefer_long_tp_trend", True)),
            "allow_tp_extend": bool(cfg.get("allow_tp_extend", True)),
        },
        "recent_mistakes_top5": mistakes
    }
    state.setdefault("last_ai_inputs", {})[symbol] = ai_input
    save_runtime_state(state)

    # OpenAI 없으면 기본값
    if openai_client is None:
        out = {
            "decision": "hold",
            "confidence": 0,
            "setup_tag": "NO_AI",
            "risk": {
                "leverage": 5,
                "risk_pct": 8,
                "sl_gap": max(1.0, v.get("ATR_PCT", 1.0) * 0.9),
                "tp_target": max(2.0, v.get("ATR_PCT", 1.0) * 2.2),
                "tp1_gap": 0.6, "tp1_size": 30,
                "tp2_gap": 1.4, "tp2_size": 30,
                "use_trailing": True,
                "trail_start": 1.0,
                "trail_gap_atr_mult": 1.2,
                "breakeven_after_tp1": True,
                "scale_in": {"enabled": False}
            },
            "global": state.get("ai_global", default_runtime_state()["ai_global"]),
            "fail_tag_if_loss": "unknown",
            "reason": "AI키 없음: 관망",
            "easy": "AI키가 없어서 자동 판단을 못해요. 지금은 관망이에요.",
            "one_liner": f"{symbol} HOLD (AI키 없음)"
        }
        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

    # 모드 룰(프롬프트)
    if mode == "SAFE":
        mode_rules = """
[안전모드]
- 애매하면 HOLD. 진입은 확실한 자리만.
- RSI '과매도/과매수' 그 자체에 진입 금지. 해소(복귀) + 확인 후 진입.
- 원금손실 최소화. 연속손실 시 더 빨리 멈춤/쿨다운 강화.
- TP는 길게 가져가되, TP1에서 리스크 회수(부분익절) 후 브레이크이븐.
"""
        conf_hint = "확신도는 쉽게 80 넘기지 말고, 진짜 좋을 때 85~95."
    else:
        mode_rules = """
[공격모드(공격+선별)]
- 공격적이되 선별이 핵심. 애매하면 HOLD.
- 손절은 짧게, 익절은 추세면 길게(트레일링/TP연장).
- RSI 과매도 진입 금지. 과매도 해소 + 확인 후 진입을 우선.
"""
        conf_hint = "확신도 남발 금지. 근거 강할 때만 80~95."

    system = f"""
너는 24시간 자동매매 트레이딩 매니저다.
목표: 원금손실 최소화 + 짧은시간 기회 포착 + 회고로 성장.
베이스: 워뇨띠 매매법(추세/눌림목/반등 타이밍) + 손익비/레버/순환매.

[핵심 금지]
- RSI < 30(과매도) 상태에서 '바로 매수 진입' 금지.
- RSI > 70(과매수) 상태에서 '바로 매도 진입' 금지.
=> 반드시 "해소(복귀)" + confirmation 최소 1개로 진입.

[필수 고려]
- 구조(HH/HL), 상위TF 컨펌(1h) 먼저.
- 레짐(TREND/RANGE/VOLATILE/MIXED)에 따라 공격/방어 조절.
- 순환매: 상대강도 점수 높을수록 우선순위.
- 손익비: TP1에서 일부 익절로 리스크 회수, 나머지는 트레일링.
- 트레일링 폭은 ATR 기반으로 유동적일 것.
- 스윕(sweep) 감지 시: 1) 바로 진입보다 "분할 진입" 또는 "확인 후 진입" 선호.

{mode_rules}
{conf_hint}

출력은 반드시 JSON 하나.
스키마:
{{
 "decision":"buy/sell/hold",
 "confidence":0-100,
 "setup_tag":"pullback_reclaim/trend_resume/sweep_reversal/range_meanrev/...",
 "risk":{{
   "leverage":1-50,
   "risk_pct":1-100,
   "sl_gap":0.3-30.0,
   "tp_target":0.3-80.0,
   "tp1_gap":0.1-10.0, "tp1_size":10-90,
   "tp2_gap":0.1-30.0, "tp2_size":10-90,
   "use_trailing":true/false,
   "trail_start":0.1-30.0,
   "trail_gap_atr_mult":0.5-5.0,
   "breakeven_after_tp1":true/false,
   "scale_in":{{"enabled":true/false, "first_pct":10-90, "second_pct":10-90, "trigger":"confirm"}}
 }},
 "global":{{
   "cooldown_minutes":0-240,
   "max_consec_losses":1-10,
   "pause_minutes":5-240,
   "news_avoid":true/false,
   "news_block_before_min":0-60,
   "news_block_after_min":0-60
 }},
 "fail_tag_if_loss":"whipsaw/premature_entry/trend_break/news/overtrade/...",
 "reason":"전문가용 근거(짧게)",
 "easy":"아주 쉬운 설명(2~4줄)",
 "one_liner":"한줄평"
}}
"""

    user = json.dumps(ai_input, ensure_ascii=False)

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.35
        )
        out = json.loads(resp.choices[0].message.content)

        # 기본값 보정
        out.setdefault("decision", "hold")
        out.setdefault("confidence", 0)
        out.setdefault("setup_tag", "unknown")
        out.setdefault("risk", {})
        out.setdefault("global", {})
        out.setdefault("fail_tag_if_loss", "unknown")
        out.setdefault("reason", "")
        out.setdefault("easy", "")
        out.setdefault("one_liner", "")

        r = out["risk"]
        g = out["global"]

        def clampf(v, lo, hi, d):
            v = safe_float(v, d)
            return max(lo, min(hi, v))

        r["leverage"] = int(clampf(r.get("leverage", 5), 1, 50, 5))
        r["risk_pct"] = clampf(r.get("risk_pct", 10), 1, 100, 10)
        r["sl_gap"] = clampf(r.get("sl_gap", 1.2), 0.3, 30.0, 1.2)
        r["tp_target"] = clampf(r.get("tp_target", 2.5), 0.3, 80.0, 2.5)
        r["tp1_gap"] = clampf(r.get("tp1_gap", 0.6), 0.1, 10.0, 0.6)
        r["tp1_size"] = int(clampf(r.get("tp1_size", 30), 10, 90, 30))
        r["tp2_gap"] = clampf(r.get("tp2_gap", 1.4), 0.1, 30.0, 1.4)
        r["tp2_size"] = int(clampf(r.get("tp2_size", 30), 10, 90, 30))
        r["use_trailing"] = bool(r.get("use_trailing", True))
        r["trail_start"] = clampf(r.get("trail_start", 1.0), 0.1, 30.0, 1.0)
        r["trail_gap_atr_mult"] = clampf(r.get("trail_gap_atr_mult", 1.2), 0.5, 5.0, 1.2)
        r["breakeven_after_tp1"] = bool(r.get("breakeven_after_tp1", True))
        r.setdefault("scale_in", {"enabled": False})
        r["scale_in"].setdefault("enabled", False)
        r["scale_in"]["enabled"] = bool(r["scale_in"].get("enabled", False))
        if r["scale_in"]["enabled"]:
            r["scale_in"]["first_pct"] = int(clampf(r["scale_in"].get("first_pct", 60), 10, 90, 60))
            r["scale_in"]["second_pct"] = int(clampf(r["scale_in"].get("second_pct", 40), 10, 90, 40))
            r["scale_in"]["trigger"] = str(r["scale_in"].get("trigger", "confirm"))

        g["cooldown_minutes"] = int(clampf(g.get("cooldown_minutes", 10), 0, 240, 10))
        g["max_consec_losses"] = int(clampf(g.get("max_consec_losses", 3), 1, 10, 3))
        g["pause_minutes"] = int(clampf(g.get("pause_minutes", 30), 5, 240, 30))
        g["news_avoid"] = bool(g.get("news_avoid", True))
        g["news_block_before_min"] = int(clampf(g.get("news_block_before_min", 15), 0, 60, 15))
        g["news_block_after_min"] = int(clampf(g.get("news_block_after_min", 15), 0, 60, 15))

        # 가드레일(옵션)
        if cfg.get("enable_hard_guardrails", True):
            if mode == "SAFE":
                r["leverage"] = min(r["leverage"], int(cfg.get("hard_max_leverage_safe", 10)))
                r["risk_pct"] = min(r["risk_pct"], float(cfg.get("hard_max_risk_pct_safe", 15.0)))
            else:
                r["leverage"] = min(r["leverage"], int(cfg.get("hard_max_leverage_aggressive", 20)))
                r["risk_pct"] = min(r["risk_pct"], float(cfg.get("hard_max_risk_pct_aggressive", 30.0)))

        # one_liner 기본
        if not out.get("one_liner"):
            out["one_liner"] = f"{symbol} {out.get('decision','hold').upper()} conf {out.get('confidence',0)}"

        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

    except Exception as e:
        out = {
            "decision": "hold",
            "confidence": 0,
            "setup_tag": "ai_error",
            "risk": {
                "leverage": 5, "risk_pct": 8,
                "sl_gap": 1.2, "tp_target": 2.5,
                "tp1_gap": 0.6, "tp1_size": 30,
                "tp2_gap": 1.4, "tp2_size": 30,
                "use_trailing": True,
                "trail_start": 1.0,
                "trail_gap_atr_mult": 1.2,
                "breakeven_after_tp1": True,
                "scale_in": {"enabled": False}
            },
            "global": state.get("ai_global", default_runtime_state()["ai_global"]),
            "fail_tag_if_loss": "unknown",
            "reason": f"AI 오류로 관망: {e}",
            "easy": "AI 호출이 실패했어요. 지금은 관망이에요.",
            "one_liner": f"{symbol} HOLD (AI err)"
        }
        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

# =========================================================
# AI 회고(한줄평 + 개선)
# =========================================================
def ai_review_trade(trade_row: dict):
    if openai_client is None:
        return "AI키 없음: 수동 회고 필요"
    system = """
너는 트레이딩 코치다.
요청: 아래 거래를 바탕으로 "한줄평" + "회고"를 작성해라.
형식:
1) 한줄평: (최대 25자)
2) 회고: 잘한 점 1 / 아쉬운 점 1 / 다음 행동 1
손절이면: 휩쏘/성급진입/구조붕괴/뉴스 등 실패유형을 반드시 짚고 개선 한 가지.
익절이면: 유지할 습관 하나 + 다음에 더 좋게 할 하나.
"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(trade_row, ensure_ascii=False)}],
            temperature=0.35
        )
        return (resp.choices[0].message.content or "").strip()
    except:
        return "AI 회고 실패"

# =========================================================
# Telegram bot thread
# =========================================================
def telegram_bot_thread():
    bot_ex = create_exchange()
    state = load_runtime_state()
    cal_cache = {"t": 0, "df": pd.DataFrame()}

    def get_calendar_cached():
        if time.time() - cal_cache["t"] > 600:
            cal_cache["df"] = fetch_econ_calendar()
            cal_cache["t"] = time.time()
        return cal_cache["df"]

    cfg = load_settings()
    if cfg.get("telegram_enabled", True):
        tg_send(tg_token, tg_id, "🚀 봇 시작!\n(Streamlit=제어판 / Telegram=보고&조회)", reply_markup=TG_MENU)

    last_manage = 0
    last_scan = 0
    last_report = 0

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()
            mode = cfg.get("trade_mode", "SAFE").upper()
            base_tf = cfg.get("base_tf", "5m")
            confirm_tf = cfg.get("confirm_tf", "1h")

            # 데일리 리셋
            try:
                bal = bot_ex.fetch_balance({"type": "swap"})
                equity = safe_float(bal["USDT"]["total"])
            except:
                equity = safe_float(state.get("day_start_equity", 0))
            maybe_roll_daily_state(state, equity)

            # 텔레그램 콜백
            if cfg.get("telegram_enabled", True) and tg_token and tg_id:
                try:
                    res = requests.get(
                        f"https://api.telegram.org/bot{tg_token}/getUpdates",
                        params={"offset": int(state.get("tg_offset", 0)) + 1, "timeout": 1},
                        timeout=6
                    ).json()

                    if res.get("ok"):
                        for up in res.get("result", []):
                            state["tg_offset"] = up["update_id"]
                            save_runtime_state(state)

                            if "callback_query" not in up:
                                continue
                            cb = up["callback_query"]
                            data = cb.get("data", "")
                            cid = cb["message"]["chat"]["id"]
                            cb_id = cb["id"]

                            if data == "balance":
                                try:
                                    bal = bot_ex.fetch_balance({"type": "swap"})
                                    eq = safe_float(bal["USDT"]["total"])
                                    fr = safe_float(bal["USDT"]["free"])
                                    tg_send(tg_token, cid, f"💰 잔고\n총자산: ${eq:,.2f}\n주문가능: ${fr:,.2f}", reply_markup=TG_MENU)
                                except:
                                    tg_send(tg_token, cid, "잔고 조회 실패", reply_markup=TG_MENU)

                            elif data == "pos":
                                ps = get_active_positions(bot_ex, TARGET_COINS)
                                if not ps:
                                    tg_send(tg_token, cid, "📌 포지션 없음", reply_markup=TG_MENU)
                                else:
                                    lines = [f"📌 포지션 현황 ({mode})"]
                                    for p in ps:
                                        sym = p.get("symbol", "")
                                        side = normalize_side(p.get("side", ""))
                                        roi = safe_float(p.get("percentage", 0))
                                        lev = p.get("leverage", "?")
                                        qty = safe_float(p.get("contracts", 0))
                                        lines.append(f"- {sym} | {side.upper()} x{lev} | 수량 {qty} | ROI {roi:.2f}%")
                                    tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "toggle":
                                cfg2 = load_settings()
                                cfg2["auto_trade"] = not cfg2.get("auto_trade", False)
                                save_settings(cfg2)
                                tg_send(tg_token, cid, f"🤖 자동매매 {'ON' if cfg2['auto_trade'] else 'OFF'}", reply_markup=TG_MENU)

                            elif data == "log_recent":
                                df = read_trade_log(12)
                                if df.empty:
                                    tg_send(tg_token, cid, "🧾 로그 없음", reply_markup=TG_MENU)
                                else:
                                    lines = ["🧾 최근 로그(한줄)"]
                                    for _, r in df.iterrows():
                                        lines.append(f"- {r.get('OneLine','')}")
                                    tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "log_file":
                                if os.path.exists(TRADE_LOG_FILE):
                                    tg_send_document(tg_token, cid, TRADE_LOG_FILE, caption="📎 trade_log.csv")
                                else:
                                    tg_send(tg_token, cid, "CSV 파일이 아직 없어요(첫 청산 이후 생성).", reply_markup=TG_MENU)

                            elif data == "transparency":
                                # 최근 AI 출력 요약
                                outs = state.get("last_ai_outputs", {})
                                if not outs:
                                    tg_send(tg_token, cid, "최근 AI 판단 기록이 없어요.\n(브리핑/스캔 눌러보세요)", reply_markup=TG_MENU)
                                else:
                                    # 가장 최근 것 비슷하게 1개만(심볼순)
                                    sym = TARGET_COINS[0]
                                    o = outs.get(sym) or next(iter(outs.values()))
                                    tg_send(tg_token, cid, f"🧠 AI 투명성(요약)\n- 지표: {', '.join(INDICATOR_LIST[:7])}...\n- 예시결정: {o.get('one_liner','')}\n- 쉬운설명: {o.get('easy','')}", reply_markup=TG_MENU)

                            elif data in ["brief", "scan"]:
                                lines = [f"📊 브리핑 ({mode})" if data == "brief" else f"🌍 스캔(5) ({mode})"]

                                # 상대강도 랭킹으로 먼저 정렬(D)
                                try:
                                    ranks = []
                                    for sym in TARGET_COINS:
                                        ohlcv = bot_ex.fetch_ohlcv(sym, base_tf, limit=250)
                                        df5 = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                                        ranks.append((sym, relative_strength_score(df5)))
                                    ranks.sort(key=lambda x: x[1], reverse=True)
                                    scan_list = [x[0] for x in ranks]
                                except:
                                    scan_list = TARGET_COINS[:]

                                for sym in scan_list:
                                    try:
                                        ohlcv5 = bot_ex.fetch_ohlcv(sym, base_tf, limit=250)
                                        df5 = pd.DataFrame(ohlcv5, columns=["time","open","high","low","close","vol"])
                                        ohlcv1 = bot_ex.fetch_ohlcv(sym, confirm_tf, limit=250)
                                        df1 = pd.DataFrame(ohlcv1, columns=["time","open","high","low","close","vol"])
                                        pack = build_pack(df5, df1, cfg)
                                        if pack is None:
                                            continue

                                        # 레이더(조건 근접할 때만 AI)
                                        if cfg.get("radar_use", True) and not radar_should_call_ai(sym, pack, mode):
                                            continue

                                        out = ai_decide(sym, pack, state, mode, cfg)

                                        # 글로벌옵션 자동 적용
                                        if cfg.get("use_ai_global", True) and isinstance(out.get("global", {}), dict):
                                            state["ai_global"] = out["global"]
                                            save_runtime_state(state)

                                        r = out.get("risk", {})
                                        lines.append(
                                            f"\n[{sym}] {out.get('decision','hold').upper()} (conf {out.get('confidence',0)}%)\n"
                                            f"- 레짐:{pack['regime']} | HTF:{pack['htf_trend']} | RS:{pack['rs_score']:.1f}\n"
                                            f"- x{r.get('leverage')} | 진입금액 {r.get('risk_pct')}% | SL {r.get('sl_gap')}% | TP {r.get('tp_target')}%\n"
                                            f"- 한줄: {out.get('one_liner','')}"
                                        )
                                    except:
                                        continue
                                tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "close_all":
                                ps = get_active_positions(bot_ex, TARGET_COINS)
                                closed = 0
                                for p in ps:
                                    sym = p.get("symbol")
                                    side = normalize_side(p.get("side", "long"))
                                    contracts = safe_float(p.get("contracts", 0))
                                    if contracts <= 0:
                                        continue
                                    if close_position_market(bot_ex, sym, side, contracts):
                                        closed += 1
                                tg_send(tg_token, cid, f"🛑 전량청산 요청 완료(대상 {closed}개)", reply_markup=TG_MENU)

                            tg_answer(tg_token, cb_id)

                except:
                    pass

            # 자동매매 OFF면 조회/보고만
            if not cfg.get("auto_trade", False):
                time.sleep(0.5)
                continue

            # pause
            if is_paused(state):
                time.sleep(1.0)
                continue

            ai_global = state.get("ai_global", default_runtime_state()["ai_global"])
            cal = get_calendar_cached()
            blocked, why = is_news_block(ai_global, cal)

            ts = time.time()

            # -------------------------------------------------
            # 1) 포지션 관리 (부분익절/브레이크이븐/트레일링/SL/TP)
            # -------------------------------------------------
            if ts - last_manage >= int(cfg.get("manage_interval_sec", 2)):
                last_manage = ts

                positions = get_active_positions(bot_ex, TARGET_COINS)
                for p in positions:
                    sym = p.get("symbol")
                    side = normalize_side(p.get("side", "long"))
                    contracts = safe_float(p.get("contracts", 0))
                    if contracts <= 0:
                        continue

                    # Bitget percentage는 보통 포지션 ROI%로 나옴(레버리지 반영 여부는 거래소에 따라 다를 수 있음)
                    roi = safe_float(p.get("percentage", 0))
                    pnl_usdt = safe_float(p.get("unrealizedPnl", 0))
                    entry_price = safe_float(p.get("entryPrice", 0))
                    mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0)) or entry_price

                    meta = state.get("trades", {}).get(sym, {})
                    if not meta:
                        meta = {
                            "entry_time": now_str(),
                            "entry_price": entry_price if entry_price > 0 else mark,
                            "qty": contracts,
                            "side": side,
                            "mode": mode,
                            "risk": {
                                "leverage": safe_float(p.get("leverage", 1), 1),
                                "risk_pct": "",
                                "sl_gap": 1.2,
                                "tp_target": 2.5,
                                "tp1_gap": 0.6, "tp1_size": 30,
                                "tp2_gap": 1.4, "tp2_size": 30,
                                "use_trailing": True,
                                "trail_start": 1.0,
                                "trail_gap_atr_mult": 1.2,
                                "breakeven_after_tp1": True,
                                "scale_in": {"enabled": False}
                            },
                            "tp1_done": False,
                            "tp2_done": False,
                            "tp_extended": False,
                            "be_enabled": False,
                            "be_roi": 0.2,    # 브레이크이븐 기준(ROI%)
                            "best_roi": roi,  # MFE 추적
                            "worst_roi": roi, # MAE 추적
                            "best_price": mark,
                            "reason": "",
                            "easy": "",
                            "setup_tag": "unknown",
                            "fail_tag_if_loss": "unknown"
                        }
                        state.setdefault("trades", {})[sym] = meta
                        save_runtime_state(state)

                    # MFE/MAE 업데이트
                    meta["best_roi"] = max(safe_float(meta.get("best_roi", roi), roi), roi)
                    meta["worst_roi"] = min(safe_float(meta.get("worst_roi", roi), roi), roi)

                    # best_price 업데이트(트레일링용)
                    best_price = safe_float(meta.get("best_price", mark), mark)
                    if side == "long":
                        best_price = max(best_price, mark)
                    else:
                        best_price = min(best_price, mark)
                    meta["best_price"] = best_price
                    save_runtime_state(state)

                    r = meta.get("risk", {})
                    lev = safe_float(r.get("leverage", p.get("leverage", 1)), 1)
                    risk_pct = r.get("risk_pct", "")
                    sl_gap = safe_float(r.get("sl_gap", 1.2), 1.2)
                    tp_target = safe_float(r.get("tp_target", 2.5), 2.5)
                    tp1_gap = safe_float(r.get("tp1_gap", 0.6), 0.6)
                    tp1_size = int(safe_float(r.get("tp1_size", 30), 30))
                    tp2_gap = safe_float(r.get("tp2_gap", 1.4), 1.4)
                    tp2_size = int(safe_float(r.get("tp2_size", 30), 30))

                    use_trailing = bool(r.get("use_trailing", True))
                    trail_start = safe_float(r.get("trail_start", 1.0), 1.0)
                    trail_gap_atr_mult = safe_float(r.get("trail_gap_atr_mult", 1.2), 1.2)
                    be_after_tp1 = bool(r.get("breakeven_after_tp1", True))

                    # -------------------------------------------------
                    # TP1: 부분익절 + (옵션) 브레이크이븐 활성화
                    # -------------------------------------------------
                    if (not meta.get("tp1_done", False)) and roi >= tp1_gap:
                        close_qty = safe_float(contracts * (tp1_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp1_done"] = True
                            if be_after_tp1:
                                meta["be_enabled"] = True
                            save_runtime_state(state)
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP1 부분익절: {sym} (+{roi:.2f}%)\n이후 원금보호(브레이크이븐) {'ON' if meta['be_enabled'] else 'OFF'}", reply_markup=TG_MENU)

                    # -------------------------------------------------
                    # TP2: 추가 부분익절
                    # -------------------------------------------------
                    if (not meta.get("tp2_done", False)) and roi >= tp2_gap:
                        close_qty = safe_float(contracts * (tp2_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp2_done"] = True
                            save_runtime_state(state)
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP2 추가익절: {sym} (+{roi:.2f}%)", reply_markup=TG_MENU)

                    # -------------------------------------------------
                    # 브레이크이븐: TP1 이후 ROI가 다시 내려오면 청산
                    # -------------------------------------------------
                    if meta.get("be_enabled", False) and roi <= safe_float(meta.get("be_roi", 0.2), 0.2):
                        ok = close_position_market(bot_ex, sym, side, contracts)
                        if ok:
                            event = "BE(원금보호)"
                            trade_row = {
                                "Time": now_str(), "Mode": meta.get("mode", mode),
                                "Symbol": sym, "Event": event, "Side": side,
                                "Qty": contracts, "EntryPrice": meta.get("entry_price", entry_price),
                                "ExitPrice": mark, "PnL_USDT": f"{pnl_usdt:.4f}",
                                "PnL_Percent": f"{roi:.2f}", "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Regime": meta.get("regime",""), "HTF_Trend": meta.get("htf_trend",""),
                                "SetupTag": meta.get("setup_tag",""), "FailTag": "",
                                "MFE_ROI": f"{safe_float(meta.get('best_roi', roi), roi):.2f}",
                                "MAE_ROI": f"{safe_float(meta.get('worst_roi', roi), roi):.2f}",
                                "Reason": str(meta.get("reason",""))[:200],
                                "Snapshot_JSON": json.dumps(meta.get("snapshot", {}), ensure_ascii=False)
                            }
                            review = ai_review_trade(trade_row)
                            trade_row["Review"] = review
                            trade_row["OneLine"] = make_oneline_summary(trade_row)
                            append_trade_log(trade_row)

                            # 손익에 따른 연속손실
                            if roi < 0:
                                state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                            else:
                                state["consec_losses"] = 0

                            if state["consec_losses"] >= int(ai_global.get("max_consec_losses", 3)):
                                state["pause_until"] = int(time.time() + int(ai_global.get("pause_minutes", 30)) * 60)

                            set_cooldown(state, sym, int(ai_global.get("cooldown_minutes", 10)))
                            state["trades"].pop(sym, None)
                            save_runtime_state(state)

                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"🛡️ 원금보호 청산: {sym} ({roi:.2f}%)\n{trade_row['OneLine']}", reply_markup=TG_MENU)
                        continue

                    # -------------------------------------------------
                    # ATR 기반 트레일링: trail_gap = ATR% * ATR_mult * leverage(근사)
                    # -------------------------------------------------
                    if use_trailing and roi >= trail_start:
                        # 현재 ATR%는 entry 기준으로 근사
                        atr_pct = safe_float(meta.get("atr_pct_at_entry", 1.0), 1.0)
                        trail_gap_roi = max(0.3, atr_pct * trail_gap_atr_mult * (lev if lev > 0 else 1))

                        # best_price -> mark drawdown(%price)
                        if side == "long":
                            dd_price = (best_price - mark) / best_price * 100 if best_price > 0 else 0
                        else:
                            dd_price = (mark - best_price) / best_price * 100 if best_price > 0 else 0

                        dd_roi = dd_price * (lev if lev > 0 else 1)  # ROI 근사
                        if dd_roi >= trail_gap_roi:
                            ok = close_position_market(bot_ex, sym, side, contracts)
                            if ok:
                                event = "TRAIL(청산)"
                                trade_row = {
                                    "Time": now_str(), "Mode": meta.get("mode", mode),
                                    "Symbol": sym, "Event": event, "Side": side,
                                    "Qty": contracts, "EntryPrice": meta.get("entry_price", entry_price),
                                    "ExitPrice": mark, "PnL_USDT": f"{pnl_usdt:.4f}",
                                    "PnL_Percent": f"{roi:.2f}", "Leverage": lev, "RiskPct": risk_pct,
                                    "TP_Target": tp_target, "SL_Target": sl_gap,
                                    "Regime": meta.get("regime",""), "HTF_Trend": meta.get("htf_trend",""),
                                    "SetupTag": meta.get("setup_tag",""), "FailTag": "",
                                    "MFE_ROI": f"{safe_float(meta.get('best_roi', roi), roi):.2f}",
                                    "MAE_ROI": f"{safe_float(meta.get('worst_roi', roi), roi):.2f}",
                                    "Reason": str(meta.get("reason",""))[:200],
                                    "Snapshot_JSON": json.dumps(meta.get("snapshot", {}), ensure_ascii=False)
                                }
                                review = ai_review_trade(trade_row)
                                trade_row["Review"] = review
                                trade_row["OneLine"] = make_oneline_summary(trade_row)
                                append_trade_log(trade_row)

                                if roi < 0:
                                    state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                                else:
                                    state["consec_losses"] = 0

                                if state["consec_losses"] >= int(ai_global.get("max_consec_losses", 3)):
                                    state["pause_until"] = int(time.time() + int(ai_global.get("pause_minutes", 30)) * 60)

                                set_cooldown(state, sym, int(ai_global.get("cooldown_minutes", 10)))
                                state["trades"].pop(sym, None)
                                save_runtime_state(state)

                                if cfg.get("telegram_enabled", True):
                                    tg_send(tg_token, tg_id, f"🏁 트레일링 청산: {sym} ({roi:.2f}%)\n{trade_row['OneLine']}", reply_markup=TG_MENU)
                            continue

                    # -------------------------------------------------
                    # SL/TP 도달 처리 + TP 연장(추세면)
                    # -------------------------------------------------
                    if roi <= -abs(sl_gap) or roi >= tp_target:
                        # TP 연장(추세에서 1회)
                        if roi >= tp_target and cfg.get("allow_tp_extend", True) and cfg.get("prefer_long_tp_trend", True):
                            if not meta.get("tp_extended", False) and meta.get("regime","") == "TREND":
                                meta["tp_extended"] = True
                                meta["risk"]["tp_target"] = float(tp_target) * float(cfg.get("tp_extend_mult", 1.7))
                                save_runtime_state(state)
                                if cfg.get("telegram_enabled", True):
                                    tg_send(tg_token, tg_id, f"📈 TP 도달 → 추세로 판단해 TP 1회 연장! {sym}\n새 TP: {meta['risk']['tp_target']:.2f}%", reply_markup=TG_MENU)
                                continue

                        event = "SL(손절)" if roi <= -abs(sl_gap) else "TP(익절)"
                        ok = close_position_market(bot_ex, sym, side, contracts)
                        if ok:
                            fail_tag = meta.get("fail_tag_if_loss","") if roi < 0 else ""
                            trade_row = {
                                "Time": now_str(), "Mode": meta.get("mode", mode),
                                "Symbol": sym, "Event": event, "Side": side,
                                "Qty": contracts, "EntryPrice": meta.get("entry_price", entry_price),
                                "ExitPrice": mark, "PnL_USDT": f"{pnl_usdt:.4f}",
                                "PnL_Percent": f"{roi:.2f}", "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Regime": meta.get("regime",""), "HTF_Trend": meta.get("htf_trend",""),
                                "SetupTag": meta.get("setup_tag",""), "FailTag": fail_tag,
                                "MFE_ROI": f"{safe_float(meta.get('best_roi', roi), roi):.2f}",
                                "MAE_ROI": f"{safe_float(meta.get('worst_roi', roi), roi):.2f}",
                                "Reason": str(meta.get("reason",""))[:200],
                                "Snapshot_JSON": json.dumps(meta.get("snapshot", {}), ensure_ascii=False)
                            }
                            review = ai_review_trade(trade_row)
                            trade_row["Review"] = review
                            trade_row["OneLine"] = make_oneline_summary(trade_row)
                            append_trade_log(trade_row)

                            if roi < 0:
                                state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                            else:
                                state["consec_losses"] = 0

                            if state["consec_losses"] >= int(ai_global.get("max_consec_losses", 3)):
                                state["pause_until"] = int(time.time() + int(ai_global.get("pause_minutes", 30)) * 60)

                            set_cooldown(state, sym, int(ai_global.get("cooldown_minutes", 10)))
                            state["trades"].pop(sym, None)
                            save_runtime_state(state)

                            if cfg.get("telegram_enabled", True):
                                emoji = "🩸" if roi < 0 else "🎉"
                                tg_send(tg_token, tg_id, f"{emoji} {event}: {sym} ({roi:.2f}%)\n{trade_row['OneLine']}", reply_markup=TG_MENU)

            # -------------------------------------------------
            # 2) 신규 진입 스캔 (로테이션 + RSI 해소 진입 강제)
            # -------------------------------------------------
            if ts - last_scan >= int(cfg.get("entry_scan_interval_sec", 10)):
                last_scan = ts

                if blocked:
                    state["last_bot_note"] = f"뉴스 회피: {why}"
                    save_runtime_state(state)
                else:
                    # 상대강도 랭킹으로 스캔 순서(D)
                    try:
                        ranks = []
                        for sym in TARGET_COINS:
                            ohlcv = bot_ex.fetch_ohlcv(sym, base_tf, limit=250)
                            df5 = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                            ranks.append((sym, relative_strength_score(df5)))
                        ranks.sort(key=lambda x: x[1], reverse=True)
                        scan_list = [x[0] for x in ranks]
                    except:
                        scan_list = TARGET_COINS[:]

                    # 모드별 확신 컷
                    conf_cut = 85 if mode == "SAFE" else 80

                    # 분할진입(2차) 먼저 처리: pending_scaleins
                    pending = state.get("pending_scaleins", {})
                    if pending:
                        for sym, info in list(pending.items()):
                            try:
                                # 이미 포지션이 없으면 삭제
                                if not get_active_positions(bot_ex, [sym]):
                                    state["pending_scaleins"].pop(sym, None)
                                    save_runtime_state(state)
                                    continue

                                # 조건(confirm): RSI>30 유지 + MACD_hist 상승 + BB_mid 위 복귀 중 하나
                                ohlcv5 = bot_ex.fetch_ohlcv(sym, base_tf, limit=250)
                                df5 = pd.DataFrame(ohlcv5, columns=["time","open","high","low","close","vol"])
                                ohlcv1 = bot_ex.fetch_ohlcv(sym, confirm_tf, limit=250)
                                df1 = pd.DataFrame(ohlcv1, columns=["time","open","high","low","close","vol"])
                                pack = build_pack(df5, df1, cfg)
                                if pack is None:
                                    continue
                                sig = pack["p5"]["signals"]
                                v = pack["p5"]["values"]

                                # confirm 조건
                                ok_confirm = (not sig["rsi_oversold"]) and (
                                    sig["confirm_bb_mid_reclaim"] or sig["confirm_macd_hist_turn"] or sig["confirm_vol_recover"]
                                )
                                if not ok_confirm:
                                    continue

                                # 2차 진입 실행
                                try:
                                    bal = bot_ex.fetch_balance({"type":"swap"})
                                    free_usdt = safe_float(bal["USDT"]["free"], 0)
                                except:
                                    continue
                                if free_usdt < 10:
                                    continue

                                lev = int(info.get("leverage", 5))
                                risk_pct = safe_float(info.get("risk_pct", 10), 10)
                                second_pct = int(info.get("second_pct", 40))

                                # 2차는 원래 risk_pct의 일부만(=second_pct)
                                use_usdt = free_usdt * (risk_pct / 100.0) * (second_pct / 100.0)
                                price = safe_float(v.get("close", 0), 0)
                                if price <= 0:
                                    continue
                                qty = (use_usdt * lev) / price
                                qty = safe_float(bot_ex.amount_to_precision(sym, qty), 0)
                                if qty <= 0:
                                    continue

                                # 방향
                                decision = info.get("decision", "buy")
                                bot_ex.create_market_order(sym, decision, qty)
                                state["pending_scaleins"].pop(sym, None)
                                state["last_bot_note"] = f"2차 분할진입 완료: {sym}"
                                save_runtime_state(state)

                                if cfg.get("telegram_enabled", True):
                                    tg_send(tg_token, tg_id, f"➕ 2차 분할진입: {sym}\n- 조건(confirm) 충족\n- 수량: {qty}", reply_markup=TG_MENU)

                            except:
                                continue

                    for sym in scan_list:
                        if is_paused(state) or in_cooldown(state, sym) or (sym in state.get("trades", {})):
                            continue
                        if get_active_positions(bot_ex, [sym]):
                            continue

                        # 잔고
                        try:
                            bal = bot_ex.fetch_balance({"type": "swap"})
                            free_usdt = safe_float(bal["USDT"]["free"], 0)
                            if free_usdt < 10:
                                continue
                        except:
                            continue

                        try:
                            # 데이터
                            ohlcv5 = bot_ex.fetch_ohlcv(sym, base_tf, limit=250)
                            df5 = pd.DataFrame(ohlcv5, columns=["time","open","high","low","close","vol"])
                            ohlcv1 = bot_ex.fetch_ohlcv(sym, confirm_tf, limit=250)
                            df1 = pd.DataFrame(ohlcv1, columns=["time","open","high","low","close","vol"])

                            pack = build_pack(df5, df1, cfg)
                            if pack is None:
                                continue

                            # 레이더
                            if cfg.get("radar_use", True) and not radar_should_call_ai(sym, pack, mode):
                                continue

                            # AI 판단
                            out = ai_decide(sym, pack, state, mode, cfg)

                            # 글로벌옵션 자동 적용
                            if cfg.get("use_ai_global", True) and isinstance(out.get("global", {}), dict):
                                state["ai_global"] = out["global"]
                                save_runtime_state(state)

                            decision = out.get("decision", "hold")
                            conf = int(out.get("confidence", 0))
                            if decision not in ["buy", "sell"] or conf < conf_cut:
                                continue

                            # -------------------------------------------------
                            # A 핵심 강제 규칙:
                            # RSI 과매도/과매수 '상태' 진입 금지
                            # 반드시 '해소' + confirmation 최소 1개
                            # -------------------------------------------------
                            sig = pack["p5"]["signals"]
                            htf = pack["htf_trend"]
                            # 롱
                            if decision == "buy":
                                # 과매도 상태면 금지
                                if sig["rsi_oversold"]:
                                    continue
                                # 눌림목이면 해소가 필수
                                if htf == "uptrend":
                                    if not sig["rsi_oversold_recovered"]:
                                        continue
                                    if not (sig["confirm_bb_mid_reclaim"] or sig["confirm_macd_hist_turn"] or sig["confirm_vol_recover"]):
                                        continue
                            # 숏
                            if decision == "sell":
                                if sig["rsi_overbought"]:
                                    continue
                                if htf == "downtrend":
                                    if not sig["rsi_overbought_recovered"]:
                                        continue
                                    if not (sig["confirm_bb_mid_reclaim"] or sig["confirm_macd_hist_turn"] or sig["confirm_vol_recover"]):
                                        continue

                            r = out.get("risk", {})
                            lev = int(safe_float(r.get("leverage", 5), 5))
                            risk_pct = safe_float(r.get("risk_pct", 10), 10)

                            # 주문 qty
                            price = safe_float(pack["p5"]["values"]["close"], 0)
                            if price <= 0:
                                continue

                            # 분할 진입 여부
                            scale_in = r.get("scale_in", {"enabled": False})
                            scale_enabled = bool(scale_in.get("enabled", False))

                            # 스윕이면 분할진입을 더 선호(보정)
                            if pack["p5"]["signals"]["sweep"]:
                                scale_enabled = True
                                if "first_pct" not in scale_in:
                                    scale_in["first_pct"] = 60
                                if "second_pct" not in scale_in:
                                    scale_in["second_pct"] = 40

                            first_pct = int(safe_float(scale_in.get("first_pct", 100), 100))
                            second_pct = int(safe_float(scale_in.get("second_pct", 0), 0))

                            # 1차 사용 금액
                            use_usdt_1 = free_usdt * (risk_pct / 100.0) * (first_pct / 100.0)
                            qty1 = (use_usdt_1 * lev) / price
                            qty1 = safe_float(bot_ex.amount_to_precision(sym, qty1), 0)
                            if qty1 <= 0:
                                continue

                            try:
                                bot_ex.set_leverage(lev, sym)
                            except:
                                pass

                            bot_ex.create_market_order(sym, decision, qty1)

                            # 구조 손절 ROI(근사): 스윙 저점(롱)/고점(숏)까지 가격% * lev
                            struct5 = pack["structure_5m"]
                            swing_low = struct5.get("swing_low", None)
                            swing_high = struct5.get("swing_high", None)
                            sl_gap = safe_float(r.get("sl_gap", 1.2), 1.2)
                            tp_target = safe_float(r.get("tp_target", 2.5), 2.5)

                            if decision == "buy" and isinstance(swing_low, (int, float)) and swing_low > 0:
                                move_pct = (price - float(swing_low)) / price * 100
                                sl_gap = max(sl_gap, move_pct * lev * 0.8)  # 약간 여유
                            if decision == "sell" and isinstance(swing_high, (int, float)) and swing_high > 0:
                                move_pct = (float(swing_high) - price) / price * 100
                                sl_gap = max(sl_gap, move_pct * lev * 0.8)

                            # ATR% 저장(트레일링용)
                            atr_pct = safe_float(pack["p5"]["values"].get("ATR_PCT", 1.0), 1.0)

                            side_txt = "long" if decision == "buy" else "short"

                            # trade meta 저장
                            state.setdefault("trades", {})[sym] = {
                                "entry_time": now_str(),
                                "entry_price": price,
                                "qty": qty1,
                                "side": side_txt,
                                "mode": mode,
                                "regime": pack.get("regime",""),
                                "htf_trend": pack.get("htf_trend",""),
                                "setup_tag": out.get("setup_tag","unknown"),
                                "fail_tag_if_loss": out.get("fail_tag_if_loss","unknown"),
                                "risk": {
                                    "leverage": lev,
                                    "risk_pct": risk_pct,
                                    "sl_gap": sl_gap,
                                    "tp_target": tp_target,
                                    "tp1_gap": safe_float(r.get("tp1_gap", 0.6), 0.6),
                                    "tp1_size": int(safe_float(r.get("tp1_size", 30), 30)),
                                    "tp2_gap": safe_float(r.get("tp2_gap", 1.4), 1.4),
                                    "tp2_size": int(safe_float(r.get("tp2_size", 30), 30)),
                                    "use_trailing": bool(r.get("use_trailing", True)),
                                    "trail_start": safe_float(r.get("trail_start", 1.0), 1.0),
                                    "trail_gap_atr_mult": safe_float(r.get("trail_gap_atr_mult", 1.2), 1.2),
                                    "breakeven_after_tp1": bool(r.get("breakeven_after_tp1", True)),
                                    "scale_in": {"enabled": scale_enabled}
                                },
                                "tp1_done": False,
                                "tp2_done": False,
                                "tp_extended": False,
                                "be_enabled": False,
                                "be_roi": 0.2,
                                "best_roi": 0.0,
                                "worst_roi": 0.0,
                                "best_price": price,
                                "atr_pct_at_entry": atr_pct,
                                "reason": out.get("reason", ""),
                                "easy": out.get("easy", ""),
                                "snapshot": {
                                    "pack_regime": pack.get("regime",""),
                                    "htf_trend": pack.get("htf_trend",""),
                                    "rs_score": pack.get("rs_score", 0),
                                    "signals": pack["p5"]["signals"],
                                    "structure_5m": pack["structure_5m"],
                                    "structure_1h": pack["structure_1h"],
                                    "values_5m": pack["p5"]["values"]
                                }
                            }
                            state["last_bot_note"] = f"진입 {sym} {side_txt} ({mode})"
                            save_runtime_state(state)

                            # 2차 분할진입 예약
                            if scale_enabled and second_pct > 0:
                                state.setdefault("pending_scaleins", {})[sym] = {
                                    "decision": decision,
                                    "leverage": lev,
                                    "risk_pct": risk_pct,
                                    "second_pct": second_pct,
                                }
                                save_runtime_state(state)

                            # 텔레그램 보고
                            if cfg.get("telegram_enabled", True):
                                msg = (
                                    f"🎯 진입: {sym} ({mode})\n"
                                    f"- 방향: {side_txt.upper()} (conf {conf}%)\n"
                                    f"- 레짐:{pack['regime']} | HTF:{pack['htf_trend']} | RS:{pack['rs_score']:.1f}\n"
                                    f"- 사용금액: {risk_pct:.1f}% (free 기준)\n"
                                    f"- 레버: x{lev}\n"
                                    f"- 목표수익(TP): +{tp_target:.2f}%\n"
                                    f"- 목표손절(SL): -{sl_gap:.2f}% (구조 반영)\n"
                                    f"- 분할진입: {'ON' if scale_enabled else 'OFF'}\n"
                                    f"- 한줄: {out.get('one_liner','')}\n"
                                    f"- 쉬운설명: {out.get('easy','')}"
                                )
                                tg_send(tg_token, tg_id, msg, reply_markup=TG_MENU)

                            time.sleep(2)

                        except:
                            continue

            # -------------------------------------------------
            # 3) 생존신고
            # -------------------------------------------------
            if cfg.get("telegram_enabled", True) and (time.time() - last_report > int(cfg.get("report_interval_sec", 900))):
                last_report = time.time()
                try:
                    bal = bot_ex.fetch_balance({"type":"swap"})
                    eq = safe_float(bal["USDT"]["total"], 0)
                    tg_send(
                        tg_token, tg_id,
                        f"💤 생존신고 ({mode})\n총자산: ${eq:,.2f}\n연속손실: {state.get('consec_losses',0)}\npause: {'ON' if is_paused(state) else 'OFF'}",
                        reply_markup=TG_MENU
                    )
                except:
                    pass

            time.sleep(0.5)

        except:
            time.sleep(2)

# =========================================================
# Streamlit UI
# =========================================================
st.title("🧩 Bitget AI Bot — Streamlit (제어판 + 모니터링)")

state = load_runtime_state()

# 상단 요약
try:
    bal = exchange.fetch_balance({"type": "swap"})
    usdt_total = safe_float(bal["USDT"]["total"], 0)
    usdt_free = safe_float(bal["USDT"]["free"], 0)
except:
    usdt_total, usdt_free = 0.0, 0.0

active_positions_ui = get_active_positions(exchange, TARGET_COINS)

c1, c2, c3, c4 = st.columns(4)
c1.metric("총자산(USDT)", f"${usdt_total:,.2f}")
c2.metric("주문가능(USDT)", f"${usdt_free:,.2f}")
c3.metric("보유 포지션", f"{len(active_positions_ui)}")
c4.metric("자동매매", "🟢 ON" if config.get("auto_trade") else "🔴 OFF")
st.caption(f"봇 메모: {state.get('last_bot_note','')}")
st.divider()

# =======================
# 사이드바 = 제어판
# =======================
with st.sidebar:
    st.header("🛠️ 제어판")

    config["auto_trade"] = st.checkbox("🤖 자동매매 ON/OFF", value=config.get("auto_trade", False))
    config["telegram_enabled"] = st.checkbox("📩 텔레그램 사용", value=config.get("telegram_enabled", True))

    st.divider()
    config["trade_mode"] = st.radio("거래 모드", ["SAFE", "AGGRESSIVE"], index=0 if config.get("trade_mode","SAFE")=="SAFE" else 1)
    st.caption("SAFE: 원금 방어 우선 / AGGRESSIVE: 공격+선별")

    st.divider()
    config["use_ai_global"] = st.checkbox("AI 글로벌옵션 자동 적용", value=config.get("use_ai_global", True))
    st.caption("ON이면 AI가 cooldown/연속손실 pause/뉴스회피를 자동 조절")

    st.divider()
    config["enable_hard_guardrails"] = st.checkbox("가드레일(추천)", value=config.get("enable_hard_guardrails", True))
    with st.expander("가드레일 세부"):
        config["hard_max_leverage_safe"] = st.slider("SAFE 최대 레버", 1, 50, int(config.get("hard_max_leverage_safe", 10)))
        config["hard_max_leverage_aggressive"] = st.slider("AGGR 최대 레버", 1, 50, int(config.get("hard_max_leverage_aggressive", 20)))
        config["hard_max_risk_pct_safe"] = st.slider("SAFE 최대 진입금액(%)", 1.0, 100.0, float(config.get("hard_max_risk_pct_safe", 15.0)))
        config["hard_max_risk_pct_aggressive"] = st.slider("AGGR 최대 진입금액(%)", 1.0, 100.0, float(config.get("hard_max_risk_pct_aggressive", 30.0)))

    st.divider()
    st.subheader("🎯 스타일")
    config["prefer_short_sl"] = st.checkbox("손절은 짧게(과매도 진입 금지와 함께)", value=config.get("prefer_short_sl", True))
    config["prefer_long_tp_trend"] = st.checkbox("추세면 익절 길게(트레일링/TP연장)", value=config.get("prefer_long_tp_trend", True))
    config["allow_tp_extend"] = st.checkbox("TP 도달 시 1회 연장", value=config.get("allow_tp_extend", True))
    config["tp_extend_mult"] = st.slider("TP 연장 배수", 1.1, 3.0, float(config.get("tp_extend_mult", 1.7)))

    st.divider()
    st.subheader("⏱️ 주기")
    config["manage_interval_sec"] = st.slider("포지션 관리(초)", 1, 10, int(config.get("manage_interval_sec", 2)))
    config["entry_scan_interval_sec"] = st.slider("진입 스캔(초)", 5, 60, int(config.get("entry_scan_interval_sec", 10)))
    config["report_interval_sec"] = st.slider("생존신고(초)", 120, 3600, int(config.get("report_interval_sec", 900)))

    st.divider()
    st.subheader("📈 차트/TF")
    config["ui_symbol"] = st.selectbox("차트 코인", TARGET_COINS, index=TARGET_COINS.index(config.get("ui_symbol", TARGET_COINS[0])))
    config["ui_interval_tf"] = st.selectbox("차트 인터벌", ["1","5","15","60","240","D"], index=["1","5","15","60","240","D"].index(config.get("ui_interval_tf","5")))
    config["confirm_tf"] = st.selectbox("상위TF 컨펌", ["1h","4h"], index=0 if config.get("confirm_tf","1h")=="1h" else 1)

    st.divider()
    st.subheader("🧠 레이더(추천)")
    config["radar_use"] = st.checkbox("AI 호출을 '기회 근접'일 때만", value=config.get("radar_use", True))

    st.divider()
    if st.button("💾 설정 저장"):
        save_settings(config)
        st.success("저장됨(봇이 다음 루프부터 반영)")

    if st.button("📡 텔레그램 메뉴 보내기"):
        tg_send(tg_token, tg_id, "✅ 메뉴 갱신", reply_markup=TG_MENU)

    st.divider()
    st.subheader("🧹 매매일지 초기화")
    if "confirm_reset_log" not in st.session_state:
        st.session_state["confirm_reset_log"] = False

    if st.button("⚠️ 매매일지 초기화(1차)"):
        st.session_state["confirm_reset_log"] = True
        st.warning("한 번 더 누르면 trade_log.csv가 삭제됩니다.")

    if st.session_state["confirm_reset_log"]:
        if st.button("🗑️ 초기화 확정(2차)"):
            try:
                if os.path.exists(TRADE_LOG_FILE):
                    os.remove(TRADE_LOG_FILE)
                st.session_state["confirm_reset_log"] = False
                st.success("trade_log.csv 초기화 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"초기화 실패: {e}")

# 봇 스레드 시작(중복 방지)
if not any(t.name == "TG_Thread" for t in threading.enumerate()):
    t = threading.Thread(target=telegram_bot_thread, daemon=True, name="TG_Thread")
    add_script_run_ctx(t)
    t.start()

# =========================================================
# 메인 영역: TradingView + 지표판 + 탭
# =========================================================
st.subheader("🕯️ TradingView 차트(다크모드)")
tv_sym = TV_SYMBOL_MAP.get(config.get("ui_symbol"), "BINANCE:BTCUSDT")
render_tradingview(tv_sym, interval=config.get("ui_interval_tf", "5"), height=560)

st.divider()

# 현 코인 지표 대시보드
try:
    sym = config.get("ui_symbol", TARGET_COINS[0])
    ohlcv5 = exchange.fetch_ohlcv(sym, config.get("base_tf","5m"), limit=250)
    df5 = pd.DataFrame(ohlcv5, columns=["time","open","high","low","close","vol"])
    ohlcv1 = exchange.fetch_ohlcv(sym, config.get("confirm_tf","1h"), limit=250)
    df1 = pd.DataFrame(ohlcv1, columns=["time","open","high","low","close","vol"])
    pack_ui = build_pack(df5, df1, config)
except:
    pack_ui = None

if pack_ui:
    v = pack_ui["p5"]["values"]
    s = pack_ui["p5"]["status"]
    sig = pack_ui["p5"]["signals"]

    st.markdown("### 🚦 10종+추가 지표 상태판(현재 코인)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재가", f"{v['close']:.4f}")
    c2.metric("RSI", f"{v['RSI']:.1f}", delta=s["RSI_flow"])
    c3.metric("ADX", f"{v['ADX']:.1f}")
    c4.metric("레짐", pack_ui["regime"])
    c5.metric("HTF추세", pack_ui["htf_trend"])

    st.info(
        f"✅ 강제 룰: RSI 과매도/과매수 '상태 진입' 금지 → '해소+확인' 진입\n"
        f"- 과매도 해소: {sig['rsi_oversold_recovered']} | 확인(BB/MACD/Vol): "
        f"{sig['confirm_bb_mid_reclaim']}/{sig['confirm_macd_hist_turn']}/{sig['confirm_vol_recover']} | "
        f"스윕: {sig['sweep']}({sig.get('sweep_type','')})"
    )

st.divider()

# 포지션 표
st.subheader("📌 현재 포지션")
if active_positions_ui:
    rows = []
    for p in active_positions_ui:
        rows.append({
            "Symbol": p.get("symbol",""),
            "Side": normalize_side(p.get("side","")),
            "Leverage": p.get("leverage",""),
            "Contracts": safe_float(p.get("contracts",0)),
            "Entry": safe_float(p.get("entryPrice",0)),
            "Mark": safe_float(p.get("markPrice",0)),
            "ROI%": safe_float(p.get("percentage",0)),
            "UnrealizedPnL": safe_float(p.get("unrealizedPnl",0)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("무포지션")

# =========================================================
# Tabs (t1~t4)
# =========================================================
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🧠 AI 전략 센터")

    colA, colB = st.columns([2, 1])
    with colA:
        auto_on = st.checkbox("🤖 자동매매 활성화(텔레그램 보고)", value=config.get("auto_trade", False))
        if auto_on != config.get("auto_trade", False):
            config["auto_trade"] = auto_on
            save_settings(config)
            st.rerun()
    with colB:
        st.caption("상태: " + ("🟢 가동중" if config.get("auto_trade") else "🔴 정지"))

    st.divider()

    cbtn1, cbtn2 = st.columns(2)

    if cbtn1.button("🔍 현재 코인 AI 분석"):
        if pack_ui is None:
            st.error("데이터 준비 실패. 새로고침 해주세요.")
        else:
            with st.spinner("AI 분석 중..."):
                out = ai_decide(config.get("ui_symbol"), pack_ui, state, config.get("trade_mode","SAFE").upper(), config)
                st.json(out)

    if cbtn2.button("🌍 5개 코인 스캔(로테이션 우선순위)"):
        results = []
        mode = config.get("trade_mode","SAFE").upper()
        with st.spinner("스캔 중(레이더 통과만 AI 호출)..."):
            ranks = []
            for coin in TARGET_COINS:
                try:
                    ohlcv = exchange.fetch_ohlcv(coin, config.get("base_tf","5m"), limit=250)
                    df5 = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                    ranks.append((coin, relative_strength_score(df5)))
                except:
                    ranks.append((coin, -999))
            ranks.sort(key=lambda x: x[1], reverse=True)
            scan_list = [x[0] for x in ranks]

            for coin in scan_list:
                try:
                    ohlcv5 = exchange.fetch_ohlcv(coin, config.get("base_tf","5m"), limit=250)
                    df5 = pd.DataFrame(ohlcv5, columns=["time","open","high","low","close","vol"])
                    ohlcv1 = exchange.fetch_ohlcv(coin, config.get("confirm_tf","1h"), limit=250)
                    df1 = pd.DataFrame(ohlcv1, columns=["time","open","high","low","close","vol"])
                    pack = build_pack(df5, df1, config)
                    if pack is None:
                        continue
                    if config.get("radar_use", True) and not radar_should_call_ai(coin, pack, mode):
                        continue
                    out = ai_decide(coin, pack, state, mode, config)
                    results.append({
                        "코인": coin,
                        "RS": f"{pack['rs_score']:.1f}",
                        "레짐": pack["regime"],
                        "HTF": pack["htf_trend"],
                        "결정": out.get("decision","hold"),
                        "확신": out.get("confidence",0),
                        "한줄": out.get("one_liner",""),
                    })
                except:
                    continue

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.info("레이더 기준으로는 '기회 근접' 코인이 없어요.")

with t2:
    st.subheader("⚡ 수동주문(간단)")
    st.caption("※ 수동주문은 실수 위험이 있으니 테스트용으로만 사용 권장")

    coin = st.selectbox("코인", TARGET_COINS, index=0)
    amount_usdt = st.number_input("사용금액(USDT)", 0.0, 100000.0, 50.0, step=10.0)
    lev = st.slider("레버리지", 1, 50, 5)

    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입"):
        try:
            exchange.set_leverage(lev, coin)
            ticker = exchange.fetch_ticker(coin)
            price = safe_float(ticker.get("last", 0), 0)
            qty = (amount_usdt * lev) / price if price > 0 else 0
            qty = safe_float(exchange.amount_to_precision(coin, qty), 0)
            if qty > 0:
                exchange.create_market_order(coin, "buy", qty)
                st.success(f"롱 진입: {coin} qty={qty}")
        except Exception as e:
            st.error(f"실패: {e}")

    if b2.button("🔴 숏 진입"):
        try:
            exchange.set_leverage(lev, coin)
            ticker = exchange.fetch_ticker(coin)
            price = safe_float(ticker.get("last", 0), 0)
            qty = (amount_usdt * lev) / price if price > 0 else 0
            qty = safe_float(exchange.amount_to_precision(coin, qty), 0)
            if qty > 0:
                exchange.create_market_order(coin, "sell", qty)
                st.success(f"숏 진입: {coin} qty={qty}")
        except Exception as e:
            st.error(f"실패: {e}")

    if b3.button("🚫 해당 코인 포지션 종료"):
        try:
            ps = get_active_positions(exchange, [coin])
            if not ps:
                st.info("포지션 없음")
            else:
                p = ps[0]
                ok = close_position_market(exchange, coin, p.get("side","long"), safe_float(p.get("contracts",0)))
                st.success("종료 완료" if ok else "종료 실패")
        except Exception as e:
            st.error(f"실패: {e}")

with t3:
    st.subheader("📅 경제 캘린더(뉴스 회피)")
    cal = fetch_econ_calendar()
    if cal.empty:
        st.info("캘린더 데이터 없음(잠시 후 다시 시도)")
    else:
        st.dataframe(cal.drop(columns=["utc_dt"], errors="ignore"), use_container_width=True, hide_index=True)

        ag = state.get("ai_global", default_runtime_state()["ai_global"])
        blocked, why = is_news_block(ag, cal)
        if blocked:
            st.warning(f"🚫 현재 뉴스 회피 구간: {why}")

with t4:
    st.subheader("📜 매매일지(한줄 + 상세)")
    df = read_trade_log(int(config.get("log_rows_ui", 200)))
    if df.empty:
        st.info("아직 거래 기록이 없어요.")
    else:
        for _, r in df.iterrows():
            st.write(f"• {r.get('OneLine','')}")
            with st.expander("상세 보기"):
                st.write({
                    "Time": r.get("Time",""),
                    "Mode": r.get("Mode",""),
                    "Symbol": r.get("Symbol",""),
                    "Event": r.get("Event",""),
                    "Side": r.get("Side",""),
                    "Qty": r.get("Qty",""),
                    "Entry": r.get("EntryPrice",""),
                    "Exit": r.get("ExitPrice",""),
                    "PnL%": r.get("PnL_Percent",""),
                    "MFE/MAE": f"{r.get('MFE_ROI','')} / {r.get('MAE_ROI','')}",
                    "Regime/HTF": f"{r.get('Regime','')} / {r.get('HTF_Trend','')}",
                    "SetupTag": r.get("SetupTag",""),
                    "FailTag": r.get("FailTag",""),
                    "Reason": r.get("Reason",""),
                    "Review": r.get("Review",""),
                    "Snapshot": r.get("Snapshot_JSON",""),
                })

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")

st.caption("⚠️ 자동매매는 손실이 발생할 수 있어요. 실전 전에는 샌드박스에서 충분히 테스트하세요.")
