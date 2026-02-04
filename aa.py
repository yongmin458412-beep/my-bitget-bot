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
from urllib.parse import quote

from openai import OpenAI
from streamlit.runtime.scriptrunner import add_script_run_ctx

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트")

IS_SANDBOX = True  # 실전이면 False
SETTINGS_FILE = "bot_settings.json"
RUNTIME_STATE_FILE = "runtime_state.json"
TRADE_LOG_FILE = "trade_log.csv"

TARGET_COINS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"]

# =========================================================
# 설정 관리
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,

        # ✅ 사용자가 관리(고정 적용)
        "max_positions": 2,
        "fixed_leverage": 5,
        "fixed_risk_pct": 10.0,

        # ✅ 손절/익절 안전장치
        "min_sl_gap": 2.5,   # 손절 최소폭(너가 원한 “너무 타이트 손절 방지”)
        "min_rr": 1.8,       # 최소 손익비
        "tp1_gap": 0.5,      # 부분익절 트리거
        "tp1_size": 30,      # 부분익절 비율
        "move_sl_to_be": True,

        # ✅ 프리징 방지(호출 주기)
        "manage_interval_sec": 2,
        "entry_scan_interval_sec": 10,

        # ✅ 손실 제한(너가 관리)
        "cooldown_minutes": 15,
        "max_consec_losses": 3,
        "pause_minutes": 60,

        # UI
        "show_tv_chart": True,
        "show_indicator_table": True,
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default.update(saved)
        except:
            pass
    return default


def save_settings(cfg):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        st.toast("✅ 설정 저장 완료", icon="💾")
    except:
        st.error("설정 저장 실패")


config = load_settings()

# =========================================================
# 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime_state():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {}
    }


def load_runtime_state():
    if not os.path.exists(RUNTIME_STATE_FILE):
        s = default_runtime_state()
        save_runtime_state(s)
        return s
    try:
        with open(RUNTIME_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        s = default_runtime_state()
        save_runtime_state(s)
        return s


def save_runtime_state(state):
    try:
        with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except:
        pass


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
        save_runtime_state(state)

# =========================================================
# trade_log.csv
# =========================================================
def append_trade_log(row: dict):
    cols = ["Time", "Symbol", "Event", "Side", "Qty", "Price", "ROI_Pct", "Note"]
    df = pd.DataFrame([{c: row.get(c, "") for c in cols}])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

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
    st.error("🚨 Bitget API 키가 secrets에 없습니다. secrets.toml 확인!")
    st.stop()

openai_client = None
if openai_key:
    try:
        openai_client = OpenAI(api_key=openai_key)
    except:
        openai_client = None

# =========================================================
# Exchange 생성 (⚠️ 스레드별로 따로 만들기 위해 함수화)
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
    return ex

@st.cache_resource
def init_exchange_ui():
    return create_exchange()

exchange = init_exchange_ui()

# =========================================================
# 보조지표 계산(ta 없이)
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

def calc_indicators(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 250:
        return df, None

    df = df.copy()

    # 10종
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

    # ATR% (손절/익절 자동 보정용)
    df["ATR"] = atr(df["high"], df["low"], df["close"], 14)
    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100

    df = df.dropna()
    if df.empty:
        return df, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    status = {
        "RSI": "🟢 과매도 탈출" if (prev["RSI"] < 30 and last["RSI"] >= 30) else ("🔴 과매수 탈출" if (prev["RSI"] > 70 and last["RSI"] <= 70) else "⚪ 중립"),
        "BB": "🟢 하단 이탈" if last["close"] < last["BB_lower"] else ("🔴 상단 돌파" if last["close"] > last["BB_upper"] else "⚪ 밴드 내"),
        "MA": "📈 골든" if (prev["MA_fast"] <= prev["MA_slow"] and last["MA_fast"] > last["MA_slow"]) else ("📉 데드" if (prev["MA_fast"] >= prev["MA_slow"] and last["MA_fast"] < last["MA_slow"]) else "⚪ 유지"),
        "MACD": "📈 골든" if (prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]) else ("📉 데드" if (prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]) else "⚪ 유지"),
        "ADX": "🔥 추세장" if last["ADX"] >= 25 else "💤 횡보장",
        "STOCH": "🟢 바닥반등" if (prev["STO_K"] <= prev["STO_D"] and last["STO_K"] > last["STO_D"] and last["STO_K"] < 30) else ("🔴 꼭대기꺾임" if (prev["STO_K"] >= prev["STO_D"] and last["STO_K"] < last["STO_D"] and last["STO_K"] > 70) else "⚪ 중립"),
        "CCI": "🟢 과매도" if last["CCI"] < -100 else ("🔴 과매수" if last["CCI"] > 100 else "⚪ 중립"),
        "MFI": "🟢 과매도" if last["MFI"] < 20 else ("🔴 과매수" if last["MFI"] > 80 else "⚪ 중립"),
        "WILLR": "🟢 과매도" if last["WILLR"] < -80 else ("🔴 과매수" if last["WILLR"] > -20 else "⚪ 중립"),
        "VOL": "🔥 급증" if (last["VOL_SMA"] > 0 and last["vol"] >= last["VOL_SMA"] * 2.0) else "⚪ 보통",
        "ATR%": f"{float(last['ATR_PCT']):.2f}%"
    }

    return df, {"status": status, "last": last, "prev": prev}

def score_signals(status: dict):
    long_score = 0
    short_score = 0
    txt = " ".join(status.values())

    if "과매도 탈출" in txt or "바닥반등" in txt:
        long_score += 2
    if "골든" in txt:
        long_score += 1
    if "하단 이탈" in txt or "과매도" in txt:
        long_score += 1

    if "과매수 탈출" in txt or "꼭대기꺾임" in txt:
        short_score += 2
    if "데드" in txt:
        short_score += 1
    if "상단 돌파" in txt or "과매수" in txt:
        short_score += 1

    return long_score, short_score

# =========================================================
# AI 전략(지표 10개 다 주고, AI가 중요한 것만 골라 설명)
# =========================================================
def generate_ai_strategy(symbol: str, df: pd.DataFrame, pack: dict, cfg: dict):
    if openai_client is None:
        return {
            "decision": "hold",
            "confidence": 0,
            "ai_reco": {"leverage": 5, "risk_pct": 10, "sl_gap": cfg["min_sl_gap"], "tp_gap": cfg["min_sl_gap"] * cfg["min_rr"]},
            "focus_indicators": ["RSI", "ADX"],
            "simple": "OpenAI 키가 없어서 관망해요.",
            "detail": "OPENAI_API_KEY를 설정하면 AI 분석이 활성화됩니다."
        }

    last = pack["last"]
    prev = pack["prev"]
    status = pack["status"]
    long_score, short_score = score_signals(status)

    # ATR 기반 손절 “추천 최소치” 만들기 (손절 너무 잦은 문제 개선)
    atr_pct = float(last["ATR_PCT"])
    atr_sl_floor = max(cfg["min_sl_gap"], atr_pct * 1.2)    # ATR%가 크면 손절폭 넓힘
    atr_tp_floor = max(atr_sl_floor * cfg["min_rr"], atr_pct * 2.0)

    snapshot = {
        "price": float(last["close"]),
        "ATR_PCT": atr_pct,
        "RSI_prev": float(prev["RSI"]), "RSI": float(last["RSI"]),
        "BB_upper": float(last["BB_upper"]), "BB_lower": float(last["BB_lower"]), "BB_mid": float(last["BB_mid"]),
        "MA_fast": float(last["MA_fast"]), "MA_slow": float(last["MA_slow"]),
        "MACD": float(last["MACD"]), "MACD_signal": float(last["MACD_signal"]), "MACD_hist": float(last["MACD_hist"]),
        "ADX": float(last["ADX"]), "PDI": float(last["PDI"]), "MDI": float(last["MDI"]),
        "STO_K": float(last["STO_K"]), "STO_D": float(last["STO_D"]),
        "CCI": float(last["CCI"]),
        "MFI": float(last["MFI"]),
        "WILLR": float(last["WILLR"]),
        "VOL": float(last["vol"]), "VOL_SMA": float(last["VOL_SMA"]),
        "status": status,
        "vote": {"long_score": long_score, "short_score": short_score},
        "user_rules": {
            "min_sl_gap": cfg["min_sl_gap"],
            "min_rr": cfg["min_rr"],
            "tp1_gap": cfg["tp1_gap"],
            "tp1_size": cfg["tp1_size"],
            "fixed_leverage": cfg["fixed_leverage"],
            "fixed_risk_pct": cfg["fixed_risk_pct"],
            "atr_sl_floor": atr_sl_floor,
            "atr_tp_floor": atr_tp_floor
        }
    }

    system_prompt = f"""
너는 "자동매매 코치"야.

중요:
- 실제 적용 레버리지/비중은 사용자가 고정값으로 관리한다.
- 너는 추천값만 제시하고, 사용자가 이해하기 쉽게 말한다.
- 확실하지 않으면 hold.

목표:
- 손절이 너무 잦지 않게 (손절폭 최소 {cfg['min_sl_gap']}% 이상, ATR 기반 추천도 참고)
- 손익비 최소 {cfg['min_rr']} 이상일 때만 진입
- TP1(부분익절)로 수익을 자주 잠근다

출력 JSON:
{{
 "decision":"buy/sell/hold",
 "confidence":0~100,
 "ai_reco":{{"leverage":3~10,"risk_pct":5~30,"sl_gap":2.5~10.0,"tp_gap":0~30.0}},
 "focus_indicators":["이번에 중요했던 지표 3~5개"],
 "simple":"초보도 이해 가능한 설명 2~4줄",
 "detail":"조금 더 자세한 설명"
}}
"""

    user_prompt = f"""
심볼: {symbol}
지표 스냅샷(JSON): {json.dumps(snapshot, ensure_ascii=False)}

규칙:
- 손익비가 별로면 hold
- 설명은 꼭 쉽게.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.25
        )
        out = json.loads(resp.choices[0].message.content)

        # 안전 보정: AI가 너무 타이트하게 주면 ATR/최소값으로 보정
        out.setdefault("ai_reco", {})
        sl = float(out["ai_reco"].get("sl_gap", atr_sl_floor))
        tp = float(out["ai_reco"].get("tp_gap", atr_tp_floor))

        sl = max(sl, cfg["min_sl_gap"], atr_sl_floor)
        tp = max(tp, sl * cfg["min_rr"], atr_tp_floor)

        out["ai_reco"]["sl_gap"] = float(sl)
        out["ai_reco"]["tp_gap"] = float(tp)

        lev = int(out["ai_reco"].get("leverage", 5))
        lev = int(min(max(lev, 3), 10))
        out["ai_reco"]["leverage"] = lev

        risk = float(out["ai_reco"].get("risk_pct", 10))
        risk = float(min(max(risk, 5), 30))
        out["ai_reco"]["risk_pct"] = risk

        out.setdefault("focus_indicators", ["RSI", "ADX"])
        out.setdefault("simple", "설명 없음")
        out.setdefault("detail", "")

        return out
    except Exception as e:
        return {
            "decision": "hold",
            "confidence": 0,
            "ai_reco": {"leverage": 5, "risk_pct": 10, "sl_gap": atr_sl_floor, "tp_gap": atr_tp_floor},
            "focus_indicators": ["RSI", "ADX", "ATR%"],
            "simple": "AI 호출 에러로 관망해요.",
            "detail": f"에러: {e}"
        }

# =========================================================
# 유틸
# =========================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def now_ts():
    return int(time.time())

def is_paused(state):
    return time.time() < safe_float(state.get("pause_until", 0))

def in_cooldown(state, symbol):
    until = safe_float(state.get("cooldowns", {}).get(symbol, 0))
    return time.time() < until

def set_cooldown(state, symbol, minutes: int):
    state.setdefault("cooldowns", {})
    state["cooldowns"][symbol] = int(time.time() + minutes * 60)
    save_runtime_state(state)

def get_active_positions(ex, symbols):
    try:
        ps = ex.fetch_positions(symbols=symbols)
        act = []
        for p in ps:
            if safe_float(p.get("contracts", 0)) > 0:
                act.append(p)
        return act
    except:
        return []

# =========================================================
# 순환매: 변동성 큰 2개만 신규 진입 후보
# =========================================================
def pick_rotation_symbols(ex, symbols, timeframe="5m", limit=60, top_n=2):
    scored = []
    for sym in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                continue
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            base = float(df["close"].iloc[-13])
            now = float(df["close"].iloc[-1])
            chg = abs((now - base) / base) * 100 if base > 0 else 0
            vol = float(df["vol"].iloc[-1])
            scored.append((sym, chg, vol))
        except:
            pass

    if not scored:
        return symbols[:top_n]

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in scored[:top_n]]

# =========================================================
# 텔레그램 + 봇 스레드(전용 exchange 사용!)  ✅ 프리징 방지 핵심
# =========================================================
def telegram_thread():
    bot_ex = create_exchange()  # ✅ 스레드 전용 거래소 인스턴스(멈춤 방지 핵심)
    state = load_runtime_state()

    def tg_send(text):
        if not tg_token or not tg_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                data={"chat_id": tg_id, "text": text},
                timeout=5
            )
        except:
            pass

    tg_send("🚀 봇 가동 시작")

    last_manage = 0
    last_entry_scan = 0
    last_report = 0
    REPORT_INTERVAL = 900

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()

            # 데일리 롤링(잔고 기반)
            try:
                bal = bot_ex.fetch_balance({"type": "swap"})
                equity = safe_float(bal["USDT"]["total"])
            except:
                equity = safe_float(state.get("day_start_equity", 0.0))

            maybe_roll_daily_state(state, equity)

            if not cfg.get("auto_trade", False):
                time.sleep(1)
                continue

            if is_paused(state):
                time.sleep(2)
                continue

            ts = time.time()

            # 1) 포지션 관리(너무 자주하지 않게)
            if ts - last_manage >= int(cfg["manage_interval_sec"]):
                last_manage = ts

                active_positions = get_active_positions(bot_ex, TARGET_COINS)

                for p in active_positions:
                    sym = p.get("symbol")
                    side = p.get("side", "long")  # long/short
                    contracts = safe_float(p.get("contracts", 0))
                    entry = safe_float(p.get("entryPrice", 0))
                    mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0))
                    roi = safe_float(p.get("percentage", 0))

                    meta = state.get("trades", {}).get(sym, {})
                    sl = float(meta.get("sl_gap", cfg["min_sl_gap"]))
                    tp = float(meta.get("tp_gap", sl * cfg["min_rr"]))
                    tp1_gap = float(meta.get("tp1_gap", cfg["tp1_gap"]))
                    tp1_size = int(meta.get("tp1_size", cfg["tp1_size"]))
                    tp1_done = bool(meta.get("tp1_done", False))

                    # TP1 부분익절
                    if (not tp1_done) and roi >= tp1_gap and contracts > 0:
                        close_qty = float(bot_ex.amount_to_precision(sym, contracts * (tp1_size / 100.0)))
                        if close_qty > 0:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, close_qty)
                            except:
                                pass

                            state.setdefault("trades", {}).setdefault(sym, {})
                            state["trades"][sym]["tp1_done"] = True
                            if cfg.get("move_sl_to_be", True):
                                state["trades"][sym]["be_price"] = entry
                            save_runtime_state(state)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym,
                                "Event": "TP1(부분익절)",
                                "Side": side,
                                "Qty": close_qty,
                                "Price": mark,
                                "ROI_Pct": f"{roi:.2f}",
                                "Note": "TP1 도달"
                            })
                            tg_send(f"✅ TP1 부분익절: {sym} ({roi:.2f}%)")

                    # 본절 방어(TP1 이후)
                    be_price = meta.get("be_price", None)
                    if be_price and contracts > 0 and roi <= 0.1:
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            bot_ex.create_market_order(sym, close_side, contracts)
                        except:
                            pass

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "BE(본절정리)",
                            "Side": side,
                            "Qty": contracts,
                            "Price": mark,
                            "ROI_Pct": f"{roi:.2f}",
                            "Note": "TP1 후 본절"
                        })
                        tg_send(f"🛡️ 본절 정리: {sym} ({roi:.2f}%)")

                        set_cooldown(state, sym, cfg["cooldown_minutes"])
                        state["trades"].pop(sym, None)
                        save_runtime_state(state)
                        continue

                    # SL/TP 청산
                    if contracts > 0:
                        if roi <= -abs(sl):
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, contracts)
                            except:
                                pass

                            state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                            if state["consec_losses"] >= cfg["max_consec_losses"]:
                                state["pause_until"] = int(time.time() + cfg["pause_minutes"] * 60)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym,
                                "Event": "SL(손절)",
                                "Side": side,
                                "Qty": contracts,
                                "Price": mark,
                                "ROI_Pct": f"{roi:.2f}",
                                "Note": f"손절폭 {sl}%"
                            })
                            tg_send(f"🩸 손절: {sym} ({roi:.2f}%) / 연속손실 {state['consec_losses']}")
                            set_cooldown(state, sym, cfg["cooldown_minutes"])
                            state["trades"].pop(sym, None)
                            save_runtime_state(state)

                        elif roi >= tp:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, contracts)
                            except:
                                pass

                            state["consec_losses"] = 0

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym,
                                "Event": "TP(익절)",
                                "Side": side,
                                "Qty": contracts,
                                "Price": mark,
                                "ROI_Pct": f"{roi:.2f}",
                                "Note": f"익절폭 {tp}%"
                            })
                            tg_send(f"🎉 익절: {sym} (+{roi:.2f}%)")
                            set_cooldown(state, sym, cfg["cooldown_minutes"])
                            state["trades"].pop(sym, None)
                            save_runtime_state(state)

            # 2) 신규 진입(너무 자주 스캔하지 않게)
            if ts - last_entry_scan >= int(cfg["entry_scan_interval_sec"]):
                last_entry_scan = ts

                active_positions = get_active_positions(bot_ex, TARGET_COINS)
                if len(active_positions) < int(cfg["max_positions"]):

                    rotation = pick_rotation_symbols(bot_ex, TARGET_COINS, top_n=min(2, len(TARGET_COINS)))

                    for sym in rotation:
                        if len(get_active_positions(bot_ex, TARGET_COINS)) >= int(cfg["max_positions"]):
                            break
                        if in_cooldown(state, sym):
                            continue
                        if get_active_positions(bot_ex, [sym]):
                            continue

                        try:
                            ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")

                            df, pack = calc_indicators(df)
                            if pack is None:
                                continue

                            # 횡보+중립이면 스킵
                            if pack["status"].get("ADX") == "💤 횡보장" and (35 <= pack["last"]["RSI"] <= 65):
                                continue

                            ai = generate_ai_strategy(sym, df, pack, cfg)
                            decision = ai.get("decision", "hold")
                            conf = int(ai.get("confidence", 0))

                            required_conf = 85 if len(active_positions) >= 1 else 80
                            if decision not in ["buy", "sell"] or conf < required_conf:
                                continue

                            # ✅ 실제 적용은 사용자 고정
                            lev = int(cfg["fixed_leverage"])
                            risk_pct = float(cfg["fixed_risk_pct"])

                            reco = ai.get("ai_reco", {})
                            sl = float(max(float(reco.get("sl_gap", cfg["min_sl_gap"])), cfg["min_sl_gap"]))
                            tp = float(max(float(reco.get("tp_gap", sl * cfg["min_rr"])), sl * cfg["min_rr"]))

                            try:
                                bot_ex.set_leverage(lev, sym)
                            except:
                                pass

                            bal = bot_ex.fetch_balance({"type": "swap"})
                            free_usdt = safe_float(bal["USDT"]["free"])
                            use_usdt = free_usdt * (risk_pct / 100.0)
                            price = float(pack["last"]["close"])
                            qty = (use_usdt * lev) / price if price > 0 else 0
                            qty = float(bot_ex.amount_to_precision(sym, qty))
                            if qty <= 0:
                                continue

                            bot_ex.create_market_order(sym, decision, qty)

                            side_txt = "long" if decision == "buy" else "short"
                            state.setdefault("trades", {})[sym] = {
                                "side": side_txt,
                                "qty": qty,
                                "applied_leverage": lev,
                                "applied_risk_pct": risk_pct,
                                "ai_reco": ai.get("ai_reco", {}),
                                "focus": ai.get("focus_indicators", []),
                                "sl_gap": sl,
                                "tp_gap": tp,
                                "tp1_gap": cfg["tp1_gap"],
                                "tp1_size": cfg["tp1_size"],
                                "tp1_done": False,
                                "be_price": None,
                                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            save_runtime_state(state)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym,
                                "Event": "ENTRY(진입)",
                                "Side": side_txt,
                                "Qty": qty,
                                "Price": price,
                                "ROI_Pct": "",
                                "Note": ai.get("simple", "")[:80]
                            })

                            tg_send(
                                f"🎯 진입: {sym}\n"
                                f"- 방향: {side_txt} (conf {conf}%)\n"
                                f"- 적용(고정): 레버 x{lev}, 비중 {risk_pct}%\n"
                                f"- 목표: TP {tp:.2f}% / SL {sl:.2f}% / TP1 +{cfg['tp1_gap']}%에 {cfg['tp1_size']}%\n"
                                f"- AI중요지표: {', '.join(ai.get('focus_indicators', []))}\n"
                                f"- 쉬운설명: {ai.get('simple','')}"
                            )

                            time.sleep(2)

                        except:
                            pass

            # 3) 생존 신고
            if time.time() - last_report > REPORT_INTERVAL:
                try:
                    bal = bot_ex.fetch_balance({"type": "swap"})
                    eq = safe_float(bal["USDT"]["total"])
                    tg_send(f"💤 생존신고: 총자산 ${eq:,.2f} / 연속손실 {state.get('consec_losses',0)}")
                except:
                    pass
                last_report = time.time()

            time.sleep(0.5)

        except:
            time.sleep(2)

# =========================================================
# 사이드바 UI
# =========================================================
st.sidebar.title("🛠️ 제어판")

if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력(선택)", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

st.sidebar.divider()
config["auto_trade"] = st.sidebar.checkbox("🤖 24시간 자동매매 ON", value=config.get("auto_trade", False))
config["max_positions"] = st.sidebar.slider("동시 포지션 수", 1, 5, int(config.get("max_positions", 2)))

st.sidebar.divider()
st.sidebar.subheader("💰 금전/리스크(내가 관리)")
config["fixed_leverage"] = st.sidebar.slider("고정 레버리지", 1, 20, int(config.get("fixed_leverage", 5)))
config["fixed_risk_pct"] = st.sidebar.slider("고정 비중(% of free USDT)", 1.0, 30.0, float(config.get("fixed_risk_pct", 10.0)))

st.sidebar.divider()
st.sidebar.subheader("🛡️ 수익실현/손실최소(내가 관리)")
config["min_sl_gap"] = st.sidebar.number_input("최소 손절폭(%)", 1.0, 15.0, float(config.get("min_sl_gap", 2.5)), step=0.1)
config["min_rr"] = st.sidebar.number_input("최소 손익비", 1.0, 5.0, float(config.get("min_rr", 1.8)), step=0.1)
config["tp1_gap"] = st.sidebar.number_input("TP1(부분익절) 트리거(%)", 0.1, 5.0, float(config.get("tp1_gap", 0.5)), step=0.1)
config["tp1_size"] = st.sidebar.slider("TP1 청산비율(%)", 10, 80, int(config.get("tp1_size", 30)))
config["move_sl_to_be"] = st.sidebar.checkbox("TP1 후 본절 방어", value=config.get("move_sl_to_be", True))

st.sidebar.divider()
st.sidebar.subheader("⏱️ 제한(연속손실/정지)")
config["cooldown_minutes"] = st.sidebar.slider("코인별 쿨다운(분)", 0, 120, int(config.get("cooldown_minutes", 15)))
config["max_consec_losses"] = st.sidebar.slider("연속손실 제한", 1, 10, int(config.get("max_consec_losses", 3)))
config["pause_minutes"] = st.sidebar.slider("연속손실 시 정지(분)", 5, 240, int(config.get("pause_minutes", 60)))

st.sidebar.divider()
st.sidebar.subheader("🧊 멈춤 방지(호출 주기)")
config["manage_interval_sec"] = st.sidebar.slider("포지션 관리 주기(초)", 1, 10, int(config.get("manage_interval_sec", 2)))
config["entry_scan_interval_sec"] = st.sidebar.slider("신규진입 스캔 주기(초)", 5, 60, int(config.get("entry_scan_interval_sec", 10)))

st.sidebar.divider()
st.sidebar.subheader("🖥️ 화면 옵션")
config["show_tv_chart"] = st.sidebar.checkbox("TradingView 차트 표시", value=config.get("show_tv_chart", True))
config["show_indicator_table"] = st.sidebar.checkbox("지표 상태표 표시", value=config.get("show_indicator_table", True))

save_settings(config)

# =========================================================
# 봇 스레드 시작(중복 실행 방지)
# =========================================================
if not any(t.name == "TG_Thread" for t in threading.enumerate()):
    t = threading.Thread(target=telegram_thread, daemon=True, name="TG_Thread")
    add_script_run_ctx(t)
    t.start()

# =========================================================
# 지갑/포지션 (UI용)
# =========================================================
def fetch_wallet_and_positions():
    bal = exchange.fetch_balance({"type": "swap"})
    usdt_free = safe_float(bal["USDT"]["free"])
    usdt_total = safe_float(bal["USDT"]["total"])
    positions = get_active_positions(exchange, TARGET_COINS)
    return usdt_free, usdt_total, positions

try:
    usdt_free, usdt_total, active_positions = fetch_wallet_and_positions()
except:
    usdt_free, usdt_total, active_positions = 0.0, 0.0, []

with st.sidebar:
    st.divider()
    st.header("내 지갑 현황")
    st.metric("총 자산(USDT)", f"${usdt_total:,.2f}")
    st.metric("주문 가능", f"${usdt_free:,.2f}")

    st.divider()
    st.subheader("보유 포지션")
    if active_positions:
        for p in active_positions:
            sym = p.get("symbol", "")
            side = p.get("side", "long")
            lev = safe_float(p.get("leverage", 0))
            roi = safe_float(p.get("percentage", 0))
            st.info(f"**{sym}** | {'🟢 Long' if side=='long' else '🔴 Short'} x{lev}\nROI: **{roi:.2f}%**")
    else:
        st.caption("현재 무포지션(관망 중)")

# =========================================================
# 메인 화면
# =========================================================
st.title("📌 비트겟 AI 워뇨띠 에이전트")

top1, top2, top3, top4 = st.columns(4)
top1.metric("총자산(USDT)", f"${usdt_total:,.2f}")
top2.metric("주문가능(USDT)", f"${usdt_free:,.2f}")
top3.metric("포지션 수", f"{len(active_positions)} / {config['max_positions']}")
top4.metric("자동매매", "🟢 ON" if config["auto_trade"] else "🔴 OFF")

st.divider()

symbol = st.selectbox("코인 선택", TARGET_COINS, index=0)
timeframe = st.selectbox("타임프레임", ["1m", "5m", "15m", "1h", "4h", "1d"], index=1)

# =========================================================
# ✅ 차트(아까 잘 됐던 iframe 방식)
# =========================================================
def tv_interval(tf: str) -> str:
    m = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    return m.get(tf, "5")

def tv_symbol_from_bitget(sym: str) -> str:
    base = sym.split("/")[0].replace(":USDT", "")
    return f"BINANCE:{base}USDT"

if config.get("show_tv_chart", True):
    tv_symbol = tv_symbol_from_bitget(symbol)
    interval = tv_interval(timeframe)
    tv_url = (
        "https://www.tradingview.com/widgetembed/"
        f"?symbol={quote(tv_symbol)}"
        f"&interval={quote(interval)}"
        "&hidesidetoolbar=0"
        "&symboledit=1"
        "&saveimage=1"
        "&toolbarbg=f1f3f6"
        "&theme=light"
        "&style=1"
        "&timezone=Asia%2FSeoul"
        "&locale=kr"
        "&withdateranges=1"
    )
    components.iframe(tv_url, height=620, scrolling=True)

st.divider()

# =========================================================
# 데이터 로드 + 지표 계산
# =========================================================
df = None
pack = None

try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=250)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df, pack = calc_indicators(df)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")

if pack is None:
    st.warning("⏳ 지표 계산용 데이터가 부족합니다. 코인/타임프레임 바꿔보세요.")
    st.stop()

status = pack["status"]
last = pack["last"]
long_score, short_score = score_signals(status)

judge = "⚪ 관망"
if long_score >= short_score + 2:
    judge = "🟢 매수 우위"
elif short_score >= long_score + 2:
    judge = "🔴 매도 우위"

st.subheader("🚦 보조지표 종합")
c1, c2, c3, c4 = st.columns(4)
c1.metric("현재가", f"{float(last['close']):,.4f}")
c2.metric("롱 점수", f"{long_score}")
c3.metric("숏 점수", f"{short_score}")
c4.metric("종합", judge)

if config.get("show_indicator_table", True):
    with st.expander("지표 상태(간단) 보기"):
        st.json(status)

with st.expander("📌 지표 체크 쉬운 가이드"):
    st.write(
        "✅ 너무 어렵게 보지 말고 이렇게만 보면 돼!\n\n"
        "1) **ADX가 25 이상**이면 → '추세가 있다' (신호 신뢰도 ↑)\n"
        "2) **RSI가 30 아래였다가 다시 올라오면** → '반등 시작' 가능성\n"
        "3) **MA/MACD 골든** → 상승 힘이 붙는 중\n"
        "4) **TP1(0.5%)에서 일부 익절** → 수익을 자주 잠그기\n"
        "5) 손절이 잦으면 → **ATR%가 커진 장**이라 손절폭을 조금 넓혀야 함(이번 코드에 자동 보정 포함)\n"
    )

# =========================================================
# 탭
# =========================================================
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🧠 AI 분석 (AI가 중요한 지표만 골라서 쉽게 설명)")
    colA, colB = st.columns(2)

    with colA:
        if st.button("🔍 현재 코인 AI 분석"):
            with st.spinner("AI 분석 중..."):
                ai = generate_ai_strategy(symbol, df, pack, config)

                decision = ai.get("decision", "hold").upper()
                conf = int(ai.get("confidence", 0))

                if decision == "BUY":
                    st.success(f"결론: 🟢 BUY (확신도 {conf}%)")
                elif decision == "SELL":
                    st.error(f"결론: 🔴 SELL (확신도 {conf}%)")
                else:
                    st.warning(f"결론: ⚪ HOLD (확신도 {conf}%)")

                st.info("✅ 쉬운 설명\n\n" + ai.get("simple", ""))

                st.write("🔎 AI가 이번에 중요하게 본 지표")
                st.write(", ".join(ai.get("focus_indicators", [])))

                with st.expander("조금 더 자세한 설명(지표 근거)"):
                    st.write(ai.get("detail", ""))

                st.divider()
                st.subheader("💡 AI 추천값(표시만) vs 내 적용값(고정)")
                reco = ai.get("ai_reco", {})
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("내 레버(고정)", f"x{config['fixed_leverage']}", delta=f"AI 추천 x{reco.get('leverage', '-')}")
                a2.metric("내 비중(고정)", f"{config['fixed_risk_pct']}%", delta=f"AI 추천 {reco.get('risk_pct','-')}%")
                a3.metric("SL(안전장치)", f"-{config['min_sl_gap']}% 이상", delta=f"AI {reco.get('sl_gap','-')}%")
                a4.metric("RR(최소)", f"{config['min_rr']} 이상", delta=f"AI TP {reco.get('tp_gap','-')}%")

    with colB:
        st.subheader("🤖 자동매매가 실제로 하는 일(정확히)")
        st.write(
            "1) 5개 코인을 보다가\n"
            "2) **변동성 큰 2개만** 골라서(순환매)\n"
            "3) 애매한 횡보는 스킵\n"
            "4) AI가 **10종 지표를 전부 보고**, 중요한 것만 골라 판단\n"
            "5) 확신도(80/85 이상)일 때만 진입\n"
            "6) +0.5% 도달 시 **부분익절** → 이후 **본절 방어**\n"
            "7) TP/SL 도달 시 청산\n"
            "8) 연속손실이면 자동 정지\n"
        )

with t2:
    st.subheader("⚡ 수동주문(원하면 구현 가능)")
    st.caption("지금은 표시용. 원하면 롱/숏/청산 버튼 실제 주문으로 붙여줄게.")
    st.line_chart(df.set_index("time")["close"])

# ---------------------------------------------------------
# 경제 캘린더(한글) : 1) JSON 시도 → 실패 시 TV 위젯으로 대체
# ---------------------------------------------------------
def fetch_econ_calendar_ko():
    # ForexFactory json 미러(가끔 막히면 None 반환)
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=7)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list):
            return None

        rows = []
        now = datetime.utcnow()
        for ev in data:
            # ev 예: {"date":"2026-02-04","time":"13:30","impact":"High","currency":"USD","title":"..."}
            date_s = ev.get("date")
            time_s = ev.get("time") or "00:00"
            if date_s is None:
                continue

            # UTC 기준으로 들어오는 경우가 많아서 '표시만' 간단히
            dt_s = f"{date_s} {time_s}"
            try:
                dt = datetime.strptime(dt_s, "%Y-%m-%d %H:%M")
            except:
                try:
                    dt = datetime.strptime(date_s, "%Y-%m-%d")
                except:
                    continue

            # 이번 주 위주
            if dt < now - timedelta(days=1) or dt > now + timedelta(days=8):
                continue

            impact = (ev.get("impact") or "").lower()
            imp_ko = "높음" if "high" in impact else ("중간" if "medium" in impact else ("낮음" if "low" in impact else ""))
            rows.append({
                "날짜": dt.strftime("%m-%d"),
                "시간(대략)": time_s,
                "통화": ev.get("currency", ""),
                "중요도": imp_ko,
                "지표": ev.get("title", ""),
                "예상": ev.get("forecast", ""),
                "이전": ev.get("previous", "")
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["날짜","시간(대략)","통화","중요도","지표","예상","이전"])
        return df.sort_values(["날짜","시간(대략)"], ascending=True)

    except:
        return None

with t3:
    st.subheader("📅 시장정보(경제 캘린더)")
    cal = fetch_econ_calendar_ko()
    if cal is not None:
        st.caption("✅ 한글 표로 보여줄게 (중요도=높음/중간/낮음)")
        st.dataframe(cal, use_container_width=True, hide_index=True)
    else:
        st.caption("⚠️ 표 캘린더가 안되면 TradingView 위젯으로 보여줄게")
        econ_html = """
<div class="tradingview-widget-container" style="height:600px; width:100%;">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
  {
    "colorTheme": "light",
    "isTransparent": false,
    "width": "100%",
    "height": "600",
    "locale": "ko",
    "importanceFilter": "0,1",
    "currencyFilter": "USD,KRW,EUR,JPY,CNY"
  }
  </script>
</div>
"""
        components.html(econ_html, height=620, scrolling=True)

with t4:
    st.subheader("📜 매매일지")
    state = load_runtime_state()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("date", state.get("date", ""))
    c2.metric("day_start_equity", f"{safe_float(state.get('day_start_equity',0)):.2f}")
    c3.metric("consec_losses", str(state.get("consec_losses", 0)))
    pu = int(state.get("pause_until", 0) or 0)
    pause_txt = "없음" if time.time() >= pu else datetime.fromtimestamp(pu).strftime("%m-%d %H:%M")
    c4.metric("pause_until", pause_txt)

    st.divider()
    with st.expander("runtime_state.json 원본 보기"):
        st.json(state)

    st.divider()
    st.markdown("### trade_log.csv")
    if os.path.exists(TRADE_LOG_FILE):
        log_df = pd.read_csv(TRADE_LOG_FILE)
        if "Time" in log_df.columns:
            log_df = log_df.sort_values("Time", ascending=False)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        csv = log_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")
    else:
        st.caption("아직 trade_log.csv가 없습니다(진입/청산이 발생하면 자동 생성).")
