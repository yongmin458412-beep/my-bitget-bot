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
st.set_page_config(layout="wide", page_title="Bitget AI Bot (Aggressive) - Control Panel")

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

# TradingView 심볼 매핑(원하면 여기 수정)
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


# =========================================================
# 설정 관리
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "telegram_enabled": True,

        # Streamlit 화면 선택
        "ui_symbol": TARGET_COINS[0],
        "ui_interval_tf": "5",

        # 공격모드(기본 ON)
        "aggressive_mode": True,

        # 봇 루프 주기
        "manage_interval_sec": 2,
        "entry_scan_interval_sec": 10,

        # 텔레그램 메뉴 전송 주기(생존신고)
        "report_interval_sec": 900,
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
    except:
        pass


config = load_settings()

# =========================================================
# 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime_state():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,

        # AI가 매번 결정하는 글로벌 옵션(손실제한/쿨다운/동시포지션 등)
        "ai_global": {
            "max_positions": 2,
            "cooldown_minutes": 10,
            "max_consec_losses": 3,
            "pause_minutes": 30,
            "news_avoid": True,
            "news_block_before_min": 15,
            "news_block_after_min": 15,
        },

        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},  # symbol -> until epoch
        "trades": {},     # symbol -> trade meta

        "tg_offset": 0,
        "last_bot_note": "",
        "last_ai_brief": {},
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
# trade_log.csv (AI 회고 포함)
# =========================================================
TRADE_LOG_COLUMNS = [
    "Time", "Symbol", "Event", "Side", "Qty", "EntryPrice", "ExitPrice",
    "PnL_USDT", "PnL_Percent",
    "Leverage", "RiskPct", "TP_Target", "SL_Target",
    "Reason", "Review"
]

def append_trade_log(row: dict):
    df = pd.DataFrame([{c: row.get(c, "") for c in TRADE_LOG_COLUMNS}])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")


def read_trade_log(n=30):
    if not os.path.exists(TRADE_LOG_FILE):
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df.head(n)
    except:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def summarize_recent_mistakes():
    df = read_trade_log(50)
    if df.empty:
        return "기록 없음"
    try:
        df["PnL_Percent"] = pd.to_numeric(df["PnL_Percent"], errors="coerce")
        worst = df.sort_values("PnL_Percent", ascending=True).head(5)
        lines = []
        for _, r in worst.iterrows():
            lines.append(f"- {r['Symbol']} {r['Side']} {r['PnL_Percent']:.2f}% ({str(r.get('Reason',''))[:40]})")
        return "\n".join(lines) if lines else "큰 손실 기록 없음"
    except:
        return "기록 요약 실패"


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
# Exchange 생성(봇 스레드와 UI 분리)
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
# 지표 계산(ta 없이 10종)
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

    # 상태 텍스트(AI 입력용)
    status = {
        "RSI": f"{prev['RSI']:.1f}->{last['RSI']:.1f}",
        "BB": "above" if last["close"] > last["BB_upper"] else ("below" if last["close"] < last["BB_lower"] else "inside"),
        "MA": "golden" if (prev["MA_fast"] <= prev["MA_slow"] and last["MA_fast"] > last["MA_slow"]) else ("dead" if (prev["MA_fast"] >= prev["MA_slow"] and last["MA_fast"] < last["MA_slow"]) else "flat"),
        "MACD": "golden" if (prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]) else ("dead" if (prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]) else "flat"),
        "ADX": float(last["ADX"]),
        "STO": f"{last['STO_K']:.1f}/{last['STO_D']:.1f}",
        "CCI": float(last["CCI"]),
        "MFI": float(last["MFI"]),
        "WILLR": float(last["WILLR"]),
        "VOL_SPIKE": True if (last["VOL_SMA"] > 0 and last["vol"] >= last["VOL_SMA"] * 2.0) else False,
        "ATR_PCT": float(last["ATR_PCT"]),
    }

    return df, {"last": last, "prev": prev, "status": status}


# =========================================================
# 텔레그램
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
        [{"text": "🧾 매매일지(최근)", "callback_data": "log_recent"},
         {"text": "📎 CSV파일", "callback_data": "log_file"}],
        [{"text": "🤖 ON/OFF", "callback_data": "toggle"},
         {"text": "🛑 전량청산", "callback_data": "close_all"}],
    ]
}

# =========================================================
# 경제캘린더(가벼운 회피용, 실패해도 봇은 계속)
# =========================================================
def fetch_econ_calendar():
    # ForexFactory JSON(간단 회피)
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

            rows.append({
                "utc_dt": dt,
                "date": dt.strftime("%m-%d"),
                "time_utc": time_s,
                "currency": ev.get("currency", ""),
                "impact_ko": imp_ko,
                "title": ev.get("title", ""),
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
    # “높음”만 회피(공격모드라도 갑툭튀 급등락 방지에 체감 큼)
    for _, r in cal_df.iterrows():
        if str(r.get("impact_ko","")) != "높음":
            continue
        dt = r.get("utc_dt")
        if not isinstance(dt, datetime):
            continue
        if dt - timedelta(minutes=before) <= now <= dt + timedelta(minutes=after):
            return (True, f"{r.get('currency','')} {r.get('title','')} ({r.get('impact_ko','')})")
    return (False, None)


# =========================================================
# AI: “모든 옵션” 매번 결정 (공격 모드)
# =========================================================
def ai_decide(symbol: str, pack: dict, state: dict, aggressive: bool = True):
    """
    return JSON:
    {
      "decision": "buy/sell/hold",
      "confidence": 0-100,
      "risk": {
        "leverage": ...,
        "risk_pct": ...,          # free USDT 중 몇 % 쓸지
        "sl_gap": ...,            # ROI% 기준 손절
        "tp_target": ...,         # ROI% 기준 최종 목표(익절)
        "tp1_gap": ..., "tp1_size": ...,
        "tp2_gap": ..., "tp2_size": ...,
        "use_trailing": true/false,
        "trail_start": ..., "trail_gap": ...
      },
      "global": {
        "max_positions": ...,
        "cooldown_minutes": ...,
        "max_consec_losses": ...,
        "pause_minutes": ...,
        "news_avoid": true/false,
        "news_block_before_min": ...,
        "news_block_after_min": ...
      },
      "reason": "...",
      "easy": "...(아주 쉽게)",
      "review_template": "...(나중 회고할 때 기준)"
    }
    """
    # OpenAI 없으면: 간단 기본(공격)
    if openai_client is None:
        last = pack["last"]
        atrp = safe_float(last.get("ATR_PCT", 1.0), 1.0)
        return {
            "decision": "hold",
            "confidence": 0,
            "risk": {
                "leverage": 5,
                "risk_pct": 10,
                "sl_gap": max(1.5, atrp * 1.2),
                "tp_target": max(3.0, atrp * 2.5),
                "tp1_gap": 0.5, "tp1_size": 30,
                "tp2_gap": 2.0, "tp2_size": 30,
                "use_trailing": True,
                "trail_start": 1.2, "trail_gap": 0.6,
            },
            "global": {
                "max_positions": 2,
                "cooldown_minutes": 10,
                "max_consec_losses": 3,
                "pause_minutes": 30,
                "news_avoid": True,
                "news_block_before_min": 15,
                "news_block_after_min": 15,
            },
            "reason": "AI키 없음(기본값).",
            "easy": "지금은 AI키가 없어서 관망/기본설정이에요.",
            "review_template": "손절이면: 변동성 대비 SL이 너무 좁았는지, 진입이 급했는지 점검"
        }

    mistakes = summarize_recent_mistakes()
    s = pack["status"]
    last = pack["last"]

    system = f"""
너는 '공격적인 자동매매 매니저'야.
목표: 빠른 수익 기회는 잡되, 손실은 회고로 개선해서 다음에 더 잘하기.
요청: 사용자는 "모든 옵션을 네가 매번 유동적으로 결정"하길 원해. (캡/제한 없음)

중요:
- 과도한 진입은 연속손실을 부른다. 대신 "기회가 좋아 보이면 과감, 애매하면 홀드".
- 너는 risk, TP/SL 구조, 트레일링, 손실 제한(연속손실, pause), 쿨다운, 동시포지션 수까지 모두 결정.
- 출력은 반드시 JSON 하나.

[최근 손실 Top5]
{mistakes}

[응답 JSON 스키마]
{{
 "decision":"buy/sell/hold",
 "confidence":0-100,
 "risk":{{
   "leverage":1-50,
   "risk_pct":1-100,
   "sl_gap":0.3-20.0,
   "tp_target":0.3-50.0,
   "tp1_gap":0.1-10.0, "tp1_size":10-90,
   "tp2_gap":0.1-30.0, "tp2_size":10-90,
   "use_trailing":true/false,
   "trail_start":0.1-30.0, "trail_gap":0.1-30.0
 }},
 "global":{{
   "max_positions":1-5,
   "cooldown_minutes":0-120,
   "max_consec_losses":1-10,
   "pause_minutes":5-240,
   "news_avoid":true/false,
   "news_block_before_min":0-60,
   "news_block_after_min":0-60
 }},
 "reason":"전문가용 근거(지표 기반)",
 "easy":"초등학생도 이해 가능한 쉬운 설명(2~4줄)",
 "review_template":"이 포지션이 끝났을 때 회고할 체크리스트 3개"
}}
"""

    user = {
        "symbol": symbol,
        "price": safe_float(last.get("close", 0)),
        "atr_pct": safe_float(last.get("ATR_PCT", 0)),
        "rsi_flow": s.get("RSI"),
        "bb": s.get("BB"),
        "ma": s.get("MA"),
        "macd": s.get("MACD"),
        "adx": safe_float(s.get("ADX", 0)),
        "vol_spike": bool(s.get("VOL_SPIKE", False)),
        "stoch": s.get("STO"),
        "cci": safe_float(s.get("CCI", 0)),
        "mfi": safe_float(s.get("MFI", 0)),
        "willr": safe_float(s.get("WILLR", 0)),
        "aggressive": aggressive,
        "open_positions": len(state.get("trades", {})),
        "consec_losses": int(state.get("consec_losses", 0)),
    }

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.35
        )
        out = json.loads(resp.choices[0].message.content)

        # ---- 기술적 검증(제한X, 오류방지용만) ----
        out.setdefault("decision", "hold")
        out.setdefault("confidence", 0)
        out.setdefault("risk", {})
        out.setdefault("global", {})
        out.setdefault("reason", "")
        out.setdefault("easy", "")
        out.setdefault("review_template", "")

        r = out["risk"]
        g = out["global"]

        # 수치화(음수/NaN 방지)
        def clamp_min(v, m, default):
            v = safe_float(v, default)
            return max(v, m)

        r["leverage"] = int(clamp_min(r.get("leverage", 5), 1, 5))
        r["risk_pct"] = clamp_min(r.get("risk_pct", 10), 1.0, 10)
        r["sl_gap"] = clamp_min(r.get("sl_gap", 2.0), 0.1, 2.0)
        r["tp_target"] = clamp_min(r.get("tp_target", 3.0), 0.1, 3.0)
        r["tp1_gap"] = clamp_min(r.get("tp1_gap", 0.5), 0.1, 0.5)
        r["tp1_size"] = int(clamp_min(r.get("tp1_size", 30), 1, 30))
        r["tp2_gap"] = clamp_min(r.get("tp2_gap", 2.0), 0.1, 2.0)
        r["tp2_size"] = int(clamp_min(r.get("tp2_size", 30), 1, 30))
        r["use_trailing"] = bool(r.get("use_trailing", True))
        r["trail_start"] = clamp_min(r.get("trail_start", 1.2), 0.1, 1.2)
        r["trail_gap"] = clamp_min(r.get("trail_gap", 0.1), 0.1, 0.6)

        g["max_positions"] = int(clamp_min(g.get("max_positions", 2), 1, 2))
        g["cooldown_minutes"] = int(max(0, safe_float(g.get("cooldown_minutes", 10), 10)))
        g["max_consec_losses"] = int(clamp_min(g.get("max_consec_losses", 3), 1, 3))
        g["pause_minutes"] = int(clamp_min(g.get("pause_minutes", 30), 5, 30))
        g["news_avoid"] = bool(g.get("news_avoid", True))
        g["news_block_before_min"] = int(max(0, safe_float(g.get("news_block_before_min", 15), 15)))
        g["news_block_after_min"] = int(max(0, safe_float(g.get("news_block_after_min", 15), 15)))

        # tp_target은 sl_gap보다 작게 주면 이상하니(제한이라기보다 논리 정합성)
        if r["tp_target"] < r["sl_gap"] * 0.3:
            r["tp_target"] = r["sl_gap"] * 0.6

        return out

    except Exception as e:
        # 실패 시 최소 기본
        last = pack["last"]
        atrp = safe_float(last.get("ATR_PCT", 1.0), 1.0)
        return {
            "decision": "hold",
            "confidence": 0,
            "risk": {
                "leverage": 5,
                "risk_pct": 10,
                "sl_gap": max(1.0, atrp * 1.2),
                "tp_target": max(2.0, atrp * 2.5),
                "tp1_gap": 0.5, "tp1_size": 30,
                "tp2_gap": 2.0, "tp2_size": 30,
                "use_trailing": True, "trail_start": 1.2, "trail_gap": 0.6,
            },
            "global": {
                "max_positions": 2,
                "cooldown_minutes": 10,
                "max_consec_losses": 3,
                "pause_minutes": 30,
                "news_avoid": True,
                "news_block_before_min": 15,
                "news_block_after_min": 15,
            },
            "reason": f"AI 오류로 관망: {e}",
            "easy": "AI 호출이 실패해서 오늘은 관망해요.",
            "review_template": "오류/네트워크 체크"
        }


# =========================================================
# AI 회고(매매일지에 후기 자동 작성)
# =========================================================
def ai_review_trade(trade_row: dict, state: dict):
    if openai_client is None:
        return "AI키 없음: 수동 회고 필요"
    system = """
너는 트레이딩 코치다.
요청: 아래 거래의 결과를 바탕으로 "짧고 이해 쉬운 후기"를 작성해라.
형식:
- 한줄 요약
- 잘한 점 2개
- 아쉬운 점 2개
- 다음엔 이렇게(행동지침 3개)
손절이면: 왜 손절 났는지 가설 2개 + 개선 3개
익절이면: 왜 먹혔는지 2개 + 다음에 유지할 것 3개
"""
    user = json.dumps(trade_row, ensure_ascii=False)
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.35
        )
        return (resp.choices[0].message.content or "").strip()
    except:
        return "AI 회고 실패"


# =========================================================
# 포지션 조회 & 주문 유틸
# =========================================================
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

def close_position_market(ex, symbol, side, contracts):
    # side: long/short
    close_side = "sell" if side == "long" else "buy"
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
# TradingView 차트(다크모드)
# =========================================================
def render_tradingview(symbol_tv: str, interval: str = "5", height: int = 520):
    # TradingView 위젯(다크모드)
    # interval: "1","5","15","60","240","D" 등
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
# 텔레그램 봇 스레드 (거래+리포트+매매일지+회고)
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

    # 시작 메시지
    cfg = load_settings()
    if cfg.get("telegram_enabled", True):
        tg_send(tg_token, tg_id, "🚀 공격모드 봇 시작!\n(자동매매 ON 시 AI가 모든 옵션을 매번 결정)", reply_markup=TG_MENU)

    last_manage = 0
    last_scan = 0
    last_report = 0

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()
            aggressive = bool(cfg.get("aggressive_mode", True))

            # 데일리 리셋
            try:
                bal = bot_ex.fetch_balance({"type": "swap"})
                equity = safe_float(bal["USDT"]["total"])
            except:
                equity = safe_float(state.get("day_start_equity", 0))
            maybe_roll_daily_state(state, equity)

            # 텔레그램 콜백 처리
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
                                    lines = ["📌 포지션 현황"]
                                    for p in ps:
                                        sym = p.get("symbol", "")
                                        side = p.get("side", "")
                                        roi = safe_float(p.get("percentage", 0))
                                        lev = p.get("leverage", "?")
                                        qty = safe_float(p.get("contracts", 0))
                                        lines.append(f"- {sym} | {side} x{lev} | 수량 {qty} | ROI {roi:.2f}%")
                                    tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "toggle":
                                cfg2 = load_settings()
                                cfg2["auto_trade"] = not cfg2.get("auto_trade", False)
                                save_settings(cfg2)
                                tg_send(tg_token, cid, f"🤖 자동매매 {'ON' if cfg2['auto_trade'] else 'OFF'}", reply_markup=TG_MENU)

                            elif data == "log_recent":
                                df = read_trade_log(10)
                                if df.empty:
                                    tg_send(tg_token, cid, "🧾 매매일지 없음", reply_markup=TG_MENU)
                                else:
                                    lines = ["🧾 최근 매매일지(10개)"]
                                    for _, r in df.iterrows():
                                        lines.append(
                                            f"- {r['Time']} | {r['Symbol']} | {r['Event']} | "
                                            f"{r['PnL_Percent']}% | 근거:{str(r.get('Reason',''))[:20]}"
                                        )
                                    tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "log_file":
                                if os.path.exists(TRADE_LOG_FILE):
                                    tg_send_document(tg_token, cid, TRADE_LOG_FILE, caption="📎 trade_log.csv")
                                else:
                                    tg_send(tg_token, cid, "CSV 파일이 아직 없어요(첫 거래 이후 생성).", reply_markup=TG_MENU)

                            elif data == "brief" or data == "scan":
                                syms = TARGET_COINS
                                lines = ["📊 브리핑" if data == "brief" else "🌍 전체스캔(5)"]
                                for sym in syms:
                                    try:
                                        ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                                        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                                        df, pack = calc_indicators(df)
                                        if pack is None:
                                            continue
                                        out = ai_decide(sym, pack, state, aggressive=aggressive)
                                        r = out.get("risk", {})
                                        lines.append(
                                            f"\n[{sym}] {out.get('decision','hold').upper()} (conf {out.get('confidence',0)}%)\n"
                                            f"- 레버 x{r.get('leverage')} | 진입금액 {r.get('risk_pct')}% | SL {r.get('sl_gap')}% | TP {r.get('tp_target')}%\n"
                                            f"- 근거(쉬움): {out.get('easy','')}"
                                        )
                                        # last_ai_brief 저장
                                        state["last_ai_brief"][sym] = out
                                        save_runtime_state(state)
                                    except:
                                        continue
                                tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data == "close_all":
                                ps = get_active_positions(bot_ex, TARGET_COINS)
                                closed = 0
                                for p in ps:
                                    sym = p.get("symbol")
                                    side = p.get("side", "long")
                                    contracts = safe_float(p.get("contracts", 0))
                                    if contracts <= 0:
                                        continue
                                    if close_position_market(bot_ex, sym, side, contracts):
                                        closed += 1
                                tg_send(tg_token, cid, f"🛑 전량청산 요청 완료(대상 {closed}개)", reply_markup=TG_MENU)

                            tg_answer(tg_token, cb_id)

                except:
                    pass

            # 자동매매 OFF면 루프는 계속(리포트/메뉴는 가능)
            if not cfg.get("auto_trade", False):
                time.sleep(0.5)
                continue

            # AI 글로벌 옵션 적용(매번 브리핑/진입 시 업데이트될 수 있음)
            ai_global = state.get("ai_global", default_runtime_state()["ai_global"])

            # pause 로직
            if is_paused(state):
                time.sleep(1.0)
                continue

            ts = time.time()

            # 1) 포지션 관리(부분익절/트레일링/SL/TP)
            if ts - last_manage >= int(cfg.get("manage_interval_sec", 2)):
                last_manage = ts

                positions = get_active_positions(bot_ex, TARGET_COINS)
                for p in positions:
                    sym = p.get("symbol")
                    side = p.get("side", "long")
                    contracts = safe_float(p.get("contracts", 0))
                    if contracts <= 0:
                        continue

                    roi = safe_float(p.get("percentage", 0))
                    mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0))

                    meta = state.get("trades", {}).get(sym, {})
                    if not meta:
                        # 혹시 state가 날아가면 최소 값으로 생성
                        meta = {
                            "entry_price": safe_float(p.get("entryPrice", mark)),
                            "qty": contracts,
                            "risk": {
                                "leverage": safe_float(p.get("leverage", 1)),
                                "risk_pct": "",
                                "sl_gap": 5.0,
                                "tp_target": 8.0,
                                "tp1_gap": 0.5, "tp1_size": 30,
                                "tp2_gap": 2.0, "tp2_size": 30,
                                "use_trailing": True, "trail_start": 1.2, "trail_gap": 0.6,
                            },
                            "tp1_done": False,
                            "tp2_done": False,
                            "best_price": mark,
                            "reason": "",
                            "easy": "",
                        }
                        state.setdefault("trades", {})[sym] = meta
                        save_runtime_state(state)

                    entry_price = safe_float(meta.get("entry_price", safe_float(p.get("entryPrice", mark))))
                    r = meta.get("risk", {})
                    lev = safe_float(r.get("leverage", p.get("leverage", 1)), 1)
                    risk_pct = r.get("risk_pct", "")
                    sl_gap = safe_float(r.get("sl_gap", 5.0), 5.0)
                    tp_target = safe_float(r.get("tp_target", 8.0), 8.0)

                    tp1_gap = safe_float(r.get("tp1_gap", 0.5), 0.5)
                    tp1_size = int(safe_float(r.get("tp1_size", 30), 30))
                    tp2_gap = safe_float(r.get("tp2_gap", 2.0), 2.0)
                    tp2_size = int(safe_float(r.get("tp2_size", 30), 30))

                    use_trailing = bool(r.get("use_trailing", True))
                    trail_start = safe_float(r.get("trail_start", 1.2), 1.2)
                    trail_gap = safe_float(r.get("trail_gap", 0.6), 0.6)

                    tp1_done = bool(meta.get("tp1_done", False))
                    tp2_done = bool(meta.get("tp2_done", False))

                    # best_price 업데이트(트레일링)
                    best_price = safe_float(meta.get("best_price", mark), mark)
                    if side == "long":
                        best_price = max(best_price, mark)
                    else:
                        best_price = min(best_price, mark)
                    meta["best_price"] = best_price
                    save_runtime_state(state)

                    # TP1 부분익절
                    if (not tp1_done) and roi >= tp1_gap:
                        close_qty = safe_float(contracts * (tp1_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp1_done"] = True
                            save_runtime_state(state)
                            append_trade_log({
                                "Time": now_str(), "Symbol": sym, "Event": "TP1(부분익절)", "Side": side,
                                "Qty": close_qty, "EntryPrice": entry_price, "ExitPrice": mark,
                                "PnL_USDT": "", "PnL_Percent": f"{roi:.2f}",
                                "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Reason": str(meta.get("reason",""))[:200], "Review": ""
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP1 부분익절: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                    # TP2 부분익절
                    if (not tp2_done) and roi >= tp2_gap:
                        close_qty = safe_float(contracts * (tp2_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp2_done"] = True
                            save_runtime_state(state)
                            append_trade_log({
                                "Time": now_str(), "Symbol": sym, "Event": "TP2(부분익절)", "Side": side,
                                "Qty": close_qty, "EntryPrice": entry_price, "ExitPrice": mark,
                                "PnL_USDT": "", "PnL_Percent": f"{roi:.2f}",
                                "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Reason": str(meta.get("reason",""))[:200], "Review": ""
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP2 부분익절: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                    # 트레일링
                    if use_trailing and roi >= trail_start:
                        if side == "long":
                            dd = (best_price - mark) / best_price * 100 if best_price > 0 else 0
                        else:
                            dd = (mark - best_price) / best_price * 100 if best_price > 0 else 0
                        if dd >= trail_gap:
                            # 전량 청산
                            ok = close_position_market(bot_ex, sym, side, contracts)
                            if ok:
                                pnl_usdt = safe_float(p.get("unrealizedPnl", 0), 0)
                                row = {
                                    "Time": now_str(), "Symbol": sym, "Event": "TRAIL(청산)", "Side": side,
                                    "Qty": contracts, "EntryPrice": entry_price, "ExitPrice": mark,
                                    "PnL_USDT": f"{pnl_usdt:.4f}", "PnL_Percent": f"{roi:.2f}",
                                    "Leverage": lev, "RiskPct": risk_pct,
                                    "TP_Target": tp_target, "SL_Target": sl_gap,
                                    "Reason": str(meta.get("reason",""))[:200],
                                }
                                review = ai_review_trade(row, state)
                                row["Review"] = review
                                append_trade_log(row)

                                if cfg.get("telegram_enabled", True):
                                    tg_send(
                                        tg_token, tg_id,
                                        f"🏁 트레일링 청산: {sym} ({roi:.2f}%)\n후기:\n{review[:600]}",
                                        reply_markup=TG_MENU
                                    )

                                # 연속손실 reset
                                if roi < 0:
                                    state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                                else:
                                    state["consec_losses"] = 0

                                set_cooldown(state, sym, int(ai_global.get("cooldown_minutes", 10)))
                                state["trades"].pop(sym, None)
                                save_runtime_state(state)
                                continue

                    # SL/TP 최종 청산
                    if roi <= -abs(sl_gap) or roi >= tp_target:
                        event = "SL(손절)" if roi <= -abs(sl_gap) else "TP(익절)"
                        ok = close_position_market(bot_ex, sym, side, contracts)
                        if ok:
                            pnl_usdt = safe_float(p.get("unrealizedPnl", 0), 0)
                            row = {
                                "Time": now_str(), "Symbol": sym, "Event": event, "Side": side,
                                "Qty": contracts, "EntryPrice": entry_price, "ExitPrice": mark,
                                "PnL_USDT": f"{pnl_usdt:.4f}", "PnL_Percent": f"{roi:.2f}",
                                "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Reason": str(meta.get("reason",""))[:200],
                            }
                            review = ai_review_trade(row, state)
                            row["Review"] = review
                            append_trade_log(row)

                            # 연속손실 처리(여기도 AI 글로벌)
                            if roi < 0:
                                state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                                if state["consec_losses"] >= int(ai_global.get("max_consec_losses", 3)):
                                    state["pause_until"] = int(time.time() + int(ai_global.get("pause_minutes", 30)) * 60)
                            else:
                                state["consec_losses"] = 0

                            if cfg.get("telegram_enabled", True):
                                tg_send(
                                    tg_token, tg_id,
                                    f"{'🩸' if roi<0 else '🎉'} {event}: {sym} ({roi:.2f}%)\n"
                                    f"근거: {str(meta.get('easy',''))}\n후기:\n{review[:600]}",
                                    reply_markup=TG_MENU
                                )

                            set_cooldown(state, sym, int(ai_global.get("cooldown_minutes", 10)))
                            state["trades"].pop(sym, None)
                            state["last_bot_note"] = event
                            save_runtime_state(state)

            # 2) 신규 진입 스캔
            if ts - last_scan >= int(cfg.get("entry_scan_interval_sec", 10)):
                last_scan = ts

                # AI global 옵션은 스캔 도중 계속 갱신 가능
                ai_global = state.get("ai_global", default_runtime_state()["ai_global"])
                max_pos = int(ai_global.get("max_positions", 2))

                positions = get_active_positions(bot_ex, TARGET_COINS)
                if len(positions) < max_pos and (not is_paused(state)):

                    # 뉴스 회피 여부(글로벌 옵션)
                    cal = get_calendar_cached()
                    blocked, why = is_news_block(ai_global, cal)
                    if blocked:
                        state["last_bot_note"] = f"뉴스 회피: {why}"
                        save_runtime_state(state)
                    else:
                        for sym in TARGET_COINS:
                            if len(get_active_positions(bot_ex, TARGET_COINS)) >= max_pos:
                                break
                            if in_cooldown(state, sym):
                                continue
                            if sym in state.get("trades", {}):
                                continue
                            # 실제 포지션이 이미 있으면 스킵
                            if get_active_positions(bot_ex, [sym]):
                                continue

                            try:
                                ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                                df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                                df, pack = calc_indicators(df)
                                if pack is None:
                                    continue

                                out = ai_decide(sym, pack, state, aggressive=aggressive)

                                # AI가 글로벌 옵션까지 주면 적용(매번 유동)
                                g = out.get("global", {})
                                if isinstance(g, dict) and g:
                                    state["ai_global"] = g
                                    save_runtime_state(state)

                                decision = out.get("decision", "hold")
                                conf = int(out.get("confidence", 0))

                                # 공격 모드이므로 conf 기준 낮춤(원하면 여기 더 낮춰도 됨)
                                req = 70 if aggressive else 80
                                if decision not in ["buy", "sell"] or conf < req:
                                    continue

                                r = out.get("risk", {})
                                lev = int(safe_float(r.get("leverage", 5), 5))
                                risk_pct = safe_float(r.get("risk_pct", 10), 10)
                                sl_gap = safe_float(r.get("sl_gap", 2.0), 2.0)
                                tp_target = safe_float(r.get("tp_target", 3.0), 3.0)

                                # TP/트레일링 옵션도 AI가 제공
                                tp1_gap = safe_float(r.get("tp1_gap", 0.5), 0.5)
                                tp1_size = int(safe_float(r.get("tp1_size", 30), 30))
                                tp2_gap = safe_float(r.get("tp2_gap", 2.0), 2.0)
                                tp2_size = int(safe_float(r.get("tp2_size", 30), 30))
                                use_trailing = bool(r.get("use_trailing", True))
                                trail_start = safe_float(r.get("trail_start", 1.2), 1.2)
                                trail_gap = safe_float(r.get("trail_gap", 0.6), 0.6)

                                # 레버 설정
                                try:
                                    bot_ex.set_leverage(lev, sym)
                                except:
                                    pass

                                # 주문 수량
                                bal = bot_ex.fetch_balance({"type": "swap"})
                                free_usdt = safe_float(bal["USDT"]["free"], 0)
                                use_usdt = free_usdt * (risk_pct / 100.0)
                                price = safe_float(pack["last"]["close"], 0)
                                if price <= 0:
                                    continue
                                qty = (use_usdt * lev) / price
                                qty = safe_float(bot_ex.amount_to_precision(sym, qty), 0)
                                if qty <= 0:
                                    continue

                                bot_ex.create_market_order(sym, decision, qty)

                                side_txt = "long" if decision == "buy" else "short"

                                state.setdefault("trades", {})[sym] = {
                                    "entry_time": now_str(),
                                    "entry_price": price,
                                    "qty": qty,
                                    "side": side_txt,
                                    "risk": {
                                        "leverage": lev,
                                        "risk_pct": risk_pct,
                                        "sl_gap": sl_gap,
                                        "tp_target": tp_target,
                                        "tp1_gap": tp1_gap, "tp1_size": tp1_size,
                                        "tp2_gap": tp2_gap, "tp2_size": tp2_size,
                                        "use_trailing": use_trailing,
                                        "trail_start": trail_start,
                                        "trail_gap": trail_gap,
                                    },
                                    "tp1_done": False,
                                    "tp2_done": False,
                                    "best_price": price,
                                    "reason": out.get("reason", ""),
                                    "easy": out.get("easy", ""),
                                    "review_template": out.get("review_template", ""),
                                }
                                state["last_bot_note"] = f"진입 {sym} {side_txt}"
                                state["last_ai_brief"][sym] = out
                                save_runtime_state(state)

                                # 텔레그램 보고(요청한 정보: 포지션/진입금액/레버/목표수익/목표손절/근거)
                                if cfg.get("telegram_enabled", True):
                                    tg_send(
                                        tg_token, tg_id,
                                        f"🎯 진입: {sym}\n"
                                        f"- 방향: {side_txt.upper()} (conf {conf}%)\n"
                                        f"- 사용금액: {risk_pct:.1f}% (free USDT 기준)\n"
                                        f"- 레버: x{lev}\n"
                                        f"- 목표수익(TP): +{tp_target:.2f}%\n"
                                        f"- 목표손절(SL): -{sl_gap:.2f}%\n"
                                        f"- TP1: +{tp1_gap:.2f}%에 {tp1_size}%\n"
                                        f"- TP2: +{tp2_gap:.2f}%에 {tp2_size}%\n"
                                        f"- 트레일링: {('ON' if use_trailing else 'OFF')} | +{trail_start:.2f}%부터 되돌림 {trail_gap:.2f}%\n"
                                        f"- 근거(쉬움): {out.get('easy','')}\n"
                                        f"- 근거(상세): {out.get('reason','')[:500]}",
                                        reply_markup=TG_MENU
                                    )

                                time.sleep(2)

                            except:
                                continue

            # 3) 생존신고
            if cfg.get("telegram_enabled", True) and (time.time() - last_report > int(cfg.get("report_interval_sec", 900))):
                last_report = time.time()
                try:
                    bal = bot_ex.fetch_balance({"type":"swap"})
                    eq = safe_float(bal["USDT"]["total"], 0)
                    tg_send(
                        tg_token, tg_id,
                        f"💤 생존신고\n총자산: ${eq:,.2f}\n연속손실: {state.get('consec_losses',0)}\n현재포지션: {len(get_active_positions(bot_ex, TARGET_COINS))}",
                        reply_markup=TG_MENU
                    )
                except:
                    pass

            time.sleep(0.5)

        except:
            time.sleep(2)


# =========================================================
# Streamlit UI (차트+포지션+매매일지)
# =========================================================
st.title("🧩 Bitget AI Bot (공격모드) — Streamlit 제어판 + Telegram 리포트")

state = load_runtime_state()

# 상단 메트릭
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

# 사이드바(사용자는 'ON/OFF' 정도만)
with st.sidebar:
    st.header("🛠️ 제어판")
    st.caption("요청대로: 옵션은 AI가 매번 결정(여기는 보고용/ONOFF 정도만)")

    config["auto_trade"] = st.checkbox("🤖 자동매매 ON", value=config.get("auto_trade", False))
    config["telegram_enabled"] = st.checkbox("📩 텔레그램 사용", value=config.get("telegram_enabled", True))
    config["aggressive_mode"] = st.checkbox("🔥 공격 모드", value=config.get("aggressive_mode", True))

    config["manage_interval_sec"] = st.slider("포지션 관리 주기(초)", 1, 10, int(config.get("manage_interval_sec", 2)))
    config["entry_scan_interval_sec"] = st.slider("진입 스캔 주기(초)", 5, 60, int(config.get("entry_scan_interval_sec", 10)))
    config["report_interval_sec"] = st.slider("생존신고 주기(초)", 120, 3600, int(config.get("report_interval_sec", 900)))

    st.divider()
    st.subheader("📈 차트 설정")
    config["ui_symbol"] = st.selectbox("차트 코인", TARGET_COINS, index=TARGET_COINS.index(config.get("ui_symbol", TARGET_COINS[0])))
    config["ui_interval_tf"] = st.selectbox("차트 인터벌", ["1","5","15","60","240","D"], index=["1","5","15","60","240","D"].index(config.get("ui_interval_tf","5")))

    st.divider()
    if st.button("💾 저장"):
        save_settings(config)
        st.success("저장됨(봇이 다음 루프부터 반영)")

    st.divider()
    st.subheader("🤖 OpenAI 키")
    if not openai_key:
        k = st.text_input("OPENAI_API_KEY", type="password")
        if k:
            config["openai_api_key"] = k
            save_settings(config)
            st.success("저장됨. 새로고침/재실행 시 적용")
    if st.button("📡 텔레그램 메뉴 보내기"):
        tg_send(tg_token, tg_id, "✅ 메뉴 갱신", reply_markup=TG_MENU)

    st.divider()
    st.subheader("🧠 AI가 방금 추천한 글로벌 옵션")
    ai_global = state.get("ai_global", {})
    st.json(ai_global)

# 봇 스레드 시작(중복 방지)
if not any(t.name == "TG_Thread" for t in threading.enumerate()):
    t = threading.Thread(target=telegram_bot_thread, daemon=True, name="TG_Thread")
    add_script_run_ctx(t)
    t.start()

# --- 메인: TradingView 다크 차트 ---
st.subheader("🕯️ TradingView 차트(다크모드)")
tv_sym = TV_SYMBOL_MAP.get(config.get("ui_symbol"), "BINANCE:BTCUSDT")
render_tradingview(tv_sym, interval=config.get("ui_interval_tf", "5"), height=560)

st.divider()

# --- 포지션 표시(제어판에서도 보여달라고 해서 유지) ---
st.subheader("📌 현재 포지션(요약)")
if active_positions_ui:
    rows = []
    for p in active_positions_ui:
        rows.append({
            "Symbol": p.get("symbol",""),
            "Side": p.get("side",""),
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

st.divider()

# --- Streamlit에서도 매매일지 표시 ---
st.subheader("🧾 매매일지(자동 회고 포함)")
log_df = read_trade_log(200)
if log_df.empty:
    st.caption("아직 거래 기록이 없어요. (첫 청산 이후 생성)")
else:
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    csv = log_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")

st.divider()

# --- 최근 AI 브리핑(요청: AI가 옵션을 매번 보고만 해주면) ---
st.subheader("🧠 최근 AI 판단(보고용)")
last_ai = state.get("last_ai_brief", {})
if not last_ai:
    st.caption("아직 AI 브리핑/진입이 없어요. 텔레그램에서 브리핑 버튼 누르거나 자동매매 ON 해보세요.")
else:
    # 최신 5개 정도
    items = list(last_ai.items())[-5:]
    for sym, out in items:
        r = out.get("risk", {})
        st.markdown(f"### {sym}")
        st.write(f"- 결론: {out.get('decision','hold').upper()} (conf {out.get('confidence',0)}%)")
        st.write(f"- 레버: x{r.get('leverage')} / 진입금액: {r.get('risk_pct')}% / SL: {r.get('sl_gap')}% / TP: {r.get('tp_target')}%")
        st.write(f"- 쉬운설명: {out.get('easy','')}")
        with st.expander("상세 근거"):
            st.write(out.get("reason",""))

st.caption("✅ Telegram에서 매매일지(최근/CSV), 포지션, 잔고, 브리핑/스캔, ON/OFF 모두 가능합니다.")
