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
st.set_page_config(layout="wide", page_title="Bitget AI Bot - Control Panel")

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
# 설정 관리 (Streamlit 제어판용)
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "telegram_enabled": True,

        # 모드: SAFE / AGGRESSIVE
        "trade_mode": "SAFE",

        # AI 글로벌옵션 자동 적용 ON/OFF
        "use_ai_global": True,

        # 리포트/루프
        "manage_interval_sec": 2,
        "entry_scan_interval_sec": 10,
        "report_interval_sec": 900,

        # UI
        "ui_symbol": TARGET_COINS[0],
        "ui_interval_tf": "5",

        # 추천 가드레일(원금 손실 최소화 목적)
        # - 사용자가 원하면 꺼도 됨(제어판에서)
        "enable_hard_guardrails": True,
        "hard_max_leverage_safe": 10,
        "hard_max_leverage_aggressive": 20,
        "hard_max_risk_pct_safe": 15.0,         # free USDT의 최대 몇 %까지 진입 자금으로 쓸지
        "hard_max_risk_pct_aggressive": 30.0,

        # 손절 짧게 / 익절 길게 기본 성향 (AI에게도 프롬프트로 전달)
        "prefer_short_sl": True,
        "prefer_long_tp_trend": True,

        # TP 연장 허용 (추세면 TP 도달 후 1회 연장)
        "allow_tp_extend": True,
        "tp_extend_mult": 1.7,  # TP 연장 배수

        # 로그 보기
        "log_rows_ui": 200,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            saved = read_json(SETTINGS_FILE, {})
            default.update(saved)
        except:
            pass
    return default

def save_settings(cfg):
    write_json(SETTINGS_FILE, cfg)

config = load_settings()

# =========================================================
# runtime_state.json (봇 상태 + AI 투명성 데이터 저장)
# =========================================================
def default_runtime_state():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},

        # 현재 진입 메타(봇 내부 관리)
        "trades": {},

        # 텔레그램 offset
        "tg_offset": 0,

        # 마지막 상태 메모
        "last_bot_note": "",

        # AI 투명성: 최근 입력/출력 저장(심볼별)
        "last_ai_inputs": {},
        "last_ai_outputs": {},

        # AI 글로벌옵션(적용값)
        "ai_global": {
            # 여기 값들은 AI가 추천해도 되고, 네가 제어판으로 override 해도 됨
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
        save_runtime_state(s)
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
# trade_log.csv (파일에는 자세히, UI에는 한줄평)
# =========================================================
TRADE_LOG_COLUMNS = [
    "Time", "Mode", "Symbol", "Event", "Side", "Qty",
    "EntryPrice", "ExitPrice", "PnL_USDT", "PnL_Percent",
    "Leverage", "RiskPct", "TP_Target", "SL_Target",
    "Reason", "Review", "OneLine"
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
    # 보기 쉬운 한줄평(네가 보는 용도)
    t = row.get("Time", "")
    sym = row.get("Symbol", "")
    ev = row.get("Event", "")
    pnlp = row.get("PnL_Percent", "")
    mode = row.get("Mode", "")
    easy = row.get("Review", "") or row.get("Reason", "")
    easy = str(easy).replace("\n", " ")
    easy = easy[:40] + ("..." if len(easy) > 40 else "")
    return f"{t} | {mode} | {sym} | {ev} | {pnlp}% | {easy}"

def summarize_recent_mistakes():
    df = read_trade_log(80)
    if df.empty:
        return "기록 없음"
    try:
        df["PnL_Percent"] = pd.to_numeric(df["PnL_Percent"], errors="coerce")
        worst = df.sort_values("PnL_Percent", ascending=True).head(5)
        lines = []
        for _, r in worst.iterrows():
            lines.append(f"- {r['Symbol']} {r.get('Side','')} {r['PnL_Percent']:.2f}% ({str(r.get('Reason',''))[:35]})")
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
# Exchange (UI용 / 봇 스레드용 분리)
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
# 지표 10종(ta 없이 구현) + 투명성 출력용
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

    status = {
        "RSI_flow": f"{prev['RSI']:.1f}->{last['RSI']:.1f}",
        "BB_pos": "above" if last["close"] > last["BB_upper"] else ("below" if last["close"] < last["BB_lower"] else "inside"),
        "MA_cross": "golden" if (prev["MA_fast"] <= prev["MA_slow"] and last["MA_fast"] > last["MA_slow"]) else ("dead" if (prev["MA_fast"] >= prev["MA_slow"] and last["MA_fast"] < last["MA_slow"]) else "flat"),
        "MACD_cross": "golden" if (prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]) else ("dead" if (prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]) else "flat"),
        "ADX": float(last["ADX"]),
        "STO": f"{last['STO_K']:.1f}/{last['STO_D']:.1f}",
        "CCI": float(last["CCI"]),
        "MFI": float(last["MFI"]),
        "WILLR": float(last["WILLR"]),
        "VOL_SPIKE": True if (last["VOL_SMA"] > 0 and last["vol"] >= last["VOL_SMA"] * 2.0) else False,
        "ATR_PCT": float(last["ATR_PCT"]),
    }

    # 투명성용 '지표 값' 묶음
    indicator_values = {
        "close": float(last["close"]),
        "RSI": float(last["RSI"]),
        "BB_upper": float(last["BB_upper"]),
        "BB_lower": float(last["BB_lower"]),
        "MA_fast": float(last["MA_fast"]),
        "MA_slow": float(last["MA_slow"]),
        "MACD": float(last["MACD"]),
        "MACD_signal": float(last["MACD_signal"]),
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

    return df, {"last": last, "prev": prev, "status": status, "values": indicator_values}

# =========================================================
# TradingView(다크모드)
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
# 텔레그램 유틸 + 메뉴
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
        [{"text": "🤖 ON/OFF", "callback_data": "toggle"},
         {"text": "🛑 전량청산", "callback_data": "close_all"}],
    ]
}

# =========================================================
# 경제캘린더(중요뉴스 시간 회피용) - 실패해도 봇은 계속
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
# 포지션/주문 유틸
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
# AI 결정(모드별: 공격+선별 / 안전)
# =========================================================
INDICATOR_LIST = [
    "RSI(흐름)", "볼린저 위치", "MA 크로스", "MACD 크로스", "ADX",
    "Stoch(K/D)", "CCI", "MFI", "Williams %R", "거래량 스파이크", "ATR%(변동성)"
]

def ai_decide(symbol: str, pack: dict, state: dict, mode: str, cfg: dict):
    """
    out JSON:
    {
      decision buy/sell/hold,
      confidence 0-100,
      risk: leverage, risk_pct, sl_gap, tp_target,
            tp1_gap,tp1_size,tp2_gap,tp2_size,
            use_trailing, trail_start, trail_gap,
      global: cooldown_minutes, max_consec_losses, pause_minutes, news_avoid, before/after,
      reason, easy,
      one_liner
    }
    """
    last = pack["values"]
    s = pack["status"]

    # --- 투명성: AI 입력 저장 ---
    ai_input = {
        "symbol": symbol,
        "timeframe": "5m",
        "indicators_used": INDICATOR_LIST,
        "indicator_values": last,
        "indicator_status": s,
        "mode": mode,
        "consec_losses": int(state.get("consec_losses", 0)),
        "open_positions": len(state.get("trades", {})),
        "goal": "원금손실 최소화 + 짧은시간 수익 극대화",
        "style": {
            "prefer_short_sl": bool(cfg.get("prefer_short_sl", True)),
            "prefer_long_tp_trend": bool(cfg.get("prefer_long_tp_trend", True)),
            "allow_tp_extend": bool(cfg.get("allow_tp_extend", True)),
        }
    }
    state.setdefault("last_ai_inputs", {})[symbol] = ai_input
    save_runtime_state(state)

    # OpenAI 없으면 관망 기본
    if openai_client is None:
        out = {
            "decision": "hold",
            "confidence": 0,
            "risk": {
                "leverage": 5,
                "risk_pct": 8,
                "sl_gap": max(0.8, float(last.get("ATR_PCT", 1.0)) * 0.8),
                "tp_target": max(1.6, float(last.get("ATR_PCT", 1.0)) * 2.0),
                "tp1_gap": 0.5, "tp1_size": 30,
                "tp2_gap": 1.2, "tp2_size": 30,
                "use_trailing": True,
                "trail_start": 1.0, "trail_gap": 0.5,
            },
            "global": state.get("ai_global", default_runtime_state()["ai_global"]),
            "reason": "AI키 없음: 관망",
            "easy": "AI키가 없어서 자동 판단을 못해요. 지금은 관망이에요.",
            "one_liner": f"{symbol} HOLD (AI키 없음)"
        }
        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

    mistakes = summarize_recent_mistakes()

    # 모드별 성향(추천)
    if mode == "SAFE":
        mode_rules = """
[안전모드]
- 애매하면 HOLD. 진입은 정말 좋은 자리만.
- 리스크(진입금액, 레버리지)는 작게. 원금 손실 최소화 최우선.
- 손절은 빠르게 인정하되(짧게), 휩쏘(가짜 흔들기)를 고려해서 너무 말도 안 되게 좁게 잡지 마.
- 추세가 확실하면: TP는 길게 + 트레일링 ON(익절은 길게).
- 연속 손실이 나오면 빨리 멈추고(pause), 쿨다운 길게.
"""
        conf_req_hint = "확신도는 쉽게 80 넘기지 말고, 진짜 좋을 때만 85~95를 줘."
    else:
        mode_rules = """
[공격모드(공격+선별)]
- 공격적이되 선별이 핵심: 애매하면 HOLD.
- 손절은 짧게(빠르게), 익절은 추세면 길게(트레일링/TP연장).
- 거래량 스파이크/추세강도(ADX)/크로스(MA/MACD)가 맞을 때만 과감.
- 연속 손실이 나오면 멈춤(pause)을 반드시 활용.
"""
        conf_req_hint = "확신도는 남발하지 말고, 근거가 강할 때만 80~95를 줘."

    system = f"""
너는 자동매매 트레이딩 매니저다.
목표: 원금 손실을 최소화하면서, 짧은 시간 수익 기회를 극대화.
중요: '공격+선별'을 기본 철학으로 두고, 애매하면 HOLD.
{mode_rules}

[최근 손실 Top5]
{mistakes}

{conf_req_hint}

출력은 반드시 JSON 하나.
스키마:
{{
 "decision":"buy/sell/hold",
 "confidence":0-100,
 "risk":{{
   "leverage":1-50,
   "risk_pct":1-100,
   "sl_gap":0.3-20.0,
   "tp_target":0.3-80.0,
   "tp1_gap":0.1-10.0, "tp1_size":10-90,
   "tp2_gap":0.1-30.0, "tp2_size":10-90,
   "use_trailing":true/false,
   "trail_start":0.1-30.0, "trail_gap":0.1-30.0
 }},
 "global":{{
   "cooldown_minutes":0-240,
   "max_consec_losses":1-10,
   "pause_minutes":5-240,
   "news_avoid":true/false,
   "news_block_before_min":0-60,
   "news_block_after_min":0-60
 }},
 "reason":"전문가용 근거(지표 기반, 짧게)",
 "easy":"아주 쉬운 설명(2~4줄)",
 "one_liner":"한줄평(텔레그램/로그용)"
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

        # --- 기술적 검증(오류 방지용) ---
        out.setdefault("decision", "hold")
        out.setdefault("confidence", 0)
        out.setdefault("risk", {})
        out.setdefault("global", {})
        out.setdefault("reason", "")
        out.setdefault("easy", "")
        out.setdefault("one_liner", "")

        r = out["risk"]
        g = out["global"]

        def minv(v, m, d):
            v = safe_float(v, d)
            return max(v, m)

        r["leverage"] = int(minv(r.get("leverage", 5), 1, 5))
        r["risk_pct"] = minv(r.get("risk_pct", 10), 1.0, 10.0)
        r["sl_gap"] = minv(r.get("sl_gap", 1.0), 0.1, 1.0)
        r["tp_target"] = minv(r.get("tp_target", 2.0), 0.1, 2.0)
        r["tp1_gap"] = minv(r.get("tp1_gap", 0.5), 0.1, 0.5)
        r["tp1_size"] = int(minv(r.get("tp1_size", 30), 1, 30))
        r["tp2_gap"] = minv(r.get("tp2_gap", 1.2), 0.1, 1.2)
        r["tp2_size"] = int(minv(r.get("tp2_size", 30), 1, 30))
        r["use_trailing"] = bool(r.get("use_trailing", True))
        r["trail_start"] = minv(r.get("trail_start", 1.0), 0.1, 1.0)
        r["trail_gap"] = minv(r.get("trail_gap", 0.5), 0.1, 0.5)

        g["cooldown_minutes"] = int(max(0, safe_float(g.get("cooldown_minutes", 10), 10)))
        g["max_consec_losses"] = int(minv(g.get("max_consec_losses", 3), 1, 3))
        g["pause_minutes"] = int(minv(g.get("pause_minutes", 30), 5, 30))
        g["news_avoid"] = bool(g.get("news_avoid", True))
        g["news_block_before_min"] = int(max(0, safe_float(g.get("news_block_before_min", 15), 15)))
        g["news_block_after_min"] = int(max(0, safe_float(g.get("news_block_after_min", 15), 15)))

        # --- 추천 가드레일(원금손실 최소화 목적 / 사용자가 OFF 가능) ---
        if cfg.get("enable_hard_guardrails", True):
            if mode == "SAFE":
                r["leverage"] = min(r["leverage"], int(cfg.get("hard_max_leverage_safe", 10)))
                r["risk_pct"] = min(r["risk_pct"], float(cfg.get("hard_max_risk_pct_safe", 15.0)))
            else:
                r["leverage"] = min(r["leverage"], int(cfg.get("hard_max_leverage_aggressive", 20)))
                r["risk_pct"] = min(r["risk_pct"], float(cfg.get("hard_max_risk_pct_aggressive", 30.0)))

        # one_liner 없으면 자동 생성
        if not out.get("one_liner"):
            out["one_liner"] = f"{symbol} {out.get('decision','hold').upper()} conf {out.get('confidence',0)}"

        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

    except Exception as e:
        out = {
            "decision": "hold", "confidence": 0,
            "risk": {"leverage": 5, "risk_pct": 8, "sl_gap": 1.0, "tp_target": 2.0,
                     "tp1_gap": 0.5, "tp1_size": 30, "tp2_gap": 1.2, "tp2_size": 30,
                     "use_trailing": True, "trail_start": 1.0, "trail_gap": 0.5},
            "global": state.get("ai_global", default_runtime_state()["ai_global"]),
            "reason": f"AI 오류로 관망: {e}",
            "easy": "AI 호출이 실패했어요. 지금은 관망이에요.",
            "one_liner": f"{symbol} HOLD (AI err)"
        }
        state.setdefault("last_ai_outputs", {})[symbol] = out
        save_runtime_state(state)
        return out

# =========================================================
# AI 회고(로그에는 자세히, UI는 한줄로)
# =========================================================
def ai_review_trade(trade_row: dict):
    if openai_client is None:
        return "AI키 없음: 수동 회고 필요"
    system = """
너는 트레이딩 코치다.
요청: 아래 거래를 바탕으로 '한줄평'을 먼저 만들고, 그 다음에 짧은 회고를 써라.
형식:
1) 한줄평: (최대 25자 정도)
2) 회고(짧게): 잘한 점 1개 / 아쉬운 점 1개 / 다음 행동 1개
손절이면: 다음에 어떻게 개선할지 1개는 꼭.
익절이면: 다음에 유지할 습관 1개는 꼭.
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
# Telegram bot thread (실시간 조회/보고는 여기)
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
        tg_send(tg_token, tg_id, "🚀 봇 시작!\n(Streamlit=제어판 / Telegram=보고&조회)", reply_markup=TG_MENU)

    last_manage = 0
    last_scan = 0
    last_report = 0

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()
            mode = cfg.get("trade_mode", "SAFE").upper()

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

                            elif data in ["brief", "scan"]:
                                lines = [f"📊 브리핑 ({mode})" if data == "brief" else f"🌍 스캔(5) ({mode})"]
                                for sym in TARGET_COINS:
                                    try:
                                        ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                                        df_ = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                                        df_, pack = calc_indicators(df_)
                                        if pack is None:
                                            continue

                                        out = ai_decide(sym, pack, state, mode, cfg)
                                        # 글로벌옵션 자동 적용
                                        if cfg.get("use_ai_global", True) and isinstance(out.get("global", {}), dict):
                                            state["ai_global"] = out["global"]
                                            save_runtime_state(state)

                                        r = out.get("risk", {})
                                        lines.append(
                                            f"\n[{sym}] {out.get('decision','hold').upper()} (conf {out.get('confidence',0)}%)\n"
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

            # 자동매매 OFF면 보고/조회만
            if not cfg.get("auto_trade", False):
                time.sleep(0.5)
                continue

            # pause 로직
            if is_paused(state):
                time.sleep(1.0)
                continue

            ai_global = state.get("ai_global", default_runtime_state()["ai_global"])

            # 뉴스 회피 체크(글로벌)
            cal = get_calendar_cached()
            blocked, why = is_news_block(ai_global, cal)

            ts = time.time()

            # 1) 포지션 관리(부분익절/트레일링/SL/TP + 추세면 TP 연장 1회)
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
                        # 상태 유실 대비
                        meta = {
                            "entry_price": safe_float(p.get("entryPrice", mark)),
                            "qty": contracts,
                            "risk": {
                                "leverage": safe_float(p.get("leverage", 1)),
                                "risk_pct": "",
                                "sl_gap": 1.0,
                                "tp_target": 2.0,
                                "tp1_gap": 0.5, "tp1_size": 30,
                                "tp2_gap": 1.2, "tp2_size": 30,
                                "use_trailing": True,
                                "trail_start": 1.0, "trail_gap": 0.5,
                            },
                            "tp1_done": False,
                            "tp2_done": False,
                            "tp_extended": False,
                            "best_price": mark,
                            "reason": "",
                            "easy": "",
                            "mode": mode,
                        }
                        state.setdefault("trades", {})[sym] = meta
                        save_runtime_state(state)

                    entry_price = safe_float(meta.get("entry_price", safe_float(p.get("entryPrice", mark))))
                    r = meta.get("risk", {})
                    lev = safe_float(r.get("leverage", p.get("leverage", 1)), 1)
                    risk_pct = r.get("risk_pct", "")
                    sl_gap = safe_float(r.get("sl_gap", 1.0), 1.0)
                    tp_target = safe_float(r.get("tp_target", 2.0), 2.0)

                    tp1_gap = safe_float(r.get("tp1_gap", 0.5), 0.5)
                    tp1_size = int(safe_float(r.get("tp1_size", 30), 30))
                    tp2_gap = safe_float(r.get("tp2_gap", 1.2), 1.2)
                    tp2_size = int(safe_float(r.get("tp2_size", 30), 30))

                    use_trailing = bool(r.get("use_trailing", True))
                    trail_start = safe_float(r.get("trail_start", 1.0), 1.0)
                    trail_gap = safe_float(r.get("trail_gap", 0.5), 0.5)

                    # best_price 갱신
                    best_price = safe_float(meta.get("best_price", mark), mark)
                    if side == "long":
                        best_price = max(best_price, mark)
                    else:
                        best_price = min(best_price, mark)
                    meta["best_price"] = best_price
                    save_runtime_state(state)

                    # TP1
                    if (not meta.get("tp1_done", False)) and roi >= tp1_gap:
                        close_qty = safe_float(contracts * (tp1_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp1_done"] = True
                            save_runtime_state(state)

                    # TP2
                    if (not meta.get("tp2_done", False)) and roi >= tp2_gap:
                        close_qty = safe_float(contracts * (tp2_size / 100.0), 0)
                        close_qty = safe_float(bot_ex.amount_to_precision(sym, close_qty), 0)
                        if close_qty > 0:
                            close_position_market(bot_ex, sym, side, close_qty)
                            meta["tp2_done"] = True
                            save_runtime_state(state)

                    # 트레일링 청산
                    if use_trailing and roi >= trail_start:
                        if side == "long":
                            dd = (best_price - mark) / best_price * 100 if best_price > 0 else 0
                        else:
                            dd = (mark - best_price) / best_price * 100 if best_price > 0 else 0
                        if dd >= trail_gap:
                            ok = close_position_market(bot_ex, sym, side, contracts)
                            if ok:
                                pnl_usdt = safe_float(p.get("unrealizedPnl", 0), 0)
                                trade_row = {
                                    "Time": now_str(), "Mode": meta.get("mode", mode),
                                    "Symbol": sym, "Event": "TRAIL(청산)", "Side": side,
                                    "Qty": contracts, "EntryPrice": entry_price, "ExitPrice": mark,
                                    "PnL_USDT": f"{pnl_usdt:.4f}", "PnL_Percent": f"{roi:.2f}",
                                    "Leverage": lev, "RiskPct": risk_pct,
                                    "TP_Target": tp_target, "SL_Target": sl_gap,
                                    "Reason": str(meta.get("reason",""))[:200],
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
                                state["last_bot_note"] = f"TRAIL 청산 {sym}"
                                save_runtime_state(state)

                                if cfg.get("telegram_enabled", True):
                                    tg_send(tg_token, tg_id, f"🏁 트레일링 청산: {sym} ({roi:.2f}%)\n{trade_row['OneLine']}", reply_markup=TG_MENU)
                            continue

                    # SL 또는 TP 도달
                    if roi <= -abs(sl_gap) or roi >= tp_target:

                        # --- TP 연장(추세면 익절 길게): TP 닿았을 때 1회 연장 ---
                        if roi >= tp_target and cfg.get("allow_tp_extend", True) and cfg.get("prefer_long_tp_trend", True):
                            if not meta.get("tp_extended", False):
                                # 안전/공격 둘 다: 트레일링 ON이면 추세 유지 가정(간단)
                                if bool(r.get("use_trailing", True)):
                                    meta["tp_extended"] = True
                                    meta["risk"]["tp_target"] = float(tp_target) * float(cfg.get("tp_extend_mult", 1.7))
                                    save_runtime_state(state)
                                    if cfg.get("telegram_enabled", True):
                                        tg_send(tg_token, tg_id, f"📈 TP 도달 → 추세로 판단해 TP 1회 연장! {sym}\n새 TP: {meta['risk']['tp_target']:.2f}%", reply_markup=TG_MENU)
                                    continue  # 지금은 청산 안 함

                        event = "SL(손절)" if roi <= -abs(sl_gap) else "TP(익절)"
                        ok = close_position_market(bot_ex, sym, side, contracts)
                        if ok:
                            pnl_usdt = safe_float(p.get("unrealizedPnl", 0), 0)
                            trade_row = {
                                "Time": now_str(), "Mode": meta.get("mode", mode),
                                "Symbol": sym, "Event": event, "Side": side,
                                "Qty": contracts, "EntryPrice": entry_price, "ExitPrice": mark,
                                "PnL_USDT": f"{pnl_usdt:.4f}", "PnL_Percent": f"{roi:.2f}",
                                "Leverage": lev, "RiskPct": risk_pct,
                                "TP_Target": tp_target, "SL_Target": sl_gap,
                                "Reason": str(meta.get("reason",""))[:200],
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
                            state["last_bot_note"] = f"{event} {sym}"
                            save_runtime_state(state)

                            if cfg.get("telegram_enabled", True):
                                emoji = "🩸" if roi < 0 else "🎉"
                                tg_send(tg_token, tg_id, f"{emoji} {event}: {sym} ({roi:.2f}%)\n{trade_row['OneLine']}", reply_markup=TG_MENU)

            # 2) 신규 진입 스캔 (공격+선별 / 안전모드)
            if ts - last_scan >= int(cfg.get("entry_scan_interval_sec", 10)):
                last_scan = ts

                if blocked:
                    state["last_bot_note"] = f"뉴스 회피: {why}"
                    save_runtime_state(state)
                else:
                    # 모드별 진입 기준(추천)
                    conf_cut = 85 if mode == "SAFE" else 80  # 공격+선별이라도 80부터
                    for sym in TARGET_COINS:
                        if is_paused(state) or in_cooldown(state, sym) or (sym in state.get("trades", {})):
                            continue
                        if get_active_positions(bot_ex, [sym]):
                            continue

                        # free USDT가 너무 적으면 신규 진입 스킵(원금 방어)
                        try:
                            bal = bot_ex.fetch_balance({"type": "swap"})
                            free_usdt = safe_float(bal["USDT"]["free"], 0)
                            if free_usdt < 10:
                                continue
                        except:
                            continue

                        try:
                            ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                            df_ = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                            df_, pack = calc_indicators(df_)
                            if pack is None:
                                continue

                            # "공격+선별" 필터 (모드별 강도 다르게)
                            s = pack["status"]
                            adx_val = safe_float(s.get("ADX", 0), 0)
                            trend_ok = (adx_val >= (25 if mode == "SAFE" else 22)) and (s.get("MA_cross") in ["golden", "flat"]) and (s.get("MACD_cross") in ["golden", "flat"])
                            reversal_ok = bool(s.get("VOL_SPIKE", False)) and (s.get("BB_pos") in ["above", "below"])
                            if mode == "SAFE":
                                # 안전모드: 추세 또는 강한 반전 둘 중 하나라도 확실해야
                                if not (trend_ok or reversal_ok):
                                    continue
                            else:
                                # 공격모드도 '선별'은 유지
                                if not (trend_ok or reversal_ok):
                                    continue

                            out = ai_decide(sym, pack, state, mode, cfg)

                            # AI 글로벌옵션 자동 적용(제어판에서 ON/OFF)
                            if cfg.get("use_ai_global", True) and isinstance(out.get("global", {}), dict):
                                state["ai_global"] = out["global"]
                                save_runtime_state(state)

                            decision = out.get("decision", "hold")
                            conf = int(out.get("confidence", 0))
                            if decision not in ["buy", "sell"] or conf < conf_cut:
                                continue

                            r = out.get("risk", {})
                            lev = int(safe_float(r.get("leverage", 5), 5))
                            risk_pct = safe_float(r.get("risk_pct", 10), 10)
                            sl_gap = safe_float(r.get("sl_gap", 1.0), 1.0)
                            tp_target = safe_float(r.get("tp_target", 2.0), 2.0)

                            tp1_gap = safe_float(r.get("tp1_gap", 0.5), 0.5)
                            tp1_size = int(safe_float(r.get("tp1_size", 30), 30))
                            tp2_gap = safe_float(r.get("tp2_gap", 1.2), 1.2)
                            tp2_size = int(safe_float(r.get("tp2_size", 30), 30))
                            use_trailing = bool(r.get("use_trailing", True))
                            trail_start = safe_float(r.get("trail_start", 1.0), 1.0)
                            trail_gap = safe_float(r.get("trail_gap", 0.5), 0.5)

                            # 레버 설정
                            try:
                                bot_ex.set_leverage(lev, sym)
                            except:
                                pass

                            # 주문 수량(리스크는 free 기준 %)
                            price = safe_float(pack["values"]["close"], 0)
                            if price <= 0:
                                continue
                            use_usdt = free_usdt * (risk_pct / 100.0)
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
                                "mode": mode,
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
                                "tp_extended": False,
                                "best_price": price,
                                "reason": out.get("reason", ""),
                                "easy": out.get("easy", ""),
                            }
                            state["last_bot_note"] = f"진입 {sym} {side_txt} ({mode})"
                            save_runtime_state(state)

                            # 텔레그램 보고(실시간 조회/보고는 텔레그램이 메인)
                            if cfg.get("telegram_enabled", True):
                                tg_send(
                                    tg_token, tg_id,
                                    f"🎯 진입: {sym} ({mode})\n"
                                    f"- 방향: {side_txt.upper()} (conf {conf}%)\n"
                                    f"- 사용금액: {risk_pct:.1f}% (free 기준)\n"
                                    f"- 레버: x{lev}\n"
                                    f"- 목표수익(TP): +{tp_target:.2f}%\n"
                                    f"- 목표손절(SL): -{sl_gap:.2f}%\n"
                                    f"- 트레일링: {('ON' if use_trailing else 'OFF')} | +{trail_start:.2f}%부터 되돌림 {trail_gap:.2f}%\n"
                                    f"- 한줄: {out.get('one_liner','')}\n"
                                    f"- 쉬운설명: {out.get('easy','')}",
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
                        f"💤 생존신고 ({mode})\n총자산: ${eq:,.2f}\n연속손실: {state.get('consec_losses',0)}\npause: {'ON' if is_paused(state) else 'OFF'}",
                        reply_markup=TG_MENU
                    )
                except:
                    pass

            time.sleep(0.5)

        except:
            time.sleep(2)

# =========================================================
# Streamlit UI (제어판 + 투명성 + 차트/포지션/로그)
# =========================================================
st.title("🧩 Bitget AI Bot — Streamlit 제어판 (보고/조회는 Telegram)")

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
# 사이드바 = 모든 제어판
# =======================
with st.sidebar:
    st.header("🛠️ 제어판 (모든 설정은 여기서)")
    st.caption("Telegram은 보고/조회만, Streamlit은 설정만!")

    config["auto_trade"] = st.checkbox("🤖 자동매매 ON/OFF", value=config.get("auto_trade", False))
    config["telegram_enabled"] = st.checkbox("📩 텔레그램 사용", value=config.get("telegram_enabled", True))

    st.divider()
    st.subheader("🎚️ 모드 선택(추천)")
    config["trade_mode"] = st.radio("거래 모드", ["SAFE", "AGGRESSIVE"], index=0 if config.get("trade_mode","SAFE")=="SAFE" else 1)
    st.caption("SAFE: 원금 방어 우선 / AGGRESSIVE: 공격+선별(애매하면 HOLD)")

    st.divider()
    st.subheader("🧠 AI 글로벌옵션 적용")
    config["use_ai_global"] = st.checkbox("AI가 추천한 글로벌옵션 자동 적용", value=config.get("use_ai_global", True))
    st.caption("ON이면 AI가 cooldown/연속손실 pause/뉴스회피 등을 자동 조절")

    st.divider()
    st.subheader("🛡️ 원금손실 최소화 가드레일(추천)")
    config["enable_hard_guardrails"] = st.checkbox("가드레일 사용(추천)", value=config.get("enable_hard_guardrails", True))
    st.caption("ON이면 모드별로 레버/진입금액 상한을 강제(원금 방어에 도움)")

    with st.expander("가드레일 세부(원하면 조정)"):
        config["hard_max_leverage_safe"] = st.slider("SAFE 최대 레버", 1, 50, int(config.get("hard_max_leverage_safe", 10)))
        config["hard_max_leverage_aggressive"] = st.slider("AGGR 최대 레버", 1, 50, int(config.get("hard_max_leverage_aggressive", 20)))
        config["hard_max_risk_pct_safe"] = st.slider("SAFE 최대 진입금액(%)", 1.0, 100.0, float(config.get("hard_max_risk_pct_safe", 15.0)))
        config["hard_max_risk_pct_aggressive"] = st.slider("AGGR 최대 진입금액(%)", 1.0, 100.0, float(config.get("hard_max_risk_pct_aggressive", 30.0)))

    st.divider()
    st.subheader("🎯 스타일")
    config["prefer_short_sl"] = st.checkbox("손절은 짧게", value=config.get("prefer_short_sl", True))
    config["prefer_long_tp_trend"] = st.checkbox("추세면 익절 길게(트레일링/TP연장)", value=config.get("prefer_long_tp_trend", True))
    config["allow_tp_extend"] = st.checkbox("TP 도달 시 1회 연장 허용", value=config.get("allow_tp_extend", True))
    config["tp_extend_mult"] = st.slider("TP 연장 배수", 1.1, 3.0, float(config.get("tp_extend_mult", 1.7)))

    st.divider()
    st.subheader("⏱️ 루프 주기")
    config["manage_interval_sec"] = st.slider("포지션 관리 주기(초)", 1, 10, int(config.get("manage_interval_sec", 2)))
    config["entry_scan_interval_sec"] = st.slider("진입 스캔 주기(초)", 5, 60, int(config.get("entry_scan_interval_sec", 10)))
    config["report_interval_sec"] = st.slider("생존신고 주기(초)", 120, 3600, int(config.get("report_interval_sec", 900)))

    st.divider()
    st.subheader("📈 차트 설정")
    config["ui_symbol"] = st.selectbox("차트 코인", TARGET_COINS, index=TARGET_COINS.index(config.get("ui_symbol", TARGET_COINS[0])))
    config["ui_interval_tf"] = st.selectbox("차트 인터벌", ["1","5","15","60","240","D"], index=["1","5","15","60","240","D"].index(config.get("ui_interval_tf","5")))

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
# 메인 화면: 차트 + 포지션 + 로그(한줄) + AI 투명성
# =========================================================
st.subheader("🕯️ TradingView 차트(다크모드)")
tv_sym = TV_SYMBOL_MAP.get(config.get("ui_symbol"), "BINANCE:BTCUSDT")
render_tradingview(tv_sym, interval=config.get("ui_interval_tf", "5"), height=560)

st.divider()

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

st.subheader("🧾 매매일지(너가 보기 편한 한줄 + 상세는 숨김)")
log_df = read_trade_log(int(config.get("log_rows_ui", 200)))
if log_df.empty:
    st.caption("아직 거래 기록이 없어요. (첫 청산 이후 생성)")
else:
    # 한줄 리스트
    for i, r in log_df.iterrows():
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
                "Leverage": r.get("Leverage",""),
                "RiskPct": r.get("RiskPct",""),
                "TP": r.get("TP_Target",""),
                "SL": r.get("SL_Target",""),
                "Reason": r.get("Reason",""),
                "Review": r.get("Review",""),
            })

    csv = log_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")

st.divider()

st.subheader("🔎 AI가 지금 무엇을 보고 판단하는지(투명성)")
st.caption("※ 실제 매매 판단/보고/조회는 텔레그램이 메인입니다. 여긴 확인용!")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### ✅ AI가 사용하는 지표 목록")
    st.write(INDICATOR_LIST)
    st.markdown("### ✅ AI가 보는 차트")
    st.write("- timeframe: 5m (고정)")
    st.write(f"- 감시 코인: {', '.join([c.split('/')[0] for c in TARGET_COINS])}")

with col_b:
    debug_symbol = st.selectbox("투명성 확인할 코인", TARGET_COINS, index=0)
    last_in = state.get("last_ai_inputs", {}).get(debug_symbol, {})
    last_out = state.get("last_ai_outputs", {}).get(debug_symbol, {})

    st.markdown("### 🧾 마지막 AI 입력(요약)")
    if last_in:
        st.json({
            "symbol": last_in.get("symbol"),
            "timeframe": last_in.get("timeframe"),
            "mode": last_in.get("mode"),
            "indicator_status": last_in.get("indicator_status"),
            "consec_losses": last_in.get("consec_losses"),
            "open_positions": last_in.get("open_positions"),
        })
        with st.expander("지표 값 전체 보기"):
            st.json(last_in.get("indicator_values", {}))
    else:
        st.caption("아직 AI가 이 코인을 판단한 기록이 없어요. (텔레그램에서 브리핑/스캔 누르거나 자동매매 ON)")

    st.markdown("### 🤖 마지막 AI 출력(결정)")
    if last_out:
        st.json(last_out)
    else:
        st.caption("아직 출력 기록 없음")

st.caption("⚠️ 자동매매는 손실이 발생할 수 있어요. 실전 전에는 반드시 샌드박스에서 충분히 테스트하세요.")
