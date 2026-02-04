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
from datetime import datetime
from openai import OpenAI
from streamlit.runtime.scriptrunner import add_script_run_ctx

# =========================================================
# ✅ (선택) ta 라이브러리 있으면 사용, 없으면 폴백으로 직접 계산
# =========================================================
try:
    import ta  # pip install ta (있으면 더 편함)
    HAS_TA = True
except Exception:
    HAS_TA = False

# =========================================================
# ⚙️ 기본 설정
# =========================================================
IS_SANDBOX = True  # 실전 매매 시 False로 변경
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"
STATE_FILE = "runtime_state.json"  # 런타임 상태(쿨다운/트레이드관리) 저장

st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (실전 운영판)")

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

# =========================================================
# ✅ 실전 운영 파라미터(우리가 합의한 것)
# =========================================================
MAX_POSITIONS = 2           # 동시 포지션 2개
RISK_PER_TRADE = 0.005      # 트레이드당 계좌 0.5% 리스크
MAX_MARGIN_PCT = 0.20       # 한 포지션 마진 최대 20% 캡(안전)

MIN_LEV, MAX_LEV = 3, 8     # 레버리지 범위(안정 운영)
ATR_MULT = 1.5              # 손절폭 = ATR% * ATR_MULT
MIN_STOP_PCT_PRICE = 0.5    # 가격기준 최소 손절폭(%)
MAX_STOP_PCT_PRICE = 2.5    # 가격기준 최대 손절폭(%)

TP1_FRACTION = 0.5          # 1R에서 절반 익절
TRAIL_R = 0.8               # TP1 이후: 고점대비 0.8R 되밀리면 트레일링 청산

COOLDOWN_AFTER_SL_MIN = 45  # 손절 후 해당 코인 재진입 금지
CONSEC_LOSS_LIMIT = 3       # 3연손절이면 휴식
PAUSE_AFTER_CONSEC_LOSS_MIN = 120
DAILY_MAX_LOSS_PCT = 0.02   # 하루 -2%면 자동매매 중지(보호)

CORR_LIMIT = 0.80           # 상관 0.8 넘으면 2번째 포지션 제외(유사베팅 방지)
SPREAD_LIMIT_PCT = 0.06     # 스프레드 0.06% 넘으면 진입 패스(체결 손해 방지)

SCAN_INTERVAL_SEC = 30      # 신규 진입 스캔 주기(초)

# =========================================================
# 🔐 Secrets 로드
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("🚨 Bitget API Key가 secrets에 없습니다 (API_KEY).")
    st.stop()

if not openai_key:
    st.error("🚨 OpenAI API Key가 secrets에 없습니다 (OPENAI_API_KEY).")
    st.stop()

openai_client = OpenAI(api_key=openai_key)

# =========================================================
# 💾 설정 로드/저장
# =========================================================
def load_settings():
    default = {
        "auto_trade": False,
        "order_usdt": 100.0,
        "leverage_ui": 5,

        # 지표 파라미터
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 지표 사용 여부(10종)
        "use_rsi": True, "use_bb": True, "use_ma": True, "use_macd": True,
        "use_stoch": True, "use_cci": True, "use_mfi": True, "use_willr": True,
        "use_adx": True, "use_vol": True,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default.update(saved)
        except:
            pass
    return default

def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        st.toast("✅ 설정 저장 완료", icon="💾")
    except:
        st.error("설정 저장 실패")

config = load_settings()

# =========================================================
# 📡 거래소 연결
# =========================================================
@st.cache_resource
def init_exchange():
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

exchange = init_exchange()

# =========================================================
# 📈 TradingView 위젯
# =========================================================
def to_tv_symbol(ccxt_symbol: str) -> str:
    # 예: BTC/USDT:USDT -> BITGET:BTCUSDT.P
    base = ccxt_symbol.split("/")[0].replace(":", "")
    quote = "USDT"
    return f"BITGET:{base}{quote}.P"

def tf_to_tv_interval(tf: str) -> str:
    m = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","2h":"120","4h":"240","1d":"D"}
    return m.get(tf, "5")

def render_tradingview(ccxt_symbol: str, timeframe: str, height: int = 520, theme: str = "dark"):
    tv_symbol = to_tv_symbol(ccxt_symbol)
    interval = tf_to_tv_interval(timeframe)
    container_id = f"tv_{uuid.uuid4().hex}"

    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%;">
      <div id="{container_id}" style="height:{height}px;width:100%;"></div>
    </div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{interval}",
        "timezone": "Asia/Seoul",
        "theme": "{theme}",
        "style": "1",
        "locale": "kr",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "allow_symbol_change": true,
        "save_image": false,
        "container_id": "{container_id}"
      }});
    </script>
    """
    components.html(html, height=height+20, scrolling=False)

# =========================================================
# 🧮 폴백 지표 계산(ta 없을 때)
# =========================================================
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def _bbands(close, period=20, dev=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + dev * std
    lower = mid - dev * std
    return upper, mid, lower

def _macd(close, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _stoch(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_line = 100 * (close - ll) / ((hh - ll) + 1e-12)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def _cci(high, low, close, n=20):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / ((0.015 * mad) + 1e-12)

def _mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = tp.diff()
    pos = mf.where(direction > 0, 0.0)
    neg = mf.where(direction < 0, 0.0).abs()
    pos_sum = pos.rolling(n).sum()
    neg_sum = neg.rolling(n).sum()
    mfr = pos_sum / (neg_sum + 1e-12)
    return 100 - (100 / (1 + mfr))

def _willr(high, low, close, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / ((hh - ll) + 1e-12)

def _adx(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean()
    plus_smooth = pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean()
    minus_smooth = pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100 * (plus_smooth / (tr_smooth + 1e-12))
    minus_di = 100 * (minus_smooth / (tr_smooth + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx

def calc_atr(df, n=14):
    high = df["high"]; low = df["low"]; close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =========================================================
# 🧮 10종 지표 계산(ta 있으면 ta / 없으면 폴백)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: dict):
    try:
        if df is None or df.empty or len(df) < 120:
            return df, {}, None

        # 컬럼 통일
        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"]

        if HAS_TA:
            df["RSI"] = ta.momentum.rsi(df["close"], window=int(cfg["rsi_period"]))

            bb = ta.volatility.BollingerBands(df["close"], window=int(cfg["bb_period"]), window_dev=float(cfg["bb_std"]))
            df["BB_upper"] = bb.bollinger_hband()
            df["BB_mid"] = bb.bollinger_mavg()
            df["BB_lower"] = bb.bollinger_lband()

            df["MA_fast"] = ta.trend.sma_indicator(df["close"], window=int(cfg["ma_fast"]))
            df["MA_slow"] = ta.trend.sma_indicator(df["close"], window=int(cfg["ma_slow"]))

            macd = ta.trend.MACD(df["close"])
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"] = macd.macd_diff()

            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=int(cfg["stoch_k"]), smooth_window=3)
            df["STO_K"] = stoch.stoch()
            df["STO_D"] = stoch.stoch_signal()

            df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
            df["MFI"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14)
            df["WILLR"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)
            df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

        else:
            df["RSI"] = _rsi(df["close"], int(cfg["rsi_period"]))
            df["BB_upper"], df["BB_mid"], df["BB_lower"] = _bbands(df["close"], int(cfg["bb_period"]), float(cfg["bb_std"]))
            df["MA_fast"] = df["close"].rolling(int(cfg["ma_fast"])).mean()
            df["MA_slow"] = df["close"].rolling(int(cfg["ma_slow"])).mean()
            df["MACD"], df["MACD_signal"], df["MACD_hist"] = _macd(df["close"])
            df["STO_K"], df["STO_D"] = _stoch(df["high"], df["low"], df["close"], int(cfg["stoch_k"]))
            df["CCI"] = _cci(df["high"], df["low"], df["close"], 20)
            df["MFI"] = _mfi(df["high"], df["low"], df["close"], df["volume"], 14)
            df["WILLR"] = _willr(df["high"], df["low"], df["close"], 14)
            df["ADX"] = _adx(df["high"], df["low"], df["close"], 14)

        # 거래량 스파이크
        df["VOL_MA"] = df["volume"].rolling(20).mean()
        df["VOL_SPIKE"] = df["volume"] / (df["VOL_MA"] + 1e-12)

        df = df.dropna()
        if df.empty:
            return df, {}, None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        status = {}

        # 상태(10종)
        status["RSI"] = "🔴 과매수" if last["RSI"] >= cfg["rsi_sell"] else ("🟢 과매도" if last["RSI"] <= cfg["rsi_buy"] else "⚪ 중립")
        status["RSI_FLOW"] = "↗️ 반등" if last["RSI"] > prev["RSI"] else "↘️ 약화"

        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

        status["MA"] = "📈 (단기>장기)" if last["MA_fast"] > last["MA_slow"] else "📉 (단기<장기)"
        status["MACD"] = "📈 골든" if last["MACD"] > last["MACD_signal"] else "📉 데드"
        status["STOCH"] = "🔴 과열" if last["STO_K"] > 80 else ("🟢 침체" if last["STO_K"] < 20 else "⚪ 중립")
        status["CCI"] = "🔴 과열" if last["CCI"] > 100 else ("🟢 침체" if last["CCI"] < -100 else "⚪ 중립")
        status["MFI"] = "🔴 과열" if last["MFI"] > 80 else ("🟢 침체" if last["MFI"] < 20 else "⚪ 중립")
        status["WILLR"] = "🔴 과열" if last["WILLR"] > -20 else ("🟢 침체" if last["WILLR"] < -80 else "⚪ 중립")
        status["ADX"] = "🔥 추세장" if last["ADX"] >= 25 else "💤 횡보장"
        status["VOL"] = "🔥 거래량 폭증" if last["VOL_SPIKE"] >= float(cfg["vol_mul"]) else "⚪ 보통"

        return df, status, last

    except Exception as e:
        print("Calc Error:", e)
        return df, {}, None

# =========================================================
# 💾 런타임 상태(쿨다운/연손절/일손실/트레이드) 저장
# =========================================================
def _now_ts():
    return int(time.time())

def _today_str():
    return datetime.now().strftime("%Y-%m-%d")

def load_runtime_state():
    base = {
        "date": _today_str(),
        "day_start_equity": None,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},  # coin -> ts
        "trades": {}      # coin -> trade_info
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            base.update(saved)
        except:
            pass
    return base

def save_runtime_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except:
        pass

def _rollover_daily(state, equity_now: float):
    today = _today_str()
    if state.get("date") != today:
        state["date"] = today
        state["day_start_equity"] = equity_now
        state["daily_realized_pnl"] = 0.0
        state["consec_losses"] = 0
        state["pause_until"] = 0
        state["cooldowns"] = {}
        save_runtime_state(state)

def _is_paused(state):
    return _now_ts() < int(state.get("pause_until", 0))

def _daily_stop_hit(state, equity_now: float):
    start = state.get("day_start_equity")
    if start is None:
        state["day_start_equity"] = equity_now
        save_runtime_state(state)
        return False
    return equity_now <= float(start) * (1.0 - DAILY_MAX_LOSS_PCT)

def _set_coin_cooldown(state, coin):
    state.setdefault("cooldowns", {})[coin] = _now_ts() + COOLDOWN_AFTER_SL_MIN * 60

def _coin_in_cooldown(state, coin):
    return _now_ts() < int(state.get("cooldowns", {}).get(coin, 0))

def _hit_consec_loss_pause(state):
    state["pause_until"] = _now_ts() + PAUSE_AFTER_CONSEC_LOSS_MIN * 60

# =========================================================
# 🧠 순환매/레짐/상관/스프레드/리스크 수량
# =========================================================
def returns_series(close: pd.Series):
    return close.pct_change().fillna(0.0)

def corr_of_returns(df_a, df_b, n=60):
    ra = returns_series(df_a["close"]).tail(n)
    rb = returns_series(df_b["close"]).tail(n)
    if len(ra) < 10 or len(rb) < 10:
        return 0.0
    return float(ra.corr(rb))

def get_spread_pct(ex, symbol):
    try:
        t = ex.fetch_ticker(symbol)
        bid = t.get("bid")
        ask = t.get("ask")
        if not bid or not ask or bid <= 0:
            return 0.0
        return (ask - bid) / bid * 100.0
    except:
        return 0.0

def rotation_score(coin_df, btc_df):
    c = coin_df["close"]
    b = btc_df["close"]
    ret_5m  = (c.iloc[-1] / c.iloc[-2]  - 1.0) if len(c) >= 2 else 0.0
    ret_15m = (c.iloc[-1] / c.iloc[-4]  - 1.0) if len(c) >= 4 else 0.0
    ret_1h  = (c.iloc[-1] / c.iloc[-13] - 1.0) if len(c) >= 13 else 0.0

    btc_1h  = (b.iloc[-1] / b.iloc[-13] - 1.0) if len(b) >= 13 else 0.0
    rs = ret_1h - btc_1h  # BTC 대비 상대강도

    vol = coin_df["volume"]
    vol_ma = vol.rolling(20).mean()
    vspike = float((vol.iloc[-1] / (vol_ma.iloc[-1] + 1e-12))) if len(vol_ma.dropna()) > 0 else 1.0
    vspike = min(max(vspike, 0.0), 5.0)

    score = (ret_1h*100*0.6 + ret_15m*100*0.3 + ret_5m*100*0.1) + (rs*100*0.8) + ((vspike-1.0)*2.0)
    return float(score), float(vspike), float(rs*100)

def btc_regime(btc_df):
    close = btc_df["close"]
    if len(close) < 30:
        return "neutral"

    ret_1h = (close.iloc[-1] / close.iloc[-13] - 1.0) * 100
    atr = calc_atr(btc_df, 14)
    atr_pct = float(atr.iloc[-1] / close.iloc[-1] * 100) if not np.isnan(atr.iloc[-1]) else 0.0

    if ret_1h <= -1.0 or atr_pct >= 1.2:
        return "risk_off"
    if ret_1h >= 1.0 and atr_pct < 1.2:
        return "risk_on"
    return "neutral"

def calc_qty_by_risk(ex, symbol, price, equity_free_usdt, leverage, stop_pct_price):
    risk_usdt = equity_free_usdt * RISK_PER_TRADE
    stop_dist = price * (stop_pct_price / 100.0)
    if stop_dist <= 0:
        return "0"

    qty_risk = risk_usdt / stop_dist

    max_margin = equity_free_usdt * MAX_MARGIN_PCT
    qty_cap = (max_margin * leverage) / price

    qty = min(qty_risk, qty_cap)
    qty = max(qty, 0.0)
    return ex.amount_to_precision(symbol, qty)

def close_side_from_position_side(pos_side: str):
    s = (pos_side or "").lower()
    return "sell" if s in ["long", "buy"] else "buy"

# =========================================================
# 🧠 AI 최종 확인 (쉬운 설명 강제)
# =========================================================
def generate_wonyousi_strategy(df, status_summary, rot_score, btc_state, hint):
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        system_prompt = f"""
너는 자동매매 봇의 '최종 확인'이다.

[규칙]
- 애매하면 HOLD.
- 레버리지는 {MIN_LEV}~{MAX_LEV}.
- 쉬운 설명(easy_reason)은 어려운 단어(RSI/MACD 등) 쓰지 말고, 2~3줄로 아주 쉽게 설명.
- 시스템이 이미 방향 힌트를 주었다. 반대로 가려면 confidence를 매우 낮추거나 HOLD로 해라.

[응답 JSON]
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0~100,
  "leverage": {MIN_LEV}~{MAX_LEV},
  "easy_reason": "쉬운 설명 2~3줄",
  "detail_reason": "짧은 근거"
}}
        """.strip()

        user_prompt = f"""
[상황]
- 현재가: {float(last["close"]):.6f}
- 바로 전: {float(prev["close"]):.6f} -> {float(last["close"]):.6f}
- 순환매 점수: {rot_score:.2f}
- BTC 상태: {btc_state}
- 시스템 힌트: {hint.get("direction_hint")} (이유: {hint.get("why")})

지표요약(참고):
{status_summary}
        """.strip()

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            response_format={"type":"json_object"},
            temperature=0.25
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"decision":"hold","confidence":0,"leverage":5,"easy_reason":"AI 오류로 관망","detail_reason":str(e)}

# =========================================================
# 🤖 텔레그램 + 자동매매 스레드 (실전 운영판)
# =========================================================
def telegram_thread(ex):
    state = load_runtime_state()
    last_scan = 0

    def tg_send(text):
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                data={"chat_id": tg_id, "text": text}
            )
        except:
            pass

    tg_send(
        "✅ 실전 운영판 시작\n"
        "- 동시 2포지션\n"
        "- 트레이드당 리스크 0.5%\n"
        "- 순환매(돈 몰리는 코인 우선)\n"
        "- 부분익절/본절/트레일링\n"
        "- 쿨다운/연손절/일손실 보호"
    )

    while True:
        try:
            cur_cfg = load_settings()
            is_auto_on = cur_cfg.get("auto_trade", False)

            # 잔고
            bal = ex.fetch_balance({"type":"swap"})
            usdt_free = float(bal["USDT"]["free"])
            usdt_total = float(bal["USDT"]["total"])

            _rollover_daily(state, usdt_total)

            # 일손실 제한
            if _daily_stop_hit(state, usdt_total):
                if is_auto_on:
                    tg_send("🛑 오늘 손실 한도(-2%) 도달. 자동매매 쉬어요.")
                time.sleep(10)
                continue

            # 연손절 휴식
            if _is_paused(state):
                time.sleep(5)
                continue

            # 포지션 조회
            try:
                positions = ex.fetch_positions(symbols=TARGET_COINS)
            except:
                positions = []
                for c in TARGET_COINS:
                    try:
                        positions += ex.fetch_positions([c])
                    except:
                        pass

            active_positions = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
            open_count = len(active_positions)

            # -------------------------
            # 1) 오픈 포지션 관리
            # -------------------------
            for p in active_positions:
                coin = p.get("symbol")
                if not coin:
                    continue

                pnl_pct = float(p.get("percentage", 0) or 0)  # ROI%
                contracts = float(p.get("contracts", 0) or 0)
                pos_side = (p.get("side") or "").lower()
                close_side = close_side_from_position_side(pos_side)

                trade = state.setdefault("trades", {}).get(coin)
                if not trade:
                    # 처음 발견한 포지션: 안전 기본값 생성
                    state["trades"][coin] = {
                        "sl": 4.0, "r": 4.0, "tp1": 4.0, "tp2": 8.0,
                        "tp1_done": False, "peak": pnl_pct,
                        "trail_floor": 0.0, "opened_ts": _now_ts(),
                        "easy_reason": "", "detail_reason": ""
                    }
                    trade = state["trades"][coin]
                    save_runtime_state(state)

                # peak 업데이트
                trade["peak"] = max(float(trade.get("peak", pnl_pct)), pnl_pct)

                # (A) 손절
                if pnl_pct <= -abs(float(trade["sl"])):
                    try:
                        qty_close = ex.amount_to_precision(coin, contracts)
                        ex.create_market_order(coin, close_side, qty_close, params={"reduceOnly": True})
                    except:
                        pass

                    state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                    _set_coin_cooldown(state, coin)

                    tg_send(
                        f"🩸 손절: {coin} ({pnl_pct:.2f}%)\n"
                        f"→ 같은 코인은 {COOLDOWN_AFTER_SL_MIN}분 쉬어요."
                    )

                    if state["consec_losses"] >= CONSEC_LOSS_LIMIT:
                        _hit_consec_loss_pause(state)
                        tg_send(f"⏸️ {CONSEC_LOSS_LIMIT}연속 손절. {PAUSE_AFTER_CONSEC_LOSS_MIN}분 휴식합니다.")

                    if coin in state["trades"]:
                        del state["trades"][coin]
                    save_runtime_state(state)
                    continue

                # (B) TP1 부분익절(1R에서 절반)
                if not trade.get("tp1_done", False) and pnl_pct >= float(trade["tp1"]):
                    try:
                        qty_part = contracts * TP1_FRACTION
                        qty_part = float(ex.amount_to_precision(coin, qty_part))
                        if qty_part > 0:
                            ex.create_market_order(coin, close_side, qty_part, params={"reduceOnly": True})
                    except:
                        pass

                    trade["tp1_done"] = True
                    trade["trail_floor"] = 0.0
                    save_runtime_state(state)

                    tg_send(
                        f"✅ 부분익절(TP1): {coin} (+{pnl_pct:.2f}%)\n"
                        f"→ 이제 손해 안 나게 이익 지키는 모드!"
                    )

                # (C) TP2 최종 익절(2R)
                if pnl_pct >= float(trade["tp2"]):
                    try:
                        qty_close = ex.amount_to_precision(coin, contracts)
                        ex.create_market_order(coin, close_side, qty_close, params={"reduceOnly": True})
                    except:
                        pass

                    state["consec_losses"] = 0
                    tg_send(f"🎉 익절(TP2): {coin} (+{pnl_pct:.2f}%)")

                    if coin in state["trades"]:
                        del state["trades"][coin]
                    save_runtime_state(state)
                    continue

                # (D) TP1 이후 트레일링(피크 - 0.8R 되밀리면 종료)
                if trade.get("tp1_done", False):
                    r = float(trade.get("r", 4.0))
                    peak = float(trade.get("peak", pnl_pct))
                    trail_floor = max(0.0, peak - (TRAIL_R * r))
                    trade["trail_floor"] = trail_floor
                    save_runtime_state(state)

                    if pnl_pct <= trail_floor:
                        try:
                            qty_close = ex.amount_to_precision(coin, contracts)
                            ex.create_market_order(coin, close_side, qty_close, params={"reduceOnly": True})
                        except:
                            pass

                        state["consec_losses"] = 0
                        tg_send(f"🟡 트레일링 청산: {coin}\n→ 이익 지키고 종료! (현재 {pnl_pct:.2f}%)")

                        if coin in state["trades"]:
                            del state["trades"][coin]
                        save_runtime_state(state)
                        continue

            # -------------------------
            # 2) 신규 진입
            # -------------------------
            if not is_auto_on:
                time.sleep(2)
                continue

            if open_count >= MAX_POSITIONS:
                time.sleep(2)
                continue

            if _now_ts() - last_scan < SCAN_INTERVAL_SEC:
                time.sleep(1)
                continue

            last_scan = _now_ts()

            # BTC 기준 데이터(레짐/순환매)
            btc_symbol = "BTC/USDT:USDT"
            btc_ohlcv = ex.fetch_ohlcv(btc_symbol, "5m", limit=200)
            btc_df = pd.DataFrame(btc_ohlcv, columns=["time","open","high","low","close","vol"])
            btc_df["time"] = pd.to_datetime(btc_df["time"], unit="ms")
            btc_df["volume"] = btc_df["vol"]
            btc_state = btc_regime(btc_df)

            # 후보 스캔(5개 코인)
            scanned = []
            for coin in TARGET_COINS:
                if _coin_in_cooldown(state, coin):
                    continue

                # 스프레드 필터
                sp = get_spread_pct(ex, coin)
                if sp >= SPREAD_LIMIT_PCT:
                    continue

                ohlcv = ex.fetch_ohlcv(coin, "5m", limit=200)
                df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df["volume"] = df["vol"]

                df_calc, status, last = calc_indicators(df, cur_cfg)
                if last is None or df_calc is None or df_calc.empty:
                    continue

                score, vspike, rs = rotation_score(df_calc, btc_df)

                direction_hint = "buy" if score >= 1.0 else ("sell" if score <= -1.0 else "hold")
                if direction_hint == "hold":
                    continue

                # BTC 위험장에서는 알트 롱 신규 진입 보수적으로
                if btc_state == "risk_off" and direction_hint == "buy" and coin != btc_symbol:
                    continue

                scanned.append({
                    "coin": coin,
                    "df": df_calc,
                    "status": status,
                    "last": last,
                    "score": score,
                    "hint": {"direction_hint": direction_hint, "why": f"순환매 점수 {score:.1f} (돈이 이쪽으로 몰리는 편)"}
                })

            scanned.sort(key=lambda x: abs(x["score"]), reverse=True)

            # 이미 포지션 1개 있으면 상관 필터 적용
            active_coin = active_positions[0]["symbol"] if open_count >= 1 else None
            active_df = None
            if active_coin:
                try:
                    aohlcv = ex.fetch_ohlcv(active_coin, "5m", limit=200)
                    active_df = pd.DataFrame(aohlcv, columns=["time","open","high","low","close","vol"])
                    active_df["volume"] = active_df["vol"]
                except:
                    active_df = None

            required_conf = 85 if open_count >= 1 else 80

            for item in scanned:
                if open_count >= MAX_POSITIONS:
                    break

                coin = item["coin"]
                df_calc = item["df"]
                status = item["status"]
                last = item["last"]
                score = item["score"]
                hint = item["hint"]

                # 두 번째 포지션이면 상관 높은 코인 제외
                if active_df is not None:
                    try:
                        c = corr_of_returns(active_df, df_calc, n=60)
                        if c >= CORR_LIMIT:
                            continue
                    except:
                        pass

                # AI 최종 확인(쉬운 설명 포함)
                ai = generate_wonyousi_strategy(df_calc, status, score, btc_state, hint)
                decision = ai.get("decision", "hold")
                conf = float(ai.get("confidence", 0))
                lev = int(ai.get("leverage", 5))
                lev = max(MIN_LEV, min(MAX_LEV, lev))

                if decision not in ["buy", "sell"] or conf < required_conf:
                    continue

                # 힌트 반대면 더 보수적으로
                if decision != hint["direction_hint"] and conf < 90:
                    continue

                price = float(last["close"])

                # ATR 기반 손절폭(가격 기준 %)
                atr = calc_atr(df_calc, 14)
                atr_pct = float(atr.iloc[-1] / price * 100) if atr is not None and not np.isnan(atr.iloc[-1]) else 1.0
                stop_pct_price = atr_pct * ATR_MULT
                stop_pct_price = max(MIN_STOP_PCT_PRICE, min(MAX_STOP_PCT_PRICE, stop_pct_price))

                # 리스크 0.5% 기반 수량 계산
                qty = calc_qty_by_risk(ex, coin, price, usdt_free, lev, stop_pct_price)
                if float(qty) <= 0:
                    continue

                # 레버리지 설정
                try:
                    ex.set_leverage(lev, coin)
                except:
                    pass

                # 진입
                try:
                    ex.create_market_order(coin, decision, qty)
                except Exception as e:
                    print("Order error:", e)
                    continue

                # 1R/2R 목표(ROI 기준)
                r_roi = stop_pct_price * lev
                sl = r_roi
                tp1 = r_roi
                tp2 = r_roi * 2

                state.setdefault("trades", {})[coin] = {
                    "sl": float(sl),
                    "r": float(r_roi),
                    "tp1": float(tp1),
                    "tp2": float(tp2),
                    "tp1_done": False,
                    "peak": 0.0,
                    "trail_floor": 0.0,
                    "opened_ts": _now_ts(),
                    "easy_reason": ai.get("easy_reason", ""),
                    "detail_reason": ai.get("detail_reason", ""),
                }
                save_runtime_state(state)

                easy = ai.get("easy_reason", "(설명 없음)")
                tg_send(
                    f"🎯 진입: {coin} / {decision.upper()} (conf {conf}%, x{lev})\n"
                    f"- 계획: 손절 -{sl:.1f}% / 1차 +{tp1:.1f}% / 2차 +{tp2:.1f}%\n"
                    f"- 쉬운 이유: {easy}"
                )

                open_count += 1
                time.sleep(3)

            time.sleep(1)

        except Exception as e:
            print("Thread Error:", e)
            time.sleep(5)

# =========================================================
# 🧾 OHLCV 로드
# =========================================================
def fetch_ohlcv_df(ex, sym: str, tf: str, limit: int = 200):
    ohlcv = ex.fetch_ohlcv(sym, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

# =========================================================
# ✅ 사이드바 UI
# =========================================================
st.sidebar.title("🛠️ 설정")

symbol = st.sidebar.selectbox("코인 선택", TARGET_COINS, index=0)
timeframe = st.sidebar.selectbox("타임프레임", ["1m","3m","5m","15m","30m","1h","4h","1d"], index=2)

st.sidebar.divider()
st.sidebar.subheader("🤖 자동매매")
auto_on = st.sidebar.checkbox("자동매매 ON (텔레그램)", value=config.get("auto_trade", False))
if auto_on != config.get("auto_trade", False):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.caption(f"동시 포지션: {MAX_POSITIONS}개 / 리스크: 0.5% / 레버리지: {MIN_LEV}~{MAX_LEV}")

st.sidebar.divider()
st.sidebar.subheader("📊 지표 파라미터")
c1, c2, c3 = st.sidebar.columns(3)
config["rsi_period"] = c1.number_input("RSI 기간", 5, 50, int(config["rsi_period"]))
config["rsi_buy"] = c2.number_input("RSI 과매도", 10, 50, int(config["rsi_buy"]))
config["rsi_sell"] = c3.number_input("RSI 과매수", 50, 90, int(config["rsi_sell"]))

c4, c5 = st.sidebar.columns(2)
config["bb_period"] = c4.number_input("BB 기간", 10, 50, int(config["bb_period"]))
config["bb_std"] = c5.number_input("BB 표준편차", 1.0, 4.0, float(config["bb_std"]))

c6, c7 = st.sidebar.columns(2)
config["ma_fast"] = c6.number_input("MA fast", 3, 50, int(config["ma_fast"]))
config["ma_slow"] = c7.number_input("MA slow", 50, 200, int(config["ma_slow"]))

config["stoch_k"] = st.sidebar.number_input("Stoch K", 5, 50, int(config["stoch_k"]))
config["vol_mul"] = st.sidebar.number_input("거래량 폭증 배수", 1.2, 5.0, float(config["vol_mul"]))

st.sidebar.divider()
if st.sidebar.button("💾 설정 저장"):
    save_settings(config)

st.sidebar.divider()
st.sidebar.header("🔍 점검")
if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":"테스트입니다. 1+1은?"}],
            max_tokens=10
        )
        st.sidebar.success("✅ 연결 성공: " + resp.choices[0].message.content)
    except Exception as e:
        st.sidebar.error(f"❌ 연결 실패: {e}")

# =========================================================
# ✅ 자동매매 스레드 시작(1회)
# =========================================================
if "bot_thread_started" not in st.session_state:
    st.session_state["bot_thread_started"] = False

if not st.session_state["bot_thread_started"]:
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_Thread")
    add_script_run_ctx(th)
    th.start()
    st.session_state["bot_thread_started"] = True

# =========================================================
# ✅ 메인 화면
# =========================================================
st.title("📌 비트겟 AI 워뇨띠 에이전트 (실전 운영판)")

top1, top2, top3 = st.columns([2, 2, 3])
with top1:
    st.metric("선택 코인", symbol)
with top2:
    st.metric("타임프레임", timeframe)
with top3:
    st.metric("자동매매", "🟢 ON" if config.get("auto_trade") else "🔴 OFF")

# 데이터 로드
data_loaded = False
df = None
status = {}
last = None

try:
    df0 = fetch_ohlcv_df(exchange, symbol, timeframe, limit=220)
    # timeframe은 화면용이므로, 지표/운영은 여기서 계산
    df, status, last = calc_indicators(df0, config)
    data_loaded = last is not None
except Exception as e:
    st.error(f"⚠️ 데이터 로딩 오류: {e}")

if not data_loaded:
    st.warning("⏳ 데이터 로딩 중... (리런해보세요)")
    st.stop()

# =========================================================
# ✅ 메인 레이아웃: 좌(차트/지표) + 우(지갑/포지션/AI)
# =========================================================
left, right = st.columns([3.2, 1.8], gap="large")

with left:
    st.subheader("📈 TradingView 차트")
    render_tradingview(symbol, timeframe, height=520, theme="dark")

    st.divider()
    st.subheader("🚦 10종 보조지표 상태판")

    rows = []
    def add_row(name, val, state):
        rows.append({"지표": name, "값": val, "상태": state})

    add_row("RSI", f"{last['RSI']:.1f}", f"{status.get('RSI','')} {status.get('RSI_FLOW','')}")
    add_row("Bollinger", f"{last['BB_mid']:.4f}", status.get("BB",""))
    add_row("MA(fast/slow)", f"{last['MA_fast']:.4f}/{last['MA_slow']:.4f}", status.get("MA",""))
    add_row("MACD", f"{last['MACD']:.6f}", status.get("MACD",""))
    add_row("Stoch(K/D)", f"{last['STO_K']:.1f}/{last['STO_D']:.1f}", status.get("STOCH",""))
    add_row("CCI", f"{last['CCI']:.1f}", status.get("CCI",""))
    add_row("MFI", f"{last['MFI']:.1f}", status.get("MFI",""))
    add_row("Williams %R", f"{last['WILLR']:.1f}", status.get("WILLR",""))
    add_row("ADX", f"{last['ADX']:.1f}", status.get("ADX",""))
    add_row("Volume Spike", f"{last['VOL_SPIKE']:.2f}x", status.get("VOL",""))

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption("※ 자동매매는 데모(IS_SANDBOX=True)에서 충분히 검증 후 실전 전환하세요.")

with right:
    st.subheader("💰 내 잔고 / 포지션")
    try:
        bal = exchange.fetch_balance({"type":"swap"})
        usdt_free = float(bal["USDT"]["free"])
        usdt_total = float(bal["USDT"]["total"])
        st.metric("총 자산(USDT)", f"${usdt_total:,.2f}")
        st.metric("주문 가능(USDT)", f"${usdt_free:,.2f}")
    except Exception as e:
        st.error(f"잔고 조회 실패: {e}")

    st.divider()
    st.subheader("📌 현재 포지션")
    try:
        positions = exchange.fetch_positions(symbols=TARGET_COINS)
        active_positions = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]

        if not active_positions:
            st.info("무포지션 (관망 중)")
        else:
            for p in active_positions:
                sym = (p.get("symbol","")).split(":")[0]
                side = (p.get("side","")).lower()
                side_label = "🟢 Long" if side in ["long","buy"] else "🔴 Short"
                pnl = float(p.get("unrealizedPnl", 0) or 0)
                roi = float(p.get("percentage", 0) or 0)
                lev = p.get("leverage", "?")
                st.info(f"**{sym}** ({side_label} x{lev})\n\n수익률: **{roi:.2f}%** / 손익: **${pnl:.2f}**")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")

    st.divider()
    st.subheader("🤖 지금 이 코인 AI 분석")
    if st.button("🔍 AI가 쉽게 설명해주기"):
        with st.spinner("AI가 최종 체크 중..."):
            # 현재 코인도 순환매 점수/레짐 기준으로 한 번 보여주기
            try:
                btc_df0 = fetch_ohlcv_df(exchange, "BTC/USDT:USDT", "5m", limit=220)
                btc_df0["volume"] = btc_df0["volume"]
                btc_calc, _, _ = calc_indicators(btc_df0, config)
                btc_state = btc_regime(btc_calc if btc_calc is not None else btc_df0)

                # current symbol 로테이션 점수(참고)
                if symbol == "BTC/USDT:USDT":
                    score = 2.0
                else:
                    score, _, _ = rotation_score(df, btc_calc if btc_calc is not None else btc_df0)

                hint = {"direction_hint": "buy" if score >= 1.0 else ("sell" if score <= -1.0 else "hold"),
                        "why": f"순환매 점수 {score:.1f}"}

                ai = generate_wonyousi_strategy(df, status, float(score), btc_state, hint)
                st.success("✅ 쉬운 설명")
                st.write(ai.get("easy_reason","(없음)"))

                with st.expander("자세한 근거(고급)"):
                    st.write(ai)
            except Exception as e:
                st.error(f"AI 분석 오류: {e}")

# =========================================================
# ✅ 탭(t1~t4)
# =========================================================
st.divider()
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🧠 자동매매 & AI분석")
    st.caption("AI는 최종 확인만 하고, 진입/청산/리스크/안전장치는 시스템이 관리합니다. (설명은 쉽게!)")

    c_auto, c_stat = st.columns([3, 1])
    with c_auto:
        auto_on2 = st.checkbox("🤖 24시간 자동매매 활성화", value=config.get("auto_trade", False))
        if auto_on2 != config.get("auto_trade", False):
            config["auto_trade"] = auto_on2
            save_settings(config)
            st.rerun()
    with c_stat:
        st.caption("상태: " + ("🟢 가동중" if config.get("auto_trade") else "🔴 정지"))

    st.divider()
    col1, col2 = st.columns(2)

    if col1.button("🔍 현재 코인: AI 쉬운 설명"):
        with st.spinner("AI 확인 중..."):
            try:
                btc_df0 = fetch_ohlcv_df(exchange, "BTC/USDT:USDT", "5m", limit=220)
                btc_calc, _, _ = calc_indicators(btc_df0, config)
                btc_state = btc_regime(btc_calc if btc_calc is not None else btc_df0)

                score = 2.0 if symbol == "BTC/USDT:USDT" else rotation_score(df, btc_calc if btc_calc is not None else btc_df0)[0]
                hint = {"direction_hint": "buy" if score >= 1.0 else ("sell" if score <= -1.0 else "hold"),
                        "why": f"순환매 점수 {score:.1f}"}

                ai = generate_wonyousi_strategy(df, status, float(score), btc_state, hint)

                st.write("### ✅ 결론")
                st.write(f"- 결정: **{ai.get('decision','hold').upper()}** / 확신도: **{ai.get('confidence',0)}%** / 레버리지: **x{ai.get('leverage',5)}**")
                st.write("### ✅ 쉬운 설명")
                st.info(ai.get("easy_reason","(없음)"))

                with st.expander("자세한 설명(고급)"):
                    st.write(ai.get("detail_reason",""))
            except Exception as e:
                st.error(f"분석 오류: {e}")

    if col2.button("🌍 전체 코인 순환매 랭킹(상위 우선)"):
        with st.spinner("5개 코인 스캔 중..."):
            try:
                btc_df0 = fetch_ohlcv_df(exchange, "BTC/USDT:USDT", "5m", limit=220)
                btc_calc, _, _ = calc_indicators(btc_df0, config)
                btc_state = btc_regime(btc_calc if btc_calc is not None else btc_df0)

                rows = []
                for c in TARGET_COINS:
                    dfx = fetch_ohlcv_df(exchange, c, "5m", limit=220)
                    dfx, stx, lastx = calc_indicators(dfx, config)
                    if lastx is None:
                        continue
                    score, vspike, rs = rotation_score(dfx, btc_calc if btc_calc is not None else btc_df0)
                    rows.append({
                        "코인": c.split("/")[0],
                        "순환매점수": round(score, 2),
                        "BTC대비강도": round(rs, 2),
                        "거래량": f"{vspike:.2f}x",
                        "힌트": "롱(유리)" if score >= 1.0 else ("숏(유리)" if score <= -1.0 else "애매")
                    })

                df_rank = pd.DataFrame(rows).sort_values(by="순환매점수", key=lambda s: s.abs(), ascending=False)
                st.caption(f"BTC 상태: **{btc_state}**  (risk_off면 알트 롱 신규진입이 보수적으로 동작)")
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"스캔 오류: {e}")

with t2:
    st.subheader("⚡ 수동주문(자리만)")
    st.caption("원하면 여기에 실제 수동 주문/청산 로직도 붙여줄 수 있어요.")
    m_amt = st.number_input("주문 금액($)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입"):
        st.info("여기 롱 진입 로직 연결 가능")
    if b2.button("🔴 숏 진입"):
        st.info("여기 숏 진입 로직 연결 가능")
    if b3.button("🚫 포지션 종료"):
        st.info("여기 종료 로직 연결 가능")

with t3:
    st.subheader("📅 시장정보")
    st.write("원하면 경제일정/뉴스를 붙여서 리스크 구간 진입을 더 보수적으로 만들 수 있어요.")

with t4:
    st.subheader("📜 매매일지(선택)")
    st.caption("지금 버전은 런타임 상태(runtime_state.json)로 운영합니다. (원하면 trade_log.csv 기록도 추가 가능)")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                st.json(json.load(f))
        except Exception as e:
            st.error(f"상태 파일 읽기 오류: {e}")
    else:
        st.info("아직 runtime_state.json이 없습니다. 자동매매가 돌면 생성됩니다.")
