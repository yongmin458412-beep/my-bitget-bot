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
# ⚙️ 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Ultimate)")

IS_SANDBOX = True  # 실전매매면 False
SETTINGS_FILE = "bot_settings.json"

RUNTIME_STATE_FILE = "runtime_state.json"   # 네가 말한 런타임 상태 파일
TRADE_LOG_FILE = "trade_log.csv"            # 체결/청산 기록(표/다운로드용)

# 감시 코인 (Bitget swap 심볼)
TARGET_COINS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"]

# =========================================================
# 💾 설정 관리
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",

        "auto_trade": False,
        "max_positions": 2,         # ✅ 동시에 잡는 포지션 수 (추천: 2)
        "leverage": 5,              # 기본 레버리지(최종은 AI가 제안)
        "order_usdt": 100.0,

        # 손익비/부분익절(꾸준한 수익실현용)
        "tp1_gap": 0.5,             # ✅ 0.5% 도달 시 부분익절
        "tp1_size": 30,             # ✅ 30% 부분익절
        "move_sl_to_be": True,      # ✅ TP1 후 손절을 본절로 당김

        # 지표 파라미터
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "adx_period": 14,
        "stoch_k": 14, "stoch_d": 3,
        "cci_period": 20,
        "mfi_period": 14,
        "willr_period": 14,
        "vol_sma": 20, "vol_mul": 2.0,

        # 지표 사용 여부(10종)
        "use_rsi": True,
        "use_bb": True,
        "use_ma": True,
        "use_macd": True,
        "use_adx": True,
        "use_stoch": True,
        "use_cci": True,
        "use_mfi": True,
        "use_willr": True,
        "use_vol": True,

        # 필터/안전장치
        "min_sl_gap": 2.5,          # ✅ 너무 타이트한 손절 금지
        "min_rr": 1.8,              # ✅ 최소 손익비(대략 1:1.8 이상)
        "cooldown_minutes": 15,     # ✅ 코인별 재진입 쿨다운
        "max_consec_losses": 3,     # ✅ 연속 손실 제한
        "pause_minutes": 60,        # ✅ 연속 손실이면 자동 정지 시간
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default.update(saved)
        except:
            pass
    return default


def save_settings(cfg: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        st.toast("✅ 설정 저장 완료", icon="💾")
    except:
        st.error("설정 저장 실패")


config = load_settings()

# =========================================================
# 🧠 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime_state():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,     # epoch seconds
        "cooldowns": {},      # {symbol: epoch_until}
        "trades": {}          # {symbol: {...}}
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


def save_runtime_state(state: dict):
    try:
        with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except:
        pass


def maybe_roll_daily_state(state: dict, current_equity: float):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state["date"] = today
        state["day_start_equity"] = float(current_equity)
        state["daily_realized_pnl"] = 0.0
        state["consec_losses"] = 0
        state["pause_until"] = 0
        state["cooldowns"] = {}
        state["trades"] = {}
        save_runtime_state(state)


# =========================================================
# 📝 매매 로그(trade_log.csv) - 보기/다운로드용
# =========================================================
def append_trade_log(row: dict):
    cols = ["Time", "Symbol", "Event", "Side", "Qty", "Price", "PnL_USDT", "PnL_Pct", "Note"]
    df = pd.DataFrame([{c: row.get(c, "") for c in cols}])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")


# =========================================================
# 🔐 Secrets 로드
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key or not api_secret or not api_password:
    st.error("🚨 Bitget API 키(secrets)가 비어있습니다. Streamlit secrets.toml 확인!")
    st.stop()

if not openai_key:
    st.warning("⚠️ OpenAI 키가 없어서 AI 분석은 HOLD만 나옵니다. (secrets 또는 설정에서 입력)")
else:
    openai_client = OpenAI(api_key=openai_key)

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
# 📈 보조지표(ta 없이 직접 계산) - 10종
# =========================================================
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return mid, upper, lower

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def adx(high, low, close, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.rolling(period).mean()
    return adx_val, plus_di, minus_di

def stoch(high, low, close, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def cci(high, low, close, period: int = 20):
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period).mean()
    mad = (tp - sma_tp).abs().rolling(period).mean()
    cci_val = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return cci_val

def mfi(high, low, close, volume, period: int = 14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = tp.diff()
    pos_mf = mf.where(direction > 0, 0.0)
    neg_mf = mf.where(direction < 0, 0.0).abs()
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / (neg_sum.replace(0, np.nan))
    return 100 - (100 / (1 + mfr))

def williams_r(high, low, close, period: int = 14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    wr = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    return wr

def calc_indicators(df: pd.DataFrame, cfg: dict):
    if df is None or df.empty or len(df) < 120:
        return df, {}, None

    df = df.copy()

    # 10종 계산
    if cfg["use_rsi"]:
        df["RSI"] = rsi(df["close"], cfg["rsi_period"])

    if cfg["use_bb"]:
        mid, up, low_ = bollinger(df["close"], cfg["bb_period"], cfg["bb_std"])
        df["BB_mid"], df["BB_upper"], df["BB_lower"] = mid, up, low_

    if cfg["use_ma"]:
        df["MA_fast"] = df["close"].rolling(cfg["ma_fast"]).mean()
        df["MA_slow"] = df["close"].rolling(cfg["ma_slow"]).mean()

    if cfg["use_macd"]:
        m, s, h = macd(df["close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = m, s, h

    if cfg["use_adx"]:
        a, pdi, mdi = adx(df["high"], df["low"], df["close"], cfg["adx_period"])
        df["ADX"], df["PDI"], df["MDI"] = a, pdi, mdi

    if cfg["use_stoch"]:
        k, d = stoch(df["high"], df["low"], df["close"], cfg["stoch_k"], cfg["stoch_d"])
        df["STO_K"], df["STO_D"] = k, d

    if cfg["use_cci"]:
        df["CCI"] = cci(df["high"], df["low"], df["close"], cfg["cci_period"])

    if cfg["use_mfi"]:
        df["MFI"] = mfi(df["high"], df["low"], df["close"], df["vol"], cfg["mfi_period"])

    if cfg["use_willr"]:
        df["WILLR"] = williams_r(df["high"], df["low"], df["close"], cfg["willr_period"])

    if cfg["use_vol"]:
        df["VOL_SMA"] = df["vol"].rolling(cfg["vol_sma"]).mean()

    df = df.dropna()
    if df.empty:
        return df, {}, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    status = {}
    # RSI 상태(“탈출” 강조)
    if cfg["use_rsi"]:
        buy_th = cfg["rsi_buy"]
        sell_th = cfg["rsi_sell"]
        if prev["RSI"] < buy_th and last["RSI"] >= buy_th:
            status["RSI"] = "🟢 과매도 탈출(반등)"
        elif prev["RSI"] > sell_th and last["RSI"] <= sell_th:
            status["RSI"] = "🔴 과매수 탈출(눌림)"
        else:
            status["RSI"] = "⚪ 중립"

    # BB
    if cfg["use_bb"]:
        if last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈(과매도)"
        elif last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파(과열)"
        else:
            status["BB"] = "⚪ 밴드 내"

    # MA
    if cfg["use_ma"]:
        if prev["MA_fast"] <= prev["MA_slow"] and last["MA_fast"] > last["MA_slow"]:
            status["MA"] = "📈 골든크로스"
        elif prev["MA_fast"] >= prev["MA_slow"] and last["MA_fast"] < last["MA_slow"]:
            status["MA"] = "📉 데드크로스"
        else:
            status["MA"] = "⚪ 유지"

    # MACD
    if cfg["use_macd"]:
        if prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
            status["MACD"] = "📈 골든크로스"
        elif prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
            status["MACD"] = "📉 데드크로스"
        else:
            status["MACD"] = "⚪ 유지"

    # ADX
    if cfg["use_adx"]:
        status["ADX"] = "🔥 추세장" if last["ADX"] >= 25 else "💤 횡보장"

    # Stoch
    if cfg["use_stoch"]:
        if prev["STO_K"] <= prev["STO_D"] and last["STO_K"] > last["STO_D"] and last["STO_K"] < 30:
            status["STOCH"] = "🟢 바닥 반등"
        elif prev["STO_K"] >= prev["STO_D"] and last["STO_K"] < last["STO_D"] and last["STO_K"] > 70:
            status["STOCH"] = "🔴 꼭대기 꺾임"
        else:
            status["STOCH"] = "⚪ 중립"

    # CCI
    if cfg["use_cci"]:
        if last["CCI"] < -100:
            status["CCI"] = "🟢 과매도"
        elif last["CCI"] > 100:
            status["CCI"] = "🔴 과매수"
        else:
            status["CCI"] = "⚪ 중립"

    # MFI
    if cfg["use_mfi"]:
        if last["MFI"] < 20:
            status["MFI"] = "🟢 과매도"
        elif last["MFI"] > 80:
            status["MFI"] = "🔴 과매수"
        else:
            status["MFI"] = "⚪ 중립"

    # Williams %R
    if cfg["use_willr"]:
        if last["WILLR"] < -80:
            status["WILLR"] = "🟢 과매도"
        elif last["WILLR"] > -20:
            status["WILLR"] = "🔴 과매수"
        else:
            status["WILLR"] = "⚪ 중립"

    # Volume
    if cfg["use_vol"]:
        if last["VOL_SMA"] > 0 and last["vol"] >= last["VOL_SMA"] * cfg["vol_mul"]:
            status["VOL"] = "🔥 거래량 급증"
        else:
            status["VOL"] = "⚪ 보통"

    return df, status, last


def score_signals(status: dict):
    """롱/숏 투표 점수화(단순판)"""
    long_score = 0
    short_score = 0

    txt = " ".join(status.values())

    if "과매도 탈출" in txt or "바닥 반등" in txt:
        long_score += 2
    if "골든크로스" in txt:
        long_score += 1
    if "하단 이탈" in txt or "과매도" in txt:
        long_score += 1

    if "과매수 탈출" in txt or "꼭대기 꺾임" in txt:
        short_score += 2
    if "데드크로스" in txt:
        short_score += 1
    if "상단 돌파" in txt or "과매수" in txt:
        short_score += 1

    # 추세장일 때는 신호 신뢰도 약간 가중
    if status.get("ADX") == "🔥 추세장":
        if long_score > short_score:
            long_score += 1
        elif short_score > long_score:
            short_score += 1

    return long_score, short_score


# =========================================================
# 🤖 AI 전략(손익비/부분익절/쉬운 설명 포함)
# =========================================================
def generate_ai_strategy(symbol: str, df: pd.DataFrame, status: dict, cfg: dict):
    if not openai_key:
        return {
            "decision": "hold",
            "confidence": 0,
            "leverage": cfg["leverage"],
            "percentage": 10,
            "sl_gap": max(cfg["min_sl_gap"], 2.5),
            "tp_gap": max(cfg["min_sl_gap"] * cfg["min_rr"], 5.0),
            "tp1_gap": cfg["tp1_gap"],
            "tp1_size": cfg["tp1_size"],
            "reason_simple": "OpenAI 키가 없어서 관망합니다.",
            "reason_detail": "OPENAI_API_KEY 설정 후 다시 시도하세요."
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    long_score, short_score = score_signals(status)

    system_prompt = f"""
너는 '수익실현을 꾸준히' 하는 신중한 트레이더다.

[목표]
- 손절이 너무 잦지 않게(=손절폭 너무 작게 금지)
- 손익비가 좋은 거래만(최소 손익비 {cfg['min_rr']} 이상)
- 부분익절로 '작게라도 자주' 실현 (TP1: {cfg['tp1_gap']}%에서 {cfg['tp1_size']}% 정리)

[규칙]
1) RSI는 '극단 구간 진입'이 아니라 '탈출(반등/눌림)'을 더 중요하게 본다.
2) 손절폭 sl_gap 은 최소 {cfg['min_sl_gap']}% 이상.
3) 레버리지는 3~10 범위 권장(과한 레버리지 금지).
4) 확신도가 낮으면 무조건 hold.

[응답 JSON]
decision: buy/sell/hold
confidence: 0~100
percentage: 5~30
leverage: 3~10
sl_gap: 2.5~7.0
tp_gap: sl_gap*{cfg['min_rr']} 이상(가능하면 더)
tp1_gap: {cfg['tp1_gap']}
tp1_size: {cfg['tp1_size']}
reason_simple: 초보도 이해 가능한 한글 2~3줄
reason_detail: 기술적 근거(지표/흐름)
"""

    user_prompt = f"""
[심볼] {symbol}
[가격] 현재가 {last['close']}
[흐름] 직전->현재 RSI: {prev.get('RSI', np.nan)} -> {last.get('RSI', np.nan)}
[추세] ADX: {last.get('ADX', np.nan)}
[요약] {status}
[점수] long_score={long_score}, short_score={short_score}

주의: "손절이 1.5% 같은 초타이트"는 금지. 손익비가 맞아야 한다.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )
        out = json.loads(resp.choices[0].message.content)

        # ✅ 안전장치 보정
        out["sl_gap"] = float(max(out.get("sl_gap", cfg["min_sl_gap"]), cfg["min_sl_gap"]))
        out["leverage"] = int(min(max(int(out.get("leverage", cfg["leverage"])), 3), 10))
        out["percentage"] = float(min(max(float(out.get("percentage", 10)), 5), 30))

        # tp_gap 최소 손익비 보정
        min_tp = out["sl_gap"] * float(cfg["min_rr"])
        out["tp_gap"] = float(max(float(out.get("tp_gap", min_tp)), min_tp))

        out["tp1_gap"] = float(cfg["tp1_gap"])
        out["tp1_size"] = int(cfg["tp1_size"])
        return out
    except Exception as e:
        return {
            "decision": "hold",
            "confidence": 0,
            "leverage": cfg["leverage"],
            "percentage": 10,
            "sl_gap": max(cfg["min_sl_gap"], 2.5),
            "tp_gap": max(cfg["min_sl_gap"] * cfg["min_rr"], 5.0),
            "tp1_gap": cfg["tp1_gap"],
            "tp1_size": cfg["tp1_size"],
            "reason_simple": "AI 호출 중 오류로 관망합니다.",
            "reason_detail": f"에러: {e}"
        }


# =========================================================
# 🧾 포지션/손익 계산 보조
# =========================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def compute_roi_pct(side: str, entry: float, mark: float, lev: float):
    if entry <= 0:
        return 0.0
    direction = 1.0 if side in ["long", "buy"] else -1.0
    return direction * ((mark - entry) / entry) * 100.0 * max(1.0, lev)

def get_active_positions(ex, symbols):
    try:
        ps = ex.fetch_positions(symbols=symbols)
        act = []
        for p in ps:
            contracts = safe_float(p.get("contracts", 0))
            if contracts > 0:
                act.append(p)
        return act
    except:
        return []

def is_paused(state: dict):
    return time.time() < safe_float(state.get("pause_until", 0))

def in_cooldown(state: dict, symbol: str):
    until = safe_float(state.get("cooldowns", {}).get(symbol, 0))
    return time.time() < until

def set_cooldown(state: dict, symbol: str, minutes: int):
    state.setdefault("cooldowns", {})
    state["cooldowns"][symbol] = int(time.time() + minutes * 60)
    save_runtime_state(state)


# =========================================================
# 🔁 “순환매(로테이션)” 간단 구현: 5개 중 ‘움직임 큰’ 2개만 신규진입
# =========================================================
def pick_rotation_symbols(ex, symbols, timeframe="5m", limit=60, top_n=2):
    # 최근 1시간(5m*12) 변동이 큰 코인 2개만 뽑기
    scored = []
    for sym in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 15:
                continue
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
            close = df["close"]
            # 1시간 전 대비 변화율(절댓값)
            base = float(close.iloc[-13]) if len(close) >= 13 else float(close.iloc[0])
            now = float(close.iloc[-1])
            chg = abs((now - base) / base) * 100 if base > 0 else 0
            # 거래량도 같이 반영(간단히)
            v = float(df["vol"].iloc[-1])
            scored.append((sym, chg, v))
        except:
            pass

    if not scored:
        return symbols[:top_n]

    # 변화율 우선, 동률이면 거래량
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in scored[:top_n]]


# =========================================================
# 🤖 텔레그램 스레드(자동매매)
# =========================================================
def telegram_thread(ex):
    state = load_runtime_state()

    offset = 0
    last_report = 0
    REPORT_INTERVAL = 900

    def tg_send(text):
        if not tg_token or not tg_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                data={"chat_id": tg_id, "text": text},
                timeout=5,
            )
        except:
            pass

    tg_send("🚀 봇 가동 시작 (자동매매 스레드)")

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()

            # 잔고로 일일 시작자산 롤링
            try:
                bal = ex.fetch_balance({"type": "swap"})
                equity = safe_float(bal["USDT"]["total"])
            except:
                equity = state.get("day_start_equity", 0.0)

            maybe_roll_daily_state(state, equity)

            # 자동매매 OFF면 텔레그램 버튼만 처리
            if not cfg.get("auto_trade", False):
                time.sleep(1)
                continue

            # 연속 손실로 일시정지
            if is_paused(state):
                time.sleep(2)
                continue

            # 1) 현재 포지션 관리 (TP1/TP/SL)
            active_positions = get_active_positions(ex, TARGET_COINS)

            for p in active_positions:
                sym = p.get("symbol")
                side = p.get("side", "long")  # bitget: long/short
                contracts = safe_float(p.get("contracts", 0))
                lev = safe_float(p.get("leverage", cfg["leverage"]))
                entry = safe_float(p.get("entryPrice", 0))
                mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0))

                roi_pct = safe_float(p.get("percentage", None))
                if roi_pct is None:
                    roi_pct = compute_roi_pct(side, entry, mark, lev)

                trade_meta = state.get("trades", {}).get(sym, {})
                sl = float(trade_meta.get("sl_gap", cfg["min_sl_gap"]))
                tp = float(trade_meta.get("tp_gap", sl * cfg["min_rr"]))
                tp1_gap = float(trade_meta.get("tp1_gap", cfg["tp1_gap"]))
                tp1_size = int(trade_meta.get("tp1_size", cfg["tp1_size"]))
                tp1_done = bool(trade_meta.get("tp1_done", False))

                # ✅ TP1 부분익절
                if (not tp1_done) and roi_pct >= tp1_gap and contracts > 0:
                    close_qty = contracts * (tp1_size / 100.0)
                    close_qty = float(ex.amount_to_precision(sym, close_qty))

                    if close_qty > 0:
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            ex.create_market_order(sym, close_side, close_qty)
                        except:
                            pass

                        # TP1 후 본절 스탑(메타만 업데이트: 실제 스탑 주문 대신 봇이 감시)
                        state.setdefault("trades", {}).setdefault(sym, {})
                        state["trades"][sym]["tp1_done"] = True
                        if cfg.get("move_sl_to_be", True):
                            state["trades"][sym]["be_price"] = entry  # 본절 기준가
                        save_runtime_state(state)

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "TP1(부분익절)",
                            "Side": side,
                            "Qty": close_qty,
                            "Price": mark,
                            "PnL_USDT": "",
                            "PnL_Pct": f"{roi_pct:.2f}",
                            "Note": "0.5% 도달 부분익절"
                        })
                        tg_send(f"✅ TP1 부분익절: {sym} ({roi_pct:.2f}%)")

                # ✅ 본절 방어(= TP1 이후, ROI가 0 근처로 되돌면 나가기)
                be_price = trade_meta.get("be_price", None)
                if be_price and contracts > 0:
                    # 단순하게 ROI <= 0.1%면 정리(원하면 0.0으로 바꿔도 됨)
                    if roi_pct <= 0.1:
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            ex.create_market_order(sym, close_side, contracts)
                        except:
                            pass

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "BE(본절정리)",
                            "Side": side,
                            "Qty": contracts,
                            "Price": mark,
                            "PnL_USDT": "",
                            "PnL_Pct": f"{roi_pct:.2f}",
                            "Note": "TP1 후 본절 방어"
                        })
                        tg_send(f"🛡️ 본절 정리: {sym} ({roi_pct:.2f}%)")

                        # 쿨다운
                        set_cooldown(state, sym, cfg["cooldown_minutes"])
                        state["trades"].pop(sym, None)
                        save_runtime_state(state)
                        continue

                # ✅ SL/TP 청산
                if contracts > 0:
                    if roi_pct <= -abs(sl):
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            ex.create_market_order(sym, close_side, contracts)
                        except:
                            pass

                        state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                        # 연속 손실이면 일시정지
                        if state["consec_losses"] >= cfg["max_consec_losses"]:
                            state["pause_until"] = int(time.time() + cfg["pause_minutes"] * 60)

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "SL(손절)",
                            "Side": side,
                            "Qty": contracts,
                            "Price": mark,
                            "PnL_USDT": "",
                            "PnL_Pct": f"{roi_pct:.2f}",
                            "Note": f"손절폭 {sl}%"
                        })
                        tg_send(f"🩸 손절: {sym} ({roi_pct:.2f}%) / 연속손실 {state['consec_losses']}")
                        set_cooldown(state, sym, cfg["cooldown_minutes"])
                        state["trades"].pop(sym, None)
                        save_runtime_state(state)

                    elif roi_pct >= tp:
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            ex.create_market_order(sym, close_side, contracts)
                        except:
                            pass

                        # 승리면 연속손실 리셋
                        state["consec_losses"] = 0

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "TP(익절)",
                            "Side": side,
                            "Qty": contracts,
                            "Price": mark,
                            "PnL_USDT": "",
                            "PnL_Pct": f"{roi_pct:.2f}",
                            "Note": f"익절폭 {tp}%"
                        })
                        tg_send(f"🎉 익절: {sym} (+{roi_pct:.2f}%)")
                        set_cooldown(state, sym, cfg["cooldown_minutes"])
                        state["trades"].pop(sym, None)
                        save_runtime_state(state)

            # 2) 신규 진입(순환매: 5개 중 2개만)
            active_positions = get_active_positions(ex, TARGET_COINS)
            if len(active_positions) < int(cfg["max_positions"]):
                rotation = pick_rotation_symbols(ex, TARGET_COINS, top_n=min(2, len(TARGET_COINS)))

                for sym in rotation:
                    if len(get_active_positions(ex, TARGET_COINS)) >= int(cfg["max_positions"]):
                        break

                    if in_cooldown(state, sym):
                        continue

                    # 이미 포지션 있으면 패스
                    already = [p for p in get_active_positions(ex, [sym]) if safe_float(p.get("contracts", 0)) > 0]
                    if already:
                        continue

                    try:
                        ohlcv = ex.fetch_ohlcv(sym, "5m", limit=200)
                        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                        df["time"] = pd.to_datetime(df["time"], unit="ms")
                        df, status, last = calc_indicators(df, cfg)
                        if last is None:
                            continue

                        # 횡보장 필터(대충) - 너무 애매하면 진입 금지
                        if cfg["use_adx"] and last.get("ADX", 0) < 18 and cfg["use_rsi"]:
                            # RSI 중립 + ADX 약하면 관망
                            if 35 <= last.get("RSI", 50) <= 65:
                                continue

                        ai = generate_ai_strategy(sym, df, status, cfg)
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))

                        # 확신도 컷(보수적으로)
                        required_conf = 85 if len(active_positions) >= 1 else 80
                        if decision not in ["buy", "sell"] or conf < required_conf:
                            continue

                        lev = int(ai["leverage"])
                        sl = float(ai["sl_gap"])
                        tp = float(ai["tp_gap"])

                        # 레버리지 설정
                        try:
                            ex.set_leverage(lev, sym)
                        except:
                            pass

                        # 수량 계산(잔고 * percentage)
                        bal = ex.fetch_balance({"type": "swap"})
                        free_usdt = safe_float(bal["USDT"]["free"])
                        use_usdt = free_usdt * (float(ai["percentage"]) / 100.0)
                        price = float(last["close"])
                        qty = (use_usdt * lev) / price if price > 0 else 0
                        qty = float(ex.amount_to_precision(sym, qty))

                        if qty <= 0:
                            continue

                        # 진입
                        ex.create_market_order(sym, decision, qty)

                        # 런타임 trade meta 저장
                        state.setdefault("trades", {})[sym] = {
                            "side": "long" if decision == "buy" else "short",
                            "qty": qty,
                            "leverage": lev,
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sl_gap": sl,
                            "tp_gap": tp,
                            "tp1_gap": cfg["tp1_gap"],
                            "tp1_size": cfg["tp1_size"],
                            "tp1_done": False,
                            "be_price": None
                        }
                        save_runtime_state(state)

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym,
                            "Event": "ENTRY(진입)",
                            "Side": state["trades"][sym]["side"],
                            "Qty": qty,
                            "Price": price,
                            "PnL_USDT": "",
                            "PnL_Pct": "",
                            "Note": ai.get("reason_simple", "")
                        })

                        tg_send(
                            f"🎯 진입: {sym}\n"
                            f"- 방향: {state['trades'][sym]['side']} (conf {conf}%)\n"
                            f"- 레버리지: x{lev}\n"
                            f"- 목표: TP {tp}% / SL {sl}%\n"
                            f"- TP1: +{cfg['tp1_gap']}%에 {cfg['tp1_size']}% 정리\n"
                            f"- 이유: {ai.get('reason_simple','')}"
                        )

                        time.sleep(5)

                    except Exception as e:
                        # 코인 하나 실패해도 계속
                        pass

            # 3) 정기 생존보고
            if time.time() - last_report > REPORT_INTERVAL:
                try:
                    bal = ex.fetch_balance({"type": "swap"})
                    eq = safe_float(bal["USDT"]["total"])
                    tg_send(f"💤 생존신고: 총자산 ${eq:,.2f} / 연속손실 {state.get('consec_losses',0)}")
                except:
                    pass
                last_report = time.time()

            time.sleep(1)

        except Exception:
            time.sleep(5)


# =========================================================
# ✅ 사이드바(설정)
# =========================================================
st.sidebar.title("🛠️ AI 에이전트 제어판")

# OpenAI 키(옵션: UI 저장)
if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

st.sidebar.divider()
config["auto_trade"] = st.sidebar.checkbox("🤖 24시간 자동매매 ON", value=config.get("auto_trade", False))
config["max_positions"] = st.sidebar.slider("동시 포지션 수(추천 2)", 1, 5, int(config.get("max_positions", 2)))
config["leverage"] = st.sidebar.slider("기본 레버리지(참고)", 1, 20, int(config.get("leverage", 5)))
config["order_usdt"] = st.sidebar.number_input("기본 주문금액(참고)", 10.0, 100000.0, float(config.get("order_usdt", 100.0)))

st.sidebar.divider()
st.sidebar.subheader("💰 수익실현(꾸준히)")
config["tp1_gap"] = st.sidebar.number_input("TP1(부분익절) 트리거(%)", 0.1, 5.0, float(config.get("tp1_gap", 0.5)), step=0.1)
config["tp1_size"] = st.sidebar.slider("TP1 청산 비율(%)", 10, 80, int(config.get("tp1_size", 30)))
config["move_sl_to_be"] = st.sidebar.checkbox("TP1 후 본절 방어", value=config.get("move_sl_to_be", True))

st.sidebar.divider()
st.sidebar.subheader("🛡️ 안전장치")
config["min_sl_gap"] = st.sidebar.number_input("최소 손절폭(%)", 1.0, 10.0, float(config.get("min_sl_gap", 2.5)), step=0.1)
config["min_rr"] = st.sidebar.number_input("최소 손익비(대략)", 1.0, 5.0, float(config.get("min_rr", 1.8)), step=0.1)
config["cooldown_minutes"] = st.sidebar.slider("코인별 쿨다운(분)", 0, 120, int(config.get("cooldown_minutes", 15)))
config["max_consec_losses"] = st.sidebar.slider("연속손실 제한", 1, 10, int(config.get("max_consec_losses", 3)))
config["pause_minutes"] = st.sidebar.slider("연속손실 시 정지(분)", 5, 240, int(config.get("pause_minutes", 60)))

st.sidebar.divider()
st.sidebar.subheader("📊 10종 보조지표 ON/OFF")
config["use_rsi"] = st.sidebar.checkbox("RSI", value=config.get("use_rsi", True))
config["use_bb"] = st.sidebar.checkbox("볼린저밴드", value=config.get("use_bb", True))
config["use_ma"] = st.sidebar.checkbox("이평(MA)", value=config.get("use_ma", True))
config["use_macd"] = st.sidebar.checkbox("MACD", value=config.get("use_macd", True))
config["use_adx"] = st.sidebar.checkbox("ADX", value=config.get("use_adx", True))
config["use_stoch"] = st.sidebar.checkbox("스토캐스틱", value=config.get("use_stoch", True))
config["use_cci"] = st.sidebar.checkbox("CCI", value=config.get("use_cci", True))
config["use_mfi"] = st.sidebar.checkbox("MFI", value=config.get("use_mfi", True))
config["use_willr"] = st.sidebar.checkbox("Williams %R", value=config.get("use_willr", True))
config["use_vol"] = st.sidebar.checkbox("거래량", value=config.get("use_vol", True))

# 저장
save_settings(config)

# =========================================================
# ✅ 자동매매 스레드(중복 실행 방지)
# =========================================================
found = any(t.name == "TG_Thread" for t in threading.enumerate())
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_Thread")
    add_script_run_ctx(t)
    t.start()

# =========================================================
# 🧾 잔고/포지션(사이드바 + 메인 요약)
# =========================================================
def fetch_wallet_and_positions():
    bal = exchange.fetch_balance({"type": "swap"})
    usdt_free = safe_float(bal["USDT"]["free"])
    usdt_total = safe_float(bal["USDT"]["total"])
    ps = get_active_positions(exchange, TARGET_COINS)
    return usdt_free, usdt_total, ps

usdt_free, usdt_total, active_positions = (0.0, 0.0, [])
try:
    usdt_free, usdt_total, active_positions = fetch_wallet_and_positions()
except:
    pass

with st.sidebar:
    st.divider()
    st.header("내 지갑 현황 (Wallet)")
    st.metric("총 자산(USDT)", f"${usdt_total:,.2f}")
    st.metric("주문 가능", f"${usdt_free:,.2f}")

    st.divider()
    st.subheader("보유 포지션")
    if active_positions:
        for p in active_positions:
            sym = p.get("symbol", "")
            side = p.get("side", "long")
            lev = safe_float(p.get("leverage", 0))
            entry = safe_float(p.get("entryPrice", 0))
            mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0))
            roi = safe_float(p.get("percentage", None))
            if roi is None:
                roi = compute_roi_pct(side, entry, mark, lev)
            st.info(f"**{sym}**  |  {'🟢 Long' if side=='long' else '🔴 Short'} x{lev}\n\n"
                    f"ROI: **{roi:.2f}%**")
    else:
        st.caption("현재 무포지션(관망 중)")

# =========================================================
# 🎯 메인 화면: 상단 요약 + 트레이딩뷰 차트 + 10종지표 + 탭
# =========================================================
st.title("📌 비트겟 AI 워뇨띠 에이전트")

top1, top2, top3, top4 = st.columns(4)
top1.metric("총자산(USDT)", f"${usdt_total:,.2f}")
top2.metric("주문가능(USDT)", f"${usdt_free:,.2f}")
top3.metric("포지션 수", f"{len(active_positions)} / {config['max_positions']}")
top4.metric("자동매매", "🟢 ON" if config["auto_trade"] else "🔴 OFF")

st.divider()

# 코인 선택(메인)
symbol = st.selectbox("차트 코인 선택", TARGET_COINS, index=0)
timeframe = st.selectbox("타임프레임", ["1m","5m","15m","1h","4h","1d"], index=1)

# =========================================================
# 📌 트레이딩뷰 차트 임베드(메인)
# - Bitget 심볼이 TradingView에서 다를 수 있어 안정적으로 BINANCE 기준 표시
# =========================================================
def tv_symbol_from_bitget(sym: str):
    base = sym.split("/")[0].replace(":USDT","")
    return f"BINANCE:{base}USDT"

tv_symbol = tv_symbol_from_bitget(symbol)

tv_html = f"""
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container" style="height:600px; width:100%;">
  <div id="tradingview_chart" style="height:600px; width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "autosize": true,
    "symbol": "{tv_symbol}",
    "interval": "{timeframe}",
    "timezone": "Asia/Seoul",
    "theme": "light",
    "style": "1",
    "locale": "ko",
    "withdateranges": true,
    "hide_side_toolbar": false,
    "allow_symbol_change": true,
    "details": true,
    "hotlist": false,
    "calendar": false,
    "container_id": "tradingview_chart"
  }}
  );
  </script>
</div>
<!-- TradingView Widget END -->
"""
components.html(tv_html, height=620, scrolling=True)

st.divider()

# =========================================================
# 📊 데이터 로드 + 10종 지표판
# =========================================================
df = None
status = {}
last = None

try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=250)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df, status, last = calc_indicators(df, config)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")

if last is None:
    st.warning("⏳ 지표 계산을 위한 데이터가 부족합니다. (타임프레임/코인 바꿔보세요)")
    st.stop()

st.subheader("🚦 10종 보조지표 종합 상태판")
long_score, short_score = score_signals(status)

judge = "⚪ 관망"
if long_score >= short_score + 2:
    judge = "🟢 매수 우위"
elif short_score >= long_score + 2:
    judge = "🔴 매도 우위"

c1, c2, c3, c4 = st.columns(4)
c1.metric("현재가", f"{float(last['close']):,.4f}")
c2.metric("롱 점수", f"{long_score}")
c3.metric("숏 점수", f"{short_score}")
c4.metric("종합", judge)

with st.expander("지표 상세 보기"):
    st.json(status)

# =========================================================
# 📌 탭 구성: t1~t4
# =========================================================
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

# ---------------- t1 ----------------
with t1:
    st.subheader("🧠 AI 분석(쉽게 설명 포함)")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔍 현재 코인 AI 분석"):
            with st.spinner("AI가 손익비/부분익절 포함해서 판단 중..."):
                ai = generate_ai_strategy(symbol, df, status, config)
                decision = ai.get("decision", "hold").upper()
                conf = ai.get("confidence", 0)

                if decision == "BUY":
                    st.success(f"결론: 🟢 BUY (확신도 {conf}%)")
                elif decision == "SELL":
                    st.error(f"결론: 🔴 SELL (확신도 {conf}%)")
                else:
                    st.warning(f"결론: ⚪ HOLD (확신도 {conf}%)")

                st.info("✅ 쉬운 설명\n\n" + ai.get("reason_simple", ""))
                with st.expander("🔍 상세 근거(조금 더 기술적)"):
                    st.write(ai.get("reason_detail", ""))

                st.write("📌 제안값")
                st.write({
                    "leverage": ai.get("leverage"),
                    "percentage(%)": ai.get("percentage"),
                    "sl_gap(%)": ai.get("sl_gap"),
                    "tp_gap(%)": ai.get("tp_gap"),
                    "tp1_gap(%)": ai.get("tp1_gap"),
                    "tp1_size(%)": ai.get("tp1_size"),
                })

    with colB:
        st.caption("자동매매는 사이드바에서 ON/OFF 됩니다. (텔레그램 연동은 스레드에서 관리)")

# ---------------- t2 ----------------
with t2:
    st.subheader("⚡ 수동 주문(뼈대)")
    st.caption("원하면 여기 수동진입/청산도 Bitget 주문까지 실제 구현해줄게.")
    amt = st.number_input("주문금액(USDT)", 10.0, 100000.0, float(config["order_usdt"]))
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입(미구현)"):
        st.info("미구현")
    if b2.button("🔴 숏 진입(미구현)"):
        st.info("미구현")
    if b3.button("🚫 포지션 종료(미구현)"):
        st.info("미구현")

# ---------------- t3 ----------------
with t3:
    st.subheader("📅 경제 캘린더(한글)")
    st.caption("중요 일정은 변동성이 커질 수 있어요. (TradingView 경제캘린더 위젯 사용) :contentReference[oaicite:1]{index=1}")

    st.info("💡 쉽게 보기\n"
            "- **중요도 높은 발표(금리/물가/CPI/고용)** 전후로 휩쏘(급등락)가 잘 나와요.\n"
            "- 그래서 자동매매는 **‘확신도 높은 자리’만** 들어가게 설계했고, **TP1(부분익절) + 본절방어**로 수익을 자주 잠그는 구조예요.\n"
            "TradingView도 경제캘린더를 통해 주요 이벤트를 추적하라고 안내합니다. :contentReference[oaicite:2]{index=2}")

    econ_html = """
<!-- TradingView Widget BEGIN -->
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
<!-- TradingView Widget END -->
"""
    components.html(econ_html, height=620, scrolling=True)

# ---------------- t4 ----------------
with t4:
    st.subheader("📜 매매일지(런타임 상태 + 로그)")

    state = load_runtime_state()

    # 상단 상태 요약
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("날짜", state.get("date", ""))
    c2.metric("시작자산", f"{safe_float(state.get('day_start_equity',0)):.2f} USDT")
    c3.metric("연속손실", str(state.get("consec_losses", 0)))
    pause_until = int(state.get("pause_until", 0) or 0)
    pause_txt = "없음" if time.time() >= pause_until else datetime.fromtimestamp(pause_until).strftime("%m-%d %H:%M")
    c4.metric("자동정지", pause_txt)

    st.divider()

    # 런타임 trades/cooldowns 보기
    colL, colR = st.columns(2)
    with colL:
        st.markdown("### 🧠 runtime_state.json (현재 상태)")
        with st.expander("원본 JSON 보기"):
            st.json(state)

    with colR:
        st.markdown("### 🔥 현재 트레이드 메타(trades)")
        trades = state.get("trades", {})
        if trades:
            st.dataframe(pd.DataFrame([{"symbol": k, **v} for k, v in trades.items()]), use_container_width=True)
        else:
            st.caption("현재 저장된 트레이드 메타가 없습니다.")

        st.markdown("### ⏱️ 쿨다운(cooldowns)")
        cds = state.get("cooldowns", {})
        if cds:
            rows = []
            now = time.time()
            for k, v in cds.items():
                left = int(max(0, v - now))
                rows.append({"symbol": k, "cooldown_until": datetime.fromtimestamp(v).strftime("%m-%d %H:%M:%S"), "sec_left": left})
            st.dataframe(pd.DataFrame(rows).sort_values("sec_left"), use_container_width=True)
        else:
            st.caption("쿨다운 없음")

    st.divider()

    # trade_log.csv 보기/다운로드
    st.markdown("### 📄 trade_log.csv (체결/청산 로그)")
    if os.path.exists(TRADE_LOG_FILE):
        try:
            log_df = pd.read_csv(TRADE_LOG_FILE)
            if "Time" in log_df.columns:
                log_df = log_df.sort_values("Time", ascending=False)
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            csv = log_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")
        except Exception as e:
            st.error(f"로그 읽기 실패: {e}")
    else:
        st.caption("아직 trade_log.csv가 없습니다. (진입/청산이 발생하면 자동 생성)")

    st.divider()
    if st.button("🧪 테스트 로그 1줄 추가"):
        append_trade_log({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": "BTC/TEST",
            "Event": "TEST",
            "Side": "long",
            "Qty": 0.001,
            "Price": 12345,
            "PnL_USDT": "",
            "PnL_Pct": "",
            "Note": "로그 출력 확인"
        })
        st.success("추가 완료! 위 표 새로고침하면 보입니다.")
