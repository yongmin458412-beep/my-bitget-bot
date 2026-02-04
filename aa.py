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
st.set_page_config(layout="wide", page_title="Bitget AI Bot Control Panel")

IS_SANDBOX = True  # 실전이면 False로 바꾸기
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

# =========================================================
# 설정 관리
# =========================================================
def load_settings():
    default = {
        # 기본
        "openai_api_key": "",
        "auto_trade": False,
        "max_positions": 2,

        # ✅ AI가 매번 자동 적용할지
        "ai_auto_apply_risk": True,

        # ✅ 사용자(제어판) 보험 캡
        "cap_max_leverage": 10,
        "cap_max_risk_pct": 20.0,
        "cap_min_sl_gap": 2.5,
        "cap_min_rr": 1.8,

        # (AI 자동 적용 OFF일 때 고정값)
        "fixed_leverage": 5,
        "fixed_risk_pct": 10.0,

        # ✅ 수익실현 구조
        "tp1_gap": 0.5,     # +0.5%에 부분익절
        "tp1_size": 30,     # 30% 청산
        "move_sl_to_be": True,  # TP1 후 본절 방어

        "use_tp2": True,
        "tp2_gap": 2.0,
        "tp2_size": 30,

        "use_trailing": True,
        "trail_start": 1.2,  # +1.2% 이상부터 트레일링 시작
        "trail_gap": 0.6,    # 최고점 대비 -0.6% 되돌림이면 청산

        # ✅ 연속손실 제한
        "cooldown_minutes": 15,
        "max_consec_losses": 3,
        "pause_minutes": 60,

        # ✅ 호출 주기(멈춤 방지)
        "manage_interval_sec": 2,
        "entry_scan_interval_sec": 12,

        # ✅ 뉴스 회피 (ForexFactory json 기반)
        "avoid_news": True,
        "news_block_before_min": 15,
        "news_block_after_min": 15,
        "news_currencies": ["USD", "KRW", "EUR", "JPY", "CNY"],
        "news_impact_only_high": True,

        # 텔레그램
        "telegram_enabled": True,
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
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {},
        "tg_offset": 0,
        "last_bot_note": "",
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


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


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
    state["cooldowns"][symbol] = int(time.time() + minutes * 60)
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
    st.error("🚨 Bitget API 키가 secrets.toml에 없습니다. (API_KEY/API_SECRET/API_PASSWORD)")
    st.stop()

openai_client = None
if openai_key:
    try:
        openai_client = OpenAI(api_key=openai_key)
    except:
        openai_client = None


# =========================================================
# Exchange 생성 (UI / 봇 스레드 분리!)
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
    # 원웨이 모드 시도 (실패해도 무시)
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
# 지표 계산(ta 미사용)
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
# AI 전략: 매번 레버/비중/SL/TP 추천 + 쉬운 설명
# =========================================================
def clamp_ai_risk(ai_reco: dict, cfg: dict, atr_pct: float):
    lev = int(ai_reco.get("leverage", 5))
    risk = float(ai_reco.get("risk_pct", cfg.get("fixed_risk_pct", 10)))
    sl = float(ai_reco.get("sl_gap", cfg["cap_min_sl_gap"]))
    tp = float(ai_reco.get("tp_gap", sl * cfg["cap_min_rr"]))

    # 최소 보장
    sl = max(sl, float(cfg["cap_min_sl_gap"]))

    # ATR 기반 SL 하한(손절 너무 잦은 문제 완화)
    if atr_pct and atr_pct > 0:
        sl = max(sl, atr_pct * 1.2)  # ATR%가 큰 장이면 SL 넓힘

    tp = max(tp, sl * float(cfg["cap_min_rr"]))

    # 캡 적용
    lev = min(max(lev, 1), int(cfg["cap_max_leverage"]))
    risk = min(max(risk, 1.0), float(cfg["cap_max_risk_pct"]))

    # 변동성 큰 장이면 레버 상한 자동 하향
    if atr_pct and atr_pct > 0:
        vol_cap = max(2, int(20 / atr_pct))  # ATR%↑ => cap↓
        lev = min(lev, vol_cap)

    rr = tp / sl if sl > 0 else 0
    return {"leverage": lev, "risk_pct": risk, "sl_gap": sl, "tp_gap": tp, "rr": rr}


def generate_ai_strategy(symbol: str, df: pd.DataFrame, pack: dict, cfg: dict):
    # OpenAI 비활성 시: 규칙 기반으로만
    last = pack["last"]
    prev = pack["prev"]
    status = pack["status"]
    long_score, short_score = score_signals(status)

    atr_pct = float(last.get("ATR_PCT", 0))
    atr_sl_floor = max(cfg["cap_min_sl_gap"], atr_pct * 1.2)
    atr_tp_floor = max(atr_sl_floor * cfg["cap_min_rr"], atr_pct * 2.0)

    if openai_client is None:
        decision = "hold"
        conf = 0
        if long_score >= short_score + 2 and status.get("ADX") != "💤 횡보장":
            decision, conf = "buy", 78
        elif short_score >= long_score + 2 and status.get("ADX") != "💤 횡보장":
            decision, conf = "sell", 78

        return {
            "decision": decision,
            "confidence": conf,
            "ai_reco": {"leverage": 5, "risk_pct": cfg.get("fixed_risk_pct", 10), "sl_gap": atr_sl_floor, "tp_gap": atr_tp_floor},
            "focus_indicators": ["RSI", "ADX", "ATR%"],
            "simple": "AI키가 없어서 기본 규칙으로만 판단했어요.",
            "detail": f"상태: {status}"
        }

    snapshot = {
        "price": float(last["close"]),
        "ATR_PCT": atr_pct,
        "RSI_prev": float(prev["RSI"]), "RSI": float(last["RSI"]),
        "BB_upper": float(last["BB_upper"]), "BB_lower": float(last["BB_lower"]),
        "MA_fast": float(last["MA_fast"]), "MA_slow": float(last["MA_slow"]),
        "MACD": float(last["MACD"]), "MACD_signal": float(last["MACD_signal"]),
        "ADX": float(last["ADX"]), "PDI": float(last["PDI"]), "MDI": float(last["MDI"]),
        "STO_K": float(last["STO_K"]), "STO_D": float(last["STO_D"]),
        "CCI": float(last["CCI"]), "MFI": float(last["MFI"]), "WILLR": float(last["WILLR"]),
        "VOL": float(last["vol"]), "VOL_SMA": float(last["VOL_SMA"]),
        "status": status,
        "vote": {"long_score": long_score, "short_score": short_score},
        "caps": {
            "cap_max_leverage": cfg["cap_max_leverage"],
            "cap_max_risk_pct": cfg["cap_max_risk_pct"],
            "cap_min_sl_gap": cfg["cap_min_sl_gap"],
            "cap_min_rr": cfg["cap_min_rr"],
            "atr_sl_floor": atr_sl_floor,
            "atr_tp_floor": atr_tp_floor,
        }
    }

    system_prompt = f"""
너는 "자동매매 코치"야.
목표: 손절 연타를 줄이고, 수익을 자주 잠그고(TP1/TP2), 좋은 타이밍만 들어가.

규칙:
- 확실하지 않으면 hold.
- 손절폭(sl_gap)은 최소 {cfg['cap_min_sl_gap']}% 이상.
- 손익비는 최소 {cfg['cap_min_rr']} 이상(= tp_gap >= sl_gap*RR).
- 레버리지/비중은 추천만 하되, 사용자는 캡을 걸어둘 거야.

출력은 JSON 하나로:
{{
 "decision":"buy/sell/hold",
 "confidence":0~100,
 "ai_reco":{{"leverage":1~20,"risk_pct":1~30,"sl_gap":1.0~12.0,"tp_gap":2.0~30.0}},
 "focus_indicators":["중요 지표 3~5개"],
 "simple":"아주 쉬운 설명 2~4줄",
 "detail":"조금 더 자세한 근거"
}}
"""

    user_prompt = f"""
심볼: {symbol}
지표/상태(JSON): {json.dumps(snapshot, ensure_ascii=False)}
설명은 꼭 쉽게.
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

        out.setdefault("ai_reco", {})
        # AI가 너무 타이트하게 내면 ATR/최소값으로 보정(실제 적용은 clamp에서)
        sl = float(out["ai_reco"].get("sl_gap", atr_sl_floor))
        tp = float(out["ai_reco"].get("tp_gap", atr_tp_floor))
        sl = max(sl, atr_sl_floor, cfg["cap_min_sl_gap"])
        tp = max(tp, sl * cfg["cap_min_rr"], atr_tp_floor)
        out["ai_reco"]["sl_gap"] = float(sl)
        out["ai_reco"]["tp_gap"] = float(tp)

        out.setdefault("focus_indicators", ["RSI", "ADX", "ATR%"])
        out.setdefault("simple", "")
        out.setdefault("detail", "")

        return out
    except Exception as e:
        return {
            "decision": "hold",
            "confidence": 0,
            "ai_reco": {"leverage": 5, "risk_pct": cfg.get("fixed_risk_pct", 10), "sl_gap": atr_sl_floor, "tp_gap": atr_tp_floor},
            "focus_indicators": ["RSI", "ADX", "ATR%"],
            "simple": "AI 호출 오류라서 관망해요.",
            "detail": f"에러: {e}"
        }


# =========================================================
# 순환매: 변동성 큰 N개만 신규진입 후보
# =========================================================
def pick_rotation_symbols(ex, symbols, timeframe="5m", limit=80, top_n=2):
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
# 포지션 조회
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


# =========================================================
# 경제 캘린더(한글) + 뉴스회피
# =========================================================
def fetch_econ_calendar_ko():
    """
    ForexFactory 주간 캘린더 JSON (가끔 막힐 수 있음)
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=7)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list):
            return pd.DataFrame()

        rows = []
        now = datetime.utcnow()

        for ev in data:
            date_s = ev.get("date")
            time_s = ev.get("time") or "00:00"
            if not date_s:
                continue

            # FF는 대체로 UTC 기반(정확 시간 변동 가능) -> '회피 필터'는 안전하게 넉넉히
            dt_s = f"{date_s} {time_s}"
            try:
                dt = datetime.strptime(dt_s, "%Y-%m-%d %H:%M")
            except:
                try:
                    dt = datetime.strptime(date_s, "%Y-%m-%d")
                except:
                    continue

            # 최근~미래 8일 정도만
            if dt < now - timedelta(days=1) or dt > now + timedelta(days=8):
                continue

            impact = (ev.get("impact") or "").lower()
            imp_ko = "높음" if "high" in impact else ("중간" if "medium" in impact else ("낮음" if "low" in impact else ""))
            rows.append({
                "utc_dt": dt,
                "날짜": dt.strftime("%m-%d"),
                "시간(UTC)": time_s,
                "통화": ev.get("currency", ""),
                "중요도": imp_ko,
                "지표": ev.get("title", ""),
                "예상": ev.get("forecast", ""),
                "이전": ev.get("previous", "")
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("utc_dt", ascending=True)

    except:
        return pd.DataFrame()


def is_in_news_block(cfg, cal_df: pd.DataFrame):
    """
    중요한 뉴스 전후 시간대면 신규진입 금지
    """
    if not cfg.get("avoid_news", True):
        return False, None

    if cal_df is None or cal_df.empty:
        return False, None

    now = datetime.utcnow()
    before = int(cfg.get("news_block_before_min", 15))
    after = int(cfg.get("news_block_after_min", 15))

    cur_list = set(cfg.get("news_currencies", ["USD"]))
    high_only = bool(cfg.get("news_impact_only_high", True))

    for _, row in cal_df.iterrows():
        cur = str(row.get("통화", "")).upper()
        imp = str(row.get("중요도", ""))
        if cur and cur not in cur_list:
            continue
        if high_only and imp != "높음":
            continue

        dt = row.get("utc_dt", None)
        if not isinstance(dt, datetime):
            continue

        if dt - timedelta(minutes=before) <= now <= dt + timedelta(minutes=after):
            # 회피 중
            title = row.get("지표", "")
            return True, f"{cur} {imp} 뉴스({title}) 전후"

    return False, None


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
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            timeout=6
        )
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


TG_MENU = {
    "inline_keyboard": [
        [{"text": "📊 브리핑(2개)", "callback_data": "brief"},
         {"text": "🌍 전체스캔(5개)", "callback_data": "scan"}],
        [{"text": "💰 잔고", "callback_data": "balance"},
         {"text": "📌 포지션", "callback_data": "pos"}],
        [{"text": "🤖 ON/OFF", "callback_data": "toggle"},
         {"text": "📰 뉴스(한글)", "callback_data": "news"}],
        [{"text": "🛑 전량청산", "callback_data": "close_all"},
         {"text": "🧾 상태", "callback_data": "status"}],
    ]
}


# =========================================================
# 핵심: 봇 스레드 (UI exchange와 분리!)
# =========================================================
def telegram_bot_thread():
    bot_ex = create_exchange()
    state = load_runtime_state()

    # 캘린더 캐시
    cal_cache = {"t": 0, "df": pd.DataFrame()}

    def get_calendar_cached():
        now = time.time()
        # 10분마다 갱신
        if now - cal_cache["t"] > 600:
            cal_cache["df"] = fetch_econ_calendar_ko()
            cal_cache["t"] = now
        return cal_cache["df"]

    # 시작 메시지
    if config.get("telegram_enabled", True):
        tg_send(tg_token, tg_id, "🚀 봇 가동 시작!\n(메뉴로 확인/조작 가능)", reply_markup=TG_MENU)

    last_manage = 0
    last_entry_scan = 0
    last_report = 0
    REPORT_INTERVAL = 900  # 15분

    while True:
        try:
            cfg = load_settings()
            state = load_runtime_state()

            # 잔고로 데일리 리셋
            try:
                bal = bot_ex.fetch_balance({"type": "swap"})
                equity = safe_float(bal["USDT"]["total"])
            except:
                equity = safe_float(state.get("day_start_equity", 0.0))
            maybe_roll_daily_state(state, equity)

            # 텔레그램 폴링 처리(메뉴 버튼)
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

                            # --- 메뉴 처리 ---
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
                                    msg = "📌 포지션 없음(관망)"
                                else:
                                    lines = ["📌 포지션 현황"]
                                    for p in ps:
                                        sym = p.get("symbol", "")
                                        side = p.get("side", "")
                                        roi = safe_float(p.get("percentage", 0))
                                        lev = p.get("leverage", "?")
                                        lines.append(f"- {sym} | {side} x{lev} | ROI {roi:.2f}%")
                                    msg = "\n".join(lines)
                                tg_send(tg_token, cid, msg, reply_markup=TG_MENU)

                            elif data == "toggle":
                                cfg2 = load_settings()
                                cfg2["auto_trade"] = not cfg2.get("auto_trade", False)
                                save_settings(cfg2)
                                tg_send(tg_token, cid, f"🤖 자동매매 {'ON' if cfg2['auto_trade'] else 'OFF'}", reply_markup=TG_MENU)

                            elif data == "status":
                                note = state.get("last_bot_note", "")
                                pause = "ON" if is_paused(state) else "OFF"
                                consec = state.get("consec_losses", 0)
                                tg_send(
                                    tg_token, cid,
                                    f"🧾 상태\n- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}\n- 정지(pause): {pause}\n- 연속손실: {consec}\n- 메모: {note}",
                                    reply_markup=TG_MENU
                                )

                            elif data == "news":
                                cal_df = get_calendar_cached()
                                if cal_df is None or cal_df.empty:
                                    tg_send(tg_token, cid, "📰 캘린더 데이터를 못 불러왔어요(사이트 차단/지연 가능).", reply_markup=TG_MENU)
                                else:
                                    # 중요도 '높음' 위주로 10개
                                    df2 = cal_df.copy()
                                    df2 = df2[df2["중요도"].isin(["높음", "중간"])]
                                    df2 = df2.head(10)
                                    lines = ["📰 경제 캘린더(UTC 기준)\n(중요도 높은 일정은 자동 회피 가능)"]
                                    for _, r in df2.iterrows():
                                        lines.append(f"- {r['날짜']} {r['시간(UTC)']} | {r['통화']} | {r['중요도']} | {r['지표']}")
                                    tg_send(tg_token, cid, "\n".join(lines), reply_markup=TG_MENU)

                            elif data in ["brief", "scan"]:
                                cal_df = get_calendar_cached()
                                blocked, reason = is_in_news_block(cfg, cal_df)
                                block_txt = f"\n🛑 현재 뉴스회피 구간: {reason}" if blocked else ""

                                syms = pick_rotation_symbols(bot_ex, TARGET_COINS, top_n=2) if data == "brief" else TARGET_COINS
                                lines = [f"📊 {'브리핑(2개)' if data=='brief' else '전체스캔(5개)'}{block_txt}"]
                                for sym in syms:
                                    try:
                                        ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                                        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                                        df, pack = calc_indicators(df)
                                        if pack is None:
                                            continue
                                        ai = generate_ai_strategy(sym, df, pack, cfg)
                                        applied = clamp_ai_risk(ai.get("ai_reco", {}), cfg, float(pack["last"].get("ATR_PCT", 0)))

                                        lines.append(
                                            f"\n[{sym}]\n"
                                            f"- 결론: {ai.get('decision','hold').upper()} (conf {ai.get('confidence',0)}%)\n"
                                            f"- 쉬운설명: {ai.get('simple','')}\n"
                                            f"- 적용(캡 반영): x{applied['leverage']} | 비중 {applied['risk_pct']:.1f}% | SL {applied['sl_gap']:.2f}% | TP {applied['tp_gap']:.2f}% (RR {applied['rr']:.2f})"
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
                                    close_side = "sell" if side == "long" else "buy"
                                    try:
                                        bot_ex.create_market_order(sym, close_side, contracts, params={"reduceOnly": True})
                                        closed += 1
                                    except:
                                        pass
                                tg_send(tg_token, cid, f"🛑 전량 청산 요청 완료 (대상 {closed}개)", reply_markup=TG_MENU)

                            tg_answer(tg_token, cb_id)

                except:
                    pass

            # 자동매매 OFF면 매매만 스킵(텔레그램 메뉴는 계속 됨)
            if not cfg.get("auto_trade", False):
                time.sleep(0.5)
                continue

            if is_paused(state):
                state["last_bot_note"] = "연속손실로 일시 정지 중"
                save_runtime_state(state)
                time.sleep(1.5)
                continue

            ts = time.time()

            # 1) 포지션 관리
            if ts - last_manage >= int(cfg["manage_interval_sec"]):
                last_manage = ts

                active_positions = get_active_positions(bot_ex, TARGET_COINS)

                for p in active_positions:
                    sym = p.get("symbol")
                    side = p.get("side", "long")  # long/short
                    contracts = safe_float(p.get("contracts", 0))
                    entry = safe_float(p.get("entryPrice", 0))
                    mark = safe_float(p.get("markPrice", 0)) or safe_float(p.get("last", 0))
                    roi = safe_float(p.get("percentage", 0))  # %로 들어옴

                    meta = state.get("trades", {}).get(sym, {})
                    sl = float(meta.get("sl_gap", cfg["cap_min_sl_gap"]))
                    tp = float(meta.get("tp_gap", sl * cfg["cap_min_rr"]))

                    tp1_gap = float(meta.get("tp1_gap", cfg["tp1_gap"]))
                    tp1_size = int(meta.get("tp1_size", cfg["tp1_size"]))
                    tp1_done = bool(meta.get("tp1_done", False))

                    use_tp2 = bool(meta.get("use_tp2", cfg.get("use_tp2", True)))
                    tp2_gap = float(meta.get("tp2_gap", cfg.get("tp2_gap", 2.0)))
                    tp2_size = int(meta.get("tp2_size", cfg.get("tp2_size", 30)))
                    tp2_done = bool(meta.get("tp2_done", False))

                    use_trailing = bool(meta.get("use_trailing", cfg.get("use_trailing", True)))
                    trail_start = float(meta.get("trail_start", cfg.get("trail_start", 1.2)))
                    trail_gap = float(meta.get("trail_gap", cfg.get("trail_gap", 0.6)))

                    # 최고가/최저가 갱신(트레일링용)
                    best_price = meta.get("best_price", None)
                    if best_price is None:
                        best_price = mark
                    if side == "long":
                        best_price = max(best_price, mark)
                    else:
                        best_price = min(best_price, mark)

                    state.setdefault("trades", {}).setdefault(sym, {})
                    state["trades"][sym]["best_price"] = best_price
                    save_runtime_state(state)

                    # TP1 부분익절
                    if (not tp1_done) and roi >= tp1_gap and contracts > 0:
                        close_qty = float(bot_ex.amount_to_precision(sym, contracts * (tp1_size / 100.0)))
                        if close_qty > 0:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, close_qty, params={"reduceOnly": True})
                            except:
                                pass

                            state["trades"][sym]["tp1_done"] = True
                            if cfg.get("move_sl_to_be", True):
                                state["trades"][sym]["be_price"] = entry
                            save_runtime_state(state)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym, "Event": "TP1(부분익절)", "Side": side,
                                "Qty": close_qty, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                                "Note": "수익 잠금"
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP1 부분익절: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                    # TP2 부분익절
                    if use_tp2 and (not tp2_done) and roi >= tp2_gap and contracts > 0:
                        close_qty = float(bot_ex.amount_to_precision(sym, contracts * (tp2_size / 100.0)))
                        if close_qty > 0:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, close_qty, params={"reduceOnly": True})
                            except:
                                pass

                            state["trades"][sym]["tp2_done"] = True
                            save_runtime_state(state)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym, "Event": "TP2(부분익절)", "Side": side,
                                "Qty": close_qty, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                                "Note": "2차 수익 잠금"
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"✅ TP2 부분익절: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                    # 본절 방어(TP1 후)
                    be_price = meta.get("be_price", None)
                    if be_price and contracts > 0 and roi <= 0.1:
                        close_side = "sell" if side == "long" else "buy"
                        try:
                            bot_ex.create_market_order(sym, close_side, contracts, params={"reduceOnly": True})
                        except:
                            pass

                        append_trade_log({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": sym, "Event": "BE(본절정리)", "Side": side,
                            "Qty": contracts, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                            "Note": "TP1 이후 본절"
                        })
                        if cfg.get("telegram_enabled", True):
                            tg_send(tg_token, tg_id, f"🛡️ 본절 정리: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                        set_cooldown(state, sym, cfg["cooldown_minutes"])
                        state["trades"].pop(sym, None)
                        save_runtime_state(state)
                        continue

                    # 트레일링 청산(가격 기반)
                    if use_trailing and roi >= trail_start and contracts > 0:
                        if side == "long":
                            dd = (best_price - mark) / best_price * 100 if best_price > 0 else 0
                        else:
                            dd = (mark - best_price) / best_price * 100 if best_price > 0 else 0
                        # dd가 trail_gap 이상이면 청산
                        if dd >= trail_gap:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, contracts, params={"reduceOnly": True})
                            except:
                                pass

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym, "Event": "TRAIL(트레일링)", "Side": side,
                                "Qty": contracts, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                                "Note": f"되돌림 {dd:.2f}%"
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"🏁 트레일링 청산: {sym} ({roi:.2f}%)", reply_markup=TG_MENU)

                            set_cooldown(state, sym, cfg["cooldown_minutes"])
                            state["trades"].pop(sym, None)
                            save_runtime_state(state)
                            continue

                    # SL/TP 청산(ROI% 기준)
                    if contracts > 0:
                        if roi <= -abs(sl):
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, contracts, params={"reduceOnly": True})
                            except:
                                pass

                            state["consec_losses"] = int(state.get("consec_losses", 0)) + 1
                            if state["consec_losses"] >= cfg["max_consec_losses"]:
                                state["pause_until"] = int(time.time() + cfg["pause_minutes"] * 60)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym, "Event": "SL(손절)", "Side": side,
                                "Qty": contracts, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                                "Note": f"SL {sl:.2f}% / 연속손실 {state['consec_losses']}"
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"🩸 손절: {sym} ({roi:.2f}%) / 연속손실 {state['consec_losses']}", reply_markup=TG_MENU)

                            set_cooldown(state, sym, cfg["cooldown_minutes"])
                            state["trades"].pop(sym, None)
                            state["last_bot_note"] = "손절 발생"
                            save_runtime_state(state)

                        elif roi >= tp:
                            close_side = "sell" if side == "long" else "buy"
                            try:
                                bot_ex.create_market_order(sym, close_side, contracts, params={"reduceOnly": True})
                            except:
                                pass

                            state["consec_losses"] = 0

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym, "Event": "TP(익절)", "Side": side,
                                "Qty": contracts, "Price": mark, "ROI_Pct": f"{roi:.2f}",
                                "Note": f"TP {tp:.2f}%"
                            })
                            if cfg.get("telegram_enabled", True):
                                tg_send(tg_token, tg_id, f"🎉 익절: {sym} (+{roi:.2f}%)", reply_markup=TG_MENU)

                            set_cooldown(state, sym, cfg["cooldown_minutes"])
                            state["trades"].pop(sym, None)
                            state["last_bot_note"] = "익절 발생"
                            save_runtime_state(state)

            # 2) 신규 진입(스캔 주기 제한)
            if ts - last_entry_scan >= int(cfg["entry_scan_interval_sec"]):
                last_entry_scan = ts

                active_positions = get_active_positions(bot_ex, TARGET_COINS)
                if len(active_positions) < int(cfg["max_positions"]):
                    # 뉴스 회피
                    cal_df = fetch_econ_calendar_ko() if cfg.get("avoid_news", True) else pd.DataFrame()
                    blocked, reason = is_in_news_block(cfg, cal_df)
                    if blocked:
                        state["last_bot_note"] = f"뉴스 회피 중: {reason}"
                        save_runtime_state(state)
                        time.sleep(1.0)
                        continue

                    rotation = pick_rotation_symbols(bot_ex, TARGET_COINS, top_n=min(2, len(TARGET_COINS)))

                    for sym in rotation:
                        # 포지션 수 다시 확인
                        if len(get_active_positions(bot_ex, TARGET_COINS)) >= int(cfg["max_positions"]):
                            break
                        if in_cooldown(state, sym):
                            continue
                        if get_active_positions(bot_ex, [sym]):
                            continue

                        try:
                            ohlcv = bot_ex.fetch_ohlcv(sym, "5m", limit=250)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df, pack = calc_indicators(df)
                            if pack is None:
                                continue

                            # 애매한 횡보 + RSI 중립이면 패스(비용/손절연타 방지)
                            if pack["status"].get("ADX") == "💤 횡보장" and (35 <= pack["last"]["RSI"] <= 65):
                                continue

                            ai = generate_ai_strategy(sym, df, pack, cfg)
                            decision = ai.get("decision", "hold")
                            conf = int(ai.get("confidence", 0))

                            required_conf = 85 if len(active_positions) >= 1 else 80
                            if decision not in ["buy", "sell"] or conf < required_conf:
                                continue

                            atr_pct = float(pack["last"].get("ATR_PCT", 0))

                            # ✅ 적용값 결정: AI 자동 적용 or 고정
                            if cfg.get("ai_auto_apply_risk", True):
                                applied = clamp_ai_risk(ai.get("ai_reco", {}), cfg, atr_pct)
                                lev = applied["leverage"]
                                risk_pct = applied["risk_pct"]
                                sl = applied["sl_gap"]
                                tp = applied["tp_gap"]
                            else:
                                lev = int(cfg.get("fixed_leverage", 5))
                                risk_pct = float(cfg.get("fixed_risk_pct", 10))
                                sl = max(cfg["cap_min_sl_gap"], atr_pct * 1.2)
                                tp = max(sl * cfg["cap_min_rr"], atr_pct * 2.0)

                            # 레버 설정
                            try:
                                bot_ex.set_leverage(int(lev), sym)
                            except:
                                pass

                            # 주문 수량
                            bal = bot_ex.fetch_balance({"type": "swap"})
                            free_usdt = safe_float(bal["USDT"]["free"])
                            use_usdt = free_usdt * (float(risk_pct) / 100.0)
                            price = float(pack["last"]["close"])
                            qty = (use_usdt * float(lev)) / price if price > 0 else 0
                            qty = float(bot_ex.amount_to_precision(sym, qty))
                            if qty <= 0:
                                continue

                            bot_ex.create_market_order(sym, decision, qty)

                            side_txt = "long" if decision == "buy" else "short"

                            # 상태 저장
                            state.setdefault("trades", {})[sym] = {
                                "side": side_txt,
                                "qty": qty,
                                "ai_reco": ai.get("ai_reco", {}),
                                "applied_leverage": lev,
                                "applied_risk_pct": risk_pct,
                                "sl_gap": float(sl),
                                "tp_gap": float(tp),
                                "rr": float(tp) / float(sl) if float(sl) else 0,
                                "atr_pct": atr_pct,

                                "tp1_gap": cfg["tp1_gap"],
                                "tp1_size": cfg["tp1_size"],
                                "tp1_done": False,
                                "be_price": None,

                                "use_tp2": cfg.get("use_tp2", True),
                                "tp2_gap": cfg.get("tp2_gap", 2.0),
                                "tp2_size": cfg.get("tp2_size", 30),
                                "tp2_done": False,

                                "use_trailing": cfg.get("use_trailing", True),
                                "trail_start": cfg.get("trail_start", 1.2),
                                "trail_gap": cfg.get("trail_gap", 0.6),
                                "best_price": price,

                                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "simple": ai.get("simple", ""),
                                "focus": ai.get("focus_indicators", []),
                            }
                            state["last_bot_note"] = f"진입: {sym} {side_txt}"
                            save_runtime_state(state)

                            append_trade_log({
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Symbol": sym,
                                "Event": "ENTRY(진입)",
                                "Side": side_txt,
                                "Qty": qty,
                                "Price": price,
                                "ROI_Pct": "",
                                "Note": (ai.get("simple", "")[:90] if ai else "")
                            })

                            if cfg.get("telegram_enabled", True):
                                tg_send(
                                    tg_token, tg_id,
                                    f"🎯 진입: {sym}\n"
                                    f"- 방향: {side_txt.upper()} (conf {conf}%)\n"
                                    f"- 적용: 레버 x{lev} | 비중 {risk_pct:.1f}%\n"
                                    f"- 목표: TP {tp:.2f}% / SL {sl:.2f}% (RR {float(tp)/float(sl):.2f})\n"
                                    f"- TP1: +{cfg['tp1_gap']}%에 {cfg['tp1_size']}% | TP2: +{cfg.get('tp2_gap',2.0)}%에 {cfg.get('tp2_size',30)}%\n"
                                    f"- 트레일링: +{cfg.get('trail_start',1.2)}%부터 되돌림 {cfg.get('trail_gap',0.6)}%\n"
                                    f"- 쉬운설명: {ai.get('simple','')}",
                                    reply_markup=TG_MENU
                                )

                            time.sleep(2)

                        except:
                            continue

            # 3) 생존 신고
            if cfg.get("telegram_enabled", True) and time.time() - last_report > REPORT_INTERVAL:
                try:
                    bal = bot_ex.fetch_balance({"type": "swap"})
                    eq = safe_float(bal["USDT"]["total"])
                    tg_send(tg_token, tg_id, f"💤 생존신고\n총자산: ${eq:,.2f}\n연속손실: {state.get('consec_losses',0)}", reply_markup=TG_MENU)
                except:
                    pass
                last_report = time.time()

            time.sleep(0.5)

        except:
            time.sleep(2)


# =========================================================
# Streamlit: 제어판 UI (텔레그램이 메인)
# =========================================================
st.title("🧩 Bitget AI Bot — 제어판(Streamlit) / 정보수신(Telegram)")

state = load_runtime_state()

# 상단 상태
c1, c2, c3, c4 = st.columns(4)
try:
    bal = exchange.fetch_balance({"type": "swap"})
    usdt_total = safe_float(bal["USDT"]["total"])
    usdt_free = safe_float(bal["USDT"]["free"])
except:
    usdt_total, usdt_free = 0.0, 0.0

active_positions_ui = get_active_positions(exchange, TARGET_COINS)

c1.metric("총자산(USDT)", f"${usdt_total:,.2f}")
c2.metric("주문가능(USDT)", f"${usdt_free:,.2f}")
c3.metric("보유 포지션", f"{len(active_positions_ui)} / {config.get('max_positions',2)}")
c4.metric("자동매매", "🟢 ON" if config.get("auto_trade") else "🔴 OFF")

st.caption(f"마지막 봇 메모: {state.get('last_bot_note','')}")

st.divider()

# 사이드바: 설정
with st.sidebar:
    st.header("🛠️ 설정(제어판)")
    st.caption("⚠️ 실제 매매 전에는 반드시 데모(샌드박스)로 충분히 테스트하세요.")

    # OpenAI 키 입력(선택)
    if not openai_key:
        k = st.text_input("OpenAI API Key(선택)", type="password")
        if k:
            config["openai_api_key"] = k
            save_settings(config)
            st.success("저장됨. 새로고침/재실행하면 적용됩니다.")

    st.divider()
    config["telegram_enabled"] = st.checkbox("텔레그램 알림/메뉴 사용", value=config.get("telegram_enabled", True))
    config["auto_trade"] = st.checkbox("🤖 자동매매 ON", value=config.get("auto_trade", False))
    config["max_positions"] = st.slider("동시 포지션 수", 1, 5, int(config.get("max_positions", 2)))

    st.divider()
    st.subheader("🧠 AI가 매번 자동 적용")
    config["ai_auto_apply_risk"] = st.checkbox("AI가 레버/비중/손익비 자동 적용", value=config.get("ai_auto_apply_risk", True))

    st.caption("👇 보험(캡): AI가 뭐라 해도 이 범위를 넘지 못함")
    config["cap_max_leverage"] = st.slider("최대 레버리지 캡", 1, 20, int(config.get("cap_max_leverage", 10)))
    config["cap_max_risk_pct"] = st.slider("최대 비중 캡(%)", 1.0, 50.0, float(config.get("cap_max_risk_pct", 20.0)))
    config["cap_min_sl_gap"] = st.number_input("최소 손절폭(%)", 0.5, 20.0, float(config.get("cap_min_sl_gap", 2.5)), step=0.1)
    config["cap_min_rr"] = st.number_input("최소 손익비(RR)", 1.0, 5.0, float(config.get("cap_min_rr", 1.8)), step=0.1)

    st.divider()
    st.subheader("🎯 수익실현 구조")
    config["tp1_gap"] = st.number_input("TP1 트리거(+%)", 0.1, 5.0, float(config.get("tp1_gap", 0.5)), step=0.1)
    config["tp1_size"] = st.slider("TP1 청산비율(%)", 10, 80, int(config.get("tp1_size", 30)))
    config["move_sl_to_be"] = st.checkbox("TP1 후 본절 방어", value=config.get("move_sl_to_be", True))

    config["use_tp2"] = st.checkbox("TP2 사용", value=config.get("use_tp2", True))
    config["tp2_gap"] = st.number_input("TP2 트리거(+%)", 0.5, 20.0, float(config.get("tp2_gap", 2.0)), step=0.1)
    config["tp2_size"] = st.slider("TP2 청산비율(%)", 10, 80, int(config.get("tp2_size", 30)))

    config["use_trailing"] = st.checkbox("트레일링 사용", value=config.get("use_trailing", True))
    config["trail_start"] = st.number_input("트레일링 시작(+%)", 0.5, 10.0, float(config.get("trail_start", 1.2)), step=0.1)
    config["trail_gap"] = st.number_input("트레일링 되돌림(%)", 0.2, 10.0, float(config.get("trail_gap", 0.6)), step=0.1)

    st.divider()
    st.subheader("📰 뉴스 회피")
    config["avoid_news"] = st.checkbox("중요 뉴스 전후 진입 금지", value=config.get("avoid_news", True))
    config["news_block_before_min"] = st.slider("뉴스 전(분)", 0, 60, int(config.get("news_block_before_min", 15)))
    config["news_block_after_min"] = st.slider("뉴스 후(분)", 0, 60, int(config.get("news_block_after_min", 15)))
    config["news_impact_only_high"] = st.checkbox("중요도 '높음'만 회피", value=config.get("news_impact_only_high", True))

    st.divider()
    st.subheader("⏱️ 멈춤 방지(호출 주기)")
    config["manage_interval_sec"] = st.slider("포지션 관리 주기(초)", 1, 10, int(config.get("manage_interval_sec", 2)))
    config["entry_scan_interval_sec"] = st.slider("신규 진입 스캔 주기(초)", 5, 60, int(config.get("entry_scan_interval_sec", 12)))

    st.divider()
    st.subheader("🧯 손실 제한")
    config["cooldown_minutes"] = st.slider("코인 쿨다운(분)", 0, 120, int(config.get("cooldown_minutes", 15)))
    config["max_consec_losses"] = st.slider("연속손실 제한", 1, 10, int(config.get("max_consec_losses", 3)))
    config["pause_minutes"] = st.slider("정지 시간(분)", 5, 240, int(config.get("pause_minutes", 60)))

    st.divider()
    if st.button("💾 설정 저장"):
        save_settings(config)
        st.success("저장 완료")

    st.divider()
    st.subheader("🔧 텔레그램 테스트")
    if st.button("📡 텔레그램 메뉴 보내기"):
        tg_send(tg_token, tg_id, "✅ 메뉴를 보냈어요.", reply_markup=TG_MENU)

    st.subheader("🤖 OpenAI 테스트")
    if st.button("OpenAI 연결 테스트"):
        if openai_key:
            try:
                test = OpenAI(api_key=openai_key)
                resp = test.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "테스트: 1+1은?"}],
                    max_tokens=10
                )
                st.success("✅ 성공: " + (resp.choices[0].message.content or "").strip())
            except Exception as e:
                st.error(f"❌ 실패: {e}")
        else:
            st.warning("OPENAI_API_KEY가 없습니다.")


# 봇 스레드 시작 (중복 방지)
if not any(t.name == "TG_Thread" for t in threading.enumerate()):
    t = threading.Thread(target=telegram_bot_thread, daemon=True, name="TG_Thread")
    add_script_run_ctx(t)
    t.start()

# 하단: 상태/로그
st.subheader("📌 현재 포지션(제어판용 요약)")
if active_positions_ui:
    for p in active_positions_ui:
        sym = p.get("symbol", "")
        side = p.get("side", "")
        roi = safe_float(p.get("percentage", 0))
        lev = p.get("leverage", "?")
        st.info(f"**{sym}** | {side} x{lev} | ROI **{roi:.2f}%**")
else:
    st.caption("무포지션")

st.divider()
st.subheader("🧾 runtime_state.json")
with st.expander("원본 보기"):
    st.json(load_runtime_state())

st.divider()
st.subheader("📜 trade_log.csv")
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
    st.caption("아직 trade_log.csv가 없습니다(진입/청산이 발생하면 자동 생성).")

st.divider()
st.caption("✅ 텔레그램이 메인(브리핑/잔고/포지션/뉴스/ONOFF). Streamlit은 설정·상태 확인용 제어판입니다.")
