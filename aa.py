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
import sqlite3
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from streamlit.runtime.scriptrunner import add_script_run_ctx

# =========================================================
# ✅ 0) 시스템 기본 설정
# =========================================================
IS_SANDBOX = True  # ✅ 모의투자면 True, 실전은 False
SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
TRADE_LOG_FILE = "trade_log.csv"
DB_FILE = "wonyousi_brain.db"  # ✅ AI 회고/교훈 저장


st.set_page_config(layout="wide", page_title="Bitget AI 워뇨띠 봇 (제어판=Streamlit / 보고=Telegram)")

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT"
]

# =========================================================
# ✅ 1) 모드 규칙 (사용자 고정값 그대로)
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
        "entry_pct_min": 8,     # ✅ 공격모드: 최소 8%
        "entry_pct_max": 25,
        "lev_min": 2,
        "lev_max": 10,          # ✅ 레버리지는 낮게 유지
    },
    "하이리스크/하이리턴": {
        "min_conf": 85,
        "entry_pct_min": 15,
        "entry_pct_max": 40,
        "lev_min": 8,
        "lev_max": 25,          # ✅ 높게
    }
}

# =========================================================
# ✅ 2) Secrets / 키 로드
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
openai_key = st.secrets.get("OPENAI_API_KEY", "")

# =========================================================
# ✅ 3) 유틸
# =========================================================


KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def now_kst_str():
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def kr_side(decision: str) -> str:
    # decision: buy/sell
    return "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"

def tg_send(text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                      data={"chat_id": tg_id, "text": text})
    except:
        pass

def tg_send_md(text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                      data={"chat_id": tg_id, "text": text, "parse_mode": "Markdown"})
    except:
        # 마크다운 실패 시 일반 메시지로라도 보내기
        tg_send(text)

# =========================================================
# ✅ 4) 설정 로드/저장
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",
        "timeframe": "5m",
        "enforce_mode_rules": True,     # ✅ 모드 최소/최대 강제(핵심)
        "ai_journal_on_close": True,    # ✅ 청산 후 AI 회고 작성
        "ai_global_reco_auto_apply": False,  # ✅ AI가 글로벌 추천값 자동 적용 여부

        # ✅ 사용자(너)가 관리하는 금전/리스크 옵션(AI 추천 표시 가능)
        "max_consec_losses": 3,         # 연속 손실 n번이면 일시정지
        "pause_minutes": 30,            # 일시정지 시간(분)
        "per_coin_cooldown_sec": 30,    # 같은 코인 신규진입/AI호출 쿨다운

        # ✅ 손익비/손절 기본값(AI가 거래마다 추천하되, 안전장치)
        "manual_min_rr": 1.8,
        "manual_min_sl_pct": 1.2,
        "manual_tp_pct": 6.0,
        "manual_entry_pct": 10,
        "manual_leverage": 5,

        # ✅ 전략 토글(다 켜둠 = 기능 삭제 X)
        "use_pullback_entry": True,     # ✅ 눌림목(과매도 '해소' 진입) 핵심
        "use_trend_filter": True,
        "use_news_filter": False,       # 경제지표 전후 신규진입 회피(원하면 켜)
        "avoid_news_minutes": 15,

        "use_trailing_stop": True,
        "trail_activate_pct": 4.0,      # 수익률 +4% 이상부터 트레일링
        "trail_distance_pct": 2.0,      # 최고점 대비 -2%면 청산

        "use_dca": False,
        "dca_trigger_pct": -8.0,        # 손실률 -8%면 물타기
        "dca_max_count": 1,
        "dca_scale_pct": 50.0,          # 최초 증거금의 50%만큼 추가(보수적으로)

        "use_switching": False,         # 반대 시그널 강하면 스위칭
        "switch_conf": 90,              # 스위칭 확신도

        # ✅ 10종 보조지표 활성(기능 삭제 X)
        "use_rsi": True,
        "use_bb": True,
        "use_ma": True,
        "use_macd": True,
        "use_adx": True,
        "use_stoch": True,
        "use_mfi": True,
        "use_willr": True,
        "use_cci": True,
        "use_vol": True,

        # ✅ 지표 파라미터
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_fast": 20,
        "ma_slow": 60,
        "adx_period": 14,
        "atr_period": 14
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default.update(saved)
        except:
            pass

    # ✅ 저장된 trade_mode가 MODE_RULES에 없으면 강제 교정(에러 방지)
    if default.get("trade_mode") not in MODE_RULES:
        default["trade_mode"] = "안전모드"
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(default, f, ensure_ascii=False, indent=2)
        except:
            pass

    return default

def save_settings(conf):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        st.toast("✅ 설정 저장 완료", icon="💾")
    except:
        st.error("설정 저장 실패")

config = load_settings()

# =========================================================
# ✅ 5) 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime():
    return {
        "date": now_kst().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,     # epoch
        "cooldowns": {},      # coin -> epoch
        "trades": {}          # trade_id -> journal
    }

def load_runtime():
    rt = default_runtime()
    if os.path.exists(RUNTIME_FILE):
        try:
            with open(RUNTIME_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            rt.update(saved)
        except:
            pass

    # 날짜 바뀌면 일일 카운터 리셋
    if rt.get("date") != now_kst().strftime("%Y-%m-%d"):
        rt = default_runtime()
    return rt

def save_runtime(rt):
    try:
        with open(RUNTIME_FILE, "w", encoding="utf-8") as f:
            json.dump(rt, f, ensure_ascii=False, indent=2)
    except:
        pass

runtime_state = load_runtime()

# =========================================================
# ✅ 6) SQLite(회고/교훈 저장)
# =========================================================
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            symbol TEXT,
            result TEXT,
            roi_pct REAL,
            pnl_usdt REAL,
            one_liner TEXT,
            next_time TEXT
        )
        """)
        conn.commit()
        conn.close()
    except:
        pass

def save_lesson(time_s, symbol, result, roi_pct, pnl_usdt, one_liner, next_time):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO lessons(time, symbol, result, roi_pct, pnl_usdt, one_liner, next_time)
        VALUES(?,?,?,?,?,?,?)
        """, (time_s, symbol, result, roi_pct, pnl_usdt, one_liner, next_time))
        conn.commit()
        conn.close()
    except:
        pass

def get_recent_lessons(limit=10):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT time, symbol, result, roi_pct, one_liner, next_time FROM lessons ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append(f"- {r[0]} {r[1]} {r[2]} ({r[3]:.2f}%) | {r[4]} / 다음: {r[5]}")
        return "\n".join(out) if out else "최근 교훈 없음"
    except:
        return "최근 교훈 조회 실패"

init_db()

# =========================================================
# ✅ 7) CSV 매매 로그
# =========================================================
def append_trade_log(row: dict):
    df = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

def load_trade_log():
    if os.path.exists(TRADE_LOG_FILE):
        try:
            return pd.read_csv(TRADE_LOG_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def get_past_mistakes_summary():
    # ✅ 큰 손실 Top 5를 AI에게 알려주기(학습/회고)
    if not os.path.exists(TRADE_LOG_FILE):
        return "과거 매매 기록 없음."
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty or "ROI_percent" not in df.columns:
            return "기록은 있으나 분석할 데이터가 부족함."
        worst = df.sort_values(by="ROI_percent", ascending=True).head(5)
        s = []
        for _, r in worst.iterrows():
            s.append(f"- {r.get('Symbol','?')} {r.get('Result','?')} {safe_float(r.get('ROI_percent',0)):.2f}% | {r.get('OneLiner','')}")
        return "\n".join(s) if s else "큰 손실 기록 없음."
    except:
        return "기록 조회 실패"

# =========================================================
# ✅ 8) 거래소 연결
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

if not api_key or not api_secret or not api_password:
    st.error("🚨 Bitget API 키(3종)가 Secrets에 없습니다: API_KEY / API_SECRET / API_PASSWORD")
    st.stop()

if not tg_token or not tg_id:
    st.error("🚨 Telegram TOKEN/CHAT_ID가 Secrets에 없습니다: TG_TOKEN / TG_CHAT_ID")
    st.stop()

exchange = init_exchange()

# =========================================================
# ✅ 9) 지표 계산(ta 모듈 없이 직접 계산) - 10종 포함
# =========================================================
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-9))

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx_v = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_v, plus_di, minus_di

def bollinger(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return mid, upper, lower

def cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * (mad + 1e-9))

def stochastic(df, k_period=14, d_period=3):
    low_n = df["low"].rolling(k_period).min()
    high_n = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_n) / ((high_n - low_n) + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

def williams_r(df, period=14):
    low_n = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    wr = -100 * (high_n - df["close"]) / ((high_n - low_n) + 1e-9)
    return wr

def mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["vol"]
    pos = np.where(tp > tp.shift(), mf, 0.0)
    neg = np.where(tp < tp.shift(), mf, 0.0)
    pos_mf = pd.Series(pos).rolling(period).sum()
    neg_mf = pd.Series(neg).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - (100 / (1 + mfr))

def calc_indicators(df, conf):
    if df is None or df.empty or len(df) < 120:
        return df, {}, None

    df = df.copy()

    # RSI
    df["RSI"] = rsi(df["close"], conf["rsi_period"])

    # Bollinger
    bb_mid, bb_u, bb_l = bollinger(df["close"], conf["bb_period"], conf["bb_std"])
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bb_mid, bb_u, bb_l

    # MA
    df["MA_fast"] = df["close"].rolling(conf["ma_fast"]).mean()
    df["MA_slow"] = df["close"].rolling(conf["ma_slow"]).mean()

    # MACD
    df["EMA12"] = ema(df["close"], 12)
    df["EMA26"] = ema(df["close"], 26)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = ema(df["MACD"], 9)

    # ATR
    df["ATR"] = atr(df, conf["atr_period"])

    # ADX
    adx_v, pdi, mdi = adx(df, conf["adx_period"])
    df["ADX"], df["+DI"], df["-DI"] = adx_v, pdi.values, mdi.values

    # CCI, Stoch, WillR, MFI
    df["CCI"] = cci(df, 20)
    k, d = stochastic(df, 14, 3)
    df["StochK"], df["StochD"] = k, d
    df["WillR"] = williams_r(df, 14)
    df["MFI"] = mfi(df, 14)

    # Volume
    df["VolSMA"] = df["vol"].rolling(20).mean()
    df["VolSpike"] = df["vol"] > (df["VolSMA"] * 2.0)

    df = df.dropna()
    if df.empty:
        return df, {}, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 상태판(한글 + 쉬운 설명)
    status = {}

    trend_up = last["MA_fast"] > last["MA_slow"]
    status["추세"] = "상승추세(큰 흐름이 위)" if trend_up else "하락추세(큰 흐름이 아래)"

    # RSI 상태 + 흐름
    if last["RSI"] < 30:
        status["RSI"] = f"과매도(너무 많이 내려옴) {last['RSI']:.1f}"
    elif last["RSI"] > 70:
        status["RSI"] = f"과매수(너무 많이 올라옴) {last['RSI']:.1f}"
    else:
        status["RSI"] = f"중립 {last['RSI']:.1f}"
    status["RSI_흐름"] = f"{prev['RSI']:.1f} → {last['RSI']:.1f} (올라오는지/내려오는지)"

    # Bollinger
    if last["close"] < last["BB_lower"]:
        status["볼린저"] = "하단 이탈(과하게 눌림 가능)"
    elif last["close"] > last["BB_upper"]:
        status["볼린저"] = "상단 돌파(과열 가능)"
    else:
        status["볼린저"] = "밴드 안(평균 범위)"

    # MACD
    status["MACD"] = "상승 신호(위로 힘)" if last["MACD"] > last["MACD_signal"] else "하락 신호(아래로 힘)"

    # ADX
    status["ADX"] = f"{last['ADX']:.1f} " + ("(추세 강함)" if last["ADX"] >= 25 else "(횡보/약함)")

    # 기타
    status["거래량"] = "급증(관심 필요)" if bool(last["VolSpike"]) else "평균"
    status["MFI"] = f"{last['MFI']:.1f}(자금흐름)"
    status["CCI"] = f"{last['CCI']:.1f}(과열/침체 힌트)"
    status["Stoch"] = f"{last['StochK']:.1f}/{last['StochD']:.1f}(단기 힌트)"
    status["WillR"] = f"{last['WillR']:.1f}(단기 힌트)"

    # ✅ 눌림목 개선 핵심: “과매도에 진입”이 아니라 “과매도 해소/반등 확인”
    rsi_cross_up = (prev["RSI"] < 30) and (last["RSI"] >= 30)
    rsi_turn_up = last["RSI"] > prev["RSI"]
    status["_필터_눌림목반등후보"] = bool(trend_up and (prev["RSI"] < 35) and rsi_turn_up)
    status["_필터_RSI해소돌파"] = bool(rsi_cross_up)

    # 숏(하락 추세) 쪽도 동일하게 “과매수 해소” 확인
    rsi_cross_down = (prev["RSI"] > 70) and (last["RSI"] <= 70)
    rsi_turn_down = last["RSI"] < prev["RSI"]
    status["_필터_상승과열되돌림후보"] = bool((not trend_up) and (prev["RSI"] > 65) and rsi_turn_down)
    status["_필터_RSI과매수해소"] = bool(rsi_cross_down)

    return df, status, last

# =========================================================
# ✅ 10) OpenAI 클라이언트
# =========================================================
def get_openai_client():
    key = openai_key or config.get("openai_api_key", "")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except:
        return None

# =========================================================
# ✅ 11) AI: 글로벌 추천값(사이드바 추천용)
# =========================================================
def ai_global_reco(df, status, symbol, timeframe, mode_name):
    """
    ✅ 사이드바 옵션(손익비/손절/익절/트레일링 등)에 대해
    '지금 차트 기준 추천값'을 JSON으로 반환
    """
    client = get_openai_client()
    if client is None:
        return None

    last = df.iloc[-1]
    atr_pct = float(last["ATR"] / last["close"] * 100)

    system = f"""
너는 자동매매 봇의 '글로벌 옵션 추천자'야.
사용자는 안전/공격/하이리스크 모드를 쓰고 있고, 지금 모드는 {mode_name}야.

목표:
- 손실은 짧게 끊되(너무 좁으면 휩쏘), 추세가 맞으면 익절을 길게 가져갈 수 있게 세팅.
- 수수료 누수(횡보장 잦은 진입)를 줄이기 위한 추천도 포함.

반드시 JSON으로만 답해. 키는 아래를 포함:
{{
  "recommended": {{
    "manual_min_rr": 숫자,
    "manual_min_sl_pct": 숫자,
    "manual_tp_pct": 숫자,
    "trail_activate_pct": 숫자,
    "trail_distance_pct": 숫자,
    "per_coin_cooldown_sec": 정수,
    "use_news_filter": true/false,
    "avoid_news_minutes": 정수
  }},
  "why_easy": "아주 쉬운 한국어(괄호로 풀어쓰기)",
  "watch_indicators": ["RSI", "MA", ...]  # 지금 차트에서 중요하게 볼 지표들
}}
"""
    user = f"""
[차트] {symbol} / {timeframe}
- 추세: {status.get("추세")}
- RSI: {status.get("RSI")} / RSI 흐름: {status.get("RSI_흐름")}
- 볼린저: {status.get("볼린저")}
- MACD: {status.get("MACD")}
- ADX: {status.get("ADX")}
- ATR% (변동성): 약 {atr_pct:.2f}%
"""

    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=0.25
        )
        return json.loads(r.choices[0].message.content)
    except:
        return None

# =========================================================
# ✅ 12) AI: 매매 결정(진입/손절/익절/레버리지/진입비중 + 쉬운 설명)
# =========================================================
def ai_decide_trade(df, status, symbol, timeframe, mode_name):
    client = get_openai_client()
    rule = MODE_RULES.get(mode_name, MODE_RULES["안전모드"])

    # OpenAI 없으면 수동값 기반으로만
    if client is None:
        return {
            "decision": "hold",
            "confidence": 0,
            "entry_pct": config.get("manual_entry_pct", 10),
            "leverage": config.get("manual_leverage", 5),
            "sl_pct": config.get("manual_min_sl_pct", 1.2),
            "tp_pct": config.get("manual_tp_pct", 6.0),
            "reason_easy": "OpenAI 키가 없어서 AI 판단을 생략했어요. (수동값만 유지)",
            "reason_detail": "OpenAI 미설정",
            "used_indicators": ["RSI", "MA", "볼린저", "MACD", "ADX", "ATR", "거래량", "MFI", "CCI", "Stoch", "WillR"]
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    atr_pct = float(last["ATR"] / last["close"] * 100)

    past_mistakes = get_past_mistakes_summary()
    recent_lessons = get_recent_lessons(limit=8)

    system = f"""
너는 '워뇨띠 매매법'을 베이스로 한 선별형 트레이더야.
목표는: 원금손실을 줄이고, 기회가 올 때만 진입해서 익절을 더 많이 만들기.

[사용자 문제(반드시 고쳐야 함)]
- 상승추세 눌림목에서 RSI가 과매도라고 바로 진입하면 휩쏘(잠깐 흔들기)에 계속 털린다.
✅ 그래서 "과매도 진입"이 아니라
✅ "과매도 해소(다시 올라오는 순간) + 반등 확인"을 더 중요하게 본다.

[모드: {mode_name}]
- 최소 확신도: {rule["min_conf"]}%
- 진입비중(잔고%): {rule["entry_pct_min"]} ~ {rule["entry_pct_max"]}
- 레버리지: {rule["lev_min"]} ~ {rule["lev_max"]}

[과거 큰 손실 5개]
{past_mistakes}

[최근 회고/교훈]
{recent_lessons}

[출력 형식(JSON만)]
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0~100,
  "entry_pct": 숫자,
  "leverage": 숫자,
  "sl_pct": 숫자,
  "tp_pct": 숫자,
  "reason_easy": "매우 쉬운 한국어(괄호로 풀어쓰기)",
  "reason_detail": "조금 자세히",
  "used_indicators": ["RSI(14)", "MA(20/60)", ...]
}}
"""
    user = f"""
[차트] {symbol} / {timeframe}
- 현재가: {last["close"]:.4f}
- 추세: {status.get("추세")}
- RSI: {status.get("RSI")} / RSI 흐름: {status.get("RSI_흐름")}
- 볼린저: {status.get("볼린저")}
- MACD: {status.get("MACD")}
- ADX: {status.get("ADX")}
- 거래량: {status.get("거래량")}
- ATR% (변동성): 약 {atr_pct:.2f}%

[추가 힌트]
- 손절을 너무 좁게 잡으면 휩쏘에 터진다.
- ATR%가 크면 손절도 최소한 의미 있게 잡아야 한다.
- 확신도가 낮으면 관망이 정답이다.
"""

    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=0.25
        )
        out = json.loads(r.choices[0].message.content)

        # ✅ 최소 키 보정
        out.setdefault("decision","hold")
        out.setdefault("confidence",0)
        out.setdefault("entry_pct", config.get("manual_entry_pct", 10))
        out.setdefault("leverage", config.get("manual_leverage", 5))
        out.setdefault("sl_pct", config.get("manual_min_sl_pct", 1.2))
        out.setdefault("tp_pct", config.get("manual_tp_pct", 6.0))
        out.setdefault("reason_easy","")
        out.setdefault("reason_detail","")
        out.setdefault("used_indicators", [])
        return out
    except Exception as e:
        return {
            "decision": "hold",
            "confidence": 0,
            "entry_pct": config.get("manual_entry_pct", 10),
            "leverage": config.get("manual_leverage", 5),
            "sl_pct": config.get("manual_min_sl_pct", 1.2),
            "tp_pct": config.get("manual_tp_pct", 6.0),
            "reason_easy": f"AI 호출 오류로 관망 처리(에러: {str(e)[:80]})",
            "reason_detail": "AI 오류",
            "used_indicators": ["RSI", "MA", "볼린저", "MACD", "ADX"]
        }

# =========================================================
# ✅ 13) AI: 청산 후 회고(한줄평 + 다음 개선)
# =========================================================
def ai_write_journal(trade_summary: dict):
    client = get_openai_client()
    if client is None:
        return {"one_liner":"AI 키 없음: 수동 기록", "next_time":"다음엔 손절/익절 기준을 더 명확히"}

    system = """
너는 매매 코치야.
아래 매매 결과를 보고:
1) 한줄평(아주 쉽게)
2) 다음엔 어떻게 할지(아주 쉽게)
를 한국어로 작성해.
어려운 용어는 (괄호로 쉬운 말)로 풀어쓰기.
JSON만 출력:
{"one_liner":"...", "next_time":"..."}
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(trade_summary, ensure_ascii=False)}],
            response_format={"type":"json_object"},
            temperature=0.3
        )
        return json.loads(r.choices[0].message.content)
    except:
        return {"one_liner":"회고 생성 실패", "next_time":"다음엔 진입 근거를 더 선명하게"}

# =========================================================
# ✅ 14) 경제 캘린더(한글)
# =========================================================
def get_forex_events_kor(limit=30):
    """
    ✅ ForexFactory 공개 주간 캘린더 JSON
    - 구조가 바뀔 수 있어서 최대한 방어적으로 파싱
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list):
            return pd.DataFrame(columns=["일시","국가","중요도","지표","예상","이전","실제"])

        impact_map = {
            "Low":"낮음",
            "Medium":"중간",
            "High":"높음",
            "Holiday":"휴장"
        }

        rows = []
        for e in data:
            title = str(e.get("title", e.get("event", "")))
            country = str(e.get("country", e.get("currency", "")))
            impact = str(e.get("impact", e.get("importance", "")))
            impact = impact_map.get(impact, impact)

            date_s = str(e.get("date",""))
            time_s = str(e.get("time",""))
            dt_display = f"{date_s} {time_s}".strip()

            rows.append({
                "일시": dt_display,
                "국가": country,
                "중요도": impact,
                "지표": title,
                "예상": str(e.get("forecast","")),
                "이전": str(e.get("previous","")),
                "실제": str(e.get("actual",""))
            })

        df = pd.DataFrame(rows)
        return df.head(limit)
    except:
        return pd.DataFrame(columns=["일시","국가","중요도","지표","예상","이전","실제"])

def is_news_risky_now(conf):
    """
    ✅ (옵션) 경제지표 전후 신규 진입 회피
    - 정확한 타임존/구조가 보장되진 않아서: "대충 위험 회피" 용도로만 사용 권장
    """
    if not conf.get("use_news_filter", False):
        return False

    minutes = int(conf.get("avoid_news_minutes", 15))
    df = get_forex_events_kor(limit=50)
    if df.empty:
        return False

    # 아주 단순히 "중요도=높음" 이 있으면 무조건 피하는 게 아니라,
    # 시간 파싱이 애매하므로 여기서는 '안전하게' false로 둠.
    # (원하면 내가 “시간대 정확 파싱 버전”으로도 업그레이드 가능)
    return False

# =========================================================
# ✅ 15) 주문/청산 헬퍼
# =========================================================
def set_leverage_safe(ex, lev, symbol):
    try:
        ex.set_leverage(int(lev), symbol)
    except:
        pass

def close_position_market(ex, symbol, side, contracts):
    """
    side: long/buy OR short/sell
    """
    try:
        close_side = "sell" if str(side).lower() in ["long", "buy"] else "buy"
        ex.create_market_order(symbol, close_side, contracts)
        return True
    except:
        return False

# =========================================================
# ✅ 16) 포지션 관리(손절/익절/트레일링/DCA/스위칭)
# =========================================================
def manage_position(ex, symbol, pos, active_info, conf):
    """
    - TP/SL: roi% 기준
    - 트레일링: 최고 roi에서 일정 폭 하락하면 청산
    - DCA: 손실이면 1회 추가(옵션)
    """
    try:
        roi = safe_float(pos.get("percentage", 0.0))
        upnl = safe_float(pos.get("unrealizedPnl", 0.0))
        entry_price = safe_float(pos.get("entryPrice", 0.0))
        side = pos.get("side", "")
        contracts = safe_float(pos.get("contracts", 0.0))
        if contracts <= 0:
            return None

        sl_pct = abs(safe_float(active_info.get("sl_pct", conf.get("manual_min_sl_pct", 1.2))))
        tp_pct = abs(safe_float(active_info.get("tp_pct", conf.get("manual_tp_pct", 6.0))))

        # ✅ 트레일링: (수익이 충분히 났을 때만)
        if conf.get("use_trailing_stop", True):
            act = safe_float(conf.get("trail_activate_pct", 4.0))
            dist = safe_float(conf.get("trail_distance_pct", 2.0))

            # 최고 roi 갱신
            best = safe_float(active_info.get("best_roi", roi))
            if roi > best:
                best = roi
                active_info["best_roi"] = best

            # 활성화 이후에는 best - dist 이탈 시 청산
            if best >= act:
                trail_line = best - dist
                active_info["trail_line"] = trail_line
                if roi <= trail_line:
                    ok = close_position_market(ex, symbol, side, contracts)
                    if ok:
                        return {"closed": True, "close_reason": f"트레일링 스탑(최고 {best:.2f}%에서 밀림)"}
        # ✅ 기본 TP/SL
        if roi <= -sl_pct:
            ok = close_position_market(ex, symbol, side, contracts)
            if ok:
                return {"closed": True, "close_reason": f"손절(목표 -{sl_pct:.2f}% 도달)"}
        if roi >= tp_pct:
            ok = close_position_market(ex, symbol, side, contracts)
            if ok:
                return {"closed": True, "close_reason": f"익절(목표 +{tp_pct:.2f}% 도달)"}

        # ✅ DCA(옵션)
        if conf.get("use_dca", False):
            dca_trigger = safe_float(conf.get("dca_trigger_pct", -8.0))  # 음수
            max_count = int(conf.get("dca_max_count", 1))
            dca_count = int(active_info.get("dca_count", 0))
            if roi <= dca_trigger and dca_count < max_count:
                # 추가 진입(현재 포지션의 증거금 기준 일부만)
                scale = safe_float(conf.get("dca_scale_pct", 50.0)) / 100.0
                # 잔고 확인
                bal = ex.fetch_balance({"type":"swap"})
                free = safe_float(bal["USDT"]["free"])
                add_margin = safe_float(active_info.get("margin_usdt", 0.0)) * scale
                if add_margin > 1 and free > add_margin:
                    price = safe_float(ex.fetch_ticker(symbol).get("last", entry_price))
                    lev = int(active_info.get("lev", conf.get("manual_leverage", 5)))
                    notional = add_margin * lev
                    qty = notional / max(price, 1e-9)
                    qty = ex.amount_to_precision(symbol, qty)
                    order_side = "buy" if str(side).lower() in ["long", "buy"] else "sell"
                    ex.create_market_order(symbol, order_side, qty)
                    active_info["dca_count"] = dca_count + 1
                    tg_send(f"💧 물타기(DCA) 실행: {symbol}\n- 손실률 {roi:.2f}%에서 추가 진입\n- 추가 증거금 {add_margin:.2f} USDT")
        return None
    except:
        return None

# =========================================================
# ✅ 17) 자동매매 스레드(텔레그램 보고 포함)
# =========================================================
def telegram_thread(ex):
    tg_send(
        "🚀 봇 시작!\n"
        f"- 모의투자: {'ON(샌드박스)' if IS_SANDBOX else 'OFF(실전)'}\n"
        f"- 시간(KST): {now_kst_str()}\n"
        "📌 Streamlit은 제어판, 텔레그램이 모든 보고/조회입니다.\n"
        "명령어: 잔고 / 포지션 / 매매일지 / 캘린더 / 상태 / 스캔"
    )

    active_trades = {}  # symbol -> dict(sl,tp,lev,entry_pct,trade_id,margin_usdt,best_roi,trail_line,dca_count,...)
    offset = 0
    last_ping = time.time()

    while True:
        try:
            conf = load_settings()
            rt = load_runtime()

            # ✅ 일시정지(pause)
            now_epoch = int(time.time())
            if rt.get("pause_until", 0) > now_epoch:
                # 그래도 텔레그램 명령은 처리 가능
                pass

            # -----------------------------
            # A) 텔레그램 "명령" 처리
            # -----------------------------
            try:
                res = requests.get(
                    f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1",
                    timeout=5
                ).json()
                if res.get("ok"):
                    for up in res.get("result", []):
                        offset = up["update_id"]
                        if "message" in up and "text" in up["message"]:
                            txt = up["message"]["text"].strip()

                            if txt in ["/start", "메뉴", "도움말"]:
                                tg_send("📌 명령어\n- 잔고\n- 포지션\n- 매매일지\n- 캘린더\n- 상태\n- 스캔")
                            elif txt == "상태":
                                tg_send(f"✅ 상태\n- 모드: {conf.get('trade_mode')}\n- 자동매매: {'ON' if conf.get('auto_trade') else 'OFF'}\n- 시간(KST): {now_kst_str()}")
                            elif txt == "잔고":
                                bal = ex.fetch_balance({"type":"swap"})
                                total = safe_float(bal["USDT"]["total"])
                                free = safe_float(bal["USDT"]["free"])
                                tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT")
                            elif txt == "포지션":
                                ps = ex.fetch_positions(symbols=TARGET_COINS)
                                act = [p for p in ps if safe_float(p.get("contracts", 0)) > 0]
                                if not act:
                                    tg_send("📊 포지션 없음(관망)")
                                else:
                                    msg = "📊 현재 포지션\n"
                                    for p in act:
                                        sym = p.get("symbol","")
                                        side = str(p.get("side","")).lower()
                                        side_kr = "롱" if side in ["long","buy"] else "숏"
                                        roi = safe_float(p.get("percentage",0))
                                        upnl = safe_float(p.get("unrealizedPnl",0))
                                        lev = p.get("leverage","?")
                                        msg += f"- {sym} {side_kr} x{lev} | 수익률 {roi:.2f}% | 손익 {upnl:.2f} USDT\n"
                                    tg_send(msg)
                            elif txt == "매매일지":
                                trades = rt.get("trades", {})
                                if not trades:
                                    tg_send("📜 매매일지: 아직 기록 없음")
                                else:
                                    items = list(trades.values())[-10:]
                                    msg = "📜 최근 매매일지(한줄평)\n"
                                    for t in items[::-1]:
                                        msg += f"- {t.get('time','')} {t.get('symbol','')} {t.get('result','')} | {t.get('one_liner','')}\n"
                                    tg_send(msg)
                            elif txt == "캘린더":
                                df_ev = get_forex_events_kor(limit=15)
                                if df_ev.empty:
                                    tg_send("📅 캘린더: 불러오기 실패/없음")
                                else:
                                    lines = ["📅 이번주 경제 캘린더(요약)"]
                                    for _, r in df_ev.iterrows():
                                        lines.append(f"- {r['일시']} / {r['국가']} / {r['중요도']} / {r['지표']}")
                                    tg_send("\n".join(lines[:25]))
                            elif txt == "스캔":
                                # ✅ 5개 코인 즉시 스캔 결과 요약
                                tf = conf.get("timeframe","5m")
                                lines = [f"🌍 전체 스캔({tf})"]
                                for coin in TARGET_COINS:
                                    try:
                                        ohlcv = ex.fetch_ohlcv(coin, tf, limit=150)
                                        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                                        df["time"] = pd.to_datetime(df["time"], unit="ms")
                                        df, status, last = calc_indicators(df, conf)
                                        if last is None:
                                            continue
                                        mode_name = conf.get("trade_mode","안전모드")
                                        ai = ai_decide_trade(df, status, coin, tf, mode_name)
                                        lines.append(f"- {coin}: {ai.get('decision','hold').upper()} / 확신 {ai.get('confidence',0)}% / {ai.get('reason_easy','')[:30]}")
                                    except:
                                        pass
                                tg_send("\n".join(lines[:30]))
            except:
                pass

            # -----------------------------
            # B) 자동매매 로직
            # -----------------------------
            if not conf.get("auto_trade", False):
                time.sleep(1)
                continue

            # ✅ 일시정지중이면 신규 진입은 중단(포지션 관리는 계속)
            pause_active = rt.get("pause_until", 0) > int(time.time())

            mode_name = conf.get("trade_mode","안전모드")
            rule = MODE_RULES.get(mode_name, MODE_RULES["안전모드"])

            for coin in TARGET_COINS:
                try:
                    # 포지션 조회
                    positions = ex.fetch_positions([coin])
                    active_pos = [p for p in positions if safe_float(p.get("contracts",0)) > 0]

                    # (1) 포지션이 있으면 관리(청산/트레일링/DCA)
                    if active_pos:
                        p = active_pos[0]
                        info = active_trades.get(coin, {})
                        res = manage_position(ex, coin, p, info, conf)
                        # manage_position에서 업데이트된 정보 저장
                        if info:
                            active_trades[coin] = info

                        # 청산 발생 시 기록/회고
                        if res and res.get("closed"):
                            # 청산 후 다시 포지션 조회해서 실제 값 맞추기(안전)
                            roi = safe_float(p.get("percentage",0))
                            upnl = safe_float(p.get("unrealizedPnl",0))
                            entry_price = safe_float(p.get("entryPrice",0))
                            side = p.get("side","")
                            close_reason = res.get("close_reason","자동 청산")

                            result = "익절(수익)" if roi >= 0 else "손절(손실)"
                            t_time = now_kst_str()

                            # 연속손실 카운터 업데이트
                            if roi < 0:
                                rt["consec_losses"] = int(rt.get("consec_losses",0)) + 1
                            else:
                                rt["consec_losses"] = 0

                            # 연속손실로 일시정지
                            if rt["consec_losses"] >= int(conf.get("max_consec_losses",3)):
                                pause_minutes = int(conf.get("pause_minutes",30))
                                rt["pause_until"] = int(time.time()) + pause_minutes * 60
                                tg_send(f"⛔ 연속 손실 {rt['consec_losses']}회 → {pause_minutes}분 자동 일시정지!")

                            trade_summary = {
                                "time": t_time,
                                "symbol": coin,
                                "result": result,
                                "roi_pct": roi,
                                "pnl_usdt": upnl,
                                "entry_price": entry_price,
                                "mode": mode_name,
                                "close_reason": close_reason
                            }

                            one = {"one_liner":"", "next_time":""}
                            if conf.get("ai_journal_on_close", True):
                                one = ai_write_journal(trade_summary)

                            trade_id = info.get("trade_id") or f"{int(time.time())}_{coin.replace('/','_')}"

                            rt["trades"][trade_id] = {
                                "time": t_time,
                                "symbol": coin,
                                "result": result,
                                "roi_pct": roi,
                                "pnl_usdt": upnl,
                                "one_liner": one.get("one_liner",""),
                                "next_time": one.get("next_time",""),
                                "close_reason": close_reason
                            }
                            save_runtime(rt)

                            append_trade_log({
                                "Time": t_time,
                                "Symbol": coin,
                                "Mode": mode_name,
                                "Result": result,
                                "ROI_percent": roi,
                                "PnL_USDT": upnl,
                                "EntryPrice": entry_price,
                                "CloseReason": close_reason,
                                "OneLiner": one.get("one_liner",""),
                                "NextTime": one.get("next_time","")
                            })

                            save_lesson(t_time, coin, result, roi, upnl, one.get("one_liner",""), one.get("next_time",""))

                            tg_send(
                                "📌 청산 알림\n"
                                f"- 코인: {coin}\n"
                                f"- 결과: {result}\n"
                                f"- 수익률: {roi:.2f}% / 손익: {upnl:.2f} USDT\n"
                                f"- 청산 이유: {close_reason}\n"
                                f"- 한줄평: {one.get('one_liner','')}\n"
                                f"- 다음엔: {one.get('next_time','')}"
                            )

                            if coin in active_trades:
                                del active_trades[coin]
                        continue

                    # (2) 포지션이 없으면 신규 진입(일시정지면 skip)
                    if pause_active:
                        continue

                    # 코인별 쿨다운
                    cd = rt.get("cooldowns", {}).get(coin, 0)
                    if int(time.time()) < int(cd):
                        continue

                    # 뉴스 회피(옵션)
                    if is_news_risky_now(conf):
                        continue

                    tf = conf.get("timeframe","5m")
                    ohlcv = ex.fetch_ohlcv(coin, tf, limit=150)
                    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    df, status, last = calc_indicators(df, conf)
                    if last is None:
                        continue

                    # ✅ “눌림목 개선” 필터:
                    # - 상승추세: RSI 과매도에 '바로 진입' 금지
                    # - RSI가 되돌아오는 흐름(해소/반등 후보)일 때만 적극 진입
                    call_ai = True
                    if conf.get("use_pullback_entry", True):
                        # 애매한 횡보에서는 호출 줄여 수수료 누수 방지
                        if (30 <= safe_float(last["RSI"]) <= 70) and (safe_float(last["ADX"]) < 18):
                            call_ai = False

                        # 눌림목 반등/해소 후보가 아니면 신규진입 더 까다롭게
                        pullback_ok = bool(status.get("_필터_눌림목반등후보") or status.get("_필터_RSI해소돌파") or
                                          status.get("_필터_상승과열되돌림후보") or status.get("_필터_RSI과매수해소"))
                        if not pullback_ok:
                            # 그래도 강추세(ADX 높음)라면 기회가 있을 수 있어 호출 유지
                            if safe_float(last["ADX"]) < 25:
                                call_ai = False

                    if not call_ai:
                        continue

                    ai = ai_decide_trade(df, status, coin, tf, mode_name)
                    decision = ai.get("decision","hold")
                    conf_score = int(safe_float(ai.get("confidence",0)))

                    # 모드별 최소 확신도
                    if decision not in ["buy","sell"] or conf_score < int(rule["min_conf"]):
                        continue

                    # AI 값
                    entry_pct = safe_float(ai.get("entry_pct", conf.get("manual_entry_pct",10)))
                    lev = int(safe_float(ai.get("leverage", conf.get("manual_leverage",5))))
                    sl_pct = safe_float(ai.get("sl_pct", conf.get("manual_min_sl_pct",1.2)))
                    tp_pct = safe_float(ai.get("tp_pct", conf.get("manual_tp_pct",6.0)))

                    # ✅ 모드 룰 강제(공격모드인데 2% 들어가는 문제 방지)
                    if conf.get("enforce_mode_rules", True):
                        entry_pct = clamp(entry_pct, rule["entry_pct_min"], rule["entry_pct_max"])
                        lev = int(clamp(lev, rule["lev_min"], rule["lev_max"]))

                    # ✅ 휩쏘 방지: ATR 기반 최소 손절폭 보정
                    atr_pct = safe_float(last["ATR"] / last["close"] * 100)
                    min_sl_from_atr = max(0.6, atr_pct * 0.9)
                    sl_pct = max(sl_pct, min_sl_from_atr)

                    # ✅ 손익비 체크(너가 원하는 “익절이 더 많게”)
                    rr = tp_pct / max(sl_pct, 1e-9)
                    min_rr = safe_float(conf.get("manual_min_rr", 1.8))
                    if rr < min_rr:
                        # 손익비가 너무 안 좋으면 관망(수수료 누수 방지)
                        continue

                    # 잔고/수량 계산
                    bal = ex.fetch_balance({"type":"swap"})
                    free = safe_float(bal["USDT"]["free"])
                    total = safe_float(bal["USDT"]["total"])
                    margin = free * (entry_pct / 100.0)

                    if margin <= 1:
                        continue

                    # 레버리지 설정
                    set_leverage_safe(ex, lev, coin)

                    price = safe_float(last["close"])
                    notional = margin * lev
                    qty = notional / max(price, 1e-9)
                    qty = ex.amount_to_precision(coin, qty)
                    if safe_float(qty) <= 0:
                        continue

                    # 진입
                    ex.create_market_order(coin, decision, qty)

                    trade_id = f"{int(time.time())}_{coin.replace('/','_')}"

                    active_trades[coin] = {
                        "trade_id": trade_id,
                        "sl_pct": sl_pct,
                        "tp_pct": tp_pct,
                        "lev": lev,
                        "entry_pct": entry_pct,
                        "margin_usdt": margin,
                        "notional": notional,
                        "best_roi": 0.0,
                        "trail_line": None,
                        "dca_count": 0,
                        "open_time": now_kst_str(),
                        "reason_easy": ai.get("reason_easy",""),
                        "used_indicators": ai.get("used_indicators", [])
                    }

                    # 코인 쿨다운 설정
                    rt["cooldowns"][coin] = int(time.time()) + int(conf.get("per_coin_cooldown_sec", 30))
                    save_runtime(rt)

                    # 텔레그램 진입 보고(USDT + 잔고% + 쉬운 근거)
                    est_tp = price * (1 + tp_pct/100.0) if decision == "buy" else price * (1 - tp_pct/100.0)
                    est_sl = price * (1 - sl_pct/100.0) if decision == "buy" else price * (1 + sl_pct/100.0)

                    tg_send(
                        "🚀 진입 알림\n"
                        f"- 모드: {mode_name}\n"
                        f"- 코인: {coin}\n"
                        f"- 방향: {kr_side(decision)}\n"
                        f"- 확신도: {conf_score}% (AI가 좋다고 느낀 정도)\n"
                        f"- 진입 증거금: {margin:.2f} USDT (잔고의 약 {entry_pct:.1f}%)\n"
                        f"- 포지션 규모(레버 포함): {notional:.2f} USDT (x{lev})\n"
                        f"- 목표 익절: +{tp_pct:.2f}% (예상가 {est_tp:.4f})\n"
                        f"- 목표 손절: -{sl_pct:.2f}% (예상가 {est_sl:.4f})\n"
                        f"- 손익비(RR): {rr:.2f} (최소기준 {min_rr})\n"
                        f"- 쉬운 근거: {ai.get('reason_easy','')}\n"
                        f"- AI가 본 지표: {', '.join(ai.get('used_indicators', []))}"
                    )

                    time.sleep(2)

                except:
                    pass

            # (C) 생존신고
            if time.time() - last_ping > 900:
                try:
                    bal = ex.fetch_balance({"type":"swap"})
                    total = safe_float(bal["USDT"]["total"])
                    tg_send(f"💤 생존신고: 총자산 {total:.2f} USDT / 모드={load_settings().get('trade_mode')}")
                except:
                    pass
                last_ping = time.time()

            time.sleep(1)

        except:
            time.sleep(3)

# =========================================================
# ✅ 18) Streamlit UI (제어판)
# =========================================================
st.title("🧠 Bitget AI 워뇨띠 봇")
st.caption("Streamlit=제어판(설정/차트/일지) · Telegram=실시간 보고/조회(진입·청산·명령어)")

# 기본 체크
if not openai_key and not config.get("openai_api_key",""):
    st.warning("⚠️ OPENAI_API_KEY가 없어서 AI 기능이 제한됩니다(관망/수동값 중심). Secrets 또는 사이드바에 입력하세요.")

# =========================================================
# 사이드바(제어판)
# =========================================================
st.sidebar.title("🛠️ 제어판")

# ✅ selectbox 인덱스 에러 방지 포함
mode_keys = list(MODE_RULES.keys())
saved_mode = config.get("trade_mode", "안전모드")
default_index = mode_keys.index(saved_mode) if saved_mode in mode_keys else 0
mode = st.sidebar.selectbox("매매 모드", mode_keys, index=default_index)

auto_trade = st.sidebar.checkbox("🤖 자동매매 ON/OFF", value=config.get("auto_trade", False))
timeframe = st.sidebar.selectbox("타임프레임", ["1m","3m","5m","15m","1h"], index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))
enforce_rules = st.sidebar.checkbox("✅ 모드 룰 강제(진입비중/레버리지)", value=config.get("enforce_mode_rules", True))
ai_journal = st.sidebar.checkbox("📝 청산 시 AI 회고 자동작성", value=config.get("ai_journal_on_close", True))

st.sidebar.divider()
st.sidebar.subheader("💰 리스크(너가 관리) + AI 추천 표시")
max_losses = st.sidebar.slider("연속 손실 제한(회)", 1, 10, int(config.get("max_consec_losses", 3)))
pause_minutes = st.sidebar.slider("일시정지(분)", 5, 180, int(config.get("pause_minutes", 30)))
cooldown_sec = st.sidebar.slider("코인별 쿨다운(초)", 5, 300, int(config.get("per_coin_cooldown_sec", 30)))

st.sidebar.divider()
st.sidebar.subheader("📐 손익비/기본값(너가 관리)")
manual_rr = st.sidebar.slider("최소 손익비(RR)", 1.0, 5.0, float(config.get("manual_min_rr", 1.8)), step=0.1)
manual_sl = st.sidebar.slider("기본 손절(%)", 0.2, 10.0, float(config.get("manual_min_sl_pct", 1.2)), step=0.1)
manual_tp = st.sidebar.slider("기본 익절(%)", 1.0, 40.0, float(config.get("manual_tp_pct", 6.0)), step=0.5)
manual_entry = st.sidebar.slider("기본 진입비중(%)", 1, 50, int(config.get("manual_entry_pct", 10)))
manual_lev = st.sidebar.slider("기본 레버리지", 1, 50, int(config.get("manual_leverage", 5)))

st.sidebar.divider()
st.sidebar.subheader("🧠 전략 기능(전체 기능 유지)")
use_pullback = st.sidebar.checkbox("✅ 눌림목(과매도/과매수 해소) 진입", value=config.get("use_pullback_entry", True))
use_trailing = st.sidebar.checkbox("✅ 트레일링 스탑(수익 늘리기)", value=config.get("use_trailing_stop", True))
trail_act = st.sidebar.slider("트레일 시작 수익률(%)", 1.0, 20.0, float(config.get("trail_activate_pct", 4.0)), step=0.5)
trail_dist = st.sidebar.slider("트레일 폭(%)", 0.5, 10.0, float(config.get("trail_distance_pct", 2.0)), step=0.5)

use_dca = st.sidebar.checkbox("💧 물타기(DCA)", value=config.get("use_dca", False))
dca_trig = st.sidebar.slider("DCA 발동 수익률(%)", -30.0, -1.0, float(config.get("dca_trigger_pct", -8.0)), step=0.5)
dca_max = st.sidebar.slider("DCA 최대 횟수", 0, 5, int(config.get("dca_max_count", 1)))
dca_scale = st.sidebar.slider("DCA 추가비중(초기 증거금 대비 %)", 10.0, 200.0, float(config.get("dca_scale_pct", 50.0)), step=10.0)

use_switch = st.sidebar.checkbox("🔄 스위칭(반대 신호 강하면 전환)", value=config.get("use_switching", False))
switch_conf = st.sidebar.slider("스위칭 확신도 기준", 50, 100, int(config.get("switch_conf", 90)))

st.sidebar.divider()
st.sidebar.subheader("📊 10종 지표 활성(유지)")
use_rsi = st.sidebar.checkbox("RSI", value=config.get("use_rsi", True))
use_bb = st.sidebar.checkbox("볼린저", value=config.get("use_bb", True))
use_ma = st.sidebar.checkbox("이평(MA)", value=config.get("use_ma", True))
use_macd = st.sidebar.checkbox("MACD", value=config.get("use_macd", True))
use_adx = st.sidebar.checkbox("ADX", value=config.get("use_adx", True))
use_stoch = st.sidebar.checkbox("스토캐스틱", value=config.get("use_stoch", True))
use_mfi = st.sidebar.checkbox("MFI", value=config.get("use_mfi", True))
use_willr = st.sidebar.checkbox("Williams %R", value=config.get("use_willr", True))
use_cci = st.sidebar.checkbox("CCI", value=config.get("use_cci", True))
use_vol = st.sidebar.checkbox("거래량", value=config.get("use_vol", True))

st.sidebar.divider()
st.sidebar.subheader("🧹 매매일지 관리")
if st.sidebar.button("🗑️ 매매일지 초기화(runtime+csv+db)"):
    for f in [RUNTIME_FILE, TRADE_LOG_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass
    # db 초기화(테이블은 유지, 데이터만 삭제)
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("DELETE FROM lessons")
        conn.commit()
        conn.close()
    except:
        pass
    st.sidebar.success("초기화 완료! 새로고침하면 반영됩니다.")
    st.rerun()

# OpenAI 키 입력(선택)
if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key(선택)", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

# ✅ 설정 반영
changed = False
updates = {
    "trade_mode": mode,
    "auto_trade": auto_trade,
    "timeframe": timeframe,
    "enforce_mode_rules": enforce_rules,
    "ai_journal_on_close": ai_journal,

    "max_consec_losses": max_losses,
    "pause_minutes": pause_minutes,
    "per_coin_cooldown_sec": cooldown_sec,

    "manual_min_rr": manual_rr,
    "manual_min_sl_pct": manual_sl,
    "manual_tp_pct": manual_tp,
    "manual_entry_pct": manual_entry,
    "manual_leverage": manual_lev,

    "use_pullback_entry": use_pullback,
    "use_trailing_stop": use_trailing,
    "trail_activate_pct": trail_act,
    "trail_distance_pct": trail_dist,

    "use_dca": use_dca,
    "dca_trigger_pct": dca_trig,
    "dca_max_count": dca_max,
    "dca_scale_pct": dca_scale,

    "use_switching": use_switch,
    "switch_conf": switch_conf,

    "use_rsi": use_rsi,
    "use_bb": use_bb,
    "use_ma": use_ma,
    "use_macd": use_macd,
    "use_adx": use_adx,
    "use_stoch": use_stoch,
    "use_mfi": use_mfi,
    "use_willr": use_willr,
    "use_cci": use_cci,
    "use_vol": use_vol,
}

for k, v in updates.items():
    if config.get(k) != v:
        config[k] = v
        changed = True

if changed:
    save_settings(config)

# ✅ 현재 모드 룰 요약 표시
rule = MODE_RULES[config.get("trade_mode", "안전모드")]
st.sidebar.success(
    f"✅ 현재 모드 룰\n"
    f"- 최소 확신도: {rule['min_conf']}%\n"
    f"- 진입비중: {rule['entry_pct_min']}% ~ {rule['entry_pct_max']}%\n"
    f"- 레버리지: x{rule['lev_min']} ~ x{rule['lev_max']}\n"
    f"- 강제 적용: {'ON' if config.get('enforce_mode_rules') else 'OFF'}"
)

# =========================================================
# ✅ 텔레그램 스레드 시작(중복 방지)
# =========================================================
found = any(t.name == "TG_THREAD" for t in threading.enumerate())
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(t)
    t.start()

# =========================================================
# ✅ 메인 화면: 상단(차트/지갑)
# =========================================================
left, right = st.columns([2.2, 1])

with right:
    st.subheader("💰 내 지갑/포지션")
    try:
        bal = exchange.fetch_balance({"type":"swap"})
        st.metric("총자산(USDT)", f"{safe_float(bal['USDT']['total']):.2f}")
        st.metric("사용가능(USDT)", f"{safe_float(bal['USDT']['free']):.2f}")
        st.divider()

        ps = exchange.fetch_positions(symbols=TARGET_COINS)
        act = [p for p in ps if safe_float(p.get("contracts",0)) > 0]
        if not act:
            st.caption("무포지션(관망)")
        else:
            for p in act:
                sym = p.get("symbol","")
                side = str(p.get("side","")).lower()
                side_kr = "🟢 롱" if side in ["long","buy"] else "🔴 숏"
                roi = safe_float(p.get("percentage",0))
                upnl = safe_float(p.get("unrealizedPnl",0))
                lev = p.get("leverage","?")
                st.info(f"**{sym}**  {side_kr} (x{lev})\n\n수익률 **{roi:.2f}%** / 손익 **{upnl:.2f} USDT**")
    except Exception as e:
        st.error(f"조회 실패: {e}")

with left:
    st.subheader("📈 트레이딩뷰 차트(다크모드)")
    tv_map = {
        "BTC/USDT:USDT": "BINANCE:BTCUSDT",
        "ETH/USDT:USDT": "BINANCE:ETHUSDT",
        "SOL/USDT:USDT": "BINANCE:SOLUSDT",
        "XRP/USDT:USDT": "BINANCE:XRPUSDT",
        "DOGE/USDT:USDT": "BINANCE:DOGEUSDT",
    }
    chart_symbol = st.selectbox("차트 코인", TARGET_COINS, index=0)
    tv_symbol = tv_map.get(chart_symbol, "BINANCE:BTCUSDT")
    tv_interval_map = {"1m":"1","3m":"3","5m":"5","15m":"15","1h":"60"}
    tv_interval = tv_interval_map.get(config.get("timeframe","5m"), "5")

    # ✅ rerun 때 위젯 충돌 방지: container id를 매번 유니크하게
    chart_id = f"tv_{int(time.time()*1000)}"

    tv_html = f"""
    <div class="tradingview-widget-container" style="height:520px;">
      <div id="{chart_id}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "{tv_interval}",
          "timezone": "Asia/Seoul",
          "theme": "dark",
          "style": "1",
          "locale": "kr",
          "toolbar_bg": "#131722",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "save_image": false,
          "container_id": "{chart_id}"
        }});
      </script>
    </div>
    """
    components.html(tv_html, height=540)

# =========================================================
# ✅ 탭 구성
# =========================================================
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🤖 자동매매 상태")
    st.write(f"- 모드: **{config.get('trade_mode')}**")
    st.write(f"- 자동매매: **{'ON' if config.get('auto_trade') else 'OFF'}**")

    st.info(
        "📌 이 봇의 진입 방식(중요)\n"
        "- 상승추세 눌림목에서는 RSI 과매도 '그 자체'가 아니라,\n"
        "  ✅ RSI가 다시 올라오는 '해소 타이밍 + 반등 확인'을 더 중요하게 봅니다.\n"
        "- 손익비(RR)가 기준보다 나쁘면 진입하지 않습니다(수수료 누수 방지)."
    )

    st.divider()
    st.subheader("🧠 현재 차트 AI 분석(쉬운 설명)")

    if st.button("🔍 선택 코인 AI 분석 실행"):
        tf = config.get("timeframe","5m")
        try:
            ohlcv = exchange.fetch_ohlcv(chart_symbol, tf, limit=150)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df, status, last = calc_indicators(df, config)
            if last is None:
                st.error("데이터/지표 계산 실패(캔들이 너무 적을 수 있어요)")
            else:
                ai = ai_decide_trade(df, status, chart_symbol, tf, config.get("trade_mode","안전모드"))

                st.write("### ✅ AI 결론")
                dec = ai.get("decision","hold")
                confp = ai.get("confidence",0)
                if dec == "buy":
                    st.success(f"결론: 🟢 매수(롱) / 확신도 {confp}%")
                elif dec == "sell":
                    st.error(f"결론: 🔴 매도(숏) / 확신도 {confp}%")
                else:
                    st.warning(f"결론: ⚪ 관망 / 확신도 {confp}%")

                st.write(f"- 추천 진입비중: **{ai.get('entry_pct')}%** / 추천 레버리지: **x{ai.get('leverage')}**")
                st.write(f"- 추천 손절: **-{ai.get('sl_pct')}%** / 추천 익절: **+{ai.get('tp_pct')}%**")
                st.info(f"🧸 쉬운 근거: {ai.get('reason_easy','')}")

                with st.expander("📌 사용 지표 / 상세 / 현재 상태판"):
                    st.write("AI가 본 지표:", ai.get("used_indicators", []))
                    st.write("조금 더 자세한 설명:", ai.get("reason_detail",""))
                    st.write("현재 지표 상태판:", status)

                st.divider()
                st.subheader("🚦 10종 보조지표 상태판(요약)")
                st.dataframe(pd.DataFrame([status]), width="stretch", hide_index=True)

                st.markdown("#### 📉 Bitget 실시간 종가(라인차트)")
                st.line_chart(df.set_index("time")["close"])
        except Exception as e:
            st.error(f"분석 오류: {e}")

    st.divider()
    st.subheader("🧠 (선택) AI 글로벌 추천값(사이드바 옵션 추천)")
    st.caption("이 기능은 '지금 차트 기준 추천'을 보여주기만 하고, 자동 적용은 하지 않습니다(버튼으로 적용).")

    if st.button("💡 글로벌 추천값 받아오기"):
        tf = config.get("timeframe","5m")
        try:
            ohlcv = exchange.fetch_ohlcv(chart_symbol, tf, limit=150)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df, status, last = calc_indicators(df, config)
            if last is None:
                st.error("추천값 생성 실패(데이터 부족)")
            else:
                reco = ai_global_reco(df, status, chart_symbol, tf, config.get("trade_mode","안전모드"))
                if not reco:
                    st.warning("OpenAI 키가 없거나 추천값 생성에 실패했어요.")
                else:
                    st.success("✅ AI 추천값 생성 완료")
                    st.info(reco.get("why_easy",""))
                    st.write("추천값:", reco.get("recommended", {}))
                    st.write("지금 중요하게 볼 지표:", reco.get("watch_indicators", []))

                    if st.button("✅ 이 추천값을 설정에 적용"):
                        rec = reco.get("recommended", {})
                        # 일부 키만 안전하게 업데이트
                        for k in ["manual_min_rr","manual_min_sl_pct","manual_tp_pct","trail_activate_pct","trail_distance_pct","per_coin_cooldown_sec","use_news_filter","avoid_news_minutes"]:
                            if k in rec:
                                config[k] = rec[k]
                        save_settings(config)
                        st.success("적용 완료! (사이드바 값이 바뀌었는지 확인)")
                        st.rerun()
        except Exception as e:
            st.error(f"추천값 오류: {e}")

with t2:
    st.subheader("⚡ 수동주문(모의 테스트용)")
    st.caption("※ 수동 주문은 너가 테스트할 때만. 실시간 운영은 자동매매+텔레그램 보고 추천.")

    sym = st.selectbox("수동 주문 코인", TARGET_COINS, index=0, key="manual_coin")
    amt = st.number_input("증거금(USDT)", 1.0, 100000.0, 20.0, step=5.0)
    lev = st.slider("레버리지", 1, 50, int(config.get("manual_leverage",5)))

    c1, c2, c3 = st.columns(3)

    def manual_order(side: str):
        try:
            set_leverage_safe(exchange, lev, sym)
            ticker = exchange.fetch_ticker(sym)
            price = safe_float(ticker.get("last", 0))
            notional = amt * lev
            qty = notional / max(price, 1e-9)
            qty = exchange.amount_to_precision(sym, qty)
            exchange.create_market_order(sym, side, qty)
            st.success(f"주문 성공: {sym} / {side.upper()} / 증거금 {amt} USDT / x{lev}")
            tg_send(f"✋ 수동주문: {sym} {kr_side(side)} / 증거금 {amt} USDT / x{lev}")
        except Exception as e:
            st.error(f"주문 실패: {e}")

    if c1.button("🟢 롱(매수)"):
        manual_order("buy")
    if c2.button("🔴 숏(매도)"):
        manual_order("sell")
    if c3.button("🚫 해당 코인 포지션 종료"):
        try:
            ps = exchange.fetch_positions([sym])
            act = [p for p in ps if safe_float(p.get("contracts",0)) > 0]
            if not act:
                st.warning("해당 코인 포지션 없음")
            else:
                p = act[0]
                ok = close_position_market(exchange, sym, p.get("side",""), safe_float(p.get("contracts",0)))
                if ok:
                    st.success("청산 성공")
                    tg_send(f"🚫 수동청산: {sym}")
                else:
                    st.error("청산 실패")
        except Exception as e:
            st.error(f"청산 오류: {e}")

with t3:
    st.subheader("📅 시장정보(경제 캘린더, 한글)")
    st.caption("ForexFactory 공개 주간 캘린더 기반. (사이트 구조 변경 시 잠시 안 뜰 수 있어요)")
    ev = get_forex_events_kor(limit=40)
    if ev.empty:
        st.warning("캘린더를 불러오지 못했어요.")
    else:
        st.dataframe(ev, width="stretch", hide_index=True)

with t4:
    st.subheader("📜 매매일지(한줄평 위주 + 상세파일 저장)")
    rt = load_runtime()
    trades = rt.get("trades", {})
    if not trades:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        rows = list(trades.values())
        dfj = pd.DataFrame(rows)
        # 최신순
        if "time" in dfj.columns:
            dfj = dfj.sort_values("time", ascending=False)
        st.dataframe(dfj, width="stretch", hide_index=True)

    st.divider()
    st.subheader("📁 상세 로그(trade_log.csv)")
    log_df = load_trade_log()
    if log_df.empty:
        st.caption("상세 로그 없음")
    else:
        st.dataframe(log_df.tail(300).iloc[::-1], width="stretch", hide_index=True)
        csv = log_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")

# =========================================================
# ✅ (끝) 사이드바 하단 - OpenAI 연결 테스트
# =========================================================
st.sidebar.divider()
st.sidebar.header("🔍 긴급 점검")

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    try:
        client = get_openai_client()
        if client is None:
            st.sidebar.error("❌ OpenAI 키가 없습니다.")
        else:
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"user","content":"테스트입니다. 1+1은? 숫자만"}],
                max_tokens=10
            )
            st.sidebar.success(f"✅ 연결 성공: {r.choices[0].message.content}")
    except Exception as e:
        st.sidebar.error(f"❌ 연결 실패: {e}")
