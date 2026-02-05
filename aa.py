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
IS_SANDBOX = True  # 실전매매면 False
SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
TRADE_LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="Bitget AI 워뇨띠 봇 (Streamlit=제어판 / Telegram=보고)")

TARGET_COINS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"]

# =========================================================
# 모드 룰 (여기만 바꾸면 성격이 바로 바뀜)
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
        "entry_pct_min": 8,     # ✅ 공격모드: 최소 8%부터 (너가 원한 “공격”)
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
# 유틸
# =========================================================
def now_kst_str():
    # 서버가 UTC일 수도 있어서 "표시용"으로만 쓰는 간단 처리
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S (KST)")

def kr_side_from_order(decision: str) -> str:
    return "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def tg_send(text: str, tg_token: str, tg_id: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_id, "text": text}
        )
    except:
        pass

# =========================================================
# 설정 저장/로드
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",
        "timeframe": "5m",
        "enforce_mode_rules": True,   # ✅ 모드 최소/최대 강제
        "ai_journal_on_close": True,  # ✅ 청산 시 AI 회고 작성
        "ai_global_reco_auto_apply": False,  # ✅ AI가 ‘글로벌 추천값’을 자동으로 적용할지
        # 수동 기준값(원하면 너가 직접 관리하는 값)
        "manual_min_rr": 1.8,
        "manual_min_sl_pct": 1.2,
        "manual_tp_pct": 6.0,
        "manual_leverage": 5,
        "manual_entry_pct": 10,
        # 지표 파라미터(기본값)
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "adx_period": 14,
        "ma_fast": 20,
        "ma_slow": 60,
        "atr_period": 14,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default.update(saved)
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
# 런타임 상태 (pause, 연속손실 등) - 너가 보여준 형태 유지
# =========================================================
def default_runtime():
    d = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
    return {
        "date": d,
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {}  # trade_id -> info
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
    return rt

def save_runtime(rt):
    try:
        with open(RUNTIME_FILE, "w", encoding="utf-8") as f:
            json.dump(rt, f, ensure_ascii=False, indent=2)
    except:
        pass

runtime_state = load_runtime()

def reset_journal_files():
    # 매매일지 초기화
    for f in [RUNTIME_FILE, TRADE_LOG_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass

# =========================================================
# 매매 로그(CSV)
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

# =========================================================
# Bitget / Secrets
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. (Streamlit Secrets 설정)")
    st.stop()
if not tg_token or not tg_id:
    st.error("🚨 Telegram TOKEN/CHAT_ID가 없습니다. (Streamlit Secrets 설정)")
    st.stop()
if not openai_key:
    st.warning("⚠️ OpenAI API Key가 없습니다. (AI 분석/회고 기능이 꺼집니다)")

# =========================================================
# 거래소 연결
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
# 지표 계산 (ta 라이브러리 없이 직접 계산)
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
    high = df["high"]
    low = df["low"]
    close = df["close"]
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
    if df is None or df.empty or len(df) < 80:
        return df, {}, None

    df = df.copy()
    df["RSI"] = rsi(df["close"], conf["rsi_period"])
    bb_mid, bb_u, bb_l = bollinger(df["close"], conf["bb_period"], conf["bb_std"])
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bb_mid, bb_u, bb_l

    df["MA_fast"] = df["close"].rolling(conf["ma_fast"]).mean()
    df["MA_slow"] = df["close"].rolling(conf["ma_slow"]).mean()

    df["EMA12"] = ema(df["close"], 12)
    df["EMA26"] = ema(df["close"], 26)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = ema(df["MACD"], 9)

    df["ATR"] = atr(df, conf["atr_period"])
    adx_v, pdi, mdi = adx(df, conf["adx_period"])
    df["ADX"], df["+DI"], df["-DI"] = adx_v, pdi.values, mdi.values

    df["CCI"] = cci(df, 20)
    k, d = stochastic(df, 14, 3)
    df["StochK"], df["StochD"] = k, d
    df["WillR"] = williams_r(df, 14)
    df["MFI"] = mfi(df, 14)

    df["VolSMA"] = df["vol"].rolling(20).mean()
    df["VolSpike"] = df["vol"] > (df["VolSMA"] * 2.0)

    df = df.dropna()
    if df.empty:
        return df, {}, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    status = {}
    # 추세(간단)
    trend_up = last["MA_fast"] > last["MA_slow"]
    status["추세"] = "상승추세(위로 가는 흐름)" if trend_up else "하락추세(아래로 가는 흐름)"

    # RSI
    if last["RSI"] < 30:
        status["RSI"] = f"과매도(너무 많이 내려온 상태) {last['RSI']:.1f}"
    elif last["RSI"] > 70:
        status["RSI"] = f"과매수(너무 많이 오른 상태) {last['RSI']:.1f}"
    else:
        status["RSI"] = f"중립 {last['RSI']:.1f}"

    # RSI 해소(중요!)
    status["RSI_흐름"] = f"{prev['RSI']:.1f} → {last['RSI']:.1f} (지금 올라오는지/내려오는지 확인)"

    # 볼밴
    if last["close"] < last["BB_lower"]:
        status["볼린저"] = "하단 이탈(과하게 눌림 가능)"
    elif last["close"] > last["BB_upper"]:
        status["볼린저"] = "상단 돌파(과열 가능)"
    else:
        status["볼린저"] = "밴드 안(평균 범위)"

    # ADX
    status["추세강도(ADX)"] = f"{last['ADX']:.1f} " + ("(추세 강함)" if last["ADX"] >= 25 else "(횡보/약함)")

    # MACD
    status["MACD"] = "상승 신호(골든 느낌)" if last["MACD"] > last["MACD_signal"] else "하락 신호(데드 느낌)"

    # 기타
    status["거래량"] = "거래량 급증(관심 필요)" if bool(last["VolSpike"]) else "평균 수준"
    status["MFI"] = f"{last['MFI']:.1f}(자금흐름)"
    status["CCI"] = f"{last['CCI']:.1f}(과열/침체 힌트)"
    status["Stoch"] = f"{last['StochK']:.1f}/{last['StochD']:.1f}(단기 과열 힌트)"
    status["WillR"] = f"{last['WillR']:.1f}(단기 과열 힌트)"

    # 눌림목/반등 조건(너가 말한 문제를 막는 1차 필터)
    # 상승추세 + RSI가 과매도였다가 다시 올라오는 순간을 더 중요하게 보기
    rsi_cross_up = (prev["RSI"] < 30) and (last["RSI"] >= 30)
    rsi_turn_up = last["RSI"] > prev["RSI"]
    status["_필터_눌림목반등후보"] = bool(trend_up and (prev["RSI"] < 35) and rsi_turn_up)

    status["_필터_RSI해소돌파"] = bool(rsi_cross_up)

    return df, status, last

# =========================================================
# AI: 결정 + 쉬운 설명 + 지표 사용내역
# =========================================================
def openai_client():
    if not openai_key:
        return None
    try:
        return OpenAI(api_key=openai_key)
    except:
        return None

def ai_decide_trade(df, status, symbol, timeframe, mode_name):
    """
    AI가:
    - 진입/관망/반대
    - 확신도
    - 진입비중(%), 레버리지, 손절/익절%
    - 쉬운 근거(한글)
    - 사용한 지표 목록
    을 JSON으로 반환
    """
    client = openai_client()
    if client is None:
        return {
            "decision": "hold",
            "confidence": 0,
            "entry_pct": config.get("manual_entry_pct", 10),
            "leverage": config.get("manual_leverage", 5),
            "sl_pct": config.get("manual_min_sl_pct", 1.2),
            "tp_pct": config.get("manual_tp_pct", 6.0),
            "reason_easy": "OpenAI 키가 없어서 AI 분석을 건너뛰었어요. (수동값으로만 동작)",
            "used_indicators": ["RSI", "볼린저", "이동평균", "ADX", "MACD"]
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    mode_rule = MODE_RULES.get(mode_name, MODE_RULES["안전모드"])

    trend = status.get("추세", "")
    adx_txt = status.get("추세강도(ADX)", "")
    rsi_flow = status.get("RSI_흐름", "")

    # “짧은 손절 + 추세 맞으면 익절 길게”를 위해 ATR 기반 최소 손절 추천 힌트 제공
    atr_pct = float(last["ATR"] / last["close"] * 100)

    system_prompt = f"""
너는 '워뇨띠 스타일'을 기본으로 하는 선별형 트레이더야.
목표는: 원금 손실은 짧게 끊고(손절은 짧게), 추세가 맞으면 익절은 길게 가져가는 것.

[중요: 너가 반드시 고쳐야 하는 버릇]
- 상승추세에서 RSI가 과매도면 "눌림목" 가능성이 커.
  ❌ RSI가 과매도라고 바로 진입하지 말고,
  ✅ RSI가 과매도에서 '해소(다시 올라오는 순간)' + 반등 확인이 있을 때 진입해.

[모드: {mode_name}]
- 이 모드의 최소 확신도: {mode_rule["min_conf"]}%
- 진입비중(잔고 대비 %): {mode_rule["entry_pct_min"]}~{mode_rule["entry_pct_max"]}
- 레버리지: {mode_rule["lev_min"]}~{mode_rule["lev_max"]}

[손절/익절 아이디어]
- 손절이 너무 작으면 휩쏘(개미털기)에 자주 맞아.
- ATR(변동성) 기반으로 "현재 시장에서 의미 있는 최소 손절폭"을 같이 고려해.
- 익절은 추세가 강할수록 더 길게 보는 편이 좋아. (단, 손익비가 좋아야 함)

[출력(JSON, 한글)]
반드시 아래 키를 모두 포함해:
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0~100,
  "entry_pct": 숫자,
  "leverage": 숫자,
  "sl_pct": 숫자,
  "tp_pct": 숫자,
  "reason_easy": "초등학생도 이해할 쉬운 한국어로(괄호로 풀어쓰기)",
  "reason_detail": "조금 더 자세히",
  "used_indicators": ["RSI(14)", "볼린저(20,2)", ...]
}}
"""

    user_prompt = f"""
[차트] {symbol} / {timeframe}
- 현재가: {last["close"]:.4f}
- 추세: {trend}
- ADX: {adx_txt}
- RSI 흐름: {rsi_flow}
- 볼린저: {status.get("볼린저", "")}
- MACD: {status.get("MACD", "")}
- 변동성(ATR%): 약 {atr_pct:.2f}% (이 값이 크면 손절을 너무 좁게 잡으면 잘 털려)

[주의]
- 확신도는 쉽게 90 주지 마. 정말 좋은 자리일 때만 높게.
- 상승추세 눌림목이면 "RSI 해소"나 "반등 확인" 없이 바로 잡지 마.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.25
        )
        out = json.loads(resp.choices[0].message.content)
        return out
    except Exception as e:
        return {
            "decision": "hold",
            "confidence": 0,
            "entry_pct": config.get("manual_entry_pct", 10),
            "leverage": config.get("manual_leverage", 5),
            "sl_pct": config.get("manual_min_sl_pct", 1.2),
            "tp_pct": config.get("manual_tp_pct", 6.0),
            "reason_easy": f"AI 호출 오류라서 관망으로 처리했어요. (에러: {str(e)[:120]})",
            "reason_detail": "오류로 인해 안전하게 HOLD",
            "used_indicators": ["RSI", "볼린저", "이동평균", "ADX", "MACD"]
        }

def ai_write_journal(trade_summary: dict):
    """
    청산 후: 한줄평 + 다음 개선점을 쉬운 한국어로 작성
    """
    client = openai_client()
    if client is None:
        return {"one_liner": "AI 키 없음: 수동 기록", "next_time": "다음엔 손절/익절 기준을 더 명확히"}

    system_prompt = """
너는 매매 코치야. 아래 매매 결과를 보고,
1) 한줄평(아주 쉽게)
2) 다음엔 어떻게 개선할지(아주 쉽게)
를 한국어로 작성해.
어려운 용어는 (괄호로 쉬운 말)로 풀어서 써.
JSON으로만 답해.
{"one_liner":"...", "next_time":"..."}
"""
    user_prompt = json.dumps(trade_summary, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(resp.choices[0].message.content)
    except:
        return {"one_liner": "기록 생성 실패", "next_time": "다음엔 진입 근거를 더 선명하게"}

# =========================================================
# 경제 캘린더 (한글로 보기)
# - ForexFactory 주간 캘린더 JSON 사용
# =========================================================
def get_forex_events_kor(limit=20):
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()

        # 데이터 구조가 리스트/딕트 둘 다 안전 처리
        events = None
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            # 흔한 키들
            for k in ["events", "data", "calendar", "result"]:
                if k in data and isinstance(data[k], list):
                    events = data[k]
                    break
        if events is None:
            return pd.DataFrame(columns=["일시", "국가", "중요도", "지표", "예상", "이전", "실제"])

        rows = []
        for e in events:
            title = str(e.get("title", e.get("event", "")))
            country = str(e.get("country", e.get("currency", "")))
            impact = str(e.get("impact", e.get("importance", "")))

            date_s = str(e.get("date", ""))
            time_s = str(e.get("time", ""))

            # 시간 파싱 (원본이 타임존을 명확히 안 주는 경우가 있어 안전하게 표시)
            dt_display = f"{date_s} {time_s}".strip()

            forecast = str(e.get("forecast", ""))
            previous = str(e.get("previous", ""))
            actual = str(e.get("actual", ""))

            # 한글 컬럼으로 정리
            rows.append({
                "일시": dt_display,
                "국가": country,
                "중요도": impact,
                "지표": title,
                "예상": forecast,
                "이전": previous,
                "실제": actual
            })

        df = pd.DataFrame(rows)
        return df.head(limit)
    except:
        return pd.DataFrame(columns=["일시", "국가", "중요도", "지표", "예상", "이전", "실제"])

# =========================================================
# 텔레그램 봇 스레드 (모든 보고는 여기로)
# =========================================================
def telegram_thread(ex):
    tg_send("🚀 봇 시작! (Streamlit=제어판 / Telegram=보고)\n"
            f"- 샌드박스: {'ON(모의)' if IS_SANDBOX else 'OFF(실전)'}\n"
            f"- 시간: {now_kst_str()}",
            tg_token, tg_id)

    active_trades = {}  # symbol -> dict(sl,tp,entry_pct,lev,open_time,trade_id)
    offset = 0
    last_ping = time.time()

    while True:
        try:
            cur_conf = load_settings()
            mode_name = cur_conf.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode_name, MODE_RULES["안전모드"])

            # 1) 텔레그램 콜백 처리(요청)
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
                                tg_send(
                                    "📌 사용 가능한 명령\n"
                                    "- 잔고\n- 포지션\n- 매매일지\n- 캘린더\n- 상태\n",
                                    tg_token, tg_id
                                )
                            elif txt == "잔고":
                                bal = ex.fetch_balance({"type": "swap"})
                                total = safe_float(bal["USDT"]["total"])
                                free = safe_float(bal["USDT"]["free"])
                                tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT", tg_token, tg_id)
                            elif txt == "포지션":
                                ps = ex.fetch_positions(symbols=TARGET_COINS)
                                active = [p for p in ps if safe_float(p.get("contracts", 0)) > 0]
                                if not active:
                                    tg_send("📊 현재 포지션: 없음(관망)", tg_token, tg_id)
                                else:
                                    msg = "📊 현재 포지션\n"
                                    for p in active:
                                        sym = p.get("symbol", "")
                                        side = p.get("side", "")
                                        roi = safe_float(p.get("percentage", 0))
                                        upnl = safe_float(p.get("unrealizedPnl", 0))
                                        lev = p.get("leverage", "?")
                                        side_kr = "롱" if str(side).lower() in ["long", "buy"] else "숏"
                                        msg += f"- {sym} / {side_kr} / 레버리지 x{lev} / 수익률 {roi:.2f}% / 손익 {upnl:.2f} USDT\n"
                                    tg_send(msg, tg_token, tg_id)
                            elif txt == "매매일지":
                                rt = load_runtime()
                                trades = rt.get("trades", {})
                                if not trades:
                                    tg_send("📜 매매일지: 아직 기록 없음", tg_token, tg_id)
                                else:
                                    # 최근 10개만
                                    items = list(trades.values())[-10:]
                                    msg = "📜 최근 매매일지(한줄평)\n"
                                    for t in items[::-1]:
                                        msg += f"- {t.get('time','')} {t.get('symbol','')} {t.get('result','')} | {t.get('one_liner','')}\n"
                                    tg_send(msg, tg_token, tg_id)
                            elif txt == "캘린더":
                                df_ev = get_forex_events_kor(limit=15)
                                if df_ev.empty:
                                    tg_send("📅 경제 캘린더: 가져오기 실패/없음", tg_token, tg_id)
                                else:
                                    # 너무 길면 잘라서 보내기
                                    lines = ["📅 이번주 경제 캘린더(요약)"]
                                    for _, r in df_ev.iterrows():
                                        lines.append(f"- {r['일시']} / {r['국가']} / {r['중요도']} / {r['지표']}")
                                    tg_send("\n".join(lines[:25]), tg_token, tg_id)
                            elif txt == "상태":
                                tg_send(f"✅ 상태\n- 모드: {mode_name}\n- 자동매매: {'ON' if cur_conf.get('auto_trade') else 'OFF'}\n- 시간: {now_kst_str()}",
                                        tg_token, tg_id)
            except:
                pass

            # 2) 자동매매 루프
            if cur_conf.get("auto_trade", False):
                for coin in TARGET_COINS:
                    try:
                        # 포지션 확인
                        positions = ex.fetch_positions([coin])
                        pos_list = [p for p in positions if safe_float(p.get("contracts", 0)) > 0]

                        # (A) 포지션이 있으면 청산 조건 체크
                        if pos_list:
                            p = pos_list[0]
                            side = str(p.get("side", "")).lower()
                            roi = safe_float(p.get("percentage", 0))
                            upnl = safe_float(p.get("unrealizedPnl", 0))
                            entry_price = safe_float(p.get("entryPrice", 0))
                            contracts = safe_float(p.get("contracts", 0))

                            tinfo = active_trades.get(coin, None)
                            if tinfo is None:
                                # 없으면 안전한 기본
                                tinfo = {"sl_pct": 2.0, "tp_pct": 6.0, "entry_pct": 0, "lev": p.get("leverage", "?"), "trade_id": None}

                            sl_pct = abs(safe_float(tinfo.get("sl_pct", 2.0)))
                            tp_pct = abs(safe_float(tinfo.get("tp_pct", 6.0)))

                            # 손절/익절 도달 시 반대주문으로 청산
                            if roi <= -sl_pct or roi >= tp_pct:
                                close_side = "sell" if side in ["long", "buy"] else "buy"
                                ex.create_market_order(coin, close_side, contracts)

                                # 회고/기록
                                result = "익절(수익)" if roi >= tp_pct else "손절(손실)"
                                trade_summary = {
                                    "time": now_kst_str(),
                                    "symbol": coin,
                                    "result": result,
                                    "roi_pct": roi,
                                    "pnl_usdt": upnl,
                                    "entry_price": entry_price,
                                    "mode": mode_name,
                                    "note": "자동 청산"
                                }

                                one = {"one_liner": "", "next_time": ""}
                                if cur_conf.get("ai_journal_on_close", True):
                                    one = ai_write_journal(trade_summary)

                                # runtime_state 업데이트
                                rt = load_runtime()
                                t_id = tinfo.get("trade_id") or f"{int(time.time())}_{coin.replace('/','_')}"
                                rt["trades"][t_id] = {
                                    "time": trade_summary["time"],
                                    "symbol": coin,
                                    "result": result,
                                    "roi_pct": roi,
                                    "pnl_usdt": upnl,
                                    "one_liner": one.get("one_liner", ""),
                                    "next_time": one.get("next_time", "")
                                }
                                save_runtime(rt)

                                # CSV 로그(상세)
                                append_trade_log({
                                    "Time": trade_summary["time"],
                                    "Symbol": coin,
                                    "Mode": mode_name,
                                    "Result": result,
                                    "ROI_percent": roi,
                                    "PnL_USDT": upnl,
                                    "EntryPrice": entry_price,
                                    "CloseType": "AUTO",
                                    "OneLiner": one.get("one_liner", ""),
                                    "NextTime": one.get("next_time", "")
                                })

                                # 텔레그램 보고(한글/쉬운 말)
                                tg_send(
                                    "📌 청산 알림\n"
                                    f"- 코인: {coin}\n"
                                    f"- 결과: {result}\n"
                                    f"- 손익: {upnl:.2f} USDT\n"
                                    f"- 수익률: {roi:.2f}%\n"
                                    f"- 한줄평: {one.get('one_liner','')}\n"
                                    f"- 다음엔: {one.get('next_time','')}",
                                    tg_token, tg_id
                                )

                                if coin in active_trades:
                                    del active_trades[coin]
                            continue

                        # (B) 포지션 없으면 신규 진입 분석
                        ohlcv = ex.fetch_ohlcv(coin, cur_conf.get("timeframe", "5m"), limit=120)
                        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                        df["time"] = pd.to_datetime(df["time"], unit="ms")
                        df, status, last = calc_indicators(df, cur_conf)
                        if last is None:
                            continue

                        # ✅ “눌림목 반등” 필터(너가 말한 반복손절 문제를 줄이기 위한 1차 방어)
                        # - 상승추세에서 RSI가 그냥 과매도라고 바로 진입하지 않도록,
                        #   RSI가 "되돌아오는 흐름"이 있을 때만 AI를 적극 호출
                        call_ai = True
                        if status.get("_필터_눌림목반등후보") or status.get("_필터_RSI해소돌파"):
                            call_ai = True
                        else:
                            # 완전 횡보에서 불필요한 진입 줄이기(수수료 누수 방지)
                            # RSI가 중립 + ADX 낮으면 관망 성향
                            if (30 <= safe_float(last["RSI"]) <= 70) and (safe_float(last["ADX"]) < 18):
                                call_ai = False

                        if not call_ai:
                            continue

                        ai = ai_decide_trade(df, status, coin, cur_conf.get("timeframe", "5m"), mode_name)

                        decision = ai.get("decision", "hold")
                        conf = int(safe_float(ai.get("confidence", 0)))

                        # 모드별 최소 확신도
                        if conf < int(rule["min_conf"]):
                            continue
                        if decision not in ["buy", "sell"]:
                            continue

                        # AI 추천값
                        entry_pct = safe_float(ai.get("entry_pct", cur_conf.get("manual_entry_pct", 10)))
                        lev = int(safe_float(ai.get("leverage", cur_conf.get("manual_leverage", 5))))
                        sl_pct = safe_float(ai.get("sl_pct", cur_conf.get("manual_min_sl_pct", 1.2)))
                        tp_pct = safe_float(ai.get("tp_pct", cur_conf.get("manual_tp_pct", 6.0)))

                        # ✅ 모드 룰 강제(공격모드인데 2%만 들어가는 문제 해결 핵심)
                        if cur_conf.get("enforce_mode_rules", True):
                            entry_pct = clamp(entry_pct, rule["entry_pct_min"], rule["entry_pct_max"])
                            lev = int(clamp(lev, rule["lev_min"], rule["lev_max"]))

                        # 손절은 “너무 좁으면 휩쏘” => ATR 기반 최소치 보정 (너가 겪은 1.5% 손절 지옥 방지)
                        atr_pct = safe_float(last["ATR"] / last["close"] * 100)
                        min_sl_from_atr = max(0.6, atr_pct * 0.9)  # 시장 변동성이 크면 손절도 조금 넓혀야 함
                        sl_pct = max(sl_pct, min_sl_from_atr)

                        # 잔고/수량 계산(정확한 USDT 보고용)
                        bal = ex.fetch_balance({"type": "swap"})
                        free_usdt = safe_float(bal["USDT"]["free"])
                        total_usdt = safe_float(bal["USDT"]["total"])

                        margin_usdt = free_usdt * (entry_pct / 100.0)
                        price = safe_float(last["close"])

                        if margin_usdt <= 1:
                            continue

                        # 레버리지 설정
                        try:
                            ex.set_leverage(lev, coin)
                        except:
                            pass

                        # 수량(명목=margin*lev)
                        notional = margin_usdt * lev
                        qty = (notional / price)
                        qty = ex.amount_to_precision(coin, qty)

                        if safe_float(qty) <= 0:
                            continue

                        # 진입
                        ex.create_market_order(coin, decision, qty)

                        # active 저장
                        trade_id = f"{int(time.time())}_{coin.replace('/','_')}"
                        active_trades[coin] = {
                            "sl_pct": sl_pct,
                            "tp_pct": tp_pct,
                            "entry_pct": entry_pct,
                            "lev": lev,
                            "trade_id": trade_id,
                            "open_time": now_kst_str(),
                            "margin_usdt": margin_usdt,
                            "notional": notional,
                            "decision": decision,
                            "confidence": conf,
                            "reason_easy": ai.get("reason_easy", ""),
                            "used_indicators": ai.get("used_indicators", [])
                        }

                        # 텔레그램 보고(전부 한글/쉬운 말)
                        est_tp_price = price * (1 + (tp_pct / 100.0)) if decision == "buy" else price * (1 - (tp_pct / 100.0))
                        est_sl_price = price * (1 - (sl_pct / 100.0)) if decision == "buy" else price * (1 + (sl_pct / 100.0))

                        tg_send(
                            "🚀 진입 알림\n"
                            f"- 모드: {mode_name}\n"
                            f"- 코인: {coin}\n"
                            f"- 방향: {kr_side_from_order(decision)}\n"
                            f"- 확신도: {conf}% (AI가 ‘좋다’고 느낀 정도)\n"
                            f"- 진입 증거금: {margin_usdt:.2f} USDT (잔고의 약 {entry_pct:.1f}%)\n"
                            f"- 포지션 규모(명목): {notional:.2f} USDT (레버리지 x{lev})\n"
                            f"- 목표 익절: +{tp_pct:.2f}% (예상가 {est_tp_price:.4f})\n"
                            f"- 목표 손절: -{sl_pct:.2f}% (예상가 {est_sl_price:.4f})\n"
                            f"- 쉬운 근거: {ai.get('reason_easy','')}\n"
                            f"- AI가 본 지표: {', '.join(ai.get('used_indicators', []))}",
                            tg_token, tg_id
                        )

                        time.sleep(3)

                    except Exception as e:
                        # 코인별 에러는 조용히 넘어감
                        # (너무 많은 에러 보고는 텔레그램/로그를 망침)
                        pass

            # 3) 생존신고(가끔)
            if time.time() - last_ping > 900:
                try:
                    bal = ex.fetch_balance({"type": "swap"})
                    total = safe_float(bal["USDT"]["total"])
                    tg_send(f"💤 생존신고: 총자산 {total:.2f} USDT / 모드={load_settings().get('trade_mode')}", tg_token, tg_id)
                except:
                    pass
                last_ping = time.time()

            time.sleep(1)

        except:
            time.sleep(3)

# =========================================================
# Streamlit UI (제어판)
# =========================================================
st.title("🧠 Bitget AI 워뇨띠 봇")
st.caption("Streamlit은 제어판 / 텔레그램이 모든 보고(진입·청산·일지·상태)")

# 사이드바: 제어판
st.sidebar.title("🛠️ 제어판")

# 모드 선택
mode = st.sidebar.selectbox("매매 모드", list(MODE_RULES.keys()), index=list(MODE_RULES.keys()).index(config.get("trade_mode", "안전모드")))
auto_trade = st.sidebar.checkbox("🤖 자동매매 ON/OFF", value=config.get("auto_trade", False))
timeframe = st.sidebar.selectbox("차트 타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))
enforce_rules = st.sidebar.checkbox("✅ 모드 룰 강제(최소 진입비중/레버리지)", value=config.get("enforce_mode_rules", True))
ai_journal = st.sidebar.checkbox("📝 청산 시 AI 회고 자동작성", value=config.get("ai_journal_on_close", True))

st.sidebar.divider()
st.sidebar.subheader("💰 수동 기준값(너가 관리) + AI는 추천만")
manual_entry_pct = st.sidebar.slider("수동 진입비중(잔고 %)", 1, 50, int(config.get("manual_entry_pct", 10)))
manual_lev = st.sidebar.slider("수동 레버리지", 1, 50, int(config.get("manual_leverage", 5)))
manual_sl = st.sidebar.slider("수동 손절(%)", 0.2, 10.0, float(config.get("manual_min_sl_pct", 1.2)), step=0.1)
manual_tp = st.sidebar.slider("수동 익절(%)", 1.0, 40.0, float(config.get("manual_tp_pct", 6.0)), step=0.5)
manual_rr = st.sidebar.slider("수동 최소 손익비(RR)", 1.0, 5.0, float(config.get("manual_min_rr", 1.8)), step=0.1)

changed = False
if config.get("trade_mode") != mode:
    config["trade_mode"] = mode; changed = True
if config.get("auto_trade") != auto_trade:
    config["auto_trade"] = auto_trade; changed = True
if config.get("timeframe") != timeframe:
    config["timeframe"] = timeframe; changed = True
if config.get("enforce_mode_rules") != enforce_rules:
    config["enforce_mode_rules"] = enforce_rules; changed = True
if config.get("ai_journal_on_close") != ai_journal:
    config["ai_journal_on_close"] = ai_journal; changed = True

for k, v in [
    ("manual_entry_pct", manual_entry_pct),
    ("manual_leverage", manual_lev),
    ("manual_min_sl_pct", manual_sl),
    ("manual_tp_pct", manual_tp),
    ("manual_min_rr", manual_rr),
]:
    if config.get(k) != v:
        config[k] = v
        changed = True

if changed:
    save_settings(config)

st.sidebar.divider()
st.sidebar.subheader("🧹 매매일지 관리")
if st.sidebar.button("🗑️ 매매일지 초기화(런타임+CSV 삭제)"):
    reset_journal_files()
    st.sidebar.success("초기화 완료! 새로고침하면 반영돼요.")

st.sidebar.divider()
st.sidebar.subheader("🔍 긴급 점검")
if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    try:
        if not openai_key:
            st.sidebar.error("❌ OpenAI 키 없음")
        else:
            c = OpenAI(api_key=openai_key)
            r = c.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "테스트: 1+1=? 한국어로 숫자만"}],
                max_tokens=10
            )
            st.sidebar.success(f"✅ 연결 성공: {r.choices[0].message.content}")
    except Exception as e:
        st.sidebar.error(f"❌ 연결 실패: {e}")

# 스레드 실행(텔레그램 봇)
found = any(t.name == "TG_THREAD" for t in threading.enumerate())
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(t)
    t.start()

# =========================================================
# 메인 화면: 차트/지표/포지션/일지
# =========================================================
c_top1, c_top2 = st.columns([2, 1])

with c_top2:
    st.subheader("💰 내 지갑/포지션")
    try:
        bal = exchange.fetch_balance({"type": "swap"})
        st.metric("총자산(USDT)", f"{safe_float(bal['USDT']['total']):.2f}")
        st.metric("사용가능(USDT)", f"{safe_float(bal['USDT']['free']):.2f}")
        st.divider()
        ps = exchange.fetch_positions(symbols=TARGET_COINS)
        active = [p for p in ps if safe_float(p.get("contracts", 0)) > 0]
        if not active:
            st.caption("현재 무포지션(관망)")
        else:
            for p in active:
                sym = p.get("symbol", "")
                side = str(p.get("side", "")).lower()
                side_kr = "🟢 롱" if side in ["long", "buy"] else "🔴 숏"
                roi = safe_float(p.get("percentage", 0))
                upnl = safe_float(p.get("unrealizedPnl", 0))
                lev = p.get("leverage", "?")
                st.info(f"**{sym}**  {side_kr} (x{lev})\n\n수익률 **{roi:.2f}%** / 손익 **{upnl:.2f} USDT**")
    except Exception as e:
        st.error(f"조회 실패: {e}")

with c_top1:
    st.subheader("📈 트레이딩뷰 차트(다크모드)")
    # 트레이딩뷰는 “시각용”이니 거래소 심볼과 100% 일치 안 해도 OK.
    # 가장 안정적으로는 BINANCE 심볼로 표시
    base = "BTCUSDT"
    tv_map = {
        "BTC/USDT:USDT": "BINANCE:BTCUSDT",
        "ETH/USDT:USDT": "BINANCE:ETHUSDT",
        "SOL/USDT:USDT": "BINANCE:SOLUSDT",
        "XRP/USDT:USDT": "BINANCE:XRPUSDT",
        "DOGE/USDT:USDT": "BINANCE:DOGEUSDT",
    }

    symbol_choice = st.selectbox("차트 코인", TARGET_COINS, index=0)
    tv_symbol = tv_map.get(symbol_choice, "BINANCE:BTCUSDT")
    tv_interval_map = {"1m":"1", "3m":"3", "5m":"5", "15m":"15", "1h":"60"}
    tv_interval = tv_interval_map.get(config.get("timeframe","5m"), "5")

    tv_html = f"""
    <div class="tradingview-widget-container" style="height:520px;">
      <div id="tv_chart"></div>
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
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """
    components.html(tv_html, height=540)

# 탭 구성
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🤖 자동매매 상태")
    st.write(f"- 현재 모드: **{config.get('trade_mode')}**")
    st.write(f"- 자동매매: **{'ON' if config.get('auto_trade') else 'OFF'}**")
    rule = MODE_RULES[config.get("trade_mode","안전모드")]
    st.info(
        "📌 모드 규칙 요약\n"
        f"- 최소 확신도: {rule['min_conf']}%\n"
        f"- 진입비중(잔고%): {rule['entry_pct_min']} ~ {rule['entry_pct_max']}\n"
        f"- 레버리지: {rule['lev_min']} ~ {rule['lev_max']}\n"
        "※ ‘모드 룰 강제’가 켜져 있으면 위 범위를 벗어나지 않게 자동 보정돼요."
    )

    st.divider()
    st.subheader("🧠 현재 차트 AI 분석(설명 쉬운 버전)")
    if st.button("🔍 선택한 코인 AI 분석"):
        tf = config.get("timeframe","5m")
        ohlcv = exchange.fetch_ohlcv(symbol_choice, tf, limit=120)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df, status, last = calc_indicators(df, config)
        if last is None:
            st.error("데이터 부족/지표 계산 실패")
        else:
            ai = ai_decide_trade(df, status, symbol_choice, tf, config.get("trade_mode","안전모드"))
            st.write("### ✅ AI 결론")
            st.write(f"- 결정: **{ai.get('decision','hold')}** (buy=롱 / sell=숏 / hold=관망)")
            st.write(f"- 확신도: **{ai.get('confidence',0)}%**")
            st.write(f"- 추천 진입비중: **{ai.get('entry_pct')}%** / 추천 레버리지: **x{ai.get('leverage')}**")
            st.write(f"- 추천 손절: **-{ai.get('sl_pct')}%** / 추천 익절: **+{ai.get('tp_pct')}%**")
            st.info(f"🧸 쉬운 근거: {ai.get('reason_easy','')}")
            with st.expander("📌 사용한 지표 / 상세 근거"):
                st.write("지표:", ai.get("used_indicators", []))
                st.write("상세:", ai.get("reason_detail",""))
                st.write("현재 지표 상태판:", status)

with t2:
    st.subheader("⚡ 수동주문(기본 골격)")
    st.caption("여긴 너가 수동으로 테스트할 때만 쓰고, 자동매매는 텔레그램 보고를 보면서 운영하면 돼.")
    amount = st.number_input("주문 증거금(USDT)", 0.0, 100000.0, 20.0, step=5.0)
    lev = st.slider("레버리지", 1, 50, 5)
    c1, c2, c3 = st.columns(3)
    if c1.button("🟢 롱(매수)"):
        st.info("수동 주문은 너가 원할 때만 추가 구현하면 돼(지금은 골격만).")
    if c2.button("🔴 숏(매도)"):
        st.info("수동 주문은 너가 원할 때만 추가 구현하면 돼(지금은 골격만).")
    if c3.button("🚫 포지션 종료"):
        st.info("수동 청산도 원하면 넣어줄게.")

with t3:
    st.subheader("📅 시장정보(경제 캘린더)")
    st.caption("ForexFactory 주간 캘린더 기반(무료 공개 데이터). 시간대는 ‘원본 기준’이라 약간 차이날 수 있어요.")
    ev = get_forex_events_kor(limit=30)
    if ev.empty:
        st.warning("캘린더를 불러오지 못했어요.")
    else:
        st.dataframe(ev, use_container_width=True, hide_index=True)

with t4:
    st.subheader("📜 매매일지(보기는 한줄평 위주, 파일엔 상세 저장)")
    rt = load_runtime()
    trades = rt.get("trades", {})
    if not trades:
        st.info("아직 기록된 매매가 없어요.")
    else:
        rows = list(trades.values())
        dfj = pd.DataFrame(rows)
        st.dataframe(dfj.iloc[::-1], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📁 상세 로그(trade_log.csv)")
    log_df = load_trade_log()
    if log_df.empty:
        st.caption("상세 로그 아직 없음")
    else:
        st.dataframe(log_df.tail(200).iloc[::-1], use_container_width=True, hide_index=True)
        csv = log_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")
