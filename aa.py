# =========================================================
#  Bitget AI Wonyoti Agent (Final Integrated)
#  - Streamlit: 제어판/차트/포지션/일지/AI 시야
#  - Telegram: 실시간 보고/조회/일지 요약
#  - AutoTrade: 데모(IS_SANDBOX=True) 기반
#
#  ⚠️ 주의: 트레이딩은 손실 위험이 큽니다. (특히 레버리지)
# =========================================================

import os
import json
import time
import uuid
import math
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import requests
import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx

import ccxt
from openai import OpenAI

# ---- optional pip ----
try:
    import ta  # pip: ta
except Exception:
    ta = None

try:
    from streamlit_autorefresh import st_autorefresh  # pip: streamlit-autorefresh
except Exception:
    st_autorefresh = None

# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Final)")

IS_SANDBOX = True  # ✅ 데모/모의투자

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"
BRAIN_DB = "wonyousi_brain.db"  # (선택) 향후 확장

# 감시 대상 코인
TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

# =========================================================
# ✅ 1) 시간 유틸 (KST, timezone-aware) - DeprecationWarning 제거
# =========================================================
from datetime import datetime, timedelta, timezone
KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")


# =========================================================
# ✅ 2) JSON 안전 저장/로드 (원자적)
# =========================================================
def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def read_json_safe(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# =========================================================
# ✅ 3) MODE_RULES (사용자 제공) - 3단계 모드
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
        "entry_pct_min": 8,     # ✅ 공격: 최소 8% ~ 25%
        "entry_pct_max": 25,
        "lev_min": 2,
        "lev_max": 10,          # ✅ 레버는 낮게
    },
    "하이리스크/하이리턴": {
        "min_conf": 85,
        "entry_pct_min": 15,
        "entry_pct_max": 40,
        "lev_min": 8,
        "lev_max": 25,          # ✅ 레버도 높게
    }
}


# =========================================================
# ✅ 4) 설정 관리 (load/save)
# =========================================================
def default_settings() -> Dict[str, Any]:
    return {
        # 공통
        "openai_api_key": "",
        "auto_trade": False,
        "trade_mode": "안전모드",
        "timeframe": "5m",
        "order_usdt": 100.0,

        # 텔레그램
        "tg_enable_reports": True,

        # 10종 지표 파라미터
        "rsi_period": 14,
        "rsi_buy": 30,
        "rsi_sell": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_fast": 7,
        "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 10종 지표 ON/OFF
        "use_rsi": True,
        "use_bb": True,
        "use_cci": True,
        "use_vol": True,
        "use_ma": True,
        "use_macd": True,
        "use_stoch": True,
        "use_mfi": True,
        "use_willr": True,
        "use_adx": True,

        # 방어/자금/전략 옵션
        "use_trailing_stop": True,
        "use_dca": True,
        "dca_trigger": -20.0,
        "dca_max_count": 1,
        "dca_add_pct": 50.0,        # (기본) 추가진입은 원진입의 50% 규모
        "use_switching": True,
        "switch_trigger": -12.0,    # 손실이 커졌는데 반대 시그널 강하면 스위칭

        "no_trade_weekend": False,

        # 연속손실/일시정지
        "loss_pause_enable": True,
        "loss_pause_after": 3,        # 연속 3번 손실이면
        "loss_pause_minutes": 30,     # 30분 정지

        # AI 추천 글로벌옵션
        "ai_reco_show": True,
        "ai_reco_apply": False,  # ✅ ON이면 AI 추천값을 자동으로 config에 반영
        "ai_reco_refresh_sec": 20,  # 추천 갱신 주기(너무 잦으면 비용/지연)

        # AI 출력 쉬운말(한글)
        "ai_easy_korean": True,
    }


def load_settings() -> Dict[str, Any]:
    cfg = default_settings()
    if os.path.exists(SETTINGS_FILE):
        saved = read_json_safe(SETTINGS_FILE, {})
        if isinstance(saved, dict):
            cfg.update(saved)
    # 예전 키 이름 호환
    if "openai_key" in cfg and not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = cfg["openai_key"]
    return cfg


def save_settings(cfg: Dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, cfg)


config = load_settings()


# =========================================================
# ✅ 5) 런타임 상태(runtime_state.json) - 사용자 포맷 유지
# =========================================================
def default_runtime() -> Dict[str, Any]:
    return {
        "date": today_kst_str(),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {}  # 심볼별 dca 횟수 등 저장
    }


def load_runtime() -> Dict[str, Any]:
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()
    # 날짜 rollover
    if rt.get("date") != today_kst_str():
        rt = default_runtime()
    # 필드 보정
    for k, v in default_runtime().items():
        if k not in rt:
            rt[k] = v
    return rt


def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ 6) 매매일지 CSV (상세 저장 + 한줄평 + 후기)
# =========================================================
def log_trade(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_amount: float,
    pnl_percent: float,
    reason: str,
    one_line: str = "",
    review: str = ""
) -> None:
    try:
        row = pd.DataFrame([{
            "Time": now_kst_str(),
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Reason": reason,
            "OneLine": one_line,
            "Review": review,
        }])

        if not os.path.exists(LOG_FILE):
            row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def read_trade_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def reset_trade_log() -> None:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception:
        pass


def get_past_mistakes_text(max_items: int = 5) -> str:
    df = read_trade_log()
    if df.empty or "PnL_Percent" not in df.columns:
        return "과거 매매 기록 없음."
    try:
        worst = df.sort_values("PnL_Percent", ascending=True).head(max_items)
        lines = []
        for _, r in worst.iterrows():
            lines.append(f"- {r.get('Coin','?')} {r.get('Side','?')} {float(r.get('PnL_Percent',0)):.2f}% 손실 | 이유: {str(r.get('Reason',''))[:40]}")
        return "\n".join(lines) if lines else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# =========================================================
# ✅ 7) Secrets (Bitget / Telegram / OpenAI)
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")

tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. Secrets에 API_KEY/API_SECRET/API_PASSWORD 설정하세요.")
    st.stop()

# OpenAI 클라이언트
openai_client = None
if openai_key:
    try:
        openai_client = OpenAI(api_key=openai_key)
    except Exception:
        openai_client = None


# =========================================================
# ✅ 8) 거래소 연결
# =========================================================
@st.cache_resource
def init_exchange():
    try:
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
    except Exception:
        return None


exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키/권한/네트워크 확인.")
    st.stop()


# =========================================================
# ✅ 9) Bitget 헬퍼 (포지션/잔고/수량 정밀)
# =========================================================
def safe_fetch_balance(ex) -> Tuple[float, float]:
    try:
        bal = ex.fetch_balance({"type": "swap"})
        free = float(bal["USDT"]["free"])
        total = float(bal["USDT"]["total"])
        return free, total
    except Exception:
        return 0.0, 0.0


def safe_fetch_positions(ex, symbols: List[str]) -> List[Dict[str, Any]]:
    try:
        return ex.fetch_positions(symbols)
    except TypeError:
        try:
            return ex.fetch_positions(symbols=symbols)
        except Exception:
            return []
    except Exception:
        return []


def get_last_price(ex, sym: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(sym)
        return float(t["last"])
    except Exception:
        return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_precision_qty(ex, sym: str, qty: float) -> float:
    try:
        return float(ex.amount_to_precision(sym, qty))
    except Exception:
        return float(qty)


def set_leverage_safe(ex, sym: str, lev: int) -> None:
    try:
        ex.set_leverage(int(lev), sym)
    except Exception:
        pass


def market_order_safe(ex, sym: str, side: str, qty: float) -> bool:
    try:
        ex.create_market_order(sym, side, qty)
        return True
    except Exception:
        return False


def close_position_market(ex, sym: str, pos_side: str, contracts: float) -> bool:
    # pos_side: long/short OR buy/sell
    if contracts <= 0:
        return False
    if pos_side in ["long", "buy"]:
        return market_order_safe(ex, sym, "sell", contracts)
    return market_order_safe(ex, sym, "buy", contracts)


def position_roi_percent(p: Dict[str, Any]) -> float:
    # ccxt 포지션 dict에서 ROI% 가져오거나 계산
    try:
        if p.get("percentage") is not None:
            return float(p.get("percentage"))
    except Exception:
        pass
    return 0.0


def position_side_normalize(p: Dict[str, Any]) -> str:
    # bitget/ccxt는 side가 long/short 또는 buy/sell로 올 수 있음
    s = (p.get("side") or p.get("positionSide") or "").lower()
    if s in ["long", "buy"]:
        return "long"
    if s in ["short", "sell"]:
        return "short"
    # fallback
    return "long"


# =========================================================
# ✅ 10) TradingView 다크모드 차트
# =========================================================
def tv_symbol_from_ccxt(sym: str) -> str:
    # BTC/USDT:USDT -> BITGET:BTCUSDT.P (가능하면)
    base = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    return f"BITGET:{base}{quote}.P"


def render_tradingview(symbol_ccxt: str, interval="5", height=560) -> None:
    tvsym = tv_symbol_from_ccxt(symbol_ccxt)
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tv_chart" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tvsym}",
          "interval": "{interval}",
          "timezone": "Asia/Seoul",
          "theme": "dark",
          "style": "1",
          "locale": "kr",
          "toolbar_bg": "#0e1117",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "withdateranges": true,
          "save_image": false,
          "container_id": "tv_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height)


# =========================================================
# ✅ 11) 지표 계산 (10종 + 상태요약 + “눌림목 해소” 감지)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.Series]]:
    status: Dict[str, Any] = {}
    if df is None or df.empty or len(df) < 120:
        return df, status, None
    if ta is None:
        status["_ERROR"] = "ta 모듈 없음(requirements.txt에 ta 추가 필요)"
        return df, status, None

    # parameters
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))
    bb_period = int(cfg.get("bb_period", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    ma_fast = int(cfg.get("ma_fast", 7))
    ma_slow = int(cfg.get("ma_slow", 99))
    stoch_k = int(cfg.get("stoch_k", 14))
    vol_mul = float(cfg.get("vol_mul", 2.0))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]

    # RSI
    if cfg.get("use_rsi", True):
        df["RSI"] = ta.momentum.rsi(close, window=rsi_period)

    # Bollinger
    if cfg.get("use_bb", True):
        bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()

    # MA
    if cfg.get("use_ma", True):
        df["MA_fast"] = ta.trend.sma_indicator(close, window=ma_fast)
        df["MA_slow"] = ta.trend.sma_indicator(close, window=ma_slow)

    # MACD
    if cfg.get("use_macd", True):
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

    # Stoch
    if cfg.get("use_stoch", True):
        df["STO_K"] = ta.momentum.stoch(high, low, close, window=stoch_k, smooth_window=3)
        df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=stoch_k, smooth_window=3)

    # CCI
    if cfg.get("use_cci", True):
        df["CCI"] = ta.trend.cci(high, low, close, window=20)

    # MFI
    if cfg.get("use_mfi", True):
        df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)

    # Williams %R
    if cfg.get("use_willr", True):
        df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # ADX
    if cfg.get("use_adx", True):
        df["ADX"] = ta.trend.adx(high, low, close, window=14)

    # Volume spike
    if cfg.get("use_vol", True):
        df["VOL_MA"] = vol.rolling(20).mean()
        df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * vol_mul)).astype(int)

    df = df.dropna()
    if df.empty or len(df) < 5:
        return df, status, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- status text (Korean) ----
    used = []

    # RSI status
    if cfg.get("use_rsi", True):
        used.append("RSI")
        rsi_now = float(last.get("RSI", 50))
        if rsi_now < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_now:.1f})"
        elif rsi_now > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_now:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_now:.1f})"

    # Bollinger
    if cfg.get("use_bb", True):
        used.append("볼린저밴드")
        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

    # MA trend
    trend = "중립"
    if cfg.get("use_ma", True):
        used.append("이동평균(MA)")
        if last["MA_fast"] > last["MA_slow"] and last["close"] > last["MA_slow"]:
            trend = "상승추세"
        elif last["MA_fast"] < last["MA_slow"] and last["close"] < last["MA_slow"]:
            trend = "하락추세"
        else:
            trend = "횡보/전환"
        status["추세"] = f"📈 {trend}"

    # MACD
    if cfg.get("use_macd", True):
        used.append("MACD")
        status["MACD"] = "📈 상승(골든)" if last["MACD"] > last["MACD_signal"] else "📉 하락(데드)"

    # ADX
    if cfg.get("use_adx", True):
        used.append("ADX(추세강도)")
        adx = float(last.get("ADX", 0))
        status["ADX"] = "🔥 추세 강함" if adx >= 25 else "💤 추세 약함"

    # volume
    if cfg.get("use_vol", True):
        used.append("거래량")
        status["거래량"] = "🔥 거래량 급증" if int(last.get("VOL_SPIKE", 0)) == 1 else "⚪ 보통"

    # ---- 핵심: “과매도에 바로 진입” 방지 -> “해소 시점(반등/반락 확인)” ----
    rsi_prev = float(prev.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0
    rsi_now = float(last.get("RSI", 50)) if cfg.get("use_rsi", True) else 50.0

    rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
    rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

    # 눌림목 후보: 상승추세 + 과매도 해소 + (ADX 너무 약하지 않음)
    adx_now = float(last.get("ADX", 0)) if cfg.get("use_adx", True) else 0.0
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_now >= 18)

    status["_used_indicators"] = used
    status["_rsi_resolve_long"] = bool(rsi_resolve_long)
    status["_rsi_resolve_short"] = bool(rsi_resolve_short)
    status["_pullback_candidate"] = bool(pullback_candidate)

    return df, status, last


# =========================================================
# ✅ 12) AI 판단 (한글 쉬운 설명 + used_indicators 포함)
# =========================================================
def ai_decide_trade(
    df: pd.DataFrame,
    status: Dict[str, Any],
    symbol: str,
    mode: str,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    반환 예:
    {
      decision: buy/sell/hold,
      confidence: 0~100,
      entry_pct: 잔고 대비 진입비중(%),
      leverage: 레버리지,
      sl_pct: 손절(%) (ROI 기준),
      tp_pct: 익절(%) (ROI 기준),
      rr: 손익비,
      used_indicators: [...],
      reason_easy: 쉬운 한글
    }
    """
    # OpenAI 없으면 hold
    if openai_client is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "OpenAI 키 없음", "used_indicators": status.get("_used_indicators", [])}

    if df is None or df.empty or status is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "데이터 부족", "used_indicators": status.get("_used_indicators", [])}

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    last = df.iloc[-1]
    prev = df.iloc[-2]

    past_mistakes = get_past_mistakes_text(5)

    # 모델에게 “과매도 해소 타이밍”을 강제
    features = {
        "symbol": symbol,
        "mode": mode,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)) if "RSI" in df.columns else None,
        "rsi_now": float(last.get("RSI", 50)) if "RSI" in df.columns else None,
        "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
        "trend": status.get("추세", ""),
        "bb": status.get("BB", ""),
        "macd": status.get("MACD", ""),
        "vol": status.get("거래량", ""),
        "rsi_resolve_long": bool(status.get("_rsi_resolve_long", False)),
        "rsi_resolve_short": bool(status.get("_rsi_resolve_short", False)),
        "pullback_candidate": bool(status.get("_pullback_candidate", False)),
    }

    # ✅ “제한을 없애달라” 요청이 있었지만, 최소한 모드 룰(min_conf, entry_pct 범위, lev 범위)은 유지
    # SL/TP는 넓게 허용
    sys = f"""
너는 '워뇨띠 스타일(눌림목/해소 타이밍) + 손익비' 기반의 자동매매 트레이더 AI다.
목표:
- 손실은 짧게(빠르게 끊기) 하지만
- 추세가 맞으면 익절은 더 길게(수익을 키우기)
- 그리고 같은 실수를 반복하지 않기(회고)

[과거 실수(요약)]
{past_mistakes}

[핵심 룰]
1) RSI가 과매도/과매수 "상태"에 들어가자마자 진입하지 말고,
   '해소되는 시점'(반등/반락 확인)에서만 진입 후보로 고려한다.
2) 상승추세에서는 롱(매수) 우선, 하락추세에서는 숏(매도) 우선. (역추세는 매우 신중)
3) 모드 규칙은 반드시 준수:
   - 최소 확신도: {rule["min_conf"]}
   - 진입 비중(%): {rule["entry_pct_min"]}~{rule["entry_pct_max"]}
   - 레버리지: {rule["lev_min"]}~{rule["lev_max"]}

[응답]
반드시 JSON만 출력한다.
설명은 '초보도 이해하는 쉬운 한글'로, 괄호로 뜻을 덧붙인다.
"""

    user = f"""
시장 데이터(JSON):
{json.dumps(features, ensure_ascii=False)}

JSON 형식:
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0-100,
  "entry_pct": {rule["entry_pct_min"]}-{rule["entry_pct_max"]},
  "leverage": {rule["lev_min"]}-{rule["lev_max"]},
  "sl_pct": 0.3-20.0,
  "tp_pct": 0.5-50.0,
  "rr": 0.5-10.0,
  "used_indicators": ["..."],
  "reason_easy": "쉬운 한글(괄호로 의미 추가)"
}}

조건:
- 확신이 낮으면 HOLD
- pullback_candidate=True(상승추세 눌림목 반등 후보)면 가산점
- 손절은 짧게, 익절은 추세 강하면 길게(ADX가 높을수록 tp_pct를 늘릴 수 있음)
- 텍스트는 영어 금지, 모두 한글로.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        out = json.loads(resp.choices[0].message.content)

        # normalize / clamp
        out["decision"] = out.get("decision", "hold")
        if out["decision"] not in ["buy", "sell", "hold"]:
            out["decision"] = "hold"

        out["confidence"] = int(clamp(int(out.get("confidence", 0)), 0, 100))

        out["entry_pct"] = float(out.get("entry_pct", rule["entry_pct_min"]))
        out["entry_pct"] = float(clamp(out["entry_pct"], rule["entry_pct_min"], rule["entry_pct_max"]))

        out["leverage"] = int(out.get("leverage", rule["lev_min"]))
        out["leverage"] = int(clamp(out["leverage"], rule["lev_min"], rule["lev_max"]))

        out["sl_pct"] = float(out.get("sl_pct", 1.2))
        out["tp_pct"] = float(out.get("tp_pct", 3.0))
        out["rr"] = float(out.get("rr", max(0.5, out["tp_pct"] / max(out["sl_pct"], 0.01))))

        used = out.get("used_indicators", status.get("_used_indicators", []))
        if not isinstance(used, list):
            used = status.get("_used_indicators", [])
        out["used_indicators"] = used

        out["reason_easy"] = str(out.get("reason_easy", ""))[:500]

        # 최소 확신도 미달이면 hold
        if out["decision"] in ["buy", "sell"] and out["confidence"] < rule["min_conf"]:
            out["decision"] = "hold"

        return out

    except Exception as e:
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": status.get("_used_indicators", [])}


# =========================================================
# ✅ 13) AI 회고(후기) 작성 (청산 시 일지에 저장)
# =========================================================
def ai_write_review(
    symbol: str,
    side: str,
    pnl_percent: float,
    reason: str
) -> Tuple[str, str]:
    """
    return: (one_line, review_long)
    """
    if openai_client is None:
        # fallback
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "OpenAI 키 없음 - 후기 자동작성 불가")

    sys = """
너는 매매 회고를 아주 쉽게 써주는 코치다.
출력은 반드시 JSON만.
영어 금지. 초보도 이해하도록 쉬운 한글로.
"""

    user = f"""
상황:
- 코인: {symbol}
- 포지션: {side}
- 결과: {pnl_percent:.2f}%
- 청산 이유: {reason}

JSON 형식:
{{
  "one_line": "한줄평(아주 짧게)",
  "review": "후기(손절이면 다음에 어떻게 개선할지 / 익절이면 다음에 무엇을 유지할지)"
}}
"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        out = json.loads(resp.choices[0].message.content)
        one = str(out.get("one_line", ""))[:120]
        rev = str(out.get("review", ""))[:800]
        return one, rev
    except Exception:
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")


# =========================================================
# ✅ 14) 경제 캘린더 (한글)
# =========================================================
def get_forex_events_kr(limit: int = 80) -> pd.DataFrame:
    """
    ForexFactory JSON(이번주) 불러와서 한글로 표기.
    네트워크 제한/실패 시 빈 DF.
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        rows = []
        for x in data[:limit]:
            impact = x.get("impact", "")
            imp_kr = {"High": "매우 중요", "Medium": "중요", "Low": "낮음"}.get(impact, impact)
            rows.append({
                "날짜": x.get("date", ""),
                "시간": x.get("time", ""),
                "국가": x.get("country", ""),
                "지표": x.get("title", ""),
                "중요도": imp_kr,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["날짜", "시간", "국가", "지표", "중요도"])


# =========================================================
# ✅ 15) 모니터 상태 (AI 시야/하트비트)
# =========================================================
def monitor_init():
    mon = read_json_safe(MONITOR_FILE, {"coins": {}}) or {"coins": {}}
    mon["_boot_time_kst"] = now_kst_str()
    mon["_last_write"] = 0
    write_json_atomic(MONITOR_FILE, mon)
    return mon


def monitor_write_throttled(mon: Dict[str, Any], min_interval_sec: float = 1.0):
    lastw = float(mon.get("_last_write", 0))
    if time.time() - lastw >= min_interval_sec:
        write_json_atomic(MONITOR_FILE, mon)
        mon["_last_write"] = time.time()


# =========================================================
# ✅ 16) 텔레그램 유틸
# =========================================================
def tg_send(text: str):
    if not tg_token or not tg_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_id, "text": text},
            timeout=10
        )
    except Exception:
        pass


def tg_send_menu():
    if not tg_token or not tg_id:
        return
    kb = {
        "inline_keyboard": [
            [{"text": "📡 상태", "callback_data": "status"},
             {"text": "👁️ AI시야", "callback_data": "vision"}],
            [{"text": "📊 포지션", "callback_data": "position"},
             {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "📜 일지(최근)", "callback_data": "log"},
             {"text": "🛑 전량청산", "callback_data": "close_all"}]
        ]
    }
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_id, "text": "✅ 메뉴 갱신", "reply_markup": json.dumps(kb)},
            timeout=10
        )
    except Exception:
        pass


def tg_answer_callback(cb_id: str):
    if not tg_token:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery",
            data={"callback_query_id": cb_id},
            timeout=10
        )
    except Exception:
        pass


# =========================================================
# ✅ 17) 자동매매 핵심 스레드 (24시간 모니터 + 매매 + 일지 + 시야)
# =========================================================
def telegram_thread(ex):
    offset = 0
    mon = monitor_init()

    tg_send("🚀 AI 봇 가동 시작! (모의투자)\n명령: 상태 / 시야 / 일지")
    tg_send_menu()

    # active_targets: 심볼별 목표/정보 저장
    active_targets: Dict[str, Dict[str, Any]] = {}

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()
            mode = cfg.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])

            # ✅ 하트비트
            mon["last_heartbeat_epoch"] = time.time()
            mon["last_heartbeat_kst"] = now_kst_str()
            mon["auto_trade"] = bool(cfg.get("auto_trade", False))
            mon["trade_mode"] = mode
            mon["pause_until"] = rt.get("pause_until", 0)
            mon["consec_losses"] = rt.get("consec_losses", 0)

            # ✅ 자동매매 ON일 때만 스캔/매매
            if cfg.get("auto_trade", False):
                # 주말 거래 금지 옵션
                if cfg.get("no_trade_weekend", False):
                    # KST 기준 토/일
                    wd = now_kst().weekday()  # 0=월 ... 5=토 6=일
                    if wd in [5, 6]:
                        mon["global_state"] = "주말 거래 OFF"
                        monitor_write_throttled(mon, 2.0)
                        time.sleep(2.0)
                        continue

                # 일시정지(연속손실)
                if cfg.get("loss_pause_enable", True) and time.time() < float(rt.get("pause_until", 0)):
                    mon["global_state"] = "일시정지 중(연속손실/보호)"
                    monitor_write_throttled(mon, 2.0)
                    time.sleep(1.0)
                else:
                    mon["global_state"] = "스캔/매매 중"

                    # 1) 포지션 관리 (손절/익절/트레일링/DCA/스위칭)
                    for sym in TARGET_COINS:
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if not act:
                            continue

                        p = act[0]
                        side = position_side_normalize(p)  # long/short
                        contracts = float(p.get("contracts") or 0)
                        entry = float(p.get("entryPrice") or 0)
                        roi = float(position_roi_percent(p))

                        # 목표가: active_targets에 없으면 fallback
                        tgt = active_targets.get(sym, {
                            "sl": 2.0,     # 손절(%) 기준
                            "tp": 5.0,     # 익절(%) 기준
                            "entry_usdt": 0.0,
                            "entry_pct": 0.0,
                            "lev": p.get("leverage", "?"),
                            "reason": ""
                        })
                        sl = float(tgt.get("sl", 2.0))
                        tp = float(tgt.get("tp", 5.0))

                        # ✅ 트레일링: 절반 익절 도달하면 손절을 당겨서 수익보호
                        if cfg.get("use_trailing_stop", True):
                            if roi >= (tp * 0.5):
                                # 본전 방어 수준 (-0.3% 정도)
                                sl = min(sl, 0.3)

                        # ✅ DCA (물타기): 손실이 일정 수준(dca_trigger) 이하일 때 1회 추가 진입
                        #    - 데모에서만 충분히 테스트 권장
                        if cfg.get("use_dca", True):
                            dca_trig = float(cfg.get("dca_trigger", -20.0))
                            dca_max = int(cfg.get("dca_max_count", 1))
                            dca_add_pct = float(cfg.get("dca_add_pct", 50.0))

                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0})
                            dca_count = int(trade_state.get("dca_count", 0))

                            if roi <= dca_trig and dca_count < dca_max:
                                # 원래 진입금의 일부만큼 추가
                                free, total = safe_fetch_balance(ex)
                                base_entry = float(tgt.get("entry_usdt", 0.0))
                                add_usdt = base_entry * (dca_add_pct / 100.0)
                                if add_usdt > free:
                                    add_usdt = free * 0.5

                                px = get_last_price(ex, sym)
                                if px and add_usdt > 5:
                                    lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                    set_leverage_safe(ex, sym, lev)
                                    qty = to_precision_qty(ex, sym, (add_usdt * lev) / px)
                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                    if ok:
                                        trade_state["dca_count"] = dca_count + 1
                                        save_runtime(rt)
                                        tg_send(f"💧 물타기(DCA)\n- 코인: {sym}\n- 추가금: {add_usdt:.2f} USDT\n- 이유: 손실 {roi:.2f}% (기준 {dca_trig}%)")
                                        mon["last_action"] = {"time_kst": now_kst_str(), "type": "DCA", "symbol": sym, "roi": roi}
                                        monitor_write_throttled(mon, 0.2)

                        # ✅ 손절
                        if roi <= -abs(sl):
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                pnl_usdt = float(p.get("unrealizedPnl") or 0)

                                one, review = ai_write_review(sym, side, roi, "자동 손절(목표 손절 도달)")
                                log_trade(sym, side, entry, exit_px, pnl_usdt, roi, "자동 손절", one_line=one, review=review)

                                # 연속손실 증가 및 일시정지 조건
                                rt["consec_losses"] = int(rt.get("consec_losses", 0)) + 1
                                if cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(f"🛑 연속손실 보호\n- 연속손실: {rt['consec_losses']}회\n- {int(cfg.get('loss_pause_minutes',30))}분 자동 정지")
                                save_runtime(rt)

                                tg_send(
                                    f"🩸 손절\n"
                                    f"- 코인: {sym}\n"
                                    f"- 수익률: {roi:.2f}%\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 이유: 목표 손절 도달\n"
                                    f"- 한줄평: {one}"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)  # dca 기록 제거
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "STOP", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                        # ✅ 익절
                        elif roi >= tp:
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                pnl_usdt = float(p.get("unrealizedPnl") or 0)

                                one, review = ai_write_review(sym, side, roi, "자동 익절(목표 익절 도달)")
                                log_trade(sym, side, entry, exit_px, pnl_usdt, roi, "자동 익절", one_line=one, review=review)

                                rt["consec_losses"] = 0
                                save_runtime(rt)

                                tg_send(
                                    f"🎉 익절\n"
                                    f"- 코인: {sym}\n"
                                    f"- 수익률: +{roi:.2f}%\n"
                                    f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                    f"- 레버: x{tgt.get('lev','?')}\n"
                                    f"- 이유: 목표 익절 도달\n"
                                    f"- 한줄평: {one}"
                                )

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                save_runtime(rt)

                                mon["last_action"] = {"time_kst": now_kst_str(), "type": "TAKE", "symbol": sym, "roi": roi}
                                monitor_write_throttled(mon, 0.2)

                    # 2) 신규 진입 스캔
                    free_usdt, total_usdt = safe_fetch_balance(ex)

                    for sym in TARGET_COINS:
                        # 이미 포지션 있으면 스킵
                        ps = safe_fetch_positions(ex, [sym])
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        if act:
                            continue

                        # 쿨다운
                        cd = float(rt.get("cooldowns", {}).get(sym, 0))
                        if time.time() < cd:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "쿨다운(잠깐 쉬는중)"
                            continue

                        # 데이터 로드
                        try:
                            ohlcv = ex.fetch_ohlcv(sym, cfg.get("timeframe", "5m"), limit=220)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except Exception as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = f"데이터 실패: {e}"
                            continue

                        df, stt, last = calc_indicators(df, cfg)
                        mon.setdefault("coins", {}).setdefault(sym, {})
                        cs = mon["coins"][sym]

                        if last is None:
                            cs.update({
                                "last_scan_kst": now_kst_str(),
                                "ai_called": False,
                                "skip_reason": "지표 계산 실패(ta/데이터 부족)"
                            })
                            continue

                        # 모니터 기록(지표/상태)
                        cs.update({
                            "last_scan_epoch": time.time(),
                            "last_scan_kst": now_kst_str(),
                            "price": float(last["close"]),
                            "trend": stt.get("추세", ""),
                            "rsi": float(last.get("RSI", 0)) if "RSI" in df.columns else None,
                            "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
                            "bb": stt.get("BB", ""),
                            "macd": stt.get("MACD", ""),
                            "vol": stt.get("거래량", ""),
                            "pullback_candidate": bool(stt.get("_pullback_candidate", False)),
                        })

                        # ✅ AI 호출 필터 (횡보/해소 아님이면 비용 절감 + 휩쏘 회피)
                        call_ai = False
                        if bool(stt.get("_pullback_candidate", False)):
                            call_ai = True
                        elif bool(stt.get("_rsi_resolve_long", False)) or bool(stt.get("_rsi_resolve_short", False)):
                            call_ai = True
                        else:
                            # ADX 강하면 트렌드 진입 후보로 AI 호출
                            adxv = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            if adxv >= 25:
                                call_ai = True

                        if not call_ai:
                            cs["ai_called"] = False
                            cs["skip_reason"] = "횡보/해소 신호 없음(휩쏘 위험)"
                            monitor_write_throttled(mon, 1.0)
                            continue

                        # AI 판단
                        ai = ai_decide_trade(df, stt, sym, mode, cfg)
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))

                        cs.update({
                            "ai_called": True,
                            "ai_decision": decision,
                            "ai_confidence": conf,
                            "ai_entry_pct": float(ai.get("entry_pct", rule["entry_pct_min"])),
                            "ai_leverage": int(ai.get("leverage", rule["lev_min"])),
                            "ai_sl_pct": float(ai.get("sl_pct", 1.2)),
                            "ai_tp_pct": float(ai.get("tp_pct", 3.0)),
                            "ai_rr": float(ai.get("rr", 1.5)),
                            "ai_used": ", ".join(ai.get("used_indicators", [])),
                            "ai_reason_easy": ai.get("reason_easy", ""),
                            "min_conf_required": int(rule["min_conf"]),
                            "skip_reason": ""
                        })
                        monitor_write_throttled(mon, 1.0)

                        # 진입 조건
                        if decision in ["buy", "sell"] and conf >= int(rule["min_conf"]):
                            entry_pct = float(ai.get("entry_pct", rule["entry_pct_min"]))
                            lev = int(ai.get("leverage", rule["lev_min"]))
                            slp = float(ai.get("sl_pct", 1.2))
                            tpp = float(ai.get("tp_pct", 3.0))

                            entry_usdt = free_usdt * (entry_pct / 100.0)
                            px = float(last["close"])
                            if entry_usdt < 5:
                                cs["skip_reason"] = "잔고 부족(진입금 너무 작음)"
                                continue

                            set_leverage_safe(ex, sym, lev)
                            qty = to_precision_qty(ex, sym, (entry_usdt * lev) / px)
                            if qty <= 0:
                                cs["skip_reason"] = "수량 계산 실패"
                                continue

                            ok = market_order_safe(ex, sym, decision, qty)
                            if ok:
                                # 목표 저장
                                active_targets[sym] = {
                                    "sl": slp, "tp": tpp,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "reason": ai.get("reason_easy", "")
                                }

                                # 쿨다운 60초
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)

                                # 텔레그램 보고
                                if cfg.get("tg_enable_reports", True):
                                    direction = "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"
                                    tg_send(
                                        f"🎯 진입\n"
                                        f"- 코인: {sym}\n"
                                        f"- 방향: {direction}\n"
                                        f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%)\n"
                                        f"- 레버리지: x{lev}\n"
                                        f"- 목표익절: +{tpp:.2f}% / 목표손절: -{slp:.2f}%\n"
                                        f"- 확신도: {conf}% (기준 {rule['min_conf']}%)\n"
                                        f"- 근거(쉬운말): {ai.get('reason_easy','')[:220]}\n"
                                        f"- AI가 본 지표: {', '.join(ai.get('used_indicators', []))}"
                                    )

                                mon["last_action"] = {
                                    "time_kst": now_kst_str(),
                                    "type": "ENTRY",
                                    "symbol": sym,
                                    "decision": decision,
                                    "conf": conf,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "tp": tpp,
                                    "sl": slp,
                                }
                                monitor_write_throttled(mon, 0.2)

                                # 다음 코인 스캔 텀
                                time.sleep(1.0)

                        time.sleep(0.4)

            # =================================================
            # 텔레그램 수신 처리 (텍스트 명령 / 콜백 버튼)
            # =================================================
            try:
                res = requests.get(
                    f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1",
                    timeout=10
                ).json()
            except Exception:
                res = {"ok": False}

            if res.get("ok"):
                for up in res.get("result", []):
                    offset = up.get("update_id", offset)

                    # 텍스트 명령
                    if "message" in up and "text" in up["message"]:
                        txt = up["message"]["text"].strip()

                        if txt == "상태":
                            cfg_live = load_settings()  # ✅ 항상 최신 파일 기준
                            free, total = safe_fetch_balance(ex)
                            rt = load_runtime()
                            tg_send(
                                f"📡 상태\n"
                                f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 잔고: {total:.2f} USDT (사용가능 {free:.2f})\n"
                                f"- 연속손실: {rt.get('consec_losses',0)}\n"
                                f"- 정지해제: {('정지중' if time.time() < float(rt.get('pause_until',0)) else '정상')}\n"
                            )


                        elif txt == "시야":
                            mon_now = read_json_safe(MONITOR_FILE, {})
                            coins = mon_now.get("coins", {}) or {}
                            lines = [
                                "👁️ AI 시야(요약)",
                                f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                                f"- 모드: {mon_now.get('trade_mode','-')}",
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                            ]
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                    f"/ RSI {cs.get('rsi','-')} / ADX {cs.get('adx','-')} "
                                    f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:30]}"
                                )
                            tg_send("\n".join(lines))

                        elif txt == "일지":
                            df_log = read_trade_log()
                            if df_log.empty:
                                tg_send("📜 일지 없음(아직 기록된 매매가 없어요)")
                            else:
                                top = df_log.head(8)
                                msg = ["📜 최근 매매일지(요약)"]
                                for _, r in top.iterrows():
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {float(r['PnL_Percent']):.2f}% | {str(r.get('OneLine',''))[:40]}")
                                tg_send("\n".join(msg))

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up["callback_query"]
                        data = cb.get("data", "")
                        cb_id = cb.get("id", "")

                        if data == "status":
                            cfg_live = load_settings()  # ✅ 항상 최신 파일 기준
                            free, total = safe_fetch_balance(ex)
                            rt = load_runtime()
                            tg_send(
                                f"📡 상태\n"
                                f"- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 잔고: {total:.2f} USDT (사용가능 {free:.2f})\n"
                                f"- 연속손실: {rt.get('consec_losses',0)}\n"
                            )

                        elif data == "vision":
                            mon_now = read_json_safe(MONITOR_FILE, {})
                            coins = mon_now.get("coins", {}) or {}
                            lines = [
                                "👁️ AI 시야(요약)",
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                            ]
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                    f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                                )
                            tg_send("\n".join(lines))

                        elif data == "balance":
                            free, total = safe_fetch_balance(ex)
                            tg_send(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT")

                        elif data == "position":
                            msg = ["📊 포지션"]
                            has = False
                            for sym in TARGET_COINS:
                                ps = safe_fetch_positions(ex, [sym])
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if act:
                                    p = act[0]
                                    has = True
                                    side = position_side_normalize(p)
                                    roi = float(position_roi_percent(p))
                                    msg.append(f"- {sym}: {('롱' if side=='long' else '숏')} (수익률 {roi:.2f}%)")
                            if not has:
                                msg.append("- 없음(관망)")
                            tg_send("\n".join(msg))

                        elif data == "log":
                            df_log = read_trade_log()
                            if df_log.empty:
                                tg_send("📜 일지 없음")
                            else:
                                top = df_log.head(8)
                                msg = ["📜 최근 매매일지(요약)"]
                                for _, r in top.iterrows():
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {float(r['PnL_Percent']):.2f}% | {str(r.get('OneLine',''))[:40]}")
                                tg_send("\n".join(msg))

                        elif data == "close_all":
                            tg_send("🛑 전량 청산 시도")
                            for sym in TARGET_COINS:
                                ps = safe_fetch_positions(ex, [sym])
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    continue
                                p = act[0]
                                side = position_side_normalize(p)
                                contracts = float(p.get("contracts") or 0)
                                close_position_market(ex, sym, side, contracts)
                            tg_send("✅ 전량 청산 요청 완료")

                        tg_answer_callback(cb_id)

            monitor_write_throttled(mon, 2.0)
            time.sleep(0.8)

        except Exception as e:
            tg_send(f"⚠️ 스레드 오류: {e}")
            time.sleep(3.0)


# =========================================================
# ✅ 18) 스레드 시작 (중복 방지)
# =========================================================
def ensure_thread_started():
    for t in threading.enumerate():
        if t.name == "TG_THREAD":
            return
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(th)
    th.start()


ensure_thread_started()


# =========================================================
# ✅ 19) Streamlit UI
# =========================================================
st.sidebar.title("🛠️ 제어판")
st.sidebar.caption("Streamlit은 제어/상태 확인용, Telegram은 실시간 보고/조회용")

# OpenAI 키 입력(선택)
if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

with st.sidebar.expander("🧪 디버그: 저장된 설정(bot_settings.json) 확인"):
    st.json(read_json_safe(SETTINGS_FILE, {}))

# 모드 선택 (MODE_RULES 기반)
mode_keys = list(MODE_RULES.keys())
safe_mode = config.get("trade_mode", "안전모드")
if safe_mode not in mode_keys:
    safe_mode = "안전모드"
config["trade_mode"] = st.sidebar.selectbox("매매 모드", mode_keys, index=mode_keys.index(safe_mode))

# 자동매매 ON/OFF
auto_on = st.sidebar.checkbox("🤖 자동매매 (텔레그램 연동)", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

# 기본 옵션
st.sidebar.divider()
config["timeframe"] = st.sidebar.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))
config["tg_enable_reports"] = st.sidebar.checkbox("📨 텔레그램 보고 활성화", value=bool(config.get("tg_enable_reports", True)))
config["use_trailing_stop"] = st.sidebar.checkbox("🚀 트레일링 스탑(수익보호)", value=bool(config.get("use_trailing_stop", True)))

st.sidebar.divider()
st.sidebar.subheader("🛡️ 방어/자금 관리")
config["loss_pause_enable"] = st.sidebar.checkbox("연속손실 보호(자동 정지)", value=bool(config.get("loss_pause_enable", True)))
c1, c2 = st.sidebar.columns(2)
config["loss_pause_after"] = c1.number_input("연속손실 N회", 1, 20, int(config.get("loss_pause_after", 3)))
config["loss_pause_minutes"] = c2.number_input("정지(분)", 1, 240, int(config.get("loss_pause_minutes", 30)))

st.sidebar.divider()
config["use_dca"] = st.sidebar.checkbox("💧 물타기(DCA)", value=bool(config.get("use_dca", True)))
c3, c4 = st.sidebar.columns(2)
config["dca_trigger"] = c3.number_input("DCA 발동(%)", -90.0, -1.0, float(config.get("dca_trigger", -20.0)), step=0.5)
config["dca_max_count"] = c4.number_input("최대 횟수", 0, 10, int(config.get("dca_max_count", 1)))
config["dca_add_pct"] = st.sidebar.slider("추가 규모(원진입 대비 %)", 10, 200, int(config.get("dca_add_pct", 50)))

st.sidebar.divider()
st.sidebar.subheader("📊 보조지표 (10종) ON/OFF")
colA, colB = st.sidebar.columns(2)
config["use_rsi"] = colA.checkbox("RSI", value=bool(config.get("use_rsi", True)))
config["use_bb"] = colB.checkbox("볼린저", value=bool(config.get("use_bb", True)))
config["use_ma"] = colA.checkbox("MA(이평)", value=bool(config.get("use_ma", True)))
config["use_macd"] = colB.checkbox("MACD", value=bool(config.get("use_macd", True)))
config["use_stoch"] = colA.checkbox("스토캐스틱", value=bool(config.get("use_stoch", True)))
config["use_cci"] = colB.checkbox("CCI", value=bool(config.get("use_cci", True)))
config["use_mfi"] = colA.checkbox("MFI", value=bool(config.get("use_mfi", True)))
config["use_willr"] = colB.checkbox("윌리엄%R", value=bool(config.get("use_willr", True)))
config["use_adx"] = colA.checkbox("ADX", value=bool(config.get("use_adx", True)))
config["use_vol"] = colB.checkbox("거래량", value=bool(config.get("use_vol", True)))

st.sidebar.divider()
st.sidebar.subheader("지표 파라미터")
r1, r2, r3 = st.sidebar.columns(3)
config["rsi_period"] = r1.number_input("RSI 기간", 5, 50, int(config.get("rsi_period", 14)))
config["rsi_buy"] = r2.number_input("과매도", 10, 50, int(config.get("rsi_buy", 30)))
config["rsi_sell"] = r3.number_input("과매수", 50, 90, int(config.get("rsi_sell", 70)))

b1, b2 = st.sidebar.columns(2)
config["bb_period"] = b1.number_input("BB 기간", 5, 50, int(config.get("bb_period", 20)))
config["bb_std"] = b2.number_input("BB 승수", 1.0, 5.0, float(config.get("bb_std", 2.0)))

m1, m2 = st.sidebar.columns(2)
config["ma_fast"] = m1.number_input("MA 단기", 3, 50, int(config.get("ma_fast", 7)))
config["ma_slow"] = m2.number_input("MA 장기", 50, 300, int(config.get("ma_slow", 99)))

st.sidebar.divider()
st.sidebar.subheader("🔍 긴급 점검")
if st.sidebar.button("📡 텔레그램 메뉴 전송"):
    tg_send_menu()

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    if openai_client is None:
        st.sidebar.error("OpenAI 연결 실패(키/설정 확인)")
    else:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                max_tokens=10
            )
            st.sidebar.success("✅ 연결 성공: " + resp.choices[0].message.content)
        except Exception as e:
            st.sidebar.error(f"❌ 실패: {e}")

# 설정 저장
save_settings(config)

# =========================================================
# ✅ Sidebar: 잔고/포지션 현황
# =========================================================
with st.sidebar:
    st.divider()
    st.header("내 지갑 현황")
    free, total = safe_fetch_balance(exchange)
    st.metric("총 자산(USDT)", f"{total:,.2f}")
    st.metric("주문 가능", f"{free:,.2f}")

    st.divider()
    st.subheader("보유 포지션(주요 5개)")
    try:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        if not act:
            st.caption("무포지션(관망)")
        else:
            for p in act:
                sym = p.get("symbol", "")
                side = position_side_normalize(p)
                roi = float(position_roi_percent(p))
                lev = p.get("leverage", "?")
                st.info(f"**{sym}** ({'🟢롱' if side=='long' else '🔴숏'} x{lev})\n수익률: **{roi:.2f}%**")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")


# =========================================================
# ✅ Main UI: 차트/지표/탭
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트 (Final)")
st.caption("Streamlit=제어판/모니터링, Telegram=실시간 보고/조회. (모의투자 IS_SANDBOX=True)")

# 코인 선택
markets = exchange.markets or {}
if markets:
    symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
    if not symbol_list:
        symbol_list = TARGET_COINS
else:
    symbol_list = TARGET_COINS

symbol = st.selectbox("코인 선택", symbol_list, index=0)

# 상단 레이아웃
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("📉 TradingView 차트 (다크모드)")
    interval_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}
    render_tradingview(symbol, interval=interval_map.get(config.get("timeframe", "5m"), "5"), height=560)

with right:
    st.subheader("🧾 실시간 지표 요약")
    if ta is None:
        st.error("ta 모듈이 없습니다. requirements.txt에 `ta` 추가 후 재배포하세요.")
    else:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df2, stt, last = calc_indicators(df, config)

            if last is None:
                st.warning("지표 계산 실패(데이터 부족)")
            else:
                st.metric("현재가", f"{float(last['close']):,.4f}")
                # 보기 좋게 주요만
                show = {
                    "RSI": stt.get("RSI", "-"),
                    "BB": stt.get("BB", "-"),
                    "MACD": stt.get("MACD", "-"),
                    "ADX": stt.get("ADX", "-"),
                    "추세": stt.get("추세", "-"),
                    "거래량": stt.get("거래량", "-"),
                    "눌림목후보(해소)": "✅" if stt.get("_pullback_candidate") else "—",
                }
                st.write(show)
        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")

st.divider()

# 탭
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("👁️ 실시간 AI 모니터링(봇 시야)")
    # 자동 새로고침(선택)
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, key="mon_refresh")  # 2초
    else:
        st.caption("자동 새로고침을 원하면 requirements.txt에 streamlit-autorefresh 추가하세요.")
        st.button("🔄 수동 새로고침")

    mon = read_json_safe(MONITOR_FILE, None)
    if not mon:
        st.warning("monitor_state.json이 아직 없습니다. (스레드 시작 확인)")
    else:
        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = (time.time() - hb) if hb else 9999

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        c2.metric("모드", mon.get("trade_mode", "-"))
        c3.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < 6 else "🔴 멈춤 의심")
        c4.metric("연속손실", str(mon.get("consec_losses", 0)))

        if age >= 6:
            st.error("⚠️ 봇 스레드가 멈췄거나(크래시) 갱신이 안될 수 있어요.")

        st.caption(f"봇 상태: {mon.get('global_state','-')} | 마지막 액션: {mon.get('last_action',{})}")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0) or 0)
            scan_age = (time.time() - last_scan) if last_scan else 9999
            rows.append({
                "코인": sym,
                "스캔(초전)": f"{scan_age:.1f}",
                "가격": cs.get("price", ""),
                "추세": cs.get("trend", ""),
                "RSI": cs.get("rsi", ""),
                "ADX": cs.get("adx", ""),
                "BB": cs.get("bb", ""),
                "MACD": cs.get("macd", ""),
                "눌림목후보": "✅" if cs.get("pullback_candidate") else "—",
                "AI호출": "✅" if cs.get("ai_called") else "—",
                "AI결론": str(cs.get("ai_decision", "-")).upper(),
                "확신도": cs.get("ai_confidence", "-"),
                "필요확신도": cs.get("min_conf_required", "-"),
                "진입%": cs.get("ai_entry_pct", "-"),
                "레버": cs.get("ai_leverage", "-"),
                "SL%": cs.get("ai_sl_pct", "-"),
                "TP%": cs.get("ai_tp_pct", "-"),
                "손익비": cs.get("ai_rr", "-"),
                "AI지표": cs.get("ai_used", ""),
                "스킵/근거": (cs.get("skip_reason") or cs.get("ai_reason_easy") or "")[:160],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("아직 스캔 데이터가 없습니다.")

    st.divider()
    st.subheader("🔍 현재 코인 AI 분석(수동 버튼)")
    if st.button("현재 코인 AI 분석 실행"):
        if openai_client is None:
            st.error("OpenAI 키 없음")
        elif ta is None:
            st.error("ta 모듈 없음")
        else:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe", "5m"), limit=220)
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if last is None:
                    st.warning("지표 계산 실패")
                else:
                    ai = ai_decide_trade(df2, stt, symbol, config.get("trade_mode", "안전모드"), config)
                    st.json(ai)
            except Exception as e:
                st.error(f"분석 오류: {e}")

with t2:
    st.subheader("⚡ 수동 주문(데모용)")
    st.caption("⚠️ 수동 주문은 실수 방지를 위해 기본은 '설명/테스트' 중심입니다.")
    amt = st.number_input("주문 금액(USDT)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    config["order_usdt"] = float(amt)
    save_settings(config)

    enable_manual = st.checkbox("수동 주문 활성화(주의!)", value=False)
    b1, b2, b3 = st.columns(3)

    if b1.button("🟢 롱 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "buy", qty)
            st.success("롱 진입 성공" if ok else "롱 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b2.button("🔴 숏 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "sell", qty)
            st.success("숏 진입 성공" if ok else "숏 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b3.button("🚫 전량 청산") and enable_manual:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        for p in act:
            sym = p.get("symbol", "")
            side = position_side_normalize(p)
            contracts = float(p.get("contracts") or 0)
            close_position_market(exchange, sym, side, contracts)
        st.success("전량 청산 요청 완료(데모)")

with t3:
    st.subheader("📅 시장정보 (경제 캘린더)")
    ev = get_forex_events_kr()
    if ev.empty:
        st.info("일정 없음/불러오기 실패(네트워크 제한일 수 있음)")
    else:
        st.dataframe(ev, width="stretch", hide_index=True)

with t4:
    st.subheader("📜 매매일지 (보기 쉽게 + 초기화)")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🔄 새로고침"):
        st.rerun()
    if c2.button("🧹 매매일지 초기화"):
        reset_trade_log()
        st.success("매매일지 초기화 완료")
        st.rerun()

    df_log = read_trade_log()
    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        show_cols = [c for c in ["Time", "Coin", "Side", "PnL_Percent", "PnL_USDT", "OneLine", "Reason", "Review"] if c in df_log.columns]
        st.dataframe(df_log[show_cols], width="stretch", hide_index=True)

        csv_bytes = df_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

    st.divider()
    st.subheader("📌 runtime_state.json (현재 상태)")
    rt = load_runtime()
    st.json(rt)
    if st.button("🧼 runtime_state 초기화(오늘 기준)"):
        write_json_atomic(RUNTIME_FILE, default_runtime())
        st.success("runtime_state.json 초기화 완료")
        st.rerun()

st.caption("⚠️ 이 봇은 모의투자(IS_SANDBOX=True)에서 충분히 검증 후 사용하세요.")
