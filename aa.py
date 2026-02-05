import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx

import ccxt
import pandas as pd
import numpy as np
import time
import requests
import threading
import os
import json
import uuid
import sqlite3

from datetime import datetime, timedelta, timezone

# === indicators ===
try:
    import ta
except Exception as e:
    ta = None

# === OpenAI ===
from openai import OpenAI


# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Monitoring+AI Vision)")

IS_SANDBOX = True  # 데모/모의투자
SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"
DB_FILE = "wonyousi_brain.db"

KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def now_kst_str():
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT"
]

# ✅ 너가 준 MODE_RULES 그대로
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
        "entry_pct_min": 8,
        "entry_pct_max": 25,
        "lev_min": 2,
        "lev_max": 10,
    },
    "하이리스크/하이리턴": {
        "min_conf": 85,
        "entry_pct_min": 15,
        "entry_pct_max": 40,
        "lev_min": 8,
        "lev_max": 25,
    }
}


# =========================================================
# ✅ 1) JSON 안전 저장/읽기(원자적)
# =========================================================
def write_json_atomic(path, data: dict):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except:
        pass

def read_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


# =========================================================
# ✅ 2) 설정 load/save
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "trade_mode": "안전모드",
        "auto_trade": False,

        "timeframe": "5m",
        "order_usdt": 100.0,

        # 지표 파라미터(기본)
        "rsi_period": 14,
        "rsi_buy": 30,
        "rsi_sell": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_fast": 7,
        "ma_slow": 99,

        # 지표 ON/OFF
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

        # 고급
        "use_trailing_stop": True,
        "no_trade_weekend": False,

        # AI 추천값 표시/적용
        "ai_reco_show": True,
        "ai_reco_apply": False,   # ✅ ON이면 AI 추천 글로벌옵션 자동 적용(원하면 켜)
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            saved = read_json_safe(SETTINGS_FILE, {})
            if isinstance(saved, dict):
                default.update(saved)
        except:
            pass
    return default

def save_settings(cfg):
    write_json_atomic(SETTINGS_FILE, cfg)


config = load_settings()


# =========================================================
# ✅ 3) 런타임 상태(runtime_state.json) - 너가 말한 포맷 유지
# =========================================================
def default_runtime():
    return {
        "date": now_kst().strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "daily_realized_pnl": 0.0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {}
    }

def load_runtime():
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()

    # 날짜 바뀌면 초기화(하루 단위)
    today = now_kst().strftime("%Y-%m-%d")
    if rt.get("date") != today:
        rt = default_runtime()
    return rt

def save_runtime(rt):
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ 4) 매매일지 CSV (상세 저장 + UI 한줄평)
# =========================================================
def log_trade(coin, side, entry_price, exit_price, pnl_amount, pnl_percent, reason, one_line=""):
    try:
        now = now_kst_str()
        row = pd.DataFrame([{
            "Time": now,
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Reason": reason,
            "OneLine": one_line
        }])
        if not os.path.exists(LOG_FILE):
            row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except:
        pass

def read_trade_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df
    except:
        return pd.DataFrame()

def reset_trade_log():
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except:
        pass


# =========================================================
# ✅ 5) Bitget / Telegram / OpenAI 키 로드
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

if not openai_key:
    st.warning("⚠️ OpenAI API Key가 없습니다. (AI 기능이 제한됩니다)")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_key)


# =========================================================
# ✅ 6) 거래소 연결
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
    except Exception as e:
        return None

exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키/권한/네트워크 확인.")
    st.stop()


# =========================================================
# ✅ 7) 텔레그램 유틸
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
    except:
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
    except:
        pass


# =========================================================
# ✅ 8) 지표 계산(10종) + “눌림목/해소” 필터
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: dict):
    """
    10종 지표 계산 + 상태 요약(한글)
    """
    status = {}
    if df is None or df.empty or len(df) < 120:
        return df, status, None

    if ta is None:
        status["_ERROR"] = "ta 라이브러리가 없음(requirements.txt에 ta 추가 필요)"
        return df, status, None

    # 기본 시계열
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]

    # 파라미터
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))
    bb_period = int(cfg.get("bb_period", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    ma_fast = int(cfg.get("ma_fast", 7))
    ma_slow = int(cfg.get("ma_slow", 99))

    # RSI
    if cfg.get("use_rsi", True):
        df["RSI"] = ta.momentum.rsi(close, window=rsi_period)

    # BB
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
        df["STO_K"] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

    # CCI
    if cfg.get("use_cci", True):
        df["CCI"] = ta.trend.cci(high, low, close, window=20)

    # MFI
    if cfg.get("use_mfi", True):
        df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)

    # WillR
    if cfg.get("use_willr", True):
        df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # ADX
    if cfg.get("use_adx", True):
        df["ADX"] = ta.trend.adx(high, low, close, window=14)

    # Volume spike(단순)
    if cfg.get("use_vol", True):
        df["VOL_MA"] = vol.rolling(20).mean()
        df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * 2)).astype(int)

    df = df.dropna()
    if df.empty:
        return df, status, None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 상태 요약(한글)
    # RSI
    rsi_val = float(last.get("RSI", 50))
    prev_rsi = float(prev.get("RSI", rsi_val))
    if cfg.get("use_rsi", True):
        if rsi_val < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_val:.1f})"
        elif rsi_val > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_val:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_val:.1f})"

    # BB
    if cfg.get("use_bb", True):
        if last["close"] > last["BB_upper"]:
            status["볼린저"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["볼린저"] = "🟢 하단 이탈"
        else:
            status["볼린저"] = "⚪ 밴드 안"

    # MA 추세
    trend = "중립"
    if cfg.get("use_ma", True):
        if last["MA_fast"] > last["MA_slow"] and last["close"] > last["MA_slow"]:
            trend = "상승추세"
        elif last["MA_fast"] < last["MA_slow"] and last["close"] < last["MA_slow"]:
            trend = "하락추세"
        else:
            trend = "횡보/전환"
        status["추세"] = f"📈 {trend}"

    # MACD
    if cfg.get("use_macd", True):
        status["MACD"] = "📈 골든(상승)" if last["MACD"] > last["MACD_signal"] else "📉 데드(하락)"

    # ADX
    adx_val = float(last.get("ADX", 0))
    if cfg.get("use_adx", True):
        status["ADX"] = "🔥 추세 강함" if adx_val >= 25 else "💤 추세 약함"

    # ✅ 핵심: “과매도에 바로 진입 금지” → “해소(반등 확인) 때 진입” 후보 표시
    # 롱 해소: prev_rsi < buy_threshold 이고 now_rsi >= buy_threshold
    rsi_resolve_long = (prev_rsi < rsi_buy) and (rsi_val >= rsi_buy)
    rsi_resolve_short = (prev_rsi > rsi_sell) and (rsi_val <= rsi_sell)

    status["_필터_RSI해소롱"] = bool(rsi_resolve_long)
    status["_필터_RSI해소숏"] = bool(rsi_resolve_short)

    # 눌림목 후보(상승추세 + RSI 과매도였다가 회복 + ADX가 너무 약하진 않음)
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_val >= 18)
    status["_필터_눌림목반등후보"] = bool(pullback_candidate)

    return df, status, last


# =========================================================
# ✅ 9) AI 판단 (쉬운 설명 + 사용 지표 목록)
# =========================================================
def ai_decide_trade(df: pd.DataFrame, status: dict, coin: str, mode: str):
    """
    return dict:
      decision(buy/sell/hold), confidence(0~100), entry_pct, leverage, sl_pct, tp_pct, rr, reason_easy, used_indicators
    """
    if openai_client is None or df is None or df.empty:
        return {"decision": "hold", "confidence": 0, "reason_easy": "OpenAI 키 없음/데이터 부족", "used_indicators": []}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])

    # 사용할 수 있는 지표들(데이터로 전달)
    pack = {
        "coin": coin,
        "mode": mode,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)),
        "rsi_now": float(last.get("RSI", 50)),
        "adx": float(last.get("ADX", 0)),
        "trend": status.get("추세", ""),
        "bb": status.get("볼린저", ""),
        "macd": status.get("MACD", ""),
        "pullback_candidate": bool(status.get("_필터_눌림목반등후보")),
        "rsi_resolve_long": bool(status.get("_필터_RSI해소롱")),
        "rsi_resolve_short": bool(status.get("_필터_RSI해소숏")),
    }

    system = f"""
너는 '워뇨띠 스타일 + 손익비' 기반의 선별형 트레이더야.
목표: 손실은 짧게(빠른 손절) 하지만 추세가 맞으면 익절은 더 길게(수익 극대화).

중요 규칙(반드시 지켜):
1) "과매도/과매수에 들어가는 것"이 아니라 "해소되는 타이밍(반등/반락 확인)"에 들어가.
2) 추세(상승/하락)가 맞는 방향으로만 유리하게 진입해.
3) 모드별 기준:
- 모드: {mode}
- 최소 확신도: {rule["min_conf"]}
- 진입 비중 범위: {rule["entry_pct_min"]}~{rule["entry_pct_max"]}% (잔고 대비)
- 레버리지 범위: {rule["lev_min"]}~{rule["lev_max"]}

응답은 JSON만. 쉬운 말로 설명해야 함.
"""

    user = f"""
시장 요약 데이터(JSON):
{json.dumps(pack, ensure_ascii=False)}

원하는 출력(JSON):
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0-100,
  "entry_pct": {rule["entry_pct_min"]}-{rule["entry_pct_max"]},
  "leverage": {rule["lev_min"]}-{rule["lev_max"]},
  "sl_pct": 0.5-6.0,
  "tp_pct": 1.0-20.0,
  "rr": 0.5-6.0,
  "used_indicators": ["RSI", "추세(MA)", "ADX", "볼린저", "MACD" ...],
  "reason_easy": "초보도 이해할 쉬운 문장으로 (괄호로 뜻도 설명)"
}}
조건:
- 확신이 낮으면 무조건 hold
- 'pullback_candidate'가 True면 (상승추세 눌림목 반등) 쪽에 가산점
- 손절은 짧게, 익절은 추세 강하면 길게(ADX가 높을수록 tp_pct 늘릴 수 있음)
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        out = json.loads(resp.choices[0].message.content)

        # 안전 클램프(모드 룰)
        out["confidence"] = int(max(0, min(100, int(out.get("confidence", 0)))))
        out["entry_pct"] = float(out.get("entry_pct", rule["entry_pct_min"]))
        out["entry_pct"] = float(np.clip(out["entry_pct"], rule["entry_pct_min"], rule["entry_pct_max"]))

        out["leverage"] = int(out.get("leverage", rule["lev_min"]))
        out["leverage"] = int(np.clip(out["leverage"], rule["lev_min"], rule["lev_max"]))

        out["sl_pct"] = float(out.get("sl_pct", 1.2))
        out["tp_pct"] = float(out.get("tp_pct", 3.0))
        out["rr"] = float(out.get("rr", 1.5))

        if out.get("decision") not in ["buy", "sell", "hold"]:
            out["decision"] = "hold"

        # 최소 확신도 미달이면 hold
        if out["decision"] in ["buy", "sell"] and out["confidence"] < rule["min_conf"]:
            out["decision"] = "hold"

        return out

    except Exception as e:
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": []}


# =========================================================
# ✅ 10) 트레이딩뷰 차트(다크모드)
# =========================================================
def tv_symbol_from_ccxt(sym: str):
    # "BTC/USDT:USDT" -> "BITGET:BTCUSDT.P" 시도
    base = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    # perpetual 추정 ".P"
    return f"BITGET:{base}{quote}.P"

def render_tradingview(symbol_ccxt: str, height=560):
    tv = tv_symbol_from_ccxt(symbol_ccxt)
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv}",
          "interval": "5",
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
# ✅ 11) 경제 캘린더(쉬운 한글) - 안정 JSON 소스(가능하면)
# =========================================================
def get_forex_events_kr():
    """
    ForexFactory 주간 캘린더 JSON(불러오면 한글로 쉽게 정리)
    실패하면 빈 DF 반환
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        rows = []
        for x in data[:60]:
            # 필요한 필드만
            dt = x.get("date", "")
            tm = x.get("time", "")
            title = x.get("title", "")
            impact = x.get("impact", "")
            country = x.get("country", "")
            # impact 한국어
            imp_kr = {"High": "매우 중요", "Medium": "중요", "Low": "낮음"}.get(impact, impact)
            rows.append({
                "날짜": dt,
                "시간": tm,
                "국가": country,
                "지표": title,
                "중요도": imp_kr
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame(columns=["날짜", "시간", "국가", "지표", "중요도"])


# =========================================================
# ✅ 12) 자동매매 스레드 + 모니터 파일(하트비트/AI시야)
# =========================================================
def safe_fetch_positions(ex, symbols):
    try:
        return ex.fetch_positions(symbols)
    except TypeError:
        return ex.fetch_positions(symbols=symbols)
    except:
        return []

def get_free_usdt(ex):
    try:
        bal = ex.fetch_balance({"type": "swap"})
        free = float(bal["USDT"]["free"])
        total = float(bal["USDT"]["total"])
        return free, total
    except:
        return 0.0, 0.0

def get_price(ex, sym):
    try:
        t = ex.fetch_ticker(sym)
        return float(t["last"])
    except:
        return None

def close_position_market(ex, sym, side, contracts):
    # side: long/short or buy/sell
    try:
        if side in ["long", "buy"]:
            ex.create_market_order(sym, "sell", contracts)
        else:
            ex.create_market_order(sym, "buy", contracts)
        return True
    except:
        return False

def telegram_thread(ex):
    offset = 0
    active_targets = {}  # {symbol: {"sl":x,"tp":y,"entry_pct":..,"lev":..,"entry_usdt":..,"reason":..}}

    # 모니터 초기화
    monitor = read_json_safe(MONITOR_FILE, {"coins": {}}) or {"coins": {}}
    monitor["_boot_time_kst"] = now_kst_str()
    monitor["_last_write"] = 0
    write_json_atomic(MONITOR_FILE, monitor)

    # 시작 알림
    tg_send("🚀 AI 봇 가동 시작(모의투자). 상태/시야는 Streamlit에서 확인 가능!\n명령: 상태 / 시야 / 일지")

    menu_kb = {
        "inline_keyboard": [
            [{"text": "📊 포지션", "callback_data": "position"},
             {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "👁️ 시야(요약)", "callback_data": "vision"},
             {"text": "📜 일지(최근)", "callback_data": "log"}],
            [{"text": "🛑 전량청산", "callback_data": "close_all"}]
        ]
    }
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_id, "text": "✅ 텔레그램 메뉴 준비 완료", "reply_markup": json.dumps(menu_kb)},
            timeout=10
        )
    except:
        pass

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()

            # ✅ 하트비트 기록
            monitor["last_heartbeat_epoch"] = time.time()
            monitor["last_heartbeat_kst"] = now_kst_str()
            monitor["auto_trade"] = bool(cfg.get("auto_trade", False))
            monitor["trade_mode"] = cfg.get("trade_mode", "안전모드")
            monitor["pause_until"] = rt.get("pause_until", 0)
            monitor["consec_losses"] = rt.get("consec_losses", 0)

            # 자동매매 ON일 때만 스캔/매매
            if cfg.get("auto_trade", False):
                # 일시정지(연속손실 등) 처리
                if time.time() < float(rt.get("pause_until", 0)):
                    # 모니터 저장은 하되 스캔은 쉬기
                    if time.time() - monitor.get("_last_write", 0) > 1:
                        write_json_atomic(MONITOR_FILE, monitor)
                        monitor["_last_write"] = time.time()
                else:
                    mode = cfg.get("trade_mode", "안전모드")
                    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])

                    # 1) 포지션 관리(손절/익절)
                    for sym in TARGET_COINS:
                        positions = safe_fetch_positions(ex, [sym])
                        act = [p for p in positions if float(p.get("contracts") or 0) > 0]
                        if not act:
                            continue

                        p = act[0]
                        side = p.get("side") or p.get("positionSide") or "long"
                        entry = float(p.get("entryPrice") or 0)
                        contracts = float(p.get("contracts") or 0)

                        mark = float(p.get("markPrice") or (get_price(ex, sym) or entry))
                        lev = float(p.get("leverage") or 1)

                        # ROI% 추정
                        roi = p.get("percentage", None)
                        if roi is None:
                            if entry > 0:
                                raw = (mark - entry) / entry * 100.0
                                roi = raw * lev if side in ["long", "buy"] else (-raw * lev)
                            else:
                                roi = 0.0
                        roi = float(roi)

                        target = active_targets.get(sym, {"sl": -2.0, "tp": 4.0})
                        sl = float(target.get("sl", -2.0))
                        tp = float(target.get("tp", 4.0))

                        # 트레일링(옵션): 수익이 충분하면 손절을 본전 근처로 끌어올림
                        if cfg.get("use_trailing_stop", True):
                            if roi >= (tp * 0.5):
                                # 본전+수수료 정도로 방어(간단)
                                sl = max(sl, -0.3)

                        if roi <= -abs(sl):
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_price(ex, sym) or mark
                                pnl_usdt = float(p.get("unrealizedPnl") or 0)
                                log_trade(sym, side, entry, exit_px, pnl_usdt, roi, "자동 손절",
                                          one_line="손절(짧게 끊음) → 다음엔 해소 확인 더 엄격")
                                rt["consec_losses"] = int(rt.get("consec_losses", 0)) + 1
                                rt["daily_realized_pnl"] = float(rt.get("daily_realized_pnl", 0)) + pnl_usdt
                                save_runtime(rt)

                                tg_send(f"🩸 손절\n- 코인: {sym}\n- 수익률: {roi:.2f}%\n- 이유: 목표 손절 도달")
                                active_targets.pop(sym, None)

                                monitor["last_action"] = {"time_kst": now_kst_str(), "type": "STOP", "symbol": sym, "roi": roi}
                                write_json_atomic(MONITOR_FILE, monitor)
                                monitor["_last_write"] = time.time()

                        elif roi >= tp:
                            ok = close_position_market(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_price(ex, sym) or mark
                                pnl_usdt = float(p.get("unrealizedPnl") or 0)
                                log_trade(sym, side, entry, exit_px, pnl_usdt, roi, "자동 익절",
                                          one_line="익절(추세 수익) → 다음에도 같은 조건을 반복")
                                rt["consec_losses"] = 0
                                rt["daily_realized_pnl"] = float(rt.get("daily_realized_pnl", 0)) + pnl_usdt
                                save_runtime(rt)

                                tg_send(f"🎉 익절\n- 코인: {sym}\n- 수익률: +{roi:.2f}%\n- 이유: 목표 익절 도달")
                                active_targets.pop(sym, None)

                                monitor["last_action"] = {"time_kst": now_kst_str(), "type": "TAKE", "symbol": sym, "roi": roi}
                                write_json_atomic(MONITOR_FILE, monitor)
                                monitor["_last_write"] = time.time()

                    # 2) 신규 진입 스캔
                    free_usdt, total_usdt = get_free_usdt(ex)

                    for sym in TARGET_COINS:
                        # 이미 포지션 있으면 신규 진입 스킵
                        positions = safe_fetch_positions(ex, [sym])
                        act = [p for p in positions if float(p.get("contracts") or 0) > 0]
                        if act:
                            continue

                        # 쿨다운(코인별)
                        cd = rt.get("cooldowns", {}).get(sym, 0)
                        if time.time() < float(cd):
                            monitor["coins"].setdefault(sym, {})
                            monitor["coins"][sym]["skip_reason"] = "쿨다운(잠깐 쉬는 중)"
                            continue

                        try:
                            ohlcv = ex.fetch_ohlcv(sym, cfg.get("timeframe", "5m"), limit=200)
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except Exception as e:
                            monitor["coins"].setdefault(sym, {})
                            monitor["coins"][sym]["skip_reason"] = f"데이터 오류: {e}"
                            continue

                        df, status, last = calc_indicators(df, cfg)
                        if last is None:
                            monitor["coins"].setdefault(sym, {})
                            monitor["coins"][sym]["skip_reason"] = "지표 계산 실패(데이터 부족/ta 없음)"
                            continue

                        # ✅ 모니터 기본 상태 기록
                        cs = monitor["coins"].get(sym, {})
                        cs.update({
                            "last_scan_epoch": time.time(),
                            "last_scan_kst": now_kst_str(),
                            "price": float(last["close"]),
                            "trend": status.get("추세", ""),
                            "rsi": float(last.get("RSI", 0)),
                            "adx": float(last.get("ADX", 0)),
                            "bb": status.get("볼린저", ""),
                            "macd": status.get("MACD", ""),
                            "pullback_candidate": bool(status.get("_필터_눌림목반등후보")),
                        })
                        monitor["coins"][sym] = cs

                        # ✅ 필터: 애매한 횡보는 AI 호출도 하지 않음(비용 절약+휘둘림 방지)
                        # - 추세 약하고 RSI 해소도 아니면 스킵
                        call_ai = False
                        if status.get("_필터_눌림목반등후보"):
                            call_ai = True
                            cs["skip_reason"] = ""
                        elif status.get("_필터_RSI해소롱") or status.get("_필터_RSI해소숏"):
                            call_ai = True
                            cs["skip_reason"] = ""
                        elif float(last.get("ADX", 0)) >= 25:
                            call_ai = True
                            cs["skip_reason"] = ""
                        else:
                            cs["ai_called"] = False
                            cs["skip_reason"] = "횡보/해소 신호 없음(휩쏘 위험)"
                            monitor["coins"][sym] = cs
                            continue

                        # ✅ AI 판단
                        ai = ai_decide_trade(df, status, sym, mode)
                        decision = ai.get("decision", "hold")
                        conf_score = int(ai.get("confidence", 0))

                        cs.update({
                            "ai_called": True,
                            "ai_decision": decision,
                            "ai_confidence": conf_score,
                            "ai_reason_easy": (ai.get("reason_easy", "")[:160]),
                            "ai_entry_pct": float(ai.get("entry_pct", rule["entry_pct_min"])),
                            "ai_leverage": int(ai.get("leverage", rule["lev_min"])),
                            "ai_sl_pct": float(ai.get("sl_pct", 1.0)),
                            "ai_tp_pct": float(ai.get("tp_pct", 3.0)),
                            "ai_rr": float(ai.get("rr", 1.5)),
                            "min_conf_required": int(rule["min_conf"]),
                            "ai_used_indicators": ai.get("used_indicators", []),
                        })
                        monitor["coins"][sym] = cs

                        # 1초에 1번만 저장
                        if time.time() - monitor.get("_last_write", 0) > 1:
                            write_json_atomic(MONITOR_FILE, monitor)
                            monitor["_last_write"] = time.time()

                        # ✅ 진입 조건 만족 시 주문
                        if decision in ["buy", "sell"] and conf_score >= rule["min_conf"]:
                            entry_pct = float(ai.get("entry_pct"))
                            lev = int(ai.get("leverage"))
                            sl_pct = float(ai.get("sl_pct"))
                            tp_pct = float(ai.get("tp_pct"))

                            # 진입 금액(USDT)
                            entry_usdt = free_usdt * (entry_pct / 100.0)
                            if entry_usdt <= 1:
                                cs["skip_reason"] = "잔고 부족(진입금 너무 작음)"
                                continue

                            price = float(last["close"])
                            qty = (entry_usdt * lev) / price

                            try:
                                qty = float(ex.amount_to_precision(sym, qty))
                            except:
                                qty = float(qty)

                            if qty <= 0:
                                cs["skip_reason"] = "수량 계산 실패"
                                continue

                            # 레버리지 설정 시도
                            try:
                                ex.set_leverage(lev, sym)
                            except:
                                pass

                            # 주문
                            try:
                                ex.create_market_order(sym, decision, qty)

                                active_targets[sym] = {
                                    "sl": sl_pct,
                                    "tp": tp_pct,
                                    "entry_pct": entry_pct,
                                    "lev": lev,
                                    "entry_usdt": entry_usdt,
                                    "reason": ai.get("reason_easy", "")
                                }

                                tg_send(
                                    "🎯 진입\n"
                                    f"- 코인: {sym}\n"
                                    f"- 방향: {'롱(상승에 베팅)' if decision=='buy' else '숏(하락에 베팅)'}\n"
                                    f"- 확신도: {conf_score}% (기준 {rule['min_conf']}%)\n"
                                    f"- 진입금: {entry_usdt:.2f} USDT (잔고의 {entry_pct:.1f}%)\n"
                                    f"- 레버리지: x{lev} (배율)\n"
                                    f"- 목표익절: +{tp_pct:.2f}% / 목표손절: -{sl_pct:.2f}%\n"
                                    f"- 근거(쉬운말): {ai.get('reason_easy','')}\n"
                                    f"- AI가 본 지표: {', '.join(ai.get('used_indicators', []))}"
                                )

                                monitor["last_action"] = {
                                    "time_kst": now_kst_str(),
                                    "type": "ENTRY",
                                    "symbol": sym,
                                    "decision": decision,
                                    "conf": conf_score,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "lev": lev
                                }
                                write_json_atomic(MONITOR_FILE, monitor)
                                monitor["_last_write"] = time.time()

                                # 코인별 쿨다운(예: 60초)
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)

                                time.sleep(2)

                            except Exception as e:
                                cs["skip_reason"] = f"주문 실패: {e}"
                                monitor["coins"][sym] = cs

                        time.sleep(0.6)

            # ✅ 텔레그램 업데이트 처리(명령/버튼)
            try:
                res = requests.get(
                    f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1",
                    timeout=10
                ).json()
            except:
                res = {"ok": False}

            if res.get("ok"):
                for up in res.get("result", []):
                    offset = up.get("update_id", offset)

                    # 텍스트 명령
                    if "message" in up and "text" in up["message"]:
                        txt = up["message"]["text"].strip()
                        if txt == "상태":
                            rt = load_runtime()
                            free, total = get_free_usdt(ex)
                            tg_send(
                                "📡 상태\n"
                                f"- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg.get('trade_mode')}\n"
                                f"- 잔고: {total:.2f} USDT (사용가능 {free:.2f})\n"
                                f"- 연속손실: {rt.get('consec_losses', 0)}"
                            )

                        elif txt == "시야":
                            mon = read_json_safe(MONITOR_FILE, {})
                            lines = []
                            lines.append("👁️ AI 시야(요약)")
                            lines.append(f"- 자동매매: {'ON' if mon.get('auto_trade') else 'OFF'}")
                            lines.append(f"- 모드: {mon.get('trade_mode','-')}")
                            lines.append(f"- 마지막 하트비트: {mon.get('last_heartbeat_kst','-')}")
                            coins = mon.get("coins", {}) or {}
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) / "
                                    f"RSI {cs.get('rsi','-'):.1f} / ADX {cs.get('adx','-'):.1f} / "
                                    f"{(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:30]}"
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
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {r['PnL_Percent']:.2f}% | {str(r.get('OneLine',''))[:40]}")
                                tg_send("\n".join(msg))

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up["callback_query"]
                        data = cb.get("data", "")
                        cb_id = cb.get("id", "")
                        cid = cb["message"]["chat"]["id"]

                        if data == "balance":
                            free, total = get_free_usdt(ex)
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
                                    side = p.get("side", "long")
                                    roi = p.get("percentage", 0.0)
                                    msg.append(f"- {sym}: {side} (수익률 {float(roi):.2f}%)")
                            if not has:
                                msg.append("- 없음(관망)")
                            tg_send("\n".join(msg))

                        elif data == "vision":
                            mon = read_json_safe(MONITOR_FILE, {})
                            lines = []
                            lines.append("👁️ AI 시야(요약)")
                            lines.append(f"- 마지막 하트비트: {mon.get('last_heartbeat_kst','-')}")
                            coins = mon.get("coins", {}) or {}
                            for sym, cs in list(coins.items())[:10]:
                                lines.append(
                                    f"- {sym}: {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                    f"/ RSI {cs.get('rsi','-'):.1f} / ADX {cs.get('adx','-'):.1f}"
                                )
                            tg_send("\n".join(lines))

                        elif data == "log":
                            df_log = read_trade_log()
                            if df_log.empty:
                                tg_send("📜 일지 없음")
                            else:
                                top = df_log.head(8)
                                msg = ["📜 최근 매매일지(요약)"]
                                for _, r in top.iterrows():
                                    msg.append(f"- {r['Time']} {r['Coin']} {r['Side']} {r['PnL_Percent']:.2f}% | {str(r.get('OneLine',''))[:40]}")
                                tg_send("\n".join(msg))

                        elif data == "close_all":
                            tg_send("🛑 전량 청산 시도")
                            for sym in TARGET_COINS:
                                ps = safe_fetch_positions(ex, [sym])
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    continue
                                p = act[0]
                                side = p.get("side", "long")
                                contracts = float(p.get("contracts") or 0)
                                close_position_market(ex, sym, side, contracts)
                            tg_send("✅ 전량 청산 요청 완료")

                        tg_answer_callback(cb_id)

            except:
                pass

            # 모니터 저장(너무 자주 X)
            if time.time() - monitor.get("_last_write", 0) > 2:
                write_json_atomic(MONITOR_FILE, monitor)
                monitor["_last_write"] = time.time()

            time.sleep(0.7)

        except Exception as e:
            tg_send(f"⚠️ 스레드 오류: {e}")
            time.sleep(3)


# 스레드 1회 실행
found = any(t.name == "TG_THREAD" for t in threading.enumerate())
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_THREAD")
    add_script_run_ctx(t)
    t.start()


# =========================================================
# ✅ 13) Streamlit UI (제어판 + 차트 + 포지션 + 일지 + AI시야)
# =========================================================
st.sidebar.title("🛠️ 제어판(설정)")
st.sidebar.caption("Streamlit은 제어/상태 확인용, 실시간 알림/보고는 텔레그램에서!")

# OpenAI 키 입력(선택)
if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

# 모드 / 자동매매
config["trade_mode"] = st.sidebar.selectbox("매매 모드", list(MODE_RULES.keys()),
                                            index=list(MODE_RULES.keys()).index(config.get("trade_mode", "안전모드")))
auto_on = st.sidebar.checkbox("🤖 자동매매(텔레그램 연동)", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.divider()
config["ai_reco_show"] = st.sidebar.checkbox("AI 추천값 표시", value=bool(config.get("ai_reco_show", True)))
config["ai_reco_apply"] = st.sidebar.checkbox("AI 추천값 자동적용(고급)", value=bool(config.get("ai_reco_apply", False)))
save_settings(config)

st.sidebar.divider()

# 코인/타임프레임
markets = exchange.markets or {}
symbol_list = [s for s in markets.keys() if markets[s].get("linear") and markets[s].get("swap")] or TARGET_COINS
symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)
timeframe = st.sidebar.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m","3m","5m","15m","1h"].index(config.get("timeframe","5m")))
config["timeframe"] = timeframe
save_settings(config)

# 지표 ON/OFF (10종)
st.sidebar.subheader("📊 지표 ON/OFF (10종)")
cols = st.sidebar.columns(2)
config["use_rsi"] = cols[0].checkbox("RSI", value=bool(config.get("use_rsi", True)))
config["use_bb"] = cols[1].checkbox("볼린저", value=bool(config.get("use_bb", True)))
config["use_ma"] = cols[0].checkbox("이평(MA)", value=bool(config.get("use_ma", True)))
config["use_macd"] = cols[1].checkbox("MACD", value=bool(config.get("use_macd", True)))
config["use_stoch"] = cols[0].checkbox("스토캐스틱", value=bool(config.get("use_stoch", True)))
config["use_cci"] = cols[1].checkbox("CCI", value=bool(config.get("use_cci", True)))
config["use_mfi"] = cols[0].checkbox("MFI", value=bool(config.get("use_mfi", True)))
config["use_willr"] = cols[1].checkbox("윌리엄%R", value=bool(config.get("use_willr", True)))
config["use_adx"] = cols[0].checkbox("ADX", value=bool(config.get("use_adx", True)))
config["use_vol"] = cols[1].checkbox("거래량", value=bool(config.get("use_vol", True)))

st.sidebar.divider()
config["use_trailing_stop"] = st.sidebar.checkbox("🚀 트레일링(수익나면 손절 끌어올림)", value=bool(config.get("use_trailing_stop", True)))
save_settings(config)

# 잔고/포지션 요약(사이드바)
st.sidebar.header("💰 내 지갑(요약)")
free, total = get_free_usdt(exchange)
st.sidebar.metric("총 자산(USDT)", f"{total:.2f}")
st.sidebar.metric("사용 가능", f"{free:.2f}")

st.sidebar.divider()
st.sidebar.subheader("📌 포지션(요약)")
try:
    ps = safe_fetch_positions(exchange, TARGET_COINS)
    act = [p for p in ps if float(p.get("contracts") or 0) > 0]
    if not act:
        st.sidebar.caption("무포지션(관망)")
    else:
        for p in act[:8]:
            symp = p.get("symbol", "")
            side = p.get("side", "long")
            roi = float(p.get("percentage") or 0)
            st.sidebar.write(f"- {symp} / {side} / {roi:.2f}%")
except:
    st.sidebar.caption("포지션 조회 실패")


# =========================================================
# ✅ 메인 화면
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트")
st.caption("Streamlit: 제어판/상태, Telegram: 실시간 보고/조회")

# 상단: 차트 + 지표
cL, cR = st.columns([2, 1], gap="large")

with cL:
    st.subheader("📉 트레이딩뷰 차트(다크모드)")
    render_tradingview(symbol, height=560)

with cR:
    st.subheader("🧾 현재 지표 요약")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, config.get("timeframe","5m"), limit=200)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df2, status, last = calc_indicators(df, config)
        if last is not None:
            st.metric("현재가", f"{float(last['close']):.4f}")
            st.write(status)
        else:
            st.warning("지표 계산 실패(데이터/ta 확인)")
    except Exception as e:
        st.error(f"데이터 오류: {e}")

st.divider()

# 탭
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("👁️ 실시간 AI 모니터링(봇 시야)")
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, key="mon_refresh")  # 2초마다 갱신
    except:
        st.caption("자동 새로고침을 쓰려면 requirements.txt에 streamlit-autorefresh 추가하세요.")
        st.button("🔄 수동 새로고침")

    mon = read_json_safe(MONITOR_FILE, None)
    if not mon:
        st.warning("모니터 파일이 아직 없습니다. (스레드 시작 확인)")
    else:
        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = time.time() - hb if hb else 9999

        a,b,c,d = st.columns(4)
        a.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        b.metric("모드", mon.get("trade_mode","-"))
        c.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < 6 else "🔴 멈춤 의심")
        d.metric("연속손실", str(mon.get("consec_losses", 0)))

        if age >= 6:
            st.error("⚠️ 자동매매 스레드가 멈췄거나(크래시) 파일 갱신이 멈춘 상태일 수 있어요.")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0))
            scan_age = time.time() - last_scan if last_scan else 9999

            rows.append({
                "코인": sym,
                "스캔(초전)": f"{scan_age:.1f}",
                "가격": cs.get("price"),
                "추세": cs.get("trend"),
                "RSI": cs.get("rsi"),
                "ADX": cs.get("adx"),
                "볼린저": cs.get("bb"),
                "MACD": cs.get("macd"),
                "눌림목후보": "✅" if cs.get("pullback_candidate") else "—",
                "AI호출": "✅" if cs.get("ai_called") else "—",
                "AI결론": str(cs.get("ai_decision","-")).upper(),
                "확신도": cs.get("ai_confidence","-"),
                "진입%": cs.get("ai_entry_pct","-"),
                "레버": cs.get("ai_leverage","-"),
                "RR": cs.get("ai_rr","-"),
                "AI가 본 지표": ", ".join(cs.get("ai_used_indicators", []) or []),
                "스킵/근거": cs.get("skip_reason") or cs.get("ai_reason_easy") or "",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("스캔(초전)"), width="stretch", hide_index=True)
        else:
            st.caption("아직 스캔 데이터 없음")

    st.divider()
    st.subheader("🔍 현재 코인 AI 분석(버튼)")
    if st.button("AI 분석 실행(현재 코인)"):
        if last is None:
            st.warning("데이터 부족")
        else:
            ai = ai_decide_trade(df2, status, symbol, config.get("trade_mode","안전모드"))
            st.write(ai)

with t2:
    st.subheader("⚡ 수동 주문(테스트용)")
    st.caption("여기는 수동 컨트롤(필요하면 더 확장 가능)")
    amt = st.number_input("주문금액(USDT)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    config["order_usdt"] = float(amt)
    save_settings(config)
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱(매수)"):
        st.info("수동 주문은 안전을 위해 기본 비활성(원하면 구현해줄게)")
    if b2.button("🔴 숏(매도)"):
        st.info("수동 주문은 안전을 위해 기본 비활성(원하면 구현해줄게)")
    if b3.button("🚫 포지션 종료"):
        st.info("수동 종료는 텔레그램 '전량청산'을 권장")

with t3:
    st.subheader("📅 시장정보(경제 캘린더)")
    ev = get_forex_events_kr()
    if ev.empty:
        st.caption("일정 없음/불러오기 실패(네트워크 제한일 수 있음)")
    else:
        st.dataframe(ev, width="stretch", hide_index=True)

with t4:
    st.subheader("📜 매매일지(보기 쉽게)")
    df_log = read_trade_log()
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔄 새로고침"):
            st.rerun()
    with col2:
        if st.button("🧹 매매일지 초기화(삭제)"):
            reset_trade_log()
            st.success("매매일지 초기화 완료")
            st.rerun()

    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        # 보기 편하게 필요한 컬럼만 위쪽에
        show_cols = [c for c in ["Time","Coin","Side","PnL_Percent","PnL_USDT","OneLine","Reason"] if c in df_log.columns]
        st.dataframe(df_log[show_cols], width="stretch", hide_index=True)

st.caption("⚠️ 투자/트레이딩은 손실 위험이 큽니다. 이 봇은 모의투자에서 충분히 검증 후 사용하세요.")
