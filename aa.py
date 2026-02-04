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

# ✅ 지표 라이브러리 (pip install ta)
import ta

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True  # 실전 매매 시 False
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트")

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT"
]

# =========================================================
# 📝 매매일지 (CSV)
# =========================================================
def log_trade(coin, side, entry_price, exit_price, pnl_amount, pnl_percent, reason):
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([{
            "Time": now,
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Reason": reason
        }])

        if not os.path.exists(LOG_FILE):
            new_data.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            new_data.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"Log Error: {e}")


def get_past_mistakes():
    try:
        if not os.path.exists(LOG_FILE):
            return "과거 매매 기록 없음."
        df = pd.read_csv(LOG_FILE)
        worst = df.sort_values(by="PnL_Percent", ascending=True).head(5)
        if worst.empty:
            return "큰 손실 기록 없음."
        s = ""
        for _, r in worst.iterrows():
            s += f"- {r['Coin']} {r['Side']} 진입 후 {r['PnL_Percent']}% (이유: {r.get('Reason','기록없음')})\n"
        return s
    except:
        return "기록 조회 실패"


# =========================================================
# 💾 설정
# =========================================================
def load_settings():
    default = {
        "openai_api_key": "",
        "auto_trade": False,
        "order_usdt": 100.0,
        "leverage": 5,

        # 지표 파라미터
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 지표 사용 여부 (10종)
        "use_rsi": True, "use_bb": True, "use_ma": True, "use_macd": True,
        "use_stoch": True, "use_cci": True, "use_mfi": True, "use_willr": True,
        "use_adx": True, "use_vol": True,

        "target_vote": 2,
        "no_trade_weekend": False
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                default.update(saved)
        except:
            pass
    return default


def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        st.toast("✅ 설정 저장 완료", icon="💾")
    except:
        st.error("설정 저장 실패")


config = load_settings()
if "order_usdt" not in st.session_state:
    st.session_state["order_usdt"] = config["order_usdt"]

# =========================================================
# 🔐 Secrets 로드
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

if not api_key:
    st.error("🚨 Bitget API Key가 secrets에 없습니다 (API_KEY).")
    st.stop()

if not openai_key:
    st.error("🚨 OpenAI API Key가 secrets에 없습니다 (OPENAI_API_KEY).")
    st.stop()

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
# 📈 TradingView 위젯
# =========================================================
def to_tv_symbol(ccxt_symbol: str) -> str:
    # 예: "BTC/USDT:USDT" -> "BITGET:BTCUSDT.P"
    base = ccxt_symbol.split("/")[0].replace(":", "")
    quote = "USDT"
    return f"BITGET:{base}{quote}.P"


def tf_to_tv_interval(tf: str) -> str:
    # TradingView interval: "1","5","15","60","240","D"
    m = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D"}
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
# 🧮 10종 지표 계산 (통합)
# =========================================================
def calc_indicators(df: pd.DataFrame, cfg: dict):
    """
    df columns required: time, open, high, low, close, volume
    returns: (df, status_dict, last_row)
    """
    try:
        if df is None or df.empty or len(df) < 120:
            return df, {}, None

        # RSI
        df["RSI"] = ta.momentum.rsi(df["close"], window=int(cfg["rsi_period"]))

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=int(cfg["bb_period"]), window_dev=float(cfg["bb_std"]))
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_mid"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        # MA (fast/slow)
        df["MA_fast"] = ta.trend.sma_indicator(df["close"], window=int(cfg["ma_fast"]))
        df["MA_slow"] = ta.trend.sma_indicator(df["close"], window=int(cfg["ma_slow"]))

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=int(cfg["stoch_k"]), smooth_window=3
        )
        df["STO_K"] = stoch.stoch()
        df["STO_D"] = stoch.stoch_signal()

        # CCI
        df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)

        # MFI
        df["MFI"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14)

        # Williams %R
        df["WILLR"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)

        # ADX
        df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

        # Volume Spike
        df["VOL_MA"] = df["volume"].rolling(20).mean()
        df["VOL_SPIKE"] = df["volume"] / (df["VOL_MA"] + 1e-9)

        df = df.dropna()
        if df.empty:
            return df, {}, None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        status = {}

        # 1) RSI
        if last["RSI"] >= cfg["rsi_sell"]:
            status["RSI"] = "🔴 과매수"
        elif last["RSI"] <= cfg["rsi_buy"]:
            status["RSI"] = "🟢 과매도"
        else:
            status["RSI"] = "⚪ 중립"

        # 2) BB
        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

        # 3) MA
        if last["MA_fast"] > last["MA_slow"]:
            status["MA"] = "📈 (단기>장기)"
        else:
            status["MA"] = "📉 (단기<장기)"

        # 4) MACD
        if last["MACD"] > last["MACD_signal"]:
            status["MACD"] = "📈 골든"
        else:
            status["MACD"] = "📉 데드"

        # 5) STOCH
        if last["STO_K"] > 80:
            status["STOCH"] = "🔴 과열"
        elif last["STO_K"] < 20:
            status["STOCH"] = "🟢 침체"
        else:
            status["STOCH"] = "⚪ 중립"

        # 6) CCI
        if last["CCI"] > 100:
            status["CCI"] = "🔴 과열"
        elif last["CCI"] < -100:
            status["CCI"] = "🟢 침체"
        else:
            status["CCI"] = "⚪ 중립"

        # 7) MFI
        if last["MFI"] > 80:
            status["MFI"] = "🔴 과열"
        elif last["MFI"] < 20:
            status["MFI"] = "🟢 침체"
        else:
            status["MFI"] = "⚪ 중립"

        # 8) WILLR (range: -100 ~ 0)
        if last["WILLR"] > -20:
            status["WILLR"] = "🔴 과열"
        elif last["WILLR"] < -80:
            status["WILLR"] = "🟢 침체"
        else:
            status["WILLR"] = "⚪ 중립"

        # 9) ADX
        status["ADX"] = "🔥 추세장" if last["ADX"] >= 25 else "💤 횡보장"

        # 10) VOL
        vmul = float(cfg["vol_mul"])
        status["VOL"] = "🔥 거래량 폭증" if last["VOL_SPIKE"] >= vmul else "⚪ 보통"

        # 보조: RSI 반등/하락(직전 대비)
        status["RSI_FLOW"] = "↗️ 반등" if last["RSI"] > prev["RSI"] else "↘️ 약화"

        return df, status, last
    except Exception as e:
        print(f"Calc Error: {e}")
        return df, {}, None


# =========================================================
# 🧠 OpenAI 전략
# =========================================================
def generate_wonyousi_strategy(df: pd.DataFrame, status_summary: dict):
    try:
        if df is None or df.empty or len(df) < 3:
            return {"decision": "hold", "confidence": 0, "reason": "데이터 부족"}

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        past_mistakes = get_past_mistakes()

        system_prompt = f"""
당신은 매우 보수적인 '스윙 트레이더'입니다.

[과거 실수]
{past_mistakes}

[원칙]
1) RSI가 과매도/과매수 '구간'에서 바로 들어가지 말고, 구간을 탈출하는 '반등/반락'을 확인 후 진입
2) 손절폭(sl_gap)은 최소 2.5% 이상
3) 레버리지는 3~10배 권장 (20배 금지)
4) 애매하면 HOLD

[응답(JSON)]
decision(buy/sell/hold), percentage(10~30), leverage(3~10), sl_gap(2.5~6), tp_gap(5~15), confidence(0~100), reason
        """.strip()

        user_prompt = f"""
[시장]
- 현재가: {last_row['close']}
- RSI: {prev_row['RSI']:.1f} -> {last_row['RSI']:.1f} ({status_summary.get('RSI_FLOW','')})
- ADX: {last_row['ADX']:.1f} ({status_summary.get('ADX','')})
- BB: {status_summary.get('BB','')}
- MACD: {status_summary.get('MACD','')}
- MA: {status_summary.get('MA','')}
- VOL: {status_summary.get('VOL','')}

반전이 확실하지 않으면 confidence 80 이상 주지 마세요.
        """.strip()

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.25
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"decision": "hold", "confidence": 0, "reason": f"AI 오류: {e}"}


# =========================================================
# 🤖 텔레그램 + 자동매매 스레드 (필요한 부분만 유지)
# =========================================================
def side_to_close_order(side: str) -> str:
    # ccxt 통합 포지션 side 케이스 방어
    s = (side or "").lower()
    if s in ["long", "buy"]:
        return "sell"
    return "buy"


def telegram_thread(ex):
    active_trades = {}
    offset = 0
    last_report_time = time.time()
    REPORT_INTERVAL = 900

    def tg_send(text):
        try:
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                          data={"chat_id": tg_id, "text": text})
        except:
            pass

    tg_send("🚀 AI 봇 가동 시작")

    while True:
        try:
            cur_cfg = load_settings()
            if cur_cfg.get("auto_trade", False):

                # 포지션 개수 확인 (컷라인)
                active_pos_count = 0
                for c in TARGET_COINS:
                    try:
                        p = ex.fetch_positions([c])
                        if any(float(x.get("contracts", 0)) > 0 for x in p):
                            active_pos_count += 1
                    except:
                        pass
                required_conf = 85 if active_pos_count >= 1 else 80

                for coin in TARGET_COINS:
                    try:
                        # 포지션 관리
                        positions = ex.fetch_positions([coin])
                        active_ps = [p for p in positions if float(p.get("contracts", 0)) > 0]
                        if active_ps:
                            p = active_ps[0]
                            pnl_pct = float(p.get("percentage", 0))
                            target = active_trades.get(coin, {"sl": 4.0, "tp": 8.0})

                            if pnl_pct <= -abs(target["sl"]) or pnl_pct >= abs(target["tp"]):
                                close_side = side_to_close_order(p.get("side"))
                                ex.create_market_order(coin, close_side, p.get("contracts"))
                                tg_send(f"✅ 청산: {coin} ({pnl_pct:.2f}%)")
                                if coin in active_trades:
                                    del active_trades[coin]
                            continue

                        # 신규 진입
                        ohlcv = ex.fetch_ohlcv(coin, "5m", limit=150)
                        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
                        df["time"] = pd.to_datetime(df["time"], unit="ms")

                        df, status, last = calc_indicators(df, cur_cfg)
                        if last is None:
                            continue

                        # 횡보장 필터
                        if 30 <= last["RSI"] <= 70 and last["ADX"] < 20:
                            continue

                        strat = generate_wonyousi_strategy(df, status)
                        decision = strat.get("decision", "hold")
                        conf = float(strat.get("confidence", 0))

                        if decision in ["buy", "sell"] and conf >= required_conf:
                            lev = int(strat.get("leverage", 5))
                            lev = max(3, min(lev, 10))

                            sl = float(strat.get("sl_gap", 3.0))
                            sl = max(2.5, sl)
                            tp = float(strat.get("tp_gap", 6.0))

                            pct = float(strat.get("percentage", 10))
                            pct = min(max(pct, 5), 30)

                            try:
                                ex.set_leverage(lev, coin)
                            except:
                                pass

                            bal = ex.fetch_balance({"type": "swap"})
                            usdt_free = float(bal["USDT"]["free"])
                            amt = usdt_free * (pct / 100.0)
                            qty = ex.amount_to_precision(coin, (amt * lev) / float(last["close"]))

                            if float(qty) > 0:
                                ex.create_market_order(coin, decision, qty)
                                active_trades[coin] = {"sl": sl, "tp": tp}
                                tg_send(f"🎯 진입: {coin} {decision.upper()} / conf={conf}% / x{lev} / TP {tp}% SL {sl}%")
                                time.sleep(10)

                    except Exception as e:
                        print("Auto Error:", coin, e)
                    time.sleep(0.8)

            # 생존 신고
            if time.time() - last_report_time >= REPORT_INTERVAL:
                try:
                    bal = ex.fetch_balance({"type": "swap"})
                    tg_send(f"💤 생존신고: USDT={float(bal['USDT']['total']):,.2f}")
                except:
                    pass
                last_report_time = time.time()

            # 버튼 콜백은 필요 시 추가
            time.sleep(1)

        except Exception as e:
            print("Thread Error:", e)
            time.sleep(5)


# =========================================================
# 🧾 데이터 로드
# =========================================================
def fetch_ohlcv_df(ex, sym: str, tf: str, limit: int = 150):
    ohlcv = ex.fetch_ohlcv(sym, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


# =========================================================
# 🧩 UI: 사이드바 (설정)
# =========================================================
st.sidebar.title("🛠️ 설정")

markets = exchange.markets
symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)

timeframe = st.sidebar.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2)

st.sidebar.divider()
st.sidebar.subheader("🤖 자동매매")
auto_on = st.sidebar.checkbox("자동매매 활성화(텔레그램)", value=config.get("auto_trade", False))
if auto_on != config.get("auto_trade", False):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📊 지표 사용(10종)")
config["use_rsi"] = st.sidebar.checkbox("RSI", value=config["use_rsi"])
config["use_bb"] = st.sidebar.checkbox("Bollinger Bands", value=config["use_bb"])
config["use_ma"] = st.sidebar.checkbox("MA (fast/slow)", value=config["use_ma"])
config["use_macd"] = st.sidebar.checkbox("MACD", value=config["use_macd"])
config["use_stoch"] = st.sidebar.checkbox("Stochastic", value=config["use_stoch"])
config["use_cci"] = st.sidebar.checkbox("CCI", value=config["use_cci"])
config["use_mfi"] = st.sidebar.checkbox("MFI", value=config["use_mfi"])
config["use_willr"] = st.sidebar.checkbox("Williams %R", value=config["use_willr"])
config["use_adx"] = st.sidebar.checkbox("ADX", value=config["use_adx"])
config["use_vol"] = st.sidebar.checkbox("Volume Spike", value=config["use_vol"])

st.sidebar.divider()
st.sidebar.subheader("지표 파라미터")
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
config["leverage"] = st.sidebar.slider("기본 레버리지(UI)", 1, 50, int(config["leverage"]))

st.sidebar.divider()
if st.sidebar.button("💾 설정 저장"):
    save_settings(config)

# ✅ OpenAI 연결 테스트
st.sidebar.divider()
st.sidebar.header("🔍 긴급 점검")
if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    try:
        test_client = OpenAI(api_key=openai_key)
        resp = test_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
            max_tokens=10
        )
        st.sidebar.success("✅ 연결 성공: " + resp.choices[0].message.content)
    except Exception as e:
        st.sidebar.error(f"❌ 연결 실패: {e}")

# =========================================================
# ✅ 자동매매 스레드 시작 (1회만)
# =========================================================
if "bot_thread_started" not in st.session_state:
    st.session_state["bot_thread_started"] = False

if not st.session_state["bot_thread_started"]:
    th = threading.Thread(target=telegram_thread, args=(exchange,), daemon=True, name="TG_Thread")
    add_script_run_ctx(th)
    th.start()
    st.session_state["bot_thread_started"] = True

# =========================================================
# 🧱 메인 화면 레이아웃 (요청한 요소 모두 “메인”에 표시)
# =========================================================
st.title("📌 비트겟 AI 워뇨띠 에이전트")

# 상단 상태 바
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
    df0 = fetch_ohlcv_df(exchange, symbol, timeframe, limit=200)
    df, status, last = calc_indicators(df0, config)
    data_loaded = last is not None
except Exception as e:
    st.error(f"⚠️ 데이터 로딩 오류: {e}")

if not data_loaded:
    st.warning("⏳ 데이터 로딩 중... (리런해보세요)")
    st.stop()

# =========================================================
# ✅ 메인: 좌(차트/지표) + 우(지갑/포지션/요약)
# =========================================================
left, right = st.columns([3.2, 1.8], gap="large")

with left:
    st.subheader("📈 TradingView 차트")
    render_tradingview(symbol, timeframe, height=520, theme="dark")

    st.divider()
    st.subheader("🚦 10종 보조지표 상태판")

    # 10종 상태만 골라 표시 (사용 체크 여부 반영)
    indi_rows = []
    def add_row(name, key, val, state):
        indi_rows.append({"지표": name, "값": val, "상태": state})

    if config["use_rsi"]:
        add_row("RSI", "RSI", f"{last['RSI']:.1f}", f"{status.get('RSI','')} {status.get('RSI_FLOW','')}")
    if config["use_bb"]:
        add_row("Bollinger", "BB", f"{last['BB_mid']:.2f}", status.get("BB",""))
    if config["use_ma"]:
        add_row("MA(fast/slow)", "MA", f"{last['MA_fast']:.2f}/{last['MA_slow']:.2f}", status.get("MA",""))
    if config["use_macd"]:
        add_row("MACD", "MACD", f"{last['MACD']:.4f}", status.get("MACD",""))
    if config["use_stoch"]:
        add_row("Stoch(K/D)", "STO", f"{last['STO_K']:.1f}/{last['STO_D']:.1f}", status.get("STOCH",""))
    if config["use_cci"]:
        add_row("CCI", "CCI", f"{last['CCI']:.1f}", status.get("CCI",""))
    if config["use_mfi"]:
        add_row("MFI", "MFI", f"{last['MFI']:.1f}", status.get("MFI",""))
    if config["use_willr"]:
        add_row("Williams %R", "WILLR", f"{last['WILLR']:.1f}", status.get("WILLR",""))
    if config["use_adx"]:
        add_row("ADX", "ADX", f"{last['ADX']:.1f}", status.get("ADX",""))
    if config["use_vol"]:
        add_row("Volume Spike", "VOL", f"{last['VOL_SPIKE']:.2f}x", status.get("VOL",""))

    st.dataframe(pd.DataFrame(indi_rows), use_container_width=True, hide_index=True)

    st.caption("※ 지표는 참고용이며, 데모(IS_SANDBOX=True)에서 충분히 검증 후 실전 전환하세요.")

with right:
    st.subheader("💰 내 잔고 / 포지션")
    try:
        bal = exchange.fetch_balance({"type": "swap"})
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
        active_positions = [p for p in positions if float(p.get("contracts", 0)) > 0]
        if not active_positions:
            st.info("무포지션 (관망 중)")
        else:
            for p in active_positions:
                sym = (p.get("symbol","")).split(":")[0]
                side = (p.get("side","")).lower()
                side_label = "🟢 Long" if side in ["long", "buy"] else "🔴 Short"
                pnl = float(p.get("unrealizedPnl", 0))
                roi = float(p.get("percentage", 0))
                lev = p.get("leverage", "?")
                st.info(f"**{sym}** ({side_label} x{lev})\n\n수익률: **{roi:.2f}%**  / 손익: **${pnl:.2f}**")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")

    st.divider()
    st.subheader("🤖 빠른 AI 요약")
    if st.button("🔍 지금 이 코인 AI 분석"):
        with st.spinner("AI가 차트를 분석 중..."):
            ai = generate_wonyousi_strategy(df, status)
            st.write(ai)

# =========================================================
# ✅ 탭(t1~t4) 유지 (요청대로)
# =========================================================
st.divider()
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지"])

with t1:
    st.subheader("🧠 워뇨띠 AI 전략 센터")

    c_auto, c_stat = st.columns([3, 1])
    with c_auto:
        auto_on2 = st.checkbox("🤖 24시간 자동매매 활성화 (텔레그램 연동)", value=config.get("auto_trade", False))
        if auto_on2 != config.get("auto_trade", False):
            config["auto_trade"] = auto_on2
            save_settings(config)
            st.rerun()
    with c_stat:
        st.caption("상태: " + ("🟢 가동중" if config.get("auto_trade") else "🔴 정지"))

    st.divider()
    col1, col2 = st.columns(2)

    if col1.button("🔍 현재 차트 분석 (This Coin)"):
        with st.spinner("AI 분석 중..."):
            ai_res = generate_wonyousi_strategy(df, status)
            decision = ai_res.get("decision", "hold").upper()
            conf = ai_res.get("confidence", 0)
            reason = ai_res.get("reason", "")

            if decision == "BUY":
                st.success(f"결론: 🟢 BUY (확신도 {conf}%)")
            elif decision == "SELL":
                st.error(f"결론: 🔴 SELL (확신도 {conf}%)")
            else:
                st.warning(f"결론: ⚪ HOLD (확신도 {conf}%)")
            st.info(f"근거: {reason}")

    if col2.button("🌍 전체 코인 스캔 (All Coins)"):
        ph = st.empty()
        ph.info("🕵️ 5개 코인 분석 중...")
        rows = []
        pb = st.progress(0.0)

        for i, c in enumerate(TARGET_COINS):
            try:
                dfx = fetch_ohlcv_df(exchange, c, "5m", limit=200)
                dfx, stx, lastx = calc_indicators(dfx, config)
                res = generate_wonyousi_strategy(dfx, stx)
                rows.append({
                    "코인": c.split("/")[0],
                    "현재가": f"{lastx['close']:.4f}",
                    "결론": res.get("decision", "hold").upper(),
                    "확신도": res.get("confidence", 0),
                    "요약": (res.get("reason", "")[:40] + "...") if res.get("reason") else ""
                })
            except Exception as e:
                rows.append({"코인": c, "결론": "ERROR", "요약": str(e)})
            pb.progress((i + 1) / len(TARGET_COINS))

        ph.success("✅ 스캔 완료")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t2:
    st.subheader("⚡ 수동주문")
    st.caption("※ 여기는 UI만 준비(버튼)해두고, 실제 주문 로직은 연결하면 됩니다.")
    m_amt = st.number_input("주문 금액 ($)", 0.0, 100000.0, float(config["order_usdt"]))
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입"):
        st.info("롱 진입 로직 연결 위치")
    if b2.button("🔴 숏 진입"):
        st.info("숏 진입 로직 연결 위치")
    if b3.button("🚫 포지션 종료"):
        st.info("포지션 종료 로직 연결 위치")

with t3:
    st.subheader("📅 시장정보")
    st.write("경제 일정/뉴스는 별도 크롤링/API로 붙이면 됩니다. (현재는 빈 화면)")

with t4:
    st.subheader("📜 매매일지 (trade_log.csv)")
    if os.path.exists(LOG_FILE):
        try:
            hist = pd.read_csv(LOG_FILE)
            if "Time" in hist.columns:
                hist = hist.sort_values(by="Time", ascending=False)
            st.dataframe(hist, use_container_width=True, hide_index=True)

            csv = hist.to_csv(index=False).encode("utf-8-sig")
            st.download_button("💾 CSV 다운로드", csv, "trade_log.csv", "text/csv")
        except Exception as e:
            st.error(f"로그 읽기 오류: {e}")
    else:
        st.info("아직 기록된 매매가 없습니다.")

    if st.button("🧪 테스트 데이터 입력"):
        log_trade("BTC/TEST", "long", 50000, 49000, -100, -2.0, "테스트")
        st.success("테스트 기록 저장 완료")
        time.sleep(0.8)
        st.rerun()
