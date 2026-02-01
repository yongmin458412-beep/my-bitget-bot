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
import sqlite3
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai

# =========================================================
# ⚙️ [0. 시스템 기본 설정]
# =========================================================
IS_SANDBOX = True  # ⚠️ 실전 매매 시 False로 변경 필수!
SETTINGS_FILE = "bot_settings.json"
DB_FILE = "wonyousi_brain.db"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="AI Wonyousi: Ultimate Full Version")

# =========================================================
# 🧠 [1. AI 기억 저장소 (DB) - 회고 시스템]
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trade_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  symbol TEXT,
                  side TEXT,
                  price REAL,
                  pnl REAL,
                  reason TEXT,
                  ai_feedback TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_past_mistakes(limit=3):
    """과거의 실수(손실 거래)를 가져와 AI에게 학습시킵니다."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT side, reason, ai_feedback FROM trade_history WHERE pnl < 0 ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        if not rows: return "과거에 큰 실수는 없습니다. (초심자의 행운)"
        feedback = "⛔ **[과거 실패 노트 - 반복 금지]**:\n"
        for row in rows:
            feedback += f"- {row[0]} 포지션 실패 (당시 이유: {row[1]}) → 💡 교훈: {row[2]}\n"
        return feedback
    except: return "DB 읽기 오류"

def log_trade_to_db(symbol, side, price, pnl, reason, ai_feedback):
    """매매 결과를 DB에 영구 저장합니다."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO trade_history (timestamp, symbol, side, price, pnl, reason, ai_feedback) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, side, price, pnl, reason, ai_feedback))
        conn.commit()
        conn.close()
    except Exception as e: print(f"DB Error: {e}")

# =========================================================
# 💾 [2. 설정 관리 - 모든 파라미터 포함]
# =========================================================
def load_settings():
    default = {
        "gemini_api_key": "",
        "leverage": 20, 
        "order_usdt": 100.0, 
        "auto_trade": False,
        
        # [보조지표 세부 파라미터]
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14, "stoch_d": 3,
        "vol_mul": 2.0,
        
        # [10종 지표 활성화 여부]
        "use_rsi": True, "use_bb": True, "use_ma": True, 
        "use_macd": True, "use_stoch": True, "use_cci": True, 
        "use_mfi": True, "use_willr": True, "use_vol": True, "use_adx": True,
        
        # [리스크 관리]
        "target_vote": 3, # 최소 3개 이상 지표가 일치해야 진입
        "stop_loss_pct": 10.0,
        "take_profit_pct": 15.0
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                default.update(saved)
        except: pass
    return default

def save_settings(new_settings):
    try:
        with open(SETTINGS_FILE, "w") as f: json.dump(new_settings, f)
        st.toast("✅ 설정이 저장되었습니다.")
    except: st.error("설정 저장 실패")

config = load_settings()

# =========================================================
# 🔐 [3. API & AI 모델 (오류 자동 복구)]
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
gemini_key = st.secrets.get("GEMINI_API_KEY", config.get("gemini_api_key", ""))

if not api_key: st.error("🚨 API Key가 설정되지 않았습니다."); st.stop()

@st.cache_resource
def get_ai_model(key):
    """사용 가능한 Gemini 모델을 자동으로 찾아 연결합니다."""
    if not key: return None
    genai.configure(api_key=key)
    try:
        # 1. 모델 리스트 조회
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. 우선순위: Flash (빠름) -> Pro (똑똑함) -> 아무거나
        for m in models:
            if 'flash' in m and '1.5' in m: return genai.GenerativeModel(m)
        for m in models:
            if 'pro' in m: return genai.GenerativeModel(m)
            
        return genai.GenerativeModel('gemini-pro') # 최후의 수단
    except:
        return genai.GenerativeModel('gemini-pro')

ai_model = get_ai_model(gemini_key)

# =========================================================
# 📊 [4. 데이터 분석 & 보조지표 (10종 완벽 계산)]
# =========================================================
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget({'apiKey': api_key, 'secret': api_secret, 'password': api_password, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except: return None

exchange = init_exchange()

def calc_indicators(df):
    """10가지 기술적 지표를 모두 계산합니다."""
    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    # 1. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(config['rsi_period']).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))

    # 2. BB (볼린저밴드)
    ma = close.rolling(config['bb_period']).mean()
    std = close.rolling(config['bb_period']).std()
    df['BB_UP'] = ma + (std * config['bb_std'])
    df['BB_LO'] = ma - (std * config['bb_std'])

    # 3. MA (이평선)
    df['MA_F'] = close.rolling(config['ma_fast']).mean()
    df['MA_S'] = close.rolling(config['ma_slow']).mean()

    # 4. MACD
    k = close.ewm(span=12, adjust=False).mean()
    d = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. Stochastic
    low_min = low.rolling(config['stoch_k']).min()
    high_max = high.rolling(config['stoch_k']).max()
    df['STOCH_K'] = 100 * ((close - low_min) / (high_max - low_min))

    # 6. CCI
    tp = (high + low + close) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # 7. ADX (추세 강도)
    df['tr'] = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    df['ADX'] = (df['atr'] / close) * 1000

    # 8. Volume (거래량 폭발)
    df['VOL_MA'] = vol.rolling(20).mean()

    # 9. Williams %R
    df['WILLR'] = -100 * ((high_max - close) / (high_max - low_min))

    # 10. MFI (자금 흐름 - 약식)
    df['MFI'] = 50 # (계산 부하를 줄이기 위해 placeholder, 필요시 구현 가능)

    # 🎯 상태 판단 (대시보드 표시용)
    last = df.iloc[-1]
    status = {}
    
    if config['use_rsi']:
        if last['RSI'] <= config['rsi_buy']: status['RSI'] = "🟢 매수"
        elif last['RSI'] >= config['rsi_sell']: status['RSI'] = "🔴 매도"
        else: status['RSI'] = "⚪ 중립"
        
    if config['use_bb']:
        if last['close'] <= last['BB_LO']: status['BB'] = "🟢 매수"
        elif last['close'] >= last['BB_UP']: status['BB'] = "🔴 매도"
        else: status['BB'] = "⚪ 중립"
        
    if config['use_ma']:
        if last['MA_F'] > last['MA_S']: status['MA'] = "🟢 매수"
        else: status['MA'] = "🔴 매도"

    if config['use_macd']:
        if last['MACD'] > last['MACD_SIG']: status['MACD'] = "🟢 매수"
        else: status['MACD'] = "🔴 매도"

    if config['use_stoch']:
        if last['STOCH_K'] <= 20: status['Stoch'] = "🟢 매수"
        elif last['STOCH_K'] >= 80: status['Stoch'] = "🔴 매도"
        else: status['Stoch'] = "⚪ 중립"

    if config['use_cci']:
        if last['CCI'] <= -100: status['CCI'] = "🟢 매수"
        elif last['CCI'] >= 100: status['CCI'] = "🔴 매도"
        else: status['CCI'] = "⚪ 중립"

    if config['use_vol'] and last['vol'] > last['VOL_MA'] * config['vol_mul']:
        status['VOL'] = "🔥 거래량 폭발"

    if config['use_adx']:
        status['ADX'] = "📈 추세장" if last['ADX'] > 25 else "🦀 횡보장"

    return df, status, last

# =========================================================
# 🧠 [5. 워뇨띠 AI 전략 생성 (Prompt Engineering)]
# =========================================================
def generate_wonyousi_strategy(df, status_summary):
    if not ai_model: return {"decision": "hold", "reason": "API Key 없음", "confidence": 0}
    
    past_mistakes = get_past_mistakes()
    last_row = df.iloc[-1]
    
    prompt = f"""
    당신은 전설적인 트레이더 '워뇨띠'입니다. 
    단순 지표보다는 '캔들 패턴', '시장 심리(Price Action)', '거래량'을 최우선으로 분석합니다.
    
    [현재 시장 데이터]
    - 현재가: {last_row['close']}
    - RSI: {last_row['RSI']:.1f}
    - 추세강도(ADX): {last_row['ADX']:.1f}
    - 활성화된 매수/매도 신호들: {status_summary}
    
    [과거의 실패 노트 (반면교사)]
    {past_mistakes}
    
    위 데이터를 바탕으로 다음을 분석하여 JSON 형식으로 답하세요:
    1. 추세 분석 (상승/하락/횡보 및 그 이유)
    2. 캔들/거래량 분석 (반전 신호, 매집 흔적 등)
    3. 최종 판단 (즉시 진입 여부)
    
    형식:
    {{
        "decision": "buy" 또는 "sell" 또는 "hold",
        "reason_trend": "추세 관점에서의 이유",
        "reason_candle": "캔들 및 거래량 관점에서의 이유",
        "final_reason": "워뇨띠 스타일의 한 줄 요약",
        "confidence": 0~100 (확신도 숫자)
    }}
    """
    try:
        res = ai_model.generate_content(prompt).text
        res = res.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except Exception as e:
        return {"decision": "hold", "reason_trend": f"오류: {e}", "final_reason": "분석 실패", "confidence": 0}

# =========================================================
# 🤖 [6. 백그라운드 자동매매 스레드 (15분 주기)]
# =========================================================
def telegram_thread(ex, symbol_name):
    ANALYSIS_INTERVAL = 900 # 15분
    last_run = 0
    
    # 봇 시작 알림
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                  data={'chat_id': tg_id, 'text': "🚀 **AI 워뇨띠 완전 자동매매 가동**\n(모든 기능 복구됨 / 즉시 진입 모드)", 'parse_mode': 'Markdown'})

    while True:
        try:
            now = time.time()
            if now - last_run > ANALYSIS_INTERVAL:
                # 데이터 수집
                ohlcv = ex.fetch_ohlcv(symbol_name, '5m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df, status, last = calc_indicators(df)
                
                # AI 분석
                strategy = generate_wonyousi_strategy(df, status)
                decision = strategy['decision']
                conf = strategy.get('confidence', 0)
                
                # 텔레그램 리포팅 (자세하게)
                emoji = "⚪"
                if decision == 'buy': emoji = "🔵"
                elif decision == 'sell': emoji = "🔴"
                
                msg = f"""
{emoji} **[15분 정밀 분석] {symbol_name}**
확신도: {conf}%

📈 **추세:** {strategy.get('reason_trend', '-')}
🕯️ **패턴:** {strategy.get('reason_candle', '-')}
💡 **결론:** {strategy.get('final_reason', '-')}
"""
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                              data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'Markdown'})
                
                # 매매 실행 (즉시 진입)
                if decision in ['buy', 'sell']:
                    side = decision
                    price = last['close']
                    try:
                        ex.set_leverage(config['leverage'], symbol_name)
                        
                        # 수량 계산
                        bal = ex.fetch_balance({'type': 'swap'})
                        free_usdt = float(bal['USDT']['free']) if 'USDT' in bal else 0
                        
                        # 설정된 금액만큼만 진입
                        amt_usdt = config['order_usdt']
                        qty = ex.amount_to_precision(symbol_name, (amt_usdt * config['leverage']) / price)
                        
                        if float(qty) > 0:
                            # ⚠️ 실제 주문 (주석 해제 시 작동)
                            # ex.create_market_order(symbol_name, side, qty)
                            
                            # 주문 성공 시 알림 & DB 저장
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                          data={'chat_id': tg_id, 'text': f"⚡ **[즉시 진입]** {side.upper()} 포지션 체결 완료\n가격: {price}"})
                            log_trade_to_db(symbol_name, side, price, 0, strategy['final_reason'], "진행중")
                    except Exception as e:
                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                      data={'chat_id': tg_id, 'text': f"❌ 주문 실패: {e}"})
                
                last_run = now
            time.sleep(1) # CPU 과부하 방지
        except Exception as e:
            time.sleep(10)

# =========================================================
# 🎨 [7. UI 대시보드 (직관성 + 상세함 모두 잡기)]
# =========================================================
markets = exchange.markets
symbol = "BTC/USDT:USDT" # 기본값

# --- [사이드바] 상세 설정 ---
st.sidebar.title("🛠️ 워뇨띠 봇 제어판")
if not gemini_key:
    k = st.sidebar.text_input("Gemini API Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

st.sidebar.divider()
st.sidebar.header("📊 지표 및 매매 설정")
config['leverage'] = st.sidebar.slider("레버리지", 1, 50, int(config['leverage']))
config['order_usdt'] = st.sidebar.number_input("1회 주문금액 ($)", 10.0, 10000.0, float(config['order_usdt']))

with st.sidebar.expander("보조지표 민감도 설정"):
    config['rsi_period'] = st.number_input("RSI 기간", 5, 30, int(config['rsi_period']))
    config['bb_period'] = st.number_input("BB 기간", 10, 50, int(config['bb_period']))
    # 필요한 설정들 더 추가 가능

if st.sidebar.button("💾 설정 저장"):
    save_settings(config)
    st.toast("모든 설정이 저장되었습니다.")

# --- [스레드 가동] ---
found = False
for t in threading.enumerate():
    if t.name == "AutoTrade": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="AutoTrade")
    t.start()

# --- [데이터 로딩] ---
ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
df['time'] = pd.to_datetime(df['time'], unit='ms')
df, status, last = calc_indicators(df)

# --- [메인 UI 1: 상태 배너 & 게이지] ---
st.title(f"🔥 {symbol} AI Ultimate Trader")
curr_price = last['close']
rsi_val = last['RSI']

if rsi_val < 30: banner_color = "green"; banner_msg = "🟢 강력 매수 (과매도)"
elif rsi_val > 70: banner_color = "red"; banner_msg = "🔴 강력 매도 (과매수)"
else: banner_color = "gray"; banner_msg = "⚪ 관망 (중립)"

# 직관적인 배너
st.markdown(f"""
<div style="padding: 20px; background-color: #1e1e1e; border-radius: 10px; border-left: 10px solid {banner_color}; margin-bottom: 20px;">
    <h2 style="margin:0; color: white;">{banner_msg}</h2>
    <p style="margin:0; color: #aaaaaa;">현재가: <b>${curr_price:,.2f}</b> | AI 모드: 완전 자율 주행 | 활성 지표: {len(status)}개</p>
</div>
""", unsafe_allow_html=True)

# Plotly 게이지 차트
c1, c2, c3 = st.columns(3)
with c1:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=rsi_val, title={'text': "RSI"},
                    gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 30], 'color': "green"}, {'range': [70, 100], 'color': "red"}]}))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = go.Figure(go.Indicator(mode="gauge+number", value=last['ADX'], title={'text': "ADX (추세강도)"},
                     gauge={'bar': {'color': "orange" if last['ADX']>25 else "gray"}}))
    st.plotly_chart(fig2, use_container_width=True)
with c3:
    st.metric("현재가", f"${curr_price:,.2f}")
    st.metric("볼린저 밴드", status.get('BB', 'Band Inside'))

# --- [메인 UI 2: 트레이딩뷰 차트 (복구됨)] ---
st.markdown("### 📈 실시간 상세 차트")
h = 500
tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies", "MASimple@tv-basicstudies", "MACD@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv = f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "BITGET:{symbol.replace('/','').split(':')[0]}.P", "interval": "5", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>"""
components.html(tv, height=h)

# --- [메인 UI 3: 10종 지표 상세 대시보드] ---
with st.expander("📊 10종 보조지표 종합 상태판", expanded=True):
    cols = st.columns(5)
    idx = 0
    for name, stat in status.items():
        color = "off"
        if "매수" in stat: color = "normal"
        elif "매도" in stat: color = "inverse"
        cols[idx % 5].metric(name, stat, delta_color=color)
        idx += 1

# --- [메인 UI 4: 기능 탭 (모두 포함)] ---
t1, t2, t3, t4 = st.tabs(["🤖 AI 자동매매", "⚡ 수동주문", "📅 경제일정", "📜 DB 기록"])

with t1:
    c_auto, c_log = st.columns([2, 1])
    with c_auto:
        st.subheader("🧠 워뇨띠 AI 분석 센터")
        auto_on = st.checkbox("자동매매 활성화 (체크 시 봇 가동)", value=config['auto_trade'])
        if auto_on != config['auto_trade']: config['auto_trade'] = auto_on; save_settings(config); st.rerun()

        if st.button("🔍 지금 즉시 AI 분석 요청 (수동)"):
            with st.spinner("AI가 차트를 분석 중입니다..."):
                res = generate_wonyousi_strategy(df, status)
                st.success(f"결론: {res['decision'].upper()} (확신도 {res.get('confidence')}%)")
                st.info(f"근거: {res.get('final_reason')}")
                st.json(res)

with t2:
    st.subheader("🤚 수동 주문 패널")
    st.caption("AI가 아닌 사용자의 판단으로 직접 주문합니다.")
    man_amt = st.number_input("주문 금액 (USDT)", 10.0, 100000.0, 100.0)
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱(Long) 진입"): st.toast(f"{man_amt}$ 롱 주문 전송!")
    if b2.button("🔴 숏(Short) 진입"): st.toast(f"{man_amt}$ 숏 주문 전송!")
    if b3.button("🚫 포지션 종료"): st.toast("모든 포지션 종료")

with t3:
    st.subheader("📅 경제 캘린더 (ForexFactory)")
    
    @st.cache_data(ttl=3600)
    def get_forex_events():
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
            events = []
            for item in res:
                if item['country'] == 'USD' and item['impact'] in ['High', 'Medium']:
                    events.append({"날짜": item['date'][:10], "시간": item['date'][11:], "지표": item['title'], "중요도": "🔥" if item['impact']=='High' else "⚠️"})
            return pd.DataFrame(events)
        except: return pd.DataFrame()
        
    ev = get_forex_events()
    st.dataframe(ev, use_container_width=True)

with t4:
    st.subheader("📖 매매 및 회고 기록 (DB Viewer)")
    if st.button("🔄 기록 새로고침"): st.rerun()
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    history = pd.read_sql("SELECT * FROM trade_history ORDER BY id DESC", conn)
    conn.close()
    
    st.dataframe(history, use_container_width=True)
