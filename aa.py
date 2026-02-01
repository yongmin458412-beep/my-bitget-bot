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
import plotly.graph_objects as go # [New] 직관적인 차트용 라이브러리
from datetime import datetime
import google.generativeai as genai

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True  # ⚠️ 실전 시 False로 변경
SETTINGS_FILE = "bot_settings.json"
DB_FILE = "wonyousi_brain.db"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="AI Wonyousi: Autonomous Trader")

# ---------------------------------------------------------
# 🧠 [Brain] AI 기억 & 회고 시스템
# ---------------------------------------------------------
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
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT side, reason, ai_feedback FROM trade_history WHERE pnl < 0 ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        if not rows: return "과거에 큰 실수는 없습니다."
        feedback = "⛔ **[과거 실패 노트]**:\n"
        for row in rows:
            feedback += f"- {row[0]} 진입 실패 (이유: {row[1]}) → 반성: {row[2]}\n"
        return feedback
    except: return ""

def log_trade_to_db(symbol, side, price, pnl, reason, ai_feedback):
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO trade_history (timestamp, symbol, side, price, pnl, reason, ai_feedback) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, side, price, pnl, reason, ai_feedback))
        conn.commit()
        conn.close()
    except: pass

# ---------------------------------------------------------
# 💾 설정 로드
# ---------------------------------------------------------
def load_settings():
    default = {
        "gemini_api_key": "",
        "leverage": 20,
        "auto_trade": False, 
        "order_usdt": 100.0,
        # 지표 설정
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        # 사용 여부
        "use_rsi": True, "use_bb": True, "use_ma": True, "use_adx": True
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                default.update(saved)
        except: pass
    return default

def save_settings(new_settings):
    with open(SETTINGS_FILE, "w") as f: json.dump(new_settings, f)

config = load_settings()

# ---------------------------------------------------------
# 🔐 API & AI 모델 (자동 감지)
# ---------------------------------------------------------
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
gemini_key = st.secrets.get("GEMINI_API_KEY", config.get("gemini_api_key", ""))

if not api_key: st.error("🚨 API Key 설정 필요"); st.stop()

@st.cache_resource
def get_ai_model(key):
    if not key: return None
    genai.configure(api_key=key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = 'gemini-pro'
        for m in models:
            if 'flash' in m: target = m; break
        return genai.GenerativeModel(target)
    except: return genai.GenerativeModel('gemini-pro')

ai_model = get_ai_model(gemini_key)

# ---------------------------------------------------------
# 🧠 [Core] 워뇨띠 AI 분석 엔진 (상세 설명 강화)
# ---------------------------------------------------------
def generate_wonyousi_strategy(df, status_summary):
    if not ai_model: return {"decision": "hold", "reason": "API Key 없음", "confidence": 0}
    
    past_mistakes = get_past_mistakes()
    last_row = df.iloc[-1]
    
    prompt = f"""
    당신은 전설적인 트레이더 '워뇨띠'입니다. 
    지금부터 비트코인 차트를 분석하고 매매 결정을 내립니다.
    
    [현재 시장 데이터]
    - 가격: {last_row['close']}
    - RSI: {last_row['RSI']:.1f}
    - 볼린저밴드 위치: {status_summary.get('BB', '중간')}
    - 추세강도(ADX): {last_row['ADX']:.1f}
    
    [과거의 실패 기록 (반면교사)]
    {past_mistakes}
    
    위 데이터를 보고 다음 3가지 관점에서 상세히 분석하세요:
    1. 추세 (상승/하락/횡보)
    2. 거래량 및 캔들 패턴 (매집/분산/반전 신호)
    3. 진입 시나리오 (리스크 관리 포함)

    결과는 오직 JSON 형식으로만 출력하세요.
    {{
        "decision": "buy" 또는 "sell" 또는 "hold",
        "reason_trend": "추세 관점에서의 이유",
        "reason_candle": "캔들/거래량 관점에서의 이유",
        "final_reason": "종합적인 한 줄 결론",
        "confidence": 0~100 (확신도),
        "stop_loss": 손절가(숫자),
        "take_profit": 익절가(숫자)
    }}
    """
    try:
        res = ai_model.generate_content(prompt).text
        res = res.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except:
        return {"decision": "hold", "reason_trend": "분석 실패", "final_reason": "AI 오류", "confidence": 0}

# ---------------------------------------------------------
# 📡 거래소 & 데이터 처리
# ---------------------------------------------------------
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
    close = df['close']
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(config['rsi_period'])).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(config['rsi_period'])).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    
    # BB
    ma = close.rolling(int(config['bb_period'])).mean()
    std = close.rolling(int(config['bb_period'])).std()
    df['BB_UP'] = ma + (std * 2); df['BB_LO'] = ma - (std * 2)
    
    # ADX
    df['high_low'] = df['high'] - df['low']
    df['ADX'] = (df['high_low'].rolling(14).mean() / close) * 1000 # 약식 계산
    
    last = df.iloc[-1]
    status = {}
    if last['RSI'] <= 30: status['RSI'] = "과매도(L)"
    elif last['RSI'] >= 70: status['RSI'] = "과매수(S)"
    else: status['RSI'] = "중립"
    
    if last['close'] <= last['BB_LO']: status['BB'] = "하단 터치"
    elif last['close'] >= last['BB_UP']: status['BB'] = "상단 터치"
    else: status['BB'] = "밴드 내"
    
    return df, status, last

# ---------------------------------------------------------
# 🤖 [Auto] 완전 자동 매매 스레드 (즉시 진입)
# ---------------------------------------------------------
def telegram_thread(ex, symbol_name):
    ANALYSIS_INTERVAL = 900 # 15분
    last_run = 0
    
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                  data={'chat_id': tg_id, 'text': "🚀 **AI 완전 자율 매매 시작**\n경제뉴스 알림 OFF / 즉시 진입 ON"})

    while True:
        try:
            now = time.time()
            if now - last_run > ANALYSIS_INTERVAL:
                # 1. 데이터 수집
                ohlcv = ex.fetch_ohlcv(symbol_name, '5m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df, status, last = calc_indicators(df)
                
                # 2. AI 분석
                strategy = generate_wonyousi_strategy(df, status)
                decision = strategy['decision']
                conf = strategy.get('confidence', 0)
                
                # 3. 텔레그램 리포팅 (자세하게)
                emoji = "⚪"
                if decision == 'buy': emoji = "🔵"
                elif decision == 'sell': emoji = "🔴"
                
                msg = f"""
{emoji} **[15분 분석] {symbol_name}**
확신도: {conf}%

📊 **추세 분석:** {strategy.get('reason_trend', '-')}
🕯️ **캔들/패턴:** {strategy.get('reason_candle', '-')}
💡 **종합 판단:** {strategy.get('final_reason', '-')}
"""
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                              data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'Markdown'})
                
                # 4. [즉시 진입] 매매 실행 로직
                if decision in ['buy', 'sell']:
                    # 여기서 실제 주문 (시장가)
                    side = decision
                    price = last['close']
                    
                    # (실제 주문 코드 예시 - 안전 위해 try로 감쌈)
                    try:
                        ex.set_leverage(config['leverage'], symbol_name)
                        bal = ex.fetch_balance({'type': 'swap'})
                        free_usdt = float(bal['USDT']['free'])
                        amount = (free_usdt * 0.2) * config['leverage'] / price # 시드 20% 투입
                        qty = ex.amount_to_precision(symbol_name, amount)
                        
                        if float(qty) > 0:
                            # ex.create_market_order(symbol_name, side, qty) # ⚠️ 주석 해제 시 실제 주문
                            
                            # 알림 및 로그
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                          data={'chat_id': tg_id, 'text': f"⚡ **즉시 진입 완료!**\n{side.upper()} @ {price}"})
                            log_trade_to_db(symbol_name, side, price, 0, strategy['final_reason'], "진행 중")
                    except Exception as e:
                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                      data={'chat_id': tg_id, 'text': f"❌ 주문 실패: {e}"})
                
                last_run = now
            time.sleep(1)
        except: time.sleep(10)

# ---------------------------------------------------------
# 🎨 [UI] 메인 대시보드 (직관성 강화)
# ---------------------------------------------------------
markets = exchange.markets
symbol = "BTC/USDT:USDT" # 기본값

# 사이드바 설정
st.sidebar.header("🛠️ 설정")
if not gemini_key:
    k = st.sidebar.text_input("Gemini Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

# 스레드 시작
found = False
for t in threading.enumerate():
    if t.name == "AutoTrade": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="AutoTrade")
    t.start()

# 데이터 로딩
ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
df['time'] = pd.to_datetime(df['time'], unit='ms')
df, status, last = calc_indicators(df)

# === [UI 1] 상단 상태 배너 ===
st.title(f"🤖 {symbol} Autonomous Trader")
curr_price = last['close']
rsi_val = last['RSI']

# 상태에 따른 색상/메시지 결정
if rsi_val < 30: 
    banner_color = "green"
    banner_msg = "🟢 강력 매수 구간 (과매도)"
elif rsi_val > 70: 
    banner_color = "red"
    banner_msg = "🔴 강력 매도 구간 (과매수)"
else: 
    banner_color = "gray"
    banner_msg = "⚪ 관망 구간 (중립)"

st.markdown(f"""
<div style="padding: 20px; background-color: #1e1e1e; border-radius: 10px; border-left: 10px solid {banner_color}; margin-bottom: 20px;">
    <h2 style="margin:0; color: white;">{banner_msg}</h2>
    <p style="margin:0; color: #aaaaaa;">현재가: <b>${curr_price:,.2f}</b> | AI 모드: 완전 자율 주행</p>
</div>
""", unsafe_allow_html=True)

# === [UI 2] 직관적인 게이지 차트 (Plotly) ===
c1, c2, c3 = st.columns(3)

with c1:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = rsi_val,
        title = {'text': "RSI (강도)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': banner_color},
                 'steps': [
                     {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                     {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}],
                 'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': rsi_val}}))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # 추세 강도(ADX) 게이지
    adx_val = last['ADX']
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = adx_val,
        title = {'text': "ADX (추세 힘)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "orange" if adx_val > 25 else "gray"},
                 'steps': [{'range': [0, 25], 'color': "rgba(255, 255, 255, 0.1)"}]}))
    fig2.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig2, use_container_width=True)

with c3:
    # 캔들 차트 (간소화)
    st.markdown("#### 📊 최근 차트 흐름")
    st.line_chart(df.set_index('time')['close'].tail(50), height=200)

# === [UI 3] AI 상세 분석 리포트 ===
st.divider()
col_ai, col_log = st.columns([2, 1])

with col_ai:
    st.subheader("🧠 AI 실시간 분석 리포트")
    if st.button("🔍 지금 바로 분석 요청 (수동)"):
        with st.spinner("AI가 차트를 뜯어보는 중..."):
            ai_res = generate_wonyousi_strategy(df, status)
            
            # 카드로 결과 표시
            st.markdown(f"""
            <div style="background-color: #262730; padding: 20px; border-radius: 10px;">
                <h3>결론: <span style="color: {'#00ff00' if ai_res['decision']=='buy' else '#ff0000'};">{ai_res['decision'].upper()}</span> (확신도 {ai_res.get('confidence')}% )</h3>
                <hr>
                <p><b>📈 추세 관점:</b> {ai_res.get('reason_trend')}</p>
                <p><b>🕯️ 캔들/패턴:</b> {ai_res.get('reason_candle')}</p>
                <p><b>💡 최종 판단:</b> {ai_res.get('final_reason')}</p>
                <hr>
                <small>추천 손절가: {ai_res.get('stop_loss')} | 익절가: {ai_res.get('take_profit')}</small>
            </div>
            """, unsafe_allow_html=True)

with col_log:
    st.subheader("📜 매매 기록 (DB)")
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    history = pd.read_sql("SELECT symbol, side, pnl, reason FROM trade_history ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    
    if not history.empty:
        st.dataframe(history, hide_index=True)
    else:
        st.info("아직 매매 기록이 없습니다.")
