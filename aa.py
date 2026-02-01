import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
import time
import requests
import json
import sqlite3
import os
from datetime import datetime
import google.generativeai as genai

# =========================================================
# ⚙️ [기본 설정 및 데이터베이스 초기화]
# =========================================================
st.set_page_config(layout="wide", page_title="AI Wonyousi Agent (Recursive Learning)")

# DB 파일 경로
DB_FILE = "trading_history.db"

# 데이터베이스 연결 및 테이블 생성 (일기장 만들기)
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # 매매 기록 테이블
    c.execute('''CREATE TABLE IF NOT EXISTS trade_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  symbol TEXT,
                  side TEXT,
                  entry_price REAL,
                  exit_price REAL,
                  leverage INTEGER,
                  pnl REAL,
                  reason TEXT,
                  ai_feedback TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 🔐 API 설정 (Secrets 우선, 없으면 입력)
# ---------------------------------------------------------
api_key = st.secrets.get("API_KEY", "")
api_secret = st.secrets.get("API_SECRET", "")
api_password = st.secrets.get("API_PASSWORD", "")
gemini_key = st.secrets.get("GEMINI_API_KEY", "")

# 사이드바 입력 (Secrets가 비어있을 경우)
with st.sidebar:
    st.title("🔧 설정 & 제어")
    if not api_key:
        api_key = st.text_input("Bitget Access Key", type="password")
        api_secret = st.text_input("Bitget Secret Key", type="password")
        api_password = st.text_input("Bitget Passphrase", type="password")
    if not gemini_key:
        gemini_key = st.text_input("Gemini API Key", type="password")

# AI 모델 초기화
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # 속도와 분석력이 좋은 1.5 Flash 사용

# 거래소 연결
def get_exchange():
    try:
        ex = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': api_password,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        return ex
    except: return None

exchange = get_exchange()

# =========================================================
# 🧠 [AI의 뇌] 분석, 판단, 회고 엔진
# =========================================================

def get_recent_feedback(limit=5):
    """과거 매매 기록(일기장)에서 교훈을 가져옵니다."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # 수익이 마이너스였던(실패한) 최근 기록을 가져옴
    c.execute("SELECT side, reason, ai_feedback FROM trade_log WHERE pnl < 0 ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    feedback_text = ""
    if rows:
        feedback_text = "⚠️ [너의 과거 실패 노트 - 이것만은 피하자]:\n"
        for row in rows:
            feedback_text += f"- {row[0]} 포지션 실패 사유: {row[1]} (반성: {row[2]})\n"
    else:
        feedback_text = "✨ 과거 실패 기록이 없습니다. 초심자의 행운을 빕니다!"
    return feedback_text

def analyze_market_ai(df, symbol):
    """AI가 차트와 과거 기록을 보고 매매 전략을 수립합니다."""
    if not gemini_key: return None

    # 1. 차트 데이터 요약 (워뇨띠 스타일: 패턴과 추세 중시)
    last_candles = df.tail(10).to_dict(orient='records')
    current_price = df.iloc[-1]['close']
    rsi = df.iloc[-1]['RSI']
    bb_width = df.iloc[-1]['BB_UP'] - df.iloc[-1]['BB_LO']
    
    # 2. 과거의 실수(피드백) 가져오기
    past_mistakes = get_recent_feedback()

    # 3. 프롬프트 작성 (워뇨띠 페르소나 + 회고적 학습)
    prompt = f"""
    당신은 전설적인 코인 트레이더 '워뇨띠'의 매매 철학을 가진 AI입니다.
    보조지표보다는 '캔들의 패턴', '거래량', '심리(Price Action)'를 중요시합니다.
    
    [현재 시장 데이터 - {symbol}]
    - 현재가: {current_price}
    - 최근 10개 캔들 데이터: {last_candles}
    - RSI: {rsi:.1f} (참고만 할 것)
    - 볼린저밴드 폭: {bb_width:.1f}
    
    {past_mistakes}
    
    위 데이터를 바탕으로 지금 당장 취해야 할 행동을 결정하세요.
    반드시 아래 JSON 형식으로만 답하세요. (설명 금지, JSON만 출력)
    
    {{
        "decision": "buy" 또는 "sell" 또는 "hold",
        "leverage": 1 ~ 20 사이의 정수 (확신이 들수록 높게),
        "entry_price": 시장가 진입 시 0, 지정가면 가격,
        "take_profit": 익절 가격,
        "stop_loss": 손절 가격,
        "reason": "왜 진입하는지, 캔들 패턴과 추세를 근거로 1줄 요약"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        st.error(f"AI 분석 실패: {e}")
        return None

def log_trade_result(symbol, side, entry, exit_price, leverage, pnl, reason):
    """매매가 끝나면 DB에 기록하고 AI에게 반성문을 쓰게 합니다."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 회고(Reflection) 생성
    reflection_prompt = f"""
    매매가 종료되었습니다. 결과를 분석해 주세요.
    - 포지션: {side}
    - 진입이유: {reason}
    - 수익금(PnL): {pnl} USDT
    
    이 매매가 성공적이었다면 무엇을 잘했는지, 실패했다면 무엇을 놓쳤는지 
    다음 매매를 위한 '한 줄 교훈'을 작성해 주세요.
    """
    try:
        feedback = model.generate_content(reflection_prompt).text
    except: feedback = "피드백 생성 실패"

    c.execute("INSERT INTO trade_log (timestamp, symbol, side, entry_price, exit_price, leverage, pnl, reason, ai_feedback) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, side, entry, exit_price, leverage, pnl, reason, feedback))
    conn.commit()
    conn.close()
    return feedback

# =========================================================
# 📊 차트 데이터 계산
# =========================================================
def fetch_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        # 보조지표 (참고용)
        df['RSI'] = 100 - (100 / (1 + df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / (-df['close'].diff().where(df['close'].diff() < 0, 0).rolling(14).mean())))
        df['BB_MA'] = df['close'].rolling(20).mean()
        df['BB_STD'] = df['close'].rolling(20).std()
        df['BB_UP'] = df['BB_MA'] + (df['BB_STD'] * 2)
        df['BB_LO'] = df['BB_MA'] - (df['BB_STD'] * 2)
        return df
    except: return pd.DataFrame()

# =========================================================
# 🖥️ 메인 UI
# =========================================================
st.header("🤖 AI Auto-Trader: Recursive Learning Edition")
st.caption("워뇨띠의 직관 + AI의 데이터 분석 + 과거 실패를 통한 자가 학습")

if not api_key or not gemini_key:
    st.warning("👈 사이드바에서 API Key를 먼저 입력해주세요.")
    st.stop()

# 코인 선택
markets = exchange.load_markets()
symbols = [m for m in markets if markets[m].get('swap')]
symbol = st.selectbox("거래 코인 선택", symbols, index=0)

# 데이터 로드
df = fetch_data(symbol)
if df.empty:
    st.error("데이터 로딩 실패")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    # 차트 출력
    st.line_chart(df.set_index('time')['close'])

with col2:
    st.subheader("현재 시장 데이터")
    last_row = df.iloc[-1]
    st.metric("현재가", f"${last_row['close']:.2f}")
    st.metric("RSI (14)", f"{last_row['RSI']:.1f}")
    
    # 수동 AI 분석 요청 버튼
    if st.button("🧠 AI 매매 전략 수립 요청"):
        with st.spinner("과거 일기장을 훑어보고 차트를 분석 중..."):
            strategy = analyze_market_ai(df, symbol)
            
            if strategy:
                st.success("분석 완료!")
                st.json(strategy)
                
                # 의사결정 시각화
                if strategy['decision'] == 'buy':
                    st.info(f"🔵 **LONG 진입 추천**\n\n이유: {strategy['reason']}\n레버리지: {strategy['leverage']}x")
                elif strategy['decision'] == 'sell':
                    st.info(f"🔴 **SHORT 진입 추천**\n\n이유: {strategy['reason']}\n레버리지: {strategy['leverage']}x")
                else:
                    st.warning(f"⚪ **관망 추천**\n\n이유: {strategy['reason']}")
                
                # 실제 주문 버튼 (안전을 위해 사용자 확인 후 실행)
                if strategy['decision'] != 'hold':
                    if st.button("🚀 위 전략대로 주문 실행"):
                        # 여기에 실제 주문 로직(create_order) 추가 가능
                        st.toast("주문이 전송되었습니다! (Demo)", icon="✅")

# =========================================================
# 📔 매매 일지 & 피드백 (회고 시스템)
# =========================================================
st.divider()
st.subheader("📜 AI의 매매 일지 & 반성문")

# 임의 데이터 추가 버튼 (테스트용)
if st.button("테스트용 가짜 데이터 입력 (DB Test)"):
    log_trade_result(symbol, "long", 50000, 49500, 10, -50.0, "상승 장악형 캔들 보고 진입")
    st.rerun()

# DB 데이터 조회
conn = sqlite3.connect(DB_FILE)
history_df = pd.read_sql("SELECT * FROM trade_log ORDER BY id DESC", conn)
conn.close()

if not history_df.empty:
    for index, row in history_df.iterrows():
        color = "green" if row['pnl'] > 0 else "red"
        with st.expander(f"[{row['timestamp']}] {row['side'].upper()} | PnL: {row['pnl']} USDT ({'성공' if row['pnl']>0 else '실패'})"):
            st.write(f"**진입 이유:** {row['reason']}")
            st.markdown(f"**AI의 회고:** :{color}[{row['ai_feedback']}]")
else:
    st.info("아직 매매 기록이 없습니다. AI가 첫 거래를 시작하면 기록이 쌓입니다.")

# =========================================================
# ⚙️ 자동매매 루프 (백그라운드 실행용 로직 예시)
# =========================================================
st.sidebar.divider()
st.sidebar.markdown("### 🤖 자동매매 상태")
auto_trade = st.sidebar.checkbox("자동매매 활성화 (Loop)")

if auto_trade:
    st.sidebar.write("🔄 시스템 가동 중...")
    placeholder = st.sidebar.empty()
    
    # 1분마다 실행된다고 가정 (실제로는 while loop 필요하지만 streamlit 특성상 새로고침으로 구현)
    if st.sidebar.button("강제 1회 실행"):
        strategy = analyze_market_ai(df, symbol)
        placeholder.json(strategy)
        # 여기서 실제 주문 로직이 들어갑니다.
        # 주문 후 포지션이 종료되면 log_trade_result()를 호출하여 DB에 기록합니다.
