import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import io

# =========================================================
# ⚙️ [설정] 환경 설정
# =========================================================
IS_SANDBOX = True # 실전시 False로 변경

st.set_page_config(layout="wide", page_title="비트겟 봇 (High Risk)")

# 세션 상태 초기화
if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = 100.0

# ---------------------------------------------------------
# 🔐 API 키 설정 (Secrets)
# ---------------------------------------------------------
try:
    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    api_password = st.secrets["API_PASSWORD"]
except:
    st.error("🚨 API 키가 설정되지 않았습니다. Streamlit Secrets를 설정해주세요.")
    st.stop()

# ---------------------------------------------------------
# 🛠️ 유틸리티 함수
# ---------------------------------------------------------
def safe_rerun():
    time.sleep(0.5)
    if hasattr(st, 'rerun'): st.rerun()
    else: st.experimental_rerun()

def safe_toast(msg):
    if hasattr(st, 'toast'): st.toast(msg)
    else: st.success(msg)

# 👇 [업그레이드] 텔레그램: 텍스트 + 차트 이미지 전송 기능
def send_telegram(token, chat_id, message, chart_df=None):
    try:
        if not token or not chat_id: return
        
        # 1. 텍스트 전송
        url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url_msg, data={'chat_id': chat_id, 'text': message})
        
        # 2. 차트 이미지 전송 (데이터가 있을 경우)
        if chart_df is not None:
            # 그래프 그리기
            plt.figure(figsize=(10, 5))
            plt.plot(chart_df['time'], chart_df['close'], label='Price', color='yellow')
            plt.plot(chart_df['time'], chart_df['MA20'], label='MA20', color='cyan', alpha=0.5)
            plt.title(f"Entry Chart Capture")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 스타일 설정 (어두운 배경)
            ax = plt.gca()
            ax.set_facecolor('black')
            plt.gcf().patch.set_facecolor('black')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # 이미지를 메모리에 저장
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='black')
            buf.seek(0)
            
            # 전송
            url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
            requests.post(url_photo, data={'chat_id': chat_id}, files={'photo': buf})
            plt.close() # 메모리 해제

    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

# ---------------------------------------------------------
# 🧮 보조지표 계산 함수
# ---------------------------------------------------------
def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 볼린저밴드 (20, 2)
    df['MA20'] = close.rolling(20).mean()
    df['STD'] = close.rolling(20).std()
    df['BB_UP'] = df['MA20'] + (df['STD'] * 2)
    df['BB_LO'] = df['MA20'] - (df['STD'] * 2)

    # 이평선 (MA)
    df['MA5'] = close.rolling(5).mean()
    df['MA50'] = close.rolling(50).mean()
    df['MA120'] = close.rolling(120).mean()
    
    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 스토캐스틱
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # 거래량 이평
    df['VOL_MA'] = df['vol'].rolling(20).mean()

    return df

# ---------------------------------------------------------
# 📡 거래소 연결
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
if not exchange: st.stop()

# ---------------------------------------------------------
# 🎨 사이드바 UI
# ---------------------------------------------------------
st.sidebar.title("🔥 야수의 심장 봇")
is_mobile = st.sidebar.checkbox("📱 모바일 모드", value=True)

markets = exchange.markets
futures_symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", futures_symbols, index=0)

st.sidebar.divider()
st.sidebar.subheader("⚔️ 공격적 세팅")
p_leverage = st.sidebar.slider("레버리지 (20배 추천)", 1, 50, 20)
min_vote = st.sidebar.slider("🎯 진입 조건 (몇 개 만족시 진입?)", 1, 5, 3, help="체크한 보조지표 중 이 숫자만큼 신호가 뜨면 진입합니다.")

st.sidebar.divider()
st.sidebar.subheader("🛡️ 리스크 관리")
use_sl_tp = st.sidebar.checkbox("자동 익절/손절 켜기", value=True)
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, 15.0, step=1.0)
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, 10.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("🔔 텔레그램")
tg_token = st.sidebar.text_input("봇 토큰", type="password")
tg_id = st.sidebar.text_input("챗 ID")

# ---------------------------------------------------------
# 📊 데이터 로딩
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol)
    curr_price = ticker['last']
    # 차트 전송을 위해 데이터를 좀 넉넉히 가져옴
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms') # 시간 변환
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    balance = exchange.fetch_balance({'type': 'swap'})
    margin_coin = 'SUSDT' if 'SBTC' in symbol else 'USDT'
    usdt_free = float(balance[margin_coin]['free']) if margin_coin in balance else 0.0
except Exception as e:
    st.error(f"데이터 로딩 에러: {e}")
    st.stop()

# ---------------------------------------------------------
# ⚡ 주문 & 알림 실행 함수
# ---------------------------------------------------------
def execute_trade(side, is_close=False, reason=""):
    try:
        if not is_close:
            exchange.set_leverage(p_leverage, symbol)
            
        qty = 0.0
        params = {}
        log_pnl = 0
        log_roi = 0
        
        if is_close:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if float(p['contracts']) > 0), None)
            if not pos: return
            qty = float(pos['contracts'])
            params = {'reduceOnly': True}
            order_side = 'sell' if pos['side'] == 'long' else 'buy'
            trade_emoji = "💰"
            log_pnl = float(pos['unrealizedPnl'])
            log_roi = float(pos['percentage'])
        else:
            input_val = st.session_state['order_usdt']
            raw_qty = (input_val * p_leverage) / curr_price
            qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            trade_emoji = "🚀"
            
        price = ticker['ask']*1.01 if order_side == 'buy' else ticker['bid']*0.99
        exchange.create_order(symbol, 'limit', order_side, qty, price, params=params)
        
        # 메시지 작성 (원화 환산 포함)
        action = "청산" if is_close else "진입"
        krw_val = curr_price * 1450 # 대략적인 환율
        msg = f"{trade_emoji} {side.upper()} {action} 체결!\n"
        msg += f"📍 이유: {reason}\n"
        msg += f"💲 가격: ${curr_price:,.2f} (약 {krw_val:,.0f}원)\n"
        msg += f"📊 레버리지: {p_leverage}배\n"
        
        if is_close:
            krw_pnl = log_pnl * 1450
            msg += f"📈 수익: ${log_pnl:.2f} ({krw_pnl:,.0f}원) | {log_roi:.2f}%"
            
        st.success(msg)
        safe_toast(msg)
        
        # 텔레그램 전송 (진입 시에만 차트 전송)
        if tg_token and tg_id: 
            send_chart = df.tail(50) if not is_close else None # 최근 50개 캔들
            send_telegram(tg_token, tg_id, msg, send_chart)
            
        safe_rerun()
        
    except Exception as e:
        st.error(f"주문 에러: {e}")

# =========================================================
# 📱 UI 구성
# =========================================================
def show_metrics():
    cols = st.columns(2) if is_mobile else st.columns(4)
    cols[0].metric("현재가", f"${curr_price:,.2f}")
    if is_mobile:
        cols[0].metric("잔고", f"${usdt_free:,.0f}")
    else:
        cols[1].metric("RSI", f"{last['RSI']:.1f}")
        cols[2].metric("잔고", f"${usdt_free:,.2f}")
        cols[3].metric("거래량", f"{last['vol']:.0f}")

def show_chart_and_position():
    # 차트
    tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
    studies_json = str(tv_studies).replace("'", '"')
    tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
    chart_h = 350 if is_mobile else 450
    
    components.html(f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
      "width": "100%", "height": {chart_h}, "symbol": "{tv_symbol}",
      "interval": "1", "timezone": "Asia/Seoul", "theme": "dark",
      "style": "1", "locale": "kr", "studies": {studies_json}, 
      "container_id": "tradingview_chart" }});
      </script>
    </div>
    """, height=chart_h)

    # 포지션
    st.subheader("💼 포지션 현황")
    active_position = None
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if float(p['contracts']) > 0:
                active_position = p
                break
                
        if active_position:
            side = active_position['side']
            roi = float(active_position['percentage'])
            pnl = float(active_position['unrealizedPnl'])
            entry = float(active_position['entryPrice'])
            lev = active_position['leverage']
            
            # 원화 환산
            krw_pnl = pnl * 1450
            
            color = "#4CAF50" if roi >= 0 else "#FF5252"
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h3 style="color: {color}; margin:0;">{side.upper()} x{lev}</h3>
                <p>평단가: ${entry:,.2f}</p>
                <p style="font-size: 1.2em; font-weight: bold;">
                   수익: ${pnl:.2f} ({krw_pnl:,.0f}원) | {roi:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if use_sl_tp:
                if roi >= tp_pct: execute_trade(side, is_close=True, reason="익절 달성")
                elif roi <= -sl_pct: execute_trade(side, is_close=True, reason="손절 방어")
                else:
                    limit_range = tp_pct + sl_pct
                    current_pos = roi + sl_pct
                    progress = min(max(current_pos / limit_range, 0.0), 1.0)
                    st.progress(progress)
        else:
            st.info("현재 포지션이 없습니다. (진입 대기중)")
    except: pass
    return active_position

def show_order_controls(active_pos):
    st.subheader("⚡ 주문 설정")
    c1, c2, c3, c4 = st.columns(4)
    def set_amt(pct): st.session_state['order_usdt'] = float(f"{usdt_free * pct:.2f}")
    if c1.button("25%"): set_amt(0.25)
    if c2.button("50%"): set_amt(0.5)
    if c3.button("75%"): set_amt(0.75)
    if c4.button("100%"): set_amt(1.0)
    
    st.number_input("주문 금액(USDT)", 0.0, usdt_free, key='order_usdt')

    b1, b2 = st.columns(2)
    if b1.button("📈 롱 진입", use_container_width=True): execute_trade('long', reason="수동")
    if b2.button("📉 숏 진입", use_container_width=True): execute_trade('short', reason="수동")
    
    if st.button("🚫 포지션 즉시 종료", use_container_width=True): 
        if active_pos: execute_trade(active_pos['side'], is_close=True, reason="수동 청산")

def show_bot_logic(active_pos):
    st.subheader("🧠 봇 전략 (투표 시스템)")
    
    # 👇 [업그레이드] 보조지표 상세 설명 및 선택
    with st.expander("🔻 보조지표 선택 (설명 포함)", expanded=True):
        st.write(f"현재 설정: 아래 지표 중 **{min_vote}개 이상** 만족 시 진입")
        
        use_rsi = st.checkbox("1. RSI 역추세", value=True, help="RSI 30이하(과매도)면 매수, 70이상(과매수)면 매도")
        use_bb = st.checkbox("2. 볼린저밴드 이탈", value=True, help="밴드 하단을 뚫고 내려가면 매수(반등 노림), 상단을 뚫으면 매도")
        use_ma_trend = st.checkbox("3. 20일/120일 이평선 지지/저항", value=True, help="가격이 20일선 위에 있으면 상승세(롱), 아래면 하락세(숏)")
        use_vol = st.checkbox("4. 거래량 폭발", value=False, help="평소보다 거래량이 2배 이상 터질 때만 진입 (가짜 움직임 방지)")
        use_macd = st.checkbox("5. MACD 골든크로스", value=False, help="MACD 선이 시그널 선을 돌파할 때 진입")

    # 신호 계산 (Signal Counting)
    votes_long = 0
    votes_short = 0
    
    # 1. RSI (역추세)
    if use_rsi:
        if last['RSI'] <= 30: votes_long += 1
        elif last['RSI'] >= 70: votes_short += 1
        
    # 2. BB (역추세: 밴드 찢고 들어올 때)
    if use_bb:
        if last['close'] <= last['BB_LO']: votes_long += 1
        elif last['close'] >= last['BB_UP']: votes_short += 1
        
    # 3. MA (추세/지지저항: 20일선 기준)
    if use_ma_trend:
        if last['close'] > last['MA20']: votes_long += 1 # 20일선 지지
        elif last['close'] < last['MA20']: votes_short += 1 # 20일선 저항
        
    # 4. 거래량 (필터)
    if use_vol:
        if last['vol'] > last['VOL_MA'] * 2.0: # 거래량 2배 터짐
            votes_long += 1
            votes_short += 1 # 방향 상관없이 거래량 터지면 가점
            
    # 5. MACD (추세)
    if use_macd:
        if last['MACD'] > last['MACD_Signal']: votes_long += 1
        elif last['MACD'] < last['MACD_Signal']: votes_short += 1

    # 최종 판단
    final_long = votes_long >= min_vote
    final_short = votes_short >= min_vote
    
    # UI 표시
    c1, c2 = st.columns(2)
    c1.metric("롱 신호 점수", f"{votes_long}/{min_vote}개")
    c2.metric("숏 신호 점수", f"{votes_short}/{min_vote}개")
    
    if final_long: st.success("🔥 롱 진입 조건 만족!")
    if final_short: st.error("🔥 숏 진입 조건 만족!")

    # 자동매매 실행
    st.divider()
    auto_on = st.checkbox("🤖 자동매매 활성화 (투표 조건 만족 시 진입)")
    if auto_on:
        if not active_pos:
            if final_long: execute_trade('long', reason=f"신호 {votes_long}개 만족")
            elif final_short: execute_trade('short', reason=f"신호 {votes_short}개 만족")
        else:
            # 포지션 있을 때 스위칭 (강력한 반대 신호가 뜨면)
            cur = active_pos['side']
            # 스위칭은 기준보다 +1점 더 높아야 실행 (잦은 매매 방지)
            if cur == 'long' and votes_short >= min_vote + 1: 
                execute_trade('long', is_close=True, reason="강력한 반대신호")
            elif cur == 'short' and votes_long >= min_vote + 1: 
                execute_trade('short', is_close=True, reason="강력한 반대신호")
        time.sleep(3)
        safe_rerun()

# =========================================================
# 🚀 메인 실행 로직
# =========================================================
st.title(f"🔥 {symbol}")

if is_mobile:
    show_metrics()
    tab1, tab2, tab3 = st.tabs(["📊 차트", "⚡ 주문", "🧠 전략"])
    with tab1: pos = show_chart_and_position()
    with tab2: show_order_controls(pos)
    with tab3: show_bot_logic(pos)
else:
    show_metrics()
    st.divider()
    pos = show_chart_and_position()
    st.divider()
    c1, c2 = st.columns([1,1])
    with c1: show_order_controls(pos)
    with c2: show_bot_logic(pos)
