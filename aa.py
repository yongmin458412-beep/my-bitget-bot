import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

# =========================================================
# ⚙️ [설정] 환경 설정
# =========================================================
IS_SANDBOX = True 

st.set_page_config(layout="wide", page_title="비트겟 봇 (Mobile Ver)")

if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = 10.0

# ---------------------------------------------------------
# 🔐 API 키 설정 (Secrets 사용)
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

def send_telegram(token, chat_id, message):
    try:
        if token and chat_id:
            if len(chat_id) > 9 and not chat_id.startswith("-"): pass 
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': message}
            requests.post(url, data=data)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

# ---------------------------------------------------------
# 🧮 보조지표 계산
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

    # 볼린저밴드
    df['MA20'] = close.rolling(20).mean()
    df['STD'] = close.rolling(20).std()
    df['BB_UP'] = df['MA20'] + (df['STD'] * 2)
    df['BB_LO'] = df['MA20'] - (df['STD'] * 2)

    # MA50
    df['MA50'] = close.rolling(50).mean()

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 스토캐스틱
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # CCI
    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean()
    def get_mad(x): return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(20).apply(get_mad)
    df['CCI'] = (tp - sma) / (0.015 * mad)

    # Williams %R
    df['WILLR'] = -100 * ((highest_high - close) / (highest_high - lowest_low))

    # Volume MA
    df['VOL_MA'] = df['vol'].rolling(20).mean()

    # ADX
    df['TR'] = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ADX'] = (df['TR'] / close).rolling(14).mean() * 100 

    return df

# ---------------------------------------------------------
# 거래소 연결
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
# 사이드바 설정
# ---------------------------------------------------------
st.sidebar.title("🛠️ 봇 설정")

# 👇 [핵심] 모바일 모드 스위치
is_mobile = st.sidebar.checkbox("📱 모바일 모드 (탭 보기)", value=True)

markets = exchange.markets
futures_symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", futures_symbols, index=0)
MARGIN_COIN = 'SUSDT' if 'SBTC' in symbol else 'USDT'

st.sidebar.divider()
st.sidebar.subheader("🎛️ 민감도 설정")
p_rsi_buy = st.sidebar.slider("RSI 매수 (이하)", 10, 40, 30)
p_rsi_sell = st.sidebar.slider("RSI 매도 (이상)", 60, 90, 70)
p_leverage = st.sidebar.slider("레버리지", 1, 125, 10)

st.sidebar.divider()
st.sidebar.subheader("🛡️ 리스크 관리")
use_sl_tp = st.sidebar.checkbox("익절/손절 자동 청산 켜기", value=True)
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, 10.0, step=0.5)
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, 5.0, step=0.5)

st.sidebar.divider()
st.sidebar.subheader("🔔 텔레그램 알림")
tg_token = st.sidebar.text_input("봇 토큰 (Token)", type="password")
tg_id = st.sidebar.text_input("챗 ID (Chat ID)")

# ---------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol)
    curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    balance = exchange.fetch_balance({'type': 'swap'})
    usdt_free = float(balance[MARGIN_COIN]['free']) if MARGIN_COIN in balance else 0.0

except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

# ---------------------------------------------------------
# 주문 함수
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
        
        action = "청산" if is_close else "진입"
        msg = f"{trade_emoji} {side.upper()} {action} 성공!\n이유: {reason}\n가격: ${curr_price:,.2f}"
        if is_close: msg += f"\n수익: ${log_pnl:.2f} ({log_roi:.2f}%)"
            
        st.success(msg)
        safe_toast(msg)
        if tg_token and tg_id: send_telegram(tg_token, tg_id, msg)
        safe_rerun()
        
    except Exception as e:
        st.error(f"주문 에러: {e}")

# ---------------------------------------------------------
# 📱 모바일 vs 데스크탑 레이아웃 분기
# ---------------------------------------------------------
st.title(f"🤖 {symbol}")

# 모바일 모드일 때는 탭을 생성하고, 아니면 그냥 변수(None) 처리
tab1, tab2, tab3 = None, None, None
if is_mobile:
    tab1, tab2, tab3 = st.tabs(["📊 차트/현황", "⚡ 주문/설정", "📝 봇 상태"])
# =========================================================
# 1. 상단 메트릭 (공통)
# =========================================================
def show_metrics():
    # 모바일이면 2열, 데스크탑이면 4열
    cols = st.columns(2) if is_mobile else st.columns(4)
    cols[0].metric("현재가", f"${curr_price:,.2f}")
    
    # RSI 표시 (모바일/데스크탑 위치 다름)
    if is_mobile:
        cols[0].metric("RSI", f"{last['RSI']:.1f}")
        cols[1].metric("잔고", f"${usdt_free:,.0f}")
        cols[1].metric("볼륨", f"{last['vol']:.0f}")
    else:
        cols[1].metric("RSI", f"{last['RSI']:.1f}")
        cols[2].metric("잔고", f"${usdt_free:,.2f}")
        cols[3].metric("거래량", f"{last['vol']:.0f}")

# ... (중간 생략) ...

# =========================================================
# 🚀 화면 그리기 (모바일 vs 데스크탑)
# =========================================================

if is_mobile:
    # 📱 모바일 뷰
    show_metrics() # 👈 여기가 st.help() 없이 깨끗해야 합니다!
    
    with tab1: # 차트 & 포지션
        pos = show_chart_and_position()
        
    with tab2: # 주문
        show_order_controls(pos)
        
    with tab3: # 봇 상태
        show_bot_logic(pos)
        
else:
    # 🖥️ 데스크탑 뷰
    show_metrics() # 👈 여기도 확인!
    st.divider()
    pos = show_chart_and_position()
    st.divider()
    c_left, c_right = st.columns([1, 1])
    with c_left: show_order_controls(pos)
    with c_right: show_bot_logic(pos)
# =========================================================
# 2. 차트 및 포지션 (Tab 1 or Main)
# =========================================================
def show_chart_and_position():
    # 차트
    tv_studies = []
    if is_mobile: 
        # 모바일에선 지표 너무 많으면 지저분하니 핵심만
        tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
    else:
        # 데스크탑은 다 보여줌
        tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies", "MACD@tv-basicstudies"]
        
    studies_json = str(tv_studies).replace("'", '"')
    tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
    
    # 모바일은 차트 높이 줄임 (450 -> 350)
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

    # 포지션 현황
    st.subheader("💼 포지션")
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
            
            color = "#4CAF50" if roi >= 0 else "#FF5252"
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h3 style="color: {color}; margin:0;">{side.upper()} 보유중</h3>
                <div style="display: flex; justify-content: space-between;">
                    <span>진입: ${entry:,.2f}</span>
                    <span style="font-weight: bold;">{roi:.2f}% (${pnl:.1f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 리스크 관리 로직 실행
            if use_sl_tp:
                if roi >= tp_pct: execute_trade(side, is_close=True, reason="익절")
                elif roi <= -sl_pct: execute_trade(side, is_close=True, reason="손절")
                else:
                    limit_range = tp_pct + sl_pct
                    current_pos = roi + sl_pct
                    progress = min(max(current_pos / limit_range, 0.0), 1.0)
                    st.progress(progress)
        else:
            st.info("포지션 없음 (대기중)")
    except: pass
    return active_position

# =========================================================
# 3. 주문 및 설정 (Tab 2 or Main)
# =========================================================
def show_order_controls(active_pos):
    st.subheader("⚡ 주문")
    
    # 퍼센트 버튼
    c1, c2, c3, c4 = st.columns(4)
    def set_amt(pct): st.session_state['order_usdt'] = float(f"{usdt_free * pct:.2f}")
    if c1.button("10%"): set_amt(0.1)
    if c2.button("25%"): set_amt(0.25)
    if c3.button("50%"): set_amt(0.5)
    if c4.button("100%"): set_amt(1.0)
    
    st.number_input("금액(USDT)", 0.0, usdt_free, key='order_usdt')

    b1, b2 = st.columns(2)
    if b1.button("📈 롱", use_container_width=True): execute_trade('long')
    if b2.button("📉 숏", use_container_width=True): execute_trade('short')
    
    if st.button("🚫 포지션 청산", use_container_width=True): 
        if active_pos: execute_trade(active_pos['side'], is_close=True, reason="수동")

# =========================================================
# 4. 봇 상태 및 로직 (Tab 3 or Main)
# =========================================================
def show_bot_logic(active_pos):
    st.subheader("🧠 봇 상태")
    
    # 지표 체크박스 (모바일에선 여기서 설정)
    if is_mobile:
        st.caption("✅ 적용할 지표 선택")
        use_rsi = st.checkbox("RSI", value=True)
        use_bb = st.checkbox("볼린저밴드", value=True)
        use_ma = st.checkbox("MA50 추세")
        use_macd = st.checkbox("MACD")
    else:
        # 데스크탑은 아래에 별도 섹션
        use_rsi = True; use_bb = True; use_ma = False; use_macd = False 
        
    signals_long = []
    signals_short = []
    
    if use_rsi:
        if last['RSI'] <= p_rsi_buy: signals_long.append(True)
        elif last['RSI'] >= p_rsi_sell: signals_short.append(True)
        else: signals_long.append(False); signals_short.append(False)

    if use_bb:
        if last['close'] <= last['BB_LO']: signals_long.append(True)
        elif last['close'] >= last['BB_UP']: signals_short.append(True)
        else: signals_long.append(False); signals_short.append(False)
        
    # (나머지 지표 생략 - 모바일 최적화를 위해 핵심만)
    
    final_long = all(signals_long) and (len(signals_long)>0)
    final_short = all(signals_short) and (len(signals_short)>0)
    
    c1, c2 = st.columns(2)
    if final_long: c1.success("🔥 롱 조건 만족")
    else: c1.warning("롱 대기")
    
    if final_short: c2.error("🔥 숏 조건 만족")
    else: c2.warning("숏 대기")

    # 자동매매 실행
    auto_on = st.checkbox("🤖 자동매매 활성화")
    if auto_on:
        if not active_pos:
            if final_long: execute_trade('long', reason="자동")
            elif final_short: execute_trade('short', reason="자동")
        else:
            # 스위칭 로직
            cur = active_pos['side']
            if cur == 'long' and final_short: execute_trade('long', is_close=True, reason="스위칭")
            elif cur == 'short' and final_long: execute_trade('short', is_close=True, reason="스위칭")
        
        time.sleep(3)
        safe_rerun()

# =========================================================
# 🚀 화면 그리기 (모바일 vs 데스크탑)
# =========================================================

if is_mobile:
    # 📱 모바일 뷰
    show_metrics() # 상단 공통
    
    with tab1: # 차트 & 포지션
        pos = show_chart_and_position()
        
    with tab2: # 주문
        show_order_controls(pos)
        
    with tab3: # 봇 상태
        show_bot_logic(pos)
        
else:
    # 🖥️ 데스크탑 뷰 (기존 방식대로 길게 나열)
    show_metrics()
    st.divider()
    pos = show_chart_and_position()
    st.divider()
    c_left, c_right = st.columns([1, 1])
    with c_left: show_order_controls(pos)
    with c_right: show_bot_logic(pos)
