import streamlit as st
import streamlit.components.v1 as components
import ccxt
import pandas as pd
import numpy as np
import time
import requests # 텔레그램 전송용 라이브러리

# =========================================================
# ⚙️ [설정] 초기 세팅
# =========================================================
IS_SANDBOX = True 
try:
    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    api_password = st.secrets["API_PASSWORD"]
except:
    st.error("🚨 API 키가 설정되지 않았습니다. Streamlit Secrets를 설정해주세요.")
    st.stop()
st.set_page_config(layout="wide", page_title="비트겟 프로 봇 V5 (알림탑재)")

if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = 10.0

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

# 👇 [신규] 텔레그램 메시지 전송 함수
def send_telegram(token, chat_id, message):
    try:
        if token and chat_id:
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
st.sidebar.title("🛠️ 봇 설정 V5")
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

# 👇 [신규] 텔레그램 설정 섹션
st.sidebar.divider()
st.sidebar.subheader("🔔 텔레그램 알림")
tg_token = st.sidebar.text_input("봇 토큰 (Token)", type="password", placeholder="12345:ABCDE...")
tg_id = st.sidebar.text_input("챗 ID (Chat ID)", placeholder="12345678")

if st.sidebar.button("📩 알림 테스트 보내기"):
    send_telegram(tg_token, tg_id, "✅ 봇 연결 성공! 알림이 잘 옵니다.")
    st.sidebar.success("전송 시도 완료!")

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
# 메인 상단 정보
# ---------------------------------------------------------
st.title(f"🤖 {symbol} 트레이딩 봇")
m1, m2, m3, m4 = st.columns(4)
m1.metric("현재가", f"${curr_price:,.2f}")
m2.metric("RSI", f"{last['RSI']:.1f}")
m3.metric("가용 잔고", f"${usdt_free:,.2f}")
m4.metric("거래량", f"{last['vol']:.0f}")

st.divider()
st.subheader("✅ 보조지표 선택")

col_c1, col_c2 = st.columns(2)
with col_c1:
    use_rsi = st.checkbox("1. 과매수/과매도 (RSI)", value=True)
    use_bb = st.checkbox("2. 가격 급등락 (볼린저밴드)", value=True)
    use_ma = st.checkbox("3. 추세 방향 (이동평균 50선)")
    use_macd = st.checkbox("4. 상승/하락 신호 (MACD)")
    use_stoch = st.checkbox("5. 최저점 잡기 (스토캐스틱)")
with col_c2:
    use_cci = st.checkbox("6. 시장 과열 (CCI)")
    use_willr = st.checkbox("7. 단기 반전 (Williams %R)")
    use_vol = st.checkbox("8. 거래량 폭발 (Volume)")
    use_adx = st.checkbox("9. 추세 강도 (ADX)")
    use_sar = st.checkbox("10. 단기 골든크로스 (MA 5/20)")

# ---------------------------------------------------------
# 차트
# ---------------------------------------------------------
tv_studies = []
if use_rsi: tv_studies.append("RSI@tv-basicstudies")
if use_bb: tv_studies.append("BB@tv-basicstudies")
if use_ma: tv_studies.append("MASimple@tv-basicstudies") 
if use_macd: tv_studies.append("MACD@tv-basicstudies")
if use_stoch: tv_studies.append("Stochastic@tv-basicstudies")
if use_cci: tv_studies.append("CCI@tv-basicstudies")
if use_willr: tv_studies.append("WilliamsR@tv-basicstudies")

studies_json = str(tv_studies).replace("'", '"')
tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"

components.html(f"""
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
  "width": "100%", "height": 450, "symbol": "{tv_symbol}",
  "interval": "1", "timezone": "Asia/Seoul", "theme": "dark",
  "style": "1", "locale": "kr", "studies": {studies_json}, 
  "container_id": "tradingview_chart" }});
  </script>
</div>
""", height=450)

# ---------------------------------------------------------
# 주문 함수 (텔레그램 연동됨)
# ---------------------------------------------------------
def execute_trade(side, is_close=False, reason=""):
    try:
        if not is_close:
            exchange.set_leverage(p_leverage, symbol)
            
        qty = 0.0
        params = {}
        
        if is_close:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if float(p['contracts']) > 0), None)
            if not pos: return
            
            qty = float(pos['contracts'])
            params = {'reduceOnly': True}
            order_side = 'sell' if pos['side'] == 'long' else 'buy'
            trade_emoji = "💰" # 청산 이모지
        else:
            input_val = st.session_state['order_usdt']
            raw_qty = (input_val * p_leverage) / curr_price
            qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            trade_emoji = "🚀" # 진입 이모지
            
        price = ticker['ask']*1.01 if order_side == 'buy' else ticker['bid']*0.99
        exchange.create_order(symbol, 'limit', order_side, qty, price, params=params)
        
        # 메시지 생성
        act = "청산" if is_close else "진입"
        msg = f"{trade_emoji} {side.upper()} {act} 성공!\n코인: {symbol}\n이유: {reason}\n가격: ${curr_price:,.2f}"
        
        # 1. 화면 알림
        st.success(msg)
        safe_toast(msg)
        
        # 2. 텔레그램 알림 발송
        if tg_token and tg_id:
            send_telegram(tg_token, tg_id, msg)
            
        safe_rerun()
        
    except Exception as e:
        st.error(f"주문 에러: {e}")

# ---------------------------------------------------------
# 포지션 & 리스크 관리
# ---------------------------------------------------------
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
        
        color = "#4CAF50" if roi >= 0 else "#FF5252"
        st.markdown(f"""
        <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
            <h3 style="color: {color}; margin:0;">{side.upper()} 포지션 보유중</h3>
            <div style="display: flex; justify-content: space-between;">
                <span>진입가: ${entry:,.2f}</span>
                <span style="font-size: 1.2em; font-weight: bold;">수익률: {roi:.2f}% (PNL: ${pnl:.2f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if use_sl_tp:
            if roi >= tp_pct:
                st.warning(f"🚀 목표 수익률({tp_pct}%) 도달! 익절합니다.")
                execute_trade(side, is_close=True, reason="익절 달성")
            elif roi <= -sl_pct:
                st.error(f"📉 손실 제한({sl_pct}%) 초과! 손절합니다.")
                execute_trade(side, is_close=True, reason="손절 방어")
            else:
                limit_range = tp_pct + sl_pct
                current_pos = roi + sl_pct
                progress = min(max(current_pos / limit_range, 0.0), 1.0)
                st.caption(f"손절(-{sl_pct}%) ------------------ 현재({roi:.2f}%) ------------------ 익절(+{tp_pct}%)")
                st.progress(progress)

    else:
        st.info("현재 보유 중인 포지션이 없습니다.")

except Exception as e:
    st.error(f"포지션 조회 에러: {e}")

# ---------------------------------------------------------
# 수동 주문 UI
# ---------------------------------------------------------
st.divider()
st.subheader("⚡ 주문 실행")

col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
def set_amount(pct): st.session_state['order_usdt'] = float(f"{usdt_free * pct:.2f}")
if col_p1.button("10%"): set_amount(0.1)
if col_p2.button("25%"): set_amount(0.25)
if col_p3.button("50%"): set_amount(0.5)
if col_p4.button("75%"): set_amount(0.75)
if col_p5.button("100%"): set_amount(1.0)

input_usdt = st.number_input("주문 금액 (USDT)", 0.0, usdt_free, st.session_state['order_usdt'], step=10.0)
st.session_state['order_usdt'] = input_usdt

b1, b2, b3 = st.columns(3)
if b1.button("📈 롱 진입", use_container_width=True): execute_trade('long')
if b2.button("📉 숏 진입", use_container_width=True): execute_trade('short')
if b3.button("🚫 포지션 정리", use_container_width=True): 
    if active_position: execute_trade(active_position['side'], is_close=True, reason="수동 청산")

# ---------------------------------------------------------
# 봇 로직
# ---------------------------------------------------------
st.divider()
st.subheader("🧠 봇 자동매매")

signals_long = []
signals_short = []
reasons = []

if use_rsi:
    if last['RSI'] <= p_rsi_buy: signals_long.append(True); reasons.append("RSI 과매도")
    elif last['RSI'] >= p_rsi_sell: signals_short.append(True); reasons.append("RSI 과매수")
    else: signals_long.append(False); signals_short.append(False)

if use_bb:
    if last['close'] <= last['BB_LO']: signals_long.append(True); reasons.append("볼린저 하단")
    elif last['close'] >= last['BB_UP']: signals_short.append(True); reasons.append("볼린저 상단")
    else: signals_long.append(False); signals_short.append(False)

if use_ma:
    if last['close'] > last['MA50']: signals_long.append(True); reasons.append("상승 추세")
    else: signals_long.append(False) 

if use_macd:
    if last['MACD'] > last['MACD_Signal']: signals_long.append(True); reasons.append("MACD 골든")
    else: signals_long.append(False)

if use_stoch:
    if last['STOCH_K'] < 20: signals_long.append(True); reasons.append("스토캐스틱 저점")
    else: signals_long.append(False)

if use_cci:
    if last['CCI'] < -100: signals_long.append(True); reasons.append("CCI 과매도")
    else: signals_long.append(False)

if use_vol:
    if last['vol'] > last['VOL_MA'] * 1.5: signals_long.append(True); reasons.append("거래량 폭발")
    else: signals_long.append(False)

active_count = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_willr, use_vol, use_adx, use_sar])
final_long = all(signals_long) and (len(signals_long) > 0)
final_short = all(signals_short) and (len(signals_short) > 0)

c_res1, c_res2 = st.columns(2)
c_res1.info(f"체크된 지표: {active_count}개")
if final_long: c_res2.success(f"🔥 롱 진입 조건 만족! ({', '.join(reasons)})")
elif final_short: c_res2.error(f"🔥 숏 진입 조건 만족! ({', '.join(reasons)})")
else: c_res2.warning("⏳ 진입 조건 대기중...")

if st.checkbox("🤖 자동매매 활성화"):
    if not active_position:
        if final_long: execute_trade('long', reason="자동 진입"); 
        elif final_short: execute_trade('short', reason="자동 진입"); 
    else:
        current_side = active_position['side']
        if current_side == 'long' and final_short:
            execute_trade('long', is_close=True, reason="반대신호 스위칭")
        elif current_side == 'short' and final_long:
            execute_trade('short', is_close=True, reason="반대신호 스위칭")

    time.sleep(3)
    safe_rerun()