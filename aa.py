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
# ⚙️ [설정] 기본 환경 (모의투자 500불 전용)
# =========================================================
IS_SANDBOX = True 

st.set_page_config(layout="wide", page_title="비트겟 봇 (Final Fixed)")

if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = 100.0

# ---------------------------------------------------------
# 🔐 API 키 & 텔레그램 키 로딩
# ---------------------------------------------------------
try:
    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    api_password = st.secrets["API_PASSWORD"]
    
    default_tg_token = st.secrets.get("TG_TOKEN", "")
    default_tg_id = st.secrets.get("TG_CHAT_ID", "")
except:
    st.error("🚨 Secrets 설정이 필요합니다.")
    st.stop()

# ---------------------------------------------------------
# 📡 거래소 연결 (객체 생성만 먼저 함)
# ---------------------------------------------------------
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget({
            'apiKey': api_key, 
            'secret': api_secret, 
            'password': api_password, 
            'enableRateLimit': True, 
            'options': {'defaultType': 'swap'}
        })
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except: return None

exchange = init_exchange()
if not exchange: st.stop()

# ---------------------------------------------------------
# 🎨 사이드바 (여기서 변수들을 먼저 정의해야 에러가 안 남!)
# ---------------------------------------------------------
st.sidebar.title("🛠️ 봇 정밀 설정")
is_mobile = st.sidebar.checkbox("📱 모바일 모드", value=True)

markets = exchange.markets
futures_symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", futures_symbols, index=0)

# 👇 [에러 해결 핵심] 레버리지 변수를 여기서 먼저 정의합니다.
st.sidebar.divider()
st.sidebar.subheader("⚖️ 전략 및 리스크")

p_leverage = st.sidebar.slider("레버리지", 1, 50, 20) # 변수 정의 완료!

# 나머지 설정들...
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, 15.0)
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, 10.0)

st.sidebar.subheader("📊 지표 세부 설정")
P = {} 
with st.sidebar.expander("1. RSI", expanded=True):
    use_rsi = st.checkbox("RSI 사용", value=True)
    P['rsi_period'] = st.number_input("RSI 기간", 5, 100, 14)
    P['rsi_buy'] = st.slider("롱 진입 (이하)", 10, 50, 30)
    P['rsi_sell'] = st.slider("숏 진입 (이상)", 50, 90, 70)

with st.sidebar.expander("2. 볼린저밴드", expanded=True):
    use_bb = st.checkbox("볼린저밴드 사용", value=True)
    P['bb_period'] = st.number_input("BB 기간", 10, 50, 20)
    P['bb_std'] = st.number_input("승수", 1.0, 3.0, 2.0, step=0.1)

with st.sidebar.expander("3. 이동평균선", expanded=False):
    use_ma = st.checkbox("이평선 사용", value=False)
    P['ma_fast'] = st.number_input("단기 이평선", 1, 100, 5)
    P['ma_slow'] = st.number_input("장기 이평선", 10, 200, 60)

with st.sidebar.expander("4. MACD", expanded=False):
    use_macd = st.checkbox("MACD 사용", value=False)

with st.sidebar.expander("5. 스토캐스틱", expanded=False):
    use_stoch = st.checkbox("스토캐스틱 사용", value=False)
    P['stoch_k'] = st.number_input("K 기간", 5, 30, 14)

with st.sidebar.expander("6. CCI", expanded=False):
    use_cci = st.checkbox("CCI 사용", value=False)

with st.sidebar.expander("7. MFI", expanded=False):
    use_mfi = st.checkbox("MFI 사용", value=False)

with st.sidebar.expander("8. Williams %R", expanded=False):
    use_willr = st.checkbox("Williams %R 사용", value=False)

with st.sidebar.expander("9. 거래량", expanded=True):
    use_vol = st.checkbox("거래량 폭발 감지", value=True)
    P['vol_mul'] = st.number_input("평소 대비 배수", 1.5, 5.0, 2.0)

with st.sidebar.expander("10. ADX", expanded=False):
    use_adx = st.checkbox("ADX 사용", value=False)

active_indicators = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_mfi, use_willr, use_vol, use_adx])
target_vote = st.sidebar.slider(
    f"🎯 진입 조건 (총 {active_indicators}개 중)", 
    1, max(1, active_indicators), min(3, active_indicators)
)

st.sidebar.divider()
st.sidebar.subheader("🔔 텔레그램")
tg_token = st.sidebar.text_input("봇 토큰", value=default_tg_token, type="password")
tg_id = st.sidebar.text_input("챗 ID", value=default_tg_id)

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

def send_telegram(token, chat_id, message, chart_df=None):
    try:
        if not token or not chat_id: return
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={'chat_id': chat_id, 'text': message})
        
        if chart_df is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(chart_df['time'], chart_df['close'], label='Price', color='yellow')
            if 'MA_SLOW' in chart_df.columns:
                plt.plot(chart_df['time'], chart_df['MA_SLOW'], label='MA(Slow)', color='cyan', alpha=0.5)
            if 'BB_UP' in chart_df.columns:
                plt.plot(chart_df['time'], chart_df['BB_UP'], color='white', alpha=0.1)
                plt.plot(chart_df['time'], chart_df['BB_LO'], color='white', alpha=0.1)

            plt.title(f"Trading Signal")
            plt.legend()
            plt.grid(True, alpha=0.2)
            
            ax = plt.gca()
            ax.set_facecolor('black')
            plt.gcf().patch.set_facecolor('black')
            ax.tick_params(colors='white')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='black')
            buf.seek(0)
            
            requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data={'chat_id': chat_id}, files={'photo': buf})
            plt.close()
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

# ---------------------------------------------------------
# 🧮 보조지표 계산
# ---------------------------------------------------------
def calculate_indicators(df, params):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['vol']
    
    # 1. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. 볼린저밴드
    df['BB_MA'] = close.rolling(params['bb_period']).mean()
    df['BB_STD'] = close.rolling(params['bb_period']).std()
    df['BB_UP'] = df['BB_MA'] + (df['BB_STD'] * params['bb_std'])
    df['BB_LO'] = df['BB_MA'] - (df['BB_STD'] * params['bb_std'])

    # 3. 이동평균선
    df['MA_FAST'] = close.rolling(params['ma_fast']).mean()
    df['MA_SLOW'] = close.rolling(params['ma_slow']).mean()

    # 4. MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. 스토캐스틱
    k_period = params['stoch_k']
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # 6. CCI
    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean()
    def get_mad(x): return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(20).apply(get_mad)
    df['CCI'] = (tp - sma) / (0.015 * mad)

    # 7. MFI
    typical_price = (high + low + close) / 3
    money_flow = typical_price * vol
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # 8. Williams %R
    df['WILLR'] = -100 * ((highest_high - close) / (highest_high - lowest_low))

    # 9. ADX
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ADX'] = (tr.rolling(14).mean() / close) * 1000

    # 10. Volume MA
    df['VOL_MA'] = vol.rolling(20).mean()

    return df

# ---------------------------------------------------------
# 📊 데이터 로딩 & 잔고 로직
# ---------------------------------------------------------
usdt_free = 0.0
margin_coin_display = "USDT"

try:
    ticker = exchange.fetch_ticker(symbol)
    curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = calculate_indicators(df, P)
    last = df.iloc[-1]
    
    balance = exchange.fetch_balance({'type': 'swap'})
    
    # 잔고 우선순위 검색
    found_assets = {}
    for coin, info in balance.items():
        if isinstance(info, dict) and 'free' in info and info['free'] > 0:
            found_assets[coin] = info['free']

    if 'USDT' in found_assets:
        usdt_free = float(found_assets['USDT'])
        margin_coin_display = "USDT (Demo)"
    elif 'SUSDT' in found_assets:
        usdt_free = float(found_assets['SUSDT'])
        margin_coin_display = "SUSDT (Demo)"
    elif 'SBTC' in found_assets:
        usdt_free = float(found_assets['SBTC'])
        margin_coin_display = "SBTC (Demo)"
    else:
        for coin, amt in balance.get('total', {}).items():
            if float(amt) > 0:
                usdt_free = float(balance[coin]['free'])
                margin_coin_display = f"{coin} (Demo)"
                break

except Exception as e:
    st.error(f"데이터 로딩 오류: {e}")
    st.stop()

# ---------------------------------------------------------
# ⚡ 주문 함수 (레버리지 설정 안전하게 적용)
# ---------------------------------------------------------
def execute_trade(side, is_close=False, reason=""):
    try:
        if not is_close:
            # ⭐ 여기서 레버리지를 설정합니다. (변수가 이미 정의된 상태)
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
        
        # 메시지
        action = "청산" if is_close else "진입"
        krw_val = curr_price * 1450
        msg = f"{trade_emoji} {side.upper()} {action} 체결!\n"
        msg += f"📍 이유: {reason}\n"
        msg += f"💲 가격: ${curr_price:,.2f} (약 {krw_val:,.0f}원)\n"
        msg += f"📊 레버리지: {p_leverage}배"
        if is_close:
            msg += f"\n📈 실현손익: ${log_pnl:.2f} ({log_roi:.2f}%)"
            
        st.success(msg)
        safe_toast(msg)
        
        if tg_token and tg_id:
            chart_data = df.tail(60) if not is_close else None
            send_telegram(tg_token, tg_id, msg, chart_data)
            
        safe_rerun()
        
    except Exception as e:
        st.error(f"주문 실패: {e}")

# =========================================================
# 📱 UI 디스플레이
# =========================================================
def show_metrics():
    # 잔고 강조
    st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; text-align: center;">
        <span style="font-size: 1.2em; color: #888;">내 잔고 ({margin_coin_display})</span><br>
        <span style="font-size: 2.5em; color: #4CAF50; font-weight: bold;">${usdt_free:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    cols[0].metric("현재가", f"${curr_price:,.2f}")
    cols[1].metric("RSI", f"{last['RSI']:.1f}")
    cols[2].metric("변동성(BB)", f"{last['BB_STD']:.2f}")

def show_chart_and_pos():
    tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
    studies_json = str(tv_studies).replace("'", '"')
    tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
    h = 350 if is_mobile else 450
    components.html(f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "{tv_symbol}", "interval": "1", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});
      </script>
    </div>""", height=h)

    st.subheader("💼 포지션 상태")
    active_pos = None
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if float(p['contracts']) > 0:
                active_pos = p
                break
        
        if active_pos:
            side = active_pos['side']
            roi = float(active_pos['percentage'])
            pnl = float(active_pos['unrealizedPnl'])
            entry = float(active_pos['entryPrice'])
            color = "#4CAF50" if roi >= 0 else "#FF5252"
            
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; background: #262730;">
                <h3 style="color: {color}; margin:0;">{side.upper()} 보유중 (x{active_pos['leverage']})</h3>
                <p>진입: ${entry:,.2f} | 수익: ${pnl:.2f} ({roi:.2f}%)</p>
            </div>""", unsafe_allow_html=True)
            
            if roi >= tp_pct: execute_trade(side, True, "익절")
            elif roi <= -sl_pct: execute_trade(side, True, "손절")
        else:
            st.info("보유 포지션 없음")
            
    except: pass
    return active_pos

def show_strategy(active_pos):
    st.subheader("🧠 전략 분석 (다수결)")
    
    long_score = 0
    short_score = 0
    reasons_L = []
    reasons_S = []
    
    # 1. RSI
    if use_rsi:
        if last['RSI'] <= P['rsi_buy']: long_score+=1; reasons_L.append(f"RSI과매도")
        elif last['RSI'] >= P['rsi_sell']: short_score+=1; reasons_S.append(f"RSI과매수")
    # 2. BB
    if use_bb:
        if last['close'] <= last['BB_LO']: long_score+=1; reasons_L.append("BB하단")
        elif last['close'] >= last['BB_UP']: short_score+=1; reasons_S.append("BB상단")
    # 3. MA
    if use_ma:
        if last['close'] > last['MA_SLOW']: long_score+=1; reasons_L.append("이평상승")
        else: short_score+=1; reasons_S.append("이평하락")
    # 4. MACD
    if use_macd:
        if last['MACD'] > last['MACD_SIG']: long_score+=1; reasons_L.append("MACD골든")
        else: short_score+=1; reasons_S.append("MACD데드")
    # 5. Stoch
    if use_stoch:
        if last['STOCH_K'] < 20: long_score+=1; reasons_L.append("스토캐과매도")
        elif last['STOCH_K'] > 80: short_score+=1; reasons_S.append("스토캐과매수")
    # 6. CCI
    if use_cci:
        if last['CCI'] < -100: long_score+=1; reasons_L.append("CCI저점")
        elif last['CCI'] > 100: short_score+=1; reasons_S.append("CCI고점")
    # 7. MFI
    if use_mfi:
        if last['MFI'] < 20: long_score+=1; reasons_L.append("MFI저점")
        elif last['MFI'] > 80: short_score+=1; reasons_S.append("MFI고점")
    # 8. WillR
    if use_willr:
        if last['WILLR'] < -80: long_score+=1; reasons_L.append("WillR저점")
        elif last['WILLR'] > -20: short_score+=1; reasons_S.append("WillR고점")
    # 9. Volume
    if use_vol:
        if last['vol'] > last['VOL_MA'] * P['vol_mul']:
            long_score+=1; short_score+=1; reasons_L.append("거래량급증"); reasons_S.append("거래량급증")
    # 10. ADX
    if use_adx:
        if last['ADX'] > 25: long_score+=1; short_score+=1;

    c1, c2 = st.columns(2)
    c1.metric("📈 롱 점수", f"{long_score} / {target_vote}", delta=f"{long_score-target_vote}")
    c2.metric("📉 숏 점수", f"{short_score} / {target_vote}", delta=f"{short_score-target_vote}")
    
    final_long = long_score >= target_vote
    final_short = short_score >= target_vote
    
    if final_long: st.success(f"🔥 롱 진입 조건 만족! ({', '.join(reasons_L)})")
    if final_short: st.error(f"🔥 숏 진입 조건 만족! ({', '.join(reasons_S)})")

    st.divider()
    if st.checkbox("🤖 자동매매 활성화"):
        if not active_pos:
            if final_long: execute_trade('long', reason=",".join(reasons_L))
            elif final_short: execute_trade('short', reason=",".join(reasons_S))
        else:
            cur = active_pos['side']
            if cur == 'long' and short_score >= target_vote + 1: execute_trade('long', True, "스위칭")
            elif cur == 'short' and long_score >= target_vote + 1: execute_trade('short', True, "스위칭")
        time.sleep(3)
        safe_rerun()

def show_order(active_pos):
    st.subheader("⚡ 수동 주문")
    c1, c2, c3, c4 = st.columns(4)
    def set_amt(pct): st.session_state['order_usdt'] = float(f"{usdt_free * pct:.2f}")
    if c1.button("20%"): set_amt(0.2)
    if c2.button("50%"): set_amt(0.5)
    if c3.button("80%"): set_amt(0.8)
    if c4.button("Full"): set_amt(1.0)
    
    st.number_input("금액 (USDT)", 0.0, usdt_free, key='order_usdt')
    b1, b2 = st.columns(2)
    if b1.button("롱 진입", use_container_width=True): execute_trade('long', reason="수동")
    if b2.button("숏 진입", use_container_width=True): execute_trade('short', reason="수동")
    if st.button("포지션 청산", use_container_width=True):
        if active_pos: execute_trade(active_pos['side'], True, "수동청산")

# =========================================================
# 🚀 메인 실행
# =========================================================
st.title(f"🔥 {symbol}")

if is_mobile:
    show_metrics()
    t1, t2, t3 = st.tabs(["📊 차트", "⚡ 주문", "🧠 전략"])
    with t1: pos = show_chart_and_pos()
    with t2: show_order(pos)
    with t3: show_strategy(pos)
else:
    show_metrics()
    st.divider()
    pos = show_chart_and_pos()
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1: show_order(pos)
    with c2: show_strategy(pos)
