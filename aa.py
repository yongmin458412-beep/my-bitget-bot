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
from datetime import datetime
import matplotlib.pyplot as plt
import io

# =========================================================
# ⚙️ [설정] 기본 환경
# =========================================================
IS_SANDBOX = True # 모의투자
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="비트겟 봇 (AI Mode)")

# ---------------------------------------------------------
# 💾 설정 및 로그 관리
# ---------------------------------------------------------
def load_settings():
    default = {
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        "use_rsi": True, "use_bb": True, "use_ma": False, 
        "use_macd": False, "use_stoch": False, "use_cci": True, "use_vol": True
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
        with open(SETTINGS_FILE, "w") as f:
            json.dump(new_settings, f)
    except: pass

config = load_settings()
if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = config['order_usdt']

# ---------------------------------------------------------
# 🔐 API 로딩
# ---------------------------------------------------------
try:
    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    api_password = st.secrets["API_PASSWORD"]
    tg_token = st.secrets.get("TG_TOKEN", "")
    tg_id = st.secrets.get("TG_CHAT_ID", "")
except: st.error("🚨 Secrets 설정 필요"); st.stop()

# ---------------------------------------------------------
# 📊 수익 분석 함수 (개선됨)
# ---------------------------------------------------------
def log_trade(action, symbol, side, price, qty, leverage, pnl=0, roi=0):
    now = datetime.now()
    margin = (price * qty) / leverage
    new_data = {
        "Time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Date": now.strftime("%Y-%m-%d"),
        "Symbol": symbol, "Action": action, "Side": side,
        "Price": price, "Qty": qty, "Margin": margin, "PnL": pnl, "ROI": roi
    }
    df = pd.DataFrame([new_data])
    if not os.path.exists(LOG_FILE): df.to_csv(LOG_FILE, index=False)
    else: df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def get_analytics():
    """오늘 수익과 전체 누적 수익을 계산"""
    if not os.path.exists(LOG_FILE): return 0.0, 0.0, 0
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return 0.0, 0.0, 0
        
        # 전체 누적
        total_pnl = df['PnL'].sum()
        
        # 오늘 누적
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df['Date'] == today]
        daily_pnl = today_df['PnL'].sum()
        
        # 매매 횟수 (청산 기준)
        trade_count = len(df[df['Action'].str.contains('청산')])
        
        return daily_pnl, total_pnl, trade_count
    except: return 0.0, 0.0, 0

# ---------------------------------------------------------
# 📡 텔레그램 (잔고 상세 표시)
# ---------------------------------------------------------
def send_telegram(message, chart_df=None):
    if not tg_token or not tg_id: return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        keyboard = {"inline_keyboard": [[{"text": "🔍 실시간 현황 확인", "callback_data": "check_status"}]]}
        payload = {'chat_id': tg_id, 'text': message, 'parse_mode': 'HTML', 'reply_markup': json.dumps(keyboard)}
        requests.post(url, data=payload)
        
        if chart_df is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(chart_df['time'], chart_df['close'], color='yellow', label='Price')
            if 'MA_SLOW' in chart_df.columns: plt.plot(chart_df['time'], chart_df['MA_SLOW'], color='cyan', alpha=0.5)
            if 'BB_UP' in chart_df.columns:
                plt.plot(chart_df['time'], chart_df['BB_UP'], color='white', alpha=0.1)
                plt.plot(chart_df['time'], chart_df['BB_LO'], color='white', alpha=0.1)
            plt.title("Snapshot"); plt.grid(True, alpha=0.2); ax = plt.gca(); ax.set_facecolor('black'); plt.gcf().patch.set_facecolor('black'); ax.tick_params(colors='white')
            buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor='black'); buf.seek(0)
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendPhoto", data={'chat_id': tg_id}, files={'photo': buf}); plt.close()
    except: pass

def telegram_listener(exchange_obj, symbol_name):
    last_update_id = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={last_update_id+1}&timeout=30"
            res = requests.get(url).json()
            if res.get('ok') and res.get('result'):
                for update in res['result']:
                    last_update_id = update['update_id']
                    if 'callback_query' in update:
                        cb = update['callback_query']; cb_id = cb['id']; chat_id = cb['message']['chat']['id']
                        if cb['data'] == 'check_status':
                            # 1. 잔고 및 자산 계산
                            try:
                                bal = exchange_obj.fetch_balance({'type': 'swap'})
                                if 'SUSDT' in bal: coin='SUSDT'
                                elif 'USDT' in bal: coin='USDT'
                                else: coin='SBTC'
                                wallet = float(bal[coin]['total']) if coin in bal else 0.0
                            except: wallet = 0.0; coin="USDT"

                            msg = ""; unrealized_pnl = 0.0
                            try:
                                positions = exchange_obj.fetch_positions([symbol_name])
                                for p in positions:
                                    if float(p['contracts']) > 0:
                                        unrealized_pnl = float(p['unrealizedPnl'])
                                        msg = f"📊 <b>포지션 현황</b>\n• {symbol_name} <b>{p['side'].upper()}</b> x{p['leverage']}\n• 수익률: <b>{float(p['percentage']):.2f}%</b>\n• 미실현손익: ${unrealized_pnl:.2f}\n------------------\n"
                                        break
                                if not msg: msg = "📉 <b>포지션 없음</b> (대기 중)\n------------------\n"
                            except: msg = "❌ 데이터 조회 실패\n"

                            equity = wallet + unrealized_pnl
                            msg += f"💰 <b>지갑 잔고 (사용가능):</b> ${wallet:,.2f}\n"
                            msg += f"💎 <b>총 추정 자산 (Equity):</b> ${equity:,.2f}"
                            
                            send_telegram(msg)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb_id})
            time.sleep(1)
        except: time.sleep(5)

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
# 🎨 사이드바
# ---------------------------------------------------------
st.sidebar.title("🛠️ 봇 정밀 설정")
is_mobile = st.sidebar.checkbox("📱 모바일 모드", value=True)

markets = exchange.markets
futures_symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", futures_symbols, index=0)

# 리스너 시작 (중복 방지)
thread_exists = False
for t in threading.enumerate():
    if t.name == "TelegramListener": thread_exists = True; break
if not thread_exists:
    t = threading.Thread(target=telegram_listener, args=(exchange, symbol), daemon=True, name="TelegramListener")
    t.start()

# 원웨이 모드 강제
try:
    exchange.set_leverage(config['leverage'], symbol)
    try: exchange.set_position_mode(hedged=False, symbol=symbol)
    except: pass
except: pass

st.sidebar.divider()
st.sidebar.subheader("📊 지표 및 전략")

P = {} 
with st.sidebar.expander("1. RSI", expanded=True):
    use_rsi = st.checkbox("RSI 사용", value=config['use_rsi'])
    P['rsi_period'] = st.number_input("RSI 기간", 5, 100, 14)
    P['rsi_buy'] = st.slider("롱 진입 (이하)", 10, 50, 30)
    P['rsi_sell'] = st.slider("숏 진입 (이상)", 50, 90, 70)

with st.sidebar.expander("2. 볼린저밴드", expanded=True):
    use_bb = st.checkbox("볼린저밴드 사용", value=config['use_bb'])
    P['bb_period'] = st.number_input("BB 기간", 10, 50, 20)
    P['bb_std'] = st.number_input("승수", 1.0, 3.0, 2.0)

with st.sidebar.expander("3. 이동평균선", expanded=False):
    use_ma = st.checkbox("이평선 사용", value=config['use_ma'])
    P['ma_fast'] = st.number_input("단기", 1, 100, 5)
    P['ma_slow'] = st.number_input("장기", 10, 200, 60)

with st.sidebar.expander("4. MACD", expanded=False):
    use_macd = st.checkbox("MACD 사용", value=config['use_macd'])

with st.sidebar.expander("5. 스토캐스틱", expanded=False):
    use_stoch = st.checkbox("스토캐스틱 사용", value=config['use_stoch'])
    P['stoch_k'] = st.number_input("K 기간", 5, 30, 14)

with st.sidebar.expander("6. CCI", expanded=True):
    use_cci = st.checkbox("CCI 사용", value=config['use_cci'])

with st.sidebar.expander("9. 거래량", expanded=True):
    use_vol = st.checkbox("거래량 감지", value=config['use_vol'])
    P['vol_mul'] = st.number_input("거래량 배수", 1.5, 5.0, 2.0)

active_indicators = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_vol])

st.sidebar.divider()
target_vote = st.sidebar.slider("🎯 진입 조건 (신호 개수)", 1, max(1, active_indicators), config['target_vote'])
p_leverage = st.sidebar.slider("레버리지", 1, 50, config['leverage'])
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, config['tp'])
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, config['sl'])

if st.sidebar.button("📡 연결 상태 정밀진단"):
    send_telegram("✅ <b>시스템 점검 완료!</b>\n이상 없습니다.")
    st.toast("진단 완료")

# ---------------------------------------------------------
# 🧮 지표 계산
# ---------------------------------------------------------
def calculate_indicators(df, params):
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_MA'] = close.rolling(params['bb_period']).mean()
    df['BB_STD'] = close.rolling(params['bb_period']).std()
    df['BB_UP'] = df['BB_MA'] + (df['BB_STD'] * params['bb_std'])
    df['BB_LO'] = df['BB_MA'] - (df['BB_STD'] * params['bb_std'])
    
    tp = (df['high'] + df['low'] + close) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - sma) / (0.015 * mad)
    
    df['VOL_MA'] = df['vol'].rolling(20).mean()
    
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ---------------------------------------------------------
# ⚡ 주문 및 스마트 관리
# ---------------------------------------------------------
def safe_rerun():
    time.sleep(0.5); 
    if hasattr(st, 'rerun'): st.rerun()
    else: st.experimental_rerun()

def execute_trade(side, is_close=False, reason=""):
    try:
        if not is_close: exchange.set_leverage(p_leverage, symbol)
        
        qty = 0.0; params = {}; log_pnl = 0; log_roi = 0
        if is_close:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if float(p['contracts']) > 0), None)
            if not pos: return
            qty = float(pos['contracts'])
            params = {'reduceOnly': True}
            order_side = 'sell' if pos['side'] == 'long' else 'buy'
            emoji = "💰"; log_pnl = float(pos['unrealizedPnl']); log_roi = float(pos['percentage'])
        else:
            input_val = st.session_state['order_usdt']
            raw_qty = (input_val * p_leverage) / curr_price
            qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            emoji = "🚀"
            
        price = ticker['ask']*1.01 if order_side == 'buy' else ticker['bid']*0.99
        exchange.create_order(symbol, 'limit', order_side, qty, price, params=params)
        
        action_name = "청산" if is_close else "진입"
        log_trade(action_name, symbol, side, curr_price, qty, p_leverage, log_pnl, log_roi)
        
        daily_pnl, total_pnl, _ = get_analytics()
        
        bal = exchange.fetch_balance({'type': 'swap'})
        if 'SUSDT' in bal: coin='SUSDT'
        elif 'USDT' in bal: coin='USDT'
        else: coin='SBTC'
        wallet = float(bal[coin]['total']) if coin in bal else 0.0
        
        msg = f"{emoji} <b>{side.upper()} {action_name} 완료</b>\n--------------------------------\n📍 <b>이유:</b> {reason}\n💲 <b>가격:</b> ${curr_price:,.2f}"
        if is_close: msg += f"\n📈 <b>실현 수익:</b> ${log_pnl:.2f} ({log_roi:.2f}%)\n📅 <b>오늘 수익:</b> ${daily_pnl:.2f}\n🏆 <b>전체 수익:</b> ${total_pnl:.2f}"
        else: msg += f"\n💸 <b>투자금:</b> ${(float(qty)*curr_price)/p_leverage:,.2f}"
        msg += f"\n--------------------------------\n💰 <b>지갑 잔고:</b> ${wallet:,.2f}"

        st.success(msg.replace("<b>", "").replace("</b>", ""))
        send_telegram(msg, df.tail(60) if not is_close else None)
        safe_rerun()
    except Exception as e: st.error(f"주문 실패: {e}")

# =========================================================
# 📊 데이터 및 메인 로직
# =========================================================
usdt_free = 0.0; margin_coin_display = "USDT"
try:
    ticker = exchange.fetch_ticker(symbol); curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = calculate_indicators(df, P)
    last = df.iloc[-1]
    
    bal = exchange.fetch_balance({'type': 'swap'})
    if 'USDT' in bal and float(bal['USDT']['free']) > 0: usdt_free = float(bal['USDT']['free']); margin_coin_display = "USDT"
    elif 'SUSDT' in bal and float(bal['SUSDT']['free']) > 0: usdt_free = float(bal['SUSDT']['free']); margin_coin_display = "SUSDT"
    elif 'SBTC' in bal and float(bal['SBTC']['free']) > 0: usdt_free = float(bal['SBTC']['free']); margin_coin_display = "SBTC"
except Exception as e: st.error(f"데이터 로딩 실패: {e}"); st.stop()

# 화면 표시
st.title(f"🔥 {symbol}")
daily_pnl, total_pnl, _ = get_analytics()
color = "#4CAF50" if total_pnl >= 0 else "#FF5252"
st.markdown(f"""<div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; justify-content: space-around;"><div style="text-align: center;"><span style="color: #888;">사용 가능 잔고</span><br><span style="font-size: 1.5em; color: white;">${usdt_free:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">오늘 수익</span><br><span style="font-size: 1.5em; color: white;">${daily_pnl:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">전체 누적</span><br><span style="font-size: 1.5em; color: {color};">${total_pnl:,.2f}</span></div></div>""", unsafe_allow_html=True)

# 차트
tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
h = 350 if is_mobile else 450
components.html(f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "{tv_symbol}", "interval": "1", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>""", height=h)

# 포지션 확인
active_pos = None
try:
    positions = exchange.fetch_positions([symbol])
    for p in positions:
        if float(p['contracts']) > 0: active_pos = p; break
except: pass

if active_pos:
    roi = float(active_pos['percentage'])
    st.markdown(f"""<div style="border: 2px solid {'#4CAF50' if roi>=0 else '#FF5252'}; padding: 10px; border-radius: 10px; background: #262730;"><h3 style="margin:0;">{active_pos['side'].upper()} (x{active_pos['leverage']})</h3><p>수익률: {roi:.2f}% | 미실현: ${float(active_pos['unrealizedPnl']):.2f}</p></div>""", unsafe_allow_html=True)

# 신호 계산
long_score = 0; short_score = 0; reasons_L = []; reasons_S = []
if use_rsi:
    if last['RSI'] <= P['rsi_buy']: long_score+=1; reasons_L.append("RSI과매도")
    elif last['RSI'] >= P['rsi_sell']: short_score+=1; reasons_S.append("RSI과매수")
if use_bb:
    if last['close'] <= last['BB_LO']: long_score+=1; reasons_L.append("BB하단")
    elif last['close'] >= last['BB_UP']: short_score+=1; reasons_S.append("BB상단")
if use_cci:
    if last['CCI'] < -100: long_score+=1; reasons_L.append("CCI저점")
    elif last['CCI'] > 100: short_score+=1; reasons_S.append("CCI고점")
if use_vol:
    if last['vol'] > last['VOL_MA'] * P['vol_mul']: long_score+=1; short_score+=1; reasons_L.append("거래량↑"); reasons_S.append("거래량↑")

c1, c2 = st.columns(2)
c1.metric("📈 롱 점수", f"{long_score}/{target_vote}")
c2.metric("📉 숏 점수", f"{short_score}/{target_vote}")

final_long = long_score >= target_vote
final_short = short_score >= target_vote

# 👇 [설정 저장]
current_settings = {
    "leverage": p_leverage, "target_vote": target_vote, "tp": tp_pct, "sl": sl_pct,
    "auto_trade": st.session_state.get('auto_trade', False),
    "use_rsi": use_rsi, "use_bb": use_bb, "use_ma": use_ma, "use_macd": use_macd,
    "use_stoch": use_stoch, "use_cci": use_cci, "use_vol": use_vol,
    "order_usdt": st.session_state.get('order_usdt', 100.0)
}
if current_settings != config: save_settings(current_settings)

# 👇 [지능형 리스크 관리 & 자동매매]
t1, t2 = st.tabs(["🤖 자동매매", "⚡ 수동주문"])
with t1:
    auto_on = st.checkbox("자동매매 활성화", value=config['auto_trade'], key="auto_trade")
    if auto_on:
        if not active_pos:
            if final_long: execute_trade('long', reason=",".join(reasons_L))
            elif final_short: execute_trade('short', reason=",".join(reasons_S))
        else:
            # 🧠 스마트 방어 로직
            cur_side = active_pos['side']
            roi = float(active_pos['percentage'])
            
            # 1. 익절은 칼같이
            if roi >= tp_pct: execute_trade(cur_side, True, "목표 달성")
            
            # 2. 손실 상황 (-10% 이하) 발생 시 판단
            elif roi <= -sl_pct:
                # Case A: 반대 신호가 강력함 -> 스위칭 (손절 후 반대 진입)
                if (cur_side == 'long' and short_score >= target_vote) or \
                   (cur_side == 'short' and long_score >= target_vote):
                    execute_trade(cur_side, True, "🚨 손절 후 스위칭 (추세 전환)")
                    time.sleep(1)
                    if cur_side == 'long': execute_trade('short', reason="스위칭 진입")
                    else: execute_trade('long', reason="스위칭 진입")
                
                # Case B: 내 방향 신호가 아직 있음 (가짜 하락) -> 버티기 (최대 -20%까지)
                elif (cur_side == 'long' and long_score > 0) or \
                     (cur_side == 'short' and short_score > 0):
                    if roi <= -20.0: # 그래도 -20% 찍으면 사망
                        execute_trade(cur_side, True, "💀 강제 청산 (최대 손절폭 도달)")
                    else:
                        st.warning(f"📉 손실 중이나 지표가 살아있어 대기합니다. (ROI: {roi:.2f}%)")
                
                # Case C: 아무 신호도 없음 -> 그냥 손절
                else:
                    execute_trade(cur_side, True, "손절 제한 (가망 없음)")

        time.sleep(3); safe_rerun()

with t2:
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
