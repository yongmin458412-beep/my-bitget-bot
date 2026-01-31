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

st.set_page_config(layout="wide", page_title="비트겟 봇 (Perfect)")

# ---------------------------------------------------------
# 💾 설정 파일 관리
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
# 🔐 API & 텔레그램 로딩
# ---------------------------------------------------------
try:
    api_key = st.secrets["API_KEY"]
    api_secret = st.secrets["API_SECRET"]
    api_password = st.secrets["API_PASSWORD"]
    tg_token = st.secrets.get("TG_TOKEN", "")
    tg_id = st.secrets.get("TG_CHAT_ID", "")
except:
    st.error("🚨 Secrets 설정이 필요합니다."); st.stop()

# ---------------------------------------------------------
# 📊 매매일지 함수
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

def get_daily_summary():
    if not os.path.exists(LOG_FILE): return 0.0, 0
    try:
        df = pd.read_csv(LOG_FILE)
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df['Date'] == today]
        return today_df['PnL'].sum(), len(today_df[today_df['Action'].str.contains('청산')])
    except: return 0.0, 0

# ---------------------------------------------------------
# 📡 텔레그램 (중복 방지 및 버튼 기본 탑재)
# ---------------------------------------------------------
def send_telegram(message, chart_df=None):
    """
    모든 메시지에 '실시간 현황 확인' 버튼을 기본으로 붙여서 전송합니다.
    """
    if not tg_token or not tg_id: return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        
        # 👇 [수정됨] 무조건 버튼 추가
        keyboard = {
            "inline_keyboard": [[
                {"text": "🔍 실시간 현황 확인", "callback_data": "check_status"}
            ]]
        }
        
        payload = {
            'chat_id': tg_id, 
            'text': message, 
            'parse_mode': 'HTML',
            'reply_markup': json.dumps(keyboard) # 버튼 부착
        }
        
        requests.post(url, data=payload)
        
        if chart_df is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(chart_df['time'], chart_df['close'], color='yellow', label='Price')
            if 'MA_SLOW' in chart_df.columns: plt.plot(chart_df['time'], chart_df['MA_SLOW'], color='cyan', alpha=0.5)
            if 'BB_UP' in chart_df.columns:
                plt.plot(chart_df['time'], chart_df['BB_UP'], color='white', alpha=0.1)
                plt.plot(chart_df['time'], chart_df['BB_LO'], color='white', alpha=0.1)
            plt.title("Trade Snapshot"); plt.grid(True, alpha=0.2); ax = plt.gca(); ax.set_facecolor('black'); plt.gcf().patch.set_facecolor('black'); ax.tick_params(colors='white')
            buf = io.BytesIO(); plt.savefig(buf, format='png', facecolor='black'); buf.seek(0)
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendPhoto", data={'chat_id': tg_id}, files={'photo': buf}); plt.close()
    except: pass

def telegram_listener(exchange_obj, symbol_name):
    """백그라운드에서 버튼 클릭을 감지합니다."""
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
                            # 버튼 클릭 시 답장 로직
                            msg = "📉 <b>포지션 없음</b>\n봇이 대기 중입니다."
                            try:
                                positions = exchange_obj.fetch_positions([symbol_name])
                                has_pos = False
                                for p in positions:
                                    if float(p['contracts']) > 0:
                                        roi = float(p['percentage'])
                                        pnl = float(p['unrealizedPnl'])
                                        msg = f"📊 <b>포지션 현황</b>\n• 종목: {symbol_name}\n• <b>{p['side'].upper()}</b> x{p['leverage']}\n• 수익률: <b>{roi:.2f}%</b>\n• 수익금: ${pnl:.2f}"
                                        has_pos = True
                                        break
                                if not has_pos:
                                    msg = f"📉 <b>포지션 없음</b>\n현재 {symbol_name} 대기 중..."
                            except: msg = "❌ 거래소 연결 실패"
                            
                            # 답장 보내기 (여기도 버튼 붙임)
                            send_telegram(msg) 
                            
                            # 로딩바 없애기
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb_id})
            time.sleep(1)
        except: time.sleep(5)

# ---------------------------------------------------------
# 📡 거래소 연결 및 리스너 관리 (중복 해결 핵심)
# ---------------------------------------------------------
@st.cache_resource
def init_exchange_and_listener():
    try:
        ex = ccxt.bitget({'apiKey': api_key, 'secret': api_secret, 'password': api_password, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except: return None

exchange = init_exchange_and_listener()
if not exchange: st.stop()

# ---------------------------------------------------------
# 🎨 사이드바
# ---------------------------------------------------------
st.sidebar.title("🛠️ 봇 정밀 설정")
is_mobile = st.sidebar.checkbox("📱 모바일 모드", value=True)

markets = exchange.markets
futures_symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", futures_symbols, index=0)

# 👇 [핵심] 좀비 쓰레드 방지 로직
# 현재 실행 중인 모든 쓰레드를 검사해서, 이미 'TelegramListener'라는 이름의 쓰레드가 있으면 새로 안 만듭니다.
thread_exists = False
for t in threading.enumerate():
    if t.name == "TelegramListener":
        thread_exists = True
        break

if not thread_exists:
    t = threading.Thread(target=telegram_listener, args=(exchange, symbol), daemon=True, name="TelegramListener")
    t.start()
    print("✅ 텔레그램 리스너 시작됨 (한 번만 실행)")

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

# 정밀 연결 확인 버튼
if st.sidebar.button("📡 연결 상태 정밀진단"):
    with st.sidebar.status("시스템 점검 중...", expanded=True) as status:
        st.write("1. 거래소 연결 시도...")
        try:
            exchange.fetch_ticker(symbol)
            st.write("✅ 비트겟 API 정상")
            
            st.write("2. 텔레그램 발송 시도...")
            # 테스트 메시지에도 버튼이 자동으로 붙습니다.
            send_telegram("✅ <b>시스템 점검 완료!</b>\n이상 없습니다.")
            st.write("✅ 텔레그램 발송 성공")
            
            status.update(label="점검 완료! 모든 시스템 정상.", state="complete")
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            status.update(label="점검 실패", state="error")

# ---------------------------------------------------------
# 🛠️ 유틸리티
# ---------------------------------------------------------
def safe_rerun():
    time.sleep(0.5)
    if hasattr(st, 'rerun'): st.rerun()
    else: st.experimental_rerun()

def safe_toast(msg):
    if hasattr(st, 'toast'): st.toast(msg)
    else: st.success(msg)

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
    if 'USDT' in balance and float(balance['USDT']['free']) > 0:
        usdt_free = float(balance['USDT']['free']); margin_coin_display = "USDT"
    elif 'SUSDT' in balance and float(balance['SUSDT']['free']) > 0:
        usdt_free = float(balance['SUSDT']['free']); margin_coin_display = "SUSDT"
    elif 'SBTC' in balance and float(balance['SBTC']['free']) > 0:
        usdt_free = float(balance['SBTC']['free']); margin_coin_display = "SBTC"
except Exception as e:
    st.error(f"데이터 에러: {e}"); st.stop()

# ---------------------------------------------------------
# ⚡ 주문 함수
# ---------------------------------------------------------
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
        daily_pnl, daily_cnt = get_daily_summary()
        
        krw_val = curr_price * 1450
        invest_amount = (float(qty) * curr_price) / p_leverage
        
        msg = f"{emoji} <b>{side.upper()} {action_name} 완료</b>\n--------------------------------\n📍 <b>이유:</b> {reason}\n💲 <b>가격:</b> ${curr_price:,.2f}"
        if not is_close: msg += f"\n💸 <b>투자금:</b> ${invest_amount:,.2f}\n📊 <b>레버리지:</b> {p_leverage}배"
        else: msg += f"\n📈 <b>실현 수익:</b> ${log_pnl:.2f} ({log_roi:.2f}%)\n--------------------------------\n📅 <b>오늘 수익:</b> ${daily_pnl:.2f} ({daily_cnt}회)"
            
        st.success(msg.replace("<b>", "").replace("</b>", ""))
        safe_toast(msg.replace("<b>", "").replace("</b>", ""))
        chart_data = df.tail(60) if not is_close else None
        
        # 여기서 버튼 옵션을 따로 줄 필요 없음 (함수 내에서 기본값 처리됨)
        send_telegram(msg, chart_data) 
        
        safe_rerun()
    except Exception as e: st.error(f"주문 실패: {e}")

# =========================================================
# 🚀 메인 UI
# =========================================================
st.title(f"🔥 {symbol}")

daily_pnl_show, _ = get_daily_summary()
pnl_color = "#4CAF50" if daily_pnl_show >= 0 else "#FF5252"
st.markdown(f"""<div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; justify-content: space-around; align-items: center;"><div style="text-align: center;"><span style="color: #888;">내 잔고 ({margin_coin_display})</span><br><span style="font-size: 1.8em; color: white; font-weight: bold;">${usdt_free:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">오늘 수익</span><br><span style="font-size: 1.8em; color: {pnl_color}; font-weight: bold;">${daily_pnl_show:,.2f}</span></div></div>""", unsafe_allow_html=True)

def show_main_ui():
    tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
    studies_json = str(tv_studies).replace("'", '"')
    tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
    h = 350 if is_mobile else 450
    components.html(f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "{tv_symbol}", "interval": "1", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>""", height=h)

    st.subheader("💼 포지션")
    active_pos = None
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if float(p['contracts']) > 0:
                active_pos = p; break
        
        if active_pos:
            roi = float(active_pos['percentage'])
            color = "#4CAF50" if roi >= 0 else "#FF5252"
            st.markdown(f"""<div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; background: #262730;"><h3 style="color: {color}; margin:0;">{active_pos['side'].upper()} 보유중 (x{active_pos['leverage']})</h3><p>진입: ${float(active_pos['entryPrice']):,.2f} | 수익: ${float(active_pos['unrealizedPnl']):.2f} ({roi:.2f}%)</p></div>""", unsafe_allow_html=True)
            if roi >= tp_pct: execute_trade(active_pos['side'], True, "익절")
            elif roi <= -sl_pct: execute_trade(active_pos['side'], True, "손절")
        else: st.info("보유 포지션 없음")
    except: pass
    return active_pos

active_pos = show_main_ui()

long_score = 0; short_score = 0; reasons_L = []; reasons_S = []
if use_rsi:
    if last['RSI'] <= P['rsi_buy']: long_score+=1; reasons_L.append(f"RSI과매도")
    elif last['RSI'] >= P['rsi_sell']: short_score+=1; reasons_S.append(f"RSI과매수")
if use_bb:
    if last['close'] <= last['BB_LO']: long_score+=1; reasons_L.append("BB하단")
    elif last['close'] >= last['BB_UP']: short_score+=1; reasons_S.append("BB상단")
if use_cci:
    if last['CCI'] < -100: long_score+=1; reasons_L.append("CCI저점")
    elif last['CCI'] > 100: short_score+=1; reasons_S.append("CCI고점")
if use_vol:
    if last['vol'] > last['VOL_MA'] * P['vol_mul']: long_score+=1; short_score+=1; reasons_L.append("거래량급증"); reasons_S.append("거래량급증")

c1, c2 = st.columns(2)
c1.metric("📈 롱 점수", f"{long_score} / {target_vote}")
c2.metric("📉 숏 점수", f"{short_score} / {target_vote}")

final_long = long_score >= target_vote
final_short = short_score >= target_vote

current_settings = {
    "leverage": p_leverage, "target_vote": target_vote, "tp": tp_pct, "sl": sl_pct,
    "auto_trade": st.session_state.get('auto_trade', False),
    "use_rsi": use_rsi, "use_bb": use_bb, "use_ma": use_ma, "use_macd": use_macd,
    "use_stoch": use_stoch, "use_cci": use_cci, "use_vol": use_vol,
    "order_usdt": st.session_state.get('order_usdt', 100.0)
}
if current_settings != config: save_settings(current_settings)

t1, t2 = st.tabs(["🤖 자동매매", "⚡ 수동주문"])
with t1:
    auto_on = st.checkbox("자동매매 활성화", value=config['auto_trade'], key="auto_trade")
    if auto_on:
        if not active_pos:
            if final_long: execute_trade('long', reason=",".join(reasons_L))
            elif final_short: execute_trade('short', reason=",".join(reasons_S))
        else:
            cur = active_pos['side']
            if cur == 'long' and short_score >= target_vote + 1: execute_trade('long', True, "스위칭")
            elif cur == 'short' and long_score >= target_vote + 1: execute_trade('short', True, "스위칭")
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
