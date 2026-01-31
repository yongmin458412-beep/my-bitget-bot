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

st.set_page_config(layout="wide", page_title="비트겟 봇 (Master)")

# ---------------------------------------------------------
# 💾 설정 파일 관리
# ---------------------------------------------------------
def load_settings():
    default = {
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        "use_rsi": True, "use_bb": True, "use_ma": False, 
        "use_macd": False, "use_stoch": False, "use_cci": True, "use_vol": True,
        # 스마트 방어 & 추매 설정
        "use_switching": True, 
        "use_dca": False, "dca_trigger": -5.0, "dca_max_count": 1
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
# 📊 매매일지 및 수익 분석
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
    if not os.path.exists(LOG_FILE): return 0.0, 0.0, 0.0, 0
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return 0.0, 0.0, 0.0, 0
        
        # 청산된 건만 계산
        closed_trades = df[df['Action'].str.contains('청산')]
        total_pnl = closed_trades['PnL'].sum()
        
        # 오늘 누적
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = closed_trades[closed_trades['Date'] == today]
        daily_pnl = today_df['PnL'].sum()
        
        # 최근 ROI (마지막 거래)
        last_roi = closed_trades.iloc[-1]['ROI'] if not closed_trades.empty else 0.0
        
        return last_roi, daily_pnl, total_pnl, len(today_df)
    except: return 0.0, 0.0, 0.0, 0

# ---------------------------------------------------------
# 📡 텔레그램 (잔고 표시 완벽 수정)
# ---------------------------------------------------------
def get_balance_details(exchange_obj):
    """
    사용자가 원하는 잔고 표시 방식:
    1. 현재 잔고 (Free): 포지션 잡고 남은 쓸 수 있는 돈 (예: 400)
    2. 총 추정 자산 (Total + PnL): 내 원금 + 현재 수익금 (예: 500 + 10 = 510)
    """
    try:
        bal = exchange_obj.fetch_balance({'type': 'swap'})
        if 'SUSDT' in bal: coin = 'SUSDT'
        elif 'USDT' in bal: coin = 'USDT'
        else: coin = 'SBTC'
        
        free = float(bal[coin]['free'])   # 사용 가능 잔고 (400)
        total = float(bal[coin]['total']) # 지갑 총액 (500, 증거금 포함)
        return coin, free, total
    except:
        return "USDT", 0.0, 0.0

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
                            # 1. 잔고 조회
                            coin, free, total = get_balance_details(exchange_obj)
                            
                            # 2. 포지션 조회
                            msg = ""; unrealized_pnl = 0.0
                            has_pos = False
                            try:
                                positions = exchange_obj.fetch_positions([symbol_name])
                                for p in positions:
                                    if float(p['contracts']) > 0:
                                        unrealized_pnl = float(p['unrealizedPnl'])
                                        roi = float(p['percentage'])
                                        msg = f"📊 <b>포지션 현황</b>\n• {symbol_name} <b>{p['side'].upper()}</b> x{p['leverage']}\n"
                                        msg += f"• 수익률: <b>{roi:.2f}%</b>\n• 수익금: ${unrealized_pnl:.2f}\n------------------\n"
                                        has_pos = True; break
                                if not has_pos: msg = f"📉 <b>포지션 없음</b> (대기 중)\n------------------\n"
                            except: msg = "❌ 데이터 조회 실패\n"

                            # 3. 총 자산 계산 (지갑총액 + 미실현손익)
                            equity = total + unrealized_pnl
                            
                            # 4. 수익 현황
                            last_roi, d_pnl, t_pnl, _ = get_analytics()

                            msg += f"💰 <b>현재 잔고 (Free):</b> ${free:,.2f}\n"
                            msg += f"💎 <b>총합 잔고 (Equity):</b> ${equity:,.2f}\n"
                            msg += f"------------------\n"
                            msg += f"📅 금일 수익: ${d_pnl:,.2f}\n"
                            msg += f"🏆 총 누적 수익: ${t_pnl:,.2f}"
                            
                            send_telegram(msg)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb_id})
            time.sleep(1)
        except: time.sleep(5)

# ---------------------------------------------------------
# 📡 거래소 연결
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

# 리스너 시작
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
st.sidebar.subheader("🛡️ 방어 및 추매 설정")
# 👇 [추가됨] 추매 및 스위칭 UI
use_switching = st.sidebar.checkbox("스위칭 허용 (반대 신호 시)", value=config['use_switching'])
use_dca = st.sidebar.checkbox("추매(물타기) 허용", value=config['use_dca'])
dca_trigger = st.sidebar.number_input("추매 발동 (ROI %)", -50.0, -1.0, config['dca_trigger'], step=0.5, help="-5.0이면 -5% 손실 시 물탑니다.")
dca_max_count = st.sidebar.number_input("최대 추매 횟수", 1, 5, config['dca_max_count'], help="안전을 위해 1~2회를 추천합니다.")

st.sidebar.divider()
st.sidebar.subheader("📊 지표 설정")

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

with st.sidebar.expander("6. CCI", expanded=True):
    use_cci = st.checkbox("CCI 사용", value=config['use_cci'])

with st.sidebar.expander("9. 거래량", expanded=True):
    use_vol = st.checkbox("거래량 감지", value=config['use_vol'])
    P['vol_mul'] = st.number_input("거래량 배수", 1.5, 5.0, 2.0)

# 나머지 지표
use_ma = config['use_ma']; use_macd = config['use_macd']; use_stoch = config['use_stoch']
active_indicators = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_vol])

st.sidebar.divider()
target_vote = st.sidebar.slider("🎯 진입 조건 (신호 개수)", 1, max(1, active_indicators), config['target_vote'])
p_leverage = st.sidebar.slider("레버리지", 1, 50, config['leverage'])
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, config['tp'])
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, config['sl'])

# 👇 [복구됨] 테스트 버튼
if st.sidebar.button("📡 텔레그램 연결 테스트"):
    send_telegram("✅ <b>연결 테스트 성공!</b>\n아래 버튼을 눌러보세요.")
    st.toast("테스트 발송 완료")

# ---------------------------------------------------------
# 🛠️ 유틸리티 & 계산
# ---------------------------------------------------------
def safe_rerun():
    time.sleep(0.5); 
    if hasattr(st, 'rerun'): st.rerun()
    else: st.experimental_rerun()

def safe_toast(msg):
    if hasattr(st, 'toast'): st.toast(msg)
    else: st.success(msg)

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
# 📊 데이터 로딩
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol); curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = calculate_indicators(df, P)
    last = df.iloc[-1]
except Exception as e:
    st.error(f"데이터 에러: {e}"); st.stop()

# ---------------------------------------------------------
# ⚡ 주문 실행 함수 (추매 포함)
# ---------------------------------------------------------
def execute_trade(side, is_close=False, reason="", qty=0.0):
    try:
        if not is_close: exchange.set_leverage(p_leverage, symbol)
        
        params = {}; log_pnl = 0; log_roi = 0
        if is_close:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if float(p['contracts']) > 0), None)
            if not pos: return
            qty = float(pos['contracts'])
            params = {'reduceOnly': True}
            order_side = 'sell' if pos['side'] == 'long' else 'buy'
            emoji = "💰"; log_pnl = float(pos['unrealizedPnl']); log_roi = float(pos['percentage'])
        else:
            # 진입 또는 추매
            if qty == 0.0: # 첫 진입 시
                input_val = st.session_state['order_usdt']
                raw_qty = (input_val * p_leverage) / curr_price
                qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            emoji = "🚀"
            
        price = ticker['ask']*1.01 if order_side == 'buy' else ticker['bid']*0.99
        exchange.create_order(symbol, 'limit', order_side, qty, price, params=params)
        
        # 로그 및 알림
        action_name = "청산" if is_close else "진입/추매"
        if is_close: log_trade(action_name, symbol, side, curr_price, qty, p_leverage, log_pnl, log_roi)
        
        last_roi, d_pnl, t_pnl, _ = get_analytics()
        coin, free, total = get_balance_details(exchange)
        # 포지션 있을 때 총 자산은 (총액 + 미실현손익)
        unrealized = log_pnl if is_close else 0.0 # 지금은 약식
        equity = total + unrealized
        
        msg = f"{emoji} <b>{side.upper()} {action_name} 완료</b>\n--------------------------------\n📍 <b>이유:</b> {reason}\n💲 <b>가격:</b> ${curr_price:,.2f}"
        if is_close: 
            msg += f"\n📈 <b>실현 수익:</b> ${log_pnl:.2f} ({log_roi:.2f}%)\n📅 <b>금일 수익:</b> ${d_pnl:.2f}\n🏆 <b>총 누적 수익:</b> ${t_pnl:.2f}"
        else: 
            msg += f"\n💸 <b>투자금(증거금):</b> ${(float(qty)*curr_price)/p_leverage:,.2f}"
        
        msg += f"\n--------------------------------\n💰 <b>현재 잔고 (Free):</b> ${free:,.2f}\n💎 <b>총 추정 자산 (Equity):</b> ${equity:,.2f}"

        st.success(msg.replace("<b>", "").replace("</b>", ""))
        send_telegram(msg, df.tail(60) if not is_close else None)
        safe_rerun()
    except Exception as e: st.error(f"주문 실패: {e}")

# =========================================================
# 🚀 메인 UI
# =========================================================
st.title(f"🔥 {symbol}")

# 상단 대시보드
coin, free, total = get_balance_details(exchange)
_, d_pnl, t_pnl, _ = get_analytics()
pnl_color = "#4CAF50" if d_pnl >= 0 else "#FF5252"

st.markdown(f"""<div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; justify-content: space-around;"><div style="text-align: center;"><span style="color: #888;">현재 잔고(Free)</span><br><span style="font-size: 1.5em; color: white;">${free:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">금일 수익</span><br><span style="font-size: 1.5em; color: {pnl_color};">${d_pnl:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">총 누적 수익</span><br><span style="font-size: 1.5em; color: {'#4CAF50' if t_pnl>=0 else '#FF5252'};">${t_pnl:,.2f}</span></div></div>""", unsafe_allow_html=True)

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
            if float(p['contracts']) > 0: active_pos = p; break
    except: pass

    if active_pos:
        roi = float(active_pos['percentage'])
        st.markdown(f"""<div style="border: 2px solid {'#4CAF50' if roi>=0 else '#FF5252'}; padding: 10px; border-radius: 10px; background: #262730;"><h3 style="margin:0;">{active_pos['side'].upper()} (x{active_pos['leverage']})</h3><p>수익률: {roi:.2f}% | 미실현: ${float(active_pos['unrealizedPnl']):.2f}</p></div>""", unsafe_allow_html=True)
    return active_pos

active_pos = show_main_ui()

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

# 설정 저장
current_settings = {
    "leverage": p_leverage, "target_vote": target_vote, "tp": tp_pct, "sl": sl_pct,
    "auto_trade": st.session_state.get('auto_trade', False),
    "use_rsi": use_rsi, "use_bb": use_bb, "use_ma": use_ma, "use_macd": use_macd,
    "use_stoch": use_stoch, "use_cci": use_cci, "use_vol": use_vol,
    "use_switching": use_switching, "use_dca": use_dca, "dca_trigger": dca_trigger, "dca_max_count": dca_max_count,
    "order_usdt": st.session_state.get('order_usdt', 100.0)
}
if current_settings != config: save_settings(current_settings)

# 👇 [지능형 자동매매 로직]
t1, t2 = st.tabs(["🤖 자동매매", "⚡ 수동주문"])
with t1:
    auto_on = st.checkbox("자동매매 활성화", value=config['auto_trade'], key="auto_trade")
    if auto_on:
        if not active_pos:
            if final_long: execute_trade('long', reason=",".join(reasons_L))
            elif final_short: execute_trade('short', reason=",".join(reasons_S))
        else:
            cur_side = active_pos['side']
            roi = float(active_pos['percentage'])
            initial_margin = float(active_pos['initialMargin'])
            current_margin = float(active_pos['margin']) # 현재 잡힌 증거금
            
            # 1. 익절
            if roi >= tp_pct: execute_trade(cur_side, True, "목표 달성")
            
            # 2. 추매 (물타기) 로직
            # 조건: 사용자가 켰고, ROI가 트리거(예:-5%) 도달했고, 현재 마진이 초기마진 * (1 + 최대횟수) 보다 작을 때
            elif use_dca and roi <= dca_trigger and current_margin < (initial_margin * (1 + dca_max_count)):
                # 추매 수량: 최초 진입금액만큼 (100% 비율)
                # 현재 비트겟 API에서 정확한 수량 계산을 위해 단순화:
                # 현재 보유 수량만큼 더 삼 (1배수 물타기)
                add_qty = float(active_pos['contracts'])
                execute_trade(cur_side, False, f"💧 추매 (ROI {roi:.2f}%)", qty=add_qty)
                time.sleep(2) # 중복 방지

            # 3. 손절 & 스위칭 로직
            elif roi <= -sl_pct:
                if use_switching and ((cur_side == 'long' and short_score >= target_vote) or \
                   (cur_side == 'short' and long_score >= target_vote)):
                    execute_trade(cur_side, True, "🚨 손절 후 스위칭")
                    time.sleep(1)
                    target_side = 'short' if cur_side == 'long' else 'long'
                    execute_trade(target_side, reason="스위칭 진입")
                else:
                    execute_trade(cur_side, True, "손절 제한")

        time.sleep(3); safe_rerun()

with t2:
    c1, c2, c3, c4 = st.columns(4)
    def set_amt(pct): st.session_state['order_usdt'] = float(f"{free * pct:.2f}")
    if c1.button("20%"): set_amt(0.2)
    if c2.button("50%"): set_amt(0.5)
    if c3.button("80%"): set_amt(0.8)
    if c4.button("Full"): set_amt(1.0)
    
    st.number_input("금액 (USDT)", 0.0, free, key='order_usdt')
    b1, b2 = st.columns(2)
    if b1.button("롱 진입", use_container_width=True): execute_trade('long', reason="수동")
    if b2.button("숏 진입", use_container_width=True): execute_trade('short', reason="수동")
    if st.button("포지션 청산", use_container_width=True):
        if active_pos: execute_trade(active_pos['side'], True, "수동청산")
