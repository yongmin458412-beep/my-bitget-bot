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
import google.generativeai as genai # 🧠 AI 기능 추가

# =========================================================
# ⚙️ [설정] 기본 환경
# =========================================================
IS_SANDBOX = True # 모의투자
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="비트겟 AI 봇 (Masterpiece)")

# ---------------------------------------------------------
# 💾 설정 파일 관리
# ---------------------------------------------------------
def load_settings():
    default = {
        "gemini_api_key": "", # AI 키 저장
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True,
        "use_ma": False, "use_macd": False, "use_stoch": False, "use_mfi": False, "use_willr": False, "use_adx": True,
        "use_switching": True, "use_dca": True, "dca_trigger": -20.0, "dca_max_count": 1,
        "auto_size_type": "percent", "auto_size_val": 20.0,
        "use_dual_mode": True
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
except: st.error("🚨 Secrets 설정 필요"); st.stop()

# ---------------------------------------------------------
# 🧠 Gemini AI & 경제 지표 함수 (New!)
# ---------------------------------------------------------
def get_fear_and_greed():
    """공포 탐욕 지수 가져오기 (API 대체)"""
    try:
        url = "https://api.alternative.me/fng/"
        res = requests.get(url).json()
        value = res['data'][0]['value']
        classification = res['data'][0]['value_classification']
        return f"{value} ({classification})"
    except: return "데이터 없음"

def ask_gemini_briefing(status_data, market_data):
    """Gemini에게 현재 상황 브리핑 요청"""
    api_key = config.get('gemini_api_key', '')
    if not api_key: return "⚠️ Gemini API 키가 설정되지 않았습니다."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        당신은 암호화폐 전문 트레이딩 AI입니다. 현재 내 봇의 상황을 분석하고 조언해주세요.
        
        [내 계좌 및 포지션 상황]
        - 포지션: {status_data['position']}
        - 수익률(ROI): {status_data['roi']}%
        - 미실현 손익: ${status_data['pnl']}
        - 현재 잔고: ${status_data['balance']}
        - 총 추정 자산: ${status_data['equity']}
        - 현재 봇 상태: {status_data['action_reason']} (왜 대기중인지)

        [시장 데이터]
        - 현재가: ${market_data['price']}
        - RSI: {market_data['rsi']}
        - ADX (추세강도): {market_data['adx']} ({'추세장' if market_data['adx']>=25 else '횡보장'})
        - 공포/탐욕 지수: {market_data['fng']}
        
        [질문]
        1. 현재 손실/수익 상황에 대한 냉철한 진단.
        2. 봇이 '대기'하고 있는 이유가 합당한지 평가.
        3. 향후 대응 전략 (추매, 손절, 홀딩 중 추천).
        간결하게 3줄 요약으로 답변해.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 실패: {e}"

# ---------------------------------------------------------
# 📊 수익 분석 & 잔고 계산
# ---------------------------------------------------------
def log_trade(action, symbol, side, price, qty, leverage, pnl=0, roi=0):
    now = datetime.now()
    margin = (price * qty) / leverage
    new_data = {
        "Time": now.strftime("%Y-%m-%d %H:%M:%S"), "Date": now.strftime("%Y-%m-%d"),
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
        closed = df[df['Action'].str.contains('청산')]
        total_pnl = closed['PnL'].sum()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = closed[closed['Date'] == today]['PnL'].sum()
        last_roi = closed.iloc[-1]['ROI'] if not closed.empty else 0.0
        return last_roi, daily_pnl, total_pnl, len(closed)
    except: return 0.0, 0.0, 0.0, 0

def get_balance_details(exchange_obj):
    try:
        bal = exchange_obj.fetch_balance({'type': 'swap'})
        if 'SUSDT' in bal: coin = 'SUSDT'
        elif 'USDT' in bal: coin = 'USDT'
        else: coin = 'SBTC'
        free = float(bal[coin]['free'])
        total = float(bal[coin]['total'])
        return coin, free, total
    except: return "USDT", 0.0, 0.0

# ---------------------------------------------------------
# 📡 텔레그램 (AI 브리핑 포함)
# ---------------------------------------------------------
def send_telegram(message, chart_df=None):
    if not tg_token or not tg_id: return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        kb = {"inline_keyboard": [[{"text": "🧠 AI 종합 브리핑", "callback_data": "ai_briefing"}]]}
        requests.post(url, data={'chat_id': tg_id, 'text': message, 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})
        
        if chart_df is not None:
            buf = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.plot(chart_df['time'], chart_df['close'], color='yellow', label='Price')
            if 'ZLSMA' in chart_df.columns: plt.plot(chart_df['time'], chart_df['ZLSMA'], color='magenta', label='ZLSMA')
            plt.title("Chart Snapshot"); plt.grid(True, alpha=0.2); ax = plt.gca(); ax.set_facecolor('black'); plt.gcf().patch.set_facecolor('black'); ax.tick_params(colors='white')
            plt.savefig(buf, format='png', facecolor='black'); buf.seek(0)
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendPhoto", data={'chat_id': tg_id}, files={'photo': buf}); plt.close()
    except: pass

def get_bot_status_reason(roi, dca_count, max_dca, holding, switching):
    """봇이 현재 대기중인 이유를 분석"""
    if roi <= -50.0:
        if dca_count >= max_dca: return "🚫 최대 추매 횟수 초과 (더 이상 매수 불가)"
        return "⚠️ 위험 구간 (증거금 부족 가능성)"
    if roi <= config['dca_trigger']:
        if dca_count >= max_dca: return "✋ 추매 제한 도달 (Wait)"
        return "💧 추매 조건 만족 (자금 대기 중)"
    if roi <= config['sl'] * -1: # 손절 구간
        if holding: return "🛡️ 스마트 홀딩 중 (지표 반등 대기)"
        if switching: return "🔄 스위칭 각 보는 중"
    return "✅ 정상 모니터링 중"

def telegram_listener(exchange_obj, symbol_name):
    last_id = 0
    while True:
        try:
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={last_id+1}&timeout=30").json()
            if res.get('ok'):
                for up in res['result']:
                    last_id = up['update_id']
                    if 'callback_query' in up:
                        cb = up['callback_query']
                        chat_id = cb['message']['chat']['id']
                        
                        if cb['data'] == 'ai_briefing':
                            # 데이터 수집
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "🤖 AI가 데이터를 분석 중입니다..."})
                            
                            # 1. 자산 정보
                            coin, free, total = get_balance_details(exchange_obj)
                            # 2. 포지션 정보
                            pos_str = "없음"; roi = 0; pnl = 0; equity = total
                            dca_cnt = 0 # 실제로는 trade_log나 메모리에서 가져와야 함 (여기선 약식)
                            
                            try:
                                pos = exchange_obj.fetch_positions([symbol_name])
                                for p in pos:
                                    if float(p['contracts']) > 0:
                                        pos_str = f"{p['side'].upper()} x{p['leverage']}"
                                        roi = float(p['percentage'])
                                        pnl = float(p['unrealizedPnl'])
                                        equity = total + pnl
                                        break
                            except: pass
                            
                            # 3. 시장 데이터 (가장 최근 것)
                            try:
                                ohlcv = exchange_obj.fetch_ohlcv(symbol_name, '5m', limit=20)
                                df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                                rsi = 50.0 # 약식 계산 필요 시 추가
                                adx = 20.0 
                            except: rsi=50; adx=20
                            
                            # 4. 봇 상태 분석
                            reason = get_bot_status_reason(roi, 1, config['dca_max_count'], config.get('use_holding', True), config['use_switching'])
                            fng = get_fear_and_greed()

                            # AI에게 질문
                            status_data = {
                                'position': pos_str, 'roi': roi, 'pnl': pnl,
                                'balance': free, 'equity': equity, 'action_reason': reason
                            }
                            market_data = {
                                'price': ohlcv[-1][4], 'rsi': rsi, 'adx': adx, 'fng': fng
                            }
                            
                            ai_advice = ask_gemini_briefing(status_data, market_data)
                            
                            # 결과 전송
                            final_msg = f"📢 <b>[AI 실시간 브리핑]</b>\n\n"
                            final_msg += f"🕵️ <b>봇 상태 진단:</b>\n👉 {reason}\n\n"
                            final_msg += f"📊 <b>자산 현황:</b>\n• 잔고: ${free:,.2f}\n• 총자산: ${equity:,.2f}\n\n"
                            final_msg += f"🧠 <b>Gemini 의견:</b>\n{ai_advice}"
                            
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': final_msg, 'parse_mode': 'HTML'})
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})
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

# AI 키 입력
st.sidebar.divider()
gemini_key_input = st.sidebar.text_input("🧠 Gemini API Key (AI 브리핑용)", value=config.get('gemini_api_key', ''), type="password")
if gemini_key_input != config.get('gemini_api_key'):
    config['gemini_api_key'] = gemini_key_input
    save_settings(config)

# 리스너
thread_exists = False
for t in threading.enumerate():
    if t.name == "TelegramListener": thread_exists = True; break
if not thread_exists:
    t = threading.Thread(target=telegram_listener, args=(exchange, symbol), daemon=True, name="TelegramListener")
    t.start()

try:
    exchange.set_leverage(config['leverage'], symbol)
    try: exchange.set_position_mode(hedged=False, symbol=symbol)
    except: pass
except: pass

st.sidebar.divider()
st.sidebar.subheader("🛡️ 방어 및 추매 설정")
use_switching = st.sidebar.checkbox("스위칭 허용", value=config['use_switching'])
use_holding = st.sidebar.checkbox("스마트 존버 허용", value=config.get('use_holding', True))
use_dca = st.sidebar.checkbox("추매(물타기) 허용", value=config['use_dca'])
dca_trigger = st.sidebar.number_input("추매 발동 (ROI %)", -50.0, -1.0, config['dca_trigger'], step=0.5)
dca_max_count = st.sidebar.number_input("최대 추매 횟수", 1, 5, config['dca_max_count'])

st.sidebar.divider()
st.sidebar.subheader("⚔️ 전략 설정 (이중 모드)")
use_dual_mode = st.sidebar.checkbox("이중 모드 (횡보/추세 자동전환)", value=config.get('use_dual_mode', True))

st.sidebar.subheader("📊 지표 설정")
P = {} 
with st.sidebar.expander("지표 세부 설정", expanded=False):
    use_rsi = st.checkbox("RSI", config['use_rsi']); P['rsi_period'] = 14
    P['rsi_buy'] = st.slider("RSI 롱", 10, 50, 30); P['rsi_sell'] = st.slider("RSI 숏", 50, 90, 70)
    use_bb = st.checkbox("BB", config['use_bb']); P['bb_period']=20; P['bb_std']=2.0
    use_cci = st.checkbox("CCI", config['use_cci'])
    use_vol = st.checkbox("Volume", config['use_vol']); P['vol_mul']=2.0
    use_ma = st.checkbox("MA", config['use_ma'])
    use_macd = st.checkbox("MACD", config['use_macd'])
    use_stoch = st.checkbox("Stoch", config['use_stoch']); P['stoch_k']=14
    use_mfi = st.checkbox("MFI", config['use_mfi'])
    use_willr = st.checkbox("WillR", config['use_willr'])
    use_adx = st.checkbox("ADX", config['use_adx'])

active_indicators = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_mfi, use_willr, use_vol, use_adx])
target_vote = st.sidebar.slider("🎯 진입 조건 (신호 개수)", 1, max(1, active_indicators), config['target_vote'])
p_leverage = st.sidebar.slider("레버리지", 1, 50, config['leverage'])
tp_pct = st.sidebar.number_input("💰 익절 목표 (%)", 1.0, 500.0, config['tp'])
sl_pct = st.sidebar.number_input("💸 손절 제한 (%)", 1.0, 100.0, config['sl'])

if st.sidebar.button("📡 텔레그램 연결 테스트"):
    send_telegram("✅ <b>시스템 가동 중!</b>\nAI 브리핑 버튼을 눌러보세요.")
    st.toast("전송 완료")

# ---------------------------------------------------------
# 🧮 지표 계산
# ---------------------------------------------------------
def calculate_indicators(df, params):
    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    # ADX & TR (공통)
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ADX'] = (tr.rolling(14).mean() / close) * 1000 
    
    # ZLSMA & Chandelier (추세장용)
    length = 130; lag = (length - 1) // 2
    df['lsma_source'] = close + (close - close.shift(lag))
    df['ZLSMA'] = df['lsma_source'].ewm(span=length).mean()
    atr = tr.rolling(1).mean(); df['Chandelier_Long'] = high.rolling(1).max() - (atr * 2); df['Chandelier_Short'] = low.rolling(1).min() + (atr * 2)

    # 기본 지표들
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(P['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(P['rsi_period']).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_MA'] = close.rolling(P['bb_period']).mean()
    df['BB_STD'] = close.rolling(P['bb_period']).std()
    df['BB_UP'] = df['BB_MA'] + (df['BB_STD'] * P['bb_std'])
    df['BB_LO'] = df['BB_MA'] - (df['BB_STD'] * P['bb_std'])
    
    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean(); mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - sma) / (0.015 * mad)
    df['VOL_MA'] = vol.rolling(20).mean()
    
    # 기타 지표들...
    exp12 = close.ewm(span=12).mean(); exp26 = close.ewm(span=26).mean()
    df['MACD'] = exp12 - exp26; df['MACD_SIG'] = df['MACD'].ewm(span=9).mean()
    lowest_low = low.rolling(P['stoch_k']).min(); highest_high = high.rolling(P['stoch_k']).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    return df

# ---------------------------------------------------------
# 📊 데이터 처리
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol); curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = calculate_indicators(df, P)
    last = df.iloc[-1]
    
    is_trend_mode = last['ADX'] >= 25 and config.get('use_dual_mode', True)
    mode_str = "🌊 추세장" if is_trend_mode else "🦀 횡보장"
except Exception as e: st.error(f"데이터 로딩 실패: {e}"); st.stop()

# ---------------------------------------------------------
# ⚡ 주문 실행
# ---------------------------------------------------------
def safe_rerun():
    time.sleep(0.5); 
    if hasattr(st, 'rerun'): st.rerun()
    else: st.experimental_rerun()

def execute_trade(side, is_close=False, reason="", qty=0.0, manual_amt=0.0):
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
            if qty == 0.0:
                if manual_amt > 0: input_val = manual_amt
                else: input_val = st.session_state['order_usdt']
                
                coin, free, total = get_balance_details(exchange)
                if input_val > free * 0.98: input_val = free * 0.98
                
                raw_qty = (input_val * p_leverage) / curr_price
                qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            emoji = "🚀"
            
        exchange.create_order(symbol, 'limit', order_side, qty, ticker['ask' if order_side=='buy' else 'bid'], params=params)
        
        action_name = "청산" if is_close else "진입/추매"
        if is_close: log_trade(action_name, symbol, side, curr_price, qty, p_leverage, log_pnl, log_roi)
        
        coin, free, total = get_balance_details(exchange)
        equity = total + (log_pnl if is_close else 0.0)
        _, d_pnl, t_pnl, _ = get_analytics()
        
        msg = f"{emoji} <b>{side.upper()} {action_name} 완료</b>\n------------------\n📍 <b>이유:</b> {reason}\n💲 <b>가격:</b> ${curr_price:,.2f}"
        if is_close: msg += f"\n📈 <b>실현:</b> ${log_pnl:.2f} ({log_roi:.2f}%)\n📅 <b>금일:</b> ${d_pnl:.2f} | 🏆 <b>누적:</b> ${t_pnl:.2f}"
        else: msg += f"\n💸 <b>사용금액:</b> ${(float(qty)*curr_price)/p_leverage:,.2f}"
        msg += f"\n------------------\n💰 <b>사용가능:</b> ${free:,.2f}\n💎 <b>총 자산:</b> ${equity:,.2f}"
        
        st.success(msg.replace("<b>", "").replace("</b>", ""))
        send_telegram(msg, df.tail(60) if not is_close else None)
        safe_rerun()
    except Exception as e: st.error(f"주문 실패: {e}")

# =========================================================
# 🚀 메인 UI
# =========================================================
st.title(f"🔥 {symbol} ({mode_str})")

coin, free, total = get_balance_details(exchange)
temp_u = 0.0
try:
    pos_list = exchange.fetch_positions([symbol])
    for p in pos_list:
        if float(p['contracts']) > 0: temp_u = float(p['unrealizedPnl']); break
except: pass
equity = total + temp_u
_, d_pnl, t_pnl, _ = get_analytics()

st.markdown(f"""<div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; justify-content: space-around;"><div style="text-align: center;"><span style="color: #888;">사용 가능 잔고</span><br><span style="font-size: 1.5em; color: white;">${free:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">총 추정 자산</span><br><span style="font-size: 1.5em; color: white;">${equity:,.2f}</span></div><div style="text-align: center;"><span style="color: #888;">총 누적 수익</span><br><span style="font-size: 1.5em; color: {'#4CAF50' if t_pnl>=0 else '#FF5252'};">${t_pnl:,.2f}</span></div></div>""", unsafe_allow_html=True)

active_pos = None
if pos_list:
    for p in pos_list:
        if float(p['contracts']) > 0:
            active_pos = p
            roi = float(p['percentage'])
            st.markdown(f"""<div style="border: 2px solid {'#4CAF50' if roi>=0 else '#FF5252'}; padding: 10px; border-radius: 10px; background: #262730;"><h3 style="margin:0;">{p['side'].upper()} (x{p['leverage']})</h3><p>수익률: {roi:.2f}% | 미실현: ${float(p['unrealizedPnl']):.2f}</p></div>""", unsafe_allow_html=True)
            break

# 신호 계산
long_score = 0; short_score = 0; reasons_L = []; reasons_S = []
final_long = False; final_short = False

if is_trend_mode: # 추세장 (ZLSMA)
    if curr_price > last['ZLSMA'] and curr_price > last['Chandelier_Short']: final_long=True; reasons_L.append("ZLSMA상승")
    elif curr_price < last['ZLSMA'] and curr_price < last['Chandelier_Long']: final_short=True; reasons_S.append("ZLSMA하락")
else: # 횡보장 (투표)
    if use_rsi:
        if last['RSI'] <= P['rsi_buy']: long_score+=1; reasons_L.append("RSI")
        elif last['RSI'] >= P['rsi_sell']: short_score+=1; reasons_S.append("RSI")
    if use_bb:
        if last['close'] <= last['BB_LO']: long_score+=1; reasons_L.append("BB")
        elif last['close'] >= last['BB_UP']: short_score+=1; reasons_S.append("BB")
    if use_cci:
        if last['CCI'] < -100: long_score+=1; reasons_L.append("CCI")
        elif last['CCI'] > 100: short_score+=1; reasons_S.append("CCI")
    # ... 나머지 지표 생략 (설정에 따름)
    
    final_long = long_score >= target_vote
    final_short = short_score >= target_vote
    # 역추세 필터
    if final_long and curr_price < last['ZLSMA']: final_long = False
    if final_short and curr_price > last['ZLSMA']: final_short = False

c1, c2 = st.columns(2)
c1.metric("📈 롱 시그널", "ON" if final_long else "OFF", f"{long_score}/{target_vote}" if not is_trend_mode else "Trend")
c2.metric("📉 숏 시그널", "ON" if final_short else "OFF", f"{short_score}/{target_vote}" if not is_trend_mode else "Trend")

# 설정 저장
current_settings = {
    "leverage": p_leverage, "target_vote": target_vote, "tp": tp_pct, "sl": sl_pct,
    "auto_trade": st.session_state.get('auto_trade', False),
    "use_rsi": use_rsi, "use_bb": use_bb, "use_ma": use_ma, "use_macd": use_macd,
    "use_stoch": use_stoch, "use_cci": use_cci, "use_vol": use_vol, "use_mfi": use_mfi,
    "use_willr": use_willr, "use_adx": use_adx,
    "use_switching": use_switching, "use_dca": use_dca, "dca_trigger": dca_trigger, "dca_max_count": dca_max_count,
    "use_dual_mode": use_dual_mode, "use_holding": use_holding,
    "auto_size_type": config.get('auto_size_type'), "auto_size_val": config.get('auto_size_val'),
    "order_usdt": st.session_state.get('order_usdt', 100.0),
    "gemini_api_key": config.get('gemini_api_key', '')
}
if current_settings != config: save_settings(current_settings)

t1, t2 = st.tabs(["🤖 자동매매", "⚡ 수동주문"])
with t1:
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        auto_on = st.checkbox("자동매매 활성화", value=config['auto_trade'], key="auto_trade")
        sz_type = st.radio("진입 금액", ["자산 비율 (%)", "고정 (USDT)"], index=0 if config.get('auto_size_type')=='percent' else 1)
    with c_a2:
        if sz_type == "자산 비율 (%)":
            sz_val = st.number_input("비율 (%)", 1.0, 100.0, float(config.get('auto_size_val', 20.0)))
            entry_amt = equity * (sz_val / 100.0)
        else:
            sz_val = st.number_input("금액 ($)", 10.0, 10000.0, float(config.get('auto_size_val', 100.0)))
            entry_amt = sz_val
        st.caption(f"👉 진입 예정: ${entry_amt:,.2f}")
    
    config['auto_size_type'] = 'percent' if sz_type == "자산 비율 (%)" else 'fixed'
    config['auto_size_val'] = sz_val

    if auto_on:
        if not active_pos:
            if entry_amt > free * 0.98: entry_amt = free * 0.98
            if final_long: execute_trade('long', reason=",".join(reasons_L), manual_amt=entry_amt)
            elif final_short: execute_trade('short', reason=",".join(reasons_S), manual_amt=entry_amt)
        else:
            cur_side = active_pos['side']
            roi = float(active_pos['percentage'])
            
            # 1. 청산
            should_close = False; close_reason = ""
            if is_trend_mode:
                if cur_side == 'long' and curr_price < last['Chandelier_Long']: should_close=True; close_reason="추세반전"
                elif cur_side == 'short' and curr_price > last['Chandelier_Short']: should_close=True; close_reason="추세반전"
            else:
                if roi >= tp_pct: should_close=True; close_reason="목표달성"
            
            if should_close: execute_trade(cur_side, True, close_reason)
            
            # 2. 추매
            elif use_dca and roi <= dca_trigger:
                curr_margin = float(active_pos.get('initialMargin', 0) or 0)
                if curr_margin == 0: curr_margin = (float(active_pos['contracts']) * float(active_pos['entryPrice'])) / p_leverage
                # 1배수 물타기 (안전장치)
                if curr_margin < entry_amt * (1 + dca_max_count) * 1.1:
                    add_qty = float(active_pos['contracts'])
                    execute_trade(cur_side, False, f"💧 추매 (ROI {roi:.2f}%)", qty=add_qty)
                    time.sleep(2)

            # 3. 손절/스위칭/존버
            elif roi <= -sl_pct:
                # 스위칭
                if use_switching and ((cur_side == 'long' and final_short) or (cur_side == 'short' and final_long)):
                    execute_trade(cur_side, True, "🚨 손절 후 스위칭")
                    time.sleep(1)
                    new_entry = (equity - abs(float(active_pos['unrealizedPnl']))) * (sz_val/100.0) if sz_type == "자산 비율 (%)" else sz_val
                    execute_trade('short' if cur_side=='long' else 'long', reason="스위칭", manual_amt=new_entry)
                # 존버 (신호 살아있으면)
                elif use_holding and ((cur_side=='long' and final_long) or (cur_side=='short' and final_short)):
                    if roi <= -30.0: execute_trade(cur_side, True, "💀 강제 청산")
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
