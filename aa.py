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
import google.generativeai as genai

# =========================================================
# ⚙️ [시스템 설정] 기본 환경 및 파일 관리
# =========================================================
IS_SANDBOX = True 
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"

st.set_page_config(layout="wide", page_title="비트겟 AI 퀀트 (Masterpiece)")

def load_settings():
    """사용자의 모든 설정을 파일에서 불러옵니다."""
    default = {
        "gemini_api_key": "",
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        
        # [지표 세부 파라미터]
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14, "vol_mul": 2.0,
        
        # [지표 활성화 여부]
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True,
        "use_ma": False, "use_macd": False, "use_stoch": False, 
        "use_mfi": False, "use_willr": False, "use_adx": True,
        
        # [스마트 방어 & 추매]
        "use_switching": True, "use_dca": True, "dca_trigger": -20.0, "dca_max_count": 1,
        "use_holding": True,
        
        # [자동매매 자금 관리 & 전략]
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
# 🔐 API 키 로딩
# ---------------------------------------------------------
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN", "")
tg_id = st.secrets.get("TG_CHAT_ID", "")
gemini_key = st.secrets.get("GEMINI_API_KEY", config.get("gemini_api_key", ""))

if not api_key: 
    st.error("🚨 비트겟 API 키가 Secrets에 설정되지 않았습니다.")
    st.stop()

# ---------------------------------------------------------
# 🧠 AI 기능 (오토파일럿 & 브리핑) - 모델명 수정됨!
# ---------------------------------------------------------
def run_autopilot(df, current_config):
    if not gemini_key: return "⚠️ Gemini API 키가 필요합니다."
    
    try:
        genai.configure(api_key=gemini_key)
        # 👇 [수정됨] gemini-pro -> gemini-1.5-flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        last = df.iloc[-1]
        summary = f"RSI:{last['RSI']:.1f}, ATR:{last['ATR']:.4f}, ADX:{last['ADX']:.1f}, Price:{last['close']}"
        
        prompt = f"""
        Act as a professional crypto quant trader. Analyze this market data: {summary}
        Current Trend Strength (ADX): {last['ADX']} (If > 25 Trend, else Range)
        
        Recommend optimal JSON settings for a trading bot:
        {{
            "rsi_buy": int (lower for trend, higher for range),
            "rsi_sell": int,
            "tp": float (take profit %),
            "sl": float (stop loss %),
            "leverage": int (safe leverage),
            "reason": "Explain why in Korean"
        }}
        Return ONLY valid JSON.
        """
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        res_json = json.loads(text)
        
        current_config['rsi_buy'] = res_json['rsi_buy']
        current_config['rsi_sell'] = res_json['rsi_sell']
        current_config['tp'] = res_json['tp']
        current_config['sl'] = res_json['sl']
        current_config['leverage'] = res_json['leverage']
        save_settings(current_config)
        
        return f"✅ **AI 최적화 완료**\n\n이유: {res_json['reason']}\n변경: RSI {res_json['rsi_buy']}/{res_json['rsi_sell']}, 레버리지 x{res_json['leverage']}"
    except Exception as e: return f"AI 오류: {e}"

def ask_gemini_briefing(status_txt, market_txt):
    if not gemini_key: return "API 키 없음"
    try:
        genai.configure(api_key=gemini_key)
        # 👇 [수정됨] 모델명 변경
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"상황: {status_txt}\n시장: {market_txt}\n트레이더로서 현재 포지션 대응 전략과 시장 전망을 3줄로 브리핑해줘."
        return model.generate_content(prompt).text
    except: return "AI 응답 지연"

# ---------------------------------------------------------
# 📡 데이터 유틸리티 & 텔레그램
# ---------------------------------------------------------
def get_balance_details(exchange_obj):
    try:
        bal = exchange_obj.fetch_balance({'type': 'swap'})
        coin = 'SUSDT' if 'SUSDT' in bal else ('USDT' if 'USDT' in bal else 'SBTC')
        return coin, float(bal[coin]['free']), float(bal[coin]['total'])
    except: return "USDT", 0.0, 0.0

def get_analytics():
    if not os.path.exists(LOG_FILE): return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return 0.0, 0.0, 0.0
        closed = df[df['Action'].str.contains('청산')]
        total_pnl = closed['PnL'].sum()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = closed[closed['Date'] == today]['PnL'].sum()
        last_roi = closed.iloc[-1]['ROI'] if not closed.empty else 0.0
        return last_roi, daily_pnl, total_pnl
    except: return 0.0, 0.0, 0.0

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

def send_telegram(message, chart_df=None):
    if not tg_token or not tg_id: return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        kb = {"inline_keyboard": [[{"text": "🧠 AI 브리핑 요청", "callback_data": "ai_briefing"}]]}
        requests.post(url, data={'chat_id': tg_id, 'text': message, 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})
        
        if chart_df is not None:
            buf = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.plot(chart_df['time'], chart_df['close'], color='yellow', label='Price')
            if 'ZLSMA' in chart_df.columns: plt.plot(chart_df['time'], chart_df['ZLSMA'], color='magenta', alpha=0.7, label='ZLSMA')
            plt.title("Chart Snapshot"); plt.grid(True, alpha=0.2); ax = plt.gca(); ax.set_facecolor('black'); plt.gcf().patch.set_facecolor('black'); ax.tick_params(colors='white')
            plt.savefig(buf, format='png', facecolor='black'); buf.seek(0)
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendPhoto", data={'chat_id': tg_id}, files={'photo': buf}); plt.close()
    except: pass

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
                        if cb['data'] == 'ai_briefing':
                            coin, free, total = get_balance_details(exchange_obj)
                            pos_txt = "없음"; u_pnl = 0
                            try:
                                positions = exchange_obj.fetch_positions([symbol_name])
                                for p in positions:
                                    if float(p['contracts']) > 0:
                                        pos_txt = f"{p['side']} x{p['leverage']} ({float(p['percentage']):.2f}%)"
                                        u_pnl = float(p['unrealizedPnl'])
                            except: pass
                            
                            ai_msg = "API 키가 없습니다."
                            if gemini_key:
                                try:
                                    genai.configure(api_key=gemini_key)
                                    # 👇 [수정됨] 모델명 변경
                                    model = genai.GenerativeModel('gemini-1.5-flash')
                                    prompt = f"현재잔고: {free}, 포지션: {pos_txt}, 미실현손익: {u_pnl}. 트레이더로서 현재 상황을 분석하고 대응 전략을 3줄로 요약해줘."
                                    ai_msg = model.generate_content(prompt).text
                                except: ai_msg = "AI 분석 실패 (API 오류)"
                            
                            equity = total + u_pnl
                            msg = f"🧠 <b>AI 브리핑</b>\n\n{ai_msg}\n\n💰 <b>잔고:</b> ${free:,.2f}\n💎 <b>총자산:</b> ${equity:,.2f}"
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cb['message']['chat']['id'], 'text': msg, 'parse_mode': 'HTML'})
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})
            time.sleep(1)
        except: time.sleep(5)

# ---------------------------------------------------------
# 📡 거래소 초기화
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
# 🎨 사이드바 (모든 설정 풀버전)
# ---------------------------------------------------------
st.sidebar.title("🛠️ 봇 제어판 (Full)")
is_mobile = st.sidebar.checkbox("📱 모바일 뷰", value=True)

markets = exchange.markets
symbols = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
symbol = st.sidebar.selectbox("코인 선택", symbols, index=0)

# Gemini Key 입력창
if not st.secrets.get("GEMINI_API_KEY"):
    g_key = st.sidebar.text_input("🧠 Gemini Key", value=config.get('gemini_api_key',''), type="password")
    if g_key != config.get('gemini_api_key'):
        config['gemini_api_key'] = g_key
        save_settings(config)
        st.rerun()

# 텔레그램 리스너
found = False
for t in threading.enumerate():
    if t.name == "TelegramListener": found = True; break
if not found:
    t = threading.Thread(target=telegram_listener, args=(exchange, symbol), daemon=True, name="TelegramListener")
    t.start()

try:
    exchange.set_leverage(config['leverage'], symbol)
    try: exchange.set_position_mode(hedged=False, symbol=symbol)
    except: pass
except: pass

st.sidebar.divider()
st.sidebar.subheader("🛡️ 스마트 방어 & 추매")
use_switching = st.sidebar.checkbox("스위칭 (Switching)", value=config['use_switching'])
use_holding = st.sidebar.checkbox("스마트 존버 (Holding)", value=config.get('use_holding', True))
use_dca = st.sidebar.checkbox("추매 (DCA) 사용", value=config['use_dca'])
c_dca1, c_dca2 = st.sidebar.columns(2)
dca_trigger = c_dca1.number_input("추매 발동 ROI(%)", -90.0, -1.0, float(config['dca_trigger']), step=0.5)
dca_max_count = c_dca2.number_input("최대 추매 횟수", 1, 10, int(config['dca_max_count']))
use_dual_mode = st.sidebar.checkbox("⚔️ 이중 모드 (횡보/추세 자동)", value=config.get('use_dual_mode', True))

st.sidebar.divider()
st.sidebar.subheader("📊 지표 세부 설정 (10개)")

with st.sidebar.expander("1. RSI & 2. BB", expanded=False):
    use_rsi = st.checkbox("RSI 사용", config['use_rsi'])
    c_r1, c_r2, c_r3 = st.columns(3)
    config['rsi_period'] = c_r1.number_input("RSI 기간", 5, 50, config['rsi_period'])
    config['rsi_buy'] = c_r2.number_input("매수 선", 10, 50, config['rsi_buy'])
    config['rsi_sell'] = c_r3.number_input("매도 선", 50, 90, config['rsi_sell'])
    use_bb = st.checkbox("BB 사용", config['use_bb'])
    c_b1, c_b2 = st.columns(2)
    config['bb_period'] = c_b1.number_input("BB 기간", 5, 50, config['bb_period'])
    config['bb_std'] = c_b2.number_input("BB 승수", 1.0, 3.0, config['bb_std'])

with st.sidebar.expander("3. 추세 (MA, MACD, ADX)", expanded=False):
    use_ma = st.checkbox("이동평균선(MA)", config['use_ma'])
    c_m1, c_m2 = st.columns(2)
    config['ma_fast'] = c_m1.number_input("단기", 3, 50, config['ma_fast'])
    config['ma_slow'] = c_m2.number_input("장기", 50, 200, config['ma_slow'])
    use_macd = st.checkbox("MACD", config['use_macd'])
    use_adx = st.checkbox("ADX (추세강도)", config['use_adx'])

with st.sidebar.expander("4. 오실레이터 (Stoch, CCI, MFI...)", expanded=False):
    use_stoch = st.checkbox("Stoch", config['use_stoch'])
    config['stoch_k'] = st.number_input("Stoch K", 5, 50, config['stoch_k'])
    use_cci = st.checkbox("CCI", config['use_cci'])
    use_mfi = st.checkbox("MFI", config['use_mfi'])
    use_willr = st.checkbox("WillR", config['use_willr'])
    use_vol = st.checkbox("Volume", config['use_vol'])
    config['vol_mul'] = st.number_input("거래량 배수", 1.0, 10.0, config['vol_mul'])

st.sidebar.divider()
active_indicators = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_mfi, use_willr, use_vol, use_adx])
target_vote = st.sidebar.slider("🎯 진입 신호 강도", 1, max(1, active_indicators), config['target_vote'])
p_leverage = st.sidebar.slider("레버리지", 1, 50, config['leverage'])
tp_pct = st.sidebar.number_input("💰 익절 (%)", 1.0, 500.0, float(config['tp']))
sl_pct = st.sidebar.number_input("💸 손절 (%)", 1.0, 100.0, float(config['sl']))

new_conf = config.copy()
new_conf.update({
    'leverage': p_leverage, 'target_vote': target_vote, 'tp': tp_pct, 'sl': sl_pct,
    'use_rsi': use_rsi, 'use_bb': use_bb, 'use_ma': use_ma, 'use_macd': use_macd,
    'use_stoch': use_stoch, 'use_cci': use_cci, 'use_mfi': use_mfi, 'use_willr': use_willr, 'use_vol': use_vol, 'use_adx': use_adx,
    'use_switching': use_switching, 'use_dca': use_dca, 'dca_trigger': dca_trigger, 'dca_max_count': dca_max_count,
    'use_holding': use_holding, 'use_dual_mode': use_dual_mode
})
if new_conf != config:
    save_settings(new_conf)
    config = new_conf

if st.sidebar.button("📡 텔레그램 테스트"):
    send_telegram("✅ <b>시스템 정상!</b>")
    st.toast("전송 완료")

# ---------------------------------------------------------
# 🧮 지표 계산 함수
# ---------------------------------------------------------
def calculate_indicators(df):
    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    df['ADX'] = (df['ATR'] / close) * 1000 
    
    length = 130; lag = (length - 1) // 2
    df['lsma_source'] = close + (close - close.shift(lag))
    df['ZLSMA'] = df['lsma_source'].ewm(span=length).mean()
    df['Chandelier_Long'] = high.rolling(1).max() - (df['ATR'] * 2)
    df['Chandelier_Short'] = low.rolling(1).min() + (df['ATR'] * 2)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(config['rsi_period']).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_MA'] = close.rolling(config['bb_period']).mean()
    df['BB_STD'] = close.rolling(config['bb_period']).std()
    df['BB_UP'] = df['BB_MA'] + (df['BB_STD'] * config['bb_std'])
    df['BB_LO'] = df['BB_MA'] - (df['BB_STD'] * config['bb_std'])
    
    df['MA_FAST'] = close.rolling(config['ma_fast']).mean()
    df['MA_SLOW'] = close.rolling(config['ma_slow']).mean()
    
    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean(); mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - sma) / (0.015 * mad)
    
    df['VOL_MA'] = vol.rolling(20).mean()
    exp12 = close.ewm(span=12).mean(); exp26 = close.ewm(span=26).mean()
    df['MACD'] = exp12 - exp26; df['MACD_SIG'] = df['MACD'].ewm(span=9).mean()
    
    lowest_low = low.rolling(config['stoch_k']).min(); highest_high = high.rolling(config['stoch_k']).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    money_flow = tp * vol
    pos_flow = money_flow.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_flow = money_flow.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi_ratio = pos_flow / neg_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['WILLR'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return df

# ---------------------------------------------------------
# 📊 차트 및 데이터 로딩
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol); curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    is_trend_mode = last['ADX'] >= 25 and config['use_dual_mode']
    mode_str = "🌊 추세장 (ZLSMA)" if is_trend_mode else "🦀 횡보장 (RSI+BB)"
    
except Exception as e: st.error(f"데이터 로딩 실패: {e}"); st.stop()

# =========================================================
# 🖥️ 메인 UI
# =========================================================
st.title(f"🔥 {symbol} : {mode_str}")

tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv_symbol = "BITGET:" + symbol.split(':')[0].replace('/', '') + ".P"
h = 500
components.html(f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "{tv_symbol}", "interval": "5", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>""", height=h)

coin, free, total = get_balance_details(exchange)
temp_pnl = 0.0
try:
    pos_list = exchange.fetch_positions([symbol])
    for p in pos_list:
        if float(p['contracts']) > 0: temp_pnl = float(p['unrealizedPnl']); break
except: pass
equity = total + temp_pnl
_, d_pnl, t_pnl = get_analytics()
pnl_color = "#4CAF50" if d_pnl >= 0 else "#FF5252"

st.markdown(f"""
<div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; display: flex; justify-content: space-around; margin-bottom: 20px;">
    <div style="text-align: center;">
        <span style="color: #bbb;">사용 가능 잔고</span><br>
        <span style="font-size: 1.8em; color: white; font-weight: bold;">${free:,.2f}</span>
    </div>
    <div style="text-align: center;">
        <span style="color: #bbb;">총 추정 자산</span><br>
        <span style="font-size: 1.8em; color: #4CAF50; font-weight: bold;">${equity:,.2f}</span>
    </div>
    <div style="text-align: center;">
        <span style="color: #bbb;">총 누적 수익</span><br>
        <span style="font-size: 1.8em; color: {'#4CAF50' if t_pnl>=0 else '#FF5252'}; font-weight: bold;">${t_pnl:,.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🤖 자동매매", "⚡ 수동주문", "📅 시장 정보", "🧠 AI 설정"])

def safe_rerun():
    time.sleep(0.5)
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
                if input_val > free * 0.98: input_val = free * 0.98
                raw_qty = (input_val * p_leverage) / curr_price
                qty = exchange.amount_to_precision(symbol, raw_qty)
            order_side = 'buy' if side == 'long' else 'sell'
            emoji = "🚀"
            
        exchange.create_order(symbol, 'limit', order_side, qty, ticker['ask' if order_side=='buy' else 'bid'], params=params)
        action_name = "청산" if is_close else "진입/추매"
        if is_close: log_trade(action_name, symbol, side, curr_price, qty, p_leverage, log_pnl, log_roi)
        
        coin, free_b, total_b = get_balance_details(exchange)
        eq = total_b + (log_pnl if is_close else 0.0)
        _, dp, tp = get_analytics()
        invest_amt = (float(qty)*curr_price)/p_leverage
        
        msg = f"{emoji} <b>{side.upper()} {action_name} 완료</b>\n------------------\n📍 <b>이유:</b> {reason}\n💲 <b>가격:</b> ${curr_price:,.2f}"
        if is_close: msg += f"\n📈 <b>실현:</b> ${log_pnl:.2f} ({log_roi:.2f}%)\n📅 <b>금일:</b> ${dp:.2f} | 🏆 <b>누적:</b> ${tp:.2f}"
        else: msg += f"\n💸 <b>사용금액:</b> ${invest_amt:,.2f}"
        msg += f"\n------------------\n💰 <b>사용가능:</b> ${free_b:,.2f}\n💎 <b>총 자산:</b> ${eq:,.2f}"
        
        st.success(msg.replace("<b>", "").replace("</b>", ""))
        send_telegram(msg, df.tail(60) if not is_close else None)
        safe_rerun()
    except Exception as e: st.error(f"주문 실패: {e}")

# [탭 1] 자동매매
with tab1:
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        auto_on = st.checkbox("자동매매 시작 (ON/OFF)", value=config['auto_trade'], key="auto_main")
        sz_type = st.radio("진입 금액 기준", ["자산 비율 (%)", "고정 금액 ($)"], index=0 if config['auto_size_type']=='percent' else 1)
    with c_a2:
        if sz_type == "자산 비율 (%)":
            sz_val = st.number_input("비율 설정 (%)", 1.0, 100.0, float(config['auto_size_val']))
            st.info(f"💵 진입 예상금액: 약 ${equity * (sz_val/100):,.2f}")
        else:
            sz_val = st.number_input("금액 설정 ($)", 10.0, 10000.0, float(config['auto_size_val']))
            st.info(f"💵 진입 고정금액: ${sz_val:,.2f}")

    if config['auto_size_val'] != sz_val or config['auto_trade'] != auto_on or config['auto_size_type'] != ('percent' if sz_type == "자산 비율 (%)" else 'fixed'):
        config['auto_size_type'] = 'percent' if sz_type == "자산 비율 (%)" else 'fixed'
        config['auto_size_val'] = sz_val
        config['auto_trade'] = auto_on
        save_settings(config)
        st.rerun()

    long_score = 0; short_score = 0; reasons_L = []; reasons_S = []
    final_long = False; final_short = False
    
    if is_trend_mode:
        if curr_price > last['ZLSMA'] and curr_price > last['Chandelier_Short']: final_long=True; reasons_L.append("ZLSMA상승")
        elif curr_price < last['ZLSMA'] and curr_price < last['Chandelier_Long']: final_short=True; reasons_S.append("ZLSMA하락")
    else:
        if use_rsi:
            if last['RSI'] <= config['rsi_buy']: long_score+=1; reasons_L.append("RSI")
            elif last['RSI'] >= config['rsi_sell']: short_score+=1; reasons_S.append("RSI")
        if use_bb:
            if last['close'] <= last['BB_LO']: long_score+=1; reasons_L.append("BB")
            elif last['close'] >= last['BB_UP']: short_score+=1; reasons_S.append("BB")
        if use_cci:
            if last['CCI'] < -100: long_score+=1; reasons_L.append("CCI")
            elif last['CCI'] > 100: short_score+=1; reasons_S.append("CCI")
        if use_stoch:
            if last['STOCH_K'] < 20: long_score+=1; reasons_L.append("Stoch")
            elif last['STOCH_K'] > 80: short_score+=1; reasons_S.append("Stoch")
        if use_vol and last['vol'] > last['VOL_MA'] * config['vol_mul']:
            long_score+=1; short_score+=1
        
        final_long = long_score >= target_vote
        final_short = short_score >= target_vote
        if final_long and curr_price < last['ZLSMA']: final_long = False
        if final_short and curr_price > last['ZLSMA']: final_short = False

    active_pos = None
    if pos_list:
        for p in pos_list:
            if float(p['contracts']) > 0: active_pos = p; break

    if auto_on:
        if not active_pos:
            entry_amt = equity * (sz_val / 100.0) if sz_type == "자산 비율 (%)" else sz_val
            if final_long: execute_trade('long', reason=",".join(reasons_L), manual_amt=entry_amt)
            elif final_short: execute_trade('short', reason=",".join(reasons_S), manual_amt=entry_amt)
        else:
            cur_side = active_pos['side']
            roi = float(active_pos['percentage'])
            
            should_close = False; close_reason = ""
            if is_trend_mode:
                if cur_side == 'long' and curr_price < last['Chandelier_Long']: should_close=True; close_reason="추세반전"
                elif cur_side == 'short' and curr_price > last['Chandelier_Short']: should_close=True; close_reason="추세반전"
            else:
                if roi >= config['tp']: should_close=True; close_reason="목표달성"
            
            if should_close: execute_trade(cur_side, True, close_reason)
            
            elif use_dca and roi <= config['dca_trigger']:
                curr_margin = float(active_pos.get('initialMargin', 0) or 0)
                if curr_margin == 0: curr_margin = (float(active_pos['contracts']) * float(active_pos['entryPrice'])) / config['leverage']
                base_amt = equity * (sz_val / 100.0) if sz_type == "자산 비율 (%)" else sz_val
                if curr_margin < base_amt * (1 + config['dca_max_count']) * 1.1:
                    add_qty = float(active_pos['contracts'])
                    execute_trade(cur_side, False, f"💧 추매 (ROI {roi:.2f}%)", qty=add_qty)
                    time.sleep(2)

            elif roi <= -config['sl']:
                if use_switching and ((cur_side == 'long' and final_short) or (cur_side == 'short' and final_long)):
                    execute_trade(cur_side, True, "🚨 손절 후 스위칭")
                    time.sleep(1)
                    new_amt = (equity - abs(float(active_pos['unrealizedPnl']))) * (sz_val / 100.0) if sz_type == "자산 비율 (%)" else sz_val
                    execute_trade('short' if cur_side=='long' else 'long', reason="스위칭", manual_amt=new_amt)
                elif use_holding:
                    if roi <= -50.0: execute_trade(cur_side, True, "💀 강제 청산")
                else:
                    execute_trade(cur_side, True, "손절 제한")
        safe_rerun()

# [탭 2] 수동주문
with tab2:
    st.write("✋ **수동 진입 컨트롤**")
    cols = st.columns(4)
    def set_manual(pct): st.session_state['order_usdt'] = float(f"{free * pct:.2f}")
    if cols[0].button("20%"): set_manual(0.2)
    if cols[1].button("50%"): set_manual(0.5)
    if cols[2].button("80%"): set_manual(0.8)
    if cols[3].button("Full"): set_manual(1.0)
    
    manual_amt = st.number_input("주문 금액 ($)", 0.0, free, key='order_usdt')
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입", use_container_width=True): execute_trade('long', reason="수동", manual_amt=manual_amt)
    if b2.button("🔴 숏 진입", use_container_width=True): execute_trade('short', reason="수동", manual_amt=manual_amt)
    if b3.button("🚫 청산", use_container_width=True): 
        if active_pos: execute_trade(active_pos['side'], True, "수동청산")

# [탭 3] 시장 정보
with tab3:
    st.subheader("📰 시장 심리 & 일정")
    try:
        fng = requests.get("https://api.alternative.me/fng/").json()['data'][0]
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("😨 공포/탐욕 지수", f"{fng['value']}", fng['value_classification'])
        if st.button("🧠 AI 브리핑 요청"):
            if gemini_key:
                genai.configure(api_key=gemini_key)
                m = genai.GenerativeModel('gemini-1.5-flash') # 모델명 수정완료
                res = m.generate_content(f"현재 비트코인 공포지수는 {fng['value']}({fng['value_classification']})이다. 시장 상황을 3줄로 요약해줘.")
                st.info(res.text)
    except: st.error("지수 로딩 실패")
    
    st.markdown("---")
    st.write("📅 **주요 일정**")
    st.markdown("""
    * **CPI (물가)**: 매월 중순 발표
    * **FOMC (금리)**: 매월/격월 말 발표
    * **PCE (소비)**: 매월 말 발표
    """)

# [탭 4] AI 설정
with tab4:
    st.subheader("🤖 AI 오토파일럿")
    st.write("현재 시장(변동성, 추세)을 분석해 설정을 자동 최적화합니다.")
    if st.button("🚀 AI 최적화 실행", type="primary"):
        with st.spinner("분석 중..."):
            res = run_autopilot(df, config)
            st.success(res)
            time.sleep(2)
            st.rerun()
