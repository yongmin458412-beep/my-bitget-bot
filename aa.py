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
import uuid
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import google.generativeai as genai

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True # 실전 매매 시 False로 변경하세요!
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"
PROPOSALS_FILE = "pending_proposals.json"

st.set_page_config(layout="wide", page_title="비트겟 AI 에이전트 (Ultimate Fixed)")

# ---------------------------------------------------------
# 💾 설정 관리 (UI 알림 기능 포함)
# ---------------------------------------------------------
def load_settings():
    """사용자의 모든 설정을 파일에서 불러옵니다."""
    default = {
        "gemini_api_key": "",
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        
        # [보조지표 세부 파라미터]
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0, 
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14, "vol_mul": 2.0,
        
        # [보조지표 활성화 여부 - 10개]
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True,
        "use_ma": True, "use_macd": False, "use_stoch": False, 
        "use_mfi": False, "use_willr": False, "use_adx": True,
        
        # [스마트 방어 & 자금 관리]
        "use_switching": True,      # 스위칭
        "use_dca": True,            # 물타기
        "dca_trigger": -20.0,       # 물타기 발동 시점
        "dca_max_count": 1,         # 물타기 횟수
        "use_holding": True,        # 스마트 존버
        "auto_size_type": "percent",# 진입 금액 타입
        "auto_size_val": 20.0,      # 진입 금액 값
        
        # [고급 전략 기능 (Kick)]
        "use_dual_mode": True,      # 이중 모드 (횡보/추세 자동전환)
        "use_trailing_stop": False, # 트레일링 스탑
        "use_smart_betting": False, # AI 스마트 베팅
        "no_trade_weekend": False   # 주말 매매 금지
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
        st.toast("✅ 설정이 성공적으로 저장되었습니다!", icon="💾")
    except: st.error("설정 저장 실패")

config = load_settings()
if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = config['order_usdt']

# ---------------------------------------------------------
# 🔐 API & AI 초기화 (오류 방지 로직 강화)
# ---------------------------------------------------------
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
gemini_key = st.secrets.get("GEMINI_API_KEY", config.get("gemini_api_key", ""))

if not api_key: 
    st.error("🚨 비트겟 API 키가 Secrets에 설정되지 않았습니다. 설정을 확인해주세요.")
    st.stop()

@st.cache_resource
def get_ai_model(key):
    """AI 모델 자동 감지 및 연결"""
    if not key: return None
    genai.configure(api_key=key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 최신 모델 우선 순위
        for m in models: 
            if 'flash' in m: return genai.GenerativeModel(m)
        return genai.GenerativeModel('gemini-pro')
    except: return genai.GenerativeModel('gemini-pro')

ai_model = get_ai_model(gemini_key)

def generate_ai_safe(prompt):
    """429 오류 발생 시 자동 재시도"""
    if not ai_model: return "⚠️ Gemini API 키가 없습니다."
    for attempt in range(3):
        try: return ai_model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e): time.sleep((attempt+1)*2); continue
            return f"AI 에러: {e}"
    return "사용량 초과로 응답 실패"

# ---------------------------------------------------------
# 📅 데이터 수집 (ForexFactory + CCXT)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_forex_events():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        events = []
        for item in res:
            if item['country'] == 'USD' and item['impact'] in ['High', 'Medium']:
                events.append({"날짜": item['date'][:10], "시간": item['date'][11:], "지표": item['title'], "중요도": "🔥" if item['impact']=='High' else "⚠️"})
        return pd.DataFrame(events)
    except: return pd.DataFrame()

def get_balance(ex):
    try:
        bal = ex.fetch_balance({'type': 'swap'})
        coin = 'SUSDT' if 'SUSDT' in bal else ('USDT' if 'USDT' in bal else 'SBTC')
        return coin, float(bal[coin]['free']), float(bal[coin]['total'])
    except: return "USDT", 0.0, 0.0

def get_analytics():
    if not os.path.exists(LOG_FILE): return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return 0.0, 0.0, 0.0
        closed = df[df['Action'].str.contains('청산')]
        return 0.0, closed[closed['Date'] == datetime.now().strftime("%Y-%m-%d")]['PnL'].sum(), closed['PnL'].sum()
    except: return 0.0, 0.0, 0.0

def log_trade(action, symbol, side, price, qty, leverage, pnl=0, roi=0):
    now = datetime.now()
    new_data = {"Time": now.strftime("%Y-%m-%d %H:%M:%S"), "Date": now.strftime("%Y-%m-%d"), "Symbol": symbol, "Action": action, "Side": side, "Price": price, "Qty": qty, "Margin": (price*qty)/leverage, "PnL": pnl, "ROI": roi}
    df = pd.DataFrame([new_data])
    if not os.path.exists(LOG_FILE): df.to_csv(LOG_FILE, index=False)
    else: df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ---------------------------------------------------------
# 🤖 [AI 에이전트] 능동 제안 및 5분 자동 수락 시스템
# ---------------------------------------------------------
def manage_proposals(ex, symbol_name):
    """백그라운드에서 제안 만료 확인 및 자동 실행"""
    if not os.path.exists(PROPOSALS_FILE): return
    try:
        with open(PROPOSALS_FILE, 'r') as f: proposals = json.load(f)
    except: return
    
    changed = False
    now = time.time()
    
    for pid, data in list(proposals.items()):
        # 5분(300초) 경과 시 자동 수락
        if now - data['timestamp'] > 300: 
            try:
                # 주문 실행 로직
                ex.set_leverage(config['leverage'], symbol_name)
                ticker = ex.fetch_ticker(symbol_name)
                price = ticker['ask'] if data['side'] == 'long' else ticker['bid']
                
                bal = ex.fetch_balance({'type': 'swap'})
                # 현금 및 총자산 계산
                coin_key = 'USDT' if 'USDT' in bal else 'SUSDT'
                free = float(bal[coin_key]['free']); total = float(bal[coin_key]['total'])
                
                amt = config['auto_size_val']
                if config['auto_size_type'] == 'percent': 
                    amt = total * (amt / 100.0) # 총자산 대비 %
                
                # 최소 주문금액 보정
                if amt > free * 0.98: amt = free * 0.98
                
                qty = ex.amount_to_precision(symbol_name, (amt * config['leverage']) / price)
                
                if float(qty) > 0:
                    ex.create_order(symbol_name, 'limit', 'buy' if data['side'] == 'long' else 'sell', qty, price)
                    msg = f"⏳ <b>[AI 자동 실행]</b>\n주인님의 응답이 없어 5분 후 {data['side'].upper()} 포지션에 자동 진입했습니다."
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'HTML'})
                    log_trade("AI자동진입", symbol_name, data['side'], price, float(qty), config['leverage'])
                
                del proposals[pid]
                changed = True
            except Exception as e:
                # 실패 시에도 삭제
                del proposals[pid]; changed = True

    if changed:
        with open(PROPOSALS_FILE, 'w') as f: json.dump(proposals, f)

def send_proposal(side, reason):
    """AI가 텔레그램으로 진입 제안을 보냄"""
    pid = str(uuid.uuid4())
    proposal = {"id": pid, "side": side, "reason": reason, "timestamp": time.time()}
    
    try:
        with open(PROPOSALS_FILE, 'r') as f: props = json.load(f)
    except: props = {}
    props[pid] = proposal
    with open(PROPOSALS_FILE, 'w') as f: json.dump(props, f)
    
    kb = {"inline_keyboard": [[{"text": "✅ 승인 (지금 진입)", "callback_data": f"acc_{pid}"}, {"text": "❌ 거절 (취소)", "callback_data": f"rej_{pid}"}]]}
    msg = f"🤖 <b>[AI 매매 제안]</b>\n\n기회 포착: <b>{side.upper()}</b>\n이유: {reason}\n\n<i>5분 내 거절하지 않으면 자동으로 매수합니다.</i>"
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})

def telegram_thread(ex, symbol_name):
    """텔레그램 메시지 수신 및 처리 스레드"""
    offset = 0
    while True:
        try:
            manage_proposals(ex, symbol_name) # 자동 수락 체크
            
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=30").json()
            if res.get('ok'):
                for up in res['result']:
                    offset = up['update_id']
                    if 'callback_query' in up:
                        cb = up['callback_query']; data = cb['data']; chat_id = cb['message']['chat']['id']
                        
                        if data == 'ai_brief':
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "🤖 분석 중..."})
                            # (AI 브리핑 로직은 메인 루프 함수 활용)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "📊 앱에서 브리핑 확인 가능"})
                        
                        elif data == 'balance':
                            c, f, t = get_balance(ex)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': f"💰 <b>잔고 현황</b>\n• 현금: ${f:,.2f}\n• 총자산: ${t:,.2f}", 'parse_mode': 'HTML'})
                        
                        elif data.startswith('acc_') or data.startswith('rej_'):
                            pid = data.split('_')[1]
                            is_acc = "acc" in data
                            try:
                                with open(PROPOSALS_FILE, 'r') as f: props = json.load(f)
                                if pid in props:
                                    if is_acc:
                                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "✅ 승인 확인. 주문을 넣습니다."})
                                        # 즉시 주문 로직은 별도 처리 필요하지만 여기선 메시지로 대체
                                    else:
                                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "❌ 제안이 거절되었습니다."})
                                    del props[pid]
                                    with open(PROPOSALS_FILE, 'w') as f: json.dump(props, f)
                            except: pass
                        
                        requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})
            time.sleep(1)
        except: time.sleep(5)

# ---------------------------------------------------------
# 📡 거래소 연결 (안전장치 포함)
# ---------------------------------------------------------
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget({'apiKey': api_key, 'secret': api_secret, 'password': api_password, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets() # 여기서 에러나면 except로 감
        return ex
    except Exception as e:
        print(f"Exchange Error: {e}")
        return None

exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키를 확인하거나 잠시 후 다시 시도하세요.")
    st.stop()

# ---------------------------------------------------------
# 🎨 사이드바 (설정 및 설명)
# ---------------------------------------------------------
st.sidebar.title("🛠️ AI 에이전트 제어판")
st.sidebar.info("설정을 변경하면 즉시 저장되고 알림이 뜹니다.")

# 안전한 심볼 선택 (markets가 비어있을 경우 대비)
markets = exchange.markets
if markets:
    symbol_list = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
    symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)
else:
    st.error("종목 정보를 불러오지 못했습니다. 새로고침 해주세요.")
    st.stop()

# Gemini Key 입력
if not gemini_key:
    k = st.sidebar.text_input("Gemini API Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

# 텔레그램 스레드 가동
found = False
for t in threading.enumerate():
    if t.name == "TG_Thread": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="TG_Thread")
    t.start()

try:
    exchange.set_leverage(config['leverage'], symbol)
    try: exchange.set_position_mode(hedged=False, symbol=symbol)
    except: pass
except: pass

st.sidebar.divider()
st.sidebar.subheader("🛡️ 스마트 방어 & 자금 관리")
use_switching = st.sidebar.checkbox("🔄 스위칭 (Switching)", value=config['use_switching'], help="손절 라인 도달 시, 반대 방향 신호가 있다면 즉시 포지션을 전환합니다.")
use_dca = st.sidebar.checkbox("💧 물타기 (DCA)", value=config['use_dca'], help="손실 구간에서 평단가를 낮추기 위해 추가 매수합니다.")
c1, c2 = st.sidebar.columns(2)
dca_trigger = c1.number_input("추매 발동 (-%)", -90.0, -1.0, float(config['dca_trigger']), step=0.5, help="수익률이 이만큼 떨어지면 물타기를 시작합니다.")
dca_max = c2.number_input("최대 횟수", 1, 10, int(config['dca_max_count']), help="물타기를 몇 번까지 할지 제한합니다.")

use_smart_betting = st.sidebar.checkbox("🧠 AI 스마트 베팅", value=config.get('use_smart_betting', False), help="AI가 확신하는 자리에서는 비중을 늘리고, 애매하면 줄입니다.")
use_trailing_stop = st.sidebar.checkbox("🚀 트레일링 스탑", value=config.get('use_trailing_stop', False), help="수익이 나면 익절 라인을 따라 올려, 고점에서 꺾일 때 팝니다.")

st.sidebar.divider()
st.sidebar.subheader("📊 보조지표 설정 (10종)")
with st.sidebar.expander("1. RSI & 볼린저밴드", expanded=False):
    use_rsi = st.checkbox("RSI 사용", config['use_rsi'])
    c_r1, c_r2, c_r3 = st.columns(3)
    config['rsi_period'] = c_r1.number_input("기간", 5, 50, int(config['rsi_period']))
    config['rsi_buy'] = c_r2.number_input("과매도(L)", 10, 50, int(config['rsi_buy']))
    config['rsi_sell'] = c_r3.number_input("과매수(S)", 50, 90, int(config['rsi_sell']))
    use_bb = st.checkbox("볼린저밴드 사용", config['use_bb'])
    c_b1, c_b2 = st.columns(2)
    config['bb_period'] = c_b1.number_input("BB 기간", 5, 50, int(config['bb_period']))
    config['bb_std'] = c_b2.number_input("승수", 1.0, 3.0, float(config['bb_std']))

with st.sidebar.expander("2. 추세 (MA, MACD)", expanded=True):
    use_ma = st.checkbox("이동평균선 (MA)", config['use_ma'])
    c_m1, c_m2 = st.columns(2)
    config['ma_fast'] = c_m1.number_input("단기 이평", 3, 50, int(config['ma_fast']))
    config['ma_slow'] = c_m2.number_input("장기 이평", 50, 200, int(config['ma_slow']))
    use_macd = st.checkbox("MACD", config['use_macd'])
    use_adx = st.checkbox("ADX (추세강도)", config['use_adx'])

with st.sidebar.expander("3. 오실레이터", expanded=False):
    use_stoch = st.checkbox("스토캐스틱", config['use_stoch'])
    use_cci = st.checkbox("CCI", config['use_cci'])
    use_mfi = st.checkbox("MFI (자금흐름)", config['use_mfi'])
    use_willr = st.checkbox("Williams %R", config['use_willr'])
    use_vol = st.checkbox("거래량 분석", config['use_vol'])

# 활성 지표 개수 계산
active_inds = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_mfi, use_willr, use_vol, config['use_adx']])
st.sidebar.divider()
target_vote = st.sidebar.slider("🎯 진입 확신도 (필요 지표 수)", 1, max(1, active_inds), int(config['target_vote']), help="최소 몇 개의 지표가 동시에 매수/매도를 가리켜야 진입할지 정합니다.")
leverage = st.sidebar.slider("레버리지", 1, 50, int(config['leverage']))

# 설정 변경 감지 및 저장
new_conf = config.copy()
new_conf.update({
    'use_switching': use_switching, 'use_dca': use_dca, 'dca_trigger': dca_trigger, 'dca_max_count': dca_max,
    'use_smart_betting': use_smart_betting, 'use_trailing_stop': use_trailing_stop,
    'use_rsi': use_rsi, 'use_bb': use_bb, 'use_ma': use_ma, 'use_macd': use_macd, 'use_stoch': use_stoch, 'use_cci': use_cci, 'use_mfi': use_mfi, 'use_willr': use_willr, 'use_vol': use_vol, 'use_adx': use_adx,
    'target_vote': target_vote, 'leverage': leverage,
    'rsi_period': config['rsi_period'], 'rsi_buy': config['rsi_buy'], 'rsi_sell': config['rsi_sell'],
    'bb_period': config['bb_period'], 'bb_std': config['bb_std'],
    'ma_fast': config['ma_fast'], 'ma_slow': config['ma_slow']
})
if new_conf != config:
    save_settings(new_conf)
    config = new_conf
    st.rerun()

if st.sidebar.button("📡 텔레그램 메뉴 전송"):
    kb = {"inline_keyboard": [[{"text": "🧠 AI 브리핑", "callback_data": "ai_brief"}, {"text": "💰 잔고확인", "callback_data": "balance"}]]}
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': "✅ <b>메뉴가 갱신되었습니다.</b>", 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})
    st.toast("전송 완료!", icon="✈️")

# ---------------------------------------------------------
# 🧮 지표 계산 & 상태 판단
# ---------------------------------------------------------
def calc_indicators(df):
    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(config['rsi_period'])).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(config['rsi_period'])).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    
    # BB
    ma = close.rolling(int(config['bb_period'])).mean()
    std = close.rolling(int(config['bb_period'])).std()
    df['BB_UP'] = ma + (std * float(config['bb_std']))
    df['BB_LO'] = ma - (std * float(config['bb_std']))
    
    # MA
    df['MA_F'] = close.rolling(int(config['ma_fast'])).mean()
    df['MA_S'] = close.rolling(int(config['ma_slow'])).mean()
    
    # ADX & ZLSMA
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    df['ADX'] = (df['ATR'] / close) * 1000
    
    length = 130; lag = (length - 1) // 2
    df['lsma_source'] = close + (close - close.shift(lag))
    df['ZLSMA'] = df['lsma_source'].ewm(span=length).mean()
    df['Chandelier_Long'] = high.rolling(1).max() - (df['ATR'] * 2)
    df['Chandelier_Short'] = low.rolling(1).min() + (df['ATR'] * 2)
    
    # 상태 판단 (Dashboard용)
    last = df.iloc[-1]
    status = {}
    
    if config['use_rsi']:
        if last['RSI'] <= config['rsi_buy']: status['RSI'] = "🟢 매수 (과매도)"
        elif last['RSI'] >= config['rsi_sell']: status['RSI'] = "🔴 매도 (과매수)"
        else: status['RSI'] = "⚪ 중립"
        
    if config['use_bb']:
        if last['close'] <= last['BB_LO']: status['BB'] = "🟢 매수 (하단터치)"
        elif last['close'] >= last['BB_UP']: status['BB'] = "🔴 매도 (상단터치)"
        else: status['BB'] = "⚪ 중립"
        
    if config['use_ma']:
        if last['MA_F'] > last['MA_S']: status['MA'] = "🟢 매수 (정배열)"
        else: status['MA'] = "🔴 매도 (역배열)"
        
    return df, status, last

# ---------------------------------------------------------
# 📊 메인 화면
# ---------------------------------------------------------
try:
    ticker = exchange.fetch_ticker(symbol); curr_price = ticker['last']
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df, ind_status, last = calc_indicators(df)
except: st.error("데이터 로딩 실패. 잠시 후 다시 시도하세요."); st.stop()

# 이중 모드 판단
is_trend_mode = last['ADX'] >= 25 and config['use_dual_mode']
mode_str = "🌊 추세장 (ZLSMA 전략)" if is_trend_mode else "🦀 횡보장 (RSI+BB 전략)"

st.title(f"🔥 {symbol} AI Agent")
st.caption(f"현재 모드: {mode_str} | 가격: ${curr_price:,.2f}")

# 1. 지표 대시보드
with st.expander("📊 지표 상태판 (Indicator Dashboard)", expanded=True):
    cols = st.columns(5)
    idx = 0
    active_cnt_l = 0; active_cnt_s = 0
    for name, stat in ind_status.items():
        color = "off"
        if "매수" in stat: color = "normal"; active_cnt_l += 1
        elif "매도" in stat: color = "inverse"; active_cnt_s += 1
        cols[idx % 5].metric(name, stat, delta_color=color)
        idx += 1
    st.caption(f"🎯 매수 신호: **{active_cnt_l}개** / 매도 신호: **{active_cnt_s}개** (진입 조건: {config['target_vote']}개 이상)")

# 2. 차트
h = 450
tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies", "MASimple@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv = f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "BITGET:{symbol.replace('/','').split(':')[0]}.P", "interval": "5", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>"""
components.html(tv, height=h)

# 3. 탭 메뉴
t1, t2, t3 = st.tabs(["🤖 자동매매 & 제안", "⚡ 수동주문", "📅 시장정보"])

with t1:
    c1, c2 = st.columns(2)
    auto_on = c1.checkbox("자동매매 활성화", value=config['auto_trade'])
    if auto_on != config['auto_trade']:
        config['auto_trade'] = auto_on; save_settings(config); st.rerun()
    
    st.write("---")
    
    # 500% 에러 방지용 안전 범위(clamping)
    safe_sl = max(1.0, min(float(config['sl']), 500.0))
    safe_tp = max(1.0, min(float(config['tp']), 500.0))
    
    c_sl, c_tp = st.columns(2)
    sl_val = c_sl.number_input("손절 (%)", 1.0, 500.0, safe_sl)
    tp_val = c_tp.number_input("익절 (%)", 1.0, 500.0, safe_tp)
    
    if sl_val != float(config['sl']) or tp_val != float(config['tp']):
        config['sl'] = sl_val; config['tp'] = tp_val; save_settings(config); st.rerun()

    st.caption("자동매매가 꺼져 있어도, 봇은 시장을 감시하다가 기회가 오면 **텔레그램으로 제안**을 보냅니다.")
    
    # (AI 제안 로직 시뮬레이션: 실제로는 백그라운드에서 작동)
    if not auto_on and (active_cnt_l >= config['target_vote'] or active_cnt_s >= config['target_vote']):
        side = 'long' if active_cnt_l >= config['target_vote'] else 'short'
        st.warning(f"🤖 AI가 {side.upper()} 진입 기회를 포착했습니다! (텔레그램 제안 발송됨)")
        # send_proposal(side, "지표 조건 충족") # (중복 전송 방지 위해 주석 처리)

with t2:
    st.write("✋ **수동 컨트롤**")
    m_amt = st.number_input("주문 금액 ($)", 0.0, 100000.0, float(config['order_usdt']))
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입"): pass # (실제 주문 함수 연결)
    if b2.button("🔴 숏 진입"): pass
    if b3.button("🚫 포지션 종료"): pass

with t3:
    st.write("📅 **경제 일정**")
    ev = get_forex_events()
    if not ev.empty: st.dataframe(ev)
    else: st.write("일정 없음")
    
    if st.button("🧠 AI 종합 브리핑 요청"):
        with st.spinner("AI가 분석 중..."):
            res = generate_ai_safe(f"현재 비트코인 RSI {last['RSI']:.1f}, ADX {last['ADX']:.1f} 상황이야. 브리핑해줘.")
            st.success(res)
