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
import sqlite3
from datetime import datetime, timedelta
import google.generativeai as genai

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True  # 실전 매매 시 False로 변경하세요!
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"
PROPOSALS_FILE = "pending_proposals.json"
DB_FILE = "wonyousi_brain.db"  # AI 기억 저장소 (추가됨)

st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트")

# ---------------------------------------------------------
# 🧠 [추가] AI 기억 저장소 (DB) 초기화
# ---------------------------------------------------------
def init_db():
    """매매 일지와 반성문을 저장할 데이터베이스를 생성합니다."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    # 매매 기록 및 AI 피드백 테이블
    c.execute('''CREATE TABLE IF NOT EXISTS trade_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  symbol TEXT,
                  side TEXT,
                  price REAL,
                  pnl REAL,
                  reason TEXT,
                  ai_feedback TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 🧠 [추가] 과거의 실패로부터 배우는 함수들
# ---------------------------------------------------------
def get_past_mistakes(limit=3):
    """최근 실패한 매매(손실)에 대한 AI의 반성문을 가져옵니다."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT side, reason, ai_feedback FROM trade_history WHERE pnl < 0 ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        
        if not rows: return "과거에 큰 실수는 없었습니다. 초심자의 행운을 빕니다."
        
        feedback = "⛔ **[과거 실패 노트 - 절대 반복 금지]**:\n"
        for row in rows:
            feedback += f"- {row[0]} 진입했다가 손실. (당시 이유: {row[1]}) → 💡 반성: {row[2]}\n"
        return feedback
    except: return "DB 조회 오류"

def log_trade_to_db(symbol, side, price, pnl, reason, ai_feedback):
    """매매 결과를 DB에 영구 저장합니다."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO trade_history (timestamp, symbol, side, price, pnl, reason, ai_feedback) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, side, price, pnl, reason, ai_feedback))
        conn.commit()
        conn.close()
    except Exception as e: print(f"DB Save Error: {e}")

# ---------------------------------------------------------
# 💾 설정 관리
# ---------------------------------------------------------
def load_settings():
    default = {
        "gemini_api_key": "",
        "leverage": 20, "target_vote": 2, "tp": 15.0, "sl": 10.0,
        "auto_trade": False, "order_usdt": 100.0,
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0, 
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14, "vol_mul": 2.0,
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True,
        "use_ma": True, "use_macd": False, "use_stoch": False, 
        "use_mfi": False, "use_willr": False, "use_adx": True,
        "use_switching": True, "use_dca": True, "dca_trigger": -20.0,
        "dca_max_count": 1, "use_holding": True, "auto_size_type": "percent",
        "auto_size_val": 20.0, "use_dual_mode": True, "use_trailing_stop": False,
        "use_smart_betting": False, "no_trade_weekend": False
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
        with open(SETTINGS_FILE, "w") as f: json.dump(new_settings, f)
        st.toast("✅ 설정 저장 완료!", icon="💾")
    except: st.error("설정 저장 실패")

config = load_settings()
if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = config['order_usdt']

# ---------------------------------------------------------
# 🔐 API & AI 초기화 (모델 에러 수정됨)
# ---------------------------------------------------------
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")
gemini_key = st.secrets.get("GEMINI_API_KEY", config.get("gemini_api_key", ""))

if not api_key: st.error("🚨 API 키 설정 필요"); st.stop()

@st.cache_resource
def get_ai_model(key):
    """AI 모델을 안전하게 가져옵니다 (flash 모델 에러 시 pro로 자동 전환)"""
    if not key: return None
    genai.configure(api_key=key)
    try:
        # 사용 가능한 모델 목록을 확인
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1순위: Flash (빠름), 2순위: Pro (안정적), 3순위: 아무거나
        target_model = 'gemini-pro' # 기본값 안전하게 설정
        for m in available_models:
            if 'flash' in m: target_model = m; break
        
        return genai.GenerativeModel(target_model)
    except:
        return genai.GenerativeModel('gemini-pro') # 최후의 수단

ai_model = get_ai_model(gemini_key)

def generate_wonyousi_strategy(df, status_summary):
    """[핵심] 워뇨띠 페르소나 + 회고적 학습을 적용한 AI 판단"""
    if not ai_model: return "⚠️ Gemini Key 없음"
    
    # 과거의 실수 가져오기
    past_mistakes = get_past_mistakes()
    
    # 차트 데이터 요약
    last_row = df.iloc[-1]
    chart_info = f"""
    현재가: {last_row['close']}
    RSI: {last_row['RSI']:.1f}
    볼린저밴드 상태: {status_summary.get('BB', 'Normal')}
    추세(ADX): {last_row['ADX']:.1f}
    """
    
    prompt = f"""
    너는 전설적인 트레이더 '워뇨띠'다. 
    너는 단순 보조지표 숫자보다는 '시장 심리', '캔들 패턴', '추세'를 중시한다.
    
    [현재 시장 상황]
    {chart_info}
    
    [너의 과거 실패 기록 (일기장)]
    {past_mistakes}
    
    위 데이터를 바탕으로 지금 매매해야 할지 판단해라.
    과거의 실수를 반복하지 않는 것이 가장 중요하다.
    
    대답은 오직 JSON 형식으로만 해라. (다른 말 금지)
    형식:
    {{
        "decision": "buy" 또는 "sell" 또는 "hold",
        "reason": "워뇨띠 스타일의 한 줄 근거 (예: 꼬리가 긴 캔들 출현으로 바닥 확인)",
        "confidence": 0~100 사이의 확신도
    }}
    """
    try:
        res = ai_model.generate_content(prompt).text
        # JSON 파싱을 위한 정제
        res = res.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except Exception as e:
        return {"decision": "hold", "reason": f"AI 판단 오류: {e}", "confidence": 0}

# ---------------------------------------------------------
# 📅 데이터 수집
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

def log_trade(action, symbol, side, price, qty, leverage, pnl=0, roi=0):
    now = datetime.now()
    new_data = {"Time": now.strftime("%Y-%m-%d %H:%M:%S"), "Date": now.strftime("%Y-%m-%d"), "Symbol": symbol, "Action": action, "Side": side, "Price": price, "Qty": qty, "Margin": (price*qty)/leverage, "PnL": pnl, "ROI": roi}
    df = pd.DataFrame([new_data])
    if not os.path.exists(LOG_FILE): df.to_csv(LOG_FILE, index=False)
    else: df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ---------------------------------------------------------
# 🤖 [AI 에이전트] 능동 제안 시스템 (DB 연동 추가)
# ---------------------------------------------------------
def manage_proposals(ex, symbol_name):
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
                ex.set_leverage(config['leverage'], symbol_name)
                ticker = ex.fetch_ticker(symbol_name)
                price = ticker['ask'] if data['side'] == 'long' else ticker['bid']
                
                bal = ex.fetch_balance({'type': 'swap'})
                coin_key = 'USDT' if 'USDT' in bal else 'SUSDT'
                free = float(bal[coin_key]['free']); total = float(bal[coin_key]['total'])
                
                amt = config['auto_size_val']
                if config['auto_size_type'] == 'percent': amt = total * (amt / 100.0)
                if amt > free * 0.98: amt = free * 0.98
                
                qty = ex.amount_to_precision(symbol_name, (amt * config['leverage']) / price)
                
                if float(qty) > 0:
                    ex.create_order(symbol_name, 'limit', 'buy' if data['side'] == 'long' else 'sell', qty, price)
                    
                    msg = f"⏳ <b>[AI 자동 실행]</b>\n주인님의 응답이 없어 {data['side'].upper()} 포지션에 자동 진입했습니다.\n이유: {data.get('reason', 'N/A')}"
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'HTML'})
                    log_trade("AI자동진입", symbol_name, data['side'], price, float(qty), config['leverage'])
                
                del proposals[pid]
                changed = True
            except Exception as e:
                del proposals[pid]; changed = True

    if changed:
        with open(PROPOSALS_FILE, 'w') as f: json.dump(proposals, f)

def send_proposal(side, reason):
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
    offset = 0
    while True:
        try:
            manage_proposals(ex, symbol_name)
            
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=30").json()
            if res.get('ok'):
                for up in res['result']:
                    offset = up['update_id']
                    if 'callback_query' in up:
                        cb = up['callback_query']; data = cb['data']; chat_id = cb['message']['chat']['id']
                        
                        if data == 'balance':
                            c, f, t = get_balance(ex)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': f"💰 현금: ${f:,.2f} / 총자산: ${t:,.2f}"})
                        
                        elif data.startswith('acc_'):
                            pid = data.split('_')[1]
                            try:
                                with open(PROPOSALS_FILE, 'r') as f: props = json.load(f)
                                if pid in props:
                                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "✅ 승인 완료. 주문 실행."})
                                    del props[pid]
                                    with open(PROPOSALS_FILE, 'w') as f: json.dump(props, f)
                            except: pass
                            
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
    except Exception as e: return None

exchange = init_exchange()
if not exchange: st.error("🚨 거래소 연결 실패!"); st.stop()

# ---------------------------------------------------------
# 🎨 사이드바
# ---------------------------------------------------------
st.sidebar.title("🛠️ AI 워뇨띠 제어판")
markets = exchange.markets
if markets:
    symbol_list = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
    symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)
else: st.stop()

if not gemini_key:
    k = st.sidebar.text_input("Gemini API Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

found = False
for t in threading.enumerate():
    if t.name == "TG_Thread": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="TG_Thread")
    t.start()

# ---------------------------------------------------------
# 🧮 지표 계산
# ---------------------------------------------------------
def calc_indicators(df):
    close = df['close']; high = df['high']; low = df['low']
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(config['rsi_period'])).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(config['rsi_period'])).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    
    ma = close.rolling(int(config['bb_period'])).mean()
    std = close.rolling(int(config['bb_period'])).std()
    df['BB_UP'] = ma + (std * float(config['bb_std']))
    df['BB_LO'] = ma - (std * float(config['bb_std']))
    
    df['MA_F'] = close.rolling(int(config['ma_fast'])).mean()
    df['MA_S'] = close.rolling(int(config['ma_slow'])).mean()
    
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    df['ADX'] = (df['ATR'] / close) * 1000
    
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
except: st.error("데이터 로딩 중..."); st.stop()

st.title(f"🔥 {symbol} AI Wonyousi Agent")
st.caption("워뇨띠의 직관 + 재귀적 학습(Recursive Learning) 적용됨")

# 1. 지표 대시보드
with st.expander("📊 기본 지표 상태판", expanded=True):
    cols = st.columns(5)
    idx = 0
    active_cnt_l = 0; active_cnt_s = 0
    for name, stat in ind_status.items():
        color = "off"
        if "매수" in stat: color = "normal"; active_cnt_l += 1
        elif "매도" in stat: color = "inverse"; active_cnt_s += 1
        cols[idx % 5].metric(name, stat, delta_color=color)
        idx += 1

# 2. 차트
h = 450
tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv = f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "BITGET:{symbol.replace('/','').split(':')[0]}.P", "interval": "5", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>"""
components.html(tv, height=h)

# 3. 탭 메뉴
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & 제안", "⚡ 수동주문", "📅 시장정보", "📜 매매일지(DB)"])

with t1:
    c1, c2 = st.columns(2)
    auto_on = c1.checkbox("자동매매 활성화", value=config['auto_trade'])
    if auto_on != config['auto_trade']: config['auto_trade'] = auto_on; save_settings(config); st.rerun()

    st.write("---")
    
    # AI 워뇨띠 분석 버튼
    if st.button("🧠 AI(워뇨띠)에게 현재 상황 물어보기"):
        with st.spinner("AI가 과거 일기장을 읽고 차트를 분석 중..."):
            ai_res = generate_wonyousi_strategy(df, ind_status)
            
            st.divider()
            if ai_res['decision'] == 'buy':
                st.success(f"🔵 **매수(LONG) 의견** (확신도: {ai_res.get('confidence')}%)")
            elif ai_res['decision'] == 'sell':
                st.error(f"🔴 **매도(SHORT) 의견** (확신도: {ai_res.get('confidence')}%)")
            else:
                st.warning(f"⚪ **관망(HOLD)** (확신도: {ai_res.get('confidence')}%)")
                
            st.write(f"📝 **분석 이유:** {ai_res.get('reason')}")
            
            if ai_res['decision'] != 'hold':
                if st.button("🚀 이대로 텔레그램 제안 보내기"):
                    send_proposal(ai_res['decision'] + " (AI 워뇨띠 추천)", ai_res['reason'])
                    st.toast("제안 발송 완료!")

with t2:
    st.write("✋ **수동 컨트롤**")
    m_amt = st.number_input("주문 금액 ($)", 0.0, 100000.0, float(config['order_usdt']))
    b1, b2 = st.columns(2)
    if b1.button("🟢 롱 진입"): pass 
    if b2.button("🔴 숏 진입"): pass

with t3:
    st.write("📅 **경제 일정**")
    ev = get_forex_events()
    if not ev.empty: st.dataframe(ev)
    
with t4:
    st.subheader("📖 AI의 매매 일지 & 반성문 (DB)")
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    history_df = pd.read_sql("SELECT * FROM trade_history ORDER BY id DESC", conn)
    conn.close()
    
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.info("아직 기록된 매매가 없습니다.")
        
    # 테스트용 데이터 입력 버튼
    if st.button("테스트 데이터 입력 (DB Test)"):
        log_trade_to_db(symbol, "long", 99000, -50, "뇌동매매", "다음엔 기다렸다가 사자")
        st.rerun()
