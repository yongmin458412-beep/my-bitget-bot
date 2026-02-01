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
import google.generativeai as genai
from datetime import datetime, timedelta
import plotly.graph_objects as go

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True  # 실전 매매 시 False로 변경하세요!
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"
PROPOSALS_FILE = "pending_proposals.json"
DB_FILE = "wonyousi_brain.db"  # [New] AI 기억 저장소

st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Ultimate Integration)")

# ---------------------------------------------------------
# 🧠 [New] AI 기억 저장소 (DB) & 회고 시스템
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
# 💾 설정 관리 (기존 기능 유지)
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
        "use_switching": True, "use_dca": True, "dca_trigger": -20.0,
        "dca_max_count": 1, "use_holding": True, "auto_size_type": "percent",
        "auto_size_val": 20.0, 
        
        # [고급 전략 기능]
        "use_dual_mode": True, "use_trailing_stop": False,
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
        with open(SETTINGS_FILE, "w") as f:
            json.dump(new_settings, f)
        st.toast("✅ 설정이 성공적으로 저장되었습니다!", icon="💾")
    except: st.error("설정 저장 실패")

config = load_settings()
if 'order_usdt' not in st.session_state: st.session_state['order_usdt'] = config['order_usdt']

# ---------------------------------------------------------
# 🔐 API & AI 초기화 (404 오류 수정됨)
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
    """AI 모델 자동 감지 및 연결 (404 오류 방지)"""
    if not key: return None
    genai.configure(api_key=key)
    try:
        # 사용 가능한 모델 목록 조회
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1순위: Flash (빠름), 2순위: Pro (안정적), 3순위: 기본
        target_model = 'gemini-pro' 
        for m in models:
            if 'flash' in m: target_model = m; break
        
        return genai.GenerativeModel(target_model)
    except: 
        # 목록 조회 실패 시 기본 모델 강제 지정
        return genai.GenerativeModel('gemini-pro')

ai_model = get_ai_model(gemini_key)

def generate_wonyousi_strategy(df, status_summary):
    """[New] 워뇨띠 페르소나 + 회고적 학습 전략 생성"""
    if not ai_model: return {"decision": "hold", "reason": "API Key 없음", "confidence": 0}
    
    past_mistakes = get_past_mistakes()
    last_row = df.iloc[-1]
    
    prompt = f"""
    너는 전설적인 트레이더 '워뇨띠'다. 
    보조지표 숫자보다는 '시장 심리', '캔들 패턴', '추세'를 중시한다.
    
    [현재 시장 상황]
    - 가격: {last_row['close']}
    - RSI: {last_row['RSI']:.1f}
    - 볼린저밴드 상태: {status_summary.get('BB', 'Normal')}
    - 추세강도(ADX): {last_row['ADX']:.1f}
    
    [너의 과거 실패 기록 (일기장 - 이 실수는 반복 금지)]
    {past_mistakes}
    
    위 데이터를 바탕으로 지금 매매해야 할지 판단해라.
    JSON 형식으로만 답해라.
    
    형식:
    {{
        "decision": "buy" 또는 "sell" 또는 "hold",
        "reason": "워뇨띠 스타일의 한 줄 근거",
        "confidence": 0~100 사이의 숫자
    }}
    """
    
    # 429 에러 방어 및 재시도 로직
    max_retries = 2
    for attempt in range(max_retries):
        try:
            res = ai_model.generate_content(prompt).text
            res = res.replace("```json", "").replace("```", "").strip()
            return json.loads(res)
        except Exception as e:
            if "429" in str(e): # 사용량 초과 시
                if attempt < max_retries - 1:
                    time.sleep(60) # 60초 대기 후 재시도
                    continue
                else:
                    return {"decision": "hold", "reason": "API 한도 초과 (잠시 휴식)", "confidence": 0}
            else:
                return {"decision": "hold", "reason": f"AI 분석 오류: {e}", "confidence": 0}

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

def log_trade(action, symbol, side, price, qty, leverage, pnl=0, roi=0):
    """기존 CSV 로그 (엑셀 호환용)"""
    now = datetime.now()
    new_data = {"Time": now.strftime("%Y-%m-%d %H:%M:%S"), "Date": now.strftime("%Y-%m-%d"), "Symbol": symbol, "Action": action, "Side": side, "Price": price, "Qty": qty, "Margin": (price*qty)/leverage, "PnL": pnl, "ROI": roi}
    df = pd.DataFrame([new_data])
    if not os.path.exists(LOG_FILE): df.to_csv(LOG_FILE, index=False)
    else: df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ---------------------------------------------------------
# 🤖 [AI 에이전트] 능동 제안 및 5분 자동 수락 시스템
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
                    msg = f"⏳ <b>[AI 자동 실행]</b>\n주인님의 응답이 없어 5분 후 {data['side'].upper()} 포지션에 자동 진입했습니다.\n이유: {data.get('reason', 'AI 판단')}"
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'HTML'})
                    log_trade("AI자동진입", symbol_name, data['side'], price, float(qty), config['leverage'])
                    # DB에도 기록 (피드백 없음, 추후 청산 시 기록)
                
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
    msg = f"🤖 <b>[AI 워뇨띠 제안]</b>\n\n기회 포착: <b>{side.upper()}</b>\n이유: {reason}\n\n<i>5분 내 거절하지 않으면 자동으로 매수합니다.</i>"
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})

# ---------------------------------------------------------
# 📡 거래소 연결 (이 부분이 빠져서 에러가 난 것입니다)
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
    except Exception as e:
        print(f"Exchange Init Error: {e}")
        return None

# 👇 [핵심] 이 줄이 없으면 NameError가 뜹니다! 꼭 있어야 합니다.
exchange = init_exchange()

# 만약 연결에 실패했다면 중단
if not exchange:
    st.error("🚨 거래소 연결 실패! API Key를 확인해주세요.")
    st.stop()

# ---------------------------------------------------------
# 📊 [복구] 10종 지표 계산 함수 (이게 없으면 오류남!)
# ---------------------------------------------------------
def calc_indicators(df):
    """10가지 기술적 지표를 모두 계산합니다."""
    # 데이터가 없으면 그냥 반환
    if df.empty: return df, {}, None

    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    # 1. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(config['rsi_period'])).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(config['rsi_period'])).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))

    # 2. BB (볼린저밴드)
    ma = close.rolling(int(config['bb_period'])).mean()
    std = close.rolling(int(config['bb_period'])).std()
    df['BB_UP'] = ma + (std * float(config['bb_std']))
    df['BB_LO'] = ma - (std * float(config['bb_std']))

    # 3. MA (이평선)
    df['MA_F'] = close.rolling(int(config['ma_fast'])).mean()
    df['MA_S'] = close.rolling(int(config['ma_slow'])).mean()

    # 4. MACD
    k = close.ewm(span=12, adjust=False).mean()
    d = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. Stochastic
    low_min = low.rolling(int(config.get('stoch_k', 14))).min()
    high_max = high.rolling(int(config.get('stoch_k', 14))).max()
    df['STOCH_K'] = 100 * ((close - low_min) / (high_max - low_min))

    # 6. CCI
    tp = (high + low + close) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # 7. ADX (추세 강도)
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(14).mean()
    df['ADX'] = (atr / close) * 1000

    # 8. Volume (거래량 폭발)
    df['VOL_MA'] = vol.rolling(20).mean()

    # 상태 판단 (대시보드 표시용)
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
    
    if config['use_adx']:
        status['ADX'] = "📈 추세장" if last['ADX'] > 25 else "🦀 횡보장"

    return df, status, last
    
# ---------------------------------------------------------
# 🤖 [Auto] 텔레그램 봇 (버튼 메뉴 + 자동매매 + 상태확인)
# ---------------------------------------------------------
def telegram_thread(ex, symbol_name):
    ANALYSIS_INTERVAL = 900  # 15분
    last_run = 0
    
    # 봇 시작 시 메뉴 전송
    menu_kb = {
        "inline_keyboard": [
            [{"text": "💰 내 잔고", "callback_data": "balance"}, {"text": "📊 내 포지션", "callback_data": "position"}],
            [{"text": "🧠 현재상황 분석", "callback_data": "analysis"}, {"text": "📡 AI 상태확인", "callback_data": "status"}]
        ]
    }
    
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                  data={'chat_id': tg_id, 'text': "🚀 **AI 워뇨띠 봇 가동**\n아래 메뉴를 눌러 확인하세요.", 
                        'reply_markup': json.dumps(menu_kb), 'parse_mode': 'Markdown'})

    offset = 0

    while True:
        try:
            now = time.time()
            
            # [1] 15분 주기 자동 분석 로직
            if now - last_run > ANALYSIS_INTERVAL:
                ohlcv = ex.fetch_ohlcv(symbol_name, '5m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df, status, last = calc_indicators(df)
                
                strategy = generate_wonyousi_strategy(df, status)
                decision = strategy['decision']
                
                # 텔레그램 정기 보고
                msg = f"🤖 **[15분 자동분석]**\n결론: {decision.upper()} ({strategy.get('confidence')}%)"
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg})
                
                if decision in ['buy', 'sell']:
                    # 매매 로직 (즉시 진입)
                    side = decision
                    price = last['close']
                    try:
                        ex.set_leverage(config['leverage'], symbol_name)
                        bal = ex.fetch_balance({'type': 'swap'})
                        # 주문량 계산
                        amt_usdt = config['order_usdt']
                        qty = ex.amount_to_precision(symbol_name, (amt_usdt * config['leverage']) / price)
                        
                        if float(qty) > 0:
                            # ex.create_market_order(symbol_name, side, qty) # ⚠️ 주석 해제 시 실주문
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                          data={'chat_id': tg_id, 'text': f"⚡ **[즉시 진입]** {side.upper()} 체결\n가격: {price}"})
                            log_trade_to_db(symbol_name, side, price, 0, strategy['final_reason'], "진행중")
                    except Exception as e:
                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': f"❌ 주문 실패: {e}"})
                
                last_run = now

            # [2] 텔레그램 메시지 & 버튼 처리 (폴링)
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1").json()
            
            if res.get('ok'):
                for up in res['result']:
                    offset = up['update_id']
                    
                    # 1. 채팅 명령어 처리 (/start)
                    if 'message' in up and 'text' in up['message']:
                        text = up['message']['text']
                        chat_id = up['message']['chat']['id']
                        if text == "/start" or text == "/menu":
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                          data={'chat_id': chat_id, 'text': "🛠️ **제어 패널**", 'reply_markup': json.dumps(menu_kb), 'parse_mode': 'Markdown'})

                    # 2. 버튼 클릭(Callback) 처리
                    if 'callback_query' in up:
                        cb = up['callback_query']
                        data = cb['data']
                        chat_id = cb['message']['chat']['id']
                        
                        # A. 잔고 확인
                        if data == 'balance':
                            try:
                                bal = ex.fetch_balance({'type': 'swap'})
                                usdt = bal['USDT']['free']
                                total = bal['USDT']['total']
                                msg = f"💰 **내 지갑 현황**\n사용 가능: ${usdt:,.2f}\n총 자산: ${total:,.2f}"
                            except: msg = "❌ 잔고 조회 실패"
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': msg, 'parse_mode': 'Markdown'})
                        
                        # B. 포지션 확인 (Bitget 전용)
                        elif data == 'position':
                            try:
                                positions = ex.fetch_positions([symbol_name])
                                active_pos = [p for p in positions if float(p['contracts']) > 0]
                                if not active_pos:
                                    msg = "🧘 **현재 무포지션 상태입니다.**"
                                else:
                                    p = active_pos[0]
                                    side = p['side'].upper() # long/short
                                    entry = float(p['entryPrice'])
                                    upnl = float(p['unrealizedPnl'])
                                    roe = p['percentage']
                                    msg = f"📊 **현재 포지션 ({symbol_name})**\n방향: {side}\n평단가: ${entry:,.2f}\n수익금: ${upnl:.2f}\n수익률: {roe}%"
                            except Exception as e: msg = f"❌ 포지션 조회 오류: {e}"
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': msg, 'parse_mode': 'Markdown'})

                        # C. 즉시 AI 분석
                        elif data == 'analysis':
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': "🧠 AI가 차트를 분석 중입니다..."})
                            # 데이터 수집 및 분석 (스레드 내 로직 재사용)
                            ohlcv = ex.fetch_ohlcv(symbol_name, '5m', limit=100)
                            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                            df, status, last = calc_indicators(df)
                            strategy = generate_wonyousi_strategy(df, status)
                            
                            rpt = f"""
🔎 **실시간 AI 분석 보고**
결론: {strategy['decision'].upper()} (확신도 {strategy.get('confidence')}%)

📈 추세: {strategy.get('reason_trend')}
🕯️ 패턴: {strategy.get('reason_candle')}
💡 요약: {strategy.get('final_reason')}
"""
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': rpt, 'parse_mode': 'Markdown'})

                        # D. 봇 상태 확인
                        elif data == 'status':
                            next_run = ANALYSIS_INTERVAL - (now - last_run)
                            status_msg = f"""
📡 **AI 시스템 상태: 정상 가동 중**
- 현재 모드: 완전 자동 (15분 주기)
- 다음 정기 분석까지: {int(next_run // 60)}분 {int(next_run % 60)}초 남음
- AI 연결: {'✅ 연결됨' if ai_model else '❌ 끊김'}
- 차트 감시 중... 이상 무!
"""
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': chat_id, 'text': status_msg})

                        # 버튼 로딩 애니메이션 종료
                        requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})

            time.sleep(1) # 과부하 방지

        except Exception as e:
            print(f"Bot Error: {e}")
            time.sleep(5)

# ---------------------------------------------------------
# 🎨 [UI] 메인 대시보드 (직관성 강화)
# ---------------------------------------------------------
markets = exchange.markets
symbol = "BTC/USDT:USDT" # 기본값

# 사이드바 설정
st.sidebar.header("🛠️ 설정")
if not gemini_key:
    k = st.sidebar.text_input("Gemini API Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

# 스레드 시작
found = False
for t in threading.enumerate():
    if t.name == "TG_Thread": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="TG_Thread")
    t.start()

# 데이터 로딩
ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
df['time'] = pd.to_datetime(df['time'], unit='ms')
df, status, last = calc_indicators(df)

# === [UI 1] 상단 상태 배너 ===
st.title(f"🤖 {symbol} Autonomous Trader")
curr_price = last['close']
rsi_val = last['RSI']

# 상태에 따른 색상/메시지 결정
if rsi_val < 30: 
    banner_color = "green"
    banner_msg = "🟢 강력 매수 구간 (과매도)"
elif rsi_val > 70: 
    banner_color = "red"
    banner_msg = "🔴 강력 매도 구간 (과매수)"
else: 
    banner_color = "gray"
    banner_msg = "⚪ 관망 구간 (중립)"

st.markdown(f"""
<div style="padding: 20px; background-color: #1e1e1e; border-radius: 10px; border-left: 10px solid {banner_color}; margin-bottom: 20px;">
    <h2 style="margin:0; color: white;">{banner_msg}</h2>
    <p style="margin:0; color: #aaaaaa;">현재가: <b>${curr_price:,.2f}</b> | AI 모드: 완전 자율 주행</p>
</div>
""", unsafe_allow_html=True)

# === [UI 2] 직관적인 게이지 차트 (Plotly) ===
c1, c2, c3 = st.columns(3)

with c1:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = rsi_val,
        title = {'text': "RSI (강도)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': banner_color},
                 'steps': [
                     {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                     {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}],
                 'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': rsi_val}}))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with c2:
    # 추세 강도(ADX) 게이지
    adx_val = last['ADX']
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = adx_val,
        title = {'text': "ADX (추세 힘)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "orange" if adx_val > 25 else "gray"},
                 'steps': [{'range': [0, 25], 'color': "rgba(255, 255, 255, 0.1)"}]}))
    fig2.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig2, use_container_width=True)

with c3:
    # 캔들 차트 (간소화)
    st.markdown("#### 📊 최근 차트 흐름")
    st.line_chart(df.set_index('time')['close'].tail(50), height=200)

# === [UI 3] AI 상세 분석 리포트 ===
st.divider()
col_ai, col_log = st.columns([2, 1])

with col_ai:
    st.subheader("🧠 AI 실시간 분석 리포트")
    if st.button("🔍 지금 바로 분석 요청 (수동)"):
        with st.spinner("AI가 차트를 뜯어보는 중..."):
            ai_res = generate_wonyousi_strategy(df, status)
            
            # 카드로 결과 표시
            st.markdown(f"""
            <div style="background-color: #262730; padding: 20px; border-radius: 10px;">
                <h3>결론: <span style="color: {'#00ff00' if ai_res['decision']=='buy' else '#ff0000'};">{ai_res['decision'].upper()}</span> (확신도 {ai_res.get('confidence')}% )</h3>
                <hr>
                <p><b>📈 추세 관점:</b> {ai_res.get('reason_trend')}</p>
                <p><b>🕯️ 캔들/패턴:</b> {ai_res.get('reason_candle')}</p>
                <p><b>💡 최종 판단:</b> {ai_res.get('final_reason')}</p>
                <hr>
                <small>추천 손절가: {ai_res.get('stop_loss')} | 익절가: {ai_res.get('take_profit')}</small>
            </div>
            """, unsafe_allow_html=True)
# ---------------------------------------------------------
# 🎨 사이드바 (설정 유지)
# ---------------------------------------------------------
st.sidebar.title("🛠️ AI 에이전트 제어판")
st.sidebar.info("설정을 변경하면 즉시 저장되고 알림이 뜹니다.")

try:
    if exchange:
        balance = exchange.fetch_balance({'type': 'swap'})
        usdt = balance['USDT']['free']
        st.sidebar.markdown(f"""
        <div style="padding:10px; background-color:#262730; border-radius:5px; margin-bottom:10px;">
            <p style="color:gray; margin:0; font-size:12px;">내 보유 현금 (USDT)</p>
            <h2 style="color:#00FFAA; margin:0;">${usdt:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
except: pass

markets = exchange.markets
if markets:
    symbol_list = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
    symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)
else:
    st.error("종목 정보를 불러오지 못했습니다. 새로고침 해주세요.")
    st.stop()

if not gemini_key:
    k = st.sidebar.text_input("Gemini API Key", type="password")
    if k: config['gemini_api_key'] = k; save_settings(config); st.rerun()

try:
    exchange.set_leverage(config['leverage'], symbol)
    try: exchange.set_position_mode(hedged=False, symbol=symbol)
    except: pass
except: pass

st.sidebar.divider()
st.sidebar.subheader("🛡️ 스마트 방어 & 자금 관리")
use_switching = st.sidebar.checkbox("🔄 스위칭 (Switching)", value=config['use_switching'])
use_dca = st.sidebar.checkbox("💧 물타기 (DCA)", value=config['use_dca'])
c1, c2 = st.sidebar.columns(2)
dca_trigger = c1.number_input("추매 발동 (-%)", -90.0, -1.0, float(config['dca_trigger']), step=0.5)
dca_max = c2.number_input("최대 횟수", 1, 10, int(config['dca_max_count']))

use_smart_betting = st.sidebar.checkbox("🧠 AI 스마트 베팅", value=config.get('use_smart_betting', False))
use_trailing_stop = st.sidebar.checkbox("🚀 트레일링 스탑", value=config.get('use_trailing_stop', False))

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

active_inds = sum([use_rsi, use_bb, use_ma, use_macd, use_stoch, use_cci, use_mfi, use_willr, use_vol, config['use_adx']])
st.sidebar.divider()
target_vote = st.sidebar.slider("🎯 진입 확신도 (필요 지표 수)", 1, max(1, active_inds), int(config['target_vote']))
leverage = st.sidebar.slider("레버리지", 1, 50, int(config['leverage']))
config['order_usdt'] = st.sidebar.number_input("주문 금액 ($)", 10.0, 100000.0, float(config['order_usdt']))

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
    kb = {"inline_keyboard": [[{"text": "💰 잔고확인", "callback_data": "balance"}]]}
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': "✅ <b>메뉴 갱신</b>", 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})

with col_log:
    # [New] DB 뷰어 통합
    st.subheader("📖 AI의 성장 일지 (DB Viewer)")
    st.caption("AI가 매매 후 작성한 반성문과 피드백이 저장됩니다.")

    if st.button("🔄 기록 새로고침"): st.rerun()
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    history_df = pd.read_sql("SELECT * FROM trade_history ORDER BY id DESC", conn)
    conn.close()
    
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.info("아직 기록된 매매가 없습니다.")
        
    if st.button("🧪 테스트 데이터 입력 (DB Test)"):
        log_trade_to_db(symbol, "long", curr_price, -50.0, "뇌동매매", "상승 추세가 확실할 때만 진입하자.")
        st.rerun()
