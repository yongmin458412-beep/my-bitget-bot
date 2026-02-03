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
import sqlite3  # [New] DB 기능 추가
from datetime import datetime, timedelta
from openai import OpenAI
# [추가] 스레드 컨텍스트 오류 해결용
from streamlit.runtime.scriptrunner import add_script_run_ctx

# =========================================================
# ⚙️ [시스템 기본 설정]
# =========================================================
IS_SANDBOX = True # 실전 매매 시 False로 변경하세요!
SETTINGS_FILE = "bot_settings.json"
LOG_FILE = "trade_log.csv"
PROPOSALS_FILE = "pending_proposals.json"
DB_FILE = "wonyousi_brain.db" # [New] AI 기억 저장소

st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Ultimate Integration)")

# [추가] 감시 대상 코인 리스트 (UI와 봇이 공유)
TARGET_COINS = [
    "BTC/USDT:USDT", 
    "ETH/USDT:USDT", 
    "SOL/USDT:USDT", 
    "XRP/USDT:USDT", 
    "DOGE/USDT:USDT"
]

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
        "openai_key": "",
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

# =========================================================
# 🔐 [3. API & OpenAI 초기화] (이 부분을 통째로 교체하세요)
# =========================================================
api_key = st.secrets.get("API_KEY")
api_secret = st.secrets.get("API_SECRET")
api_password = st.secrets.get("API_PASSWORD")
tg_token = st.secrets.get("TG_TOKEN")
tg_id = st.secrets.get("TG_CHAT_ID")

# OpenAI 키 로드
openai_key = st.secrets.get("OPENAI_API_KEY", config.get("openai_api_key", ""))

# 비트겟 키 확인
if not api_key: 
    st.error("🚨 Bitget API Key가 설정되지 않았습니다.")
    st.stop()

# OpenAI 키 확인 및 연결
if not openai_key:
    st.error("🚨 OpenAI API Key가 없습니다. Secrets에 설정을 확인해주세요.")
    st.stop()
else:
    # 여기서 SyntaxError가 났던 부분입니다. 깔끔하게 다시 작성됨.
    openai_client = OpenAI(api_key=openai_key)

# =========================================================
# 🤖 [핵심] 멀티 코인 스나이퍼 봇 (24시간 감시 + 즉시 체결)
# =========================================================
def telegram_thread(ex, main_symbol):
    """
    기능:
    1. 주요 코인 5종(BTC, ETH, SOL, XRP, DOGE) 순차 감시
    2. '확실한 자리' 포착 시 15분 주기 무시하고 즉시 매매/보고
    3. 별일 없으면 15분마다 생존 신고 (요약 리포트)
    4. 텔레그램 버튼(전체 스캔, 잔고 등) 클릭 시 즉시 반응
    """
    
    # 1. 감시할 코인 리스트
    TARGET_COINS = [
        "BTC/USDT:USDT", 
        "ETH/USDT:USDT", 
        "SOL/USDT:USDT", 
        "XRP/USDT:USDT", 
        "DOGE/USDT:USDT"
    ]
    
    # 2. 메뉴 버튼 설정
    menu_kb = {
        "inline_keyboard": [
            [{"text": "🧠 전체 코인 스캔", "callback_data": "scan_all"}, {"text": "💰 내 잔고", "callback_data": "balance"}],
            [{"text": "📊 잡힌 포지션", "callback_data": "position"}, {"text": "📅 경제 캘린더", "callback_data": "calendar"}]
        ]
    }

    # 시작 알림
    start_msg = f"🚀 **워뇨띠 스나이퍼 봇 가동**\n감시 대상: {', '.join([c.split('/')[0] for c in TARGET_COINS])}\n24시간 감시를 시작합니다."
    try:
        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                      data={'chat_id': tg_id, 'text': start_msg, 'reply_markup': json.dumps(menu_kb), 'parse_mode': 'Markdown'})
    except Exception as e:
        print(f"Telegram Init Error: {e}")

    last_report_time = time.time()
    REPORT_INTERVAL = 900  # 15분 (정기 보고 주기)
    offset = 0

    while True:
        try:
            # 설정 파일 실시간 로드 (자동매매 ON/OFF 확인)
            try:
                cur_config = load_settings()
                is_auto_on = cur_config.get('auto_trade', False)
            except:
                is_auto_on = False

            # === [A] 24시간 순찰 (자동매매 켜져 있을 때만) ===
            if is_auto_on:
                for coin in TARGET_COINS:
                    try:
                        # 1. 데이터 조회 (빠르게)
                        ohlcv = ex.fetch_ohlcv(coin, '5m', limit=60)
                        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                        df, status, last = calc_indicators(df)

                        # 2. [비용 절약 필터] 특이사항 있을 때만 GPT 호출
                        is_interesting = (
                            last['RSI'] < 30 or last['RSI'] > 70 or 
                            last['ADX'] > 30 or 
                            "터치" in status.get('BB', '') or
                            "폭발" in status.get('Vol', '')
                        )

                        if is_interesting:
                            # 3. GPT-4o 정밀 분석
                            strategy = generate_wonyousi_strategy(df, status)
                            decision = strategy['decision']
                            conf = strategy.get('confidence', 0)

                            # 4. [즉시 행동] 확신도 70% 이상 매수/매도 시그널
                            if decision in ['buy', 'sell'] and conf >= 70:
                                # 주문 실행 로직 (필요시 주석 해제)
                                # ex.set_leverage(config['leverage'], coin)
                                # amt_usdt = config['order_usdt']
                                # price = last['close']
                                # qty = ex.amount_to_precision(coin, (amt_usdt * config['leverage']) / price)
                                
                                # if float(qty) > 0:
                                    # ex.create_market_order(coin, decision, qty) 
                                    
                                    # 즉시 알림 전송
                                    trade_msg = f"""
🚨 **[긴급 포착! 즉시 진입]**
코인: **{coin}**
포지션: **{decision.upper()}** (확신도 {conf}%)
💡 **이유:** {strategy.get('final_reason')}
"""
                                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                                  data={'chat_id': tg_id, 'text': trade_msg, 'parse_mode': 'Markdown'})
                                    
                                    # 매매 후 중복 진입 방지 대기
                                    time.sleep(10)

                    except Exception as e:
                        print(f"Scan Error ({coin}): {e}")
                    
                    # 코인 간 조회 간격 (API 제한 방지)
                    time.sleep(0.5)

            # === [B] 15분 정기 보고 (생존 신고) ===
            if time.time() - last_report_time > REPORT_INTERVAL:
                alive_msg = f"💤 **[15분 정기 점검]**\n현재 자동매매: {'ON' if is_auto_on else 'OFF'}\n특이사항 없음. 계속 감시합니다."
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                              data={'chat_id': tg_id, 'text': alive_msg, 'parse_mode': 'Markdown'})
                last_report_time = time.time()

            # === [C] 텔레그램 버튼/명령어 처리 ===
            try:
                # 타임아웃을 짧게 주어 루프 흐름을 방해하지 않음
                res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1").json()
                
                if res.get('ok'):
                    for up in res['result']:
                        offset = up['update_id']
                        
                        # 텍스트 메시지 처리
                        if 'message' in up and 'text' in up['message']:
                            txt = up['message']['text']
                            cid = up['message']['chat']['id']
                            if txt == "/start" or txt == "/menu":
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': "📋 **메뉴 호출**", 'reply_markup': json.dumps(menu_kb)})

                        # 버튼 클릭(Callback) 처리
                        if 'callback_query' in up:
                            cb = up['callback_query']; data = cb['data']; cid = cb['message']['chat']['id']
                            
                            if data == 'scan_all':
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': "🔍 전체 코인을 강제로 스캔합니다... (잠시만 기다려주세요)"})
                                
                                report = "🌍 **전체 코인 분석 결과**\n"
                                for coin in TARGET_COINS:
                                    try:
                                        ohlcv = ex.fetch_ohlcv(coin, '5m', limit=100)
                                        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                                        df, status, last = calc_indicators(df)
                                        res_ai = generate_wonyousi_strategy(df, status)
                                        
                                        icon = "🟢" if res_ai['decision'] == 'buy' else "🔴" if res_ai['decision'] == 'sell' else "⚪"
                                        report += f"{icon} **{coin.split('/')[0]}**: {res_ai['decision'].upper()} ({res_ai.get('confidence',0)}%)\n"
                                    except Exception as e:
                                        report += f"❌ {coin.split('/')[0]}: 오류\n"
                                
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': report, 'parse_mode': 'Markdown'})
                            
                            elif data == 'balance':
                                try:
                                    bal = ex.fetch_balance({'type': 'swap'})
                                    usdt = bal['USDT']['free']
                                    msg = f"💰 **보유 USDT:** ${usdt:,.2f}"
                                except: msg = "❌ 조회 실패"
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': msg, 'parse_mode': 'Markdown'})

                            elif data == 'position':
                                try:
                                    msg = "📊 **현재 포지션 현황**\n"
                                    has_pos = False
                                    for c in TARGET_COINS:
                                        poss = ex.fetch_positions([c])
                                        active = [p for p in poss if float(p['contracts']) > 0]
                                        if active:
                                            p = active[0]
                                            msg += f"- {c}: {p['side'].upper()} (수익 ${float(p['unrealizedPnl']):.2f})\n"
                                            has_pos = True
                                    if not has_pos: msg += "현재 무포지션입니다."
                                except: msg = "❌ 조회 실패"
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': msg, 'parse_mode': 'Markdown'})

                            elif data == 'calendar':
                                try:
                                    evs = get_forex_events()
                                    if evs.empty: msg = "📅 예정된 주요 경제 일정이 없습니다."
                                    else:
                                        msg = "📅 **주요 경제 일정**\n"
                                        for _, r in evs.iterrows(): msg += f"{r['시간']} | {r['지표']} ({r['중요도']})\n"
                                except: msg = "❌ 캘린더 오류"
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                                              data={'chat_id': cid, 'text': msg})

                            # 로딩바 제거
                            requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})

            except Exception as e:
                print(f"Telegram Update Error: {e}")
                time.sleep(1)

        except Exception as e:
            print(f"Bot Loop Error: {e}")
            time.sleep(5)
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
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키를 확인하거나 잠시 후 다시 시도하세요.")
    st.stop()

# ---------------------------------------------------------
# 🎨 사이드바 (설정 유지)
# ---------------------------------------------------------
st.sidebar.title("🛠️ AI 에이전트 제어판")
st.sidebar.info("설정을 변경하면 즉시 저장되고 알림이 뜹니다.")

markets = exchange.markets
if markets:
    symbol_list = [s for s in markets if markets[s].get('linear') and markets[s].get('swap')]
    symbol = st.sidebar.selectbox("코인 선택", symbol_list, index=0)
else:
    st.error("종목 정보를 불러오지 못했습니다. 새로고침 해주세요.")
    st.stop()

if not openai_key:
    k = st.sidebar.text_input("OpenAI API Key 입력", type="password")
    if k: 
        config['openai_api_key'] = k
        save_settings(config)
        st.rerun()
        
found = False
for t in threading.enumerate():
    if t.name == "TG_Thread": found = True; break
if not found:
    t = threading.Thread(target=telegram_thread, args=(exchange, symbol), daemon=True, name="TG_Thread")
    add_script_run_ctx(t) # 👈 [핵심] 이 줄을 추가하면 경고가 사라집니다!
    t.start()

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
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': "✅ <b>메뉴 갱신</b>", 'parse_mode': 'HTML', 'reply_markup': json.dumps(kb)})

# ---------------------------------------------------------
# 🧮 지표 계산 (기존 로직 유지)
# ---------------------------------------------------------
def calc_indicators(df):
    """10가지 기술적 지표 계산 및 상태 판단"""
    if df.empty: return df, {}, None

    close = df['close']; high = df['high']; low = df['low']; vol = df['vol']
    
    # --- [1. 지표 계산] ---
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

    # MACD
    k = close.ewm(span=12, adjust=False).mean()
    d = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Stochastic
    low_min = low.rolling(14).min()
    high_max = high.rolling(14).max()
    df['STOCH_K'] = 100 * ((close - low_min) / (high_max - low_min))

    # CCI
    tp = (high + low + close) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # ADX
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(14).mean()
    df['ADX'] = (atr / close) * 1000

    # Volume MA
    df['VOL_MA'] = vol.rolling(20).mean()

    # --- [2. 상태 판단 (Dashboard 표시용)] ---
    last = df.iloc[-1]
    status = {}
    
    # 1. RSI
    if config.get('use_rsi', True):
        if last['RSI'] <= config['rsi_buy']: status['RSI'] = "🟢 과매도"
        elif last['RSI'] >= config['rsi_sell']: status['RSI'] = "🔴 과매수"
        else: status['RSI'] = "⚪ 중립"
    
    # 2. BB
    if config.get('use_bb', True):
        if last['close'] <= last['BB_LO']: status['BB'] = "🟢 하단터치"
        elif last['close'] >= last['BB_UP']: status['BB'] = "🔴 상단터치"
        else: status['BB'] = "⚪ 밴드내"

    # 3. MA
    if config.get('use_ma', True):
        if last['MA_F'] > last['MA_S']: status['MA'] = "🟢 골든크로스"
        else: status['MA'] = "🔴 데드크로스"

    # 4. MACD
    if config.get('use_macd', True):
        if last['MACD'] > last['MACD_SIG']: status['MACD'] = "🟢 상승신호"
        else: status['MACD'] = "🔴 하락신호"

    # 5. Stochastic
    if config.get('use_stoch', True):
        if last['STOCH_K'] <= 20: status['Stoch'] = "🟢 저점"
        elif last['STOCH_K'] >= 80: status['Stoch'] = "🔴 고점"
        else: status['Stoch'] = "⚪ 중립"

    # 6. CCI
    if config.get('use_cci', True):
        if last['CCI'] <= -100: status['CCI'] = "🟢 과매도"
        elif last['CCI'] >= 100: status['CCI'] = "🔴 과매수"
        else: status['CCI'] = "⚪ 중립"

    # 7. Volume
    if config.get('use_vol', True):
        if last['vol'] > last['VOL_MA'] * 2.0: status['Vol'] = "🔥 거래량폭발"
        else: status['Vol'] = "⚪ 일반"

    # 8. ADX
    if config.get('use_adx', True):
        status['ADX'] = "📈 강한추세" if last['ADX'] > 25 else "🦀 횡보장"

    # [지표 상태판 코드 근처에 추가]
    # === [메인 UI 3: 10종 지표 상세 대시보드] ===
    with st.expander("📊 10종 보조지표 종합 상태판", expanded=True):
        cols = st.columns(5)
        idx = 0
        
        # 👇 [수정 1] 개수를 세기 위해 변수를 0으로 초기화합니다.
        active_cnt_l = 0
        active_cnt_s = 0
        
        for name, stat in status.items():
            color = "off"
            # 👇 [수정 2] 반복문을 돌면서 매수/매도 개수를 셉니다.
            if "매수" in stat: 
                color = "normal"
                active_cnt_l += 1
            elif "매도" in stat: 
                color = "inverse"
                active_cnt_s += 1
                
            cols[idx % 5].metric(name, stat, delta_color=color)
            idx += 1
    
        # 👇 [수정 3] 다 세어진 개수를 화면에 표시합니다.
        st.caption("💡 **범례:** 🟢 매수신호(Buy) | 🔴 매도신호(Sell) | ⚪ 중립(Neutral)")
        st.caption(f"🎯 **종합 집계:** 매수 신호 **{active_cnt_l}개** / 매도 신호 **{active_cnt_s}개**")
        
    return df, status, last

# 👇 [여기서부터 복사] calc_indicators 함수 바로 밑에 붙여넣으세요!

def generate_wonyousi_strategy(df, status_summary):
    """OpenAI GPT-4o를 이용한 정밀 분석 (연결 보장형)"""
    
    # 1. 함수 안에서 직접 키를 가져와서 연결 (오류 방지)
    try:
        my_key = st.secrets.get("OPENAI_API_KEY")
        if not my_key:
            return {"decision": "hold", "final_reason": "API Key 설정 오류", "confidence": 0}
        client = OpenAI(api_key=my_key)
    except Exception as e:
        return {"decision": "hold", "final_reason": f"OpenAI 연결 실패: {e}", "confidence": 0}

    # 2. 데이터 준비
    # (만약 get_past_mistakes 함수가 없다면 빈 리스트 처리)
    try: past_mistakes = get_past_mistakes()
    except: past_mistakes = "없음"
    
    last_row = df.iloc[-1]
    
    system_msg = """
    당신은 전설적인 트레이더 '워뇨띠'입니다.
    - 캔들 패턴, 거래량, 추세를 최우선으로 분석합니다.
    - 확실하지 않으면 '관망(hold)'하세요.
    - 응답은 반드시 JSON 형식이어야 합니다.
    """
    
    user_msg = f"""
    [시장 데이터]
    - 현재가: {last_row['close']}
    - RSI: {last_row['RSI']:.1f}
    - 볼린저밴드: {status_summary.get('BB', 'Normal')}
    - ADX: {last_row['ADX']:.1f}
    - 매수/매도 신호: {status_summary}
    
    [과거 실수]
    {past_mistakes}
    
    매매 판단을 JSON으로 주세요.
    Key: decision(buy/sell/hold), reason_trend, reason_candle, final_reason, confidence(int)
    """
    
    # 3. AI에게 질문
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        return {"decision": "hold", "final_reason": f"분석 중 에러: {e}", "confidence": 0}

# 👆 [여기까지 복사]


# ---------------------------------------------------------
# 📅 데이터 수집 (ForexFactory) - UI 표시용 함수 (복구)
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
        
# ---------------------------------------------------------
# 📊 메인 화면 (UI 통합)
# ---------------------------------------------------------
# [이 코드로 덮어씌우세요]
# [데이터 로딩 부분 수정]
try:
    # 1. 시세 조회
    ticker = exchange.fetch_ticker(symbol)
    curr_price = ticker['last']
    
    # 2. 캔들 데이터 조회
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=200)
    
    # 3. 데이터프레임 변환
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # 4. 지표 계산 (변수명을 status로 통일!)
    df, status, last = calc_indicators(df)  # 👈 여기가 핵심입니다! (ind_status -> status)

except Exception as e:
    st.error(f"🚨 데이터 로딩 실패! 원인: {e}")
    st.stop()

# 1. 추세 모드 판단 로직 (이 줄이 빠져서 에러가 난 것입니다)
# ADX가 25 이상이면 추세장, 아니면 횡보장으로 판단
is_trend_mode = last['ADX'] >= 25 

# 2. 모드 이름 설정
mode_str = "🌊 추세장 (강한 상승/하락)" if is_trend_mode else "🦀 횡보장 (박스권)"

# 3. 타이틀 출력
st.title(f"🔥 {symbol} GPT-4o Trader")
st.caption(f"모드: {mode_str} | 현재가: ${curr_price:,.2f}")
    
is_trend_mode = last['ADX'] >= 25 and config['use_dual_mode']

# === [메인 UI 3: 10종 지표 상세 대시보드] ===
with st.expander("📊 지표 상태판 (Indicator Dashboard)", expanded=True):
    cols = st.columns(5)
    idx = 0
    
    # 개수 세기 초기화
    active_cnt_l = 0
    active_cnt_s = 0
    
    # 👇 [핵심 수정] ind_status를 status로 변경했습니다!
    for name, stat in status.items():
        color = "off"
        if "매수" in stat: 
            color = "normal"
            active_cnt_l += 1
        elif "매도" in stat: 
            color = "inverse"
            active_cnt_s += 1
            
        cols[idx % 5].metric(name, stat, delta_color=color)
        idx += 1

    st.caption("💡 **범례:** 🟢 매수신호(Buy) | 🔴 매도신호(Sell) | ⚪ 중립(Neutral)")
    st.caption(f"🎯 **종합 집계:** 매수 신호 **{active_cnt_l}개** / 매도 신호 **{active_cnt_s}개**")
    
h = 450
tv_studies = ["RSI@tv-basicstudies", "BB@tv-basicstudies", "MASimple@tv-basicstudies"]
studies_json = str(tv_studies).replace("'", '"')
tv = f"""<div class="tradingview-widget-container"><div id="tradingview_chart"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{ "width": "100%", "height": {h}, "symbol": "BITGET:{symbol.replace('/','').split(':')[0]}.P", "interval": "5", "theme": "dark", "studies": {studies_json}, "container_id": "tradingview_chart" }});</script></div>"""
components.html(tv, height=h)

# 4개의 탭으로 확장 (새 기능 포함)
t1, t2, t3, t4 = st.tabs(["🤖 자동매매 & AI분석", "⚡ 수동주문", "📅 시장정보", "📜 매매일지(DB)"])

# [수정할 위치: 탭1(t1) 내부의 수동 분석 버튼 코드]

with t1:
    st.subheader("🧠 워뇨띠 AI 전략 센터")
    
    # 자동매매 스위치
    c_auto, c_stat = st.columns([3, 1])
    with c_auto:
        auto_on = st.checkbox("🤖 24시간 자동매매 활성화 (텔레그램 연동)", value=config.get('auto_trade', False))
        if auto_on != config.get('auto_trade', False):
            config['auto_trade'] = auto_on
            save_settings(config)
            st.rerun()
    with c_stat:
        st.caption("상태: " + ("🟢 가동중" if auto_on else "🔴 정지"))

    st.divider()

    # 👇 [수정됨] 버튼을 2개로 분리 (컬럼 활용)
    col_btn1, col_btn2 = st.columns(2)

    # 버튼 1: 현재 차트만 분석
    if col_btn1.button("🔍 현재 차트 분석 (This Coin)"):
        with st.spinner(f"'{symbol}' 차트를 정밀 분석 중입니다..."):
            try:
                ai_res = generate_wonyousi_strategy(df, status)
                
                decision = ai_res.get('decision', 'hold').upper()
                conf = ai_res.get('confidence', 0)
                reason = ai_res.get('final_reason', ai_res.get('reason', '알 수 없음'))

                if decision == 'BUY':
                    st.success(f"결론: 🟢 **매수 (BUY)** (확신도 {conf}%)")
                elif decision == 'SELL':
                    st.error(f"결론: 🔴 **매도 (SELL)** (확신도 {conf}%)")
                else:
                    st.warning(f"결론: ⚪ **관망 (HOLD)** (확신도 {conf}%)")
                
                st.info(f"💡 **근거:** {reason}")
            except Exception as e:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")

    # 버튼 2: 전체 코인 스캔
    if col_btn2.button("🌍 전체 코인 스캔 (All Coins)"):
        status_placeholder = st.empty()
        status_placeholder.info("🕵️ 5개 코인을 순차적으로 분석 중... (약 10~20초 소요)")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, coin in enumerate(TARGET_COINS):
            try:
                # 데이터 가져오기
                ohlcv_t = exchange.fetch_ohlcv(coin, '5m', limit=100)
                df_t = pd.DataFrame(ohlcv_t, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df_t['time'] = pd.to_datetime(df_t['time'], unit='ms')
                df_t, stat_t, last_t = calc_indicators(df_t)
                
                # AI 분석
                res = generate_wonyousi_strategy(df_t, stat_t)
                
                # 결과 저장
                results.append({
                    "코인": coin.split('/')[0],
                    "현재가": f"${last_t['close']:,.2f}",
                    "결론": res['decision'].upper(),
                    "확신도": f"{res.get('confidence',0)}%",
                    "근거": res.get('final_reason', '요약 불가')[:30] + "..." # 너무 길어서 자름
                })
            except Exception as e:
                results.append({"코인": coin, "결론": "Error", "근거": str(e)})
            
            progress_bar.progress((i + 1) / len(TARGET_COINS))
        
        status_placeholder.success("✅ 전체 스캔 완료!")
        st.dataframe(pd.DataFrame(results))
        
        with t2:
    st.write("✋ **수동 컨트롤**")
    m_amt = st.number_input("주문 금액 ($)", 0.0, 100000.0, float(config['order_usdt']))
    b1, b2, b3 = st.columns(3)
    if b1.button("🟢 롱 진입"): pass
    if b2.button("🔴 숏 진입"): pass
    if b3.button("🚫 포지션 종료"): pass

with t3:
    st.write("📅 **경제 일정**")
    ev = get_forex_events()
    if not ev.empty: st.dataframe(ev)
    else: st.write("일정 없음")

with t4:
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

# [여기서부터 파일 맨 끝에 추가하세요]
# ---------------------------------------------------------
# 🔍 [디버깅] 사이드바 맨 아래 OpenAI 연결 테스트 버튼
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🔍 긴급 점검")

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    try:
        # 1. 키 확인
        if not openai_key:
            st.sidebar.error("❌ API 키가 없습니다. secrets.toml을 확인하세요.")
        else:
            # 2. 간단한 인사 요청
            test_client = OpenAI(api_key=openai_key)
            response = test_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                max_tokens=10
            )
            ans = response.choices[0].message.content
            st.sidebar.success(f"✅ 연결 성공!\n응답: {ans}")
            
    except Exception as e:
        # 에러 내용을 붉은색으로 자세히 보여줌
        st.sidebar.error(f"❌ 연결 실패!\n원인: {e}")
        
        # 자주 발생하는 에러 친절 설명
        if "insufficient_quota" in str(e):
            st.sidebar.warning("💰 잔고 부족! OpenAI API 설정 페이지에서 'Credit Balance'를 충전해야 합니다. (ChatGPT Plus 결제와는 다릅니다)")
        elif "invalid_api_key" in str(e):
            st.sidebar.warning("🔑 키 오류! sk-로 시작하는 키가 맞는지, 공백은 없는지 확인하세요.")
