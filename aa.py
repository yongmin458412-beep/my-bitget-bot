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
# =========================================================
# 📝 매매 일지 시스템 (CSV 저장 + AI 피드백)
# =========================================================
LOG_FILE = "trade_log.csv"

def log_trade(coin, side, entry_price, exit_price, pnl_amount, pnl_percent, reason):
    """매매 종료 시 기록을 남깁니다."""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([{
            "Time": now,
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "Reason": reason
        }])
        
        # 파일이 없으면 새로 만들고, 있으면 이어붙이기
        if not os.path.exists(LOG_FILE):
            new_data.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Log Error: {e}")

def get_past_mistakes():
    """AI에게 '너 지난번에 이렇게 잃었어'라고 알려줄 데이터를 가져옵니다."""
    try:
        if not os.path.exists(LOG_FILE): return "과거 매매 기록 없음."
        
        df = pd.read_csv(LOG_FILE)
        # 손실이 가장 컸던(수익률이 낮은) 순서대로 5개 추출
        worst_trades = df.sort_values(by='PnL_Percent', ascending=True).head(5)
        
        summary = ""
        for _, row in worst_trades.iterrows():
            summary += f"- {row['Coin']} {row['Side']} 진입했다가 {row['PnL_Percent']}% 손실 (이유: {row.get('Reason', '기록없음')})\n"
        
        return summary if summary else "큰 손실 기록 없음."
    except:
        return "기록 조회 실패"
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
    [수정됨] 
    1. '전체 스캔' 클릭 시 무응답 버그 수정 (결과 취합 후 전송)
    2. 포지션/잔고 조회 기능 유지
    3. 자동매매(AI 스나이퍼) 로직 유지
    """
    
    menu_kb = {
        "inline_keyboard": [
            [{"text": "📊 내 포지션", "callback_data": "position"}, {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "🌍 전체 스캔 (AI)", "callback_data": "scan_all"}, {"text": "🛑 긴급 청산", "callback_data": "close_all"}]
        ]
    }

    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                  data={'chat_id': tg_id, 'text': "🤖 **AI 봇 재가동**\n스트림릿 사이드바에서 잔고 확인이 가능합니다.", 'reply_markup': json.dumps(menu_kb)})

    active_trades = {} 
    last_report_time = time.time()
    REPORT_INTERVAL = 900
    offset = 0

    while True:
        try:
            cur_config = load_settings()
            is_auto_on = cur_config.get('auto_trade', False)
            
            # [A] 🛡️ 자동매매 감시 로직 (기존과 동일)
            if is_auto_on:
                # ... (이전 답변의 자동매매 코드가 길어서 생략하지만, 실제로는 여기에 있어야 합니다) ...
                # ... (코드가 너무 길어지므로, 핵심인 '전체 스캔' 수정 부분만 아래 [C]에서 강조합니다) ...
                
                # (자동매매 로직은 이미 적용되어 있다고 가정하고, 아래 [C] 부분만 확실히 바꿉니다)
                pass 

            # [C] 버튼 및 명령어 처리
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1").json()
            if res.get('ok'):
                for up in res['result']:
                    offset = up['update_id']
                    if 'callback_query' in up:
                        cb = up['callback_query']; data = cb['data']; cid = cb['message']['chat']['id']
                        
                        # 🔥 [수정 완료] 전체 스캔 로직
                        if data == 'scan_all':
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': "🕵️ **전체 코인 정밀 분석 시작...**\n(약 10~20초 소요됩니다. 잠시만 기다려주세요.)"})
                            
                            report_msg = "🌍 **AI 시장 분석 결과**\n"
                            
                            for coin in TARGET_COINS:
                                try:
                                    # 데이터 조회
                                    ohlcv = ex.fetch_ohlcv(coin, '5m', limit=60)
                                    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                                    df, status, last = calc_indicators(df)
                                    
                                    # AI 분석 (간략화된 버전 혹은 풀버전)
                                    # 여기서는 풀버전 사용
                                    strategy = generate_wonyousi_strategy(df, status)
                                    
                                    decision = strategy.get('decision', 'hold').upper()
                                    conf = strategy.get('confidence', 0)
                                    
                                    icon = "⚪"
                                    if decision == 'BUY': icon = "🟢"
                                    elif decision == 'SELL': icon = "🔴"
                                    
                                    report_msg += f"{icon} **{coin.split('/')[0]}**: {decision} ({conf}%)\n"
                                    
                                except Exception as e:
                                    report_msg += f"⚠️ {coin.split('/')[0]}: 분석 실패\n"
                                
                                # API 제한 방지
                                time.sleep(1)
                            
                            # 최종 결과 전송
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': report_msg, 'parse_mode': 'Markdown'})

                        elif data == 'balance':
                            try:
                                bal = ex.fetch_balance({'type': 'swap'})
                                msg = f"💰 **내 지갑 현황**\n총 자산: ${bal['USDT']['total']:,.2f}\n주문 가능: ${bal['USDT']['free']:,.2f}"
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': msg, 'parse_mode': 'Markdown'})
                            except: pass

                        elif data == 'position':
                            # (이전 코드 유지)
                            msg = "📊 **포지션 현황**\n"
                            try:
                                has_pos = False
                                for c in TARGET_COINS:
                                    ps = ex.fetch_positions([c])
                                    p = [x for x in ps if float(x['contracts']) > 0]
                                    if p:
                                        has_pos = True
                                        side = "Long" if p[0]['side']=='long' else "Short"
                                        roe = float(p[0]['percentage'])
                                        msg += f"- {c}: {side} ({roe:.2f}%)\n"
                                if not has_pos: msg += "현재 무포지션"
                            except: msg = "조회 실패"
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': msg})

                        elif data == 'close_all':
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': "🛑 긴급 청산 실행!"})
                            # (청산 로직은 이전과 동일)
                            for c in TARGET_COINS:
                                try:
                                    ps = ex.fetch_positions([c])
                                    if ps and float(ps[0]['contracts']) > 0:
                                        ex.create_market_order(c, 'sell' if ps[0]['side']=='buy' else 'buy', ps[0]['contracts'])
                                except: pass
                        
                        requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})
            
            time.sleep(1)

        except Exception as e:
            print(f"TG Loop Error: {e}")
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
    """
    보조지표 계산 함수 (에러 방지 강화판)
    """
    try:
        # 데이터가 없으면 바로 안전하게 리턴
        if df is None or df.empty or len(df) < 20:
            return df, {}, None

        # 1. RSI (14)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        # 2. 볼린저밴드 (20, 2)
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()

        # 3. MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # 4. 이동평균선 (SMA)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_60'] = ta.trend.sma_indicator(df['close'], window=60)

        # 5. 스토캐스틱
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()

        # 6. ADX (추세 강도)
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        # 7. CCI
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)

        # 8. Williams %R
        df['Williams'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

        # 9. 파라볼릭 SAR
        df['SAR'] = ta.trend.psar_down(df['high'], df['low'], df['close'])

        # 10. OBV (거래량)
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['vol'])

        # --- 상태 평가 ---
        last = df.iloc[-1]
        status = {}

        # RSI
        if last['RSI'] > 70: status['RSI'] = "🔴 과매수"
        elif last['RSI'] < 30: status['RSI'] = "🟢 과매도"
        else: status['RSI'] = "⚪ 중립"

        # 볼린저밴드
        if last['close'] > last['BB_upper']: status['BB'] = "🔴 상단 터치"
        elif last['close'] < last['BB_lower']: status['BB'] = "🟢 하단 터치"
        else: status['BB'] = "⚪ 밴드 내"

        # 이동평균선
        if last['close'] > last['SMA_20'] > last['SMA_60']: status['MA'] = "🚀 정배열"
        elif last['close'] < last['SMA_20'] < last['SMA_60']: status['MA'] = "📉 역배열"
        else: status['MA'] = "⚠️ 혼조세"

        # MACD
        if last['MACD'] > last['MACD_signal']: status['MACD'] = "📈 골든크로스"
        else: status['MACD'] = "📉 데드크로스"
        
        # 거래량 (OBV) - 간단한 전일 대비
        if len(df) > 1 and df.iloc[-1]['OBV'] > df.iloc[-2]['OBV']:
            status['Vol'] = "🔥 매수세 유입"
        else:
            status['Vol'] = "💧 매도세 우위"

        return df, status, last

    except Exception as e:
        print(f"Indicator Error: {e}")
        # 🔥 여기가 핵심: 에러가 나도 3개를 반드시 돌려줌
        return df, {}, None

    # [추가] 경제 캘린더 크롤링 함수
def get_forex_events():
    """
    네이버 금융/인베스팅닷컴 등에서 주요 경제 일정을 가져오는 함수 (에러 시 빈 데이터 반환)
    """
    try:
        # 간단한 예시로, 실제 크롤링 대신 현재 시간 기준 가짜 데이터를 반환하거나 
        # 혹은 외부 라이브러리가 필요 없는 안전한 빈 DataFrame을 반환하여 에러를 막습니다.
        # (실제 크롤링 코드는 복잡하고 사이트 구조 변경에 취약하므로, 일단 에러 방지용 코드를 넣습니다)
        
        # 만약 실제 크롤링 코드를 원하시면 requests/BeautifulSoup이 필요합니다.
        # 여기서는 에러를 막기 위해 '일정 없음' 상태로 반환합니다.
        df = pd.DataFrame(columns=['날짜', '시간', '지표', '중요도'])
        return df
    except Exception as e:
        print(f"Calendar Error: {e}")
        return pd.DataFrame()
    
def generate_wonyousi_strategy(df, status_summary):
    """
    [전략 수정: 스윙/반등 확인형]
    1. 과매도/과매수 해소 시점(Reversal) 포착
    2. 레버리지 축소 + 손절폭 확대 (개미털기 방지)
    3. 확신도 기준 상향
    """
    try:
        my_key = st.secrets.get("OPENAI_API_KEY")
        if not my_key: return {"decision": "hold", "confidence": 0}
        client = OpenAI(api_key=my_key)
    except: return {"decision": "hold", "confidence": 0}

    # 최근 데이터 2개를 가져와서 추세 변화를 봅니다.
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    past_mistakes = get_past_mistakes()

    system_prompt = f"""
    당신은 신중한 '스윙 트레이더'입니다.
    
    [과거 실수]
    {past_mistakes}
    
    [핵심 전략]
    1. **진입 타이밍:** 과매도(RSI 30)나 과매수(RSI 70) 구간에 '진입'하는 게 아니라, 그 구간을 **'탈출할 때(반등)'** 진입하세요. (떨어지는 칼날 잡기 금지)
    2. **손절/익절:** 세력의 노이즈(휩소)를 견딜 수 있게 손절폭(sl_gap)을 넉넉히 잡으세요. (최소 2.5% 이상)
    3. **레버리지:** 손절폭이 넓으므로 레버리지는 **3~10배**로 낮게 잡으세요. 20배는 금지입니다.
    
    [응답 형식 (JSON)]
    {{
        "decision": "buy" / "sell" / "hold",
        "percentage": 10~30,
        "leverage": 3~10 (저배율 권장),
        "sl_gap": 2.5~6.0 (넉넉한 손절폭),
        "tp_gap": 5.0~15.0 (큰 익절폭),
        "confidence": 0~100,
        "reason": "타이밍과 손익비에 대한 상세 근거"
    }}
    """
    
    user_prompt = f"""
    [시장 데이터 흐름]
    - 현재가: {last_row['close']}
    - RSI 흐름: {prev_row['RSI']:.1f} -> {last_row['RSI']:.1f} (반등 중인지 확인!)
    - ADX: {last_row['ADX']:.1f}
    - 볼린저밴드: {status_summary.get('BB', '중간')}
    
    RSI가 극단적 수치에서 돌아오고 있나요? 확실한 반전 신호가 아니면 80점 이상 주지 마세요.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.3 # 매우 냉철하게
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"decision": "hold", "final_reason": f"에러: {e}", "confidence": 0}
        

# 👇 [여기서부터 복사] calc_indicators 함수 바로 밑에 붙여넣으세요!

def telegram_thread(ex, main_symbol):
    """
    [강화된 봇]
    1. 확신도 80%/85% 컷트라인 적용
    2. 강제 손절폭 확대 (최소 2.5%)
    3. 과매도/과매수 해소 감지
    """
    
    menu_kb = {
        "inline_keyboard": [
            [{"text": "📊 포지션 현황", "callback_data": "position"}, {"text": "💰 잔고 조회", "callback_data": "balance"}],
            [{"text": "🌍 전체 스캔", "callback_data": "scan_all"}, {"text": "🛑 긴급 청산", "callback_data": "close_all"}]
        ]
    }

    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", 
                  data={'chat_id': tg_id, 'text': "🛡️ **신중한 워뇨띠 모드 ON**\n- 신규 80% / 추가 85% 이상만 진입\n- 과매도/과매수 '해소' 시점 공략\n- 손절폭 2.5% 이상 강제", 'reply_markup': json.dumps(menu_kb)})

    active_trades = {} 
    last_report_time = time.time()
    REPORT_INTERVAL = 900
    offset = 0

    while True:
        try:
            cur_config = load_settings()
            is_auto_on = cur_config.get('auto_trade', False)
            
            if is_auto_on:
                # 1. 진입 장벽 설정 (80% / 85%)
                active_pos_count = 0
                for c in TARGET_COINS:
                    try:
                        p = ex.fetch_positions([c])
                        if any(float(x['contracts']) > 0 for x in p): active_pos_count += 1
                    except: pass
                
                # 🔥 [핵심] 컷트라인 상향 조정
                required_conf = 85 if active_pos_count >= 1 else 80

                for coin in TARGET_COINS:
                    try:
                        # === 포지션 관리 (청산 로직) ===
                        positions = ex.fetch_positions([coin])
                        active_ps = [p for p in positions if float(p['contracts']) > 0]
                        
                        if active_ps:
                            p = active_ps[0]
                            entry = float(p['entryPrice'])
                            side = p['side']
                            pnl_pct = float(p['percentage'])
                            pnl_usdt = float(p['unrealizedPnl'])
                            
                            # 메모리에서 목표가 가져오기 (없으면 안전하게 넓은 범위 적용)
                            target_info = active_trades.get(coin, {'sl': -4.0, 'tp': 8.0}) 
                            
                            if pnl_pct <= -abs(target_info['sl']):
                                ex.create_market_order(coin, 'sell' if side=='buy' else 'buy', p['contracts'])
                                log_trade(coin, side, entry, float(ex.fetch_ticker(coin)['last']), pnl_usdt, pnl_pct, "자동 손절")
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': f"🩸 **[손절]** {coin} ({pnl_pct:.2f}%)"})
                                if coin in active_trades: del active_trades[coin]
                            
                            elif pnl_pct >= target_info['tp']:
                                ex.create_market_order(coin, 'sell' if side=='buy' else 'buy', p['contracts'])
                                log_trade(coin, side, entry, float(ex.fetch_ticker(coin)['last']), pnl_usdt, pnl_pct, "자동 익절")
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': f"🎉 **[익절]** {coin} (+{pnl_pct:.2f}%)"})
                                if coin in active_trades: del active_trades[coin]
                            continue 
                        
                        # === 신규 진입 분석 ===
                        ohlcv = ex.fetch_ohlcv(coin, '5m', limit=60)
                        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                        df, status, last = calc_indicators(df)
                        
                        # 🔥 [필터 수정] RSI가 30/70 근처였다가 돌아오는 경우에만 AI 호출 (비용 절약 + 타이밍)
                        # 단순히 과열이라고 부르지 않고, 변동성이 있을 때 부름
                        if 30 <= last['RSI'] <= 70 and last['ADX'] < 20:
                            continue # 애매한 횡보장은 패스

                        strategy = generate_wonyousi_strategy(df, status)
                        decision = strategy.get('decision', 'hold')
                        conf = strategy.get('confidence', 0)
                        
                        # 컷트라인 통과 확인
                        if decision in ['buy', 'sell'] and conf >= required_conf:
                            
                            lev = int(strategy.get('leverage', 5))
                            pct = float(strategy.get('percentage', 10))
                            sl = float(strategy.get('sl_gap', 3.0)) # 기본값도 3.0으로 상향
                            tp = float(strategy.get('tp_gap', 6.0))
                            
                            # 🛡️ 강제 안전장치: 손절폭이 2.5%보다 작으면 강제로 2.5%로 늘림
                            if sl < 2.5: sl = 2.5 
                            
                            # 레버리지 안전장치: 10배 초과 금지 (사용자 요청 반영)
                            if lev > 10: lev = 10

                            try: ex.set_leverage(lev, coin)
                            except: pass
                            
                            bal = ex.fetch_balance({'type': 'swap'})
                            amt = float(bal['USDT']['free']) * (pct / 100.0)
                            price = last['close']
                            qty = ex.amount_to_precision(coin, (amt * lev) / price)
                            
                            if float(qty) > 0:
                                ex.create_market_order(coin, decision, qty)
                                active_trades[coin] = {'sl': sl, 'tp': tp}
                                
                                msg = f"""
🎯 **[AI 정밀 타격]** {coin}
진입: **{decision.upper()}** (확신도 {conf}%)
레버리지: x{lev}
목표: +{tp}% / -{sl}%
근거: {strategy.get('reason')}
"""
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': msg, 'parse_mode': 'Markdown'})
                                time.sleep(10)
                                
                    except Exception as e:
                        print(f"Scan Err ({coin}): {e}")
                    time.sleep(1)

            # [B] 정기 보고 (기존 코드 유지)
            if time.time() - last_report_time > REPORT_INTERVAL:
                try:
                    bal = ex.fetch_balance({'type': 'swap'})
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': tg_id, 'text': f"💤 생존 신고 (자산: ${bal['USDT']['total']:,.2f})"})
                    last_report_time = time.time()
                except: pass

            # [C] 버튼 처리 (기존과 동일하므로 생략하지 않고 핵심만 유지)
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates?offset={offset+1}&timeout=1").json()
            if res.get('ok'):
                for up in res['result']:
                    offset = up['update_id']
                    if 'callback_query' in up:
                        cb = up['callback_query']; data = cb['data']; cid = cb['message']['chat']['id']
                        
                        if data == 'position':
                            # (포지션 조회 코드 - 위 답변과 동일)
                            msg = "📊 **포지션 현황**\n"
                            has = False
                            for c in TARGET_COINS:
                                try:
                                    ps = ex.fetch_positions([c])
                                    p = [x for x in ps if float(x['contracts'])>0]
                                    if p:
                                        msg += f"{c}: {p[0]['side']} (수익 {float(p[0]['percentage']):.2f}%)\n"
                                        has = True
                                except: pass
                            if not has: msg += "없음"
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': msg})
                        
                        elif data == 'balance':
                            try:
                                bal = ex.fetch_balance({'type': 'swap'})
                                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': f"💰 잔고: ${bal['USDT']['total']:,.2f}"})
                            except: pass

                        elif data == 'close_all':
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={'chat_id': cid, 'text': "🛑 전량 청산!"})
                            for c in TARGET_COINS:
                                try:
                                    ps = ex.fetch_positions([c])
                                    if ps and float(ps[0]['contracts']) > 0:
                                        ex.create_market_order(c, 'sell' if ps[0]['side']=='buy' else 'buy', ps[0]['contracts'])
                                except: pass
                                
                        requests.post(f"https://api.telegram.org/bot{tg_token}/answerCallbackQuery", data={'callback_query_id': cb['id']})
            time.sleep(1)

        except Exception as e:
            print(f"Main Err: {e}")
            time.sleep(5)
            
# 👆 [여기까지 복사]
# =========================================================
# [메인 UI 0] 사이드바 설정 (여기가 제일 위에 있어야 함!)
# =========================================================
st.title("🤖 워뇨띠의 매매노트 (Bitget AI Bot)")

with st.sidebar:
    st.header("⚙️ 기본 설정")
    # 🔥 [핵심 수정] 여기서 timeframe을 먼저 만들어야 에러가 안 납니다.
    symbol = st.text_input("코인 심볼 (티커)", value="BTC/USDT:USDT")
    timeframe = st.selectbox("시간봉 선택", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=2) 
    
    st.divider()
    # (나머지 사이드바 코드들... 잔고 조회 등)

# =========================================================
# [메인 로직] 데이터 로딩 (설정이 끝난 뒤에 실행)
# =========================================================
df = None
status = {}
last = None

try:
    # 위에서 만든 timeframe 변수를 여기서 사용합니다.
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    
    # ... (이하 데이터 처리 코드 동일)

# =========================================================
# [메인 로직] 데이터 로딩 및 처리
# =========================================================
# 1. 변수 초기화 (NameError 방지용 핵심 코드!)
df = None
status = {}
last = None

try:
    # 2. OHLCV 데이터 가져오기
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    
    if not ohlcv:
        st.error("🚨 데이터 로딩 실패: 거래소에서 차트 데이터를 가져오지 못했습니다.")
    else:
        # 3. 데이터프레임 변환
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')

        # 4. 보조지표 계산
        df, status, last = calc_indicators(df)

except Exception as e:
    st.error(f"🚨 시스템 오류 발생: {e}")
    print(f"Main Logic Error: {e}")


# =========================================================
# [메인 UI 1] 시장 데이터 브리핑 (Dashboard)
# =========================================================
st.subheader(f"📊 {symbol} 실시간 현황")

# 🔥 이제 last가 None이어도 에러가 나지 않습니다.
if last is not None:
    # 1. 추세 판단 (ADX 기준)
    is_trend = last['ADX'] >= 25
    trend_str = "🔥 강력한 추세장" if is_trend else "💤 지루한 횡보장"
    
    # 2. 4단 컬럼 데이터 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("현재가 (Price)", f"${last['close']:,.2f}")
    
    with col2:
        rsi_val = last['RSI']
        # RSI 색상 처리
        rsi_color = "normal"
        if rsi_val > 70: rsi_color = "inverse" # 빨강
        elif rsi_val < 30: rsi_color = "off"     # 초록/회색
        st.metric("RSI (강도)", f"{rsi_val:.1f}", delta=status.get('RSI'), delta_color=rsi_color)
        
    with col3:
        adx_val = last['ADX']
        st.metric("ADX (추세)", f"{adx_val:.1f}", delta=trend_str)
        
    with col4:
        # 볼린저밴드 위치
        bb_width = last['BB_upper'] - last['BB_lower']
        if bb_width > 0:
            bb_pos = (last['close'] - last['BB_lower']) / bb_width
            st.metric("BB 위치", f"{bb_pos*100:.0f}%", delta=status.get('BB'))
        else:
            st.metric("BB 위치", "계산 불가")

else:
    # 데이터가 로딩되지 않았을 때
    st.warning("⚠️ 차트 데이터를 불러오는 중이거나, 데이터가 부족합니다.")
    
# =========================================================
# [메인 UI 3] 10종 지표 종합 요약 (심플 버전)
# =========================================================
st.divider()

# 1. 매수/매도 개수 계산
active_cnt_l = 0
active_cnt_s = 0
for _, stat in status.items():
    if "매수" in stat: active_cnt_l += 1
    elif "매도" in stat: active_cnt_s += 1

# 2. 종합 점수 및 디자인 설정
total_score = active_cnt_l - active_cnt_s

if total_score >= 3:
    sentiment = "🚀 매수 우위"
    bg_color = "#d4edda"; text_color = "#155724"; border_color = "#c3e6cb"
elif total_score >= 1:
    sentiment = "📈 약한 매수"
    bg_color = "#e2e6ea"; text_color = "#0c5460"; border_color = "#bee5eb"
elif total_score <= -3:
    sentiment = "📉 매도 우위"
    bg_color = "#f8d7da"; text_color = "#721c24"; border_color = "#f5c6cb"
elif total_score <= -1:
    sentiment = "🔻 약한 매도"
    bg_color = "#fff3cd"; text_color = "#856404"; border_color = "#ffeeba"
else:
    sentiment = "⚖️ 중립 (관망)"
    bg_color = "#f8f9fa"; text_color = "#383d41"; border_color = "#d6d8db"

# 3. [수정됨] 폰트 크기를 줄인 컴팩트 배너
st.markdown(f"""
<div style="
    padding: 10px; 
    border-radius: 8px; 
    background-color: {bg_color}; 
    color: {text_color}; 
    border: 1px solid {border_color};
    text-align: center;
    margin-bottom: 10px;">
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{sentiment}</div>
    <div style="font-size: 13px;">
        매수 시그널 <b>{active_cnt_l}</b>개 vs 매도 시그널 <b>{active_cnt_s}</b>개
    </div>
</div>
""", unsafe_allow_html=True)

# 4. 상세 내역은 '접어두기'로 숨김 (필요할 때만 클릭)
with st.expander("🔍 지표 상세 확인하기"):
    cols = st.columns(5)
    idx = 0
    for name, stat in status.items():
        # 텍스트 색상 단순화
        if "매수" in stat: color = "green"
        elif "매도" in stat: color = "red"
        else: color = "off"
        
        cols[idx % 5].caption(f"{name}") # 글씨 작게
        cols[idx % 5].markdown(f":{color}[{stat}]")
        idx += 1


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
    # [수정됨] CSV 파일 뷰어로 변경
    st.subheader("📖 AI의 성장 일지 (Trade Log)")
    st.caption("AI가 매매 후 작성한 기록과 반성문이 이곳에 저장됩니다.")
    
    col_ref, col_down = st.columns([1, 4])
    if col_ref.button("🔄 기록 새로고침"): 
        st.rerun()
    
    # 1. CSV 파일 읽어오기
    if os.path.exists(LOG_FILE):
        try:
            history_df = pd.read_csv(LOG_FILE)
            
            # 최신순 정렬 (Time 컬럼 기준)
            if 'Time' in history_df.columns:
                history_df = history_df.sort_values(by='Time', ascending=False)
            
            # 2. 데이터 표시
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # 3. 다운로드 버튼 제공
            with col_down:
                csv = history_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 엑셀로 다운로드", csv, "trade_log.csv", "text/csv")
                
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    else:
        st.info("📭 아직 기록된 매매가 없습니다.")
        
    st.divider()
    
    # 4. 테스트 버튼 (새로운 log_trade 함수 형식에 맞춤)
    if st.button("🧪 테스트 데이터 입력 (기록 확인용)"):
        # 가짜 데이터: 코인, 포지션, 진입가, 청산가, 손익금, 수익률, 이유
        log_trade("BTC/TEST", "long", 50000, 49000, -100, -2.0, "테스트: 손절 로직 확인용")
        st.success("테스트 데이터가 입력되었습니다! 위 표를 확인하세요.")
        time.sleep(1)
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


# =========================================================
# 💰 [사이드바] 실시간 내 잔고 & 포지션 현황
# =========================================================
with st.sidebar:
    st.divider()
    st.header("내 지갑 현황 (Wallet)")
    
    try:
        # 1. 잔고 조회
        balance = exchange.fetch_balance({'type': 'swap'})
        usdt_free = balance['USDT']['free']
        usdt_total = balance['USDT']['total']
        
        st.metric("총 자산 (USDT)", f"${usdt_total:,.2f}")
        st.metric("주문 가능", f"${usdt_free:,.2f}")
        
        # 2. 포지션 조회
        st.divider()
        st.subheader("보유 포지션")
        
        # 전체 심볼에 대해 포지션 조회는 느리므로, 주요 코인만 조회하거나 전체 조회
        # (Bitget은 fetch_positions()에 인자가 없으면 전체를 가져옵니다)
        positions = exchange.fetch_positions(symbols=TARGET_COINS) 
        active_positions = [p for p in positions if float(p['contracts']) > 0]
        
        if active_positions:
            for p in active_positions:
                symbol = p['symbol'].split(':')[0]
                side = "🟢 Long" if p['side'] == 'long' else "🔴 Short"
                pnl = float(p['unrealizedPnl'])
                roi = float(p['percentage'])
                lev = p['leverage']
                
                # 카드 형태로 표시
                st.info(f"**{symbol}** ({side} x{lev})\n"
                        f"수익: **{roi:.2f}%** (${pnl:.2f})")
        else:
            st.caption("현재 무포지션 (관망 중)")
            
    except Exception as e:
        st.error(f"데이터 조회 실패: {e}")
