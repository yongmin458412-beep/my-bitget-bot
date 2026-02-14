# =========================================================
#  Bitget AI Wonyoti Agent (Final Integrated) - 유지보수/확장판
#  - Streamlit: 제어판/차트/포지션/일지/AI 시야/백테스트/내보내기
#  - Telegram: 실시간 보고/조회/일지 요약 + (채널/그룹 분리 지원) + /menu
#  - AutoTrade: 데모(IS_SANDBOX=True) 기본
#
#  ⚠️ 주의: 트레이딩은 손실 위험이 큽니다. (특히 레버리지)
#
#  requirements.txt 추천(있으면 사용, 없어도 동작하도록 optional import 처리):
#  - streamlit
#  - ccxt
#  - openai
#  - requests
#  - pandas
#  - numpy
#  - ta
#  - streamlit-autorefresh
#  - orjson
#  - tenacity
#  - diskcache
#  - pandas_ta
#  - scipy
#  - feedparser
#  - cachetools
#  - openpyxl              # Excel 내보내기
#  - gspread               # Google Sheets (선택)
#  - google-auth           # Google Sheets (선택)
#  - deep-translator       # 한글화(선택, 없으면 AI/룰 기반)
#  - loguru                # 로그(선택)
# =========================================================

import os
import re
import json
import time
import uuid
import math
import threading
import traceback
import socket
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests
import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except Exception:
    # 구버전/환경 차이 대응(스레드 컨텍스트 미지원이어도 봇이 죽지 않게)
    def add_script_run_ctx(_th):  # type: ignore
        return None

import ccxt
from openai import OpenAI

# ---- optional pip ----
try:
    import ta  # pip: ta
except Exception:
    ta = None

try:
    from streamlit_autorefresh import st_autorefresh  # pip: streamlit-autorefresh
except Exception:
    st_autorefresh = None

try:
    import orjson
except Exception:
    orjson = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter
except Exception:
    retry = None
    stop_after_attempt = None
    wait_exponential_jitter = None

try:
    from diskcache import Cache
except Exception:
    Cache = None

try:
    import pandas_ta as pta
except Exception:
    pta = None

try:
    from scipy.signal import argrelextrema
except Exception:
    argrelextrema = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None

try:
    import openpyxl  # noqa: F401  # pip: openpyxl
except Exception:
    openpyxl = None

try:
    import gspread  # pip: gspread
    from google.oauth2.service_account import Credentials as GoogleCredentials  # pip: google-auth
except Exception:
    gspread = None
    GoogleCredentials = None

try:
    from deep_translator import GoogleTranslator  # pip: deep-translator
except Exception:
    GoogleTranslator = None

try:
    from loguru import logger  # pip: loguru
except Exception:
    logger = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager as mfont
    from matplotlib import transforms as mtransforms
    from matplotlib.patches import Rectangle
except Exception:
    plt = None
    mdates = None
    mfont = None
    mtransforms = None
    Rectangle = None

try:
    import koreanize_matplotlib as _koreanize_matplotlib  # Nanum 폰트 번들
except Exception:
    _koreanize_matplotlib = None

# =========================================================
# ✅ 글로벌 네트워크 타임아웃(안전장치)
# - 일부 라이브러리(feedparser/urllib 등)는 timeout을 지정하지 않으면 영구 대기할 수 있음
# - socket default timeout을 걸어 Streamlit UI/봇 스레드가 "멈춘 것처럼" 보이는 문제를 완화
# =========================================================
try:
    socket.setdefaulttimeout(15)
except Exception:
    pass


# =========================================================
# ✅ 빌드/버전 토큰(운영 디버깅용)
# - Streamlit은 rerun 시에도 daemon thread가 남을 수 있어, "지금 어떤 코드가 돌아가고 있는지"
#   확인하기 쉽게 토큰을 만든다.
# =========================================================
def _code_version_token() -> str:
    try:
        p = str(__file__ or "").strip()
        if not p:
            return "unknown"
        mtime = int(os.path.getmtime(p))
        return f"{os.path.basename(p)}@{mtime}"
    except Exception:
        return "unknown"


CODE_VERSION = _code_version_token()


# =========================================================
# ✅ 0) 기본 설정
# =========================================================
st.set_page_config(layout="wide", page_title="비트겟 AI 워뇨띠 에이전트 (Final Integrated)")

IS_SANDBOX = True  # ✅ 데모/모의투자 (실전 전환은 파일 하단 안내 참고)

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
MONITOR_FILE = "monitor_state.json"

DETAIL_DIR = "trade_details"
DAILY_REPORT_DIR = "daily_reports"
EVENT_IMAGE_DIR = "trade_event_images"
os.makedirs(DETAIL_DIR, exist_ok=True)
os.makedirs(DAILY_REPORT_DIR, exist_ok=True)
os.makedirs(EVENT_IMAGE_DIR, exist_ok=True)

_cache = Cache("cache") if Cache else None  # 선택(디스크 캐시)

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

# OpenAI 호출 타임아웃(초) - 스레드 멈춤 방지
OPENAI_TIMEOUT_SEC = 20

# HTTP 요청 타임아웃(초)
HTTP_TIMEOUT_SEC = 12

# 외부 시황 갱신 hard-timeout(초) - 네트워크/번역 지연으로 봇 스레드가 멈춘 것처럼 보이는 문제 완화
EXTERNAL_CONTEXT_TIMEOUT_SEC = max(10, HTTP_TIMEOUT_SEC + 4)

_THREAD_POOL = ThreadPoolExecutor(max_workers=4)
_THREAD_POOL_LOCK = threading.RLock()
_THREAD_POOL_CREATED_EPOCH = time.time()

# =========================================================
# ✅ CCXT Hard-timeout(스레드/화면 정체 방지)
# - ccxt/requests가 환경 이슈로 길게 정체되면 Streamlit 화면/봇 스레드가 멈춘 것처럼 보일 수 있음
# - ThreadPool + timeout으로 "최대 대기시간"을 보장하고, 연속 timeout 시 회로차단(circuit breaker)
#   로 스레드/리소스 누수를 완화한다.
# =========================================================
CCXT_TIMEOUT_SEC_PUBLIC = 12
CCXT_TIMEOUT_SEC_PRIVATE = 15

_CCXT_POOL = ThreadPoolExecutor(max_workers=2)
_CCXT_POOL_LOCK = threading.RLock()

_CCXT_CB_LOCK = threading.RLock()
_CCXT_CB_UNTIL_EPOCH = 0.0
_CCXT_CB_REASON = ""
_CCXT_TIMEOUT_EPOCHS = deque(maxlen=12)
_CCXT_CB_OPEN_AFTER_TIMEOUTS = 3
_CCXT_CB_WINDOW_SEC = 60.0
_CCXT_CB_COOLDOWN_SEC = 45.0


def _ccxt_cb_is_open() -> bool:
    try:
        with _CCXT_CB_LOCK:
            return time.time() < float(_CCXT_CB_UNTIL_EPOCH or 0.0)
    except Exception:
        return False


def _ccxt_cb_open(reason: str, duration_sec: float):
    try:
        with _CCXT_CB_LOCK:
            global _CCXT_CB_UNTIL_EPOCH, _CCXT_CB_REASON
            _CCXT_CB_UNTIL_EPOCH = time.time() + float(duration_sec)
            _CCXT_CB_REASON = str(reason or "")[:240]
    except Exception:
        pass


def _ccxt_record_timeout(where: str = ""):
    try:
        now_ts = time.time()
        _CCXT_TIMEOUT_EPOCHS.append(now_ts)
        # 최근 window 내 timeout 횟수 계산
        cnt = 0
        for t0 in list(_CCXT_TIMEOUT_EPOCHS):
            if (now_ts - float(t0)) <= float(_CCXT_CB_WINDOW_SEC):
                cnt += 1
        if cnt >= int(_CCXT_CB_OPEN_AFTER_TIMEOUTS):
            _ccxt_cb_open(reason=f"ccxt_timeout_burst({where})", duration_sec=float(_CCXT_CB_COOLDOWN_SEC))
    except Exception:
        pass


def _ccxt_call_with_timeout(fn, timeout_sec: int, where: str = "", context: Optional[Dict[str, Any]] = None):
    """
    ccxt 호출을 hard-timeout으로 감싸 Streamlit/TG_THREAD 정체를 완화.
    - timeout이 연속으로 발생하면 circuit breaker가 열려, 일정 시간 동안 ccxt 호출을 즉시 실패 처리한다.
    """
    if _ccxt_cb_is_open():
        raise FuturesTimeoutError("ccxt_circuit_open")

    global _CCXT_POOL
    got = False
    try:
        got = bool(_CCXT_POOL_LOCK.acquire(timeout=0.8))
    except Exception:
        got = False
    if not got:
        raise FuturesTimeoutError("ccxt_pool_lock_timeout")
    try:
        try:
            fut = _CCXT_POOL.submit(fn)
        except RuntimeError as e:
            # executor shutdown 등에서 복구
            msg = str(e or "").lower()
            if "cannot schedule new futures" in msg or "shutdown" in msg:
                _CCXT_POOL = ThreadPoolExecutor(max_workers=2)
                fut = _CCXT_POOL.submit(fn)
            else:
                raise
    finally:
        try:
            _CCXT_POOL_LOCK.release()
        except Exception:
            pass

    try:
        return fut.result(timeout=int(timeout_sec))
    except FuturesTimeoutError:
        _ccxt_record_timeout(where=where)
        raise


def ccxt_health_snapshot() -> Dict[str, Any]:
    try:
        with _CCXT_CB_LOCK:
            until = float(_CCXT_CB_UNTIL_EPOCH or 0.0)
            reason = str(_CCXT_CB_REASON or "")
        return {
            "circuit_open": bool(time.time() < until) if until else False,
            "circuit_until_kst": _epoch_to_kst_str(until) if until else "",
            "circuit_reason": reason,
            "timeouts_recent": len(list(_CCXT_TIMEOUT_EPOCHS)),
        }
    except Exception:
        return {"circuit_open": False}


# =========================================================
# ✅ 1) 시간 유틸 (KST)
# =========================================================
KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")


def next_midnight_kst_epoch() -> float:
    """KST 기준 다음날 00:00:00 epoch."""
    try:
        dt = now_kst()
        dt2 = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return float(dt2.timestamp())
    except Exception:
        return time.time() + 60 * 60 * 6


def _parse_time_kst(s: str) -> Optional[datetime]:
    try:
        # "YYYY-MM-DD HH:MM:SS"
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except Exception:
        return None


def _dt_to_epoch(dt: datetime) -> float:
    try:
        return dt.timestamp()
    except Exception:
        return time.time()


def _epoch_to_kst_str(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return now_kst_str()


# =========================================================
# ✅ 2) JSON 안전 저장/로드 (원자적)
# =========================================================
def _json_default(obj: Any):
    # JSON 직렬화 실패로 monitor_state.json 갱신이 멈추는 문제 방지(실시간 UI 갱신 핵심)
    try:
        if isinstance(obj, datetime):
            return obj.isoformat()
    except Exception:
        pass
    try:
        if isinstance(obj, (set, tuple)):
            return list(obj)
    except Exception:
        pass
    try:
        # numpy scalar(예: np.float64) 방어
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        if orjson:
            with open(tmp, "wb") as f:
                opt = 0
                try:
                    opt |= orjson.OPT_SERIALIZE_NUMPY  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    opt |= orjson.OPT_NON_STR_KEYS  # type: ignore[attr-defined]
                except Exception:
                    pass
                f.write(orjson.dumps(data, default=_json_default, option=opt))
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        # 파일 I/O 에러가 봇을 죽이면 안 됨
        try:
            # 마지막 보루: 값들을 문자열로 덤프(구조는 유지)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            pass


_READ_JSON_LAST_ERROR: Dict[str, str] = {}


def read_json_safe(path: str, default=None):
    try:
        if orjson:
            with open(path, "rb") as f:
                v = orjson.loads(f.read())
                _READ_JSON_LAST_ERROR[str(path)] = ""
                return v
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
            _READ_JSON_LAST_ERROR[str(path)] = ""
            return v
    except Exception:
        try:
            _READ_JSON_LAST_ERROR[str(path)] = traceback.format_exc(limit=2)
        except Exception:
            _READ_JSON_LAST_ERROR[str(path)] = "read_json_safe failed"
        return default


def safe_json_dumps(x: Any, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        try:
            s = str(x)
        except Exception:
            s = ""
    if len(s) > limit:
        return s[:limit] + "..."
    return s


# =========================================================
# ✅ 2.5) 상세일지 저장/조회
# =========================================================
def save_trade_detail(trade_id: str, payload: Dict[str, Any]) -> None:
    try:
        write_json_atomic(os.path.join(DETAIL_DIR, f"{trade_id}.json"), payload)
    except Exception:
        pass


def load_trade_detail(trade_id: str) -> Optional[Dict[str, Any]]:
    try:
        return read_json_safe(os.path.join(DETAIL_DIR, f"{trade_id}.json"), None)
    except Exception:
        return None


def list_recent_trade_ids(limit: int = 10) -> List[str]:
    try:
        files = [f for f in os.listdir(DETAIL_DIR) if f.endswith(".json")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(DETAIL_DIR, x)), reverse=True)
        return [os.path.splitext(f)[0] for f in files[:limit]]
    except Exception:
        return []


# =========================================================
# ✅ 2.6) Streamlit/pyarrow 호환: DataFrame 안전 변환
# - object 타입에 숫자/문자/딕트 혼재 시 Arrow 변환이 터질 수 있음
#   (사용자 로그: "Expected bytes, got a 'int' object")
# =========================================================
def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    try:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].apply(
                    lambda v: safe_json_dumps(v, limit=400)
                    if isinstance(v, (dict, list))
                    else ("" if v is None else str(v))
                )
        return out
    except Exception:
        try:
            return df.astype(str)
        except Exception:
            return pd.DataFrame()


# =========================================================
# ✅ 2.7) Streamlit DataFrame 표시 호환(버전 차이 대응)
# - 일부 Streamlit 버전에서 st.dataframe(width="stretch") / hide_index / use_container_width 호환 문제
# - UI 기능이 "작동 안 함"처럼 보이는 런타임 오류를 줄인다.
# =========================================================
def st_dataframe_safe(data, **kwargs):
    """
    Streamlit 버전 차이로 인한 파라미터 TypeError를 흡수하면서 최대한 표시.
    - 최신 Streamlit(2025+): `use_container_width`가 deprecate → `width="stretch"` 우선 사용
    - 구버전 Streamlit: `width` 미지원이면 `use_container_width=True/False`로 폴백
    """
    try:
        # ✅ 최신 Streamlit 권장: width="stretch"/"content"
        # - 호출자가 use_container_width를 줬다면(레거시), 가능한 경우 width로 변환해 경고를 없앤다.
        if "use_container_width" in kwargs and "width" not in kwargs:
            try:
                kwargs["width"] = "stretch" if bool(kwargs.get("use_container_width")) else "content"
            except Exception:
                kwargs["width"] = "stretch"
            kwargs.pop("use_container_width", None)
        kwargs.setdefault("width", "stretch")
        return st.dataframe(data, **kwargs)
    except TypeError:
        # 구버전 Streamlit: width 미지원 → use_container_width로 폴백
        try:
            w = kwargs.pop("width", None)
            if "use_container_width" not in kwargs:
                if w == "content":
                    kwargs["use_container_width"] = False
                else:
                    kwargs["use_container_width"] = True
            return st.dataframe(data, **kwargs)
        except TypeError:
            # 지원하지 않는 kwargs 제거 후 재시도
            for k in ["use_container_width", "hide_index", "column_config", "column_order", "width"]:
                kwargs.pop(k, None)
            try:
                return st.dataframe(data, **kwargs)
            except Exception:
                return st.dataframe(data)
        try:
            return st.dataframe(data)
        except Exception:
            return st.dataframe(data)
    except Exception:
        return st.dataframe(data)


# =========================================================
# ✅ 3) MODE_RULES (기존 유지)
# =========================================================
MODE_RULES = {
    # ✅ 사용자 요구 반영:
    # - 안전모드: 진입비중↓, 레버↓, 확신도↑일 때만 진입
    # - 공격모드: 진입비중/레버 "중간", 확신도 "중간"이어도 진입
    # - 하이리스크/하이리턴: 진입비중↑, 레버↑, 확신도↑일 때만 진입
    "안전모드": {"min_conf": 85, "entry_pct_min": 2, "entry_pct_max": 7, "lev_min": 2, "lev_max": 6},
    "공격모드": {"min_conf": 75, "entry_pct_min": 7, "entry_pct_max": 22, "lev_min": 4, "lev_max": 12},
    "하이리스크/하이리턴": {"min_conf": 72, "entry_pct_min": 18, "entry_pct_max": 40, "lev_min": 12, "lev_max": 25},
}


# =========================================================
# ✅ 4) 설정 관리 (load/save)
# =========================================================
def default_settings() -> Dict[str, Any]:
    return {
        # ✅ 설정 마이그레이션(기본값 변경/추가 기능 반영)
        "settings_schema_version": 11,
        "openai_api_key": "",
        # ✅ 사용자 기본값 프리셋(요청): 하이리스크/하이리턴 + 자동매매 ON
        "auto_trade": True,
        "trade_mode": "하이리스크/하이리턴",
        "timeframe": "5m",
        "order_usdt": 100.0,

        # Telegram (기본 유지)
        "tg_enable_reports": True,  # 이벤트 알림(진입/청산 등)
        "tg_send_entry_reason": False,
        # ✅ 텔레그램 메시지 가독성(요구사항):
        # - 코인/선물 용어를 모르는 사람도 이해하도록 "쉬운 한글 + 핵심만" 모드(기본 ON)
        # - OFF면 기존(상세) 메시지를 유지
        "tg_simple_messages": True,

        # ✅ 주기 리포트/시야 리포트
        "tg_enable_periodic_report": True,
        "report_interval_min": 15,
        # ✅ 하트비트(15분) 전송: 사용자 요청으로 기본 OFF
        "tg_enable_heartbeat_report": False,
        "tg_heartbeat_interval_sec": 900,
        # ✅ 메시지별 알림(푸시) 제어:
        # - silent=True면 Telegram에서 '무음 전송'(disable_notification)로 보냄
        # - 채널/그룹을 사용자가 '완전 음소거'했다면, 봇이 푸시를 강제로 켤 수는 없음(텔레그램 정책)
        "tg_heartbeat_silent": True,
        "tg_periodic_report_silent": True,
        # ✅ 사용자 요구: 알림(푸시)은 진입/청산(익절/손절)만 (기본 ON)
        "tg_notify_entry_exit_only": True,
        # ✅ 사용자 요구: 진입/청산(익절/손절)은 채널에서만 확인 → 관리자 DM 복사는 기본 OFF
        "tg_trade_alert_to_admin": False,
        # ✅ 사용자 요구: AI 시야 리포트(자동 전송)는 기본 OFF (필요할 때만 /vision 으로 조회)
        "tg_enable_hourly_vision_report": False,
        "vision_report_interval_min": 60,
        # ✅ 진입/청산 이벤트 차트 이미지 전송
        "tg_send_trade_images": True,
        "tg_send_entry_image": True,
        "tg_send_exit_image": True,
        "tg_image_chart_bars": 140,
        "tg_image_sr_lines": 3,
        "tg_image_volume_nodes": 4,
        "tg_image_show_indicators": True,
        "tg_image_show_pattern_overlay": True,

        # ✅ 텔레그램 라우팅: channel/group (secrets로 설정 권장)
        "tg_route_events_to": "channel",  # "channel"|"group"|"both"
        "tg_route_queries_to": "group",   # "group"|"channel"|"both"

        # 지표 파라미터
        "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
        "bb_period": 20, "bb_std": 2.0,
        "ma_fast": 7, "ma_slow": 99,
        "stoch_k": 14,
        "vol_mul": 2.0,

        # 지표 ON/OFF
        "use_rsi": True, "use_bb": True, "use_cci": True, "use_vol": True, "use_ma": True,
        "use_macd": True, "use_stoch": True, "use_mfi": True, "use_willr": True, "use_adx": True,
        # ✅ 추가: 스퀴즈 모멘텀(Squeeze Momentum) - 초단기/추세 전환 포착용
        "use_sqz": True,
        "sqz_bb_length": 20,
        "sqz_bb_mult": 2.0,
        "sqz_kc_length": 20,
        "sqz_kc_mult": 1.5,
        "sqz_mom_length": 20,
        # SQZ 모멘텀을 "가격 대비 %"로 환산한 기준(너무 크면 신호가 안 나고, 너무 작으면 과다신호)
        "sqz_mom_threshold_pct": 0.05,
        # SQZ를 최우선으로 반영(요구: 80% 이상 의존)
        "sqz_dependency_enable": True,
        "sqz_dependency_weight": 0.80,      # 0~1 (기본 0.80)
        "sqz_dependency_gate_entry": True,  # SQZ가 중립이면 진입 억제
        "sqz_dependency_override_ai": True, # SQZ가 반대면 AI buy/sell을 hold로 강제

        # ✅ (추가) 주력 지표(요구): Lorentzian / KNN / Logistic / SQZ / RSI
        # - 5개 중 3개 이상이 같은 방향으로 수렴할 때만 진입(비용/휩쏘 방지)
        # - 스캔 단계에서 먼저 계산하고, 진입 시에만 AI를 호출해 TP/SL/SR를 유도리 있게 설계
        "entry_convergence_enable": True,
        "entry_convergence_min_votes": 3,
        # ML 시그널 계산(외부 라이브러리 없이 numpy/pandas로만)
        "ml_enable": True,
        "ml_lookback": 220,          # 학습(과거 N샘플)
        "ml_horizon": 1,             # 라벨(미래 h봉): close[t+h] > close[t]
        "ml_feature_ma_period": 20,
        "ml_feature_vol_ma_period": 20,
        "ml_min_train_samples": 80,
        # KNN
        "ml_knn_k": 15,
        "ml_knn_prob_long": 0.56,
        "ml_knn_prob_short": 0.44,
        # Lorentzian KNN
        "ml_lor_k": 15,
        "ml_lor_prob_long": 0.56,
        "ml_lor_prob_short": 0.44,
        # Logistic regression(간이 GD)
        "ml_logit_steps": 120,
        "ml_logit_lr": 0.15,
        "ml_logit_l2": 0.01,
        "ml_logit_prob_long": 0.56,
        "ml_logit_prob_short": 0.44,
        # RSI 방향(중립 구간은 0)
        "ml_rsi_neutral_band": 3.0,  # 50±3 구간은 중립
        # 캐시(같은 봉에서는 ML도 1회만 계산)
        "ml_cache_enable": True,
        # 차트 패턴 감지(진입 보조): M/W, 쌍봉/쌍바닥, 삼중천정/삼중바닥, 삼각수렴, 박스, 쐐기, 헤드앤숄더
        "use_chart_patterns": True,
        "pattern_lookback": 220,
        "pattern_pivot_order": 4,
        "pattern_tolerance_pct": 0.60,
        "pattern_min_retrace_pct": 0.35,
        "pattern_flat_slope_pct": 0.03,
        "pattern_breakout_buffer_pct": 0.08,
        "pattern_call_strength_min": 0.45,
        "pattern_gate_entry": True,
        "pattern_gate_strength": 0.65,
        "pattern_override_ai": True,
        # ✅ 멀티 타임프레임 캔들패턴(요구)
        # - 1m/3m/5m/15m/30m/1h/2h/4h를 함께 보고 패턴 bias를 합산
        "pattern_mtf_enable": True,
        "pattern_mtf_timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"],
        "pattern_mtf_cache_sec": 90,
        "pattern_mtf_merge_weight": 0.60,

        # 방어/전략
        "use_trailing_stop": True,
        # ✅ 청산 정책: AI가 정한 TP/SL만 사용
        # - ON이면 SR/트레일링/본절보호/강제익절/손절확인 등 다른 청산 로직을 모두 무시하고
        #   "AI 목표 ROI" (tp/sl) 에 닿을 때만 청산한다.
        "exit_ai_targets_only": True,
        # ✅ AI 목표(ROI) 자체를 SR/매물대 기준 가격에서 역산해 동기화
        # - ON이면 동일한 +14.4/-9.0 반복을 줄이고, 코인/구간별로 목표가가 달라짐
        "exit_ai_targets_sync_from_sr": True,
        # ✅ (핵심) 강제 청산 정책: "수익을 손실로 마감"하지 않기
        # - 진입 판단(AI)은 유지하되, 청산(Exit)만큼은 아래 규칙을 우선 적용한다.
        # - ON이면 기존 TP/SL/SR/부분익절/트레일링(기존)보다 아래 정책이 우선한다.
        "exit_trailing_protect_enable": True,
        "exit_trailing_protect_check_sec": 1.0,
        # ✅ AI 목표(TP/SL)를 1순위로 적용(요구)
        # - 강제 Exit(수익보존) 정책을 켠 상태에서도, AI가 잡은 TP/SL(ROI%)을 우선 트리거로 사용한다.
        # - 단, 본절/부분익절/추적손절(수익보존)은 안전망으로 계속 동작한다.
        "exit_trailing_protect_ai_targets_priority": True,
        # 포지션 보유 중에는 스캔/AI 호출로 루프가 길어져 청산 타이밍을 놓칠 수 있어,
        # 강제 Exit 정책 사용 시 기본은 "포지션이 있을 때 신규 스캔/진입을 쉬고" 청산 모니터링에 집중한다.
        "exit_trailing_protect_pause_scan_while_in_position": False,
        "exit_trailing_protect_sl_roi": 15.0,                 # 기본 손절: -15%
        "exit_trailing_protect_be_roi": 10.0,                 # 1단계: +10% → 본전(진입가) 보호
        "exit_trailing_protect_partial_roi": 30.0,            # 2단계: +30% → 50% 익절(부분청산)
        "exit_trailing_protect_partial_close_pct": 50.0,      # 부분청산 비율(%)
        "exit_trailing_protect_trail_start_roi": 50.0,        # 3단계: +50% 이후부터 추적손절 활성
        "exit_trailing_protect_trail_dd_roi": 10.0,           # 최고점 대비 -10%면 전량 청산
        # ✅ 본절(BE) 라인 터치 시 즉시청산 대신 "한 번 더 판단" (요구)
        # - confirm_n회 연속 터치 + 차트 불리 판정일 때만 본절청산
        # - 차트가 유리하면 홀딩하고 다시 기회를 본다.
        "be_recheck_enable": True,
        "be_recheck_confirm_n": 2,
        "be_recheck_window_sec": 180.0,
        "be_recheck_hold_score_min": 2,
        "be_recheck_hold_cooldown_sec": 20.0,
        "be_recheck_retry_sec": 3.0,

        # ✅ 스윙 진입금(총자산 %)을 공포/탐욕(FNG)에 따라 자동 조정(요구)
        # - 하이리스크/하이리턴 + 스윙일 때, "너무 작게(예: 4%)" 들어가는 문제를 방지
        # - 기본: FNG 0/100에서 min, 50에서 max(삼각형 형태)
        "swing_fng_entry_pct_enable": True,
        "swing_fng_entry_pct_min": 8.0,
        "swing_fng_entry_pct_max": 15.0,
        # ✅ Time-based Exit(요구): 진입 후 일정 시간이 지났는데 "목표 수익의 X%"도 못 가면 기회비용 정리
        # - 사용자 요청: "시간초과 강제청산"은 사용하지 않음(목표 TP/SL 도달 전까지 홀딩)
        # - 호환을 위해 설정 키는 유지(현재 로직에서는 비활성)
        "time_exit_enable": False,
        "time_exit_bars": 24,            # 5m 기준 24 bars = 2시간
        "time_exit_target_frac": 0.3,    # 목표 수익의 30% 미만이면 정리
        "time_exit_partial_enable": True,  # 전량 청산 대신 50% 부분 청산
        "time_exit_partial_close_pct": 50.0,  # 시간 초과 시 부분 청산 비율(%)

        # ✅ Fail-safe(요구: "수익을 못 내거나 전부 잃으면 AI는 꺼진다")
        # - 실제로 "수익 보장"은 불가능하므로, 현실적인 안전장치로 자동매매를 강제 종료한다.
        # - 1) 계좌 드로다운이 임계치 이상이면 OFF
        # - 2) 일정 횟수 거래 후에도 당일 실현손익이 0 이하이면 OFF(과매매/손실 누적 방지)
        "fail_safe_enable": False,
        "fail_safe_drawdown_enable": False,
        "fail_safe_drawdown_from_peak_pct": 18.0,   # peak equity 대비 -18%면 OFF (기존 30% → 강화)
        "fail_safe_profit_guard_enable": False,      # ON: 연속 손실 누적 방지
        "fail_safe_profit_guard_min_trades": 6,     # 6회 거래 후에도
        "fail_safe_profit_guard_min_pnl_usdt": -30.0, # -30 USDT 이하 손실이면 OFF

        # ✅ 진입 고정(요구사항): 레버 20배 고정 + 잔고 20% 진입
        # - 기존 모드/AI의 entry_pct/leverage 출력은 "표시용"으로만 남기고, 실제 주문은 아래 값을 사용
        "fixed_leverage_enable": False,
        "fixed_leverage": 20,
        "fixed_entry_pct_enable": False,
        "fixed_entry_pct": 20.0,
        # ✅ (추가) 하이리스크/하이리턴 모드에서만 고정 진입(요구)
        # - entry_usdt = 총자산(total) * 20%
        # - leverage = 20x
        # - 다른 모드에서는 AI/룰 기반(기존)
        "highrisk_fixed_size_enable": True,
        "highrisk_fixed_entry_pct_total": 20.0,
        "highrisk_fixed_leverage": 20,
        # cross/isolated 선택(거래소/계정 설정에 따라 실패할 수 있으니 safe 적용)
        "margin_mode": "cross",  # "cross"|"isolated"
        # ✅ ATR 기반 레버리지(요구): 변동성이 크면 레버↓, 변동성이 작으면 레버↑
        # - fixed_leverage_enable=OFF일 때만 적용
        "atr_leverage_enable": True,
        "atr_leverage_window": 14,
        # atr_price_pct(%)가 low 이하이면 max_lev, high 이상이면 min_lev (사이 구간은 선형 보간)
        "atr_leverage_low_pct": 0.35,
        "atr_leverage_high_pct": 1.20,
        "atr_leverage_min": 5,
        "atr_leverage_max": 20,
        # ✅ 포지션 사이징 보호(요구): 한 번의 거래에서 잃는 돈을 전체 시드의 2~3% 이내로 제한
        # - 손절(ROI%) 기준으로 "진입금(마진)"을 자동 감산한다.
        "max_risk_per_trade_enable": True,
        "max_risk_per_trade_pct": 2.5,
        "max_risk_per_trade_usdt": 0.0,
        # ✅ 모드별 최대손실(%) 오버라이드: 진입금이 너무 낮아지는 문제 완화(특히 하이리스크)
        "max_risk_per_trade_pct_safe": 2.5,
        "max_risk_per_trade_pct_attack": 3.5,
        "max_risk_per_trade_pct_highrisk": 5.0,
        # ✅ (선택) Kelly sizing: AI confidence(확신도) + 손익비(rr)로 entry_pct 상한을 계산(기본 OFF)
        # - 과대진입을 줄이는 용도로만 사용(기본은 min(AI entry_pct, Kelly cap))
        "kelly_sizing_enable": False,
        "kelly_fraction_mult": 0.5,   # half-kelly 권장
        "kelly_max_entry_pct": 20.0,  # Kelly cap 상한(% of free)

        # ✅ 손절(ROI) 확인(휩쏘 방지):
        # - SR(지지/저항) 가격 이탈 손절은 즉시 실행
        # - ROI(퍼센트) 손절은 n회 연속 조건일 때만 실행(순간 위꼬리/툭 찍고 복구 방지)
        "sl_confirm_enable": True,
        "sl_confirm_n": 1,
        "sl_confirm_window_sec": 600.0,  # ✅ 6초→600초: 메인 루프 주기(수 분)가 6초보다 길어 손절 확인이 누적되지 않던 문제 수정
        # ✅ 청산 후 재진입 쿨다운(과매매/수수료/AI호출 낭비 방지)
        # - "bars"는 현재 단기 timeframe 기준 봉 개수(예: 5m에서 2 bars = 10분)
        "cooldown_after_exit_tp_bars": 1,
        "cooldown_after_exit_sl_bars": 3,
        "cooldown_after_exit_protect_bars": 2,
        "use_dca": True,
        "dca_trigger": -32.0,
        "dca_max_count": 2,
        "dca_daily_pnl_limit_enable": True,   # 당일 손익이 기준 이하면 DCA 금지
        "dca_daily_pnl_limit_usdt": -20.0,    # 당일 -20 USDT 이하면 DCA 스킵
        # ✅ 추매 규모(기본=기존 % 방식 유지)
        # - dca_add_usdt > 0 이면 "USDT(마진)" 기준으로 추매 금액을 고정(사용자 요구)
        # - 0이면 기존처럼 원진입 대비 %로 계산
        "dca_add_pct": 50.0,
        "dca_add_usdt": 0.0,
        "use_switching": True, "switch_trigger": -12.0,  # (옵션만 유지: 기존 코드도 로직 미구현)
        "no_trade_weekend": False,

        # 연속손실 보호
        "loss_pause_enable": True, "loss_pause_after": 5, "loss_pause_minutes": 15,
        # ✅ 추가 방어(사용자 선택): 서킷브레이커/일일 손실 한도
        # - loss_pause: "잠깐 쉼" (기존)
        # - circuit_breaker: (사용자 요청) 자동매매 OFF 하지 않음 → 경고/기록만
        # - daily_loss_limit: "하루 손실 한도" 도달 시 자동매매 OFF
        "circuit_breaker_enable": False,
        "circuit_breaker_after": 10,  # 연속 손실 N번이면 경고(기본 OFF)
        "daily_loss_limit_enable": False,
        "daily_loss_limit_pct": 8.0,   # day_start_equity 대비 -8%면 정지(0이면 미사용)
        "daily_loss_limit_usdt": 0.0,  # -USDT 기준(0이면 미사용)

        # AI 추천
        "ai_reco_show": True,
        "ai_reco_apply": False,
        "ai_reco_refresh_sec": 20,
        "ai_easy_korean": True,
        # ✅ AI 호출 비용 절감:
        # - 자동 스캔에서 AI는 "같은 봉(단기 TF)에서는 1회만" 호출하고, 이후에는 캐시를 재사용한다.
        # - (강제스캔 /scan 은 예외)
        "ai_scan_once_per_bar": True,
        # ✅ 진입 필터 강화(요구): 거래량(스파이크) + 이격도(Disparity) 조건
        # - 횡보 박스(거래량 없음)에서 RSI 해소만 보고 진입하는 실수를 줄이기 위해 AI 호출 자체를 제한한다.
        # - /scan 강제스캔은 이 필터를 우회(사용자 의도)한다.
        "ai_call_require_volume_spike": False,
        "ai_call_volume_spike_mul": 1.3,
        "ai_call_volume_spike_period": 20,
        "ai_call_require_disparity": False,
        "ai_call_disparity_ma_period": 20,
        "ai_call_disparity_max_abs_pct": 6.0,
        # ✅ 진입이 너무 안 되는 경우를 대비한 완화 옵션
        # - 0이면 모드별 기본값(안전 20 / 공격 17 / 하이리스크 15)을 사용
        "ai_call_adx_threshold": 0,
        # - True면 volume/disparity 조건 미달 시 "AI 호출 자체"를 막음(비용↓, 보수↑)
        # - False면 AI는 호출하되(캐시/봉당 1회), 결과에 따라 entry를 줄이거나 hold를 기대(진입 기회↑)
        "ai_call_filters_block_ai": False,
        # ✅ AI가 buy/sell을 유지할 수 있는 최소 확신 바닥값(그 이하면 강제 hold)
        "ai_decision_min_conf_floor": 55,

        # 🌍 외부 시황 통합
        "use_external_context": True,
        "macro_blackout_minutes": 30,
        "external_refresh_sec": 60,
        "news_enable": True,
        "news_refresh_sec": 300,
        "news_max_headlines": 12,
        # 뉴스 한글화는 외부 네트워크(번역기) 사용 시 느려질 수 있어 시간 예산을 둔다.
        # - 예산 초과 시 남은 헤드라인은 룰 기반 보정(_translate_ko_rule)만 적용
        "news_translate_budget_sec": 10,
        "external_koreanize_enable": True,
        "external_ai_translate_enable": False,  # 외부시황 번역에 AI 사용(비용↑, 기본 OFF)

        # ✅ 매일 아침 BTC 경제뉴스 5개 브리핑
        # ✅ 아침 브리핑(기본 OFF): 사용자 요구
        "daily_btc_brief_enable": False,
        "daily_btc_brief_hour_kst": 9,
        "daily_btc_brief_minute_kst": 0,
        "daily_btc_brief_max_items": 5,
        "daily_btc_brief_ai_summarize": True,  # OpenAI 키 있을 때만 동작

        # ✅ 스타일(스캘핑/스윙) 자동 선택/전환
        # - regime_mode: Telegram /mode로도 변경 가능(auto|scalping|swing)
        # - regime_switch_control: 시간락 없이 흔들림 방지(confirm2/hysteresis/off)
        "regime_mode": "auto",                 # "auto"|"scalping"|"swing"
        "regime_switch_control": "confirm2",   # "confirm2"|"hysteresis"|"off"
        # confirm2 상세: n회 연속 동일 레짐일 때만 전환(기본 2)
        # - 플립백(바로 되돌림) 방지: 직전 전환의 반대방향으로는 더 많은 확인(기본 3)
        # ✅ 기본값 튜닝: 너무 잦은 전환(플립플롭) 방지
        "regime_confirm_n": 3,
        "regime_confirm_n_flipback": 5,
        "regime_hysteresis_step": 0.2,
        "regime_hysteresis_enter_swing": 0.75,
        "regime_hysteresis_enter_scalp": 0.25,
        # ✅ (옵션) 하이리스크/하이리턴 모드 신규진입 제한:
        # - ON이면 auto 레짐에서 "스윙(단기+장기 정렬)"일 때만 신규 진입
        # - OFF(기본)이면 스캘핑 진입도 허용하되, 모드의 레버/진입비중 범위는 유지
        "highrisk_entry_requires_swing": False,
        # ✅ 포지션 제한(요구): 총 포지션 5개까지 + 낮은 확신 포지션은 최대 2개
        "max_open_positions_total": 5,
        "max_open_positions_low_conf": 2,
        # "낮은 확신" 기준(%) - 이 값 미만이면 low-conf로 카운트
        # - 기본: 80 (하이리스크/하이리턴 min_conf=72 기준, 72~79를 low로 보고 2개까지만 허용)
        "low_conf_position_threshold": 80,
        # ✅ 확신/시그널/필터에 따라 진입 비중 자동 조절(완화)
        "entry_size_scale_by_signal_enable": True,
        # ✅ 소프트 진입(요구): 확신도가 min_conf에 살짝 못 미쳐도, 아주 작게/보수적으로 진입해 기회를 만든다.
        # - 안전모드는 기본 OFF(0 gap)로 유지
        "soft_entry_enable": True,
        "soft_entry_conf_gap_safe": 5,
        "soft_entry_conf_gap_attack": 12,
        "soft_entry_conf_gap_highrisk": 15,
        # 소프트 진입일 때 진입비중/레버를 더 줄이는 계수
        "soft_entry_entry_pct_mult": 0.70,
        "soft_entry_leverage_mult": 0.85,
        # 소프트 진입일 때 허용하는 최소 진입비중(% of free) / 최소 레버리지
        "soft_entry_entry_pct_floor": 2.0,
        "soft_entry_leverage_floor": 2,
        # ✅ 스타일 AI 보조(선택): 레짐 전환/표시에서 불필요한 OpenAI 호출을 줄이기 위해 분리 옵션 제공
        # - style_auto_enable=True여도, 아래 옵션이 OFF면 스타일은 "룰 기반"만 사용
        # - 사용자가 원할 때만 ON (비용/지연/요금제 429 방지)
        "style_entry_ai_enable": True,   # 신규진입(스타일 선택)에 AI 사용
        "style_switch_ai_enable": True,  # 포지션 보유 중 스타일 전환 판단에 AI 사용
        "style_ai_cache_sec": 600,        # 동일 입력의 스타일 AI 결과 캐시(초)
        "style_auto_enable": True,
        "style_lock_minutes": 20,  # 전환 최소 유지 시간
        "scalp_max_hold_minutes": 25,          # 스캘핑 포지션 최대 보유(넘으면 스윙 전환 검토)
        "scalp_to_swing_min_roi": -12.0,       # 너무 큰 손실이면 전환 대신 정리 유도(기본)
        "scalp_to_swing_require_long_align": True,  # 장기추세까지 맞아야 스윙 전환
        # ✅ 스캘핑→스윙 전환(보유시간) 안전장치:
        # - 이미 익절에 거의 도달했거나(또는 충분히 수익 구간이면) 전환으로 손절만 넓히지 않게 스킵
        "scalp_to_swing_skip_when_roi_ge_tp_frac": 0.85,  # TP의 85% 이상 수익이면 전환 스킵
        "scalp_to_swing_skip_when_tp_slack_roi": 1.0,     # TP까지 남은 ROI가 1%p 이하이면 전환 스킵
        "scalp_disable_dca": True,             # 스캘핑은 기본 추매 금지
        "scalp_tp_roi_min": 0.8,
        "scalp_tp_roi_max": 6.0,
        "scalp_sl_roi_min": 0.8,
        "scalp_sl_roi_max": 5.0,
        "scalp_entry_pct_mult": 0.65,
        "scalp_lev_cap": 8,
        # ✅ 스캘핑: "가격 변동폭(%)" 기준 가드레일(레버가 높아도 TP/SL이 과도해지지 않게)
        "scalp_sl_price_pct_min": 0.25,
        "scalp_sl_price_pct_max": 0.75,
        "scalp_tp_price_pct_min": 0.35,
        "scalp_tp_price_pct_max": 1.20,
        "scalp_rr_min_price": 1.20,  # 가격 기준 최소 RR(TP>=SL*RR)
        # ✅ 스캘핑: 하드 익절(ROI%) - TP가 비정상적으로 커져도 이 값 이상이면 익절
        "scalp_hard_take_enable": True,
        "scalp_hard_take_roi_pct": 25.0,
        # ✅ 본전 보호(브레이크이븐): 수익이 어느 정도 나면 SL을 진입가 근처로 끌어올림(가격 기준)
        "trail_breakeven_enable": True,
        "trail_breakeven_at_roi_scalp": 6.0,
        "trail_breakeven_at_roi_swing": 12.0,
        "trail_breakeven_offset_price_pct": 0.05,

        "swing_tp_roi_min": 8.0,
        "swing_tp_roi_max": 80.0,
        # ✅ 스윙 손절(ROI%)은 너무 짧으면 휩쏘로 잘리는 문제가 커서 기본을 더 넓게(요구사항)
        "swing_sl_roi_min": 12.0,
        "swing_sl_roi_max": 30.0,
        "swing_entry_pct_mult": 1.0,
        "swing_lev_cap": 25,

        # ✅ 스윙: 부분익절/순환매도(옵션)
        "swing_partial_tp_enable": True,
        # TP(목표익절)의 비율로 단계 실행(예: TP의 35% 도달 시 1차 부분익절)
        "swing_partial_tp1_at_tp_frac": 0.35, "swing_partial_tp1_close_pct": 33,
        "swing_partial_tp2_at_tp_frac": 0.60, "swing_partial_tp2_close_pct": 33,
        "swing_partial_tp3_at_tp_frac": 0.85, "swing_partial_tp3_close_pct": 34,
        # ✅ (추가) 부분익절 "청산수량"을 USDT(마진)로 지정(사용자 요구)
        # - 0이면 기존 % 청산을 사용
        # - 값이 있으면 해당 USDT(마진)만큼의 포지션을 청산(레버 반영: qty = (usdt*lev)/price)
        "swing_partial_tp1_close_usdt": 0.0,
        "swing_partial_tp2_close_usdt": 0.0,
        "swing_partial_tp3_close_usdt": 0.0,

        # ✅ 사용자 요구: 스윙 순환매 기본 ON
        "swing_recycle_enable": True,
        "swing_recycle_cooldown_min": 10,
        "swing_recycle_max_count": 3,
        "swing_recycle_reentry_roi": 2.5,

        # ✅ 외부 시황 위험 시 신규진입 감산(완전 금지 X)
        "entry_risk_reduce_enable": True,
        "entry_risk_reduce_factor": 0.65,

        # ✅ 지지/저항(SR) 기반 손절/익절
        "use_sr_stop": True,
        "sr_timeframe": "15m",
        "sr_lookback": 220,
        "sr_pivot_order": 6,
        "sr_atr_period": 14,
        "sr_buffer_atr_mult": 0.25,
        "sr_rr_min": 1.5,
        "sr_levels_cache_sec": 60,
        # ✅ 스윙 전용 SR 파라미터(더 큰 매물대/완만한 SL/TP)
        "sr_timeframe_swing": "1h",
        "sr_lookback_swing": 320,
        "sr_pivot_order_swing": 8,
        "sr_buffer_atr_mult_swing": 0.45,
        "sr_rr_min_swing": 2.0,

        # ✅ 추세 필터 정책(기능 유지/확장)
        "trend_filter_enabled": True,
        "trend_filter_timeframe": "1h",
        "trend_filter_cache_sec": 60,
        # "STRICT"=기존처럼 역추세 금지, "ALLOW_SCALP"=역추세 허용하되 스캘핑 강제, "OFF"=미사용
        "trend_filter_policy": "ALLOW_SCALP",

        # ✅ 내보내기(일별 엑셀/구글시트)
        "export_daily_enable": True,
        # ✅ 사용자 요구: Excel 저장 기본 OFF, Google Sheets 저장 기본 ON
        "export_excel_enable": False,
        "export_gsheet_enable": True,  # secrets(GSHEET_ENABLED/...) 설정 필요
        "export_gsheet_spreadsheet_id": "",  # 비워두면 secrets의 GSHEET_ID 사용
        # ✅ Google Sheets 원본 로그(TRADE/EVENT/SCAN) 레거시 모드 허용 여부(기본 OFF)
        # - 사용자 요구: 구글시트에는 "매매일지 + 시간대/일별 총합"만(=trades_only)
        "gsheet_allow_legacy_logs": False,
        # ✅ Google Sheets 표(서식) 자동 적용(권장): 1회만 적용되며, UI에서 강제 재적용 가능
        "gsheet_auto_format_enable": True,
        # ✅ 관리자(명령/버튼) 응답 출력 위치
        # - 요구: "관리자가 봇을 작동하면, 답변은 채널로"
        # - 옵션: "channel"|"admin"|"both"
        "tg_admin_replies_to": "channel",
    }


def load_settings() -> Dict[str, Any]:
    base = default_settings()
    saved = {}
    saved_ver = 0
    if os.path.exists(SETTINGS_FILE):
        saved = read_json_safe(SETTINGS_FILE, {}) or {}
        if isinstance(saved, dict):
            try:
                saved_ver = int(saved.get("settings_schema_version", 0) or 0)
            except Exception:
                saved_ver = 0
    cfg = dict(base)
    if isinstance(saved, dict):
        cfg.update(saved)
    # 이전 키 호환
    if "openai_key" in cfg and not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = cfg["openai_key"]
    # 누락 키 보정
    for k, v in base.items():
        if k not in cfg:
            cfg[k] = v
    # ✅ 기본값 마이그레이션(사용자 요구 반영)
    try:
        base_ver = int(base.get("settings_schema_version", 0) or 0)
    except Exception:
        base_ver = 0
    if saved_ver < base_ver:
        changed = False
        # v2: Google Sheets 기본/순환매 기본/스윙 손절 기본 확장
        if saved_ver < 2:
            try:
                if bool(cfg.get("export_excel_enable", True)) is True:
                    cfg["export_excel_enable"] = False
                    changed = True
            except Exception:
                pass
            try:
                if bool(cfg.get("export_gsheet_enable", False)) is False:
                    cfg["export_gsheet_enable"] = True
                    changed = True
            except Exception:
                pass
            try:
                if bool(cfg.get("swing_recycle_enable", False)) is False:
                    cfg["swing_recycle_enable"] = True
                    changed = True
            except Exception:
                pass
            # 스윙은 스캘핑처럼 -2~-3%에 잘리는 문제를 줄이기 위해 기본 손절 하한을 넓게 유지
            try:
                if float(cfg.get("swing_sl_roi_min", 0.0) or 0.0) < 12.0:
                    cfg["swing_sl_roi_min"] = 12.0
                    changed = True
            except Exception:
                pass
        # v3: 스타일 선택 AI 호출 분리(기본 OFF) + 레짐 전환 플립플롭 완화 기본값
        if saved_ver < 3:
            try:
                if "style_entry_ai_enable" not in saved:
                    cfg["style_entry_ai_enable"] = False
                    changed = True
            except Exception:
                pass
            try:
                # 과거 기본값(2/3)에서 너무 자주 바뀌는 환경이 있어 기본을 더 보수적으로
                if int(cfg.get("regime_confirm_n", 0) or 0) < 3:
                    cfg["regime_confirm_n"] = 3
                    changed = True
            except Exception:
                pass
            try:
                if int(cfg.get("regime_confirm_n_flipback", 0) or 0) < 5:
                    cfg["regime_confirm_n_flipback"] = 5
                    changed = True
            except Exception:
                pass
        # v4: 스캘핑 목표 과대(익절 미발동) 완화 + 스윙 목표 확장(차이 명확화)
        if saved_ver < 4:
            # 스캘핑 가격 기준 상한/하드익절/본전보호 기본값 조정(기존 기본값을 쓰던 경우에만)
            try:
                if float(cfg.get("scalp_sl_price_pct_max", 0.0) or 0.0) == 1.0:
                    cfg["scalp_sl_price_pct_max"] = 0.75
                    changed = True
            except Exception:
                pass
            try:
                if float(cfg.get("scalp_tp_price_pct_max", 0.0) or 0.0) == 1.6:
                    cfg["scalp_tp_price_pct_max"] = 1.20
                    changed = True
            except Exception:
                pass
            try:
                if float(cfg.get("scalp_hard_take_roi_pct", 0.0) or 0.0) == 35.0:
                    cfg["scalp_hard_take_roi_pct"] = 25.0
                    changed = True
            except Exception:
                pass
            try:
                if float(cfg.get("trail_breakeven_at_roi_scalp", 0.0) or 0.0) == 8.0:
                    cfg["trail_breakeven_at_roi_scalp"] = 6.0
                    changed = True
            except Exception:
                pass
            # 스윙 목표 확장(기존 기본값을 쓰던 경우에만)
            try:
                if float(cfg.get("swing_tp_roi_min", 0.0) or 0.0) == 3.0:
                    cfg["swing_tp_roi_min"] = 8.0
                    changed = True
            except Exception:
                pass
            try:
                if float(cfg.get("swing_tp_roi_max", 0.0) or 0.0) == 50.0:
                    cfg["swing_tp_roi_max"] = 80.0
                    changed = True
            except Exception:
                pass
        # v5: AI 시야 리포트 자동 전송 기본 OFF
        if saved_ver < 5:
            try:
                cfg["tg_enable_hourly_vision_report"] = False
                changed = True
            except Exception:
                pass
        # v6: 관리자 DM으로 거래알림(진입/청산) 복사 전송 기본 OFF
        if saved_ver < 6:
            try:
                cfg["tg_trade_alert_to_admin"] = False
                changed = True
            except Exception:
                pass
        # v7: 고정 레버/고정 진입비중 기본 OFF (ATR 레버/리스크 캡 기반으로 유연화)
        if saved_ver < 7:
            try:
                cfg["fixed_leverage_enable"] = False
                changed = True
            except Exception:
                pass
            try:
                cfg["fixed_entry_pct_enable"] = False
                changed = True
            except Exception:
                pass
        # v8: 강제 수익보존(Exit 정책) 기본값을 요청 사양(10/30/50/10)으로 정렬
        # - 사용자가 이전 기본값을 그대로 쓰고 있던 경우에만 업데이트(커스텀 보호)
        if saved_ver < 8:
            try:
                be0 = float(cfg.get("exit_trailing_protect_be_roi", 0.0) or 0.0)
                part0 = float(cfg.get("exit_trailing_protect_partial_roi", 0.0) or 0.0)
                ts0 = float(cfg.get("exit_trailing_protect_trail_start_roi", 0.0) or 0.0)
                dd0 = float(cfg.get("exit_trailing_protect_trail_dd_roi", 0.0) or 0.0)
                # 이전 기본값(8/40/60/12) or 누락(0)인 경우만 교체
                if (be0 in [0.0, 8.0]) and (part0 in [0.0, 40.0]) and (ts0 in [0.0, 60.0]) and (dd0 in [0.0, 12.0]):
                    cfg["exit_trailing_protect_be_roi"] = 10.0
                    cfg["exit_trailing_protect_partial_roi"] = 30.0
                    cfg["exit_trailing_protect_trail_start_roi"] = 50.0
                    cfg["exit_trailing_protect_trail_dd_roi"] = 10.0
                    changed = True
            except Exception:
                pass
        # v9: 리부트 후에도 "신규진입 스타일 AI 호출" 체크가 유지되도록 기본값 정렬
        if saved_ver < 9:
            try:
                if bool(cfg.get("style_entry_ai_enable", False)) is False:
                    cfg["style_entry_ai_enable"] = True
                    changed = True
            except Exception:
                pass
        # v10: 청산은 AI 목표만 사용(요청)
        if saved_ver < 10:
            try:
                if bool(cfg.get("exit_ai_targets_only", False)) is False:
                    cfg["exit_ai_targets_only"] = True
                    changed = True
            except Exception:
                pass
        # v11: 차트 패턴 감지 기본값 추가(미설정 키만 보정)
        if saved_ver < 11:
            try:
                if "use_chart_patterns" not in saved:
                    cfg["use_chart_patterns"] = True
                    changed = True
            except Exception:
                pass
            try:
                if "pattern_gate_entry" not in saved:
                    cfg["pattern_gate_entry"] = True
                    changed = True
            except Exception:
                pass
            try:
                if "pattern_override_ai" not in saved:
                    cfg["pattern_override_ai"] = True
                    changed = True
            except Exception:
                pass
        cfg["settings_schema_version"] = base_ver
        if changed:
            try:
                save_settings(cfg)
            except Exception:
                pass
    return cfg


def save_settings(cfg: Dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, cfg)


config = load_settings()


# =========================================================
# ✅ 5) 런타임 상태(runtime_state.json)
# =========================================================
def default_runtime() -> Dict[str, Any]:
    return {
        "date": today_kst_str(),
        "day_start_equity": 0.0,
        # ✅ 당일 peak equity(드로다운 fail-safe용)
        "peak_equity": 0.0,
        "peak_equity_kst": "",
        "daily_realized_pnl": 0.0,
        # ✅ 당일 통계(실패/과매매 방지용)
        "daily_trade_count": 0,
        "daily_win_count": 0,
        "daily_loss_count": 0,
        "consec_losses": 0,
        "pause_until": 0,
        "cooldowns": {},
        "trades": {},
        # ✅ 직전 청산 기록(재진입/근거 표시용)
        "last_exit": {},
        # ✅ 일별 브리핑/내보내기/상태 보존
        "daily_btc_brief": {},
        "last_export_date": "",
        "open_targets": {},  # sym -> active_targets snapshot
        # ✅ Telegram /scan 강제 스캔 요청
        "force_scan": {},
        # ✅ 워커 리스(중복 스레드/워치독 복구 시 안전장치)
        "worker_lease": {"id": "", "until_epoch": 0.0, "updated_kst": "", "owner": ""},
        "revoked_worker_ids": [],
    }


def load_runtime() -> Dict[str, Any]:
    rt = read_json_safe(RUNTIME_FILE, None)
    if not isinstance(rt, dict):
        rt = default_runtime()
    if rt.get("date") != today_kst_str():
        # 날짜 바뀌면 일일 상태 초기화
        # 단, 오버나잇 포지션의 목표정보(open_targets)는 보존해 -2% 같은 fallback 청산을 방지
        prev_rt = dict(rt) if isinstance(rt, dict) else {}
        rt = default_runtime()
        try:
            ot = prev_rt.get("open_targets", {})
            if isinstance(ot, dict):
                rt["open_targets"] = ot
        except Exception:
            pass
    base = default_runtime()
    for k, v in base.items():
        if k not in rt:
            rt[k] = v
    return rt


def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)


# =========================================================
# ✅ Fail-safe: 조건 충족 시 자동매매 OFF
# =========================================================
def _fail_safe_disable_auto_trade(cfg: Dict[str, Any], rt: Dict[str, Any], mon: Optional[Dict[str, Any]], reason: str, detail: str = "") -> None:
    """
    - 안전장치 발동 시 auto_trade를 끄고, 당일 재가동을 막기 위해 pause_until을 자정까지 올린다.
    - "AI가 사라진다"는 표현 대신, 실제 동작은 '자동매매 OFF'로 구현한다.
    """
    try:
        already_reason = str(rt.get("auto_trade_stop_reason", "") or "")
        already_epoch = float(rt.get("auto_trade_stop_epoch", 0) or 0.0)
        # 같은 이유로 연속 알림 스팸 방지(5분)
        if (not bool(cfg.get("auto_trade", False))) and already_reason == str(reason or "") and already_epoch > 0 and (time.time() - already_epoch) < 300:
            return
    except Exception:
        pass
    try:
        if bool(cfg.get("auto_trade", False)):
            cfg["auto_trade"] = False
            save_settings(cfg)
    except Exception:
        pass
    try:
        rt["pause_until"] = max(float(rt.get("pause_until", 0) or 0.0), float(next_midnight_kst_epoch()))
        rt["auto_trade_stop_reason"] = str(reason or "")
        rt["auto_trade_stop_kst"] = now_kst_str()
        rt["auto_trade_stop_epoch"] = float(time.time())
        save_runtime(rt)
    except Exception:
        pass
    try:
        if mon is not None:
            mon_add_event(mon, "AUTO_TRADE_OFF", "", f"fail_safe:{reason}", {"detail": detail, "code": CODE_VERSION})
    except Exception:
        pass
    try:
        msg = "⛔️ 자동매매 OFF(안전장치)\n" f"- 이유: {reason}\n"
        if detail:
            msg += f"- 상세: {detail}\n"
        msg += f"- 시각: {now_kst_str()}"
        tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg)
    except Exception:
        pass


def maybe_trigger_fail_safe(cfg: Dict[str, Any], rt: Dict[str, Any], total_equity: float, mon: Optional[Dict[str, Any]] = None, where: str = "") -> bool:
    """
    사용자 요구(의도): "수익을 내지 못하거나 전부 잃으면 AI는 없어진다"
    현실 구현: '손실 확대/과매매'를 막는 fail-safe로 자동매매를 강제 종료한다.
    """
    try:
        if not bool(cfg.get("fail_safe_enable", True)):
            return False
    except Exception:
        return False

    triggered = False

    # 1) Drawdown from peak (당일 peak 기준)
    try:
        if bool(cfg.get("fail_safe_drawdown_enable", True)) and float(total_equity or 0.0) > 0:
            pk = _as_float(rt.get("peak_equity", 0.0), 0.0)
            if pk <= 0 or float(total_equity) > float(pk):
                rt["peak_equity"] = float(total_equity)
                rt["peak_equity_kst"] = now_kst_str()
                pk = float(total_equity)
                try:
                    save_runtime(rt)
                except Exception:
                    pass
            dd_pct = ((float(pk) - float(total_equity)) / float(pk) * 100.0) if float(pk) > 0 else 0.0
            lim_dd = float(cfg.get("fail_safe_drawdown_from_peak_pct", 30.0) or 30.0)
            if lim_dd > 0 and dd_pct >= float(lim_dd):
                _fail_safe_disable_auto_trade(
                    cfg,
                    rt,
                    mon,
                    reason="DRAWDOWN_LIMIT",
                    detail=f"peak {pk:.2f} → now {float(total_equity):.2f} ({dd_pct:.1f}% ≥ {lim_dd:.1f}%) @ {where}",
                )
                triggered = True
    except Exception:
        pass

    # 2) Profit guard (오늘 일정 횟수 거래 후에도 수익이 없으면 OFF)
    try:
        if (not triggered) and bool(cfg.get("fail_safe_profit_guard_enable", False)):
            n_need = int(cfg.get("fail_safe_profit_guard_min_trades", 10) or 10)
            min_pnl = float(cfg.get("fail_safe_profit_guard_min_pnl_usdt", 0.0) or 0.0)
            n_tr = int(rt.get("daily_trade_count", 0) or 0)
            pnl = float(rt.get("daily_realized_pnl", 0.0) or 0.0)
            if n_need > 0 and n_tr >= n_need and pnl <= float(min_pnl):
                _fail_safe_disable_auto_trade(
                    cfg,
                    rt,
                    mon,
                    reason="PROFIT_GUARD",
                    detail=f"trades {n_tr} / pnl {pnl:+.2f} ≤ {min_pnl:+.2f} @ {where}",
                )
                triggered = True
    except Exception:
        pass

    return bool(triggered)


# =========================================================
# ✅ 5.2) 워커 리스(Watchdog 복구 시 중복매매 방지)
# - TG_THREAD가 "살아있지만 멈춘 상태"일 수 있어 watchdog이 새 스레드를 띄울 수 있음
# - 이때 중복 주문을 막기 위해, 런타임에 리스(owner)를 두고 1개 워커만 "리더"로 동작한다.
# =========================================================
WORKER_LEASE_TTL_SEC = 45.0


def _runtime_revoked_ids(rt: Dict[str, Any]) -> List[str]:
    try:
        xs = rt.get("revoked_worker_ids", []) or []
        if isinstance(xs, list):
            return [str(x) for x in xs if str(x)]
    except Exception:
        pass
    return []


def runtime_is_worker_revoked(worker_id: str) -> bool:
    try:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        rt = load_runtime()
        return wid in set(_runtime_revoked_ids(rt))
    except Exception:
        return False


def runtime_worker_lease_touch(worker_id: str, owner: str = "TG_THREAD", ttl_sec: float = WORKER_LEASE_TTL_SEC) -> bool:
    """
    리스(leader) 확보/연장.
    - True: 이 worker_id가 리더(동작 허용)
    - False: 다른 리더가 있음/본인 revoked
    """
    wid = str(worker_id or "").strip()
    if not wid:
        return False
    try:
        rt = load_runtime()
        if wid in set(_runtime_revoked_ids(rt)):
            return False
        lease = rt.get("worker_lease", {}) or {}
        if not isinstance(lease, dict):
            lease = {}
        cur_id = str(lease.get("id", "") or "").strip()
        until = float(lease.get("until_epoch", 0) or 0)
        now_ts = time.time()
        if (not cur_id) or (cur_id == wid) or (now_ts >= until):
            lease["id"] = wid
            lease["owner"] = str(owner or "")[:40]
            lease["until_epoch"] = now_ts + float(ttl_sec)
            lease["updated_kst"] = now_kst_str()
            rt["worker_lease"] = lease
            save_runtime(rt)
            return True
        return False
    except Exception:
        return False


def runtime_worker_lease_get() -> Dict[str, Any]:
    try:
        rt = load_runtime()
        lease = rt.get("worker_lease", {}) or {}
        if not isinstance(lease, dict):
            lease = {}
        return {
            "id": str(lease.get("id", "") or ""),
            "until_epoch": float(lease.get("until_epoch", 0) or 0),
            "until_kst": _epoch_to_kst_str(float(lease.get("until_epoch", 0) or 0)) if float(lease.get("until_epoch", 0) or 0) else "",
            "owner": str(lease.get("owner", "") or ""),
            "revoked_ids": _runtime_revoked_ids(rt),
        }
    except Exception:
        return {"id": "", "until_epoch": 0.0, "until_kst": "", "owner": "", "revoked_ids": []}


def runtime_worker_revoke(worker_id: str, reason: str = "") -> None:
    """
    watchdog에서 호출: 기존 워커를 revoked로 표시하고 lease를 해제해 새 워커가 리더가 되게 한다.
    """
    wid = str(worker_id or "").strip()
    if not wid:
        return
    try:
        rt = load_runtime()
        revoked = _runtime_revoked_ids(rt)
        if wid not in revoked:
            revoked.append(wid)
        # 과도 누적 방지
        revoked = revoked[-40:]
        rt["revoked_worker_ids"] = revoked
        lease = rt.get("worker_lease", {}) or {}
        if isinstance(lease, dict) and str(lease.get("id", "") or "").strip() == wid:
            lease["until_epoch"] = 0.0
            lease["updated_kst"] = now_kst_str()
            lease["owner"] = f"REVOKED:{str(reason or '')[:30]}"
            rt["worker_lease"] = lease
        save_runtime(rt)
    except Exception:
        return


# =========================================================
# ✅ 6) 매매일지 CSV (기존 유지 + 표시용 이모티콘/내보내기 확장)
# =========================================================
def _read_csv_header_cols(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = (f.readline() or "").strip()
        if header.startswith("\ufeff"):
            header = header.lstrip("\ufeff")
        cols = [c.strip() for c in header.split(",") if c.strip()]
        return cols
    except Exception:
        return []


def log_trade(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_amount: float,
    pnl_percent: float,
    reason: str,
    one_line: str = "",
    review: str = "",
    trade_id: str = "",
    balance_before_total: Optional[float] = None,
    balance_after_total: Optional[float] = None,
    balance_before_free: Optional[float] = None,
    balance_after_free: Optional[float] = None,
) -> None:
    # ⚠️ CSV 컬럼 호환성 유지: 기존 컬럼 유지하면서 안전하게 append
    base_cols = [
        "Time",
        "Coin",
        "Side",
        "Entry",
        "Exit",
        "PnL_USDT",
        "PnL_Percent",
        "BalanceBefore_Total",
        "BalanceAfter_Total",
        "BalanceBefore_Free",
        "BalanceAfter_Free",
        "Reason",
        "OneLine",
        "Review",
        "TradeID",
    ]
    try:
        row_dict = {
            "Time": now_kst_str(),
            "Coin": coin,
            "Side": side,
            "Entry": entry_price,
            "Exit": exit_price,
            "PnL_USDT": pnl_amount,
            "PnL_Percent": pnl_percent,
            "BalanceBefore_Total": "" if balance_before_total is None else float(balance_before_total),
            "BalanceAfter_Total": "" if balance_after_total is None else float(balance_after_total),
            "BalanceBefore_Free": "" if balance_before_free is None else float(balance_before_free),
            "BalanceAfter_Free": "" if balance_after_free is None else float(balance_after_free),
            "Reason": reason,
            "OneLine": one_line,
            "Review": review,
            "TradeID": trade_id,
        }

        if not os.path.exists(LOG_FILE):
            pd.DataFrame([row_dict], columns=base_cols).to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        else:
            existing_cols = _read_csv_header_cols(LOG_FILE)
            cols = existing_cols if existing_cols else base_cols
            # 기존 파일에 신규 컬럼이 없다면 헤더/기록을 업그레이드(삭제 없이 컬럼만 추가)
            try:
                missing = [c for c in base_cols if c not in cols]
                if missing:
                    df_old = pd.read_csv(LOG_FILE)
                    for c in missing:
                        if c not in df_old.columns:
                            df_old[c] = ""
                    new_cols = cols + missing
                    tmp = LOG_FILE + ".tmp"
                    df_old.to_csv(tmp, index=False, encoding="utf-8-sig")
                    os.replace(tmp, LOG_FILE)
                    cols = new_cols
            except Exception:
                pass
            # 기존 파일 헤더와 컬럼 순서 맞춤(누락값은 공백)
            out = {c: row_dict.get(c, "") for c in cols}
            pd.DataFrame([out], columns=cols).to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    # ✅ Google Sheets 매매일지(요구사항: TRADE 이벤트) - CSV와 동일한 정보를 payload로 남김
    try:
        if gsheet_is_enabled():
            gsheet_log_trade(
                stage="JOURNAL",
                symbol=str(coin or ""),
                trade_id=str(trade_id or ""),
                message=str(reason or "")[:160],
                payload={
                    "time_kst": row_dict.get("Time"),
                    "coin": row_dict.get("Coin"),
                    "side": row_dict.get("Side"),
                    "entry": row_dict.get("Entry"),
                    "exit": row_dict.get("Exit"),
                    "pnl_usdt": row_dict.get("PnL_USDT"),
                    "pnl_pct": row_dict.get("PnL_Percent"),
                    "balance_before_total": row_dict.get("BalanceBefore_Total"),
                    "balance_after_total": row_dict.get("BalanceAfter_Total"),
                    "balance_before_free": row_dict.get("BalanceBefore_Free"),
                    "balance_after_free": row_dict.get("BalanceAfter_Free"),
                    "reason": row_dict.get("Reason"),
                    "one_line": row_dict.get("OneLine"),
                    "review": str(row_dict.get("Review", ""))[:800],
                    "trade_id": row_dict.get("TradeID"),
                },
            )
    except Exception:
        pass


def read_trade_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        if "Time" in df.columns:
            df = df.sort_values("Time", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def reset_trade_log() -> None:
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception:
        pass


def get_past_mistakes_text(max_items: int = 5) -> str:
    df = read_trade_log()
    if df.empty or "PnL_Percent" not in df.columns:
        return "과거 매매 기록 없음."
    try:
        lines = []

        # 1) 전체 승률 및 최근 20회 승률
        try:
            pnl_col = df["PnL_Percent"].astype(float)
            total = len(df)
            wins = int((pnl_col > 0).sum())
            wr_all = wins / max(1, total) * 100.0
            recent = df.tail(20)
            pnl_recent = recent["PnL_Percent"].astype(float)
            wr_recent = float((pnl_recent > 0).sum()) / max(1, len(recent)) * 100.0
            avg_win = float(pnl_col[pnl_col > 0].mean()) if (pnl_col > 0).any() else 0.0
            avg_loss = float(pnl_col[pnl_col < 0].mean()) if (pnl_col < 0).any() else 0.0
            lines.append(f"[성과 요약] 전체 승률 {wr_all:.1f}% ({wins}/{total}) | 최근20회 승률 {wr_recent:.1f}% | 평균수익 {avg_win:.2f}% | 평균손실 {avg_loss:.2f}%")
        except Exception:
            pass

        # 2) 최악 손실 상위 N개
        try:
            worst = df.sort_values("PnL_Percent", ascending=True).head(max_items)
            lines.append("[최악 손실]")
            for _, r in worst.iterrows():
                lines.append(
                    f"- {r.get('Coin','?')} {r.get('Side','?')} {float(r.get('PnL_Percent',0)):.2f}% 손실 | 이유: {str(r.get('Reason',''))[:40]}"
                )
        except Exception:
            pass

        # 3) 최고 수익 상위 3개 (성공 패턴 학습)
        try:
            best = df[df["PnL_Percent"].astype(float) > 0].sort_values("PnL_Percent", ascending=False).head(3)
            if not best.empty:
                lines.append("[최고 수익 패턴 - 이런 진입을 우선시해라]")
                for _, r in best.iterrows():
                    lines.append(
                        f"- {r.get('Coin','?')} {r.get('Side','?')} +{float(r.get('PnL_Percent',0)):.2f}% 수익 | {str(r.get('Reason',''))[:40]}"
                    )
        except Exception:
            pass

        return "\n".join(lines) if lines else "큰 손실 기록 없음."
    except Exception:
        return "기록 조회 실패"


# =========================================================
# ✅ 6.5) 일별 내보내기(엑셀/구글시트)
# =========================================================
def _day_df_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df is None or df.empty or "Time" not in df.columns:
        return pd.DataFrame()
    try:
        # Time이 "YYYY-MM-DD HH:MM:SS"
        return df[df["Time"].astype(str).str.startswith(str(date_str))].copy()
    except Exception:
        return pd.DataFrame()


def _trade_day_summary(df_day: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "date": today_kst_str(),
        "trades": 0,
        "win_rate_pct": 0.0,
        "total_pnl_usdt": 0.0,
        "avg_pnl_pct": 0.0,
        "max_dd_pct": 0.0,
        "profit_factor": 0.0,
    }
    if df_day is None or df_day.empty:
        return out
    try:
        pnl_pct = pd.to_numeric(df_day.get("PnL_Percent", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        pnl_usdt = pd.to_numeric(df_day.get("PnL_USDT", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        out["trades"] = int(len(df_day))
        wins = (pnl_pct > 0).sum()
        out["win_rate_pct"] = float(wins / max(1, len(df_day)) * 100.0)
        out["total_pnl_usdt"] = float(pnl_usdt.sum())
        out["avg_pnl_pct"] = float(pnl_pct.mean())
        # 간이 MDD: 누적 PnL% 기준(정확한 equity curve는 아님)
        eq = pnl_pct.cumsum()
        dd = (eq - eq.cummax()).min() if len(eq) else 0.0
        out["max_dd_pct"] = float(dd)
        gains = pnl_usdt[pnl_usdt > 0].sum()
        losses = (-pnl_usdt[pnl_usdt < 0]).sum()
        out["profit_factor"] = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        return out
    except Exception:
        return out


def export_trade_log_daily(date_str: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    - daily_reports/ 아래 날짜별 파일 생성
    - openpyxl 있으면 xlsx, 없으면 csv로 fallback
    - Google Sheets는 설정/시크릿 있을 때만
    """
    df = read_trade_log()
    df_day = _day_df_filter(df, date_str)
    summary = _trade_day_summary(df_day)
    summary["date"] = date_str
    out = {"ok": True, "date": date_str, "rows": int(len(df_day)), "excel_path": "", "csv_path": "", "gsheet": ""}

    try:
        # 표시용 이모티콘 컬럼 추가(파일 내보내기에도 반영)
        if df_day is not None and not df_day.empty and "PnL_Percent" in df_day.columns:
            pnl_pct = pd.to_numeric(df_day["PnL_Percent"], errors="coerce")
            df_day = df_day.copy()
            df_day.insert(
                0,
                "상태",
                pnl_pct.apply(lambda v: "🟢 수익" if pd.notna(v) and float(v) > 0 else ("🔴 손실" if pd.notna(v) and float(v) < 0 else "⚪ 보합")),
            )
    except Exception:
        pass

    if not cfg.get("export_daily_enable", True):
        out["ok"] = False
        out["error"] = "export_daily_enable=OFF"
        return out

    # Excel
    if cfg.get("export_excel_enable", True):
        try:
            xlsx_path = os.path.join(DAILY_REPORT_DIR, f"trade_log_{date_str}.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
                df_day.to_excel(writer, sheet_name="trades", index=False)
            out["excel_path"] = xlsx_path
        except Exception as e:
            out["excel_path"] = ""
            out["excel_error"] = str(e)

    # CSV fallback(항상 생성해두면 편함)
    try:
        csv_path = os.path.join(DAILY_REPORT_DIR, f"trade_log_{date_str}.csv")
        df_day.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out["csv_path"] = csv_path
    except Exception:
        pass

    # Google Sheets (optional)
    # - 요구사항: GSHEET_ENABLED == "true" 일 때만 동작
    if cfg.get("export_gsheet_enable", False) and str(st.secrets.get("GSHEET_ENABLED", "")).strip().lower() == "true":
        try:
            res = export_trade_log_to_gsheet(date_str, df_day, summary, cfg)
            out["gsheet"] = res.get("msg", "")
            if not res.get("ok", False):
                out["gsheet_error"] = res.get("error", "")
        except Exception as e:
            out["gsheet_error"] = str(e)
    elif cfg.get("export_gsheet_enable", False):
        out["gsheet_error"] = "GSHEET_ENABLED != 'true'"

    return out


def _get_gsheet_client_from_secrets() -> Optional[Any]:
    """
    Streamlit secrets 예시:
    - [gcp_service_account] (dict 형태)
    - 혹은 GOOGLE_SERVICE_ACCOUNT_JSON (JSON 문자열)
    """
    if gspread is None or GoogleCredentials is None:
        return None
    try:
        # ✅ 요구사항: GSHEET_SERVICE_ACCOUNT_JSON (멀티라인 포함) 지원
        info = None
        try:
            info = _gsheet_service_account_info()  # type: ignore[name-defined]
        except Exception:
            info = None
        if not info:
            return None
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def export_trade_log_to_gsheet(date_str: str, df_day: pd.DataFrame, summary: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if gspread is None:
        return {"ok": False, "error": "gspread 미설치(requirements.txt에 gspread/google-auth 추가)", "msg": ""}
    if str(st.secrets.get("GSHEET_ENABLED", "")).strip().lower() != "true":
        return {"ok": False, "error": "GSHEET_ENABLED != 'true'", "msg": ""}
    client = _get_gsheet_client_from_secrets()
    if client is None:
        return {"ok": False, "error": "Google 서비스 계정 secrets 없음", "msg": ""}

    sid = (cfg.get("export_gsheet_spreadsheet_id") or "").strip() or str(st.secrets.get("GSHEET_SPREADSHEET_ID") or "").strip() or str(st.secrets.get("GSHEET_ID") or "").strip()
    if not sid:
        return {"ok": False, "error": "GSHEET_SPREADSHEET_ID 미설정(secrets 또는 설정)", "msg": ""}

    try:
        sh = client.open_by_key(sid)
        # 날짜별 워크시트 생성/갱신
        title = str(date_str)
        try:
            ws = sh.worksheet(title)
        except Exception:
            ws = sh.add_worksheet(title=title, rows=2000, cols=30)

        # summary 먼저
        ws.clear()
        sum_rows = [["key", "value"]] + [[k, str(v)] for k, v in summary.items()]
        ws.update("A1", sum_rows)

        # trades 테이블
        start_row = len(sum_rows) + 2
        if df_day is not None and not df_day.empty:
            df2 = df_day.copy()
            df2 = df2.fillna("")
            values = [df2.columns.tolist()] + df2.astype(str).values.tolist()
            ws.update(f"A{start_row}", values)
        return {"ok": True, "msg": f"Google Sheets 업데이트 완료({title})"}
    except Exception as e:
        return {"ok": False, "error": str(e), "msg": ""}


# =========================================================
# ✅ 7) Secrets (Bitget / Telegram / OpenAI)
# =========================================================
try:
    # 일부 환경/스레드에서 st.secrets 접근이 예외가 날 수 있어 fallback 스냅샷 유지
    _SECRETS_SNAPSHOT = dict(st.secrets)
except Exception:
    _SECRETS_SNAPSHOT = {}


def _sget(key: str, default: Any = "") -> Any:
    try:
        return st.secrets.get(key, default)
    except Exception:
        try:
            return _SECRETS_SNAPSHOT.get(key, default)
        except Exception:
            return default


def _sget_str(key: str, default: str = "") -> str:
    try:
        v = _sget(key, default)
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return str(default).strip()


def _parse_id_set(csv_like: str) -> set:
    s = str(csv_like or "").strip()
    if not s:
        return set()
    out = set()
    for p in re.split(r"[,\s]+", s):
        p = p.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _boolish(v: Any) -> bool:
    return str(v or "").strip().lower() in ["true", "1", "yes", "y", "on"]


# ✅ Bitget Secrets (요구사항 규격)
api_key = _sget_str("BITGET_API_KEY") or _sget_str("API_KEY")
api_secret = _sget_str("BITGET_API_SECRET") or _sget_str("API_SECRET")
api_password = _sget_str("BITGET_API_PASSPHRASE") or _sget_str("API_PASSWORD")

# ✅ Telegram Secrets (요구사항 규격)
tg_token = _sget_str("TG_TOKEN")
tg_target_chat_id = _sget_str("TG_TARGET_CHAT_ID") or _sget_str("TG_CHAT_ID")

# (확장) TG_CHANNEL_ID / TG_GROUP_ID가 있으면 자동 감지해 라우팅
tg_channel_id = _sget_str("TG_CHANNEL_ID") or _sget_str("TG_CHAT_ID_CHANNEL") or _sget_str("TG_CHAT_ID_CHANNEL_ID")
tg_group_id = _sget_str("TG_GROUP_ID") or _sget_str("TG_CHAT_ID_GROUP") or _sget_str("TG_CHAT_ID_GROUP_ID")

tg_id_default = tg_target_chat_id
if tg_channel_id or tg_group_id:
    tg_id_channel = tg_channel_id or tg_target_chat_id
    tg_id_group = tg_group_id or tg_target_chat_id
else:
    tg_id_channel = tg_target_chat_id
    tg_id_group = tg_target_chat_id

TG_ADMIN_IDS = _parse_id_set(_sget_str("TG_ADMIN_USER_IDS"))

if not api_key:
    st.error("🚨 Bitget API Key가 없습니다. Secrets에 BITGET_API_KEY/BITGET_API_SECRET/BITGET_API_PASSPHRASE 설정하세요.")
    st.stop()


_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}
_OPENAI_CLIENT_LOCK = threading.RLock()


# =========================================================
# ✅ OpenAI Health/Suspension (쿼터/레이트리밋 대응)
# - 429(insufficient_quota) 같은 오류가 반복되면 스캔/스레드가 "계속 오류"처럼 보일 수 있어
#   일정 시간 OpenAI 호출을 자동 중지(suspend)해서 스팸/부하를 줄인다.
# - 키를 바꾸면(suffix/len 변화) 자동으로 suspend를 해제한다.
# =========================================================
_OPENAI_HEALTH_LOCK = threading.RLock()
_OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
_OPENAI_SUSPENDED_REASON = ""
_OPENAI_SUSPENDED_KEY_FPR = ""
_OPENAI_LAST_ERROR_SUMMARY = ""
_OPENAI_LAST_ERROR_EPOCH = 0.0


def _openai_key_fingerprint(key: str) -> str:
    try:
        k = str(key or "")
        if not k:
            return ""
        suf = k[-4:] if len(k) >= 4 else k
        return f"len{len(k)}..{suf}"
    except Exception:
        return ""


def _openai_err_kind(err: BaseException) -> str:
    """
    OpenAI 오류를 대략 분류(라이브러리 버전 차이/에러 형태 차이를 흡수).
    """
    try:
        name = str(type(err).__name__ or "").lower()
    except Exception:
        name = ""
    try:
        s = str(err or "").lower()
    except Exception:
        s = ""

    # quota/결제 부족
    if "insufficient_quota" in s or "exceeded your current quota" in s or "plan and billing" in s:
        return "insufficient_quota"
    # 잘못된 키
    if "invalid_api_key" in s or "incorrect api key" in s or "api key" in s and "invalid" in s:
        return "invalid_api_key"
    # rate limit
    if "ratelimit" in name or ("rate limit" in s and "insufficient_quota" not in s):
        return "rate_limit"
    # timeout
    if "timeout" in s or "timed out" in s:
        return "timeout"
    return "other"


def openai_health_info(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    returns:
      - available: bool
      - status: OK|NO_KEY|SUSPENDED
      - message: human readable(KO)
      - until_kst: str (when suspended)
    """
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    if not key:
        return {"available": False, "status": "NO_KEY", "message": "OpenAI 키 없음", "until_kst": ""}

    fpr = _openai_key_fingerprint(key)
    now = time.time()
    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        # 키가 바뀌면 suspend 해제
        if _OPENAI_SUSPENDED_KEY_FPR and _OPENAI_SUSPENDED_KEY_FPR != fpr:
            _OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
            _OPENAI_SUSPENDED_REASON = ""
            _OPENAI_SUSPENDED_KEY_FPR = ""

        if now < float(_OPENAI_SUSPENDED_UNTIL_EPOCH or 0.0) and _OPENAI_SUSPENDED_KEY_FPR == fpr:
            until_kst = _epoch_to_kst_str(float(_OPENAI_SUSPENDED_UNTIL_EPOCH))
            reason = str(_OPENAI_SUSPENDED_REASON or "").strip() or "일시 중지"
            return {"available": False, "status": "SUSPENDED", "message": f"OpenAI 일시중지: {reason}", "until_kst": until_kst}

    return {"available": True, "status": "OK", "message": "OpenAI OK", "until_kst": ""}


def openai_suspend(cfg: Optional[Dict[str, Any]], reason: str, duration_sec: int, err: Optional[BaseException] = None) -> None:
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    fpr = _openai_key_fingerprint(key)
    until = time.time() + float(max(5, int(duration_sec)))
    msg_err = ""
    try:
        msg_err = str(err)[:240] if err is not None else ""
    except Exception:
        msg_err = ""

    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        global _OPENAI_LAST_ERROR_SUMMARY, _OPENAI_LAST_ERROR_EPOCH
        _OPENAI_SUSPENDED_UNTIL_EPOCH = float(until)
        _OPENAI_SUSPENDED_REASON = str(reason or "").strip()[:120]
        _OPENAI_SUSPENDED_KEY_FPR = str(fpr or "")
        _OPENAI_LAST_ERROR_SUMMARY = msg_err
        _OPENAI_LAST_ERROR_EPOCH = time.time()

    try:
        gsheet_log_event(
            "OPENAI_SUSPEND",
            message=str(reason or "suspend"),
            payload={"until_kst": _epoch_to_kst_str(float(until)), "duration_sec": int(duration_sec), "err": msg_err},
        )
    except Exception:
        pass


def openai_handle_failure(err: BaseException, cfg: Optional[Dict[str, Any]], where: str = "") -> str:
    """
    OpenAI 실패를 분류하고, 필요 시 suspend 설정.
    returns: kind string
    """
    kind = _openai_err_kind(err)
    # quota 부족은 모델을 바꿔도 해결되지 않으므로 길게 suspend
    if kind == "insufficient_quota":
        openai_suspend(cfg, reason="insufficient_quota(API 결제/크레딧: ChatGPT와 별개)", duration_sec=6 * 60 * 60, err=err)
    elif kind == "invalid_api_key":
        openai_suspend(cfg, reason="invalid_api_key(키 오류)", duration_sec=10 * 60, err=err)
    elif kind == "rate_limit":
        openai_suspend(cfg, reason="rate_limit(잠시 대기)", duration_sec=120, err=err)
    elif kind == "timeout":
        openai_suspend(cfg, reason="timeout(잠시 대기)", duration_sec=60, err=err)
    else:
        # 기타 오류도 짧게 suspend 해서 스팸/부하 완화
        openai_suspend(cfg, reason="openai_error(잠시 대기)", duration_sec=45, err=err)
    return kind


def openai_clear_suspension(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    수동 테스트/운영자가 결제/쿼터를 복구한 직후 즉시 재시도할 수 있게 suspend를 해제.
    - 자동매매/스캔 루프에서는 사용하지 않는 것이 안전.
    """
    cfg = cfg or {}
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    fpr = _openai_key_fingerprint(key)
    with _OPENAI_HEALTH_LOCK:
        global _OPENAI_SUSPENDED_UNTIL_EPOCH, _OPENAI_SUSPENDED_REASON, _OPENAI_SUSPENDED_KEY_FPR
        if not _OPENAI_SUSPENDED_KEY_FPR:
            return
        if fpr and _OPENAI_SUSPENDED_KEY_FPR != fpr:
            # 다른 키면 이미 openai_health_info()에서 자동 해제되지만, 안전하게 클리어
            pass
        _OPENAI_SUSPENDED_UNTIL_EPOCH = 0.0
        _OPENAI_SUSPENDED_REASON = ""
        _OPENAI_SUSPENDED_KEY_FPR = ""
    try:
        gsheet_log_event("OPENAI_UNSUSPEND", message="manual_clear", payload={"code": CODE_VERSION})
    except Exception:
        pass


def get_openai_client(cfg: Dict[str, Any]) -> Optional[OpenAI]:
    # ✅ secrets 규격(요구사항): OPENAI_API_KEY
    # - 일부 환경에서 st.secrets.get 호환 이슈를 피하기 위해 _sget_str 사용
    key = _sget_str("OPENAI_API_KEY") or str(cfg.get("openai_api_key", "") or "").strip()
    if not key:
        return None
    # suspend 상태면 호출하지 않음(스팸/부하 방지)
    try:
        h = openai_health_info(cfg)
        if not bool(h.get("available", False)):
            return None
    except Exception:
        pass
    with _OPENAI_CLIENT_LOCK:
        if key in _OPENAI_CLIENT_CACHE:
            return _OPENAI_CLIENT_CACHE[key]
        try:
            c = OpenAI(api_key=key)
            _OPENAI_CLIENT_CACHE[key] = c
            return c
        except Exception:
            return None


def _call_with_timeout(fn, timeout_sec: int):
    # 스레드가 멈추는 걸 방지하기 위해 OpenAI 같은 외부 호출에 hard-timeout을 건다.
    global _THREAD_POOL, _THREAD_POOL_CREATED_EPOCH
    # ✅ safety: executor lock이 어떤 이유로든 오래 잡히면(비정상) 호출 스레드가 영구 정체될 수 있음
    # → lock 획득도 hard-timeout 처리
    got = False
    try:
        got = bool(_THREAD_POOL_LOCK.acquire(timeout=0.8))
    except Exception:
        got = False
    if not got:
        raise FuturesTimeoutError("thread_pool_lock_timeout")
    try:
        try:
            fut = _THREAD_POOL.submit(fn)
        except RuntimeError as e:
            # Streamlit 재기동/프로세스 종료 단계 등에서 executor가 shutdown 상태가 될 수 있음
            # → 1회는 새 executor로 복구 시도(봇 스레드 지속성 강화)
            msg = str(e or "").lower()
            if "cannot schedule new futures" in msg or "shutdown" in msg:
                _THREAD_POOL = ThreadPoolExecutor(max_workers=4)
                _THREAD_POOL_CREATED_EPOCH = time.time()
                fut = _THREAD_POOL.submit(fn)
            else:
                raise
    finally:
        try:
            _THREAD_POOL_LOCK.release()
        except Exception:
            pass
    try:
        return fut.result(timeout=timeout_sec)
    except FuturesTimeoutError as e:
        # timeout 발생 시 future를 취소 시도해 워커 누적을 완화
        try:
            fut.cancel()
        except Exception:
            pass
        raise FuturesTimeoutError(f"timeout({int(timeout_sec)}s)") from e


def openai_chat_create_with_fallback(
    client: OpenAI,
    models: List[str],
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_sec: int = OPENAI_TIMEOUT_SEC,
) -> Tuple[str, Any]:
    """
    OpenAI 호출 모델 fallback:
    - 일부 계정/환경에서 특정 모델이 없을 수 있어(예: gpt-4o 미지원) 순차 시도
    - 성공 시 (model_used, response) 반환
    """
    last_err: Optional[BaseException] = None
    tried: List[str] = []
    for m in models:
        m2 = str(m or "").strip()
        if not m2:
            continue
        tried.append(m2)
        try:
            def _do(use_response_format: bool = True):
                kwargs: Dict[str, Any] = {
                    "model": m2,
                    "messages": messages,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                }
                if response_format is not None and use_response_format:
                    kwargs["response_format"] = response_format
                return client.chat.completions.create(**kwargs)

            resp = _call_with_timeout(_do, timeout_sec)
            return m2, resp
        except FuturesTimeoutError as e:
            last_err = e
            continue
        except TypeError as e:
            # 일부 openai 라이브러리/환경에서 response_format 파라미터가 지원되지 않을 수 있음
            # (예: "got an unexpected keyword argument 'response_format'")
            msg = str(e or "")
            if response_format is not None and ("response_format" in msg):
                try:
                    resp = _call_with_timeout(lambda: _do(use_response_format=False), timeout_sec)
                    return m2, resp
                except Exception as e2:
                    last_err = e2
                    continue
            last_err = e
            continue
        except Exception as e:
            # 모델 자체가 response_format을 지원하지 않는 경우도 있어, 1회는 response_format 없이 재시도
            msg = str(e or "")
            if response_format is not None and ("response_format" in msg.lower()):
                try:
                    resp = _call_with_timeout(lambda: _do(use_response_format=False), timeout_sec)
                    return m2, resp
                except Exception as e2:
                    last_err = e2
                    continue
            # quota/키오류 등은 모델 바꿔도 해결되지 않으므로 즉시 중단
            kind = ""
            try:
                kind = _openai_err_kind(e)
            except Exception:
                kind = ""
            if kind in ["insufficient_quota", "invalid_api_key"]:
                raise e
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"OpenAI call failed (models_tried={tried})")


# =========================================================
# ✅ 7.5) Google Sheets Logger
# - 기본(권장): TRADES_ONLY 모드
#   - 구글시트에는 "매매일지(TRADE_LOG)" + "시간대별/일별 총합"만 저장
#   - SCAN/EVENT는 구글시트에 저장하지 않음(레이트리밋/가독성 문제)
# - (옵션) 레거시 모드: TRADE/EVENT/SCAN 원본 로그를 그대로 append
#
# 🔧 secrets(옵션)
# - GSHEET_MODE="trades_only"(기본) | "legacy"
# =========================================================

# (레거시) 원본 로그 헤더
GSHEET_HEADER = ["time_kst", "type", "stage", "symbol", "tf", "signal", "score", "trade_id", "message", "payload_json"]

# (기본) 매매일지(=trade_log.csv) 헤더
# - 요청: Google Sheets에는 "한글 + 직관" 형태로 보이게
# - 내부 CSV(trade_log.csv)는 기존 컬럼을 유지하며, 시트에 올릴 때만 한글 헤더/표현으로 매핑한다.
GSHEET_TRADE_JOURNAL_HEADER_EN = ["Time", "Coin", "Side", "Entry", "Exit", "PnL_USDT", "PnL_Percent", "Reason", "OneLine", "Review", "TradeID"]
GSHEET_TRADE_JOURNAL_HEADER = [
    "상태",
    "시간(KST)",
    "코인",
    "방향",
    "진입가",
    "청산가",
    "손익(USDT)",
    "수익률(%)",
    "진입전 총자산(USDT)",
    "청산후 총자산(USDT)",
    "진입전 가용(USDT)",
    "청산후 가용(USDT)",
    "사유",
    "한줄평",
    "후기",
    "일지ID",
]

# (기본) 시간대/일별 총합 헤더(한글)
GSHEET_HOURLY_SUMMARY_HEADER_EN = ["Hour(KST)", "Trades", "WinRate(%)", "TotalPnL(USDT)", "AvgPnL(%)", "ProfitFactor", "AsOf(KST)"]
GSHEET_DAILY_SUMMARY_HEADER_EN = ["Date(KST)", "Trades", "WinRate(%)", "TotalPnL(USDT)", "AvgPnL(%)", "MaxDD(%)", "ProfitFactor", "AsOf(KST)"]
GSHEET_HOURLY_SUMMARY_HEADER = ["시간대(KST)", "거래수", "승률(%)", "총손익(USDT)", "평균수익률(%)", "PF", "갱신시각(KST)"]
GSHEET_DAILY_SUMMARY_HEADER = ["날짜(KST)", "거래수", "승률(%)", "총손익(USDT)", "평균수익률(%)", "최대DD(%)", "PF", "갱신시각(KST)"]

# ✅ 회고 모음(손실 트레이드만) - Google Sheets 전용(요구사항)
GSHEET_REVIEWS_HEADER = [
    "시간(KST)",
    "코인",
    "방향",
    "수익률(%)",
    "손익(USDT)",
    "진입전 총자산(USDT)",
    "청산후 총자산(USDT)",
    "사유",
    "한줄평",
    "후기(개선점)",
    "일지ID",
]

# trades_only 동기화 상태(중복 append 방지)
GSHEET_SYNC_STATE_FILE = "gsheet_sync_state.json"

_GSHEET_TRADE_SYNC_EVENT = threading.Event()

# ✅ SCAN은 빈도가 매우 높을 수 있으니, TRADE/EVENT를 우선 처리(요구사항)
_GSHEET_QUEUE_HIGH = deque()  # TRADE/EVENT
_GSHEET_QUEUE_SCAN = deque()  # SCAN
_GSHEET_QUEUE_LOCK = threading.RLock()
_GSHEET_CACHE_LOCK = threading.RLock()
_GSHEET_CACHE: Dict[str, Any] = {
    "ws": None,
    "header_ok": False,
    "last_init_epoch": 0.0,
    "last_err": "",
    "last_tb": "",
    "service_account_email": "",
    "worksheet": "",
    "spreadsheet_id": "",
    "last_append_epoch": 0.0,
    "last_append_kst": "",
    "last_append_type": "",
    "last_append_stage": "",
    "next_append_high_epoch": 0.0,
    "next_append_scan_epoch": 0.0,
    "quota_cooldown_until_epoch": 0.0,
    "last_429_epoch": 0.0,
}

_GSHEET_NOTIFY_LOCK = threading.RLock()
_GSHEET_LAST_NOTIFY_EPOCH = 0.0
_GSHEET_LAST_NOTIFY_MSG = ""

# ✅ SCAN 로그는 빈도가 매우 높아 Google Sheets API rate-limit(429)을 유발할 수 있음
# - UI(monitor_state.json)에는 전체 SCAN 과정을 남기되,
# - 시트에는 stage/심볼별로 일정 간격(throttle) 샘플링해서 누적한다.
_GSHEET_SCAN_THROTTLE_LOCK = threading.RLock()
_GSHEET_SCAN_LAST: Dict[str, float] = {}
_GSHEET_SCAN_THROTTLE_SEC = 20.0
_GSHEET_SCAN_THROTTLE_MAX_KEYS = 1500
_GSHEET_SCAN_ALWAYS_STAGES = {
    "ai_result",
    "trade_opened",
    "ai_error",
    "fetch_short_fail",
    "fetch_long_fail",
    "support_resistance_fail",
}

# ✅ Google Sheets write quota(분당 write 요청 수) 방어:
# - SCAN은 주기적으로 묶어서만 append (요청 수 감소)
# - 429(Quota exceeded) 발생 시 일정 시간 쿨다운 후 재시도
_GSHEET_MIN_APPEND_HIGH_SEC = 1.0
_GSHEET_MIN_APPEND_SCAN_SEC = 6.0
_GSHEET_QUOTA_COOLDOWN_SEC = 65.0


def gsheet_is_enabled() -> bool:
    # secrets 우선 (요구사항)
    return _boolish(_sget_str("GSHEET_ENABLED"))


def gsheet_mode() -> str:
    """
    Google Sheets 기록 모드
    - "trades_only"(기본): 매매일지 + 시간대/일별 총합만 기록
    - "legacy": 기존처럼 TRADE/EVENT/SCAN 원본 로그를 기록
    """
    try:
        m = str(_sget_str("GSHEET_MODE") or "").strip().lower()
        if m in ["legacy", "raw", "logs", "full"]:
            # 기본은 trades_only(사용자 요구). 레거시 모드는 "secrets + 설정" 둘 다 명시 허용 시에만 켠다.
            # - 실수로 SCAN/EVENT가 시트에 쌓여 quota(429) 및 가독성 문제가 생기는 것을 방지.
            legacy_secret_ok = _boolish(_sget_str("GSHEET_LEGACY_LOGS"))
            if not legacy_secret_ok:
                return "trades_only"
            try:
                cfg = load_settings()
                if bool(cfg.get("gsheet_allow_legacy_logs", False)):
                    return "legacy"
            except Exception:
                pass
            return "trades_only"
        return "trades_only"
    except Exception:
        return "trades_only"


def _gsheet_trade_ws_names() -> Dict[str, str]:
    stg = _gsheet_get_settings()
    base = str(stg.get("worksheet", "") or "").strip() or "TRADES"
    return {
        "trade": base,
        "hourly": f"{base}_HOURLY",
        "daily": f"{base}_DAILY",
        "calendar": f"{base}_CALENDAR",
        "reviews": f"{base}_REVIEWS",
    }


def _gsheet_get_settings() -> Dict[str, str]:
    sid = _sget_str("GSHEET_SPREADSHEET_ID") or _sget_str("GSHEET_ID")
    ws_name = _sget_str("GSHEET_WORKSHEET") or "BOT_LOG"
    sa_json = _sget_str("GSHEET_SERVICE_ACCOUNT_JSON") or _sget_str("GOOGLE_SERVICE_ACCOUNT_JSON")
    return {"spreadsheet_id": sid, "worksheet": ws_name, "service_account_json": sa_json}


def _gsheet_service_account_info() -> Optional[Dict[str, Any]]:
    """
    secrets 형태 다양성 흡수:
    - GSHEET_SERVICE_ACCOUNT_JSON: JSON 문자열(요구사항)
    - 혹시 dict로 넣은 경우도 방어적으로 지원
    - (호환) [gcp_service_account] dict
    """
    try:
        raw = _sget("GSHEET_SERVICE_ACCOUNT_JSON", None)
        if raw is None or raw == "":
            raw = _sget("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        if (raw is None or raw == "") and ("gcp_service_account" in st.secrets) and isinstance(st.secrets.get("gcp_service_account"), dict):
            return dict(st.secrets.get("gcp_service_account") or {})
        if isinstance(raw, dict):
            return dict(raw)
        s = str(raw or "").strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception as je:
            # 스트림릿 secrets(TOML)에서 "\n" escape가 실제 개행으로 풀리면 JSON이 깨질 수 있음
            # → 사용자에게 원인을 알려주기 위해 last_err를 남긴다.
            try:
                _GSHEET_CACHE["last_err"] = f"GSHEET_SERVICE_ACCOUNT_JSON 파싱 실패: {je}"
            except Exception:
                pass
            # ✅ Streamlit secrets(TOML) 멀티라인 문자열에서 private_key의 개행이 실제 개행으로 들어오면 JSON이 깨짐
            #    → private_key 값 내부의 개행만 \\n으로 이스케이프해서 1회 복구 시도
            try:
                if s.startswith("{") and '"private_key"' in s and ("\n" in s or "\r" in s):
                    m = re.search(r'"private_key"\s*:\s*"(.*?)"', s, flags=re.S)
                    if m:
                        pk = m.group(1)
                        if "\n" in pk or "\r" in pk:
                            pk_fixed = pk.replace("\r\n", "\n").replace("\n", "\\n")
                            s2 = s[: m.start(1)] + pk_fixed + s[m.end(1) :]
                            v2 = json.loads(s2)
                            if isinstance(v2, dict):
                                try:
                                    _GSHEET_CACHE["last_err"] = ""
                                except Exception:
                                    pass
                                return dict(v2)
            except Exception:
                pass
            try:
                import ast as _ast

                # 혹시 dict가 str()로 변환돼 들어온 경우(단, 안전한 literal_eval만 사용)
                if s.startswith("{") and ("'client_email'" in s or "'private_key'" in s):
                    v = _ast.literal_eval(s)
                    if isinstance(v, dict):
                        return dict(v)
            except Exception:
                pass
            return None
    except Exception:
        return None


def _gsheet_service_account_email() -> str:
    try:
        info = _gsheet_service_account_info() or {}
        return str(info.get("client_email") or "").strip()
    except Exception:
        return ""


def _gsheet_exception_detail(err: BaseException, limit: int = 900) -> str:
    """
    gspread/google 오류는 str(err)가 비어있거나(특히 SpreadsheetNotFound/RetryError),
    APIError의 response 본문에만 정보가 있는 경우가 있어 최대한 detail을 뽑아낸다.
    """
    try:
        name = str(type(err).__name__ or "Exception")
    except Exception:
        name = "Exception"
    try:
        msg = str(err or "").strip()
    except Exception:
        msg = ""
    detail = msg if msg else name

    # gspread.exceptions.APIError / googleapiclient.errors.HttpError 등 response/body가 있는 케이스
    try:
        resp = getattr(err, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            text = getattr(resp, "text", None)
            if code:
                detail = f"{detail} | http={code}"
            if text:
                t = str(text).strip()
                if t:
                    t = t.replace("\n", " ")[:400]
                    detail = f"{detail} | body={t}"
    except Exception:
        pass

    # 일부 에러는 args[0]에 dict로 내려오는 경우가 있음
    try:
        if not msg and getattr(err, "args", None):
            a0 = err.args[0]
            if isinstance(a0, dict):
                s = safe_json_dumps(a0, limit=420).replace("\n", " ")
                if s:
                    detail = f"{name} | {s}"
    except Exception:
        pass

    try:
        if len(detail) > int(limit):
            detail = detail[: int(limit)] + "..."
    except Exception:
        pass
    return detail


def _gsheet_notify_connect_issue(where: str, msg: str, min_interval_sec: float = 300.0) -> None:
    """
    Google Sheets 연결/권한 문제를 관리자 DM으로 안내(과다 스팸 방지).
    """
    try:
        if not TG_ADMIN_IDS:
            return
        now = time.time()
        with _GSHEET_NOTIFY_LOCK:
            global _GSHEET_LAST_NOTIFY_EPOCH, _GSHEET_LAST_NOTIFY_MSG
            if (now - float(_GSHEET_LAST_NOTIFY_EPOCH or 0.0)) < float(min_interval_sec):
                return
            if msg and msg == _GSHEET_LAST_NOTIFY_MSG and (now - float(_GSHEET_LAST_NOTIFY_EPOCH or 0.0)) < float(min_interval_sec) * 2:
                return
            _GSHEET_LAST_NOTIFY_EPOCH = now
            _GSHEET_LAST_NOTIFY_MSG = msg
        stg = _gsheet_get_settings()
        email = _gsheet_service_account_email()
        hint = ""
        if email and stg.get("spreadsheet_id"):
            hint = f"\n- 서비스계정 이메일: {email}\n- 공유: 시트에 위 이메일을 '편집자'로 공유해야 합니다."
        tb_txt = ""
        try:
            tb_txt = str(_GSHEET_CACHE.get("last_tb", "") or "")
        except Exception:
            tb_txt = ""
        notify_admin_error(
            where,
            RuntimeError(msg),
            context={"spreadsheet_id": stg.get("spreadsheet_id", ""), "worksheet": stg.get("worksheet", ""), "service_account_email": email, "code": CODE_VERSION},
            tb=tb_txt,
            min_interval_sec=min_interval_sec,
        )
        if hint:
            tg_send(hint, target="admin")
    except Exception:
        pass


def _gsheet_connect_ws() -> Optional[Any]:
    if not gsheet_is_enabled():
        return None
    if gspread is None or GoogleCredentials is None:
        _GSHEET_CACHE["last_err"] = "gspread/google-auth 미설치(requirements.txt 확인)"
        return None

    stg = _gsheet_get_settings()
    sid = stg.get("spreadsheet_id", "").strip()
    ws_name = stg.get("worksheet", "BOT_LOG").strip() or "BOT_LOG"
    info = _gsheet_service_account_info()
    if not sid:
        _GSHEET_CACHE["last_err"] = "GSHEET_SPREADSHEET_ID 누락"
        return None
    if not info:
        cur = str(_GSHEET_CACHE.get("last_err", "") or "").strip()
        _GSHEET_CACHE["last_err"] = cur or "GSHEET_SERVICE_ACCOUNT_JSON 누락/파싱 실패"
        return None

    try:
        try:
            _GSHEET_CACHE["service_account_email"] = str((info or {}).get("client_email") or "").strip()
        except Exception:
            _GSHEET_CACHE["service_account_email"] = ""
        _GSHEET_CACHE["worksheet"] = ws_name
        _GSHEET_CACHE["spreadsheet_id"] = sid
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sid)
        try:
            ws = sh.worksheet(ws_name)
        except Exception:
            # 없으면 생성
            ws = sh.add_worksheet(title=ws_name, rows=5000, cols=len(GSHEET_HEADER) + 5)
        return ws
    except Exception as e:
        detail = _gsheet_exception_detail(e, limit=900)
        low = detail.lower()
        # 가장 흔한 케이스: 시트 미공유/ID 오타 → SpreadsheetNotFound(메시지 비어있을 수 있음)
        if "spreadsheetnotfound" in low:
            detail = "SpreadsheetNotFound (시트를 서비스계정 이메일에 '편집자'로 공유 + GSHEET_SPREADSHEET_ID 확인)"
        elif "permission" in low or "forbidden" in low:
            detail = f"권한 문제(Forbidden): 시트 공유/드라이브 권한/스코프 확인 | {detail}".strip()
        elif ("api" in low and "enable" in low) or "has not been used" in low:
            detail = f"API 활성화 필요: Google Sheets/Drive API | {detail}".strip()
        _GSHEET_CACHE["last_err"] = f"GSHEET 연결 실패: {detail}".strip()
        try:
            _GSHEET_CACHE["last_tb"] = traceback.format_exc()
        except Exception:
            _GSHEET_CACHE["last_tb"] = ""
        return None


def _gsheet_connect_spreadsheet() -> Optional[Any]:
    """
    trades_only 모드에서 여러 워크시트를 다루기 위해 Spreadsheet 객체를 연결한다.
    - 실패해도 봇이 죽지 않게 last_err에 남기고 None 반환.
    """
    if not gsheet_is_enabled():
        return None
    if gspread is None or GoogleCredentials is None:
        try:
            _GSHEET_CACHE["last_err"] = "gspread/google-auth 미설치(requirements.txt 확인)"
        except Exception:
            pass
        return None

    stg = _gsheet_get_settings()
    sid = str(stg.get("spreadsheet_id", "") or "").strip()
    info = _gsheet_service_account_info()
    if not sid:
        try:
            _GSHEET_CACHE["last_err"] = "GSHEET_SPREADSHEET_ID 누락"
        except Exception:
            pass
        return None
    if not info:
        cur = str(_GSHEET_CACHE.get("last_err", "") or "").strip()
        try:
            _GSHEET_CACHE["last_err"] = cur or "GSHEET_SERVICE_ACCOUNT_JSON 누락/파싱 실패"
        except Exception:
            pass
        return None

    try:
        try:
            _GSHEET_CACHE["service_account_email"] = str((info or {}).get("client_email") or "").strip()
        except Exception:
            _GSHEET_CACHE["service_account_email"] = ""
        _GSHEET_CACHE["spreadsheet_id"] = sid
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sid)
        return sh
    except Exception as e:
        detail = _gsheet_exception_detail(e, limit=900)
        low = detail.lower()
        if "spreadsheetnotfound" in low:
            detail = "SpreadsheetNotFound (시트를 서비스계정 이메일에 '편집자'로 공유 + GSHEET_SPREADSHEET_ID 확인)"
        elif "permission" in low or "forbidden" in low:
            detail = f"권한 문제(Forbidden): 시트 공유/드라이브 권한/스코프 확인 | {detail}".strip()
        elif ("api" in low and "enable" in low) or "has not been used" in low:
            detail = f"API 활성화 필요: Google Sheets/Drive API | {detail}".strip()
        try:
            _GSHEET_CACHE["last_err"] = f"GSHEET 연결 실패: {detail}".strip()
            _GSHEET_CACHE["last_tb"] = traceback.format_exc()
        except Exception:
            pass
        return None


def _gsheet_sync_state_default() -> Dict[str, Any]:
    return {
        "synced_trade_ids": [],
        "synced_review_ids": [],
        # ✅ 구글시트 초기화(Reset) 기준 시각: 이 시각 이후의 trade_log.csv만 시트에 반영
        "reset_epoch": 0.0,
        "reset_kst": "",
        "last_trade_sync_epoch": 0.0,
        "last_trade_sync_kst": "",
        "last_summary_sync_epoch": 0.0,
        "last_summary_sync_kst": "",
        "last_calendar_sync_epoch": 0.0,
        "last_calendar_sync_kst": "",
        "last_review_sync_epoch": 0.0,
        "last_review_sync_kst": "",
        "trade_ws_title": "",
        "hourly_ws_title": "",
        "daily_ws_title": "",
        "calendar_ws_title": "",
        "reviews_ws_title": "",
        # ✅ 서식(표) 자동 적용 상태(중복 batchUpdate 방지)
        "format_version_applied": 0,
        "format_applied_epoch": 0.0,
        "format_applied_kst": "",
        "format_trade_title": "",
        "format_hourly_title": "",
        "format_daily_title": "",
        "format_calendar_title": "",
        "format_reviews_title": "",
    }


def _gsheet_sync_state_load() -> Dict[str, Any]:
    st0 = read_json_safe(GSHEET_SYNC_STATE_FILE, None)
    if not isinstance(st0, dict):
        st0 = _gsheet_sync_state_default()
    base = _gsheet_sync_state_default()
    for k, v in base.items():
        if k not in st0:
            st0[k] = v
    # 타입 보정
    try:
        if not isinstance(st0.get("synced_trade_ids", []), list):
            st0["synced_trade_ids"] = []
    except Exception:
        st0["synced_trade_ids"] = []
    try:
        if not isinstance(st0.get("synced_review_ids", []), list):
            st0["synced_review_ids"] = []
    except Exception:
        st0["synced_review_ids"] = []
    return st0


def _gsheet_sync_state_save(st0: Dict[str, Any]) -> None:
    try:
        write_json_atomic(GSHEET_SYNC_STATE_FILE, st0)
    except Exception:
        pass


def _gsheet_row_looks_like_legacy_header(row: List[str]) -> bool:
    try:
        if not row:
            return False
        r0 = str(row[0] or "").strip().lower()
        if r0 != "time_kst":
            return False
        r1 = str(row[1] or "").strip().lower() if len(row) >= 2 else ""
        r2 = str(row[2] or "").strip().lower() if len(row) >= 3 else ""
        return (r1 == "type") and (r2 == "stage")
    except Exception:
        return False


def _gsheet_row_looks_like_trade_header(row: List[str]) -> bool:
    try:
        if not row:
            return False
        # 정확히 일치하면 좋지만, 일부는 BOM/공백이 섞일 수 있어 trim 비교
        # - 한글 헤더(현재) 또는 과거 영문 헤더 둘 다 허용
        a_ko = [str(x or "").strip() for x in row[: len(GSHEET_TRADE_JOURNAL_HEADER)]]
        b_ko = [str(x or "").strip() for x in GSHEET_TRADE_JOURNAL_HEADER]
        if a_ko == b_ko:
            return True
        a_en = [str(x or "").strip() for x in row[: len(GSHEET_TRADE_JOURNAL_HEADER_EN)]]
        b_en = [str(x or "").strip() for x in GSHEET_TRADE_JOURNAL_HEADER_EN]
        return a_en == b_en
    except Exception:
        return False


def _gsheet_get_or_create_worksheet(sh: Any, title: str, rows: int, cols: int) -> Any:
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=max(200, int(rows)), cols=max(12, int(cols)))


def _gsheet_prepare_trades_only_sheets(sh: Any) -> Optional[Dict[str, Any]]:
    """
    trades_only 모드용 시트 준비:
    - <base> (매매일지)
    - <base>_HOURLY
    - <base>_DAILY
    - 기존 <base>가 레거시 로그 형식이면 <base>_RAW 로 rename 후 새로 생성
    """
    names = _gsheet_trade_ws_names()
    base = names["trade"]
    hourly = names["hourly"]
    daily = names["daily"]
    calendar_ws = names["calendar"]
    reviews_ws = names.get("reviews", f"{base}_REVIEWS")

    try:
        ws_trade = None
        try:
            ws_trade = sh.worksheet(base)
        except Exception:
            ws_trade = None

        if ws_trade is not None:
            first = []
            try:
                first = ws_trade.row_values(1) or []
            except Exception:
                first = []
            # 레거시 로그 시트면 rename 후, base 이름으로 새 매매일지 시트 생성
            if _gsheet_row_looks_like_legacy_header(first):
                new_title = f"{base}_RAW"
                # 충돌 방지
                try:
                    sh.worksheet(new_title)
                    # 이미 있으면 숫자 suffix
                    for i in range(2, 30):
                        cand = f"{new_title}_{i}"
                        try:
                            sh.worksheet(cand)
                            continue
                        except Exception:
                            new_title = cand
                            break
                except Exception:
                    pass
                try:
                    ws_trade.update_title(new_title)
                    # 사용자에게 DM으로 안내(스팸 방지)
                    try:
                        tg_send(
                            f"📎 Google Sheets: 기존 '{base}' 시트가 SCAN/원본 로그 형식이라 '{new_title}'로 보관하고,\n"
                            f"새 매매일지 시트 '{base}'를 생성합니다.",
                            target="admin",
                        )
                    except Exception:
                        pass
                    ws_trade = None
                except Exception as e:
                    # rename 실패 시에도 동작은 계속: 새 시트를 다른 이름으로 만들고 안내
                    notify_admin_error("GSHEET_MIGRATE", e, context={"from": base, "to": new_title}, min_interval_sec=300.0)
                    ws_trade = None
                    base = f"{names['trade']}_TRADE_LOG"

        if ws_trade is None:
            ws_trade = _gsheet_get_or_create_worksheet(sh, base, rows=5000, cols=len(GSHEET_TRADE_JOURNAL_HEADER) + 2)
        else:
            # 기존 시트가 예전(컬럼 수가 적음)일 수 있어, 한글 헤더/추가 컬럼을 위해 cols 확장
            try:
                need_cols = int(len(GSHEET_TRADE_JOURNAL_HEADER) + 2)
                cur_cols = int(getattr(ws_trade, "col_count", 0) or 0)
                if cur_cols and cur_cols < need_cols:
                    ws_trade.resize(cols=need_cols)
            except Exception:
                pass

        # 헤더 확인/생성
        try:
            first2 = ws_trade.row_values(1) or []
        except Exception:
            first2 = []
        if not first2:
            try:
                ws_trade.append_row(GSHEET_TRADE_JOURNAL_HEADER, value_input_option="USER_ENTERED")
            except Exception:
                pass
        else:
            if not _gsheet_row_looks_like_trade_header(first2):
                # 헤더가 다른데 데이터가 이미 있으면 건드리지 않고 새 시트로 우회
                try:
                    vals = ws_trade.get_all_values() or []
                except Exception:
                    vals = []
                if len(vals) > 1:
                    alt = f"{base}_TRADE_LOG"
                    ws_trade = _gsheet_get_or_create_worksheet(sh, alt, rows=5000, cols=len(GSHEET_TRADE_JOURNAL_HEADER) + 2)
                    try:
                        first3 = ws_trade.row_values(1) or []
                    except Exception:
                        first3 = []
                    if not first3:
                        try:
                            ws_trade.append_row(GSHEET_TRADE_JOURNAL_HEADER, value_input_option="USER_ENTERED")
                        except Exception:
                            pass
                    try:
                        tg_send(f"📎 Google Sheets: 매매일지 시트가 '{alt}'로 생성되었습니다(기존 헤더 충돌).", target="admin")
                    except Exception:
                        pass
                else:
                    # 데이터가 거의 없으면 헤더만 교체(삭제는 하지 않음)
                    try:
                        ws_trade.update("A1", [GSHEET_TRADE_JOURNAL_HEADER])
                    except Exception:
                        pass
            else:
                # 과거 영문 헤더인 경우, 데이터는 유지하고 헤더만 한글로 교체(직관성 개선)
                try:
                    a1 = [str(x or "").strip() for x in first2[: len(GSHEET_TRADE_JOURNAL_HEADER_EN)]]
                    b1 = [str(x or "").strip() for x in GSHEET_TRADE_JOURNAL_HEADER_EN]
                    if a1 == b1:
                        ws_trade.update("A1", [GSHEET_TRADE_JOURNAL_HEADER])
                except Exception:
                    pass

        ws_hourly = _gsheet_get_or_create_worksheet(sh, hourly, rows=2000, cols=len(GSHEET_HOURLY_SUMMARY_HEADER) + 2)
        ws_daily = _gsheet_get_or_create_worksheet(sh, daily, rows=2000, cols=len(GSHEET_DAILY_SUMMARY_HEADER) + 2)
        # ✅ 달력형 일별 요약(요구사항)
        ws_calendar = _gsheet_get_or_create_worksheet(sh, calendar_ws, rows=140, cols=10)
        # ✅ 회고 모음(손실 트레이드만)
        ws_reviews = _gsheet_get_or_create_worksheet(sh, reviews_ws, rows=4000, cols=len(GSHEET_REVIEWS_HEADER) + 2)
        try:
            first_r = ws_reviews.row_values(1) or []
        except Exception:
            first_r = []
        if not first_r:
            try:
                ws_reviews.append_row(GSHEET_REVIEWS_HEADER, value_input_option="USER_ENTERED")
            except Exception:
                pass
        return {
            "ws_trade": ws_trade,
            "ws_hourly": ws_hourly,
            "ws_daily": ws_daily,
            "ws_calendar": ws_calendar,
            "ws_reviews": ws_reviews,
            "trade_title": base,
            "hourly_title": hourly,
            "daily_title": daily,
            "calendar_title": calendar_ws,
            "reviews_title": reviews_ws,
        }
    except Exception as e:
        # ✅ Google Sheets 503 등은 "일시 장애"일 가능성이 높아, 과도한 DM 스팸을 피한다.
        # - 오류 상세는 _GSHEET_CACHE에 남기고, 워커(GSHEET_SYNC)에서 throttle된 알림을 처리한다.
        try:
            detail = _gsheet_exception_detail(e, limit=900)
        except Exception:
            detail = str(e)[:900]
        try:
            with _GSHEET_CACHE_LOCK:
                _GSHEET_CACHE["last_err"] = f"GSHEET_PREPARE 실패: {detail}"
                _GSHEET_CACHE["last_tb"] = traceback.format_exc()
        except Exception:
            pass
        # 503(Service Unavailable)이면 잠깐 쉬었다가 재시도(무한 반복 방지)
        try:
            code = None
            resp = getattr(e, "response", None)
            if resp is not None:
                code = getattr(resp, "status_code", None)
            if code is None:
                m = str(e or "").lower()
                if " 503" in m or "[503]" in m or "service is currently unavailable" in m:
                    code = 503
            if int(code or 0) == 503:
                with _GSHEET_CACHE_LOCK:
                    _GSHEET_CACHE["service_unavailable_until_epoch"] = time.time() + 60 * 3
                    _GSHEET_CACHE["service_unavailable_kst"] = now_kst_str()
        except Exception:
            pass
        return None


# =========================================================
# ✅ 7.5.1) Google Sheets: 표(서식) 자동 적용 (trades_only 전용)
# - batchUpdate 1회로 3개 시트(매매일지/시간대/일별)에 서식을 적용
# - gspread-formatting 없이도 동작(추가 설치 불필요)
# - 레이트리밋 방지: sync state에 "버전+시트명"을 저장해 1회만 적용
# =========================================================

GSHEET_FORMAT_VERSION = 4


def _gsheet_auto_format_enabled() -> bool:
    """
    우선순위:
    1) secrets: GSHEET_AUTO_FORMAT (true/false)
    2) settings: gsheet_auto_format_enable (기본 True)
    """
    try:
        v = str(_sget_str("GSHEET_AUTO_FORMAT") or "").strip()
        if v:
            return bool(_boolish(v))
    except Exception:
        pass
    try:
        cfg = load_settings()
        return bool(cfg.get("gsheet_auto_format_enable", True))
    except Exception:
        return True


def _gsheet_format_is_already_applied(
    st0: Dict[str, Any],
    trade_title: str,
    hourly_title: str,
    daily_title: str,
    calendar_title: str = "",
    reviews_title: str = "",
) -> bool:
    try:
        ver = int(st0.get("format_version_applied", 0) or 0)
        if ver != int(GSHEET_FORMAT_VERSION):
            return False
        if str(st0.get("format_trade_title", "") or "") != str(trade_title or ""):
            return False
        if str(st0.get("format_hourly_title", "") or "") != str(hourly_title or ""):
            return False
        if str(st0.get("format_daily_title", "") or "") != str(daily_title or ""):
            return False
        if str(st0.get("format_calendar_title", "") or "") != str(calendar_title or ""):
            return False
        if str(st0.get("format_reviews_title", "") or "") != str(reviews_title or ""):
            return False
        return True
    except Exception:
        return False


def _gsheet_fetch_metadata_safe(sh: Any) -> Dict[str, Any]:
    try:
        # includeGridData=false: formatting/metadata만 필요(응답 크기↓)
        return sh.fetch_sheet_metadata(params={"includeGridData": "false"}) or {}
    except Exception:
        try:
            return sh.fetch_sheet_metadata() or {}
        except Exception:
            return {}


def _gsheet_batch_update_safe(sh: Any, body: Dict[str, Any]) -> Any:
    """
    gspread 버전 차이/래핑 차이 대응:
    - Spreadsheet.batch_update 가 있으면 사용
    - 없으면 low-level client.request로 fallback
    """
    try:
        if hasattr(sh, "batch_update"):
            return sh.batch_update(body)
    except Exception:
        pass
    # fallback: direct REST call via gspread client
    try:
        sid = str(getattr(sh, "id", "") or getattr(sh, "spreadsheet_id", "") or "").strip()
        if not sid:
            # gspread Spreadsheet는 보통 .id가 존재
            sid = str(getattr(getattr(sh, "client", None), "spreadsheet_id", "") or "").strip()
        if not sid:
            raise RuntimeError("spreadsheet_id_not_found")
        client = getattr(sh, "client", None)
        if client is None or not hasattr(client, "request"):
            raise RuntimeError("gspread_client_request_missing")
        return client.request("post", f"spreadsheets/{sid}:batchUpdate", json=body)
    except Exception:
        # 호출부에서 예외를 잡아 관리자 알림 처리
        raise


def _gsheet_meta_by_sheet_id(md: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    try:
        for s in (md.get("sheets") or []):
            try:
                props = (s or {}).get("properties") or {}
                sid = int(props.get("sheetId", -1))
                if sid >= 0:
                    out[sid] = dict(s or {})
            except Exception:
                continue
    except Exception:
        return {}
    return out


def _gsheet_build_cleanup_requests(sheet_id: int, sheet_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []
    try:
        # basic filter 제거(중복 방지)
        if (sheet_meta or {}).get("basicFilter"):
            reqs.append({"clearBasicFilter": {"sheetId": int(sheet_id)}})
    except Exception:
        pass
    # conditional formats 제거(중복 방지)
    try:
        cfs = (sheet_meta or {}).get("conditionalFormats") or []
        if isinstance(cfs, list) and cfs:
            for idx in range(len(cfs) - 1, -1, -1):
                reqs.append({"deleteConditionalFormatRule": {"sheetId": int(sheet_id), "index": int(idx)}})
    except Exception:
        pass
    # banding 제거(중복 방지)
    try:
        brs = (sheet_meta or {}).get("bandedRanges") or []
        if isinstance(brs, list) and brs:
            for br in brs:
                bid = (br or {}).get("bandedRangeId")
                if bid is None:
                    continue
                try:
                    reqs.append({"deleteBanding": {"bandedRangeId": int(bid)}})
                except Exception:
                    continue
    except Exception:
        pass
    return reqs


def _gsheet_color(hex_rgb: str) -> Dict[str, float]:
    h = str(hex_rgb or "").strip().lstrip("#")
    if len(h) != 6:
        return {"red": 1.0, "green": 1.0, "blue": 1.0}
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return {"red": float(r), "green": float(g), "blue": float(b)}
    except Exception:
        return {"red": 1.0, "green": 1.0, "blue": 1.0}


def _gsheet_build_table_requests(
    sheet_id: int,
    row_count: int,
    col_count: int,
    *,
    header_bg: str = "#1f2937",
    header_fg: str = "#ffffff",
    band1: str = "#f8fafc",
    band2: str = "#ffffff",
    default_col_width_px: int = 140,
    col_width_px: Optional[Dict[int, int]] = None,
    wrap_cols: Optional[List[int]] = None,
    number_formats: Optional[List[Tuple[int, int, str]]] = None,  # [(start_col, end_col_excl, pattern)]
    right_align_cols: Optional[List[Tuple[int, int]]] = None,      # [(start_col, end_col_excl)]
    cond_formats: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    sheetId 기준으로 "표 형태" 서식 요청을 생성한다.
    - cond_formats: addConditionalFormatRule 용 rule dict 리스트
    """
    sid = int(sheet_id)
    rc = max(2, int(row_count))
    cc = max(1, int(col_count))
    reqs: List[Dict[str, Any]] = []

    # 1) Freeze header row
    reqs.append(
        {
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }
        }
    )

    # 2) Header row style
    reqs.append(
        {
            "repeatCell": {
                "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": cc},
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _gsheet_color(header_bg),
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                        "wrapStrategy": "WRAP",
                        "textFormat": {"foregroundColor": _gsheet_color(header_fg), "bold": True},
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }
        }
    )

    # 3) Default column width
    reqs.append(
        {
            "updateDimensionProperties": {
                "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 0, "endIndex": cc},
                "properties": {"pixelSize": int(default_col_width_px)},
                "fields": "pixelSize",
            }
        }
    )

    # 4) Specific column widths
    try:
        if col_width_px:
            for c, w in sorted(col_width_px.items(), key=lambda kv: kv[0]):
                c0 = int(c)
                if c0 < 0 or c0 >= cc:
                    continue
                reqs.append(
                    {
                        "updateDimensionProperties": {
                            "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": c0, "endIndex": c0 + 1},
                            "properties": {"pixelSize": int(w)},
                            "fields": "pixelSize",
                        }
                    }
                )
    except Exception:
        pass

    # 5) Wrap long-text columns (data rows only)
    try:
        if wrap_cols:
            for c in wrap_cols:
                c0 = int(c)
                if c0 < 0 or c0 >= cc:
                    continue
                reqs.append(
                    {
                        "repeatCell": {
                            "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": rc, "startColumnIndex": c0, "endColumnIndex": c0 + 1},
                            "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
                            "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)",
                        }
                    }
                )
    except Exception:
        pass

    # 6) Number formats
    try:
        if number_formats:
            for start_c, end_c, pattern in number_formats:
                sc = int(start_c)
                ec = int(end_c)
                if sc < 0:
                    sc = 0
                if ec > cc:
                    ec = cc
                if ec <= sc:
                    continue
                reqs.append(
                    {
                        "repeatCell": {
                            "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": rc, "startColumnIndex": sc, "endColumnIndex": ec},
                            "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": str(pattern)}}},
                            "fields": "userEnteredFormat.numberFormat",
                        }
                    }
                )
    except Exception:
        pass

    # 7) Right-align numeric columns (optional)
    try:
        if right_align_cols:
            for sc0, ec0 in right_align_cols:
                sc = int(sc0)
                ec = int(ec0)
                if sc < 0:
                    sc = 0
                if ec > cc:
                    ec = cc
                if ec <= sc:
                    continue
                reqs.append(
                    {
                        "repeatCell": {
                            "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": rc, "startColumnIndex": sc, "endColumnIndex": ec},
                            "cell": {"userEnteredFormat": {"horizontalAlignment": "RIGHT"}},
                            "fields": "userEnteredFormat.horizontalAlignment",
                        }
                    }
                )
    except Exception:
        pass

    # 8) Add banding (header 포함)
    try:
        reqs.append(
            {
                "addBanding": {
                    "bandedRange": {
                        "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": rc, "startColumnIndex": 0, "endColumnIndex": cc},
                        "rowProperties": {
                            "headerColor": _gsheet_color(header_bg),
                            "firstBandColor": _gsheet_color(band1),
                            "secondBandColor": _gsheet_color(band2),
                        },
                    }
                }
            }
        )
    except Exception:
        pass

    # 9) Basic filter
    try:
        reqs.append({"setBasicFilter": {"filter": {"range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": rc, "startColumnIndex": 0, "endColumnIndex": cc}}}})
    except Exception:
        pass

    # 10) Conditional formats
    try:
        if cond_formats:
            idx = 0
            for rule in cond_formats:
                if not isinstance(rule, dict):
                    continue
                reqs.append({"addConditionalFormatRule": {"rule": rule, "index": int(idx)}})
                idx += 1
    except Exception:
        pass

    return reqs


def _gsheet_build_pnl_cond_formats(sheet_id: int, row_count: int, start_col: int, end_col: int) -> List[Dict[str, Any]]:
    sid = int(sheet_id)
    rc = max(2, int(row_count))
    sc = int(start_col)
    ec = int(end_col)
    # green for >0, red for <0
    rng = {"sheetId": sid, "startRowIndex": 1, "endRowIndex": rc, "startColumnIndex": sc, "endColumnIndex": ec}
    green = {
        "ranges": [rng],
        "booleanRule": {
            "condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0"}]},
            "format": {"backgroundColor": _gsheet_color("#e6f4ea"), "textFormat": {"foregroundColor": _gsheet_color("#137333"), "bold": True}},
        },
    }
    red = {
        "ranges": [rng],
        "booleanRule": {
            "condition": {"type": "NUMBER_LESS", "values": [{"userEnteredValue": "0"}]},
            "format": {"backgroundColor": _gsheet_color("#fce8e6"), "textFormat": {"foregroundColor": _gsheet_color("#a50e0e"), "bold": True}},
        },
    }
    return [green, red]


def _gsheet_apply_trades_only_format_internal(
    sh: Any,
    sheets: Dict[str, Any],
    st0: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    try:
        trade_title = str(sheets.get("trade_title", "") or "")
        hourly_title = str(sheets.get("hourly_title", "") or "")
        daily_title = str(sheets.get("daily_title", "") or "")
        calendar_title = str(sheets.get("calendar_title", "") or "")
        reviews_title = str(sheets.get("reviews_title", "") or "")
        ws_trade = sheets.get("ws_trade")
        ws_hourly = sheets.get("ws_hourly")
        ws_daily = sheets.get("ws_daily")
        ws_calendar = sheets.get("ws_calendar")
        ws_reviews = sheets.get("ws_reviews")
        if ws_trade is None or ws_hourly is None or ws_daily is None or ws_calendar is None or ws_reviews is None:
            return {"ok": False, "error": "missing_worksheets"}

        if not _gsheet_auto_format_enabled():
            return {"ok": True, "skipped": True, "reason": "auto_format_disabled"}

        if (not force) and _gsheet_format_is_already_applied(st0, trade_title, hourly_title, daily_title, calendar_title, reviews_title):
            return {"ok": True, "skipped": True, "reason": "already_applied"}

        md = _gsheet_fetch_metadata_safe(sh)
        by_id = _gsheet_meta_by_sheet_id(md)

        reqs: List[Dict[str, Any]] = []

        # ---- Trade journal ----
        try:
            sid = int(getattr(ws_trade, "id", -1))
            rc = int(getattr(ws_trade, "row_count", 5000) or 5000)
            cc = int(getattr(ws_trade, "col_count", len(GSHEET_TRADE_JOURNAL_HEADER)) or len(GSHEET_TRADE_JOURNAL_HEADER))
            cc = max(cc, len(GSHEET_TRADE_JOURNAL_HEADER))
            sm = by_id.get(sid, {})
            reqs += _gsheet_build_cleanup_requests(sid, sm)
            reqs += _gsheet_build_table_requests(
                sid,
                rc,
                cc,
                default_col_width_px=140,
                col_width_px={
                    0: 70,    # 상태
                    1: 165,   # 시간(KST)
                    2: 120,   # 코인
                    3: 80,    # 방향
                    4: 110,   # 진입가
                    5: 110,   # 청산가
                    6: 120,   # 손익(USDT)
                    7: 95,    # 수익률(%)
                    8: 150,   # 진입전 총자산
                    9: 150,   # 청산후 총자산
                    10: 150,  # 진입전 가용
                    11: 150,  # 청산후 가용
                    12: 240,  # 사유
                    13: 240,  # 한줄평
                    14: 420,  # 후기
                    15: 160,  # 일지ID
                },
                wrap_cols=[12, 13, 14],
                number_formats=[
                    (4, 6, "0.########"),  # 진입가/청산가
                    (6, 7, "0.00"),        # 손익(USDT)
                    (7, 8, "0.00"),        # 수익률(%)
                    (8, 12, "0.00"),       # 잔고/가용
                ],
                right_align_cols=[(4, 12)],
                cond_formats=_gsheet_build_pnl_cond_formats(sid, rc, 6, 8),
            )
        except Exception:
            pass

        # ---- Hourly summary ----
        try:
            sid = int(getattr(ws_hourly, "id", -1))
            rc = int(getattr(ws_hourly, "row_count", 2000) or 2000)
            cc = int(getattr(ws_hourly, "col_count", len(GSHEET_HOURLY_SUMMARY_HEADER)) or len(GSHEET_HOURLY_SUMMARY_HEADER))
            cc = max(cc, len(GSHEET_HOURLY_SUMMARY_HEADER))
            sm = by_id.get(sid, {})
            reqs += _gsheet_build_cleanup_requests(sid, sm)
            reqs += _gsheet_build_table_requests(
                sid,
                rc,
                cc,
                default_col_width_px=150,
                col_width_px={0: 185, 6: 185},
                number_formats=[
                    (1, 2, "0"),     # Trades
                    (2, 3, "0.0"),   # WinRate
                    (3, 4, "0.00"),  # TotalPnL
                    (4, 5, "0.00"),  # AvgPnL
                    (5, 6, "0.00"),  # ProfitFactor
                ],
                right_align_cols=[(1, 6)],
                cond_formats=_gsheet_build_pnl_cond_formats(sid, rc, 3, 4),
            )
        except Exception:
            pass

        # ---- Daily summary ----
        try:
            sid = int(getattr(ws_daily, "id", -1))
            rc = int(getattr(ws_daily, "row_count", 2000) or 2000)
            cc = int(getattr(ws_daily, "col_count", len(GSHEET_DAILY_SUMMARY_HEADER)) or len(GSHEET_DAILY_SUMMARY_HEADER))
            cc = max(cc, len(GSHEET_DAILY_SUMMARY_HEADER))
            sm = by_id.get(sid, {})
            reqs += _gsheet_build_cleanup_requests(sid, sm)
            reqs += _gsheet_build_table_requests(
                sid,
                rc,
                cc,
                default_col_width_px=155,
                col_width_px={0: 150, 7: 185},
                number_formats=[
                    (1, 2, "0"),     # Trades
                    (2, 3, "0.0"),   # WinRate
                    (3, 4, "0.00"),  # TotalPnL
                    (4, 5, "0.00"),  # AvgPnL
                    (5, 6, "0.00"),  # MaxDD
                    (6, 7, "0.00"),  # ProfitFactor
                ],
                right_align_cols=[(1, 7)],
                cond_formats=_gsheet_build_pnl_cond_formats(sid, rc, 3, 4),
            )
        except Exception:
            pass

        # ---- Reviews (loss-only) ----
        try:
            sid = int(getattr(ws_reviews, "id", -1))
            rc = int(getattr(ws_reviews, "row_count", 4000) or 4000)
            cc = int(getattr(ws_reviews, "col_count", len(GSHEET_REVIEWS_HEADER)) or len(GSHEET_REVIEWS_HEADER))
            cc = max(cc, len(GSHEET_REVIEWS_HEADER))
            sm = by_id.get(sid, {})
            reqs += _gsheet_build_cleanup_requests(sid, sm)
            reqs += _gsheet_build_table_requests(
                sid,
                rc,
                cc,
                default_col_width_px=150,
                col_width_px={
                    0: 165,  # 시간
                    1: 120,  # 코인
                    2: 80,   # 방향
                    3: 95,   # 수익률
                    4: 120,  # 손익
                    5: 150,  # 진입전 총자산
                    6: 150,  # 청산후 총자산
                    7: 240,  # 사유
                    8: 240,  # 한줄평
                    9: 520,  # 후기
                    10: 160, # 일지ID
                },
                wrap_cols=[7, 8, 9],
                number_formats=[
                    (3, 5, "0.00"),  # 수익률/손익
                    (5, 7, "0.00"),  # 잔고
                ],
                right_align_cols=[(3, 7)],
                cond_formats=_gsheet_build_pnl_cond_formats(sid, rc, 3, 5),
            )
        except Exception:
            pass

        # ---- Calendar ----
        try:
            sid = int(getattr(ws_calendar, "id", -1))
            rc = int(getattr(ws_calendar, "row_count", 140) or 140)
            cc = int(getattr(ws_calendar, "col_count", 8) or 8)
            cc = max(cc, 8)
            sm = by_id.get(sid, {})
            reqs += _gsheet_build_cleanup_requests(sid, sm)

            # Freeze top 2 rows + left 1 column
            reqs.append(
                {
                    "updateSheetProperties": {
                        "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 2, "frozenColumnCount": 1}},
                        "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
                    }
                }
            )

            # Column widths: A(label)=140, B~H=120
            col_widths = {0: 140, 1: 120, 2: 120, 3: 120, 4: 120, 5: 120, 6: 120, 7: 120}
            for c0, w in col_widths.items():
                reqs.append(
                    {
                        "updateDimensionProperties": {
                            "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": int(c0), "endIndex": int(c0) + 1},
                            "properties": {"pixelSize": int(w)},
                            "fields": "pixelSize",
                        }
                    }
                )

            # Header row 1 (A1:H1)
            reqs.append(
                {
                    "repeatCell": {
                        "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": 8},
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": _gsheet_color("#202124"),
                                "horizontalAlignment": "LEFT",
                                "textFormat": {"foregroundColor": _gsheet_color("#ffffff"), "bold": True, "fontSize": 12},
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,textFormat)",
                    }
                }
            )

            # Header row 2 (A2:H2) day names
            reqs.append(
                {
                    "repeatCell": {
                        "range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": 2, "startColumnIndex": 0, "endColumnIndex": 8},
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": _gsheet_color("#303134"),
                                "horizontalAlignment": "CENTER",
                                "textFormat": {"foregroundColor": _gsheet_color("#ffffff"), "bold": True},
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,textFormat)",
                    }
                }
            )

            # Label column A (body rows only; A3:A22 정도면 충분)
            reqs.append(
                {
                    "repeatCell": {
                        "range": {"sheetId": sid, "startRowIndex": 2, "endRowIndex": 20, "startColumnIndex": 0, "endColumnIndex": 1},
                        "cell": {"userEnteredFormat": {"backgroundColor": _gsheet_color("#f1f3f4"), "textFormat": {"bold": True}}},
                        "fields": "userEnteredFormat(backgroundColor,textFormat)",
                    }
                }
            )

            # Center align calendar grid (B3:H20)
            reqs.append(
                {
                    "repeatCell": {
                        "range": {"sheetId": sid, "startRowIndex": 2, "endRowIndex": 20, "startColumnIndex": 1, "endColumnIndex": 8},
                        "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
                        "fields": "userEnteredFormat(horizontalAlignment,verticalAlignment)",
                    }
                }
            )

            # Number formats for PnL rows / Trades rows
            pnl_rows = [3, 6, 9, 12, 15, 18]  # 0-based row indices for "손익(USDT)" rows
            trade_rows = [4, 7, 10, 13, 16, 19]  # "거래수" rows
            for r0 in pnl_rows:
                reqs.append(
                    {
                        "repeatCell": {
                            "range": {"sheetId": sid, "startRowIndex": int(r0), "endRowIndex": int(r0) + 1, "startColumnIndex": 1, "endColumnIndex": 8},
                            "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
                            "fields": "userEnteredFormat.numberFormat",
                        }
                    }
                )
            for r0 in trade_rows:
                reqs.append(
                    {
                        "repeatCell": {
                            "range": {"sheetId": sid, "startRowIndex": int(r0), "endRowIndex": int(r0) + 1, "startColumnIndex": 1, "endColumnIndex": 8},
                            "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "0"}}},
                            "fields": "userEnteredFormat.numberFormat",
                        }
                    }
                )

            # Conditional formats (PnL rows only)
            rngs = [
                {"sheetId": sid, "startRowIndex": int(r0), "endRowIndex": int(r0) + 1, "startColumnIndex": 1, "endColumnIndex": 8}
                for r0 in pnl_rows
            ]
            green_rule = {
                "ranges": rngs,
                "booleanRule": {
                    "condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0"}]},
                    "format": {"backgroundColor": _gsheet_color("#e6f4ea"), "textFormat": {"foregroundColor": _gsheet_color("#137333"), "bold": True}},
                },
            }
            red_rule = {
                "ranges": rngs,
                "booleanRule": {
                    "condition": {"type": "NUMBER_LESS", "values": [{"userEnteredValue": "0"}]},
                    "format": {"backgroundColor": _gsheet_color("#fce8e6"), "textFormat": {"foregroundColor": _gsheet_color("#a50e0e"), "bold": True}},
                },
            }
            reqs.append({"addConditionalFormatRule": {"rule": green_rule, "index": 0}})
            reqs.append({"addConditionalFormatRule": {"rule": red_rule, "index": 1}})
        except Exception:
            pass

        if not reqs:
            return {"ok": True, "skipped": True, "reason": "no_requests"}

        # 1회 batchUpdate로 적용
        _gsheet_batch_update_safe(sh, {"requests": reqs})

        # 상태 업데이트(중복 적용 방지)
        try:
            st0["format_version_applied"] = int(GSHEET_FORMAT_VERSION)
            st0["format_applied_epoch"] = time.time()
            st0["format_applied_kst"] = now_kst_str()
            st0["format_trade_title"] = trade_title
            st0["format_hourly_title"] = hourly_title
            st0["format_daily_title"] = daily_title
            st0["format_calendar_title"] = calendar_title
            st0["format_reviews_title"] = reviews_title
        except Exception:
            pass
        return {"ok": True, "applied": True, "requests": int(len(reqs))}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def gsheet_apply_trades_only_format(force: bool = False, timeout_sec: int = 35) -> Dict[str, Any]:
    """
    UI/운영자가 수동으로 서식 적용을 강제할 때 사용.
    - trades_only 모드에서만 동작.
    """
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gsheet_mode() == "legacy":
        return {"ok": False, "error": "GSHEET_MODE=legacy(서식 자동 적용은 trades_only 전용)"}
    if gspread is None or GoogleCredentials is None:
        return {"ok": False, "error": "gspread/google-auth 미설치(requirements.txt 확인)"}

    def _do():
        sh = _gsheet_connect_spreadsheet()
        if sh is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 연결 실패")
            raise RuntimeError(err)
        sheets = _gsheet_prepare_trades_only_sheets(sh)
        if sheets is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 시트 준비 실패")
            raise RuntimeError(err)
        st0 = _gsheet_sync_state_load()
        res = _gsheet_apply_trades_only_format_internal(sh, sheets, st0, force=bool(force))
        if not bool(res.get("ok", False)):
            raise RuntimeError(str(res.get("error", "") or "format_failed"))
        _gsheet_sync_state_save(st0)
        return res

    try:
        return _call_with_timeout(_do, max(15, int(timeout_sec)))
    except Exception as e:
        notify_admin_error("GSHEET_FORMAT", e, context={"force": bool(force), "code": CODE_VERSION}, tb=traceback.format_exc(), min_interval_sec=180.0)
        return {"ok": False, "error": str(e)}


def _gsheet_sync_seed_from_sheet(ws_trade: Any, trade_id_col_index_1based: int = 11, max_ids: int = 6000) -> List[str]:
    """
    state 파일이 없는 환경(배포 재시작 등)에서도 중복 append를 줄이기 위해,
    시트에서 TradeID 컬럼을 읽어 synced_trade_ids를 시드한다.
    """
    try:
        col = ws_trade.col_values(int(trade_id_col_index_1based))  # network
        # 1행 헤더 제거 + 뒤쪽만 유지
        ids = [str(x or "").strip() for x in col[1:] if str(x or "").strip()]
        if len(ids) > int(max_ids):
            ids = ids[-int(max_ids) :]
        return ids
    except Exception:
        return []


def _gsheet_tradeid_col_index_1based(ws_trade: Any) -> int:
    """
    시트 헤더에서 '일지ID/TradeID' 컬럼을 찾아 1-based index 반환.
    - 헤더가 없거나 탐지 실패 시 11(구버전) 또는 마지막 컬럼 fallback.
    """
    try:
        header = []
        try:
            header = ws_trade.row_values(1) or []
        except Exception:
            header = []
        cand = {"tradeid", "일지id", "journalid", "logid"}
        for i, v in enumerate(header):
            key = str(v or "").strip().lower().replace(" ", "")
            if key in cand:
                return int(i + 1)
        # 구버전 영문 헤더(11번째)
        if len(header) >= 11:
            return 11
        if len(header) >= 1:
            return int(len(header))
        # 헤더가 비어있으면 최신 헤더 기준 마지막
        return int(len(GSHEET_TRADE_JOURNAL_HEADER))
    except Exception:
        return 11


def _trade_log_to_hourly_daily(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    trade_log.csv(df) -> (hourly_summary_df, daily_summary_df)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=GSHEET_HOURLY_SUMMARY_HEADER), pd.DataFrame(columns=GSHEET_DAILY_SUMMARY_HEADER)

    dfx = df.copy()
    try:
        dfx["Time_dt"] = pd.to_datetime(dfx["Time"].astype(str), errors="coerce")
        dfx = dfx.dropna(subset=["Time_dt"]).copy()
    except Exception:
        dfx["Time_dt"] = pd.NaT
    if dfx.empty:
        return pd.DataFrame(columns=GSHEET_HOURLY_SUMMARY_HEADER), pd.DataFrame(columns=GSHEET_DAILY_SUMMARY_HEADER)

    dfx["PnL_USDT_num"] = pd.to_numeric(dfx.get("PnL_USDT"), errors="coerce").fillna(0.0)
    dfx["PnL_Pct_num"] = pd.to_numeric(dfx.get("PnL_Percent"), errors="coerce").fillna(0.0)

    dfx["_hour_kst"] = dfx["Time_dt"].dt.strftime("%Y-%m-%d %H:00")
    dfx["_date_kst"] = dfx["Time_dt"].dt.strftime("%Y-%m-%d")

    asof = now_kst_str()

    def _pf(pnl_usdt: pd.Series) -> float:
        try:
            gains = float(pnl_usdt[pnl_usdt > 0].sum())
            losses = float((-pnl_usdt[pnl_usdt < 0]).sum())
            if losses > 0:
                return float(gains / losses)
            return float("inf") if gains > 0 else 0.0
        except Exception:
            return 0.0

    # Hourly
    h_rows = []
    try:
        g = dfx.groupby("_hour_kst", dropna=False)
        for k, sub in g:
            pnl_pct = sub["PnL_Pct_num"]
            pnl_usdt = sub["PnL_USDT_num"]
            trades = int(len(sub))
            win_rate = float((pnl_pct > 0).sum() / max(1, trades) * 100.0)
            h_rows.append(
                {
                    "시간대(KST)": str(k),
                    "거래수": trades,
                    "승률(%)": round(win_rate, 2),
                    "총손익(USDT)": round(float(pnl_usdt.sum()), 6),
                    "평균수익률(%)": round(float(pnl_pct.mean()) if trades else 0.0, 4),
                    "PF": round(_pf(pnl_usdt), 4) if trades else 0.0,
                    "갱신시각(KST)": asof,
                }
            )
    except Exception:
        pass
    df_h = pd.DataFrame(h_rows, columns=GSHEET_HOURLY_SUMMARY_HEADER)
    if not df_h.empty:
        try:
            df_h = df_h.sort_values("시간대(KST)", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    # Daily
    d_rows = []
    try:
        g2 = dfx.groupby("_date_kst", dropna=False)
        for k, sub in g2:
            pnl_pct = sub["PnL_Pct_num"]
            pnl_usdt = sub["PnL_USDT_num"]
            trades = int(len(sub))
            win_rate = float((pnl_pct > 0).sum() / max(1, trades) * 100.0)
            # 간이 MDD: 누적 pnl_pct 기준
            eq = pnl_pct.cumsum()
            dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
            d_rows.append(
                {
                    "날짜(KST)": str(k),
                    "거래수": trades,
                    "승률(%)": round(win_rate, 2),
                    "총손익(USDT)": round(float(pnl_usdt.sum()), 6),
                    "평균수익률(%)": round(float(pnl_pct.mean()) if trades else 0.0, 4),
                    "최대DD(%)": round(dd, 4),
                    "PF": round(_pf(pnl_usdt), 4) if trades else 0.0,
                    "갱신시각(KST)": asof,
                }
            )
    except Exception:
        pass
    df_d = pd.DataFrame(d_rows, columns=GSHEET_DAILY_SUMMARY_HEADER)
    if not df_d.empty:
        try:
            df_d = df_d.sort_values("날짜(KST)", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    return df_h, df_d


def _daily_summary_to_calendar_values(df_d: pd.DataFrame, year: int, month: int) -> List[List[Any]]:
    """
    일별 요약(df_d)을 "달력 형태"의 값 테이블로 변환한다.
    - 시트 레이아웃: A=라벨, B~H=월~일
    - 각 주마다 3행 블록: 날짜 / 손익(USDT) / 거래수
    """
    import calendar as _cal

    y = int(year)
    m = int(month)
    asof = now_kst_str()

    # map: YYYY-MM-DD -> {pnl, trades}
    pnl_map: Dict[str, float] = {}
    trades_map: Dict[str, int] = {}
    try:
        if df_d is not None and (not df_d.empty) and ("날짜(KST)" in df_d.columns):
            for _, r in df_d.iterrows():
                ds = str(r.get("날짜(KST)", "") or "").strip()
                if not ds:
                    continue
                try:
                    pnl_map[ds] = float(pd.to_numeric(r.get("총손익(USDT)"), errors="coerce") or 0.0)
                except Exception:
                    pnl_map[ds] = float(r.get("총손익(USDT)", 0.0) or 0.0)
                try:
                    trades_map[ds] = int(pd.to_numeric(r.get("거래수"), errors="coerce") or 0)
                except Exception:
                    try:
                        trades_map[ds] = int(r.get("거래수", 0) or 0)
                    except Exception:
                        trades_map[ds] = 0
    except Exception:
        pnl_map = {}
        trades_map = {}

    cal = _cal.Calendar(firstweekday=0)  # 0=월
    weeks = cal.monthdatescalendar(y, m) or []
    # 6주 고정(표 크기 일정)
    while len(weeks) < 6:
        if weeks:
            last = weeks[-1][-1]
            nxt = [last + timedelta(days=i) for i in range(1, 8)]
            weeks.append(nxt)
        else:
            # fallback: 빈 6주
            base = datetime(y, m, 1).date()
            weeks = [[base + timedelta(days=i) for i in range(7)] for _ in range(6)]
            break
    weeks = weeks[:6]

    # Header rows (8 cols: A..H)
    values: List[List[Any]] = []
    values.append([f"📅 {y:04d}-{m:02d} 매매 달력(KST)", f"업데이트: {asof}", "", "", "", "", "", ""])
    values.append(["", "월", "화", "수", "목", "금", "토", "일"])

    for wk in weeks:
        # 날짜(일자)
        row_day = ["날짜"]
        row_pnl = ["손익(USDT)"]
        row_tr = ["거래수"]
        for d in wk[:7]:
            try:
                in_month = (d.year == y and d.month == m)
            except Exception:
                in_month = False
            if not in_month:
                row_day.append("")
                row_pnl.append("")
                row_tr.append("")
                continue
            ds = f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
            row_day.append(str(d.day))
            row_pnl.append(float(pnl_map.get(ds, 0.0)))
            row_tr.append(int(trades_map.get(ds, 0)))
        values.append(row_day)
        values.append(row_pnl)
        values.append(row_tr)

    # 20 rows fixed: 2 + 6*3
    if len(values) < 20:
        for _ in range(20 - len(values)):
            values.append([""] * 8)
    return values[:20]


def gsheet_sync_trades_only(force_summary: bool = False, timeout_sec: int = 35) -> Dict[str, Any]:
    """
    ✅ 사용자 요구 반영:
    - Google Sheets에는 매매일지(trade_log.csv)만 append
    - 시간대별/일별 총합은 별도 워크시트로 갱신
    """
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gspread is None or GoogleCredentials is None:
        return {"ok": False, "error": "gspread/google-auth 미설치(requirements.txt 확인)"}

    def _do():
        sh = _gsheet_connect_spreadsheet()
        if sh is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 연결 실패")
            raise RuntimeError(err)

        sheets = _gsheet_prepare_trades_only_sheets(sh)
        if sheets is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 시트 준비 실패")
            raise RuntimeError(err)

        ws_trade = sheets["ws_trade"]
        ws_hourly = sheets["ws_hourly"]
        ws_daily = sheets["ws_daily"]
        ws_calendar = sheets.get("ws_calendar")
        ws_reviews = sheets.get("ws_reviews")
        # 상태 캐시(진단용)
        try:
            with _GSHEET_CACHE_LOCK:
                _GSHEET_CACHE["ws"] = ws_trade
                _GSHEET_CACHE["header_ok"] = True
                _GSHEET_CACHE["worksheet"] = str(sheets.get("trade_title", "") or _gsheet_get_settings().get("worksheet", ""))
                _GSHEET_CACHE["last_init_epoch"] = time.time()
        except Exception:
            pass

        # 상태 로드(중복 append 방지)
        st0 = _gsheet_sync_state_load()
        try:
            st0["trade_ws_title"] = str(sheets.get("trade_title", "") or "")
            st0["hourly_ws_title"] = str(sheets.get("hourly_title", "") or "")
            st0["daily_ws_title"] = str(sheets.get("daily_title", "") or "")
            st0["calendar_ws_title"] = str(sheets.get("calendar_title", "") or "")
            st0["reviews_ws_title"] = str(sheets.get("reviews_title", "") or "")
        except Exception:
            pass

        # ✅ Google Sheets 표(서식) 자동 적용(권장)
        # - 1회만 적용(버전+시트명으로 중복 방지)
        # - 실패해도 매매일지 sync는 계속 진행
        try:
            fmt = _gsheet_apply_trades_only_format_internal(sh, sheets, st0, force=False)
            if isinstance(fmt, dict) and (not bool(fmt.get("ok", True))):
                notify_admin_error("GSHEET_FORMAT_AUTO", RuntimeError(str(fmt.get("error", "format_failed"))), context={"code": CODE_VERSION}, min_interval_sec=300.0)
        except Exception as _e:
            notify_admin_error("GSHEET_FORMAT_AUTO", _e, context={"code": CODE_VERSION}, tb=traceback.format_exc(), min_interval_sec=300.0)

        synced_list = st0.get("synced_trade_ids", []) or []
        if not synced_list:
            # 시트에서 seed(배포 재시작/파일 초기화 대비)
            tid_col = _gsheet_tradeid_col_index_1based(ws_trade)
            seeded = _gsheet_sync_seed_from_sheet(ws_trade, trade_id_col_index_1based=int(tid_col), max_ids=6000)
            if seeded:
                synced_list = seeded
                st0["synced_trade_ids"] = list(seeded)

        synced = set([str(x or "").strip() for x in (synced_list or []) if str(x or "").strip()])
        synced_review_list = st0.get("synced_review_ids", []) or []
        if (not synced_review_list) and ws_reviews is not None:
            try:
                seeded_r = _gsheet_sync_seed_from_sheet(ws_reviews, trade_id_col_index_1based=len(GSHEET_REVIEWS_HEADER), max_ids=6000)
                if seeded_r:
                    synced_review_list = seeded_r
                    st0["synced_review_ids"] = list(seeded_r)
            except Exception:
                pass
        synced_reviews = set([str(x or "").strip() for x in (synced_review_list or []) if str(x or "").strip()])

        df = read_trade_log()
        if df is None or df.empty:
            # 거래가 없더라도 summary를 주기적으로 갱신할 필요는 없음
            st0["last_trade_sync_epoch"] = time.time()
            st0["last_trade_sync_kst"] = now_kst_str()
            _gsheet_sync_state_save(st0)
            return {"ok": True, "appended": 0, "summary": False}

        # 오래된 순서로 append (시트 보기 자연스러움)
        try:
            df2 = df.copy()
            df2["Time_dt"] = pd.to_datetime(df2["Time"].astype(str), errors="coerce")
            df2 = df2.sort_values("Time_dt", ascending=True)
        except Exception:
            df2 = df.iloc[::-1]

        # ✅ 구글시트 초기화(Reset) 이후에는 reset_kst 이후의 로그만 반영
        reset_kst = str(st0.get("reset_kst", "") or "").strip()
        if reset_kst:
            try:
                df2 = df2[df2["Time"].astype(str) >= reset_kst].copy()
            except Exception:
                pass

        new_rows = []
        new_ids: List[str] = []
        for _, r in df2.iterrows():
            try:
                tid = str(r.get("TradeID", "") or "").strip()
                if not tid:
                    # 비상 fallback(중복 가능성 낮게): Time+Coin+Side
                    tid = f"NOID:{str(r.get('Time',''))}|{str(r.get('Coin',''))}|{str(r.get('Side',''))}"
                if tid in synced:
                    continue
                # Google Sheets(한글) 행 구성
                def _cell(col: str, limit: int = 0) -> str:
                    try:
                        v = r.get(col, "")
                        if v is None:
                            s = ""
                        else:
                            # pandas NaN 처리
                            try:
                                if pd.isna(v):
                                    s = ""
                                else:
                                    s = str(v)
                            except Exception:
                                s = str(v)
                        if limit and len(s) > limit:
                            return s[:limit]
                        return s
                    except Exception:
                        return ""

                # 상태/방향(직관)
                try:
                    pnl_pct_f = float(pd.to_numeric(r.get("PnL_Percent", 0), errors="coerce") or 0.0)
                except Exception:
                    pnl_pct_f = 0.0
                status_txt = "🟢 수익" if pnl_pct_f > 0 else ("🔴 손실" if pnl_pct_f < 0 else "⚪ 보합")
                side_raw = str(r.get("Side", "") or "").strip().lower()
                side_ko = "롱" if side_raw in ["long", "buy"] else ("숏" if side_raw in ["short", "sell"] else side_raw)

                row = [
                    status_txt,
                    _cell("Time"),
                    _cell("Coin"),
                    side_ko,
                    _cell("Entry"),
                    _cell("Exit"),
                    _cell("PnL_USDT"),
                    _cell("PnL_Percent"),
                    _cell("BalanceBefore_Total"),
                    _cell("BalanceAfter_Total"),
                    _cell("BalanceBefore_Free"),
                    _cell("BalanceAfter_Free"),
                    _cell("Reason", limit=200),
                    _cell("OneLine", limit=200),
                    _cell("Review", limit=800),
                    (_cell("TradeID", limit=60) or str(tid)[:60]),
                ]
                new_rows.append(row)
                new_ids.append(tid)
            except Exception:
                continue

        appended = 0
        if new_rows:
            # append_rows(지원 시)로 요청 수 최소화
            if hasattr(ws_trade, "append_rows"):
                ws_trade.append_rows(new_rows, value_input_option="USER_ENTERED")  # type: ignore[attr-defined]
            else:
                for row in new_rows:
                    ws_trade.append_row(row, value_input_option="USER_ENTERED")
            appended = len(new_rows)

            # synced ids 갱신(순서 유지) + cap
            st_list = st0.get("synced_trade_ids", []) or []
            for tid in new_ids:
                if tid and (tid not in synced):
                    st_list.append(tid)
                    synced.add(tid)
            if len(st_list) > 8000:
                st_list = st_list[-8000:]
            st0["synced_trade_ids"] = st_list

        # ✅ 회고 모음(손실 트레이드만) append (요구사항)
        appended_reviews = 0
        try:
            if ws_reviews is not None:
                review_rows = []
                review_new_ids: List[str] = []
                for _, r in df2.iterrows():
                    try:
                        tid = str(r.get("TradeID", "") or "").strip()
                        if not tid:
                            tid = f"NOID:{str(r.get('Time',''))}|{str(r.get('Coin',''))}|{str(r.get('Side',''))}"
                        if tid in synced_reviews:
                            continue
                        pnl_pct_f = float(pd.to_numeric(r.get("PnL_Percent", 0), errors="coerce") or 0.0)
                        pnl_usdt_f = float(pd.to_numeric(r.get("PnL_USDT", 0), errors="coerce") or 0.0)
                        # 손실만
                        if not (pnl_pct_f < 0 or pnl_usdt_f < 0):
                            continue
                        side_raw = str(r.get("Side", "") or "").strip().lower()
                        side_ko = "롱" if side_raw in ["long", "buy"] else ("숏" if side_raw in ["short", "sell"] else side_raw)
                        def _c(col: str, limit: int = 0) -> str:
                            try:
                                v = r.get(col, "")
                                if v is None:
                                    s = ""
                                else:
                                    try:
                                        if pd.isna(v):
                                            s = ""
                                        else:
                                            s = str(v)
                                    except Exception:
                                        s = str(v)
                                if limit and len(s) > limit:
                                    return s[:limit]
                                return s
                            except Exception:
                                return ""
                        row_r = [
                            _c("Time"),
                            _c("Coin"),
                            side_ko,
                            _c("PnL_Percent"),
                            _c("PnL_USDT"),
                            _c("BalanceBefore_Total"),
                            _c("BalanceAfter_Total"),
                            _c("Reason", limit=200),
                            _c("OneLine", limit=200),
                            _c("Review", limit=1200),
                            (_c("TradeID", limit=60) or str(tid)[:60]),
                        ]
                        review_rows.append(row_r)
                        review_new_ids.append(tid)
                        if len(review_rows) >= 200:
                            break
                    except Exception:
                        continue

                if review_rows:
                    if hasattr(ws_reviews, "append_rows"):
                        ws_reviews.append_rows(review_rows, value_input_option="USER_ENTERED")  # type: ignore[attr-defined]
                    else:
                        for row in review_rows:
                            ws_reviews.append_row(row, value_input_option="USER_ENTERED")
                    appended_reviews = len(review_rows)
                    st_list_r = st0.get("synced_review_ids", []) or []
                    for tid in review_new_ids:
                        if tid and (tid not in synced_reviews):
                            st_list_r.append(tid)
                            synced_reviews.add(tid)
                    if len(st_list_r) > 8000:
                        st_list_r = st_list_r[-8000:]
                    st0["synced_review_ids"] = st_list_r
                    st0["last_review_sync_epoch"] = time.time()
                    st0["last_review_sync_kst"] = now_kst_str()
        except Exception:
            pass

        # summary 갱신 조건(쓰기 절감)
        now_ts = time.time()
        last_sum = float(st0.get("last_summary_sync_epoch", 0) or 0)
        summary_due = force_summary or (appended > 0) or ((now_ts - last_sum) >= 60 * 30)
        did_summary = False
        did_calendar = False
        if summary_due:
            df_h, df_d = _trade_log_to_hourly_daily(df2)
            # update (clear+update: 표가 짧아질 때 잔여행 방지)
            try:
                ws_hourly.clear()
                vals_h = [GSHEET_HOURLY_SUMMARY_HEADER]
                if df_h is not None and not df_h.empty:
                    vals_h += df_h.astype(str).values.tolist()
                ws_hourly.update("A1", vals_h)
            except Exception:
                pass
            try:
                ws_daily.clear()
                vals_d = [GSHEET_DAILY_SUMMARY_HEADER]
                if df_d is not None and not df_d.empty:
                    vals_d += df_d.astype(str).values.tolist()
                ws_daily.update("A1", vals_d)
            except Exception:
                pass
            # ✅ 달력형 일별 요약(요구사항): 현재 월 기준
            try:
                if ws_calendar is not None:
                    n0 = now_kst()
                    cal_vals = _daily_summary_to_calendar_values(df_d, int(n0.year), int(n0.month))
                    ws_calendar.update("A1", cal_vals)
                    did_calendar = True
                    st0["last_calendar_sync_epoch"] = now_ts
                    st0["last_calendar_sync_kst"] = now_kst_str()
            except Exception:
                pass
            did_summary = True
            st0["last_summary_sync_epoch"] = now_ts
            st0["last_summary_sync_kst"] = now_kst_str()

        st0["last_trade_sync_epoch"] = now_ts
        st0["last_trade_sync_kst"] = now_kst_str()
        _gsheet_sync_state_save(st0)

        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_append_epoch"] = now_ts
            _GSHEET_CACHE["last_append_kst"] = now_kst_str()
            _GSHEET_CACHE["last_append_type"] = "TRADE_LOG"
            _GSHEET_CACHE["last_append_stage"] = "TRADES_ONLY_SYNC"
            _GSHEET_CACHE["last_err"] = ""
            _GSHEET_CACHE["last_tb"] = ""
        return {"ok": True, "appended": appended, "reviews": appended_reviews, "summary": did_summary, "calendar": did_calendar}

    try:
        res = _call_with_timeout(_do, timeout_sec)
        return res if isinstance(res, dict) else {"ok": True}
    except FuturesTimeoutError:
        detail = "TimeoutError"
        try:
            with _GSHEET_CACHE_LOCK:
                _GSHEET_CACHE["last_err"] = f"GSHEET sync 실패: {detail}"
                try:
                    _GSHEET_CACHE["last_tb"] = traceback.format_exc()
                except Exception:
                    _GSHEET_CACHE["last_tb"] = ""
                # 타임아웃 연속 시 호출 폭주 방지(짧은 쿨다운)
                _GSHEET_CACHE["service_unavailable_until_epoch"] = max(
                    float(_GSHEET_CACHE.get("service_unavailable_until_epoch", 0) or 0.0),
                    time.time() + 45.0,
                )
                _GSHEET_CACHE["service_unavailable_kst"] = now_kst_str()
        except Exception:
            pass
        _gsheet_notify_connect_issue("GSHEET_SYNC", "GSHEET sync 실패: TimeoutError", min_interval_sec=600.0)
        return {"ok": False, "error": detail, "timeout": True}
    except Exception as e:
        # 503(Service Unavailable)은 일시 장애일 가능성이 높음 → 잠깐 쉬었다가 재시도
        detail = ""
        try:
            detail = _gsheet_exception_detail(e, limit=900)
        except Exception:
            detail = str(e)[:900]
        low = str(detail or "").lower()
        try:
            if ("http=503" in low) or ("[503]" in low) or ("service is currently unavailable" in low) or (" 503" in low):
                with _GSHEET_CACHE_LOCK:
                    _GSHEET_CACHE["service_unavailable_until_epoch"] = time.time() + 60 * 3
                    _GSHEET_CACHE["service_unavailable_kst"] = now_kst_str()
        except Exception:
            pass
        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_err"] = f"GSHEET sync 실패: {detail}"
            try:
                _GSHEET_CACHE["last_tb"] = traceback.format_exc()
            except Exception:
                _GSHEET_CACHE["last_tb"] = ""
        # 503은 사용자 조치가 거의 없으므로 알림 간격을 더 길게(스팸 방지)
        min_int = 180.0
        try:
            if ("http=503" in low) or ("[503]" in low) or ("service is currently unavailable" in low) or (" 503" in low):
                min_int = 1800.0
        except Exception:
            min_int = 180.0
        _gsheet_notify_connect_issue("GSHEET_SYNC", str(_GSHEET_CACHE.get("last_err", "") or detail), min_interval_sec=min_int)
        return {"ok": False, "error": str(detail or e)}


def gsheet_reset_trades_only(timeout_sec: int = 45) -> Dict[str, Any]:
    """
    ✅ UI 버튼용: trades_only 구글시트 매매일지 초기화
    - 매매일지/시간대/일별/달력/회고 시트를 clear + 헤더 재작성
    - reset_kst를 기록해, reset_kst 이전 trade_log.csv가 다시 올라가지 않게 방지
    """
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gsheet_mode() == "legacy":
        return {"ok": False, "error": "GSHEET_MODE=legacy(초기화는 trades_only 전용)"}
    if gspread is None or GoogleCredentials is None:
        return {"ok": False, "error": "gspread/google-auth 미설치(requirements.txt 확인)"}

    def _do():
        sh = _gsheet_connect_spreadsheet()
        if sh is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 연결 실패")
            raise RuntimeError(err)

        sheets = _gsheet_prepare_trades_only_sheets(sh)
        if sheets is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 시트 준비 실패")
            raise RuntimeError(err)

        ws_trade = sheets["ws_trade"]
        ws_hourly = sheets["ws_hourly"]
        ws_daily = sheets["ws_daily"]
        ws_calendar = sheets.get("ws_calendar")
        ws_reviews = sheets.get("ws_reviews")

        # 1) sync state 먼저 초기화(동시 실행 시 재업로드 방지)
        st0 = _gsheet_sync_state_load()
        st0["reset_epoch"] = float(time.time())
        st0["reset_kst"] = now_kst_str()
        st0["synced_trade_ids"] = []
        st0["synced_review_ids"] = []
        st0["last_trade_sync_epoch"] = 0.0
        st0["last_trade_sync_kst"] = ""
        st0["last_summary_sync_epoch"] = 0.0
        st0["last_summary_sync_kst"] = ""
        st0["last_calendar_sync_epoch"] = 0.0
        st0["last_calendar_sync_kst"] = ""
        st0["last_review_sync_epoch"] = 0.0
        st0["last_review_sync_kst"] = ""
        # 서식은 다음 sync에서 다시 1회 적용되게(안정/가독성)
        st0["format_version_applied"] = 0
        st0["format_applied_epoch"] = 0.0
        st0["format_applied_kst"] = ""
        st0["format_trade_title"] = ""
        st0["format_hourly_title"] = ""
        st0["format_daily_title"] = ""
        st0["format_calendar_title"] = ""
        st0["format_reviews_title"] = ""
        _gsheet_sync_state_save(st0)

        # 2) 시트 clear + 헤더 재작성
        try:
            ws_trade.clear()
        except Exception:
            pass
        ws_trade.update("A1", [GSHEET_TRADE_JOURNAL_HEADER])

        try:
            ws_hourly.clear()
        except Exception:
            pass
        ws_hourly.update("A1", [GSHEET_HOURLY_SUMMARY_HEADER])

        try:
            ws_daily.clear()
        except Exception:
            pass
        ws_daily.update("A1", [GSHEET_DAILY_SUMMARY_HEADER])

        if ws_reviews is not None:
            try:
                ws_reviews.clear()
            except Exception:
                pass
            ws_reviews.update("A1", [GSHEET_REVIEWS_HEADER])

        if ws_calendar is not None:
            try:
                ws_calendar.clear()
            except Exception:
                pass
            try:
                n0 = now_kst()
                cal_vals = _daily_summary_to_calendar_values(pd.DataFrame(), int(n0.year), int(n0.month))
                ws_calendar.update("A1", cal_vals)
            except Exception:
                pass

        try:
            _GSHEET_TRADE_SYNC_EVENT.set()
        except Exception:
            pass

        return {
            "ok": True,
            "reset_kst": str(st0.get("reset_kst", "") or ""),
            "trade_sheet": str(sheets.get("trade_title", "") or ""),
            "hourly_sheet": str(sheets.get("hourly_title", "") or ""),
            "daily_sheet": str(sheets.get("daily_title", "") or ""),
            "calendar_sheet": str(sheets.get("calendar_title", "") or ""),
            "reviews_sheet": str(sheets.get("reviews_title", "") or ""),
        }

    try:
        return _call_with_timeout(_do, max(20, int(timeout_sec)))
    except Exception as e:
        notify_admin_error("GSHEET_RESET", e, context={"code": CODE_VERSION}, tb=traceback.format_exc(), min_interval_sec=180.0)
        return {"ok": False, "error": str(e)}


def gsheet_status_snapshot() -> Dict[str, Any]:
    try:
        stg = _gsheet_get_settings()
        with _GSHEET_QUEUE_LOCK:
            qh = len(_GSHEET_QUEUE_HIGH)
            qs = len(_GSHEET_QUEUE_SCAN)
        with _GSHEET_CACHE_LOCK:
            last_init = float(_GSHEET_CACHE.get("last_init_epoch", 0) or 0)
            cd_until = float(_GSHEET_CACHE.get("quota_cooldown_until_epoch", 0) or 0)
            n_high = float(_GSHEET_CACHE.get("next_append_high_epoch", 0) or 0)
            n_scan = float(_GSHEET_CACHE.get("next_append_scan_epoch", 0) or 0)
            snap = {
                "enabled": bool(gsheet_is_enabled()),
                "mode": gsheet_mode(),
                "spreadsheet_id": stg.get("spreadsheet_id", ""),
                "worksheet": stg.get("worksheet", ""),
                "service_account_email": _gsheet_service_account_email(),
                "connected": bool(_GSHEET_CACHE.get("ws", None) is not None),
                "header_ok": bool(_GSHEET_CACHE.get("header_ok", False)),
                "queue_high": qh,
                "queue_scan": qs,
                "last_init_kst": _epoch_to_kst_str(last_init) if last_init else "",
                "last_append_kst": str(_GSHEET_CACHE.get("last_append_kst", "") or ""),
                "last_append_type": str(_GSHEET_CACHE.get("last_append_type", "") or ""),
                "last_append_stage": str(_GSHEET_CACHE.get("last_append_stage", "") or ""),
                "cooldown_until_kst": _epoch_to_kst_str(cd_until) if cd_until else "",
                "next_append_high_kst": _epoch_to_kst_str(n_high) if n_high else "",
                "next_append_scan_kst": _epoch_to_kst_str(n_scan) if n_scan else "",
                "min_append_sec": {"high": float(_GSHEET_MIN_APPEND_HIGH_SEC), "scan": float(_GSHEET_MIN_APPEND_SCAN_SEC)},
                "scan_throttle_sec": float(_GSHEET_SCAN_THROTTLE_SEC),
                "last_err": str(_GSHEET_CACHE.get("last_err", "") or ""),
            }
            # trades_only 상태 추가
            if gsheet_mode() != "legacy":
                try:
                    st0 = _gsheet_sync_state_load()
                    names = _gsheet_trade_ws_names()
                    snap.update(
                        {
                            "trade_sheet": str(st0.get("trade_ws_title") or names.get("trade", "")),
                            "hourly_sheet": str(st0.get("hourly_ws_title") or names.get("hourly", "")),
                            "daily_sheet": str(st0.get("daily_ws_title") or names.get("daily", "")),
                            "calendar_sheet": str(st0.get("calendar_ws_title") or names.get("calendar", "")),
                            "reviews_sheet": str(st0.get("reviews_ws_title") or names.get("reviews", "")),
                            "last_trade_sync_kst": str(st0.get("last_trade_sync_kst", "") or ""),
                            "last_summary_sync_kst": str(st0.get("last_summary_sync_kst", "") or ""),
                            "last_calendar_sync_kst": str(st0.get("last_calendar_sync_kst", "") or ""),
                            "last_review_sync_kst": str(st0.get("last_review_sync_kst", "") or ""),
                            "synced_trade_ids": int(len(st0.get("synced_trade_ids", []) or [])),
                            "synced_review_ids": int(len(st0.get("synced_review_ids", []) or [])),
                            "format_version_applied": int(st0.get("format_version_applied", 0) or 0),
                            "format_applied_kst": str(st0.get("format_applied_kst", "") or ""),
                        }
                    )
                except Exception:
                    pass
            return snap
    except Exception:
        return {"enabled": bool(gsheet_is_enabled()), "last_err": str(_GSHEET_CACHE.get("last_err", "") if isinstance(_GSHEET_CACHE, dict) else "")}


def _col_name_1based(n: int) -> str:
    n = int(max(1, int(n)))
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def _ws_get_values_preview(ws: Any, max_rows: int = 120, max_cols: int = 16) -> List[List[str]]:
    if ws is None:
        return []
    rows = int(max(2, int(max_rows)))
    cols = int(max(1, int(max_cols)))
    rng = f"A1:{_col_name_1based(cols)}{rows}"
    try:
        vals = ws.get(rng)
        if isinstance(vals, list) and vals:
            return vals
    except Exception:
        pass
    try:
        vals2 = ws.get_all_values()
        if isinstance(vals2, list):
            return vals2[:rows]
    except Exception:
        pass
    return []


def _sheet_values_to_df(values: List[List[Any]]) -> pd.DataFrame:
    if not values or not isinstance(values, list):
        return pd.DataFrame()
    try:
        header = [str(x or "").strip() for x in (values[0] or [])]
    except Exception:
        return pd.DataFrame()
    header = [h if h else f"col_{i+1}" for i, h in enumerate(header)]
    rows = []
    for r in values[1:]:
        rr = list(r or [])
        if len(rr) < len(header):
            rr += [""] * (len(header) - len(rr))
        rows.append(rr[: len(header)])
    if not rows:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(rows, columns=header)


def _to_float_unsafe(v: Any) -> float:
    try:
        s = str(v or "").strip()
        if not s:
            return float("nan")
        s = s.replace(",", "").replace("%", "").replace("USDT", "").strip()
        if not s:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _render_gsheet_table_image(
    title: str,
    df_show: pd.DataFrame,
    *,
    tag: str = "sheet",
    subtitle: str = "",
) -> Optional[str]:
    if plt is None or df_show is None or df_show.empty:
        return None
    try:
        has_kr_font = _ensure_trade_image_font()
        dfx = df_show.copy().fillna("")
        cols = [str(c or "") for c in dfx.columns.tolist()]
        rows = [[str(x or "") for x in row] for row in dfx.values.tolist()]
        nrows = int(len(rows))
        h = float(clamp(2.6 + (0.42 * float(nrows + 1)), 3.5, 15.0))
        fig, ax = plt.subplots(figsize=(13.4, h), dpi=125)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#11161c")
        ax.axis("off")

        ttl = _plot_text_sanitize(str(title or "").strip(), has_kr_font=has_kr_font, max_len=120)
        if ttl:
            ax.set_title(ttl, fontsize=12, color="#f8fafc", pad=10)
        if subtitle:
            stxt = _plot_text_sanitize(str(subtitle), has_kr_font=has_kr_font, max_len=180)
            if stxt:
                ax.text(0.5, 0.985, stxt, transform=ax.transAxes, ha="center", va="top", fontsize=8.2, color="#94a3b8")

        tbl = ax.table(
            cellText=rows,
            colLabels=cols,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.1 if nrows <= 18 else 7.6)
        tbl.scale(1.0, 1.20)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#334155")
            cell.set_linewidth(0.55)
            if r == 0:
                cell.set_facecolor("#1f2937")
                cell.get_text().set_color("#f8fafc")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("#0f172a" if (r % 2 == 1) else "#111827")
                cell.get_text().set_color("#e2e8f0")
                try:
                    col_name = cols[c] if c < len(cols) else ""
                    txt = str(rows[r - 1][c] if c < len(rows[r - 1]) else "")
                    if ("손익" in col_name) or ("PnL" in col_name):
                        vv = _to_float_unsafe(txt)
                        if math.isfinite(vv):
                            if vv > 0:
                                cell.get_text().set_color("#86efac")
                            elif vv < 0:
                                cell.get_text().set_color("#fda4af")
                except Exception:
                    pass

        ts = now_kst().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_gsheet_{re.sub(r'[^A-Za-z0-9]+', '_', str(tag or 'sheet'))[:24]}_{uuid.uuid4().hex[:6]}.png"
        out_path = os.path.join(EVENT_IMAGE_DIR, fname)
        fig.tight_layout(pad=0.9)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        _cleanup_event_images()
        return out_path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def gsheet_build_journal_snapshot(kind: str = "today", timeout_sec: int = 22) -> Dict[str, Any]:
    """
    텔레그램 버튼용:
    - kind: today | daily | monthly | trades
    - Google Sheets trades_only 시트 기반으로 표 이미지를 생성
    """
    k = str(kind or "today").lower().strip()
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gsheet_mode() == "legacy":
        return {"ok": False, "error": "legacy 모드에서는 미지원(trades_only 필요)"}
    if plt is None:
        return {"ok": False, "error": "matplotlib 미설치"}

    def _do():
        sh = _gsheet_connect_spreadsheet()
        if sh is None:
            raise RuntimeError(str(_GSHEET_CACHE.get("last_err", "") or "gsheet_connect_failed"))
        sheets = _gsheet_prepare_trades_only_sheets(sh)
        if not sheets:
            raise RuntimeError("gsheet_prepare_failed")
        ws_trade = sheets.get("ws_trade")
        ws_daily = sheets.get("ws_daily")
        if ws_trade is None or ws_daily is None:
            raise RuntimeError("ws_trade/ws_daily_missing")

        vals_trade = _ws_get_values_preview(ws_trade, max_rows=90, max_cols=18)
        vals_daily = _ws_get_values_preview(ws_daily, max_rows=370, max_cols=10)
        df_trade = _sheet_values_to_df(vals_trade)
        df_daily = _sheet_values_to_df(vals_daily)

        if k == "trades":
            if df_trade.empty:
                return {"ok": False, "error": "매매일지 시트가 비어있습니다."}
            keep = [c for c in ["시간(KST)", "코인", "방향", "손익(USDT)", "수익률(%)", "한줄평", "일지ID"] if c in df_trade.columns]
            show = df_trade[keep].head(14).copy() if keep else df_trade.head(14).copy()
            img = _render_gsheet_table_image("구글시트 최근 매매일지", show, tag="log_trades", subtitle=f"rows={len(show)}")
            if not img:
                return {"ok": False, "error": "이미지 생성 실패"}
            return {"ok": True, "image": img, "caption": "📜 구글시트 최근 매매일지", "rows": len(show)}

        if df_daily.empty:
            return {"ok": False, "error": "일별 요약 시트가 비어있습니다."}

        date_col = "날짜(KST)" if "날짜(KST)" in df_daily.columns else (df_daily.columns[0] if len(df_daily.columns) else "")
        trades_col = "거래수" if "거래수" in df_daily.columns else ""
        pnl_col = "총손익(USDT)" if "총손익(USDT)" in df_daily.columns else ""
        wr_col = "승률(%)" if "승률(%)" in df_daily.columns else ""
        avg_col = "평균수익률(%)" if "평균수익률(%)" in df_daily.columns else ""
        pf_col = "PF" if "PF" in df_daily.columns else ""

        if k == "today":
            today = today_kst_str()
            one = df_daily[df_daily[date_col].astype(str).str.startswith(today)].head(1).copy() if date_col else pd.DataFrame()
            if one.empty:
                rt = load_runtime()
                one = pd.DataFrame(
                    [
                        {
                            "날짜(KST)": today,
                            "거래수": int(rt.get("daily_trade_count", 0) or 0),
                            "승률(%)": round((float(rt.get("daily_win_count", 0) or 0) / max(1, int(rt.get("daily_trade_count", 0) or 0))) * 100.0, 2),
                            "총손익(USDT)": round(float(rt.get("daily_realized_pnl", 0.0) or 0.0), 4),
                            "평균수익률(%)": 0.0,
                            "PF": "-",
                        }
                    ]
                )
            keep = [c for c in ["날짜(KST)", "거래수", "승률(%)", "총손익(USDT)", "평균수익률(%)", "PF"] if c in one.columns]
            show = one[keep].copy() if keep else one.copy()
            img = _render_gsheet_table_image(f"금일 손익 ({today})", show, tag="log_today")
            if not img:
                return {"ok": False, "error": "이미지 생성 실패"}
            return {"ok": True, "image": img, "caption": f"📌 금일 손익 ({today})", "rows": len(show)}

        if k == "monthly":
            dfm = df_daily.copy()
            if not date_col:
                return {"ok": False, "error": "일별 날짜 컬럼 없음"}
            dfm["월"] = dfm[date_col].astype(str).str.slice(0, 7)
            if trades_col:
                dfm["_trades"] = pd.to_numeric(dfm[trades_col], errors="coerce").fillna(0.0)
            else:
                dfm["_trades"] = 0.0
            if pnl_col:
                dfm["_pnl"] = dfm[pnl_col].map(_to_float_unsafe).fillna(0.0)
            else:
                dfm["_pnl"] = 0.0
            if wr_col:
                dfm["_wr"] = dfm[wr_col].map(_to_float_unsafe).fillna(0.0)
            else:
                dfm["_wr"] = 0.0
            grp = (
                dfm.groupby("월", dropna=False)
                .agg({"_trades": "sum", "_pnl": "sum", "_wr": "mean"})
                .reset_index()
                .sort_values("월", ascending=False)
                .head(12)
                .copy()
            )
            grp.rename(columns={"_trades": "거래수", "_pnl": "총손익(USDT)", "_wr": "평균승률(%)"}, inplace=True)
            grp["거래수"] = grp["거래수"].astype(int)
            grp["총손익(USDT)"] = grp["총손익(USDT)"].map(lambda x: f"{float(x):+.2f}")
            grp["평균승률(%)"] = grp["평균승률(%)"].map(lambda x: f"{float(x):.2f}")
            img = _render_gsheet_table_image("월별 손익 요약", grp, tag="log_monthly", subtitle="최근 12개월")
            if not img:
                return {"ok": False, "error": "이미지 생성 실패"}
            return {"ok": True, "image": img, "caption": "📆 월별 손익 요약", "rows": len(grp)}

        keep = [c for c in [date_col, trades_col, wr_col, pnl_col, avg_col, pf_col] if c]
        show = df_daily[keep].head(20).copy() if keep else df_daily.head(20).copy()
        img = _render_gsheet_table_image("일별 손익 요약", show, tag="log_daily", subtitle=f"rows={len(show)}")
        if not img:
            return {"ok": False, "error": "이미지 생성 실패"}
        return {"ok": True, "image": img, "caption": "🗓️ 일별 손익 요약", "rows": len(show)}

    try:
        return _call_with_timeout(_do, timeout_sec=max(12, int(timeout_sec)))
    except Exception as e:
        return {"ok": False, "error": str(e)}


def gsheet_test_append_row(timeout_sec: int = 20) -> Dict[str, Any]:
    """
    수동 진단용:
    - 연결 + 헤더 + append_row를 즉시 수행해서 권한/설정 문제를 바로 확인.
    """
    if not gsheet_is_enabled():
        return {"ok": False, "error": "GSHEET_ENABLED=false"}
    if gspread is None or GoogleCredentials is None:
        return {"ok": False, "error": "gspread/google-auth 미설치(requirements.txt 확인)"}

    # trades_only 모드: 실제 운영 sync를 실행해서 연결/권한/쓰기까지 확인
    if gsheet_mode() != "legacy":
        res = gsheet_sync_trades_only(force_summary=True, timeout_sec=max(20, int(timeout_sec)))
        return {"ok": bool(res.get("ok", False)), "error": str(res.get("error", "") or "")}

    def _do():
        ws = _gsheet_connect_ws()
        if ws is None:
            err = str(_GSHEET_CACHE.get("last_err", "") or "unknown_error")
            raise RuntimeError(err)
        _gsheet_ensure_header(ws)
        rec = {
            "time_kst": now_kst_str(),
            "type": "EVENT",
            "stage": "GSHEET_TEST",
            "symbol": "",
            "tf": "",
            "signal": "",
            "score": "",
            "trade_id": "",
            "message": f"manual_test code={CODE_VERSION}",
            "payload_json": {"code": CODE_VERSION},
        }
        row = [
            rec["time_kst"],
            rec["type"],
            rec["stage"],
            rec["symbol"],
            rec["tf"],
            rec["signal"],
            rec["score"],
            rec["trade_id"],
            rec["message"],
            safe_json_dumps(rec["payload_json"], limit=1800),
        ]
        ws.append_row(row, value_input_option="USER_ENTERED")
        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_append_epoch"] = time.time()
            _GSHEET_CACHE["last_append_kst"] = now_kst_str()
            _GSHEET_CACHE["last_append_type"] = "EVENT"
            _GSHEET_CACHE["last_append_stage"] = "GSHEET_TEST"
        return True

    try:
        _call_with_timeout(_do, timeout_sec)
        return {"ok": True}
    except Exception as e:
        with _GSHEET_CACHE_LOCK:
            _GSHEET_CACHE["last_err"] = f"GSHEET 테스트 실패: {e}"
        _gsheet_notify_connect_issue("GSHEET_TEST", f"GSHEET 테스트 실패: {e}", min_interval_sec=120.0)
        return {"ok": False, "error": str(e)}


def _gsheet_ensure_header(ws: Any) -> None:
    try:
        if _GSHEET_CACHE.get("header_ok"):
            return
        first = []
        try:
            first = ws.row_values(1)  # network
        except Exception:
            first = []
        if not first or (first and str(first[0]).strip().lower() != "time_kst"):
            ws.append_row(GSHEET_HEADER, value_input_option="USER_ENTERED")
        _GSHEET_CACHE["header_ok"] = True
    except Exception:
        pass


def gsheet_enqueue(rec: Dict[str, Any]) -> None:
    if not gsheet_is_enabled():
        return
    # trades_only 모드에서는 원본 로그(TRADE/EVENT/SCAN) 큐를 사용하지 않는다.
    if gsheet_mode() != "legacy":
        return
    try:
        rr = dict(rec or {})
        rr.setdefault("time_kst", now_kst_str())
        rr.setdefault("type", "EVENT")
        rr.setdefault("stage", "")
        rr.setdefault("symbol", "")
        rr.setdefault("tf", "")
        rr.setdefault("signal", "")
        rr.setdefault("score", "")
        rr.setdefault("trade_id", "")
        rr.setdefault("message", "")
        payload = rr.get("payload_json", "")
        if not isinstance(payload, str):
            rr["payload_json"] = safe_json_dumps(payload, limit=1800)
        with _GSHEET_QUEUE_LOCK:
            typ = str(rr.get("type", "EVENT")).strip().upper()
            if typ in ["TRADE", "EVENT"]:
                _GSHEET_QUEUE_HIGH.append(rr)
                # 과도 누적 방지(중요 로그는 최대한 유지)
                while len(_GSHEET_QUEUE_HIGH) > 600:
                    _GSHEET_QUEUE_HIGH.popleft()
            else:
                _GSHEET_QUEUE_SCAN.append(rr)
                # 과도 누적 방지(SCAN은 오래된 것부터 버림)
                while len(_GSHEET_QUEUE_SCAN) > 1800:
                    _GSHEET_QUEUE_SCAN.popleft()
    except Exception:
        pass


def gsheet_log_trade(stage: str, symbol: str, trade_id: str = "", message: str = "", payload: Optional[Dict[str, Any]] = None):
    # trades_only 모드: trade_log.csv 동기화 트리거만 수행(실제 append는 워커가 처리)
    if gsheet_mode() != "legacy":
        try:
            # trade_log.csv가 갱신되는 타이밍(=log_trade 호출)에서만 sync 트리거
            # - stage=JOURNAL은 log_trade()에서 호출됨
            stg = str(stage or "").strip().upper()
            if stg in ["JOURNAL", "TRADE_LOG", "LOG"]:
                _GSHEET_TRADE_SYNC_EVENT.set()
        except Exception:
            pass
        return
    gsheet_enqueue(
        {
            "type": "TRADE",
            "stage": stage,
            "symbol": symbol,
            "trade_id": trade_id,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_log_event(stage: str, message: str = "", payload: Optional[Dict[str, Any]] = None):
    # trades_only 모드에서는 EVENT를 구글시트에 남기지 않음(사용자 요구)
    if gsheet_mode() != "legacy":
        return
    gsheet_enqueue(
        {
            "type": "EVENT",
            "stage": stage,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_log_scan(stage: str, symbol: str, tf: str = "", signal: str = "", score: Any = "", message: str = "", payload: Optional[Dict[str, Any]] = None):
    # trades_only 모드에서는 SCAN을 구글시트에 남기지 않음(사용자 요구)
    if gsheet_mode() != "legacy":
        return
    # ✅ SCAN은 매우 자주 발생하므로(특히 5코인 * 다단계),
    #    시트에는 stage/심볼별로 throttle 샘플링해서 API 에러/레이트리밋을 줄인다.
    try:
        stg = str(stage or "").strip()
        sym = str(symbol or "").strip()
        stg_key = stg.lower()
        # 중요한 단계는 무조건 기록
        if stg_key and (stg_key not in _GSHEET_SCAN_ALWAYS_STAGES) and sym and sym != "*":
            now = time.time()
            k = f"{sym}|{stg_key}"
            with _GSHEET_SCAN_THROTTLE_LOCK:
                last = float(_GSHEET_SCAN_LAST.get(k, 0) or 0)
                if (now - last) < float(_GSHEET_SCAN_THROTTLE_SEC):
                    return
                _GSHEET_SCAN_LAST[k] = now
                if len(_GSHEET_SCAN_LAST) > int(_GSHEET_SCAN_THROTTLE_MAX_KEYS):
                    # 오래된 것부터 정리(메모리 누수 방지)
                    for kk in sorted(_GSHEET_SCAN_LAST, key=_GSHEET_SCAN_LAST.get)[:300]:
                        _GSHEET_SCAN_LAST.pop(kk, None)
    except Exception:
        pass

    gsheet_enqueue(
        {
            "type": "SCAN",
            "stage": stage,
            "symbol": symbol,
            "tf": tf,
            "signal": signal,
            "score": score,
            "message": message,
            "payload_json": payload or {},
        }
    )


def gsheet_worker_thread():
    backoff = 1.0
    while True:
        batch: List[Dict[str, Any]] = []
        batch_is_high = False  # TRADE/EVENT batch
        try:
            if not gsheet_is_enabled():
                time.sleep(2.0)
                continue

            # =================================================
            # trades_only 모드: 매매일지 + 시간대/일별 총합만 sync
            # =================================================
            if gsheet_mode() != "legacy":
                # quota cooldown
                now_ts = time.time()
                with _GSHEET_CACHE_LOCK:
                    cd_until = float(_GSHEET_CACHE.get("quota_cooldown_until_epoch", 0) or 0)
                if cd_until and now_ts < cd_until:
                    time.sleep(1.0)
                    continue
                # service unavailable cooldown (503 등)
                try:
                    with _GSHEET_CACHE_LOCK:
                        su_until = float(_GSHEET_CACHE.get("service_unavailable_until_epoch", 0) or 0)
                    if su_until and now_ts < su_until:
                        time.sleep(1.0)
                        continue
                except Exception:
                    pass

                # 이벤트 기반(매매일지 발생 시) + 주기적 self-heal
                try:
                    woke = _GSHEET_TRADE_SYNC_EVENT.wait(timeout=5.0)
                    if woke:
                        _GSHEET_TRADE_SYNC_EVENT.clear()
                except Exception:
                    woke = False

                do_sync = woke
                if not do_sync:
                    try:
                        st0 = _gsheet_sync_state_load()
                        last_ts = float(st0.get("last_trade_sync_epoch", 0) or 0)
                        if (now_ts - last_ts) >= 60 * 5:
                            do_sync = True
                    except Exception:
                        do_sync = True

                if do_sync:
                    try:
                        res = gsheet_sync_trades_only(force_summary=bool(woke), timeout_sec=35)
                    except FuturesTimeoutError:
                        res = {"ok": False, "error": "TimeoutError", "timeout": True}
                    except Exception as e:
                        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    if not bool(res.get("ok", False)):
                        msg = str(res.get("error", "") or "GSHEET sync 실패")
                        low = msg.lower()
                        is_timeout = bool(res.get("timeout", False)) or ("timeout" in low)
                        # 429 quota 대응
                        if ("http=429" in low) or ("quota exceeded" in low) or ("429" in low and "quota" in low):
                            with _GSHEET_CACHE_LOCK:
                                _GSHEET_CACHE["quota_cooldown_until_epoch"] = time.time() + float(_GSHEET_QUOTA_COOLDOWN_SEC)
                                _GSHEET_CACHE["last_429_epoch"] = time.time()
                            backoff = max(backoff, float(_GSHEET_QUOTA_COOLDOWN_SEC))
                        # timeout 대응(네트워크 지연/일시 장애): 짧은 쿨다운 후 재시도
                        if is_timeout:
                            try:
                                with _GSHEET_CACHE_LOCK:
                                    _GSHEET_CACHE["service_unavailable_until_epoch"] = max(
                                        float(_GSHEET_CACHE.get("service_unavailable_until_epoch", 0) or 0.0),
                                        time.time() + 45.0,
                                    )
                                    _GSHEET_CACHE["service_unavailable_kst"] = now_kst_str()
                            except Exception:
                                pass
                            backoff = max(backoff, 2.0)
                        # 503 service unavailable 대응(일시 장애): 조금 길게 쉬었다가 재시도
                        if ("http=503" in low) or ("[503]" in low) or ("service is currently unavailable" in low) or (" 503" in low):
                            try:
                                with _GSHEET_CACHE_LOCK:
                                    _GSHEET_CACHE["service_unavailable_until_epoch"] = time.time() + 60 * 3
                                    _GSHEET_CACHE["service_unavailable_kst"] = now_kst_str()
                            except Exception:
                                pass
                            backoff = max(backoff, 10.0)
                        # 503은 사용자가 할 수 있는 조치가 거의 없으므로 알림 간격을 더 길게(스팸 방지)
                        min_int = 180.0
                        try:
                            if ("http=503" in low) or ("[503]" in low) or ("service is currently unavailable" in low) or (" 503" in low):
                                min_int = 1800.0
                            elif is_timeout:
                                min_int = 600.0
                        except Exception:
                            min_int = 180.0
                        _gsheet_notify_connect_issue("GSHEET_THREAD", msg, min_interval_sec=min_int)
                        time.sleep(backoff)
                        backoff = float(clamp(backoff * 1.4, 1.0, 90.0))
                        continue
                    backoff = 1.0
                time.sleep(0.8)
                continue

            # ✅ secrets 변경(시트/워크시트) 감지 시 즉시 재연결
            try:
                stg_now = _gsheet_get_settings()
                with _GSHEET_CACHE_LOCK:
                    sid_now = str(stg_now.get("spreadsheet_id", "") or "").strip()
                    ws_now = str(stg_now.get("worksheet", "") or "").strip()
                    sid_old = str(_GSHEET_CACHE.get("spreadsheet_id", "") or "").strip()
                    ws_old = str(_GSHEET_CACHE.get("worksheet", "") or "").strip()
                    if (sid_now and sid_old and sid_now != sid_old) or (ws_now and ws_old and ws_now != ws_old):
                        _GSHEET_CACHE["ws"] = None
                        _GSHEET_CACHE["header_ok"] = False
                        _GSHEET_CACHE["last_init_epoch"] = 0.0
                        _GSHEET_CACHE["last_err"] = ""
            except Exception:
                pass

            # ✅ write quota 보호(429):
            # - SCAN은 append 요청 빈도를 제한(묶어서 보냄)
            # - 429 발생 시 일정 시간 쿨다운 후 재시도
            now_ts = time.time()
            with _GSHEET_CACHE_LOCK:
                cd_until = float(_GSHEET_CACHE.get("quota_cooldown_until_epoch", 0) or 0)
                next_high = float(_GSHEET_CACHE.get("next_append_high_epoch", 0) or 0)
                next_scan = float(_GSHEET_CACHE.get("next_append_scan_epoch", 0) or 0)
            if cd_until and now_ts < cd_until:
                # 쿨다운 중에는 SCAN backlog를 줄여서(최신 위주) 메모리/요청 폭주를 방지
                try:
                    with _GSHEET_QUEUE_LOCK:
                        while len(_GSHEET_QUEUE_SCAN) > 900:
                            _GSHEET_QUEUE_SCAN.popleft()
                except Exception:
                    pass
                time.sleep(min(2.0, max(0.2, cd_until - now_ts)))
                continue

            # ✅ batch pop: Google Sheets API 호출 수를 줄이기 위해 크게 묶어서 append
            has_high = False
            has_scan = False
            with _GSHEET_QUEUE_LOCK:
                has_high = bool(_GSHEET_QUEUE_HIGH)
                has_scan = bool(_GSHEET_QUEUE_SCAN)
                if has_high and now_ts >= next_high:
                    while _GSHEET_QUEUE_HIGH and len(batch) < 25:
                        batch.append(_GSHEET_QUEUE_HIGH.popleft())
                    if batch:
                        batch_is_high = True
                elif (not batch) and has_scan and now_ts >= next_scan:
                    # SCAN은 더 크게 묶어서 요청 수를 줄인다
                    while _GSHEET_QUEUE_SCAN and len(batch) < 200:
                        batch.append(_GSHEET_QUEUE_SCAN.popleft())
            if not batch:
                # 큐가 있지만 rate-limit 때문에 못 보내는 경우, 다음 가능 시각까지 대기
                wait_sec = 0.35
                try:
                    if has_high and now_ts < next_high:
                        wait_sec = max(0.2, min(1.0, next_high - now_ts))
                    elif (not has_high) and has_scan and now_ts < next_scan:
                        wait_sec = max(0.2, min(1.0, next_scan - now_ts))
                except Exception:
                    wait_sec = 0.35
                time.sleep(wait_sec)
                continue

            # 연결 캐시
            ws = None
            with _GSHEET_CACHE_LOCK:
                ws = _GSHEET_CACHE.get("ws", None)
                last_init = float(_GSHEET_CACHE.get("last_init_epoch", 0) or 0)
                # 오래됐으면 재연결 시도(네트워크/세션 이슈 대비)
                if ws is None or (time.time() - last_init) > 60 * 30:
                    ws = _gsheet_connect_ws()
                    _GSHEET_CACHE["ws"] = ws
                    _GSHEET_CACHE["header_ok"] = False
                    _GSHEET_CACHE["last_init_epoch"] = time.time()

            if ws is None:
                # 연결 실패면 재시도 위해 되돌려놓고 backoff
                try:
                    msg = str(_GSHEET_CACHE.get("last_err", "") or "GSHEET 연결 실패")
                    _gsheet_notify_connect_issue("GSHEET_CONNECT", msg, min_interval_sec=300.0)
                except Exception:
                    pass
                with _GSHEET_QUEUE_LOCK:
                    if batch_is_high:
                        for r in reversed(batch):
                            _GSHEET_QUEUE_HIGH.appendleft(r)
                    else:
                        for r in reversed(batch):
                            _GSHEET_QUEUE_SCAN.appendleft(r)
                time.sleep(backoff)
                backoff = float(clamp(backoff * 1.4, 1.0, 12.0))
                continue

            _gsheet_ensure_header(ws)

            rows = []
            for rec in batch:
                rows.append(
                    [
                        str(rec.get("time_kst", "")),
                        str(rec.get("type", "")),
                        str(rec.get("stage", "")),
                        str(rec.get("symbol", "")),
                        str(rec.get("tf", "")),
                        str(rec.get("signal", "")),
                        str(rec.get("score", "")),
                        str(rec.get("trade_id", "")),
                        str(rec.get("message", ""))[:500],
                        str(rec.get("payload_json", ""))[:1800],
                    ]
                )

            def _append_batch():
                # gspread 5+ 에서 append_rows 지원(요청 수 절감)
                if hasattr(ws, "append_rows"):
                    return ws.append_rows(rows, value_input_option="USER_ENTERED")  # type: ignore[attr-defined]
                # fallback: 구버전은 append_row 루프
                for row in rows:
                    ws.append_row(row, value_input_option="USER_ENTERED")
                return True

            # ✅ 내부 즉시 재시도(tenacity)는 quota 429에서 요청 수를 더 늘릴 수 있어
            #    워커 레벨(backoff/cooldown + requeue)로만 재시도한다.
            _append_batch()

            try:
                last_rec = batch[-1] if batch else {}
                with _GSHEET_CACHE_LOCK:
                    _GSHEET_CACHE["last_append_epoch"] = time.time()
                    _GSHEET_CACHE["last_append_kst"] = now_kst_str()
                    _GSHEET_CACHE["last_append_type"] = str(last_rec.get("type", "") or "")
                    _GSHEET_CACHE["last_append_stage"] = str(last_rec.get("stage", "") or "")
                    # 다음 append 가능 시각(요청 수 제한)
                    if batch_is_high:
                        _GSHEET_CACHE["next_append_high_epoch"] = time.time() + float(_GSHEET_MIN_APPEND_HIGH_SEC)
                    else:
                        _GSHEET_CACHE["next_append_scan_epoch"] = time.time() + float(_GSHEET_MIN_APPEND_SCAN_SEC)
                    _GSHEET_CACHE["last_err"] = ""
                    _GSHEET_CACHE["last_tb"] = ""
            except Exception:
                pass
            backoff = 1.0
        except Exception as e:
            # 실패해도 봇은 살아야 함(오류는 관리자에게 알림)
            try:
                with _GSHEET_CACHE_LOCK:
                    # tenacity RetryError는 메시지가 빈 경우가 많아, root cause를 뽑아서 보여준다.
                    root = e
                    wrapped = ""
                    try:
                        if type(e).__name__ == "RetryError" or hasattr(e, "last_attempt"):
                            la = getattr(e, "last_attempt", None)
                            if la is not None and hasattr(la, "exception"):
                                ex0 = la.exception()
                                if ex0 is not None:
                                    root = ex0
                                    wrapped = type(e).__name__
                    except Exception:
                        root = e
                        wrapped = ""
                    _GSHEET_CACHE["last_err"] = f"GSHEET append 실패: {_gsheet_exception_detail(root, limit=900)}"
                    _GSHEET_CACHE["last_tb"] = traceback.format_exc()
            except Exception:
                pass
            # 429(quota exceeded)면 분당 제한이 풀릴 때까지 쿨다운
            try:
                with _GSHEET_CACHE_LOCK:
                    last_err_txt = str(_GSHEET_CACHE.get("last_err", "") or "").lower()
                if ("http=429" in last_err_txt) or ("quota exceeded" in last_err_txt):
                    with _GSHEET_CACHE_LOCK:
                        _GSHEET_CACHE["quota_cooldown_until_epoch"] = time.time() + float(_GSHEET_QUOTA_COOLDOWN_SEC)
                        _GSHEET_CACHE["last_429_epoch"] = time.time()
                        # 다음 append도 쿨다운 이후로 밀어둠
                        cd2 = float(_GSHEET_CACHE.get("quota_cooldown_until_epoch", 0) or 0)
                        _GSHEET_CACHE["next_append_scan_epoch"] = max(float(_GSHEET_CACHE.get("next_append_scan_epoch", 0) or 0), cd2)
                        _GSHEET_CACHE["next_append_high_epoch"] = max(float(_GSHEET_CACHE.get("next_append_high_epoch", 0) or 0), cd2)
                    # SCAN은 오래된 것부터 더 적극적으로 제거(최신 위주)
                    try:
                        with _GSHEET_QUEUE_LOCK:
                            while len(_GSHEET_QUEUE_SCAN) > 900:
                                _GSHEET_QUEUE_SCAN.popleft()
                    except Exception:
                        pass
                    backoff = max(backoff, float(_GSHEET_QUOTA_COOLDOWN_SEC))
            except Exception:
                pass
            # 실패한 batch는 되돌려서(특히 TRADE/EVENT) 나중에 재시도
            try:
                if batch:
                    with _GSHEET_QUEUE_LOCK:
                        if batch_is_high:
                            for r in reversed(batch):
                                _GSHEET_QUEUE_HIGH.appendleft(r)
                        else:
                            for r in reversed(batch):
                                _GSHEET_QUEUE_SCAN.appendleft(r)
            except Exception:
                pass
            try:
                # 관리자 DM에 root cause를 최대한 전달
                with _GSHEET_CACHE_LOCK:
                    msg2 = str(_GSHEET_CACHE.get("last_err", "") or str(e))
                    tb2 = str(_GSHEET_CACHE.get("last_tb", "") or "")
                notify_admin_error(
                    "GSHEET_THREAD",
                    RuntimeError(msg2),
                    context={"batch_len": len(batch), "batch_is_high": bool(batch_is_high)},
                    tb=tb2,
                    min_interval_sec=120.0,
                )
            except Exception:
                notify_admin_error("GSHEET_THREAD", e, min_interval_sec=120.0)
            time.sleep(backoff)
            backoff = float(clamp(backoff * 1.5, 1.0, 90.0))


# tenacity가 있을 때만 사용하는 데코레이터를 늦게 정의(옵션 의존성)
def _retry_wrapper_append_row(fn):  # noqa: D401
    """append_row retry wrapper (tenacity optional)"""
    if retry is None:
        return fn

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1.0, max=6.0))
    def _inner():
        return fn()

    return _inner


# =========================================================
# ✅ 8) 거래소 연결
# =========================================================
@st.cache_resource
def init_exchange():
    try:
        ex = ccxt.bitget(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "timeout": 15000,  # 네트워크 hang 방지
                "options": {"defaultType": "swap"},
            }
        )
        ex.set_sandbox_mode(IS_SANDBOX)
        ex.load_markets()
        return ex
    except Exception:
        return None


def create_exchange_client_uncached() -> Optional[Any]:
    """
    ⚠️ ccxt exchange 객체는 스레드-세이프하지 않을 수 있어,
    Streamlit UI(메인 스레드)와 트레이딩 루프(TG_THREAD)를 분리하기 위한 전용 인스턴스.
    - 실패 시 None 반환(호출부에서 cached exchange로 fallback)
    """
    try:
        ex = ccxt.bitget(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
                "timeout": 15000,
                "options": {"defaultType": "swap"},
            }
        )
        ex.set_sandbox_mode(IS_SANDBOX)
        try:
            ex.load_markets()
        except Exception:
            # 네트워크 문제로 load_markets가 실패해도, 기존 cached exchange의 markets를 복사해
            # thread 전용 인스턴스를 최대한 살려둔다(스레드 정체/공유객체 fallback 방지).
            try:
                ex_cached = globals().get("exchange")
                mk = getattr(ex_cached, "markets", None) if ex_cached is not None else None
                if isinstance(mk, dict) and mk:
                    ex.markets = mk
            except Exception:
                pass
        return ex
    except Exception:
        return None


exchange = init_exchange()
if not exchange:
    st.error("🚨 거래소 연결 실패! API 키/권한/네트워크 확인.")
    st.stop()


# =========================================================
# ✅ 9) Bitget 헬퍼
# =========================================================
def safe_fetch_balance(ex) -> Tuple[float, float]:
    try:
        bal = _ccxt_call_with_timeout(lambda: ex.fetch_balance({"type": "swap"}), CCXT_TIMEOUT_SEC_PRIVATE, where="fetch_balance")
        free = float(bal["USDT"]["free"])
        total = float(bal["USDT"]["total"])
        return free, total
    except FuturesTimeoutError:
        try:
            setattr(ex, "_wonyoti_ccxt_timeout_epoch", time.time())
            setattr(ex, "_wonyoti_ccxt_timeout_where", "fetch_balance")
        except Exception:
            pass
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0


def safe_fetch_positions(ex, symbols: List[str]) -> List[Dict[str, Any]]:
    try:
        def _do():
            try:
                return ex.fetch_positions(symbols)
            except TypeError:
                return ex.fetch_positions(symbols=symbols)

        out = _ccxt_call_with_timeout(_do, CCXT_TIMEOUT_SEC_PRIVATE, where="fetch_positions")
        return out or []
    except FuturesTimeoutError:
        try:
            setattr(ex, "_wonyoti_ccxt_timeout_epoch", time.time())
            setattr(ex, "_wonyoti_ccxt_timeout_where", "fetch_positions")
        except Exception:
            pass
        return []
    except Exception:
        return []


def get_last_price(ex, sym: str) -> Optional[float]:
    try:
        t = _ccxt_call_with_timeout(lambda: ex.fetch_ticker(sym), CCXT_TIMEOUT_SEC_PUBLIC, where="fetch_ticker")
        return float(t["last"])
    except FuturesTimeoutError:
        try:
            setattr(ex, "_wonyoti_ccxt_timeout_epoch", time.time())
            setattr(ex, "_wonyoti_ccxt_timeout_where", "fetch_ticker")
        except Exception:
            pass
        return None
    except Exception:
        return None


def safe_fetch_ohlcv(ex, sym: str, tf: str, limit: int = 220) -> Optional[List[List[Any]]]:
    try:
        tf2 = str(tf or "").strip() or "5m"
        lim = int(limit or 220)
        return _ccxt_call_with_timeout(lambda: ex.fetch_ohlcv(sym, tf2, limit=lim), CCXT_TIMEOUT_SEC_PUBLIC, where="fetch_ohlcv")
    except FuturesTimeoutError:
        try:
            setattr(ex, "_wonyoti_ccxt_timeout_epoch", time.time())
            setattr(ex, "_wonyoti_ccxt_timeout_where", "fetch_ohlcv")
        except Exception:
            pass
        return None
    except Exception:
        return None


def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo


def _timeframe_seconds(tf: str, default_sec: int = 300) -> int:
    """
    "1m"|"3m"|"5m"|"15m"|"1h"|"4h"|"1d" 등을 초로 변환.
    실패 시 default_sec 반환.
    """
    try:
        s = str(tf or "").strip().lower()
        m = re.match(r"^(\d+)\s*([mhdw])$", s)
        if not m:
            return int(default_sec)
        n = int(m.group(1))
        u = m.group(2)
        if n <= 0:
            return int(default_sec)
        if u == "m":
            return int(n * 60)
        if u == "h":
            return int(n * 60 * 60)
        if u == "d":
            return int(n * 24 * 60 * 60)
        if u == "w":
            return int(n * 7 * 24 * 60 * 60)
        return int(default_sec)
    except Exception:
        return int(default_sec)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        # bool은 숫자 취급하면 UI/로그가 헷갈릴 수 있어 별도 처리
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            # NaN/inf 방어
            try:
                if math.isnan(x) or math.isinf(x):
                    return float(default)
            except Exception:
                pass
            return x
        s = str(v).strip()
        if not s:
            return float(default)
        if s.lower() in ["none", "null", "nan"]:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return int(round(_as_float(v, float(default))))
    except Exception:
        return int(default)


def to_precision_qty(ex, sym: str, qty: float) -> float:
    try:
        return float(ex.amount_to_precision(sym, qty))
    except Exception:
        return float(qty)


def set_leverage_safe(ex, sym: str, lev: int) -> None:
    try:
        ex.set_leverage(int(lev), sym)
    except Exception:
        pass


def set_margin_mode_safe(ex, sym: str, mode: str) -> None:
    """
    Bitget 선물 마진 모드 설정(cross/isolated).
    - 거래소/계정/심볼에 따라 실패할 수 있으니, 실패해도 봇이 죽지 않게 safe 처리.
    """
    try:
        m = str(mode or "").strip().lower()
        if m not in ["cross", "isolated"]:
            return
        # ccxt 표준 메서드: set_margin_mode(mode, symbol, params)
        if hasattr(ex, "set_margin_mode"):
            try:
                ex.set_margin_mode(m, sym)  # type: ignore[attr-defined]
            except TypeError:
                ex.set_margin_mode(m, sym, {})  # type: ignore[attr-defined]
    except Exception:
        pass


def market_order_safe(ex, sym: str, side: str, qty: float, params: Optional[Dict[str, Any]] = None) -> bool:
    try:
        params = params or {}
        q = to_precision_qty(ex, sym, float(qty or 0.0))
        if q <= 0:
            return False
        # create_market_order는 거래소/ccxt 버전에 따라 params 지원이 다를 수 있어 표준 create_order로 호출
        _ccxt_call_with_timeout(
            lambda: ex.create_order(sym, "market", side, q, None, params),
            CCXT_TIMEOUT_SEC_PRIVATE,
            where="create_market_order",
            context={"symbol": sym, "side": side, "qty": q, "params": params},
        )
        return True
    except Exception:
        return False


def market_order_safe_ex(ex, sym: str, side: str, qty: float, params: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    try:
        params = params or {}
        q = to_precision_qty(ex, sym, float(qty or 0.0))
        if q <= 0:
            return False, "qty<=0"
        _ccxt_call_with_timeout(
            lambda: ex.create_order(sym, "market", side, q, None, params),
            CCXT_TIMEOUT_SEC_PRIVATE,
            where="create_market_order",
            context={"symbol": sym, "side": side, "qty": q, "params": params},
        )
        return True, ""
    except Exception as e:
        return False, str(e)


def close_position_market(ex, sym: str, pos_side: str, contracts: float) -> bool:
    ok, _err = close_position_market_ex(ex, sym, pos_side, contracts)
    return bool(ok)


def close_position_market_ex(ex, sym: str, pos_side: str, contracts: float) -> Tuple[bool, str]:
    """
    Bitget 선물 포지션 청산을 최대한 '확실하게' 수행.
    - hedge 모드에서는 holdSide가 필요할 수 있음
    - reduceOnly가 없으면 반대 포지션이 열릴 수 있어 우선 reduceOnly로 시도
    """
    try:
        qty = float(contracts or 0.0)
    except Exception:
        qty = 0.0
    if qty <= 0:
        return False, "contracts<=0"

    ps = str(pos_side or "").lower().strip()
    if ps in ["long", "buy"]:
        side = "sell"
        hold_side = "long"
    elif ps in ["short", "sell"]:
        side = "buy"
        hold_side = "short"
    else:
        # fallback: long으로 취급
        side = "sell"
        hold_side = "long"

    # 우선순위: reduceOnly + holdSide → reduceOnly → holdSide → bare
    params_try: List[Dict[str, Any]] = []
    if hold_side:
        params_try.append({"reduceOnly": True, "holdSide": hold_side})
        params_try.append({"reduceOnly": "true", "holdSide": hold_side})
    params_try.append({"reduceOnly": True})
    params_try.append({"reduceOnly": "true"})
    if hold_side:
        params_try.append({"holdSide": hold_side})
    params_try.append({})

    last_err = ""
    for params in params_try:
        ok, err = market_order_safe_ex(ex, sym, side, qty, params=params)
        if ok:
            return True, ""
        last_err = err or last_err
    return False, (last_err or "close_failed")


def position_roi_percent(p: Dict[str, Any]) -> float:
    try:
        if p.get("percentage") is not None:
            return float(p.get("percentage"))
    except Exception:
        pass
    return 0.0


def estimate_roi_from_price(entry_price: float, last_price: float, side: str, leverage: float) -> float:
    """
    가격/진입가/레버로 ROI%(=대략적인 ROE%)를 추정한다.
    - 거래소가 percentage를 제공하지 않거나 지연될 때 보조 지표로 사용.
    """
    try:
        ep = float(entry_price or 0.0)
        lp = float(last_price or 0.0)
        lev = float(leverage or 1.0)
        if ep <= 0 or lp <= 0:
            return 0.0
        if lev <= 0:
            lev = 1.0
        if str(side) == "long":
            pct = (lp - ep) / ep * 100.0
        else:
            pct = (ep - lp) / ep * 100.0
        return float(pct * lev)
    except Exception:
        return 0.0


def position_side_normalize(p: Dict[str, Any]) -> str:
    s = (p.get("side") or p.get("positionSide") or "").lower()
    if s in ["long", "buy"]:
        return "long"
    if s in ["short", "sell"]:
        return "short"
    return "long"


def _pos_leverage(p: Dict[str, Any]) -> float:
    try:
        v = p.get("leverage", None)
        if v is None:
            return 1.0
        return float(v)
    except Exception:
        return 1.0


# =========================================================
# ✅ 9.3) (핵심) 추세 계산 캐시
# =========================================================
_TREND_CACHE: Dict[str, Dict[str, Any]] = {}  # {"BTC/USDT:USDT|1h": {"ts":..., "trend":"하락추세"}}


def compute_ma_trend_from_df(df: pd.DataFrame, fast: int = 7, slow: int = 99) -> str:
    try:
        if df is None or df.empty or len(df) < slow + 5:
            return "중립"
        close = df["close"].astype(float)
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        last_close = float(close.iloc[-1])
        f = float(ma_fast.iloc[-1])
        s = float(ma_slow.iloc[-1])
        if f > s and last_close > s:
            return "상승추세"
        if f < s and last_close < s:
            return "하락추세"
        return "횡보/전환"
    except Exception:
        return "중립"


def get_htf_trend_cached(ex, sym: str, tf: str, fast: int, slow: int, cache_sec: int = 60) -> str:
    key = f"{sym}|{tf}"
    now = time.time()
    if key in _TREND_CACHE:
        if (now - float(_TREND_CACHE[key].get("ts", 0))) < cache_sec:
            return str(_TREND_CACHE[key].get("trend", "중립"))
    try:
        ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=max(220, slow + 50))
        if not ohlcv:
            return "중립"
        try:
            last_bar_ms = int((ohlcv[-1] or [0])[0] or 0)
        except Exception:
            last_bar_ms = 0
        # ✅ "봉 스냅샷" 안정화:
        # - 일부 거래소/환경에서는 마지막 봉 timestamp가 자주 흔들릴 수 있어(초 단위 변동 등),
        #   레짐(confirm2/hysteresis) 전환 토큰이 몇 초마다 바뀌며 스타일이 플립플롭하는 문제가 생길 수 있다.
        # - 타임프레임 경계로 내림(round down)해 동일 봉에서는 토큰이 고정되게 만든다.
        try:
            tf_sec = int(_timeframe_seconds(str(tf or ""), 0) or 0)
            if tf_sec > 0 and last_bar_ms > 0:
                unit = int(tf_sec) * 1000
                last_bar_ms = int((last_bar_ms // unit) * unit)
        except Exception:
            pass
        hdf = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        trend = compute_ma_trend_from_df(hdf, fast=fast, slow=slow)
        _TREND_CACHE[key] = {"ts": now, "trend": trend, "last_bar_ms": last_bar_ms}
        return trend
    except Exception:
        return "중립"


# =========================================================
# ✅ 9.5) SR(지지/저항) 기반 SL/TP 계산
# =========================================================
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def pivot_levels(df: pd.DataFrame, order: int = 6, max_levels: int = 12) -> Tuple[List[float], List[float]]:
    if df is None or df.empty or len(df) < order * 4:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    if argrelextrema is not None:
        hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    else:
        hi_idx, lo_idx = [], []
        for i in range(order, len(df) - order):
            if highs[i] == np.max(highs[i - order:i + order + 1]):
                hi_idx.append(i)
            if lows[i] == np.min(lows[i - order:i + order + 1]):
                lo_idx.append(i)

    resistances = sorted(list(set(np.round(highs[hi_idx], 8))), reverse=True)[:max_levels] if len(highs) else []
    supports = sorted(list(set(np.round(lows[lo_idx], 8))))[:max_levels] if len(lows) else []
    return supports, resistances


def volume_profile_nodes(df: pd.DataFrame, bins: int = 60, top_n: int = 8) -> List[float]:
    """
    간이 매물대(Volume Profile) 노드 추정.
    - OHLCV의 typical price(hlc3) 기준으로 구간(bins)별 거래량을 누적해 상위 top_n 가격대를 반환.
    - 외부 라이브러리 없이 numpy/pandas로만 계산(가벼운 근사치).
    """
    if df is None or df.empty:
        return []
    try:
        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns or "vol" not in df.columns:
            return []
        prices = ((df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0).astype(float)
        vols = df["vol"].astype(float)
        prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
        if prices.empty:
            return []
        # vols도 prices 인덱스에 맞춤
        vols = vols.loc[prices.index].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if int(bins) < 10:
            bins = 10
        if int(bins) > 240:
            bins = 240
        top_n = int(max(1, min(int(top_n), 20)))

        mn = float(prices.min())
        mx = float(prices.max())
        if not (math.isfinite(mn) and math.isfinite(mx)) or mx <= mn:
            return []
        edges = np.linspace(mn, mx, int(bins) + 1)
        idx = np.digitize(prices.values, edges) - 1
        idx = np.clip(idx, 0, int(bins) - 1)
        vol_by_bin = np.bincount(idx, weights=vols.values, minlength=int(bins)).astype(float)
        if len(vol_by_bin) <= 0:
            return []
        top_idx = np.argsort(vol_by_bin)[::-1][:top_n]
        nodes = []
        for i in top_idx:
            try:
                i2 = int(i)
                nodes.append(float((edges[i2] + edges[i2 + 1]) / 2.0))
            except Exception:
                continue
        nodes = [x for x in nodes if math.isfinite(float(x))]
        nodes = sorted(list(set([float(round(x, 8)) for x in nodes])))
        return nodes[:top_n]
    except Exception:
        return []


def sr_stop_take(
    entry_price: float,
    side: str,
    htf_df: pd.DataFrame,
    atr_period: int = 14,
    pivot_order: int = 6,
    buffer_atr_mult: float = 0.25,
    rr_min: float = 1.5,
) -> Optional[Dict[str, Any]]:
    if htf_df is None or htf_df.empty:
        return None

    atr = calc_atr(htf_df, atr_period)
    supports, resistances = pivot_levels(htf_df, pivot_order)
    buf = atr * buffer_atr_mult if atr > 0 else entry_price * 0.0015

    if side == "buy":
        below = [s for s in supports if s < entry_price]
        sl_price = (max(below) - buf) if below else (entry_price - max(buf, entry_price * 0.003))
        risk = entry_price - sl_price
        if risk <= 0:
            return None
        above = [r for r in resistances if r > entry_price]
        tp_candidate = min(above) if above else None
        tp_by_rr = entry_price + risk * rr_min
        tp_price = tp_candidate if (tp_candidate and tp_candidate > tp_by_rr) else tp_by_rr
    else:
        above = [r for r in resistances if r > entry_price]
        sl_price = (min(above) + buf) if above else (entry_price + max(buf, entry_price * 0.003))
        risk = sl_price - entry_price
        if risk <= 0:
            return None
        below = [s for s in supports if s < entry_price]
        tp_candidate = max(below) if below else None
        tp_by_rr = entry_price - risk * rr_min
        tp_price = tp_candidate if (tp_candidate and tp_candidate < tp_by_rr) else tp_by_rr

    return {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "atr": float(atr),
        "supports": supports,
        "resistances": resistances,
    }


def _sr_params_for_style(style: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    스타일별 SR 파라미터를 반환한다.
    - 스윙: 더 큰 매물대/완만한 버퍼(손절/익절이 너무 타이트해지는 문제 완화)
    - 스캘핑: 기존(기본) SR
    """
    st = str(style or "").strip()
    if st == "스윙":
        return {
            "tf": str(cfg.get("sr_timeframe_swing", "1h") or "1h"),
            "lookback": int(cfg.get("sr_lookback_swing", 320) or 320),
            "pivot_order": int(cfg.get("sr_pivot_order_swing", 8) or 8),
            "buffer_atr_mult": float(cfg.get("sr_buffer_atr_mult_swing", 0.45) or 0.45),
            "rr_min": float(cfg.get("sr_rr_min_swing", 2.0) or 2.0),
        }
    return {
        "tf": str(cfg.get("sr_timeframe", "15m") or "15m"),
        "lookback": int(cfg.get("sr_lookback", 220) or 220),
        "pivot_order": int(cfg.get("sr_pivot_order", 6) or 6),
        "buffer_atr_mult": float(cfg.get("sr_buffer_atr_mult", 0.25) or 0.25),
        "rr_min": float(cfg.get("sr_rr_min", 1.5) or 1.5),
    }


def _sr_price_bounds_from_price_pct(entry_price: float, side: str, sl_price_pct: float, tp_price_pct: float) -> Tuple[float, float]:
    """
    ROI%가 아니라 "가격 변동폭%" 기준으로 SL/TP 최소 기준가를 계산한다.
    - buy(롱): sl_bound = entry*(1 - sl_pct), tp_bound = entry*(1 + tp_pct)
    - sell(숏): sl_bound = entry*(1 + sl_pct), tp_bound = entry*(1 - tp_pct)
    """
    px = float(entry_price or 0.0)
    slp = float(max(0.0, sl_price_pct))
    tpp = float(max(0.0, tp_price_pct))
    if str(side or "").lower().strip() == "buy":
        return px * (1.0 - slp / 100.0), px * (1.0 + tpp / 100.0)
    return px * (1.0 + slp / 100.0), px * (1.0 - tpp / 100.0)


def _sr_pick_sl_tp_price(
    *,
    entry_price: float,
    side: str,
    sl_bound: float,
    tp_bound: float,
    supports: List[float],
    resistances: List[float],
    volume_nodes: Optional[List[float]] = None,
    buf: float,
    ai_sl_price: Optional[float] = None,
    ai_tp_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    SR 후보 + AI 후보를 섞어서 최종 SL/TP 가격을 선택한다.
    목표:
    - SR 기반(지지/저항) 원칙 유지
    - 하지만 ROI/리스크 가드레일이 만든 "최소 가격 손절폭"보다 더 타이트한 SR/AI 라인은 채택하지 않음
      (요구사항: 스윙인데 -2~-3%에서 잘리는 문제 완화)
    """
    px = float(entry_price or 0.0)
    s = str(side or "").lower().strip()
    out = {"sl_price": None, "tp_price": None, "sl_source": "", "tp_source": ""}

    def _f(x) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, str) and not x.strip():
                return None
            v = float(x)
            if not math.isfinite(v):
                return None
            return v
        except Exception:
            return None

    ai_sl = _f(ai_sl_price)
    ai_tp = _f(ai_tp_price)
    buf2 = float(buf or 0.0)
    vp = [float(x) for x in (volume_nodes or []) if x is not None and math.isfinite(float(x))]

    if s == "buy":
        # SL candidates: AI or supports-buf (must be below entry and <= sl_bound)
        sl_cands: List[Tuple[float, str]] = []
        if ai_sl is not None and ai_sl < px:
            sl_cands.append((ai_sl, "AI"))
        for lv in (supports or []):
            try:
                sp = float(lv) - buf2
                if sp < px:
                    sl_cands.append((sp, "SR"))
            except Exception:
                continue
        for lv in vp:
            try:
                sp = float(lv)
                if sp < px:
                    sl_cands.append((sp, "VP"))
            except Exception:
                continue
        sl_ok = [(p, src) for (p, src) in sl_cands if p <= float(sl_bound)]
        if sl_ok:
            # 가장 덜 타이트(=entry에 가장 가까운) 기준으로 선택
            p_sel, src_sel = max(sl_ok, key=lambda x: x[0])
            out["sl_price"] = float(p_sel)
            out["sl_source"] = str(src_sel)
        else:
            out["sl_price"] = float(sl_bound)
            out["sl_source"] = "ROI"

        # TP candidates: AI or resistances (must be above entry and >= tp_bound)
        tp_cands: List[Tuple[float, str]] = []
        if ai_tp is not None and ai_tp > px:
            tp_cands.append((ai_tp, "AI"))
        for lv in (resistances or []):
            try:
                rp = float(lv)
                if rp > px:
                    tp_cands.append((rp, "SR"))
            except Exception:
                continue
        for lv in vp:
            try:
                rp = float(lv)
                if rp > px:
                    tp_cands.append((rp, "VP"))
            except Exception:
                continue
        tp_ok = [(p, src) for (p, src) in tp_cands if p >= float(tp_bound)]
        if tp_ok:
            p_sel, src_sel = min(tp_ok, key=lambda x: x[0])
            out["tp_price"] = float(p_sel)
            out["tp_source"] = str(src_sel)
        else:
            out["tp_price"] = float(tp_bound)
            out["tp_source"] = "ROI"

    else:
        # sell(숏)
        # SL candidates: AI or resistances+buf (must be above entry and >= sl_bound)
        sl_cands2: List[Tuple[float, str]] = []
        if ai_sl is not None and ai_sl > px:
            sl_cands2.append((ai_sl, "AI"))
        for lv in (resistances or []):
            try:
                rp = float(lv) + buf2
                if rp > px:
                    sl_cands2.append((rp, "SR"))
            except Exception:
                continue
        for lv in vp:
            try:
                rp = float(lv)
                if rp > px:
                    sl_cands2.append((rp, "VP"))
            except Exception:
                continue
        sl_ok2 = [(p, src) for (p, src) in sl_cands2 if p >= float(sl_bound)]
        if sl_ok2:
            # 가장 덜 타이트(=entry에 가장 가까운) 기준으로 선택
            p_sel, src_sel = min(sl_ok2, key=lambda x: x[0])
            out["sl_price"] = float(p_sel)
            out["sl_source"] = str(src_sel)
        else:
            out["sl_price"] = float(sl_bound)
            out["sl_source"] = "ROI"

        # TP candidates: AI or supports (must be below entry and <= tp_bound)
        tp_cands2: List[Tuple[float, str]] = []
        if ai_tp is not None and ai_tp < px:
            tp_cands2.append((ai_tp, "AI"))
        for lv in (supports or []):
            try:
                sp = float(lv)
                if sp < px:
                    tp_cands2.append((sp, "SR"))
            except Exception:
                continue
        for lv in vp:
            try:
                sp = float(lv)
                if sp < px:
                    tp_cands2.append((sp, "VP"))
            except Exception:
                continue
        tp_ok2 = [(p, src) for (p, src) in tp_cands2 if p <= float(tp_bound)]
        if tp_ok2:
            p_sel, src_sel = max(tp_ok2, key=lambda x: x[0])
            out["tp_price"] = float(p_sel)
            out["tp_source"] = str(src_sel)
        else:
            out["tp_price"] = float(tp_bound)
            out["tp_source"] = "ROI"

    return out


def sr_prices_for_style(
    ex,
    sym: str,
    *,
    entry_price: float,
    side: str,
    style: str,
    cfg: Dict[str, Any],
    sl_price_pct: float,
    tp_price_pct: float,
    ai_sl_price: Optional[float] = None,
    ai_tp_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    최종 SL/TP 가격을 계산(SR 기반 + AI 후보 + ROI 바운드 보정).
    - 네트워크/계산 실패 시에도 None을 반환하고 상위에서 fallback 하도록 한다.
    """
    out = {
        "ok": False,
        "sl_price": None,
        "tp_price": None,
        "sl_source": "",
        "tp_source": "",
        "tf": "",
        "lookback": 0,
        "pivot_order": 0,
        "buffer_atr_mult": 0.0,
        "rr_min": 0.0,
        "atr": 0.0,
        "supports": [],
        "resistances": [],
        "volume_nodes": [],
    }
    if not sym:
        return out
    try:
        params = _sr_params_for_style(style, cfg)
        sr_tf = str(params.get("tf") or "")
        sr_lb = int(params.get("lookback") or 0)
        piv = int(params.get("pivot_order") or 0)
        buf_mul = float(params.get("buffer_atr_mult") or 0.0)
        rr_min = float(params.get("rr_min") or 0.0)

        out.update({"tf": sr_tf, "lookback": sr_lb, "pivot_order": piv, "buffer_atr_mult": buf_mul, "rr_min": rr_min})

        htf = safe_fetch_ohlcv(ex, sym, sr_tf, limit=max(120, sr_lb))
        if not htf:
            return out
        hdf = pd.DataFrame(htf, columns=["time", "open", "high", "low", "close", "vol"])
        try:
            hdf["time"] = pd.to_datetime(hdf["time"], unit="ms")
        except Exception:
            pass

        atr = calc_atr(hdf, int(cfg.get("sr_atr_period", 14)))
        out["atr"] = float(atr)
        supports, resistances = pivot_levels(hdf, order=max(3, piv))
        vp_nodes = volume_profile_nodes(hdf, bins=60, top_n=8)
        out["supports"] = list(supports or [])
        out["resistances"] = list(resistances or [])
        out["volume_nodes"] = list(vp_nodes or [])
        buf = (atr * buf_mul) if atr > 0 else float(entry_price) * 0.0015

        sl_bound, tp_bound = _sr_price_bounds_from_price_pct(float(entry_price), str(side), float(sl_price_pct), float(tp_price_pct))
        picked = _sr_pick_sl_tp_price(
            entry_price=float(entry_price),
            side=str(side),
            sl_bound=float(sl_bound),
            tp_bound=float(tp_bound),
            supports=list(supports or []),
            resistances=list(resistances or []),
            volume_nodes=list(vp_nodes or []),
            buf=float(buf),
            ai_sl_price=ai_sl_price,
            ai_tp_price=ai_tp_price,
        )
        out["sl_price"] = picked.get("sl_price", None)
        out["tp_price"] = picked.get("tp_price", None)
        out["sl_source"] = str(picked.get("sl_source", "") or "")
        out["tp_source"] = str(picked.get("tp_source", "") or "")
        out["ok"] = bool(out["sl_price"] is not None and out["tp_price"] is not None)
        return out
    except Exception:
        return out


def plan_swing_management_levels(
    *,
    entry_price: float,
    side: str,
    tp_price: Optional[float],
    sl_price: Optional[float],
    supports: Optional[List[float]] = None,
    resistances: Optional[List[float]] = None,
    volume_nodes: Optional[List[float]] = None,
) -> Dict[str, Any]:
    out = {
        "partial_tp1_price": None,
        "partial_tp2_price": None,
        "dca_price": None,
    }
    try:
        ep = float(entry_price)
        if ep <= 0:
            return out
        s = str(side or "").lower().strip()
        tp = float(tp_price) if tp_price is not None and math.isfinite(float(tp_price)) else None
        sl = float(sl_price) if sl_price is not None and math.isfinite(float(sl_price)) else None
        sups = [float(x) for x in (supports or []) if x is not None and math.isfinite(float(x))]
        ress = [float(x) for x in (resistances or []) if x is not None and math.isfinite(float(x))]
        vps = [float(x) for x in (volume_nodes or []) if x is not None and math.isfinite(float(x))]

        def _nearest(arr: List[float], x: float) -> Optional[float]:
            if not arr:
                return None
            return min(arr, key=lambda z: abs(float(z) - float(x)))

        if s in ["buy", "long"]:
            if tp is not None and tp > ep:
                up_levels = sorted(set([x for x in (ress + vps) if ep < x < tp]))
                t1 = ep + (tp - ep) * 0.38
                t2 = ep + (tp - ep) * 0.72
                p1 = _nearest(up_levels, t1) if up_levels else t1
                p2 = _nearest(up_levels, t2) if up_levels else t2
                if p1 is not None and p2 is not None:
                    if float(p1) > float(p2):
                        p1, p2 = p2, p1
                    if abs(float(p2) - float(p1)) < (abs(tp - ep) * 0.08):
                        p2 = ep + (tp - ep) * 0.80
                    out["partial_tp1_price"] = float(clamp(float(p1), ep * 1.0001, tp * 0.9990))
                    out["partial_tp2_price"] = float(clamp(float(p2), ep * 1.0002, tp * 0.9995))
            if sl is not None and sl < ep:
                dn_levels = sorted(set([x for x in (sups + vps) if sl < x < ep]))
                td = ep - (ep - sl) * 0.62
                dd = _nearest(dn_levels, td) if dn_levels else td
                if dd is not None:
                    out["dca_price"] = float(clamp(float(dd), sl * 1.001, ep * 0.999))
        else:
            if tp is not None and tp < ep:
                dn_levels = sorted(set([x for x in (sups + vps) if tp < x < ep]), reverse=True)
                t1 = ep - (ep - tp) * 0.38
                t2 = ep - (ep - tp) * 0.72
                p1 = _nearest(dn_levels, t1) if dn_levels else t1
                p2 = _nearest(dn_levels, t2) if dn_levels else t2
                if p1 is not None and p2 is not None:
                    if float(p1) < float(p2):
                        p1, p2 = p2, p1
                    if abs(float(p2) - float(p1)) < (abs(ep - tp) * 0.08):
                        p2 = ep - (ep - tp) * 0.80
                    out["partial_tp1_price"] = float(clamp(float(p1), tp * 1.001, ep * 0.9999))
                    out["partial_tp2_price"] = float(clamp(float(p2), tp * 1.0005, ep * 0.9998))
            if sl is not None and sl > ep:
                up_levels = sorted(set([x for x in (ress + vps) if ep < x < sl]))
                td = ep + (sl - ep) * 0.62
                dd = _nearest(up_levels, td) if up_levels else td
                if dd is not None:
                    out["dca_price"] = float(clamp(float(dd), ep * 1.001, sl * 0.999))
        return out
    except Exception:
        return out


# ✅ SR 레벨 캐시(스캔 과정 표시/안정성/요청 과다 방지)
_SR_CACHE: Dict[str, Dict[str, Any]] = {}


def get_sr_levels_cached(ex, sym: str, tf: str, pivot_order: int = 6, cache_sec: int = 60, limit: int = 220) -> Dict[str, Any]:
    key = f"{sym}|{tf}|{pivot_order}|{limit}"
    now = time.time()
    try:
        if key in _SR_CACHE and (now - float(_SR_CACHE[key].get("ts", 0) or 0)) < float(cache_sec):
            return dict(_SR_CACHE[key])
    except Exception:
        pass
    out = {"ts": now, "tf": tf, "supports": [], "resistances": []}
    try:
        ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=int(limit))
        if not ohlcv:
            raise RuntimeError("ohlcv_empty")
        hdf = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        supports, resistances = pivot_levels(hdf, order=int(pivot_order))
        out["supports"] = supports
        out["resistances"] = resistances
    except Exception:
        pass
    try:
        _SR_CACHE[key] = dict(out)
    except Exception:
        pass
    return out


# =========================================================
# ✅ 10) TradingView 다크모드 차트 (기존 유지)
# =========================================================
def tv_symbol_from_ccxt(sym: str) -> str:
    base = sym.split("/")[0]
    quote = sym.split("/")[1].split(":")[0]
    return f"BITGET:{base}{quote}.P"


def render_tradingview(symbol_ccxt: str, interval: str = "5", height: int = 560) -> None:
    tvsym = tv_symbol_from_ccxt(symbol_ccxt)
    # Streamlit 재실행 시에도 차트가 안 바뀌는 현상 방지: symbol/interval별로 container_id를 바꿔 강제 리렌더
    try:
        cid = re.sub(r"[^A-Za-z0-9_]", "_", f"tv_{tvsym}_{interval}")
    except Exception:
        cid = "tv_chart"
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="{cid}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{tvsym}",
          "interval": "{interval}",
          "timezone": "Asia/Seoul",
          "theme": "dark",
          "style": "1",
          "locale": "kr",
          "toolbar_bg": "#0e1117",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "withdateranges": true,
          "save_image": false,
          "container_id": "{cid}"
        }});
      </script>
    </div>
    """
    components.html(html, height=height)


# =========================================================
# ✅ 11) 지표 계산 (기존 유지)
# =========================================================
def _rolling_linreg_last(series: pd.Series, length: int) -> pd.Series:
    """
    TradingView의 linreg(src, length, 0)와 유사하게,
    각 롤링 윈도우에서 회귀직선의 '마지막 시점 값'을 반환.
    (Squeeze Momentum 계산용)
    """
    try:
        n = int(length)
    except Exception:
        n = 20
    n = max(2, n)
    # x = 0..n-1
    x = np.arange(n, dtype=float)
    x_mean = float((n - 1) / 2.0)
    denom = float(np.sum((x - x_mean) ** 2)) if n >= 2 else 0.0

    def _calc(y: np.ndarray) -> float:
        try:
            yy = np.asarray(y, dtype=float)
            if yy.size != n:
                return float("nan")
            y_mean = float(np.mean(yy))
            if denom <= 0:
                return float(yy[-1])
            num = float(np.sum((x - x_mean) * (yy - y_mean)))
            slope = num / denom
            intercept = y_mean - slope * x_mean
            return float(intercept + slope * float(n - 1))
        except Exception:
            return float("nan")

    try:
        return series.rolling(n).apply(_calc, raw=True)
    except Exception:
        # rolling/apply가 실패하면 NaN series 반환
        try:
            return pd.Series([np.nan] * len(series), index=series.index)
        except Exception:
            return pd.Series(dtype=float)


def _local_extrema_idx(arr: np.ndarray, order: int = 4, mode: str = "max") -> List[int]:
    try:
        a = np.asarray(arr, dtype=float)
        if a.size < (order * 2 + 3):
            return []
        if argrelextrema is not None:
            if str(mode) == "max":
                idx = argrelextrema(a, np.greater_equal, order=order)[0]
            else:
                idx = argrelextrema(a, np.less_equal, order=order)[0]
            return [int(i) for i in idx.tolist()]
        out = []
        for i in range(order, len(a) - order):
            w = a[i - order:i + order + 1]
            if str(mode) == "max":
                if float(a[i]) >= float(np.max(w)):
                    out.append(int(i))
            else:
                if float(a[i]) <= float(np.min(w)):
                    out.append(int(i))
        return out
    except Exception:
        return []


def _pick_last_n_with_min_sep(indices: List[int], n_need: int, min_sep: int) -> List[int]:
    try:
        picks: List[int] = []
        for idx in sorted([int(x) for x in indices], reverse=True):
            if not picks or (int(picks[-1]) - int(idx) >= int(min_sep)):
                picks.append(int(idx))
            if len(picks) >= int(n_need):
                break
        if len(picks) < int(n_need):
            return []
        return list(sorted(picks))
    except Exception:
        return []


def detect_chart_patterns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "detected": [],
        "bullish": [],
        "bearish": [],
        "neutral": [],
        "bias": 0,
        "strength": 0.0,
        "summary": "패턴 없음",
        "score_long": 0.0,
        "score_short": 0.0,
    }
    try:
        if df is None or df.empty or len(df) < 80:
            return out
        lb = int(cfg.get("pattern_lookback", 220) or 220)
        lb = int(max(80, min(800, lb)))
        d = df.tail(lb).copy()
        high = pd.to_numeric(d["high"], errors="coerce").values.astype(float)
        low = pd.to_numeric(d["low"], errors="coerce").values.astype(float)
        close = pd.to_numeric(d["close"], errors="coerce").values.astype(float)
        n = len(d)
        if n < 80:
            return out
        order = int(cfg.get("pattern_pivot_order", 4) or 4)
        order = int(max(2, min(12, order)))
        tol_pct = float(cfg.get("pattern_tolerance_pct", 0.60) or 0.60)
        tol_pct = float(max(0.05, min(3.0, abs(tol_pct)))) / 100.0
        retrace_pct = float(cfg.get("pattern_min_retrace_pct", 0.35) or 0.35)
        retrace_pct = float(max(0.05, min(6.0, abs(retrace_pct)))) / 100.0
        flat_slope_pct = float(cfg.get("pattern_flat_slope_pct", 0.03) or 0.03)
        flat_slope_pct = float(max(0.003, min(1.0, abs(flat_slope_pct))))
        breakout_buf_pct = float(cfg.get("pattern_breakout_buffer_pct", 0.08) or 0.08)
        breakout_buf = float(max(0.0, min(1.0, abs(breakout_buf_pct)))) / 100.0
        min_sep = int(max(2, order))
        last_close = float(close[-1])
        if not math.isfinite(last_close) or last_close <= 0:
            return out

        highs_idx = _local_extrema_idx(high, order=order, mode="max")
        lows_idx = _local_extrema_idx(low, order=order, mode="min")
        if len(highs_idx) < 2 and len(lows_idx) < 2:
            return out

        bull_items: List[Tuple[str, float]] = []
        bear_items: List[Tuple[str, float]] = []
        neutral_items: List[Tuple[str, float]] = []
        seen = set()

        def _add(name: str, side: int, score: float) -> None:
            key = str(name).strip()
            if not key or key in seen:
                return
            seen.add(key)
            s = float(max(0.05, min(2.0, score)))
            out["detected"].append(key)
            if side > 0:
                bull_items.append((key, s))
            elif side < 0:
                bear_items.append((key, s))
            else:
                neutral_items.append((key, s))

        p2 = _pick_last_n_with_min_sep(highs_idx, 2, min_sep)
        if len(p2) == 2:
            i1, i2 = int(p2[0]), int(p2[1])
            if i2 > i1:
                h1, h2 = float(high[i1]), float(high[i2])
                h_avg = max((h1 + h2) * 0.5, 1e-9)
                sim = abs(h1 - h2) / h_avg
                valley = float(np.min(low[i1:i2 + 1]))
                retr = max(0.0, (h_avg - valley) / h_avg)
                if sim <= tol_pct and retr >= retrace_pct:
                    is_break = float(close[-1]) <= float(valley) * (1.0 - breakout_buf)
                    _add("M자형(쌍봉)", -1, 1.35 if is_break else 0.75)
                    _add("쌍봉(Double Top)", -1, 1.25 if is_break else 0.70)

        t2 = _pick_last_n_with_min_sep(lows_idx, 2, min_sep)
        if len(t2) == 2:
            i1, i2 = int(t2[0]), int(t2[1])
            if i2 > i1:
                l1, l2 = float(low[i1]), float(low[i2])
                l_avg = max((l1 + l2) * 0.5, 1e-9)
                sim = abs(l1 - l2) / l_avg
                peak = float(np.max(high[i1:i2 + 1]))
                retr = max(0.0, (peak - l_avg) / max(peak, 1e-9))
                if sim <= tol_pct and retr >= retrace_pct:
                    is_break = float(close[-1]) >= float(peak) * (1.0 + breakout_buf)
                    _add("W자형(쌍바닥)", 1, 1.35 if is_break else 0.75)
                    _add("쌍바닥(Double Bottom)", 1, 1.25 if is_break else 0.70)

        p3 = _pick_last_n_with_min_sep(highs_idx, 3, min_sep)
        if len(p3) == 3:
            i1, i2, i3 = [int(x) for x in p3]
            hvals = [float(high[i1]), float(high[i2]), float(high[i3])]
            havg = max(float(np.mean(hvals)), 1e-9)
            dev = max([abs(x - havg) / havg for x in hvals])
            v1 = float(np.min(low[i1:i2 + 1])) if i2 > i1 else float(low[i2])
            v2 = float(np.min(low[i2:i3 + 1])) if i3 > i2 else float(low[i3])
            vneck = min(v1, v2)
            retr = max(0.0, (havg - vneck) / havg)
            if dev <= tol_pct * 1.2 and retr >= retrace_pct:
                is_break = float(close[-1]) <= float(vneck) * (1.0 - breakout_buf)
                _add("삼중천정(Triple Top)", -1, 1.45 if is_break else 0.80)

        t3 = _pick_last_n_with_min_sep(lows_idx, 3, min_sep)
        if len(t3) == 3:
            i1, i2, i3 = [int(x) for x in t3]
            lvals = [float(low[i1]), float(low[i2]), float(low[i3])]
            lavg = max(float(np.mean(lvals)), 1e-9)
            dev = max([abs(x - lavg) / lavg for x in lvals])
            p1v = float(np.max(high[i1:i2 + 1])) if i2 > i1 else float(high[i2])
            p2v = float(np.max(high[i2:i3 + 1])) if i3 > i2 else float(high[i3])
            pneck = max(p1v, p2v)
            retr = max(0.0, (pneck - lavg) / max(pneck, 1e-9))
            if dev <= tol_pct * 1.2 and retr >= retrace_pct:
                is_break = float(close[-1]) >= float(pneck) * (1.0 + breakout_buf)
                _add("삼중바닥(Triple Bottom)", 1, 1.45 if is_break else 0.80)

        if len(p3) == 3:
            i1, i2, i3 = [int(x) for x in p3]
            s1, hd, s2 = float(high[i1]), float(high[i2]), float(high[i3])
            shoulder_avg = max((s1 + s2) * 0.5, 1e-9)
            shoulder_sim = abs(s1 - s2) / shoulder_avg
            head_up = (hd - shoulder_avg) / shoulder_avg
            if shoulder_sim <= tol_pct * 1.7 and head_up >= max(0.002, tol_pct * 0.8):
                n1 = float(np.min(low[i1:i2 + 1])) if i2 > i1 else float(low[i2])
                n2 = float(np.min(low[i2:i3 + 1])) if i3 > i2 else float(low[i3])
                neck = (n1 + n2) * 0.5
                is_break = float(close[-1]) <= float(neck) * (1.0 - breakout_buf)
                _add("헤드앤숄더", -1, 1.55 if is_break else 0.85)

        if len(t3) == 3:
            i1, i2, i3 = [int(x) for x in t3]
            s1, hd, s2 = float(low[i1]), float(low[i2]), float(low[i3])
            shoulder_avg = max((s1 + s2) * 0.5, 1e-9)
            shoulder_sim = abs(s1 - s2) / shoulder_avg
            head_dn = (shoulder_avg - hd) / shoulder_avg
            if shoulder_sim <= tol_pct * 1.7 and head_dn >= max(0.002, tol_pct * 0.8):
                n1 = float(np.max(high[i1:i2 + 1])) if i2 > i1 else float(high[i2])
                n2 = float(np.max(high[i2:i3 + 1])) if i3 > i2 else float(high[i3])
                neck = (n1 + n2) * 0.5
                is_break = float(close[-1]) >= float(neck) * (1.0 + breakout_buf)
                _add("역헤드앤숄더", 1, 1.55 if is_break else 0.85)

        hi_recent = _pick_last_n_with_min_sep(highs_idx, min(6, len(highs_idx)), min_sep)
        lo_recent = _pick_last_n_with_min_sep(lows_idx, min(6, len(lows_idx)), min_sep)
        if len(hi_recent) >= 3 and len(lo_recent) >= 3:
            xh = np.asarray(hi_recent, dtype=float)
            yh = np.asarray([float(high[i]) for i in hi_recent], dtype=float)
            xl = np.asarray(lo_recent, dtype=float)
            yl = np.asarray([float(low[i]) for i in lo_recent], dtype=float)
            sh, ih = np.polyfit(xh, yh, 1)
            sl, il = np.polyfit(xl, yl, 1)
            sh_pct = float(sh / max(last_close, 1e-9) * 100.0)
            sl_pct = float(sl / max(last_close, 1e-9) * 100.0)
            win = int(max(24, min(72, n // 2)))
            old_h = high[-win:-win // 2] if win // 2 > 0 else high[-win:]
            old_l = low[-win:-win // 2] if win // 2 > 0 else low[-win:]
            new_h = high[-win // 2:] if win // 2 > 0 else high[-win:]
            new_l = low[-win // 2:] if win // 2 > 0 else low[-win:]
            width_old = float(np.max(old_h) - np.min(old_l)) if len(old_h) and len(old_l) else 0.0
            width_new = float(np.max(new_h) - np.min(new_l)) if len(new_h) and len(new_l) else 0.0
            squeeze_ratio = float(width_new / width_old) if width_old > 0 else 1.0
            converging = bool(squeeze_ratio < 0.92)
            top_now = float(sh * float(n - 1) + ih)
            bot_now = float(sl * float(n - 1) + il)
            up_break = float(close[-1]) >= top_now * (1.0 + breakout_buf)
            dn_break = float(close[-1]) <= bot_now * (1.0 - breakout_buf)
            flat = float(flat_slope_pct)

            if (sh_pct < -flat) and (sl_pct > flat) and converging:
                if up_break:
                    _add("대칭삼각수렴 상방이탈", 1, 1.40)
                elif dn_break:
                    _add("대칭삼각수렴 하방이탈", -1, 1.40)
                else:
                    _add("대칭삼각수렴", 0, 0.55)

            if abs(sh_pct) <= flat and (sl_pct > flat) and converging:
                if up_break:
                    _add("상승삼각수렴 상방이탈", 1, 1.45)
                elif dn_break:
                    _add("상승삼각수렴 하방이탈", -1, 1.10)
                else:
                    _add("상승삼각수렴", 1, 0.80)

            if (sh_pct < -flat) and abs(sl_pct) <= flat and converging:
                if dn_break:
                    _add("하락삼각수렴 하방이탈", -1, 1.45)
                elif up_break:
                    _add("하락삼각수렴 상방이탈", 1, 1.10)
                else:
                    _add("하락삼각수렴", -1, 0.80)

            if abs(sh_pct) <= flat and abs(sl_pct) <= flat and (0.78 <= squeeze_ratio <= 1.22):
                rng_hi = float(np.percentile(high[-win:], 92))
                rng_lo = float(np.percentile(low[-win:], 8))
                if float(close[-1]) >= rng_hi * (1.0 + breakout_buf):
                    _add("박스권 상방이탈", 1, 1.20)
                elif float(close[-1]) <= rng_lo * (1.0 - breakout_buf):
                    _add("박스권 하방이탈", -1, 1.20)
                else:
                    _add("박스권 횡보", 0, 0.45)

            if (sh_pct > flat) and (sl_pct > flat) and converging:
                if dn_break:
                    _add("상승쐐기 하방이탈", -1, 1.35)
                else:
                    _add("상승쐐기", -1, 0.70)

            if (sh_pct < -flat) and (sl_pct < -flat) and converging:
                if up_break:
                    _add("하락쐐기 상방이탈", 1, 1.35)
                else:
                    _add("하락쐐기", 1, 0.70)

        bull_score = float(sum(x[1] for x in bull_items))
        bear_score = float(sum(x[1] for x in bear_items))
        diff = bull_score - bear_score
        if diff >= 0.35:
            out["bias"] = 1
        elif diff <= -0.35:
            out["bias"] = -1
        else:
            out["bias"] = 0
        base_score = max(bull_score, bear_score, 0.0)
        strength = float(min(1.0, (base_score / 3.0) + min(0.45, abs(diff) / 3.0)))
        if out["bias"] == 0:
            strength = float(min(strength, 0.60))
        out["strength"] = float(strength)
        out["score_long"] = float(bull_score)
        out["score_short"] = float(bear_score)
        out["bullish"] = [x[0] for x in bull_items[:8]]
        out["bearish"] = [x[0] for x in bear_items[:8]]
        out["neutral"] = [x[0] for x in neutral_items[:8]]
        if not out["detected"]:
            out["summary"] = "패턴 없음"
        else:
            side_txt = "롱 우세" if out["bias"] == 1 else ("숏 우세" if out["bias"] == -1 else "중립")
            out["summary"] = f"{side_txt} | " + ", ".join(out["detected"][:3])
        return out
    except Exception:
        return out


_PATTERN_MTF_CACHE: Dict[str, Dict[str, Any]] = {}
_PATTERN_MTF_LOCK = threading.RLock()


def _pattern_mtf_timeframes(cfg: Dict[str, Any]) -> List[str]:
    try:
        raw = cfg.get("pattern_mtf_timeframes", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"])
        if not isinstance(raw, list):
            raw = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]
    except Exception:
        raw = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]
    out: List[str] = []
    seen = set()
    for tf in raw:
        t = str(tf or "").strip().lower()
        if not t:
            continue
        if _timeframe_seconds(t, 0) <= 0:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if not out:
        out = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]
    return out[:12]


def get_chart_patterns_mtf_cached(ex, sym: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": False,
        "symbol": str(sym or ""),
        "bias": 0,
        "strength": 0.0,
        "score_long": 0.0,
        "score_short": 0.0,
        "summary": "",
        "rows": [],
        "timeframes": [],
    }
    try:
        if not bool(cfg.get("use_chart_patterns", True)):
            return out
        if not bool(cfg.get("pattern_mtf_enable", True)):
            return out
        tfs = _pattern_mtf_timeframes(cfg)
        cache_sec = int(cfg.get("pattern_mtf_cache_sec", 90) or 90)
        cache_sec = int(clamp(cache_sec, 10, 600))
        lb = int(cfg.get("pattern_lookback", 220) or 220)
        cache_key = f"{sym}|{'/'.join(tfs)}|{lb}|{int(cfg.get('pattern_pivot_order',4) or 4)}"
        now_ts = time.time()
        try:
            with _PATTERN_MTF_LOCK:
                ent = _PATTERN_MTF_CACHE.get(cache_key)
                if isinstance(ent, dict):
                    if (now_ts - float(ent.get("ts", 0) or 0.0)) < float(cache_sec):
                        c = ent.get("data", {})
                        if isinstance(c, dict):
                            return dict(c)
        except Exception:
            pass

        rows: List[Dict[str, Any]] = []
        score_long = 0.0
        score_short = 0.0
        w_sum = 0.0
        for tf in tfs:
            try:
                ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=max(120, lb))
                if not ohlcv or len(ohlcv) < 80:
                    continue
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                pat = detect_chart_patterns(df, cfg)
                bias = int(pat.get("bias", 0) or 0)
                strength = float(pat.get("strength", 0.0) or 0.0)
                sec = float(_timeframe_seconds(tf, 300))
                w = float(max(1.0, pow(max(sec, 60.0) / 300.0, 0.35)))
                w_sum += w
                if bias > 0:
                    score_long += float(strength) * w
                elif bias < 0:
                    score_short += float(strength) * w
                rows.append(
                    {
                        "tf": tf,
                        "bias": bias,
                        "strength": round(float(strength), 4),
                        "summary": str(pat.get("summary", "") or ""),
                        "detected": list((pat.get("detected") or [])[:3]),
                        "weight": round(w, 4),
                    }
                )
            except Exception:
                continue

        if not rows:
            out["enabled"] = True
            out["timeframes"] = tfs
            out["summary"] = "MTF 패턴 없음"
            try:
                with _PATTERN_MTF_LOCK:
                    _PATTERN_MTF_CACHE[cache_key] = {"ts": now_ts, "data": dict(out)}
            except Exception:
                pass
            return out

        diff = float(score_long - score_short)
        thr = float(max(0.25, w_sum * 0.06))
        if diff >= thr:
            bias_all = 1
        elif diff <= -thr:
            bias_all = -1
        else:
            bias_all = 0
        top_score = float(max(score_long, score_short, 0.0))
        strength_all = float(clamp(top_score / max(w_sum, 1e-9), 0.0, 1.0))
        if bias_all == 0:
            strength_all = float(min(strength_all, 0.6))
        side_txt = "롱우세" if bias_all == 1 else ("숏우세" if bias_all == -1 else "중립")
        rows_show = sorted(rows, key=lambda r: float(r.get("weight", 1.0)) * float(r.get("strength", 0.0)), reverse=True)[:5]
        tags = []
        for r in rows_show:
            t = str(r.get("summary", "") or "").strip()
            tf = str(r.get("tf", "") or "")
            if t:
                tags.append(f"{tf}:{t}")
        summary = f"MTF {side_txt} | " + (" / ".join(tags) if tags else "패턴 없음")
        out = {
            "enabled": True,
            "symbol": str(sym or ""),
            "bias": int(bias_all),
            "strength": float(strength_all),
            "score_long": float(score_long),
            "score_short": float(score_short),
            "summary": str(summary)[:320],
            "rows": rows,
            "timeframes": tfs,
        }
        try:
            with _PATTERN_MTF_LOCK:
                _PATTERN_MTF_CACHE[cache_key] = {"ts": now_ts, "data": dict(out)}
                if len(_PATTERN_MTF_CACHE) > 3000:
                    items = sorted(_PATTERN_MTF_CACHE.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0) or 0))
                    for k0, _ in items[:600]:
                        _PATTERN_MTF_CACHE.pop(k0, None)
        except Exception:
            pass
        return out
    except Exception:
        return out


def merge_pattern_bias(base_bias: int, base_strength: float, mtf_bias: int, mtf_strength: float, merge_weight: float = 0.6) -> Tuple[int, float]:
    try:
        b0 = int(base_bias or 0)
    except Exception:
        b0 = 0
    try:
        s0 = float(base_strength or 0.0)
    except Exception:
        s0 = 0.0
    try:
        b1 = int(mtf_bias or 0)
    except Exception:
        b1 = 0
    try:
        s1 = float(mtf_strength or 0.0)
    except Exception:
        s1 = 0.0
    w = float(clamp(float(merge_weight), 0.0, 1.0))
    if b1 == 0:
        return b0, float(clamp(s0, 0.0, 1.0))
    if b0 == 0:
        return b1, float(clamp(s1, 0.0, 1.0))
    if b0 == b1:
        s = float(clamp((s0 * (1.0 - w)) + (s1 * w) + 0.10, 0.0, 1.0))
        return b0, s
    score0 = float(s0 * (1.0 - w))
    score1 = float(s1 * w)
    if score1 > score0:
        return b1, float(clamp(score1, 0.0, 1.0))
    return b0, float(clamp(score0, 0.0, 1.0))


def calc_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.Series]]:
    status: Dict[str, Any] = {}
    if df is None or df.empty or len(df) < 120:
        return df, status, None
    # ✅ 지표 라이브러리 호환:
    # - 1순위: ta (기존)
    # - 2순위: pandas_ta (ta 미설치 환경에서 기능 복구)
    use_ta = ta is not None
    use_pta = (not use_ta) and (pta is not None)
    if not use_ta and not use_pta:
        status["_ERROR"] = "ta/pandas_ta 모듈 없음(requirements.txt에 ta 또는 pandas_ta 추가 필요)"
        return df, status, None
    status["_backend"] = "ta" if use_ta else "pandas_ta"
    if use_pta:
        status["_INFO"] = "ta 미설치 → pandas_ta로 지표 계산"

    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))
    bb_period = int(cfg.get("bb_period", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    ma_fast = int(cfg.get("ma_fast", 7))
    ma_slow = int(cfg.get("ma_slow", 99))
    stoch_k = int(cfg.get("stoch_k", 14))
    vol_mul = float(cfg.get("vol_mul", 2.0))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]
    orig_cols = set(df.columns)

    if cfg.get("use_rsi", True):
        try:
            if use_ta:
                df["RSI"] = ta.momentum.rsi(close, window=rsi_period)
            else:
                df["RSI"] = pta.rsi(close, length=rsi_period)
        except Exception as e:
            status["_RSI_ERROR"] = str(e)[:160]

    if cfg.get("use_bb", True):
        try:
            if use_ta:
                bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
                df["BB_upper"] = bb.bollinger_hband()
                df["BB_lower"] = bb.bollinger_lband()
                df["BB_mid"] = bb.bollinger_mavg()
            else:
                bb = pta.bbands(close, length=bb_period, std=bb_std)
                if isinstance(bb, pd.DataFrame) and not bb.empty:
                    col_u = next((c for c in bb.columns if str(c).startswith("BBU_")), "")
                    col_l = next((c for c in bb.columns if str(c).startswith("BBL_")), "")
                    col_m = next((c for c in bb.columns if str(c).startswith("BBM_")), "")
                    if col_u:
                        df["BB_upper"] = bb[col_u]
                    if col_l:
                        df["BB_lower"] = bb[col_l]
                    if col_m:
                        df["BB_mid"] = bb[col_m]
        except Exception as e:
            status["_BB_ERROR"] = str(e)[:160]

    if cfg.get("use_ma", True):
        try:
            if use_ta:
                df["MA_fast"] = ta.trend.sma_indicator(close, window=ma_fast)
                df["MA_slow"] = ta.trend.sma_indicator(close, window=ma_slow)
            else:
                df["MA_fast"] = pta.sma(close, length=ma_fast)
                df["MA_slow"] = pta.sma(close, length=ma_slow)
        except Exception as e:
            status["_MA_ERROR"] = str(e)[:160]

    if cfg.get("use_macd", True):
        try:
            if use_ta:
                macd = ta.trend.MACD(close)
                df["MACD"] = macd.macd()
                df["MACD_signal"] = macd.macd_signal()
            else:
                macd = pta.macd(close)
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    col_macd = next((c for c in macd.columns if str(c).startswith("MACD_") and not str(c).startswith("MACDh_") and not str(c).startswith("MACDs_")), "")
                    col_sig = next((c for c in macd.columns if str(c).startswith("MACDs_")), "")
                    if col_macd:
                        df["MACD"] = macd[col_macd]
                    if col_sig:
                        df["MACD_signal"] = macd[col_sig]
        except Exception as e:
            status["_MACD_ERROR"] = str(e)[:160]

    if cfg.get("use_stoch", True):
        try:
            if use_ta:
                df["STO_K"] = ta.momentum.stoch(high, low, close, window=stoch_k, smooth_window=3)
                df["STO_D"] = ta.momentum.stoch_signal(high, low, close, window=stoch_k, smooth_window=3)
            else:
                stoch = pta.stoch(high, low, close, k=stoch_k, d=3, smooth_k=3)
                if isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    col_k = next((c for c in stoch.columns if str(c).startswith("STOCHk_")), "")
                    col_d = next((c for c in stoch.columns if str(c).startswith("STOCHd_")), "")
                    if col_k:
                        df["STO_K"] = stoch[col_k]
                    if col_d:
                        df["STO_D"] = stoch[col_d]
        except Exception as e:
            status["_STOCH_ERROR"] = str(e)[:160]

    if cfg.get("use_cci", True):
        try:
            if use_ta:
                df["CCI"] = ta.trend.cci(high, low, close, window=20)
            else:
                df["CCI"] = pta.cci(high, low, close, length=20)
        except Exception as e:
            status["_CCI_ERROR"] = str(e)[:160]

    if cfg.get("use_mfi", True):
        try:
            if use_ta:
                df["MFI"] = ta.volume.money_flow_index(high, low, close, vol, window=14)
            else:
                df["MFI"] = pta.mfi(high, low, close, vol, length=14)
        except Exception as e:
            status["_MFI_ERROR"] = str(e)[:160]

    if cfg.get("use_willr", True):
        try:
            if use_ta:
                df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)
            else:
                df["WILLR"] = pta.willr(high, low, close, length=14)
        except Exception as e:
            status["_WILLR_ERROR"] = str(e)[:160]

    if cfg.get("use_adx", True):
        try:
            if use_ta:
                df["ADX"] = ta.trend.adx(high, low, close, window=14)
            else:
                adx = pta.adx(high, low, close, length=14)
                if isinstance(adx, pd.DataFrame) and not adx.empty:
                    col_adx = next((c for c in adx.columns if str(c).startswith("ADX_")), "")
                    if col_adx:
                        df["ADX"] = adx[col_adx]
        except Exception as e:
            status["_ADX_ERROR"] = str(e)[:160]

    if cfg.get("use_vol", True):
        try:
            df["VOL_MA"] = vol.rolling(20).mean()
            df["VOL_SPIKE"] = (df["vol"] > (df["VOL_MA"] * vol_mul)).astype(int)
        except Exception as e:
            status["_VOL_ERROR"] = str(e)[:160]

    # ✅ Squeeze Momentum (LazyBear 유사) - pandas/numpy로 직접 계산(추가 의존성 없음)
    # - 스퀴즈(변동성 압축) 이후 모멘텀 방향/세기를 주요 진입/필터로 사용
    if cfg.get("use_sqz", True):
        try:
            bb_len = int(cfg.get("sqz_bb_length", 20) or 20)
            bb_mult = float(cfg.get("sqz_bb_mult", 2.0) or 2.0)
            kc_len = int(cfg.get("sqz_kc_length", 20) or 20)
            kc_mult = float(cfg.get("sqz_kc_mult", 1.5) or 1.5)
            mom_len = int(cfg.get("sqz_mom_length", kc_len) or kc_len)

            bb_len = max(5, bb_len)
            kc_len = max(5, kc_len)
            mom_len = max(5, mom_len)

            # Bollinger Bands
            bb_mid = close.rolling(bb_len).mean()
            bb_std0 = close.rolling(bb_len).std(ddof=0)
            bb_upper0 = bb_mid + bb_mult * bb_std0
            bb_lower0 = bb_mid - bb_mult * bb_std0

            # Keltner Channels (TR 기반)
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            range_ma = tr.rolling(kc_len).mean()
            kc_mid = close.rolling(kc_len).mean()
            kc_upper0 = kc_mid + kc_mult * range_ma
            kc_lower0 = kc_mid - kc_mult * range_ma

            sqz_on = (bb_lower0 > kc_lower0) & (bb_upper0 < kc_upper0)

            # Momentum source
            hh = high.rolling(mom_len).max()
            ll = low.rolling(mom_len).min()
            m1 = (hh + ll) / 2.0
            m2 = (m1 + close.rolling(mom_len).mean()) / 2.0
            src = close - m2

            sqz_mom = _rolling_linreg_last(src, mom_len)
            df["SQZ_ON"] = sqz_on.astype(int)
            df["SQZ_MOM"] = sqz_mom
            # 가격 대비 % (모멘텀 강도 비교용)
            try:
                df["SQZ_MOM_PCT"] = (sqz_mom.astype(float) / close.astype(float)) * 100.0
            except Exception:
                df["SQZ_MOM_PCT"] = np.nan
        except Exception as e:
            status["_SQZ_ERROR"] = str(e)[:160]

    if pta is not None:
        try:
            df["ATR_ref"] = pta.atr(df["high"], df["low"], df["close"], length=14)
        except Exception:
            pass

    # ✅ 일부 지표가 전부 NaN이면 dropna()가 전체를 비울 수 있으므로, all-NaN 컬럼은 제거
    try:
        new_cols = [c for c in df.columns if c not in orig_cols]
        dropped = []
        for c in new_cols:
            try:
                if df[c].isna().all():
                    df.drop(columns=[c], inplace=True)
                    dropped.append(c)
            except Exception:
                continue
        if dropped:
            status["_DROP_ALL_NAN_COLS"] = dropped[:25]
    except Exception:
        pass

    # dropna는 유지(기존 동작)하되, 전부 비어버리면 close 기준으로라도 복구 시도
    df2 = df.dropna()
    if df2.empty or len(df2) < 5:
        try:
            df2 = df.dropna(subset=["close"])
        except Exception:
            df2 = df2
    if df2.empty or len(df2) < 5:
        return df2, status, None

    last = df2.iloc[-1]
    prev = df2.iloc[-2] if len(df2) >= 2 else last

    used = []

    # RSI
    if cfg.get("use_rsi", True) and "RSI" in df2.columns:
        used.append("RSI")
        rsi_now = float(last.get("RSI", 50))
        if rsi_now < rsi_buy:
            status["RSI"] = f"🟢 과매도({rsi_now:.1f})"
        elif rsi_now > rsi_sell:
            status["RSI"] = f"🔴 과매수({rsi_now:.1f})"
        else:
            status["RSI"] = f"⚪ 중립({rsi_now:.1f})"

    # BB
    if cfg.get("use_bb", True) and all(c in df2.columns for c in ["BB_upper", "BB_lower"]):
        used.append("볼린저밴드")
        if last["close"] > last["BB_upper"]:
            status["BB"] = "🔴 상단 돌파"
        elif last["close"] < last["BB_lower"]:
            status["BB"] = "🟢 하단 이탈"
        else:
            status["BB"] = "⚪ 밴드 내"

    # MA 추세(단기: 현재 timeframe 기준)
    trend = "중립"
    if cfg.get("use_ma", True):
        used.append("이동평균(MA)")
        try:
            if all(c in df2.columns for c in ["MA_fast", "MA_slow"]):
                if last["MA_fast"] > last["MA_slow"] and last["close"] > last["MA_slow"]:
                    trend = "상승추세"
                elif last["MA_fast"] < last["MA_slow"] and last["close"] < last["MA_slow"]:
                    trend = "하락추세"
                else:
                    trend = "횡보/전환"
            else:
                # 최소 기능: close만으로도 추세 산출(표시용)
                trend = compute_ma_trend_from_df(df2, fast=ma_fast, slow=ma_slow)
        except Exception:
            trend = "중립"
        status["추세"] = f"📈 {trend}"

    # MACD
    if cfg.get("use_macd", True) and all(c in df2.columns for c in ["MACD", "MACD_signal"]):
        used.append("MACD")
        status["MACD"] = "📈 상승(골든)" if last["MACD"] > last["MACD_signal"] else "📉 하락(데드)"

    # ADX
    if cfg.get("use_adx", True) and "ADX" in df2.columns:
        used.append("ADX(추세강도)")
        adx = float(last.get("ADX", 0))
        status["ADX"] = "🔥 추세 강함" if adx >= 25 else "💤 추세 약함"

    # Volume
    if cfg.get("use_vol", True) and "VOL_SPIKE" in df2.columns:
        used.append("거래량")
        status["거래량"] = "🔥 거래량 급증" if int(last.get("VOL_SPIKE", 0)) == 1 else "⚪ 보통"

    # Squeeze Momentum (SQZ)
    if cfg.get("use_sqz", True) and "SQZ_MOM_PCT" in df2.columns:
        used.append("스퀴즈모멘텀(SQZ)")
        try:
            mom_pct_now = float(last.get("SQZ_MOM_PCT", 0.0) or 0.0)
        except Exception:
            mom_pct_now = 0.0
        try:
            mom_pct_prev = float(prev.get("SQZ_MOM_PCT", 0.0) or 0.0)
        except Exception:
            mom_pct_prev = mom_pct_now
        slope = float(mom_pct_now - mom_pct_prev)
        try:
            thr = float(cfg.get("sqz_mom_threshold_pct", 0.05) or 0.05)
        except Exception:
            thr = 0.05
        thr = max(0.001, float(abs(thr)))
        bias = 0
        if mom_pct_now >= thr:
            bias = 1
        elif mom_pct_now <= -thr:
            bias = -1
        strength = float(min(1.0, abs(mom_pct_now) / (thr * 4.0))) if thr > 0 else 0.0
        sqz_on_now = False
        try:
            if "SQZ_ON" in df2.columns:
                sqz_on_now = int(last.get("SQZ_ON", 0) or 0) == 1
        except Exception:
            sqz_on_now = False
        arrow = "↗" if slope > 0 else ("↘" if slope < 0 else "→")
        if sqz_on_now:
            status["SQZ"] = f"🟡 압축중 | 모멘텀 {mom_pct_now:+.2f}% {arrow}"
        else:
            if bias == 1:
                status["SQZ"] = f"🟢 상승 모멘텀 {mom_pct_now:+.2f}% {arrow}"
            elif bias == -1:
                status["SQZ"] = f"🔴 하락 모멘텀 {mom_pct_now:+.2f}% {arrow}"
            else:
                status["SQZ"] = f"⚪ 모멘텀 약함 {mom_pct_now:+.2f}% {arrow}"
        status["_sqz_mom_pct"] = float(mom_pct_now)
        status["_sqz_slope"] = float(slope)
        status["_sqz_bias"] = int(bias)
        status["_sqz_strength"] = float(strength)
        status["_sqz_on"] = bool(sqz_on_now)

    if bool(cfg.get("use_chart_patterns", True)):
        try:
            pat = detect_chart_patterns(df2, cfg)
        except Exception:
            pat = {}
        try:
            used.append("차트패턴")
            status["패턴"] = str((pat or {}).get("summary", "패턴 없음"))
            status["_pattern_bias"] = int((pat or {}).get("bias", 0) or 0)
            status["_pattern_strength"] = float((pat or {}).get("strength", 0.0) or 0.0)
            status["_pattern_tags"] = list((pat or {}).get("detected", []) or [])
            status["_pattern_bullish"] = list((pat or {}).get("bullish", []) or [])
            status["_pattern_bearish"] = list((pat or {}).get("bearish", []) or [])
            status["_pattern_neutral"] = list((pat or {}).get("neutral", []) or [])
            status["_pattern_score_long"] = float((pat or {}).get("score_long", 0.0) or 0.0)
            status["_pattern_score_short"] = float((pat or {}).get("score_short", 0.0) or 0.0)
        except Exception:
            pass

    # RSI 해소
    rsi_prev = float(prev.get("RSI", 50)) if (cfg.get("use_rsi", True) and "RSI" in df2.columns) else 50.0
    rsi_now = float(last.get("RSI", 50)) if (cfg.get("use_rsi", True) and "RSI" in df2.columns) else 50.0
    rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
    rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

    adx_now = float(last.get("ADX", 0)) if (cfg.get("use_adx", True) and "ADX" in df2.columns) else 0.0
    pullback_candidate = (trend == "상승추세") and rsi_resolve_long and (adx_now >= 18)

    status["_used_indicators"] = used
    status["_rsi_resolve_long"] = bool(rsi_resolve_long)
    status["_rsi_resolve_short"] = bool(rsi_resolve_short)
    status["_pullback_candidate"] = bool(pullback_candidate)

    return df2, status, last


# =========================================================
# ✅ 11.2) (추가) ML/주력 지표 시그널 (Lorentzian / KNN / Logistic / SQZ / RSI)
# - 외부 라이브러리 없이 numpy/pandas로만 계산
# - 같은 봉에서는 캐시 재사용(비용/지연 감소)
# =========================================================
_ML_SIGNAL_CACHE_LOCK = threading.RLock()
_ML_SIGNAL_CACHE: Dict[str, Dict[str, Any]] = {}


def _ml_sigmoid(z: np.ndarray) -> np.ndarray:
    try:
        z2 = np.clip(z, -20.0, 20.0)
    except Exception:
        z2 = z
    try:
        return 1.0 / (1.0 + np.exp(-z2))
    except Exception:
        # 마지막 방어
        try:
            return 1.0 / (1.0 + np.exp(-np.asarray(z2, dtype=float)))
        except Exception:
            return np.asarray([0.5] * int(len(z))) if hasattr(z, "__len__") else np.asarray([0.5])


def _ml_zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd = np.where(sd <= 1e-9, 1.0, sd)
        return mu.astype(float), sd.astype(float)
    except Exception:
        d = int(X.shape[1]) if getattr(X, "ndim", 0) == 2 else 1
        return np.zeros(d, dtype=float), np.ones(d, dtype=float)


def _ml_zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    try:
        return (X - mu) / sd
    except Exception:
        try:
            return (np.asarray(X, dtype=float) - np.asarray(mu, dtype=float)) / np.asarray(sd, dtype=float)
        except Exception:
            return np.asarray(X, dtype=float)


def _ml_knn_prob(X: np.ndarray, y: np.ndarray, x: np.ndarray, k: int = 15, metric: str = "euclid") -> float:
    """
    y: 0/1
    metric:
      - euclid: L2
      - lorentz: sum(log(1+|diff|))
    """
    try:
        n = int(X.shape[0])
        if n <= 5:
            return 0.5
        kk = int(max(3, min(int(k), n)))
        diff = X - x.reshape(1, -1)
        if str(metric) == "lorentz":
            dist = np.sum(np.log1p(np.abs(diff)), axis=1)
        else:
            dist = np.sqrt(np.sum(diff * diff, axis=1))
        # k nearest
        idx = np.argpartition(dist, kk - 1)[:kk]
        yy = y[idx]
        # prob long = mean(1)
        p = float(np.mean(yy))
        if not math.isfinite(p):
            return 0.5
        return float(clamp(p, 0.0, 1.0))
    except Exception:
        return 0.5


def _ml_logit_prob(X: np.ndarray, y: np.ndarray, x: np.ndarray, steps: int = 120, lr: float = 0.15, l2: float = 0.01) -> float:
    """
    간이 Logistic Regression:
    - 경량 GD(steps 제한)
    - L2 정규화(과적합/발산 방지)
    """
    try:
        n, d = int(X.shape[0]), int(X.shape[1])
        if n <= 30 or d <= 0:
            return 0.5
        st = int(max(20, min(int(steps), 400)))
        lr0 = float(clamp(float(lr), 0.01, 0.6))
        l2_0 = float(clamp(float(l2), 0.0, 0.2))

        Xb = np.concatenate([np.ones((n, 1), dtype=float), X.astype(float)], axis=1)
        w = np.zeros((d + 1,), dtype=float)

        yy = y.astype(float)
        for _ in range(st):
            z = Xb @ w
            p = _ml_sigmoid(z)
            grad = (Xb.T @ (p - yy)) / float(n)
            # L2 (bias 제외)
            grad[1:] = grad[1:] + l2_0 * w[1:]
            w = w - lr0 * grad

        xb = np.concatenate([np.ones((1,), dtype=float), x.astype(float)], axis=0)
        p1 = float(_ml_sigmoid(np.asarray([float(xb @ w)])).reshape(-1)[0])
        if not math.isfinite(p1):
            return 0.5
        return float(clamp(p1, 0.0, 1.0))
    except Exception:
        return 0.5


def ml_signals_and_convergence(
    df: pd.DataFrame,
    status: Dict[str, Any],
    cfg: Dict[str, Any],
    cache_key: str = "",
) -> Dict[str, Any]:
    """
    반환:
      - rsi_sig, sqz_sig, pattern_sig, knn_sig, lor_sig, logit_sig (-1/0/1)
      - knn_prob, lor_prob, logit_prob (0..1)
      - votes_long, votes_short, votes_max, dir("buy"/"sell"/"hold"), detail
    """
    try:
        if bool(cfg.get("ml_cache_enable", True)) and cache_key:
            with _ML_SIGNAL_CACHE_LOCK:
                cached = _ML_SIGNAL_CACHE.get(cache_key)
                if isinstance(cached, dict) and cached:
                    return dict(cached)
    except Exception:
        pass

    out: Dict[str, Any] = {
        "rsi_sig": 0,
        "sqz_sig": 0,
        "pattern_sig": 0,
        "knn_sig": 0,
        "lor_sig": 0,
        "logit_sig": 0,
        "knn_prob": 0.5,
        "lor_prob": 0.5,
        "logit_prob": 0.5,
        "votes_long": 0,
        "votes_short": 0,
        "votes_max": 0,
        "dir": "hold",
        "detail": "",
    }

    try:
        if (df is None) or df.empty or len(df) < 80:
            return out
    except Exception:
        return out

    # RSI sig
    rsi_now = None
    try:
        if "RSI" in df.columns:
            v = df["RSI"].iloc[-1]
            rsi_now = float(v) if (v is not None and pd.notna(v)) else None
    except Exception:
        rsi_now = None
    try:
        band = float(cfg.get("ml_rsi_neutral_band", 3.0) or 3.0)
    except Exception:
        band = 3.0
    band = float(max(0.0, abs(band)))
    if rsi_now is not None:
        if float(rsi_now) >= 50.0 + band:
            out["rsi_sig"] = 1
        elif float(rsi_now) <= 50.0 - band:
            out["rsi_sig"] = -1
        else:
            out["rsi_sig"] = 0

    # SQZ sig
    try:
        bias = int(status.get("_sqz_bias", 0) or 0)
        if bias in [-1, 0, 1]:
            out["sqz_sig"] = int(bias)
        else:
            out["sqz_sig"] = 0
    except Exception:
        out["sqz_sig"] = 0

    try:
        if bool(cfg.get("use_chart_patterns", True)):
            p_bias = int(status.get("_pattern_bias", 0) or 0)
            p_strength = float(status.get("_pattern_strength", 0.0) or 0.0)
            if p_bias in [-1, 1] and p_strength >= 0.20:
                out["pattern_sig"] = int(p_bias)
    except Exception:
        out["pattern_sig"] = 0

    if not bool(cfg.get("ml_enable", True)):
        # convergence는 RSI/SQZ만으로도 계산 가능
        pass
    else:
        try:
            close = pd.to_numeric(df["close"], errors="coerce")
            vol = pd.to_numeric(df["vol"], errors="coerce") if "vol" in df.columns else pd.Series([np.nan] * len(df), index=df.index)

            ma_p = int(cfg.get("ml_feature_ma_period", 20) or 20)
            ma_p = max(5, ma_p)
            vma_p = int(cfg.get("ml_feature_vol_ma_period", 20) or 20)
            vma_p = max(5, vma_p)

            ret1 = close.pct_change(1) * 100.0
            ret3 = close.pct_change(3) * 100.0
            ma = close.rolling(ma_p).mean()
            disp = (close - ma) / ma * 100.0
            vma = vol.rolling(vma_p).mean()
            vol_ratio = vol / vma

            rsi = pd.to_numeric(df["RSI"], errors="coerce") if "RSI" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
            rsi_norm = (rsi - 50.0) / 50.0
            sqz = pd.to_numeric(df["SQZ_MOM_PCT"], errors="coerce") if "SQZ_MOM_PCT" in df.columns else pd.Series([0.0] * len(df), index=df.index)

            feat = pd.DataFrame(
                {
                    "ret1": ret1,
                    "ret3": ret3,
                    "rsi_norm": rsi_norm,
                    "sqz": sqz,
                    "vol_ratio": vol_ratio - 1.0,
                    "disp": disp,
                }
            )
            feat = feat.replace([np.inf, -np.inf], np.nan)

            # 현재 피처(x_cur)
            if feat.iloc[-1].isna().any():
                return out
            x_cur = feat.iloc[-1].astype(float).values

            # 라벨(y): 미래 h봉 후 상승이면 1, 아니면 0
            h = int(cfg.get("ml_horizon", 1) or 1)
            h = max(1, min(h, 10))
            y = (close.shift(-h) > close).astype(float)

            train_df = feat.copy()
            train_df["y"] = y
            train_df = train_df.dropna()
            if train_df.empty:
                return out

            lookback = int(cfg.get("ml_lookback", 220) or 220)
            lookback = max(60, min(lookback, 1200))
            if len(train_df) > lookback:
                train_df = train_df.iloc[-lookback:]

            min_n = int(cfg.get("ml_min_train_samples", 80) or 80)
            if len(train_df) < min_n:
                return out

            y_train = train_df["y"].astype(float).values
            X_train_raw = train_df.drop(columns=["y"]).astype(float).values

            # 스케일링
            mu, sd = _ml_zscore_fit(X_train_raw)
            X_train = _ml_zscore_apply(X_train_raw, mu, sd)
            x1 = _ml_zscore_apply(x_cur.reshape(1, -1), mu, sd).reshape(-1)

            # KNN(Euclid)
            k_knn = int(cfg.get("ml_knn_k", 15) or 15)
            p_knn = _ml_knn_prob(X_train, y_train, x1, k=k_knn, metric="euclid")
            out["knn_prob"] = float(p_knn)
            pl = float(cfg.get("ml_knn_prob_long", 0.56) or 0.56)
            ps = float(cfg.get("ml_knn_prob_short", 0.44) or 0.44)
            if p_knn >= pl:
                out["knn_sig"] = 1
            elif p_knn <= ps:
                out["knn_sig"] = -1

            # Lorentzian KNN
            k_lor = int(cfg.get("ml_lor_k", 15) or 15)
            p_lor = _ml_knn_prob(X_train, y_train, x1, k=k_lor, metric="lorentz")
            out["lor_prob"] = float(p_lor)
            pl2 = float(cfg.get("ml_lor_prob_long", 0.56) or 0.56)
            ps2 = float(cfg.get("ml_lor_prob_short", 0.44) or 0.44)
            if p_lor >= pl2:
                out["lor_sig"] = 1
            elif p_lor <= ps2:
                out["lor_sig"] = -1

            # Logistic regression
            st = int(cfg.get("ml_logit_steps", 120) or 120)
            lr0 = float(cfg.get("ml_logit_lr", 0.15) or 0.15)
            l2 = float(cfg.get("ml_logit_l2", 0.01) or 0.01)
            p_log = _ml_logit_prob(X_train, y_train, x1, steps=st, lr=lr0, l2=l2)
            out["logit_prob"] = float(p_log)
            pl3 = float(cfg.get("ml_logit_prob_long", 0.56) or 0.56)
            ps3 = float(cfg.get("ml_logit_prob_short", 0.44) or 0.44)
            if p_log >= pl3:
                out["logit_sig"] = 1
            elif p_log <= ps3:
                out["logit_sig"] = -1
        except Exception:
            pass

    # votes
    sigs = {
        "RSI": int(out.get("rsi_sig", 0) or 0),
        "SQZ": int(out.get("sqz_sig", 0) or 0),
        "KNN": int(out.get("knn_sig", 0) or 0),
        "LOR": int(out.get("lor_sig", 0) or 0),
        "LOGIT": int(out.get("logit_sig", 0) or 0),
    }
    if bool(cfg.get("use_chart_patterns", True)):
        sigs["PATTERN"] = int(out.get("pattern_sig", 0) or 0)
    v_long = sum(1 for v in sigs.values() if int(v) == 1)
    v_short = sum(1 for v in sigs.values() if int(v) == -1)
    out["votes_long"] = int(v_long)
    out["votes_short"] = int(v_short)
    out["votes_max"] = int(max(v_long, v_short))
    try:
        need = int(cfg.get("entry_convergence_min_votes", 3) or 3)
    except Exception:
        need = 3
    if v_long >= need and v_long > v_short:
        out["dir"] = "buy"
    elif v_short >= need and v_short > v_long:
        out["dir"] = "sell"
    else:
        out["dir"] = "hold"
    # detail
    try:
        def _sg(x: int) -> str:
            return "롱" if x == 1 else ("숏" if x == -1 else "중립")

        out["detail"] = (
            f"RSI:{_sg(int(out.get('rsi_sig',0)))} | SQZ:{_sg(int(out.get('sqz_sig',0)))} | "
            f"PATTERN:{_sg(int(out.get('pattern_sig',0)))} | "
            f"KNN:{_sg(int(out.get('knn_sig',0)))}({float(out.get('knn_prob',0.5)):.2f}) | "
            f"LOR:{_sg(int(out.get('lor_sig',0)))}({float(out.get('lor_prob',0.5)):.2f}) | "
            f"LOGIT:{_sg(int(out.get('logit_sig',0)))}({float(out.get('logit_prob',0.5)):.2f}) | "
            f"VOTE L{v_long}/S{v_short}"
        )[:240]
    except Exception:
        out["detail"] = ""

    # cache store + prune
    try:
        if bool(cfg.get("ml_cache_enable", True)) and cache_key:
            with _ML_SIGNAL_CACHE_LOCK:
                _ML_SIGNAL_CACHE[cache_key] = dict(out)
                if len(_ML_SIGNAL_CACHE) > 400:
                    # 오래된 것 절반 정리(순서 보장 X → key 정렬로 대충)
                    for k in list(_ML_SIGNAL_CACHE.keys())[:200]:
                        _ML_SIGNAL_CACHE.pop(k, None)
    except Exception:
        pass

    return out


# =========================================================
# ✅ 12) 외부 시황 통합(거시/심리/레짐/뉴스) - 캐시/한글화/안정성 강화
# =========================================================
_ext_cache = TTLCache(maxsize=12, ttl=60) if TTLCache else None
_translate_cache = TTLCache(maxsize=256, ttl=60 * 60 * 24) if TTLCache else None  # 24h


def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT_SEC, attempts: int = 3):
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    try:
        attempts_i = max(1, int(attempts or 1))
    except Exception:
        attempts_i = 1

    if retry is None or stop_after_attempt is None or wait_exponential_jitter is None:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    @retry(stop=stop_after_attempt(attempts_i), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def _do():
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r.json()

    try:
        return _do()
    except Exception:
        return None


def _translate_ko_rule(text: str) -> str:
    """AI/번역기 없이도 최소한 읽히게 만드는 룰 기반 '한글화 보정'."""
    t = str(text or "")
    rep = {
        "Extreme Fear": "극공포",
        "Fear": "공포",
        "Neutral": "중립",
        "Greed": "탐욕",
        "Extreme Greed": "극탐욕",
        "High": "매우 중요",
        "Medium": "중요",
        "Low": "낮음",
        "United States": "미국",
        "Euro Zone": "유로존",
        "Japan": "일본",
        "China": "중국",
        "United Kingdom": "영국",
        "Germany": "독일",
        "France": "프랑스",
        "Korea": "한국",
        "Bitcoin": "비트코인",
        "BTC": "BTC",
        "ETF": "ETF",
        "Inflation": "인플레이션",
        "Interest Rate": "금리",
        "Rate Decision": "금리결정",
        "CPI": "CPI(소비자물가)",
        "PPI": "PPI(생산자물가)",
        "FOMC": "FOMC(연준회의)",
        "Nonfarm Payrolls": "NFP(비농업 고용)",
        "Unemployment Rate": "실업률",
        "Retail Sales": "소매판매",
        "GDP": "GDP",
        "PMI": "PMI",
        "Core": "근원",
        "YoY": "전년대비",
        "MoM": "전월대비",
    }
    for k, v in rep.items():
        t = t.replace(k, v)
    return t


def translate_to_korean(text: str, cfg: Dict[str, Any], use_cache: bool = True) -> str:
    """
    우선순위:
    1) deep-translator(선택) -> 2) OpenAI(설정 ON + 키 존재) -> 3) 룰 기반 보정 -> 4) 원문
    """
    s = str(text or "").strip()
    if not s:
        return ""
    if use_cache and _translate_cache is not None:
        try:
            k = f"ko:{hash(s)}"
            if k in _translate_cache:
                return _translate_cache[k]
        except Exception:
            pass

    out = s

    # deep-translator (네트워크 hang 방지: hard-timeout)
    if GoogleTranslator is not None:
        try:
            def _do_trans():
                return GoogleTranslator(source="auto", target="ko").translate(s)

            out = _call_with_timeout(_do_trans, 4)
        except Exception:
            out = s

    # OpenAI 번역(옵션)
    if out == s and cfg.get("external_ai_translate_enable", False):
        client = get_openai_client(cfg)
        if client is not None:
            try:
                def _do():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "너는 번역기다. 입력 문장을 자연스러운 한국어로만 번역해라. 다른 말 금지."},
                            {"role": "user", "content": s},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    )

                resp = _call_with_timeout(_do, OPENAI_TIMEOUT_SEC)
                out = (resp.choices[0].message.content or "").strip()
                if not out:
                    out = s
            except Exception:
                out = s

    # 룰 기반 보정
    if cfg.get("external_koreanize_enable", True):
        out = _translate_ko_rule(out)

    if use_cache and _translate_cache is not None:
        try:
            _translate_cache[f"ko:{hash(s)}"] = out
        except Exception:
            pass
    return out


def fetch_fear_greed(cfg: Dict[str, Any]):
    # 외부시황은 "참고"용 → 과도한 재시도는 봇 루프를 멈추게 할 수 있어 attempts를 낮춘다.
    data = _http_get_json("https://api.alternative.me/fng/?limit=1&format=json", timeout=6, attempts=2)
    if not data or "data" not in data or not data["data"]:
        return None
    d0 = data["data"][0]
    try:
        v = int(d0.get("value", 0))
        cls = str(d0.get("value_classification", ""))
        cls_ko = translate_to_korean(cls, cfg)
        # 이모티콘
        emo = "😱" if v <= 25 else ("🙂" if v <= 55 else ("😋" if v <= 75 else "🤑"))
        return {"value": v, "classification": cls_ko, "emoji": emo, "timestamp": str(d0.get("timestamp", ""))}
    except Exception:
        return None


def fetch_coingecko_global():
    data = _http_get_json("https://api.coingecko.com/api/v3/global", timeout=8, attempts=2)
    if not data or "data" not in data:
        return None
    g = data["data"]
    mcp = g.get("market_cap_percentage", {}) or {}
    try:
        return {
            "btc_dominance": float(mcp.get("btc", 0.0)),
            "eth_dominance": float(mcp.get("eth", 0.0)),
            "total_mcap_usd": float((g.get("total_market_cap", {}) or {}).get("usd", 0.0)),
            "mcap_change_24h_pct": float(g.get("market_cap_change_percentage_24h_usd", 0.0)),
        }
    except Exception:
        return None


def _country_to_ko(country: str, cfg: Dict[str, Any]) -> str:
    c = str(country or "").strip()
    m = {
        "USD": "미국",
        "US": "미국",
        "EUR": "유로존",
        "EU": "유로존",
        "JPY": "일본",
        "JP": "일본",
        "CNY": "중국",
        "CN": "중국",
        "GBP": "영국",
        "UK": "영국",
        "CHF": "스위스",
        "CAD": "캐나다",
        "AUD": "호주",
        "NZD": "뉴질랜드",
        "KRW": "한국",
        "KR": "한국",
    }
    return m.get(c, translate_to_korean(c, cfg))


def fetch_upcoming_high_impact_events(cfg: Dict[str, Any], within_minutes: int = 30, limit: int = 80):
    data = _http_get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=8, attempts=2)
    if not isinstance(data, list):
        return []
    now = now_kst()
    out = []
    for x in data[:limit]:
        try:
            if str(x.get("impact", "")) != "High":
                continue
            dt_str = str(x.get("date", ""))
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo:
                    dt = dt.astimezone(KST)
                else:
                    dt = dt.replace(tzinfo=KST)
            except Exception:
                continue

            diff_min = (dt - now).total_seconds() / 60.0
            if 0 <= diff_min <= within_minutes:
                title = str(x.get("title", ""))
                title_ko = translate_to_korean(title, cfg)
                country_ko = _country_to_ko(str(x.get("country", "")), cfg)
                out.append(
                    {
                        "time_kst": dt.strftime("%m-%d %H:%M"),
                        "title": f"🚨 {title_ko}",
                        "country": country_ko,
                        "impact": "매우 중요",
                    }
                )
        except Exception:
            continue
    return out


def fetch_news_headlines_rss(cfg: Dict[str, Any], max_items: int = 12):
    if feedparser is None:
        return []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    items = []
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    for url in feeds:
        try:
            # feedparser.parse(url)은 내부 네트워크 fetch가 hang될 수 있음 → requests로 timeout 보장 후 parse
            r = requests.get(url, timeout=8, headers=headers)
            r.raise_for_status()
            d = feedparser.parse(r.content)
            for e in (d.entries or [])[: max_items * 2]:
                title = str(getattr(e, "title", "")).strip()
                if not title:
                    continue
                items.append(title)
        except Exception:
            continue
    uniq, seen = [], set()
    for t in items:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    uniq = uniq[:max_items]
    # 한글화(옵션)
    if cfg.get("external_koreanize_enable", True):
        # ✅ 번역기는 느릴 수 있음(특히 deep-translator) → 시간 예산 초과 시 룰 기반만 적용
        try:
            raw = cfg.get("news_translate_budget_sec", 10)
            budget = float(10.0 if raw is None else raw)
        except Exception:
            budget = 10.0
        budget = max(0.0, budget)
        if budget <= 0:
            uniq = [_translate_ko_rule(t) for t in uniq]
        else:
            t0 = time.time()
            out_titles = []
            for t in uniq:
                if (time.time() - t0) > budget:
                    out_titles.append(_translate_ko_rule(t))
                    continue
                out_titles.append(translate_to_korean(t, cfg))
            uniq = out_titles
    return uniq


def fetch_daily_btc_brief(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    매일 아침: BTC 관련 경제뉴스 5개 선정
    - feedparser 없으면 빈 값 반환
    - OpenAI 키 있으면 요약/한글화 강화(옵션)
    """
    date_str = today_kst_str()
    if _ext_cache is not None and f"daily_btc_brief:{date_str}" in _ext_cache:
        return _ext_cache[f"daily_btc_brief:{date_str}"]

    out = {"date": date_str, "items": [], "asof_kst": now_kst_str(), "source": "rss"}
    if feedparser is None:
        out["source"] = "feedparser_missing"
        return out

    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    keywords = [
        "bitcoin",
        "btc",
        "etf",
        "fed",
        "fomc",
        "cpi",
        "ppi",
        "rate",
        "inflation",
        "macro",
        "economy",
        "jobs",
        "nfp",
        "powell",
        "interest",
        "treasury",
        "yield",
    ]

    raw_titles: List[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (WonyotiAgent/1.0)"}
    for url in feeds:
        try:
            r = requests.get(url, timeout=8, headers=headers)
            r.raise_for_status()
            d = feedparser.parse(r.content)
            for e in (d.entries or [])[:60]:
                title = str(getattr(e, "title", "")).strip()
                if not title:
                    continue
                low = title.lower()
                if any(k in low for k in keywords):
                    raw_titles.append(title)
        except Exception:
            continue

    # 중복 제거 + 상위 N개
    uniq: List[str] = []
    seen = set()
    for t in raw_titles:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
        if len(uniq) >= int(cfg.get("daily_btc_brief_max_items", 5)):
            break

    if not uniq:
        out["items"] = []
        if _ext_cache is not None:
            _ext_cache[f"daily_btc_brief:{date_str}"] = out
        return out

    # 한글화/요약
    items_ko = [translate_to_korean(t, cfg) for t in uniq]

    if cfg.get("daily_btc_brief_ai_summarize", True):
        client = get_openai_client(cfg)
        if client is not None:
            try:
                # 예측 대상(거래 대상 코인)도 같이 전달
                payload = {"date": date_str, "titles": items_ko, "targets": TARGET_COINS}

                def _do():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "너는 암호화폐 트레이딩용 아침 브리핑 에디터다.\n"
                                    "입력된 제목 리스트에서 '비트코인/거시경제' 관점으로 중요한 5개를 골라,"
                                    "각 항목을 아주 짧고 쉬운 한국어 한줄로 정리해라.\n"
                                    "추가로, targets(코인 리스트)에 대해 오늘 하루의 방향성(롱/숏/관망)을 아주 보수적으로 '예측'해라.\n"
                                    "- 예측은 참고용이며, 과장 금지.\n"
                                    "출력은 반드시 JSON만.\n"
                                    '형식: {"items":[{"emoji":"📰","title":"...","note":"한줄 요약"}], "bias":"중립|보수|공격", "risk":"낮음|보통|높음", "outlook":[{"symbol":"BTC/USDT:USDT","dir":"롱|숏|관망","confidence":0-100,"note":"아주 짧게"}]}'
                                ),
                            },
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                        max_tokens=700,
                    )

                resp = _call_with_timeout(_do, OPENAI_TIMEOUT_SEC)
                jj = json.loads(resp.choices[0].message.content)
                items = jj.get("items", [])
                if isinstance(items, list) and items:
                    out["items"] = items[: int(cfg.get("daily_btc_brief_max_items", 5))]
                    out["bias"] = str(jj.get("bias", "중립"))
                    out["risk"] = str(jj.get("risk", "보통"))
                    # 코인/방향 예측(선택)
                    try:
                        outlk = jj.get("outlook", []) or jj.get("signals", [])
                        if isinstance(outlk, list):
                            out["outlook"] = outlk[: min(10, len(outlk))]
                    except Exception:
                        pass
                    out["source"] = "openai"
                else:
                    out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]
            except Exception:
                out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]
    else:
        out["items"] = [{"emoji": "📰", "title": t, "note": ""} for t in items_ko]

    if _ext_cache is not None:
        _ext_cache[f"daily_btc_brief:{date_str}"] = out
    return out


def build_external_context(cfg: Dict[str, Any], rt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not cfg.get("use_external_context", True):
        return {"enabled": False}

    ttl = int(cfg.get("external_refresh_sec", 60))
    cache_key = f"ext:{today_kst_str()}:{ttl}"
    if _ext_cache is not None and cache_key in _ext_cache:
        return _ext_cache[cache_key]

    blackout = int(cfg.get("macro_blackout_minutes", 30))
    high_events = fetch_upcoming_high_impact_events(cfg, within_minutes=blackout)

    fg = fetch_fear_greed(cfg)
    cg = fetch_coingecko_global()

    headlines: List[str] = []
    if cfg.get("news_enable", True):
        headlines = fetch_news_headlines_rss(cfg, max_items=int(cfg.get("news_max_headlines", 12)))

    daily_brief = {}
    try:
        # 런타임에 저장된 브리핑이 있으면 우선 사용, 없으면 즉시 가져오진 않음(아침 스케줄에서 처리)
        if rt and isinstance(rt.get("daily_btc_brief"), dict) and rt["daily_btc_brief"].get("date") == today_kst_str():
            daily_brief = rt["daily_btc_brief"]
    except Exception:
        daily_brief = {}

    ext = {
        "enabled": True,
        "blackout_minutes": blackout,
        "high_impact_events_soon": high_events,
        "fear_greed": fg,
        "global": cg,
        "headlines": headlines,
        "daily_btc_brief": daily_brief,
        "asof_kst": now_kst_str(),
    }

    if _ext_cache is not None:
        _ext_cache[cache_key] = ext
    return ext


def external_risk_multiplier(ext: Dict[str, Any], cfg: Dict[str, Any], include_fng: bool = True) -> float:
    """
    외부 시황이 위험하면 신규 진입을 "감산/보수"로 조정(완전 금지 X).
    """
    if not cfg.get("entry_risk_reduce_enable", True):
        return 1.0
    mul = 1.0
    try:
        evs = (ext or {}).get("high_impact_events_soon") or []
        if evs:
            mul *= float(cfg.get("entry_risk_reduce_factor", 0.65))
    except Exception:
        pass
    if bool(include_fng):
        try:
            fg = (ext or {}).get("fear_greed") or {}
            v = int(fg.get("value", -1)) if fg else -1
            if 0 <= v <= 15:  # 극공포: 진입 크기 0 (진입 금지 신호)
                mul *= 0.0
            elif 0 <= v <= 25:  # 공포
                mul *= 0.70
            elif v >= 80:  # 극탐욕: 포지션 크기 0.7배로 제한
                mul *= 0.70
            elif v >= 75:  # 탐욕
                mul *= 0.80
        except Exception:
            pass
    try:
        brief = (ext or {}).get("daily_btc_brief") or {}
        risk = str(brief.get("risk", "")).strip()
        if risk == "높음":
            mul *= 0.8
    except Exception:
        pass
    return float(clamp(mul, 0.2, 1.0))


def swing_entry_pct_total_by_fng(ext: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
    """
    스윙 진입금(총자산 %)을 공포/탐욕(FNG) 지수로 8~15% 범위에서 자동 설정.
    - 기본: v=50에서 최대, v=0/100에서 최소(삼각형)
    """
    try:
        if not bool(cfg.get("swing_fng_entry_pct_enable", True)):
            return None
    except Exception:
        return None
    try:
        fg = (ext or {}).get("fear_greed") or {}
        v0 = fg.get("value", None)
        if v0 is None or str(v0).strip() == "":
            return None
        v = float(v0)
        if not math.isfinite(v):
            return None
        v = float(clamp(v, 0.0, 100.0))
    except Exception:
        return None
    try:
        pmin = float(cfg.get("swing_fng_entry_pct_min", 8.0) or 8.0)
        pmax = float(cfg.get("swing_fng_entry_pct_max", 15.0) or 15.0)
        if not (math.isfinite(pmin) and math.isfinite(pmax)):
            return None
        if pmax < pmin:
            pmin, pmax = pmax, pmin
        pmin = float(clamp(pmin, 0.5, 95.0))
        pmax = float(clamp(pmax, pmin, 95.0))
    except Exception:
        pmin, pmax = 8.0, 15.0
    try:
        # factor: 1 at 50, 0 at 0/100
        factor = 1.0 - (abs(float(v) - 50.0) / 50.0)
        factor = float(clamp(factor, 0.0, 1.0))
        return float(pmin + (pmax - pmin) * factor)
    except Exception:
        return None


# =========================================================
# ✅ 12.9) 외부 시황 비동기 갱신(봇 스레드 정체 방지)
# - 외부 RSS/캘린더/번역 등은 네트워크/SDK 이슈로 장시간 블로킹될 수 있음
# - TG_THREAD(스캔/하트비트)가 멈춘 것처럼 보이는 원인을 줄이기 위해,
#   외부시황은 별도 스레드에서 갱신하고 TG_THREAD는 "스냅샷"만 사용한다.
# =========================================================
_EXT_SNAPSHOT_LOCK = threading.RLock()
_EXT_SNAPSHOT: Dict[str, Any] = {"enabled": False, "asof_kst": now_kst_str(), "_source": "init"}
_EXT_INFLIGHT = False
_EXT_LAST_START_EPOCH = 0.0
_EXT_LAST_DONE_EPOCH = 0.0
_EXT_LAST_ERROR = ""


def external_context_snapshot() -> Dict[str, Any]:
    try:
        # ✅ safety: 잠재적 데드락/정체 방지(스레드 정체 → UI에 "멈춤 의심" 유발)
        got = False
        try:
            got = bool(_EXT_SNAPSHOT_LOCK.acquire(timeout=0.25))
        except Exception:
            got = False
        try:
            snap = dict(_EXT_SNAPSHOT or {})
            inflight = bool(_EXT_INFLIGHT)
            last_start = float(_EXT_LAST_START_EPOCH or 0.0)
            last_done = float(_EXT_LAST_DONE_EPOCH or 0.0)
            last_err = str(_EXT_LAST_ERROR or "")
        finally:
            if got:
                try:
                    _EXT_SNAPSHOT_LOCK.release()
                except Exception:
                    pass
        age_sec = (time.time() - last_done) if last_done else None
        snap["_inflight"] = inflight
        snap["_age_sec"] = float(age_sec) if age_sec is not None else None
        snap["_last_start_epoch"] = last_start
        snap["_last_done_epoch"] = last_done
        if last_err:
            snap["_last_err"] = last_err[:240]
        return snap
    except Exception:
        return {"enabled": False, "asof_kst": now_kst_str(), "_source": "snapshot_error"}


def _external_context_worker(cfg: Dict[str, Any], rt: Dict[str, Any]):
    global _EXT_INFLIGHT, _EXT_LAST_DONE_EPOCH, _EXT_LAST_ERROR
    err_msg = ""
    ext: Dict[str, Any] = {}
    try:
        ext0 = build_external_context(cfg, rt=rt)
        if isinstance(ext0, dict):
            ext = ext0
        else:
            ext = {"enabled": False, "error": "external_context_invalid", "asof_kst": now_kst_str()}
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        ext = {"enabled": False, "error": err_msg[:240], "asof_kst": now_kst_str()}
        notify_admin_error("EXTERNAL_CONTEXT_THREAD", e, tb=traceback.format_exc(), min_interval_sec=180.0)

    try:
        ext["_code_version"] = CODE_VERSION
    except Exception:
        pass

    try:
        with _EXT_SNAPSHOT_LOCK:
            _EXT_SNAPSHOT.clear()
            _EXT_SNAPSHOT.update(ext)
            _EXT_LAST_DONE_EPOCH = time.time()
            _EXT_LAST_ERROR = err_msg
            _EXT_INFLIGHT = False
    except Exception:
        try:
            _EXT_INFLIGHT = False
        except Exception:
            pass


def external_context_refresh_maybe(cfg: Dict[str, Any], rt: Dict[str, Any], force: bool = False) -> bool:
    """
    외부 시황 갱신을 "비동기"로 트리거한다.
    - 반환: 이번 호출에서 worker를 새로 시작했으면 True
    """
    global _EXT_INFLIGHT, _EXT_LAST_START_EPOCH, _EXT_LAST_ERROR
    try:
        if not bool(cfg.get("use_external_context", True)):
            with _EXT_SNAPSHOT_LOCK:
                _EXT_SNAPSHOT.clear()
                _EXT_SNAPSHOT.update({"enabled": False, "asof_kst": now_kst_str(), "_source": "disabled"})
                _EXT_LAST_ERROR = ""
                _EXT_INFLIGHT = False
            return False
    except Exception:
        pass

    try:
        refresh_sec = int(cfg.get("external_refresh_sec", 60) or 60)
    except Exception:
        refresh_sec = 60
    refresh_sec = max(15, refresh_sec)

    now_ts = time.time()
    got = False
    try:
        got = bool(_EXT_SNAPSHOT_LOCK.acquire(timeout=0.35))
    except Exception:
        got = False
    if not got:
        # 잠재적 교착/정체 방지: 이번 턴은 갱신 트리거를 건너뛴다(봇은 계속)
        return False
    try:
        if _EXT_INFLIGHT:
            # 오래 걸리는 작업이 이미 수행 중이면 중복 실행하지 않음(스레드 누수 방지)
            return False
        if (not force) and _EXT_LAST_DONE_EPOCH and (now_ts - float(_EXT_LAST_DONE_EPOCH or 0.0)) < refresh_sec:
            return False
        _EXT_INFLIGHT = True
        _EXT_LAST_START_EPOCH = now_ts
        _EXT_LAST_ERROR = ""
    finally:
        try:
            _EXT_SNAPSHOT_LOCK.release()
        except Exception:
            pass

    th = threading.Thread(
        target=_external_context_worker,
        args=(dict(cfg or {}), dict(rt or {})),
        daemon=True,
        name="EXTERNAL_CONTEXT_THREAD",
    )
    try:
        add_script_run_ctx(th)
    except Exception:
        pass
    th.start()
    return True


# =========================================================
# ✅ 13) AI 판단 + 리스크 매니저(기존 유지/강화)
# =========================================================
def _atr_price_pct(df: pd.DataFrame, window: int = 14) -> float:
    try:
        if df is None or df.empty or len(df) < window + 5:
            return 0.0
        if ta is not None:
            atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=window)
            v = float(atr.iloc[-1])
        else:
            # ta 미설치 환경에서도 최소 기능 유지(수동 ATR)
            v = float(calc_atr(df, period=window))
        c = float(df["close"].iloc[-1])
        if c <= 0:
            return 0.0
        return (v / c) * 100.0
    except Exception:
        return 0.0


def _swing_stop_price_pct(df: pd.DataFrame, decision: str, lookback: int = 40, buffer_atr_mul: float = 0.25) -> float:
    try:
        if df is None or df.empty or len(df) < lookback + 5:
            return 0.0
        recent = df.tail(lookback)
        last_close = float(df["close"].iloc[-1])
        atr_pct = _atr_price_pct(df, 14)
        buf_pct = atr_pct * buffer_atr_mul

        if decision == "buy":
            swing = float(recent["low"].min())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 - buf_pct / 100.0)
            return max(0.0, ((last_close - stop_price) / last_close) * 100.0)

        if decision == "sell":
            swing = float(recent["high"].max())
            if last_close <= 0:
                return 0.0
            stop_price = swing * (1.0 + buf_pct / 100.0)
            return max(0.0, ((stop_price - last_close) / last_close) * 100.0)

        return 0.0
    except Exception:
        return 0.0


def _rr_min_by_mode(mode: str) -> float:
    if mode == "안전모드":
        return 2.0
    if mode == "공격모드":
        return 2.5
    return 3.0


def _rr_min_by_style(style: str) -> float:
    # 스타일별 최소 손익비 가이드
    if style == "스캘핑":
        return 1.2
    if style == "스윙":
        # ✅ 스윙은 스캘핑보다 "훨씬 길게" 가져가는 전략이므로 RR 하한을 더 높게
        return 2.8
    return 1.5


def _risk_guardrail(out: Dict[str, Any], df: pd.DataFrame, decision: str, mode: str, style: str, external: Dict[str, Any]) -> Dict[str, Any]:
    lev = max(1, int(out.get("leverage", 1)))
    sl_roi = float(out.get("sl_pct", 1.2))
    tp_roi = float(out.get("tp_pct", 3.0))
    rr = float(out.get("rr", 0))

    sl_price_pct_now = sl_roi / max(lev, 1)

    atr_pct = _atr_price_pct(df, 14)
    min_price_stop = max(0.25, atr_pct * 0.9)

    swing_stop = _swing_stop_price_pct(df, decision, lookback=40, buffer_atr_mul=0.25)
    if swing_stop > 0:
        swing_stop = min(swing_stop, max(min_price_stop * 3.0, atr_pct * 3.0))
    recommended_price_stop = max(min_price_stop, swing_stop)

    notes = []

    # 외부시황: 극공포면 SL 여유 약간 추가
    try:
        fg = (external or {}).get("fear_greed") or {}
        v = int(fg.get("value", -1)) if fg else -1
        if 0 <= v <= 25:
            recommended_price_stop = max(recommended_price_stop, min_price_stop * 1.2)
            notes.append("외부시황: 극공포 → 손절 여유 추가")
    except Exception:
        pass

    if sl_price_pct_now < recommended_price_stop:
        sl_price_pct_now = recommended_price_stop
        sl_roi = sl_price_pct_now * lev
        notes.append(f"손절폭(가격기준) 확장({recommended_price_stop:.2f}%)")

    rr_min_mode = _rr_min_by_mode(mode)
    rr_min_style = _rr_min_by_style(style)
    rr_min = max(rr_min_mode, rr_min_style)

    if rr <= 0:
        rr = max(rr_min, tp_roi / max(sl_roi, 0.01))

    if tp_roi < sl_roi * rr_min:
        tp_roi = sl_roi * rr_min
        notes.append(f"손익비 최소 {rr_min:.1f} 확보(익절 상향)")

    rr = max(rr, tp_roi / max(sl_roi, 0.01))

    out["sl_pct"] = float(sl_roi)
    out["tp_pct"] = float(tp_roi)
    out["rr"] = float(rr)
    out["sl_price_pct"] = float(sl_roi / max(lev, 1))
    out["tp_price_pct"] = float(tp_roi / max(lev, 1))
    out["risk_note"] = " / ".join(notes) if notes else "보정 없음"
    return out


def ai_decide_trade(
    df: pd.DataFrame,
    status: Dict[str, Any],
    symbol: str,
    mode: str,
    cfg: Dict[str, Any],
    external: Dict[str, Any],
    trend_long: str = "",
    sr_context: Optional[Dict[str, Any]] = None,
    chart_style_hint: str = "",
) -> Dict[str, Any]:
    """
    ✅ 기존 기능 유지: AI가 buy/sell/hold + entry/leverage/sl/tp/rr/근거(JSON)
    ✅ 안정성 강화: timeout + 예외 처리
    """
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가"))
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return {"decision": "hold", "confidence": 0, "reason_easy": msg, "used_indicators": status.get("_used_indicators", [])}
    if df is None or df.empty or status is None:
        return {"decision": "hold", "confidence": 0, "reason_easy": "데이터 부족", "used_indicators": status.get("_used_indicators", [])}

    rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    past_mistakes = get_past_mistakes_text(5)

    # daily brief를 포함한 외부시황(이미 thread에서 build했으면 그걸 쓰게 external 파라미터로 전달)
    ext = external or {}
    ext_enabled = False
    try:
        if isinstance(ext, dict) and ext:
            ext_enabled = bool(ext.get("enabled", True))
    except Exception:
        ext_enabled = False
    daily_brief = (ext.get("daily_btc_brief") or {}) if (ext_enabled and isinstance(ext, dict)) else {}

    features = {
        "symbol": symbol,
        "mode": mode,
        "price": float(last["close"]),
        "rsi_prev": float(prev.get("RSI", 50)) if "RSI" in df.columns else None,
        "rsi_now": float(last.get("RSI", 50)) if "RSI" in df.columns else None,
        "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
        "trend_short": status.get("추세", ""),  # 단기추세(timeframe)
        "trend_long": str(trend_long or ""),
        "bb": status.get("BB", ""),
        "macd": status.get("MACD", ""),
        "vol": status.get("거래량", ""),
        "rsi_resolve_long": bool(status.get("_rsi_resolve_long", False)),
        "rsi_resolve_short": bool(status.get("_rsi_resolve_short", False)),
        "pullback_candidate": bool(status.get("_pullback_candidate", False)),
        "atr_price_pct": _atr_price_pct(df, 14),
        "sqz": {
            "text": status.get("SQZ", ""),
            "on": bool(status.get("_sqz_on", False)),
            "mom_pct": float(status.get("_sqz_mom_pct", 0.0) or 0.0),
            "bias": int(status.get("_sqz_bias", 0) or 0),
            "strength": float(status.get("_sqz_strength", 0.0) or 0.0),
        },
        "chart_patterns": {
            "summary": status.get("패턴", ""),
            "bias": int(status.get("_pattern_bias", 0) or 0),
            "strength": float(status.get("_pattern_strength", 0.0) or 0.0),
            "detected": list(status.get("_pattern_tags", []) or []),
            "bullish": list(status.get("_pattern_bullish", []) or []),
            "bearish": list(status.get("_pattern_bearish", []) or []),
        },
        "chart_patterns_mtf": status.get("_pattern_mtf", {}) if isinstance(status.get("_pattern_mtf", {}), dict) else {},
        "ml_signals": status.get("_ml_signals", {}) if isinstance(status.get("_ml_signals", {}), dict) else {},
        "sr_context": sr_context or {},
        "chart_style_hint": str(chart_style_hint or ""),
        "external": (
            {
                "fear_greed": ext.get("fear_greed"),
                "high_impact_events_soon": (ext.get("high_impact_events_soon") or [])[:3],
                "global": ext.get("global"),
                "daily_btc_brief": daily_brief,
            }
            if ext_enabled
            else {}
        ),
    }

    fg_txt = ""
    try:
        fg = (ext or {}).get("fear_greed") or {} if ext_enabled else {}
        if fg:
            fg_txt = f"- 공포탐욕지수: {fg.get('emoji','')} {int(fg.get('value', 0))} / {fg.get('classification','')}"
    except Exception:
        fg_txt = ""

    ev_txt = ""
    try:
        evs = (ext or {}).get("high_impact_events_soon") or [] if ext_enabled else []
        if evs:
            ev_txt = "- 중요 이벤트(임박): " + " | ".join([f"{e.get('country','')} {e.get('title','')}" for e in evs[:3]])
    except Exception:
        ev_txt = ""

    brief_txt = ""
    try:
        items = (daily_brief or {}).get("items") or []
        if items:
            brief_txt = "- 오늘 아침 BTC 브리핑(요약): " + " / ".join([str(i.get("title", ""))[:40] for i in items[:3]])
    except Exception:
        brief_txt = ""

    ext_hdr = "[외부 시황(참고)]\n" + "\n".join([x for x in [fg_txt, ev_txt, brief_txt] if x]) if ext_enabled else "[외부 시황] (스캘핑/단기 판단: 적용하지 않음)"

    # ✅ 소액 탐색 진입(soft entry) 힌트: 확신이 min_conf에 조금 못 미쳐도, 아주 작게/보수적으로 진입 가능
    soft_entry_hint = ""
    try:
        if bool(cfg.get("soft_entry_enable", True)):
            gap = 0
            if str(mode) == "안전모드":
                gap = int(cfg.get("soft_entry_conf_gap_safe", 0) or 0)
            elif str(mode) == "공격모드":
                gap = int(cfg.get("soft_entry_conf_gap_attack", 8) or 8)
            else:
                gap = int(cfg.get("soft_entry_conf_gap_highrisk", 6) or 6)
            gap = int(max(0, gap))
            if gap > 0:
                min_soft = int(max(0, int(rule.get("min_conf", 0) or 0) - int(gap)))
                if min_soft > 0 and min_soft < int(rule.get("min_conf", 0) or 0):
                    soft_entry_hint = (
                        f"\n4) (소액 탐색 진입)\n"
                        f"- 확신도가 {min_soft}~{int(rule.get('min_conf',0))-1}%이면, '소액 탐색'으로 buy/sell을 줄 수 있다.\n"
                        f"- 이때는 entry_pct/leverage를 최소값 근처로, tp_pct/sl_pct도 보수적으로(짧게) 설정해라."
                    )
    except Exception:
        soft_entry_hint = ""

    # ✅ 최근 승률 기반 AI 전략 힌트 생성
    winrate_hint = ""
    try:
        df_log = read_trade_log()
        if not df_log.empty and "PnL_Percent" in df_log.columns:
            recent20 = df_log.tail(20)
            pnl_r = recent20["PnL_Percent"].astype(float)
            wr = float((pnl_r > 0).sum()) / max(1, len(recent20)) * 100.0
            if wr < 40:
                winrate_hint = f"\n[전략 조정] 최근 승률 {wr:.0f}%(저조) → 확신도 높은 진입만 허용, RR 3.0 이상 셋업 우선"
            elif wr > 65:
                winrate_hint = f"\n[전략 조정] 최근 승률 {wr:.0f}%(양호) → 추세 추종 적극 진입, 눌림목/돌파 모두 허용"
            else:
                winrate_hint = f"\n[전략 조정] 최근 승률 {wr:.0f}%(보통) → 강한 추세 + 명확한 해소 신호에서만 진입"
    except Exception:
        winrate_hint = ""

    sys = f"""
	너는 '워뇨띠 스타일(눌림목/해소 타이밍) + 손익비' 기반의 자동매매 트레이더 AI다.
{winrate_hint}

	[과거 실수 & 성과 요약]
{past_mistakes}

{ext_hdr}

			[핵심 룰]
			1) RSI 과매도/과매수 '상태'에 즉시 진입하지 말고, '해소되는 시점'에서만 진입 후보.
				2) 상승추세에서는 롱 우선, 하락추세에서는 숏 우선. (역추세는 더 짧게/보수적으로)
				3) SQZ(스퀴즈 모멘텀) 신호를 진입 판단의 80% 이상으로 반영해라. (모멘텀 방향/세기 우선)
				4) chart_patterns(M/W, 쌍봉/쌍바닥, 삼중천정/삼중바닥, 삼각수렴, 박스, 쐐기, 헤드앤숄더)을 반드시 참고해라.
				   - pattern bias와 반대 방향이면 보수적으로 hold를 우선해라.
				4-1) chart_patterns_mtf는 1m/3m/5m/15m/30m/1h/2h/4h 종합 패턴이다.
				   - 단기 패턴과 MTF 패턴이 같은 방향이면 신뢰도를 높이고, 반대면 보수적으로 접근해라.
				5) ml_signals(주력 지표 수렴: Lorentzian/KNN/Logistic/SQZ/RSI/패턴)을 반드시 따른다.
				   - ml_signals.dir이 "buy"면 decision은 buy만 가능(반대 방향 금지)
				   - ml_signals.dir이 "sell"면 decision은 sell만 가능(반대 방향 금지)
				   - ml_signals.dir이 "hold"면 hold
				6) 모드 규칙 반드시 준수:
		   - 최소 확신도: {rule["min_conf"]}
		   - 진입 비중(%): {rule["entry_pct_min"]}~{rule["entry_pct_max"]}
		   - 레버리지: {rule["lev_min"]}~{rule["lev_max"]}
		{soft_entry_hint}

	[중요]
	- sl_pct / tp_pct는 ROI%(레버 반영 수익률)로 출력한다.
	- 변동성(atr_price_pct)이 작으면 손절을 너무 타이트하게 잡지 마라.
	- sr_context(지지/저항) 정보를 참고해, 가능하면 sl_price/tp_price(가격)를 함께 지정해라.
	  - buy(롱): sl_price는 price보다 낮게, tp_price는 price보다 높게
	  - sell(숏): sl_price는 price보다 높게, tp_price는 price보다 낮게
	- sr_context.volume_nodes(매물대/거래량 집중 구간)가 있으면, TP/SL은 "매물 많은 강한 레벨"을 우선으로 잡아라.
	- 목표 TP/SL은 '수익보존(트레일링)'과 결합될 수 있게, 너무 비현실적으로 멀거나 가깝게 잡지 마라.
[생존(중요)]
- 이 시스템은 손실 확대/과매매가 감지되면 자동매매를 강제 종료한다.
- 영어 금지. 쉬운 한글.
- 반드시 JSON만 출력.
- 하이리스크/하이리턴 모드: 추세가 있으면 적극적으로 진입해라. 과도한 hold는 기회 손실이다.
- 안전모드: 확신이 애매하면 'hold'를 선택해라. (무리한 진입 금지)
"""

    user = f"""
시장 데이터(JSON):
{json.dumps(features, ensure_ascii=False)}

JSON 형식:
{{
  "decision": "buy"|"sell"|"hold",
  "confidence": 0-100,
  "entry_pct": {rule["entry_pct_min"]}-{rule["entry_pct_max"]},
  "leverage": {rule["lev_min"]}-{rule["lev_max"]},
  "sl_pct": 0.3-50.0,
  "tp_pct": 0.5-150.0,
  "rr": 0.5-10.0,
  "sl_price": number|null,
  "tp_price": number|null,
  "used_indicators": ["..."],
  "reason_easy": "쉬운 한글"
}}
"""
    try:
        # 모델 fallback (gpt-4o 미지원 계정/환경 대응)
        models = [
            str(cfg.get("openai_model_trade", "") or "").strip(),
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        # 중복 제거(순서 유지)
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=900,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        out["_openai_model"] = model_used

        out["decision"] = out.get("decision", "hold")
        if out["decision"] not in ["buy", "sell", "hold"]:
            out["decision"] = "hold"

        out["confidence"] = int(clamp(_as_int(out.get("confidence", 0), 0), 0, 100))

        # OpenAI가 null(None)을 줄 수 있으므로 숫자 변환은 항상 안전 변환 사용
        out["entry_pct"] = float(_as_float(out.get("entry_pct", rule["entry_pct_min"]), float(rule["entry_pct_min"])))
        out["entry_pct"] = float(clamp(out["entry_pct"], rule["entry_pct_min"], rule["entry_pct_max"]))

        out["leverage"] = int(_as_int(out.get("leverage", rule["lev_min"]), int(rule["lev_min"])))
        out["leverage"] = int(clamp(out["leverage"], rule["lev_min"], rule["lev_max"]))

        out["sl_pct"] = float(_as_float(out.get("sl_pct", 1.2), 1.2))
        out["tp_pct"] = float(_as_float(out.get("tp_pct", 3.0), 3.0))
        out["rr"] = float(_as_float(out.get("rr", max(0.5, out["tp_pct"] / max(out["sl_pct"], 0.01))), max(0.5, out["tp_pct"] / max(out["sl_pct"], 0.01))))

        # (선택) 가격 기반 SL/TP (SR 기반)
        try:
            sp = out.get("sl_price", None)
            tp = out.get("tp_price", None)
            sp_f = float(sp) if sp is not None and str(sp).strip() != "" else None
            tp_f = float(tp) if tp is not None and str(tp).strip() != "" else None
            px_now = float(last["close"])
            dec0 = str(out.get("decision", "hold"))
            if dec0 == "buy":
                if sp_f is not None and (sp_f <= 0 or sp_f >= px_now):
                    sp_f = None
                if tp_f is not None and (tp_f <= px_now):
                    tp_f = None
            elif dec0 == "sell":
                if sp_f is not None and (sp_f <= px_now):
                    sp_f = None
                if tp_f is not None and (tp_f <= 0 or tp_f >= px_now):
                    tp_f = None
            else:
                sp_f = None
                tp_f = None
            out["sl_price"] = sp_f
            out["tp_price"] = tp_f
        except Exception:
            out["sl_price"] = None
            out["tp_price"] = None

        used = out.get("used_indicators", status.get("_used_indicators", []))
        if not isinstance(used, list):
            used = status.get("_used_indicators", [])
        out["used_indicators"] = used

        out["reason_easy"] = str(out.get("reason_easy", ""))[:500]

        # ✅ 이전에는 min_conf 미만이면 강제로 hold로 바꿨지만,
        # 사용자는 "조건이 애매해도 소액/보수적으로 진입"을 원하므로 여기서는 decision을 유지한다.
        # (실제 주문은 스캔 루프에서 soft-entry/포지션 제한으로 제어)
        try:
            if out["decision"] in ["buy", "sell"] and out["confidence"] < int(rule["min_conf"]):
                out["below_min_conf"] = True
                # 너무 낮은 확신(바닥값)은 강제 hold
                floor0 = int(cfg.get("ai_decision_min_conf_floor", 60) or 60)
                if int(out["confidence"]) < int(floor0):
                    out["decision"] = "hold"
        except Exception:
            pass

        return out

    except FuturesTimeoutError:
        return {"decision": "hold", "confidence": 0, "reason_easy": "AI 타임아웃(대기 너무 김)", "used_indicators": status.get("_used_indicators", [])}
    except Exception as e:
        openai_handle_failure(e, cfg, where="DECIDE_TRADE")
        notify_admin_error("AI:DECIDE_TRADE", e, context={"symbol": symbol, "mode": mode}, tb=traceback.format_exc(), min_interval_sec=120.0)
        return {"decision": "hold", "confidence": 0, "reason_easy": f"AI 오류: {e}", "used_indicators": status.get("_used_indicators", [])}


# ✅ 스타일 AI 호출 캐시(스캔/포지션 루프에서 반복 호출되면 비용/지연/429가 쉽게 발생)
_AI_STYLE_CACHE_LOCK = threading.RLock()
_AI_STYLE_CACHE: Dict[str, Dict[str, Any]] = {}


def ai_decide_style(symbol: str, decision: str, trend_short: str, trend_long: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    룰 기반으로 애매할 때만 AI로 스캘핑/스윙 판단.
    비용/지연 최소화를 위해 기본은 룰 기반.
    """
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return {"style": "스캘핑", "confidence": 55, "reason": f"{msg} → 룰 기반(보수적으로 스캘핑)"}

    cache_sec = int(cfg.get("style_ai_cache_sec", 600) or 0)
    key = f"{symbol}|{decision}|{trend_short}|{trend_long}"
    if cache_sec > 0:
        try:
            with _AI_STYLE_CACHE_LOCK:
                ent = _AI_STYLE_CACHE.get(key)
                if ent:
                    ts = float(ent.get("ts", 0) or 0)
                    if ts and (time.time() - ts) < float(cache_sec):
                        out_cached = ent.get("out", {})
                        if isinstance(out_cached, dict) and out_cached:
                            return dict(out_cached)
        except Exception:
            pass

    payload = {
        "symbol": symbol,
        "decision": decision,
        "trend_short": trend_short,
        "trend_long": trend_long,
    }
    sys = (
        "너는 트레이딩 스타일 분류기다.\n"
        "단기/장기 추세와 방향(decision)을 보고 지금은 '스캘핑'이 유리한지 '스윙'이 유리한지 결정한다.\n"
        "출력은 반드시 JSON만.\n"
        '형식: {"style":"스캘핑"|"스윙","confidence":0-100,"reason":"쉬운 한글"}'
    )
    try:
        models = [
            str(cfg.get("openai_model_style", "") or "").strip(),
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        _model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            # 분류 태스크는 온도를 낮춰 흔들림(스캘핑↔스윙 플랩)을 줄인다.
            temperature=0.0,
            max_tokens=250,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        style = str(out.get("style", "스캘핑"))
        if style not in ["스캘핑", "스윙"]:
            style = "스캘핑"
        conf = int(clamp(int(out.get("confidence", 55)), 0, 100))
        reason = str(out.get("reason", ""))[:240]
        res = {"style": style, "confidence": conf, "reason": reason}
        if cache_sec > 0:
            try:
                with _AI_STYLE_CACHE_LOCK:
                    _AI_STYLE_CACHE[key] = {"ts": time.time(), "out": dict(res)}
                    # 메모리 누수 방지: 너무 커지면 오래된 것 일부 삭제
                    if len(_AI_STYLE_CACHE) > 2500:
                        # ts 기준 정렬 후 앞쪽(오래된) 정리
                        items = sorted(_AI_STYLE_CACHE.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0) or 0))
                        for k0, _ in items[:500]:
                            _AI_STYLE_CACHE.pop(k0, None)
            except Exception:
                pass
        return res
    except Exception as e:
        openai_handle_failure(e, cfg, where="DECIDE_STYLE")
        notify_admin_error("AI:DECIDE_STYLE", e, context={"symbol": symbol}, tb=traceback.format_exc(), min_interval_sec=180.0)
        return {"style": "스캘핑", "confidence": 55, "reason": "스타일 AI 판단 실패 → 스캘핑"}


def decide_style_rule_based(decision: str, trend_short: str, trend_long: str) -> Tuple[str, int, str]:
    """
    ✅ 핵심 요구 반영:
    - 단기/장기 추세가 모두 같은 방향이면 '스윙'
    - 단기만 맞으면 '스캘핑'(역추세 허용 but 짧게)
    """
    ts = str(trend_short or "")
    tl = str(trend_long or "")
    d = str(decision or "")

    def _align(tr: str, dec: str) -> bool:
        if dec == "buy":
            return "상승" in tr
        if dec == "sell":
            return "하락" in tr
        return False

    short_ok = _align(ts, d)
    long_ok = _align(tl, d)

    if short_ok and long_ok:
        return "스윙", 85, "단기+장기 추세가 같은 방향 → 스윙 유리"
    if short_ok and not long_ok:
        return "스캘핑", 82, "단기만 같은 방향(역추세/전환 구간) → 스캘핑 유리"
    if (not short_ok) and long_ok:
        return "스캘핑", 65, "장기만 같은 방향(단기 흔들림) → 보수적으로 스캘핑"
    return "스캘핑", 55, "추세 애매/불일치 → 스캘핑(보수)"


def apply_style_envelope(ai: Dict[str, Any], style: str, cfg: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI 출력은 유지하되, 스타일별 상한/하한으로 보정한다(기능 축소 X, 안전장치).
    """
    out = dict(ai or {})
    try:
        entry_pct = float(out.get("entry_pct", rule["entry_pct_min"]))
        lev = int(out.get("leverage", rule["lev_min"]))
        sl = float(out.get("sl_pct", 1.2))
        tp = float(out.get("tp_pct", 3.0))

        if style == "스캘핑":
            entry_pct = float(clamp(entry_pct * float(cfg.get("scalp_entry_pct_mult", 0.65)), rule["entry_pct_min"], rule["entry_pct_max"]))
            # ✅ 모드(MODE_RULES)의 레버 범위를 우선 존중:
            # - 하이리스크/하이리턴(예: lev_min=12)에서 scalp_lev_cap=8 때문에 레버가 8로 고정되는 문제 방지
            try:
                cap_cfg = int(cfg.get("scalp_lev_cap", rule["lev_max"]) or rule["lev_max"])
            except Exception:
                cap_cfg = int(rule.get("lev_max", lev) or lev)
            try:
                rule_min = int(rule.get("lev_min", 1) or 1)
                rule_max = int(rule.get("lev_max", cap_cfg) or cap_cfg)
            except Exception:
                rule_min, rule_max = 1, cap_cfg
            # cap이 모드 최소보다 작으면(하이리스크 등) cap 자체를 무시하고 모드 범위 내에서 유지
            cap = cap_cfg if cap_cfg >= rule_min else rule_max
            lev = int(min(lev, int(cap)))
            sl = float(clamp(sl, float(cfg.get("scalp_sl_roi_min", 0.8)), float(cfg.get("scalp_sl_roi_max", 5.0))))
            tp = float(clamp(tp, float(cfg.get("scalp_tp_roi_min", 0.8)), float(cfg.get("scalp_tp_roi_max", 6.0))))

        elif style == "스윙":
            entry_pct = float(clamp(entry_pct * float(cfg.get("swing_entry_pct_mult", 1.0)), rule["entry_pct_min"], rule["entry_pct_max"]))
            lev = int(min(lev, int(cfg.get("swing_lev_cap", rule["lev_max"]))))
            sl = float(clamp(sl, float(cfg.get("swing_sl_roi_min", 12.0)), float(cfg.get("swing_sl_roi_max", 30.0))))
            tp = float(clamp(tp, float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))

        out["entry_pct"] = entry_pct
        out["leverage"] = lev
        out["sl_pct"] = sl
        out["tp_pct"] = tp
        out["rr"] = float(out.get("rr", tp / max(sl, 0.01)))
    except Exception:
        pass
    return out


def apply_scalp_price_guardrails(out: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    스캘핑에서 레버가 높을 때 ROI% 기준 TP/SL이 과도하게 커져
    (+50%인데도 익절 안 하는 등) 문제가 생기는 것을 방지하기 위해,
    "가격 변동폭(%)" 기준으로 TP/SL을 재한정한다.
    """
    res = dict(out or {})
    try:
        lev = int(_as_int(res.get("leverage", rule.get("lev_min", 1)), int(rule.get("lev_min", 1) or 1)))
        lev = max(1, lev)

        sl_min = float(cfg.get("scalp_sl_price_pct_min", 0.25))
        sl_max = float(cfg.get("scalp_sl_price_pct_max", 1.0))
        tp_min = float(cfg.get("scalp_tp_price_pct_min", 0.35))
        tp_max = float(cfg.get("scalp_tp_price_pct_max", 1.6))
        rr_min_price = float(cfg.get("scalp_rr_min_price", 1.2))

        atr_pct = _atr_price_pct(df, 14)
        # 기본 추천(변동성 기반): 너무 작으면 수수료/노이즈로만 끝나는 것을 줄임
        sl_reco = max(sl_min, float(atr_pct) * 1.15)
        tp_reco = max(tp_min, float(atr_pct) * 1.85)

        # 기존 값(있으면)도 참고
        sl_price_pct0 = _as_float(res.get("sl_price_pct", None), 0.0)
        tp_price_pct0 = _as_float(res.get("tp_price_pct", None), 0.0)
        if sl_price_pct0 <= 0:
            sl_roi0 = _as_float(res.get("sl_pct", None), 0.0)
            sl_price_pct0 = abs(float(sl_roi0)) / max(float(lev), 1.0) if lev else abs(float(sl_roi0))
        if tp_price_pct0 <= 0:
            tp_roi0 = _as_float(res.get("tp_pct", None), 0.0)
            tp_price_pct0 = abs(float(tp_roi0)) / max(float(lev), 1.0) if lev else abs(float(tp_roi0))

        sl_price_pct = max(float(sl_price_pct0), float(sl_reco))
        tp_price_pct = max(float(tp_price_pct0), float(tp_reco))

        sl_price_pct = float(clamp(sl_price_pct, sl_min, sl_max))
        tp_price_pct = float(clamp(tp_price_pct, tp_min, tp_max))

        # 가격 기준 RR 하한(너무 작은 TP 방지)
        if rr_min_price > 0 and tp_price_pct < (sl_price_pct * rr_min_price):
            tp_price_pct = float(clamp(sl_price_pct * rr_min_price, tp_min, tp_max))

        res["sl_price_pct"] = float(sl_price_pct)
        res["tp_price_pct"] = float(tp_price_pct)
        res["sl_pct"] = float(sl_price_pct * float(lev))
        res["tp_pct"] = float(tp_price_pct * float(lev))
        res["rr"] = float(res["tp_pct"] / max(abs(float(res["sl_pct"])), 0.01))
        res["risk_note"] = str(res.get("risk_note", "") or "").strip()
        res["_scalp_price_guardrail"] = {
            "atr_price_pct": float(atr_pct),
            "sl_price_pct": float(sl_price_pct),
            "tp_price_pct": float(tp_price_pct),
            "sl_roi_pct": float(res["sl_pct"]),
            "tp_roi_pct": float(res["tp_pct"]),
        }
    except Exception:
        return res
    return res


# =========================================================
# ✅ 14) AI 회고(후기) (기존 유지 + 안정성)
# =========================================================
def ai_write_review(symbol: str, side: str, pnl_percent: float, reason: str, cfg: Dict[str, Any]) -> Tuple[str, str]:
    h = openai_health_info(cfg)
    client = get_openai_client(cfg)
    if client is None:
        one = "익절" if pnl_percent >= 0 else "손절"
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        return (f"{one}({pnl_percent:.2f}%)", f"{msg} - 후기 자동작성 불가")

    sys = "너는 매매 회고를 아주 쉽게 써주는 코치다. 출력은 반드시 JSON만. 영어 금지."
    user = f"""
상황:
- 코인: {symbol}
- 포지션: {side}
- 결과: {pnl_percent:.2f}%
- 청산 이유: {reason}

JSON 형식:
{{
  "one_line": "한줄평(아주 짧게)",
  "review": "후기(손절이면 다음에 개선 / 익절이면 유지할 점)"
}}
    """
    try:
        models = [
            str(cfg.get("openai_model_review", "") or "").strip(),
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
        models2: List[str] = []
        for m in models:
            m = str(m or "").strip()
            if not m:
                continue
            if m not in models2:
                models2.append(m)

        _model_used, resp = openai_chat_create_with_fallback(
            client=client,
            models=models2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500,
            timeout_sec=OPENAI_TIMEOUT_SEC,
        )
        out = json.loads(resp.choices[0].message.content)
        return str(out.get("one_line", ""))[:120], str(out.get("review", ""))[:800]
    except Exception as e:
        openai_handle_failure(e, cfg, where="WRITE_REVIEW")
        notify_admin_error("AI:WRITE_REVIEW", e, context={"symbol": symbol}, tb=traceback.format_exc(), min_interval_sec=180.0)
        one = "익절" if pnl_percent >= 0 else "손절"
        return (f"{one}({pnl_percent:.2f}%)", "후기 작성 실패")


# =========================================================
# ✅ 15) 모니터 상태(하트비트) + 이벤트 링버퍼
# =========================================================
def monitor_init():
    mon = read_json_safe(MONITOR_FILE, {"coins": {}, "events": [], "scan_process": []}) or {"coins": {}, "events": [], "scan_process": []}
    mon["_boot_time_kst"] = now_kst_str()
    mon["_last_write"] = 0
    write_json_atomic(MONITOR_FILE, mon)
    return mon


def monitor_write_throttled(mon: Dict[str, Any], min_interval_sec: float = 1.0):
    lastw = float(mon.get("_last_write", 0))
    if time.time() - lastw >= min_interval_sec:
        write_json_atomic(MONITOR_FILE, mon)
        mon["_last_write"] = time.time()


def mon_add_event(mon: Dict[str, Any], ev_type: str, symbol: str = "", message: str = "", extra: Optional[Dict[str, Any]] = None):
    try:
        ev = {"time_kst": now_kst_str(), "type": ev_type, "symbol": symbol, "message": message, "extra": extra or {}}
        mon.setdefault("events", [])
        mon["events"].append(ev)
        mon["events"] = mon["events"][-250:]
        # Google Sheets EVENT 누적(비동기 큐)
        try:
            gsheet_log_event(stage=ev_type, message=f"{symbol} {message}".strip(), payload={"symbol": symbol, **(extra or {})})
        except Exception:
            pass
    except Exception:
        pass


def mon_add_scan(mon: Dict[str, Any], stage: str, symbol: str, tf: str = "", signal: str = "", score: Any = "", message: str = "", extra: Optional[Dict[str, Any]] = None):
    """
    SCAN Process 로그(요구사항):
    - stage: fetch_short/fetch_long/support_resistance/rule_signal/ai_call/ai_result/trade_opened/trade_skipped/in_position ...
    - monitor_state.json에 저장되어 UI/Telegram이 항상 최신을 볼 수 있게 함
    """
    try:
        ts = time.time()
        rec = {
            "time_kst": now_kst_str(),
            "stage": stage,
            "symbol": symbol,
            "tf": tf,
            "signal": signal,
            "score": score,
            "message": message,
            "extra": extra or {},
        }
        mon.setdefault("scan_process", [])
        mon["scan_process"].append(rec)
        mon["scan_process"] = mon["scan_process"][-400:]
        mon["last_scan_epoch"] = ts
        mon["last_scan_kst"] = rec.get("time_kst", "")
        # 코인별 진행상황(요구사항: "어떤 단계로 분석중인지" 직관적으로)
        try:
            sym0 = str(symbol or "").strip()
            if sym0 and sym0 != "*":
                mon.setdefault("coins", {}).setdefault(sym0, {})
                mon["coins"][sym0]["scan_stage"] = stage
                mon["coins"][sym0]["scan_stage_kst"] = rec.get("time_kst", "")
                mon["coins"][sym0]["last_scan_epoch"] = ts
                mon["coins"][sym0]["last_scan_kst"] = rec.get("time_kst", "")
        except Exception:
            pass
        # ✅ 스캔이 길어져도 UI에서 "멈춤 의심"이 과도하게 뜨지 않게, 스캔 단계도 heartbeat로 간주
        mon["last_heartbeat_epoch"] = ts
        mon["last_heartbeat_kst"] = rec.get("time_kst", "")
        # Google Sheets에도 SCAN 누적(비동기 큐)
        try:
            gsheet_log_scan(stage=stage, symbol=symbol, tf=tf, signal=signal, score=score, message=message, payload=extra or {})
        except Exception:
            pass
        # ✅ 블로킹 호출(ex.fetch_ohlcv 등) 직전에 stage/heartbeat가 화면에 보이도록 파일을 주기적으로 flush
        try:
            monitor_write_throttled(mon, min_interval_sec=0.8)
        except Exception:
            pass
    except Exception:
        pass


def mon_recent_events(mon: Dict[str, Any], within_min: int = 15) -> List[Dict[str, Any]]:
    try:
        evs = mon.get("events", []) or []
        now = now_kst()
        out = []
        for e in reversed(evs):
            dt = _parse_time_kst(str(e.get("time_kst", "")))
            if not dt:
                continue
            if (now - dt).total_seconds() <= within_min * 60:
                out.append(e)
            else:
                break
        return list(reversed(out))
    except Exception:
        return []


# =========================================================
# ✅ 16) 텔레그램 유틸 (timeout/retry + 채널/그룹 라우팅)
# =========================================================
def _tg_post(url: str, data: Dict[str, Any], timeout_sec: Optional[float] = None):
    """
    Telegram Bot API POST helper.
    - TG_THREAD(트레이딩 루프)가 네트워크로 멈춰 보이지 않게, 전송은 별도 워커(TG_SEND_THREAD)에서 수행한다.
    - 여기서는 timeout을 항상 지정(영구 대기 방지)한다.
    """
    to = float(timeout_sec or HTTP_TIMEOUT_SEC)
    # requests timeout: (connect, read)
    # - connect는 짧게, read는 설정값 사용
    timeout = (min(4.0, max(1.0, to * 0.5)), max(2.0, to))
    if retry is None:
        return requests.post(url, data=data, timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.6, max=3.0))
    def _do():
        r = requests.post(url, data=data, timeout=timeout)
        r.raise_for_status()
        return r

    return _do()


# =========================================================
# ✅ 16.1) Telegram Send Worker (daemon)
# - 요구사항/실전 이슈: TG_THREAD가 requests.post(sendMessage)에서 블로킹되면
#   하트비트/스캔/매매가 "멈춘 것처럼" 보일 수 있음.
# - 해결: 전송은 큐에 넣고 TG_SEND_THREAD가 처리한다.
# =========================================================
_TG_SEND_QUEUE_HIGH = deque()
_TG_SEND_QUEUE_NORMAL = deque()
_TG_SEND_QUEUE_LOCK = threading.RLock()
_TG_SEND_QUEUE_EVENT = threading.Event()
_TG_SEND_QUEUE_MAX_HIGH = 300
_TG_SEND_QUEUE_MAX_NORMAL = 1200
_TG_SEND_LAST_ERR = ""
_TG_SEND_LAST_ERR_KST = ""


def tg_enqueue(method: str, data: Dict[str, Any], *, priority: str = "normal") -> None:
    """
    method: "sendMessage" | "sendPhoto" | "answerCallbackQuery" | ...
    priority: "high"(admin/중요) | "normal"
    """
    if not tg_token:
        return
    m = str(method or "").strip()
    if not m:
        return
    rec = {"method": m, "data": dict(data or {}), "priority": str(priority or "normal"), "attempt": 0, "ts": time.time()}
    try:
        with _TG_SEND_QUEUE_LOCK:
            if str(priority).lower() == "high":
                _TG_SEND_QUEUE_HIGH.append(rec)
                while len(_TG_SEND_QUEUE_HIGH) > int(_TG_SEND_QUEUE_MAX_HIGH):
                    _TG_SEND_QUEUE_HIGH.popleft()
            else:
                _TG_SEND_QUEUE_NORMAL.append(rec)
                while len(_TG_SEND_QUEUE_NORMAL) > int(_TG_SEND_QUEUE_MAX_NORMAL):
                    _TG_SEND_QUEUE_NORMAL.popleft()
        _TG_SEND_QUEUE_EVENT.set()
    except Exception:
        return


def telegram_send_worker_thread():
    """
    Telegram sendMessage/answerCallbackQuery worker.
    - 네트워크 장애/레이트리밋이 있어도 TG_THREAD는 계속 돈다.
    """
    backoff = 0.5
    while True:
        rec = None
        try:
            if not tg_token:
                time.sleep(2.0)
                continue
            rec = None
            with _TG_SEND_QUEUE_LOCK:
                if _TG_SEND_QUEUE_HIGH:
                    rec = _TG_SEND_QUEUE_HIGH.popleft()
                elif _TG_SEND_QUEUE_NORMAL:
                    rec = _TG_SEND_QUEUE_NORMAL.popleft()
            if rec is None:
                _TG_SEND_QUEUE_EVENT.wait(timeout=2.0)
                try:
                    _TG_SEND_QUEUE_EVENT.clear()
                except Exception:
                    pass
                continue

            method = str(rec.get("method", "") or "").strip()
            data = rec.get("data", {}) or {}
            if not method:
                continue

            url = f"https://api.telegram.org/bot{tg_token}/{method}"
            # send worker는 너무 오래 붙잡지 않게 timeout을 조금 더 짧게
            if method == "sendPhoto":
                file_path = str(data.pop("__file_path", "") or "").strip()
                timeout_sec = min(float(HTTP_TIMEOUT_SEC), 12.0)
                timeout = (min(4.0, max(1.0, timeout_sec * 0.5)), max(2.0, timeout_sec))
                if (not file_path) or (not os.path.exists(file_path)):
                    raise RuntimeError("sendPhoto 파일 없음")
                with open(file_path, "rb") as fp:
                    resp = requests.post(url, data=data, files={"photo": fp}, timeout=timeout)
                    resp.raise_for_status()
            else:
                _tg_post(url, data, timeout_sec=min(float(HTTP_TIMEOUT_SEC), 10.0))
            backoff = 0.5
        except Exception as e:
            # 재시도는 제한적으로만(무한루프/스팸 방지)
            try:
                global _TG_SEND_LAST_ERR, _TG_SEND_LAST_ERR_KST
                _TG_SEND_LAST_ERR = str(e)[:500]
                _TG_SEND_LAST_ERR_KST = now_kst_str()
            except Exception:
                pass
            try:
                att = int(rec.get("attempt", 0) or 0) + 1 if isinstance(rec, dict) else 99
            except Exception:
                att = 99
            if isinstance(rec, dict) and att <= 2:
                try:
                    rec["attempt"] = att
                    pri = str(rec.get("priority", "normal")).lower()
                    with _TG_SEND_QUEUE_LOCK:
                        if pri == "high":
                            _TG_SEND_QUEUE_HIGH.appendleft(rec)
                        else:
                            _TG_SEND_QUEUE_NORMAL.appendleft(rec)
                except Exception:
                    pass
            time.sleep(float(clamp(backoff, 0.5, 8.0)))
            backoff = float(clamp(backoff * 1.4, 0.5, 10.0))


def tg_admin_chat_ids() -> List[str]:
    """
    Telegram Bot API에서 개인 DM의 chat_id는 보통 user_id와 동일합니다.
    - 단, 봇이 해당 사용자에게 DM을 보내려면 사용자가 먼저 봇을 시작(/start)해야 합니다.
    """
    try:
        if not TG_ADMIN_IDS:
            return []
        ids = []
        for x in sorted(list(TG_ADMIN_IDS)):
            try:
                ids.append(str(int(x)))
            except Exception:
                continue
        return ids
    except Exception:
        return []


def tg_send_chat(chat_id: Any, text: str):
    """특정 chat_id(채널/그룹/개인)로 직접 전송."""
    if not tg_token:
        return
    if chat_id is None:
        return
    cid = str(chat_id).strip()
    if not cid:
        return
    try:
        tg_enqueue("sendMessage", {"chat_id": cid, "text": text}, priority="normal")
    except Exception:
        pass


def _tg_chat_id_by_target(target: str, cfg: Dict[str, Any]) -> List[str]:
    target = (target or "default").lower()
    if target == "channel":
        return [tg_id_channel] if tg_id_channel else []
    if target == "group":
        return [tg_id_group] if tg_id_group else []
    if target == "admin":
        ids = tg_admin_chat_ids()
        if ids:
            return ids
        # fallback: 기존 동작(그룹/디폴트)
        if tg_id_group:
            return [tg_id_group]
        return [tg_id_default] if tg_id_default else []
    if target == "both":
        ids = []
        if tg_id_channel:
            ids.append(tg_id_channel)
        if tg_id_group and tg_id_group != tg_id_channel:
            ids.append(tg_id_group)
        return ids
    # default: 이전 동작 유지
    return [tg_id_default] if tg_id_default else []


def tg_send(text: str, target: str = "default", cfg: Optional[Dict[str, Any]] = None, *, silent: bool = False, parse_mode: str = ""):
    if not tg_token:
        return
    # 요구사항: Telegram 상태/라우팅이 전역 config가 아니라 최신 load_settings() 기준으로 일치
    cfg = cfg or load_settings()
    ids = _tg_chat_id_by_target(target, cfg)
    pri = "high" if str(target or "").lower().strip() == "admin" else "normal"
    for cid in ids:
        if not cid:
            continue
        try:
            data = {"chat_id": cid, "text": text}
            if bool(silent):
                data["disable_notification"] = True
            pm = str(parse_mode or "").strip()
            if pm:
                data["parse_mode"] = pm
            tg_enqueue("sendMessage", data, priority=pri)
        except Exception:
            pass


def tg_send_photo(photo_path: str, caption: str = "", target: str = "default", cfg: Optional[Dict[str, Any]] = None, *, silent: bool = False):
    if not tg_token:
        return
    path = str(photo_path or "").strip()
    if (not path) or (not os.path.exists(path)):
        return
    cfg = cfg or load_settings()
    ids = _tg_chat_id_by_target(target, cfg)
    pri = "high" if str(target or "").lower().strip() == "admin" else "normal"
    cap = str(caption or "").strip()
    if len(cap) > 1000:
        cap = cap[:1000]
    for cid in ids:
        if not cid:
            continue
        try:
            data: Dict[str, Any] = {"chat_id": cid, "__file_path": path}
            if cap:
                data["caption"] = cap
            if bool(silent):
                data["disable_notification"] = True
            tg_enqueue("sendPhoto", data, priority=pri)
        except Exception:
            continue


def tg_send_photo_chat(chat_id: Any, photo_path: str, caption: str = "", *, silent: bool = False):
    if not tg_token:
        return
    if chat_id is None:
        return
    path = str(photo_path or "").strip()
    if (not path) or (not os.path.exists(path)):
        return
    cid = str(chat_id).strip()
    if not cid:
        return
    try:
        data: Dict[str, Any] = {"chat_id": cid, "__file_path": path}
        cap = str(caption or "").strip()
        if cap:
            data["caption"] = cap[:1000]
        if bool(silent):
            data["disable_notification"] = True
        tg_enqueue("sendPhoto", data, priority="normal")
    except Exception:
        pass


def tg_send_menu(cfg: Optional[Dict[str, Any]] = None):
    if not tg_token:
        return
    cfg = cfg or load_settings()
    kb = {
        "inline_keyboard": [
            [{"text": "📡 상태", "callback_data": "status"}, {"text": "👁️ AI시야", "callback_data": "vision"}],
            [{"text": "📊 포지션", "callback_data": "position"}, {"text": "💰 잔고", "callback_data": "balance"}],
            [{"text": "📜 매매일지", "callback_data": "log"}, {"text": "🧾 일지상세", "callback_data": "log_detail_help"}],
            [{"text": "🔎 강제스캔", "callback_data": "scan"}, {"text": "🎚️ /mode", "callback_data": "mode_help"}],
            [{"text": "📎 시트", "callback_data": "gsheet"}, {"text": "🛑 전량청산", "callback_data": "close_all"}],
        ]
    }
    # ✅ 사용자의 요구: TG_TARGET_CHAT_ID는 채널로(알림/결과),
    #    관리/버튼은 TG_ADMIN_USER_IDS(관리자 DM)로 보내기.
    # - admin ids가 있으면 admin에게, 없으면 group(default)에게.
    to_ids = tg_admin_chat_ids() or ([tg_id_group] if tg_id_group else ([tg_id_default] if tg_id_default else []))
    if not to_ids:
        return
    try:
        for cid in to_ids:
            tg_enqueue(
                "sendMessage",
                {
                    "chat_id": cid,
                    "text": "✅ /menu\n/status /positions /scan /mode auto|scalping|swing /log <id> /gsheet\n(매매일지 버튼에서 금일/일별/월별 표 확인 가능)",
                    "reply_markup": json.dumps(kb, ensure_ascii=False),
                },
                priority="high",
            )
    except Exception:
        pass


def tg_answer_callback(cb_id: str):
    if not tg_token:
        return
    try:
        tg_enqueue("answerCallbackQuery", {"callback_query_id": cb_id}, priority="high")
    except Exception:
        pass


# =========================================================
# ✅ 16.1.5) Telegram 메시지: 쉬운말(핵심만) 포맷터 (요구사항)
# =========================================================
def _tg_simple_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    try:
        cfg = cfg or load_settings()
        return bool(cfg.get("tg_simple_messages", True))
    except Exception:
        return True


def _tg_fmt_pct(v: Any, digits: int = 2, signed: bool = True) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "-"
        return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"
    except Exception:
        return "-"


def _tg_fmt_usdt(v: Any, digits: int = 2, signed: bool = True) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "-"
        return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    except Exception:
        return "-"


def _tg_pct_compact(v: Any) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "-"
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        s = f"{x:.1f}"
        return s.rstrip("0").rstrip(".")
    except Exception:
        return "-"


def _tg_fmt_target_roi(v: Any, *, sign: str = "+", min_visible: float = 0.05) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "-"
        ax = abs(float(x))
        if ax < float(min_visible):
            return "-"
        sgn = "+" if str(sign) == "+" else "-"
        return f"{sgn}{ax:.2f}%"
    except Exception:
        return "-"


def _tg_trailing_protect_policy_line(cfg: Optional[Dict[str, Any]] = None) -> str:
    try:
        cfg = cfg or load_settings()
    except Exception:
        cfg = cfg or {}
    try:
        if not bool((cfg or {}).get("exit_trailing_protect_enable", False)):
            return ""
    except Exception:
        return ""
    sl_fixed = _as_float((cfg or {}).get("exit_trailing_protect_sl_roi", 15.0), 15.0)
    be_roi = _as_float((cfg or {}).get("exit_trailing_protect_be_roi", 10.0), 10.0)
    part_roi = _as_float((cfg or {}).get("exit_trailing_protect_partial_roi", 30.0), 30.0)
    part_pct = _as_float((cfg or {}).get("exit_trailing_protect_partial_close_pct", 50.0), 50.0)
    trail_start = _as_float((cfg or {}).get("exit_trailing_protect_trail_start_roi", 50.0), 50.0)
    trail_dd = _as_float((cfg or {}).get("exit_trailing_protect_trail_dd_roi", 10.0), 10.0)
    try:
        ai_prio = bool((cfg or {}).get("exit_trailing_protect_ai_targets_priority", False))
    except Exception:
        ai_prio = False
    prefix = "AI목표우선 + " if ai_prio else ""
    return (
        f"{prefix}수익보존(기본손절 -{_tg_pct_compact(abs(sl_fixed))}% | 본절 +{_tg_pct_compact(be_roi)}% | "
        f"부분익절 +{_tg_pct_compact(part_roi)}%({_tg_pct_compact(part_pct)}%) | "
        f"추적손절 +{_tg_pct_compact(trail_start)}%후 최고점-{_tg_pct_compact(trail_dd)}%)"
    )


def _tg_style_easy(style: str) -> str:
    s = str(style or "").strip()
    if s == "스캘핑":
        return "스캘핑"
    if s == "스윙":
        return "스윙"
    return s or "-"


def _tg_dir_easy(decision_or_side: str) -> str:
    d = str(decision_or_side or "").strip().lower()
    if d in ["buy", "long"]:
        return "롱"
    if d in ["sell", "short"]:
        return "숏"
    return "-"


def _tg_bal_line(
    before_total: Optional[float],
    after_total: Optional[float],
    before_free: Optional[float],
    after_free: Optional[float],
) -> str:
    try:
        bt = f"{float(before_total):.2f}" if before_total is not None else "-"
    except Exception:
        bt = "-"
    try:
        at = f"{float(after_total):.2f}" if after_total is not None else "-"
    except Exception:
        at = "-"
    try:
        bf = f"{float(before_free):.2f}" if before_free is not None else "-"
    except Exception:
        bf = "-"
    try:
        af = f"{float(after_free):.2f}" if after_free is not None else "-"
    except Exception:
        af = "-"
    return f"- 잔액(총/가용): {bt}→{at} / {bf}→{af} USDT"


def _tg_quote_block(text: str, prefix: str = "  └ ") -> str:
    """
    텔레그램 parse_mode 없이도 '인용/문단'처럼 보이게 만드는 간단 인덴트.
    """
    try:
        s = str(text or "").strip()
    except Exception:
        s = ""
    if not s:
        return ""
    lines: List[str] = []
    for ln in s.splitlines():
        ln2 = str(ln).strip()
        if not ln2:
            continue
        lines.append(ln2)
    if not lines:
        return ""
    # 너무 길면 2줄까지만(가독성/스팸 방지)
    out_lines: List[str] = []
    for ln in lines[:2]:
        out_lines.append(prefix + (ln[:180] + ("…" if len(ln) > 180 else "")))
    return "\n".join(out_lines)


def tg_msg_entry_simple(
    *,
    symbol: str,
    style: str,
    decision: str,
    lev: Any,
    entry_usdt: float,
    entry_pct_plan: Optional[float],
    tp_pct_roi: Optional[float],
    sl_pct_roi: Optional[float],
    bal_before_total: Optional[float],
    bal_after_total: Optional[float],
    bal_before_free: Optional[float],
    bal_after_free: Optional[float],
    one_line: str,
    trade_id: str,
    exit_policy_line: str = "",
) -> str:
    try:
        entry_usdt_f = float(entry_usdt)
    except Exception:
        entry_usdt_f = 0.0
    pct_free = None
    pct_total = None
    try:
        if bal_before_free is not None and float(bal_before_free) > 0:
            pct_free = (float(entry_usdt_f) / float(bal_before_free)) * 100.0
    except Exception:
        pct_free = None
    try:
        if bal_before_total is not None and float(bal_before_total) > 0:
            pct_total = (float(entry_usdt_f) / float(bal_before_total)) * 100.0
    except Exception:
        pct_total = None
    pct_txt = ""
    try:
        if pct_free is not None and math.isfinite(float(pct_free)) and pct_total is not None and math.isfinite(float(pct_total)):
            pct_txt = f" (가용 {float(pct_free):.1f}% / 총자산 {float(pct_total):.1f}%)"
        elif pct_free is not None and math.isfinite(float(pct_free)):
            pct_txt = f" (가용 {float(pct_free):.1f}%)"
        elif pct_total is not None and math.isfinite(float(pct_total)):
            pct_txt = f" (총자산 {float(pct_total):.1f}%)"
        elif entry_pct_plan is not None:
            pct0 = float(entry_pct_plan)
            if math.isfinite(float(pct0)):
                pct_txt = f" ({pct0:.1f}%)"
    except Exception:
        pct_txt = ""
    try:
        tp_v = float(tp_pct_roi) if tp_pct_roi is not None else None
    except Exception:
        tp_v = None
    try:
        sl_v = float(sl_pct_roi) if sl_pct_roi is not None else None
    except Exception:
        sl_v = None
    tp_txt = _tg_fmt_target_roi(tp_v, sign="+", min_visible=0.05) if tp_v is not None and math.isfinite(float(tp_v)) else "-"
    sl_txt = _tg_fmt_target_roi(sl_v, sign="-", min_visible=0.05) if sl_v is not None and math.isfinite(float(sl_v)) else "-"
    try:
        bf_txt = f"{float(bal_before_free):.2f}" if bal_before_free is not None else "-"
    except Exception:
        bf_txt = "-"
    try:
        af_txt = f"{float(bal_after_free):.2f}" if bal_after_free is not None else "-"
    except Exception:
        af_txt = "-"
    q = _tg_quote_block(one_line)
    if not q:
        q = "  └ -"
    target_label = "목표손익비(익절/손절)"
    return (
        "🎯 진입\n"
        f"- 코인: {symbol}\n"
        f"- 방식: {_tg_style_easy(style)}\n"
        f"- 포지션: {_tg_dir_easy(decision)}\n"
        f"- 레버리지: x{lev}\n"
        "\n"
        f"- 진입금액(마진): {entry_usdt_f:.2f} USDT{pct_txt}\n"
        f"- {target_label}: 익절 {tp_txt} / 손절 {sl_txt}\n"
        f"- 진입전 사용가능 금액: {bf_txt} USDT\n"
        f"- 진입후 사용가능 금액: {af_txt} USDT\n"
        "\n"
        "- 한줄:\n"
        f"{q}\n"
        f"- ID: {trade_id}"
    )


def tg_msg_exit_simple(
    *,
    title: str,
    symbol: str,
    style: str,
    side: str,
    lev: Any,
    roi_pct: float,
    pnl_usdt: float,
    contracts: float,
    bal_before_total: Optional[float],
    bal_after_total: Optional[float],
    bal_before_free: Optional[float],
    bal_after_free: Optional[float],
    one_line: str,
    trade_id: str,
) -> str:
    try:
        af_txt = f"{float(bal_after_free):.2f}" if bal_after_free is not None else "-"
    except Exception:
        af_txt = "-"
    try:
        at_txt = f"{float(bal_after_total):.2f}" if bal_after_total is not None else "-"
    except Exception:
        at_txt = "-"
    q = _tg_quote_block(one_line)
    if not q:
        q = "  └ -"
    return (
        f"{title}\n"
        f"- 코인: {symbol}\n"
        f"- 방식: {_tg_style_easy(style)}\n"
        f"- 포지션: {_tg_dir_easy(side)}\n"
        f"- 레버리지: x{lev}\n"
        "\n"
        f"- 결과: {_tg_fmt_pct(roi_pct)} (손익 {_tg_fmt_usdt(pnl_usdt)} USDT)\n"
        f"- 청산수량: {contracts}\n"
        f"- 청산후 사용가능 금액: {af_txt} USDT (총자산 {at_txt} USDT)\n"
        "\n"
        "- 한줄:\n"
        f"{q}\n"
        f"- ID: {trade_id}"
    )


# =========================================================
# ✅ 16.2) 오류 알림(관리자 DM) - 요구사항
# - "코드에서 나오는 모든 오류"를 TG_ADMIN_USER_IDS로 전송(스팸 방지용 dedup/쿨다운 포함)
# =========================================================
_ERR_NOTIFY_LOCK = threading.RLock()
_ERR_NOTIFY_LAST: Dict[str, float] = {}


def notify_admin_error(where: str, err: BaseException, context: Optional[Dict[str, Any]] = None, tb: str = "", min_interval_sec: float = 60.0):
    """
    안전한 오류 알림:
    - Telegram 전송 실패가 또 다른 예외를 만들지 않게 100% swallow
    - 동일 오류는 min_interval_sec 동안 중복 전송 방지
    """
    try:
        if not tg_token:
            return
        if not TG_ADMIN_IDS:
            return
        where_s = str(where or "unknown")[:120]
        msg_s = str(err)[:300]
        sig = f"{where_s}|{type(err).__name__}|{msg_s}"

        now = time.time()
        with _ERR_NOTIFY_LOCK:
            last = float(_ERR_NOTIFY_LAST.get(sig, 0) or 0)
            if (now - last) < float(min_interval_sec):
                return
            _ERR_NOTIFY_LAST[sig] = now
            # 메모리 누수 방지(최대 300개 유지)
            if len(_ERR_NOTIFY_LAST) > 300:
                # 오래된 것부터 제거
                for k in sorted(_ERR_NOTIFY_LAST, key=_ERR_NOTIFY_LAST.get)[:80]:
                    _ERR_NOTIFY_LAST.pop(k, None)

        tb_text = tb or ""
        if not tb_text:
            try:
                tb_text = traceback.format_exc()
            except Exception:
                tb_text = ""
        tb_short = ""
        if tb_text:
            try:
                tb_lines = tb_text.strip().splitlines()
                tb_short = "\n".join(tb_lines[-8:])
            except Exception:
                tb_short = ""

        ctx_txt = ""
        if context:
            try:
                ctx_txt = safe_json_dumps(context, limit=900)
            except Exception:
                ctx_txt = str(context)[:900]

        text = (
            f"🧨 오류 알림\n"
            f"- where: {where_s}\n"
            f"- time_kst: {now_kst_str()}\n"
            f"- code: {CODE_VERSION}\n"
            f"- error: {type(err).__name__}: {msg_s}\n"
        )
        if ctx_txt:
            text += f"- ctx: {ctx_txt}\n"
        if tb_short:
            text += f"- tb(last):\n{tb_short}\n"

        # Telegram 길이 제한 보호
        if len(text) > 3500:
            text = text[:3500] + "..."

        # 관리자 DM으로만 전송
        tg_send(text, target="admin", cfg=load_settings())
        try:
            # Google Sheets에도 ERROR 이벤트 남김(가능할 때만)
            # ✅ Google Sheets 자체 장애일 때는 무한 루프/스팸을 막기 위해 시트로 ERROR를 다시 쓰지 않는다.
            if not str(where_s).upper().startswith("GSHEET"):
                gsheet_log_event("ERROR", message=f"{where_s}: {type(err).__name__}", payload={"msg": msg_s, "ctx": context or {}})
        except Exception:
            pass
    except Exception:
        pass


# =========================================================
# ✅ 16.3) Global excepthook (best-effort)
# - 잡히지 않은 예외(특히 스레드)도 관리자 DM으로 전달
# =========================================================
def install_global_error_hooks():
    try:
        import sys as _sys
        import threading as _threading

        def _fmt_tb(exc_type, exc, tb_obj) -> str:
            try:
                return "".join(traceback.format_exception(exc_type, exc, tb_obj))
            except Exception:
                try:
                    return traceback.format_exc()
                except Exception:
                    return ""

        # sys.excepthook (메인 스레드 unhandled)
        def _sys_hook(exc_type, exc, tb_obj):  # type: ignore
            try:
                notify_admin_error("SYS_EXCEPTHOOK", exc, tb=_fmt_tb(exc_type, exc, tb_obj), min_interval_sec=10.0)
            except Exception:
                pass
            # 기본 훅도 호출(가능하면)
            try:
                _sys.__excepthook__(exc_type, exc, tb_obj)
            except Exception:
                pass

        _sys.excepthook = _sys_hook

        # threading.excepthook (Python 3.8+)
        if hasattr(_threading, "excepthook"):
            _orig_thread_hook = _threading.excepthook

            def _th_hook(args):  # type: ignore
                try:
                    where = f"THREAD_EXCEPTHOOK:{getattr(args.thread, 'name', '')}"
                    notify_admin_error(where, args.exc_value, tb=_fmt_tb(args.exc_type, args.exc_value, args.exc_traceback), min_interval_sec=10.0)
                except Exception:
                    pass
                try:
                    _orig_thread_hook(args)
                except Exception:
                    pass

            _threading.excepthook = _th_hook

    except Exception:
        pass


# =========================================================
# ✅ 16.5) Telegram Update Long Polling Thread (daemon)
# - 요구사항: getUpdates long polling을 별도 스레드로 수행(트레이딩 루프 멈춤 방지)
# =========================================================
_TG_UPDATES_QUEUE: List[Dict[str, Any]] = []
_TG_UPDATES_LOCK = threading.RLock()


def tg_updates_push(up: Dict[str, Any]) -> None:
    try:
        with _TG_UPDATES_LOCK:
            _TG_UPDATES_QUEUE.append(up)
            if len(_TG_UPDATES_QUEUE) > 400:
                _TG_UPDATES_QUEUE[:] = _TG_UPDATES_QUEUE[-300:]
    except Exception:
        pass


def tg_updates_pop_all(max_items: int = 50) -> List[Dict[str, Any]]:
    try:
        with _TG_UPDATES_LOCK:
            if not _TG_UPDATES_QUEUE:
                return []
            items = _TG_UPDATES_QUEUE[:max_items]
            del _TG_UPDATES_QUEUE[: len(items)]
        return items
    except Exception:
        return []


def telegram_polling_thread():
    """
    Telegram long polling(getUpdates).
    - TG_TOKEN 없으면 비활성
    - 네트워크 오류에도 지속 실행(backoff)
    """
    offset = 0
    backoff = 1.0
    start_epoch = time.time()
    last_ok_epoch = 0.0
    last_ok_kst = ""
    consec_fail = 0
    while True:
        if not tg_token:
            time.sleep(2.0)
            continue
        try:
            url = f"https://api.telegram.org/bot{tg_token}/getUpdates"
            # Telegram 서버 long-poll timeout(초). requests read timeout보다 작게 유지.
            params = {"offset": offset + 1, "timeout": 35}
            # long-poll은 read timeout을 넉넉히(텔레그램/네트워크 지연 대비), connect timeout은 짧게(장애 시 빠른 복구)
            r = requests.get(url, params=params, timeout=(6.0, 90.0))
            data = {}
            try:
                data = r.json()
            except Exception:
                data = {"ok": False}

            if data.get("ok"):
                backoff = 1.0
                consec_fail = 0
                last_ok_epoch = time.time()
                last_ok_kst = now_kst_str()
                for up in data.get("result", []) or []:
                    try:
                        offset = max(offset, int(up.get("update_id", offset)))
                    except Exception:
                        pass
                    tg_updates_push(up)
            else:
                consec_fail += 1
                time.sleep(0.4)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Telegram 네트워크 장애는 흔할 수 있어, "지속 장애"일 때만 관리자에게 알림(스팸 방지)
            consec_fail += 1
            try:
                base = float(last_ok_epoch) if last_ok_epoch else float(start_epoch)
                outage_sec = int(time.time() - base)
            except Exception:
                outage_sec = 0
            # ✅ 장시간 지속될 때만 알림(스팸 방지)
            # - ReadTimeout은 일시적으로 발생할 수 있어, 10분 이상 + 연속 실패 누적 시만
            if outage_sec >= 600 and consec_fail >= 10:
                notify_admin_error(
                    "TG_POLL_THREAD",
                    e,
                    context={"offset": offset, "outage_sec": outage_sec, "consecutive_fail": consec_fail, "last_ok_kst": last_ok_kst},
                    min_interval_sec=1800.0,
                )
            time.sleep(backoff)
            backoff = float(clamp(backoff * 1.5, 1.0, 30.0))
        except Exception as e:
            # 폴링 오류도 관리자에게 알림(과다 스팸 방지: 120s dedup)
            consec_fail += 1
            notify_admin_error(
                "TG_POLL_THREAD",
                e,
                context={"offset": offset, "consecutive_fail": consec_fail, "last_ok_kst": last_ok_kst},
                min_interval_sec=120.0,
            )
            time.sleep(backoff)
            backoff = float(clamp(backoff * 1.5, 1.0, 15.0))


def tg_is_admin(user_id: Optional[int]) -> bool:
    try:
        uid = int(user_id or 0)
    except Exception:
        uid = 0
    # TG_ADMIN_USER_IDS가 비어있으면 제한 없이 허용
    if not TG_ADMIN_IDS:
        return True
    return uid in TG_ADMIN_IDS


# =========================================================
# ✅ 17) 자동매매 핵심 스레드 (기능 유지 + 주기보고 + 스타일전환 + 안정성)
# =========================================================
def _fmt_pos_line(sym: str, side: str, lev: Any, roi: float, upnl: float, style: str = "") -> str:
    emo = "🟢" if roi >= 0 else "🔴"
    s_txt = f" | 스타일:{style}" if style else ""
    return f"{emo} {sym} {('롱' if side=='long' else '숏')} x{lev} | 수익률 {roi:.2f}% | 손익 {upnl:.2f} USDT{s_txt}"


def _norm_symbol_key(sym: Any) -> str:
    try:
        return "".join(ch for ch in str(sym).upper() if ch.isalnum())
    except Exception:
        return ""


def _resolve_open_target_for_symbol(
    sym: Any,
    active_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    rt_open_targets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sym_s = str(sym or "")
    srcs = []
    if isinstance(active_targets, dict):
        srcs.append(active_targets)
    if isinstance(rt_open_targets, dict):
        srcs.append(rt_open_targets)
    for src in srcs:
        try:
            t = src.get(sym_s, None)
            if isinstance(t, dict) and t:
                return t
        except Exception:
            pass
    nk = _norm_symbol_key(sym_s)
    if not nk:
        return {}
    for src in srcs:
        try:
            for k, v in src.items():
                if not isinstance(v, dict):
                    continue
                kk = _norm_symbol_key(k)
                if kk and (kk == nk or kk.endswith(nk) or nk.endswith(kk)):
                    return v
        except Exception:
            continue
    return {}


def _fmt_pos_block(
    sym: str,
    side: str,
    lev: Any,
    roi: float,
    upnl: float,
    style: str = "",
    tgt: Optional[Dict[str, Any]] = None,
) -> str:
    emo = "🟢" if roi >= 0 else "🔴"
    side_txt = "롱" if str(side) == "long" else "숏"
    tgt = tgt if isinstance(tgt, dict) else {}
    style_t = str(tgt.get("style", "") or "").strip()
    style_txt = str(style_t or style or "-").strip() or "-"
    try:
        lev_txt = f"x{lev}"
    except Exception:
        lev_txt = f"x{str(lev)}"
    tp_txt = _tg_fmt_target_roi(tgt.get("tp", None), sign="+", min_visible=0.05)
    sl_txt = _tg_fmt_target_roi(tgt.get("sl", None), sign="-", min_visible=0.05)
    rr_txt = "-"
    try:
        tp_v = float(_as_float(tgt.get("tp", None), float("nan")))
        sl_v = float(_as_float(tgt.get("sl", None), float("nan")))
        if math.isfinite(tp_v) and math.isfinite(sl_v) and abs(tp_v) >= 0.05 and abs(sl_v) >= 0.05:
            rr_txt = f"{(abs(tp_v) / max(abs(sl_v), 0.01)):.2f}"
    except Exception:
        rr_txt = "-"

    entry_line = "- 진입금액(마진): -"
    try:
        entry_usdt = float(_as_float(tgt.get("entry_usdt", None), float("nan")))
    except Exception:
        entry_usdt = float("nan")
    if math.isfinite(entry_usdt) and entry_usdt > 0:
        p_free = None
        p_total = None
        try:
            bal_free = float(_as_float(tgt.get("bal_entry_free", None), float("nan")))
            if math.isfinite(bal_free) and bal_free > 0:
                p_free = (entry_usdt / bal_free) * 100.0
        except Exception:
            p_free = None
        try:
            bal_total = float(_as_float(tgt.get("bal_entry_total", None), float("nan")))
            if math.isfinite(bal_total) and bal_total > 0:
                p_total = (entry_usdt / bal_total) * 100.0
        except Exception:
            p_total = None
        pct_parts = []
        if p_free is not None and math.isfinite(float(p_free)):
            pct_parts.append(f"가용 {float(p_free):.1f}%")
        if p_total is not None and math.isfinite(float(p_total)):
            pct_parts.append(f"총자산 {float(p_total):.1f}%")
        if (not pct_parts):
            try:
                p_plan = float(_as_float(tgt.get("entry_pct", None), float("nan")))
                if math.isfinite(p_plan) and p_plan > 0:
                    pct_parts.append(f"계획 {p_plan:.1f}%")
            except Exception:
                pass
        pct_txt = f" ({' / '.join(pct_parts)})" if pct_parts else ""
        entry_line = f"- 진입금액(마진): {entry_usdt:.2f} USDT{pct_txt}"

    target_line = f"- 목표(익절/손절): 익절 {tp_txt} / 손절 {sl_txt}"
    if rr_txt != "-":
        target_line += f" (RR {rr_txt})"
    if tp_txt == "-" and sl_txt == "-":
        target_line += " (목표 미동기화)"
    return (
        f"{emo} {sym}\n"
        f"- 방식: {style_txt}\n"
        f"- 포지션: {side_txt} | 레버리지: {lev_txt}\n"
        f"- 수익률: {_tg_fmt_pct(roi)} (손익 {_tg_fmt_usdt(upnl)} USDT)\n"
        f"{target_line}\n"
        f"{entry_line}"
    )


def tg_send_position_chart_images(
    ex,
    positions: List[Dict[str, Any]],
    active_targets: Dict[str, Dict[str, Any]],
    rt_open_targets: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    route_mode: str = "channel",
    admin_uid: Optional[int] = None,
    fallback_chat_id: Optional[int] = None,
) -> None:
    if not positions:
        return
    try:
        free_now, total_now = safe_fetch_balance(ex)
    except Exception:
        free_now, total_now = 0.0, 0.0
    for p in positions[:8]:
        try:
            sym = str(p.get("symbol", "") or "")
            if not sym:
                continue
            side = position_side_normalize(p)
            roi = float(position_roi_percent(p))
            upnl = float(p.get("unrealizedPnl") or 0.0)
            lev_live = _as_float(p.get("leverage", None), float("nan"))
            tgt0 = _resolve_open_target_for_symbol(sym, active_targets, rt_open_targets)
            style = str((tgt0 or {}).get("style", "") or "").strip() or "포지션"
            entry_px = _as_float((tgt0 or {}).get("entry_price", None), float("nan"))
            if not math.isfinite(entry_px) or entry_px <= 0:
                entry_px = _as_float(p.get("entryPrice", None), float("nan"))
            lev0 = _as_float((tgt0 or {}).get("lev", None), float("nan"))
            lev_use = lev_live if math.isfinite(lev_live) else lev0
            one_line = str((tgt0 or {}).get("reason", "") or (tgt0 or {}).get("style_reason", "") or "").strip()
            img_path = build_trade_event_image(
                ex,
                sym,
                cfg,
                event_type="POSITION",
                side=str(side),
                style=str(style),
                entry_price=(float(entry_px) if math.isfinite(entry_px) and entry_px > 0 else None),
                sl_price=(_as_float((tgt0 or {}).get("sl_price", None), float("nan")) if (tgt0 or {}).get("sl_price", None) is not None else None),
                tp_price=(_as_float((tgt0 or {}).get("tp_price", None), float("nan")) if (tgt0 or {}).get("tp_price", None) is not None else None),
                partial_tp1_price=(_as_float((tgt0 or {}).get("partial_tp1_price", None), float("nan")) if (tgt0 or {}).get("partial_tp1_price", None) is not None else None),
                partial_tp2_price=(_as_float((tgt0 or {}).get("partial_tp2_price", None), float("nan")) if (tgt0 or {}).get("partial_tp2_price", None) is not None else None),
                dca_price=(_as_float((tgt0 or {}).get("dca_price", None), float("nan")) if (tgt0 or {}).get("dca_price", None) is not None else None),
                sl_roi_pct=(_as_float((tgt0 or {}).get("sl", None), float("nan")) if (tgt0 or {}).get("sl", None) is not None else None),
                tp_roi_pct=(_as_float((tgt0 or {}).get("tp", None), float("nan")) if (tgt0 or {}).get("tp", None) is not None else None),
                leverage=(float(lev_use) if math.isfinite(lev_use) else None),
                roi_pct=float(roi),
                pnl_usdt=float(upnl),
                remain_free=(float(free_now) if math.isfinite(float(free_now)) and float(free_now) > 0 else None),
                remain_total=(float(total_now) if math.isfinite(float(total_now)) and float(total_now) > 0 else None),
                one_line=one_line,
                used_indicators=[],
                pattern_hint="",
                mtf_pattern={},
                trade_id=str((tgt0 or {}).get("trade_id", "") or ""),
            )
            if not img_path:
                continue
            cap = (
                "📷 포지션 차트\n"
                f"- {sym} | {_tg_style_easy(style)} | {_tg_dir_easy(side)}\n"
                f"- 현재: {_tg_fmt_pct(roi)} ({_tg_fmt_usdt(upnl)} USDT)"
            )
            how = str(route_mode or "channel").lower().strip()
            if how == "both":
                tg_send_photo(img_path, caption=cap, target="channel", cfg=cfg, silent=False)
                if admin_uid is not None:
                    tg_send_photo_chat(admin_uid, img_path, caption=cap, silent=False)
                elif TG_ADMIN_IDS:
                    tg_send_photo(img_path, caption=cap, target="admin", cfg=cfg, silent=False)
                elif fallback_chat_id is not None:
                    tg_send_photo_chat(fallback_chat_id, img_path, caption=cap, silent=False)
            elif how == "admin":
                if admin_uid is not None:
                    tg_send_photo_chat(admin_uid, img_path, caption=cap, silent=False)
                elif TG_ADMIN_IDS:
                    tg_send_photo(img_path, caption=cap, target="admin", cfg=cfg, silent=False)
                elif fallback_chat_id is not None:
                    tg_send_photo_chat(fallback_chat_id, img_path, caption=cap, silent=False)
                else:
                    tg_send_photo(img_path, caption=cap, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg, silent=False)
            else:
                tg_send_photo(img_path, caption=cap, target="channel", cfg=cfg, silent=False)
        except Exception:
            continue


def _style_for_entry(
    symbol: str,
    decision: str,
    trend_short: str,
    trend_long: str,
    cfg: Dict[str, Any],
    allow_ai: bool = True,
) -> Dict[str, Any]:
    style, conf, reason = decide_style_rule_based(decision, trend_short, trend_long)
    # 애매하면 AI로 2차 판단
    if allow_ai and cfg.get("style_auto_enable", True) and conf <= 60:
        ai = ai_decide_style(symbol, decision, trend_short, trend_long, cfg)
        # AI가 스윙이라고 강하게 말하면 반영
        if int(ai.get("confidence", 0)) >= 70:
            style = ai.get("style", style)
            conf = int(ai.get("confidence", conf))
            reason = str(ai.get("reason", reason))
    return {"style": style, "confidence": conf, "reason": reason}


def _trend_align(trend_txt: str, side: str) -> bool:
    if side == "long":
        return "상승" in (trend_txt or "")
    if side == "short":
        return "하락" in (trend_txt or "")
    return False


def _trend_clean_for_reason(trend_txt: Any) -> str:
    try:
        s = str(trend_txt or "").strip()
    except Exception:
        return ""
    if not s:
        return ""
    # 예: "📈 상승추세" / "🧭 1h 상승추세" -> "상승추세"
    try:
        s = s.replace("📈", "").replace("🧭", "").strip()
        # 선행 이모지/기호 제거
        s = re.sub(r"^[^0-9A-Za-z가-힣]+", "", s).strip()
        # "1h 상승추세" -> "상승추세" (타임프레임 제거)
        s = re.sub(r"^(?:\\d+[mhdw]|\\d+h)\\s+", "", s).strip()
    except Exception:
        pass
    return s


def _rsi_state_ko(rsi: Optional[float], cfg: Dict[str, Any]) -> str:
    try:
        if rsi is None:
            return ""
        v = float(rsi)
        if not math.isfinite(v):
            return ""
    except Exception:
        return ""
    try:
        rsi_buy = float(cfg.get("rsi_buy", 30) or 30)
        rsi_sell = float(cfg.get("rsi_sell", 70) or 70)
    except Exception:
        rsi_buy, rsi_sell = 30.0, 70.0
    if v <= rsi_buy:
        return "과매도"
    if v >= rsi_sell:
        return "과매수"
    return "중립"


def _cleanup_event_images(max_files: int = 700, keep_hours: int = 48) -> None:
    try:
        if not os.path.isdir(EVENT_IMAGE_DIR):
            return
        files = [os.path.join(EVENT_IMAGE_DIR, f) for f in os.listdir(EVENT_IMAGE_DIR) if f.lower().endswith(".png")]
        if not files:
            return
        now_ts = time.time()
        old_sec = max(1, int(keep_hours)) * 3600
        for p in files:
            try:
                if (now_ts - os.path.getmtime(p)) > old_sec:
                    os.remove(p)
            except Exception:
                continue
        files = [os.path.join(EVENT_IMAGE_DIR, f) for f in os.listdir(EVENT_IMAGE_DIR) if f.lower().endswith(".png")]
        if len(files) <= int(max_files):
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        for p in files[: max(0, len(files) - int(max_files))]:
            try:
                os.remove(p)
            except Exception:
                continue
    except Exception:
        pass


def _draw_candles_simple(ax, df: pd.DataFrame) -> None:
    if plt is None or mdates is None or Rectangle is None:
        return
    if df is None or df.empty:
        return
    d = df.copy()
    if "time" not in d.columns:
        return
    try:
        d["time"] = pd.to_datetime(d["time"])
    except Exception:
        return
    try:
        d["open"] = pd.to_numeric(d["open"], errors="coerce")
        d["high"] = pd.to_numeric(d["high"], errors="coerce")
        d["low"] = pd.to_numeric(d["low"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
    except Exception:
        return
    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    if d.empty:
        return
    x = mdates.date2num(d["time"].dt.to_pydatetime())
    if len(x) <= 1:
        width = 0.00045
    else:
        dx = float(np.median(np.diff(x)))
        width = float(max(0.0002, dx * 0.65))
    up = "#00d084"
    dn = "#ff4d4f"
    for i in range(len(d)):
        xx = x[i]
        oo = float(d["open"].iloc[i])
        hh = float(d["high"].iloc[i])
        ll = float(d["low"].iloc[i])
        cc = float(d["close"].iloc[i])
        color = up if cc >= oo else dn
        ax.vlines(xx, ll, hh, color=color, linewidth=0.8, alpha=0.95)
        y0 = min(oo, cc)
        h0 = max(abs(cc - oo), 1e-9)
        rect = Rectangle((xx - width / 2.0, y0), width, h0, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.95)
        ax.add_patch(rect)
    ax.xaxis_date()
    ax.grid(True, color="#2d3238", linestyle="--", linewidth=0.5, alpha=0.35)


_TRADE_IMG_FONT_READY = False
_TRADE_IMG_FONT_OK = False
_TRADE_IMG_FONT_NAME = ""


def _register_packaged_korean_fonts() -> None:
    if mfont is None or _koreanize_matplotlib is None:
        return
    try:
        font_dir = os.path.join(os.path.dirname(str(_koreanize_matplotlib.__file__ or "")), "fonts")
        if not os.path.isdir(font_dir):
            return
        for fn in os.listdir(font_dir):
            if not str(fn).lower().endswith((".ttf", ".otf")):
                continue
            fp = os.path.join(font_dir, fn)
            try:
                mfont.fontManager.addfont(fp)
            except Exception:
                continue
    except Exception:
        pass


def _ensure_trade_image_font() -> bool:
    global _TRADE_IMG_FONT_READY, _TRADE_IMG_FONT_OK, _TRADE_IMG_FONT_NAME
    if plt is None:
        return False
    if _TRADE_IMG_FONT_READY:
        return bool(_TRADE_IMG_FONT_OK)
    _TRADE_IMG_FONT_READY = True
    _TRADE_IMG_FONT_OK = False
    _TRADE_IMG_FONT_NAME = ""
    if mfont is None:
        try:
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        return False
    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "NotoSansCJKkr",
        "Malgun Gothic",
        "AppleGothic",
        "Apple SD Gothic Neo",
    ]
    found_name = ""
    try:
        names = [str(getattr(f, "name", "") or "") for f in (mfont.fontManager.ttflist or [])]
        name_set = {n.lower(): n for n in names if n}
        for cand in preferred:
            if cand.lower() in name_set:
                found_name = name_set[cand.lower()]
                break
        if not found_name:
            hint_words = ["nanum", "notosanscjk", "malgun", "applegothic", "apple sd gothic", "noto sans kr", "noto sans cjk"]
            for p in (mfont.findSystemFonts(fontpaths=None, fontext="ttf") or [])[:2000]:
                n = os.path.basename(str(p)).lower()
                if any(w in n for w in hint_words):
                    try:
                        mfont.fontManager.addfont(p)
                    except Exception:
                        continue
            names2 = [str(getattr(f, "name", "") or "") for f in (mfont.fontManager.ttflist or [])]
            name_set2 = {n.lower(): n for n in names2 if n}
            for cand in preferred:
                if cand.lower() in name_set2:
                    found_name = name_set2[cand.lower()]
                    break
        if not found_name:
            _register_packaged_korean_fonts()
            names3 = [str(getattr(f, "name", "") or "") for f in (mfont.fontManager.ttflist or [])]
            name_set3 = {n.lower(): n for n in names3 if n}
            for cand in preferred:
                if cand.lower() in name_set3:
                    found_name = name_set3[cand.lower()]
                    break
    except Exception:
        found_name = ""
    try:
        plt.rcParams["axes.unicode_minus"] = False
        if found_name:
            plt.rcParams["font.family"] = [found_name]
            _TRADE_IMG_FONT_OK = True
            _TRADE_IMG_FONT_NAME = str(found_name)
        else:
            plt.rcParams["font.family"] = ["DejaVu Sans"]
            _TRADE_IMG_FONT_OK = False
            _TRADE_IMG_FONT_NAME = "DejaVu Sans"
    except Exception:
        _TRADE_IMG_FONT_OK = False
    return bool(_TRADE_IMG_FONT_OK)


def _price_from_roi_target(entry_price: float, side: str, roi_pct: float, leverage: float, kind: str) -> Optional[float]:
    try:
        px = float(entry_price)
        lev = float(leverage)
        roi = abs(float(roi_pct))
        if px <= 0 or lev <= 0 or roi <= 0:
            return None
        move = (roi / lev) / 100.0
        s = str(side or "").lower().strip()
        if s in ["buy", "long"]:
            if kind == "tp":
                return px * (1.0 + move)
            return px * (1.0 - move)
        if kind == "tp":
            return px * (1.0 - move)
        return px * (1.0 + move)
    except Exception:
        return None


def _plot_text_sanitize(text: Any, *, has_kr_font: bool, max_len: int = 220) -> str:
    try:
        s = str(text or "").replace("\n", " ").strip()
    except Exception:
        return ""
    if not s:
        return ""
    if not has_kr_font:
        s = re.sub(r"[^\x20-\x7E]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    if len(s) > int(max_len):
        return s[: int(max_len)] + "…"
    return s


def _style_plot_label(style: str, has_kr_font: bool) -> str:
    s = str(style or "").strip()
    if has_kr_font:
        return s or "-"
    if s == "스캘핑":
        return "SCALP"
    if s == "스윙":
        return "SWING"
    s2 = _plot_text_sanitize(s, has_kr_font=False, max_len=24)
    return s2 or "STYLE"


def _pivot_indices_for_plot(high: np.ndarray, low: np.ndarray, order: int) -> Tuple[List[int], List[int]]:
    hi_idx: List[int] = []
    lo_idx: List[int] = []
    try:
        if len(high) < max(10, order * 4):
            return hi_idx, lo_idx
        if argrelextrema is not None:
            try:
                hi_idx = [int(x) for x in argrelextrema(high, np.greater_equal, order=order)[0].tolist()]
                lo_idx = [int(x) for x in argrelextrema(low, np.less_equal, order=order)[0].tolist()]
                return hi_idx, lo_idx
            except Exception:
                hi_idx, lo_idx = [], []
        for i in range(order, len(high) - order):
            try:
                if high[i] >= np.max(high[i - order : i + order + 1]):
                    hi_idx.append(int(i))
                if low[i] <= np.min(low[i - order : i + order + 1]):
                    lo_idx.append(int(i))
            except Exception:
                continue
    except Exception:
        pass
    return hi_idx, lo_idx


def _draw_pattern_overlay(
    ax,
    d: pd.DataFrame,
    cfg: Dict[str, Any],
    detected_patterns: List[str],
    has_kr_font: bool,
) -> None:
    try:
        if d is None or d.empty:
            return
        if not bool(cfg.get("tg_image_show_pattern_overlay", True)):
            return
        high = d["high"].astype(float).values
        low = d["low"].astype(float).values
        close = d["close"].astype(float).values
        time_vals = pd.to_datetime(d["time"])
        x_num = mdates.date2num(time_vals.dt.to_pydatetime())
        order = int(cfg.get("pattern_pivot_order", 4) or 4)
        order = int(clamp(order, 3, 12))
        hi_idx, lo_idx = _pivot_indices_for_plot(high, low, order=order)
        if not hi_idx and not lo_idx:
            return

        pats = [str(p or "") for p in (detected_patterns or []) if str(p or "").strip()]
        ptxt = " | ".join(pats)

        def _draw_points(idx_list: List[int], price_arr: np.ndarray, color: str, label: str):
            if not idx_list:
                return
            xs = [float(x_num[i]) for i in idx_list if 0 <= int(i) < len(x_num)]
            ys = [float(price_arr[i]) for i in idx_list if 0 <= int(i) < len(price_arr)]
            if not xs or not ys:
                return
            ax.scatter(xs, ys, s=28, color=color, edgecolors="#111827", linewidths=0.4, zorder=9, alpha=0.95)
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.45, zorder=8)
            try:
                ax.text(
                    xs[-1],
                    ys[-1],
                    label,
                    color=color,
                    fontsize=8.2,
                    ha="left",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "#0b1220", "edgecolor": color, "alpha": 0.75},
                )
            except Exception:
                pass

        # M/W/헤드앤숄더 류 위치 표시
        if ("쌍바닥" in ptxt) or ("Double Bottom" in ptxt):
            _draw_points(sorted(lo_idx)[-2:], low, "#22c55e", "W")
        if ("쌍봉" in ptxt) or ("Double Top" in ptxt):
            _draw_points(sorted(hi_idx)[-2:], high, "#ef4444", "M")
        if ("삼중바닥" in ptxt) or ("Triple Bottom" in ptxt):
            _draw_points(sorted(lo_idx)[-3:], low, "#10b981", "3B")
        if ("삼중천정" in ptxt) or ("Triple Top" in ptxt):
            _draw_points(sorted(hi_idx)[-3:], high, "#f97316", "3T")
        if ("헤드앤숄더" in ptxt) and ("역헤드앤숄더" not in ptxt):
            _draw_points(sorted(hi_idx)[-3:], high, "#fb7185", "H&S")
        if "역헤드앤숄더" in ptxt:
            _draw_points(sorted(lo_idx)[-3:], low, "#34d399", "iH&S")

        # 삼각/쐐기/박스 위치(최근 피벗 추세선 근사)
        if ("삼각" in ptxt) or ("쐐기" in ptxt) or ("박스권" in ptxt):
            hi_recent = sorted(hi_idx)[-6:]
            lo_recent = sorted(lo_idx)[-6:]
            if len(hi_recent) >= 3 and len(lo_recent) >= 3:
                xh = np.asarray(hi_recent, dtype=float)
                yh = np.asarray([float(high[i]) for i in hi_recent], dtype=float)
                xl = np.asarray(lo_recent, dtype=float)
                yl = np.asarray([float(low[i]) for i in lo_recent], dtype=float)
                try:
                    sh, ih = np.polyfit(xh, yh, 1)
                    sl, il = np.polyfit(xl, yl, 1)
                    xx = np.arange(max(0, len(close) - 80), len(close), dtype=float)
                    y_top = sh * xx + ih
                    y_bot = sl * xx + il
                    ax.plot(mdates.date2num(time_vals.iloc[xx.astype(int)].dt.to_pydatetime()), y_top, color="#f59e0b", linewidth=1.0, linestyle="--", alpha=0.55)
                    ax.plot(mdates.date2num(time_vals.iloc[xx.astype(int)].dt.to_pydatetime()), y_bot, color="#38bdf8", linewidth=1.0, linestyle="--", alpha=0.55)
                except Exception:
                    pass

        # 패턴 요약 라벨(한글 폰트 없으면 영어/ASCII만)
        p_label = _plot_text_sanitize(", ".join(pats[:3]), has_kr_font=has_kr_font, max_len=80)
        if p_label:
            try:
                ax.text(
                    0.99,
                    0.99,
                    f"Pattern: {p_label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.0,
                    color="#e2e8f0",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "#111827", "edgecolor": "#374151", "alpha": 0.72},
                )
            except Exception:
                pass
    except Exception:
        pass


def build_trade_event_image(
    ex,
    sym: str,
    cfg: Dict[str, Any],
    *,
    event_type: str,
    side: str,
    style: str,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    partial_tp1_price: Optional[float] = None,
    partial_tp2_price: Optional[float] = None,
    dca_price: Optional[float] = None,
    sl_roi_pct: Optional[float] = None,
    tp_roi_pct: Optional[float] = None,
    leverage: Optional[float] = None,
    roi_pct: Optional[float] = None,
    pnl_usdt: Optional[float] = None,
    remain_free: Optional[float] = None,
    remain_total: Optional[float] = None,
    one_line: str = "",
    used_indicators: Optional[List[str]] = None,
    pattern_hint: str = "",
    mtf_pattern: Optional[Dict[str, Any]] = None,
    trade_id: str = "",
) -> Optional[str]:
    if plt is None or mdates is None or Rectangle is None:
        return None
    try:
        tf = str(cfg.get("timeframe", "5m") or "5m")
        bars = int(cfg.get("tg_image_chart_bars", 140) or 140)
        bars = int(clamp(bars, 80, 500))
        ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=max(120, bars))
        if not ohlcv or len(ohlcv) < 50:
            return None
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        d = df.tail(bars).copy()
        if d.empty:
            return None

        has_kr_font = _ensure_trade_image_font()
        show_indicator_panels = bool(cfg.get("tg_image_show_indicators", True))
        if show_indicator_panels:
            fig, axes = plt.subplots(
                nrows=4,
                ncols=1,
                sharex=True,
                figsize=(13.2, 8.8),
                dpi=120,
                gridspec_kw={"height_ratios": [5.0, 1.35, 1.55, 1.35], "hspace": 0.05},
            )
            ax = axes[0]
            ax_rsi = axes[1]
            ax_macd = axes[2]
            ax_sqz = axes[3]
            panel_axes = [ax_rsi, ax_macd, ax_sqz]
        else:
            fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=120)
            ax_rsi = None
            ax_macd = None
            ax_sqz = None
            panel_axes = []
        fig.patch.set_facecolor("#0e1117")
        for axx in [ax] + panel_axes:
            axx.set_facecolor("#11161c")
            axx.tick_params(colors="#cfd8dc", labelsize=8)
            for spine in axx.spines.values():
                spine.set_color("#263238")
                spine.set_linewidth(0.8)
        _draw_candles_simple(ax, d)

        panel_df = pd.DataFrame()
        try:
            d2, _, _ = calc_indicators(df.copy(), cfg)
            if isinstance(d2, pd.DataFrame) and not d2.empty:
                panel_df = d2.copy()
                panel_df["time"] = pd.to_datetime(panel_df["time"])
                panel_df = panel_df[panel_df["time"] >= pd.to_datetime(d["time"].iloc[0])].copy()
        except Exception:
            panel_df = pd.DataFrame()

        now_px = float(d["close"].iloc[-1])
        sl_line = float(sl_price) if (sl_price is not None and math.isfinite(float(sl_price))) else None
        tp_line = float(tp_price) if (tp_price is not None and math.isfinite(float(tp_price))) else None
        pt1_line = float(partial_tp1_price) if (partial_tp1_price is not None and math.isfinite(float(partial_tp1_price))) else None
        pt2_line = float(partial_tp2_price) if (partial_tp2_price is not None and math.isfinite(float(partial_tp2_price))) else None
        dca_line = float(dca_price) if (dca_price is not None and math.isfinite(float(dca_price))) else None
        lev_line = float(leverage) if (leverage is not None and math.isfinite(float(leverage))) else None
        ep = float(entry_price) if (entry_price is not None and math.isfinite(float(entry_price))) else None
        if sl_line is None and ep is not None and sl_roi_pct is not None and lev_line is not None and lev_line > 0:
            sl_line = _price_from_roi_target(ep, str(side), float(sl_roi_pct), float(lev_line), "sl")
        if tp_line is None and ep is not None and tp_roi_pct is not None and lev_line is not None and lev_line > 0:
            tp_line = _price_from_roi_target(ep, str(side), float(tp_roi_pct), float(lev_line), "tp")

        try:
            sr_params = _sr_params_for_style(str(style), cfg)
            sr_tf = str(sr_params.get("tf", cfg.get("sr_timeframe", "15m")) or "15m")
            sr_lb = int(sr_params.get("lookback", cfg.get("sr_lookback", 220)) or 220)
            sr_po = int(sr_params.get("pivot_order", cfg.get("sr_pivot_order", 6)) or 6)
            sr_cache = int(cfg.get("sr_levels_cache_sec", 60) or 60)
            sr_levels = get_sr_levels_cached(ex, sym, sr_tf, pivot_order=sr_po, cache_sec=sr_cache, limit=sr_lb)
            supports = [float(x) for x in (sr_levels.get("supports") or []) if math.isfinite(float(x))]
            resistances = [float(x) for x in (sr_levels.get("resistances") or []) if math.isfinite(float(x))]
            n_sr = int(cfg.get("tg_image_sr_lines", 3) or 3)
            n_sr = int(clamp(n_sr, 1, 6))
            sup_near = sorted([x for x in supports if x <= now_px], reverse=True)[:n_sr]
            res_near = sorted([x for x in resistances if x >= now_px])[:n_sr]
            for lv in sup_near:
                ax.axhline(float(lv), color="#64b5f6", linewidth=0.8, linestyle=":", alpha=0.55)
            for lv in res_near:
                ax.axhline(float(lv), color="#ff8a65", linewidth=0.8, linestyle=":", alpha=0.55)
        except Exception:
            pass

        try:
            vp_n = int(cfg.get("tg_image_volume_nodes", 4) or 4)
            vp_n = int(clamp(vp_n, 0, 8))
            if vp_n > 0:
                vp_nodes = volume_profile_nodes(d, bins=60, top_n=vp_n)
                for lv in (vp_nodes or [])[:vp_n]:
                    if math.isfinite(float(lv)):
                        ax.axhline(float(lv), color="#b388ff", linewidth=0.7, linestyle="-.", alpha=0.40)
        except Exception:
            pass

        detected_patterns: List[str] = []
        if bool(cfg.get("use_chart_patterns", True)):
            try:
                pat_src = panel_df if (isinstance(panel_df, pd.DataFrame) and not panel_df.empty) else d
                pat = detect_chart_patterns(pat_src.copy(), cfg)
                detected_patterns = [str(x) for x in (pat or {}).get("detected", []) if str(x).strip()]
            except Exception:
                detected_patterns = []
        if not detected_patterns and str(pattern_hint or "").strip():
            try:
                tmp = str(pattern_hint or "")
                for sep in ["|", "/", ",", ";"]:
                    tmp = tmp.replace(sep, ",")
                detected_patterns = [x.strip() for x in tmp.split(",") if x.strip()]
            except Exception:
                detected_patterns = []
        _draw_pattern_overlay(ax, d, cfg, detected_patterns, has_kr_font)

        ax.axhline(float(now_px), color="#a0aec0", linewidth=1.0, alpha=0.35, linestyle=":")
        if ep is not None:
            ax.axhline(float(ep), color="#00bcd4", linewidth=1.4, linestyle="-", alpha=0.98)
        if exit_price is not None and math.isfinite(float(exit_price)):
            ax.axhline(float(exit_price), color="#ffd166", linewidth=1.3, linestyle="-", alpha=0.98)
        if sl_line is not None:
            ax.axhline(float(sl_line), color="#ff4d4f", linewidth=1.8, linestyle="--", alpha=0.98)
        if tp_line is not None:
            ax.axhline(float(tp_line), color="#00d084", linewidth=1.8, linestyle="--", alpha=0.98)
        if pt1_line is not None:
            ax.axhline(float(pt1_line), color="#2dd4bf", linewidth=1.2, linestyle="-.", alpha=0.90)
        if pt2_line is not None:
            ax.axhline(float(pt2_line), color="#14b8a6", linewidth=1.2, linestyle="-.", alpha=0.90)
        if dca_line is not None:
            ax.axhline(float(dca_line), color="#f59e0b", linewidth=1.2, linestyle=":", alpha=0.95)

        if mtransforms is not None:
            try:
                trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
                if tp_line is not None:
                    lbl = f"목표익절 {float(tp_line):.6g}" if has_kr_font else f"TP {float(tp_line):.6g}"
                    ax.text(
                        0.995,
                        float(tp_line),
                        lbl,
                        transform=trans,
                        va="center",
                        ha="right",
                        fontsize=8.2,
                        color="#a7f3d0",
                        bbox={"boxstyle": "round,pad=0.20", "facecolor": "#06241a", "edgecolor": "#0f5132", "alpha": 0.90},
                    )
                if sl_line is not None:
                    lbl = f"목표손절 {float(sl_line):.6g}" if has_kr_font else f"SL {float(sl_line):.6g}"
                    ax.text(
                        0.995,
                        float(sl_line),
                        lbl,
                        transform=trans,
                        va="center",
                        ha="right",
                        fontsize=8.2,
                        color="#fecaca",
                        bbox={"boxstyle": "round,pad=0.20", "facecolor": "#2a0b11", "edgecolor": "#7f1d1d", "alpha": 0.90},
                    )
                if pt1_line is not None:
                    lbl = f"분할익절1 {float(pt1_line):.6g}" if has_kr_font else f"PT1 {float(pt1_line):.6g}"
                    ax.text(
                        0.995,
                        float(pt1_line),
                        lbl,
                        transform=trans,
                        va="center",
                        ha="right",
                        fontsize=8.0,
                        color="#99f6e4",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#032c2a", "edgecolor": "#115e59", "alpha": 0.88},
                    )
                if pt2_line is not None:
                    lbl = f"분할익절2 {float(pt2_line):.6g}" if has_kr_font else f"PT2 {float(pt2_line):.6g}"
                    ax.text(
                        0.995,
                        float(pt2_line),
                        lbl,
                        transform=trans,
                        va="center",
                        ha="right",
                        fontsize=8.0,
                        color="#ccfbf1",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#042f2e", "edgecolor": "#0f766e", "alpha": 0.88},
                    )
                if dca_line is not None:
                    lbl = f"추매라인 {float(dca_line):.6g}" if has_kr_font else f"DCA {float(dca_line):.6g}"
                    ax.text(
                        0.995,
                        float(dca_line),
                        lbl,
                        transform=trans,
                        va="center",
                        ha="right",
                        fontsize=8.0,
                        color="#fde68a",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#2d1b05", "edgecolor": "#92400e", "alpha": 0.88},
                    )
            except Exception:
                pass

        if show_indicator_panels and (ax_rsi is not None) and (ax_macd is not None) and (ax_sqz is not None):
            px_df = panel_df if (isinstance(panel_df, pd.DataFrame) and not panel_df.empty) else d
            try:
                px_df = px_df.copy()
                px_df["time"] = pd.to_datetime(px_df["time"])
                x = mdates.date2num(px_df["time"].dt.to_pydatetime())
            except Exception:
                x = np.array([], dtype=float)
                px_df = pd.DataFrame()

            for axp in [ax_rsi, ax_macd, ax_sqz]:
                axp.grid(True, color="#2d3238", linestyle="--", linewidth=0.45, alpha=0.32)
                axp.tick_params(colors="#94a3b8", labelsize=7.8)

            if (not px_df.empty) and ("RSI" in px_df.columns) and len(x) == len(px_df):
                rsi = pd.to_numeric(px_df["RSI"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(rsi)
                if valid.any():
                    ax_rsi.plot(x[valid], rsi[valid], color="#f59e0b", linewidth=1.2, alpha=0.95)
                rsi_buy = float(cfg.get("rsi_buy", 30) or 30)
                rsi_sell = float(cfg.get("rsi_sell", 70) or 70)
                ax_rsi.axhline(rsi_buy, color="#22c55e", linestyle="--", linewidth=0.9, alpha=0.8)
                ax_rsi.axhline(rsi_sell, color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.8)
                ax_rsi.set_ylim(0, 100)
                try:
                    rv = float(rsi[np.isfinite(rsi)][-1])
                    rstate = "중립"
                    if rv <= rsi_buy:
                        rstate = "과매도"
                    elif rv >= rsi_sell:
                        rstate = "과매수"
                    rtxt = f"RSI {rv:.1f} ({rstate})" if has_kr_font else f"RSI {rv:.1f}"
                    ax_rsi.text(
                        0.995,
                        0.82,
                        rtxt,
                        transform=ax_rsi.transAxes,
                        ha="right",
                        va="center",
                        fontsize=7.8,
                        color="#fde68a",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#1f2937", "edgecolor": "#374151", "alpha": 0.86},
                    )
                except Exception:
                    pass
            else:
                ax_rsi.text(0.01, 0.6, "RSI N/A", transform=ax_rsi.transAxes, color="#94a3b8", fontsize=7.6)

            if (not px_df.empty) and all(c in px_df.columns for c in ["MACD", "MACD_signal"]) and len(x) == len(px_df):
                macd = pd.to_numeric(px_df["MACD"], errors="coerce").to_numpy(dtype=float)
                sig = pd.to_numeric(px_df["MACD_signal"], errors="coerce").to_numpy(dtype=float)
                hist = macd - sig
                valid = np.isfinite(macd) & np.isfinite(sig)
                if valid.any():
                    ax_macd.plot(x[valid], macd[valid], color="#60a5fa", linewidth=1.0, alpha=0.95)
                    ax_macd.plot(x[valid], sig[valid], color="#f43f5e", linewidth=1.0, alpha=0.95)
                    h2 = hist.copy()
                    h2[~np.isfinite(h2)] = 0.0
                    bw = float(max(0.00018, np.median(np.diff(x[valid])) * 0.68)) if valid.sum() > 1 else 0.00028
                    bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in h2]
                    ax_macd.bar(x, h2, width=bw, color=bar_colors, alpha=0.35, linewidth=0.0)
                ax_macd.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.45)
                try:
                    mv = float(macd[np.isfinite(macd)][-1])
                    sv = float(sig[np.isfinite(sig)][-1])
                    mstate = "골든" if mv > sv else "데드"
                    mtxt = f"MACD ({mstate})" if has_kr_font else f"MACD ({mstate})"
                    ax_macd.text(
                        0.995,
                        0.82,
                        mtxt,
                        transform=ax_macd.transAxes,
                        ha="right",
                        va="center",
                        fontsize=7.8,
                        color="#bfdbfe",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#0f172a", "edgecolor": "#334155", "alpha": 0.86},
                    )
                except Exception:
                    pass
            else:
                ax_macd.text(0.01, 0.6, "MACD N/A", transform=ax_macd.transAxes, color="#94a3b8", fontsize=7.6)

            if (not px_df.empty) and ("SQZ_MOM_PCT" in px_df.columns) and len(x) == len(px_df):
                sqz = pd.to_numeric(px_df["SQZ_MOM_PCT"], errors="coerce").to_numpy(dtype=float)
                sqz_on = np.zeros(len(px_df), dtype=int)
                if "SQZ_ON" in px_df.columns:
                    try:
                        sqz_on = pd.to_numeric(px_df["SQZ_ON"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
                    except Exception:
                        sqz_on = np.zeros(len(px_df), dtype=int)
                sqz2 = sqz.copy()
                sqz2[~np.isfinite(sqz2)] = 0.0
                bw = float(max(0.00018, np.median(np.diff(x)) * 0.70)) if len(x) > 1 else 0.00028
                colors = []
                for i, v in enumerate(sqz2):
                    if int(sqz_on[i]) == 1:
                        colors.append("#facc15")
                    else:
                        colors.append("#22c55e" if v >= 0 else "#ef4444")
                ax_sqz.bar(x, sqz2, width=bw, color=colors, alpha=0.52, linewidth=0.0)
                ax_sqz.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.45)
                try:
                    sv = float(sqz[np.isfinite(sqz)][-1])
                    st = "압축중" if (len(sqz_on) > 0 and int(sqz_on[-1]) == 1) else ("상승" if sv > 0 else "하락")
                    stxt = f"SQZ {sv:+.2f}% ({st})" if has_kr_font else f"SQZ {sv:+.2f}% ({st})"
                    ax_sqz.text(
                        0.995,
                        0.82,
                        stxt,
                        transform=ax_sqz.transAxes,
                        ha="right",
                        va="center",
                        fontsize=7.8,
                        color="#fde68a",
                        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#111827", "edgecolor": "#374151", "alpha": 0.86},
                    )
                except Exception:
                    pass
            else:
                ax_sqz.text(0.01, 0.6, "SQZ N/A", transform=ax_sqz.transAxes, color="#94a3b8", fontsize=7.6)

            ax_rsi.set_ylabel("RSI", color="#94a3b8", fontsize=7.8)
            ax_macd.set_ylabel("MACD", color="#94a3b8", fontsize=7.8)
            ax_sqz.set_ylabel("SQZ%", color="#94a3b8", fontsize=7.8)

        sym_s = str(sym or "")
        side_s = ("롱" if str(side).lower() in ["buy", "long"] else "숏") if has_kr_font else ("LONG" if str(side).lower() in ["buy", "long"] else "SHORT")
        evt_map_ko = {"ENTRY": "진입", "TP": "익절", "SL": "손절", "PROTECT": "수익보호", "TAKE_FORCE": "강제익절", "POSITION": "포지션"}
        evt_map_en = {"ENTRY": "ENTRY", "TP": "TAKE", "SL": "STOP", "PROTECT": "PROTECT", "TAKE_FORCE": "TAKE_FORCE", "POSITION": "POSITION"}
        evt_key = str(event_type or "").upper().strip()
        evt_txt = evt_map_ko.get(evt_key, evt_key) if has_kr_font else evt_map_en.get(evt_key, evt_key)
        style_txt = _style_plot_label(style, has_kr_font)
        ttl = f"{evt_txt} | {sym_s} | {style_txt} {side_s} | TF {tf}"
        ax.set_title(ttl, color="white", fontsize=11, pad=10)
        if show_indicator_panels and ax_sqz is not None:
            for axh in [ax, ax_rsi, ax_macd]:
                axh.tick_params(labelbottom=False)
            ax_sqz.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for lab in ax_sqz.get_xticklabels():
                lab.set_rotation(0)
                lab.set_horizontalalignment("center")
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

        ind_txt = ", ".join([str(x) for x in (used_indicators or []) if str(x).strip()])[:120]
        ind_txt = _plot_text_sanitize(ind_txt, has_kr_font=has_kr_font, max_len=140)
        pat_txt = _plot_text_sanitize(pattern_hint, has_kr_font=has_kr_font, max_len=180)
        mtf_txt = ""
        try:
            if isinstance(mtf_pattern, dict) and bool(mtf_pattern):
                mtf_txt = _plot_text_sanitize(str(mtf_pattern.get("summary", "") or ""), has_kr_font=has_kr_font, max_len=200)
        except Exception:
            mtf_txt = ""
        one = _plot_text_sanitize(one_line, has_kr_font=has_kr_font, max_len=200)
        info_lines = []
        if ep is not None:
            info_lines.append(f"진입가 {float(ep):.6g}" if has_kr_font else f"Entry {float(ep):.6g}")
        if tp_line is not None:
            info_lines.append(f"목표익절 {float(tp_line):.6g}" if has_kr_font else f"TP target {float(tp_line):.6g}")
        if sl_line is not None:
            info_lines.append(f"목표손절 {float(sl_line):.6g}" if has_kr_font else f"SL target {float(sl_line):.6g}")
        if pt1_line is not None:
            info_lines.append(f"분할익절1 {float(pt1_line):.6g}" if has_kr_font else f"Partial TP1 {float(pt1_line):.6g}")
        if pt2_line is not None:
            info_lines.append(f"분할익절2 {float(pt2_line):.6g}" if has_kr_font else f"Partial TP2 {float(pt2_line):.6g}")
        if dca_line is not None:
            info_lines.append(f"추매라인 {float(dca_line):.6g}" if has_kr_font else f"DCA line {float(dca_line):.6g}")
        if tp_roi_pct is not None and math.isfinite(float(tp_roi_pct)):
            info_lines.append(f"목표익절 ROI +{float(abs(tp_roi_pct)):.2f}%" if has_kr_font else f"TP ROI +{float(abs(tp_roi_pct)):.2f}%")
        if sl_roi_pct is not None and math.isfinite(float(sl_roi_pct)):
            info_lines.append(f"목표손절 ROI -{float(abs(sl_roi_pct)):.2f}%" if has_kr_font else f"SL ROI -{float(abs(sl_roi_pct)):.2f}%")
        if roi_pct is not None and math.isfinite(float(roi_pct)):
            info_lines.append(f"결과 ROI {float(roi_pct):+.2f}%" if has_kr_font else f"Result ROI {float(roi_pct):+.2f}%")
        if pnl_usdt is not None and math.isfinite(float(pnl_usdt)):
            info_lines.append(f"손익 {float(pnl_usdt):+.2f} USDT" if has_kr_font else f"PnL {float(pnl_usdt):+.2f} USDT")
        if remain_free is not None and math.isfinite(float(remain_free)):
            if remain_total is not None and math.isfinite(float(remain_total)):
                info_lines.append(f"잔액(가용/총) {float(remain_free):.2f}/{float(remain_total):.2f}" if has_kr_font else f"Balance (free/total) {float(remain_free):.2f}/{float(remain_total):.2f}")
            else:
                info_lines.append(f"잔액(가용) {float(remain_free):.2f}" if has_kr_font else f"Balance (free) {float(remain_free):.2f}")
        if ind_txt and (not show_indicator_panels):
            info_lines.append(f"지표: {ind_txt}" if has_kr_font else f"Indicators: {ind_txt}")
        pat_info = ", ".join(detected_patterns[:4]) if detected_patterns else pat_txt
        pat_info = _plot_text_sanitize(pat_info, has_kr_font=has_kr_font, max_len=160)
        if pat_info:
            info_lines.append(f"패턴(단기): {pat_info}" if has_kr_font else f"Pattern(short): {pat_info}")
        if mtf_txt:
            info_lines.append(f"패턴(MTF): {mtf_txt[:180]}" if has_kr_font else f"Pattern(MTF): {mtf_txt[:180]}")
        if one:
            info_lines.append(f"근거: {one}" if has_kr_font else f"Reason: {one}")
        box_text = "\n".join(info_lines[:10])
        if box_text:
            ax.text(
                0.01,
                0.01,
                box_text,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8.2,
                color="#e5e7eb",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#0b1220", "edgecolor": "#334155", "alpha": 0.82},
            )

        ts = now_kst().strftime("%Y%m%d_%H%M%S")
        sid = re.sub(r"[^A-Za-z0-9]+", "_", sym_s)[:40].strip("_") or "SYM"
        tid = re.sub(r"[^A-Za-z0-9]+", "", str(trade_id or ""))[:16]
        fname = f"{ts}_{sid}_{str(event_type).lower()}_{tid or uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(EVENT_IMAGE_DIR, fname)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        _cleanup_event_images()
        return out_path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def chart_snapshot_for_reason(ex, sym: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    행동(손절/익절/본절/추매/순환매) 직전에,
    '왜 그렇게 했는지' 설명을 만들기 위한 최소 차트 스냅샷(룰 기반).
    - AI 호출 없음
    - 네트워크/ta 문제로 실패해도 봇이 멈추지 않음
    """
    out: Dict[str, Any] = {"time_kst": now_kst_str(), "symbol": str(sym)}
    try:
        tf = str(cfg.get("timeframe", "5m") or "5m")
        out["tf"] = tf
        fast = int(cfg.get("ma_fast", 7) or 7)
        slow = int(cfg.get("ma_slow", 99) or 99)
        out["trend_short"] = str(get_htf_trend_cached(ex, sym, tf, fast=fast, slow=slow, cache_sec=20))
        htf_tf = str(cfg.get("trend_filter_timeframe", "1h") or "1h")
        out["htf_tf"] = htf_tf
        out["trend_long"] = str(
            get_htf_trend_cached(ex, sym, htf_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60) or 60))
        )
    except Exception:
        pass

    # RSI/MACD/ADX는 선택(ta 있을 때만)
    try:
        tf = str(out.get("tf") or cfg.get("timeframe", "5m") or "5m")
        ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=220)
        if not ohlcv or len(ohlcv) < 40:
            return out
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        close = df["close"].astype(float)
        if ta is None:
            return out

        if bool(cfg.get("use_rsi", True)):
            try:
                rsi_p = int(cfg.get("rsi_period", 14) or 14)
                rsi_s = ta.momentum.rsi(close, window=rsi_p)
                rsi_v = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else None
                out["rsi"] = rsi_v
                out["rsi_state"] = _rsi_state_ko(rsi_v, cfg)
            except Exception:
                pass

        if bool(cfg.get("use_macd", True)):
            try:
                m = ta.trend.MACD(close)
                macd_v = float(m.macd().iloc[-1])
                sig_v = float(m.macd_signal().iloc[-1])
                if pd.notna(macd_v) and pd.notna(sig_v):
                    out["macd_state"] = "골든" if macd_v > sig_v else "데드"
            except Exception:
                pass

        if bool(cfg.get("use_adx", True)):
            try:
                adx_s = ta.trend.adx(df["high"].astype(float), df["low"].astype(float), close, window=14)
                adx_v = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else None
                out["adx"] = adx_v
            except Exception:
                pass
    except Exception:
        pass
    return out


def _fmt_indicator_line_for_reason(entry_snap: Optional[Dict[str, Any]], now_snap: Optional[Dict[str, Any]]) -> str:
    if not isinstance(now_snap, dict):
        return ""
    parts: List[str] = []
    try:
        ts = _trend_clean_for_reason(now_snap.get("trend_short", ""))
        if ts:
            parts.append(f"단기추세:{ts}")
    except Exception:
        pass
    try:
        tl = _trend_clean_for_reason(now_snap.get("trend_long", ""))
        if tl:
            parts.append(f"장기추세:{tl}")
    except Exception:
        pass
    try:
        r1 = now_snap.get("rsi", None)
        if r1 is not None:
            r1f = float(r1)
            if isinstance(entry_snap, dict) and entry_snap.get("rsi", None) is not None:
                r0f = float(entry_snap.get("rsi", 0.0))
                parts.append(f"RSI:{r0f:.0f}→{r1f:.0f}({str(now_snap.get('rsi_state','') or '')})")
            else:
                parts.append(f"RSI:{r1f:.0f}({str(now_snap.get('rsi_state','') or '')})")
    except Exception:
        pass
    try:
        ms = str(now_snap.get("macd_state", "") or "").strip()
        if ms:
            parts.append(f"MACD:{ms}")
    except Exception:
        pass
    try:
        adx_v = now_snap.get("adx", None)
        if adx_v is not None:
            adx_f = float(adx_v)
            if math.isfinite(adx_f):
                parts.append(f"ADX:{adx_f:.0f}")
    except Exception:
        pass
    return " | ".join(parts)[:220]


def build_exit_one_line(
    *,
    base_reason: str,
    entry_snap: Optional[Dict[str, Any]],
    now_snap: Optional[Dict[str, Any]],
) -> str:
    """
    텔레그램/일지에 들어갈 '짧은 근거(2줄)' 생성.
    - 1줄: 행동 이유(한국어)
    - 2줄: 단기/장기추세 + RSI + MACD 등 핵심 변화
    """
    base = str(base_reason or "").strip() or "-"
    ind = _fmt_indicator_line_for_reason(entry_snap, now_snap)
    if ind:
        return f"{base}\n{ind}"
    return base


def be_recheck_should_hold(
    side: str,
    entry_snap: Optional[Dict[str, Any]],
    now_snap: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    본절(BE) 라인 터치 시 즉시 청산하지 않고, 차트 상태를 한 번 더 평가한다.
    - hold=True: 본절 터치여도 홀딩
    - hold=False: 본절 청산 진행
    """
    if not isinstance(now_snap, dict):
        return False, "차트확인 실패"

    side0 = str(side or "").lower().strip()
    trend_s = _trend_clean_for_reason(now_snap.get("trend_short", ""))
    trend_l = _trend_clean_for_reason(now_snap.get("trend_long", ""))
    macd = str(now_snap.get("macd_state", "") or "").strip()
    rsi_state = str(now_snap.get("rsi_state", "") or "").strip()
    adx_v = _as_float(now_snap.get("adx", 0.0), 0.0)

    score = 0
    tags: List[str] = []

    if side0 == "long":
        if "상승" in trend_s:
            score += 2
            tags.append("단기상승")
        elif "하락" in trend_s:
            score -= 2
            tags.append("단기하락")
        if "상승" in trend_l:
            score += 1
            tags.append("장기상승")
        elif "하락" in trend_l:
            score -= 1
            tags.append("장기하락")
        if macd == "골든":
            score += 1
            tags.append("MACD골든")
        elif macd == "데드":
            score -= 1
            tags.append("MACD데드")
        if rsi_state == "과매도":
            score += 1
            tags.append("RSI과매도")
        elif rsi_state == "과매수":
            score -= 1
            tags.append("RSI과매수")
    else:
        if "하락" in trend_s:
            score += 2
            tags.append("단기하락")
        elif "상승" in trend_s:
            score -= 2
            tags.append("단기상승")
        if "하락" in trend_l:
            score += 1
            tags.append("장기하락")
        elif "상승" in trend_l:
            score -= 1
            tags.append("장기상승")
        if macd == "데드":
            score += 1
            tags.append("MACD데드")
        elif macd == "골든":
            score -= 1
            tags.append("MACD골든")
        if rsi_state == "과매수":
            score += 1
            tags.append("RSI과매수")
        elif rsi_state == "과매도":
            score -= 1
            tags.append("RSI과매도")

    if adx_v >= 22:
        # 추세 강도 높을 때는 현재 포지션 방향 유리/불리 판정을 조금 더 강하게 반영
        if score > 0:
            score += 1
            tags.append(f"ADX강({adx_v:.0f})")
        elif score < 0:
            score -= 1
            tags.append(f"ADX역({adx_v:.0f})")

    try:
        hold_min = int(cfg.get("be_recheck_hold_score_min", 2) or 2)
    except Exception:
        hold_min = 2
    hold = bool(score >= hold_min)

    # entry 대비 RSI 변화 간단 보조 설명
    try:
        if isinstance(entry_snap, dict) and entry_snap.get("rsi", None) is not None and now_snap.get("rsi", None) is not None:
            r0 = float(entry_snap.get("rsi", 0.0))
            r1 = float(now_snap.get("rsi", 0.0))
            tags.append(f"RSI {r0:.0f}→{r1:.0f}")
    except Exception:
        pass

    note = f"점수 {score:+d}/{int(hold_min)} | " + ", ".join(tags[:5]) if tags else f"점수 {score:+d}/{int(hold_min)}"
    return hold, note[:220]


def _maybe_switch_style_for_open_position(
    ex,
    sym: str,
    pos_side: str,
    tgt: Dict[str, Any],
    cfg: Dict[str, Any],
    mon: Dict[str, Any],
) -> Dict[str, Any]:
    """
    포지션 보유 중 차트 상황이 바뀌면 스타일을 전환(스윙->스캘핑 청산모드, 스캘핑->스윙 전환)
    """
    try:
        # ✅ 사용자가 자동 스타일 전환을 끈 경우(auto 레짐만):
        # - 불필요한 전환/AI 호출/혼란을 막기 위해 현재 스타일을 유지한다.
        try:
            regime_mode0 = str(cfg.get("regime_mode", "auto") or "auto").lower().strip()
        except Exception:
            regime_mode0 = "auto"
        if regime_mode0 == "auto" and (not bool(cfg.get("style_auto_enable", True))):
            try:
                tgt["style_reco"] = str(tgt.get("style", "스캘핑") or "스캘핑")
                tgt["style_reco_note"] = "자동 전환 OFF"
            except Exception:
                pass
            return tgt

        fast = int(cfg.get("ma_fast", 7))
        slow = int(cfg.get("ma_slow", 99))

        short_tf = str(cfg.get("timeframe", "5m"))
        long_tf = str(cfg.get("trend_filter_timeframe", "1h"))

        short_trend = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=25)
        long_trend = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))

        cur_style = str(tgt.get("style", "스캘핑"))
        # 추천 스타일(룰 기반)
        dec = "buy" if pos_side == "long" else "sell"
        # ✅ 강제 Exit(수익보존) 정책이 ON이면, 포지션 관리 루프가 AI 호출로 지연되지 않게 스타일 전환 AI는 잠시 비활성
        allow_ai_switch = bool(cfg.get("style_switch_ai_enable", False)) and (not bool(cfg.get("exit_trailing_protect_enable", False)))
        rec = _style_for_entry(sym, dec, short_trend, long_trend, cfg, allow_ai=bool(allow_ai_switch))
        rec_style = rec.get("style", cur_style)
        # ✅ 레짐(스캘핑/스윙) 강제/자동 선택
        # 요구사항: "시간 기반 최소유지기간(style_lock_minutes) 강제 금지"
        # 대신 confirm2/hysteresis로 흔들림 방지
        regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
        if regime_mode in ["scalping", "scalp", "short"]:
            rec_style = "스캘핑"
        elif regime_mode in ["swing", "long"]:
            rec_style = "스윙"

        # ✅ 스캘핑→스윙(보유시간) 전환을 이미 실행한 포지션은, 같은 포지션에서 곧바로 스윙→스캘핑으로 되돌리는 플립플롭을 막는다.
        # - 전환 직후 손절만 넓어지고(익절은 그대로) 반복 전환이 발생하면, 수익을 반납하거나 수수료/혼란만 커질 수 있음
        try:
            if (
                str(regime_mode or "") == "auto"
                and bool(tgt.get("_hold_convert_to_swing", False))
                and str(tgt.get("style", "")) == "스윙"
                and str(rec_style or "") == "스캘핑"
            ):
                tgt["style_reco"] = "스윙"
                tgt["style_reco_note"] = "보유시간 전환(스윙) 완료 → 되돌림 방지"
                tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                return tgt
        except Exception:
            pass

        # ✅ confirm2/hysteresis가 "같은 데이터 스냅샷"으로 몇 초 만에 누적되어
        #    레짐이 빠르게 바뀌는 현상을 방지:
        # - get_htf_trend_cached는 cache_sec 동안 같은 값을 반환할 수 있으므로,
        #   카운트/바이어스는 "새 스냅샷(캐시 갱신/새 봉)"에서만 갱신한다.
        try:
            sm = _TREND_CACHE.get(f"{sym}|{short_tf}", {}) or {}
            lm = _TREND_CACHE.get(f"{sym}|{long_tf}", {}) or {}
            # ✅ 레짐 confirm2/hysteresis는 "몇 초마다"가 아니라, "새 봉(캔들) 기준"으로만 누적되게 한다.
            # - ts(cache 갱신 시각)를 토큰에 포함하면 5~25초 단위로 누적되어 전환이 너무 잦아질 수 있음
            # - 따라서 last_bar_ms(봉 timestamp)만 사용
            short_bar = int(sm.get("last_bar_ms", 0) or 0)
            long_bar = int(lm.get("last_bar_ms", 0) or 0)
            trend_snap_token = f"{short_tf}|{short_bar}|{long_tf}|{long_bar}"
        except Exception:
            trend_snap_token = ""

        # ✅ 플립플롭 방지(시간락 없이):
        # - 장기추세가 계속 같은 방향인데, 단기추세가 "횡보/전환"처럼 중립으로 흔들리는 것만으로
        #   스윙→스캘핑으로 급전환하지 않게 한다.
        # - (중요) 레짐 강제 모드에서는 이 가드를 적용하지 않는다.
        try:
            if regime_mode == "auto" and cur_style == "스윙" and rec_style == "스캘핑":
                def _trend_state(t: str) -> str:
                    tt = str(t or "")
                    if "상승" in tt:
                        return "up"
                    if "하락" in tt:
                        return "down"
                    if ("횡보" in tt) or ("전환" in tt):
                        return "side"
                    return "neutral"

                def _align_state(state: str, side: str) -> bool:
                    if side == "long":
                        return state == "up"
                    if side == "short":
                        return state == "down"
                    return False

                def _opp_state(state: str, side: str) -> bool:
                    if side == "long":
                        return state == "down"
                    if side == "short":
                        return state == "up"
                    return False

                st_short = _trend_state(short_trend)
                st_long = _trend_state(long_trend)
                long_align = _align_state(st_long, str(pos_side or ""))
                short_opp = _opp_state(st_short, str(pos_side or ""))
                if long_align and (not short_opp) and st_short in ["side", "neutral"]:
                    tgt["style_reco"] = "스윙"
                    tgt["style_reco_note"] = "장기추세 유지 + 단기 중립 흔들림 → 스윙 유지"
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt
        except Exception:
            pass

        switch_ctl = str(cfg.get("regime_switch_control", "confirm2")).lower().strip()  # confirm2|hysteresis|off
        if regime_mode == "auto" and rec_style == cur_style:
            # 연속 확인 로직이 "연속"이 되도록, 동일 스타일이 나오면 pending을 초기화
            try:
                tgt["_pending_style"] = ""
                tgt["_pending_style_count"] = 0
                tgt["_pending_style_snap_token"] = ""
            except Exception:
                pass
        if regime_mode == "auto" and rec_style != cur_style:
            if switch_ctl == "confirm2":
                # 기본 2회 확인(confirm2) + "플립백(바로 되돌리기)" 방지:
                # - 직전 전환의 반대 방향으로 되돌리려면 더 많은 확인이 필요(시간락 없이 흔들림 방지)
                required_n = 2
                flipback_n = 3
                try:
                    required_n = int(cfg.get("regime_confirm_n", 2) or 2)
                except Exception:
                    required_n = 2
                try:
                    flipback_n = int(cfg.get("regime_confirm_n_flipback", 3) or 3)
                except Exception:
                    flipback_n = 3
                required_n = max(2, min(8, required_n))
                flipback_n = max(required_n, min(10, flipback_n))
                try:
                    last_from = str(tgt.get("_last_style_switch_from", "") or "")
                    last_to = str(tgt.get("_last_style_switch_to", "") or "")
                    # 현재 스타일이 직전 전환 "to"이고, 이번 추천이 직전 "from"으로 되돌리기라면 더 엄격
                    if last_to and last_from and (last_to == cur_style) and (rec_style == last_from):
                        required_n = max(required_n, flipback_n)
                except Exception:
                    pass
                pending = str(tgt.get("_pending_style", ""))
                cnt = int(tgt.get("_pending_style_count", 0) or 0)
                snap_prev = str(tgt.get("_pending_style_snap_token", "") or "")
                if pending == rec_style:
                    # 같은 스냅샷(캐시 갱신 전)에서는 카운트를 올리지 않는다
                    if trend_snap_token and trend_snap_token == snap_prev:
                        cnt = cnt
                    else:
                        cnt += 1
                        tgt["_pending_style_snap_token"] = trend_snap_token
                else:
                    pending = rec_style
                    cnt = 1
                    tgt["_pending_style_snap_token"] = trend_snap_token
                tgt["_pending_style"] = pending
                tgt["_pending_style_count"] = cnt
                if cnt < int(required_n):
                    # 2회 연속 동일 레짐일 때만 전환
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt
                # 전환 확정
                tgt["_pending_style"] = ""
                tgt["_pending_style_count"] = 0
                tgt["_pending_style_snap_token"] = ""
            elif switch_ctl == "hysteresis":
                bias = float(tgt.get("_regime_bias", 0.5) or 0.5)  # 0=스캘핑, 1=스윙
                step = float(cfg.get("regime_hysteresis_step", 0.55))
                enter_swing = float(cfg.get("regime_hysteresis_enter_swing", 0.75))
                enter_scalp = float(cfg.get("regime_hysteresis_enter_scalp", 0.25))
                snap_prev = str(tgt.get("_regime_bias_snap_token", "") or "")
                if (not trend_snap_token) or trend_snap_token != snap_prev:
                    if rec_style == "스윙":
                        bias = min(1.0, bias + step)
                    else:
                        bias = max(0.0, bias - step)
                    tgt["_regime_bias"] = bias
                    tgt["_regime_bias_snap_token"] = trend_snap_token
                # 임계값을 넘을 때만 전환
                if cur_style == "스캘핑" and bias < enter_swing:
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt
                if cur_style == "스윙" and bias > enter_scalp:
                    tgt["style_reco"] = rec_style
                    tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                    tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                    return tgt

        # ✅ 같은 스냅샷(봉)에서 전환이 반복되는 경우(환경/중복 워커/네트워크 흔들림 등) 1회만 허용
        try:
            last_sw_tok = str(tgt.get("_last_style_switch_snap_token", "") or "")
            if trend_snap_token and trend_snap_token == last_sw_tok and rec_style != cur_style:
                tgt["style_reco"] = rec_style
                tgt["style_reco_note"] = "같은 봉(스냅샷) 반복 → 전환 스킵"
                tgt["trend_short_now"] = f"{short_tf} {short_trend}"
                tgt["trend_long_now"] = f"{long_tf} {long_trend}"
                return tgt
        except Exception:
            pass

        if rec_style != cur_style:
            # 전환 전 목표(RR) 스냅샷
            try:
                old_tp = float(tgt.get("tp", 0) or 0.0)
            except Exception:
                old_tp = 0.0
            try:
                old_sl = float(tgt.get("sl", 0) or 0.0)
            except Exception:
                old_sl = 0.0
            rr_old = (old_tp / max(abs(old_sl), 0.01)) if (old_tp or old_sl) else 0.0
            # flip-back 방지용 메타(시간락 없이 흔들림 제어)
            try:
                tgt["_last_style_switch_from"] = str(cur_style)
                tgt["_last_style_switch_to"] = str(rec_style)
            except Exception:
                pass
            # 전환 기록
            tgt["style"] = rec_style
            tgt["style_confidence"] = int(rec.get("confidence", 0))
            tgt["style_reason"] = str(rec.get("reason", ""))[:240]
            tgt["style_last_switch_epoch"] = time.time()
            try:
                tgt["_last_style_switch_snap_token"] = str(trend_snap_token or "")
            except Exception:
                pass
            tgt["trend_short_now"] = f"{short_tf} {short_trend}"
            tgt["trend_long_now"] = f"{long_tf} {long_trend}"

            # 전환 시 목표 보정: 스윙->스캘핑이면 "빨리 청산" 모드로 목표 낮춤
            if rec_style == "스캘핑":
                # 기존 TP/SL이 너무 크면 스캘핑 범위로 조임
                tgt["tp"] = float(clamp(float(tgt.get("tp", 3.0)), float(cfg.get("scalp_tp_roi_min", 0.8)), float(cfg.get("scalp_tp_roi_max", 6.0))))
                tgt["sl"] = float(clamp(float(tgt.get("sl", 2.0)), float(cfg.get("scalp_sl_roi_min", 0.8)), float(cfg.get("scalp_sl_roi_max", 5.0))))
                tgt["scalp_exit_mode"] = True
            else:
                tgt["tp"] = float(clamp(float(tgt.get("tp", 6.0)), float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))
                tgt["sl"] = float(clamp(float(tgt.get("sl", 3.0)), float(cfg.get("swing_sl_roi_min", 12.0)), float(cfg.get("swing_sl_roi_max", 30.0))))
                tgt["scalp_exit_mode"] = False
                # ✅ 스윙은 스캘핑보다 RR/목표폭이 "확연히" 커야 하므로,
                #    전환 시점에도 RR 하한을 강제해 TP를 충분히 늘린다.
                try:
                    mode_now = str(cfg.get("trade_mode", "안전모드") or "안전모드")
                    rr_min_now = max(float(_rr_min_by_mode(mode_now)), float(_rr_min_by_style("스윙")))
                    sl_now = float(tgt.get("sl", 0) or 0.0)
                    tp_now = float(tgt.get("tp", 0) or 0.0)
                    tp_need = abs(sl_now) * float(rr_min_now)
                    if tp_now < tp_need:
                        tp_cap = float(cfg.get("swing_tp_roi_max", 50.0))
                        tgt["tp"] = float(clamp(tp_need, float(cfg.get("swing_tp_roi_min", 3.0)), tp_cap))
                except Exception:
                    pass

            # ✅ 스타일 전환 시 SR 가격 라인도 함께 재계산(스윙인데 -2~-3%에 잘리는 문제 완화)
            try:
                lev0 = float(tgt.get("lev", 1) or 1)
            except Exception:
                lev0 = 1.0
            try:
                sl_roi0 = float(tgt.get("sl", 0) or 0.0)
            except Exception:
                sl_roi0 = 0.0
            try:
                tp_roi0 = float(tgt.get("tp", 0) or 0.0)
            except Exception:
                tp_roi0 = 0.0
            sl_price_pct0 = abs(sl_roi0) / max(lev0, 1.0) if lev0 else abs(sl_roi0)
            tp_price_pct0 = abs(tp_roi0) / max(lev0, 1.0) if lev0 else abs(tp_roi0)
            tgt["sl_price_pct"] = float(sl_price_pct0)
            tgt["tp_price_pct"] = float(tp_price_pct0)
            try:
                # 이전 AI 가격 라인은 스위치 이후에는 참고만(재계산 SR이 우선)
                tgt["sl_price_ai"] = None
                tgt["tp_price_ai"] = None
            except Exception:
                pass
            try:
                if cfg.get("use_sr_stop", True):
                    dec2 = "buy" if str(pos_side) == "long" else "sell"
                    entry_px0 = 0.0
                    try:
                        entry_px0 = float(tgt.get("entry_price", 0) or 0.0)
                    except Exception:
                        entry_px0 = 0.0
                    if entry_px0 <= 0:
                        entry_px0 = float(get_last_price(ex, sym) or 0.0)
                    if entry_px0 > 0:
                        sr_res2 = sr_prices_for_style(
                            ex,
                            sym,
                            entry_price=float(entry_px0),
                            side=str(dec2),
                            style=str(rec_style),
                            cfg=cfg,
                            sl_price_pct=float(sl_price_pct0),
                            tp_price_pct=float(tp_price_pct0),
                            ai_sl_price=None,
                            ai_tp_price=None,
                        )
                        if isinstance(sr_res2, dict):
                            tgt["sl_price"] = sr_res2.get("sl_price", tgt.get("sl_price"))
                            tgt["tp_price"] = sr_res2.get("tp_price", tgt.get("tp_price"))
                            tgt["sl_price_source"] = str(sr_res2.get("sl_source", "") or "")
                            tgt["tp_price_source"] = str(sr_res2.get("tp_source", "") or "")
                            tgt["sr_used"] = {
                                "tf": sr_res2.get("tf", ""),
                                "lookback": sr_res2.get("lookback", 0),
                                "pivot_order": sr_res2.get("pivot_order", 0),
                                "buffer_atr_mult": sr_res2.get("buffer_atr_mult", 0.0),
                                "rr_min": sr_res2.get("rr_min", 0.0),
                            }
                    # SR 실패 시에도 ROI 바운드로 가격 라인을 확보(가격 기반 stop 조건 유지)
                    if tgt.get("sl_price") is None or tgt.get("tp_price") is None:
                        try:
                            slb2, tpb2 = _sr_price_bounds_from_price_pct(float(entry_px0), str(dec2), float(sl_price_pct0), float(tp_price_pct0))
                            if tgt.get("sl_price") is None:
                                tgt["sl_price"] = float(slb2)
                                if not str(tgt.get("sl_price_source", "") or ""):
                                    tgt["sl_price_source"] = "ROI"
                            if tgt.get("tp_price") is None:
                                tgt["tp_price"] = float(tpb2)
                                if not str(tgt.get("tp_price_source", "") or ""):
                                    tgt["tp_price_source"] = "ROI"
                        except Exception:
                            pass
            except Exception:
                pass

            # 전환 후 목표(RR)
            try:
                new_tp = float(tgt.get("tp", 0) or 0.0)
            except Exception:
                new_tp = 0.0
            try:
                new_sl = float(tgt.get("sl", 0) or 0.0)
            except Exception:
                new_sl = 0.0
            rr_new = (new_tp / max(abs(new_sl), 0.01)) if (new_tp or new_sl) else 0.0

            mon_add_event(
                mon,
                "STYLE_SWITCH",
                sym,
                f"{cur_style} → {rec_style} | 목표손익비(익절/손절) +{old_tp:.2f}%/-{old_sl:.2f}% → +{new_tp:.2f}%/-{new_sl:.2f}%",
                {"reason": tgt.get("style_reason", ""), "rr_old": rr_old, "rr_new": rr_new, "tp_old": old_tp, "tp_new": new_tp, "sl_old": old_sl, "sl_new": new_sl},
            )
            # 사용자 체감용: 스타일 전환 즉시 알림(채널/이벤트 라우팅)
            try:
                trade_id0 = str(tgt.get("trade_id", "") or "") or "-"
                lev0 = tgt.get("lev", "?")
                if _tg_simple_enabled(cfg):
                    q = _tg_quote_block(str(tgt.get("style_reason", "") or ""))
                    if not q:
                        q = "  └ -"
                    msg = (
                        "🔄 방식 전환\n"
                        f"- 코인: {sym}\n"
                        f"- {cur_style} → {rec_style}\n"
                        f"- 포지션: {_tg_dir_easy(pos_side)}\n"
                        f"- 레버리지: x{lev0}\n"
                        "\n"
                        f"- 목표손익비(익절/손절): 익절 {old_tp:+.2f}% / 손절 -{abs(old_sl):.2f}% → 익절 {new_tp:+.2f}% / 손절 -{abs(new_sl):.2f}%\n"
                        "\n"
                        "- 한줄:\n"
                        f"{q}\n"
                        f"- ID: {trade_id0}"
                    )
                else:
                    msg = (
                        f"🔄 방식 바뀜\n- 코인: {sym}\n- {cur_style} → {rec_style}\n- 손익비(이익:손실): {rr_old:.2f} → {rr_new:.2f}\n"
                        f"- 목표(익절/손절): +{old_tp:.2f}%/-{old_sl:.2f}% → +{new_tp:.2f}%/-{new_sl:.2f}%\n- 이유: {str(tgt.get('style_reason','') or '')[:140]}"
                    )
                tg_send(
                    msg,
                    target=cfg.get("tg_route_events_to", "channel"),
                    cfg=cfg,
                    silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                )
            except Exception:
                pass
        else:
            tgt["style_reco"] = rec_style
            tgt["trend_short_now"] = f"{short_tf} {short_trend}"
            tgt["trend_long_now"] = f"{long_tf} {long_trend}"

    except Exception:
        pass
    return tgt


def _should_convert_scalp_to_swing(tgt: Dict[str, Any], roi: float, cfg: Dict[str, Any], long_align: bool) -> bool:
    try:
        # ✅ 강제 수익보존(트레일링 보호) 모드에서는 스타일 전환으로 목표/리스크가 흔들리면 안 됨
        if bool(cfg.get("exit_trailing_protect_enable", False)):
            return False
        if str(tgt.get("style", "")) != "스캘핑":
            return False
        entry_epoch = float(tgt.get("entry_epoch", 0) or 0)
        if not entry_epoch:
            return False
        hold_min = (time.time() - entry_epoch) / 60.0
        if hold_min < float(cfg.get("scalp_max_hold_minutes", 25)):
            return False
        # ✅ 이미 익절에 가까우면(또는 충분히 수익 구간이면) 스윙 전환으로 SL만 넓히지 않는다
        try:
            tp_roi = float(tgt.get("tp", 0) or 0.0)
        except Exception:
            tp_roi = 0.0
        if tp_roi > 0:
            try:
                frac = float(cfg.get("scalp_to_swing_skip_when_roi_ge_tp_frac", 0.85) or 0.85)
            except Exception:
                frac = 0.85
            frac = float(clamp(frac, 0.55, 0.98))
            try:
                slack = float(cfg.get("scalp_to_swing_skip_when_tp_slack_roi", 1.0) or 1.0)
            except Exception:
                slack = 1.0
            slack = float(max(0.0, slack))
            if roi >= (tp_roi * frac):
                return False
            if (roi > 0) and ((tp_roi - roi) <= slack):
                return False
        # 너무 큰 손실이면 전환보다 정리가 낫다(기본)
        if roi < float(cfg.get("scalp_to_swing_min_roi", -12.0)):
            return False
        if cfg.get("scalp_to_swing_require_long_align", True) and not long_align:
            return False
        return True
    except Exception:
        return False


def _try_scalp_to_swing_dca(ex, sym: str, side: str, cur_px: float, tgt: Dict[str, Any], rt: Dict[str, Any], cfg: Dict[str, Any], mon: Dict[str, Any]) -> bool:
    """
    스캘핑 포지션이 스윙으로 전환해도 될 때(장기추세 align 등) 1회 추매 + 목표 RR 상향
    """
    try:
        if not cfg.get("use_dca", True):
            return False
        # 추매는 스윙 전환 시점에만 허용(스캘핑 기본 추매X)
        trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
        dca_count = int(trade_state.get("dca_count", 0))
        dca_max = max(0, int(cfg.get("dca_max_count", 2)))
        if dca_count >= max(1, dca_max):
            return False

        # ✅ 당일 손익 제한: 오늘 손실이 기준 이하면 DCA 금지
        try:
            if bool(cfg.get("dca_daily_pnl_limit_enable", True)):
                daily_limit = float(cfg.get("dca_daily_pnl_limit_usdt", -20.0) or -20.0)
                day_summary = _trade_day_summary(_day_df_filter(read_trade_log(), today_kst_str()))
                day_pnl = float(day_summary.get("total_pnl_usdt", 0.0) or 0.0)
                if day_pnl <= daily_limit:
                    mon_add_event(mon, "DCA_SKIP", sym, f"당일 손익 제한({day_pnl:.2f} USDT ≤ {daily_limit:.2f})", {})
                    return False
        except Exception:
            pass

        free, _ = safe_fetch_balance(ex)
        base_entry = float(tgt.get("entry_usdt", 0.0))
        dca_add_pct = float(cfg.get("dca_add_pct", 50.0))
        dca_add_usdt_cfg = 0.0
        try:
            dca_add_usdt_cfg = float(cfg.get("dca_add_usdt", 0.0) or 0.0)
        except Exception:
            dca_add_usdt_cfg = 0.0
        # ✅ (추가) USDT 기준 추매(마진) 우선, 없으면 기존 % 방식 유지
        add_usdt = float(dca_add_usdt_cfg) if float(dca_add_usdt_cfg) > 0 else (base_entry * (dca_add_pct / 100.0))
        if add_usdt > free:
            add_usdt = free * 0.5
        if add_usdt < 5:
            return False

        lev = int(float(tgt.get("lev", MODE_RULES.get(cfg.get("trade_mode", "안전모드"), MODE_RULES["안전모드"])["lev_min"])) or 1)
        set_leverage_safe(ex, sym, lev)
        qty = to_precision_qty(ex, sym, (add_usdt * lev) / max(cur_px, 1e-9))
        if qty <= 0:
            return False
        ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
        if ok:
            trade_state["dca_count"] = dca_count + 1
            save_runtime(rt)
            mon_add_event(mon, "DCA_CONVERT", sym, f"스캘핑→스윙 전환 추매 {add_usdt:.2f} USDT", {"add_usdt": add_usdt})
            try:
                # 실제 마진 추정(근사): notional/lev
                try:
                    margin_est = (float(qty) * float(cur_px)) / max(float(lev), 1.0)
                except Exception:
                    margin_est = float(add_usdt)
                try:
                    tg_send(
                        f"💧 전환추매(DCA)\n- 코인: {sym}\n- 스타일: 스캘핑→스윙\n- 추가금(마진): {float(add_usdt):.2f} USDT (추정 {float(margin_est):.2f})\n- 추가수량: {qty}\n- 레버: x{lev}\n- 일지ID: {str(tgt.get('trade_id','') or '-')}",
                        target=cfg.get("tg_route_events_to", "channel"),
                        cfg=cfg,
                    )
                except Exception:
                    pass
                gsheet_log_trade(
                    stage="DCA_CONVERT",
                    symbol=sym,
                    trade_id=str(tgt.get("trade_id", "") or ""),
                    message=f"add_usdt={add_usdt:.2f}",
                    payload={"add_usdt": add_usdt, "qty": qty, "lev": lev},
                )
            except Exception:
                pass
            return True
    except Exception:
        return False
    return False


def _swing_partial_tp_levels(tp_roi: float, cfg: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """
    returns: [(trigger_roi, close_frac, label), ...]
    """
    try:
        steps = [
            (float(cfg.get("swing_partial_tp1_at_tp_frac", 0.35)), float(cfg.get("swing_partial_tp1_close_pct", 33)) / 100.0, "TP1"),
            (float(cfg.get("swing_partial_tp2_at_tp_frac", 0.60)), float(cfg.get("swing_partial_tp2_close_pct", 33)) / 100.0, "TP2"),
            (float(cfg.get("swing_partial_tp3_at_tp_frac", 0.85)), float(cfg.get("swing_partial_tp3_close_pct", 34)) / 100.0, "TP3"),
        ]
        out = []
        for frac, close_frac, label in steps:
            if frac <= 0 or close_frac <= 0:
                continue
            out.append((max(0.1, tp_roi * frac), float(clamp(close_frac, 0.01, 0.95)), label))
        # 트리거 기준 오름차순 정렬
        out.sort(key=lambda x: x[0])
        return out
    except Exception:
        return []


def telegram_thread(ex):
    offset = 0
    mon = monitor_init()

    # runtime에서 open_targets 복구(스레드 재시작에도 목표/스타일 일부 유지)
    rt_boot = load_runtime()
    active_targets: Dict[str, Dict[str, Any]] = {}
    try:
        ot = rt_boot.get("open_targets", {}) or {}
        if isinstance(ot, dict):
            active_targets.update({k: v for k, v in ot.items() if isinstance(v, dict)})
    except Exception:
        pass

    # ✅ 워커 ID + 리스(중복 스레드/복구 시 안전장치)
    worker_id = uuid.uuid4().hex[:8]
    try:
        mon["worker_id"] = worker_id
        mon["worker_owner"] = "TG_THREAD"
        monitor_write_throttled(mon, 0.2)
    except Exception:
        pass
    try:
        # 최초 리스 확보(실패해도 봇은 계속; watchdog/다른 워커가 리더일 수 있음)
        runtime_worker_lease_touch(worker_id, owner="TG_THREAD", ttl_sec=WORKER_LEASE_TTL_SEC)
    except Exception:
        pass
    next_lease_touch_epoch = 0.0

    # ✅ 시작 EVENT (Google Sheets/모니터)
    try:
        mon_add_event(mon, "START", "", "봇 시작", {"sandbox": bool(IS_SANDBOX)})
        gsheet_log_event("START", message="bot_started", payload={"sandbox": bool(IS_SANDBOX), "boot_time_kst": mon.get("_boot_time_kst", "")})
    except Exception:
        pass

    # 부팅 메시지(그룹: 메뉴, 채널: 시작 알림)
    cfg_boot = load_settings()
    boot_msg = f"🚀 AI 봇 가동 시작! (모의투자)\n- code: {CODE_VERSION}\n명령: /menu /status /positions /scan /mode /log /gsheet"
    tg_send(boot_msg, target="channel", cfg=cfg_boot)
    # ✅ 요구: TG_TARGET_CHAT_ID는 채널(브로드캐스트), 관리는 관리자 DM으로(중복/스팸 방지)
    if TG_ADMIN_IDS:
        tg_send(boot_msg, target="admin", cfg=cfg_boot)
    elif tg_id_group and tg_id_group != tg_id_channel:
        tg_send(boot_msg, target="group", cfg=cfg_boot)
    tg_send_menu(cfg=cfg_boot)

    # 주기 작업 스케줄러 상태
    next_report_ts = 0.0
    next_heartbeat_ts = 0.0  # 요구사항: 15분(900초) 고정 하트비트
    next_vision_ts = 0.0
    last_daily_brief_date = ""
    last_daily_brief_attempt_epoch = 0.0
    last_export_attempt_epoch = 0.0

    backoff_sec = 1.0

    while True:
        try:
            cfg = load_settings()
            rt = load_runtime()
            # ✅ 워커 revoke/리스 체크(중복매매 방지)
            try:
                if worker_id and (worker_id in set(_runtime_revoked_ids(rt))):
                    mon_add_event(mon, "WORKER_REVOKED", "", "revoked_by_watchdog", {"worker_id": worker_id})
                    break
            except Exception:
                pass
            try:
                now_ts_lease = time.time()
                if now_ts_lease >= float(next_lease_touch_epoch or 0.0):
                    ok_lease = runtime_worker_lease_touch(worker_id, owner="TG_THREAD", ttl_sec=WORKER_LEASE_TTL_SEC)
                    next_lease_touch_epoch = now_ts_lease + 12.0
                    try:
                        mon["worker_is_leader"] = bool(ok_lease)
                        mon["worker_lease"] = runtime_worker_lease_get()
                        monitor_write_throttled(mon, 0.6)
                    except Exception:
                        pass
                    if not ok_lease:
                        # 다른 리더가 확정이면 현재 워커는 종료(중복 스캔/주문 방지)
                        try:
                            lease0 = runtime_worker_lease_get()
                            lid = str(lease0.get("id", "") or "")
                            until0 = float(lease0.get("until_epoch", 0) or 0)
                            if lid and lid != worker_id and time.time() < until0:
                                mon_add_event(mon, "LEASE_LOST", "", f"leader={lid}", {"until_kst": lease0.get("until_kst", "")})
                                break
                        except Exception:
                            pass
            except Exception:
                pass
            mode = cfg.get("trade_mode", "안전모드")
            rule = MODE_RULES.get(mode, MODE_RULES["안전모드"])
            ccxt_timeout_epoch_loop_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
            ccxt_timeout_where_loop_start = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "")

            # =========================================================
            # ✅ 루프 하트비트(즉시 기록)
            # - 외부시황/거래소 호출이 느리거나 일시 장애여도 UI에서 "멈춤 의심"이 과도하게 뜨지 않게
            # - (중요) trade heartbeat(텔레그램 15분 리포트)와 별개로, '스레드 생존' 신호다.
            # =========================================================
            try:
                now_str0 = now_kst_str()
                mon["loop_stage"] = "LOOP_START"
                mon["loop_stage_kst"] = now_str0
                mon["last_heartbeat_epoch"] = time.time()
                mon["last_heartbeat_kst"] = now_str0
                mon["auto_trade"] = bool(cfg.get("auto_trade", False))
                mon["trade_mode"] = mode
                mon["pause_until"] = rt.get("pause_until", 0)
                mon["consec_losses"] = rt.get("consec_losses", 0)
                mon["trend_filter_policy"] = cfg.get("trend_filter_policy", "ALLOW_SCALP")
                mon["code_version"] = CODE_VERSION
                monitor_write_throttled(mon, 0.5)
            except Exception:
                pass

            # ✅ 일일 시작 총자산(day_start_equity) 초기화(일일 손실 한도 계산용)
            try:
                if bool(cfg.get("daily_loss_limit_enable", False)):
                    dse = float(rt.get("day_start_equity", 0) or 0.0)
                    if dse <= 0:
                        now_ts0 = time.time()
                        last_try = float(rt.get("_day_start_equity_try_epoch", 0) or 0.0)
                        if (now_ts0 - last_try) >= 60.0:
                            rt["_day_start_equity_try_epoch"] = now_ts0
                            _free0, _total0 = safe_fetch_balance(ex)
                            if float(_total0) > 0:
                                rt["day_start_equity"] = float(_total0)
                                save_runtime(rt)
                                try:
                                    mon_add_event(mon, "DAY_START", "", f"day_start_equity={float(_total0):.2f}", {"day_start_equity": float(_total0)})
                                except Exception:
                                    pass
            except Exception:
                pass

            # ✅ 매일 아침 브리핑(한 번만)
            try:
                if cfg.get("daily_btc_brief_enable", False):
                    h = int(cfg.get("daily_btc_brief_hour_kst", 9))
                    m = int(cfg.get("daily_btc_brief_minute_kst", 0))
                    now = now_kst()
                    today = today_kst_str()
                    # 이미 저장되어 있으면 사용
                    if rt.get("daily_btc_brief", {}).get("date") == today:
                        last_daily_brief_date = today
                    # 스케줄 시각 이후, 오늘 브리핑이 없으면 생성
                    if last_daily_brief_date != today and (now.hour > h or (now.hour == h and now.minute >= m)):
                        # 실패 시 무한 루프/멈춤 방지: 재시도는 20분 간격으로만
                        if (time.time() - float(last_daily_brief_attempt_epoch or 0.0)) < 20 * 60:
                            pass
                        else:
                            last_daily_brief_attempt_epoch = time.time()
                            # UI에 "어디서 멈췄는지" 보이도록 stage 먼저 기록
                            try:
                                mon["loop_stage"] = "DAILY_BRIEF"
                                mon["loop_stage_kst"] = now_kst_str()
                                mon["last_heartbeat_epoch"] = time.time()
                                mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                                monitor_write_throttled(mon, 0.2)
                            except Exception:
                                pass

                            brief = {}
                            try:
                                # feedparser/외부 네트워크가 걸려도 스레드가 오래 멈추지 않도록 hard-timeout
                                brief = _call_with_timeout(lambda: fetch_daily_btc_brief(cfg), 35)
                            except FuturesTimeoutError:
                                mon_add_event(mon, "DAILY_BRIEF_TIMEOUT", "", "daily brief timeout", {"timeout_sec": 35})
                                notify_admin_error("DAILY_BRIEF", RuntimeError("timeout"), context={"timeout_sec": 35}, min_interval_sec=300.0)
                                brief = {}
                            except Exception as e:
                                mon_add_event(mon, "DAILY_BRIEF_FAIL", "", f"{type(e).__name__}: {e}"[:140], {})
                                notify_admin_error("DAILY_BRIEF", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                                brief = {}

                            if isinstance(brief, dict) and brief:
                                rt["daily_btc_brief"] = brief
                                save_runtime(rt)
                                last_daily_brief_date = today
                                # 채널로 브리핑 전송
                                try:
                                    if brief.get("items"):
                                        bias = str(brief.get("bias", "중립"))
                                        risk = str(brief.get("risk", "보통"))
                                        lines = [f"🌅 오늘 아침 BTC 브리핑 ({today})", f"- 시황 톤: {bias} | 리스크: {risk}"]
                                        for it in brief["items"][: int(cfg.get("daily_btc_brief_max_items", 5))]:
                                            emo = str(it.get("emoji", "📰"))
                                            title = str(it.get("title", ""))[:90]
                                            note = str(it.get("note", ""))[:90]
                                            if note:
                                                lines.append(f"{emo} {title}\n   └ {note}")
                                            else:
                                                lines.append(f"{emo} {title}")
                                        # 🔮 (옵션) 코인/방향 예측(outlook)
                                        try:
                                            outlook = brief.get("outlook") or brief.get("signals") or []
                                            if isinstance(outlook, list) and outlook:
                                                lines.append("🔮 전망(예측, 참고용)")
                                                for s0 in outlook[:10]:
                                                    sym0 = str(s0.get("symbol", "") or "").strip()
                                                    d0 = str(s0.get("dir", "") or s0.get("direction", "") or "").strip()
                                                    conf0 = s0.get("confidence", "")
                                                    note0 = str(s0.get("note", "") or "").strip()
                                                    if sym0:
                                                        if note0:
                                                            lines.append(f"- {sym0}: {d0} ({conf0}%) | {note0[:60]}")
                                                        else:
                                                            lines.append(f"- {sym0}: {d0} ({conf0}%)")
                                        except Exception:
                                            pass
                                        tg_send("\n".join(lines), target="channel", cfg=cfg)
                                except Exception:
                                    pass
            except Exception:
                pass

            # 외부 시황(스냅샷) 갱신 트리거(비동기) + 즉시 스냅샷 반영
            # - 외부 네트워크/번역/RSS가 멈추면 TG_THREAD까지 멈춘 것처럼 보일 수 있으므로,
            #   TG_THREAD에서는 절대 블로킹하지 않는다.
            try:
                mon["loop_stage"] = "EXTERNAL_CONTEXT"
                mon["loop_stage_kst"] = now_kst_str()
                mon["last_heartbeat_epoch"] = time.time()
                mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                monitor_write_throttled(mon, 0.2)
            except Exception:
                pass
            try:
                external_context_refresh_maybe(cfg, rt, force=False)
            except Exception:
                pass
            try:
                ext = external_context_snapshot()
                if not bool(cfg.get("use_external_context", True)):
                    mon["loop_stage"] = "EXTERNAL_CONTEXT_OFF"
                    mon["loop_stage_kst"] = now_kst_str()
            except Exception as e:
                ext = {"enabled": False, "error": str(e)[:240], "asof_kst": now_kst_str(), "_source": "snapshot_fail"}
                notify_admin_error("EXTERNAL_CONTEXT", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                mon_add_event(mon, "EXTERNAL_FAIL", "", f"{e}"[:140], {})
            mon["external"] = ext

            # ✅ 일별 내보내기 자동(새벽 00시대, 전일 기준)
            try:
                if cfg.get("export_daily_enable", True):
                    now0 = now_kst()
                    if now0.hour == 0 and now0.minute < 10:
                        today = today_kst_str()
                        if str(rt.get("last_export_date", "")) != today:
                            # 실패 시 반복 시도로 스레드가 멈추지 않게, 재시도는 10분 간격으로만
                            if (time.time() - float(last_export_attempt_epoch or 0.0)) < 10 * 60:
                                pass
                            else:
                                last_export_attempt_epoch = time.time()
                                # stage 먼저 기록
                                try:
                                    mon["loop_stage"] = "DAILY_EXPORT"
                                    mon["loop_stage_kst"] = now_kst_str()
                                    mon["last_heartbeat_epoch"] = time.time()
                                    mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                                    monitor_write_throttled(mon, 0.2)
                                except Exception:
                                    pass

                                yday = (now0 - timedelta(days=1)).strftime("%Y-%m-%d")
                                res = {}
                                try:
                                    # Excel/CSV/gsheet export가 네트워크/파일 이슈로 오래 걸려도 스레드가 멈추지 않게 hard-timeout
                                    res = _call_with_timeout(lambda: export_trade_log_daily(yday, cfg), 40)
                                except FuturesTimeoutError:
                                    mon_add_event(mon, "EXPORT_TIMEOUT", "", "daily export timeout", {"timeout_sec": 40})
                                    notify_admin_error(
                                        "DAILY_EXPORT", RuntimeError("timeout"), context={"timeout_sec": 40, "yday": yday}, min_interval_sec=300.0
                                    )
                                    res = {"ok": False, "error": "timeout"}
                                except Exception as e:
                                    mon_add_event(mon, "EXPORT_FAIL", "", f"{type(e).__name__}: {e}"[:140], {"yday": yday})
                                    notify_admin_error("DAILY_EXPORT", e, tb=traceback.format_exc(), min_interval_sec=180.0)
                                    res = {"ok": False, "error": str(e)[:240]}
                                # 스팸/정체 방지: 하루 1회만 시도하도록 날짜는 시도 시점에 고정
                                rt["last_export_date"] = today
                                save_runtime(rt)
                                # 채널로 완료 보고(스팸 방지: 하루 1회)
                                try:
                                    if isinstance(res, dict) and res.get("ok"):
                                        msg = (
                                            f"📤 일별 일지 내보내기({yday})\n"
                                            f"- rows: {res.get('rows')}\n"
                                            f"- xlsx: {res.get('excel_path','')}\n"
                                            f"- csv: {res.get('csv_path','')}\n"
                                            f"- gsheet: {res.get('gsheet','')}"
                                        )
                                        tg_send(
                                            msg,
                                            target=cfg.get("tg_route_events_to", "channel"),
                                            cfg=cfg,
                                            silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                                        )
                                except Exception:
                                    pass
            except Exception:
                pass

            # 하트비트
            mon["last_heartbeat_epoch"] = time.time()
            mon["last_heartbeat_kst"] = now_kst_str()
            mon["auto_trade"] = bool(cfg.get("auto_trade", False))
            mon["trade_mode"] = mode
            mon["pause_until"] = rt.get("pause_until", 0)
            mon["consec_losses"] = rt.get("consec_losses", 0)
            mon["trend_filter_policy"] = cfg.get("trend_filter_policy", "ALLOW_SCALP")

            # ✅ 하트비트(요구사항: 15분=900초마다)
            try:
                if tg_token and bool(cfg.get("tg_enable_heartbeat_report", False)):
                    if next_heartbeat_ts <= 0:
                        # 부팅 직후 첫 하트비트는 조금 지연(스팸 방지)
                        next_heartbeat_ts = time.time() + 20
                    if time.time() >= next_heartbeat_ts:
                        # stage 먼저 기록(잔고/포지션 조회가 느려도 UI에 표시)
                        try:
                            mon["loop_stage"] = "TG_HEARTBEAT"
                            mon["loop_stage_kst"] = now_kst_str()
                            mon["last_heartbeat_epoch"] = time.time()
                            mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                            monitor_write_throttled(mon, 0.2)
                        except Exception:
                            pass
                        free, total = safe_fetch_balance(ex)
                        realized = float(rt.get("daily_realized_pnl", 0.0) or 0.0)
                        regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                        regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")

                        # 포지션 요약
                        pos_blocks: List[str] = []
                        ps = safe_fetch_positions(ex, TARGET_COINS)
                        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                        rt_open_targets = {}
                        try:
                            rt_open_targets = (rt.get("open_targets", {}) or {}) if isinstance(rt, dict) else {}
                        except Exception:
                            rt_open_targets = {}
                        if act:
                            for p in act[:10]:
                                sym = p.get("symbol", "")
                                side = position_side_normalize(p)
                                roi = float(position_roi_percent(p))
                                upnl = float(p.get("unrealizedPnl") or 0.0)
                                lev = p.get("leverage", "?")
                                tgt0 = _resolve_open_target_for_symbol(sym, active_targets, rt_open_targets)
                                style = str((tgt0 or {}).get("style", ""))
                                pos_blocks.append(_fmt_pos_block(sym, side, lev, roi, upnl, style=style, tgt=tgt0))
                        else:
                            pos_blocks.append("⚪ 무포지션(관망)")

                        last_scan_kst = mon.get("last_scan_kst", "-")
                        last_hb_kst = mon.get("last_heartbeat_kst", "-")
                        pos_txt = "\n\n".join([str(x) for x in pos_blocks if str(x).strip()]) or "⚪ 무포지션(관망)"
                        txt = "\n".join(
                            [
                                "💓 하트비트(15분)",
                                f"- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}",
                                f"- 모드: {mode}",
                                f"- 레짐: {regime_txt}",
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})",
                                f"- 리얼손익(오늘): {realized:.2f} USDT",
                                "",
                                "📊 포지션",
                                pos_txt,
                                "",
                                f"- 마지막 스캔: {last_scan_kst}",
                                f"- 마지막 하트비트: {last_hb_kst}",
                            ]
                        )
                        tg_send(
                            txt,
                            target=cfg.get("tg_route_events_to", "channel"),
                            cfg=cfg,
                            silent=bool(cfg.get("tg_heartbeat_silent", True)),
                        )
                        try:
                            mon["last_tg_heartbeat_epoch"] = time.time()
                            mon["last_tg_heartbeat_kst"] = now_kst_str()
                        except Exception:
                            pass
                        try:
                            gsheet_log_event("HEARTBEAT", message=f"regime={regime_txt} pos={len(act)} bal={total:.2f}", payload={"regime": regime_txt, "positions": len(act), "total": total, "free": free})
                        except Exception:
                            pass
                        try:
                            hb_int = int(cfg.get("tg_heartbeat_interval_sec", 900) or 900)
                        except Exception:
                            hb_int = 900
                        next_heartbeat_ts = time.time() + float(clamp(hb_int, 60, 7200))
            except Exception:
                pass

            # ✅ 주기 리포트(15분 기본)
            try:
                if cfg.get("tg_enable_periodic_report", True):
                    interval = max(3, int(cfg.get("report_interval_min", 15)))
                    # 하트비트(15분)는 별도 고정 스케줄이므로, 동일(15)이면 중복 전송 방지
                    if bool(cfg.get("tg_enable_heartbeat_report", False)) and interval == 15:
                        # heartbeat가 이미 15분 고정으로 전송되므로, 별도 주기 리포트는 스킵
                        next_report_ts = 0.0
                    else:
                        if next_report_ts <= 0:
                            next_report_ts = time.time() + interval * 60
                        if time.time() >= next_report_ts:
                            try:
                                mon["loop_stage"] = "PERIODIC_REPORT"
                                mon["loop_stage_kst"] = now_kst_str()
                                mon["last_heartbeat_epoch"] = time.time()
                                mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                                monitor_write_throttled(mon, 0.2)
                            except Exception:
                                pass
                            free, total = safe_fetch_balance(ex)
                            # 포지션 요약
                            pos_blocks: List[str] = []
                            ps = safe_fetch_positions(ex, TARGET_COINS)
                            act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                            if act:
                                rt_open_targets = {}
                                try:
                                    rt_open_targets = (rt.get("open_targets", {}) or {}) if isinstance(rt, dict) else {}
                                except Exception:
                                    rt_open_targets = {}
                                for p in act[:8]:
                                    sym = p.get("symbol", "")
                                    side = position_side_normalize(p)
                                    roi = float(position_roi_percent(p))
                                    upnl = float(p.get("unrealizedPnl") or 0.0)
                                    lev = p.get("leverage", "?")
                                    tgt0 = _resolve_open_target_for_symbol(sym, active_targets, rt_open_targets)
                                    style = str((tgt0 or {}).get("style", ""))
                                    block = _fmt_pos_block(sym, side, lev, roi, upnl, style=style, tgt=tgt0)
                                    pos_blocks.append(block)
                            else:
                                pos_blocks.append("⚪ 무포지션(관망)")
                            pos_txt = "\n\n".join([x for x in pos_blocks if str(x or "").strip()])

                            # 외부 시황 요약
                            fg = (ext or {}).get("fear_greed") or {}
                            fg_line = ""
                            if fg:
                                fg_line = f"{fg.get('emoji','')} 공포탐욕 {fg.get('value','?')} ({fg.get('classification','')})"
                            ev_soon = (ext or {}).get("high_impact_events_soon") or []
                            ev_soon_line = " / ".join([f"{x.get('country','')} {x.get('title','')[:18]}" for x in ev_soon[:2]]) if ev_soon else "없음"
                            regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            realized = float(rt.get("daily_realized_pnl", 0.0) or 0.0)

                            # 신규진입 가능 여부(자동매매 ON인데 진입을 안 하면 즉시 확인)
                            entry_ok_txt = "가능"
                            try:
                                pu = float(rt.get("pause_until", 0) or 0.0)
                            except Exception:
                                pu = 0.0
                            try:
                                weekend_block2 = bool(cfg.get("no_trade_weekend", False)) and (now_kst().weekday() in [5, 6])
                            except Exception:
                                weekend_block2 = False
                            try:
                                paused_now2 = bool(cfg.get("loss_pause_enable", True)) and (time.time() < float(pu))
                            except Exception:
                                paused_now2 = False
                            if not bool(cfg.get("auto_trade", False)):
                                entry_ok_txt = "불가(auto_trade=OFF)"
                            elif weekend_block2:
                                entry_ok_txt = "불가(주말)"
                            elif paused_now2 and pu > 0:
                                entry_ok_txt = f"불가(정지 ~{_epoch_to_kst_str(float(pu))[11:16]})"

                            txt = "\n".join(
                                [
                                    f"🕒 {interval}분 상황보고",
                                    f"- 자동매매: {'ON' if cfg.get('auto_trade') else 'OFF'}",
                                    f"- 신규진입: {entry_ok_txt}",
                                    f"- 모드: {mode}",
                                    f"- 레짐: {regime_txt}",
                                    f"- 잔고: {total:.2f} USDT (가용 {free:.2f})",
                                    f"- 리얼손익(오늘): {realized:.2f} USDT",
                                    "",
                                    "📊 포지션",
                                    pos_txt,
                                    "",
                                    f"🌍 외부시황: {fg_line}",
                                    f"🚨 이벤트 임박: {ev_soon_line}",
                                ]
                            )
                            tgt = cfg.get("tg_route_events_to", "channel")
                            tg_send(txt, target=tgt, cfg=cfg, silent=bool(cfg.get("tg_periodic_report_silent", True)))
                            try:
                                gsheet_log_event(
                                    "PERIODIC_REPORT",
                                    message=f"interval={interval} pos={len(act)}",
                                    payload={"interval_min": interval, "positions": len(act), "total": total, "free": free},
                                )
                            except Exception:
                                pass
                            next_report_ts = time.time() + interval * 60
            except Exception:
                pass

            # ✅ CCXT timeout 감지 → exchange 인스턴스 교체(동시 호출/정체 완화)
            # - hard-timeout으로 반환되더라도, 백그라운드 작업이 계속 돌 수 있어 같은 인스턴스를 계속 쓰면 꼬일 수 있음
            # - timeout이 감지되면 다음 루프부터 새 인스턴스로 교체해 안정성을 우선한다.
            try:
                t_after = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                if t_after and t_after > float(ccxt_timeout_epoch_loop_start or 0):
                    where_now = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "").strip()
                    mon_add_event(mon, "CCXT_TIMEOUT", "", f"{where_now or 'unknown'}", {"where": where_now, "code": CODE_VERSION})
                    ex_new = create_exchange_client_uncached()
                    if ex_new is not None:
                        ex = ex_new
                        mon_add_event(mon, "CCXT_REFRESH", "", "exchange refreshed", {"reason": where_now or "timeout"})
                        # loop-start 마커 갱신
                        ccxt_timeout_epoch_loop_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                        ccxt_timeout_where_loop_start = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "")
            except Exception:
                pass

            # ✅ 1시간마다 AI 시야 리포트(채널)
            try:
                if cfg.get("tg_enable_hourly_vision_report", False):
                    interval = max(10, int(cfg.get("vision_report_interval_min", 60)))
                    if next_vision_ts <= 0:
                        next_vision_ts = time.time() + interval * 60
                    if time.time() >= next_vision_ts:
                        mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                        coins = mon_now.get("coins", {}) or {}
                        lines = [
                            "👁️ AI 시야 리포트",
                            f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                            f"- 모드: {mon_now.get('trade_mode','-')}",
                            f"- 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                        ]
                        for sym, cs in list(coins.items())[:12]:
                            style = str(cs.get("style_reco", "")) or str(cs.get("style", ""))
                            style_txt = f"[{style}]" if style else ""
                            lines.append(
                                f"- {sym}: {style_txt} {str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                            )
                        tg_send(
                            "\n".join(lines),
                            target="channel",
                            cfg=cfg,
                            silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                        )
                        next_vision_ts = time.time() + interval * 60
            except Exception:
                pass

            # ✅ /scan 강제스캔 요청(runtime_state.json)
            force_scan_req = rt.get("force_scan", {}) if isinstance(rt.get("force_scan", {}), dict) else {}
            force_scan_id = str(force_scan_req.get("id", "") or "")
            force_scan_done = bool(force_scan_req.get("done", False))
            force_scan_only = bool(force_scan_req.get("scan_only", True))
            force_scan_symbols = force_scan_req.get("symbols", [])
            if not isinstance(force_scan_symbols, list):
                force_scan_symbols = []
            try:
                force_scan_ts = float(force_scan_req.get("requested_at_epoch", 0) or 0)
            except Exception:
                force_scan_ts = 0.0
            force_scan_pending = bool(force_scan_id) and (not force_scan_done) and (time.time() - force_scan_ts < 60 * 10)

            # ✅ 스캔 루프는 항상 실행(시야/스캔 갱신)
            # - 신규 진입(주문)만 auto_trade/paused/weekend 정책으로 제어
            if True:
                trade_enabled = bool(cfg.get("auto_trade", False))
                force_scan_syms_set = set(force_scan_symbols or [])
                force_scan_summary_lines: List[str] = []

                # 주말 거래 금지: 신규진입만 제한(스캔/시야는 계속)
                # - weekend_block_now는 entry_allowed_global에 반영

                # 일시정지(연속손실)
                paused_now = cfg.get("loss_pause_enable", True) and time.time() < float(rt.get("pause_until", 0))
                if False and paused_now and trade_enabled and not force_scan_pending:
                    mon["global_state"] = "일시정지 중(연속손실/보호)"
                    monitor_write_throttled(mon, 2.0)
                    time.sleep(1.0)
                else:
                    # 신규 진입 허용 여부(강제스캔 scan_only면 '강제로 추가 호출된 AI'로는 진입 금지)
                    weekend_block_now = cfg.get("no_trade_weekend", False) and (now_kst().weekday() in [5, 6])
                    entry_allowed_global = trade_enabled and (not paused_now) and (not weekend_block_now)

                    # 상태 표시(사용자 체감 개선)
                    if force_scan_pending:
                        mon["global_state"] = "강제 스캔 중(/scan)"
                    elif not trade_enabled:
                        mon["global_state"] = "스캔 중(자동매매 OFF)"
                    elif paused_now:
                        mon["global_state"] = "스캔 중(정지: 연속손실 보호)"
                    elif weekend_block_now:
                        mon["global_state"] = "스캔 중(주말: 신규진입 OFF)"
                    else:
                        mon["global_state"] = "스캔/매매 중"

                    # ✅ CCXT timeout 발생 시 exchange refresh 플래그
                    need_exchange_refresh = False

                    # 1) 포지션 관리
                    open_pos_snapshot = []

                    # ✅ 포지션은 1회 스냅샷으로 사용(API 호출 최소화)
                    pos_by_sym: Dict[str, Dict[str, Any]] = {}
                    try:
                        # stage 먼저 기록(fetch_positions가 느려도 UI에 표시)
                        try:
                            mon["loop_stage"] = "FETCH_POSITIONS"
                            mon["loop_stage_kst"] = now_kst_str()
                            mon["last_heartbeat_epoch"] = time.time()
                            mon["last_heartbeat_kst"] = mon["loop_stage_kst"]
                            monitor_write_throttled(mon, 0.2)
                        except Exception:
                            pass
                        _to_before_pos = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                        ps_all = safe_fetch_positions(ex, TARGET_COINS)
                        _to_after_pos = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                        if _to_after_pos and _to_after_pos > _to_before_pos:
                            need_exchange_refresh = True
                        for p0 in (ps_all or []):
                            try:
                                sym0 = str(p0.get("symbol") or "").strip()
                                if not sym0:
                                    continue
                                if float(p0.get("contracts") or 0) > 0:
                                    pos_by_sym[sym0] = p0
                            except Exception:
                                continue
                    except Exception:
                        pos_by_sym = {}

                    # ✅ 포지션 스냅샷에서 timeout이 발생했다면, 같은 exchange 인스턴스를 계속 쓰지 않도록 즉시 교체
                    if need_exchange_refresh:
                        try:
                            where_now = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "").strip()
                            mon_add_event(mon, "CCXT_REFRESH", "", "exchange refreshed(after fetch_positions timeout)", {"where": where_now, "code": CODE_VERSION})
                            ex_new = create_exchange_client_uncached()
                            if ex_new is not None:
                                ex = ex_new
                                pos_by_sym = {}
                                # loop-start 마커 갱신(새 인스턴스 기준)
                                ccxt_timeout_epoch_loop_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                                ccxt_timeout_where_loop_start = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "")
                                need_exchange_refresh = False
                        except Exception:
                            pass

                    # ✅ 포지션 관리는 "항상" 수행해야 함(자동매매 OFF/일시정지/주말이어도 청산은 계속 필요)
                    for sym in TARGET_COINS:
                        p = pos_by_sym.get(sym)
                        if not p:
                            continue
                        ai_exit_only = bool(cfg.get("exit_ai_targets_only", False))
                        side = position_side_normalize(p)
                        contracts = float(p.get("contracts") or 0)
                        entry = float(p.get("entryPrice") or 0)
                        roi = float(position_roi_percent(p))
                        cur_px = get_last_price(ex, sym) or entry
                        lev_live = _pos_leverage(p)
                        upnl = float(p.get("unrealizedPnl") or 0.0)
                        # ✅ percentage 없을 때 가격 기반 ROI 폴백(거래소 지연/미제공 대응)
                        if p.get("percentage") is None and float(entry or 0) > 0 and float(cur_px or 0) > 0:
                            roi = float(estimate_roi_from_price(float(entry), float(cur_px), str(side), float(lev_live or 1)))

                        tgt = active_targets.get(
                            sym,
                            {
                                "sl": (None if ai_exit_only else 2.0),
                                "tp": (None if ai_exit_only else 5.0),
                                "entry_usdt": 0.0,
                                "entry_pct": 0.0,
                                "lev": p.get("leverage", "?"),
                                "reason": "",
                                "trade_id": "",
                                "sl_price": None,
                                "tp_price": None,
                                "sl_price_pct": None,
                                "style": "스캘핑",
                                "entry_epoch": time.time(),
                                "style_last_switch_epoch": time.time(),
                            },
                        )

                        # ✅ 수동포지션/복구포지션에서도 타겟/스타일 상태를 "in-memory"에 고정
                        # - active_targets에 없으면 매 루프 default dict로 재생성되어
                        #   스타일 전환/confirm2 상태가 리셋되며, 같은 이유로 반복 전환(플랩)될 수 있음.
                        try:
                            if not isinstance(tgt, dict):
                                tgt = {}
                            default_sl = (None if ai_exit_only else 2.0)
                            default_tp = (None if ai_exit_only else 5.0)
                            base_tgt = {
                                "sl": default_sl,
                                "tp": default_tp,
                                "entry_usdt": 0.0,
                                "entry_pct": 0.0,
                                "entry_price": float(entry) if entry else 0.0,
                                "lev": p.get("leverage", "?"),
                                "reason": "",
                                "trade_id": "",
                                "sl_price": None,
                                "tp_price": None,
                                "sl_price_pct": None,
                                "tp_price_pct": None,
                                "sl_price_source": "",
                                "tp_price_source": "",
                                "sr_used": {},
                                "sl_price_ai": None,
                                "tp_price_ai": None,
                                "style": "스캘핑",
                                "entry_epoch": time.time(),
                                "style_last_switch_epoch": time.time(),
                            }
                            for k0, v0 in base_tgt.items():
                                if k0 not in tgt:
                                    tgt[k0] = v0
                            # entry_price는 거래소 포지션 값으로 매 루프 보정(수동포지션/복구포지션 대응)
                            try:
                                if float(entry or 0) > 0:
                                    tgt["entry_price"] = float(entry)
                            except Exception:
                                pass
                            active_targets[sym] = tgt
                        except Exception:
                            pass

                        forced_exit = bool(cfg.get("exit_trailing_protect_enable", False)) and (not ai_exit_only)
                        # ✅ 스타일 자동 전환(포지션 보유 중)
                        # - 강제 Exit(수익보존) 정책이 ON이면, 스타일 전환/목표(tp/sl) 보정을 멈추고 "진입 당시 값"을 고정한다.
                        #   (스윙↔스캘핑 반복 전환 + 목표 손익비가 계속 바뀌는 현상 방지)
                        if (not forced_exit) and (not ai_exit_only):
                            tgt = _maybe_switch_style_for_open_position(ex, sym, side, tgt, cfg, mon)
                        style_now = str(tgt.get("style", "스캘핑"))
                        try:
                            tgt["exit_trailing_protect_enable"] = bool(forced_exit)
                        except Exception:
                            pass

                        # 저장(스레드 재시작 대비)
                        rt.setdefault("open_targets", {})[sym] = tgt
                        save_runtime(rt)

                        sl = float(abs(_as_float(tgt.get("sl", None), 0.0)))
                        tp = float(abs(_as_float(tgt.get("tp", None), 0.0)))
                        if not ai_exit_only:
                            if (not math.isfinite(sl)) or sl <= 0:
                                sl = 2.0
                            if (not math.isfinite(tp)) or tp <= 0:
                                tp = 5.0
                        trade_id = str(tgt.get("trade_id") or "")
                        ai_targets_ready = bool(math.isfinite(sl) and math.isfinite(tp) and sl > 0 and tp > 0)
                        if ai_exit_only and (not ai_targets_ready):
                            try:
                                now_ep = time.time()
                                last_warn = float(tgt.get("ai_exit_missing_warn_epoch", 0) or 0.0)
                                if (now_ep - last_warn) >= 120.0:
                                    tgt["ai_exit_missing_warn_epoch"] = float(now_ep)
                                    tgt["ai_exit_missing_warn_kst"] = now_kst_str()
                                    mon_add_event(
                                        mon,
                                        "AI_EXIT_WAIT",
                                        sym,
                                        "AI 목표 TP/SL 없음: 청산 대기",
                                        {"trade_id": trade_id, "code": CODE_VERSION},
                                    )
                            except Exception:
                                pass

                        # ✅ 스윙은 "길게 가져가는" 매매:
                        # - 스윙인데 -2~-3% 같은 짧은 손절로 잘리는 문제를 줄이기 위해,
                        #   오픈 포지션에서도 하한(SL)과 최소 손익비(RR)를 강제 보정한다.
                        try:
                            if (not forced_exit) and (not ai_exit_only) and style_now == "스윙":
                                changed_targets = False
                                sl_min = float(cfg.get("swing_sl_roi_min", 12.0))
                                if sl < sl_min:
                                    sl = float(sl_min)
                                    tgt["sl"] = float(sl_min)
                                    changed_targets = True

                                rr_min_now = max(float(_rr_min_by_mode(str(mode))), float(_rr_min_by_style("스윙")))
                                tp_need = abs(float(sl)) * float(rr_min_now)
                                if tp < float(tp_need):
                                    tp_cap = float(cfg.get("swing_tp_roi_max", 50.0))
                                    tp_new = float(clamp(tp_need, float(cfg.get("swing_tp_roi_min", 3.0)), tp_cap))
                                    tp = float(tp_new)
                                    tgt["tp"] = float(tp_new)
                                    changed_targets = True

                                if changed_targets:
                                    # 가격 기준 퍼센트 갱신(레버 기준)
                                    try:
                                        lev0 = float(tgt.get("lev", lev_live) or lev_live or 1.0)
                                    except Exception:
                                        lev0 = float(lev_live or 1.0) or 1.0
                                    if lev0 <= 0:
                                        lev0 = 1.0
                                    tgt["sl_price_pct"] = float(abs(float(sl)) / max(float(lev0), 1.0))
                                    tgt["tp_price_pct"] = float(abs(float(tp)) / max(float(lev0), 1.0))

                                    # SR 가격 라인도 최신 목표(가격폭)에 맞춰 재계산(가능할 때만)
                                    try:
                                        if cfg.get("use_sr_stop", True):
                                            dec2 = "buy" if side == "long" else "sell"
                                            try:
                                                entry_px0 = float(tgt.get("entry_price", entry) or entry or 0.0)
                                            except Exception:
                                                entry_px0 = float(entry or 0.0)
                                            if entry_px0 > 0:
                                                sr_res2 = sr_prices_for_style(
                                                    ex,
                                                    sym,
                                                    entry_price=float(entry_px0),
                                                    side=str(dec2),
                                                    style="스윙",
                                                    cfg=cfg,
                                                    sl_price_pct=float(tgt.get("sl_price_pct", 0.0) or 0.0),
                                                    tp_price_pct=float(tgt.get("tp_price_pct", 0.0) or 0.0),
                                                    ai_sl_price=None,
                                                    ai_tp_price=None,
                                                )
                                                if isinstance(sr_res2, dict):
                                                    tgt["sl_price"] = sr_res2.get("sl_price", tgt.get("sl_price"))
                                                    tgt["tp_price"] = sr_res2.get("tp_price", tgt.get("tp_price"))
                                                    tgt["sl_price_source"] = str(sr_res2.get("sl_source", "") or "")
                                                    tgt["tp_price_source"] = str(sr_res2.get("tp_source", "") or "")
                                                    tgt["sr_used"] = {
                                                        "tf": sr_res2.get("tf", ""),
                                                        "lookback": sr_res2.get("lookback", 0),
                                                        "pivot_order": sr_res2.get("pivot_order", 0),
                                                        "buffer_atr_mult": sr_res2.get("buffer_atr_mult", 0.0),
                                                        "rr_min": sr_res2.get("rr_min", 0.0),
                                                    }
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # ✅ 스캘핑: 포지션 보유 중에도 "가격%" 가드레일을 유지해 TP/SL 과도 방지
                        # - (중요) 스캘핑인데 TP/SL이 커져 +50%가 넘어도 익절을 못 하는 문제를 줄임
                        try:
                            if (not forced_exit) and (not ai_exit_only) and style_now == "스캘핑":
                                changed_targets = False
                                try:
                                    lev0 = float(tgt.get("lev", lev_live) or lev_live or 1.0)
                                except Exception:
                                    lev0 = float(lev_live or 1.0) or 1.0
                                if lev0 <= 0:
                                    lev0 = 1.0

                                # 현재 TP/SL(ROI%) -> 가격% 변환
                                try:
                                    sl_price_pct0 = abs(float(sl)) / max(float(lev0), 1.0)
                                except Exception:
                                    sl_price_pct0 = 0.0
                                try:
                                    tp_price_pct0 = abs(float(tp)) / max(float(lev0), 1.0)
                                except Exception:
                                    tp_price_pct0 = 0.0

                                sl_min = float(cfg.get("scalp_sl_price_pct_min", 0.25))
                                sl_max = float(cfg.get("scalp_sl_price_pct_max", 1.0))
                                tp_min = float(cfg.get("scalp_tp_price_pct_min", 0.35))
                                tp_max = float(cfg.get("scalp_tp_price_pct_max", 1.6))
                                rr_min_price = float(cfg.get("scalp_rr_min_price", 1.2))

                                sl_price_pct = float(clamp(float(sl_price_pct0), sl_min, sl_max))
                                tp_price_pct = float(clamp(float(tp_price_pct0), tp_min, tp_max))
                                # 가격 기준 RR 하한(너무 작은 TP 방지)
                                if rr_min_price > 0 and tp_price_pct < (sl_price_pct * rr_min_price):
                                    tp_price_pct = float(clamp(sl_price_pct * rr_min_price, tp_min, tp_max))

                                sl_new = float(sl_price_pct * float(lev0))
                                tp_new = float(tp_price_pct * float(lev0))

                                if (abs(sl_new - float(sl)) > 0.05) or (abs(tp_new - float(tp)) > 0.05):
                                    sl = float(sl_new)
                                    tp = float(tp_new)
                                    tgt["sl"] = float(sl_new)
                                    tgt["tp"] = float(tp_new)
                                    changed_targets = True

                                # price%도 저장(후속 트레일링/SR 보정에 사용)
                                tgt["sl_price_pct"] = float(sl_price_pct)
                                tgt["tp_price_pct"] = float(tp_price_pct)

                                # SR/ROI 바운드 가격 라인 갱신(필요할 때만)
                                try:
                                    if changed_targets or (tgt.get("sl_price") is None) or (tgt.get("tp_price") is None):
                                        dec2 = "buy" if side == "long" else "sell"
                                        try:
                                            entry_px0 = float(tgt.get("entry_price", entry) or entry or 0.0)
                                        except Exception:
                                            entry_px0 = float(entry or 0.0)
                                        if entry_px0 > 0:
                                            if cfg.get("use_sr_stop", True):
                                                sr_res2 = sr_prices_for_style(
                                                    ex,
                                                    sym,
                                                    entry_price=float(entry_px0),
                                                    side=str(dec2),
                                                    style="스캘핑",
                                                    cfg=cfg,
                                                    sl_price_pct=float(sl_price_pct),
                                                    tp_price_pct=float(tp_price_pct),
                                                    ai_sl_price=tgt.get("sl_price_ai", None),
                                                    ai_tp_price=tgt.get("tp_price_ai", None),
                                                )
                                                if isinstance(sr_res2, dict):
                                                    tgt["sl_price"] = sr_res2.get("sl_price", tgt.get("sl_price"))
                                                    tgt["tp_price"] = sr_res2.get("tp_price", tgt.get("tp_price"))
                                                    tgt["sl_price_source"] = str(sr_res2.get("sl_source", "") or "")
                                                    tgt["tp_price_source"] = str(sr_res2.get("tp_source", "") or "")
                                                    tgt["sr_used"] = {
                                                        "tf": sr_res2.get("tf", ""),
                                                        "lookback": sr_res2.get("lookback", 0),
                                                        "pivot_order": sr_res2.get("pivot_order", 0),
                                                        "buffer_atr_mult": sr_res2.get("buffer_atr_mult", 0.0),
                                                        "rr_min": sr_res2.get("rr_min", 0.0),
                                                    }
                                            # SR 실패 시 fallback ROI bounds
                                            if tgt.get("sl_price") is None or tgt.get("tp_price") is None:
                                                slb2, tpb2 = _sr_price_bounds_from_price_pct(float(entry_px0), str(dec2), float(sl_price_pct), float(tp_price_pct))
                                                if tgt.get("sl_price") is None:
                                                    tgt["sl_price"] = float(slb2)
                                                    tgt["sl_price_source"] = str(tgt.get("sl_price_source", "") or "") or "ROI"
                                                if tgt.get("tp_price") is None:
                                                    tgt["tp_price"] = float(tpb2)
                                                    tgt["tp_price_source"] = str(tgt.get("tp_price_source", "") or "") or "ROI"
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # 트레일링(기존): 강제 수익보존 Exit 정책이 ON이면 사용하지 않음(Exit는 강제 정책이 우선)
                        if (not forced_exit) and (not ai_exit_only) and cfg.get("use_trailing_stop", True):
                            if roi >= (tp * 0.5):
                                lev_now = float(tgt.get("lev", p.get("leverage", 1))) or 1.0
                                base_price_sl = float(tgt.get("sl_price_pct") or max(0.25, float(sl) / max(lev_now, 1)))
                                trail_price_pct = max(0.20, base_price_sl * 0.60)
                                trail_roi = trail_price_pct * lev_now
                                sl = min(sl, max(1.2, float(trail_roi)))

                        # SR 가격 트리거
                        hit_sl_by_price = False
                        hit_tp_by_price = False
                        sl_price = tgt.get("sl_price")
                        tp_price = tgt.get("tp_price")
                        # ✅ 본전 보호(브레이크이븐): 수익이 어느 정도 나면 SL을 진입가 근처로 끌어올림(가격 기준)
                        # - 손절이 아니라 "수익 보호" 목적(연속 손절 카운트에도 포함하지 않게 별도 처리)
                        try:
                            if (not forced_exit) and (not ai_exit_only) and bool(cfg.get("trail_breakeven_enable", True)):
                                prev_sl_src = ""
                                try:
                                    prev_sl_src = str(tgt.get("sl_price_source", "") or "").strip().upper()
                                except Exception:
                                    prev_sl_src = ""
                                at_roi = float(
                                    cfg.get(
                                        "trail_breakeven_at_roi_scalp" if str(style_now) == "스캘핑" else "trail_breakeven_at_roi_swing",
                                        8.0,
                                    )
                                )
                                off_pct = float(cfg.get("trail_breakeven_offset_price_pct", 0.05))
                                if float(roi) >= float(at_roi):
                                    try:
                                        entry_px_be = float(tgt.get("entry_price", entry) or entry or 0.0)
                                    except Exception:
                                        entry_px_be = float(entry or 0.0)
                                    if entry_px_be > 0:
                                        if side == "long":
                                            be_price = entry_px_be * (1.0 + (off_pct / 100.0))
                                            if float(be_price) < float(cur_px):
                                                if sl_price is None or float(sl_price) < float(be_price):
                                                    sl_price = float(be_price)
                                                    tgt["sl_price"] = float(be_price)
                                                    tgt["sl_price_source"] = "BE"
                                                    try:
                                                        tgt["be_arm_price"] = float(be_price)
                                                        tgt["be_arm_at_roi"] = float(at_roi)
                                                        tgt["be_arm_offset_pct"] = float(off_pct)
                                                        if prev_sl_src != "BE":
                                                            tgt["be_arm_time_kst"] = now_kst_str()
                                                            tgt["be_arm_epoch"] = time.time()
                                                            tgt["be_arm_roi"] = float(roi)
                                                            if not str(tgt.get("be_arm_ind", "") or "").strip():
                                                                snap_be = chart_snapshot_for_reason(ex, sym, cfg)
                                                                entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                                                tgt["be_arm_ind"] = _fmt_indicator_line_for_reason(entry_snap, snap_be)
                                                    except Exception:
                                                        pass
                                        else:
                                            be_price = entry_px_be * (1.0 - (off_pct / 100.0))
                                            if float(be_price) > float(cur_px):
                                                if sl_price is None or float(sl_price) > float(be_price):
                                                    sl_price = float(be_price)
                                                    tgt["sl_price"] = float(be_price)
                                                    tgt["sl_price_source"] = "BE"
                                                    try:
                                                        tgt["be_arm_price"] = float(be_price)
                                                        tgt["be_arm_at_roi"] = float(at_roi)
                                                        tgt["be_arm_offset_pct"] = float(off_pct)
                                                        if prev_sl_src != "BE":
                                                            tgt["be_arm_time_kst"] = now_kst_str()
                                                            tgt["be_arm_epoch"] = time.time()
                                                            tgt["be_arm_roi"] = float(roi)
                                                            if not str(tgt.get("be_arm_ind", "") or "").strip():
                                                                snap_be = chart_snapshot_for_reason(ex, sym, cfg)
                                                                entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                                                tgt["be_arm_ind"] = _fmt_indicator_line_for_reason(entry_snap, snap_be)
                                                    except Exception:
                                                        pass
                        except Exception:
                            pass
                        if (not forced_exit) and (not ai_exit_only) and cfg.get("use_sr_stop", True):
                            if sl_price is not None:
                                if side == "long" and cur_px <= float(sl_price):
                                    hit_sl_by_price = True
                                if side == "short" and cur_px >= float(sl_price):
                                    hit_sl_by_price = True
                            if tp_price is not None:
                                if side == "long" and cur_px >= float(tp_price):
                                    hit_tp_by_price = True
                                if side == "short" and cur_px <= float(tp_price):
                                    hit_tp_by_price = True

                        # ✅ 스윙: 부분익절(순환매도 옵션) - 요구사항 반영
                        if (not forced_exit) and (not ai_exit_only) and style_now == "스윙" and cfg.get("swing_partial_tp_enable", True) and contracts > 0:
                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                            done = set(trade_state.get("partial_tp_done", []) or [])
                            # ✅ 스윙 분할익절:
                            # - 우선순위: 진입 시점에 지정된 SR/매물대 2개 가격 라인(TP1/TP2)
                            # - 없으면 기존 ROI 비율 트리거로 fallback
                            levels_exec: List[Tuple[str, float, str, float]] = []
                            try:
                                p1 = _as_float(tgt.get("partial_tp1_price", None), float("nan"))
                                p2 = _as_float(tgt.get("partial_tp2_price", None), float("nan"))
                                if math.isfinite(p1):
                                    levels_exec.append(("TP1", float(cfg.get("swing_partial_tp1_close_pct", 33)) / 100.0, "price", float(p1)))
                                if math.isfinite(p2):
                                    levels_exec.append(("TP2", float(cfg.get("swing_partial_tp2_close_pct", 33)) / 100.0, "price", float(p2)))
                            except Exception:
                                levels_exec = []
                            if not levels_exec:
                                for trig_roi, close_frac, label in _swing_partial_tp_levels(tp, cfg):
                                    levels_exec.append((str(label), float(close_frac), "roi", float(trig_roi)))
                            contracts_left = contracts
                            for label, close_frac, trig_kind, trig_v in levels_exec:
                                if label in done:
                                    continue
                                if contracts_left <= 0:
                                    continue
                                hit_partial = False
                                if str(trig_kind) == "price":
                                    if side == "long" and float(cur_px) >= float(trig_v):
                                        hit_partial = True
                                    elif side == "short" and float(cur_px) <= float(trig_v):
                                        hit_partial = True
                                else:
                                    if float(roi) >= float(trig_v):
                                        hit_partial = True
                                if hit_partial:
                                    # ✅ (추가) 부분익절 청산수량을 USDT(마진)로 지정 가능
                                    close_usdt_cfg = 0.0
                                    try:
                                        if label == "TP1":
                                            close_usdt_cfg = float(cfg.get("swing_partial_tp1_close_usdt", 0.0) or 0.0)
                                        elif label == "TP2":
                                            close_usdt_cfg = float(cfg.get("swing_partial_tp2_close_usdt", 0.0) or 0.0)
                                        elif label == "TP3":
                                            close_usdt_cfg = float(cfg.get("swing_partial_tp3_close_usdt", 0.0) or 0.0)
                                    except Exception:
                                        close_usdt_cfg = 0.0
                                    try:
                                        lev_for_calc = float(lev_live or 0.0)
                                    except Exception:
                                        lev_for_calc = 0.0
                                    if lev_for_calc <= 0:
                                        try:
                                            lev_for_calc = float(tgt.get("lev", 1) or 1)
                                        except Exception:
                                            lev_for_calc = 1.0
                                    close_mode = "pct"
                                    if float(close_usdt_cfg) > 0 and float(cur_px) > 0:
                                        close_mode = "usdt"
                                        close_qty_raw = (float(close_usdt_cfg) * float(lev_for_calc)) / max(float(cur_px), 1e-9)
                                        close_qty = to_precision_qty(ex, sym, min(float(contracts_left), float(close_qty_raw)))
                                    else:
                                        close_qty = to_precision_qty(ex, sym, contracts_left * close_frac)
                                    # 너무 작은 수량은 스킵
                                    if close_qty <= 0:
                                        done.add(label)
                                        continue
                                    ok, err_close = close_position_market_ex(ex, sym, side, close_qty)
                                    if ok:
                                        done.add(label)
                                        # 청산 마진(추정): notional/lev
                                        try:
                                            close_margin_est = (float(close_qty) * float(cur_px)) / max(float(lev_for_calc), 1.0)
                                        except Exception:
                                            close_margin_est = 0.0
                                        # 순환매도(재진입)용 메모리: 부분익절 수량 누적 + 타임스탬프
                                        try:
                                            trade_state["last_partial_tp_epoch"] = time.time()
                                            trade_state["recycle_qty"] = float(trade_state.get("recycle_qty", 0.0) or 0.0) + float(close_qty)
                                        except Exception:
                                            pass
                                        trade_state["partial_tp_done"] = list(done)
                                        save_runtime(rt)
                                        contracts_left = max(0.0, contracts_left - close_qty)
                                        close_txt = f"{float(close_usdt_cfg):.2f}USDT" if close_mode == "usdt" else f"{close_frac*100:.0f}%"
                                        trig_note = f"{label}@{float(trig_v):.6g}" if str(trig_kind) == "price" else f"{label}@ROI{float(trig_v):.2f}%"
                                        mon_add_event(mon, "PARTIAL_TP", sym, f"{label} 부분익절({close_txt})", {"roi": roi, "qty": close_qty, "margin_usdt_est": close_margin_est, "mode": close_mode, "trigger": trig_note})
                                        try:
                                            gsheet_log_trade(
                                                stage="PARTIAL_TP",
                                                symbol=sym,
                                                trade_id=trade_id,
                                                message=f"{label} close_qty={close_qty}",
                                                payload={"label": label, "roi": roi, "qty": close_qty, "contracts_left": contracts_left, "margin_usdt_est": close_margin_est, "mode": close_mode, "trigger": trig_note},
                                            )
                                        except Exception:
                                            pass
                                        # 텔레그램 채널 보고
                                        if _tg_simple_enabled(cfg):
                                            msg = (
                                                f"🧩 부분익절({label})\n"
                                                f"- 코인: {sym}\n"
                                                f"- 방식: 스윙\n"
                                                f"- 포지션: {_tg_dir_easy(side)}\n"
                                                "\n"
                                                f"- 지금 수익률: {_tg_fmt_pct(roi)}\n"
                                                f"- 청산금액(마진): {float(close_margin_est):.2f} USDT\n"
                                                f"- 청산수량: {close_qty}\n"
                                                f"- 남은수량: {contracts_left}\n"
                                                "\n"
                                                f"- ID: {trade_id or '-'}"
                                            )
                                        else:
                                            msg = (
                                                f"🧩 부분익절({label})\n- 코인: {sym}\n- 스타일: 스윙\n- ROI: +{roi:.2f}%\n- 청산수량: {close_qty}\n- 청산마진(추정): {close_margin_est:.2f} USDT\n- 남은수량: {contracts_left}\n- 일지ID: {trade_id or '-'}"
                                            )
                                        tg_send(
                                            msg,
                                            target=cfg.get("tg_route_events_to", "channel"),
                                            cfg=cfg,
                                            silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                                        )
                                        # 상세일지 기록
                                        if trade_id:
                                            d = load_trade_detail(trade_id) or {}
                                            evs = d.get("events", []) or []
                                            evs.append({"time": now_kst_str(), "type": "PARTIAL_TP", "label": label, "roi": roi, "qty": close_qty, "margin_usdt_est": close_margin_est, "mode": close_mode})
                                            d["events"] = evs
                                            save_trade_detail(trade_id, d)
                                    else:
                                        mon_add_event(mon, "ORDER_FAIL", sym, f"부분익절 실패({label})", {"err": err_close, "qty": close_qty, "roi": roi, "trade_id": trade_id})
                                        try:
                                            notify_admin_error(
                                                where="ORDER:PARTIAL_TP_CLOSE",
                                                err=RuntimeError(str(err_close)),
                                                context={"symbol": sym, "label": label, "qty": close_qty, "roi": roi, "trade_id": trade_id},
                                                tb="",
                                                min_interval_sec=60.0,
                                            )
                                        except Exception:
                                            pass

                        # ✅ 스윙: 순환매도(부분익절 후 재진입/리밸런싱) - 옵션 ON일 때만
                        if (not forced_exit) and style_now == "스윙" and cfg.get("swing_recycle_enable", False) and contracts > 0:
                            try:
                                trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                                rc = int(trade_state.get("recycle_count", 0) or 0)
                                rc_max = int(cfg.get("swing_recycle_max_count", 2))
                                cooldown = int(cfg.get("swing_recycle_cooldown_min", 20)) * 60
                                last_tp_epoch = float(trade_state.get("last_partial_tp_epoch", 0) or 0)
                                qty_avail = float(trade_state.get("recycle_qty", 0.0) or 0.0)
                                reentry_roi = float(cfg.get("swing_recycle_reentry_roi", 0.8))

                                if rc < rc_max and qty_avail > 0 and last_tp_epoch > 0:
                                    if (time.time() - last_tp_epoch) >= cooldown and roi <= reentry_roi:
                                        # 추세가 계속 같은 방향이면 재진입(리밸런싱)
                                        short_tf = str(cfg.get("timeframe", "5m"))
                                        long_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                                        fast = int(cfg.get("ma_fast", 7))
                                        slow = int(cfg.get("ma_slow", 99))
                                        short_tr = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=30)
                                        long_tr = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))

                                        if _trend_align(short_tr, side) and _trend_align(long_tr, side):
                                            lev = int(float(tgt.get("lev", 1)) or 1)
                                            free, _ = safe_fetch_balance(ex)
                                            margin_need = (qty_avail * cur_px) / max(lev, 1)
                                            if margin_need <= free * 0.9:
                                                set_leverage_safe(ex, sym, lev)
                                                qty_re = to_precision_qty(ex, sym, qty_avail)
                                                if qty_re > 0:
                                                        ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty_re)
                                                        if ok:
                                                            trade_state["recycle_count"] = rc + 1
                                                            trade_state["recycle_qty"] = max(0.0, qty_avail - float(qty_re))
                                                            save_runtime(rt)
                                                            # ✅ 순환매도도 "왜 재진입하는지" 남기기(차트 스냅샷, AI 호출 없음)
                                                            snap_re = {}
                                                            try:
                                                                snap_re = chart_snapshot_for_reason(ex, sym, cfg)
                                                            except Exception:
                                                                snap_re = {}
                                                            entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                                            why_re = _fmt_indicator_line_for_reason(entry_snap, snap_re) or f"단기:{short_tr} | 장기:{long_tr}"
                                                            mon_add_event(mon, "RECYCLE_REENTRY", sym, f"재진입 {qty_re}", {"roi": roi, "trend": f"{short_tr}/{long_tr}"})
                                                            try:
                                                                gsheet_log_trade(
                                                                    stage="RECYCLE_REENTRY",
                                                                    symbol=sym,
                                                                    trade_id=trade_id,
                                                                    message=f"qty={qty_re}",
                                                                    payload={"roi": roi, "qty": qty_re, "trend": f"{short_tr}/{long_tr}", "recycle_count": rc + 1},
                                                                )
                                                            except Exception:
                                                                pass
                                                            if _tg_simple_enabled(cfg):
                                                                why_line = f"- 근거: {why_re}\n" if why_re else ""
                                                                msg = (
                                                                    "♻️ 순환매도(재진입)\n"
                                                                    f"- 코인: {sym}\n"
                                                                    f"- 방식: 스윙\n"
                                                                    f"- 포지션: {_tg_dir_easy(side)}\n"
                                                                    "\n"
                                                                    f"- 재진입금액(마진): {float(margin_need):.2f} USDT\n"
                                                                    f"- 재진입수량: {qty_re}\n"
                                                                    f"- 지금 수익률: {_tg_fmt_pct(roi)}\n"
                                                                    f"{why_line}"
                                                                    "\n"
                                                                    f"- ID: {trade_id or '-'}"
                                                                )
                                                            else:
                                                                msg = (
                                                                    f"♻️ 순환매도 재진입\n- 코인: {sym}\n- 스타일: 스윙\n- 재진입수량: {qty_re}\n"
                                                                    f"- 조건: ROI {roi:.2f}% <= {reentry_roi}%\n- 단기({short_tf}): {short_tr}\n- 장기({long_tf}): {long_tr}\n"
                                                                    f"- 일지ID: {trade_id or '-'}"
                                                                )
                                                            tg_send(
                                                                msg,
                                                                target=cfg.get("tg_route_events_to", "channel"),
                                                                cfg=cfg,
                                                                silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                                                            )
                                                            if trade_id:
                                                                d = load_trade_detail(trade_id) or {}
                                                                evs = d.get("events", []) or []
                                                                evs.append({"time": now_kst_str(), "type": "RECYCLE_REENTRY", "roi": roi, "qty": qty_re})
                                                                d["events"] = evs
                                                                save_trade_detail(trade_id, d)
                            except Exception:
                                pass

                        # ✅ 스캘핑 -> 스윙 전환 조건(보유시간/정렬) + 필요시 추매
                        try:
                            short_tf = str(cfg.get("timeframe", "5m"))
                            long_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                            fast = int(cfg.get("ma_fast", 7))
                            slow = int(cfg.get("ma_slow", 99))
                            short_tr = get_htf_trend_cached(ex, sym, short_tf, fast=fast, slow=slow, cache_sec=25)
                            long_tr = get_htf_trend_cached(ex, sym, long_tf, fast=fast, slow=slow, cache_sec=int(cfg.get("trend_filter_cache_sec", 60)))
                            long_align = _trend_align(long_tr, side)
                            if _should_convert_scalp_to_swing(tgt, roi, cfg, long_align=long_align):
                                # 전환 + (선택) 1회 추매
                                try:
                                    old_tp_conv = float(tgt.get("tp", 0) or 0.0)
                                except Exception:
                                    old_tp_conv = 0.0
                                try:
                                    old_sl_conv = float(tgt.get("sl", 0) or 0.0)
                                except Exception:
                                    old_sl_conv = 0.0
                                did_dca = _try_scalp_to_swing_dca(ex, sym, side, cur_px, tgt, rt, cfg, mon)
                                tgt["style"] = "스윙"
                                tgt["style_reason"] = f"스캘핑 장기화({cfg.get('scalp_max_hold_minutes',25)}m+) → 스윙 전환"
                                tgt["style_last_switch_epoch"] = time.time()
                                # ✅ 보유시간 전환 플래그(같은 포지션에서 스윙→스캘핑 되돌림 방지)
                                try:
                                    tgt["_hold_convert_to_swing"] = True
                                except Exception:
                                    pass
                                # 스윙 목표로 확장
                                tgt["tp"] = float(clamp(max(tp, float(cfg.get("swing_tp_roi_min", 3.0))), float(cfg.get("swing_tp_roi_min", 3.0)), float(cfg.get("swing_tp_roi_max", 50.0))))
                                tgt["sl"] = float(clamp(max(sl, float(cfg.get("swing_sl_roi_min", 12.0))), float(cfg.get("swing_sl_roi_min", 12.0)), float(cfg.get("swing_sl_roi_max", 30.0))))
                                # ✅ 스윙 전환이면 "손절폭을 넓힌 만큼 익절도 같이" 늘려서 손익비가 나빠지지 않게 한다.
                                # - (중요) 손절만 넓히고 익절은 그대로면, 이미 수익 중인 포지션에서 되레 수익 반납 리스크만 커질 수 있음
                                try:
                                    mode_now = str(cfg.get("trade_mode", "안전모드") or "안전모드")
                                    rr_min_now = max(float(_rr_min_by_mode(mode_now)), float(_rr_min_by_style("스윙")))
                                    sl_now = float(tgt.get("sl", 0) or 0.0)
                                    tp_now = float(tgt.get("tp", 0) or 0.0)
                                    tp_need = abs(sl_now) * float(rr_min_now)
                                    if tp_now < tp_need:
                                        tp_cap = float(cfg.get("swing_tp_roi_max", 50.0))
                                        tgt["tp"] = float(clamp(tp_need, float(cfg.get("swing_tp_roi_min", 3.0)), tp_cap))
                                except Exception:
                                    pass
                                # ✅ 전환 시 SR 가격 라인도 스윙 기준으로 재계산(너무 타이트한 SL 방지)
                                try:
                                    lev0 = float(tgt.get("lev", 1) or 1)
                                except Exception:
                                    lev0 = 1.0
                                try:
                                    sl_roi0 = float(tgt.get("sl", 0) or 0.0)
                                except Exception:
                                    sl_roi0 = 0.0
                                try:
                                    tp_roi0 = float(tgt.get("tp", 0) or 0.0)
                                except Exception:
                                    tp_roi0 = 0.0
                                sl_price_pct0 = abs(sl_roi0) / max(lev0, 1.0) if lev0 else abs(sl_roi0)
                                tp_price_pct0 = abs(tp_roi0) / max(lev0, 1.0) if lev0 else abs(tp_roi0)
                                tgt["sl_price_pct"] = float(sl_price_pct0)
                                tgt["tp_price_pct"] = float(tp_price_pct0)
                                try:
                                    tgt["sl_price_ai"] = None
                                    tgt["tp_price_ai"] = None
                                except Exception:
                                    pass
                                try:
                                    if cfg.get("use_sr_stop", True):
                                        dec2 = "buy" if str(side) == "long" else "sell"
                                        entry_px0 = 0.0
                                        try:
                                            entry_px0 = float(tgt.get("entry_price", 0) or 0.0)
                                        except Exception:
                                            entry_px0 = 0.0
                                        if entry_px0 <= 0:
                                            entry_px0 = float(entry or 0.0)
                                        if entry_px0 <= 0:
                                            entry_px0 = float(cur_px or 0.0)
                                        if entry_px0 > 0:
                                            sr_res2 = sr_prices_for_style(
                                                ex,
                                                sym,
                                                entry_price=float(entry_px0),
                                                side=str(dec2),
                                                style="스윙",
                                                cfg=cfg,
                                                sl_price_pct=float(sl_price_pct0),
                                                tp_price_pct=float(tp_price_pct0),
                                                ai_sl_price=None,
                                                ai_tp_price=None,
                                            )
                                            if isinstance(sr_res2, dict):
                                                tgt["sl_price"] = sr_res2.get("sl_price", tgt.get("sl_price"))
                                                tgt["tp_price"] = sr_res2.get("tp_price", tgt.get("tp_price"))
                                                tgt["sl_price_source"] = str(sr_res2.get("sl_source", "") or "")
                                                tgt["tp_price_source"] = str(sr_res2.get("tp_source", "") or "")
                                                tgt["sr_used"] = {
                                                    "tf": sr_res2.get("tf", ""),
                                                    "lookback": sr_res2.get("lookback", 0),
                                                    "pivot_order": sr_res2.get("pivot_order", 0),
                                                    "buffer_atr_mult": sr_res2.get("buffer_atr_mult", 0.0),
                                                    "rr_min": sr_res2.get("rr_min", 0.0),
                                                }
                                        # SR 실패 시에도 ROI 바운드로 가격 라인 확보
                                        if tgt.get("sl_price") is None or tgt.get("tp_price") is None:
                                            try:
                                                slb2, tpb2 = _sr_price_bounds_from_price_pct(float(entry_px0), str(dec2), float(sl_price_pct0), float(tp_price_pct0))
                                                if tgt.get("sl_price") is None:
                                                    tgt["sl_price"] = float(slb2)
                                                    if not str(tgt.get("sl_price_source", "") or ""):
                                                        tgt["sl_price_source"] = "ROI"
                                                if tgt.get("tp_price") is None:
                                                    tgt["tp_price"] = float(tpb2)
                                                    if not str(tgt.get("tp_price_source", "") or ""):
                                                        tgt["tp_price_source"] = "ROI"
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                active_targets[sym] = tgt
                                rt.setdefault("open_targets", {})[sym] = tgt
                                save_runtime(rt)
                                mon_add_event(mon, "SCALP_TO_SWING", sym, f"전환 완료(추매:{'Y' if did_dca else 'N'})", {"roi": roi})
                                try:
                                    gsheet_log_trade(
                                        stage="SCALP_TO_SWING",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message=f"did_dca={'Y' if did_dca else 'N'}",
                                        payload={"roi": roi, "did_dca": bool(did_dca)},
                                    )
                                except Exception:
                                    pass
                                try:
                                    new_tp_conv = float(tgt.get("tp", 0) or 0.0)
                                except Exception:
                                    new_tp_conv = 0.0
                                try:
                                    new_sl_conv = float(tgt.get("sl", 0) or 0.0)
                                except Exception:
                                    new_sl_conv = 0.0
                                if _tg_simple_enabled(cfg):
                                    q = _tg_quote_block(str(tgt.get("style_reason", "") or ""))
                                    if not q:
                                        q = "  └ -"
                                    msg = (
                                        "🔄 방식 전환\n"
                                        f"- 코인: {sym}\n"
                                        f"- 스캘핑 → 스윙\n"
                                        f"- 포지션: {_tg_dir_easy(side)}\n"
                                        "\n"
                                        f"- 목표손익비(익절/손절): 익절 {old_tp_conv:+.2f}% / 손절 -{abs(old_sl_conv):.2f}% → 익절 {new_tp_conv:+.2f}% / 손절 -{abs(new_sl_conv):.2f}%\n"
                                        f"- 지금 수익률: {_tg_fmt_pct(roi)}\n"
                                        "\n"
                                        "- 한줄:\n"
                                        f"{q}\n"
                                        f"- ID: {trade_id or '-'}"
                                    )
                                else:
                                    msg = (
                                        f"🔄 스타일 전환\n- 코인: {sym}\n- 스캘핑 → 스윙\n- 이유: {tgt.get('style_reason','')}\n- ROI: {roi:.2f}%\n"
                                        f"- (전환추매): {'있음' if did_dca else '없음'}\n- 일지ID: {trade_id or '-'}"
                                    )
                                tg_send(
                                    msg,
                                    target=cfg.get("tg_route_events_to", "channel"),
                                    cfg=cfg,
                                    silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                                )
                        except Exception:
                            pass

                        # ✅ DCA: 스캘핑은 기본 금지(요구사항), 스윙에서만 허용
                        if cfg.get("use_dca", True) and not (style_now == "스캘핑" and cfg.get("scalp_disable_dca", True)):
                            dca_trig = float(cfg.get("dca_trigger", -20.0))
                            dca_max = int(cfg.get("dca_max_count", 1))
                            dca_add_pct = float(cfg.get("dca_add_pct", 50.0))
                            dca_add_usdt_cfg = 0.0
                            try:
                                dca_add_usdt_cfg = float(cfg.get("dca_add_usdt", 0.0) or 0.0)
                            except Exception:
                                dca_add_usdt_cfg = 0.0

                            trade_state = rt.setdefault("trades", {}).setdefault(sym, {"dca_count": 0, "partial_tp_done": [], "recycle_count": 0})
                            dca_count = int(trade_state.get("dca_count", 0))

                            dca_ready = False
                            dca_trigger_note = ""
                            try:
                                dca_price_line = _as_float(tgt.get("dca_price", None), float("nan"))
                            except Exception:
                                dca_price_line = float("nan")
                            if math.isfinite(dca_price_line) and str(style_now) == "스윙":
                                if (side == "long" and float(cur_px) <= float(dca_price_line)) or (side == "short" and float(cur_px) >= float(dca_price_line)):
                                    dca_ready = True
                                    dca_trigger_note = f"추매라인({float(dca_price_line):.6g}) 도달"
                            elif roi <= dca_trig:
                                dca_ready = True
                                dca_trigger_note = f"ROI {roi:.2f}% <= {dca_trig:.2f}%"

                            if dca_ready and dca_count < dca_max:
                                free, _ = safe_fetch_balance(ex)
                                base_entry = float(tgt.get("entry_usdt", 0.0))
                                # ✅ (추가) USDT 기준 추매(마진) 우선, 없으면 기존 % 방식 유지
                                add_usdt = float(dca_add_usdt_cfg) if float(dca_add_usdt_cfg) > 0 else (base_entry * (dca_add_pct / 100.0))
                                if add_usdt > free:
                                    add_usdt = free * 0.5

                                if cur_px and add_usdt > 5:
                                    lev = int(float(tgt.get("lev", rule["lev_min"])) or rule["lev_min"])
                                    set_leverage_safe(ex, sym, lev)
                                    qty = to_precision_qty(ex, sym, (add_usdt * lev) / cur_px)
                                    ok = market_order_safe(ex, sym, "buy" if side == "long" else "sell", qty)
                                    if ok:
                                        trade_state["dca_count"] = dca_count + 1
                                        save_runtime(rt)
                                        # ✅ 추매도 "왜 하는지" 남기기(차트 스냅샷, AI 호출 없음)
                                        snap_dca = {}
                                        try:
                                            snap_dca = chart_snapshot_for_reason(ex, sym, cfg)
                                        except Exception:
                                            snap_dca = {}
                                        entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                        why_dca = _fmt_indicator_line_for_reason(entry_snap, snap_dca)
                                        # 실제 마진 추정(근사): notional/lev
                                        try:
                                            margin_est = (float(qty) * float(cur_px)) / max(float(lev), 1.0)
                                        except Exception:
                                            margin_est = float(add_usdt)
                                        if _tg_simple_enabled(cfg):
                                            why_line = f"- 근거: {why_dca}\n" if why_dca else ""
                                            msg = (
                                                "💧 추매(DCA)\n"
                                                f"- 코인: {sym}\n"
                                                f"- 방식: {_tg_style_easy(style_now)}\n"
                                                f"- 포지션: {_tg_dir_easy(side)}\n"
                                                "\n"
                                                f"- 추가금액(마진): {float(add_usdt):.2f} USDT\n"
                                                f"- 트리거: {dca_trigger_note}\n"
                                                f"- 지금 수익률: {_tg_fmt_pct(roi)}\n"
                                                f"{why_line}"
                                                "\n"
                                                f"- ID: {trade_id or '-'}"
                                            )
                                        else:
                                            msg = (
                                                f"💧 물타기(DCA)\n- 코인: {sym}\n- 스타일: {style_now}\n- 추가금(마진): {float(add_usdt):.2f} USDT (추정 {float(margin_est):.2f})\n"
                                                f"- 추가수량: {qty}\n- 레버: x{lev}\n- 트리거: {dca_trigger_note}\n- 일지ID: {trade_id or '-'}"
                                            )
                                        tg_send(
                                            msg,
                                            target=cfg.get("tg_route_events_to", "channel"),
                                            cfg=cfg,
                                            silent=bool(cfg.get("tg_notify_entry_exit_only", True)),
                                        )
                                        mon_add_event(mon, "DCA", sym, f"DCA {add_usdt:.2f} USDT", {"roi": roi, "trigger": dca_trigger_note})
                                        try:
                                            gsheet_log_trade(
                                                stage="DCA",
                                                symbol=sym,
                                                trade_id=trade_id,
                                                message=f"add_usdt={add_usdt:.2f}",
                                                payload={"roi": roi, "add_usdt": add_usdt, "qty": qty, "lev": lev, "dca_count": dca_count + 1, "trigger": dca_trigger_note},
                                            )
                                        except Exception:
                                            pass

                        # 스캘핑 전환 청산 모드: 목표를 더 보수적으로(빨리 끝내기)
                        scalp_exit_mode = bool(tgt.get("scalp_exit_mode", False))
                        if scalp_exit_mode:
                            tp = min(tp, float(cfg.get("scalp_tp_roi_max", 6.0)))
                            sl = min(sl, float(cfg.get("scalp_sl_roi_max", 5.0)))

                        # ✅ forced_exit(수익보존)에서도 "AI 목표(TP/SL)"를 우선 적용할 수 있도록(요구)
                        tp_from_ai = False
                        sl_from_ai = False
                        try:
                            tp_plan_roi = float(abs(float(tp)))
                        except Exception:
                            tp_plan_roi = 0.0
                        try:
                            sl_plan_roi = float(abs(float(sl)))
                        except Exception:
                            sl_plan_roi = 0.0

                        # =================================================
                        # ✅ 강제 수익 보존(Trailing Protect) Exit 정책
                        # - 진입(AI)은 유지하되, 청산은 AI를 배제하고 아래 규칙을 우선 적용
                        #   1) +10% → 본전(진입가) 보호
                        #   2) +30% → 50% 부분익절(시장가)
                        #   3) +50% 이후: 최고점 대비 -10%면 전량 청산
                        #   4) 기본 손절: -15%면 전량 손절
                        # =================================================
                        forced_take_reason = ""
                        forced_take_detail = ""
                        forced_trail_hit = False
                        try:
                            if bool(forced_exit):
                                # percentage가 비어있거나 지연될 수 있어, 가격 기반 ROI 추정도 같이 사용(보조)
                                try:
                                    if p.get("percentage") is None:
                                        roi = float(estimate_roi_from_price(float(entry), float(cur_px), str(side), float(lev_live)))
                                except Exception:
                                    pass

                                sl_fixed = float(cfg.get("exit_trailing_protect_sl_roi", 15.0) or 15.0)
                                be_roi = float(cfg.get("exit_trailing_protect_be_roi", 10.0) or 10.0)
                                part_roi = float(cfg.get("exit_trailing_protect_partial_roi", 30.0) or 30.0)
                                part_pct = float(cfg.get("exit_trailing_protect_partial_close_pct", 50.0) or 50.0)
                                trail_start = float(cfg.get("exit_trailing_protect_trail_start_roi", 50.0) or 50.0)
                                trail_dd = float(cfg.get("exit_trailing_protect_trail_dd_roi", 10.0) or 10.0)

                                # 강제 정책에서는 기존 TP/SL 목표는 "표시용"으로만 두고, Exit는 고정 기준을 사용
                                sl = float(abs(sl_fixed))
                                tp = 999999.0
                                try:
                                    if bool(cfg.get("exit_trailing_protect_ai_targets_priority", True)):
                                        if tp_plan_roi > 0 and math.isfinite(float(tp_plan_roi)):
                                            tp = float(tp_plan_roi)
                                            tp_from_ai = True
                                        if sl_plan_roi > 0 and math.isfinite(float(sl_plan_roi)):
                                            sl = float(min(abs(float(sl_fixed)), float(sl_plan_roi)))
                                            sl_from_ai = True
                                except Exception:
                                    tp_from_ai = False
                                    sl_from_ai = False

                                # 1) 본전 보호(진입가): +be_roi% 넘으면 SL을 진입가로 고정
                                try:
                                    if (not bool(tgt.get("forced_be_armed", False))) and (float(roi) >= float(be_roi)):
                                        be_price = float(entry) if float(entry or 0.0) > 0 else float(tgt.get("entry_price", 0.0) or 0.0)
                                        if be_price > 0:
                                            tgt["forced_be_armed"] = True
                                            tgt["forced_be_price"] = float(be_price)
                                            tgt["sl_price"] = float(be_price)
                                            tgt["sl_price_source"] = "BE"
                                            tgt["be_arm_time_kst"] = now_kst_str()
                                            tgt["be_arm_epoch"] = time.time()
                                            tgt["be_arm_roi"] = float(roi)
                                            tgt["be_arm_at_roi"] = float(be_roi)
                                            tgt["be_arm_offset_pct"] = 0.0
                                            tgt["be_arm_price"] = float(be_price)
                                            if not str(tgt.get("be_arm_ind", "") or "").strip():
                                                try:
                                                    snap_be = chart_snapshot_for_reason(ex, sym, cfg)
                                                    entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                                    tgt["be_arm_ind"] = _fmt_indicator_line_for_reason(entry_snap, snap_be)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                                # BE 트리거: 가격이 진입가(본전)로 되돌아오면
                                # - 즉시청산하지 않고, 차트 상태를 한 번 더 판단해 홀딩/청산 결정(요구)
                                try:
                                    if bool(tgt.get("forced_be_armed", False)):
                                        be_price = float(tgt.get("forced_be_price", 0.0) or tgt.get("be_arm_price", 0.0) or 0.0)
                                        if be_price > 0:
                                            be_touch = False
                                            if (str(side) == "long" and float(cur_px) <= float(be_price)) or (str(side) == "short" and float(cur_px) >= float(be_price)):
                                                be_touch = True
                                            if be_touch:
                                                if not bool(cfg.get("be_recheck_enable", True)):
                                                    tgt["sl_price"] = float(be_price)
                                                    tgt["sl_price_source"] = "BE"
                                                    hit_sl_by_price = True
                                                else:
                                                    now_ep = time.time()
                                                    hold_until = float(tgt.get("be_recheck_hold_until_epoch", 0) or 0.0)
                                                    if now_ep < hold_until:
                                                        # 직전 재판단에서 "잠깐 홀딩"이 결정된 구간
                                                        hit_sl_by_price = False
                                                    else:
                                                        try:
                                                            entry_snap_be = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                                                        except Exception:
                                                            entry_snap_be = None
                                                        try:
                                                            now_snap_be = chart_snapshot_for_reason(ex, sym, cfg)
                                                        except Exception:
                                                            now_snap_be = {}

                                                        hold_decision, hold_note = be_recheck_should_hold(
                                                            str(side),
                                                            entry_snap_be if isinstance(entry_snap_be, dict) else None,
                                                            now_snap_be if isinstance(now_snap_be, dict) else None,
                                                            cfg,
                                                        )
                                                        tgt["be_recheck_last_note"] = str(hold_note or "")
                                                        tgt["be_recheck_last_kst"] = now_kst_str()
                                                        try:
                                                            tgt["be_recheck_last_ind"] = _fmt_indicator_line_for_reason(
                                                                entry_snap_be if isinstance(entry_snap_be, dict) else None,
                                                                now_snap_be if isinstance(now_snap_be, dict) else None,
                                                            )
                                                        except Exception:
                                                            pass

                                                        if hold_decision:
                                                            # 차트가 아직 유리하면 본절 터치여도 청산하지 않고 홀딩
                                                            try:
                                                                hold_cd = float(cfg.get("be_recheck_hold_cooldown_sec", 20.0) or 20.0)
                                                            except Exception:
                                                                hold_cd = 20.0
                                                            tgt["be_recheck_touch_count"] = 0
                                                            tgt["be_recheck_hold_until_epoch"] = now_ep + max(1.0, hold_cd)
                                                            tgt["be_recheck_last_decision"] = "HOLD"
                                                            hit_sl_by_price = False
                                                            try:
                                                                mon_add_event(
                                                                    mon,
                                                                    "BE_HOLD",
                                                                    sym,
                                                                    f"본절 재판단 홀딩 | {str(hold_note)[:120]}",
                                                                    {"trade_id": trade_id, "price": cur_px, "be_price": be_price},
                                                                )
                                                            except Exception:
                                                                pass
                                                        else:
                                                            try:
                                                                win_sec = float(cfg.get("be_recheck_window_sec", 180.0) or 180.0)
                                                            except Exception:
                                                                win_sec = 180.0
                                                            try:
                                                                need_n = max(1, int(cfg.get("be_recheck_confirm_n", 2) or 2))
                                                            except Exception:
                                                                need_n = 2
                                                            try:
                                                                last_ep = float(tgt.get("be_recheck_last_touch_epoch", 0) or 0.0)
                                                            except Exception:
                                                                last_ep = 0.0
                                                            try:
                                                                cnt = int(tgt.get("be_recheck_touch_count", 0) or 0)
                                                            except Exception:
                                                                cnt = 0
                                                            if (now_ep - last_ep) > max(5.0, win_sec):
                                                                cnt = 0
                                                            cnt += 1
                                                            tgt["be_recheck_touch_count"] = int(cnt)
                                                            tgt["be_recheck_last_touch_epoch"] = float(now_ep)

                                                            if int(cnt) >= int(need_n):
                                                                tgt["sl_price"] = float(be_price)
                                                                tgt["sl_price_source"] = "BE"
                                                                tgt["be_recheck_last_decision"] = "CLOSE"
                                                                tgt["be_recheck_close_note"] = str(hold_note or "")
                                                                hit_sl_by_price = True
                                                            else:
                                                                # 첫 터치는 바로 청산하지 않고 짧게 다시 본다.
                                                                try:
                                                                    retry_sec = float(cfg.get("be_recheck_retry_sec", 3.0) or 3.0)
                                                                except Exception:
                                                                    retry_sec = 3.0
                                                                tgt["be_recheck_hold_until_epoch"] = now_ep + max(1.0, retry_sec)
                                                                tgt["be_recheck_last_decision"] = "PENDING"
                                                                hit_sl_by_price = False
                                                                try:
                                                                    mon_add_event(
                                                                        mon,
                                                                        "BE_RECHECK",
                                                                        sym,
                                                                        f"본절 재확인 {int(cnt)}/{int(need_n)} | {str(hold_note)[:100]}",
                                                                        {"trade_id": trade_id, "price": cur_px, "be_price": be_price},
                                                                    )
                                                                except Exception:
                                                                    pass
                                except Exception:
                                    pass

                                # 2) +part_roi% 넘기면 50% 부분익절(한 번만)
                                try:
                                    if (not bool(tgt.get("forced_partial_done", False))) and float(roi) >= float(part_roi) and float(contracts) > 0:
                                        frac = float(part_pct) / 100.0
                                        frac = float(clamp(frac, 0.05, 0.95))
                                        close_qty = to_precision_qty(ex, sym, float(contracts) * frac)
                                        if close_qty > 0 and float(close_qty) < float(contracts):
                                            ok, err_close = close_position_market_ex(ex, sym, side, close_qty)
                                            if ok:
                                                tgt["forced_partial_done"] = True
                                                tgt["forced_partial_time_kst"] = now_kst_str()
                                                tgt["forced_partial_roi"] = float(roi)
                                                tgt["forced_partial_qty"] = float(close_qty)
                                                try:
                                                    margin_est = (float(close_qty) * float(cur_px)) / max(float(lev_live), 1.0)
                                                except Exception:
                                                    margin_est = 0.0
                                                mon_add_event(mon, "FORCED_PARTIAL_TP", sym, f"+{part_roi:.0f}% 50% 익절", {"roi": roi, "qty": close_qty, "margin_usdt_est": margin_est})
                                                try:
                                                    gsheet_log_trade(
                                                        stage="FORCED_PARTIAL_TP",
                                                        symbol=sym,
                                                        trade_id=trade_id,
                                                        message=f"roi>={part_roi} close_pct={part_pct}",
                                                        payload={"roi": roi, "qty": close_qty, "margin_usdt_est": margin_est, "part_roi": part_roi, "part_pct": part_pct},
                                                    )
                                                except Exception:
                                                    pass
                                                if _tg_simple_enabled(cfg):
                                                    msg = (
                                                        "🧩 부분익절(강제)\n"
                                                        f"- 코인: {sym}\n"
                                                        f"- 방식: {_tg_style_easy(style_now)}\n"
                                                        f"- 포지션: {_tg_dir_easy(side)}\n"
                                                        "\n"
                                                        f"- 지금 수익률: {_tg_fmt_pct(roi)}\n"
                                                        f"- 청산비율: {float(part_pct):.0f}%\n"
                                                        f"- 청산금액(마진): {float(margin_est):.2f} USDT\n"
                                                        "\n"
                                                        f"- ID: {trade_id or '-'}"
                                                    )
                                                else:
                                                    msg = f"🧩 부분익절(강제)\n- 코인: {sym}\n- ROI: +{roi:.2f}%\n- 청산비율: {float(part_pct):.0f}%\n- 청산수량: {close_qty}\n- 청산마진(추정): {margin_est:.2f} USDT\n- 일지ID: {trade_id or '-'}"
                                                tg_send(
                                                    msg,
                                                    target=cfg.get("tg_route_events_to", "channel"),
                                                    cfg=cfg,
                                                    silent=True,
                                                )
                                                if trade_id:
                                                    d = load_trade_detail(trade_id) or {}
                                                    evs = d.get("events", []) or []
                                                    evs.append({"time": now_kst_str(), "type": "FORCED_PARTIAL_TP", "roi": roi, "qty": close_qty, "margin_usdt_est": margin_est, "part_roi": part_roi, "part_pct": part_pct})
                                                    d["events"] = evs
                                                    save_trade_detail(trade_id, d)
                                                rt.setdefault("open_targets", {})[sym] = tgt
                                                save_runtime(rt)
                                            else:
                                                mon_add_event(mon, "ORDER_FAIL", sym, "강제 부분익절 실패", {"err": err_close, "qty": close_qty, "roi": roi, "trade_id": trade_id})
                                                try:
                                                    notify_admin_error(
                                                        where="ORDER:FORCED_PARTIAL_TP_CLOSE",
                                                        err=RuntimeError(str(err_close)),
                                                        context={"symbol": sym, "qty": close_qty, "roi": roi, "trade_id": trade_id},
                                                        tb="",
                                                        min_interval_sec=60.0,
                                                    )
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                                # 3) +trail_start% 넘긴 이후: 최고점 대비 -trail_dd%면 전량 청산
                                try:
                                    if float(roi) >= float(trail_start):
                                        tgt["forced_trail_active"] = True
                                    if bool(tgt.get("forced_trail_active", False)):
                                        pk_raw = tgt.get("forced_peak_roi", None)
                                        try:
                                            pk = float(pk_raw) if (pk_raw is not None and str(pk_raw).strip() != "") else float(roi)
                                        except Exception:
                                            pk = float(roi)
                                        # 첫 활성화 시점에 peak를 반드시 저장(이 값이 없으면 드로다운 감지가 안 됨)
                                        if pk_raw is None or str(pk_raw).strip() == "":
                                            tgt["forced_peak_roi"] = float(pk)
                                            tgt["forced_peak_time_kst"] = now_kst_str()
                                        # peak 갱신
                                        if float(roi) > float(pk):
                                            pk = float(roi)
                                            tgt["forced_peak_roi"] = float(pk)
                                            tgt["forced_peak_time_kst"] = now_kst_str()
                                        # 드로다운 체크
                                        if float(roi) <= (float(pk) - float(trail_dd)):
                                            forced_trail_hit = True
                                            forced_take_reason = "익절(추적손절)"
                                            forced_take_detail = f"최고 {pk:.1f}% → 현재 {float(roi):.1f}% (-{(pk - float(roi)):.1f}%)"
                                            tgt["force_take_reason"] = forced_take_reason
                                            tgt["force_take_detail"] = forced_take_detail
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # 강제 정책의 전량 청산(추적손절) → do_take로 처리(익절 로그/메시지 흐름 재사용)
                        hard_take = bool(forced_exit and forced_trail_hit)
                        if (not forced_exit) and (not ai_exit_only) and (not hard_take):
                            try:
                                if str(style_now) == "스캘핑" and bool(cfg.get("scalp_hard_take_enable", True)):
                                    ht = float(cfg.get("scalp_hard_take_roi_pct", 35.0))
                                    if float(roi) >= float(ht):
                                        hard_take = True
                            except Exception:
                                hard_take = False

                        # 사용자 요청: "시간초과 강제청산" 비활성화
                        # - 목표 TP/SL(및 SR 가격 트리거)이 올 때까지 홀딩한다.
                        # - 기존 설정 키(time_exit_*)는 호환을 위해 유지하지만, 강제청산은 실행하지 않는다.

                        # ✅ ROI 손절은 "확인 n회"로 한 번 더 생각(휩쏘 방지)
                        roi_stop_hit = bool(ai_targets_ready and (float(roi) <= -abs(float(sl))))
                        roi_stop_confirmed = roi_stop_hit
                        if (not forced_exit) and (not ai_exit_only) and roi_stop_hit and (not bool(hit_sl_by_price)) and bool(cfg.get("sl_confirm_enable", True)):
                            try:
                                n_need = max(1, int(cfg.get("sl_confirm_n", 2) or 2))
                            except Exception:
                                n_need = 2
                            try:
                                win_sec = float(cfg.get("sl_confirm_window_sec", 600.0) or 600.0)
                            except Exception:
                                win_sec = 6.0
                            now_ep = time.time()
                            try:
                                last_ep = float(tgt.get("sl_confirm_last_epoch", 0) or 0)
                            except Exception:
                                last_ep = 0.0
                            try:
                                last_cnt = int(tgt.get("sl_confirm_count", 0) or 0)
                            except Exception:
                                last_cnt = 0
                            cnt = 1 if (now_ep - last_ep) > float(win_sec) else (last_cnt + 1)
                            tgt["sl_confirm_last_epoch"] = float(now_ep)
                            tgt["sl_confirm_count"] = int(cnt)
                            roi_stop_confirmed = bool(int(cnt) >= int(n_need))
                        else:
                            # 조건이 풀리면 카운트 리셋
                            if not roi_stop_hit:
                                try:
                                    tgt["sl_confirm_count"] = 0
                                    tgt["sl_confirm_last_epoch"] = 0.0
                                except Exception:
                                    pass

                        if ai_exit_only:
                            do_stop = bool(roi_stop_hit)
                            do_take = bool(ai_targets_ready and (float(roi) >= float(tp)))
                            sl_from_ai = True
                            tp_from_ai = True
                        else:
                            do_stop = bool(hit_sl_by_price) or bool(roi_stop_confirmed)
                            do_take = hit_tp_by_price or hard_take or (roi >= tp)

                        # 손절
                        if do_stop:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            sl_src = str(tgt.get("sl_price_source", "") or "").strip().upper()
                            is_protect = bool(hit_sl_by_price) and (sl_src == "BE")
                            is_loss = (not is_protect) and ((float(pnl_usdt_snapshot) < 0.0) or (float(roi) < 0.0))
                            reason_ko = ""
                            if is_protect:
                                reason_ko = "수익보호(본전)"
                            elif bool(hit_sl_by_price):
                                reason_ko = "손절(지지/저항 이탈)"
                            else:
                                reason_ko = "손절(AI목표)" if bool(sl_from_ai) else "손절(목표 손절)"
                            # ✅ 차트 근거(룰 기반) 스냅샷: "왜 정리했는지"를 명확히 남긴다(AI 호출 없음)
                            entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                            snap_now = {}
                            try:
                                snap_now = chart_snapshot_for_reason(ex, sym, cfg)
                            except Exception:
                                snap_now = {}
                            sl_price_now = tgt.get("sl_price", None)
                            sl_line_txt = ""
                            try:
                                if sl_price_now is not None:
                                    sl_line_txt = f" (라인 {float(sl_price_now):.6g})"
                            except Exception:
                                sl_line_txt = ""
                            if is_protect:
                                arm_roi = None
                                arm_at = None
                                be_close_note = ""
                                try:
                                    v = tgt.get("be_arm_roi", None)
                                    arm_roi = float(v) if v is not None else None
                                except Exception:
                                    arm_roi = None
                                try:
                                    v = tgt.get("be_arm_at_roi", None)
                                    arm_at = float(v) if v is not None else None
                                except Exception:
                                    arm_at = None
                                try:
                                    be_close_note = str(tgt.get("be_recheck_close_note", "") or "").strip()
                                except Exception:
                                    be_close_note = ""
                                if arm_roi is not None and arm_at is not None:
                                    base_reason = f"수익이 {arm_roi:+.1f}%까지 나서(기준 {arm_at:.1f}%) 본절 라인을 올렸고, 그 라인에 닿아 정리했어요{sl_line_txt}"
                                elif arm_roi is not None:
                                    base_reason = f"수익이 {arm_roi:+.1f}%까지 나서 본절 라인을 올렸고, 그 라인에 닿아 정리했어요{sl_line_txt}"
                                else:
                                    base_reason = f"수익이 나서 본절 라인을 올렸고, 그 라인에 닿아 정리했어요{sl_line_txt}"
                                if be_close_note:
                                    base_reason = f"{base_reason} | 재판단: {be_close_note}"
                            elif bool(hit_sl_by_price):
                                base_reason = f"지지/저항 이탈로 손절했어요{sl_line_txt}"
                            else:
                                base_reason = f"손실이 목표손절(-{abs(float(sl)):.1f}%)에 닿아서 정리했어요"
                            one_rule = build_exit_one_line(base_reason=base_reason, entry_snap=entry_snap, now_snap=snap_now)
                            ok, err_close = close_position_market_ex(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                # ✅ AI 회고 비용 절감: 손실일 때만 AI 회고 작성(사용자 요구)
                                review = ""
                                if is_loss:
                                    try:
                                        _ai_one, _ai_review = ai_write_review(sym, side, roi, reason_ko, cfg)
                                        review = str(_ai_review or "")
                                    except Exception:
                                        review = ""
                                # ✅ 텔레그램/일지에는 "차트 기반 한줄 근거"를 우선 기록
                                one = str(one_rule or "").strip() or ("본전으로 지킴" if is_protect else "정리 완료")
                                # ✅ 매매일지/구글시트에 "진입 전/청산 후 잔액"을 같이 기록(요구사항)
                                bb_total = None
                                bb_free = None
                                try:
                                    v0 = tgt.get("bal_entry_total", "")
                                    bb_total = float(v0) if (v0 is not None and str(v0).strip() != "") else None
                                except Exception:
                                    bb_total = None
                                try:
                                    v1 = tgt.get("bal_entry_free", "")
                                    bb_free = float(v1) if (v1 is not None and str(v1).strip() != "") else None
                                except Exception:
                                    bb_free = None
                                log_trade(
                                    sym,
                                    side,
                                    entry,
                                    exit_px,
                                    pnl_usdt_snapshot,
                                    roi,
                                    reason_ko,
                                    one_line=one,
                                    review=review,
                                    trade_id=trade_id,
                                    balance_before_total=bb_total,
                                    balance_after_total=total_after,
                                    balance_before_free=bb_free,
                                    balance_after_free=free_after,
                                )
                                try:
                                    gsheet_log_trade(
                                        stage="EXIT_PROTECT" if is_protect else "EXIT_SL",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message="protect_be" if is_protect else "auto_sl",
                                        payload={
                                            "roi": roi,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "entry": entry,
                                            "exit": exit_px,
                                            "hit_sr": bool(hit_sl_by_price),
                                            "sl_price_source": sl_src,
                                            "style": style_now,
                                        },
                                    )
                                except Exception:
                                    pass

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update(
                                        {
                                            "exit_time": now_kst_str(),
                                            "exit_price": exit_px,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "pnl_pct": roi,
                                            "result": "PROTECT" if is_protect else "SL",
                                            "exit_reason_detail": one,
                                            "exit_snapshot": snap_now,
                                            "review": review,
                                            "balance_after_total": total_after,
                                            "balance_after_free": free_after,
                                        }
                                    )
                                    try:
                                        if bool(is_protect):
                                            d["be_arm"] = {
                                                "time_kst": str(tgt.get("be_arm_time_kst", "") or ""),
                                                "epoch": float(tgt.get("be_arm_epoch", 0) or 0.0),
                                                "roi": _as_float(tgt.get("be_arm_roi", None), 0.0),
                                                "at_roi": _as_float(tgt.get("be_arm_at_roi", None), 0.0),
                                                "offset_price_pct": _as_float(tgt.get("be_arm_offset_pct", None), 0.0),
                                                "price": _as_float(tgt.get("be_arm_price", None), 0.0),
                                                "ind": str(tgt.get("be_arm_ind", "") or "")[:220],
                                            }
                                    except Exception:
                                        pass
                                    save_trade_detail(trade_id, d)

                                # ✅ 일일 손익/연속손실 업데이트 + 방어 로직
                                try:
                                    # day_start_equity가 비어있으면(0) 진입 시점 총자산으로 초기화(가능한 경우)
                                    if float(rt.get("day_start_equity", 0) or 0) <= 0 and bb_total is not None and float(bb_total) > 0:
                                        rt["day_start_equity"] = float(bb_total)
                                    rt["daily_realized_pnl"] = float(rt.get("daily_realized_pnl", 0) or 0.0) + float(pnl_usdt_snapshot)
                                except Exception:
                                    pass
                                # ✅ 당일 통계(거래수/승패) 업데이트 (fail-safe profit guard용)
                                try:
                                    rt["daily_trade_count"] = int(rt.get("daily_trade_count", 0) or 0) + 1
                                    if float(pnl_usdt_snapshot) > 0:
                                        rt["daily_win_count"] = int(rt.get("daily_win_count", 0) or 0) + 1
                                    elif float(pnl_usdt_snapshot) < 0:
                                        rt["daily_loss_count"] = int(rt.get("daily_loss_count", 0) or 0) + 1
                                except Exception:
                                    pass

                                if is_loss:
                                    rt["consec_losses"] = int(rt.get("consec_losses", 0) or 0) + 1
                                else:
                                    rt["consec_losses"] = 0

                                # 1) 기존 기능: 연속손실 pause(잠깐 쉼)
                                if is_loss and cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(
                                        f"🛑 연속 손실로 잠깐 멈춤\n- 연속손실: {rt['consec_losses']}번\n- {int(cfg.get('loss_pause_minutes',30))}분 쉬기",
                                        target=cfg.get("tg_route_events_to", "channel"),
                                        cfg=cfg,
                                    )
                                    mon_add_event(mon, "PAUSE", "", "loss_pause", {"consec": rt["consec_losses"]})

                                # 2) 서킷브레이커(사용자 요청): 연속 손실이어도 자동매매 OFF 하지 않음 → 경고/기록만
                                try:
                                    if is_loss and bool(cfg.get("circuit_breaker_enable", False)) and int(rt["consec_losses"]) >= int(cfg.get("circuit_breaker_after", 12)):
                                        now_ep = time.time()
                                        last_warn = float(rt.get("circuit_breaker_warn_epoch", 0) or 0.0)
                                        # 과도한 알림/스팸 방지(5분에 1회)
                                        if (now_ep - last_warn) >= 300.0:
                                            rt["circuit_breaker_warn_epoch"] = now_ep
                                            tg_send(
                                                f"⚠️ 서킷브레이커(경고)\n- 연속손실: {rt['consec_losses']}번\n- 자동매매는 계속 ON(요청 설정)",
                                                target=cfg.get("tg_route_events_to", "channel"),
                                                cfg=cfg,
                                                silent=True,
                                            )
                                            mon_add_event(mon, "CIRCUIT_WARN", "", "circuit_breaker", {"consec_losses": rt["consec_losses"]})
                                except Exception:
                                    pass

                                # 3) 추가 기능: 일일 손실 한도(도달 시 자동매매 OFF)
                                try:
                                    if bool(cfg.get("daily_loss_limit_enable", False)):
                                        lim_pct = float(cfg.get("daily_loss_limit_pct", 0.0) or 0.0)
                                        lim_usdt = float(cfg.get("daily_loss_limit_usdt", 0.0) or 0.0)
                                        day_pnl = float(rt.get("daily_realized_pnl", 0.0) or 0.0)
                                        dse0 = float(rt.get("day_start_equity", 0.0) or 0.0)
                                        day_pct = (day_pnl / dse0 * 100.0) if dse0 > 0 else 0.0
                                        hit_usdt = (lim_usdt > 0) and (day_pnl <= -abs(lim_usdt))
                                        hit_pct = (lim_pct > 0) and (day_pct <= -abs(lim_pct))
                                        if hit_usdt or hit_pct:
                                            if bool(cfg.get("auto_trade", False)):
                                                cfg["auto_trade"] = False
                                                save_settings(cfg)
                                            rt["pause_until"] = max(float(rt.get("pause_until", 0) or 0.0), float(next_midnight_kst_epoch()))
                                            rt["auto_trade_stop_reason"] = "DAILY_LOSS_LIMIT"
                                            rt["auto_trade_stop_kst"] = now_kst_str()
                                            tg_send(
                                                f"⛔️ 일일 손실 한도 도달\n- 오늘 손익: {day_pnl:+.2f} USDT ({day_pct:+.2f}%)\n- 자동매매를 껐어요(내일 0시까지).",
                                                target=cfg.get("tg_route_events_to", "channel"),
                                                cfg=cfg,
                                            )
                                            mon_add_event(mon, "AUTO_TRADE_OFF", "", "daily_loss_limit", {"day_pnl": day_pnl, "day_pct": day_pct})
                                except Exception:
                                    pass

                                # ✅ Fail-safe(요구): 수익 못 내거나 큰 손실이면 자동매매 OFF
                                try:
                                    w = "EXIT_PROTECT" if bool(is_protect) else "EXIT_SL"
                                    if maybe_trigger_fail_safe(cfg, rt, float(total_after), mon=mon, where=w):
                                        entry_allowed_global = False
                                except Exception:
                                    pass

                                save_runtime(rt)

                                emo = "🟢" if roi >= 0 else "🔴"
                                try:
                                    bb_total_s = f"{float(bb_total):.2f}" if bb_total is not None else "-"
                                except Exception:
                                    bb_total_s = "-"
                                try:
                                    bb_free_s = f"{float(bb_free):.2f}" if bb_free is not None else "-"
                                except Exception:
                                    bb_free_s = "-"
                                if _tg_simple_enabled(cfg):
                                    msg = tg_msg_exit_simple(
                                        title="🛡️ 수익보호" if is_protect else "🩸 손절",
                                        symbol=str(sym),
                                        style=str(style_now),
                                        side=str(side),
                                        lev=tgt.get("lev", "?"),
                                        roi_pct=float(roi),
                                        pnl_usdt=float(pnl_usdt_snapshot),
                                        contracts=float(contracts),
                                        bal_before_total=bb_total,
                                        bal_after_total=float(total_after),
                                        bal_before_free=bb_free,
                                        bal_after_free=float(free_after),
                                        one_line=str(one),
                                        trade_id=str(trade_id or "-"),
                                    )
                                else:
                                    msg = (
                                        f"{emo} {('수익보호' if is_protect else '손절')}\n- 코인: {sym}\n- 스타일: {style_now}\n- 수익률: {roi:+.2f}% (손익 {pnl_usdt_snapshot:+.2f} USDT)\n"
                                        f"- 진입가→청산가: {float(entry):.6g} → {float(exit_px):.6g}\n"
                                        f"- 청산수량(contracts): {contracts}\n"
                                        f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                        f"- 레버: x{tgt.get('lev','?')}\n"
                                        f"- 잔고(총/가용): {bb_total_s}→{total_after:.2f} / {bb_free_s}→{free_after:.2f} USDT\n"
                                        f"- 이유: {reason_ko}\n"
                                        f"- 한줄평: {one}\n- 일지ID: {trade_id or '없음'}"
                                    )
                                tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                try:
                                    if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                        tg_send(msg, target="admin", cfg=cfg, silent=False)
                                except Exception:
                                    pass
                                try:
                                    if bool(cfg.get("tg_send_trade_images", True)) and bool(cfg.get("tg_send_exit_image", True)):
                                        img_path = build_trade_event_image(
                                            ex,
                                            sym,
                                            cfg,
                                            event_type=("PROTECT" if is_protect else "SL"),
                                            side=str(side),
                                            style=str(style_now),
                                            entry_price=float(entry),
                                            exit_price=float(exit_px),
                                            sl_price=(float(tgt.get("sl_price")) if tgt.get("sl_price") is not None else None),
                                            tp_price=(float(tgt.get("tp_price")) if tgt.get("tp_price") is not None else None),
                                            partial_tp1_price=(float(tgt.get("partial_tp1_price")) if tgt.get("partial_tp1_price") is not None else None),
                                            partial_tp2_price=(float(tgt.get("partial_tp2_price")) if tgt.get("partial_tp2_price") is not None else None),
                                            dca_price=(float(tgt.get("dca_price")) if tgt.get("dca_price") is not None else None),
                                            sl_roi_pct=(float(tgt.get("sl")) if tgt.get("sl") is not None else None),
                                            tp_roi_pct=(float(tgt.get("tp")) if tgt.get("tp") is not None else None),
                                            leverage=(float(tgt.get("lev")) if tgt.get("lev") is not None else None),
                                            roi_pct=float(roi),
                                            pnl_usdt=float(pnl_usdt_snapshot),
                                            remain_free=float(free_after),
                                            remain_total=float(total_after),
                                            one_line=str(one),
                                            used_indicators=[],
                                            pattern_hint=str(_fmt_indicator_line_for_reason(entry_snap, snap_now)),
                                            mtf_pattern={},
                                            trade_id=str(trade_id or ""),
                                        )
                                        if img_path:
                                            cap = (
                                                f"📷 청산 차트\n"
                                                f"- {sym} | {style_now} | {_tg_dir_easy(side)}\n"
                                                f"- 결과: {_tg_fmt_pct(float(roi))} ({_tg_fmt_usdt(float(pnl_usdt_snapshot))} USDT)"
                                            )
                                            tg_send_photo(img_path, caption=cap, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                            if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                                tg_send_photo(img_path, caption=cap, target="admin", cfg=cfg, silent=False)
                                            if trade_id:
                                                d0 = load_trade_detail(str(trade_id)) or {}
                                                d0["exit_chart_image"] = str(img_path)
                                                save_trade_detail(str(trade_id), d0)
                                except Exception:
                                    pass

                                # ✅ 청산 후 재진입 쿨다운 + 직전 청산 기록(과매매/수수료/AI호출 낭비 방지)
                                try:
                                    tf_sec = int(_timeframe_seconds(str(cfg.get("timeframe", "5m") or "5m"), 300))
                                    if bool(is_protect):
                                        bars = int(cfg.get("cooldown_after_exit_protect_bars", 2) or 0)
                                    else:
                                        bars = int(cfg.get("cooldown_after_exit_sl_bars", 3) or 0)
                                    bars = max(0, bars)
                                    if tf_sec > 0 and bars > 0:
                                        rt.setdefault("cooldowns", {})[sym] = time.time() + float(tf_sec) * float(bars)
                                except Exception:
                                    pass
                                try:
                                    rt.setdefault("last_exit", {})[sym] = {
                                        "time_kst": now_kst_str(),
                                        "epoch": float(time.time()),
                                        "type": "PROTECT" if bool(is_protect) else "SL",
                                        "symbol": str(sym),
                                        "side": str(side),
                                        "style": str(style_now),
                                        "roi": float(roi),
                                        "pnl_usdt": float(pnl_usdt_snapshot),
                                        "trade_id": str(trade_id or ""),
                                    }
                                except Exception:
                                    pass

                                # ✅ 청산 후 재진입 쿨다운 + 직전 청산 기록(과매매/수수료/AI호출 낭비 방지)
                                try:
                                    tf_sec = int(_timeframe_seconds(str(cfg.get("timeframe", "5m") or "5m"), 300))
                                    bars = int(cfg.get("cooldown_after_exit_tp_bars", 1) or 0)
                                    bars = max(0, bars)
                                    if tf_sec > 0 and bars > 0:
                                        rt.setdefault("cooldowns", {})[sym] = time.time() + float(tf_sec) * float(bars)
                                except Exception:
                                    pass
                                try:
                                    rt.setdefault("last_exit", {})[sym] = {
                                        "time_kst": now_kst_str(),
                                        "epoch": float(time.time()),
                                        "type": "TP",
                                        "symbol": str(sym),
                                        "side": str(side),
                                        "style": str(style_now),
                                        "roi": float(roi),
                                        "pnl_usdt": float(pnl_usdt_snapshot),
                                        "trade_id": str(trade_id or ""),
                                        "hard_take": bool(hard_take),
                                    }
                                except Exception:
                                    pass

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                rt.setdefault("open_targets", {}).pop(sym, None)
                                save_runtime(rt)

                                mon_add_event(mon, "PROTECT" if is_protect else "STOP", sym, f"{reason_ko} ROI {roi:.2f}%", {"trade_id": trade_id, "reason": reason_ko})
                                monitor_write_throttled(mon, 0.2)
                            else:
                                mon_add_event(mon, "ORDER_FAIL", sym, "청산 실패(손절)", {"err": err_close, "roi": roi, "sl": sl, "trade_id": trade_id})
                                try:
                                    notify_admin_error(
                                        where="ORDER:EXIT_SL",
                                        err=RuntimeError(str(err_close)),
                                        context={"symbol": sym, "roi": roi, "sl": sl, "trade_id": trade_id, "mode": mode, "style": style_now},
                                        tb="",
                                        min_interval_sec=60.0,
                                    )
                                except Exception:
                                    pass

                        # 익절
                        elif do_take:
                            pnl_usdt_snapshot = float(p.get("unrealizedPnl") or 0.0)
                            force_take_reason = str(tgt.get("force_take_reason", "") or "").strip()
                            force_take_detail = str(tgt.get("force_take_detail", "") or "").strip()
                            if force_take_reason:
                                take_reason_ko = force_take_reason
                            else:
                                if bool(hit_tp_by_price):
                                    take_reason_ko = "익절(저항/목표 도달)"
                                elif bool(hard_take):
                                    take_reason_ko = "익절(강제)"
                                else:
                                    take_reason_ko = "익절(AI목표)" if bool(tp_from_ai) else "익절(목표 익절)"
                            is_loss_take = (float(pnl_usdt_snapshot) < 0.0) or (float(roi) < 0.0)
                            # ✅ 차트 근거(룰 기반) 스냅샷: "왜 익절했는지"를 명확히 남긴다(AI 호출 없음)
                            entry_snap = tgt.get("entry_snapshot") if isinstance(tgt.get("entry_snapshot"), dict) else None
                            snap_now = {}
                            try:
                                snap_now = chart_snapshot_for_reason(ex, sym, cfg)
                            except Exception:
                                snap_now = {}
                            tp_price_now = tgt.get("tp_price", None)
                            tp_line_txt = ""
                            try:
                                if tp_price_now is not None:
                                    tp_line_txt = f" (라인 {float(tp_price_now):.6g})"
                            except Exception:
                                tp_line_txt = ""
                            if force_take_reason:
                                base_reason = f"{force_take_reason}" + (f" | {force_take_detail}" if force_take_detail else "")
                            elif bool(hard_take):
                                base_reason = "수익이 많이 나서 일단 챙겼어요(강제익절)"
                            elif bool(hit_tp_by_price):
                                base_reason = f"저항/목표가에 닿아서 익절했어요{tp_line_txt}"
                            else:
                                if bool(tp_from_ai):
                                    base_reason = f"AI목표익절(+{abs(float(tp)):.1f}%)에 닿아서 익절했어요"
                                else:
                                    base_reason = f"목표익절(+{abs(float(tp)):.1f}%)에 닿아서 익절했어요"
                            one_rule = build_exit_one_line(base_reason=base_reason, entry_snap=entry_snap, now_snap=snap_now)
                            ok, err_close = close_position_market_ex(ex, sym, side, contracts)
                            if ok:
                                exit_px = get_last_price(ex, sym) or entry
                                free_after, total_after = safe_fetch_balance(ex)

                                # ✅ AI 회고 비용 절감: 손실일 때만 AI 회고 작성
                                review = ""
                                if is_loss_take:
                                    try:
                                        _ai_one, _ai_review = ai_write_review(sym, side, roi, take_reason_ko, cfg)
                                        review = str(_ai_review or "")
                                    except Exception:
                                        review = ""
                                # ✅ 텔레그램/일지에는 "차트 기반 한줄 근거"를 우선 기록
                                one = str(one_rule or "").strip() or "익절 성공"
                                # ✅ 매매일지/구글시트에 "진입 전/청산 후 잔액"을 같이 기록(요구사항)
                                bb_total = None
                                bb_free = None
                                try:
                                    v0 = tgt.get("bal_entry_total", "")
                                    bb_total = float(v0) if (v0 is not None and str(v0).strip() != "") else None
                                except Exception:
                                    bb_total = None
                                try:
                                    v1 = tgt.get("bal_entry_free", "")
                                    bb_free = float(v1) if (v1 is not None and str(v1).strip() != "") else None
                                except Exception:
                                    bb_free = None
                                log_trade(
                                    sym,
                                    side,
                                    entry,
                                    exit_px,
                                    pnl_usdt_snapshot,
                                    roi,
                                    take_reason_ko,
                                    one_line=one,
                                    review=review,
                                    trade_id=trade_id,
                                    balance_before_total=bb_total,
                                    balance_after_total=total_after,
                                    balance_before_free=bb_free,
                                    balance_after_free=free_after,
                                )
                                try:
                                    gsheet_log_trade(
                                        stage="EXIT_TP",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message="hard_take" if bool(hard_take) else "auto_tp",
                                        payload={
                                            "roi": roi,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "entry": entry,
                                            "exit": exit_px,
                                            "hit_sr": bool(hit_tp_by_price),
                                            "tp_price_source": str(tgt.get("tp_price_source", "") or ""),
                                            "style": style_now,
                                        },
                                    )
                                except Exception:
                                    pass

                                if trade_id:
                                    d = load_trade_detail(trade_id) or {}
                                    d.update(
                                        {
                                            "exit_time": now_kst_str(),
                                            "exit_price": exit_px,
                                            "pnl_usdt": pnl_usdt_snapshot,
                                            "pnl_pct": roi,
                                            "result": "TP",
                                            "exit_reason_detail": one,
                                            "exit_snapshot": snap_now,
                                            "review": review,
                                            "balance_after_total": total_after,
                                            "balance_after_free": free_after,
                                        }
                                    )
                                    save_trade_detail(trade_id, d)

                                # ✅ 일일 손익/연속손실 업데이트 + 방어 로직
                                try:
                                    if float(rt.get("day_start_equity", 0) or 0) <= 0 and bb_total is not None and float(bb_total) > 0:
                                        rt["day_start_equity"] = float(bb_total)
                                    rt["daily_realized_pnl"] = float(rt.get("daily_realized_pnl", 0) or 0.0) + float(pnl_usdt_snapshot)
                                except Exception:
                                    pass
                                # ✅ 당일 통계(거래수/승패) 업데이트 (fail-safe profit guard용)
                                try:
                                    rt["daily_trade_count"] = int(rt.get("daily_trade_count", 0) or 0) + 1
                                    if float(pnl_usdt_snapshot) > 0:
                                        rt["daily_win_count"] = int(rt.get("daily_win_count", 0) or 0) + 1
                                    elif float(pnl_usdt_snapshot) < 0:
                                        rt["daily_loss_count"] = int(rt.get("daily_loss_count", 0) or 0) + 1
                                except Exception:
                                    pass

                                if is_loss_take:
                                    rt["consec_losses"] = int(rt.get("consec_losses", 0) or 0) + 1
                                else:
                                    rt["consec_losses"] = 0

                                # 1) 기존 기능: 연속손실 pause(잠깐 쉼)
                                if is_loss_take and cfg.get("loss_pause_enable", True) and rt["consec_losses"] >= int(cfg.get("loss_pause_after", 3)):
                                    rt["pause_until"] = time.time() + int(cfg.get("loss_pause_minutes", 30)) * 60
                                    tg_send(
                                        f"🛑 연속 손실로 잠깐 멈춤\n- 연속손실: {rt['consec_losses']}번\n- {int(cfg.get('loss_pause_minutes',30))}분 쉬기",
                                        target=cfg.get("tg_route_events_to", "channel"),
                                        cfg=cfg,
                                    )
                                    mon_add_event(mon, "PAUSE", "", "loss_pause", {"consec": rt["consec_losses"]})

                                # 2) 서킷브레이커(사용자 요청): 연속 손실이어도 자동매매 OFF 하지 않음 → 경고/기록만
                                try:
                                    if is_loss_take and bool(cfg.get("circuit_breaker_enable", False)) and int(rt["consec_losses"]) >= int(cfg.get("circuit_breaker_after", 12)):
                                        now_ep = time.time()
                                        last_warn = float(rt.get("circuit_breaker_warn_epoch", 0) or 0.0)
                                        # 과도한 알림/스팸 방지(5분에 1회)
                                        if (now_ep - last_warn) >= 300.0:
                                            rt["circuit_breaker_warn_epoch"] = now_ep
                                            tg_send(
                                                f"⚠️ 서킷브레이커(경고)\n- 연속손실: {rt['consec_losses']}번\n- 자동매매는 계속 ON(요청 설정)",
                                                target=cfg.get("tg_route_events_to", "channel"),
                                                cfg=cfg,
                                                silent=True,
                                            )
                                            mon_add_event(mon, "CIRCUIT_WARN", "", "circuit_breaker", {"consec_losses": rt["consec_losses"]})
                                except Exception:
                                    pass

                                # 3) 일일 손실 한도(도달 시 자동매매 OFF)
                                try:
                                    if bool(cfg.get("daily_loss_limit_enable", False)):
                                        lim_pct = float(cfg.get("daily_loss_limit_pct", 0.0) or 0.0)
                                        lim_usdt = float(cfg.get("daily_loss_limit_usdt", 0.0) or 0.0)
                                        day_pnl = float(rt.get("daily_realized_pnl", 0.0) or 0.0)
                                        dse0 = float(rt.get("day_start_equity", 0.0) or 0.0)
                                        day_pct = (day_pnl / dse0 * 100.0) if dse0 > 0 else 0.0
                                        hit_usdt = (lim_usdt > 0) and (day_pnl <= -abs(lim_usdt))
                                        hit_pct = (lim_pct > 0) and (day_pct <= -abs(lim_pct))
                                        if hit_usdt or hit_pct:
                                            if bool(cfg.get("auto_trade", False)):
                                                cfg["auto_trade"] = False
                                                save_settings(cfg)
                                            rt["pause_until"] = max(float(rt.get("pause_until", 0) or 0.0), float(next_midnight_kst_epoch()))
                                            rt["auto_trade_stop_reason"] = "DAILY_LOSS_LIMIT"
                                            rt["auto_trade_stop_kst"] = now_kst_str()
                                            tg_send(
                                                f"⛔️ 일일 손실 한도 도달\n- 오늘 손익: {day_pnl:+.2f} USDT ({day_pct:+.2f}%)\n- 자동매매를 껐어요(내일 0시까지).",
                                                target=cfg.get("tg_route_events_to", "channel"),
                                                cfg=cfg,
                                            )
                                            mon_add_event(mon, "AUTO_TRADE_OFF", "", "daily_loss_limit", {"day_pnl": day_pnl, "day_pct": day_pct})
                                except Exception:
                                    pass

                                # ✅ Fail-safe(요구): 수익 못 내거나 큰 손실이면 자동매매 OFF
                                try:
                                    if maybe_trigger_fail_safe(cfg, rt, float(total_after), mon=mon, where="EXIT_TP"):
                                        entry_allowed_global = False
                                except Exception:
                                    pass

                                save_runtime(rt)

                                try:
                                    bb_total_s = f"{float(bb_total):.2f}" if bb_total is not None else "-"
                                except Exception:
                                    bb_total_s = "-"
                                try:
                                    bb_free_s = f"{float(bb_free):.2f}" if bb_free is not None else "-"
                                except Exception:
                                    bb_free_s = "-"
                                # ✅ 손실인데 '익절'로 보이는 혼동 방지:
                                # - 시간초과 정리/강제 정리 등은 ROI/PnL이 음수면 '정리'로 표기
                                title_txt = "🎉 익절(강제)" if bool(hard_take) else "🎉 익절"
                                try:
                                    if bool(is_loss_take):
                                        r0 = str(tgt.get("force_take_reason", "") or take_reason_ko or "").strip()
                                        if "시간초과" in r0:
                                            title_txt = "⏳ 시간초과 정리(강제)" if bool(hard_take) else "⏳ 시간초과 정리"
                                        else:
                                            title_txt = "🩸 정리(강제)" if bool(hard_take) else "🩸 정리"
                                except Exception:
                                    title_txt = title_txt
                                if _tg_simple_enabled(cfg):
                                    msg = tg_msg_exit_simple(
                                        title=str(title_txt),
                                        symbol=str(sym),
                                        style=str(style_now),
                                        side=str(side),
                                        lev=tgt.get("lev", "?"),
                                        roi_pct=float(roi),
                                        pnl_usdt=float(pnl_usdt_snapshot),
                                        contracts=float(contracts),
                                        bal_before_total=bb_total,
                                        bal_after_total=float(total_after),
                                        bal_before_free=bb_free,
                                        bal_after_free=float(free_after),
                                        one_line=str(one),
                                        trade_id=str(trade_id or "-"),
                                    )
                                else:
                                    msg = (
                                        f"{title_txt}\n- 코인: {sym}\n- 스타일: {style_now}\n- 수익률: {roi:+.2f}% (손익 {pnl_usdt_snapshot:+.2f} USDT)\n"
                                        f"- 진입가→청산가: {float(entry):.6g} → {float(exit_px):.6g}\n"
                                        f"- 청산수량(contracts): {contracts}\n"
                                        f"- 진입금: {float(tgt.get('entry_usdt',0)):.2f} USDT (잔고 {float(tgt.get('entry_pct',0)):.1f}%)\n"
                                        f"- 레버: x{tgt.get('lev','?')}\n"
                                        f"- 잔고(총/가용): {bb_total_s}→{total_after:.2f} / {bb_free_s}→{free_after:.2f} USDT\n"
                                        f"- 이유: {take_reason_ko}\n"
                                        f"- 한줄평: {one}\n- 일지ID: {trade_id or '없음'}"
                                    )
                                tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                try:
                                    if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                        tg_send(msg, target="admin", cfg=cfg, silent=False)
                                except Exception:
                                    pass
                                try:
                                    if bool(cfg.get("tg_send_trade_images", True)) and bool(cfg.get("tg_send_exit_image", True)):
                                        img_path = build_trade_event_image(
                                            ex,
                                            sym,
                                            cfg,
                                            event_type=("TAKE_FORCE" if bool(hard_take) else "TP"),
                                            side=str(side),
                                            style=str(style_now),
                                            entry_price=float(entry),
                                            exit_price=float(exit_px),
                                            sl_price=(float(tgt.get("sl_price")) if tgt.get("sl_price") is not None else None),
                                            tp_price=(float(tgt.get("tp_price")) if tgt.get("tp_price") is not None else None),
                                            partial_tp1_price=(float(tgt.get("partial_tp1_price")) if tgt.get("partial_tp1_price") is not None else None),
                                            partial_tp2_price=(float(tgt.get("partial_tp2_price")) if tgt.get("partial_tp2_price") is not None else None),
                                            dca_price=(float(tgt.get("dca_price")) if tgt.get("dca_price") is not None else None),
                                            sl_roi_pct=(float(tgt.get("sl")) if tgt.get("sl") is not None else None),
                                            tp_roi_pct=(float(tgt.get("tp")) if tgt.get("tp") is not None else None),
                                            leverage=(float(tgt.get("lev")) if tgt.get("lev") is not None else None),
                                            roi_pct=float(roi),
                                            pnl_usdt=float(pnl_usdt_snapshot),
                                            remain_free=float(free_after),
                                            remain_total=float(total_after),
                                            one_line=str(one),
                                            used_indicators=[],
                                            pattern_hint=str(_fmt_indicator_line_for_reason(entry_snap, snap_now)),
                                            mtf_pattern={},
                                            trade_id=str(trade_id or ""),
                                        )
                                        if img_path:
                                            cap = (
                                                f"📷 청산 차트\n"
                                                f"- {sym} | {style_now} | {_tg_dir_easy(side)}\n"
                                                f"- 결과: {_tg_fmt_pct(float(roi))} ({_tg_fmt_usdt(float(pnl_usdt_snapshot))} USDT)"
                                            )
                                            tg_send_photo(img_path, caption=cap, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                            if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                                tg_send_photo(img_path, caption=cap, target="admin", cfg=cfg, silent=False)
                                            if trade_id:
                                                d0 = load_trade_detail(str(trade_id)) or {}
                                                d0["exit_chart_image"] = str(img_path)
                                                save_trade_detail(str(trade_id), d0)
                                except Exception:
                                    pass

                                active_targets.pop(sym, None)
                                rt.setdefault("trades", {}).pop(sym, None)
                                rt.setdefault("open_targets", {}).pop(sym, None)
                                save_runtime(rt)

                                mon_add_event(mon, "TAKE", sym, f"ROI +{roi:.2f}%", {"trade_id": trade_id})
                                monitor_write_throttled(mon, 0.2)
                            else:
                                mon_add_event(mon, "ORDER_FAIL", sym, "청산 실패(익절)", {"err": err_close, "roi": roi, "tp": tp, "trade_id": trade_id})
                                try:
                                    notify_admin_error(
                                        where="ORDER:EXIT_TP",
                                        err=RuntimeError(str(err_close)),
                                        context={"symbol": sym, "roi": roi, "tp": tp, "trade_id": trade_id, "mode": mode, "style": style_now},
                                        tb="",
                                        min_interval_sec=60.0,
                                    )
                                except Exception:
                                    pass

                        open_pos_snapshot.append(
                            {
                                "symbol": sym,
                                "side": side,
                                "roi": roi,
                                "upnl": upnl,
                                "lev": lev_live,
                                "style": style_now,
                                # ✅ 표시용 목표(tp/sl)는 "진입 당시 목표"(AI/룰)를 유지
                                "tp": _as_float(tgt.get("tp", tp), 0.0),
                                "sl": _as_float(tgt.get("sl", sl), 0.0),
                                # ✅ 실제 청산은 '수익보존' 정책이 우선일 수 있어, 혼동 방지용으로 함께 저장
                                "exit_policy": ("AI_TARGET_ONLY" if bool(ai_exit_only) else ("TRAIL_PROTECT" if bool(forced_exit) else "TARGET")),
                                "exit_rule": ("AI TP/SL only" if bool(ai_exit_only) else (_tg_trailing_protect_policy_line(cfg) if bool(forced_exit) else "")),
                                "exit_sl_roi": float(abs(float(_as_float(tgt.get("sl", sl), 0.0)))),
                                "trade_id": trade_id,
                            }
                        )

                    # ✅ 자동매매가 OFF/정지/주말이어도 포지션 스냅샷은 UI에 표시
                    if (not open_pos_snapshot) and (not entry_allowed_global) and pos_by_sym:
                        try:
                            for sym in TARGET_COINS:
                                p = pos_by_sym.get(sym)
                                if not p:
                                    continue
                                side = position_side_normalize(p)
                                roi = float(position_roi_percent(p))
                                upnl = float(p.get("unrealizedPnl") or 0.0)
                                lev_live = _pos_leverage(p)
                                tgt0 = (active_targets.get(sym, {}) or {})
                                style_now = str(tgt0.get("style", "") or "")
                                tp = float(tgt0.get("tp", 0.0) or 0.0)
                                sl = float(tgt0.get("sl", 0.0) or 0.0)
                                trade_id = str(tgt0.get("trade_id") or "")
                                open_pos_snapshot.append(
                                    {
                                        "symbol": sym,
                                        "side": side,
                                        "roi": roi,
                                        "upnl": upnl,
                                        "lev": lev_live,
                                        "style": style_now,
                                        "tp": tp,
                                        "sl": sl,
                                        "exit_policy": ("AI_TARGET_ONLY" if bool(cfg.get("exit_ai_targets_only", False)) else ("TRAIL_PROTECT" if bool(cfg.get("exit_trailing_protect_enable", False)) else "TARGET")),
                                        "exit_rule": ("AI TP/SL only" if bool(cfg.get("exit_ai_targets_only", False)) else (_tg_trailing_protect_policy_line(cfg) if bool(cfg.get("exit_trailing_protect_enable", False)) else "")),
                                        "exit_sl_roi": float(abs(sl)),
                                        "trade_id": trade_id,
                                    }
                                )
                        except Exception:
                            pass

                    mon["open_positions"] = open_pos_snapshot

                    # 2) 신규 진입 스캔
                    risk_mul = external_risk_multiplier(ext, cfg)
                    mon["entry_risk_multiplier"] = risk_mul
                    # ✅ 극공포(공포탐욕 0~15): 신규 진입 전면 금지
                    if entry_allowed_global and float(risk_mul) <= 0.0:
                        entry_allowed_global = False
                        try:
                            fg_v = int(((ext or {}).get("fear_greed") or {}).get("value", -1))
                            mon_add_event(mon, "ENTRY_BLOCK_FEAR", "", f"극공포 지수({fg_v}) → 신규 진입 금지", {"fear_greed": fg_v})
                        except Exception:
                            pass
                    free_usdt = 0.0
                    total_usdt = 0.0
                    if entry_allowed_global:
                        _to_before_bal = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                        free_usdt, total_usdt = safe_fetch_balance(ex)
                        _to_after_bal = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                        if _to_after_bal and _to_after_bal > _to_before_bal:
                            need_exchange_refresh = True
                            entry_allowed_global = False
                            free_usdt = 0.0
                            total_usdt = 0.0
                        # ✅ Fail-safe 체크(잔고 기반): 즉시 신규진입 차단
                        if entry_allowed_global and float(total_usdt or 0.0) > 0:
                            if maybe_trigger_fail_safe(cfg, rt, float(total_usdt), mon=mon, where="BALANCE"):
                                entry_allowed_global = False
                                trade_enabled = False
                                free_usdt = 0.0
                                total_usdt = 0.0

                    # ✅ 잔고 조회에서 timeout이 발생했다면, 스캔 전에 인스턴스 교체(동시 호출 꼬임 방지)
                    if need_exchange_refresh:
                        try:
                            where_now = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "").strip()
                            mon_add_event(mon, "CCXT_REFRESH", "", "exchange refreshed(after balance timeout)", {"where": where_now, "code": CODE_VERSION})
                            ex_new = create_exchange_client_uncached()
                            if ex_new is not None:
                                ex = ex_new
                                ccxt_timeout_epoch_loop_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                                ccxt_timeout_where_loop_start = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "")
                                need_exchange_refresh = False
                        except Exception:
                            pass
                    active_syms = set(pos_by_sym.keys())
                    # ✅ 포지션 제한(총/낮은 확신) - 신규 진입에서 사용
                    try:
                        max_pos_total = int(cfg.get("max_open_positions_total", 5) or 5)
                    except Exception:
                        max_pos_total = 5
                    try:
                        max_pos_low_conf = int(cfg.get("max_open_positions_low_conf", 2) or 2)
                    except Exception:
                        max_pos_low_conf = 2
                    try:
                        low_conf_th = int(cfg.get("low_conf_position_threshold", 92) or 92)
                    except Exception:
                        low_conf_th = 92

                    scan_cycle_start = time.time()
                    ccxt_timeout_epoch_scan_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                    # ✅ 강제 Exit(수익보존) 정책 사용 시:
                    # - 포지션 보유 중에는 스캔/AI 호출로 루프가 길어져 청산 타이밍을 놓칠 수 있어,
                    #   기본은 "포지션 모니터링 우선"으로 스캔을 잠깐 쉰다.
                    skip_scan_loop = False
                    try:
                        if bool(cfg.get("exit_trailing_protect_enable", False)) and bool(cfg.get("exit_trailing_protect_pause_scan_while_in_position", True)):
                            # 포지션을 더 열 수 있으면(최대 갯수 미만) 스캔은 계속 돌려 신규 진입 기회를 유지
                            # - 스캔 자체는 가볍고(대상 코인 5개), AI는 봉당 1회 캐시로 비용을 제어
                            max_pos_total = int(cfg.get("max_open_positions_total", 5) or 5)
                            if active_syms and (len(active_syms) >= max(1, max_pos_total)) and (not bool(force_scan_pending)):
                                skip_scan_loop = True
                                mon_add_scan(mon, stage="scan_skipped", symbol="*", tf=str(cfg.get("timeframe", "")), message="포지션 가득(최대치) → 스캔 잠시 중단")
                    except Exception:
                        skip_scan_loop = False

                    for sym in (TARGET_COINS if (not skip_scan_loop) else []):
                        # 포지션 있으면 스킵
                        if sym in active_syms:
                            mon_add_scan(mon, stage="in_position", symbol=sym, tf=str(cfg.get("timeframe", "")), message="이미 포지션 보유")
                            continue

                        # 현재(단기) 봉 timestamp(ms) - AI 재호출 최소화(같은 봉에서는 캐시 사용)
                        short_last_bar_ms = 0

                        # 쿨다운
                        cd = float(rt.get("cooldowns", {}).get(sym, 0))
                        if time.time() < cd:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "쿨다운(잠깐 쉬는중)"
                            mon_add_scan(mon, stage="trade_skipped", symbol=sym, tf=str(cfg.get("timeframe", "")), message="쿨다운")
                            continue

                        # 데이터 로드(단기: cfg timeframe)
                        try:
                            mon_add_scan(mon, stage="fetch_short", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="OHLCV 로드")
                            _to_before = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                            ohlcv = safe_fetch_ohlcv(ex, sym, str(cfg.get("timeframe", "5m")), limit=220)
                            if not ohlcv:
                                _to_after = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                                if _to_after and _to_after > _to_before:
                                    raise FuturesTimeoutError("fetch_ohlcv_timeout")
                                raise RuntimeError("ohlcv_empty")
                            try:
                                short_last_bar_ms = int((ohlcv[-1] or [0])[0] or 0)
                            except Exception:
                                short_last_bar_ms = 0
                            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                            df["time"] = pd.to_datetime(df["time"], unit="ms")
                        except FuturesTimeoutError as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = "ccxt timeout(ohlcv)"
                            mon_add_scan(mon, stage="fetch_short_fail", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message=str(e)[:140])
                            need_exchange_refresh = True
                            # 강제스캔 요약에도 반영
                            try:
                                if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                    force_scan_summary_lines.append(f"- {sym}: fetch_short_timeout")
                            except Exception:
                                pass
                            break
                        except Exception as e:
                            mon.setdefault("coins", {}).setdefault(sym, {})
                            mon["coins"][sym]["skip_reason"] = f"데이터 실패: {e}"
                            mon_add_scan(mon, stage="fetch_short_fail", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message=str(e)[:140])
                            # 강제스캔 요약에도 반영
                            try:
                                if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                    force_scan_summary_lines.append(f"- {sym}: fetch_short_fail | {str(e)[:80]}")
                            except Exception:
                                pass
                            continue

                        df, stt, last = calc_indicators(df, cfg)
                        mon.setdefault("coins", {}).setdefault(sym, {})
                        cs = mon["coins"][sym]

                        if last is None:
                            cs.update({"last_scan_kst": now_kst_str(), "ai_called": False, "skip_reason": "지표 계산 실패(ta/데이터 부족)"})
                            mon_add_scan(mon, stage="rule_signal", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="지표 계산 실패")
                            try:
                                if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                    force_scan_summary_lines.append(f"- {sym}: indicator_fail(ta/데이터 부족)")
                            except Exception:
                                pass
                            continue

                        # 장기추세(1h) 계산 + 캐시
                        htf_tf = str(cfg.get("trend_filter_timeframe", "1h"))
                        htf_trend = get_htf_trend_cached(
                            ex,
                            sym,
                            htf_tf,
                            fast=int(cfg.get("ma_fast", 7)),
                            slow=int(cfg.get("ma_slow", 99)),
                            cache_sec=int(cfg.get("trend_filter_cache_sec", 60)),
                        )
                        cs["trend_htf"] = f"🧭 {htf_tf} {htf_trend}"
                        mon_add_scan(mon, stage="fetch_long", symbol=sym, tf=htf_tf, signal=htf_trend, message="장기추세 계산")
                        # ccxt timeout 감지(백그라운드 작업 동시 실행 꼬임 방지)
                        try:
                            _to_after_long = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                            if _to_after_long and _to_after_long > float(ccxt_timeout_epoch_scan_start or 0):
                                mon_add_scan(mon, stage="fetch_long_fail", symbol=sym, tf=htf_tf, message="ccxt timeout(fetch_long)")
                                need_exchange_refresh = True
                                break
                        except Exception:
                            pass

                        # 모니터 기록(단기/장기 같이)
                        cs.update(
                            {
                                "last_scan_epoch": time.time(),
                                "last_scan_kst": now_kst_str(),
                                "price": float(last["close"]),
                                "short_last_bar_ms": int(short_last_bar_ms or 0),
                                "trend_short": stt.get("추세", ""),  # 단기추세(timeframe)
                                "trend_long": cs.get("trend_htf", ""),  # 장기추세(1h)
                                "rsi": float(last.get("RSI", 0)) if "RSI" in df.columns else None,
                                "adx": float(last.get("ADX", 0)) if "ADX" in df.columns else None,
                                "bb": stt.get("BB", ""),
                                "macd": stt.get("MACD", ""),
                                "vol": stt.get("거래량", ""),
                                "sqz": stt.get("SQZ", ""),
                                "sqz_mom_pct": stt.get("_sqz_mom_pct", ""),
                                "sqz_bias": stt.get("_sqz_bias", ""),
                                "pattern": stt.get("패턴", ""),
                                "pattern_bias": stt.get("_pattern_bias", 0),
                                "pattern_strength": stt.get("_pattern_strength", 0.0),
                                "pattern_mtf_summary": "",
                                "pullback_candidate": bool(stt.get("_pullback_candidate", False)),
                            }
                        )

                        # ✅ 멀티 타임프레임 패턴(1m/3m/5m/15m/30m/1h/2h/4h) 반영
                        try:
                            pat_mtf = get_chart_patterns_mtf_cached(ex, sym, cfg)
                        except Exception:
                            pat_mtf = {}
                        try:
                            if isinstance(pat_mtf, dict) and bool(pat_mtf.get("enabled", False)):
                                stt["_pattern_mtf"] = dict(pat_mtf)
                                cs["pattern_mtf_summary"] = str(pat_mtf.get("summary", "") or "")
                                cs["pattern_mtf_bias"] = int(pat_mtf.get("bias", 0) or 0)
                                cs["pattern_mtf_strength"] = float(pat_mtf.get("strength", 0.0) or 0.0)
                                merge_w = float(cfg.get("pattern_mtf_merge_weight", 0.60) or 0.60)
                                mb, ms = merge_pattern_bias(
                                    int(stt.get("_pattern_bias", 0) or 0),
                                    float(stt.get("_pattern_strength", 0.0) or 0.0),
                                    int(pat_mtf.get("bias", 0) or 0),
                                    float(pat_mtf.get("strength", 0.0) or 0.0),
                                    merge_weight=merge_w,
                                )
                                stt["_pattern_bias"] = int(mb)
                                stt["_pattern_strength"] = float(ms)
                                stt["패턴"] = str((stt.get("패턴", "") or "패턴 없음")).strip()
                                mtf_short = str(pat_mtf.get("summary", "") or "")
                                if mtf_short:
                                    stt["패턴"] = f"{stt['패턴']} | {mtf_short[:120]}"
                                cs["pattern"] = stt.get("패턴", "")
                                cs["pattern_bias"] = int(stt.get("_pattern_bias", 0) or 0)
                                cs["pattern_strength"] = float(stt.get("_pattern_strength", 0.0) or 0.0)
                                mon_add_scan(
                                    mon,
                                    stage="pattern_mtf",
                                    symbol=sym,
                                    tf="1m~4h",
                                    signal=str(stt.get("_pattern_bias", 0)),
                                    score=float(stt.get("_pattern_strength", 0.0) or 0.0),
                                    message=str(pat_mtf.get("summary", "") or "")[:120],
                                )
                        except Exception:
                            pass

                        # ✅ S/R 계산(스캔 과정 표시용) - 캐시 사용
                        sr_ctx: Optional[Dict[str, Any]] = None
                        try:
                            sr_tf0 = str(cfg.get("sr_timeframe", "15m"))
                            sr_lb0 = int(cfg.get("sr_lookback", 220))
                            sr_cache0 = int(cfg.get("sr_levels_cache_sec", 60))
                            sr_levels = get_sr_levels_cached(
                                ex,
                                sym,
                                sr_tf0,
                                pivot_order=int(cfg.get("sr_pivot_order", 6)),
                                cache_sec=sr_cache0,
                                limit=sr_lb0,
                            )
                            # ccxt timeout 감지(SR 계산 중)
                            try:
                                _to_after_sr = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                                if _to_after_sr and _to_after_sr > float(ccxt_timeout_epoch_scan_start or 0):
                                    mon_add_scan(mon, stage="support_resistance_fail", symbol=sym, tf=sr_tf0, message="ccxt timeout(SR)")
                                    need_exchange_refresh = True
                                    break
                            except Exception:
                                pass
                            supports = list(sr_levels.get("supports") or [])
                            resistances = list(sr_levels.get("resistances") or [])
                            px0 = float(last["close"])
                            near_sup = max([s for s in supports if s < px0], default=None) if supports else None
                            near_res = min([r for r in resistances if r > px0], default=None) if resistances else None
                            cs["sr_tf"] = sr_tf0
                            cs["sr_support_near"] = near_sup
                            cs["sr_resistance_near"] = near_res
                            sr_ctx = {
                                "tf": sr_tf0,
                                "support_near": near_sup,
                                "resistance_near": near_res,
                                "supports": supports[:8],
                                "resistances": resistances[:8],
                            }
                            # ✅ 매물대(간이 Volume Profile) 노드(매물 집중 구간)도 같이 제공(AI 목표가/트레일링 기준 참고)
                            try:
                                vp_nodes = volume_profile_nodes(df, bins=60, top_n=8)
                            except Exception:
                                vp_nodes = []
                            if vp_nodes:
                                sr_ctx["volume_nodes"] = vp_nodes[:8]
                                cs["volume_nodes"] = vp_nodes[:8]
                            mon_add_scan(
                                mon,
                                stage="support_resistance",
                                symbol=sym,
                                tf=sr_tf0,
                                signal="S/R",
                                score="",
                                message=f"sup={near_sup} res={near_res}",
                                extra={"support": near_sup, "resistance": near_res},
                            )
                        except Exception as e:
                            mon_add_scan(mon, stage="support_resistance", symbol=sym, tf=str(cfg.get("sr_timeframe", "")), message=f"SR 실패: {e}"[:140])

                        # ✅ 주력 지표(요구): Lorentzian / KNN / Logistic / SQZ / RSI
                        # - 3개 이상 수렴할 때만 진입(그 전엔 AI 호출도 막아 비용 절감)
                        ml_cache_key = ""
                        try:
                            if bool(cfg.get("ml_cache_enable", True)) and int(short_last_bar_ms or 0) > 0:
                                ml_cache_key = f"{sym}|{int(short_last_bar_ms)}"
                        except Exception:
                            ml_cache_key = ""
                        ml = ml_signals_and_convergence(df, stt, cfg, cache_key=str(ml_cache_key or ""))
                        try:
                            cs["ml_dir"] = str(ml.get("dir", "hold"))
                            cs["ml_votes"] = int(ml.get("votes_max", 0) or 0)
                            cs["ml_detail"] = str(ml.get("detail", ""))[:240]
                            cs["ml_knn_prob"] = float(ml.get("knn_prob", 0.5) or 0.5)
                            cs["ml_lor_prob"] = float(ml.get("lor_prob", 0.5) or 0.5)
                            cs["ml_logit_prob"] = float(ml.get("logit_prob", 0.5) or 0.5)
                            cs["ml_rsi_sig"] = int(ml.get("rsi_sig", 0) or 0)
                            cs["ml_sqz_sig"] = int(ml.get("sqz_sig", 0) or 0)
                            cs["ml_knn_sig"] = int(ml.get("knn_sig", 0) or 0)
                            cs["ml_lor_sig"] = int(ml.get("lor_sig", 0) or 0)
                            cs["ml_logit_sig"] = int(ml.get("logit_sig", 0) or 0)
                        except Exception:
                            pass
                        try:
                            stt["ML"] = str(ml.get("detail", ""))[:240]
                            stt["_ml_signals"] = dict(ml) if isinstance(ml, dict) else {}
                        except Exception:
                            pass
                        mon_add_scan(
                            mon,
                            stage="ml_signal",
                            symbol=sym,
                            tf=str(cfg.get("timeframe", "5m")),
                            signal=str(ml.get("dir", "hold")),
                            score=int(ml.get("votes_max", 0) or 0),
                            message=str(ml.get("detail", ""))[:120],
                        )

                        # AI 호출 필터(완화 + 모드/추세 기반)
                        # - "해소 신호가 없으면 AI 자체를 안 부른다"가 너무 보수적이라 무포지션이 길어질 수 있음
                        # - 강한 시그널(눌림목/RSI해소/밴드이탈)은 우선 호출
                        # - 그 외에는 ADX/거래량/모멘텀을 조합해 "추세 지속" 가능성이 있을 때 호출
                        call_ai = False
                        try:
                            sig_pullback = bool(stt.get("_pullback_candidate", False))
                            sig_rsi_resolve = bool(stt.get("_rsi_resolve_long", False)) or bool(stt.get("_rsi_resolve_short", False))
                            adxv = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            pattern_bias = int(stt.get("_pattern_bias", 0) or 0)
                            pattern_strength = float(stt.get("_pattern_strength", 0.0) or 0.0)
                            pattern_call_min = float(cfg.get("pattern_call_strength_min", 0.45) or 0.45)
                            pattern_strong = bool(cfg.get("use_chart_patterns", True)) and (abs(pattern_bias) == 1) and (pattern_strength >= pattern_call_min)

                            # 모드별 ADX 임계(진입이 너무 안 되는 문제 완화)
                            adx_th = float(cfg.get("ai_call_adx_threshold", 0) or 0)
                            if adx_th <= 0:
                                if str(mode) == "안전모드":
                                    adx_th = 20.0
                                elif str(mode) == "공격모드":
                                    adx_th = 17.0
                                else:
                                    adx_th = 15.0
                            # 추세 신호만으로도 AI를 부를 때 필요한 최소 ADX(너무 보수적이면 무포지션이 길어짐)
                            trend_min_adx = float(cfg.get("ai_call_trend_min_adx", 0) or 0)
                            if trend_min_adx <= 0:
                                if str(mode) == "안전모드":
                                    trend_min_adx = 12.0
                                elif str(mode) == "공격모드":
                                    trend_min_adx = 8.0
                                else:
                                    trend_min_adx = 6.0

                            trend_txt = str(stt.get("추세", "") or "")
                            macd_txt = str(stt.get("MACD", "") or "")
                            bb_txt = str(stt.get("BB", "") or "")
                            macd_cross = ("골든" in macd_txt) or ("데드" in macd_txt)

                            vol_spike = False
                            try:
                                vol_spike = ("VOL_SPIKE" in df.columns) and int(last.get("VOL_SPIKE", 0) or 0) == 1
                            except Exception:
                                vol_spike = False

                            rsi50_cross = False
                            try:
                                if "RSI" in df.columns and len(df) >= 3:
                                    rsi_prev = float(df["RSI"].iloc[-2])
                                    rsi_now = float(df["RSI"].iloc[-1])
                                    rsi50_cross = (rsi_prev < 50 <= rsi_now) or (rsi_prev > 50 >= rsi_now)
                            except Exception:
                                rsi50_cross = False

                            rsi_extreme = False
                            try:
                                if "RSI" in df.columns:
                                    rsi_now2 = float(last.get("RSI", 50))
                                    rsi_buy0 = float(cfg.get("rsi_buy", 30) or 30)
                                    rsi_sell0 = float(cfg.get("rsi_sell", 70) or 70)
                                    mrg = float(cfg.get("ai_call_rsi_extreme_margin", 5.0) or 5.0)
                                    # 과매도/과매수 근처(해소 전)도 "기회"로 보고 AI 호출(안전모드는 제외)
                                    rsi_extreme = (rsi_now2 <= (rsi_buy0 + mrg)) or (rsi_now2 >= (rsi_sell0 - mrg))
                            except Exception:
                                rsi_extreme = False

                            # ✅ SQZ(스퀴즈 모멘텀) 기반: 모멘텀 방향/세기가 기준 이상이면 AI 호출
                            sqz_mom_pct = 0.0
                            sqz_thr = 0.05
                            try:
                                sqz_mom_pct = float(stt.get("_sqz_mom_pct", 0.0) or 0.0)
                                sqz_thr = float(cfg.get("sqz_mom_threshold_pct", 0.05) or 0.05)
                            except Exception:
                                sqz_mom_pct = 0.0
                                sqz_thr = 0.05
                            try:
                                sqz_thr = float(max(0.001, abs(float(sqz_thr))))
                            except Exception:
                                sqz_thr = 0.05
                            sqz_strong = bool(cfg.get("use_sqz", True)) and bool(cfg.get("sqz_dependency_enable", True)) and (abs(float(sqz_mom_pct)) >= float(sqz_thr))

                            # 강한 시그널 우선
                            if pattern_strong:
                                call_ai = True
                            elif sqz_strong:
                                call_ai = True
                            elif sig_pullback or sig_rsi_resolve:
                                call_ai = True
                            elif ("상단 돌파" in bb_txt) or ("하단 이탈" in bb_txt):
                                call_ai = True
                            # ADX 추세강도 기반
                            elif adxv >= adx_th:
                                call_ai = True
                            # MACD 교차는 추세 전환/지속 후보(특히 공격/하이리스크에서 기회 포착)
                            elif (str(mode) != "안전모드") and macd_cross and (adxv >= max(6.0, float(trend_min_adx) - 3.0)):
                                call_ai = True
                            # RSI가 과매도/과매수 근처면(해소 전)도 AI를 부를 수 있게 완화(공격/하이리스크)
                            elif (str(mode) != "안전모드") and rsi_extreme:
                                call_ai = True
                            # 추세 지속/모멘텀(거래량/RSI50/ MACD) 기반
                            elif (
                                (("상승" in trend_txt) or ("하락" in trend_txt))
                                and (vol_spike or rsi50_cross or macd_cross)
                                and (adxv >= max(12.0, adx_th - 5.0))
                            ):
                                call_ai = True
                            # 추세 신호 단독으로도 AI 호출 허용(하이리스크 기회 포착)
                            elif (("상승" in trend_txt) or ("하락" in trend_txt)) and adxv >= max(float(trend_min_adx), adx_th - 8.0):
                                call_ai = True
                        except Exception:
                            call_ai = False

                        # ✅ (필수) 3-of-N 수렴 게이트: 진입/AI 호출은 이 조건을 최우선으로 적용
                        # - legacy call_ai 로직은 "참고"로만 남기고, 실제로는 수렴 조건이 우선한다.
                        try:
                            if bool(cfg.get("entry_convergence_enable", True)):
                                need = int(cfg.get("entry_convergence_min_votes", 3) or 3)
                                ml_dir = str(ml.get("dir", "hold") or "hold")
                                ml_votes = int(ml.get("votes_max", 0) or 0)
                                if (ml_dir in ["buy", "sell"]) and (ml_votes >= int(need)):
                                    call_ai = True
                                else:
                                    call_ai = False
                                    try:
                                        cs["skip_reason"] = f"지표 수렴 부족({ml_votes}/{need})"
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # ✅ 추가 필터(요구): 거래량 스파이크 + 이격도(Disparity) 체크
                        # - call_ai=True라도, 조건을 만족하지 않으면 AI 호출을 막아 비용/휩쏘 진입을 줄인다.
                        # - /scan 강제스캔은 아래 forced_ai에서 우회한다.
                        filter_msgs: List[str] = []
                        vol_ratio: Optional[float] = None
                        disparity_pct: Optional[float] = None
                        try:
                            if call_ai and bool(cfg.get("ai_call_require_volume_spike", True)):
                                per = max(5, int(cfg.get("ai_call_volume_spike_period", 20) or 20))
                                mul = float(cfg.get("ai_call_volume_spike_mul", 1.5) or 1.5)
                                try:
                                    vv = df["vol"].astype(float)
                                    if len(vv) >= per + 1:
                                        v_now = float(vv.iloc[-1])
                                        v_ma = float(vv.iloc[-(per + 1):-1].mean())
                                        if v_ma > 0:
                                            vol_ratio = float(v_now / v_ma)
                                            if float(vol_ratio) < float(mul):
                                                filter_msgs.append(f"거래량 부족({vol_ratio:.2f}x < {mul:.2f}x)")
                                except Exception:
                                    pass

                            if call_ai and bool(cfg.get("ai_call_require_disparity", True)):
                                ma_p = max(5, int(cfg.get("ai_call_disparity_ma_period", 20) or 20))
                                max_abs = float(cfg.get("ai_call_disparity_max_abs_pct", 4.0) or 4.0)
                                try:
                                    cc = df["close"].astype(float)
                                    if len(cc) >= ma_p:
                                        ma = float(cc.rolling(ma_p).mean().iloc[-1])
                                        if ma > 0:
                                            px_now = float(cc.iloc[-1])
                                            disparity_pct = float((px_now - ma) / ma * 100.0)
                                            if abs(float(disparity_pct)) > float(max_abs):
                                                filter_msgs.append(f"이격도 과다({abs(disparity_pct):.1f}% > {max_abs:.1f}%)")
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        if call_ai:
                            try:
                                mon_add_scan(
                                    mon,
                                    stage="rule_filter",
                                    symbol=sym,
                                    tf=str(cfg.get("timeframe", "5m")),
                                    signal="vol/disparity/pattern",
                                    score="",
                                    message=(
                                        "PASS"
                                        if not filter_msgs
                                        else (
                                            (("BLOCK: " if bool(cfg.get("ai_call_filters_block_ai", False)) else "WARN: ") + " / ".join(filter_msgs))[:180]
                                        )
                                    ),
                                    extra={
                                        "vol_ratio": vol_ratio,
                                        "disparity_pct": disparity_pct,
                                        "pattern_bias": int(stt.get("_pattern_bias", 0) or 0),
                                        "pattern_strength": float(stt.get("_pattern_strength", 0.0) or 0.0),
                                    },
                                )
                            except Exception:
                                pass
                        if call_ai and filter_msgs and bool(cfg.get("ai_call_filters_block_ai", False)):
                            call_ai = False
                            try:
                                cs["skip_reason"] = " / ".join(filter_msgs)[:160]
                            except Exception:
                                pass
                        elif call_ai and filter_msgs:
                            try:
                                cs["prefilter_note"] = " / ".join(filter_msgs)[:160]
                            except Exception:
                                pass

                        # ✅ /scan 강제스캔: 원래 call_ai=False인 경우에만 AI를 "추가로" 호출(주문은 막기 위해 플래그 보관)
                        forced_ai = False
                        try:
                            if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)) and (not call_ai):
                                call_ai = True
                                forced_ai = True
                        except Exception:
                            forced_ai = False

                        # ✅ rule_signal 단계 기록
                        try:
                            sigs = []
                            if bool(stt.get("_pullback_candidate", False)):
                                sigs.append("pullback")
                            if bool(stt.get("_rsi_resolve_long", False)):
                                sigs.append("rsi_resolve_long")
                            if bool(stt.get("_rsi_resolve_short", False)):
                                sigs.append("rsi_resolve_short")
                            try:
                                pb = int(stt.get("_pattern_bias", 0) or 0)
                                if pb == 1:
                                    sigs.append("pattern_long")
                                elif pb == -1:
                                    sigs.append("pattern_short")
                            except Exception:
                                pass
                            adxv2 = float(last.get("ADX", 0)) if "ADX" in df.columns else 0.0
                            mon_add_scan(
                                mon,
                                stage="rule_signal",
                                symbol=sym,
                                tf=str(cfg.get("timeframe", "5m")),
                                signal=",".join(sigs) if sigs else "none",
                                score=adxv2,
                                message=("AI 호출(강제스캔)" if forced_ai else ("AI 호출" if call_ai else "AI 스킵(휩쏘 위험)")),
                                extra={"pullback": bool(stt.get("_pullback_candidate", False)), "adx": adxv2},
                            )
                        except Exception:
                            pass

                        if not call_ai:
                            cs["ai_called"] = False
                            # pre-filter(거래량/이격도)에서 이미 skip_reason을 남겼다면 덮어쓰지 않음
                            if not str(cs.get("skip_reason", "") or "").strip():
                                cs["skip_reason"] = "횡보/해소 신호 없음(휩쏘 위험)"
                            monitor_write_throttled(mon, 1.0)
                            mon_add_scan(mon, stage="trade_skipped", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="call_ai=False")
                            continue

                        # ✅ 비용 절감: 신규진입이 꺼져 있으면 AI를 자동 호출하지 않음(강제스캔만 예외)
                        # - 사용자가 버튼(/scan)으로 요청할 때만 AI 호출
                        if (not entry_allowed_global) and (not forced_ai):
                            try:
                                cs["ai_called"] = False
                                cs["ai_decision"] = "-"
                                cs["ai_confidence"] = ""
                                cs["ai_reason_easy"] = ""
                                cs["skip_reason"] = "신규진입 OFF: AI 생략(/scan으로 수동 호출)"
                            except Exception:
                                pass
                            monitor_write_throttled(mon, 1.0)
                            mon_add_scan(mon, stage="ai_skipped", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="entry_disabled(auto_trade/paused/weekend)")
                            continue

                        # AI 판단
                        # ✅ 비용 절감: 같은 봉에서는 AI를 재호출하지 않고 캐시 재사용(강제스캔 제외)
                        use_cached_ai = False
                        try:
                            if (not forced_ai) and bool(cfg.get("ai_scan_once_per_bar", True)):
                                last_ai_bar = int(cs.get("ai_last_called_bar_ms", 0) or 0)
                                cur_bar = int(short_last_bar_ms or 0)
                                if cur_bar > 0 and last_ai_bar == cur_bar and str(cs.get("ai_decision", "") or "").strip():
                                    use_cached_ai = True
                        except Exception:
                            use_cached_ai = False

                        if use_cached_ai:
                            mon_add_scan(mon, stage="ai_cached", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="같은 봉: 캐시 재사용")
                            try:
                                ai = {
                                    "decision": str(cs.get("ai_decision", "hold") or "hold"),
                                    "confidence": int(_as_int(cs.get("ai_confidence", 0), 0)),
                                    "entry_pct": float(_as_float(cs.get("ai_entry_pct", rule["entry_pct_min"]), float(rule["entry_pct_min"]))),
                                    "leverage": int(_as_int(cs.get("ai_leverage", rule["lev_min"]), int(rule["lev_min"]))),
                                    "sl_pct": float(_as_float(cs.get("ai_sl_pct", 1.2), 1.2)),
                                    "tp_pct": float(_as_float(cs.get("ai_tp_pct", 3.0), 3.0)),
                                    "rr": float(_as_float(cs.get("ai_rr", 1.5), 1.5)),
                                    "used_indicators": [x.strip() for x in str(cs.get("ai_used", "") or "").split(",") if x.strip()],
                                    "reason_easy": str(cs.get("ai_reason_easy", "") or ""),
                                    "sl_price": cs.get("ai_sl_price", None),
                                    "tp_price": cs.get("ai_tp_price", None),
                                }
                            except Exception:
                                ai = {"decision": "hold", "confidence": 0, "reason_easy": "ai_cache_parse_fail", "used_indicators": stt.get("_used_indicators", [])}
                        else:
                            mon_add_scan(mon, stage="ai_call", symbol=sym, tf=str(cfg.get("timeframe", "5m")), message="AI 판단 요청")
                        # ✅ 요구: 스윙 판단일 때만 외부시황을 AI에 제공(스캘핑/단기=차트만)
                        try:
                            tr_s = str(stt.get("추세", "") or "")
                            tr_l = str(htf_trend or "")
                            chart_style_hint = "스윙" if (("상승" in tr_s and "상승" in tr_l) or ("하락" in tr_s and "하락" in tr_l)) else "스캘핑"
                        except Exception:
                            chart_style_hint = "스캘핑"
                        cs["chart_style_hint"] = chart_style_hint
                        try:
                            mon_add_scan(mon, stage="style_hint", symbol=sym, tf=str(cfg.get("timeframe", "5m")), signal=chart_style_hint, message="차트 기반 스타일 힌트")
                        except Exception:
                            pass
                        ext_for_ai = ext if chart_style_hint == "스윙" else {"enabled": False}
                        if not use_cached_ai:
                            ai = ai_decide_trade(
                                df,
                                stt,
                                sym,
                                mode,
                                cfg,
                                external=ext_for_ai,
                                trend_long=str(htf_trend or ""),
                                sr_context=sr_ctx,
                                chart_style_hint=chart_style_hint,
                            )
                            try:
                                cs["ai_last_called_epoch"] = float(time.time())
                                cs["ai_last_called_bar_ms"] = int(short_last_bar_ms or 0)
                                cs["ai_sl_price"] = ai.get("sl_price", None)
                                cs["ai_tp_price"] = ai.get("tp_price", None)
                            except Exception:
                                pass
                        decision = ai.get("decision", "hold")
                        conf = int(ai.get("confidence", 0))
                        mon_add_scan(mon, stage="ai_result", symbol=sym, tf=str(cfg.get("timeframe", "5m")), signal=str(decision), score=conf, message=str(ai.get("reason_easy", ""))[:80])

                        # ✅ 주력 지표 수렴(3-of-N)과 AI 방향이 다르면 진입하지 않음(비용/과오류 방지)
                        try:
                            if bool(cfg.get("entry_convergence_enable", True)):
                                ml_dir = str(ml.get("dir", "hold") or "hold")
                                if (ml_dir in ["buy", "sell"]) and (str(decision) in ["buy", "sell"]) and (str(decision) != ml_dir):
                                    raw = str(decision)
                                    decision = "hold"
                                    conf = int(max(0, int(round(float(conf) * 0.25))))
                                    try:
                                        cs["skip_reason"] = f"지표 수렴({ml_dir}) vs AI({raw}) 불일치"
                                    except Exception:
                                        pass
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=raw,
                                        score=int(ai.get("confidence", 0) or 0),
                                        message=str(cs.get("skip_reason", ""))[:140],
                                        extra={"ml_dir": ml_dir, "ml_votes": int(ml.get("votes_max", 0) or 0)},
                                    )
                        except Exception:
                            pass

                        # ✅ SQZ 의존도(요구: 80%+): SQZ 모멘텀이 반대/중립이면 진입을 강하게 억제
                        sqz_skip_reason = ""
                        try:
                            raw_decision = str(decision or "hold")
                            raw_conf = int(conf)
                            if raw_decision in ["buy", "sell"] and bool(cfg.get("use_sqz", True)) and bool(cfg.get("sqz_dependency_enable", True)):
                                w = float(cfg.get("sqz_dependency_weight", 0.80) or 0.80)
                                w = float(clamp(w, 0.0, 1.0))
                                gate = bool(cfg.get("sqz_dependency_gate_entry", True))
                                override = bool(cfg.get("sqz_dependency_override_ai", True))
                                bias = int(stt.get("_sqz_bias", 0) or 0)
                                mom_pct = float(stt.get("_sqz_mom_pct", 0.0) or 0.0)
                                aligned = (bias == 1 and raw_decision == "buy") or (bias == -1 and raw_decision == "sell")
                                opposed = (bias == 1 and raw_decision == "sell") or (bias == -1 and raw_decision == "buy")

                                if opposed:
                                    conf = int(round(float(conf) * max(0.0, 1.0 - w)))
                                    if override:
                                        decision = "hold"
                                        sqz_skip_reason = f"SQZ 반대 모멘텀({mom_pct:+.2f}%)"
                                elif bias == 0 and gate:
                                    conf = int(round(float(conf) * max(0.0, 1.0 - w)))
                                    decision = "hold"
                                    sqz_skip_reason = f"SQZ 중립 모멘텀({mom_pct:+.2f}%)"
                                elif aligned:
                                    # 정방향이면 confidence 소폭 보정(최대 100)
                                    conf = int(min(100, int(conf) + int(round(10.0 * w))))

                                if sqz_skip_reason:
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(raw_decision),
                                        score=int(raw_conf),
                                        message=sqz_skip_reason,
                                        extra={"sqz_mom_pct": mom_pct, "sqz_bias": bias, "w": w},
                                    )
                        except Exception:
                            sqz_skip_reason = ""

                        pattern_skip_reason = ""
                        try:
                            raw_decision2 = str(decision or "hold")
                            raw_conf2 = int(conf)
                            if raw_decision2 in ["buy", "sell"] and bool(cfg.get("use_chart_patterns", True)):
                                p_bias = int(stt.get("_pattern_bias", 0) or 0)
                                p_strength = float(stt.get("_pattern_strength", 0.0) or 0.0)
                                p_gate = float(cfg.get("pattern_gate_strength", 0.65) or 0.65)
                                p_gate = float(clamp(p_gate, 0.05, 1.0))
                                aligned = (p_bias == 1 and raw_decision2 == "buy") or (p_bias == -1 and raw_decision2 == "sell")
                                opposed = (p_bias == 1 and raw_decision2 == "sell") or (p_bias == -1 and raw_decision2 == "buy")
                                if aligned:
                                    conf = int(min(100, int(conf) + int(round(8.0 * max(0.0, min(1.0, p_strength))))))
                                elif opposed and bool(cfg.get("pattern_gate_entry", True)) and p_strength >= p_gate:
                                    conf = int(round(float(conf) * max(0.0, 1.0 - min(0.85, 0.35 + (p_strength * 0.45)))))
                                    if bool(cfg.get("pattern_override_ai", True)):
                                        decision = "hold"
                                    pattern_skip_reason = f"패턴 반대({p_strength:.2f})"
                                if pattern_skip_reason:
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(raw_decision2),
                                        score=int(raw_conf2),
                                        message=pattern_skip_reason,
                                        extra={"pattern_bias": p_bias, "pattern_strength": p_strength},
                                    )
                        except Exception:
                            pattern_skip_reason = ""

                        skip_reason_merged = " / ".join([x for x in [sqz_skip_reason, pattern_skip_reason] if str(x).strip()])
                        # 강제스캔 요약 라인(요구사항: /scan 결과는 짧게)
                        try:
                            if force_scan_pending and ((not force_scan_syms_set) or (sym in force_scan_syms_set)):
                                force_scan_summary_lines.append(f"- {sym}: {str(decision).upper()}({conf}%) | {str(ai.get('reason_easy',''))[:60]}")
                        except Exception:
                            pass

                        cs.update(
                            {
                                "ai_called": True,
                                "ai_decision": decision,
                                "ai_confidence": conf,
                                "ai_entry_pct": float(ai.get("entry_pct", rule["entry_pct_min"])),
                                "ai_leverage": int(ai.get("leverage", rule["lev_min"])),
                                "ai_sl_pct": float(ai.get("sl_pct", 1.2)),
                                "ai_tp_pct": float(ai.get("tp_pct", 3.0)),
                                "ai_rr": float(ai.get("rr", 1.5)),
                                "ai_used": ", ".join(ai.get("used_indicators", [])),
                                "ai_reason_easy": ai.get("reason_easy", ""),
                                "pattern": stt.get("패턴", ""),
                                "pattern_bias": int(stt.get("_pattern_bias", 0) or 0),
                                "pattern_strength": float(stt.get("_pattern_strength", 0.0) or 0.0),
                                "min_conf_required": int(rule["min_conf"]),
                                "skip_reason": skip_reason_merged,
                            }
                        )
                        monitor_write_throttled(mon, 1.0)

                        # 진입(STRICT + SOFT)
                        min_conf_strict = int(rule.get("min_conf", 0) or 0)
                        min_conf_soft = int(min_conf_strict)
                        is_soft_entry = False
                        try:
                            if decision in ["buy", "sell"] and bool(cfg.get("soft_entry_enable", True)):
                                if str(mode) == "안전모드":
                                    gap = int(cfg.get("soft_entry_conf_gap_safe", 0) or 0)
                                elif str(mode) == "공격모드":
                                    gap = int(cfg.get("soft_entry_conf_gap_attack", 8) or 8)
                                else:
                                    gap = int(cfg.get("soft_entry_conf_gap_highrisk", 6) or 6)
                                min_conf_soft = int(max(0, int(min_conf_strict) - int(max(0, gap))))
                        except Exception:
                            min_conf_soft = int(min_conf_strict)

                        # ✅ buy/sell인데 확신도가 낮아 진입을 못 하면, 스킵 사유를 남겨 원인 파악을 쉽게 한다.
                        try:
                            if decision in ["buy", "sell"] and int(conf) < int(min_conf_soft):
                                cs["skip_reason"] = f"확신도 부족({int(conf)}% < {int(min_conf_soft)}%)"
                                mon_add_scan(
                                    mon,
                                    stage="trade_skipped",
                                    symbol=sym,
                                    tf=str(cfg.get("timeframe", "5m")),
                                    signal=str(decision),
                                    score=conf,
                                    message=str(cs.get("skip_reason", ""))[:140],
                                    extra={"min_conf_strict": int(min_conf_strict), "min_conf_soft": int(min_conf_soft)},
                                )
                        except Exception:
                            pass

                        if decision in ["buy", "sell"] and conf >= int(min_conf_soft):
                            is_soft_entry = bool(int(conf) < int(min_conf_strict))
                            # ✅ 강제스캔(scan_only) 또는 auto_trade OFF/정지/주말이면 신규진입 금지
                            if (not entry_allowed_global) or (forced_ai and force_scan_only):
                                try:
                                    why = "entry_disabled"
                                    if forced_ai and force_scan_only:
                                        why = "force_scan(scan_only)"
                                    elif not trade_enabled:
                                        why = "auto_trade=OFF"
                                    elif paused_now:
                                        why = "paused(loss_protect)"
                                    elif cfg.get("no_trade_weekend", False) and (now_kst().weekday() in [5, 6]):
                                        why = "weekend_block"
                                    cs["skip_reason"] = f"신규진입 금지({why})"
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message=f"신규진입 금지({why})",
                                        extra={"forced_ai": forced_ai, "force_scan_only": force_scan_only, "trade_enabled": trade_enabled},
                                    )
                                except Exception:
                                    pass
                                continue
                            # ✅ 포지션 제한: 총 포지션 수 / 낮은 확신 포지션 수
                            try:
                                if len(active_syms) >= max(1, int(max_pos_total)):
                                    cs["skip_reason"] = f"포지션 제한({len(active_syms)}/{int(max_pos_total)})"
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message="max_open_positions_total",
                                        extra={"active": len(active_syms), "max": int(max_pos_total)},
                                    )
                                    continue
                            except Exception:
                                pass
                            try:
                                low_open = 0
                                if int(max_pos_low_conf) > 0 and int(low_conf_th) > 0:
                                    for s0 in active_syms:
                                        c0 = (active_targets.get(s0, {}) or {}).get("entry_confidence", None)
                                        c0i = int(_as_int(c0, 0)) if c0 is not None else 0
                                        if c0i and c0i < int(low_conf_th):
                                            low_open += 1
                                    if low_open >= int(max_pos_low_conf) and int(conf) < int(low_conf_th):
                                        cs["skip_reason"] = f"낮은 확신 포지션 한도({low_open}/{int(max_pos_low_conf)}). {int(low_conf_th)}%+만 추가"
                                        mon_add_scan(
                                            mon,
                                            stage="trade_skipped",
                                            symbol=sym,
                                            tf=str(cfg.get("timeframe", "5m")),
                                            signal=str(decision),
                                            score=conf,
                                            message="max_open_positions_low_conf",
                                            extra={"low_open": low_open, "max_low": int(max_pos_low_conf), "threshold": int(low_conf_th)},
                                        )
                                        continue
                            except Exception:
                                pass
                            px = float(last["close"])

                            # ✅ 스타일 결정 (단기/장기 추세로 스캘핑/스윙)
                            style_info = _style_for_entry(
                                sym,
                                decision,
                                stt.get("추세", ""),
                                htf_trend,
                                cfg,
                                allow_ai=bool(cfg.get("style_entry_ai_enable", False)),
                            )
                            style = style_info.get("style", "스캘핑")
                            cs["style_reco"] = style
                            cs["style_confidence"] = int(style_info.get("confidence", 0))
                            cs["style_reason"] = str(style_info.get("reason", ""))[:240]
                            # ✅ /mode 레짐 강제(auto|scalping|swing)
                            regime_mode = str(cfg.get("regime_mode", "auto")).lower().strip()
                            if regime_mode in ["scalping", "scalp", "short"]:
                                style = "스캘핑"
                                cs["style_reco"] = "스캘핑"
                                cs["style_confidence"] = 100
                                cs["style_reason"] = "레짐 강제: scalping"
                            elif regime_mode in ["swing", "long"]:
                                style = "스윙"
                                cs["style_reco"] = "스윙"
                                cs["style_confidence"] = 100
                                cs["style_reason"] = "레짐 강제: swing"

                            # ✅ 추세 필터 정책(기존 "금지" 기능 유지 + 새로운 "허용-스캘핑" 추가)
                            if cfg.get("trend_filter_enabled", True) and cfg.get("trend_filter_policy", "ALLOW_SCALP") == "STRICT":
                                is_down = ("하락" in str(htf_trend))
                                is_up = ("상승" in str(htf_trend))
                                if is_down and decision == "buy":
                                    cs["skip_reason"] = f"장기추세({htf_tf}) 하락이라 롱 금지(STRICT)"
                                    continue
                                if is_up and decision == "sell":
                                    cs["skip_reason"] = f"장기추세({htf_tf}) 상승이라 숏 금지(STRICT)"
                                    continue
                            elif cfg.get("trend_filter_enabled", True) and cfg.get("trend_filter_policy", "ALLOW_SCALP") == "ALLOW_SCALP" and regime_mode == "auto":
                                # 역추세면 스캘핑 강제
                                is_down = ("하락" in str(htf_trend))
                                is_up = ("상승" in str(htf_trend))
                                if is_down and decision == "buy":
                                    style = "스캘핑"
                                    cs["style_reco"] = "스캘핑"
                                    cs["style_reason"] = f"장기추세({htf_tf}) 하락 → 역추세는 스캘핑만"
                                if is_up and decision == "sell":
                                    style = "스캘핑"
                                    cs["style_reco"] = "스캘핑"
                                    cs["style_reason"] = f"장기추세({htf_tf}) 상승 → 역추세는 스캘핑만"

                            # ✅ 왜 스캘핑/스윙인지(단기/장기 추세 포함) 더 직관적으로 남김
                            try:
                                r0 = str(cs.get("style_reason", "") or "").strip()
                                ts0 = str(stt.get("추세", "") or "").strip()
                                tl0 = f"{htf_tf} {htf_trend}".strip()
                                if ts0 or tl0:
                                    r0 = (r0 + f" | 단기:{ts0 or '-'} / 장기:{tl0 or '-'}").strip()
                                cs["style_reason"] = r0[:240]
                            except Exception:
                                pass

                            # ✅ 하이리스크/하이리턴 모드 신규진입 제한(선택):
                            # - 사용자가 원하면(auto에서) "스윙(단기+장기 정렬)"일 때만 신규 진입하도록 제한 가능
                            # - 기본값은 OFF(진입 허용)이며, 이때도 MODE_RULES의 레버/진입비중 범위를 우선 존중한다.
                            if (
                                str(mode) == "하이리스크/하이리턴"
                                and str(regime_mode) == "auto"
                                and bool(cfg.get("highrisk_entry_requires_swing", False))
                                and str(style) != "스윙"
                            ):
                                cs["skip_reason"] = "하이리스크: 스윙(단기+장기 정렬)만 진입(설정)"
                                try:
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message="highrisk_requires_swing",
                                        extra={"mode": str(mode), "style": str(style), "trend_short": str(stt.get("추세", "")), "trend_long": str(htf_trend)},
                                    )
                                except Exception:
                                    pass
                                continue

                            # 스타일별 envelope + 리스크가드레일
                            ai2 = apply_style_envelope(ai, style, cfg, rule)
                            # ✅ ATR 기반 레버리지(요구): 변동성이 크면 레버↓(손절/익절 가격폭 일관성을 위해, risk_guardrail 이전에 결정)
                            try:
                                if (not bool(cfg.get("fixed_leverage_enable", False))) and bool(cfg.get("atr_leverage_enable", True)):
                                    w = int(cfg.get("atr_leverage_window", 14) or 14)
                                    atr_pct = float(_atr_price_pct(df, max(7, w)))
                                    lo_pct = float(cfg.get("atr_leverage_low_pct", 0.35) or 0.35)
                                    hi_pct = float(cfg.get("atr_leverage_high_pct", 1.20) or 1.20)
                                    min_lev_cfg = int(cfg.get("atr_leverage_min", 5) or 5)
                                    max_lev_cfg = int(cfg.get("atr_leverage_max", 20) or 20)

                                    # 모드/스타일 상한 반영
                                    lev_min_allowed = int(rule.get("lev_min", 1) or 1)
                                    lev_max_allowed = int(rule.get("lev_max", max_lev_cfg) or max_lev_cfg)
                                    try:
                                        if str(style) == "스캘핑":
                                            lev_max_allowed = min(lev_max_allowed, int(cfg.get("scalp_lev_cap", lev_max_allowed) or lev_max_allowed))
                                        elif str(style) == "스윙":
                                            lev_max_allowed = min(lev_max_allowed, int(cfg.get("swing_lev_cap", lev_max_allowed) or lev_max_allowed))
                                    except Exception:
                                        pass
                                    lev_min_allowed = max(1, lev_min_allowed, int(min_lev_cfg))
                                    lev_max_allowed = max(lev_min_allowed, min(int(max_lev_cfg), lev_max_allowed))

                                    lev_atr = float(lev_max_allowed)
                                    if atr_pct > 0 and hi_pct > lo_pct:
                                        if atr_pct <= lo_pct:
                                            lev_atr = float(lev_max_allowed)
                                        elif atr_pct >= hi_pct:
                                            lev_atr = float(lev_min_allowed)
                                        else:
                                            t = float((atr_pct - lo_pct) / (hi_pct - lo_pct))
                                            lev_atr = float(lev_max_allowed - t * (lev_max_allowed - lev_min_allowed))
                                    lev_atr_i = int(clamp(int(round(lev_atr)), lev_min_allowed, lev_max_allowed))
                                    ai2["leverage"] = int(lev_atr_i)
                                    ai2["leverage_source"] = "ATR"
                                    ai2["atr_price_pct"] = float(atr_pct)
                            except Exception:
                                pass
                            # ✅ 요구: 스윙만 외부시황 반영(스캘핑=차트만)
                            ext_for_risk = ext if str(style) == "스윙" else {"enabled": False}
                            ai2 = _risk_guardrail(ai2, df, decision, mode, style, ext_for_risk)
                            # ✅ 스캘핑: 레버가 높을 때 TP/SL이 과도해지는 문제(익절 미발동 등) 방지
                            if str(style) == "스캘핑":
                                ai2 = apply_scalp_price_guardrails(ai2, df, cfg, rule)

                            entry_pct = float(ai2.get("entry_pct", rule["entry_pct_min"]))
                            lev = int(ai2.get("leverage", rule["lev_min"]))
                            slp = float(ai2.get("sl_pct", 1.2))
                            tpp = float(ai2.get("tp_pct", 3.0))

                            # ✅ 요구사항: 레버 20배 고정 + 잔고 20% 진입(고정값 우선)
                            entry_pct_src = "AI"
                            lev_src = "AI"
                            try:
                                if bool(cfg.get("fixed_entry_pct_enable", False)):
                                    entry_pct = float(cfg.get("fixed_entry_pct", 20.0) or 20.0)
                                    ai2["entry_pct"] = float(entry_pct)
                                    entry_pct_src = "FIXED"
                            except Exception:
                                pass
                            try:
                                if bool(cfg.get("fixed_leverage_enable", False)):
                                    lev = int(cfg.get("fixed_leverage", 20) or 20)
                                    ai2["leverage"] = int(lev)
                                    lev_src = "FIXED"
                            except Exception:
                                pass

                            # ✅ Kelly sizing(선택): AI entry_pct가 과대할 때만 상한으로 눌러준다(half-kelly)
                            kelly_cap_pct: Optional[float] = None
                            try:
                                if (not bool(cfg.get("fixed_entry_pct_enable", False))) and bool(cfg.get("kelly_sizing_enable", False)):
                                    rr0 = float(_as_float(ai2.get("rr", 0.0), 0.0))
                                    p0 = float(clamp(float(conf) / 100.0, 0.05, 0.95))
                                    if rr0 > 0:
                                        f_star = p0 - ((1.0 - p0) / float(rr0))
                                    else:
                                        f_star = 0.0
                                    f_star = float(clamp(float(f_star), 0.0, 1.0))
                                    mult = float(cfg.get("kelly_fraction_mult", 0.5) or 0.5)
                                    f_use = float(clamp(float(f_star) * float(mult), 0.0, 1.0))
                                    cap_max = float(cfg.get("kelly_max_entry_pct", 20.0) or 20.0)
                                    kelly_cap_pct = float(clamp(float(f_use) * 100.0, 0.0, max(0.0, cap_max)))
                                    if kelly_cap_pct > 0:
                                        entry_pct = float(min(float(entry_pct), float(kelly_cap_pct)))
                                        ai2["entry_pct"] = float(entry_pct)
                                        ai2["entry_pct_kelly_cap"] = float(kelly_cap_pct)
                                        entry_pct_src = f"{entry_pct_src}+KELLY"
                            except Exception:
                                kelly_cap_pct = None

                            # ✅ 진입 조건 완화: "시그널/필터/확신도"에 따라 진입비중 자동 조절
                            # - 진입조건이 완벽하지 않거나(거래량/이격도 경고 등) 확신이 낮으면 비중을 자동으로 줄여 과매매/손실을 완화
                            # - 반대로 조건이 잘 맞고(conf 높음) 강한 시그널이면 비중을 약간 키운다(모드 범위 내)
                            try:
                                if bool(cfg.get("entry_size_scale_by_signal_enable", True)):
                                    # soft-entry면 entry_pct 하한을 더 낮게 허용(작게 진입)
                                    entry_floor = float(rule["entry_pct_min"])
                                    try:
                                        if bool(is_soft_entry):
                                            entry_floor = float(cfg.get("soft_entry_entry_pct_floor", 2.0) or 2.0)
                                    except Exception:
                                        entry_floor = float(rule["entry_pct_min"])

                                    sig_pull = bool(stt.get("_pullback_candidate", False))
                                    sig_rsi = bool(stt.get("_rsi_resolve_long", False)) or bool(stt.get("_rsi_resolve_short", False))
                                    conf0 = int(conf)
                                    base_min_conf = int(rule.get("min_conf", 0) or 0)

                                    # 확신도 기반
                                    if conf0 >= int(low_conf_th):
                                        conf_factor = 1.15
                                    elif conf0 >= int(base_min_conf + 4):
                                        conf_factor = 1.0
                                    else:
                                        conf_factor = 0.85

                                    # 시그널 강도 기반
                                    sig_factor = 1.10 if sig_pull else (1.0 if sig_rsi else 0.85)

                                    # 필터 경고(거래량/이격도) 기반: 경고가 있으면 보수적으로
                                    pre_factor = 0.75 if (isinstance(filter_msgs, list) and filter_msgs) else 1.0

                                    # ✅ SQZ 모멘텀 정방향이면 진입비중을 더 키우고(진입금↑),
                                    #    중립이면 약간 보수적으로(요구: SQZ 의존도 80%+)
                                    sqz_factor = 1.0
                                    try:
                                        if bool(cfg.get("use_sqz", True)) and bool(cfg.get("sqz_dependency_enable", True)):
                                            bias0 = int(stt.get("_sqz_bias", 0) or 0)
                                            str0 = float(stt.get("_sqz_strength", 0.0) or 0.0)
                                            w0 = float(cfg.get("sqz_dependency_weight", 0.80) or 0.80)
                                            w0 = float(clamp(w0, 0.0, 1.0))
                                            if bias0 != 0:
                                                sqz_factor = 1.0 + min(0.35, 0.25 * float(str0) * max(0.8, float(w0)))
                                            else:
                                                sqz_factor = 0.90
                                    except Exception:
                                        sqz_factor = 1.0

                                    f_max = 1.35
                                    try:
                                        if str(mode) == "공격모드":
                                            f_max = 1.45
                                        elif str(mode) == "하이리스크/하이리턴":
                                            f_max = 1.60
                                    except Exception:
                                        f_max = 1.35

                                    f = float(clamp(float(conf_factor) * float(sig_factor) * float(pre_factor) * float(sqz_factor), 0.35, float(f_max)))
                                    entry_pct_scaled = float(clamp(float(entry_pct) * f, float(entry_floor), float(rule["entry_pct_max"])))
                                    if abs(entry_pct_scaled - float(entry_pct)) > 1e-9:
                                        entry_pct = entry_pct_scaled
                                        ai2["entry_pct"] = float(entry_pct)
                                        ai2["entry_pct_scale_factor"] = float(f)
                                        ai2["entry_pct_scale_note"] = f"conf={conf0} pull={int(sig_pull)} rsi={int(sig_rsi)} warn={int(bool(filter_msgs))}"
                            except Exception:
                                pass

                            # ✅ soft-entry: 확신이 살짝 부족하면 "아주 작게/보수적으로" 진입(비중↓/레버↓)
                            try:
                                if bool(is_soft_entry) and bool(cfg.get("soft_entry_enable", True)):
                                    if not bool(cfg.get("fixed_entry_pct_enable", False)):
                                        mult_e = float(cfg.get("soft_entry_entry_pct_mult", 0.55) or 0.55)
                                        floor_e = float(cfg.get("soft_entry_entry_pct_floor", 2.0) or 2.0)
                                        entry_pct = float(max(float(floor_e), float(entry_pct) * float(clamp(mult_e, 0.1, 1.0))))
                                        ai2["entry_pct"] = float(entry_pct)
                                        ai2["entry_tier"] = "SOFT"
                                    if not bool(cfg.get("fixed_leverage_enable", False)):
                                        mult_l = float(cfg.get("soft_entry_leverage_mult", 0.75) or 0.75)
                                        floor_l = int(cfg.get("soft_entry_leverage_floor", 2) or 2)
                                        lev2 = int(round(float(lev) * float(clamp(mult_l, 0.1, 1.0))))
                                        lev2 = int(max(int(floor_l), min(int(lev), int(lev2))))
                                        lev = int(max(1, lev2))
                                        ai2["leverage"] = int(lev)
                                        ai2["entry_tier"] = "SOFT"
                            except Exception:
                                pass

                            # ✅ 외부시황 위험 감산은 스윙에서만 적용
                            # - 스윙 진입금(FNG 기반)을 별도로 쓰는 경우에는 FNG 감산을 중복 적용하지 않도록,
                            #   여기서는 "이벤트/브리핑" 요인만 반영한 multiplier를 사용한다.
                            entry_risk_mul = float(risk_mul) if str(style) == "스윙" else 1.0
                            try:
                                if str(style) == "스윙" and bool(cfg.get("swing_fng_entry_pct_enable", True)):
                                    entry_risk_mul = float(external_risk_multiplier(ext, cfg, include_fng=False))
                            except Exception:
                                entry_risk_mul = entry_risk_mul
                            entry_usdt = free_usdt * (entry_pct / 100.0) * entry_risk_mul

                            # ✅ (요구) 하이리스크/하이리턴 모드에서만: 총자산 20% 진입 + 레버 20x 고정
                            # - 스캘핑/스윙 스타일 캡보다 우선(사용자 요구)
                            try:
                                if str(mode) == "하이리스크/하이리턴" and bool(cfg.get("highrisk_fixed_size_enable", True)):
                                    lev_fix = int(cfg.get("highrisk_fixed_leverage", 20) or 20)
                                    lev_fix = int(clamp(lev_fix, 1, 125))
                                    lev = int(lev_fix)
                                    ai2["leverage"] = int(lev)
                                    ai2["leverage_source"] = "HIGHRISK_FIXED"
                                    lev_src = "HIGHRISK_FIXED"

                                    pct_total = float(cfg.get("highrisk_fixed_entry_pct_total", 20.0) or 20.0)
                                    pct_total = float(clamp(pct_total, 0.5, 95.0))
                                    try:
                                        # ✅ 요구: 스윙이면 공포/탐욕 지수(FNG)에 따라 총자산 8~15% 진입
                                        if str(style) == "스윙":
                                            pct_fng = swing_entry_pct_total_by_fng(ext, cfg)
                                            if pct_fng is not None:
                                                pct_total = float(pct_fng)
                                                ai2["entry_pct_total_fng"] = float(pct_fng)
                                                ai2["entry_pct_total_source"] = "FNG_SWING"
                                    except Exception:
                                        pass
                                    base_eq = float(total_usdt) if float(total_usdt) > 0 else float(free_usdt)
                                    entry_usdt_target = float(base_eq) * (float(pct_total) / 100.0)
                                    # 스윙: 외부 리스크(이벤트/브리핑) 감산만 반영(공포/탐욕은 pct_total로 이미 반영)
                                    try:
                                        if str(style) == "스윙" and bool(cfg.get("swing_fng_entry_pct_enable", True)):
                                            # 최종 pct_total은 8~15% 범위 내로 고정
                                            pmin = float(cfg.get("swing_fng_entry_pct_min", 8.0) or 8.0)
                                            pmax = float(cfg.get("swing_fng_entry_pct_max", 15.0) or 15.0)
                                            pmin = float(clamp(pmin, 0.5, 95.0))
                                            pmax = float(clamp(pmax, pmin, 95.0))
                                            # 감산 적용 후에도 8~15를 유지(너무 작게 들어가는 문제 방지)
                                            pct_eff = float(clamp(float(pct_total) * float(entry_risk_mul), float(pmin), float(pmax)))
                                            entry_usdt = float(base_eq) * (float(pct_eff) / 100.0)
                                            ai2["entry_pct_total_effective"] = float(pct_eff)
                                        else:
                                            entry_usdt = float(entry_usdt_target) * float(entry_risk_mul)
                                    except Exception:
                                        entry_usdt = float(entry_usdt_target) * float(entry_risk_mul)
                                    # free를 넘으면 주문 실패 → free 내로 제한
                                    entry_usdt = float(min(entry_usdt, float(free_usdt) * 0.99))
                                    ai2["entry_usdt_target_total"] = float(entry_usdt_target)
                                    ai2["entry_usdt"] = float(entry_usdt)
                                    ai2["entry_pct_total"] = float(pct_total)
                                    entry_pct_src = "HIGHRISK_FIXED"
                                    try:
                                        # 표시용: free 대비 %로도 환산(텔레그램/일지에 "(몇%)" 표시용)
                                        entry_pct = float(entry_usdt / float(free_usdt) * 100.0) if float(free_usdt) > 0 else float(entry_pct)
                                        ai2["entry_pct"] = float(entry_pct)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            # ✅ Max Risk Per Trade(요구): 손절(ROI%) 기준으로 1회 최대 손실을 2~3%로 제한
                            try:
                                if bool(cfg.get("max_risk_per_trade_enable", True)):
                                    # forced exit(수익보존) 정책이면 실제 손절은 고정(-15%)이므로 그 기준을 우선 사용
                                    sl_for_risk = float(abs(float(slp)))
                                    try:
                                        if bool(cfg.get("exit_trailing_protect_enable", False)):
                                            sl_forced = float(cfg.get("exit_trailing_protect_sl_roi", 15.0) or 15.0)
                                            sl_for_risk = float(max(sl_for_risk, abs(sl_forced)))
                                    except Exception:
                                        pass
                                    # max loss 계산(퍼센트/USDT 중 더 엄격한 쪽)
                                    base_eq = float(total_usdt) if float(total_usdt) > 0 else float(free_usdt)
                                    # ✅ 모드별 리스크캡(진입금이 너무 낮아지는 문제 완화)
                                    lim_pct_base = float(cfg.get("max_risk_per_trade_pct", 2.5) or 0.0)
                                    lim_pct = float(lim_pct_base)
                                    try:
                                        if str(mode) == "안전모드":
                                            lim_pct = float(cfg.get("max_risk_per_trade_pct_safe", lim_pct_base) or lim_pct_base)
                                        elif str(mode) == "공격모드":
                                            lim_pct = float(cfg.get("max_risk_per_trade_pct_attack", lim_pct_base) or lim_pct_base)
                                        else:
                                            lim_pct = float(cfg.get("max_risk_per_trade_pct_highrisk", lim_pct_base) or lim_pct_base)
                                    except Exception:
                                        lim_pct = float(lim_pct_base)
                                    # SQZ 정방향+강한 모멘텀(하이리스크)일 때는 리스크캡을 약간 완화(진입금↑)
                                    try:
                                        if bool(cfg.get("use_sqz", True)) and bool(cfg.get("sqz_dependency_enable", True)):
                                            bias0 = int(stt.get("_sqz_bias", 0) or 0)
                                            str0 = float(stt.get("_sqz_strength", 0.0) or 0.0)
                                            aligned0 = (bias0 == 1 and str(decision) == "buy") or (bias0 == -1 and str(decision) == "sell")
                                            if aligned0 and float(str0) >= 0.6 and str(mode) == "하이리스크/하이리턴":
                                                lim_pct = float(lim_pct) * (1.0 + min(0.25, 0.15 * float(str0)))
                                    except Exception:
                                        pass
                                    lim_usdt = float(cfg.get("max_risk_per_trade_usdt", 0.0) or 0.0)
                                    max_loss_pct = (base_eq * abs(lim_pct) / 100.0) if lim_pct > 0 else float("inf")
                                    max_loss_abs = abs(lim_usdt) if lim_usdt > 0 else float("inf")
                                    max_loss = float(min(max_loss_pct, max_loss_abs))
                                    if sl_for_risk > 0 and max_loss != float("inf") and entry_usdt > 0:
                                        risk_now = float(entry_usdt * (sl_for_risk / 100.0))
                                        if risk_now > max_loss:
                                            entry_usdt_cap = float(max_loss * 100.0 / sl_for_risk)
                                            entry_usdt = float(min(entry_usdt, entry_usdt_cap))
                                            ai2["entry_usdt_risk_cap"] = float(entry_usdt)
                                            ai2["risk_cap_usdt"] = float(max_loss)
                                            ai2["risk_sl_for_risk"] = float(sl_for_risk)
                            except Exception:
                                pass

                            # entry_pct는 최종 entry_usdt 기준으로 다시 계산(표시/일지 일관성)
                            try:
                                if free_usdt > 0:
                                    entry_pct = float((float(entry_usdt) / float(free_usdt)) * 100.0)
                                    ai2["entry_pct"] = float(entry_pct)
                            except Exception:
                                pass
                            if entry_usdt < 5:
                                cs["skip_reason"] = "잔고 부족(진입금 너무 작음)"
                                continue

                            # margin mode(cross/isolated) + leverage
                            try:
                                set_margin_mode_safe(ex, sym, str(cfg.get("margin_mode", "cross")))
                            except Exception:
                                pass
                            set_leverage_safe(ex, sym, lev)
                            qty = to_precision_qty(ex, sym, (entry_usdt * lev) / max(px, 1e-9))
                            if qty <= 0:
                                cs["skip_reason"] = "수량 계산 실패"
                                continue

                            # ✅ 리스(리더) 확인: watchdog 복구 중 중복 주문 방지
                            try:
                                lease_now = runtime_worker_lease_get()
                                if str(lease_now.get("id", "") or "").strip() and str(lease_now.get("id", "") or "").strip() != str(worker_id):
                                    cs["skip_reason"] = "리스 상실(리더 아님)"
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message="리스 상실(리더 아님) → 주문 스킵",
                                        extra={"leader": lease_now.get("id", ""), "until_kst": lease_now.get("until_kst", "")},
                                    )
                                    continue
                            except Exception:
                                pass

                            ok, err_order = market_order_safe_ex(ex, sym, decision, qty)
                            if not ok:
                                try:
                                    msg0 = f"주문 실패: {str(err_order or '')}".strip()
                                    if len(msg0) > 160:
                                        msg0 = msg0[:160] + "..."
                                    cs["skip_reason"] = msg0
                                    mon_add_scan(
                                        mon,
                                        stage="trade_skipped",
                                        symbol=sym,
                                        tf=str(cfg.get("timeframe", "5m")),
                                        signal=str(decision),
                                        score=conf,
                                        message=msg0,
                                        extra={"qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style},
                                    )
                                    mon_add_event(mon, "ORDER_FAIL", sym, "ENTRY 주문 실패", {"err": str(err_order or ""), "qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style})
                                except Exception:
                                    pass
                                continue
                            else:
                                trade_id = uuid.uuid4().hex[:10]
                                mon_add_scan(
                                    mon,
                                    stage="trade_opened",
                                    symbol=sym,
                                    tf=str(cfg.get("timeframe", "5m")),
                                    signal=str(decision),
                                    score=conf,
                                    message=f"주문 체결, trade_id={trade_id}",
                                    extra={"qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style},
                                )
                                try:
                                    gsheet_log_trade(
                                        stage="ENTRY",
                                        symbol=sym,
                                        trade_id=trade_id,
                                        message=f"{decision} style={style} conf={conf}",
                                        payload={"qty": qty, "entry_usdt": entry_usdt, "lev": lev, "style": style, "tp": tpp, "sl": slp},
                                    )
                                except Exception:
                                    pass

                                # ✅ SL/TP 가격 라인(지지/저항 + AI 후보 + ROI 바운드)
                                sl_price = None
                                tp_price = None
                                sl_price_source = ""
                                tp_price_source = ""
                                sr_used: Dict[str, Any] = {}
                                ai_sl_price = ai2.get("sl_price", None)
                                ai_tp_price = ai2.get("tp_price", None)
                                try:
                                    sl_price_pct = float(ai2.get("sl_price_pct", float(slp) / max(int(lev), 1)))
                                except Exception:
                                    sl_price_pct = float(slp) / max(int(lev), 1)
                                try:
                                    tp_price_pct = float(ai2.get("tp_price_pct", float(tpp) / max(int(lev), 1)))
                                except Exception:
                                    tp_price_pct = float(tpp) / max(int(lev), 1)
                                if cfg.get("use_sr_stop", True):
                                    try:
                                        sr_res = sr_prices_for_style(
                                            ex,
                                            sym,
                                            entry_price=float(px),
                                            side=str(decision),
                                            style=str(style),
                                            cfg=cfg,
                                            sl_price_pct=float(sl_price_pct),
                                            tp_price_pct=float(tp_price_pct),
                                            ai_sl_price=ai_sl_price,
                                            ai_tp_price=ai_tp_price,
                                        )
                                        if isinstance(sr_res, dict):
                                            sr_used = dict(sr_res)
                                            sl_price = sr_res.get("sl_price", None)
                                            tp_price = sr_res.get("tp_price", None)
                                            sl_price_source = str(sr_res.get("sl_source", "") or "")
                                            tp_price_source = str(sr_res.get("tp_source", "") or "")
                                    except Exception:
                                        pass
                                # SR 계산 실패/값 비정상 시에도 최소한 "가격 기준 SL/TP"는 ROI 바운드로 확보
                                if sl_price is None or tp_price is None:
                                    try:
                                        slb, tpb = _sr_price_bounds_from_price_pct(float(px), str(decision), float(sl_price_pct), float(tp_price_pct))
                                        sl_price = float(slb)
                                        tp_price = float(tpb)
                                        if not sl_price_source:
                                            sl_price_source = "ROI"
                                        if not tp_price_source:
                                            tp_price_source = "ROI"
                                    except Exception:
                                        pass

                                # ✅ 목표 ROI를 SR/매물대 기준 가격에서 역산해 동기화
                                # - exit_ai_targets_only=True일 때도 코인/구간별 목표가가 달라지도록 함
                                try:
                                    if bool(cfg.get("exit_ai_targets_sync_from_sr", True)) and (sl_price is not None) and (tp_price is not None):
                                        px0 = float(px)
                                        lev0 = float(max(1, int(lev)))
                                        if px0 > 0 and lev0 > 0:
                                            if str(decision) == "buy":
                                                sl_price_pct_sync = max(0.0, ((px0 - float(sl_price)) / px0) * 100.0)
                                                tp_price_pct_sync = max(0.0, ((float(tp_price) - px0) / px0) * 100.0)
                                            else:
                                                sl_price_pct_sync = max(0.0, ((float(sl_price) - px0) / px0) * 100.0)
                                                tp_price_pct_sync = max(0.0, ((px0 - float(tp_price)) / px0) * 100.0)
                                            if sl_price_pct_sync > 0 and tp_price_pct_sync > 0:
                                                sl_price_pct = float(sl_price_pct_sync)
                                                tp_price_pct = float(tp_price_pct_sync)
                                                slp = float(sl_price_pct * lev0)
                                                tpp = float(tp_price_pct * lev0)
                                                ai2["sl_pct"] = float(slp)
                                                ai2["tp_pct"] = float(tpp)
                                                ai2["sl_price_pct"] = float(sl_price_pct)
                                                ai2["tp_price_pct"] = float(tp_price_pct)
                                                ai2["rr"] = float(float(tpp) / max(abs(float(slp)), 0.01))
                                except Exception:
                                    pass

                                # ✅ 스윙 전용: 첫 진입 시 분할익절 2개 + 추매라인 1개를 SR/매물대 기반으로 지정
                                partial_tp1_price = None
                                partial_tp2_price = None
                                dca_price = None
                                try:
                                    if str(style) == "스윙":
                                        swing_levels = plan_swing_management_levels(
                                            entry_price=float(px),
                                            side=str(decision),
                                            tp_price=(float(tp_price) if tp_price is not None else None),
                                            sl_price=(float(sl_price) if sl_price is not None else None),
                                            supports=list((sr_used or {}).get("supports", []) or []),
                                            resistances=list((sr_used or {}).get("resistances", []) or []),
                                            volume_nodes=list((sr_used or {}).get("volume_nodes", []) or []),
                                        )
                                        partial_tp1_price = swing_levels.get("partial_tp1_price", None)
                                        partial_tp2_price = swing_levels.get("partial_tp2_price", None)
                                        dca_price = swing_levels.get("dca_price", None)
                                except Exception:
                                    partial_tp1_price = None
                                    partial_tp2_price = None
                                    dca_price = None

                                # 목표 저장
                                # ✅ 진입 시점 차트 스냅샷(손절/익절/본절/추매/순환매 근거용)
                                entry_rsi = None
                                entry_adx = None
                                try:
                                    v = last.get("RSI", None) if isinstance(last, pd.Series) else None
                                    entry_rsi = float(v) if (v is not None and pd.notna(v)) else None
                                except Exception:
                                    entry_rsi = None
                                try:
                                    v = last.get("ADX", None) if isinstance(last, pd.Series) else None
                                    entry_adx = float(v) if (v is not None and pd.notna(v)) else None
                                except Exception:
                                    entry_adx = None

                                active_targets[sym] = {
                                    "sl": slp,
                                    "tp": tpp,
                                    "entry_usdt": entry_usdt,
                                    "entry_pct": entry_pct,
                                    "entry_confidence": int(conf),
                                    "entry_prefilter_note": " / ".join(filter_msgs)[:180] if isinstance(filter_msgs, list) and filter_msgs else "",
                                    "entry_tier": "SOFT" if bool(is_soft_entry) else "STRICT",
                                    "lev": lev,
                                    "entry_price": float(px),
                                    "entry_snapshot": {
                                        "time_kst": now_kst_str(),
                                        "price": float(px),
                                        "trend_short": str(stt.get("추세", "") or ""),
                                        "trend_long": str(htf_trend or ""),
                                        "rsi": entry_rsi,
                                        "adx": entry_adx,
                                        "macd": str(stt.get("MACD", "") or ""),
                                    },
                                    # ✅ 잔고 스냅샷(시트/일지에 표시용)
                                    "bal_entry_total": float(total_usdt) if "total_usdt" in locals() else "",
                                    "bal_entry_free": float(free_usdt) if "free_usdt" in locals() else "",
                                    # ✅ 진입 직후 잔고(총/가용) 스냅샷(요구사항: "진입후 잔액")
                                    "bal_entry_after_total": "",
                                    "bal_entry_after_free": "",
                                    "reason": ai2.get("reason_easy", ""),
                                    "trade_id": trade_id,
                                    "sl_price": sl_price,
                                    "tp_price": tp_price,
                                    "sl_price_pct": float(sl_price_pct),
                                    "tp_price_pct": float(tp_price_pct),
                                    "sl_price_source": sl_price_source,
                                    "tp_price_source": tp_price_source,
                                    "sr_used": {"tf": sr_used.get("tf", ""), "lookback": sr_used.get("lookback", 0), "pivot_order": sr_used.get("pivot_order", 0), "buffer_atr_mult": sr_used.get("buffer_atr_mult", 0.0), "rr_min": sr_used.get("rr_min", 0.0)},
                                    "partial_tp1_price": partial_tp1_price,
                                    "partial_tp2_price": partial_tp2_price,
                                    "dca_price": dca_price,
                                    "sl_price_ai": ai_sl_price,
                                    "tp_price_ai": ai_tp_price,
                                    "style": style,
                                    "style_confidence": int(cs.get("style_confidence", 0)),
                                    "style_reason": str(cs.get("style_reason", ""))[:240],
                                    "entry_epoch": time.time(),
                                    "style_last_switch_epoch": time.time(),
                                }

                                # ✅ 진입 직후 잔고(총/가용) 스냅샷 갱신(가능할 때만; 실패해도 봇은 계속)
                                try:
                                    free_a, total_a = safe_fetch_balance(ex)
                                    active_targets[sym]["bal_entry_after_total"] = float(total_a)
                                    active_targets[sym]["bal_entry_after_free"] = float(free_a)
                                except Exception:
                                    pass

                                rt.setdefault("open_targets", {})[sym] = active_targets[sym]
                                save_runtime(rt)
                                try:
                                    active_syms.add(sym)
                                except Exception:
                                    pass

                                # 상세일지
                                save_trade_detail(
                                    trade_id,
                                    {
                                        "trade_id": trade_id,
                                        "time": now_kst_str(),
                                        "coin": sym,
                                        "decision": decision,
                                        "confidence": conf,
                                        "entry_price": px,
                                        "entry_usdt": entry_usdt,
                                        "entry_pct": entry_pct,
                                        "lev": lev,
                                        "balance_before_total": float(total_usdt) if "total_usdt" in locals() else "",
                                        "balance_before_free": float(free_usdt) if "free_usdt" in locals() else "",
                                        "balance_after_total": active_targets.get(sym, {}).get("bal_entry_after_total", ""),
                                        "balance_after_free": active_targets.get(sym, {}).get("bal_entry_after_free", ""),
                                        "sl_pct_roi": slp,
                                        "tp_pct_roi": tpp,
                                        "sl_price_sr": sl_price,
                                        "tp_price_sr": tp_price,
                                        "sl_price_ai": ai_sl_price,
                                        "tp_price_ai": ai_tp_price,
                                        "sl_price_source": sl_price_source,
                                        "tp_price_source": tp_price_source,
                                        "sr_used": {"tf": sr_used.get("tf", ""), "lookback": sr_used.get("lookback", 0), "pivot_order": sr_used.get("pivot_order", 0), "buffer_atr_mult": sr_used.get("buffer_atr_mult", 0.0), "rr_min": sr_used.get("rr_min", 0.0)},
                                        "partial_tp1_price": partial_tp1_price,
                                        "partial_tp2_price": partial_tp2_price,
                                        "dca_price": dca_price,
                                        "used_indicators": ai2.get("used_indicators", []),
                                        "reason_easy": ai2.get("reason_easy", ""),
                                        "raw_status": stt,
                                        "trend_short": stt.get("추세", ""),
                                        "trend_long": f"🧭 {htf_tf} {htf_trend}",
                                        "style": style,
                                        "style_confidence": int(cs.get("style_confidence", 0)),
                                        "style_reason": str(cs.get("style_reason", ""))[:240],
                                        "events": [],
                                        "external_used": (
                                            {
                                                "fear_greed": (ext or {}).get("fear_greed"),
                                                "high_impact_events_soon": ((ext or {}).get("high_impact_events_soon") or [])[:3],
                                                "asof_kst": (ext or {}).get("asof_kst", ""),
                                                "daily_btc_brief": (ext or {}).get("daily_btc_brief", {}),
                                            }
                                            if str(style) == "스윙"
                                            else {"enabled": False}
                                        ),
                                    },
                                )

                                # 쿨다운
                                rt.setdefault("cooldowns", {})[sym] = time.time() + 60
                                save_runtime(rt)

                                # 텔레그램 보고
                                if cfg.get("tg_enable_reports", True):
                                    direction = "롱(상승에 베팅)" if decision == "buy" else "숏(하락에 베팅)"
                                    try:
                                        rr0 = float(ai2.get("rr", 0.0) or 0.0)
                                    except Exception:
                                        rr0 = 0.0
                                    if rr0 <= 0:
                                        try:
                                            rr0 = float(tpp) / max(abs(float(slp)), 0.01)
                                        except Exception:
                                            rr0 = 0.0
                                    if _tg_simple_enabled(cfg):
                                        # ✅ 쉬운말(핵심만)
                                        bb_total = None
                                        bb_free = None
                                        ba_total = None
                                        ba_free = None
                                        try:
                                            bb_total = float(total_usdt) if "total_usdt" in locals() else None
                                        except Exception:
                                            bb_total = None
                                        try:
                                            bb_free = float(free_usdt) if "free_usdt" in locals() else None
                                        except Exception:
                                            bb_free = None
                                        try:
                                            v = (active_targets.get(sym, {}) or {}).get("bal_entry_after_total", "")
                                            ba_total = float(v) if (v is not None and str(v).strip() != "") else None
                                        except Exception:
                                            ba_total = None
                                        try:
                                            v = (active_targets.get(sym, {}) or {}).get("bal_entry_after_free", "")
                                            ba_free = float(v) if (v is not None and str(v).strip() != "") else None
                                        except Exception:
                                            ba_free = None
                                        # ✅ 텔레그램 '한줄'에는:
                                        # - AI 진입 근거(차트/지표/SR)
                                        # - 왜 스캘핑/스윙인지(스타일 이유)
                                        # 를 2줄로 함께 보여준다(가독성: 인용 2줄까지만 표시).
                                        ai_reason = str(ai2.get("reason_easy", "") or "").strip()
                                        style_reason = str(cs.get("style_reason", "") or "").strip()
                                        # ✅ 직전 청산 정보(본절/손절/익절)도 함께 표시: "왜 또 들어갔는지" 이해하기 쉽게
                                        last_exit_note = ""
                                        try:
                                            le_map = rt.get("last_exit", {}) or {}
                                            le = le_map.get(sym) if isinstance(le_map, dict) else None
                                            if isinstance(le, dict):
                                                try:
                                                    le_ep = float(le.get("epoch", 0) or 0.0)
                                                except Exception:
                                                    le_ep = 0.0
                                                max_age = float(_timeframe_seconds(str(cfg.get("timeframe", "5m") or "5m"), 300)) * 20.0
                                                if le_ep > 0 and (time.time() - le_ep) <= max_age:
                                                    le_type = str(le.get("type", "") or "").strip().upper()
                                                    type_ko = {"PROTECT": "본절", "SL": "손절", "TP": "익절"}.get(le_type, le_type or "")
                                                    tm = str(le.get("time_kst", "") or "").strip()
                                                    tm_hm = tm[11:16] if len(tm) >= 16 else tm
                                                    le_roi = None
                                                    try:
                                                        v = le.get("roi", None)
                                                        le_roi = float(v) if v is not None else None
                                                    except Exception:
                                                        le_roi = None
                                                    if type_ko:
                                                        if le_roi is not None:
                                                            last_exit_note = f"직전:{type_ko}({_tg_fmt_pct(le_roi)}) {tm_hm}".strip()
                                                        else:
                                                            last_exit_note = f"직전:{type_ko} {tm_hm}".strip()
                                        except Exception:
                                            last_exit_note = ""

                                        one_line_parts: List[str] = []
                                        if ai_reason:
                                            one_line_parts.append(ai_reason)
                                        line2_parts: List[str] = []
                                        if style_reason and (style_reason not in ai_reason):
                                            line2_parts.append(style_reason)
                                        if last_exit_note:
                                            line2_parts.append(last_exit_note)
                                        if line2_parts:
                                            one_line_parts.append(" | ".join(line2_parts)[:220])
                                        one_line0 = "\n".join(one_line_parts).strip()
                                        if not one_line0:
                                            one_line0 = "-"
                                        msg = tg_msg_entry_simple(
                                            symbol=str(sym),
                                            style=str(style),
                                            decision=str(decision),
                                            lev=lev,
                                            entry_usdt=float(entry_usdt),
                                            entry_pct_plan=float(entry_pct) if entry_pct is not None else None,
                                            tp_pct_roi=float(tpp) if tpp is not None else None,
                                            sl_pct_roi=float(slp) if slp is not None else None,
                                            bal_before_total=bb_total,
                                            bal_after_total=ba_total,
                                            bal_before_free=bb_free,
                                            bal_after_free=ba_free,
                                            one_line=one_line0,
                                            trade_id=str(trade_id),
                                        )
                                        try:
                                            if str(style) == "스윙":
                                                t0 = active_targets.get(sym, {}) if isinstance(active_targets.get(sym, {}), dict) else {}
                                                p1 = _as_float(t0.get("partial_tp1_price", None), float("nan"))
                                                p2 = _as_float(t0.get("partial_tp2_price", None), float("nan"))
                                                dp = _as_float(t0.get("dca_price", None), float("nan"))
                                                lines_extra: List[str] = []
                                                if math.isfinite(p1) or math.isfinite(p2):
                                                    s1 = f"{p1:.6g}" if math.isfinite(p1) else "-"
                                                    s2 = f"{p2:.6g}" if math.isfinite(p2) else "-"
                                                    lines_extra.append(f"- 분할익절 라인: 1차 {s1} / 2차 {s2}")
                                                if math.isfinite(dp):
                                                    lines_extra.append(f"- 추매 라인: {dp:.6g}")
                                                if lines_extra:
                                                    msg += "\n" + "\n".join(lines_extra)
                                        except Exception:
                                            pass
                                    else:
                                        # ✅ 기존(상세) 메시지 유지
                                        try:
                                            sl_price_pct0 = float(ai2.get("sl_price_pct", float(slp) / max(int(lev), 1)) or 0.0)
                                        except Exception:
                                            sl_price_pct0 = float(slp) / max(int(lev), 1)
                                        try:
                                            tp_price_pct0 = float(ai2.get("tp_price_pct", float(tpp) / max(int(lev), 1)) or 0.0)
                                        except Exception:
                                            tp_price_pct0 = float(tpp) / max(int(lev), 1)
                                        msg = (
                                            f"🎯 진입\n- 코인: {sym}\n- 스타일: {style}\n- 방향: {direction}\n"
                                            f"- 스타일이유: {str(cs.get('style_reason','') or '').strip()[:180]}\n"
                                            f"- 진입금: {entry_usdt:.2f} USDT (잔고 {entry_pct:.1f}%)\n"
                                            f"- 레버리지: x{lev}\n"
                                            f"- 목표손익비(익절/손절): 익절 +{tpp:.2f}% / 손절 -{slp:.2f}% | (참고) RR {rr0:.2f}\n"
                                            f"- 가격기준(TP/SL): +{tp_price_pct0:.2f}% / -{sl_price_pct0:.2f}%\n"
                                            f"- 단기추세({cfg.get('timeframe','5m')}): {stt.get('추세','-')}\n"
                                            f"- 장기추세({htf_tf}): 🧭 {htf_trend}\n"
                                            f"- 외부리스크 감산: x{entry_risk_mul:.2f} ({'스윙만 적용' if str(style)=='스윙' else '스캘핑=미적용'})\n"
                                        )
                                        if sl_price is not None and tp_price is not None:
                                            src_txt = ""
                                            try:
                                                src_txt = f" ({sl_price_source or '-'} / {tp_price_source or '-'})"
                                            except Exception:
                                                src_txt = ""
                                            msg += f"- SR기준가: TP {tp_price:.6g} / SL {sl_price:.6g}{src_txt}\n"
                                        try:
                                            if str(style) == "스윙":
                                                p1 = _as_float((active_targets.get(sym, {}) or {}).get("partial_tp1_price", None), float("nan"))
                                                p2 = _as_float((active_targets.get(sym, {}) or {}).get("partial_tp2_price", None), float("nan"))
                                                dp = _as_float((active_targets.get(sym, {}) or {}).get("dca_price", None), float("nan"))
                                                if math.isfinite(p1) or math.isfinite(p2):
                                                    s1 = f"{p1:.6g}" if math.isfinite(p1) else "-"
                                                    s2 = f"{p2:.6g}" if math.isfinite(p2) else "-"
                                                    msg += f"- 분할익절 라인: 1차 {s1} / 2차 {s2}\n"
                                                if math.isfinite(dp):
                                                    msg += f"- 추매 라인: {dp:.6g}\n"
                                        except Exception:
                                            pass
                                        msg += f"- 확신도: {conf}% (기준 {rule['min_conf']}%)\n- 일지ID: {trade_id}\n"
                                        if cfg.get("tg_send_entry_reason", False):
                                            # 요구사항: 텔레그램에는 '긴 근거'를 보내지 않고, /log <id>로 조회
                                            msg += (
                                                f"- 근거(짧게): {str(ai2.get('reason_easy',''))[:120]}\n"
                                                f"- 자세한 근거: /log {trade_id}\n"
                                                f"- AI지표: {', '.join(ai2.get('used_indicators', []))}\n"
                                            )
                                    tg_send(msg, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                    # ✅ 채널을 음소거해도 중요한 알림을 놓치지 않도록(사용자 요구): 관리자 DM에도 복사
                                    try:
                                        if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                            tg_send(msg, target="admin", cfg=cfg, silent=False)
                                    except Exception:
                                        pass
                                    # ✅ 진입 근거 이미지(캔들 + SR/매물대 + 지표/패턴 요약)
                                    try:
                                        if bool(cfg.get("tg_send_trade_images", True)) and bool(cfg.get("tg_send_entry_image", True)):
                                            aft = active_targets.get(sym, {}) if isinstance(active_targets.get(sym, {}), dict) else {}
                                            afree = _as_float(aft.get("bal_entry_after_free", None), float("nan"))
                                            atotal = _as_float(aft.get("bal_entry_after_total", None), float("nan"))
                                            if not math.isfinite(float(afree)):
                                                afree = None
                                            if not math.isfinite(float(atotal)):
                                                atotal = None
                                            one_img = str(locals().get("one_line0", "") or ai2.get("reason_easy", "") or "").strip()
                                            img_path = build_trade_event_image(
                                                ex,
                                                sym,
                                                cfg,
                                                event_type="ENTRY",
                                                side=str(decision),
                                                style=str(style),
                                                entry_price=float(px),
                                                sl_price=(float(sl_price) if sl_price is not None else None),
                                                tp_price=(float(tp_price) if tp_price is not None else None),
                                                partial_tp1_price=(float(aft.get("partial_tp1_price")) if aft.get("partial_tp1_price") is not None else None),
                                                partial_tp2_price=(float(aft.get("partial_tp2_price")) if aft.get("partial_tp2_price") is not None else None),
                                                dca_price=(float(aft.get("dca_price")) if aft.get("dca_price") is not None else None),
                                                sl_roi_pct=float(abs(slp)) if (slp is not None) else None,
                                                tp_roi_pct=float(abs(tpp)) if (tpp is not None) else None,
                                                leverage=float(lev) if (lev is not None) else None,
                                                remain_free=(float(afree) if afree is not None else None),
                                                remain_total=(float(atotal) if atotal is not None else None),
                                                one_line=one_img,
                                                used_indicators=list(ai2.get("used_indicators", []) or []),
                                                pattern_hint=str(stt.get("패턴", "") or ""),
                                                mtf_pattern=(stt.get("_pattern_mtf", {}) if isinstance(stt.get("_pattern_mtf", {}), dict) else {}),
                                                trade_id=str(trade_id),
                                            )
                                            if img_path:
                                                cap = (
                                                    f"📷 진입 차트\n"
                                                    f"- {sym} | {style} | {_tg_dir_easy(decision)}\n"
                                                    f"- 목표: 익절 {_tg_fmt_target_roi(tpp, sign='+', min_visible=0.05)} / 손절 {_tg_fmt_target_roi(slp, sign='-', min_visible=0.05)}"
                                                )
                                                tg_send_photo(img_path, caption=cap, target=cfg.get("tg_route_events_to", "channel"), cfg=cfg, silent=False)
                                                if bool(cfg.get("tg_trade_alert_to_admin", True)) and tg_admin_chat_ids():
                                                    tg_send_photo(img_path, caption=cap, target="admin", cfg=cfg, silent=False)
                                                if trade_id:
                                                    d0 = load_trade_detail(str(trade_id)) or {}
                                                    d0["entry_chart_image"] = str(img_path)
                                                    save_trade_detail(str(trade_id), d0)
                                    except Exception:
                                        pass

                                mon_add_event(mon, "ENTRY", sym, f"{decision} {style} conf{conf}", {"trade_id": trade_id})
                                monitor_write_throttled(mon, 0.2)
                                time.sleep(1.0)

                        else:
                            # AI 결과가 HOLD이거나, 확신도/조건 미달로 진입하지 않음
                            mon_add_scan(
                                mon,
                                stage="trade_skipped",
                                symbol=sym,
                                tf=str(cfg.get("timeframe", "5m")),
                                signal=str(decision),
                                score=conf,
                                message="진입 조건 미달/보류",
                                extra={"decision": decision, "confidence": conf, "min_conf": int(rule.get("min_conf", 0))},
                            )

                        time.sleep(0.4)

                # 스캔 사이클 시간(멈춤 감지/표시용)
                try:
                    if "scan_cycle_start" in locals():
                        mon["scan_cycle_sec"] = float(time.time() - float(scan_cycle_start))
                        mon["last_scan_cycle_kst"] = now_kst_str()
                except Exception:
                    pass

                # ✅ 스캔 도중 CCXT timeout이 발생하면 exchange 인스턴스를 교체(정체/동시호출 꼬임 방지)
                try:
                    _to_after_scan = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                except Exception:
                    _to_after_scan = 0.0
                if locals().get("need_exchange_refresh") or (_to_after_scan and _to_after_scan > float(ccxt_timeout_epoch_loop_start or 0)):
                    try:
                        where_now = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "").strip()
                        mon_add_event(mon, "CCXT_REFRESH", "", "exchange refreshed(after scan timeout)", {"where": where_now, "code": CODE_VERSION})
                        ex_new = create_exchange_client_uncached()
                        if ex_new is not None:
                            ex = ex_new
                            # loop-start 마커 갱신(새 인스턴스 기준)
                            ccxt_timeout_epoch_loop_start = float(getattr(ex, "_wonyoti_ccxt_timeout_epoch", 0) or 0)
                            ccxt_timeout_where_loop_start = str(getattr(ex, "_wonyoti_ccxt_timeout_where", "") or "")
                    except Exception:
                        pass

                # ✅ 강제스캔 결과 전송 및 요청 해제(1회)
                if force_scan_pending and force_scan_id:
                    try:
                        lines = [f"🔎 강제스캔 결과: {force_scan_id}", f"- 시각(KST): {now_kst_str()}"]
                        if force_scan_summary_lines:
                            lines += force_scan_summary_lines[:12]
                        else:
                            lines.append("- (수집된 결과 없음)")
                        # ✅ 요구: TG_TARGET_CHAT_ID는 채널(브로드캐스트), 관리/버튼/강제스캔 결과는 관리자 DM으로
                        try:
                            force_by = int(force_scan_req.get("requested_by", 0) or 0)
                        except Exception:
                            force_by = 0
                        if TG_ADMIN_IDS and force_by:
                            tg_send_chat(force_by, "\n".join(lines))
                        elif TG_ADMIN_IDS:
                            tg_send("\n".join(lines), target="admin", cfg=cfg)
                        else:
                            tg_send("\n".join(lines), target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)
                        mon_add_event(mon, "SCAN_DONE", "", f"id={force_scan_id}", {"symbols": list(force_scan_syms_set)[:50], "scan_only": force_scan_only})
                        gsheet_log_event("SCAN_DONE", message=f"id={force_scan_id}", payload={"symbols": list(force_scan_syms_set)[:50], "scan_only": force_scan_only})
                    except Exception:
                        pass
                    try:
                        rt["force_scan"] = {}
                        save_runtime(rt)
                    except Exception:
                        pass

            # 텔레그램 수신 처리(요구사항: long polling 스레드(getUpdates) -> 큐 처리)
            updates = tg_updates_pop_all(max_items=80)
            for up in updates:
                try:
                    # 텍스트 명령
                    if "message" in up and "text" in (up.get("message") or {}):
                        msg0 = up.get("message") or {}
                        txt = str(msg0.get("text") or "").strip()
                        chat_id = ((msg0.get("chat") or {}) if isinstance(msg0.get("chat"), dict) else {}).get("id", None)
                        from0 = msg0.get("from") or {}
                        uid = from0.get("id", None)
                        is_admin = tg_is_admin(uid)

                        def _reply_to_chat(m: str):
                            # /status처럼 누구나 허용되는 응답은 "요청이 온 채팅"으로 답장
                            if chat_id is not None:
                                tg_send_chat(chat_id, m)
                            else:
                                tg_send(m, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)

                        def _reply_admin_dm(m: str):
                            # ✅ 요구: "관리자가 봇을 작동하면, 답변은 채널로"
                            # - 설정: tg_admin_replies_to = channel|admin|both
                            how = str(cfg.get("tg_admin_replies_to", "channel") or "channel").lower().strip()
                            if how == "channel":
                                tg_send(m, target="channel", cfg=cfg)
                                return
                            if how == "both":
                                tg_send(m, target="channel", cfg=cfg)
                                # fallthrough to admin DM
                            # admin DM
                            if TG_ADMIN_IDS:
                                if uid is not None:
                                    tg_send_chat(uid, m)
                                else:
                                    tg_send(m, target="admin", cfg=cfg)
                            else:
                                _reply_to_chat(m)

                        def _deny():
                            _reply_to_chat("⛔️ 관리자만 사용할 수 있는 명령입니다.")

                        low = txt.lower().strip()

                        # /menu (관리자) - TG_ADMIN_USER_IDS 설정 시, /status 외에는 관리자만 허용
                        if low.startswith("/menu") or low in ["menu", "메뉴"]:
                            if TG_ADMIN_IDS and not is_admin:
                                _deny()
                            else:
                                tg_send_menu(cfg=cfg)

                        # /status (누구나 허용)
                        elif low.startswith("/status") or txt == "상태":
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt2 = load_runtime()
                            mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                            regime_mode = str(cfg_live.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            h = openai_health_info(cfg_live)
                            ai_txt = "OK" if bool(h.get("available", False)) else str(h.get("message", "OFF"))
                            until = str(h.get("until_kst", "")).strip()
                            if until and (not bool(h.get("available", False))):
                                ai_txt = f"{ai_txt} (~{until} KST)"
                            msg = (
                                f"📡 상태\n- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 레짐: {regime_txt}\n"
                                f"- OpenAI: {ai_txt}\n"
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})\n"
                                f"- 연속손실: {rt2.get('consec_losses',0)}\n"
                                f"- 정지해제: {('정지중' if time.time() < float(rt2.get('pause_until',0)) else '정상')}\n"
                                f"- 마지막 스캔: {mon_now.get('last_scan_kst','-')}\n"
                                f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}\n"
                            )
                            _reply_to_chat(msg)

                        # /gsheet (관리자) - 상태/연결 테스트
                        elif low.startswith("/gsheet") or txt in ["시트", "구글시트", "gsheet"]:
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                do_test = False
                                do_format = False
                                if len(parts) >= 2:
                                    arg = str(parts[1]).lower().strip()
                                    do_test = arg in ["test", "t", "ping", "check"]
                                    do_format = arg in ["format", "fmt", "style", "서식", "포맷"]
                                if do_test:
                                    res = gsheet_test_append_row(timeout_sec=25)
                                    if res.get("ok"):
                                        _reply_admin_dm("✅ Google Sheets TEST 성공(GSHEET_TEST)")
                                    else:
                                        _reply_admin_dm(f"❌ Google Sheets TEST 실패: {res.get('error','')}")
                                if do_format:
                                    res = gsheet_apply_trades_only_format(force=True, timeout_sec=35)
                                    if res.get("ok"):
                                        _reply_admin_dm(f"✅ Google Sheets 서식 적용 완료(requests={res.get('requests','')})")
                                    else:
                                        _reply_admin_dm(f"❌ Google Sheets 서식 적용 실패: {res.get('error','')}")
                                stg = gsheet_status_snapshot()
                                msg = (
                                    "📎 Google Sheets 상태\n"
                                    f"- mode: {stg.get('mode','')}\n"
                                    f"- enabled: {stg.get('enabled')}\n"
                                    f"- connected: {stg.get('connected')}\n"
                                    f"- spreadsheet_id: {stg.get('spreadsheet_id')}\n"
                                    f"- worksheet: {stg.get('worksheet')}\n"
                                    f"- trade_sheet: {stg.get('trade_sheet','')}\n"
                                    f"- hourly_sheet: {stg.get('hourly_sheet','')}\n"
                                    f"- daily_sheet: {stg.get('daily_sheet','')}\n"
                                    f"- calendar_sheet: {stg.get('calendar_sheet','')}\n"
                                    f"- reviews_sheet: {stg.get('reviews_sheet','')}\n"
                                    f"- service_account_email: {stg.get('service_account_email')}\n"
                                    f"- synced_trade_ids: {stg.get('synced_trade_ids','')}\n"
                                    f"- synced_review_ids: {stg.get('synced_review_ids','')}\n"
                                    f"- last_trade_sync: {stg.get('last_trade_sync_kst','')}\n"
                                    f"- last_summary_sync: {stg.get('last_summary_sync_kst','')}\n"
                                    f"- last_review_sync: {stg.get('last_review_sync_kst','')}\n"
                                    f"- queue_high/scan(legacy): {stg.get('queue_high')}/{stg.get('queue_scan')}\n"
                                    f"- last_append: {stg.get('last_append_kst')} ({stg.get('last_append_type')}:{stg.get('last_append_stage')})\n"
                                    f"- last_err: {stg.get('last_err')}\n"
                                    "사용법: /gsheet test | /gsheet format"
                                )
                                _reply_admin_dm(msg[:3500])

                        # /positions (관리자)
                        elif low.startswith("/positions") or txt == "포지션":
                            if not is_admin:
                                _deny()
                            else:
                                blocks: List[str] = []
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    _reply_admin_dm("📊 포지션\n\n- ⚪ 없음(관망)")
                                else:
                                    rt_open_targets = {}
                                    try:
                                        rt_open_targets = (rt.get("open_targets", {}) or {}) if isinstance(rt, dict) else {}
                                    except Exception:
                                        rt_open_targets = {}
                                    for p in act:
                                        sym = p.get("symbol", "")
                                        side = position_side_normalize(p)
                                        roi = float(position_roi_percent(p))
                                        upnl = float(p.get("unrealizedPnl") or 0.0)
                                        lev = p.get("leverage", "?")
                                        tgt0 = _resolve_open_target_for_symbol(sym, active_targets, rt_open_targets)
                                        style = str((tgt0 or {}).get("style", ""))
                                        blocks.append(_fmt_pos_block(sym, side, lev, roi, upnl, style=style, tgt=tgt0))
                                    _reply_admin_dm("📊 포지션\n\n" + "\n\n".join(blocks))
                                    try:
                                        tg_send_position_chart_images(
                                            ex,
                                            act,
                                            active_targets,
                                            rt_open_targets,
                                            cfg,
                                            route_mode=str(cfg.get("tg_admin_replies_to", "channel") or "channel"),
                                            admin_uid=(int(uid) if uid is not None else None),
                                            fallback_chat_id=(int(chat_id) if chat_id is not None else None),
                                        )
                                    except Exception:
                                        pass

                        # /scan (관리자) - 강제스캔(스캔만, 주문X)
                        elif low.startswith("/scan") or txt == "스캔":
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                sym_arg = parts[1].strip().upper() if len(parts) >= 2 else ""
                                # 심볼 필터(간단): "BTC" 또는 "BTC/USDT:USDT" 형태 지원
                                syms = list(TARGET_COINS)
                                if sym_arg:
                                    if "/" in sym_arg:
                                        syms = [s for s in TARGET_COINS if s.upper().startswith(sym_arg)]
                                    else:
                                        syms = [s for s in TARGET_COINS if s.upper().startswith(f"{sym_arg}/")]
                                if not syms:
                                    _reply_admin_dm("대상 심볼이 없습니다. 예) /scan BTC 또는 /scan BTC/USDT:USDT")
                                else:
                                    rid = uuid.uuid4().hex[:8]
                                    rt2 = load_runtime()
                                    rt2["force_scan"] = {
                                        "id": rid,
                                        "requested_at_epoch": time.time(),
                                        "requested_at_kst": now_kst_str(),
                                        "requested_by": int(uid or 0),
                                        "symbols": syms,
                                        "scan_only": True,  # 안전: 강제스캔은 기본 주문X
                                        "done": False,
                                    }
                                    save_runtime(rt2)
                                    try:
                                        mon_add_event(mon, "SCAN_REQUEST", "", f"force_scan id={rid}", {"symbols": syms, "by": uid})
                                        gsheet_log_event("SCAN_REQUEST", message=f"id={rid}", payload={"symbols": syms, "by": uid})
                                    except Exception:
                                        pass
                                    _reply_admin_dm(f"🔎 강제스캔 요청 완료: {rid}\n- 대상: {', '.join(syms)}\n- 주의: 강제스캔은 '스캔만' 수행(주문X)")

                        # /mode auto|scalping|swing (관리자)
                        elif low.startswith("/mode") or low.startswith("모드"):
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) < 2:
                                    _reply_admin_dm("사용법: /mode auto|scalping|swing")
                                else:
                                    arg = str(parts[1]).lower().strip()
                                    if arg in ["auto", "a"]:
                                        m = "auto"
                                    elif arg in ["scalping", "scalp", "short", "s"]:
                                        m = "scalping"
                                    elif arg in ["swing", "long", "l"]:
                                        m = "swing"
                                    else:
                                        m = ""
                                    if not m:
                                        _reply_admin_dm("사용법: /mode auto|scalping|swing")
                                    else:
                                        cfg2 = load_settings()
                                        cfg2["regime_mode"] = m
                                        save_settings(cfg2)
                                        try:
                                            mon_add_event(mon, "MODE_CHANGE", "", f"regime_mode={m}", {"by": uid})
                                            gsheet_log_event("MODE_CHANGE", message=f"regime_mode={m}", payload={"by": uid})
                                        except Exception:
                                            pass
                                        _reply_admin_dm(f"✅ 레짐 변경: {m}")

                        # /vision (관리자)
                        elif low.startswith("/vision") or txt == "시야":
                            if not is_admin:
                                _deny()
                            else:
                                mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                                coins = mon_now.get("coins", {}) or {}
                                lines = [
                                    "👁️ AI 시야(요약)",
                                    f"- 자동매매: {'ON' if mon_now.get('auto_trade') else 'OFF'}",
                                    f"- 모드: {mon_now.get('trade_mode','-')}",
                                    f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}",
                                ]
                                for sym, cs in list(coins.items())[:10]:
                                    style = str(cs.get("style_reco", "")) or ""
                                    stxt = f"[{style}] " if style else ""
                                    lines.append(
                                        f"- {sym}: {stxt}{str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                        f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                        f"/ 패턴 {str(cs.get('pattern','-'))[:18]} "
                                        f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:30]}"
                                    )
                                _reply_admin_dm("\n".join(lines))

                        # /log 또는 /log <id> (관리자)
                        elif low.startswith("/log") or txt == "일지":
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) >= 2 and parts[1].strip():
                                    tid = parts[1].strip()
                                    d = load_trade_detail(tid)
                                    if not d:
                                        _reply_admin_dm("해당 ID를 찾지 못했어요.")
                                    else:
                                        evs = d.get("events", []) or []
                                        ev_short = []
                                        for e in evs[-6:]:
                                            try:
                                                ev_short.append(f"- {e.get('time','')} {e.get('type','')}: {str(e)[:60]}")
                                            except Exception:
                                                continue
                                        msg = (
                                            f"🧾 /log {tid}\n"
                                            f"- 코인: {d.get('coin')}\n"
                                            f"- 스타일: {d.get('style','-')} ({d.get('style_confidence','-')}%)\n"
                                            f"- 방향: {d.get('decision')}\n"
                                            f"- 확신도: {d.get('confidence')}\n"
                                            f"- 진입: {d.get('time','-')} @ {d.get('entry_price')}\n"
                                            f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                            f"- 레버: x{d.get('lev')}\n"
                                            f"- TP/SL(ROI): +{d.get('tp_pct_roi')}% / -{d.get('sl_pct_roi')}%\n"
                                            f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                            f"- 한줄근거: {str(d.get('reason_easy',''))[:800]}\n"
                                        )
                                        if d.get("exit_time"):
                                            msg += (
                                                f"- 청산: {d.get('exit_time')} @ {d.get('exit_price')}\n"
                                                f"- 결과: {d.get('result','-')} | PnL {float(d.get('pnl_usdt',0)):.2f} USDT | ROI {float(d.get('pnl_pct',0)):.2f}%\n"
                                            )
                                        if ev_short:
                                            msg += "최근 이벤트:\n" + "\n".join(ev_short)
                                        # 텔레그램 길이 제한 보호
                                        _reply_admin_dm(msg[:3500])
                                else:
                                    df_log = read_trade_log()
                                    if df_log.empty:
                                        _reply_admin_dm("📜 일지 없음(아직 기록된 매매가 없어요)")
                                    else:
                                        top = df_log.head(8)
                                        msg = ["📜 최근 매매일지(요약)"]
                                        for _, r in top.iterrows():
                                            tid = str(r.get("TradeID", "") or "")
                                            pnl = float(r.get("PnL_Percent", 0) or 0)
                                            emo = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                                            msg.append(
                                                f"- {emo} {r['Time']} {r['Coin']} {r['Side']} {pnl:.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}"
                                            )
                                        _reply_admin_dm("\n".join(msg))

                        # (호환) 일지상세 /detail (관리자)
                        elif txt.startswith("일지상세") or low.startswith("/detail"):
                            if not is_admin:
                                _deny()
                            else:
                                parts = txt.split()
                                if len(parts) < 2:
                                    _reply_admin_dm("사용법: 일지상세 <ID>\n(예: 일지상세 a1b2c3d4e5)")
                                else:
                                    tid = parts[1].strip()
                                    d = load_trade_detail(tid)
                                    if not d:
                                        _reply_admin_dm("해당 ID를 찾지 못했어요.")
                                    else:
                                        _reply_admin_dm(
                                            (
                                                f"🧾 일지상세 {tid}\n"
                                                f"- 코인: {d.get('coin')}\n"
                                                f"- 스타일: {d.get('style','-')} ({d.get('style_confidence','-')}%)\n"
                                                f"- 방향: {d.get('decision')}\n"
                                                f"- 확신도: {d.get('confidence')}\n"
                                                f"- 진입가: {d.get('entry_price')}\n"
                                                f"- 진입금: {float(d.get('entry_usdt',0)):.2f} USDT (잔고 {float(d.get('entry_pct',0)):.1f}%)\n"
                                                f"- 레버: x{d.get('lev')}\n"
                                                f"- 단기추세: {d.get('trend_short','-')}\n"
                                                f"- 장기추세: {d.get('trend_long','-')}\n"
                                                f"- SR TP/SL: {d.get('tp_price_sr')} / {d.get('sl_price_sr')}\n"
                                                f"- 한줄근거: {str(d.get('reason_easy',''))[:200]}\n"
                                                f"- 사용지표: {', '.join(d.get('used_indicators', []))[:200]}\n"
                                            )[:3500]
                                        )

                    # 콜백 버튼
                    if "callback_query" in up:
                        cb = up.get("callback_query") or {}
                        data = str(cb.get("data", "") or "")
                        cb_id = str(cb.get("id", "") or "")
                        uid = (cb.get("from") or {}).get("id", None)
                        is_admin = tg_is_admin(uid)
                        cb_chat_id = (((cb.get("message") or {}).get("chat") or {}) if isinstance((cb.get("message") or {}).get("chat"), dict) else {}).get("id", None)

                        def _cb_reply(m: str):
                            # ✅ 요구: "관리자가 봇을 작동하면, 답변은 채널로"
                            how = str(cfg.get("tg_admin_replies_to", "channel") or "channel").lower().strip()
                            if how == "channel":
                                tg_send(m, target="channel", cfg=cfg)
                                return
                            if how == "both":
                                tg_send(m, target="channel", cfg=cfg)
                                # fallthrough to admin DM
                            # admin DM
                            if TG_ADMIN_IDS:
                                if uid is not None:
                                    tg_send_chat(uid, m)
                                else:
                                    tg_send(m, target="admin", cfg=cfg)
                            else:
                                # fallback: 버튼이 있던 채팅으로 답장
                                if cb_chat_id is not None:
                                    tg_send_chat(cb_chat_id, m)
                                else:
                                    tg_send(m, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg)

                        def _cb_reply_kb(m: str, kb_obj: Dict[str, Any]):
                            try:
                                markup = json.dumps(kb_obj, ensure_ascii=False)
                            except Exception:
                                _cb_reply(m)
                                return
                            how = str(cfg.get("tg_admin_replies_to", "channel") or "channel").lower().strip()

                            def _send_to_chat(chat_id_val: Any):
                                try:
                                    if chat_id_val is None:
                                        return
                                    cid = str(chat_id_val).strip()
                                    if not cid:
                                        return
                                    tg_enqueue("sendMessage", {"chat_id": cid, "text": m, "reply_markup": markup}, priority="high")
                                except Exception:
                                    pass

                            if how == "channel":
                                ids = _tg_chat_id_by_target("channel", cfg)
                                if ids:
                                    for cid in ids:
                                        _send_to_chat(cid)
                                else:
                                    _cb_reply(m)
                                return

                            if how == "both":
                                ids = _tg_chat_id_by_target("channel", cfg)
                                for cid in ids:
                                    _send_to_chat(cid)
                                if uid is not None:
                                    _send_to_chat(uid)
                                elif TG_ADMIN_IDS:
                                    for cid in tg_admin_chat_ids():
                                        _send_to_chat(cid)
                                elif cb_chat_id is not None:
                                    _send_to_chat(cb_chat_id)
                                return

                            if uid is not None:
                                _send_to_chat(uid)
                            elif TG_ADMIN_IDS:
                                for cid in tg_admin_chat_ids():
                                    _send_to_chat(cid)
                            elif cb_chat_id is not None:
                                _send_to_chat(cb_chat_id)
                            else:
                                ids = _tg_chat_id_by_target(cfg.get("tg_route_queries_to", "group"), cfg)
                                for cid in ids:
                                    _send_to_chat(cid)

                        def _cb_send_photo(path: str, caption: str = ""):
                            p = str(path or "").strip()
                            if (not p) or (not os.path.exists(p)):
                                return
                            how = str(cfg.get("tg_admin_replies_to", "channel") or "channel").lower().strip()
                            if how == "channel":
                                tg_send_photo(p, caption=caption, target="channel", cfg=cfg, silent=False)
                                return
                            if how == "both":
                                tg_send_photo(p, caption=caption, target="channel", cfg=cfg, silent=False)
                                if uid is not None:
                                    tg_send_photo_chat(uid, p, caption=caption, silent=False)
                                elif TG_ADMIN_IDS:
                                    tg_send_photo(p, caption=caption, target="admin", cfg=cfg, silent=False)
                                elif cb_chat_id is not None:
                                    tg_send_photo_chat(cb_chat_id, p, caption=caption, silent=False)
                                return
                            if uid is not None:
                                tg_send_photo_chat(uid, p, caption=caption, silent=False)
                            elif TG_ADMIN_IDS:
                                tg_send_photo(p, caption=caption, target="admin", cfg=cfg, silent=False)
                            elif cb_chat_id is not None:
                                tg_send_photo_chat(cb_chat_id, p, caption=caption, silent=False)
                            else:
                                tg_send_photo(p, caption=caption, target=cfg.get("tg_route_queries_to", "group"), cfg=cfg, silent=False)

                        if data == "status":
                            # 누구나
                            cfg_live = load_settings()
                            free, total = safe_fetch_balance(ex)
                            rt2 = load_runtime()
                            regime_mode = str(cfg_live.get("regime_mode", "auto")).lower().strip()
                            regime_txt = "AUTO" if regime_mode == "auto" else ("SCALPING" if regime_mode.startswith("scal") else "SWING")
                            _cb_reply(
                                f"📡 상태\n- 자동매매: {'ON' if cfg_live.get('auto_trade') else 'OFF'}\n"
                                f"- 모드: {cfg_live.get('trade_mode','-')}\n"
                                f"- 레짐: {regime_txt}\n"
                                f"- 잔고: {total:.2f} USDT (가용 {free:.2f})\n"
                                f"- 연속손실: {rt2.get('consec_losses',0)}\n"
                            )

                        elif data == "vision":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                mon_now = read_json_safe(MONITOR_FILE, {}) or {}
                                coins = mon_now.get("coins", {}) or {}
                                lines = ["👁️ AI 시야(요약)", f"- 마지막 하트비트: {mon_now.get('last_heartbeat_kst','-')}"]
                                for sym, cs in list(coins.items())[:10]:
                                    style = str(cs.get("style_reco", "")) or ""
                                    stxt = f"[{style}] " if style else ""
                                    lines.append(
                                        f"- {sym}: {stxt}{str(cs.get('ai_decision','-')).upper()}({cs.get('ai_confidence','-')}%) "
                                        f"/ 단기 {cs.get('trend_short','-')} / 장기 {cs.get('trend_long','-')} "
                                        f"/ 패턴 {str(cs.get('pattern','-'))[:18]} "
                                        f"/ {str(cs.get('ai_reason_easy') or cs.get('skip_reason') or '')[:35]}"
                                    )
                                _cb_reply("\n".join(lines))

                        elif data == "balance":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                free, total = safe_fetch_balance(ex)
                                _cb_reply(f"💰 잔고\n- 총자산: {total:.2f} USDT\n- 사용가능: {free:.2f} USDT")

                        elif data == "position":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                blocks: List[str] = []
                                ps = safe_fetch_positions(ex, TARGET_COINS)
                                act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                if not act:
                                    _cb_reply("📊 포지션\n\n- ⚪ 없음(관망)")
                                else:
                                    rt_open_targets = {}
                                    try:
                                        rt_open_targets = (rt.get("open_targets", {}) or {}) if isinstance(rt, dict) else {}
                                    except Exception:
                                        rt_open_targets = {}
                                    for p in act:
                                        sym = p.get("symbol", "")
                                        side = position_side_normalize(p)
                                        roi = float(position_roi_percent(p))
                                        upnl = float(p.get("unrealizedPnl") or 0.0)
                                        lev = p.get("leverage", "?")
                                        tgt0 = _resolve_open_target_for_symbol(sym, active_targets, rt_open_targets)
                                        style = str((tgt0 or {}).get("style", ""))
                                        blocks.append(_fmt_pos_block(sym, side, lev, roi, upnl, style=style, tgt=tgt0))
                                    _cb_reply("📊 포지션\n\n" + "\n\n".join(blocks))
                                    try:
                                        tg_send_position_chart_images(
                                            ex,
                                            act,
                                            active_targets,
                                            rt_open_targets,
                                            cfg,
                                            route_mode=str(cfg.get("tg_admin_replies_to", "channel") or "channel"),
                                            admin_uid=(int(uid) if uid is not None else None),
                                            fallback_chat_id=(int(cb_chat_id) if cb_chat_id is not None else None),
                                        )
                                    except Exception:
                                        pass

                        elif data == "log":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                kb_log = {
                                    "inline_keyboard": [
                                        [{"text": "📌 금일 손익", "callback_data": "log_today"}, {"text": "🗓️ 일별 손익", "callback_data": "log_daily"}],
                                        [{"text": "📆 월별 손익", "callback_data": "log_monthly"}, {"text": "📋 최근 거래표", "callback_data": "log_trades"}],
                                        [{"text": "🧾 일지상세 도움", "callback_data": "log_detail_help"}, {"text": "📜 최근 텍스트", "callback_data": "log_recent"}],
                                    ]
                                }
                                _cb_reply_kb(
                                    "📜 매매일지 메뉴\n- 버튼을 누르면 구글시트 표를 이미지로 보냅니다.",
                                    kb_log,
                                )

                        elif data == "log_recent":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                df_log = read_trade_log()
                                if df_log.empty:
                                    _cb_reply("📜 일지 없음")
                                else:
                                    top = df_log.head(8)
                                    msg = ["📜 최근 매매일지(요약)"]
                                    for _, r in top.iterrows():
                                        tid = str(r.get("TradeID", "") or "")
                                        pnl = float(r.get("PnL_Percent", 0) or 0)
                                        emo = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                                        msg.append(
                                            f"- {emo} {r['Time']} {r['Coin']} {r['Side']} {pnl:.2f}% | {str(r.get('OneLine',''))[:40]} | ID:{tid}"
                                        )
                                    _cb_reply("\n".join(msg))

                        elif data in ["log_today", "log_daily", "log_monthly", "log_trades"]:
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                kind_map = {
                                    "log_today": "today",
                                    "log_daily": "daily",
                                    "log_monthly": "monthly",
                                    "log_trades": "trades",
                                }
                                kind = kind_map.get(data, "today")
                                res = gsheet_build_journal_snapshot(kind=kind, timeout_sec=26)
                                if not bool(res.get("ok", False)):
                                    _cb_reply(f"⚠️ 구글시트 표 조회 실패\n- {str(res.get('error','unknown'))[:220]}")
                                else:
                                    img = str(res.get("image", "") or "")
                                    cap = str(res.get("caption", "📎 구글시트 표") or "📎 구글시트 표")
                                    rows = int(res.get("rows", 0) or 0)
                                    _cb_reply(f"{cap}\n- 행 수: {rows}")
                                    _cb_send_photo(img, caption=cap)

                        elif data == "log_detail_help":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🧾 일지 조회\n- /log : 매매일지 메뉴\n- /log <ID> : 상세\n- (호환) 일지상세 <ID>")

                        elif data == "scan":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                rid = uuid.uuid4().hex[:8]
                                rt2 = load_runtime()
                                rt2["force_scan"] = {
                                    "id": rid,
                                    "requested_at_epoch": time.time(),
                                    "requested_at_kst": now_kst_str(),
                                    "requested_by": int(uid or 0),
                                    "symbols": list(TARGET_COINS),
                                    "scan_only": True,
                                    "done": False,
                                }
                                save_runtime(rt2)
                                try:
                                    mon_add_event(mon, "SCAN_REQUEST", "", f"force_scan id={rid}", {"symbols": list(TARGET_COINS), "by": uid})
                                    gsheet_log_event("SCAN_REQUEST", message=f"id={rid}", payload={"symbols": list(TARGET_COINS), "by": uid})
                                except Exception:
                                    pass
                                _cb_reply(f"🔎 강제스캔 요청 완료: {rid}\n- 주의: 스캔만 수행(주문X)")

                        elif data == "mode_help":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🎚️ /mode 사용법\n- /mode auto\n- /mode scalping\n- /mode swing")

                        elif data == "gsheet":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                stg = gsheet_status_snapshot()
                                msg = (
                                    "📎 Google Sheets 상태\n"
                                    f"- enabled: {stg.get('enabled')}\n"
                                    f"- connected: {stg.get('connected')}\n"
                                    f"- spreadsheet_id: {stg.get('spreadsheet_id')}\n"
                                    f"- worksheet: {stg.get('worksheet')}\n"
                                    f"- service_account_email: {stg.get('service_account_email')}\n"
                                    f"- queue_high/scan: {stg.get('queue_high')}/{stg.get('queue_scan')}\n"
                                    f"- last_append: {stg.get('last_append_kst')} ({stg.get('last_append_type')}:{stg.get('last_append_stage')})\n"
                                    f"- last_err: {stg.get('last_err')}\n"
                                    "테스트: /gsheet test"
                                )
                                _cb_reply(msg[:3500])

                        elif data == "close_all":
                            if not is_admin:
                                _cb_reply("⛔️ 관리자만 사용할 수 있는 버튼입니다.")
                            else:
                                _cb_reply("🛑 전량 청산 시도")
                                for sym in TARGET_COINS:
                                    ps = safe_fetch_positions(ex, [sym])
                                    act = [p for p in ps if float(p.get("contracts") or 0) > 0]
                                    if not act:
                                        continue
                                    p = act[0]
                                    side = position_side_normalize(p)
                                    contracts = float(p.get("contracts") or 0)
                                    close_position_market(ex, sym, side, contracts)
                                _cb_reply("✅ 전량 청산 요청 완료")
                                try:
                                    mon_add_event(mon, "CLOSE_ALL", "", "close_all requested", {"by": uid})
                                    gsheet_log_event("CLOSE_ALL", message="close_all", payload={"by": uid})
                                except Exception:
                                    pass

                        if cb_id:
                            tg_answer_callback(cb_id)

                except Exception as _e:
                    # 업데이트 처리 중 오류도 EVENT로 남김(봇은 계속)
                    try:
                        mon_add_event(mon, "TG_UPDATE_ERROR", "", "TG update 처리 오류", {"err": str(_e)[:240]})
                        gsheet_log_event("TG_UPDATE_ERROR", message=str(_e)[:240])
                    except Exception:
                        pass
                    notify_admin_error("TG_UPDATE_HANDLER", _e, tb=traceback.format_exc(), min_interval_sec=60.0)

            monitor_write_throttled(mon, 2.0)
            backoff_sec = 1.0
            time.sleep(0.8)

        except Exception as e:
            # 스레드가 죽지 않도록 backoff
            try:
                try:
                    mon_add_event(mon, "THREAD_ERROR", "", "TG_THREAD_LOOP 예외", {"err": str(e)[:240]})
                    mon["loop_stage"] = "ERROR"
                    mon["loop_stage_kst"] = now_kst_str()
                    monitor_write_throttled(mon, 0.2)
                except Exception:
                    pass
                notify_admin_error("TG_THREAD_LOOP", e, tb=traceback.format_exc(), min_interval_sec=45.0)
                err = f"{e}"
                if len(err) > 500:
                    err = err[:500] + "..."
                # ✅ 요구: 오류는 관리자 DM으로(채널 스팸 방지)
                if not TG_ADMIN_IDS:
                    tg_send(f"⚠️ 스레드 오류: {err}", target="channel", cfg=load_settings())
            except Exception:
                pass
            time.sleep(backoff_sec)
            backoff_sec = float(clamp(backoff_sec * 1.6, 1.0, 15.0))


# =========================================================
# ✅ 17.5) Watchdog: 하트비트 멈춤 감시/경고/재시작 시도
# =========================================================
def watchdog_thread():
    warned = False
    last_restart_epoch = 0.0
    while True:
        try:
            mon = read_json_safe(MONITOR_FILE, {}) or {}
            hb = float(mon.get("last_heartbeat_epoch", 0) or 0)
            age = (time.time() - hb) if hb else 9999
            cfg = load_settings()
            if age >= 60 and not warned:
                warned = True
                msg = f"🧯 워치독 경고: 하트비트 {age:.0f}초 정체(스레드 멈춤 의심)"
                tg_send(msg, target="channel", cfg=cfg)
                tg_send(msg, target="admin", cfg=cfg)
                # ✅ 어디서 멈췄는지 관리자에게 스택 일부 전달(원인 추적용)
                try:
                    import sys as _sys

                    frames = _sys._current_frames()
                    lines_out: List[str] = []
                    for t in threading.enumerate():
                        if str(getattr(t, "name", "") or "").startswith("TG_THREAD"):
                            fr = frames.get(getattr(t, "ident", None))
                            if fr is None:
                                continue
                            try:
                                stk = traceback.format_stack(fr)
                                tail = "".join(stk[-18:])
                            except Exception:
                                tail = ""
                            lines_out.append(f"[{t.name}]\\n{tail}".strip())
                    if lines_out:
                        text2 = "🧩 스택 스냅샷(일부)\\n" + "\\n\\n".join(lines_out)
                        if len(text2) > 3500:
                            text2 = text2[:3500] + "..."
                        tg_send(text2, target="admin", cfg=cfg)
                except Exception:
                    pass
            if age < 30:
                warned = False

            # ✅ 하트비트가 오래 정체되면(살아있어도) 워커를 revoke + recovery 스레드를 띄워 복구
            try:
                if age >= 90 and (time.time() - float(last_restart_epoch or 0.0)) >= 180:
                    wid = str(mon.get("worker_id", "") or "").strip()
                    if wid:
                        runtime_worker_revoke(wid, reason=f"watchdog_stale_{int(age)}s")
                    # recovery thread 중복 방지
                    has_recovery = False
                    for t in threading.enumerate():
                        if str(t.name or "").startswith("TG_THREAD_RECOVERY") and t.is_alive():
                            has_recovery = True
                            break
                    if not has_recovery:
                        ex2 = create_exchange_client_uncached() or exchange
                        th = threading.Thread(
                            target=telegram_thread,
                            args=(ex2,),
                            daemon=True,
                            name=f"TG_THREAD_RECOVERY_{int(time.time())}",
                        )
                        add_script_run_ctx(th)
                        th.start()
                        msg3 = f"🧯 워치독 복구: recovery 스레드 시작(age={age:.0f}s, revoked={wid or '-'})"
                        tg_send(msg3, target="admin", cfg=cfg)
                        tg_send(msg3, target="channel", cfg=cfg)
                    last_restart_epoch = time.time()
            except Exception:
                pass

            # 스레드가 아예 없으면 재시작
            alive = False
            for t in threading.enumerate():
                if t.name == "TG_THREAD" and t.is_alive():
                    alive = True
                    break
            if not alive:
                try:
                    ex2 = create_exchange_client_uncached() or exchange
                    th = threading.Thread(target=telegram_thread, args=(ex2,), daemon=True, name="TG_THREAD")
                    add_script_run_ctx(th)
                    th.start()
                    msg2 = "🧯 워치독: TG_THREAD 재시작 시도"
                    tg_send(msg2, target="channel", cfg=cfg)
                    tg_send(msg2, target="admin", cfg=cfg)
                except Exception:
                    pass

        except Exception:
            pass
        time.sleep(5.0)


# =========================================================
# ✅ 18) 스레드 시작(중복 방지) - TG_THREAD + WATCHDOG
# =========================================================
def ensure_threads_started():
    has_tg = False
    has_wd = False
    has_poll = False
    has_gs = False
    has_send = False
    for t in threading.enumerate():
        if t.name == "TG_THREAD":
            has_tg = True
        if t.name == "TG_POLL_THREAD":
            has_poll = True
        if t.name == "GSHEET_THREAD":
            has_gs = True
        if t.name == "TG_SEND_THREAD":
            has_send = True
        if t.name == "WATCHDOG_THREAD":
            has_wd = True
    if not has_send:
        # Telegram send worker (sendMessage) - 네트워크 블로킹으로 TG_THREAD가 멈추는 현상 완화
        ths = threading.Thread(target=telegram_send_worker_thread, args=(), daemon=True, name="TG_SEND_THREAD")
        add_script_run_ctx(ths)
        ths.start()
    if not has_poll:
        # Telegram long polling(getUpdates) 전용 스레드 (요구사항)
        thp = threading.Thread(target=telegram_polling_thread, args=(), daemon=True, name="TG_POLL_THREAD")
        add_script_run_ctx(thp)
        thp.start()
    if not has_gs:
        # Google Sheets append_row 전용 워커 (요구사항)
        thg = threading.Thread(target=gsheet_worker_thread, args=(), daemon=True, name="GSHEET_THREAD")
        add_script_run_ctx(thg)
        thg.start()
    if not has_tg:
        ex2 = create_exchange_client_uncached() or exchange
        th = threading.Thread(target=telegram_thread, args=(ex2,), daemon=True, name="TG_THREAD")
        add_script_run_ctx(th)
        th.start()
    if not has_wd:
        wd = threading.Thread(target=watchdog_thread, args=(), daemon=True, name="WATCHDOG_THREAD")
        add_script_run_ctx(wd)
        wd.start()


# 전역 예외 훅 설치(가능한 경우): 스레드/런타임에서 잡히지 않은 오류를 관리자 DM으로
install_global_error_hooks()
ensure_threads_started()


# =========================================================
# ✅ 19) Streamlit UI
# =========================================================
st.sidebar.title("🛠️ 제어판")
st.sidebar.caption("Streamlit=제어/상태 확인용, Telegram=실시간 보고/조회용")

openai_key_secret = _sget_str("OPENAI_API_KEY")
if not openai_key_secret and not config.get("openai_api_key"):
    k = st.sidebar.text_input("OpenAI API Key 입력(선택)", type="password")
    if k:
        config["openai_api_key"] = k
        save_settings(config)
        st.rerun()

with st.sidebar.expander("🧪 디버그: 저장된 설정(bot_settings.json) 확인"):
    st.json(read_json_safe(SETTINGS_FILE, {}))
    st.caption("⚠️ 아래 버튼은 `bot_settings.json`을 '기본값(프리셋)'으로 덮어씁니다.")
    _reset_ok = st.checkbox("덮어쓰기 확인", value=False, key="reset_settings_confirm")
    if st.button("♻️ 기본값(프리셋) 적용", disabled=not _reset_ok):
        try:
            config.clear()
            config.update(default_settings())
            save_settings(config)
            st.success("✅ 기본값(프리셋) 적용 완료")
            st.rerun()
        except Exception as e:
            st.error(f"적용 실패: {e}")

mode_keys = list(MODE_RULES.keys())
safe_mode = config.get("trade_mode", "안전모드")
if safe_mode not in mode_keys:
    safe_mode = "안전모드"
config["trade_mode"] = st.sidebar.selectbox("매매 모드", mode_keys, index=mode_keys.index(safe_mode))

auto_on = st.sidebar.checkbox("🤖 자동매매 (텔레그램 연동)", value=bool(config.get("auto_trade", False)))
if auto_on != bool(config.get("auto_trade", False)):
    config["auto_trade"] = auto_on
    save_settings(config)
    st.rerun()

st.sidebar.divider()
config["timeframe"] = st.sidebar.selectbox(
    "타임프레임",
    ["1m", "3m", "5m", "15m", "1h"],
    index=["1m", "3m", "5m", "15m", "1h"].index(config.get("timeframe", "5m")),
)
config["tg_enable_reports"] = st.sidebar.checkbox("📨 텔레그램 이벤트 알림(진입/청산 등)", value=bool(config.get("tg_enable_reports", True)))
config["tg_send_entry_reason"] = st.sidebar.checkbox("📌 텔레그램에 진입근거(긴글)도 보내기", value=bool(config.get("tg_send_entry_reason", False)))
config["tg_simple_messages"] = st.sidebar.checkbox(
    "🧓 텔레그램 쉬운말(핵심만)",
    value=bool(config.get("tg_simple_messages", True)),
    help="진입/익절/손절/부분익절/추매 등 알림을 어려운 용어 없이 '핵심 정보'만 보내도록 합니다.",
)

st.sidebar.subheader("🧠 AI 비용 절감")
config["ai_scan_once_per_bar"] = st.sidebar.checkbox(
    "스캔 AI: 같은 봉 재호출 금지(권장)",
    value=bool(config.get("ai_scan_once_per_bar", True)),
    help="자동 스캔에서 같은 봉(단기 TF)에서는 AI를 다시 부르지 않고 마지막 결과를 재사용합니다. /scan은 예외입니다.",
)
with st.sidebar.expander("진입 전 AI 호출 필터(거래량/이격도)"):
    config["ai_call_require_volume_spike"] = st.checkbox(
        "거래량 스파이크 없으면 AI 호출 안함",
        value=bool(config.get("ai_call_require_volume_spike", True)),
        help="현재 봉 거래량이 최근 평균보다 충분히 커야만 AI를 호출합니다(박스권/힘 없는 해소 진입 방지).",
    )
    v1, v2 = st.columns(2)
    config["ai_call_volume_spike_mul"] = v1.number_input("스파이크 배수", 1.0, 10.0, float(config.get("ai_call_volume_spike_mul", 1.5) or 1.5), step=0.1)
    config["ai_call_volume_spike_period"] = v2.number_input("평균 기간", 5, 120, int(config.get("ai_call_volume_spike_period", 20) or 20), step=1)
    st.divider()
    config["ai_call_require_disparity"] = st.checkbox(
        "이격도 과하면 AI 호출 안함",
        value=bool(config.get("ai_call_require_disparity", True)),
        help="가격이 MA(기본 20)에서 너무 멀면(과열/급락) 눌림목이 아니라 추세 꺾임일 수 있어 AI 호출을 막습니다.",
    )
    d1, d2 = st.columns(2)
    config["ai_call_disparity_max_abs_pct"] = d1.number_input("최대 |이격도|%", 0.5, 30.0, float(config.get("ai_call_disparity_max_abs_pct", 4.0) or 4.0), step=0.5)
    config["ai_call_disparity_ma_period"] = d2.number_input("이격도 MA 기간", 5, 120, int(config.get("ai_call_disparity_ma_period", 20) or 20), step=1)

st.sidebar.subheader("⏱️ 주기 리포트")
config["tg_enable_heartbeat_report"] = st.sidebar.checkbox(
    "💓 하트비트(요약) 전송",
    value=bool(config.get("tg_enable_heartbeat_report", False)),
    help="기본은 OFF. 켜면 지정한 주기마다 잔고/포지션 요약을 보냅니다.",
)
config["tg_heartbeat_interval_sec"] = st.sidebar.number_input(
    "하트비트 주기(초)",
    60,
    7200,
    int(config.get("tg_heartbeat_interval_sec", 900)),
    step=60,
)
config["tg_heartbeat_silent"] = st.sidebar.checkbox(
    "하트비트는 무음(알림X)",
    value=bool(config.get("tg_heartbeat_silent", True)),
    help="무음 전송(disable_notification)로 보내서 알림을 줄입니다.",
)
st.sidebar.divider()
config["tg_enable_periodic_report"] = st.sidebar.checkbox(
    "상황보고 전송",
    value=bool(config.get("tg_enable_periodic_report", True)),
    help="요청대로 OpenAI를 부르지 않는 '상태 요약'만 전송합니다.",
)
config["report_interval_min"] = st.sidebar.number_input("상황보고 주기(분)", 3, 120, int(config.get("report_interval_min", 15)))
config["tg_periodic_report_silent"] = st.sidebar.checkbox(
    "상황보고는 무음(알림X)",
    value=bool(config.get("tg_periodic_report_silent", True)),
    help="무음 전송(disable_notification)로 보내서 알림을 줄입니다.",
)
config["tg_enable_hourly_vision_report"] = st.sidebar.checkbox("1시간 AI시야 리포트(채널)", value=bool(config.get("tg_enable_hourly_vision_report", False)))
config["vision_report_interval_min"] = st.sidebar.number_input("AI시야 리포트 주기(분)", 10, 240, int(config.get("vision_report_interval_min", 60)))

st.sidebar.subheader("🔔 알림(푸시) 제어")
config["tg_notify_entry_exit_only"] = st.sidebar.checkbox(
    "알림은 진입/청산만(권장)",
    value=bool(config.get("tg_notify_entry_exit_only", True)),
    help="ON이면 DCA/부분익절/방식전환 같은 '중간 이벤트'는 무음 전송으로 보냅니다.",
)
config["tg_trade_alert_to_admin"] = st.sidebar.checkbox(
    "진입/청산도 관리자 DM으로 복사",
    value=bool(config.get("tg_trade_alert_to_admin", False)),
    help="사용자 요청: 기본은 OFF(관리자는 버튼만). 켜면 진입/청산 알림을 관리자 DM으로 한 번 더 보냅니다. (관리자는 먼저 봇에 /start 필요)",
)

st.sidebar.subheader("📡 텔레그램 라우팅")
config["tg_route_events_to"] = st.sidebar.selectbox("이벤트(진입/익절/손절/보고) 전송 대상", ["channel", "group", "both"], index=["channel", "group", "both"].index(config.get("tg_route_events_to", "channel")))
config["tg_route_queries_to"] = st.sidebar.selectbox("조회/버튼 응답 전송 대상", ["group", "channel", "both"], index=["group", "channel", "both"].index(config.get("tg_route_queries_to", "group")))
config["tg_admin_replies_to"] = st.sidebar.selectbox(
    "관리자 명령 응답 위치",
    ["channel", "admin", "both"],
    index=["channel", "admin", "both"].index(config.get("tg_admin_replies_to", "channel")) if config.get("tg_admin_replies_to", "channel") in ["channel", "admin", "both"] else 0,
    help="관리자가 DM으로 /scan /mode /positions 등을 실행했을 때, 결과를 어디로 보낼지 선택합니다.",
)
st.sidebar.caption("※ TG_CHAT_ID_GROUP / TG_CHAT_ID_CHANNEL secrets를 설정하면 채널/그룹 분리가 됩니다.")

st.sidebar.divider()
st.sidebar.subheader("🧭 추세/스타일 정책")
config["trend_filter_enabled"] = st.sidebar.checkbox("장기추세(1h) 정책 사용", value=bool(config.get("trend_filter_enabled", True)))
config["trend_filter_timeframe"] = "1h"
config["trend_filter_policy"] = st.sidebar.selectbox("정책", ["ALLOW_SCALP", "STRICT", "OFF"], index=["ALLOW_SCALP", "STRICT", "OFF"].index(config.get("trend_filter_policy", "ALLOW_SCALP")))
st.sidebar.caption("ALLOW_SCALP: 역추세 허용(스캘핑 강제) / STRICT: 역추세 금지 / OFF: 미사용")

config["regime_mode"] = st.sidebar.selectbox(
    "레짐 모드(/mode)",
    ["auto", "scalping", "swing"],
    index=["auto", "scalping", "swing"].index(str(config.get("regime_mode", "auto")).lower() if str(config.get("regime_mode", "auto")).lower() in ["auto", "scalping", "swing"] else "auto"),
)
config["regime_switch_control"] = st.sidebar.selectbox(
    "레짐 흔들림 방지(시간락 없음)",
    ["confirm2", "hysteresis", "off"],
    index=["confirm2", "hysteresis", "off"].index(str(config.get("regime_switch_control", "confirm2")).lower() if str(config.get("regime_switch_control", "confirm2")).lower() in ["confirm2", "hysteresis", "off"] else "confirm2"),
)
with st.sidebar.expander("confirm2 상세(선택)"):
    c_c1, c_c2 = st.columns(2)
    config["regime_confirm_n"] = c_c1.number_input("confirm n", 2, 8, int(config.get("regime_confirm_n", 2)))
    config["regime_confirm_n_flipback"] = c_c2.number_input("flipback n", 2, 10, int(config.get("regime_confirm_n_flipback", 3)))
with st.sidebar.expander("히스테리시스 상세(선택)"):
    c_h1, c_h2, c_h3 = st.columns(3)
    config["regime_hysteresis_step"] = c_h1.number_input("step", 0.05, 1.0, float(config.get("regime_hysteresis_step", 0.55)), step=0.05)
    config["regime_hysteresis_enter_swing"] = c_h2.number_input("enter swing", 0.1, 0.99, float(config.get("regime_hysteresis_enter_swing", 0.75)), step=0.05)
    config["regime_hysteresis_enter_scalp"] = c_h3.number_input("enter scalp", 0.01, 0.9, float(config.get("regime_hysteresis_enter_scalp", 0.25)), step=0.05)

config["highrisk_entry_requires_swing"] = st.sidebar.checkbox(
    "하이리스크: 스윙만 신규진입(선택)",
    value=bool(config.get("highrisk_entry_requires_swing", False)),
    help="하이리스크/하이리턴 모드에서 auto 레짐일 때만 적용됩니다. ON이면 단기+장기 추세 정렬(스윙)에서만 신규 진입합니다.",
)

config["style_auto_enable"] = st.sidebar.checkbox("스캘핑/스윙 자동 선택/전환", value=bool(config.get("style_auto_enable", True)))
config["style_entry_ai_enable"] = st.sidebar.checkbox(
    "🤖 신규진입 스타일 선택에 AI 사용(비용↑)",
    value=bool(config.get("style_entry_ai_enable", True)),
    help="신규 진입 시 스캘핑/스윙 선택을 OpenAI로 한 번 더 보조합니다. 기본 ON.",
)
config["style_switch_ai_enable"] = st.sidebar.checkbox(
    "🤖 포지션 스타일 전환에 AI 사용(비용↑)",
    value=bool(config.get("style_switch_ai_enable", False)),
    help="포지션 보유 중 스타일 전환 판단에 OpenAI를 추가로 호출합니다. 기본은 룰 기반(비용/429 방지).",
)
config["style_ai_cache_sec"] = st.sidebar.number_input("스타일 AI 캐시(초)", 0, 36000, int(config.get("style_ai_cache_sec", 600)))
config["style_lock_minutes"] = st.sidebar.number_input("스타일 전환 락(분) [DEPRECATED]", 0, 180, int(config.get("style_lock_minutes", 20)))
st.sidebar.caption("※ 요구사항 반영: 시간 기반 최소유지기간은 사용하지 않습니다(레짐 흔들림 방지=confirm2/hysteresis).")

st.sidebar.subheader("🧩 스윙 분할익절/순환")
config["swing_partial_tp_enable"] = st.sidebar.checkbox("스윙: 1/2/3차 분할익절", value=bool(config.get("swing_partial_tp_enable", True)))
with st.sidebar.expander("분할익절 상세 설정"):
    p1a, p1b = st.columns(2)
    config["swing_partial_tp1_at_tp_frac"] = p1a.number_input("1차: TP비율", 0.05, 0.95, float(config.get("swing_partial_tp1_at_tp_frac", 0.35)), step=0.05)
    config["swing_partial_tp1_close_pct"] = p1b.number_input("1차: 청산%", 1, 90, int(config.get("swing_partial_tp1_close_pct", 33)))
    p2a, p2b = st.columns(2)
    config["swing_partial_tp2_at_tp_frac"] = p2a.number_input("2차: TP비율", 0.05, 0.95, float(config.get("swing_partial_tp2_at_tp_frac", 0.60)), step=0.05)
    config["swing_partial_tp2_close_pct"] = p2b.number_input("2차: 청산%", 1, 90, int(config.get("swing_partial_tp2_close_pct", 33)))
    p3a, p3b = st.columns(2)
    config["swing_partial_tp3_at_tp_frac"] = p3a.number_input("3차: TP비율", 0.05, 0.99, float(config.get("swing_partial_tp3_at_tp_frac", 0.85)), step=0.05)
    config["swing_partial_tp3_close_pct"] = p3b.number_input("3차: 청산%", 1, 95, int(config.get("swing_partial_tp3_close_pct", 34)))
    st.caption("※ (선택) 아래 USDT(마진) 값을 0보다 크게 설정하면, 해당 단계는 '청산%' 대신 USDT 기준으로 청산합니다.")
    u1, u2, u3 = st.columns(3)
    config["swing_partial_tp1_close_usdt"] = u1.number_input("1차: 청산 USDT", 0.0, 1000000.0, float(config.get("swing_partial_tp1_close_usdt", 0.0) or 0.0), step=5.0)
    config["swing_partial_tp2_close_usdt"] = u2.number_input("2차: 청산 USDT", 0.0, 1000000.0, float(config.get("swing_partial_tp2_close_usdt", 0.0) or 0.0), step=5.0)
    config["swing_partial_tp3_close_usdt"] = u3.number_input("3차: 청산 USDT", 0.0, 1000000.0, float(config.get("swing_partial_tp3_close_usdt", 0.0) or 0.0), step=5.0)

config["swing_recycle_enable"] = st.sidebar.checkbox("스윙: 순환매도(부분익절 후 재진입)", value=bool(config.get("swing_recycle_enable", False)))
with st.sidebar.expander("순환매도 상세 설정"):
    r1, r2, r3 = st.columns(3)
    config["swing_recycle_cooldown_min"] = r1.number_input("쿨다운(분)", 1, 240, int(config.get("swing_recycle_cooldown_min", 20)))
    config["swing_recycle_max_count"] = r2.number_input("최대횟수", 0, 10, int(config.get("swing_recycle_max_count", 2)))
    config["swing_recycle_reentry_roi"] = r3.number_input("재진입ROI(%)", 0.1, 20.0, float(config.get("swing_recycle_reentry_roi", 0.8)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🧱 지지/저항(SR) 손절/익절")
config["use_sr_stop"] = st.sidebar.checkbox("SR 기반 가격 손절/익절 사용", value=bool(config.get("use_sr_stop", True)))
c_sr1, c_sr2 = st.sidebar.columns(2)
config["sr_timeframe"] = c_sr1.selectbox("SR 타임프레임", ["5m", "15m", "1h", "4h"], index=["5m", "15m", "1h", "4h"].index(config.get("sr_timeframe", "15m")))
config["sr_pivot_order"] = c_sr2.number_input("피벗 민감도", 3, 10, int(config.get("sr_pivot_order", 6)))
c_sr_lb1, c_sr_lb2 = st.sidebar.columns(2)
config["sr_lookback"] = c_sr_lb1.number_input("SR Lookback", 120, 800, int(config.get("sr_lookback", 220)), step=10)
config["sr_levels_cache_sec"] = c_sr_lb2.number_input("SR Cache(초)", 5, 600, int(config.get("sr_levels_cache_sec", 60)), step=5)
c_sr3, c_sr4 = st.sidebar.columns(2)
config["sr_atr_period"] = c_sr3.number_input("ATR 기간", 7, 30, int(config.get("sr_atr_period", 14)))
config["sr_buffer_atr_mult"] = c_sr4.number_input("버퍼(ATR배)", 0.05, 2.0, float(config.get("sr_buffer_atr_mult", 0.25)), step=0.05)
config["sr_rr_min"] = st.sidebar.number_input("SR 최소 RR", 1.0, 5.0, float(config.get("sr_rr_min", 1.5)), step=0.1)
with st.sidebar.expander("스윙(Swing) SR 상세(선택)"):
    c_sw1, c_sw2 = st.columns(2)
    config["sr_timeframe_swing"] = c_sw1.selectbox(
        "스윙 SR TF",
        ["15m", "1h", "4h"],
        index=["15m", "1h", "4h"].index(config.get("sr_timeframe_swing", "1h")) if config.get("sr_timeframe_swing", "1h") in ["15m", "1h", "4h"] else 1,
    )
    config["sr_lookback_swing"] = c_sw2.number_input("스윙 SR Lookback", 120, 800, int(config.get("sr_lookback_swing", 320)), step=10)
    c_sw3, c_sw4 = st.columns(2)
    config["sr_pivot_order_swing"] = c_sw3.number_input("스윙 피벗", 3, 12, int(config.get("sr_pivot_order_swing", 8)))
    config["sr_buffer_atr_mult_swing"] = c_sw4.number_input("스윙 버퍼(ATR배)", 0.05, 2.0, float(config.get("sr_buffer_atr_mult_swing", 0.45)), step=0.05)
    config["sr_rr_min_swing"] = st.number_input("스윙 SR 최소 RR", 1.0, 6.0, float(config.get("sr_rr_min_swing", 2.0)), step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🛡️ 방어/자금 관리")
config["loss_pause_enable"] = st.sidebar.checkbox("연속손실 보호(자동 정지)", value=bool(config.get("loss_pause_enable", True)))
c1, c2 = st.sidebar.columns(2)
config["loss_pause_after"] = c1.number_input("연속손실 N회", 1, 20, int(config.get("loss_pause_after", 3)))
config["loss_pause_minutes"] = c2.number_input("정지(분)", 1, 240, int(config.get("loss_pause_minutes", 30)))
with st.sidebar.expander("진입 사이징/레버(고정/ATR/리스크캡/Kelly)"):
    config["margin_mode"] = st.selectbox(
        "마진 모드",
        ["cross", "isolated"],
        index=["cross", "isolated"].index(str(config.get("margin_mode", "cross") or "cross")) if str(config.get("margin_mode", "cross") or "cross") in ["cross", "isolated"] else 0,
        help="계정/마켓 설정에 따라 실패할 수 있습니다. 실패해도 봇은 죽지 않고 주문만 시도합니다.",
    )
    st.divider()
    c_fx1, c_fx2 = st.columns(2)
    config["fixed_entry_pct_enable"] = c_fx1.checkbox("진입비중 고정", value=bool(config.get("fixed_entry_pct_enable", False)))
    config["fixed_leverage_enable"] = c_fx2.checkbox("레버 고정", value=bool(config.get("fixed_leverage_enable", False)))
    c_fx3, c_fx4 = st.columns(2)
    config["fixed_entry_pct"] = c_fx3.number_input("고정 진입비중(%)", 1.0, 100.0, float(config.get("fixed_entry_pct", 20.0) or 20.0), step=1.0)
    config["fixed_leverage"] = c_fx4.number_input("고정 레버", 1, 125, int(config.get("fixed_leverage", 20) or 20), step=1)
    st.caption("※ 고정이 ON이면 AI/ATR 출력은 '표시용'이고, 실제 주문은 고정값으로 들어갑니다.")
    st.divider()
    config["atr_leverage_enable"] = st.checkbox("ATR 기반 레버리지(권장)", value=bool(config.get("atr_leverage_enable", True)))
    a1, a2, a3 = st.columns(3)
    config["atr_leverage_low_pct"] = a1.number_input("ATR low(%)", 0.05, 10.0, float(config.get("atr_leverage_low_pct", 0.35) or 0.35), step=0.05)
    config["atr_leverage_high_pct"] = a2.number_input("ATR high(%)", 0.1, 30.0, float(config.get("atr_leverage_high_pct", 1.20) or 1.20), step=0.1)
    config["atr_leverage_window"] = a3.number_input("ATR 기간", 7, 50, int(config.get("atr_leverage_window", 14) or 14), step=1)
    a4, a5 = st.columns(2)
    config["atr_leverage_min"] = a4.number_input("ATR 최소 레버", 1, 125, int(config.get("atr_leverage_min", 5) or 5), step=1)
    config["atr_leverage_max"] = a5.number_input("ATR 최대 레버", 1, 125, int(config.get("atr_leverage_max", 20) or 20), step=1)
    st.caption("※ ATR%가 커질수록 레버리지를 낮추고, ATR%가 작을수록 높입니다.")
    st.divider()
    config["max_risk_per_trade_enable"] = st.checkbox("1회 최대손실 제한(권장)", value=bool(config.get("max_risk_per_trade_enable", True)))
    r1, r2 = st.columns(2)
    config["max_risk_per_trade_pct"] = r1.number_input("최대손실(%)", 0.1, 20.0, float(config.get("max_risk_per_trade_pct", 2.5) or 2.5), step=0.1)
    config["max_risk_per_trade_usdt"] = r2.number_input("최대손실(USDT)", 0.0, 100000000.0, float(config.get("max_risk_per_trade_usdt", 0.0) or 0.0), step=10.0)
    st.caption("※ 퍼센트/USDT 중 더 엄격한 기준으로 '진입금(마진)'을 자동 감산합니다.")
    st.caption("※ 모드별로 진입금이 너무 낮다면, 아래 오버라이드를 올려주세요(특히 하이리스크).")
    mr1, mr2, mr3 = st.columns(3)
    config["max_risk_per_trade_pct_safe"] = mr1.number_input("안전(%)", 0.1, 50.0, float(config.get("max_risk_per_trade_pct_safe", 2.5) or 2.5), step=0.1)
    config["max_risk_per_trade_pct_attack"] = mr2.number_input("공격(%)", 0.1, 50.0, float(config.get("max_risk_per_trade_pct_attack", 3.5) or 3.5), step=0.1)
    config["max_risk_per_trade_pct_highrisk"] = mr3.number_input("하이리스크(%)", 0.1, 50.0, float(config.get("max_risk_per_trade_pct_highrisk", 5.0) or 5.0), step=0.1)
    st.divider()
    config["kelly_sizing_enable"] = st.checkbox("Kelly cap(선택)", value=bool(config.get("kelly_sizing_enable", False)))
    k1, k2 = st.columns(2)
    config["kelly_fraction_mult"] = k1.number_input("Kelly 배수(half=0.5)", 0.05, 1.0, float(config.get("kelly_fraction_mult", 0.5) or 0.5), step=0.05)
    config["kelly_max_entry_pct"] = k2.number_input("Kelly 상한(%)", 1.0, 100.0, float(config.get("kelly_max_entry_pct", 20.0) or 20.0), step=1.0)
    st.caption("※ 현재는 AI 확신도(confidence)를 승률(p)로 근사합니다. (보수적으로 cap로만 사용)")
    st.divider()
    # 사용자 요청: 시간초과 강제청산은 사용하지 않음(목표 TP/SL 도달 전까지 홀딩)
    config["time_exit_enable"] = False
    st.checkbox("시간초과 정리(기회비용) [비활성화]", value=False, disabled=True)

with st.sidebar.expander("Fail-safe(자동매매 강제 종료)"):
    config["fail_safe_enable"] = st.checkbox("Fail-safe 사용", value=bool(config.get("fail_safe_enable", True)))
    st.caption("수익을 '보장'할 수는 없어서, 손실 확대/과매매를 막는 안전장치로 자동매매를 강제 OFF합니다.")
    st.divider()
    config["fail_safe_drawdown_enable"] = st.checkbox("드로다운 제한(피크 대비)", value=bool(config.get("fail_safe_drawdown_enable", True)))
    config["fail_safe_drawdown_from_peak_pct"] = st.number_input(
        "피크 대비 손실(%)",
        5.0,
        99.0,
        float(config.get("fail_safe_drawdown_from_peak_pct", 30.0) or 30.0),
        step=1.0,
        help="예: 30이면, 오늘 최고 자산(피크) 대비 -30% 이상 내려가면 자동매매를 끕니다.",
    )
    st.divider()
    config["fail_safe_profit_guard_enable"] = st.checkbox("수익 가드(거래 후 수익 없으면 OFF)", value=bool(config.get("fail_safe_profit_guard_enable", False)))
    pg1, pg2 = st.columns(2)
    config["fail_safe_profit_guard_min_trades"] = pg1.number_input(
        "최소 거래수",
        1,
        200,
        int(config.get("fail_safe_profit_guard_min_trades", 10) or 10),
        step=1,
        help="이 거래 수 이상 진행한 뒤에도 수익이 없으면 자동매매를 끕니다.",
    )
    config["fail_safe_profit_guard_min_pnl_usdt"] = pg2.number_input(
        "최소 실현손익(USDT)",
        -1000000.0,
        1000000.0,
        float(config.get("fail_safe_profit_guard_min_pnl_usdt", 0.0) or 0.0),
        step=1.0,
        help="0이면 '수익(+)이 아닌 경우' OFF. -50이면 -50 USDT 이하이면 OFF.",
    )

with st.sidebar.expander("추가 방어(서킷브레이커/일일 손실 한도)"):
    config["circuit_breaker_enable"] = st.checkbox("서킷브레이커 경고(연속 손실 알림)", value=bool(config.get("circuit_breaker_enable", False)))
    config["circuit_breaker_after"] = st.number_input("연속 손실 N번 → 경고", 3, 50, int(config.get("circuit_breaker_after", 12)), step=1)
    st.divider()
    config["daily_loss_limit_enable"] = st.checkbox("일일 최대 손실 한도 사용(도달 시 자동매매 OFF)", value=bool(config.get("daily_loss_limit_enable", False)))
    dl1, dl2 = st.columns(2)
    config["daily_loss_limit_pct"] = dl1.number_input("하루 손실 한도(%)", 0.0, 100.0, float(config.get("daily_loss_limit_pct", 8.0) or 0.0), step=0.5)
    config["daily_loss_limit_usdt"] = dl2.number_input("하루 손실 한도(USDT)", 0.0, 100000000.0, float(config.get("daily_loss_limit_usdt", 0.0) or 0.0), step=10.0)
    st.caption("※ 하루 손실은 '일일 시작 총자산(day_start_equity)' 대비로 계산합니다(가능하면 자동 설정).")

st.sidebar.divider()
config["use_trailing_stop"] = st.sidebar.checkbox("🚀 트레일링 스탑(수익보호)", value=bool(config.get("use_trailing_stop", True)))
config["exit_ai_targets_only"] = st.sidebar.checkbox(
    "🎯 청산은 AI 목표만 사용",
    value=bool(config.get("exit_ai_targets_only", True)),
    help="ON이면 익절/손절은 AI 목표 TP/SL에 닿을 때만 실행합니다. SR/본절/강제추적손절 등 다른 청산 규칙은 무시합니다.",
)
with st.sidebar.expander("손절 확인(휩쏘 방지)"):
    config["sl_confirm_enable"] = st.checkbox("ROI 손절은 확인 후 실행", value=bool(config.get("sl_confirm_enable", True)))
    c_slc1, c_slc2 = st.columns(2)
    config["sl_confirm_n"] = c_slc1.number_input("확인 횟수", 1, 5, int(config.get("sl_confirm_n", 2)), step=1)
    config["sl_confirm_window_sec"] = c_slc2.number_input("시간창(초)", 30.0, 3600.0, float(config.get("sl_confirm_window_sec", 600.0) or 600.0), step=30.0)
    st.caption("※ SR(지지/저항) 가격 이탈 손절은 즉시 실행됩니다.")
with st.sidebar.expander("청산 후 재진입 쿨다운(과매매 방지)"):
    cd1, cd2, cd3 = st.columns(3)
    config["cooldown_after_exit_tp_bars"] = cd1.number_input("익절(봉)", 0, 30, int(config.get("cooldown_after_exit_tp_bars", 1) or 0), step=1)
    config["cooldown_after_exit_sl_bars"] = cd2.number_input("손절(봉)", 0, 60, int(config.get("cooldown_after_exit_sl_bars", 3) or 0), step=1)
    config["cooldown_after_exit_protect_bars"] = cd3.number_input("본절(봉)", 0, 60, int(config.get("cooldown_after_exit_protect_bars", 2) or 0), step=1)
    st.caption("※ 현재 단기 타임프레임 기준 봉 개수입니다. (예: 5m에서 2봉=10분)")
config["use_dca"] = st.sidebar.checkbox("💧 물타기(DCA) (스윙 중심)", value=bool(config.get("use_dca", True)))
c3, c4 = st.sidebar.columns(2)
config["dca_trigger"] = c3.number_input("DCA 발동(%)", -90.0, -1.0, float(config.get("dca_trigger", -20.0)), step=0.5)
config["dca_max_count"] = c4.number_input("최대 횟수", 0, 10, int(config.get("dca_max_count", 1)))
config["dca_add_pct"] = st.sidebar.slider("추가 규모(원진입 대비 %)", 10, 200, int(config.get("dca_add_pct", 50)))
config["dca_add_usdt"] = st.sidebar.number_input(
    "추가 규모(USDT, 마진) [우선]",
    0.0,
    1000000.0,
    float(config.get("dca_add_usdt", 0.0) or 0.0),
    step=5.0,
    help="0보다 크면, DCA는 % 대신 이 USDT(마진) 금액을 사용합니다. (선물: qty≈(usdt*레버)/가격)",
)

st.sidebar.divider()
st.sidebar.subheader("🪙 외부 시황")
config["use_external_context"] = st.sidebar.checkbox("외부 시황 통합", value=bool(config.get("use_external_context", True)))
config["external_koreanize_enable"] = st.sidebar.checkbox("외부시황 한글화(가능한 범위)", value=bool(config.get("external_koreanize_enable", True)))
config["external_ai_translate_enable"] = st.sidebar.checkbox("외부시황 AI 번역(비용↑)", value=bool(config.get("external_ai_translate_enable", False)))
config["news_translate_budget_sec"] = st.sidebar.number_input(
    "뉴스 번역 시간예산(초, 0=룰기반만)",
    0,
    60,
    int(config.get("news_translate_budget_sec", 10)),
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("🌅 아침 브리핑")
config["daily_btc_brief_enable"] = st.sidebar.checkbox("매일 아침 BTC 경제뉴스 5개", value=bool(config.get("daily_btc_brief_enable", False)))
cc_b1, cc_b2 = st.sidebar.columns(2)
config["daily_btc_brief_hour_kst"] = cc_b1.number_input("시(KST)", 0, 23, int(config.get("daily_btc_brief_hour_kst", 9)))
config["daily_btc_brief_minute_kst"] = cc_b2.number_input("분(KST)", 0, 59, int(config.get("daily_btc_brief_minute_kst", 0)))

st.sidebar.divider()
st.sidebar.subheader("📤 일별 내보내기")
config["export_daily_enable"] = st.sidebar.checkbox("일별 내보내기 활성화", value=bool(config.get("export_daily_enable", True)))
config["export_excel_enable"] = st.sidebar.checkbox("Excel(xlsx) 저장", value=bool(config.get("export_excel_enable", False)))
config["export_gsheet_enable"] = st.sidebar.checkbox("Google Sheets 저장", value=bool(config.get("export_gsheet_enable", True)))

st.sidebar.divider()
st.sidebar.subheader("📊 보조지표 (12종) ON/OFF")
colA, colB = st.sidebar.columns(2)
config["use_rsi"] = colA.checkbox("RSI", value=bool(config.get("use_rsi", True)))
config["use_bb"] = colB.checkbox("볼린저", value=bool(config.get("use_bb", True)))
config["use_ma"] = colA.checkbox("MA(이평)", value=bool(config.get("use_ma", True)))
config["use_macd"] = colB.checkbox("MACD", value=bool(config.get("use_macd", True)))
config["use_stoch"] = colA.checkbox("스토캐스틱", value=bool(config.get("use_stoch", True)))
config["use_cci"] = colB.checkbox("CCI", value=bool(config.get("use_cci", True)))
config["use_mfi"] = colA.checkbox("MFI", value=bool(config.get("use_mfi", True)))
config["use_willr"] = colB.checkbox("윌리엄%R", value=bool(config.get("use_willr", True)))
config["use_adx"] = colA.checkbox("ADX", value=bool(config.get("use_adx", True)))
config["use_vol"] = colB.checkbox("거래량", value=bool(config.get("use_vol", True)))
config["use_sqz"] = colA.checkbox("SQZ(스퀴즈)", value=bool(config.get("use_sqz", True)))
config["use_chart_patterns"] = colB.checkbox("차트패턴", value=bool(config.get("use_chart_patterns", True)))

st.sidebar.divider()
st.sidebar.subheader("지표 파라미터")
r1, r2, r3 = st.sidebar.columns(3)
config["rsi_period"] = r1.number_input("RSI 기간", 5, 50, int(config.get("rsi_period", 14)))
config["rsi_buy"] = r2.number_input("과매도", 10, 50, int(config.get("rsi_buy", 30)))
config["rsi_sell"] = r3.number_input("과매수", 50, 90, int(config.get("rsi_sell", 70)))

b1, b2 = st.sidebar.columns(2)
config["bb_period"] = b1.number_input("BB 기간", 5, 50, int(config.get("bb_period", 20)))
config["bb_std"] = b2.number_input("BB 승수", 1.0, 5.0, float(config.get("bb_std", 2.0)))

m1, m2 = st.sidebar.columns(2)
config["ma_fast"] = m1.number_input("MA 단기", 3, 50, int(config.get("ma_fast", 7)))
config["ma_slow"] = m2.number_input("MA 장기", 50, 300, int(config.get("ma_slow", 99)))

with st.sidebar.expander("🔥 스퀴즈 모멘텀(SQZ) 설정"):
    config["sqz_dependency_enable"] = st.checkbox("SQZ 의존(진입 필터)", value=bool(config.get("sqz_dependency_enable", True)))
    config["sqz_dependency_gate_entry"] = st.checkbox("SQZ 중립이면 진입 억제", value=bool(config.get("sqz_dependency_gate_entry", True)))
    config["sqz_dependency_override_ai"] = st.checkbox("SQZ 반대면 AI 신호 무시", value=bool(config.get("sqz_dependency_override_ai", True)))
    config["sqz_dependency_weight"] = st.slider("SQZ 의존도(가중치)", 0.5, 1.0, float(config.get("sqz_dependency_weight", 0.80) or 0.80), step=0.05)
    c_sq1, c_sq2 = st.columns(2)
    config["sqz_mom_threshold_pct"] = c_sq1.number_input("모멘텀 기준(%)", 0.005, 1.0, float(config.get("sqz_mom_threshold_pct", 0.05) or 0.05), step=0.01)
    config["sqz_bb_length"] = c_sq2.number_input("BB 길이", 5, 80, int(config.get("sqz_bb_length", 20) or 20), step=1)
    c_sq3, c_sq4 = st.columns(2)
    config["sqz_bb_mult"] = c_sq3.number_input("BB 배수", 0.5, 6.0, float(config.get("sqz_bb_mult", 2.0) or 2.0), step=0.1)
    config["sqz_kc_length"] = c_sq4.number_input("KC 길이", 5, 80, int(config.get("sqz_kc_length", 20) or 20), step=1)
    c_sq5, c_sq6 = st.columns(2)
    config["sqz_kc_mult"] = c_sq5.number_input("KC 배수", 0.5, 6.0, float(config.get("sqz_kc_mult", 1.5) or 1.5), step=0.1)
    config["sqz_mom_length"] = c_sq6.number_input("모멘텀 길이", 5, 120, int(config.get("sqz_mom_length", 20) or 20), step=1)

with st.sidebar.expander("📐 차트 패턴 설정"):
    config["pattern_gate_entry"] = st.checkbox("패턴 반대면 진입 억제", value=bool(config.get("pattern_gate_entry", True)))
    config["pattern_override_ai"] = st.checkbox("강한 반대패턴이면 AI 신호 무시", value=bool(config.get("pattern_override_ai", True)))
    c_pt1, c_pt2 = st.columns(2)
    config["pattern_lookback"] = c_pt1.number_input("탐지 봉 수", 80, 800, int(config.get("pattern_lookback", 220) or 220), step=20)
    config["pattern_pivot_order"] = c_pt2.number_input("피벗 민감도", 2, 12, int(config.get("pattern_pivot_order", 4) or 4), step=1)
    c_pt3, c_pt4 = st.columns(2)
    config["pattern_tolerance_pct"] = c_pt3.number_input("고점/저점 허용오차(%)", 0.05, 3.0, float(config.get("pattern_tolerance_pct", 0.60) or 0.60), step=0.05)
    config["pattern_min_retrace_pct"] = c_pt4.number_input("최소 되돌림(%)", 0.05, 6.0, float(config.get("pattern_min_retrace_pct", 0.35) or 0.35), step=0.05)
    c_pt5, c_pt6 = st.columns(2)
    config["pattern_flat_slope_pct"] = c_pt5.number_input("수평기준 기울기(%/bar)", 0.003, 1.0, float(config.get("pattern_flat_slope_pct", 0.03) or 0.03), step=0.005)
    config["pattern_breakout_buffer_pct"] = c_pt6.number_input("이탈 확인 버퍼(%)", 0.0, 1.0, float(config.get("pattern_breakout_buffer_pct", 0.08) or 0.08), step=0.01)
    c_pt7, c_pt8 = st.columns(2)
    config["pattern_call_strength_min"] = c_pt7.number_input("AI호출 최소강도", 0.05, 1.0, float(config.get("pattern_call_strength_min", 0.45) or 0.45), step=0.05)
    config["pattern_gate_strength"] = c_pt8.number_input("진입차단 강도", 0.05, 1.0, float(config.get("pattern_gate_strength", 0.65) or 0.65), step=0.05)

st.sidebar.divider()
st.sidebar.subheader("🔍 긴급 점검")
if st.sidebar.button("📡 텔레그램 메뉴 전송(/menu)"):
    tg_send_menu(cfg=config)

if st.sidebar.button("🤖 OpenAI 연결 테스트"):
    # 운영자가 결제/쿼터를 복구한 직후 즉시 재시도할 수 있게 수동 clear
    openai_clear_suspension(config)
    h = openai_health_info(config)
    client = get_openai_client(config)
    if client is None:
        msg = str(h.get("message", "OpenAI 사용 불가")).strip()
        until = str(h.get("until_kst", "")).strip()
        if until:
            msg = f"{msg} (~{until} KST)"
        st.sidebar.error(f"❌ OpenAI 사용 불가: {msg}")
        if "insufficient_quota" in msg:
            st.sidebar.caption("OpenAI 결제/크레딧(Quota) 부족입니다. OpenAI 콘솔에서 Billing/크레딧을 확인하세요.")
        elif str(h.get("status")) == "NO_KEY":
            st.sidebar.caption("Streamlit secrets에 OPENAI_API_KEY를 설정하세요.")
    else:
        models_to_try = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
        last_err: Optional[BaseException] = None
        tried = []
        for m in models_to_try:
            tried.append(m)
            try:
                def _do():
                    return client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "테스트입니다. 1+1은?"}],
                        temperature=0.0,
                        max_tokens=16,
                    )

                resp = _call_with_timeout(_do, max(OPENAI_TIMEOUT_SEC, 30))
                out = (resp.choices[0].message.content or "").strip()
                st.sidebar.success(f"✅ 연결 성공({m}): {out}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                # quota/키오류면 더 시도해도 의미 없음
                kind = _openai_err_kind(e)
                openai_handle_failure(e, config, where="UI_OPENAI_TEST")
                if kind in ["insufficient_quota", "invalid_api_key"]:
                    break
                continue
        if last_err is not None:
            st.sidebar.error(f"❌ 실패: {last_err}")
            notify_admin_error("UI:OPENAI_TEST", last_err, context={"models_tried": tried})

# ✅ Google Sheets 연결 테스트(요구사항)
config["gsheet_auto_format_enable"] = st.sidebar.checkbox(
    "📊 Google Sheets 표 서식 자동 적용(권장)",
    value=bool(config.get("gsheet_auto_format_enable", True)),
    help="매매일지/시간대/일별 시트를 '표'처럼 보기 좋게 1회 자동 서식 적용합니다.",
)
if st.sidebar.button("📎 Google Sheets 연결 테스트"):
    try:
        res = gsheet_test_append_row(timeout_sec=25)
        if res.get("ok"):
            st.sidebar.success("✅ Google Sheets append_row 성공(GSHEET_TEST)")
        else:
            st.sidebar.error(f"❌ Google Sheets 실패: {res.get('error','')}")
        stg = gsheet_status_snapshot()
        email = str(stg.get("service_account_email", "")).strip()
        if email:
            st.sidebar.caption(f"서비스계정 이메일(시트 공유 필요): {email}")
        else:
            st.sidebar.caption("서비스계정 이메일을 읽지 못했어요(GSHEET_SERVICE_ACCOUNT_JSON 확인).")
        with st.sidebar.expander("Google Sheets 상태(디버그)"):
            st.json(stg)
    except Exception as e:
        st.sidebar.error(f"❌ 테스트 오류: {e}")
        notify_admin_error("UI:GSHEET_TEST", e, context={"code": CODE_VERSION})

# ✅ Google Sheets 표(서식) 강제 적용(요구사항)
if st.sidebar.button("📊 Google Sheets 표 서식 적용(강제)"):
    try:
        res = gsheet_apply_trades_only_format(force=True, timeout_sec=35)
        if res.get("ok"):
            st.sidebar.success("✅ 서식 적용 완료")
        else:
            st.sidebar.error(f"❌ 서식 적용 실패: {res.get('error','')}")
    except Exception as e:
        st.sidebar.error(f"❌ 서식 적용 오류: {e}")
        notify_admin_error("UI:GSHEET_FORMAT", e, context={"code": CODE_VERSION})

save_settings(config)

with st.sidebar:
    st.divider()
    st.header("내 지갑 현황")
    free, total = safe_fetch_balance(exchange)
    st.metric("총 자산(USDT)", f"{total:,.2f}")
    st.metric("주문 가능", f"{free:,.2f}")

    st.divider()
    st.subheader("보유 포지션(주요 5개)")
    try:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        if not act:
            st.caption("무포지션(관망)")
        else:
            for p in act[:5]:
                sym = p.get("symbol", "")
                side = position_side_normalize(p)
                roi = float(position_roi_percent(p))
                lev = p.get("leverage", "?")
                upnl = float(p.get("unrealizedPnl") or 0.0)
                emo = "🟢" if roi >= 0 else "🔴"
                st.info(f"**{emo} {sym}** ({'롱' if side=='long' else '숏'} x{lev})\nROI: **{roi:.2f}%** (PnL {upnl:.2f} USDT)")
    except Exception as e:
        st.error(f"포지션 조회 실패: {e}")


# =========================================================
# ✅ Main UI
# =========================================================
st.title("📈 비트겟 AI 워뇨띠 에이전트 (Final Integrated)")
st.caption("Streamlit=제어판/모니터링, Telegram=실시간 보고/조회. (모의투자 IS_SANDBOX=True)")

markets = exchange.markets or {}
if markets:
    symbol_list = [s for s in markets if markets[s].get("linear") and markets[s].get("swap")]
    if not symbol_list:
        symbol_list = TARGET_COINS
else:
    symbol_list = TARGET_COINS

symbol = st.selectbox("코인 선택", symbol_list, index=0)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("📉 TradingView 차트 (다크모드)")
    interval_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}
    render_tradingview(symbol, interval=interval_map.get(config.get("timeframe", "5m"), "5"), height=560)

with right:
    st.subheader("🧾 실시간 지표 요약")
    if ta is None and pta is None:
        st.error("ta/pandas_ta 모듈이 없습니다. requirements.txt에 `ta` 또는 `pandas_ta` 추가 후 재배포하세요.")
    else:
        try:
            ohlcv = safe_fetch_ohlcv(exchange, symbol, str(config.get("timeframe", "5m")), limit=220)
            if not ohlcv:
                raise RuntimeError("ohlcv_empty_or_timeout")
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df2, stt, last = calc_indicators(df, config)

            # 장기추세(1h)도 같이 표시
            htf_tf = "1h"
            htf_trend = get_htf_trend_cached(
                exchange,
                symbol,
                htf_tf,
                fast=int(config.get("ma_fast", 7)),
                slow=int(config.get("ma_slow", 99)),
                cache_sec=int(config.get("trend_filter_cache_sec", 60)),
            )

            if last is None:
                # 지표가 부족해도 장기추세/스타일은 표시(사용자 체감 개선)
                st.warning("지표 계산 실패(데이터 부족/지표 계산 오류)")
                # UI 표시에서는 OpenAI를 호출하지 않음(스트림릿 rerun/자동새로고침으로 비용 폭증 방지)
                style_hint = _style_for_entry(symbol, "buy", "", htf_trend, config, allow_ai=False)
                st.write(
                    {
                        "장기추세(1h)": f"🧭 {htf_trend}",
                        "추천 스타일(롱 관점)": f"{style_hint.get('style','-')} ({style_hint.get('confidence','-')}%)",
                        "상태": stt.get("_ERROR") or stt.get("_INFO") or "-",
                    }
                )
            else:
                st.metric("현재가", f"{float(last['close']):,.4f}")
                # 스타일 추천(현재 차트 기준)
                # UI 표시에서는 OpenAI를 호출하지 않음(스트림릿 rerun/자동새로고침으로 비용 폭증 방지)
                style_hint = _style_for_entry(symbol, "buy", stt.get("추세", ""), htf_trend, config, allow_ai=False)
                show = {
                    "단기추세(현재봉)": stt.get("추세", "-"),
                    "장기추세(1h)": f"🧭 {htf_trend}",
                    "추천 스타일(롱 관점)": f"{style_hint.get('style','-')} ({style_hint.get('confidence','-')}%)",
                    "RSI": stt.get("RSI", "-"),
                    "BB": stt.get("BB", "-"),
                    "MACD": stt.get("MACD", "-"),
                    "ADX": stt.get("ADX", "-"),
                    "거래량": stt.get("거래량", "-"),
                    "SQZ": stt.get("SQZ", "-"),
                    "차트패턴": stt.get("패턴", "-"),
                    "눌림목후보(해소)": "✅" if stt.get("_pullback_candidate") else "—",
                    "지표엔진": stt.get("_backend", "-"),
                }
                st.write(show)

                if config.get("use_sr_stop", True):
                    try:
                        sr_tf = config.get("sr_timeframe", "15m")
                        sr_lb = int(config.get("sr_lookback", 220))
                        htf = safe_fetch_ohlcv(exchange, symbol, str(sr_tf), limit=sr_lb)
                        if not htf:
                            raise RuntimeError("sr_ohlcv_empty_or_timeout")
                        hdf = pd.DataFrame(htf, columns=["time", "open", "high", "low", "close", "vol"])
                        hdf["time"] = pd.to_datetime(hdf["time"], unit="ms")
                        sr = sr_stop_take(
                            entry_price=float(last["close"]),
                            side="buy",
                            htf_df=hdf,
                            atr_period=int(config.get("sr_atr_period", 14)),
                            pivot_order=int(config.get("sr_pivot_order", 6)),
                            buffer_atr_mult=float(config.get("sr_buffer_atr_mult", 0.25)),
                            rr_min=float(config.get("sr_rr_min", 1.5)),
                        )
                        if sr:
                            st.caption(f"SR(참고): 롱 기준 TP {sr['tp_price']:.6g} / SL {sr['sl_price']:.6g}")
                    except Exception as e:
                        notify_admin_error("UI:SR_CALC", e, context={"symbol": symbol, "tf": str(config.get("sr_timeframe", ""))}, min_interval_sec=120.0)

        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            notify_admin_error("UI:INDICATOR_SUMMARY", e, context={"symbol": symbol, "tf": str(config.get("timeframe", ""))})

st.divider()

# ✅ 메인 화면에서도 바로 보이는 스캔 요약(요구: 5개 코인 분석 과정을 실시간으로 확인)
try:
    with st.expander("🔎 실시간 스캔 요약(최근)", expanded=False):
        mon0 = read_json_safe(MONITOR_FILE, {}) or {}
        coins0 = mon0.get("coins", {}) or {}
        rows0 = []
        for sym0 in TARGET_COINS:
            cs0 = (coins0.get(sym0) or {}) if isinstance(coins0, dict) else {}
            rows0.append(
                {
                    "코인": sym0,
                    "단계": cs0.get("scan_stage", ""),
                    "단계시각": cs0.get("scan_stage_kst", ""),
                    "AI": str(cs0.get("ai_decision", "-")).upper() if cs0 else "-",
                    "확신": cs0.get("ai_confidence", ""),
                    "스킵/근거": (cs0.get("skip_reason") or cs0.get("ai_reason_easy") or "")[:60],
                }
            )
        st_dataframe_safe(df_for_display(pd.DataFrame(rows0)), hide_index=True)

        scan0 = mon0.get("scan_process") or []
        if scan0:
            n0 = st.number_input("최근 로그 개수", 10, 200, 40, step=10, key="main_scan_n")
            df0 = pd.DataFrame(list(scan0)[-int(n0) :]).iloc[::-1].reset_index(drop=True)
            st_dataframe_safe(df_for_display(df0), hide_index=True)
        else:
            st.caption("SCAN 로그 없음(봇이 아직 스캔을 시작하지 않았거나 파일 접근 문제일 수 있음)")
except Exception:
    pass

tabs = st.tabs(["🤖 자동매매 & AI시야", "⚡ 수동주문", "📅 시장정보", "📜 매매일지", "🧪 간이 백테스트"])
t1, t2, t3, t4, t5 = tabs

with t1:
    st.subheader("👁️ 실시간 AI 모니터링(봇 시야)")
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, key="mon_refresh")
    else:
        st.caption("자동 새로고침을 원하면 requirements.txt에 streamlit-autorefresh 추가")

    # ✅ 모니터 파일 진단(사용자 환경에서 UI가 안 바뀌는 문제를 빨리 찾기 위함)
    try:
        mon_abs = os.path.abspath(MONITOR_FILE)
        mon_exists = os.path.exists(mon_abs)
        mon_mtime = os.path.getmtime(mon_abs) if mon_exists else 0.0
        mon_size = os.path.getsize(mon_abs) if mon_exists else 0
        try:
            th_names = [t.name for t in threading.enumerate()]
            th_core = [n for n in th_names if any(k in n for k in ["TG_THREAD", "TG_POLL_THREAD", "WATCHDOG_THREAD", "GSHEET_THREAD"])]
        except Exception:
            th_core = []
        mon_diag = {
            "path": mon_abs,
            "exists": mon_exists,
            "mtime_kst": _epoch_to_kst_str(mon_mtime) if mon_mtime else "",
            "size_bytes": mon_size,
            "last_read_error": (_READ_JSON_LAST_ERROR.get(MONITOR_FILE) or _READ_JSON_LAST_ERROR.get(mon_abs) or "").strip(),
            "threads": th_core,
        }
        with st.expander("🧪 monitor_state.json 진단(출력/갱신 문제 확인)", expanded=False):
            st.write(mon_diag)
            if mon_diag.get("last_read_error"):
                st.code(mon_diag.get("last_read_error", "")[:1200])
            if st.button("monitor_state.json tail 보기", key="btn_mon_tail"):
                try:
                    with open(mon_abs, "r", encoding="utf-8") as f:
                        txt = f.read()
                    st.code(txt[-2500:] if len(txt) > 2500 else txt)
                except Exception as e:
                    st.error(f"tail 읽기 실패: {e}")
    except Exception:
        pass

    mon = read_json_safe(MONITOR_FILE, None)
    if not mon:
        st.warning("monitor_state.json이 아직 없습니다. (스레드 시작 확인)")
    else:
        # ✅ Google Sheets 상태(요구사항)
        try:
            st.subheader("📎 Google Sheets 상태")
            st.write(gsheet_status_snapshot())
            st.caption("※ 서비스계정 이메일로 스프레드시트를 '편집자'로 공유해야 append 됩니다.")
        except Exception:
            pass
        # 외부 시황 요약(항상 보이게)
        st.subheader("🌍 외부 시황 요약(한글/이모티콘)")
        ext = (mon.get("external") or {})
        if not ext or not ext.get("enabled", False):
            st.caption("외부 시황 통합 OFF")
        else:
            st.write(
                {
                    "갱신시각(KST)": ext.get("asof_kst"),
                    "중요이벤트(임박)": len(ext.get("high_impact_events_soon") or []),
                    "공포탐욕": (ext.get("fear_greed") or {}),
                    "도미넌스/시총": (ext.get("global") or {}),
                    "아침브리핑": (ext.get("daily_btc_brief") or {}),
                    "진입감산배수": mon.get("entry_risk_multiplier", 1.0),
                }
            )
            evs = ext.get("high_impact_events_soon") or []
            if evs:
                st.warning("⚠️ 중요 이벤트 임박(신규진입 보수적으로)")
                st_dataframe_safe(df_for_display(pd.DataFrame(evs)), hide_index=True)
            hd = ext.get("headlines") or []
            if hd:
                st.caption("뉴스 헤드라인(요약용)")
                st.write(hd[:10])

        hb = float(mon.get("last_heartbeat_epoch", 0))
        age = (time.time() - hb) if hb else 9999
        try:
            scan_cycle_sec = float(mon.get("scan_cycle_sec", 0) or 0)
        except Exception:
            scan_cycle_sec = 0.0
        # 요구사항: heartbeat lag가 scan_interval*4 이상이면 '멈춤 의심'
        stale_thresh = max(60.0, float(scan_cycle_sec) * 4.0) if scan_cycle_sec > 0 else 60.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("자동매매", "ON" if mon.get("auto_trade") else "OFF")
        c2.metric("모드", mon.get("trade_mode", "-"))
        c3.metric("하트비트", f"{age:.1f}초 전", "🟢 작동중" if age < stale_thresh else "🔴 멈춤 의심")
        c4.metric("연속손실", str(mon.get("consec_losses", 0)))

        if age >= stale_thresh:
            st.error(f"⚠️ 봇 스레드가 멈췄거나(크래시) 갱신이 안될 수 있어요. ({stale_thresh:.0f}초 이상)")

        st.caption(
            f"봇 상태: {mon.get('global_state','-')} | stage: {mon.get('loop_stage','-')}@{mon.get('loop_stage_kst','-')} | code: {mon.get('code_version','-')}"
        )
        try:
            h = openai_health_info(load_settings())
            ai_txt = "OK" if bool(h.get("available", False)) else str(h.get("message", "OFF"))
            until = str(h.get("until_kst", "")).strip()
            if until and (not bool(h.get("available", False))):
                ai_txt = f"{ai_txt} (~{until} KST)"
            st.caption(f"OpenAI: {ai_txt}")
        except Exception:
            pass

        # ✅ 포지션/진입 정보(직관적 표시)
        st.subheader("📊 현재 포지션(스타일/목표 포함)")
        ops = mon.get("open_positions") or []
        if ops:
            st.caption(f"현재 포지션 수: {len(ops)}")
            st_dataframe_safe(df_for_display(pd.DataFrame(ops)), hide_index=True)
        else:
            st.caption("⚪ 포지션 없음(관망)")

        # ✅ 최근 이벤트(가독성 강화)
        st.subheader("🧾 최근 이벤트(봇 로그)")
        evs = (mon.get("events") or [])[-30:]
        if evs:
            st_dataframe_safe(df_for_display(pd.DataFrame(evs[::-1])), hide_index=True)
        else:
            st.caption("이벤트 없음")

        # ✅ AI/Scan Process (요구사항: 단계별 스캔 과정 표시)
        st.subheader("🧠 AI/Scan Process (최근)")
        scan_logs = (mon.get("scan_process") or [])
        if scan_logs:
            max_show = st.number_input("표시 개수(N)", 20, 400, 120, step=10)
            try:
                df_scan = pd.DataFrame(scan_logs[-int(max_show):])
                df_scan = df_scan.iloc[::-1].reset_index(drop=True)
                st_dataframe_safe(df_for_display(df_scan), hide_index=True)
            except Exception:
                st_dataframe_safe(df_for_display(pd.DataFrame(scan_logs[-int(max_show):][::-1])), hide_index=True)
        else:
            st.caption("SCAN 로그 없음")

        rows = []
        coins = mon.get("coins", {}) or {}
        for sym, cs in coins.items():
            last_scan = float(cs.get("last_scan_epoch", 0) or 0)
            scan_age = (time.time() - last_scan) if last_scan else 9999
            rows.append(
                {
                    "코인": sym,
                    "스캔(초전)": f"{scan_age:.1f}",
                    "스캔단계": cs.get("scan_stage", ""),
                    "단계시각": cs.get("scan_stage_kst", ""),
                    "가격": cs.get("price", ""),
                    "단기추세": cs.get("trend_short", ""),
                    "장기추세(1h)": cs.get("trend_long", ""),
                    "추천스타일": cs.get("style_reco", ""),
                    "스타일확신": cs.get("style_confidence", ""),
                    "RSI": cs.get("rsi", ""),
                    "ADX": cs.get("adx", ""),
                    "BB": cs.get("bb", ""),
                    "MACD": cs.get("macd", ""),
                    "SQZ": cs.get("sqz", ""),
                    "패턴": cs.get("pattern", ""),
                    "패턴방향": cs.get("pattern_bias", ""),
                    "패턴강도": cs.get("pattern_strength", ""),
                    "수렴(ML)": str(cs.get("ml_dir", ""))[:10],
                    "표(ML)": cs.get("ml_votes", ""),
                    "ML상세": (cs.get("ml_detail", "") or "")[:120],
                    "눌림목후보": "✅" if cs.get("pullback_candidate") else "—",
                    "AI호출": "✅" if cs.get("ai_called") else "—",
                    "AI결론": str(cs.get("ai_decision", "-")).upper(),
                    "확신도": cs.get("ai_confidence", "-"),
                    "필요확신도": cs.get("min_conf_required", "-"),
                    "진입%": cs.get("ai_entry_pct", "-"),
                    "레버": cs.get("ai_leverage", "-"),
                    "SL%": cs.get("ai_sl_pct", "-"),
                    "TP%": cs.get("ai_tp_pct", "-"),
                    "손익비": cs.get("ai_rr", "-"),
                    "AI지표": cs.get("ai_used", ""),
                    "스킵/근거": (cs.get("skip_reason") or cs.get("ai_reason_easy") or "")[:160],
                }
            )
        if rows:
            st_dataframe_safe(df_for_display(pd.DataFrame(rows)), hide_index=True)
        else:
            st.info("아직 스캔 데이터가 없습니다.")

    st.divider()
    st.subheader("🔍 현재 코인 AI 분석(수동 버튼)")
    if st.button("현재 코인 AI 분석 실행"):
        # 수동 실행은 운영자가 즉시 재시도할 수 있게 suspend를 클리어
        openai_clear_suspension(config)
        if get_openai_client(config) is None:
            h = openai_health_info(config)
            msg = str(h.get("message", "OpenAI 사용 불가")).strip()
            until = str(h.get("until_kst", "")).strip()
            if until:
                msg = f"{msg} (~{until} KST)"
            st.error(msg)
        elif ta is None and pta is None:
            st.error("ta/pandas_ta 모듈 없음")
        else:
            try:
                # 외부시황은 비동기 스냅샷 사용(수동 버튼이 UI를 멈추지 않게)
                try:
                    external_context_refresh_maybe(load_settings(), load_runtime(), force=True)
                except Exception:
                    pass
                ext_now = external_context_snapshot()
                ohlcv = safe_fetch_ohlcv(exchange, symbol, str(config.get("timeframe", "5m")), limit=220)
                if not ohlcv:
                    raise RuntimeError("ohlcv_empty_or_timeout")
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if last is None:
                    st.warning("지표 계산 실패")
                else:
                    # 수동 분석에서도 주력 지표 수렴(ML) 정보를 AI에 제공
                    try:
                        ml0 = ml_signals_and_convergence(df2, stt, config, cache_key="")
                        stt["_ml_signals"] = dict(ml0) if isinstance(ml0, dict) else {}
                    except Exception:
                        pass
                    ai = ai_decide_trade(df2, stt, symbol, config.get("trade_mode", "안전모드"), config, external=ext_now)
                    # 스타일 힌트
                    htf_trend = get_htf_trend_cached(exchange, symbol, "1h", int(config.get("ma_fast", 7)), int(config.get("ma_slow", 99)), int(config.get("trend_filter_cache_sec", 60)))
                    # 수동 분석에서도 스타일 힌트는 룰 기반만 사용(불필요한 추가 OpenAI 호출 방지)
                    style_info = _style_for_entry(symbol, ai.get("decision", "hold"), stt.get("추세", ""), htf_trend, config, allow_ai=False)
                    st.json({"ai": ai, "style": style_info, "htf_trend": htf_trend})
            except Exception as e:
                st.error(f"분석 오류: {e}")
                notify_admin_error("UI:MANUAL_AI_ANALYSIS", e, context={"symbol": symbol, "tf": str(config.get("timeframe", ""))})

with t2:
    st.subheader("⚡ 수동 주문(데모용)")
    st.caption("⚠️ 수동 주문은 실수 방지를 위해 기본은 '설명/테스트' 중심입니다.")
    amt = st.number_input("주문 금액(USDT)", 0.0, 100000.0, float(config.get("order_usdt", 100.0)))
    config["order_usdt"] = float(amt)
    save_settings(config)

    enable_manual = st.checkbox("수동 주문 활성화(주의!)", value=False)
    b1, b2, b3 = st.columns(3)

    if b1.button("🟢 롱 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "buy", qty)
            st.success("롱 진입 성공" if ok else "롱 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b2.button("🔴 숏 진입") and enable_manual:
        px = get_last_price(exchange, symbol)
        free, _ = safe_fetch_balance(exchange)
        if px and amt > 0 and amt < free:
            lev = MODE_RULES[config["trade_mode"]]["lev_min"]
            set_leverage_safe(exchange, symbol, lev)
            qty = to_precision_qty(exchange, symbol, (amt * lev) / px)
            ok = market_order_safe(exchange, symbol, "sell", qty)
            st.success("숏 진입 성공" if ok else "숏 진입 실패")
        else:
            st.warning("잔고/가격/금액 확인 필요")

    if b3.button("🚫 전량 청산") and enable_manual:
        ps = safe_fetch_positions(exchange, TARGET_COINS)
        act = [p for p in ps if float(p.get("contracts") or 0) > 0]
        for p in act:
            sym = p.get("symbol", "")
            side = position_side_normalize(p)
            contracts = float(p.get("contracts") or 0)
            close_position_market(exchange, sym, side, contracts)
        st.success("전량 청산 요청 완료(데모)")

with t3:
    st.subheader("📅 시장정보(외부 시황)")
    try:
        cex1, cex2 = st.columns([1, 4])
        if cex1.button("🔄 강제갱신"):
            try:
                external_context_refresh_maybe(load_settings(), load_runtime(), force=True)
            except Exception:
                pass
        ext = external_context_snapshot()
        if not (ext or {}).get("enabled"):
            st.info("외부 시황 통합 OFF")
            st.caption(
                f"asof={ext.get('asof_kst','-')} | inflight={ext.get('_inflight','-')} | age_sec={ext.get('_age_sec','-')} | err={ext.get('_last_err','') or ext.get('error','')}"
            )
        else:
            st.json(ext)
    except Exception as e:
        st.error(f"시장정보 로딩 오류: {e}")
        notify_admin_error("UI:MARKET_INFO", e, min_interval_sec=120.0)

with t4:
    st.subheader("📜 매매일지 (이모티콘/색상 + 일별 내보내기)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    if c1.button("🔄 새로고침"):
        st.rerun()
    if c2.button("🧹 매매일지 초기화"):
        reset_trade_log()
        st.success("매매일지 초기화 완료")
        st.rerun()
    if c3.button("📤 오늘 일지 내보내기"):
        try:
            res = export_trade_log_daily(today_kst_str(), config)
            if res.get("ok"):
                st.success(f"내보내기 완료: rows={res.get('rows')} | xlsx={res.get('excel_path','')} | csv={res.get('csv_path','')}")
            else:
                st.error(f"내보내기 실패: {res.get('error','')}")
        except Exception as e:
            st.error(f"내보내기 오류: {e}")
            notify_admin_error("UI:EXPORT_TODAY", e, min_interval_sec=120.0)
    if c4.button("🧨 구글시트 매매일지 초기화(오늘부터)"):
        try:
            res = gsheet_reset_trades_only(timeout_sec=55)
            if bool(res.get("ok", False)):
                st.success(f"구글시트 초기화 완료 ✅ (reset_kst={res.get('reset_kst','')})")
            else:
                st.error(f"구글시트 초기화 실패: {res.get('error','')}")
        except Exception as e:
            st.error(f"구글시트 초기화 오류: {e}")
            notify_admin_error("UI:GSHEET_RESET", e, min_interval_sec=120.0)

    df_log = read_trade_log()
    if df_log.empty:
        st.info("아직 기록된 매매가 없습니다.")
    else:
        # 표시 개선: 이모티콘 + 색상
        df_show = df_log.copy()
        try:
            df_show["PnL_Percent"] = pd.to_numeric(df_show.get("PnL_Percent"), errors="coerce")
            df_show["PnL_USDT"] = pd.to_numeric(df_show.get("PnL_USDT"), errors="coerce")
            df_show.insert(
                0,
                "상태",
                df_show["PnL_Percent"].apply(lambda v: "🟢" if pd.notna(v) and float(v) > 0 else ("🔴" if pd.notna(v) and float(v) < 0 else "⚪")),
            )
        except Exception:
            pass

        # ✅ 잔고(진입 전/청산 후)까지 같이 보기(요구사항: 직관화)
        show_cols = [
            c
            for c in [
                "상태",
                "Time",
                "Coin",
                "Side",
                "Entry",
                "Exit",
                "PnL_Percent",
                "PnL_USDT",
                "BalanceBefore_Total",
                "BalanceAfter_Total",
                "BalanceBefore_Free",
                "BalanceAfter_Free",
                "OneLine",
                "Reason",
                "Review",
                "TradeID",
            ]
            if c in df_show.columns
        ]

        def _color_pnl(v):
            try:
                x = float(v)
            except Exception:
                return ""
            if x > 0:
                return "background-color: rgba(0, 200, 0, 0.18); color: #00c853;"
            if x < 0:
                return "background-color: rgba(220, 0, 0, 0.18); color: #ff1744;"
            return ""

        try:
            # pandas 2.2+: Styler.applymap deprecate → map 사용
            sty0 = df_show[show_cols].style
            if hasattr(sty0, "map"):
                sty = sty0.map(_color_pnl, subset=["PnL_Percent", "PnL_USDT"])  # type: ignore[attr-defined]
            else:
                sty = sty0.applymap(_color_pnl, subset=["PnL_Percent", "PnL_USDT"])
            st_dataframe_safe(sty, hide_index=True)
        except Exception:
            st_dataframe_safe(df_for_display(df_show[show_cols]), hide_index=True)

        csv_bytes = df_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV 다운로드", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

        df_s = None
        try:
            df_s = df_log.copy()
            df_s["PnL_USDT"] = pd.to_numeric(df_s.get("PnL_USDT"), errors="coerce").fillna(0.0)
            df_s["PnL_Percent"] = pd.to_numeric(df_s.get("PnL_Percent"), errors="coerce").fillna(0.0)
        except Exception:
            df_s = None

        # ✅ 코인별 승률/손익 요약 + 손절 패턴(요구사항)
        try:
            st.divider()
            st.subheader("📊 코인별 성적(승률/손익)")
            if df_s is None:
                raise RuntimeError("trade_log_parse_failed")
            df_s["is_win"] = df_s["PnL_Percent"] > 0
            g = df_s.groupby("Coin", dropna=False)
            df_coin = (
                g.agg(
                    거래수=("PnL_Percent", "count"),
                    승률_=("is_win", "mean"),
                    총손익_USDT=("PnL_USDT", "sum"),
                    평균수익률_=("PnL_Percent", "mean"),
                    최대손실_=("PnL_Percent", "min"),
                    최대수익_=("PnL_Percent", "max"),
                )
                .reset_index()
            )
            df_coin["승률(%)"] = (df_coin["승률_"].astype(float) * 100.0).round(1)
            df_coin = df_coin.drop(columns=["승률_"])
            df_coin = df_coin.sort_values(["총손익_USDT", "승률(%)"], ascending=[False, False])
            st_dataframe_safe(df_for_display(df_coin), hide_index=True)

            st.subheader("🧩 손절/익절 사유 분포(최근 기준)")
            if "Reason" in df_s.columns:
                df_reason = df_s.copy()
                df_reason["Reason"] = df_reason["Reason"].astype(str).fillna("")
                df_reason["사유"] = df_reason["Reason"].str.slice(0, 60)
                df_r = df_reason.groupby("사유").size().reset_index(name="건수").sort_values("건수", ascending=False).head(25)
                st_dataframe_safe(df_for_display(df_r), hide_index=True)
        except Exception as e:
            st.warning(f"코인별 요약/패턴 분석 실패: {e}")

        try:
            st.divider()
            st.subheader("🧠 회고 모음(손실만)")
            if df_s is None:
                raise RuntimeError("trade_log_parse_failed")
            df_l = df_s[(df_s["PnL_Percent"] < 0)].copy()
            if df_l.empty:
                st.caption("손실 기록이 아직 없어요.")
            else:
                cols_l = [c for c in ["Time", "Coin", "Side", "PnL_Percent", "PnL_USDT", "Reason", "OneLine", "Review", "TradeID"] if c in df_l.columns]
                # 후기(review)가 비어있으면 표시에서 제외(가독성)
                try:
                    df_l["Review"] = df_l.get("Review", "").astype(str)
                    df_l2 = df_l[df_l["Review"].str.strip() != ""]
                except Exception:
                    df_l2 = df_l
                if df_l2.empty:
                    st.caption("후기(Review)가 저장된 손실 기록이 아직 없어요.")
                else:
                    st_dataframe_safe(df_for_display(df_l2[cols_l].head(80)), hide_index=True)
        except Exception:
            pass

    st.divider()
    st.subheader("🧾 상세일지 조회(TradeID)")
    tid = st.text_input("TradeID 입력 (텔레그램 '일지'에 ID가 나옵니다)")
    if st.button("상세일지 열기"):
        if not tid.strip():
            st.warning("TradeID를 입력해줘.")
        else:
            d = load_trade_detail(tid.strip())
            if not d:
                st.error("해당 ID를 찾지 못했어.")
            else:
                st.json(d)

    st.divider()
    st.subheader("📌 runtime_state.json (현재 상태)")
    rt = load_runtime()
    st.json(rt)
    if st.button("🧼 runtime_state 초기화(오늘 기준)"):
        write_json_atomic(RUNTIME_FILE, default_runtime())
        st.success("runtime_state.json 초기화 완료")
        st.rerun()

with t5:
    st.subheader("🧪 간이 백테스트(가벼운 규칙 기반, 버튼 실행형)")
    st.caption("실제 주문이 아니라 과거 OHLCV로 '대략' 성능을 확인합니다. (기본 OFF, 클릭 시 실행)")

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    bt_symbol = bt_col1.selectbox("심볼", symbol_list, index=symbol_list.index(symbol) if symbol in symbol_list else 0)
    bt_tf = bt_col2.selectbox("타임프레임", ["1m", "3m", "5m", "15m", "1h"], index=["1m", "3m", "5m", "15m", "1h"].index(config.get("timeframe", "5m")))
    bt_n = bt_col3.number_input("최근 N봉", 200, 2000, 600, step=50)

    bt_style = st.selectbox("전략 스타일", ["스캘핑", "스윙"], index=0)
    run_bt = st.button("▶️ 백테스트 실행")

    if run_bt:
        if ta is None and pta is None:
            st.error("ta/pandas_ta 모듈 없음")
        else:
            try:
                ohlcv = safe_fetch_ohlcv(exchange, bt_symbol, str(bt_tf), limit=int(bt_n))
                if not ohlcv:
                    raise RuntimeError("ohlcv_empty_or_timeout")
                df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df2, stt, last = calc_indicators(df, config)
                if df2 is None or df2.empty:
                    st.error("데이터 부족")
                else:
                    # 간이 시뮬: RSI 해소 + MA 추세 기반
                    trades = []
                    in_pos = False
                    side = None
                    entry_px = 0.0
                    peak = 0.0
                    equity = 0.0
                    max_equity = 0.0
                    max_dd = 0.0

                    # 스타일별 목표(대략)
                    tp = 1.8 if bt_style == "스캘핑" else 6.0
                    sl = 1.2 if bt_style == "스캘핑" else 3.0

                    for i in range(2, len(df2)):
                        row = df2.iloc[i]
                        prev = df2.iloc[i - 1]
                        price = float(row["close"])

                        # 간이 신호
                        trend = "횡보/전환"
                        if "MA_fast" in df2.columns and "MA_slow" in df2.columns:
                            if float(row["MA_fast"]) > float(row["MA_slow"]) and price > float(row["MA_slow"]):
                                trend = "상승추세"
                            elif float(row["MA_fast"]) < float(row["MA_slow"]) and price < float(row["MA_slow"]):
                                trend = "하락추세"
                        rsi_prev = float(prev.get("RSI", 50))
                        rsi_now = float(row.get("RSI", 50))
                        rsi_buy = float(config.get("rsi_buy", 30))
                        rsi_sell = float(config.get("rsi_sell", 70))

                        rsi_resolve_long = (rsi_prev < rsi_buy) and (rsi_now >= rsi_buy)
                        rsi_resolve_short = (rsi_prev > rsi_sell) and (rsi_now <= rsi_sell)

                        if not in_pos:
                            if trend == "상승추세" and rsi_resolve_long:
                                in_pos = True
                                side = "long"
                                entry_px = price
                                peak = price
                            elif trend == "하락추세" and rsi_resolve_short:
                                in_pos = True
                                side = "short"
                                entry_px = price
                                peak = price
                        else:
                            # ROI 계산(레버 무시, 단순 퍼센트)
                            if side == "long":
                                roi = ((price - entry_px) / entry_px) * 100.0
                                peak = max(peak, price)
                            else:
                                roi = ((entry_px - price) / entry_px) * 100.0
                                peak = min(peak, price)

                            if roi >= tp or roi <= -sl:
                                trades.append(roi)
                                equity += roi
                                max_equity = max(max_equity, equity)
                                max_dd = min(max_dd, equity - max_equity)
                                in_pos = False
                                side = None
                                entry_px = 0.0

                    if trades:
                        wins = sum(1 for x in trades if x > 0)
                        win_rate = wins / len(trades) * 100.0
                        gains = sum(x for x in trades if x > 0)
                        losses = -sum(x for x in trades if x < 0)
                        pf = gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0
                        total_ret = sum(trades)
                        avg_r = float(np.mean(trades))
                        st.metric("총 수익률(단순합)", f"{total_ret:.2f}%")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("거래수", str(len(trades)))
                        c2.metric("승률", f"{win_rate:.1f}%")
                        c3.metric("PF", f"{pf:.2f}" if pf != float("inf") else "inf")
                        c4.metric("MDD(단순)", f"{max_dd:.2f}%")
                        st.caption(f"평균 R(간이): {avg_r:.2f}")
                        st.write(pd.DataFrame({"trade_roi_pct": trades}).tail(50))
                    else:
                        st.warning("조건에 맞는 거래가 없었습니다.")
            except Exception as e:
                st.error(f"백테스트 오류: {e}")
                notify_admin_error("UI:BACKTEST", e, context={"symbol": bt_symbol, "tf": bt_tf, "n": int(bt_n)}, min_interval_sec=120.0)


st.caption("⚠️ 이 봇은 모의투자(IS_SANDBOX=True)에서 충분히 검증 후 사용하세요.")


# =========================================================
# ✅ [중요] 모의투자 → 실전 전환 방법(자동 전환 절대 없음, 사용자가 직접 변경)
# =========================================================
# 1) 이 파일 상단의 IS_SANDBOX = True 를 False 로 변경
#    - IS_SANDBOX = False
# 2) Bitget 실계정 API 키를 Streamlit Secrets에 설정:
#    - (요구사항) BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
#    - (호환) API_KEY / API_SECRET / API_PASSWORD 도 자동 인식
#    - (권한) 선물(SWAP) 주문/포지션 조회 권한 필요
# 3) Telegram 채널/그룹 분리를 원하면 Secrets에 추가:
#    - (요구사항) TG_TOKEN / TG_TARGET_CHAT_ID
#    - (확장) TG_GROUP_ID / TG_CHANNEL_ID (있으면 자동 라우팅: 채널=알림/하트비트, 그룹=명령)
#      * 채널로 보내려면 봇이 채널 관리자여야 합니다.
# 4) 실전 전에는 반드시:
#    - 주문 수량/레버/SL/TP 로직을 소액으로 점검
#    - 거래소 최소수량/정밀도/슬리피지/수수료 고려
#    - 예기치 못한 버그/네트워크 장애 대비(위험 제한, 손실 감내 범위 설정)
# =========================================================

# =========================================================
# ✅ 검증 체크리스트(요구사항)
# =========================================================
# - Streamlit 실행 시 UI가 정상 표시되는가?
# - TG_TOKEN/TG_TARGET_CHAT_ID 설정 시 메시지가 정상 발송되는가?
#   - 채널 사용 시 봇을 채널 관리자(게시 권한)로 추가해야 함.
# - Telegram 명령이 동작하는가?
#   - /status (누구나)
#   - /positions /scan /mode auto|scalping|swing /log <id> (관리자: TG_ADMIN_USER_IDS 설정 시 제한)
# - GSHEET_ENABLED="true"일 때 Google Sheets에 append_row가 동작하는가?
#   - GSHEET_SERVICE_ACCOUNT_JSON 을 json.loads로 읽음
#   - GSHEET_SPREADSHEET_ID / GSHEET_WORKSHEET 로 워크시트 열고 없으면 생성
#   - 서비스계정 이메일로 스프레드시트 공유 필요
#   - TRADE/EVENT/SCAN 로그가 누적되는가?
# - 레짐 전환에 시간락(style_lock_minutes) 강제가 없는가?
#   - confirm2/hysteresis/off 로만 흔들림 제어
# - 기존 기능이 삭제되지 않았는가? (Streamlit 탭/수동주문/일지/백테스트/외부시황/내보내기 등)
# =========================================================
