import os, re, json, time, uuid, math, threading, traceback, socket, csv
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
    def add_script_run_ctx(_th): return None

import ccxt

try:
    from openai import OpenAI as _OpenAIClient
    OPENAI_AVAILABLE = True
except Exception:
    _OpenAIClient = None
    OPENAI_AVAILABLE = False

try:
    import ta
except Exception:
    ta = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    import orjson
except Exception:
    orjson = None

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
    import openpyxl
except Exception:
    openpyxl = None

try:
    import gspread
    from google.oauth2.service_account import Credentials as GoogleCredentials
except Exception:
    gspread = None
    GoogleCredentials = None

try:
    from diskcache import Cache
except Exception:
    Cache = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from loguru import logger
except Exception:
    logger = None

try:
    socket.setdefaulttimeout(15)
except Exception:
    pass

st.set_page_config(layout="wide", page_title="Bitget AI Bot v2")

IS_SANDBOX = True

SETTINGS_FILE = "bot_settings.json"
RUNTIME_FILE = "runtime_state.json"
LOG_FILE = "trade_log.csv"
LOSS_REVIEW_FILE = "loss_review.csv"
MONITOR_FILE = "monitor_state.json"
AI_INSIGHTS_FILE = "ai_insights.json"
DETAIL_DIR = "trade_details"
DAILY_REPORT_DIR = "daily_reports"
os.makedirs(DETAIL_DIR, exist_ok=True)
os.makedirs(DAILY_REPORT_DIR, exist_ok=True)

_cache = Cache("cache") if Cache else None

TARGET_COINS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

HTTP_TIMEOUT_SEC = 12
EXTERNAL_CONTEXT_TIMEOUT_SEC = 16
OPENAI_TIMEOUT_SEC = 25
CCXT_TIMEOUT_SEC_PUBLIC = 12
CCXT_TIMEOUT_SEC_PRIVATE = 15
WORKER_LEASE_TTL_SEC = 90.0

_THREAD_POOL = ThreadPoolExecutor(max_workers=4)
_CCXT_POOL = ThreadPoolExecutor(max_workers=2)
_CCXT_POOL_LOCK = threading.RLock()
_CCXT_CB_LOCK = threading.RLock()
_CCXT_CB_UNTIL_EPOCH = 0.0
_CCXT_CB_REASON = ""
_CCXT_TIMEOUT_EPOCHS = deque(maxlen=12)
_CCXT_CB_OPEN_AFTER_TIMEOUTS = 3
_CCXT_CB_WINDOW_SEC = 60.0
_CCXT_CB_COOLDOWN_SEC = 45.0

KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(KST)

def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")

def _epoch_to_kst_str(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return now_kst_str()

def _parse_time_kst(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except Exception:
        return None

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "" or v != v:
            return default
        return float(v)
    except Exception:
        return default

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _json_default(obj: Any):
    try:
        if isinstance(obj, datetime): return obj.isoformat()
    except Exception: pass
    try:
        if isinstance(obj, (set, tuple)): return list(obj)
    except Exception: pass
    try:
        if isinstance(obj, np.generic): return obj.item()
    except Exception: pass
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
                try: opt |= orjson.OPT_SERIALIZE_NUMPY
                except Exception: pass
                try: opt |= orjson.OPT_NON_STR_KEYS
                except Exception: pass
                f.write(orjson.dumps(data, default=_json_default, option=opt))
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            pass

def read_json_safe(path: str, default=None):
    try:
        if orjson:
            with open(path, "rb") as f:
                return orjson.loads(f.read())
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_json_dumps(x: Any, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        try: s = str(x)
        except Exception: s = ""
    return s[:limit] + "..." if len(s) > limit else s

def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    try:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].apply(
                    lambda v: safe_json_dumps(v, 400) if isinstance(v, (dict, list))
                    else ("" if v is None else str(v))
                )
        return out
    except Exception:
        try: return df.astype(str)
        except Exception: return pd.DataFrame()

def st_dataframe_safe(data, **kwargs):
    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        df = df_for_display(df)
        kw = {k: v for k, v in kwargs.items()}
        kw.setdefault("use_container_width", True)
        try:
            st.dataframe(df, **kw)
        except TypeError:
            kw2 = {k: v for k, v in kw.items() if k in ("height", "use_container_width")}
            st.dataframe(df, **kw2)
    except Exception as e:
        st.warning(f"DataFrame 표시 오류: {e}")

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
        cnt = sum(1 for t0 in list(_CCXT_TIMEOUT_EPOCHS) if (now_ts - float(t0)) <= _CCXT_CB_WINDOW_SEC)
        if cnt >= _CCXT_CB_OPEN_AFTER_TIMEOUTS:
            _ccxt_cb_open(reason=f"timeout_burst({where})", duration_sec=_CCXT_CB_COOLDOWN_SEC)
    except Exception:
        pass

def _ccxt_call_with_timeout(fn, timeout_sec: int, where: str = ""):
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
            msg = str(e).lower()
            if "cannot schedule" in msg or "shutdown" in msg:
                _CCXT_POOL = ThreadPoolExecutor(max_workers=2)
                fut = _CCXT_POOL.submit(fn)
            else:
                raise
    finally:
        try: _CCXT_POOL_LOCK.release()
        except Exception: pass
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
            "circuit_open": time.time() < until if until else False,
            "circuit_until_kst": _epoch_to_kst_str(until) if until else "",
            "circuit_reason": reason,
            "timeouts_recent": len(list(_CCXT_TIMEOUT_EPOCHS)),
        }
    except Exception:
        return {"circuit_open": False}

def default_settings() -> Dict[str, Any]:
    return {
        "bitget_api_key": "",
        "bitget_api_secret": "",
        "bitget_passphrase": "",
        "tg_token": "",
        "tg_chat_id": "",
        "tg_channel_id": "",
        "tg_admin_ids": "",
        "tg_simple_msg": True,
        "openai_api_key": "",
        "openai_model_fast": "gpt-4o-mini",
        "openai_model_deep": "gpt-4o",
        "gsheet_enabled": False,
        "gsheet_spreadsheet_id": "",
        "gsheet_service_account_json": "",
        "target_coins": TARGET_COINS,
        "leverage": 10,
        "max_positions": 3,
        "position_size_pct": 10.0,
        "mode": "auto",
        "style": "scalp",
        "taker_fee_rate": 0.0006,
        "scalp_tp_safety_mul": 3.0,
        "scalp_sl_pct": 0.8,
        "swing_tp_pct": 3.0,
        "swing_sl_pct": 1.5,
        "trailing_stop_enabled": True,
        "trailing_start_roi": 50.0,
        "trailing_lock_pct": 0.3,
        "circuit_breaker_enabled": True,
        "circuit_breaker_n": 5,
        "circuit_breaker_pause_min": 30,
        "daily_loss_limit_pct": 5.0,
        "daily_loss_limit_enabled": True,
        "ai_params_enabled": True,
        "ai_params_cache_min": 30,
        "periodic_report_interval_min": 30,
        "loss_review_batch_n": 5,
        "htf_tf": "1h",
        "htf_fast_ma": 7,
        "htf_slow_ma": 99,
        "scan_interval_sec": 60,
        "scan_tf": "5m",
        "pivot_order": 6,
        "atr_period": 14,
        "use_fear_greed": True,
        "use_news": True,
        "fear_greed_bear_threshold": 25,
        "fear_greed_bull_threshold": 75,
    }

def load_settings() -> Dict[str, Any]:
    saved = read_json_safe(SETTINGS_FILE, {})
    base = default_settings()
    base.update({k: v for k, v in saved.items() if k in base})
    try:
        from streamlit import secrets as _sec
        _map = {
            "BITGET_API_KEY": "bitget_api_key",
            "BITGET_API_SECRET": "bitget_api_secret",
            "BITGET_PASSPHRASE": "bitget_passphrase",
            "TG_TOKEN": "tg_token",
            "TG_CHAT_ID": "tg_chat_id",
            "OPENAI_API_KEY": "openai_api_key",
        }
        for sk, dk in _map.items():
            try:
                v = _sec.get(sk) or _sec.get(sk.lower())
                if v: base[dk] = str(v)
            except Exception:
                pass
    except Exception:
        pass
    return base

def save_settings(cfg: Dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, cfg)

def default_runtime() -> Dict[str, Any]:
    return {
        "running": False,
        "positions": {},
        "last_heartbeat": "",
        "consecutive_losses": 0,
        "daily_loss_pct": 0.0,
        "daily_loss_date": "",
        "circuit_breaker_until": 0.0,
        "circuit_breaker_reason": "",
        "daily_stopped": False,
        "worker_id": "",
        "worker_leases": {},
        "revoked_worker_ids": [],
        "ai_params_cache": {},
        "ai_params_cache_time": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "coin_stats": {},
        "ai_call_count_today": 0,
        "ai_call_date": "",
    }

def load_runtime() -> Dict[str, Any]:
    saved = read_json_safe(RUNTIME_FILE, {})
    base = default_runtime()
    base.update({k: v for k, v in saved.items() if k in base})
    return base

def save_runtime(rt: Dict[str, Any]) -> None:
    write_json_atomic(RUNTIME_FILE, rt)

def _runtime_revoked_ids(rt: Dict[str, Any]) -> List[str]:
    v = rt.get("revoked_worker_ids", [])
    return list(v) if isinstance(v, (list, tuple)) else []

def runtime_is_worker_revoked(worker_id: str) -> bool:
    try:
        rt = load_runtime()
        return worker_id in _runtime_revoked_ids(rt)
    except Exception:
        return False

def runtime_worker_lease_touch(worker_id: str, owner: str = "TG_THREAD", ttl_sec: float = WORKER_LEASE_TTL_SEC) -> bool:
    try:
        rt = load_runtime()
        if worker_id in _runtime_revoked_ids(rt):
            return False
        leases = rt.get("worker_leases", {})
        now_ts = time.time()
        for wid, lease in list(leases.items()):
            if wid == worker_id: continue
            exp = _as_float(lease.get("expires_at", 0))
            if now_ts < exp and lease.get("owner") == owner:
                return False
        lease_entry = leases.get(worker_id, {})
        lease_entry["owner"] = owner
        lease_entry["expires_at"] = now_ts + ttl_sec
        lease_entry["last_touch"] = now_kst_str()
        leases[worker_id] = lease_entry
        rt["worker_leases"] = leases
        save_runtime(rt)
        return True
    except Exception:
        return True

def runtime_worker_revoke(worker_id: str, reason: str = "") -> None:
    try:
        rt = load_runtime()
        ids = _runtime_revoked_ids(rt)
        if worker_id not in ids:
            ids.append(worker_id)
        rt["revoked_worker_ids"] = ids[-50:]
        leases = rt.get("worker_leases", {})
        leases.pop(worker_id, None)
        rt["worker_leases"] = leases
        save_runtime(rt)
    except Exception:
        pass

_CB_LOCK = threading.RLock()
_CB_STATE: Dict[str, Any] = {}

def cb_is_paused() -> Tuple[bool, str]:
    try:
        with _CB_LOCK:
            until = _as_float(_CB_STATE.get("until", 0))
            reason = _CB_STATE.get("reason", "")
        if time.time() < until:
            return True, reason
        return False, ""
    except Exception:
        return False, ""

def cb_activate(reason: str, pause_min: int) -> None:
    try:
        with _CB_LOCK:
            _CB_STATE["until"] = time.time() + pause_min * 60
            _CB_STATE["reason"] = reason
            _CB_STATE["activated_at"] = now_kst_str()
    except Exception:
        pass

def cb_reset() -> None:
    try:
        with _CB_LOCK:
            _CB_STATE.clear()
    except Exception:
        pass

def cb_status() -> Dict[str, Any]:
    try:
        with _CB_LOCK:
            until = _as_float(_CB_STATE.get("until", 0))
            return {
                "paused": time.time() < until,
                "until_kst": _epoch_to_kst_str(until) if until else "",
                "reason": _CB_STATE.get("reason", ""),
            }
    except Exception:
        return {"paused": False, "until_kst": "", "reason": ""}

_DAILY_LOCK = threading.RLock()
_DAILY_STATE: Dict[str, Any] = {}

def daily_loss_check_and_update(pnl_usdt: float, balance: float, cfg: Dict[str, Any]) -> bool:
    if not cfg.get("daily_loss_limit_enabled", True):
        return False
    try:
        with _DAILY_LOCK:
            today = today_kst_str()
            if _DAILY_STATE.get("date") != today:
                _DAILY_STATE["date"] = today
                _DAILY_STATE["loss_usdt"] = 0.0
                _DAILY_STATE["stopped"] = False
            if pnl_usdt < 0:
                _DAILY_STATE["loss_usdt"] = _as_float(_DAILY_STATE.get("loss_usdt", 0)) + abs(pnl_usdt)
            if balance > 0:
                loss_pct = _DAILY_STATE["loss_usdt"] / balance * 100
            else:
                loss_pct = 0.0
            limit = _as_float(cfg.get("daily_loss_limit_pct", 5.0))
            if loss_pct >= limit and not _DAILY_STATE.get("stopped"):
                _DAILY_STATE["stopped"] = True
                return True
            return False
    except Exception:
        return False

def daily_loss_status() -> Dict[str, Any]:
    try:
        with _DAILY_LOCK:
            return dict(_DAILY_STATE)
    except Exception:
        return {}

def daily_loss_reset() -> None:
    try:
        with _DAILY_LOCK:
            _DAILY_STATE.clear()
    except Exception:
        pass

LOG_COLS = [
    "trade_id", "timestamp", "symbol", "side", "style",
    "leverage", "entry_price", "exit_price", "qty",
    "pnl_usdt", "pnl_pct", "roi_pct",
    "sl_price", "tp_price", "sl_pct", "tp_pct",
    "exit_reason", "duration_sec",
    "trend_short", "trend_long",
    "fear_greed", "ai_score",
    "review", "review_summary",
]

LOSS_REVIEW_COLS = [
    "trade_id", "timestamp", "symbol", "side", "style",
    "entry_price", "exit_price", "pnl_usdt", "roi_pct",
    "exit_reason", "market_context",
    "rule_analysis", "ai_review", "reviewed_at",
]

def _ensure_csv(path: str, cols: List[str]) -> None:
    if not os.path.exists(path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(cols)
        except Exception:
            pass

def log_trade(
    trade_id: str, symbol: str, side: str, style: str,
    leverage: float, entry_price: float, exit_price: float, qty: float,
    pnl_usdt: float, pnl_pct: float, roi_pct: float,
    sl_price: float, tp_price: float, sl_pct: float, tp_pct: float,
    exit_reason: str, duration_sec: float,
    trend_short: str = "", trend_long: str = "",
    fear_greed: Any = "", ai_score: Any = "",
    review: str = "", review_summary: str = "",
) -> None:
    _ensure_csv(LOG_FILE, LOG_COLS)
    row = [
        trade_id, now_kst_str(), symbol, side, style,
        leverage, entry_price, exit_price, qty,
        round(pnl_usdt, 4), round(pnl_pct, 4), round(roi_pct, 4),
        sl_price, tp_price, sl_pct, tp_pct,
        exit_reason, round(duration_sec, 1),
        trend_short, trend_long,
        fear_greed, ai_score,
        review, review_summary,
    ]
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass

def log_loss_review(
    trade_id: str, symbol: str, side: str, style: str,
    entry_price: float, exit_price: float, pnl_usdt: float, roi_pct: float,
    exit_reason: str, market_context: str = "",
    rule_analysis: str = "", ai_review: str = "",
) -> None:
    _ensure_csv(LOSS_REVIEW_FILE, LOSS_REVIEW_COLS)
    row = [
        trade_id, now_kst_str(), symbol, side, style,
        entry_price, exit_price, round(pnl_usdt, 4), round(roi_pct, 4),
        exit_reason, market_context,
        rule_analysis, ai_review, now_kst_str(),
    ]
    try:
        with open(LOSS_REVIEW_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass

def read_trade_log() -> pd.DataFrame:
    _ensure_csv(LOG_FILE, LOG_COLS)
    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8")
        if df.empty: return df
        for c in ["pnl_usdt", "roi_pct", "pnl_pct", "leverage"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame(columns=LOG_COLS)

def read_loss_review_log() -> pd.DataFrame:
    _ensure_csv(LOSS_REVIEW_FILE, LOSS_REVIEW_COLS)
    try:
        df = pd.read_csv(LOSS_REVIEW_FILE, encoding="utf-8")
        return df
    except Exception:
        return pd.DataFrame(columns=LOSS_REVIEW_COLS)

def reset_trade_log() -> None:
    try:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_COLS)
    except Exception:
        pass

def get_coin_stats(df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    if df is None:
        df = read_trade_log()
    if df.empty:
        return {}
    stats = {}
    for sym in df["symbol"].unique() if "symbol" in df.columns else []:
        sub = df[df["symbol"] == sym]
        wins = len(sub[sub["pnl_usdt"] > 0])
        losses = len(sub[sub["pnl_usdt"] <= 0])
        total = len(sub)
        pnl = sub["pnl_usdt"].sum() if "pnl_usdt" in sub.columns else 0
        stats[sym] = {
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl": round(pnl, 4),
        }
    return stats

def get_recent_losses(n: int = 10) -> pd.DataFrame:
    df = read_trade_log()
    if df.empty: return df
    if "pnl_usdt" in df.columns:
        return df[df["pnl_usdt"] < 0].tail(n)
    return pd.DataFrame()

def rule_based_loss_analysis(
    symbol: str, side: str, entry_price: float, exit_price: float,
    roi_pct: float, exit_reason: str,
    trend_short: str = "", trend_long: str = "",
    fear_greed: Any = "", style: str = "",
) -> str:
    lines = []
    if roi_pct < -2.0:
        lines.append("과도한 손실: SL 비율이 너무 큼")
    trend_conflict = False
    if side == "long" and trend_short in ("down", "bear", "하락"):
        trend_conflict = True
    if side == "short" and trend_short in ("up", "bull", "상승"):
        trend_conflict = True
    if trend_conflict:
        lines.append(f"추세 불일치: {side} 포지션인데 단기추세={trend_short}")
    if isinstance(fear_greed, (int, float)):
        fg = int(fear_greed)
        if side == "long" and fg < 25:
            lines.append(f"극단적 공포 구간({fg})에서 롱 진입")
        if side == "short" and fg > 75:
            lines.append(f"극단적 탐욕 구간({fg})에서 숏 진입")
    if exit_reason in ("sl_hit", "stop_loss"):
        lines.append("손절 트리거 - 진입 타이밍 또는 SL 위치 재검토 필요")
    if not lines:
        lines.append("특이 패턴 없음 - 시장 변동성에 의한 손실로 추정")
    return " | ".join(lines)

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

_AI_LOCK = threading.RLock()
_AI_SUSPEND_UNTIL = 0.0
_AI_SUSPEND_REASON = ""
_AI_CALL_COUNT = 0
_AI_CALL_DATE = ""

def _get_openai_client(cfg: Dict[str, Any]) -> Optional[Any]:
    if not OPENAI_AVAILABLE:
        return None
    key = cfg.get("openai_api_key", "").strip()
    if not key:
        return None
    try:
        return _OpenAIClient(api_key=key)
    except Exception:
        return None

def _ai_is_suspended() -> bool:
    try:
        with _AI_LOCK:
            return time.time() < _AI_SUSPEND_UNTIL
    except Exception:
        return False

def _ai_suspend(reason: str, sec: int = 300) -> None:
    try:
        with _AI_LOCK:
            global _AI_SUSPEND_UNTIL, _AI_SUSPEND_REASON
            _AI_SUSPEND_UNTIL = time.time() + sec
            _AI_SUSPEND_REASON = reason
    except Exception:
        pass

def _ai_track_call() -> None:
    try:
        with _AI_LOCK:
            global _AI_CALL_COUNT, _AI_CALL_DATE
            today = today_kst_str()
            if _AI_CALL_DATE != today:
                _AI_CALL_DATE = today
                _AI_CALL_COUNT = 0
            _AI_CALL_COUNT += 1
    except Exception:
        pass

def gemini_call(prompt: str, cfg: Dict[str, Any], model_key: str = "openai_model_fast", timeout_sec: int = OPENAI_TIMEOUT_SEC) -> str:
    if _ai_is_suspended():
        return ""
    client = _get_openai_client(cfg)
    if client is None:
        return ""
    model_name = cfg.get(model_key, "gpt-4o-mini")
    try:
        def _call():
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                timeout=timeout_sec,
            )
            return resp.choices[0].message.content.strip()
        fut = _THREAD_POOL.submit(_call)
        result = fut.result(timeout=timeout_sec + 5)
        _ai_track_call()
        return result or ""
    except FuturesTimeoutError:
        _ai_suspend("timeout", 120)
        return ""
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "429" in err or "rate" in err or "insufficient_quota" in err:
            _ai_suspend("rate_limit", 600)
        elif "api_key" in err or "api key" in err or "invalid" in err:
            _ai_suspend("invalid_key", 3600)
        return ""

def gemini_call_json(prompt: str, cfg: Dict[str, Any], model_key: str = "openai_model_fast") -> Dict[str, Any]:
    raw = gemini_call(prompt, cfg, model_key)
    if not raw:
        return {}
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {}

def gemini_health_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with _AI_LOCK:
            susp = time.time() < _AI_SUSPEND_UNTIL
            reason = _AI_SUSPEND_REASON
            count = _AI_CALL_COUNT
            date = _AI_CALL_DATE
        return {
            "available": OPENAI_AVAILABLE,
            "key_set": bool(cfg.get("openai_api_key", "").strip()),
            "suspended": susp,
            "suspend_reason": reason,
            "call_count_today": count,
            "call_date": date,
        }
    except Exception:
        return {"available": False}

def gemini_batch_loss_review(cfg: Dict[str, Any]) -> str:
    df = read_loss_review_log()
    if df.empty:
        return ""
    unreviewed = df[df.get("ai_review", pd.Series([""] * len(df))).fillna("") == ""] if "ai_review" in df.columns else df
    batch_n = _as_int(cfg.get("loss_review_batch_n", 5))
    if len(unreviewed) < batch_n:
        return ""
    recent = unreviewed.tail(batch_n)
    lines = []
    for _, row in recent.iterrows():
        lines.append(
            f"- {row.get('symbol','')} {row.get('side','')} | ROI:{row.get('roi_pct','')}% | "
            f"사유:{row.get('exit_reason','')} | 컨텍스트:{row.get('market_context','')}"
        )
    prompt = (
        "당신은 암호화폐 선물 트레이더 AI 코치입니다.\n"
        f"최근 {batch_n}건의 손절 내역을 분석하고, 공통 패턴, 개선 방안, 핵심 교훈을 한국어로 3-4문장으로 요약하세요.\n\n"
        "손절 내역:\n" + "\n".join(lines) + "\n\n"
        "JSON 형식으로 반환하세요: {\"patterns\": \"...\", \"improvements\": \"...\", \"lesson\": \"...\"}"
    )
    result = gemini_call_json(prompt, cfg, model_key="openai_model_deep")
    if result:
        summary = f"패턴: {result.get('patterns','')}\n개선: {result.get('improvements','')}\n교훈: {result.get('lesson','')}"
        try:
            insights = read_json_safe(AI_INSIGHTS_FILE, {"reviews": []})
            insights["reviews"].append({
                "timestamp": now_kst_str(),
                "summary": summary,
                "trade_count": batch_n,
            })
            insights["reviews"] = insights["reviews"][-50:]
            write_json_atomic(AI_INSIGHTS_FILE, insights)
        except Exception:
            pass
        return summary
    return ""

def init_exchange(cfg: Dict[str, Any]) -> Optional[Any]:
    key = cfg.get("bitget_api_key", "").strip()
    secret = cfg.get("bitget_api_secret", "").strip()
    pw = cfg.get("bitget_passphrase", "").strip()
    if not key or not secret:
        return None
    try:
        ex = ccxt.bitget({
            "apiKey": key,
            "secret": secret,
            "password": pw,
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })
        if IS_SANDBOX:
            ex.set_sandbox_mode(True)
        ex.load_markets()
        return ex
    except Exception:
        return None

_EX_LOCK = threading.RLock()
_EX_INSTANCE = None
_EX_CREATED_AT = 0.0
_EX_TTL = 300.0

def get_exchange(cfg: Dict[str, Any]) -> Optional[Any]:
    global _EX_INSTANCE, _EX_CREATED_AT
    with _EX_LOCK:
        if _EX_INSTANCE is not None and (time.time() - _EX_CREATED_AT) < _EX_TTL:
            return _EX_INSTANCE
        _EX_INSTANCE = init_exchange(cfg)
        _EX_CREATED_AT = time.time()
        return _EX_INSTANCE

def safe_fetch_balance(ex) -> Tuple[float, float]:
    try:
        def _f(): return ex.fetch_balance({"type": "swap"})
        b = _ccxt_call_with_timeout(_f, CCXT_TIMEOUT_SEC_PRIVATE, "fetch_balance")
        usdt = b.get("USDT", {})
        total = _as_float(usdt.get("total", 0))
        free = _as_float(usdt.get("free", 0))
        return total, free
    except Exception:
        return 0.0, 0.0

def safe_fetch_positions(ex, symbols: List[str]) -> List[Dict[str, Any]]:
    try:
        def _f(): return ex.fetch_positions(symbols)
        raw = _ccxt_call_with_timeout(_f, CCXT_TIMEOUT_SEC_PRIVATE, "fetch_positions")
        return [p for p in (raw or []) if _as_float(p.get("contracts", 0)) > 0]
    except Exception:
        return []

def get_last_price(ex, sym: str) -> Optional[float]:
    try:
        def _f(): return ex.fetch_ticker(sym)
        t = _ccxt_call_with_timeout(_f, CCXT_TIMEOUT_SEC_PUBLIC, "fetch_ticker")
        return _as_float(t.get("last", 0)) or None
    except Exception:
        return None

def safe_fetch_ohlcv(ex, sym: str, tf: str, limit: int = 220) -> Optional[List]:
    try:
        def _f(): return ex.fetch_ohlcv(sym, tf, limit=limit)
        data = _ccxt_call_with_timeout(_f, CCXT_TIMEOUT_SEC_PUBLIC, "fetch_ohlcv")
        return data if data and len(data) >= 10 else None
    except Exception:
        return None

def ohlcv_to_df(data: List) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    return df.dropna(subset=["close"])

def to_precision_qty(ex, sym: str, qty: float) -> float:
    try:
        market = ex.market(sym)
        precision = market.get("precision", {}).get("amount", 3)
        factor = 10 ** int(precision)
        return math.floor(qty * factor) / factor
    except Exception:
        return round(qty, 3)

def set_leverage_safe(ex, sym: str, lev: int) -> None:
    try:
        ex.set_leverage(lev, sym, params={"marginMode": "isolated"})
    except Exception:
        pass

def market_order_safe(ex, sym: str, side: str, qty: float) -> bool:
    try:
        ex.create_market_order(sym, side, qty)
        return True
    except Exception:
        return False

def close_position_market(ex, sym: str, pos_side: str, contracts: float) -> bool:
    try:
        side = "sell" if pos_side == "long" else "buy"
        ex.create_market_order(sym, side, contracts, params={
            "reduceOnly": True,
            "positionSide": pos_side.upper(),
        })
        return True
    except Exception:
        return False

def position_roi_percent(p: Dict[str, Any]) -> float:
    try:
        pnl = _as_float(p.get("unrealizedPnl", 0))
        margin = _as_float(p.get("initialMargin", 0)) or _as_float(p.get("collateral", 0))
        if margin <= 0: return 0.0
        return pnl / margin * 100
    except Exception:
        return 0.0

def position_side_normalize(p: Dict[str, Any]) -> str:
    s = str(p.get("side", "") or p.get("positionSide", "")).lower()
    if s in ("long", "buy", "net"): return "long"
    if s in ("short", "sell"): return "short"
    c = _as_float(p.get("contracts", 0))
    return "long" if c > 0 else "short"

def _pos_leverage(p: Dict[str, Any]) -> float:
    v = _as_float(p.get("leverage", 1))
    return v if v > 0 else 1.0

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        if len(tr) < period: return float(tr.mean()) if len(tr) > 0 else 0.0
        return float(tr[-period:].mean())
    except Exception:
        return 0.0

def calc_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.Series]]:
    signals = {}
    last = None
    try:
        df = df.copy()
        c = df["close"]
        df["ma7"] = c.rolling(7).mean()
        df["ma25"] = c.rolling(25).mean()
        df["ma99"] = c.rolling(99).mean()
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["bb_mid"] = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * std20
        df["bb_lower"] = df["bb_mid"] - 2 * std20
        if ta:
            try:
                atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], c, window=14)
                df["atr"] = atr_ind.average_true_range()
            except Exception:
                df["atr"] = 0.0
        else:
            df["atr"] = 0.0
        last = df.iloc[-1]
        rsi_v = _as_float(last.get("rsi", 50))
        macd_v = _as_float(last.get("macd", 0))
        macd_sig = _as_float(last.get("macd_signal", 0))
        ma7_v = _as_float(last.get("ma7", 0))
        ma25_v = _as_float(last.get("ma25", 0))
        ma99_v = _as_float(last.get("ma99", 0))
        signals["rsi"] = rsi_v
        signals["macd"] = macd_v
        signals["macd_hist"] = _as_float(last.get("macd_hist", 0))
        signals["ma7"] = ma7_v
        signals["ma25"] = ma25_v
        signals["ma99"] = ma99_v
        signals["atr"] = _as_float(last.get("atr", 0))
        bull_signals = sum([
            rsi_v > 55,
            macd_v > macd_sig,
            ma7_v > ma25_v,
            ma25_v > ma99_v,
        ])
        bear_signals = sum([
            rsi_v < 45,
            macd_v < macd_sig,
            ma7_v < ma25_v,
            ma25_v < ma99_v,
        ])
        if bull_signals >= 3: signals["overall"] = "bull"
        elif bear_signals >= 3: signals["overall"] = "bear"
        else: signals["overall"] = "neutral"
    except Exception:
        pass
    return df, signals, last

def compute_ma_trend(df: pd.DataFrame, fast: int = 7, slow: int = 99) -> str:
    try:
        c = df["close"]
        if len(c) < slow + 2: return "neutral"
        fast_ma = c.rolling(fast).mean().iloc[-1]
        slow_ma = c.rolling(slow).mean().iloc[-1]
        if fast_ma > slow_ma * 1.001: return "up"
        if fast_ma < slow_ma * 0.999: return "down"
        return "neutral"
    except Exception:
        return "neutral"

_HTF_CACHE: Dict[str, Any] = {}
_HTF_LOCK = threading.RLock()

def get_htf_trend_cached(ex, sym: str, tf: str, fast: int, slow: int, cache_sec: int = 60) -> str:
    key = f"{sym}_{tf}_{fast}_{slow}"
    with _HTF_LOCK:
        entry = _HTF_CACHE.get(key, {})
        if time.time() - _as_float(entry.get("ts", 0)) < cache_sec:
            return entry.get("trend", "neutral")
    data = safe_fetch_ohlcv(ex, sym, tf, limit=max(slow + 20, 120))
    if not data:
        return "neutral"
    df = ohlcv_to_df(data)
    trend = compute_ma_trend(df, fast, slow)
    with _HTF_LOCK:
        _HTF_CACHE[key] = {"trend": trend, "ts": time.time()}
    return trend

def calc_min_tp_roi(leverage: float, fee_rate: float = 0.0006) -> float:
    round_trip_fee_roi = fee_rate * 2 * leverage
    return round_trip_fee_roi * 100 * 3

def pivot_levels(df: pd.DataFrame, order: int = 6, max_levels: int = 12) -> Tuple[List[float], List[float]]:
    supports, resistances = [], []
    try:
        highs = df["high"].values
        lows = df["low"].values
        n = len(highs)
        if n < order * 2 + 1:
            return [], []
        if argrelextrema:
            res_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
            sup_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        else:
            res_idx = [i for i in range(order, n - order) if highs[i] == max(highs[i-order:i+order+1])]
            sup_idx = [i for i in range(order, n - order) if lows[i] == min(lows[i-order:i+order+1])]
        resistances = sorted(set(round(highs[i], 6) for i in res_idx), reverse=True)[:max_levels]
        supports = sorted(set(round(lows[i], 6) for i in sup_idx))[:max_levels]
    except Exception:
        pass
    return supports, resistances

def sr_stop_take(
    entry_price: float, side: str, supports: List[float], resistances: List[float],
    atr: float, cfg: Dict[str, Any], style: str = "scalp",
    ai_params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    leverage = _as_float(cfg.get("leverage", 10), 10)
    fee_rate = _as_float(cfg.get("taker_fee_rate", 0.0006), 0.0006)
    min_tp_roi = calc_min_tp_roi(leverage, fee_rate)
    if ai_params and ai_params.get("tp_roi") and ai_params.get("sl_roi"):
        tp_roi = _as_float(ai_params["tp_roi"])
        sl_roi = _as_float(ai_params["sl_roi"])
        if tp_roi >= min_tp_roi and sl_roi > 0:
            tp_price_pct = tp_roi / leverage / 100
            sl_price_pct = sl_roi / leverage / 100
            if side == "long":
                return entry_price * (1 - sl_price_pct), entry_price * (1 + tp_price_pct)
            else:
                return entry_price * (1 + sl_price_pct), entry_price * (1 - tp_price_pct)
    if style == "scalp":
        sl_pct = _as_float(cfg.get("scalp_sl_pct", 0.8)) / 100
        safety = _as_float(cfg.get("scalp_tp_safety_mul", 3.0))
        tp_roi_target = max(min_tp_roi, min_tp_roi * safety)
        tp_pct = tp_roi_target / leverage / 100
    else:
        sl_pct = _as_float(cfg.get("swing_sl_pct", 1.5)) / 100
        tp_pct = _as_float(cfg.get("swing_tp_pct", 3.0)) / 100
    if side == "long":
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + tp_pct)
        if supports:
            near_sup = [s for s in supports if s < entry_price]
            if near_sup:
                sr_sl = max(near_sup) * 0.998
                sl = max(sl, sr_sl) if style == "scalp" else min(sl, sr_sl)
        if resistances:
            near_res = [r for r in resistances if r > entry_price]
            if near_res:
                sr_tp = min(near_res) * 0.998
                if sr_tp > entry_price * (1 + calc_min_tp_roi(leverage, fee_rate) / leverage / 100):
                    tp = min(tp, sr_tp)
    else:
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - tp_pct)
        if resistances:
            near_res = [r for r in resistances if r > entry_price]
            if near_res:
                sr_sl = min(near_res) * 1.002
                sl = min(sl, sr_sl) if style == "scalp" else max(sl, sr_sl)
        if supports:
            near_sup = [s for s in supports if s < entry_price]
            if near_sup:
                sr_tp = max(near_sup) * 1.002
                if sr_tp < entry_price * (1 - calc_min_tp_roi(leverage, fee_rate) / leverage / 100):
                    tp = max(tp, sr_tp)
    return sl, tp

_SR_CACHE: Dict[str, Any] = {}
_SR_LOCK = threading.RLock()

def get_sr_levels_cached(ex, sym: str, tf: str, pivot_order: int = 6, cache_sec: int = 60) -> Dict[str, Any]:
    key = f"{sym}_{tf}_{pivot_order}"
    with _SR_LOCK:
        entry = _SR_CACHE.get(key, {})
        if time.time() - _as_float(entry.get("ts", 0)) < cache_sec:
            return entry
    data = safe_fetch_ohlcv(ex, sym, tf, limit=220)
    if not data:
        return {"supports": [], "resistances": [], "atr": 0.0}
    df = ohlcv_to_df(data)
    supports, resistances = pivot_levels(df, order=pivot_order)
    atr = calc_atr(df)
    result = {"supports": supports, "resistances": resistances, "atr": atr, "ts": time.time()}
    with _SR_LOCK:
        _SR_CACHE[key] = result
    return result

def calc_trailing_sl(
    entry_price: float, side: str, current_roi: float,
    trail_start_roi: float, lock_pct: float, leverage: float,
) -> Optional[float]:
    if current_roi < trail_start_roi:
        return None
    locked_roi = current_roi * lock_pct
    if locked_roi <= 0:
        return None
    price_move_pct = locked_roi / leverage / 100
    if side == "long":
        return entry_price * (1 + price_move_pct)
    else:
        return entry_price * (1 - price_move_pct)

def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT_SEC):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def fetch_fear_greed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = _http_get_json("https://api.alternative.me/fng/?limit=1")
        if data and data.get("data"):
            d = data["data"][0]
            return {"value": int(d.get("value", 50)), "label": d.get("value_classification", "")}
    except Exception:
        pass
    return {"value": 50, "label": "Neutral"}

def fetch_coingecko_global() -> Dict[str, Any]:
    try:
        data = _http_get_json("https://api.coingecko.com/api/v3/global")
        if data and data.get("data"):
            d = data["data"]
            return {
                "market_cap_change_24h": round(_as_float(d.get("market_cap_change_percentage_24h_usd", 0)), 2),
                "btc_dominance": round(_as_float(d.get("market_cap_percentage", {}).get("btc", 0)), 2),
            }
    except Exception:
        pass
    return {}

def fetch_news_headlines_rss(cfg: Dict[str, Any], max_items: int = 8) -> List[str]:
    if not feedparser:
        return []
    feeds = [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ]
    headlines = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items // len(feeds) + 1]:
                title = entry.get("title", "").strip()
                if title:
                    headlines.append(title)
        except Exception:
            pass
    return headlines[:max_items]

_EXT_CTX_LOCK = threading.RLock()
_EXT_CTX_STATE: Dict[str, Any] = {}
_EXT_CTX_LAST_UPDATE = 0.0
_EXT_CTX_UPDATE_INTERVAL = 300.0

def build_external_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    try:
        if cfg.get("use_fear_greed", True):
            ctx["fear_greed"] = fetch_fear_greed(cfg)
    except Exception:
        pass
    try:
        ctx["global"] = fetch_coingecko_global()
    except Exception:
        pass
    try:
        if cfg.get("use_news", True):
            ctx["news"] = fetch_news_headlines_rss(cfg)
    except Exception:
        pass
    ctx["updated_at"] = now_kst_str()
    return ctx

def external_context_snapshot() -> Dict[str, Any]:
    with _EXT_CTX_LOCK:
        return dict(_EXT_CTX_STATE)

def external_context_refresh_maybe(cfg: Dict[str, Any], force: bool = False) -> bool:
    global _EXT_CTX_LAST_UPDATE
    with _EXT_CTX_LOCK:
        if not force and (time.time() - _EXT_CTX_LAST_UPDATE) < _EXT_CTX_UPDATE_INTERVAL:
            return False
    def _worker():
        global _EXT_CTX_LAST_UPDATE
        try:
            fut = _THREAD_POOL.submit(build_external_context, cfg)
            ctx = fut.result(timeout=EXTERNAL_CONTEXT_TIMEOUT_SEC)
            with _EXT_CTX_LOCK:
                _EXT_CTX_STATE.clear()
                _EXT_CTX_STATE.update(ctx)
                _EXT_CTX_LAST_UPDATE = time.time()
        except Exception:
            pass
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return True

def external_risk_multiplier(ext: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    try:
        fg = _as_float(ext.get("fear_greed", {}).get("value", 50))
        bear_thresh = _as_float(cfg.get("fear_greed_bear_threshold", 25))
        bull_thresh = _as_float(cfg.get("fear_greed_bull_threshold", 75))
        if fg < bear_thresh or fg > bull_thresh:
            return 0.7
        return 1.0
    except Exception:
        return 1.0

_AI_PARAMS_LOCK = threading.RLock()
_AI_PARAMS_CACHE: Dict[str, Any] = {}
_AI_PARAMS_CACHE_TIME = 0.0

def ai_get_optimal_params(
    sym: str, df: pd.DataFrame, signals: Dict[str, Any],
    ext: Dict[str, Any], cfg: Dict[str, Any], rt: Dict[str, Any],
) -> Dict[str, Any]:
    cache_min = _as_float(cfg.get("ai_params_cache_min", 30))
    with _AI_PARAMS_LOCK:
        if (time.time() - _AI_PARAMS_CACHE_TIME) < cache_min * 60:
            cached = _AI_PARAMS_CACHE.get(sym)
            if cached:
                return cached
    leverage = _as_float(cfg.get("leverage", 10))
    fee_rate = _as_float(cfg.get("taker_fee_rate", 0.0006))
    min_tp = round(calc_min_tp_roi(leverage, fee_rate), 2)
    fg = ext.get("fear_greed", {}).get("value", 50)
    recent_losses = rt.get("consecutive_losses", 0)
    coin_stats_raw = rt.get("coin_stats", {}).get(sym, {})
    win_rate = _as_float(coin_stats_raw.get("win_rate", 50))
    last_close = _as_float(df["close"].iloc[-1]) if not df.empty else 0
    atr = _as_float(signals.get("atr", 0))
    atr_pct = atr / last_close * 100 if last_close > 0 else 0
    prompt = (
        f"당신은 암호화폐 선물 트레이딩 AI입니다. 최적 매매 파라미터를 반환하세요.\n\n"
        f"심볼: {sym}\n"
        f"레버리지: {leverage}x\n"
        f"수수료율: {fee_rate*100:.4f}% (왕복 최소 TP ROI: {min_tp:.2f}%)\n"
        f"RSI: {signals.get('rsi', 50):.1f}\n"
        f"MACD 히스토그램: {signals.get('macd_hist', 0):.4f}\n"
        f"ATR%: {atr_pct:.3f}%\n"
        f"시장 추세: {signals.get('overall', 'neutral')}\n"
        f"공포탐욕지수: {fg}\n"
        f"최근 연속 손절: {recent_losses}회\n"
        f"이 코인 승률: {win_rate:.1f}%\n\n"
        f"반환 형식(JSON):\n"
        f'{{"style":"scalp|swing","tp_roi":{min_tp*1.5:.1f},"sl_roi":숫자,"leverage":숫자,"confidence":0~100,"reason":"한줄설명"}}\n'
        f"주의: tp_roi는 반드시 {min_tp:.2f} 이상, sl_roi는 양수(%), leverage는 1~20"
    )
    result = gemini_call_json(prompt, cfg)
    defaults = {
        "style": cfg.get("style", "scalp"),
        "tp_roi": min_tp * 2,
        "sl_roi": min_tp * 0.8,
        "leverage": leverage,
        "confidence": 50,
        "reason": "기본값",
    }
    if result:
        for k in ["style", "tp_roi", "sl_roi", "leverage", "confidence", "reason"]:
            if k in result:
                defaults[k] = result[k]
        defaults["tp_roi"] = max(_as_float(defaults["tp_roi"]), min_tp)
        defaults["sl_roi"] = max(_as_float(defaults["sl_roi"]), min_tp * 0.3)
        defaults["leverage"] = clamp(_as_int(defaults["leverage"]), 1, 20)
    with _AI_PARAMS_LOCK:
        global _AI_PARAMS_CACHE_TIME
        _AI_PARAMS_CACHE[sym] = defaults
        _AI_PARAMS_CACHE_TIME = time.time()
    return defaults

def ai_decide_trade(
    sym: str, df: pd.DataFrame, signals: Dict[str, Any],
    ext: Dict[str, Any], cfg: Dict[str, Any], rt: Dict[str, Any],
    trend_short: str = "neutral", trend_long: str = "neutral",
) -> Dict[str, Any]:
    leverage = _as_float(cfg.get("leverage", 10))
    fee_rate = _as_float(cfg.get("taker_fee_rate", 0.0006))
    min_tp = calc_min_tp_roi(leverage, fee_rate)
    fg = ext.get("fear_greed", {}).get("value", 50)
    rsi = _as_float(signals.get("rsi", 50))
    macd_hist = _as_float(signals.get("macd_hist", 0))
    overall = signals.get("overall", "neutral")
    long_score, short_score = 0, 0
    if overall == "bull": long_score += 2
    elif overall == "bear": short_score += 2
    if rsi < 35: long_score += 1
    elif rsi > 65: short_score += 1
    if macd_hist > 0: long_score += 1
    elif macd_hist < 0: short_score += 1
    if trend_short == "up": long_score += 1
    elif trend_short == "down": short_score += 1
    if trend_long == "up": long_score += 1
    elif trend_long == "down": short_score += 1
    if fg < 25: short_score += 1
    elif fg > 75: long_score += 1
    risk_mul = external_risk_multiplier(ext, cfg)
    consecutive = rt.get("consecutive_losses", 0)
    if consecutive >= 3: risk_mul *= 0.5
    if long_score >= 3 and long_score > short_score:
        decision = "long"
        score = long_score
    elif short_score >= 3 and short_score > long_score:
        decision = "short"
        score = short_score
    else:
        decision = "hold"
        score = 0
    return {
        "decision": decision,
        "score": score,
        "long_score": long_score,
        "short_score": short_score,
        "risk_multiplier": risk_mul,
        "min_tp_roi": min_tp,
    }

def ai_write_review(sym: str, side: str, roi_pct: float, reason: str, cfg: Dict[str, Any]) -> str:
    if not cfg.get("openai_api_key", "").strip():
        return rule_based_loss_analysis(sym, side, 0, 0, roi_pct, reason)
    prompt = (
        f"암호화폐 선물 트레이더 AI 코치입니다.\n"
        f"심볼: {sym}, 방향: {side}, ROI: {roi_pct:.2f}%, 청산사유: {reason}\n"
        f"이 거래의 원인을 분석하고 다음 거래를 위한 교훈을 1-2문장으로 작성하세요."
    )
    result = gemini_call(prompt, cfg)
    return result or rule_based_loss_analysis(sym, side, 0, 0, roi_pct, reason)

_TG_QUEUE: deque = deque(maxlen=200)
_TG_QUEUE_LOCK = threading.RLock()

def tg_enqueue(method: str, data: Dict[str, Any], priority: str = "normal") -> None:
    with _TG_QUEUE_LOCK:
        if priority == "high":
            _TG_QUEUE.appendleft({"method": method, "data": data})
        else:
            _TG_QUEUE.append({"method": method, "data": data})

def _tg_post(url: str, data: Dict[str, Any], timeout_sec: float = 10.0):
    try:
        r = requests.post(url, json=data, timeout=timeout_sec)
        return r.json()
    except Exception:
        return None

def telegram_send_worker_thread():
    while True:
        try:
            item = None
            with _TG_QUEUE_LOCK:
                if _TG_QUEUE:
                    item = _TG_QUEUE.popleft()
            if item is None:
                time.sleep(0.3)
                continue
            cfg = load_settings()
            token = cfg.get("tg_token", "").strip()
            if not token:
                time.sleep(1)
                continue
            url = f"https://api.telegram.org/bot{token}/{item['method']}"
            _tg_post(url, item["data"])
            time.sleep(0.05)
        except Exception:
            time.sleep(1)

def tg_send_chat(chat_id: Any, text: str, parse_mode: str = "HTML") -> None:
    if not chat_id or not text:
        return
    tg_enqueue("sendMessage", {"chat_id": str(chat_id), "text": text[:4096], "parse_mode": parse_mode})

def tg_send(text: str, cfg: Optional[Dict[str, Any]] = None, target: str = "default", priority: str = "normal") -> None:
    if cfg is None:
        cfg = load_settings()
    chat_id = cfg.get("tg_chat_id", "").strip()
    if not chat_id or not text:
        return
    tg_enqueue("sendMessage", {"chat_id": chat_id, "text": text[:4096], "parse_mode": "HTML"}, priority)

def _fmt_dir(side: str) -> str:
    return "🟢 <b>롱</b>" if side == "long" else "🔴 <b>숏</b>"

def _fmt_pnl(v: float) -> str:
    emoji = "📈" if v >= 0 else "📉"
    sign = "+" if v >= 0 else ""
    return f"{emoji} {sign}{v:.2f}"

def tg_msg_entry(
    sym: str, side: str, leverage: int, entry_price: float,
    sl_price: float, tp_price: float, sl_pct: float, tp_pct: float,
    style: str, balance: float, ai_params: Optional[Dict] = None,
    trend_short: str = "", trend_long: str = "",
) -> str:
    style_label = "⚡ 스캘핑" if style == "scalp" else "🌊 스윙"
    trend_line = ""
    if trend_short or trend_long:
        trend_line = f"\n┃ 추세  단기 <code>{trend_short}</code> / 장기 <code>{trend_long}</code>"
    ai_line = ""
    if ai_params and ai_params.get("reason"):
        ai_line = f"\n┃ AI    <i>{ai_params['reason'][:60]}</i>"
    return (
        f"┏━━━ 진입 {style_label} ━━━\n"
        f"┃ {_fmt_dir(side)} {sym}\n"
        f"┃ 레버리지  <code>{leverage}x</code>    잔고  <code>{balance:.1f} USDT</code>\n"
        f"┃ 진입가  <code>{entry_price:,.4f}</code>\n"
        f"┃ 손절가  <code>{sl_price:,.4f}</code>  <i>(-{sl_pct:.2f}%)</i>\n"
        f"┃ 익절가  <code>{tp_price:,.4f}</code>  <i>(+{tp_pct:.2f}%)</i>"
        f"{trend_line}{ai_line}\n"
        f"┗━━━━━━━━━━━━━━━━━━━"
    )

def tg_msg_exit(
    sym: str, side: str, style: str,
    entry_price: float, exit_price: float,
    roi_pct: float, pnl_usdt: float,
    exit_reason: str, duration_sec: float,
    balance: float, review: str = "",
) -> str:
    reason_labels = {
        "tp_hit": "🎯 익절",
        "sl_hit": "🛑 손절",
        "trailing_sl": "🔄 트레일링 손절",
        "manual": "✋ 수동 청산",
        "daily_limit": "⛔ 일일 손실 한도",
        "circuit_breaker": "⛔ 서킷브레이커",
        "timeout": "⏱ 시간초과",
    }
    reason_label = reason_labels.get(exit_reason, exit_reason)
    dur_min = int(duration_sec // 60)
    dur_sec = int(duration_sec % 60)
    result_icon = "✅" if roi_pct >= 0 else "❌"
    review_line = ""
    if review:
        short_review = review[:100] + ("..." if len(review) > 100 else "")
        review_line = f"\n┃\n┃ 💭 회고\n┃ <blockquote>{short_review}</blockquote>"
    return (
        f"┏━━━ 청산 {result_icon} ━━━\n"
        f"┃ {_fmt_dir(side)} {sym}\n"
        f"┃ 사유  {reason_label}\n"
        f"┃ 진입  <code>{entry_price:,.4f}</code> → 청산  <code>{exit_price:,.4f}</code>\n"
        f"┃ ROI  {_fmt_pnl(roi_pct)}%    PNL  {_fmt_pnl(pnl_usdt)} USDT\n"
        f"┃ 보유  <code>{dur_min}분 {dur_sec}초</code>    잔고  <code>{balance:.1f} USDT</code>"
        f"{review_line}\n"
        f"┗━━━━━━━━━━━━━━━━━━━"
    )

def tg_msg_periodic_report(
    positions: List[Dict[str, Any]], balance: float, free: float,
    daily_status: Dict[str, Any], cb_status_dict: Dict[str, Any],
    coin_stats: Dict[str, Dict], consecutive_losses: int,
) -> str:
    lines = [
        f"┏━━━ 정기 리포트 {now_kst_str()[11:16]} ━━━",
        f"┃ 잔고  <code>{balance:.2f} USDT</code>  가용  <code>{free:.2f} USDT</code>",
    ]
    if cb_status_dict.get("paused"):
        lines.append(f"┃ ⛔ 서킷브레이커  {cb_status_dict.get('reason','')}  (해제: {cb_status_dict.get('until_kst','')[11:16]})")
    if daily_status.get("stopped"):
        lines.append(f"┃ ⛔ 일일 손실 한도 도달")
    if consecutive_losses > 0:
        lines.append(f"┃ 연속 손절  <code>{consecutive_losses}회</code>")
    if positions:
        lines.append("┃")
        lines.append("┃ 📊 <b>보유 포지션</b>")
        for p in positions:
            sym = p.get("symbol", "")
            side = position_side_normalize(p)
            roi = position_roi_percent(p)
            upnl = _as_float(p.get("unrealizedPnl", 0))
            lev = _pos_leverage(p)
            roi_icon = "📈" if roi >= 0 else "📉"
            lines.append(f"┃  {roi_icon} {sym} {side.upper()} {lev:.0f}x  ROI <code>{roi:+.2f}%</code>  PNL <code>{upnl:+.2f}</code>")
    else:
        lines.append("┃ 포지션 없음")
    if coin_stats:
        lines.append("┃")
        lines.append("┃ 🏆 <b>코인별 승률</b>")
        sorted_coins = sorted(coin_stats.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)
        for sym, s in sorted_coins[:5]:
            short_sym = sym.split("/")[0]
            lines.append(f"┃  {short_sym}  {s['win_rate']:.0f}%  ({s['wins']}승/{s['losses']}패)  {s['total_pnl']:+.2f}$")
    lines.append("┗━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

def tg_msg_circuit_breaker(reason: str, pause_min: int) -> str:
    return (
        f"⛔ <b>서킷브레이커 발동</b>\n"
        f"<blockquote>사유: {reason}\n"
        f"일시정지: {pause_min}분</blockquote>"
    )

def tg_msg_daily_limit(loss_pct: float, limit_pct: float) -> str:
    return (
        f"⛔ <b>일일 손실 한도 도달</b>\n"
        f"<blockquote>손실: {loss_pct:.2f}% / 한도: {limit_pct:.2f}%\n"
        f"오늘 매매를 중단합니다.</blockquote>"
    )

def tg_send_menu(cfg: Optional[Dict[str, Any]] = None) -> None:
    if cfg is None:
        cfg = load_settings()
    chat_id = cfg.get("tg_chat_id", "").strip()
    if not chat_id:
        return
    text = (
        "🤖 <b>봇 메뉴</b>\n\n"
        "/status — 현재 상태 + 포지션\n"
        "/report — 정기 리포트\n"
        "/ai_report — AI 분석 리포트 (AI 호출)\n"
        "/ai_review — 최근 손절 AI 회고 (AI 호출)\n"
        "/log — 최근 매매일지 5건\n"
        "/loss — 최근 손절 내역\n"
        "/stop — 봇 일시정지\n"
        "/start — 봇 재시작\n"
        "/cb_reset — 서킷브레이커 해제\n"
        "/menu — 메뉴 표시"
    )
    tg_send_chat(chat_id, text)

def tg_answer_callback(token: str, cb_id: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
        requests.post(url, json={"callback_query_id": cb_id}, timeout=5)
    except Exception:
        pass

_TG_POLL_LOCK = threading.RLock()
_TG_UPDATES: deque = deque(maxlen=100)

def tg_updates_push(up: Dict[str, Any]) -> None:
    with _TG_POLL_LOCK:
        _TG_UPDATES.append(up)

def tg_updates_pop_all(max_items: int = 50) -> List[Dict[str, Any]]:
    with _TG_POLL_LOCK:
        items = list(_TG_UPDATES)[:max_items]
        _TG_UPDATES.clear()
        return items

def telegram_polling_thread():
    offset = None
    while True:
        try:
            cfg = load_settings()
            token = cfg.get("tg_token", "").strip()
            if not token:
                time.sleep(5)
                continue
            params: Dict[str, Any] = {"timeout": 10, "allowed_updates": ["message", "callback_query"]}
            if offset is not None:
                params["offset"] = offset
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if not data.get("ok"):
                time.sleep(5)
                continue
            for up in data.get("result", []):
                uid = up.get("update_id", 0)
                if offset is None or uid >= offset:
                    offset = uid + 1
                tg_updates_push(up)
        except Exception:
            time.sleep(5)

def tg_is_admin(user_id: Optional[int], cfg: Dict[str, Any]) -> bool:
    try:
        admin_str = cfg.get("tg_admin_ids", "").strip()
        if not admin_str:
            return True
        admin_ids = {s.strip() for s in admin_str.split(",") if s.strip()}
        return str(user_id) in admin_ids
    except Exception:
        return False

_MON_LOCK = threading.RLock()
_MON_STATE: Dict[str, Any] = {}
_MON_EVENTS: deque = deque(maxlen=200)
_MON_SCANS: deque = deque(maxlen=500)

def monitor_init():
    with _MON_LOCK:
        _MON_STATE.update({
            "running": False, "last_heartbeat": "", "status": "idle",
            "balance": 0.0, "free_balance": 0.0, "positions": [],
            "scan_count": 0, "trade_count": 0, "error_count": 0,
        })

def mon_add_event(ev_type: str, symbol: str = "", message: str = "") -> None:
    try:
        _MON_EVENTS.append({
            "ts": now_kst_str(), "type": ev_type,
            "symbol": symbol, "message": message,
        })
    except Exception:
        pass

def mon_add_scan(stage: str, symbol: str, signal: str = "", score: Any = "", message: str = "") -> None:
    try:
        _MON_SCANS.append({
            "ts": now_kst_str(), "stage": stage, "symbol": symbol,
            "signal": signal, "score": score, "message": message,
        })
    except Exception:
        pass

def mon_recent_events(within_min: int = 15) -> List[Dict[str, Any]]:
    cutoff = now_kst() - timedelta(minutes=within_min)
    result = []
    for ev in list(_MON_EVENTS):
        try:
            dt = _parse_time_kst(ev["ts"])
            if dt and dt >= cutoff:
                result.append(ev)
        except Exception:
            pass
    return result

def monitor_write_throttled(min_interval_sec: float = 1.0):
    with _MON_LOCK:
        last = _as_float(_MON_STATE.get("_last_write", 0))
        if time.time() - last < min_interval_sec:
            return
        _MON_STATE["_last_write"] = time.time()
        try:
            write_json_atomic(MONITOR_FILE, dict(_MON_STATE))
        except Exception:
            pass

def _handle_tg_command(text: str, cfg: Dict[str, Any], rt: Dict[str, Any], ex) -> Optional[str]:
    cmd = text.strip().split()[0].lower().lstrip("/")
    chat_id = cfg.get("tg_chat_id", "").strip()
    if cmd in ("status", "report"):
        try:
            balance, free = safe_fetch_balance(ex) if ex else (0.0, 0.0)
            positions = safe_fetch_positions(ex, cfg.get("target_coins", TARGET_COINS)) if ex else []
            coin_stats = get_coin_stats()
            msg = tg_msg_periodic_report(
                positions, balance, free,
                daily_loss_status(), cb_status(),
                coin_stats, rt.get("consecutive_losses", 0)
            )
            return msg
        except Exception as e:
            return f"상태 조회 오류: {e}"
    elif cmd == "ai_report":
        ext = external_context_snapshot()
        fg_val = ext.get("fear_greed", {}).get("value", 50)
        news = ext.get("news", [])
        news_text = "\n".join(f"- {h}" for h in news[:5])
        prompt = (
            f"현재 시장 분석 리포트를 한국어로 작성하세요.\n"
            f"공포탐욕지수: {fg_val}\n"
            f"최근 뉴스:\n{news_text}\n"
            f"주요 코인(BTC,ETH,SOL)의 현재 시장 컨디션과 매매 방향 추천을 간결하게 알려주세요."
        )
        result = gemini_call(prompt, cfg, model_key="openai_model_fast")
        if result:
            return f"🤖 <b>AI 시장 분석</b>\n<blockquote>{result[:800]}</blockquote>"
        return "OpenAI API 키를 확인하세요."
    elif cmd == "ai_review":
        result = gemini_batch_loss_review(cfg)
        if result:
            return f"🔍 <b>AI 손절 회고</b>\n<blockquote>{result[:800]}</blockquote>"
        batch_n = _as_int(cfg.get("loss_review_batch_n", 5))
        df = read_loss_review_log()
        unreviewed_count = len(df) if not df.empty else 0
        return f"회고 데이터 부족 (현재 {unreviewed_count}건 / 필요 {batch_n}건)"
    elif cmd == "log":
        df = read_trade_log()
        if df.empty:
            return "매매 기록 없음"
        recent = df.tail(5)
        lines = ["📋 <b>최근 매매 5건</b>"]
        for _, row in recent.iterrows():
            pnl = _as_float(row.get("pnl_usdt", 0))
            roi = _as_float(row.get("roi_pct", 0))
            icon = "✅" if pnl >= 0 else "❌"
            lines.append(
                f"{icon} {row.get('symbol','').split('/')[0]} {row.get('side','')} "
                f"ROI <code>{roi:+.2f}%</code> PNL <code>{pnl:+.2f}$</code>"
            )
        return "\n".join(lines)
    elif cmd == "loss":
        df = get_recent_losses(5)
        if df.empty:
            return "최근 손절 기록 없음"
        lines = ["🔴 <b>최근 손절 5건</b>"]
        for _, row in df.iterrows():
            roi = _as_float(row.get("roi_pct", 0))
            reason = row.get("exit_reason", "")
            lines.append(
                f"❌ {row.get('symbol','').split('/')[0]} {row.get('side','')} "
                f"ROI <code>{roi:.2f}%</code>  <i>{reason}</i>"
            )
        return "\n".join(lines)
    elif cmd == "stop":
        rt["running"] = False
        save_runtime(rt)
        return "⏸ 봇이 일시정지됩니다."
    elif cmd == "start":
        rt["running"] = True
        save_runtime(rt)
        cb_reset()
        daily_loss_reset()
        return "▶️ 봇이 재시작됩니다."
    elif cmd == "cb_reset":
        cb_reset()
        return "✅ 서킷브레이커가 해제되었습니다."
    elif cmd == "menu":
        return None
    return None

def _process_tg_updates(cfg: Dict[str, Any], rt: Dict[str, Any], ex) -> None:
    updates = tg_updates_pop_all()
    for up in updates:
        try:
            msg = up.get("message", {})
            user_id = msg.get("from", {}).get("id")
            if not tg_is_admin(user_id, cfg):
                continue
            text = msg.get("text", "").strip()
            if not text.startswith("/"):
                continue
            if text.split()[0].lower().lstrip("/") == "menu":
                tg_send_menu(cfg)
                continue
            reply = _handle_tg_command(text, cfg, rt, ex)
            if reply:
                chat_id = cfg.get("tg_chat_id", "").strip()
                tg_send_chat(chat_id, reply)
        except Exception:
            pass

def telegram_thread(ex_factory=None):
    worker_id = str(uuid.uuid4())[:8]
    rt = load_runtime()
    cfg = load_settings()

    if not runtime_worker_lease_touch(worker_id, "TG_THREAD"):
        return

    rt["running"] = True
    rt["worker_id"] = worker_id
    save_runtime(rt)

    last_heartbeat = time.time()
    last_report = time.time()
    last_cfg_reload = time.time()
    last_ext_refresh = 0.0

    report_interval = _as_float(cfg.get("periodic_report_interval_min", 30)) * 60

    with _MON_LOCK:
        _MON_STATE["running"] = True
        _MON_STATE["worker_id"] = worker_id

    try:
        ex = ex_factory(cfg) if ex_factory else get_exchange(cfg)
        if ex is None:
            tg_send("❌ 거래소 연결 실패. API 키를 확인하세요.", cfg)
            return

        tg_send(
            f"▶️ <b>봇 시작</b>\n"
            f"<blockquote>모드: {'데모' if IS_SANDBOX else '실전'}\n"
            f"코인: {len(cfg.get('target_coins', TARGET_COINS))}개\n"
            f"레버리지: {cfg.get('leverage', 10)}x\n"
            f"스타일: {cfg.get('style', 'scalp')}</blockquote>",
            cfg
        )
        tg_send_menu(cfg)

        while True:
            try:
                if runtime_is_worker_revoked(worker_id):
                    break

                now_ts = time.time()

                if now_ts - last_cfg_reload > 60:
                    cfg = load_settings()
                    last_cfg_reload = now_ts
                    report_interval = _as_float(cfg.get("periodic_report_interval_min", 30)) * 60

                rt = load_runtime()
                if not rt.get("running", True):
                    time.sleep(5)
                    continue

                paused, pause_reason = cb_is_paused()
                if paused:
                    with _MON_LOCK:
                        _MON_STATE["status"] = f"cb_paused: {pause_reason}"
                    time.sleep(10)
                    continue

                d_status = daily_loss_status()
                if d_status.get("stopped"):
                    with _MON_LOCK:
                        _MON_STATE["status"] = "daily_limit_stopped"
                    time.sleep(30)
                    continue

                external_context_refresh_maybe(cfg, force=(now_ts - last_ext_refresh > _EXT_CTX_UPDATE_INTERVAL))
                if now_ts - last_ext_refresh > _EXT_CTX_UPDATE_INTERVAL:
                    last_ext_refresh = now_ts

                _process_tg_updates(cfg, rt, ex)

                balance, free = safe_fetch_balance(ex)
                open_positions = safe_fetch_positions(ex, cfg.get("target_coins", TARGET_COINS))

                with _MON_LOCK:
                    _MON_STATE["balance"] = balance
                    _MON_STATE["free_balance"] = free
                    _MON_STATE["last_heartbeat"] = now_kst_str()
                    _MON_STATE["status"] = "running"
                    _MON_STATE["positions"] = [
                        {
                            "symbol": p.get("symbol", ""),
                            "side": position_side_normalize(p),
                            "roi": round(position_roi_percent(p), 2),
                            "upnl": round(_as_float(p.get("unrealizedPnl", 0)), 4),
                        }
                        for p in open_positions
                    ]

                last_heartbeat = now_ts
                rt["last_heartbeat"] = now_kst_str()

                pos_by_sym: Dict[str, Dict] = {}
                for p in open_positions:
                    sym = p.get("symbol", "")
                    if sym:
                        pos_by_sym[sym] = p

                for sym, tgt in list(rt.get("positions", {}).items()):
                    try:
                        if sym not in pos_by_sym:
                            _on_position_closed_externally(sym, tgt, rt, cfg, balance)
                            continue
                        pos = pos_by_sym[sym]
                        _manage_open_position(ex, sym, pos, tgt, rt, cfg, balance)
                    except Exception:
                        pass

                positions_to_del = [s for s in rt.get("positions", {}) if s not in pos_by_sym]
                for s in positions_to_del:
                    rt["positions"].pop(s, None)

                max_pos = _as_int(cfg.get("max_positions", 3))
                current_count = len(rt.get("positions", {}))

                if current_count < max_pos:
                    ext = external_context_snapshot()
                    scan_interval = _as_float(cfg.get("scan_interval_sec", 60))
                    coins = cfg.get("target_coins", TARGET_COINS)

                    for sym in coins:
                        if sym in rt.get("positions", {}):
                            continue
                        if len(rt.get("positions", {})) >= max_pos:
                            break
                        try:
                            _scan_and_enter(ex, sym, rt, cfg, ext, balance, free)
                        except Exception:
                            pass
                        time.sleep(1)

                if now_ts - last_report >= report_interval:
                    coin_stats = get_coin_stats()
                    update_coin_stats_runtime(rt, coin_stats)
                    msg = tg_msg_periodic_report(
                        open_positions, balance, free,
                        daily_loss_status(), cb_status(),
                        coin_stats, rt.get("consecutive_losses", 0)
                    )
                    tg_send(msg, cfg)
                    last_report = now_ts

                save_runtime(rt)
                monitor_write_throttled()

                scan_interval = _as_float(cfg.get("scan_interval_sec", 60))
                time.sleep(min(scan_interval, 30))

            except Exception as loop_err:
                mon_add_event("error", message=str(loop_err)[:200])
                time.sleep(10)

    except Exception as outer_err:
        tg_send(f"❌ 봇 오류: {str(outer_err)[:200]}", cfg)
    finally:
        runtime_worker_revoke(worker_id, "thread_ended")
        with _MON_LOCK:
            _MON_STATE["running"] = False
            _MON_STATE["status"] = "stopped"

def update_coin_stats_runtime(rt: Dict[str, Any], coin_stats: Dict[str, Dict]) -> None:
    try:
        rt["coin_stats"] = {
            sym: {
                "win_rate": s.get("win_rate", 0),
                "wins": s.get("wins", 0),
                "losses": s.get("losses", 0),
                "total_pnl": s.get("total_pnl", 0),
            }
            for sym, s in coin_stats.items()
        }
    except Exception:
        pass

def _on_position_closed_externally(sym: str, tgt: Dict[str, Any], rt: Dict[str, Any], cfg: Dict[str, Any], balance: float) -> None:
    try:
        entry_price = _as_float(tgt.get("entry_price", 0))
        side = tgt.get("side", "long")
        style = tgt.get("style", "scalp")
        leverage = _as_float(tgt.get("leverage", cfg.get("leverage", 10)))
        entry_time = _as_float(tgt.get("entry_time", time.time()))
        duration = time.time() - entry_time
        trade_id = tgt.get("trade_id", str(uuid.uuid4())[:8])
        rt["positions"].pop(sym, None)
        mon_add_event("position_closed_external", sym, "외부에서 청산됨")
    except Exception:
        pass

def _manage_open_position(
    ex, sym: str, pos: Dict[str, Any], tgt: Dict[str, Any],
    rt: Dict[str, Any], cfg: Dict[str, Any], balance: float,
) -> None:
    side = position_side_normalize(pos)
    roi = position_roi_percent(pos)
    upnl = _as_float(pos.get("unrealizedPnl", 0))
    contracts = _as_float(pos.get("contracts", 0))
    entry_price = _as_float(pos.get("entryPrice", 0)) or _as_float(tgt.get("entry_price", 0))
    cur_price = get_last_price(ex, sym) or entry_price
    leverage = _pos_leverage(pos)
    style = tgt.get("style", cfg.get("style", "scalp"))
    trade_id = tgt.get("trade_id", str(uuid.uuid4())[:8])
    entry_time = _as_float(tgt.get("entry_time", time.time()))

    sl_price = _as_float(tgt.get("sl_price", 0))
    tp_price = _as_float(tgt.get("tp_price", 0))
    trailing_sl = _as_float(tgt.get("trailing_sl", 0))

    exit_reason = None

    if cfg.get("trailing_stop_enabled", True):
        trail_start = _as_float(cfg.get("trailing_start_roi", 50))
        lock_pct = _as_float(cfg.get("trailing_lock_pct", 0.3))
        new_trail = calc_trailing_sl(entry_price, side, roi, trail_start, lock_pct, leverage)
        if new_trail is not None:
            if side == "long" and new_trail > trailing_sl:
                tgt["trailing_sl"] = new_trail
                trailing_sl = new_trail
            elif side == "short" and (trailing_sl == 0 or new_trail < trailing_sl):
                tgt["trailing_sl"] = new_trail
                trailing_sl = new_trail

    if trailing_sl > 0:
        if side == "long" and cur_price <= trailing_sl:
            exit_reason = "trailing_sl"
        elif side == "short" and cur_price >= trailing_sl:
            exit_reason = "trailing_sl"

    if exit_reason is None and sl_price > 0:
        if side == "long" and cur_price <= sl_price:
            exit_reason = "sl_hit"
        elif side == "short" and cur_price >= sl_price:
            exit_reason = "sl_hit"

    if exit_reason is None and tp_price > 0:
        if side == "long" and cur_price >= tp_price:
            exit_reason = "tp_hit"
        elif side == "short" and cur_price <= tp_price:
            exit_reason = "tp_hit"

    if exit_reason:
        ok = close_position_market(ex, sym, side, contracts)
        if ok:
            _on_exit(ex, sym, side, style, entry_price, cur_price, contracts, leverage,
                     sl_price, tp_price, trade_id, entry_time, exit_reason, rt, cfg, balance)
            rt["positions"].pop(sym, None)

def _on_exit(
    ex, sym: str, side: str, style: str,
    entry_price: float, exit_price: float, contracts: float, leverage: float,
    sl_price: float, tp_price: float,
    trade_id: str, entry_time: float, exit_reason: str,
    rt: Dict[str, Any], cfg: Dict[str, Any], balance: float,
) -> None:
    try:
        duration = time.time() - entry_time
        price_diff = exit_price - entry_price if side == "long" else entry_price - exit_price
        pnl_pct = price_diff / entry_price * 100
        roi_pct = pnl_pct * leverage
        notional = entry_price * contracts
        pnl_usdt = notional * price_diff / entry_price
        sl_pct = abs(entry_price - sl_price) / entry_price * 100 if sl_price else 0
        tp_pct = abs(tp_price - entry_price) / entry_price * 100 if tp_price else 0

        review = ""
        if pnl_usdt < 0:
            rt["consecutive_losses"] = rt.get("consecutive_losses", 0) + 1
            rt["total_losses"] = rt.get("total_losses", 0) + 1
            ext = external_context_snapshot()
            fg_val = ext.get("fear_greed", {}).get("value", "")
            rule_analysis = rule_based_loss_analysis(sym, side, entry_price, exit_price, roi_pct, exit_reason, fear_greed=fg_val)
            review = rule_analysis
            log_loss_review(
                trade_id, sym, side, style, entry_price, exit_price,
                pnl_usdt, roi_pct, exit_reason,
                market_context=f"FG:{fg_val}",
                rule_analysis=rule_analysis,
            )
            stopped = daily_loss_check_and_update(pnl_usdt, balance, cfg)
            if stopped:
                msg = tg_msg_daily_limit(
                    daily_loss_status().get("loss_usdt", 0) / max(balance, 1) * 100,
                    _as_float(cfg.get("daily_loss_limit_pct", 5.0))
                )
                tg_send(msg, cfg, priority="high")
            cb_n = _as_int(cfg.get("circuit_breaker_n", 5))
            if cfg.get("circuit_breaker_enabled", True) and rt["consecutive_losses"] >= cb_n:
                pause_min = _as_int(cfg.get("circuit_breaker_pause_min", 30))
                reason = f"{cb_n}연속 손절"
                cb_activate(reason, pause_min)
                tg_send(tg_msg_circuit_breaker(reason, pause_min), cfg)
        else:
            rt["consecutive_losses"] = 0
            rt["total_wins"] = rt.get("total_wins", 0) + 1

        rt["total_trades"] = rt.get("total_trades", 0) + 1

        log_trade(
            trade_id, sym, side, style, leverage,
            entry_price, exit_price, contracts,
            pnl_usdt, pnl_pct, roi_pct,
            sl_price, tp_price, sl_pct, tp_pct,
            exit_reason, duration,
            review=review,
        )

        new_balance, _ = safe_fetch_balance(ex)
        msg = tg_msg_exit(
            sym, side, style, entry_price, exit_price,
            roi_pct, pnl_usdt, exit_reason, duration,
            new_balance if new_balance > 0 else balance,
            review=review,
        )
        tg_send(msg, cfg)
        mon_add_event("exit", sym, f"{exit_reason} ROI:{roi_pct:.2f}%")
    except Exception:
        pass

def _scan_and_enter(
    ex, sym: str, rt: Dict[str, Any], cfg: Dict[str, Any],
    ext: Dict[str, Any], balance: float, free: float,
) -> None:
    data = safe_fetch_ohlcv(ex, sym, cfg.get("scan_tf", "5m"), limit=220)
    if data is None:
        mon_add_scan("skip", sym, message="OHLCV 없음")
        return

    df = ohlcv_to_df(data)
    df, signals, last = calc_indicators(df, cfg)
    if last is None:
        return

    trend_short = get_htf_trend_cached(
        ex, sym, cfg.get("scan_tf", "5m"),
        cfg.get("htf_fast_ma", 7), cfg.get("htf_slow_ma", 99), 60
    )
    trend_long = get_htf_trend_cached(
        ex, sym, cfg.get("htf_tf", "1h"),
        cfg.get("htf_fast_ma", 7), cfg.get("htf_slow_ma", 99), 120
    )

    decision_result = ai_decide_trade(sym, df, signals, ext, cfg, rt, trend_short, trend_long)
    decision = decision_result.get("decision", "hold")
    score = decision_result.get("score", 0)
    risk_mul = decision_result.get("risk_multiplier", 1.0)

    mon_add_scan("scan", sym, signal=decision, score=score)

    if decision == "hold":
        return

    ai_params: Dict[str, Any] = {}
    if cfg.get("ai_params_enabled", True):
        try:
            ai_params = ai_get_optimal_params(sym, df, signals, ext, cfg, rt)
            style = ai_params.get("style", cfg.get("style", "scalp"))
            leverage = clamp(_as_int(ai_params.get("leverage", cfg.get("leverage", 10))), 1, 20)
        except Exception:
            style = cfg.get("style", "scalp")
            leverage = _as_int(cfg.get("leverage", 10))
    else:
        style = cfg.get("style", "scalp")
        leverage = _as_int(cfg.get("leverage", 10))

    sr = get_sr_levels_cached(ex, sym, cfg.get("scan_tf", "5m"), cfg.get("pivot_order", 6))
    supports = sr.get("supports", [])
    resistances = sr.get("resistances", [])
    atr = sr.get("atr", 0.0)

    entry_price = get_last_price(ex, sym)
    if not entry_price:
        return

    sl_price, tp_price = sr_stop_take(
        entry_price, decision, supports, resistances, atr, cfg, style, ai_params
    )

    pos_size_pct = _as_float(cfg.get("position_size_pct", 10.0)) * risk_mul
    pos_usdt = free * pos_size_pct / 100
    if pos_usdt < 5:
        return

    qty_raw = pos_usdt * leverage / entry_price
    qty = to_precision_qty(ex, sym, qty_raw)
    if qty <= 0:
        return

    set_leverage_safe(ex, sym, leverage)

    order_side = "buy" if decision == "long" else "sell"
    ok = market_order_safe(ex, sym, order_side, qty)
    if not ok:
        mon_add_scan("entry_fail", sym, message="주문 실패")
        return

    trade_id = str(uuid.uuid4())[:8]
    tgt_entry = {
        "trade_id": trade_id,
        "side": decision,
        "style": style,
        "leverage": leverage,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "trailing_sl": 0.0,
        "qty": qty,
        "entry_time": time.time(),
        "ai_params": ai_params,
        "trend_short": trend_short,
        "trend_long": trend_long,
    }
    rt.setdefault("positions", {})[sym] = tgt_entry

    new_balance, new_free = safe_fetch_balance(ex)
    sl_pct = abs(entry_price - sl_price) / entry_price * 100
    tp_pct = abs(tp_price - entry_price) / entry_price * 100
    msg = tg_msg_entry(
        sym, decision, leverage, entry_price,
        sl_price, tp_price, sl_pct, tp_pct,
        style, new_balance if new_balance > 0 else balance,
        ai_params, trend_short, trend_long,
    )
    tg_send(msg, cfg)
    mon_add_event("entry", sym, f"{decision} {style} {leverage}x")

_WATCHDOG_LOCK = threading.RLock()

def watchdog_thread():
    heartbeat_timeout = 120
    while True:
        try:
            time.sleep(60)
            rt = load_runtime()
            if not rt.get("running", False):
                continue
            hb_str = rt.get("last_heartbeat", "")
            if not hb_str:
                continue
            hb_dt = _parse_time_kst(hb_str)
            if hb_dt is None:
                continue
            elapsed = (now_kst() - hb_dt).total_seconds()
            if elapsed > heartbeat_timeout:
                cfg = load_settings()
                tg_send(
                    f"⚠️ <b>워치독 경고</b>\n"
                    f"<blockquote>하트비트 {int(elapsed)}초 이상 없음\n"
                    f"마지막: {hb_str}</blockquote>",
                    cfg
                )
                with _MON_LOCK:
                    _MON_STATE["status"] = "watchdog_warning"
        except Exception:
            time.sleep(30)

_THREADS_STARTED = False
_THREADS_LOCK = threading.RLock()

def ensure_threads_started():
    global _THREADS_STARTED
    with _THREADS_LOCK:
        if _THREADS_STARTED:
            return

        cfg = load_settings()

        tg_send_worker = threading.Thread(target=telegram_send_worker_thread, daemon=True, name="TG_SEND")
        tg_send_worker.start()
        try: add_script_run_ctx(tg_send_worker)
        except Exception: pass

        tg_poll = threading.Thread(target=telegram_polling_thread, daemon=True, name="TG_POLL")
        tg_poll.start()
        try: add_script_run_ctx(tg_poll)
        except Exception: pass

        wd = threading.Thread(target=watchdog_thread, daemon=True, name="WATCHDOG")
        wd.start()

        rt = load_runtime()
        if rt.get("running", False):
            def _start_bot():
                time.sleep(2)
                telegram_thread(get_exchange)
            bot = threading.Thread(target=_start_bot, daemon=True, name="TG_THREAD")
            bot.start()
            try: add_script_run_ctx(bot)
            except Exception: pass

        _THREADS_STARTED = True

def gsheet_is_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("gsheet_enabled") and cfg.get("gsheet_spreadsheet_id", "").strip())

def _gsheet_connect(cfg: Dict[str, Any]) -> Optional[Any]:
    if not gspread or not GoogleCredentials:
        return None
    try:
        sa_json = cfg.get("gsheet_service_account_json", "").strip()
        if not sa_json:
            return None
        info = json.loads(sa_json)
        scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = GoogleCredentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc
    except Exception:
        return None

def gsheet_sync_trades(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not gsheet_is_enabled(cfg):
        return {"status": "disabled"}
    try:
        gc = _gsheet_connect(cfg)
        if gc is None:
            return {"status": "connect_failed"}
        sh = gc.open_by_key(cfg["gsheet_spreadsheet_id"])
        df = read_trade_log()
        if df.empty:
            return {"status": "no_data"}
        try:
            ws = sh.worksheet("trades")
        except Exception:
            ws = sh.add_worksheet("trades", rows=5000, cols=len(LOG_COLS))
        ws.clear()
        header = LOG_COLS
        rows = [header] + df.fillna("").astype(str).values.tolist()
        ws.update(rows)
        try:
            df_loss = read_loss_review_log()
            if not df_loss.empty:
                try:
                    ws2 = sh.worksheet("loss_review")
                except Exception:
                    ws2 = sh.add_worksheet("loss_review", rows=2000, cols=len(LOSS_REVIEW_COLS))
                ws2.clear()
                rows2 = [LOSS_REVIEW_COLS] + df_loss.fillna("").astype(str).values.tolist()
                ws2.update(rows2)
        except Exception:
            pass
        return {"status": "ok", "rows": len(df)}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}

_GSHEET_WORKER_QUEUE: deque = deque(maxlen=50)
_GSHEET_WORKER_LOCK = threading.RLock()

def gsheet_enqueue_trade(rec: Dict[str, Any]) -> None:
    with _GSHEET_WORKER_LOCK:
        _GSHEET_WORKER_QUEUE.append(rec)

def gsheet_worker_thread():
    last_sync = 0.0
    sync_interval = 300.0
    while True:
        try:
            time.sleep(10)
            now_ts = time.time()
            if now_ts - last_sync < sync_interval:
                continue
            cfg = load_settings()
            if not gsheet_is_enabled(cfg):
                continue
            gsheet_sync_trades(cfg)
            last_sync = now_ts
        except Exception:
            time.sleep(60)

def tv_symbol_from_ccxt(sym: str) -> str:
    try:
        base = sym.split("/")[0].replace(":", "")
        return f"BITGET:{base}USDT.P"
    except Exception:
        return "BTCUSDT"

def render_tradingview(symbol_ccxt: str, interval: str = "5", height: int = 500) -> None:
    tv_sym = tv_symbol_from_ccxt(symbol_ccxt)
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px">
    <div id="tv_chart"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
        "width": "100%","height": {height},
        "symbol": "{tv_sym}","interval": "{interval}",
        "timezone": "Asia/Seoul","theme": "dark","style": "1",
        "locale": "kr","toolbar_bg": "#1a1a2e",
        "enable_publishing": false,"hide_side_toolbar": false,
        "container_id": "tv_chart"
    }});
    </script></div>"""
    components.html(html, height=height + 20)

def export_trade_log_excel(date_str: str) -> Optional[bytes]:
    if openpyxl is None:
        return None
    try:
        df = read_trade_log()
        if "timestamp" in df.columns:
            df_day = df[df["timestamp"].astype(str).str.startswith(date_str)]
        else:
            df_day = df
        import io
        buf = io.BytesIO()
        df_day.to_excel(buf, index=False)
        return buf.getvalue()
    except Exception:
        return None

def _render_ui():
    ensure_threads_started()
    monitor_init()

    cfg = load_settings()
    rt = load_runtime()

    tabs = st.tabs(["🏠 대시보드", "⚙️ 설정", "📊 차트", "📋 매매일지", "🔴 손절회고", "🏆 코인 분석", "🤖 AI 관리", "📤 내보내기"])

    with tabs[0]:
        _tab_dashboard(cfg, rt)
    with tabs[1]:
        _tab_settings(cfg)
    with tabs[2]:
        _tab_chart(cfg)
    with tabs[3]:
        _tab_trade_log(cfg)
    with tabs[4]:
        _tab_loss_review(cfg)
    with tabs[5]:
        _tab_coin_analysis()
    with tabs[6]:
        _tab_ai_management(cfg, rt)
    with tabs[7]:
        _tab_export(cfg)

def _tab_dashboard(cfg: Dict[str, Any], rt: Dict[str, Any]):
    if st_autorefresh:
        st_autorefresh(interval=10000, key="dash_refresh")

    col1, col2, col3, col4 = st.columns(4)
    mon = read_json_safe(MONITOR_FILE, {})
    balance = _as_float(mon.get("balance", 0))
    free = _as_float(mon.get("free_balance", 0))
    running = mon.get("running", False)
    status = mon.get("status", "stopped")

    col1.metric("💰 잔고", f"{balance:.2f} USDT")
    col2.metric("🔓 가용 잔고", f"{free:.2f} USDT")
    col3.metric("🤖 봇 상태", "실행중" if running else "정지", delta="ON" if running else "OFF")
    col4.metric("⛔ 연속 손절", rt.get("consecutive_losses", 0))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("제어")
        c1, c2, c3 = st.columns(3)
        if c1.button("▶️ 봇 시작", use_container_width=True):
            rt["running"] = True
            save_runtime(rt)
            tg_send("▶️ 봇 재시작 (UI)", cfg)
            st.success("봇이 시작됩니다.")
        if c2.button("⏸ 봇 정지", use_container_width=True):
            rt["running"] = False
            save_runtime(rt)
            tg_send("⏸ 봇 정지 (UI)", cfg)
            st.warning("봇이 정지됩니다.")
        if c3.button("🔄 CB 해제", use_container_width=True):
            cb_reset()
            daily_loss_reset()
            st.success("서킷브레이커 해제")

    with col_b:
        st.subheader("상태")
        cb_s = cb_status()
        if cb_s.get("paused"):
            st.error(f"⛔ 서킷브레이커 발동\n{cb_s.get('reason','')} → {cb_s.get('until_kst','')[11:16]}까지")
        d_status = daily_loss_status()
        if d_status.get("stopped"):
            st.error("⛔ 일일 손실 한도 도달 - 오늘 매매 중단")
        ccxt_h = ccxt_health_snapshot()
        if ccxt_h.get("circuit_open"):
            st.warning(f"⚡ CCXT 회로 차단: {ccxt_h.get('circuit_reason','')}")
        st.caption(f"하트비트: {mon.get('last_heartbeat', '-')}")

    st.divider()
    st.subheader("📊 보유 포지션")
    positions = mon.get("positions", [])
    if positions:
        rows = []
        for p in positions:
            roi = _as_float(p.get("roi", 0))
            rows.append({
                "심볼": p.get("symbol", ""),
                "방향": p.get("side", ""),
                "ROI(%)": f"{roi:+.2f}",
                "미실현손익": f"{_as_float(p.get('upnl', 0)):+.4f}",
            })
        st_dataframe_safe(rows)
    else:
        st.info("보유 포지션 없음")

    st.divider()
    st.subheader("📡 최근 이벤트")
    events = mon_recent_events(within_min=15)
    if events:
        for ev in reversed(events[-10:]):
            icon = {"entry": "🟢", "exit": "⬜", "error": "🔴"}.get(ev.get("type", ""), "•")
            st.caption(f"{icon} {ev.get('ts','')[11:16]} {ev.get('symbol','')} {ev.get('message','')}")
    else:
        st.caption("최근 이벤트 없음")

def _tab_settings(cfg: Dict[str, Any]):
    st.header("⚙️ 설정")
    with st.form("settings_form"):
        st.subheader("🔑 API 키")
        c1, c2 = st.columns(2)
        cfg["bitget_api_key"] = c1.text_input("Bitget API Key", cfg.get("bitget_api_key", ""), type="password")
        cfg["bitget_api_secret"] = c2.text_input("Bitget Secret", cfg.get("bitget_api_secret", ""), type="password")
        cfg["bitget_passphrase"] = st.text_input("Bitget Passphrase", cfg.get("bitget_passphrase", ""), type="password")
        cfg["openai_api_key"] = st.text_input("OpenAI API Key", cfg.get("openai_api_key", ""), type="password")

        st.subheader("📱 텔레그램")
        c1, c2 = st.columns(2)
        cfg["tg_token"] = c1.text_input("TG Bot Token", cfg.get("tg_token", ""), type="password")
        cfg["tg_chat_id"] = c2.text_input("TG Chat ID", cfg.get("tg_chat_id", ""))
        cfg["tg_admin_ids"] = st.text_input("관리자 ID (쉼표 구분)", cfg.get("tg_admin_ids", ""))

        st.subheader("📈 매매 설정")
        c1, c2, c3 = st.columns(3)
        cfg["leverage"] = c1.number_input("레버리지", 1, 50, int(cfg.get("leverage", 10)))
        cfg["max_positions"] = c2.number_input("최대 포지션", 1, 10, int(cfg.get("max_positions", 3)))
        cfg["position_size_pct"] = c3.number_input("포지션 크기(%)", 1.0, 100.0, float(cfg.get("position_size_pct", 10.0)))
        c1, c2 = st.columns(2)
        cfg["style"] = c1.selectbox("기본 스타일", ["scalp", "swing"], index=0 if cfg.get("style","scalp")=="scalp" else 1)
        cfg["scan_interval_sec"] = c2.number_input("스캔 간격(초)", 10, 300, int(cfg.get("scan_interval_sec", 60)))
        cfg["taker_fee_rate"] = st.number_input("Taker 수수료율", 0.0001, 0.01, float(cfg.get("taker_fee_rate", 0.0006)), format="%.4f")
        fee = _as_float(cfg.get("taker_fee_rate", 0.0006))
        lev = _as_float(cfg.get("leverage", 10))
        min_tp = calc_min_tp_roi(lev, fee)
        st.info(f"현재 설정 기준 스캘핑 최소 TP: **{min_tp:.2f}% ROI** (수수료 3x 안전마진)")

        st.subheader("🛡️ 리스크 관리")
        c1, c2 = st.columns(2)
        cfg["circuit_breaker_enabled"] = c1.checkbox("서킷브레이커 사용", cfg.get("circuit_breaker_enabled", True))
        cfg["daily_loss_limit_enabled"] = c2.checkbox("일일 손실 한도 사용", cfg.get("daily_loss_limit_enabled", True))
        c1, c2, c3 = st.columns(3)
        cfg["circuit_breaker_n"] = c1.number_input("연속 손절 횟수", 1, 20, int(cfg.get("circuit_breaker_n", 5)))
        cfg["circuit_breaker_pause_min"] = c2.number_input("일시정지 시간(분)", 5, 1440, int(cfg.get("circuit_breaker_pause_min", 30)))
        cfg["daily_loss_limit_pct"] = c3.number_input("일일 손실 한도(%)", 0.1, 50.0, float(cfg.get("daily_loss_limit_pct", 5.0)))

        st.subheader("🔄 트레일링 스탑")
        cfg["trailing_stop_enabled"] = st.checkbox("트레일링 스탑 사용", cfg.get("trailing_stop_enabled", True))
        c1, c2 = st.columns(2)
        cfg["trailing_start_roi"] = c1.number_input("활성화 ROI(%)", 10.0, 200.0, float(cfg.get("trailing_start_roi", 50.0)))
        cfg["trailing_lock_pct"] = c2.number_input("수익 잠금 비율", 0.1, 1.0, float(cfg.get("trailing_lock_pct", 0.3)))

        st.subheader("🤖 AI 설정")
        cfg["ai_params_enabled"] = st.checkbox("AI 파라미터 자동화", cfg.get("ai_params_enabled", True))
        c1, c2 = st.columns(2)
        cfg["ai_params_cache_min"] = c1.number_input("AI 파라미터 캐시(분)", 1, 120, int(cfg.get("ai_params_cache_min", 30)))
        cfg["loss_review_batch_n"] = c2.number_input("손절 회고 배치 수", 1, 20, int(cfg.get("loss_review_batch_n", 5)))
        cfg["periodic_report_interval_min"] = st.number_input("정기 보고 간격(분)", 5, 1440, int(cfg.get("periodic_report_interval_min", 30)))

        st.subheader("📊 Google Sheets")
        cfg["gsheet_enabled"] = st.checkbox("Google Sheets 동기화", cfg.get("gsheet_enabled", False))
        cfg["gsheet_spreadsheet_id"] = st.text_input("Spreadsheet ID", cfg.get("gsheet_spreadsheet_id", ""))
        cfg["gsheet_service_account_json"] = st.text_area("Service Account JSON", cfg.get("gsheet_service_account_json", ""), height=100)

        if st.form_submit_button("💾 저장", type="primary", use_container_width=True):
            save_settings(cfg)
            st.success("✅ 설정이 저장되었습니다.")
            st.rerun()

def _tab_chart(cfg: Dict[str, Any]):
    st.header("📊 TradingView 차트")
    coins = cfg.get("target_coins", TARGET_COINS)
    c1, c2 = st.columns(2)
    sym = c1.selectbox("코인", coins)
    interval = c2.selectbox("인터벌", ["1", "3", "5", "15", "30", "60", "240", "D"], index=2)
    render_tradingview(sym, interval, height=600)

def _tab_trade_log(cfg: Dict[str, Any]):
    st.header("📋 매매일지")
    df = read_trade_log()
    if df.empty:
        st.info("매매 기록이 없습니다.")
        return

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    wins = len(df[df["pnl_usdt"] > 0]) if "pnl_usdt" in df.columns else 0
    losses = total - wins
    win_rate = wins / total * 100 if total > 0 else 0
    total_pnl = df["pnl_usdt"].sum() if "pnl_usdt" in df.columns else 0

    col1.metric("전체 거래", total)
    col2.metric("승률", f"{win_rate:.1f}%")
    col3.metric("승/패", f"{wins}/{losses}")
    col4.metric("총 PNL", f"{total_pnl:+.2f} USDT")

    st.divider()
    st_dataframe_safe(df.tail(100).sort_values("timestamp", ascending=False) if "timestamp" in df.columns else df.tail(100))

    if st.button("🗑️ 매매일지 초기화"):
        reset_trade_log()
        st.success("초기화 완료")
        st.rerun()

def _tab_loss_review(cfg: Dict[str, Any]):
    st.header("🔴 손절 회고")
    df = read_loss_review_log()
    if df.empty:
        st.info("손절 회고 기록이 없습니다.")
        return

    col1, col2 = st.columns(2)
    col1.metric("총 손절 회고", len(df))

    ai_insights = read_json_safe(AI_INSIGHTS_FILE, {"reviews": []})
    reviews = ai_insights.get("reviews", [])
    if reviews:
        col2.metric("AI 배치 회고", len(reviews))
        st.subheader("🤖 AI 패턴 분석 기록")
        for rev in reversed(reviews[-5:]):
            with st.expander(f"📅 {rev.get('timestamp','')[:16]} ({rev.get('trade_count','')}건 분석)"):
                st.text(rev.get("summary", ""))

    st.divider()
    st.subheader("손절 내역")
    st_dataframe_safe(df.sort_values("timestamp", ascending=False) if "timestamp" in df.columns else df)

    if st.button("🤖 AI 회고 지금 실행"):
        with st.spinner("OpenAI 분석 중..."):
            result = gemini_batch_loss_review(cfg)
        if result:
            st.success("분석 완료")
            st.text_area("결과", result, height=150)
        else:
            batch_n = cfg.get("loss_review_batch_n", 5)
            st.warning(f"데이터 부족 (최소 {batch_n}건 필요)")

def _tab_coin_analysis():
    st.header("🏆 코인별 분석")
    df = read_trade_log()
    if df.empty:
        st.info("매매 데이터가 없습니다.")
        return

    stats = get_coin_stats(df)
    if not stats:
        st.info("분석할 데이터가 없습니다.")
        return

    rows = []
    for sym, s in sorted(stats.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
        rows.append({
            "코인": sym.split("/")[0],
            "전체": s["total"],
            "승": s["wins"],
            "패": s["losses"],
            "승률(%)": s["win_rate"],
            "총PNL(USDT)": s["total_pnl"],
        })
    st_dataframe_safe(rows)

    if df is not None and "pnl_usdt" in df.columns and "timestamp" in df.columns:
        st.subheader("일별 P&L")
        try:
            df["date"] = df["timestamp"].astype(str).str[:10]
            daily = df.groupby("date")["pnl_usdt"].sum().reset_index()
            daily.columns = ["날짜", "PNL(USDT)"]
            st.bar_chart(daily.set_index("날짜"))
        except Exception:
            pass

    st.subheader("손절 사유 분포")
    if "exit_reason" in df.columns:
        reason_counts = df[df["pnl_usdt"] < 0]["exit_reason"].value_counts()
        if not reason_counts.empty:
            st.bar_chart(reason_counts)

def _tab_ai_management(cfg: Dict[str, Any], rt: Dict[str, Any]):
    st.header("🤖 AI 관리")
    gh = gemini_health_info(cfg)
    col1, col2, col3 = st.columns(3)
    col1.metric("OpenAI 사용가능", "✅" if gh.get("available") else "❌")
    col2.metric("오늘 AI 호출", gh.get("call_count_today", 0))
    col3.metric("일시정지", "⛔" if gh.get("suspended") else "✅")
    if gh.get("suspended"):
        st.warning(f"정지 사유: {gh.get('suspend_reason', '')}")

    st.divider()
    st.subheader("현재 AI 파라미터 캐시")
    with _AI_PARAMS_LOCK:
        cache_data = dict(_AI_PARAMS_CACHE)
        cache_time = _AI_PARAMS_CACHE_TIME
    if cache_data:
        st.caption(f"캐시 시간: {_epoch_to_kst_str(cache_time)}")
        for sym, params in cache_data.items():
            with st.expander(sym.split("/")[0]):
                c1, c2, c3 = st.columns(3)
                c1.metric("스타일", params.get("style", "-"))
                c2.metric("TP ROI", f"{_as_float(params.get('tp_roi', 0)):.2f}%")
                c3.metric("SL ROI", f"{_as_float(params.get('sl_roi', 0)):.2f}%")
                st.caption(f"신뢰도: {params.get('confidence', 0)} | {params.get('reason', '')}")
    else:
        st.info("AI 파라미터 캐시 없음")

    if st.button("🔄 AI 파라미터 캐시 초기화"):
        with _AI_PARAMS_LOCK:
            _AI_PARAMS_CACHE.clear()
            global _AI_PARAMS_CACHE_TIME
            _AI_PARAMS_CACHE_TIME = 0.0
        st.success("캐시 초기화 완료")

    st.divider()
    st.subheader("AI 시장 분석 (수동 호출)")
    if st.button("🤖 지금 AI 분석 요청 (AI 호출)"):
        ext = external_context_snapshot()
        fg_val = ext.get("fear_greed", {}).get("value", 50)
        news = ext.get("news", [])
        news_text = "\n".join(f"- {h}" for h in news[:5])
        prompt = (
            f"현재 암호화폐 시장 분석 리포트를 한국어로 작성하세요.\n"
            f"공포탐욕지수: {fg_val}\n"
            f"최근 뉴스:\n{news_text}\n"
            f"BTC, ETH, SOL 시장 컨디션과 단기 방향성을 간결하게 3-4문장으로 요약하세요."
        )
        with st.spinner("OpenAI 분석 중..."):
            result = gemini_call(prompt, cfg)
        if result:
            st.text_area("AI 분석 결과", result, height=200)
        else:
            st.error("OpenAI API 키를 확인하세요.")

def _tab_export(cfg: Dict[str, Any]):
    st.header("📤 내보내기")
    date_str = st.date_input("날짜 선택").strftime("%Y-%m-%d") if True else today_kst_str()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Excel 내보내기")
        if st.button("📥 Excel 다운로드"):
            data = export_trade_log_excel(date_str)
            if data:
                st.download_button(
                    "💾 다운로드", data=data,
                    file_name=f"trades_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("openpyxl 패키지가 필요합니다: pip install openpyxl")

    with col2:
        st.subheader("Google Sheets 동기화")
        if gsheet_is_enabled(cfg):
            if st.button("📊 Google Sheets 동기화"):
                with st.spinner("동기화 중..."):
                    result = gsheet_sync_trades(cfg)
                if result.get("status") == "ok":
                    st.success(f"✅ {result.get('rows', 0)}건 동기화 완료")
                else:
                    st.error(f"오류: {result.get('error', result.get('status', ''))}")
        else:
            st.info("설정에서 Google Sheets를 활성화하세요.")

    st.divider()
    st.subheader("데이터 현황")
    df_trade = read_trade_log()
    df_loss = read_loss_review_log()
    col1, col2 = st.columns(2)
    col1.metric("매매 기록", len(df_trade))
    col2.metric("손절 회고", len(df_loss))

_render_ui()
