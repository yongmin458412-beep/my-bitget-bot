from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _as_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, str):
            vv = v.replace(",", "").strip()
            if not vv:
                return 0.0
            return float(vv)
        return float(v)
    except Exception:
        return 0.0


def _spread_bps_from_bid_ask(bid: Any, ask: Any) -> Optional[float]:
    b = _as_float(bid)
    a = _as_float(ask)
    if not (b > 0 and a > 0 and a >= b):
        return None
    mid = (a + b) / 2.0
    if mid <= 0:
        return None
    return float(((a - b) / mid) * 10000.0)


def _extract_quote_volume(ticker: Dict[str, Any]) -> float:
    qv = _as_float(ticker.get("quoteVolume"))
    if qv > 0:
        return qv

    info = ticker.get("info") if isinstance(ticker.get("info"), dict) else {}
    for k in (
        "quoteVolume",
        "quoteVol",
        "usdtVolume",
        "turnover",
        "quoteTurnover",
        "amount",
        "quote_volume",
        "volumeQuote",
    ):
        qv2 = _as_float(info.get(k))
        if qv2 > 0:
            return qv2

    base_volume = _as_float(ticker.get("baseVolume"))
    last = _as_float(ticker.get("last"))
    if base_volume > 0 and last > 0:
        return float(base_volume * last)
    return 0.0


def _extract_bid_ask(ticker: Dict[str, Any]) -> Tuple[float, float]:
    bid = _as_float(ticker.get("bid"))
    ask = _as_float(ticker.get("ask"))
    if bid > 0 and ask > 0:
        return bid, ask
    info = ticker.get("info") if isinstance(ticker.get("info"), dict) else {}
    bid2 = _as_float(info.get("bidPr")) or _as_float(info.get("bidPx")) or _as_float(info.get("bidPrice"))
    ask2 = _as_float(info.get("askPr")) or _as_float(info.get("askPx")) or _as_float(info.get("askPrice"))
    return bid2, ask2


@dataclass
class UniverseResult:
    symbols: List[str]
    refreshed: bool
    refresh_at_epoch: float
    next_refresh_epoch: float
    ttl_sec: int
    reason_code: str
    stats: Dict[str, Any]
    top_rows: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "refreshed": bool(self.refreshed),
            "refresh_at_epoch": float(self.refresh_at_epoch),
            "next_refresh_epoch": float(self.next_refresh_epoch),
            "ttl_sec": int(self.ttl_sec),
            "reason_code": str(self.reason_code),
            "stats": dict(self.stats or {}),
            "top_rows": [dict(x) for x in (self.top_rows or [])],
        }


class BitgetUniverseBuilder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {
            "key": "",
            "expires_at": 0.0,
            "result": None,
        }

    def get_universe(
        self,
        ex: Any,
        *,
        top_n: int,
        max_spread_bps: float,
        ttl_sec: int,
        min_quote_volume: float = 0.0,
        force_refresh: bool = False,
    ) -> UniverseResult:
        now = time.time()
        n = int(max(1, min(300, int(top_n or 20))))
        spread_cap = float(max(0.1, float(max_spread_bps or 20.0)))
        ttl = int(max(15, min(3600, int(ttl_sec or 180))))
        min_qv = float(max(0.0, float(min_quote_volume or 0.0)))
        cache_key = f"n={n}|spread={spread_cap:.6f}|ttl={ttl}|qv={min_qv:.2f}"

        with self._lock:
            cached = self._cache.get("result")
            expires_at = float(self._cache.get("expires_at", 0.0) or 0.0)
            key = str(self._cache.get("key", "") or "")
            if (not force_refresh) and cached and (key == cache_key) and now < expires_at:
                c = cached if isinstance(cached, UniverseResult) else None
                if c is not None:
                    return UniverseResult(
                        symbols=list(c.symbols),
                        refreshed=False,
                        refresh_at_epoch=float(c.refresh_at_epoch),
                        next_refresh_epoch=float(expires_at),
                        ttl_sec=int(ttl),
                        reason_code="CACHE_HIT",
                        stats=dict(c.stats or {}),
                        top_rows=[dict(x) for x in (c.top_rows or [])],
                    )

        res = self._build_universe(
            ex,
            top_n=n,
            max_spread_bps=spread_cap,
            ttl_sec=ttl,
            min_quote_volume=min_qv,
        )
        with self._lock:
            self._cache["key"] = cache_key
            self._cache["expires_at"] = float(res.next_refresh_epoch)
            self._cache["result"] = res
        return res

    def _build_universe(
        self,
        ex: Any,
        *,
        top_n: int,
        max_spread_bps: float,
        ttl_sec: int,
        min_quote_volume: float,
    ) -> UniverseResult:
        now = time.time()
        reason_code = "OK"
        stats: Dict[str, Any] = {
            "candidate_total": 0,
            "accepted_total": 0,
            "skipped_inactive": 0,
            "skipped_non_linear_swap": 0,
            "skipped_non_usdt": 0,
            "skipped_low_volume": 0,
            "skipped_spread_missing": 0,
            "skipped_spread_wide": 0,
            "ticker_errors": 0,
        }
        top_rows: List[Dict[str, Any]] = []
        symbols: List[str] = []

        markets: Dict[str, Any] = {}
        try:
            mk = getattr(ex, "markets", None)
            if isinstance(mk, dict) and mk:
                markets = mk
            else:
                loaded = ex.load_markets()
                if isinstance(loaded, dict):
                    markets = loaded
        except Exception:
            markets = {}

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for key, market in (markets or {}).items():
            if not isinstance(market, dict):
                continue
            is_swap = bool(market.get("swap"))
            is_linear = bool(market.get("linear"))
            if not (is_swap and is_linear):
                stats["skipped_non_linear_swap"] = int(stats["skipped_non_linear_swap"]) + 1
                continue
            if market.get("active") is False:
                stats["skipped_inactive"] = int(stats["skipped_inactive"]) + 1
                continue
            quote = str(market.get("quote", "") or "").upper().strip()
            settle = str(market.get("settle", "") or "").upper().strip()
            if (quote != "USDT") and (settle != "USDT"):
                stats["skipped_non_usdt"] = int(stats["skipped_non_usdt"]) + 1
                continue
            sym = str(market.get("symbol", "") or key or "").strip()
            if not sym:
                continue
            candidates.append((sym, market))

        stats["candidate_total"] = int(len(candidates))
        if not candidates:
            reason_code = "NO_CANDIDATES"
            return UniverseResult(
                symbols=[],
                refreshed=True,
                refresh_at_epoch=float(now),
                next_refresh_epoch=float(now + ttl_sec),
                ttl_sec=int(ttl_sec),
                reason_code=reason_code,
                stats=stats,
                top_rows=[],
            )

        tickers: Dict[str, Any] = {}
        ticker_reason = ""
        try:
            syms = [s for s, _ in candidates]
            t_all = ex.fetch_tickers(syms)
            if isinstance(t_all, dict):
                tickers = t_all
        except TypeError:
            try:
                t_all = ex.fetch_tickers()
                if isinstance(t_all, dict):
                    tickers = t_all
            except Exception as e:
                ticker_reason = f"FETCH_TICKERS_FAIL:{type(e).__name__}"
        except Exception as e:
            ticker_reason = f"FETCH_TICKERS_FAIL:{type(e).__name__}"

        for sym, market in candidates:
            ticker = tickers.get(sym) if isinstance(tickers, dict) else None
            if not isinstance(ticker, dict):
                mid = str(market.get("id", "") or "").strip()
                if mid and isinstance(tickers, dict):
                    t_alt = tickers.get(mid)
                    if isinstance(t_alt, dict):
                        ticker = t_alt
            if not isinstance(ticker, dict):
                stats["ticker_errors"] = int(stats["ticker_errors"]) + 1
                continue

            qv = _extract_quote_volume(ticker)
            if qv <= float(min_quote_volume):
                stats["skipped_low_volume"] = int(stats["skipped_low_volume"]) + 1
                continue

            bid, ask = _extract_bid_ask(ticker)
            spread_bps = _spread_bps_from_bid_ask(bid, ask)
            if spread_bps is None or (not math.isfinite(float(spread_bps))):
                stats["skipped_spread_missing"] = int(stats["skipped_spread_missing"]) + 1
                continue
            if float(spread_bps) > float(max_spread_bps):
                stats["skipped_spread_wide"] = int(stats["skipped_spread_wide"]) + 1
                continue

            top_rows.append(
                {
                    "symbol": str(sym),
                    "quote_volume": float(qv),
                    "spread_bps": float(spread_bps),
                    "bid": float(bid),
                    "ask": float(ask),
                }
            )

        if ticker_reason and not top_rows:
            reason_code = ticker_reason
        if not top_rows:
            reason_code = "NO_SYMBOL_AFTER_FILTERS"
        else:
            top_rows.sort(key=lambda x: float(x.get("quote_volume", 0.0)), reverse=True)
            symbols = [str(x.get("symbol", "")) for x in top_rows[: int(top_n)] if str(x.get("symbol", "")).strip()]
            stats["accepted_total"] = int(len(symbols))
            if not symbols:
                reason_code = "NO_TOP_SYMBOLS"

        return UniverseResult(
            symbols=symbols,
            refreshed=True,
            refresh_at_epoch=float(now),
            next_refresh_epoch=float(now + ttl_sec),
            ttl_sec=int(ttl_sec),
            reason_code=str(reason_code),
            stats=stats,
            top_rows=top_rows[: int(top_n)],
        )
