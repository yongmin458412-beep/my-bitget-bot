from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema  # type: ignore
except Exception:
    argrelextrema = None


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _cluster_levels(levels: List[Tuple[float, float]], cluster_bps: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not levels:
        return out
    levels2 = sorted([(float(p), float(s)) for p, s in levels if p > 0], key=lambda x: x[0])
    if not levels2:
        return out
    cur_prices: List[float] = [levels2[0][0]]
    cur_strength = float(levels2[0][1])
    for price, strength in levels2[1:]:
        pivot = float(np.mean(cur_prices))
        if pivot <= 0:
            pivot = price
        dist_bps = abs((price - pivot) / pivot) * 10000.0
        if dist_bps <= cluster_bps:
            cur_prices.append(price)
            cur_strength += float(strength)
        else:
            p = float(np.mean(cur_prices))
            out.append({"price": p, "strength": float(cur_strength), "touches": int(len(cur_prices))})
            cur_prices = [price]
            cur_strength = float(strength)
    p = float(np.mean(cur_prices))
    out.append({"price": p, "strength": float(cur_strength), "touches": int(len(cur_prices))})
    return out


def _volume_profile_nodes(df: pd.DataFrame, bins: int = 48, top_n: int = 6) -> List[Dict[str, Any]]:
    if df is None or df.empty or len(df) < 30:
        return []
    try:
        lo = float(df["low"].astype(float).min())
        hi = float(df["high"].astype(float).max())
        if not (hi > lo > 0):
            return []
        edges = np.linspace(lo, hi, int(max(12, min(160, int(bins or 48)))) + 1)
        mids = (edges[:-1] + edges[1:]) / 2.0
        vol_acc = np.zeros(len(mids), dtype=float)
        closes = df["close"].astype(float).values
        vols = df["vol"].astype(float).values
        idx = np.clip(np.digitize(closes, edges) - 1, 0, len(mids) - 1)
        for i in range(len(idx)):
            vol_acc[int(idx[i])] += float(vols[i])
        if not np.any(vol_acc > 0):
            return []
        rank = np.argsort(vol_acc)[::-1][: int(max(1, min(20, int(top_n or 6))))]
        maxv = float(np.max(vol_acc)) if len(vol_acc) else 0.0
        nodes: List[Dict[str, Any]] = []
        for ri in rank:
            nodes.append(
                {
                    "price": float(mids[int(ri)]),
                    "strength": float((vol_acc[int(ri)] / max(maxv, 1e-12)) * 10.0),
                    "touches": 0,
                }
            )
        nodes.sort(key=lambda x: float(x.get("price", 0.0)))
        return nodes
    except Exception:
        return []


def compute_sr_context(
    df: pd.DataFrame,
    timeframe: str,
    lookback_bars: int = 220,
    pivot_order: int = 6,
    cluster_bps: float = 12.0,
    volume_bins: int = 48,
    volume_nodes_top_n: int = 6,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "timeframe": str(timeframe or ""),
        "lookback_bars": int(lookback_bars),
        "supports": [],
        "resistances": [],
        "volume_nodes": [],
    }
    if df is None or df.empty:
        return out
    try:
        dfx = df.tail(int(max(40, min(1200, int(lookback_bars or 220))))).copy()
        highs = dfx["high"].astype(float).values
        lows = dfx["low"].astype(float).values
        vols = dfx["vol"].astype(float).values
        order = int(max(2, min(24, int(pivot_order or 6))))

        if argrelextrema is not None:
            hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
            lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        else:
            hi_idx = []
            lo_idx = []
            for i in range(order, len(highs) - order):
                if highs[i] >= np.max(highs[i - order : i + order + 1]):
                    hi_idx.append(i)
                if lows[i] <= np.min(lows[i - order : i + order + 1]):
                    lo_idx.append(i)
            hi_idx = np.array(hi_idx, dtype=int)
            lo_idx = np.array(lo_idx, dtype=int)

        highs_lv: List[Tuple[float, float]] = []
        lows_lv: List[Tuple[float, float]] = []
        for i in hi_idx:
            highs_lv.append((float(highs[int(i)]), float(max(1.0, vols[int(i)]))))
        for i in lo_idx:
            lows_lv.append((float(lows[int(i)]), float(max(1.0, vols[int(i)]))))

        res = _cluster_levels(highs_lv, float(max(2.0, cluster_bps)))
        sup = _cluster_levels(lows_lv, float(max(2.0, cluster_bps)))
        res.sort(key=lambda x: float(x.get("price", 0.0)))
        sup.sort(key=lambda x: float(x.get("price", 0.0)))

        out["supports"] = sup
        out["resistances"] = res
        out["volume_nodes"] = _volume_profile_nodes(dfx, bins=volume_bins, top_n=volume_nodes_top_n)
        return out
    except Exception:
        return out
