from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema
except Exception:
    argrelextrema = None


def _cluster_levels(levels: List[float], cluster_bps: float) -> List[Dict[str, Any]]:
    if not levels:
        return []
    vals = sorted([float(x) for x in levels if float(x) > 0.0])
    if not vals:
        return []
    out: List[Dict[str, Any]] = []
    cur: List[float] = [vals[0]]
    for p in vals[1:]:
        pivot = float(np.mean(cur))
        tol = max(1e-12, abs(pivot) * (float(cluster_bps) / 10000.0))
        if abs(p - pivot) <= tol:
            cur.append(float(p))
        else:
            out.append({"price": float(np.mean(cur)), "touches": int(len(cur))})
            cur = [float(p)]
    out.append({"price": float(np.mean(cur)), "touches": int(len(cur))})
    out.sort(key=lambda x: float(x["price"]))
    return out


def compute_sr_context(
    df: pd.DataFrame,
    *,
    timeframe: str = "5m",
    lookback_bars: int = 220,
    pivot_order: int = 6,
    cluster_bps: float = 8.0,
    volume_bins: int = 24,
) -> Dict[str, Any]:
    """
    Build SR context from OHLCV:
    - supports/resistances from local extrema + clustering
    - volume_nodes from simple volume profile bins
    """
    out: Dict[str, Any] = {
        "timeframe": str(timeframe or "5m"),
        "supports": [],
        "resistances": [],
        "volume_nodes": [],
        "lookback_bars": int(lookback_bars),
        "reason_code": "OK",
    }
    if df is None or len(df) < max(20, int(pivot_order) * 4):
        out["reason_code"] = "DATA_TOO_SHORT"
        return out

    hdf = df.tail(int(max(40, lookback_bars))).copy()
    highs = hdf["high"].astype(float).values
    lows = hdf["low"].astype(float).values
    closes = hdf["close"].astype(float).values
    vols = hdf["vol"].astype(float).values if "vol" in hdf.columns else np.ones_like(closes)
    n = len(hdf)

    if argrelextrema is not None:
        hi_idx = argrelextrema(highs, np.greater_equal, order=int(max(2, pivot_order)))[0]
        lo_idx = argrelextrema(lows, np.less_equal, order=int(max(2, pivot_order)))[0]
    else:
        hi_idx = []
        lo_idx = []
        w = int(max(2, pivot_order))
        for i in range(w, n - w):
            seg_h = highs[i - w : i + w + 1]
            seg_l = lows[i - w : i + w + 1]
            if highs[i] >= float(np.max(seg_h)):
                hi_idx.append(i)
            if lows[i] <= float(np.min(seg_l)):
                lo_idx.append(i)
        hi_idx = np.array(hi_idx, dtype=int)
        lo_idx = np.array(lo_idx, dtype=int)

    r_raw = [float(highs[i]) for i in hi_idx if 0 <= int(i) < n]
    s_raw = [float(lows[i]) for i in lo_idx if 0 <= int(i) < n]
    r_clus = _cluster_levels(r_raw, cluster_bps=float(cluster_bps))
    s_clus = _cluster_levels(s_raw, cluster_bps=float(cluster_bps))

    # strength boost from proximity volume
    v_med = float(np.median(vols)) if len(vols) else 1.0
    v_med = max(v_med, 1e-12)
    for lvl in r_clus:
        p = float(lvl["price"])
        tol = abs(p) * (float(cluster_bps) / 10000.0)
        mask = np.abs(closes - p) <= max(tol, abs(p) * 0.0005)
        v_score = float(np.sum(vols[mask]) / v_med) if np.any(mask) else 0.0
        lvl["strength"] = float(lvl["touches"] + min(8.0, v_score * 0.12))
    for lvl in s_clus:
        p = float(lvl["price"])
        tol = abs(p) * (float(cluster_bps) / 10000.0)
        mask = np.abs(closes - p) <= max(tol, abs(p) * 0.0005)
        v_score = float(np.sum(vols[mask]) / v_med) if np.any(mask) else 0.0
        lvl["strength"] = float(lvl["touches"] + min(8.0, v_score * 0.12))

    r_clus.sort(key=lambda x: (float(x.get("strength", 0.0)), float(x.get("price", 0.0))), reverse=True)
    s_clus.sort(key=lambda x: (float(x.get("strength", 0.0)), float(x.get("price", 0.0))), reverse=True)

    # simple volume profile nodes
    nodes: List[Dict[str, Any]] = []
    try:
        lo = float(np.min(lows))
        hi = float(np.max(highs))
        if hi > lo and int(volume_bins) >= 8:
            bins = np.linspace(lo, hi, int(volume_bins) + 1)
            idx = np.digitize(closes, bins) - 1
            agg = np.zeros(int(volume_bins), dtype=float)
            for i, b in enumerate(idx):
                if 0 <= int(b) < int(volume_bins):
                    agg[int(b)] += float(vols[i])
            mids = (bins[:-1] + bins[1:]) / 2.0
            rows = [{"price": float(mids[i]), "strength": float(agg[i])} for i in range(len(agg))]
            rows.sort(key=lambda x: float(x["strength"]), reverse=True)
            vmax = float(rows[0]["strength"]) if rows else 0.0
            for r in rows[: min(8, len(rows))]:
                denom = vmax if vmax > 0 else 1.0
                nodes.append({"price": float(r["price"]), "strength": float(r["strength"] / denom)})
    except Exception:
        nodes = []

    out["supports"] = sorted(s_clus[:12], key=lambda x: float(x.get("price", 0.0)))
    out["resistances"] = sorted(r_clus[:12], key=lambda x: float(x.get("price", 0.0)))
    out["volume_nodes"] = sorted(nodes, key=lambda x: float(x.get("strength", 0.0)), reverse=True)
    return out

