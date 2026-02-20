from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None


def clamp(v: Any, lo: Any, hi: Any):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(int(v))
        if np is not None and isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return float(default)
            return x
        if isinstance(v, (int, float)):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return float(default)
            return x
        s = str(v).strip()
        if not s:
            return float(default)
        if s.lower() in ["none", "null", "nan"]:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(v)
        if np is not None and isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, int):
            return int(v)
        return int(round(as_float(v, float(default))))
    except Exception:
        return int(default)


def timeframe_seconds(tf: str, default_sec: int = 300) -> int:
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


def roi_pct_from_price_move(price_move_pct: float, leverage: float) -> float:
    lev = max(1.0, abs(as_float(leverage, 1.0)))
    return float(max(0.0, abs(as_float(price_move_pct, 0.0))) * lev)


def price_move_pct_from_roi(roi_pct: float, leverage: float) -> float:
    lev = max(1.0, abs(as_float(leverage, 1.0)))
    return float(max(0.0, abs(as_float(roi_pct, 0.0))) / lev)


def pick_tp_sl_from_sr(
    *,
    entry_price: float,
    side: str,
    leverage: float,
    supports: Optional[List[Dict[str, Any]]] = None,
    resistances: Optional[List[Dict[str, Any]]] = None,
    volume_nodes: Optional[List[Dict[str, Any]]] = None,
    atr_price_pct: float = 0.0,
    style_tp_min_roi: float = 0.0,
    style_tp_max_roi: float = 0.0,
    style_sl_max_roi: float = 0.0,
    rr_floor: float = 1.5,
    sr_front_run_bps: float = 5.0,
    sr_breathing_bps: float = 8.0,
    sr_breathing_atr_mult: float = 1.2,
) -> Dict[str, Any]:
    px = max(0.0, as_float(entry_price, 0.0))
    lev = max(1.0, abs(as_float(leverage, 1.0)))
    sd = str(side or "").strip().lower()
    sup = sorted(
        [dict(x) for x in (supports or []) if as_float((x or {}).get("price"), 0.0) > 0],
        key=lambda x: as_float(x.get("price"), 0.0),
    )
    res = sorted(
        [dict(x) for x in (resistances or []) if as_float((x or {}).get("price"), 0.0) > 0],
        key=lambda x: as_float(x.get("price"), 0.0),
    )
    nodes = sorted(
        [dict(x) for x in (volume_nodes or []) if as_float((x or {}).get("price"), 0.0) > 0],
        key=lambda x: as_float(x.get("price"), 0.0),
    )

    out: Dict[str, Any] = {
        "ok": False,
        "tp_price": None,
        "sl_price": None,
        "tp_source": "",
        "sl_source": "",
        "tp_roi": 0.0,
        "sl_roi": 0.0,
        "rr": 0.0,
    }
    if px <= 0:
        return out

    front_run = max(0.0, as_float(sr_front_run_bps, 0.0)) / 10000.0
    breathe_price_pct = max(
        max(0.0, as_float(sr_breathing_bps, 0.0)) / 10000.0 * 100.0,
        max(0.0, as_float(atr_price_pct, 0.0)) * max(0.0, as_float(sr_breathing_atr_mult, 1.2)),
    )

    def _price_to_move_pct(target: float) -> float:
        if px <= 0:
            return 0.0
        return abs((float(target) - px) / px) * 100.0

    tp_price = None
    sl_price = None
    tp_source = ""
    sl_source = ""

    if sd in ["buy", "long"]:
        below = [as_float(x.get("price"), 0.0) for x in sup if as_float(x.get("price"), 0.0) < px]
        above = [as_float(x.get("price"), 0.0) for x in res if as_float(x.get("price"), 0.0) > px]
        if below:
            base_sl = max(below)
            sl_price = float(base_sl * (1.0 - breathe_price_pct / 100.0))
            sl_source = "SR"
        elif nodes:
            nbelow = [as_float(x.get("price"), 0.0) for x in nodes if as_float(x.get("price"), 0.0) < px]
            if nbelow:
                base_sl = max(nbelow)
                sl_price = float(base_sl * (1.0 - breathe_price_pct / 100.0))
                sl_source = "VOLUME_NODE"
        if above:
            base_tp = min(above)
            tp_price = float(base_tp * (1.0 - front_run))
            tp_source = "SR"
        elif nodes:
            nabove = [as_float(x.get("price"), 0.0) for x in nodes if as_float(x.get("price"), 0.0) > px]
            if nabove:
                base_tp = min(nabove)
                tp_price = float(base_tp * (1.0 - front_run))
                tp_source = "VOLUME_NODE"
    else:
        above = [as_float(x.get("price"), 0.0) for x in res if as_float(x.get("price"), 0.0) > px]
        below = [as_float(x.get("price"), 0.0) for x in sup if as_float(x.get("price"), 0.0) < px]
        if above:
            base_sl = min(above)
            sl_price = float(base_sl * (1.0 + breathe_price_pct / 100.0))
            sl_source = "SR"
        elif nodes:
            nabove = [as_float(x.get("price"), 0.0) for x in nodes if as_float(x.get("price"), 0.0) > px]
            if nabove:
                base_sl = min(nabove)
                sl_price = float(base_sl * (1.0 + breathe_price_pct / 100.0))
                sl_source = "VOLUME_NODE"
        if below:
            base_tp = max(below)
            tp_price = float(base_tp * (1.0 + front_run))
            tp_source = "SR"
        elif nodes:
            nbelow = [as_float(x.get("price"), 0.0) for x in nodes if as_float(x.get("price"), 0.0) < px]
            if nbelow:
                base_tp = max(nbelow)
                tp_price = float(base_tp * (1.0 + front_run))
                tp_source = "VOLUME_NODE"

    tp_price_move = _price_to_move_pct(tp_price) if tp_price else 0.0
    sl_price_move = _price_to_move_pct(sl_price) if sl_price else 0.0
    tp_roi = roi_pct_from_price_move(tp_price_move, lev)
    sl_roi = roi_pct_from_price_move(sl_price_move, lev)

    tp_min = max(0.0, as_float(style_tp_min_roi, 0.0))
    tp_max = max(tp_min, as_float(style_tp_max_roi, 0.0))
    sl_max = max(0.1, as_float(style_sl_max_roi, 0.0))
    rr_req = max(1.0, as_float(rr_floor, 1.5))

    if tp_roi <= 0 and tp_min > 0:
        tp_roi = tp_min
        tp_source = "ROI_FALLBACK"
    if sl_roi <= 0:
        sl_roi = min(sl_max, max(0.2, tp_roi / max(rr_req, 1.0)))
        sl_source = "ATR_FALLBACK"

    tp_roi = min(max(tp_roi, tp_min), tp_max if tp_max > 0 else tp_roi)
    sl_roi = min(max(sl_roi, 0.2), sl_max)

    if tp_roi > 0 and sl_roi > 0:
        rr = float(tp_roi / max(sl_roi, 1e-9))
    else:
        rr = 0.0
    if rr < rr_req:
        return out

    out.update(
        {
            "ok": True,
            "tp_price": float(tp_price) if tp_price else None,
            "sl_price": float(sl_price) if sl_price else None,
            "tp_source": str(tp_source or "ROI_FALLBACK"),
            "sl_source": str(sl_source or "ATR_FALLBACK"),
            "tp_roi": float(tp_roi),
            "sl_roi": float(sl_roi),
            "rr": float(rr),
        }
    )
    return out
