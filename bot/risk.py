from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

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


def _roi_to_price_pct(roi_pct: float, leverage: float) -> float:
    lev = max(1.0, float(abs(leverage) or 1.0))
    return float(abs(roi_pct)) / lev


def _price_pct_to_roi(price_pct: float, leverage: float) -> float:
    lev = max(1.0, float(abs(leverage) or 1.0))
    return float(abs(price_pct)) * lev


def pick_tp_sl_from_sr(
    *,
    entry_price: float,
    side: str,
    leverage: float,
    sr_context: Optional[Dict[str, Any]],
    atr_price_pct: float,
    rr_floor: float,
    tp_roi_cap: Optional[float] = None,
    sl_roi_cap: Optional[float] = None,
    sr_front_run_bps: float = 5.0,
    sr_breathing_bps: float = 10.0,
    sr_breathing_atr_mult: float = 1.0,
    fallback_tp_roi: float = 8.0,
    fallback_sl_roi: float = 3.0,
) -> Dict[str, Any]:
    """
    Prefer SR-based TP/SL prices.
    - Never shrinks SL to satisfy RR.
    - If RR is too low at nearest TP, tries farther TP or ATR fallback TP.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "reason_code": "INIT",
        "tp_price": None,
        "sl_price": None,
        "tp_roi": 0.0,
        "sl_roi": 0.0,
        "tp_price_pct": 0.0,
        "sl_price_pct": 0.0,
        "tp_price_source": "",
        "sl_price_source": "",
        "rr": 0.0,
    }
    try:
        px = float(entry_price or 0.0)
        if px <= 0:
            out["reason_code"] = "BAD_ENTRY_PRICE"
            return out
        side0 = str(side or "").lower().strip()
        is_long = side0 in ["buy", "long"]
        lev = max(1.0, float(abs(leverage) or 1.0))
        rr_need = max(1.0, float(rr_floor or 1.0))

        sr = dict(sr_context or {})
        supports = [float((x or {}).get("price", 0.0)) for x in (sr.get("supports") or []) if float((x or {}).get("price", 0.0)) > 0]
        resistances = [float((x or {}).get("price", 0.0)) for x in (sr.get("resistances") or []) if float((x or {}).get("price", 0.0)) > 0]
        volume_nodes = [float((x or {}).get("price", 0.0)) for x in (sr.get("volume_nodes") or []) if float((x or {}).get("price", 0.0)) > 0]
        atr_pct = max(0.0, float(atr_price_pct or 0.0))

        breathing_price_pct = max(float(sr_breathing_bps) / 10000.0 * 100.0, atr_pct * float(max(0.0, sr_breathing_atr_mult)))
        front_run_pct = max(0.0, float(sr_front_run_bps) / 10000.0 * 100.0)

        # SL from nearest protective SR + breathing room
        sl_price = 0.0
        sl_src = ""
        if is_long:
            below = sorted([s for s in supports if s < px], reverse=True)
            if below:
                raw = float(below[0])
                sl_price = raw * (1.0 - breathing_price_pct / 100.0)
                sl_src = "SR"
        else:
            above = sorted([r for r in resistances if r > px])
            if above:
                raw = float(above[0])
                sl_price = raw * (1.0 + breathing_price_pct / 100.0)
                sl_src = "SR"
        if sl_price <= 0:
            sl_price_pct_fb = _roi_to_price_pct(float(fallback_sl_roi or 2.0), lev)
            if is_long:
                sl_price = px * (1.0 - sl_price_pct_fb / 100.0)
            else:
                sl_price = px * (1.0 + sl_price_pct_fb / 100.0)
            sl_src = "ATR_FALLBACK"

        sl_price_pct = abs((px - sl_price) / px) * 100.0
        sl_roi = _price_pct_to_roi(sl_price_pct, lev)
        if sl_roi_cap is not None and float(sl_roi_cap) > 0:
            sl_roi = min(float(sl_roi), float(sl_roi_cap))
            sl_price_pct = _roi_to_price_pct(sl_roi, lev)
            sl_price = px * (1.0 - sl_price_pct / 100.0) if is_long else px * (1.0 + sl_price_pct / 100.0)
            if sl_src == "SR":
                sl_src = "ROI_GUARDRAIL"

        # TP candidates (nearest SR first, then farther levels)
        tp_candidates: list[tuple[float, str]] = []
        if is_long:
            for r in sorted([x for x in resistances if x > px]):
                tp_candidates.append((r * (1.0 - front_run_pct / 100.0), "SR"))
            for vn in sorted([x for x in volume_nodes if x > px]):
                tp_candidates.append((vn * (1.0 - front_run_pct / 100.0), "VOLUME_NODE"))
        else:
            for s in sorted([x for x in supports if x < px], reverse=True):
                tp_candidates.append((s * (1.0 + front_run_pct / 100.0), "SR"))
            for vn in sorted([x for x in volume_nodes if x < px], reverse=True):
                tp_candidates.append((vn * (1.0 + front_run_pct / 100.0), "VOLUME_NODE"))

        tp_price = 0.0
        tp_src = ""
        for candidate, src in tp_candidates:
            if candidate <= 0:
                continue
            tp_price_pct = abs((candidate - px) / px) * 100.0
            tp_roi = _price_pct_to_roi(tp_price_pct, lev)
            if tp_roi_cap is not None and float(tp_roi_cap) > 0:
                tp_roi = min(float(tp_roi), float(tp_roi_cap))
                tp_price_pct = _roi_to_price_pct(tp_roi, lev)
            rr = tp_price_pct / max(sl_price_pct, 1e-9)
            if rr >= rr_need:
                tp_price = px * (1.0 + tp_price_pct / 100.0) if is_long else px * (1.0 - tp_price_pct / 100.0)
                tp_src = src
                break

        if tp_price <= 0:
            tp_price_pct_fb = _roi_to_price_pct(float(fallback_tp_roi or 3.0), lev)
            need_pct = max(tp_price_pct_fb, sl_price_pct * rr_need)
            tp_price = px * (1.0 + need_pct / 100.0) if is_long else px * (1.0 - need_pct / 100.0)
            tp_src = "ATR_FALLBACK"

        tp_price_pct = abs((tp_price - px) / px) * 100.0
        tp_roi = _price_pct_to_roi(tp_price_pct, lev)
        if tp_roi_cap is not None and float(tp_roi_cap) > 0:
            tp_roi = min(float(tp_roi), float(tp_roi_cap))
            tp_price_pct = _roi_to_price_pct(tp_roi, lev)
            tp_price = px * (1.0 + tp_price_pct / 100.0) if is_long else px * (1.0 - tp_price_pct / 100.0)
            if tp_src == "SR":
                tp_src = "ROI_GUARDRAIL"

        rr_final = tp_price_pct / max(sl_price_pct, 1e-9)
        out.update(
            {
                "ok": bool(tp_price > 0 and sl_price > 0 and rr_final >= 1.0),
                "reason_code": "OK" if bool(tp_price > 0 and sl_price > 0 and rr_final >= 1.0) else "SR_PLAN_INVALID",
                "tp_price": float(tp_price),
                "sl_price": float(sl_price),
                "tp_roi": float(tp_roi),
                "sl_roi": float(sl_roi),
                "tp_price_pct": float(tp_price_pct),
                "sl_price_pct": float(sl_price_pct),
                "tp_price_source": str(tp_src or "ROI_FALLBACK"),
                "sl_price_source": str(sl_src or "ROI_FALLBACK"),
                "rr": float(rr_final),
            }
        )
        return out
    except Exception as e:
        out["reason_code"] = f"ERROR:{type(e).__name__}"
        return out
