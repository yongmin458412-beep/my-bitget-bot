"""Structural target discovery and merging."""

from __future__ import annotations

from math import isfinite
from typing import Any

from core.enums import Side
from strategy.base import SignalContext

from .structure_levels import build_structure_snapshot

# 단타 기준: 타겟이 이 배수를 초과하면 제외 (세션 레벨 등 원거리 타겟 필터링)
_DEFAULT_MAX_TARGET_ATR = 3.5
# ATR 기반 합성 타겟 배수 (구조적 타겟이 부족할 때 보충)
_SYNTHETIC_ATR_MULTIPLES = [0.8, 1.5, 2.5, 3.5]


def merge_close_targets(targets: list[dict[str, Any]], threshold_pct: float) -> list[dict[str, Any]]:
    """Merge targets that are effectively the same price zone."""

    if not targets:
        return []
    ordered = sorted(targets, key=lambda item: item["price"])
    merged: list[dict[str, Any]] = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        reference = max(abs(last["price"]), abs(item["price"]), 1e-9)
        if abs(item["price"] - last["price"]) / reference <= threshold_pct:
            reasons = list(dict.fromkeys(last.get("reasons", [last.get("reason")]) + item.get("reasons", [item.get("reason")])))
            last["price"] = (last["price"] + item["price"]) / 2.0
            last["priority"] = min(last["priority"], item["priority"])
            last["reasons"] = reasons
            last["reason"] = "+".join(reasons)
            last["inside_range_middle"] = bool(last.get("inside_range_middle")) or bool(item.get("inside_range_middle"))
            continue
        merged.append(dict(item))
    return merged


def detect_structural_targets(
    signal_context: SignalContext,
    setup_context: dict[str, Any],
    *,
    merge_threshold_pct: float = 0.0015,
    range_middle_exclusion: float = 0.2,
    max_target_atr_multiple: float = _DEFAULT_MAX_TARGET_ATR,
) -> list[dict[str, Any]]:
    """Return ordered structural TP candidates for the setup.

    단타 우선순위:
      1순위 — 근거리 스윙 (진입 TF 기준)
      2순위 — 레인지 반대편 경계
      3순위 — 이퀄 하이/로우
      4순위 — 상위 TF 스윙
      5순위 — 세션 레벨 (전일/아시아/런던) — max_target_atr_multiple 이내일 때만

    ATR 기반 합성 타겟: 구조적 타겟이 2개 미만이면 entry ± ATR 배수로 보충.
    """

    side: Side = setup_context["side"]
    entry_price = float(setup_context["entry_price"])
    atr_value = float(setup_context.get("atr") or 0.0)
    structure = build_structure_snapshot(
        df_entry=signal_context.frames["3m"],
        df_structure=signal_context.frames["15m"],
    )
    range_boundaries = setup_context.get("range_boundaries") or structure["range_boundaries"]
    equal_levels = structure["equal_levels"]
    entry_swings = structure["recent_swings_entry"]
    structure_swings = structure["recent_swings_structure"]
    session_levels = structure["session_levels"]

    # ── 우선순위 그룹별로 분리 ────────────────────────────────────────────
    # 그룹 A: 근거리 구조 (단타 핵심)
    near_candidates: list[tuple[str, float | None]] = []
    # 그룹 B: 세션 레벨 (원거리 — ATR 필터 적용)
    session_candidates: list[tuple[str, float | None]] = []

    if side == Side.LONG:
        near_candidates = [
            ("recent_swing_high", next((p for p in entry_swings.get("swing_highs", []) if p > entry_price), None)),
            ("range_opposite_side", range_boundaries.get("range_high")),
            *[("equal_highs", v) for v in equal_levels.get("equal_highs", [])],
            ("higher_timeframe_swing_high", next((p for p in structure_swings.get("swing_highs", []) if p > entry_price), None)),
        ]
        session_candidates = [
            ("asia_high", session_levels.get("asia_high")),
            ("london_high", session_levels.get("london_high")),
            ("previous_high", session_levels.get("prev_day_high")),
        ]
    else:
        near_candidates = [
            ("recent_swing_low", next((p for p in reversed(entry_swings.get("swing_lows", [])) if p < entry_price), None)),
            ("range_opposite_side", range_boundaries.get("range_low")),
            *[("equal_lows", v) for v in equal_levels.get("equal_lows", [])],
            ("higher_timeframe_swing_low", next((p for p in reversed(structure_swings.get("swing_lows", [])) if p < entry_price), None)),
        ]
        session_candidates = [
            ("asia_low", session_levels.get("asia_low")),
            ("london_low", session_levels.get("london_low")),
            ("previous_low", session_levels.get("prev_day_low")),
        ]

    min_target_distance = max(entry_price * merge_threshold_pct, atr_value * 0.25, 1e-9)
    max_target_distance = atr_value * max_target_atr_multiple if atr_value > 0 else float("inf")

    range_mid = float(range_boundaries.get("range_mid") or 0.0)
    range_width = float(range_boundaries.get("range_width") or 0.0)
    middle_band = range_width * range_middle_exclusion

    def _make_target(reason: str, raw_price: float | None, priority: int, *, allow_far: bool = False) -> dict[str, Any] | None:
        if raw_price is None:
            return None
        price = float(raw_price)
        if not isfinite(price):
            return None
        if side == Side.LONG and price <= entry_price:
            return None
        if side == Side.SHORT and price >= entry_price:
            return None
        distance = abs(price - entry_price)
        if distance < min_target_distance:
            return None
        # 세션 레벨은 max_target_distance 초과 시 제외
        if not allow_far and distance > max_target_distance:
            return None
        inside_range_middle = range_width > 0 and abs(price - range_mid) <= middle_band
        return {
            "price": price,
            "reason": reason,
            "reasons": [reason],
            "priority": priority,
            "inside_range_middle": inside_range_middle,
        }

    targets: list[dict[str, Any]] = []

    # 그룹 A: 근거리 구조 타겟 (거리 무관하게 수집, 이후 정렬에서 가까운 것이 TP1)
    for priority, (reason, raw_price) in enumerate(near_candidates, start=1):
        t = _make_target(reason, raw_price, priority, allow_far=True)
        if t:
            targets.append(t)

    # 그룹 B: 세션 레벨 — max_target_atr_multiple 이내일 때만 포함
    for priority, (reason, raw_price) in enumerate(session_candidates, start=len(near_candidates) + 1):
        t = _make_target(reason, raw_price, priority, allow_far=False)
        if t:
            targets.append(t)

    # ── ATR 합성 타겟 보충 ───────────────────────────────────────────────
    # 구조적 타겟이 3개 미만이면 entry ± ATR 배수로 보충 (단타 기준선 역할)
    if atr_value > 0 and len(targets) < 3:
        for mult in _SYNTHETIC_ATR_MULTIPLES:
            synth_price = entry_price + atr_value * mult if side == Side.LONG else entry_price - atr_value * mult
            if not any(abs(t["price"] - synth_price) / max(abs(synth_price), 1e-9) <= merge_threshold_pct for t in targets):
                inside_range_middle = range_width > 0 and abs(synth_price - range_mid) <= middle_band
                targets.append({
                    "price": synth_price,
                    "reason": f"atr_target_{mult:.1f}x",
                    "reasons": [f"atr_target_{mult:.1f}x"],
                    "priority": 99,
                    "inside_range_middle": inside_range_middle,
                })
            if len(targets) >= 2:
                break

    targets = merge_close_targets(targets, merge_threshold_pct)
    # 최종 정렬: LONG → 가격 낮은 것(가까운 것)이 TP1, SHORT → 가격 높은 것(가까운 것)이 TP1
    targets.sort(key=lambda item: item["price"], reverse=side == Side.SHORT)
    for idx, target in enumerate(targets, start=1):
        target["priority"] = idx
    return targets
