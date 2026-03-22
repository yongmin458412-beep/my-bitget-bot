"""Stop-loss construction helpers."""

from __future__ import annotations

from core.enums import Side


def apply_atr_buffer(stop_price: float, atr_value: float, side: Side, multiplier: float = 0.2) -> float:
    """Apply an ATR buffer to a structural stop."""

    if side == Side.LONG:
        return stop_price - atr_value * multiplier
    return stop_price + atr_value * multiplier


def hard_stop_from_pct(entry_price: float, side: Side, stop_pct: float) -> float:
    """Fallback hard stop percentage."""

    if side == Side.LONG:
        return entry_price * (1 - stop_pct)
    return entry_price * (1 + stop_pct)


def choose_safer_stop(
    *,
    entry_price: float,
    structural_stop: float,
    side: Side,
    atr_value: float,
    stop_mode: str = "structure",
    use_atr_buffer: bool = True,
    atr_buffer_multiplier: float = 0.2,
    hard_stop_pct: float | None = None,
    structural_stop_already_buffered: bool = False,
) -> float:
    """Prefer structural stop with ATR buffer, with optional hard-stop cap."""

    if stop_mode == "hard_percent":
        if hard_stop_pct is None:
            return structural_stop
        return hard_stop_from_pct(entry_price, side, hard_stop_pct)

    stop = structural_stop
    if use_atr_buffer and not structural_stop_already_buffered and atr_value > 0:
        stop = apply_atr_buffer(structural_stop, atr_value, side, multiplier=atr_buffer_multiplier)
    if hard_stop_pct is None:
        return stop
    hard_stop = hard_stop_from_pct(entry_price, side, hard_stop_pct)
    if side == Side.LONG:
        return min(stop, hard_stop)
    return max(stop, hard_stop)
