"""Candidate strategy routing, deduplication, and conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from core.enums import RegimeType, Side, StrategyName
from core.settings import StrategyRouterConfig
from core.utils import unique_preserve_order
from market.market_regime import RegimeSnapshot

from .base import StrategySignal


SUPPORTED_STRATEGY_GROUPS: tuple[StrategyName, ...] = (
    StrategyName.BREAK_RETEST,
    StrategyName.LIQUIDITY_RAID,
    StrategyName.FAIR_VALUE_GAP,
    StrategyName.ORDER_BLOCK,
    StrategyName.CHOCH,
)


@dataclass(slots=True)
class SignalRoutingDecision:
    """Normalized candidate-selection result for one symbol."""

    candidates: list[StrategySignal]
    chosen: StrategySignal | None
    rejected: list[StrategySignal]
    decision: str


def filter_strategy_registry(
    strategy_registry: Iterable[tuple[StrategyName, Any]],
    *,
    enabled_lookup: dict[str, bool],
    allowed: set[StrategyName],
    max_active_groups: int,
) -> list[tuple[StrategyName, Any]]:
    """Return only supported, enabled, regime-allowed strategy groups."""

    filtered: list[tuple[StrategyName, Any]] = []
    for strategy_name, strategy in strategy_registry:
        if strategy_name not in SUPPORTED_STRATEGY_GROUPS:
            continue
        if not enabled_lookup.get(strategy_name.value, True):
            continue
        if strategy_name not in allowed:
            continue
        filtered.append((strategy_name, strategy))
    limit = max(1, max_active_groups)
    return filtered[:limit]


def get_candidate_signals(
    strategies: Iterable[tuple[StrategyName, Any]],
    context: Any,
    allowed: set[StrategyName],
) -> list[StrategySignal]:
    """Evaluate only the strategies that the regime currently permits."""

    candidates: list[StrategySignal] = []
    for strategy_name, strategy in strategies:
        if strategy_name not in allowed:
            continue
        signal = strategy.evaluate(context)
        if signal is None:
            continue
        signal.display_name = signal.display_name or signal.strategy.display_name
        signal.regime = context.regime
        candidates.append(signal)
    return candidates


def apply_strategy_cooldowns(
    signals: list[StrategySignal],
    *,
    recent_history: Iterable[dict[str, Any]],
    strategy_cooldown_minutes: int,
    symbol_cooldown_minutes: int,
) -> tuple[list[StrategySignal], list[StrategySignal]]:
    """Filter out signals that violate recent per-strategy or per-symbol cooldowns."""

    active: list[StrategySignal] = []
    blocked: list[StrategySignal] = []
    now = datetime.now(tz=UTC)
    latest_by_key: dict[tuple[str, str], datetime] = {}
    latest_by_symbol: dict[str, datetime] = {}

    for row in recent_history:
        symbol = str(row.get("symbol") or "")
        strategy = str(row.get("strategy") or "")
        closed_at = str(row.get("closed_at") or row.get("created_at") or "")
        if not symbol or not strategy or not closed_at:
            continue
        try:
            ts = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        latest_by_key[(symbol, strategy)] = max(ts, latest_by_key.get((symbol, strategy), ts))
        latest_by_symbol[symbol] = max(ts, latest_by_symbol.get(symbol, ts))

    for signal in signals:
        strategy_key = signal.strategy.value
        strategy_ts = latest_by_key.get((signal.symbol, strategy_key))
        symbol_ts = latest_by_symbol.get(signal.symbol)
        blocked_reason = ""
        if strategy_ts and (now - strategy_ts) < timedelta(minutes=max(1, strategy_cooldown_minutes)):
            blocked_reason = f"strategy_cooldown:{strategy_key}"
        elif symbol_ts and (now - symbol_ts) < timedelta(minutes=max(1, symbol_cooldown_minutes)):
            blocked_reason = "symbol_cooldown"
        if blocked_reason:
            signal.trade_rejected_reason = blocked_reason
            signal.blockers.append(blocked_reason)
            blocked.append(signal)
            continue
        active.append(signal)
    return active, blocked


def _target_anchor(signal: StrategySignal) -> float:
    """Return the nearest target anchor price used for overlap calculations."""

    if signal.target_plan and isinstance(signal.target_plan[0], dict):
        return float(signal.target_plan[0].get("price") or signal.tp1_price)
    return float(signal.tp1_price)


def compute_overlap_score(left: StrategySignal, right: StrategySignal) -> float:
    """Estimate whether two signals describe the same trade story."""

    if left.symbol != right.symbol:
        return 0.0
    score = 0.0
    if left.side == right.side:
        score += 0.28
    entry_reference = max(abs(left.entry_price), abs(right.entry_price), 1e-9)
    if abs(left.entry_price - right.entry_price) / entry_reference <= 0.0025:
        score += 0.20
    stop_reference = max(abs(left.stop_price), abs(right.stop_price), 1e-9)
    if abs(left.stop_price - right.stop_price) / stop_reference <= 0.0035:
        score += 0.18
    target_reference = max(abs(_target_anchor(left)), abs(_target_anchor(right)), 1e-9)
    if abs(_target_anchor(left) - _target_anchor(right)) / target_reference <= 0.004:
        score += 0.20
    if set(left.tags).intersection(right.tags):
        score += 0.14
    return round(min(score, 1.0), 4)


def deduplicate_signals(
    signals: list[StrategySignal],
    *,
    merge_same_direction_signals: bool,
    reject_overlapping_strategy_signals: bool,
    overlap_threshold: float,
) -> SignalRoutingDecision:
    """Merge or reject same-direction signals that describe the same structure."""

    if not merge_same_direction_signals or len(signals) <= 1:
        return SignalRoutingDecision(candidates=signals, chosen=None, rejected=[], decision="no_dedupe_needed")

    ordered = sorted(signals, key=lambda item: item.score, reverse=True)
    survivors: list[StrategySignal] = []
    rejected: list[StrategySignal] = []

    for candidate in ordered:
        merged = False
        for primary in survivors:
            overlap = compute_overlap_score(primary, candidate)
            if primary.side != candidate.side or overlap < overlap_threshold:
                continue
            merged = True
            primary.overlap_score = max(primary.overlap_score, overlap)
            candidate.overlap_score = overlap
            primary.candidate_strategies = unique_preserve_order(
                primary.candidate_strategies
                + [primary.strategy.value, candidate.strategy.value]
            )
            if reject_overlapping_strategy_signals:
                candidate.trade_rejected_reason = "overlapping_strategy_signal"
                candidate.rejected_strategies = unique_preserve_order(candidate.rejected_strategies + [candidate.strategy.value])
                candidate.rejection_reasons.append("same_direction_overlap")
                rejected.append(candidate)
            else:
                primary.tags = unique_preserve_order(primary.tags + candidate.tags)
            break
        if not merged:
            candidate.candidate_strategies = unique_preserve_order(candidate.candidate_strategies + [candidate.strategy.value])
            survivors.append(candidate)

    return SignalRoutingDecision(candidates=survivors, chosen=None, rejected=rejected, decision="same_direction_dedupe")


def resolve_signal_conflicts(
    signals: list[StrategySignal],
    *,
    regime: RegimeSnapshot,
    block_opposite_signals_same_window: bool,
    score_gap_threshold: float,
) -> SignalRoutingDecision:
    """Resolve opposite-direction conflicts according to regime-specific rules."""

    if not signals:
        return SignalRoutingDecision(candidates=[], chosen=None, rejected=[], decision="no_candidates")
    if len(signals) == 1:
        chosen = signals[0]
        chosen.chosen_strategy = chosen.strategy.value
        chosen.conflict_resolution_decision = "single_candidate"
        return SignalRoutingDecision(candidates=signals, chosen=chosen, rejected=[], decision="single_candidate")

    longs = [signal for signal in signals if signal.side == Side.LONG]
    shorts = [signal for signal in signals if signal.side == Side.SHORT]
    if not longs or not shorts:
        chosen = max(signals, key=lambda item: item.score)
        chosen.chosen_strategy = chosen.strategy.value
        chosen.conflict_resolution_decision = "same_direction_candidates"
        rejected = [item for item in signals if item is not chosen]
        for item in rejected:
            item.trade_rejected_reason = item.trade_rejected_reason or "lower_score_duplicate"
            item.rejected_strategies = unique_preserve_order(item.rejected_strategies + [item.strategy.value])
        return SignalRoutingDecision(candidates=signals, chosen=chosen, rejected=rejected, decision="same_direction_candidates")

    if not block_opposite_signals_same_window:
        chosen = max(signals, key=lambda item: item.score)
        chosen.chosen_strategy = chosen.strategy.value
        chosen.conflict_resolution_decision = "opposite_allowed_highest_score"
        rejected = [item for item in signals if item is not chosen]
        return SignalRoutingDecision(candidates=signals, chosen=chosen, rejected=rejected, decision="opposite_allowed_highest_score")

    best_long = max(longs, key=lambda item: item.score)
    best_short = max(shorts, key=lambda item: item.score)
    score_gap = abs(best_long.score - best_short.score)
    strict_no_trade = regime.regime in {RegimeType.RANGING, RegimeType.EVENT_RISK, RegimeType.DEAD_MARKET}
    if strict_no_trade or score_gap < score_gap_threshold:
        rejected = list(signals)
        for item in rejected:
            item.trade_rejected_reason = "opposite_signal_conflict"
            item.rejected_strategies = unique_preserve_order(item.rejected_strategies + [item.strategy.value])
            item.rejection_reasons.append(regime.regime.display_name)
            item.conflict_resolution_decision = "no_trade_opposite_conflict"
        return SignalRoutingDecision(candidates=signals, chosen=None, rejected=rejected, decision="no_trade_opposite_conflict")

    chosen = best_long if best_long.score > best_short.score else best_short
    chosen.chosen_strategy = chosen.strategy.value
    chosen.conflict_resolution_decision = "highest_score_after_opposite_conflict"
    rejected = [item for item in signals if item is not chosen]
    for item in rejected:
        item.trade_rejected_reason = "opposite_signal_conflict"
        item.rejected_strategies = unique_preserve_order(item.rejected_strategies + [item.strategy.value])
        item.rejection_reasons.append("lower_score_than_primary")
    return SignalRoutingDecision(candidates=signals, chosen=chosen, rejected=rejected, decision="highest_score_after_opposite_conflict")


def choose_primary_signal(
    signals: list[StrategySignal],
    *,
    regime: RegimeSnapshot,
    router: StrategyRouterConfig,
) -> SignalRoutingDecision:
    """Apply dedupe and conflict resolution, returning a final primary signal."""

    deduped = deduplicate_signals(
        signals,
        merge_same_direction_signals=router.merge_same_direction_signals,
        reject_overlapping_strategy_signals=router.reject_overlapping_strategy_signals,
        overlap_threshold=router.same_direction_overlap_threshold,
    )
    conflict = resolve_signal_conflicts(
        deduped.candidates,
        regime=regime,
        block_opposite_signals_same_window=router.block_opposite_signals_same_window,
        score_gap_threshold=router.opposite_signal_score_gap,
    )
    rejected = deduped.rejected + conflict.rejected
    if conflict.chosen is not None:
        conflict.chosen.candidate_strategies = unique_preserve_order(
            conflict.chosen.candidate_strategies + [signal.strategy.value for signal in signals]
        )
    return SignalRoutingDecision(
        candidates=signals,
        chosen=conflict.chosen,
        rejected=rejected,
        decision=conflict.decision,
    )
