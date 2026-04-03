"""Risk blocker evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.settings import RiskConfig


@dataclass(slots=True)
class GuardCheckResult:
    """Risk-guard result."""

    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        """Whether the trade can proceed."""

        return not self.blockers


class TradeGuard:
    """Guardrails around entries and mode transitions."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        symbol: str,
        open_positions_count: int,
        symbol_has_position: bool,
        daily_loss_r: float,
        consecutive_losses: int,
        daily_order_count: int,
        paused: bool,
        news_blocked: bool,
        funding_blocked: bool,
        account_drawdown_pct: float,
        unrealized_loss_pct: float,
        symbol_cooldown_until: datetime | None = None,
        current_portfolio_heat_pct: float = 0.0,
    ) -> GuardCheckResult:
        """Return blockers and warnings for a proposed entry."""

        result = GuardCheckResult()
        now = datetime.now(tz=UTC)
        if paused:
            result.blockers.append("paused")
        # 안전망: 최대 포지션 수 초과 (hard cap)
        if open_positions_count >= self.config.max_concurrent_positions:
            result.blockers.append("max_concurrent_positions")
        # 포트폴리오 히트 한도 초과 (증거금 합계 / 총자산)
        if current_portfolio_heat_pct >= self.config.max_portfolio_heat_pct:
            result.blockers.append("portfolio_heat_limit")
        if symbol_has_position and not self.config.allow_multiple_positions_per_symbol:
            result.blockers.append("symbol_position_exists")
        # ── 서킷 브레이커 (토글 제어) ─────────────────────────────────────
        if getattr(self.config, "daily_loss_limit_enabled", False) and daily_loss_r <= -self.config.max_daily_loss_r:
            result.blockers.append("daily_loss_limit")
        if getattr(self.config, "consecutive_loss_cooldown_enabled", False) and consecutive_losses >= self.config.max_consecutive_losses:
            result.blockers.append("consecutive_loss_cooldown")
        if getattr(self.config, "drawdown_limit_enabled", False) and account_drawdown_pct >= self.config.max_account_drawdown_pct:
            result.blockers.append("account_drawdown_limit")
        if getattr(self.config, "kill_switch_enabled", False) and unrealized_loss_pct >= self.config.kill_switch_unrealized_loss_pct:
            result.blockers.append("kill_switch_triggered")
        # ──────────────────────────────────────────────────────────────────
        if daily_order_count >= self.config.max_daily_orders:
            result.blockers.append("daily_order_limit")
        if news_blocked:
            result.blockers.append("news_block")
        if funding_blocked:
            result.blockers.append("funding_window")
        if symbol_cooldown_until and symbol_cooldown_until > now:
            result.blockers.append(f"symbol_cooldown_until:{symbol_cooldown_until.isoformat()}")
        return result

