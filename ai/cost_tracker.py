"""AI API 비용 추적 — 일일/월간 한도 관리."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from core.logger import get_logger
from core.persistence import SQLitePersistence


@dataclass(slots=True)
class CostSnapshot:
    daily_total: float
    monthly_total: float
    daily_remaining: float
    monthly_remaining: float
    call_count_today: int


class AICostTracker:
    """AI API 비용 추적 + 일일/월간 하드리밋."""

    def __init__(
        self,
        persistence: SQLitePersistence,
        daily_limit_usd: float = 2.0,
        monthly_limit_usd: float = 50.0,
    ) -> None:
        self.persistence = persistence
        self.daily_limit = daily_limit_usd
        self.monthly_limit = monthly_limit_usd
        self.logger = get_logger(__name__)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.persistence.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_cost_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                purpose TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )

    def can_call(self, estimated_cost: float = 0.001) -> bool:
        """일일/월간 한도 체크. 초과 시 False."""
        snap = self.get_snapshot()
        if snap.daily_total + estimated_cost > self.daily_limit:
            return False
        if snap.monthly_total + estimated_cost > self.monthly_limit:
            return False
        return True

    def record_call(
        self,
        provider: str,
        cost_usd: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        purpose: str = "",
    ) -> None:
        """AI 호출 비용 기록."""
        now = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.persistence.execute(
            """
            INSERT INTO ai_cost_log (provider, cost_usd, tokens_in, tokens_out, purpose, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (provider, cost_usd, tokens_in, tokens_out, purpose, now),
        )

    def get_snapshot(self) -> CostSnapshot:
        """현재 비용 현황."""
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        month = datetime.now(tz=UTC).strftime("%Y-%m")

        daily_rows = self.persistence.fetchall(
            "SELECT COALESCE(SUM(cost_usd), 0) as total, COUNT(*) as cnt FROM ai_cost_log WHERE created_at >= ?",
            (f"{today}T00:00:00",),
        )
        monthly_rows = self.persistence.fetchall(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM ai_cost_log WHERE created_at >= ?",
            (f"{month}-01T00:00:00",),
        )
        daily_total = float(daily_rows[0]["total"]) if daily_rows else 0.0
        daily_count = int(daily_rows[0]["cnt"]) if daily_rows else 0
        monthly_total = float(monthly_rows[0]["total"]) if monthly_rows else 0.0

        return CostSnapshot(
            daily_total=daily_total,
            monthly_total=monthly_total,
            daily_remaining=max(0, self.daily_limit - daily_total),
            monthly_remaining=max(0, self.monthly_limit - monthly_total),
            call_count_today=daily_count,
        )
