"""Main orchestration for the Bitget futures trading system."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai.client import OpenAIResponsesClient
from ai.summarizer import AISummarizer
from core.charting import TradeChartSpec, render_trade_chart
from core.enums import BotStatus, OrderType, ProductType, RegimeType, Side, SignalStatus, StrategyName, TradingMode
from core.logger import get_logger, setup_logging
from core.persistence import SQLitePersistence
from core.settings import SettingsManager, build_settings_snapshot
from core.state_store import StateStore
from core.utils import dump_json, load_json, round_to_step
from execution.fill_handler import FillHandler
from execution.order_manager import OrderManager
from execution.router import OrderRouter
from execution.sltp_manager import SLTPManager
from exchange.bitget_demo import BitgetDemoExchange
from exchange.bitget_live import BitgetLiveExchange
from exchange.bitget_models import ContractConfig, OrderRequest
from journal.performance import PerformanceAnalyzer
from journal.trade_journal import TradeJournal
from market.klines import KlineService
from market.indicators import atr
from market.market_regime import MarketRegimeClassifier
from market.orderbook import OrderBookService
from market.sr_levels import SupportResistanceDetector
from market.universe import UniverseManager
from news.analyzer import NewsAnalyzer
from news.collector import NewsCollector
from news.impact_filter import NewsImpactFilter
from risk.risk_engine import RiskEngine
from strategy.base import SignalContext, StrategySignal
from risk.risk_engine import ApprovedTrade
from strategy.break_retest import BreakRetestStrategy
from strategy.choch import CHoCHStrategy
from strategy.conflict_router import (
    apply_strategy_cooldowns,
    choose_primary_signal,
    filter_strategy_registry,
    get_candidate_signals,
)
from strategy.ev_filter import ExpectedValueFilter
from strategy.fair_value_gap import FairValueGapStrategy
from strategy.liquidity_raid import LiquidityRaidStrategy
from strategy.order_block import OrderBlockStrategy
from strategy.rr_filter import evaluate_trade_viability
from strategy.signal_score import SignalScorer
from telegram_bot.bot import TelegramBotService
from telegram_bot.commands import CommandProvider, TelegramCommandRouter
from telegram_bot.formatters import format_daily_summary, format_entry_fill_alert


ROOT_DIR = Path(__file__).resolve().parents[1]
CONTROL_COMMANDS_PATH = ROOT_DIR / "state" / "control_commands.json"


class TradingApplication(CommandProvider):
    """Main application runtime."""

    def __init__(self) -> None:
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load()
        setup_logging(ROOT_DIR / self.settings.logging.directory, self.settings.logging.level)
        self.logger = get_logger(__name__)
        self.persistence = SQLitePersistence(self.settings.db_path)
        self.state_store = StateStore(self.settings.state_path, self.persistence)
        self.state_store.load()
        self.stop_event = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []
        self.contracts_by_symbol: dict[str, Any] = {}
        self.latest_tickers: dict[str, Any] = {}
        self.latest_news_alerts: list[dict[str, Any]] = []
        self.last_balance: dict[str, Any] = {"balance": 0.0, "used_margin": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0}
        self.last_daily_summary_date: str | None = None
        self.last_weekly_summary_key: str | None = None
        self.last_heartbeat_sent_at: datetime | None = None
        # stop_reason 반복 손절 쿨다운: {stop_reason: cooldown_until_datetime}
        self._stop_reason_cooldowns: dict[str, datetime] = {}
        self._build_components()

    def _build_components(self) -> None:
        """Instantiate components from current settings."""

        if self.settings.mode == TradingMode.DEMO:
            self.exchange = BitgetDemoExchange(self.settings)
        else:
            self.exchange = BitgetLiveExchange(self.settings)
        self.klines = KlineService(self.exchange)
        self.orderbooks = OrderBookService(self.exchange)
        self.universe_manager = UniverseManager(
            self.settings,
            self.exchange,
            self.persistence,
            self.state_store,
            self.klines,
            self.orderbooks,
        )
        self.regime_classifier = MarketRegimeClassifier()
        self.level_detector = SupportResistanceDetector()
        self.break_retest = BreakRetestStrategy(
            retest_tolerance_atr=self.settings.strategy.retest_tolerance_atr,
            volume_multiple=self.settings.strategy.confirmation_volume_multiple,
            range_middle_exclusion=self.settings.strategy.range_middle_exclusion,
            merge_nearby_target_threshold_pct=self.settings.strategy.merge_nearby_target_threshold_pct,
        )
        self.liquidity_raid = LiquidityRaidStrategy(
            volume_multiple=self.settings.strategy.confirmation_volume_multiple,
            range_middle_exclusion=self.settings.strategy.range_middle_exclusion,
            merge_nearby_target_threshold_pct=self.settings.strategy.merge_nearby_target_threshold_pct,
        )
        self.fair_value_gap = FairValueGapStrategy(
            merge_nearby_target_threshold_pct=self.settings.strategy.merge_nearby_target_threshold_pct,
        )
        self.order_block = OrderBlockStrategy(
            merge_nearby_target_threshold_pct=self.settings.strategy.merge_nearby_target_threshold_pct,
        )
        self.choch = CHoCHStrategy(
            merge_nearby_target_threshold_pct=self.settings.strategy.merge_nearby_target_threshold_pct,
        )
        self.strategy_registry: list[tuple[StrategyName, Any]] = [
            (StrategyName.BREAK_RETEST, self.break_retest),
            (StrategyName.LIQUIDITY_RAID, self.liquidity_raid),
            (StrategyName.FAIR_VALUE_GAP, self.fair_value_gap),
            (StrategyName.ORDER_BLOCK, self.order_block),
            (StrategyName.CHOCH, self.choch),
        ]
        self.signal_scorer = SignalScorer(
            trend_weight=self.settings.strategy.trend_weight,
            raid_weight=self.settings.strategy.raid_weight,
            breakout_weight=self.settings.strategy.breakout_weight,
            momentum_weight=self.settings.strategy.momentum_weight,
        )
        self.ev_filter = ExpectedValueFilter(self.settings.ev)
        self.risk_engine = RiskEngine(self.settings)
        self.order_router = OrderRouter(self.settings)
        self.order_manager = OrderManager(self.settings, self.exchange, self.persistence, self.state_store)
        self.sltp_manager = SLTPManager(self.settings)
        self.trade_journal = TradeJournal(self.persistence)
        self.performance = PerformanceAnalyzer(self.persistence)
        self.ai_client = OpenAIResponsesClient(self.settings, self.persistence)
        self.ai_summarizer = AISummarizer(self.ai_client)
        self.news_collector = NewsCollector(self.persistence)
        self.news_analyzer = NewsAnalyzer(self.ai_summarizer, self.persistence, self.settings)
        self.news_filter = NewsImpactFilter(self.persistence)
        self.fill_handler = FillHandler(self.persistence, self.state_store, self.trade_journal)
        self.telegram_router = TelegramCommandRouter(self)
        self.telegram_bot = TelegramBotService(self.settings, self.telegram_router)

    async def start(self) -> None:
        """Start background tasks."""

        self.state_store.set_status(BotStatus.STARTING.value)
        await self.telegram_bot.start()
        await self._notify_startup_attempt()
        await self._restore_runtime_guarded()
        # API 키 로딩 확인 (Railway 디버그용)
        _key = self.settings.secrets.bitget_demo_api_key or self.settings.secrets.bitget_api_key
        _key_preview = f"{_key[:6]}...{_key[-4:]}" if len(_key) > 10 else ("(비어있음)" if not _key else _key)
        self.logger.info(f"API 키 상태: {_key_preview} | 모드: {self.settings.mode.value}")
        self._log_event("시스템 시작", "봇이 시작되었고 상태 복원을 완료했습니다.")
        await self.telegram_bot.broadcast_admins("봇 재시작/장애 복구\n- 상태 복원 완료\n- 기본 모드 안전 확인")
        self.state_store.set_status(BotStatus.RUNNING.value)
        self.tasks = [
            asyncio.create_task(self._universe_loop(), name="universe_loop"),
            asyncio.create_task(self._signal_loop(), name="signal_loop"),
            asyncio.create_task(self._news_loop(), name="news_loop"),
            asyncio.create_task(self._order_sync_loop(), name="order_sync_loop"),
            asyncio.create_task(self._health_loop(), name="health_loop"),
        ]

    async def stop(self) -> None:
        """Stop the application gracefully."""

        self.state_store.set_status(BotStatus.STOPPING.value)
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.telegram_bot.stop()
        await self.telegram_bot.close()
        await self.exchange.close()
        self.state_store.set_status(BotStatus.STOPPED.value)

    async def run_forever(self) -> None:
        """Run until external shutdown."""

        await self.start()
        await self.stop_event.wait()
        await self.stop()

    async def _restore_runtime(self) -> None:
        """Restore pending orders, positions, and managed exits."""

        await self.order_manager.restore_open_orders()
        positions = await self.exchange.get_positions()
        for position in positions:
            existing = self.state_store.state.open_positions.get(position.symbol, {})
            payload = {
                "symbol": position.symbol,
                "mode": self.exchange.mode.value,
                "side": position.side.value,
                "entry_price": position.entry_price,
                "mark_price": position.mark_price,
                "stop_price": position.stop_loss,
                "tp1_price": position.take_profit,
                "tp2_price": position.take_profit,
                "tp3_price": existing.get("tp3_price"),
                "quantity": position.size,
                "strategy": position.raw.get("strategy", ""),
                "stop_reason": existing.get("stop_reason", ""),
                "target_plan": existing.get("target_plan", []),
                "rr_to_tp1": existing.get("rr_to_tp1", 0.0),
                "rr_to_tp2": existing.get("rr_to_tp2", 0.0),
                "rr_to_best_target": existing.get("rr_to_best_target", 0.0),
                "opened_at": existing.get("opened_at"),
                "tp1_done": existing.get("tp1_done", False),
                "tp2_done": existing.get("tp2_done", False),
                "tp3_done": existing.get("tp3_done", False),
                "break_even_moved": existing.get("break_even_moved", False),
                "remaining_quantity": existing.get("remaining_quantity", position.size),
            }
            self.state_store.update_position(position.symbol, payload)
            if position.size > 0 and position.stop_loss and position.take_profit:
                self.sltp_manager.register_payload(payload)
        self.persistence.snapshot_settings(
            self.settings.mode.value,
            build_settings_snapshot(self.settings),
            datetime.now(tz=UTC).isoformat(timespec="seconds"),
        )

    async def _restore_runtime_guarded(self) -> None:
        """Restore runtime state without blocking startup forever."""

        timeout_seconds = max(5, self.settings.telegram.startup_alert_timeout_seconds)
        try:
            await asyncio.wait_for(self._restore_runtime(), timeout=timeout_seconds)
        except TimeoutError:
            self._log_event("상태 복원 지연", f"{timeout_seconds}초 안에 상태 복원이 끝나지 않아 빈 상태로 계속 진행합니다.", level="WARNING")
            await self.telegram_bot.broadcast_admins(
                f"상태 복원 지연\n- {timeout_seconds}초 초과\n- 텔레그램/봇 루프는 계속 시작합니다."
            )
        except Exception as exc:  # noqa: BLE001
            self._log_event("상태 복원 실패", str(exc), level="ERROR")
            await self.telegram_bot.broadcast_admins(
                f"상태 복원 실패\n- 오류: {exc}\n- 빈 상태로 계속 시작합니다."
            )

    async def _notify_startup_attempt(self) -> None:
        """Send a Telegram startup attempt message as early as possible."""

        if not self.settings.telegram.startup_alert_enabled:
            return
        bot_info = await self.telegram_bot.verify_bot()
        bot_name = bot_info.get("username") if bot_info else "unknown"
        await self.telegram_bot.broadcast_admins(
            (
                f"봇 시작 시도\n"
                f"- 모드: {self.settings.mode.value}\n"
                f"- 텔레그램 봇: {bot_name}\n"
                f"- PID: {os.getpid()}\n"
                f"- 시각: {datetime.now(tz=UTC).isoformat(timespec='seconds')}"
            )
        )

    async def _universe_loop(self) -> None:
        """Refresh trackable and active symbols."""

        while not self.stop_event.is_set():
            try:
                snapshot = await self.universe_manager.refresh()
                self.contracts_by_symbol = {item.symbol: item for item in snapshot.tracked_contracts}
                tickers = await self.exchange.get_all_tickers()
                self.latest_tickers = {item.symbol: item for item in tickers}
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log_event("유니버스 갱신 실패", str(exc), level="ERROR")
            await asyncio.sleep(self.settings.universe.refresh_seconds)

    async def _signal_loop(self) -> None:
        """Scan active symbols and place entry orders."""

        while not self.stop_event.is_set():
            try:
                if self.state_store.state.paused:
                    await asyncio.sleep(self.settings.runtime.scan_interval_seconds)
                    continue
                await self._refresh_balance_cache()
                open_positions = self.state_store.state.open_positions
                active_symbols = list(self.state_store.state.active_universe)
                for symbol in active_symbols:
                    if symbol in open_positions:
                        continue
                    if any(order.get("symbol") == symbol for order in self.state_store.state.open_orders.values()):
                        continue
                    contract = self.contracts_by_symbol.get(symbol)
                    if contract is None:
                        continue
                    await self._evaluate_symbol(contract)
                    if self.stop_event.is_set():
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log_event("시그널 루프 오류", str(exc), level="ERROR")
            await asyncio.sleep(self.settings.runtime.scan_interval_seconds)

    async def _evaluate_symbol(self, contract: Any) -> None:
        """Evaluate a single symbol."""

        sym = contract.symbol
        entry_tf = self.settings.timeframes.entry      # 3m
        confirm_tf = self.settings.timeframes.confirm  # 5m
        structure_tf = self.settings.timeframes.structure  # 15m
        # ✅ 1H 제거 — 1m~15m 봉만 사용 (단기 구조 기반)
        frames_raw = await self.klines.get_multi_timeframe(
            contract.product_type,
            contract.symbol,
            [entry_tf, confirm_tf, structure_tf],
            limit=250,
        )
        frames = {
            "3m": frames_raw[entry_tf],
            "5m": frames_raw[confirm_tf],
            "15m": frames_raw[structure_tf],
        }
        levels = self.level_detector.build_levels(frames["3m"], frames["15m"])
        # ✅ regime 분류도 15m 기반으로 (1H 대신 15m 반복 사용)
        regime = self.regime_classifier.classify(frames["15m"], frames["15m"])
        ticker = self.latest_tickers.get(contract.symbol) or await self.exchange.rest.get_ticker(contract.product_type, contract.symbol)
        if ticker is None or ticker.bid_price <= 0 or ticker.ask_price <= 0:
            return
        depth = await self.orderbooks.get_metrics(contract.product_type, contract.symbol)
        historical_stats = self._historical_stats(contract.symbol)
        news_blocks = self.news_filter.active_blocks(contract.symbol)
        context = SignalContext(
            symbol=contract.symbol,
            product_type=contract.product_type.value,
            frames=frames,
            levels=levels.levels,
            level_tags=levels.tags,
            regime=regime.regime,
            regime_notes=regime.notes,
            ticker=ticker.model_dump(mode="json"),
            orderbook=depth.to_payload(),
            historical_stats=historical_stats,
            blocked_by_news=bool(news_blocks),
            news_penalty=0.2 if news_blocks else 0.0,
            trend_quality_score=float(regime.trend_quality_score),
        )

        allowed_strategies = set(regime.allowed_strategies) if self.settings.strategy_router.regime_router_enabled else {name for name, _ in self.strategy_registry}
        enabled_lookup = self.settings.strategy.enabled
        available_registry = filter_strategy_registry(
            self.strategy_registry,
            enabled_lookup=enabled_lookup,
            allowed=allowed_strategies,
            max_active_groups=self.settings.strategy_router.max_active_strategy_groups,
        )
        if not available_registry:
            self.logger.debug("전략 필터링 후 0개", extra={"extra_data": {"symbol": sym, "regime": str(regime.regime), "allowed": [str(s) for s in allowed_strategies]}})
            return
        candidates = get_candidate_signals(available_registry, context, allowed_strategies)
        blocked_candidates: list[StrategySignal] = []
        recent_history = self._recent_strategy_history(contract.symbol)
        candidates, cooldown_blocked = apply_strategy_cooldowns(
            candidates,
            recent_history=recent_history,
            strategy_cooldown_minutes=self.settings.strategy_router.strategy_cooldown_minutes,
            symbol_cooldown_minutes=self.settings.strategy_router.symbol_cooldown_minutes,
        )
        blocked_candidates.extend(cooldown_blocked)
        if not candidates:
            if blocked_candidates:
                self.logger.info("신호 블로킹됨", extra={"extra_data": {"symbol": sym, "blocked_count": len(blocked_candidates), "reasons": [c.blockers for c in blocked_candidates[:3]]}})
            for candidate in blocked_candidates:
                self.trade_journal.log_signal(
                    self._signal_payload(
                        candidate,
                        SignalStatus.BLOCKED.value,
                        expected_value=float(candidate.ev_metrics.get("expected_value_proxy", -1.0)),
                        blockers=list(dict.fromkeys(candidate.blockers)),
                    )
                )
            return

        approved_candidates: list[StrategySignal] = []
        for candidate in candidates:
            risk_distance = max(abs(candidate.entry_price - candidate.stop_price), 1e-9)
            candidate.fees_r = (candidate.entry_price * (self.settings.ev.estimated_maker_fee_bps * 2 / 10_000)) / risk_distance
            candidate.slippage_r = (
                candidate.entry_price
                * ((self.settings.ev.slippage_bps + ticker.spread_bps) / 10_000)
            ) / risk_distance
            viability = evaluate_trade_viability(
                context,
                candidate.stop_price,
                candidate.target_plan,
                candidate.fees_r,
                candidate.slippage_r,
                entry_price=candidate.entry_price,
                side=candidate.side,
                strategy_name=candidate.strategy.value,
                min_rr_to_tp1_break_retest=self.settings.ev.min_rr_to_tp1_break_retest,
                preferred_rr_to_tp2_break_retest=self.settings.ev.preferred_rr_to_tp2_break_retest,
                min_rr_to_tp1_liquidity_raid=self.settings.ev.min_rr_to_tp1_liquidity_raid,
                preferred_rr_to_tp2_liquidity_raid=self.settings.ev.preferred_rr_to_tp2_liquidity_raid,
                min_rr_to_tp1_fair_value_gap=self.settings.ev.min_rr_to_tp1_fair_value_gap,
                preferred_rr_to_tp2_fair_value_gap=self.settings.ev.preferred_rr_to_tp2_fair_value_gap,
                min_rr_to_tp1_order_block=self.settings.ev.min_rr_to_tp1_order_block,
                preferred_rr_to_tp2_order_block=self.settings.ev.preferred_rr_to_tp2_order_block,
                min_rr_to_tp1_choch=self.settings.ev.min_rr_to_tp1_choch,
                preferred_rr_to_tp2_choch=self.settings.ev.preferred_rr_to_tp2_choch,
                reject_trade_if_targets_are_inside_range_middle=self.settings.strategy.reject_trade_if_targets_are_inside_range_middle,
                min_stop_distance_pct=self.settings.risk.min_stop_distance_pct,
            )
            # 👑 SL 기반 TP 계산: SL이 1이면 TP는 2배
            if candidate.stop_price is not None and candidate.stop_price > 0:
                from strategy.rr_filter import compute_quadrant_targets
                quadrants = compute_quadrant_targets(
                    entry_price=candidate.entry_price,
                    stop_price=candidate.stop_price,
                    side=candidate.side
                )
                if quadrants:
                    candidate.tp1_price = quadrants.get("tp1")
                    candidate.tp2_price = quadrants.get("tp2")
                    candidate.tp3_price = quadrants.get("tp3")
                    candidate.tp4_price = quadrants.get("tp4")
                    # TP1-3: 각 50% | TP4: 전량청산
                    candidate.target_plan = [
                        {"price": quadrants["tp1"], "reason": "tp1_50pct_of_remaining", "priority": 1},
                        {"price": quadrants["tp2"], "reason": "tp2_50pct_of_remaining", "priority": 2},
                        {"price": quadrants["tp3"], "reason": "tp3_50pct_of_remaining", "priority": 3},
                        {"price": quadrants["tp4"], "reason": "tp4_full_close", "priority": 4},
                    ]
                    # RR 재계산 (1:2 구조)
                    candidate.rr_to_tp1 = 0.5   # TP1 = 0.5R
                    candidate.rr_to_tp2 = 1.0   # TP2 = 1.0R
                    candidate.rr_to_best_target = 2.0  # 최종 TP4 = 2.0R
                    candidate.expected_r = 2.0
                else:
                    candidate.rr_to_tp1 = viability.rr_to_tp1
                    candidate.rr_to_tp2 = viability.rr_to_tp2
                    candidate.rr_to_best_target = viability.rr_to_best_target
                    candidate.expected_r = viability.rr_to_tp2 or viability.rr_to_best_target or candidate.expected_r
                    if viability.target_plan:
                        candidate.target_plan = viability.target_plan
                        candidate.tp1_price = float(viability.target_plan[0]["price"])
                        if len(viability.target_plan) > 1:
                            candidate.tp2_price = float(viability.target_plan[1]["price"])
                        if len(viability.target_plan) > 2:
                            candidate.tp3_price = float(viability.target_plan[2]["price"])
            else:
                candidate.rr_to_tp1 = viability.rr_to_tp1
                candidate.rr_to_tp2 = viability.rr_to_tp2
                candidate.rr_to_best_target = viability.rr_to_best_target
                candidate.expected_r = viability.rr_to_tp2 or viability.rr_to_best_target or candidate.expected_r
                if viability.target_plan:
                    candidate.target_plan = viability.target_plan
                    candidate.tp1_price = float(viability.target_plan[0]["price"])
                    if len(viability.target_plan) > 1:
                        candidate.tp2_price = float(viability.target_plan[1]["price"])
                    if len(viability.target_plan) > 2:
                        candidate.tp3_price = float(viability.target_plan[2]["price"])
            candidate.ev_metrics = {
                **candidate.ev_metrics,
                **viability.to_dict(),
            }
            if not viability.approved:
                candidate.trade_rejected_reason = viability.reject_reason
                candidate.blockers.append(viability.reject_reason)
                blocked_candidates.append(candidate)
                continue
            candidate.score = self.signal_scorer.score(candidate, regime.regime).final_score
            candidate.display_name = candidate.display_name or candidate.strategy.display_name
            candidate.regime = regime.regime
            approved_candidates.append(candidate)

        for candidate in blocked_candidates:
            self.trade_journal.log_signal(
                self._signal_payload(
                    candidate,
                    SignalStatus.BLOCKED.value,
                    expected_value=float(candidate.ev_metrics.get("expected_value_proxy", -1.0)),
                    blockers=list(dict.fromkeys(candidate.blockers)),
                )
            )

        if not approved_candidates:
            if blocked_candidates:
                self.logger.info("모든 후보 viability 실패", extra={"extra_data": {"symbol": sym, "blocked": len(blocked_candidates), "reasons": [c.trade_rejected_reason for c in blocked_candidates[:3]]}})
            return

        if len(approved_candidates) > self.settings.strategy_router.max_candidate_signals_per_symbol:
            approved_candidates = sorted(approved_candidates, key=lambda item: item.score, reverse=True)[
                : self.settings.strategy_router.max_candidate_signals_per_symbol
            ]

        routing_decision = choose_primary_signal(
            approved_candidates,
            regime=regime,
            router=self.settings.strategy_router,
        )
        for rejected_signal in routing_decision.rejected:
            blockers = list(dict.fromkeys(rejected_signal.blockers + ([rejected_signal.trade_rejected_reason] if rejected_signal.trade_rejected_reason else [])))
            self.trade_journal.log_signal(
                self._signal_payload(
                    rejected_signal,
                    SignalStatus.BLOCKED.value,
                    expected_value=float(rejected_signal.ev_metrics.get("expected_value_proxy", -1.0)),
                    blockers=blockers,
                )
            )

        best_signal = routing_decision.chosen
        if best_signal is None:
            return
        if best_signal.score < self.settings.strategy.min_signal_score:
            self.logger.info("신호 스코어 미달", extra={"extra_data": {"symbol": sym, "strategy": best_signal.strategy.value, "score": best_signal.score, "min_score": self.settings.strategy.min_signal_score}})
            blockers = list(dict.fromkeys(best_signal.blockers + ["low_score"]))
            self.trade_journal.log_signal(
                self._signal_payload(best_signal, SignalStatus.BLOCKED.value, expected_value=-1.0, blockers=blockers)
            )
            return
        # stop_reason 쿨다운 체크: 같은 손절 이유 N회 연속 시 해당 패턴 차단
        signal_stop_reason = getattr(best_signal, "stop_reason", None) or ""
        if signal_stop_reason and self._is_stop_reason_cooled_down(signal_stop_reason):
            cooldown_until = self._stop_reason_cooldowns.get(signal_stop_reason)
            remain_min = int((cooldown_until - datetime.now(UTC)).total_seconds() / 60) if cooldown_until else 0
            self.logger.info(
                "🚫 stop_reason 쿨다운 — 진입 차단",
                extra={"extra_data": {"symbol": sym, "stop_reason": signal_stop_reason, "remain_minutes": remain_min}},
            )
            blockers = list(dict.fromkeys(best_signal.blockers + [f"stop_reason_cooldown:{signal_stop_reason}"]))
            self.trade_journal.log_signal(
                self._signal_payload(best_signal, SignalStatus.BLOCKED.value, expected_value=-1.0, blockers=blockers)
            )
            return
        self.logger.info("✅ 진입 신호 승인!", extra={"extra_data": {"symbol": sym, "strategy": best_signal.strategy.value, "side": best_signal.side, "score": best_signal.score, "entry": best_signal.entry_price}})
        best_signal.chosen_strategy = best_signal.strategy.value
        best_signal.conflict_resolution_decision = best_signal.conflict_resolution_decision or routing_decision.decision
        best_signal.candidate_strategies = best_signal.candidate_strategies or [item.strategy.value for item in approved_candidates]
        if routing_decision.rejected:
            best_signal.rejected_strategies = [item.strategy.value for item in routing_decision.rejected]
            best_signal.rejection_reasons = [
                item.trade_rejected_reason or item.conflict_resolution_decision or "rejected"
                for item in routing_decision.rejected
            ]

        funding_minutes_away = await self._funding_minutes_away(contract)
        ev = self.ev_filter.evaluate(
            best_signal,
            historical_win_rate=historical_stats["win_rate"],
            historical_avg_win_r=historical_stats["avg_win_r"],
            historical_avg_loss_r=historical_stats["avg_loss_r"],
            spread_bps=ticker.spread_bps,
            funding_minutes_away=funding_minutes_away,
            news_penalty=context.news_penalty,
        )
        best_signal.ev_metrics = {
            **best_signal.ev_metrics,
            "expected_value": ev.expected_value,
            "p_win": ev.p_win,
            "avg_win_r": ev.avg_win_r,
            "avg_loss_r": ev.avg_loss_r,
            "fees_r": ev.fees_r,
            "slippage_r": ev.slippage_r,
            "event_penalty": ev.event_penalty,
            "funding_penalty": ev.funding_penalty,
            "spread_penalty": ev.spread_penalty,
            "approved": ev.approved,
            "reject_reason": ev.reject_reason,
            "rr_to_tp1": ev.rr_to_tp1,
            "rr_to_tp2": ev.rr_to_tp2,
            "rr_to_best_target": ev.rr_to_best_target,
        }
        if not ev.approved:
            self.logger.info("EV 필터 거절", extra={"extra_data": {"symbol": sym, "reason": ev.reject_reason, "ev": ev.expected_value, "rr_tp1": ev.rr_to_tp1, "p_win": ev.p_win}})
            blockers = list(dict.fromkeys(best_signal.blockers + [ev.reject_reason or "ev_filter"]))
            payload = self._signal_payload(best_signal, SignalStatus.BLOCKED.value, expected_value=ev.expected_value, blockers=blockers)
            self.trade_journal.log_signal(payload)
            return

        runtime_metrics = self._runtime_metrics()
        approved = self.risk_engine.approve(
            signal=best_signal,
            contract=contract,
            account_equity=self.last_balance.get("balance", 0.0),
            open_positions=self.state_store.state.open_positions,
            runtime_metrics=runtime_metrics,
            atr_value=best_signal.risk_per_unit,
            news_blocked=bool(news_blocks),
            funding_blocked=funding_minutes_away is not None and funding_minutes_away <= self.settings.risk.funding_block_before_minutes,
        )
        if approved is None:
            # max_concurrent_positions 차단 시: 횡보 포지션 먼저 정리 후 재시도
            if "max_concurrent_positions" in best_signal.blockers:
                evicted = await self._evict_stale_positions()
                if evicted:
                    # 슬롯이 생겼으므로 재승인 시도
                    best_signal.blockers.clear()
                    approved = self.risk_engine.approve(
                        signal=best_signal,
                        contract=contract,
                        account_equity=self.last_balance.get("balance", 0.0),
                        open_positions=self.state_store.state.open_positions,
                        runtime_metrics=self._runtime_metrics(),
                        atr_value=best_signal.risk_per_unit,
                        news_blocked=bool(news_blocks),
                        funding_blocked=funding_minutes_away is not None and funding_minutes_away <= self.settings.risk.funding_block_before_minutes,
                    )
            if approved is None:
                self.logger.info("리스크 엔진 거절", extra={"extra_data": {"symbol": sym, "blockers": best_signal.blockers[:3]}})
                payload = self._signal_payload(
                    best_signal,
                    SignalStatus.BLOCKED.value,
                    expected_value=ev.expected_value,
                    blockers=list(dict.fromkeys(best_signal.blockers)),
                )
                self.trade_journal.log_signal(payload)
                return
        best_signal.tp1_price = approved.tp1_price
        best_signal.tp2_price = approved.tp2_price
        best_signal.tp3_price = approved.tp3_price
        best_signal.tp4_price = approved.tp4_price
        best_signal.target_plan = approved.target_plan
        best_signal.rr_to_tp1 = approved.rr_to_tp1
        best_signal.rr_to_tp2 = approved.rr_to_tp2
        best_signal.rr_to_best_target = approved.rr_to_best_target

        # 단타 핵심 레벨만 차트에 표시 (세션/전일 레벨 제외)
        _sr_label_map = {
            "swing_high_4h": ("매물대 고점", "#f97316", "--"),
            "swing_low_4h": ("매물대 저점", "#f97316", "--"),
            "range_high_recent": ("레인지 고점", "#6b7280", "--"),
            "range_low_recent": ("레인지 저점", "#6b7280", "--"),
        }
        if best_signal.rationale is None:
            best_signal.rationale = {}
        existing_chart_levels = list(best_signal.rationale.get("chart_levels", []))
        existing_prices = {float(item.get("price", 0)) for item in existing_chart_levels}
        for key, (label, color, ls) in _sr_label_map.items():
            price = levels.levels.get(key)
            if price and float(price) not in existing_prices:
                existing_chart_levels.append({"label": label, "price": float(price), "color": color, "linestyle": ls})
                existing_prices.add(float(price))
        best_signal.rationale["chart_levels"] = existing_chart_levels

        payload = self._signal_payload(best_signal, SignalStatus.APPROVED.value, expected_value=ev.expected_value)
        self.trade_journal.log_signal(payload)

        routing = self.order_router.build_entry_order(
            approved_trade=approved,
            contract=contract,
            best_bid=ticker.bid_price,
            best_ask=ticker.ask_price,
            spread_bps=ticker.spread_bps,
            volatility_score=min(1.0, abs(ticker.change_24h)),
        )

        try:
            result = await self.order_manager.submit_entry(approved, contract, routing.order, routing.rationale)
        except Exception as exc:  # noqa: BLE001
            self._log_event("진입 주문 실패", f"{contract.symbol}: {exc}", level="ERROR")
            return

        if routing.order.order_type == OrderType.MARKET:
            fill_price = self._best_execution_price(ticker, best_signal.side.value)
            fill_payload = {
                **self.state_store.state.open_orders.get(result.client_order_id, {}),
                "client_order_id": result.client_order_id,
                "exchange_order_id": result.exchange_order_id,
                "symbol": contract.symbol,
                "mode": self.exchange.mode.value,
                "side": best_signal.side.value,
                "order_type": routing.order.order_type.value,
                "price": fill_price,
                "avg_fill_price": fill_price,
                "quantity": approved.quantity,
                "filled_quantity": approved.quantity,
                "stop_price": approved.stop_price,
                "tp1_price": approved.tp1_price,
                "tp2_price": approved.tp2_price,
                "tp3_price": approved.tp3_price,
                "strategy": best_signal.strategy.value,
                "signal_id": best_signal.signal_id,
                "tags": best_signal.tags,
                "rationale": best_signal.rationale,
                "stop_reason": approved.stop_reason,
                "target_plan": approved.target_plan,
                "rr_to_tp1": approved.rr_to_tp1,
                "rr_to_tp2": approved.rr_to_tp2,
                "rr_to_best_target": approved.rr_to_best_target,
                "ev_metrics": approved.ev_metrics,
                "leverage": approved.leverage,
                "display_name": best_signal.display_name,
                "regime": best_signal.regime.value,
                "candidate_strategies": best_signal.candidate_strategies,
                "chosen_strategy": best_signal.chosen_strategy or best_signal.strategy.value,
                "rejected_strategies": best_signal.rejected_strategies,
                "rejection_reasons": best_signal.rejection_reasons,
                "overlap_score": best_signal.overlap_score,
                "conflict_resolution_decision": best_signal.conflict_resolution_decision,
            }
            await self._finalize_market_entry(
                contract=contract,
                fill_payload=fill_payload,
                chart_label="진입",
                chart_notes=[
                    f"전략: {best_signal.display_name or best_signal.strategy.value}",
                    f"레짐: {best_signal.regime.display_name}",
                    f"태그: {', '.join(best_signal.tags[:3])}",
                    f"손절사유: {approved.stop_reason}",
                    f"타깃: {', '.join(item.get('reason', '-') for item in approved.target_plan[:3])}",
                    f"RR(TP1/TP2): {approved.rr_to_tp1:.2f} / {approved.rr_to_tp2:.2f}",
                    f"충돌해결: {best_signal.conflict_resolution_decision or 'single_candidate'}",
                ],
            )

    async def _news_loop(self) -> None:
        """Collect and analyze news periodically."""

        while not self.stop_event.is_set():
            try:
                if not self.settings.news.enabled:
                    await asyncio.sleep(self.settings.news.poll_seconds)
                    continue
                items = await self.news_collector.fetch_all()
                for item in items:
                    analysis = await self.news_analyzer.analyze_item(item)
                    alert_payload = {
                        "title": item.title,
                        "summary_ko": analysis.summary_ko,
                        "impact_level": analysis.impact_level,
                        "direction_bias": analysis.direction_bias,
                        "should_block_new_entries": analysis.should_block_new_entries,
                    }
                    self.latest_news_alerts.insert(0, alert_payload)
                    self.latest_news_alerts = self.latest_news_alerts[:20]
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log_event("뉴스 수집 오류", str(exc), level="ERROR")
            await asyncio.sleep(self.settings.news.poll_seconds)

    async def _order_sync_loop(self) -> None:
        """Sync orders, detect fills, and manage exits."""

        while not self.stop_event.is_set():
            try:
                await self._process_control_commands()
                await self._process_pending_fills()
                await self.order_manager.sync_pending_orders()
                await self._refresh_positions_from_exchange()
                await self._process_managed_exits()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                _tb_str = _tb.format_exc()
                self._log_event("주문 동기화 오류", f"{exc}\n{_tb_str}", level="ERROR")
                self.logger.error(
                    "주문 동기화 루프 예외",
                    extra={"extra_data": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": _tb_str,
                        "open_orders": list(self.state_store.state.open_orders.keys()),
                        "open_positions": list(self.state_store.state.open_positions.keys()),
                    }},
                )
            await asyncio.sleep(5)

    async def _health_loop(self) -> None:
        """Periodic health, reload, and summary loop."""

        while not self.stop_event.is_set():
            try:
                self.state_store.state.last_healthcheck_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
                self.state_store.save()
                await self._refresh_balance_cache()
                await self._process_control_commands()
                await self._reload_if_changed()
                await self._update_daily_stats()
                await self._maybe_send_summaries()
                await self._maybe_send_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log_event("헬스체크 오류", str(exc), level="ERROR")
            await asyncio.sleep(self.settings.runtime.healthcheck_seconds)

    async def _maybe_send_heartbeat(self) -> None:
        """Optionally send periodic heartbeat notifications."""

        if not self.settings.telegram.heartbeat_alert_enabled:
            return
        now = datetime.now(tz=UTC)
        minutes = max(5, self.settings.telegram.heartbeat_minutes)
        if self.last_heartbeat_sent_at and (now - self.last_heartbeat_sent_at) < timedelta(minutes=minutes):
            return
        self.last_heartbeat_sent_at = now
        await self.telegram_bot.broadcast_admins(
            (
                f"봇 헬스체크\n"
                f"- 상태: {self.state_store.state.bot_status}\n"
                f"- 모드: {self.settings.mode.value}\n"
                f"- 활성 유니버스: {len(self.state_store.state.active_universe)}\n"
                f"- 열린 포지션: {len(self.state_store.state.open_positions)}\n"
                f"- 잔고: {self.last_balance.get('balance', 0):,.2f} USDT"
            )
        )

    async def _refresh_positions_from_exchange(self) -> None:
        """Refresh runtime position state."""

        positions = await self.exchange.get_positions()
        seen = set()
        for position in positions:
            existing = self.state_store.state.open_positions.get(position.symbol, {})
            raw = position.raw or {}
            payload = {
                "symbol": position.symbol,
                "mode": self.exchange.mode.value,
                "side": position.side.value,
                "entry_price": position.entry_price,
                "mark_price": position.mark_price,
                "stop_price": position.stop_loss or existing.get("stop_price"),
                "tp1_price": existing.get("tp1_price"),
                "tp2_price": existing.get("tp2_price"),
                "tp3_price": existing.get("tp3_price"),
                "tp3_done": existing.get("tp3_done", False),
                "quantity": position.size,
                "unrealized_pnl": position.unrealized_pnl,
                "used_margin": position.used_margin,
                "leverage": existing.get("leverage", raw.get("leverage")),
                "strategy": existing.get("strategy", raw.get("strategy", "")),
                "display_name": existing.get("display_name", raw.get("display_name", "")),
                "rationale": existing.get("rationale", {}),
                "tags": existing.get("tags", []),
                "market_regime": existing.get("market_regime", ""),
                "chosen_strategy": existing.get("chosen_strategy", existing.get("strategy", "")),
                "candidate_strategies": existing.get("candidate_strategies", []),
                "rejected_strategies": existing.get("rejected_strategies", []),
                "rejection_reasons": existing.get("rejection_reasons", []),
                "overlap_score": existing.get("overlap_score", 0.0),
                "conflict_resolution_decision": existing.get("conflict_resolution_decision", ""),
                "stop_reason": existing.get("stop_reason"),
                "target_plan": existing.get("target_plan", []),
                "rr_to_tp1": existing.get("rr_to_tp1", 0.0),
                "rr_to_tp2": existing.get("rr_to_tp2", 0.0),
                "rr_to_best_target": existing.get("rr_to_best_target", 0.0),
                "remaining_quantity": existing.get("remaining_quantity", position.size),
                "tp1_done": existing.get("tp1_done", False),
                "tp2_done": existing.get("tp2_done", False),
                "break_even_moved": existing.get("break_even_moved", False),
            }
            seen.add(position.symbol)
            self.state_store.update_position(position.symbol, payload)
            if position.symbol not in self.sltp_manager._managed and payload["stop_price"] and payload["tp1_price"]:
                self.sltp_manager.register_payload(payload)

        for symbol in list(self.state_store.state.open_positions.keys()):
            if symbol not in seen:
                self.state_store.remove_position(symbol)
                self.sltp_manager.remove(symbol)

    async def _process_pending_fills(self) -> None:
        """Look up fills for pending orders (both entry and close)."""

        open_order_count = len(self.state_store.state.open_orders)
        if open_order_count:
            self.logger.info(
                "주문 체결 조회 시작",
                extra={"extra_data": {"open_order_count": open_order_count, "symbols": list(self.state_store.state.open_orders.keys())}},
            )

        for client_oid, order in list(self.state_store.state.open_orders.items()):
            contract = self.contracts_by_symbol.get(order["symbol"])
            if contract is None:
                self.logger.warning(
                    "주문의 심볼이 유니버스에 없음 — 건너뜀",
                    extra={"extra_data": {"client_order_id": client_oid, "symbol": order.get("symbol")}},
                )
                continue

            # Check if this is a reduce-only (close) order
            is_close_order = bool(order.get("reduce_only"))

            fills = await self.exchange.rest.get_fills(
                contract.product_type,
                symbol=contract.symbol,
                order_id=order.get("exchange_order_id"),
            )
            self.logger.info(
                "체결 조회 결과",
                extra={"extra_data": {
                    "symbol": contract.symbol,
                    "client_order_id": client_oid,
                    "exchange_order_id": order.get("exchange_order_id"),
                    "is_close_order": is_close_order,
                    "fill_count": len(fills),
                    "latest_fill_size": float(fills[-1].size) if fills else None,
                    "latest_fill_price": float(fills[-1].price) if fills else None,
                }},
            )
            if not fills:
                # Only apply fallback logic for entry orders, not close orders
                if is_close_order:
                    continue

                fallback_result = await self.order_manager.replace_stale_limit_with_market(
                    contract=contract,
                    order_payload=order,
                )
                if fallback_result is None:
                    continue
                ticker = self.latest_tickers.get(contract.symbol) or await self.exchange.rest.get_ticker(
                    contract.product_type,
                    contract.symbol,
                )
                fill_price = self._best_execution_price(ticker, str(order.get("side", "long")))
                fill_payload = {
                    **order,
                    "client_order_id": fallback_result.client_order_id,
                    "exchange_order_id": fallback_result.exchange_order_id,
                    "order_type": OrderType.MARKET.value,
                    "status": fallback_result.status.value,
                    "reason": "maker_timeout_fallback",
                    "avg_fill_price": fill_price,
                    "price": fill_price,
                    "filled_quantity": float(order.get("quantity") or 0.0),
                    "quantity": float(order.get("quantity") or 0.0),
                }
                await self._finalize_market_entry(
                    contract=contract,
                    fill_payload=fill_payload,
                    chart_label="진입",
                    chart_notes=[
                        f"전략: {order.get('strategy', '-')}",
                        "체결 경로: 지정가 미체결 -> 시장가 fallback",
                    ],
                )
                continue

            fill = fills[-1]

            # ✅ Handle close order fills
            if is_close_order:
                if not fill or fill.price is None or fill.size is None:
                    self.logger.warning(
                        "Close order fill invalid - missing price or size",
                        extra={"extra_data": {"symbol": contract.symbol, "fill": str(fill)}}
                    )
                    continue
                position = self.state_store.state.open_positions.get(contract.symbol)
                close_reason = order.get("reason", "manual_close")
                is_partial = close_reason in {"partial_tp1", "partial_tp2", "partial_tp3"}

                if is_partial:
                    # 분할익절: _finalize_position_close 하지 않고 remaining_quantity만 업데이트
                    self.sltp_manager.confirm_partial_fill(contract.symbol, float(fill.size))
                    self.state_store.remove_order(client_oid)
                    self.logger.info(
                        "Partial TP fill confirmed — position kept open",
                        extra={"extra_data": {"symbol": contract.symbol, "reason": close_reason, "filled_qty": float(fill.size), "fill_price": float(fill.price)}}
                    )
                elif position:
                    # 원래 청산 사유 보존 (stop_out, final_target_exit, time_stop 등)
                    human_reason = self._human_action_reason(close_reason)
                    chart_label = self._exit_chart_label(close_reason)
                    try:
                        await self._finalize_position_close(
                            contract=contract,
                            position=position,
                            exit_price=float(fill.price),
                            exit_reason=close_reason,
                            chart_label=chart_label,
                            chart_notes=[f"{human_reason} (체결가: {float(fill.price):,.4f})", f"손절사유: {position.get('stop_reason', '')}"],
                        )
                    finally:
                        # 에러 여부 무관하게 반드시 order 제거 (재시도 루프 방지)
                        self.state_store.remove_order(client_oid)
                    self.logger.info(
                        "Close order filled and position closed",
                        extra={"extra_data": {"symbol": contract.symbol, "fill_price": float(fill.price), "filled_qty": float(fill.size)}}
                    )
                else:
                    self.logger.warning(
                        "Close order filled but position not found in state",
                        extra={"extra_data": {"symbol": contract.symbol, "fill_price": float(fill.price)}}
                    )
                    self.state_store.remove_order(client_oid)
                continue

            # Entry order fill
            if not fill or fill.price is None or fill.size is None or fill.size <= 0:
                self.logger.warning(
                    "진입 체결 무효 — size 누락 또는 0 (알림 생략, 주문 유지)",
                    extra={"extra_data": {
                        "symbol": contract.symbol,
                        "client_order_id": client_oid,
                        "exchange_order_id": order.get("exchange_order_id"),
                        "fill_price": float(fill.price) if fill and fill.price is not None else None,
                        "fill_size": float(fill.size) if fill and fill.size is not None else None,
                        "fill_raw": fill.raw if fill else None,
                    }}
                )
                continue

            # 부분체결 감지: fill 합계 < 주문수량 99% → 완전체결 대기, 알림 금지
            valid_fills = [f for f in fills if f.size and f.size > 0 and f.price and f.price > 0]
            total_filled = sum(float(f.size) for f in valid_fills)
            order_quantity = float(order.get("quantity") or 0.0)
            if order_quantity > 0 and total_filled < order_quantity * 0.99:
                self.logger.info(
                    "진입 부분 체결 — 완전 체결 대기 (알림/처리 보류)",
                    extra={"extra_data": {
                        "symbol": contract.symbol,
                        "client_order_id": client_oid,
                        "total_filled": total_filled,
                        "order_quantity": order_quantity,
                        "fill_pct": round(total_filled / order_quantity * 100, 2),
                    }},
                )
                continue

            # 전량(또는 99%↑) 체결: 가중평균가 계산 후 진입 확정
            total_value = sum(float(f.price) * float(f.size) for f in valid_fills)
            avg_fill_price = total_value / max(total_filled, 1e-10)
            self.logger.info(
                "진입 체결 확인 — _finalize_market_entry 호출",
                extra={"extra_data": {
                    "symbol": contract.symbol,
                    "client_order_id": client_oid,
                    "exchange_order_id": order.get("exchange_order_id"),
                    "avg_fill_price": avg_fill_price,
                    "total_filled": total_filled,
                    "order_quantity": order_quantity,
                    "side": order.get("side"),
                    "strategy": order.get("strategy"),
                    "leverage": order.get("leverage"),
                }},
            )
            fill_payload = {
                **order,
                "client_order_id": client_oid,
                "exchange_order_id": order.get("exchange_order_id"),
                "avg_fill_price": avg_fill_price,
                "price": avg_fill_price,
                "filled_quantity": total_filled,
                "quantity": order.get("quantity") or total_filled,
            }
            try:
                await self._finalize_market_entry(
                    contract=contract,
                    fill_payload=fill_payload,
                    chart_label="진입",
                    chart_notes=[
                        f"전략: {order.get('strategy', '-')}",
                        "체결 경로: pending fill sync",
                    ],
                )
                self.logger.info(
                    "진입 완료 처리 성공",
                    extra={"extra_data": {"symbol": contract.symbol, "client_order_id": client_oid}},
                )
            except Exception as _exc:
                self.logger.error(
                    "진입 완료 처리 중 예외 — 주문은 강제 제거됨",
                    extra={"extra_data": {
                        "symbol": contract.symbol,
                        "client_order_id": client_oid,
                        "error": str(_exc),
                    }},
                )
                raise
            finally:
                # 에러 여부 무관하게 반드시 order 제거 (재시도 루프 방지)
                self.state_store.remove_order(client_oid)

    async def _process_managed_exits(self) -> None:
        """Apply TP/SL logic to active positions."""

        for symbol, position in list(self.state_store.state.open_positions.items()):
            ticker = self.latest_tickers.get(symbol)
            mark_price = position.get("mark_price")
            if not mark_price and ticker is not None:
                mark_price = ticker.last_price or ticker.mark_price
            if not mark_price:
                continue
            actions = self.sltp_manager.evaluate_price(
                symbol, float(mark_price),
                open_position_count=len(self.state_store.state.open_positions),
            )
            if not actions:
                continue
            contract = self.contracts_by_symbol.get(symbol)
            if contract is None:
                continue
            for action in actions:
                if action["action"] == "move_stop":
                    position["stop_price"] = action["new_stop_price"]
                    position["break_even_moved"] = True
                    self.state_store.update_position(symbol, position)
                    continue

                await self.order_manager.close_position_market(
                    symbol=symbol,
                    product_type=contract.product_type,
                    margin_coin=contract.margin_coin,
                    side=position["side"],
                    quantity=float(action["quantity"]),
                    close_reason=action["action"],
                )
                if action["action"] in {"partial_tp1", "partial_tp2", "partial_tp3"}:
                    position["quantity"] = max(0.0, float(position.get("quantity", 0.0)) - float(action["quantity"]))
                    position["remaining_quantity"] = position["quantity"]
                    if action["action"] == "partial_tp1":
                        position["tp1_done"] = True
                    if action["action"] == "partial_tp2":
                        position["tp2_done"] = True
                    if action["action"] == "partial_tp3":
                        position["tp3_done"] = True
                    self.state_store.update_position(symbol, position)
                    _entry_p = float(position.get("entry_price") or 0.0)
                    _close_qty = float(action["quantity"])
                    _lev = float(position.get("leverage") or 1.0) or 1.0
                    _side_str = str(position.get("side", "long"))
                    if _entry_p > 0 and _close_qty > 0:
                        if _side_str == "long":
                            _pnl_usdt = (mark_price - _entry_p) * _close_qty
                            _pnl_pct = (mark_price - _entry_p) / _entry_p * 100 * _lev
                        else:
                            _pnl_usdt = (_entry_p - mark_price) * _close_qty
                            _pnl_pct = (_entry_p - mark_price) / _entry_p * 100 * _lev
                        _profit_str = f"\n- 수익: {_pnl_pct:+.2f}% ({_pnl_usdt:+.4f} USDT)"
                    else:
                        _profit_str = ""
                    await self.telegram_bot.broadcast_admins(
                        f"{self._human_action_reason(action['action'])}\n- 심볼: {symbol}\n- 수량: {action['quantity']}{_profit_str}"
                    )
                    await self._broadcast_trade_chart(
                        contract=contract,
                        event_label="부분 청산",
                        side=str(position.get("side", "long")),
                        entry_price=float(position.get("entry_price") or 0.0),
                        current_price=float(mark_price),
                        stop_price=float(position.get("stop_price") or 0.0) or None,
                        tp1_price=float(position.get("tp1_price") or 0.0) or None,
                        tp2_price=float(position.get("tp2_price") or 0.0) or None,
                        tp3_price=float(position.get("tp3_price") or 0.0) or None,
                        tp4_price=float(position.get("tp4_price") or 0.0) or None,
                        final_target_price=self._primary_target_from_plan(
                            position.get("target_plan") or [],
                            str(position.get("side", "long")),
                            float(position.get("entry_price") or 0.0),
                        ),
                        quantity=float(action["quantity"]),
                        entry_notional_usdt=float(position.get("entry_price") or 0.0) * float(position.get("quantity") or 0.0),
                        remaining_notional_usdt=float(mark_price) * float(position.get("quantity") or 0.0),
                        leverage=float(position.get("leverage", 0.0) or 0.0) or None,
                        notes=[f"사유: {self._human_action_reason(action['action'])}", f"손절사유: {position.get('stop_reason', '')}"],
                        strategy_name=str(position.get("display_name") or position.get("chosen_strategy") or position.get("strategy", "")),
                        current_regime=str(position.get("market_regime", "")),
                        conflict_resolution_summary=str(position.get("conflict_resolution_decision", "")),
                        stop_reason=str(position.get("stop_reason", "")),
                        target_reasons=[item.get("reason", "") for item in (position.get("target_plan") or [])[:3] if isinstance(item, dict)],
                        entry_reason_title=str((position.get("rationale") or {}).get("entry_reason_title", "")),
                        entry_reason_lines=list((position.get("rationale") or {}).get("entry_reason_lines", [])),
                        chart_levels=list((position.get("rationale") or {}).get("chart_levels", [])),
                        chart_zones=list((position.get("rationale") or {}).get("chart_zones", [])),
                        chart_marker=dict((position.get("rationale") or {}).get("chart_marker", {})),
                        rr_to_tp1=float(position.get("rr_to_tp1") or 0.0) or None,
                        rr_to_tp2=float(position.get("rr_to_tp2") or 0.0) or None,
                    )
                else:
                    # ✅ FIXED: Don't close immediately. Wait for actual fill.
                    # close_position_market order is persisted above.
                    # Fill will be detected in _process_pending_fills -> _finalize_position_close with actual fill_price
                    self.logger.info(
                        "Close order submitted, waiting for fill",
                        extra={"extra_data": {"symbol": symbol, "reason": action["action"], "mark_price": mark_price}}
                    )

    async def _refresh_balance_cache(self) -> None:
        """Update cached balance summary."""

        accounts = await self.exchange.get_accounts()
        balance = sum(float(item.usdt_equity) for item in accounts)
        used_margin = sum(float(item.crossed_margin + item.isolated_margin) for item in accounts)
        unrealized = sum(float(item.unrealized_pnl) for item in accounts)
        self.last_balance = {
            "balance": balance,
            "used_margin": used_margin,
            "unrealized_pnl": unrealized,
            "realized_pnl": self.performance.summarize().realized_pnl,
        }

    async def _reload_if_changed(self) -> None:
        """Reload settings when bot_settings.json changes."""

        changed = self.settings_manager.reload_if_changed()
        if changed is None:
            return
        old_mode = self.settings.mode
        self.settings = changed
        await self.exchange.close()
        await self.telegram_bot.stop()
        await self.telegram_bot.close()
        self._build_components()
        await self.telegram_bot.start()
        self._log_event("설정 리로드", "bot_settings.json 변경사항을 반영했습니다.")
        await self.telegram_bot.broadcast_admins(
            f"설정 리로드\n- 모드: {old_mode.value} -> {self.settings.mode.value}"
        )

    def _load_control_commands(self) -> dict[str, Any]:
        """Load queued dashboard control commands from disk."""

        return load_json(CONTROL_COMMANDS_PATH, default={"pending": [], "history": []})

    def _save_control_commands(self, payload: dict[str, Any]) -> None:
        """Persist control command queue to disk."""

        dump_json(CONTROL_COMMANDS_PATH, payload)

    async def _process_control_commands(self) -> None:
        """Process dashboard-issued control commands."""

        queue = self._load_control_commands()
        pending = [item for item in queue.get("pending", []) if isinstance(item, dict)]
        if not pending:
            return

        history = [item for item in queue.get("history", []) if isinstance(item, dict)]
        processed_any = False
        for command in pending:
            command_id = str(command.get("id") or "")
            action = str(command.get("action") or "")
            payload = command.get("payload") or {}
            result_text = "처리되지 않음"
            status = "ignored"
            try:
                if action == "close_all_positions":
                    result_text = await self._close_all_positions_internal(
                        enforce_permission=False,
                        source="dashboard",
                    )
                    status = "processed"
                    self._log_event("대시보드 모두청산 요청", result_text, level="WARNING")
                else:
                    result_text = f"지원하지 않는 제어 명령: {action}"
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                result_text = str(exc)
                self._log_event("대시보드 명령 실패", f"{action}: {exc}", level="ERROR")

            history.append(
                {
                    **command,
                    "payload": payload,
                    "status": status,
                    "result": result_text,
                    "processed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                    "command_id": command_id,
                }
            )
            processed_any = True

        if processed_any:
            self._save_control_commands({"pending": [], "history": history[-100:]})

    async def _update_daily_stats(self) -> None:
        """Recalculate daily stats from journal rows."""

        summary = self.performance.summarize()
        today = datetime.now(tz=UTC).date().isoformat()
        self.persistence.save_daily_stats(
            {
                "trade_date": today,
                "mode": self.settings.mode.value,
                "realized_pnl": summary.realized_pnl,
                "unrealized_pnl": self.last_balance.get("unrealized_pnl", 0.0),
                "pnl_r": 0.0,
                "fees": 0.0,
                "trade_count": summary.trade_count,
                "win_count": int(round(summary.win_rate * summary.trade_count)),
                "loss_count": summary.trade_count - int(round(summary.win_rate * summary.trade_count)),
                "max_drawdown": summary.max_drawdown,
                "metadata": {"profit_factor": summary.profit_factor, "expectancy": summary.expectancy},
                "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
        )

    async def _maybe_send_summaries(self) -> None:
        """Send daily and weekly summaries via Telegram."""

        now = datetime.now()
        perf = self.performance.summarize()
        daily_key = now.date().isoformat()
        if now.hour == self.settings.news.daily_summary_hour_kst and self.last_daily_summary_date != daily_key:
            self.last_daily_summary_date = daily_key
            await self.telegram_bot.broadcast_admins(
                format_daily_summary(
                    {
                        "title": "일일 요약",
                        "trade_count": perf.trade_count,
                        "win_rate": perf.win_rate,
                        "realized_pnl": perf.realized_pnl,
                        "profit_factor": perf.profit_factor,
                        "expectancy": perf.expectancy,
                    }
                )
            )
        weekly_key = f"{now.isocalendar().year}-W{now.isocalendar().week}"
        if now.weekday() == 0 and now.hour == self.settings.news.daily_summary_hour_kst and self.last_weekly_summary_key != weekly_key:
            self.last_weekly_summary_key = weekly_key
            await self.telegram_bot.broadcast_admins(
                format_daily_summary(
                    {
                        "title": "주간 요약",
                        "trade_count": perf.trade_count,
                        "win_rate": perf.win_rate,
                        "realized_pnl": perf.realized_pnl,
                        "profit_factor": perf.profit_factor,
                        "expectancy": perf.expectancy,
                    }
                )
            )

    async def _funding_minutes_away(self, contract: Any) -> int | None:
        """Return minutes until the next funding event."""

        try:
            funding = await self.exchange.rest.get_next_funding_time(contract.product_type, contract.symbol)
            raw = funding.get("nextSettleTime") or funding.get("nextFundingTime")
            if not raw:
                return None
            next_time = datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
            delta = next_time - datetime.now(tz=UTC)
            return max(0, int(delta.total_seconds() // 60))
        except Exception:  # noqa: BLE001
            return None

    def _historical_stats(self, symbol: str) -> dict[str, float]:
        """Compute recent win/loss statistics for EV estimation."""

        rows = self.persistence.fetchall(
            "SELECT * FROM trades WHERE symbol = ? AND status = 'closed' ORDER BY id DESC LIMIT 30",
            (symbol,),
        )
        if not rows:
            return {"win_rate": 0.45, "avg_win_r": 1.25, "avg_loss_r": 1.0}
        wins = [float(row.get("pnl_r") or 0.0) for row in rows if float(row.get("pnl_r") or 0.0) > 0]
        losses = [abs(float(row.get("pnl_r") or 0.0)) for row in rows if float(row.get("pnl_r") or 0.0) <= 0]
        return {
            "win_rate": len(wins) / len(rows),
            "avg_win_r": sum(wins) / len(wins) if wins else 1.2,
            "avg_loss_r": sum(losses) / len(losses) if losses else 1.0,
        }

    def _recent_strategy_history(self, symbol: str) -> list[dict[str, Any]]:
        """Return recent signal/trade history for cooldown and conflict routing."""

        rows = self.persistence.fetchall(
            """
            SELECT symbol, strategy, created_at, closed_at
            FROM trades
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (symbol,),
        )
        rows.extend(
            self.persistence.fetchall(
                """
                SELECT symbol, strategy, created_at, NULL AS closed_at
                FROM signals
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (symbol,),
            )
        )
        return rows

    def _runtime_metrics(self) -> dict[str, Any]:
        """Build current risk metrics."""

        today = datetime.now(tz=UTC).date().isoformat()
        daily = self.persistence.fetchone("SELECT * FROM daily_stats WHERE trade_date = ?", (today,)) or {}
        closed_trades = self.persistence.fetchall(
            "SELECT pnl_r, closed_at FROM trades WHERE status = 'closed' ORDER BY id DESC LIMIT 20"
        )
        consecutive_losses = 0
        for row in closed_trades:
            if float(row.get("pnl_r") or 0.0) < 0:
                consecutive_losses += 1
            else:
                break
        return {
            "daily_loss_r": float(daily.get("pnl_r") or 0.0),
            "consecutive_losses": consecutive_losses,
            "daily_order_count": len(
                self.persistence.fetchall(
                    "SELECT id FROM orders WHERE created_at >= ?",
                    (today,),
                )
            ),
            "paused": self.state_store.state.paused,
            "account_drawdown_pct": (daily.get("max_drawdown") or 0.0),
            "unrealized_loss_pct": abs(min(0.0, self.last_balance.get("unrealized_pnl", 0.0))) / max(self.last_balance.get("balance", 1.0), 1.0) * 100,
        }

    def _signal_payload(
        self,
        signal: Any,
        status: str,
        *,
        expected_value: float,
        blockers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Serialize a signal for persistence."""

        return {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "strategy": signal.strategy.value,
            "display_name": signal.display_name or signal.strategy.display_name,
            "side": signal.side.value,
            "status": status,
            "market_regime": signal.regime.value if isinstance(signal.regime, RegimeType) else str(signal.regime or ""),
            "score": signal.score,
            "expected_value": expected_value,
            "expected_r": signal.expected_r,
            "fees_r": signal.fees_r,
            "slippage_r": signal.slippage_r,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "tp1_price": signal.tp1_price,
            "tp2_price": signal.tp2_price,
            "tp3_price": signal.tp3_price,
            "tp4_price": signal.tp4_price,
            "stop_reason": signal.stop_reason,
            "target_plan": signal.target_plan,
            "target_reasons": [item.get("reason", "") for item in signal.target_plan[:3]],
            "rr_to_tp1": signal.rr_to_tp1,
            "rr_to_tp2": signal.rr_to_tp2,
            "rr_to_best_target": signal.rr_to_best_target,
            "candidate_strategies": signal.candidate_strategies,
            "chosen_strategy": signal.chosen_strategy or signal.strategy.value,
            "rejected_strategies": signal.rejected_strategies,
            "rejection_reasons": signal.rejection_reasons,
            "overlap_score": signal.overlap_score,
            "conflict_resolution_decision": signal.conflict_resolution_decision,
            "trade_rejected_reason": signal.trade_rejected_reason,
            "ev_metrics": signal.ev_metrics,
            "tags": signal.tags,
            "rationale": signal.rationale,
            "blockers": blockers or signal.blockers,
            "created_at": signal.created_at,
        }

    def _estimate_exit_pnl(self, position: dict[str, Any], exit_price: float) -> float:
        """Approximate realized PnL in quote terms."""

        entry = float(position.get("entry_price") or 0.0)
        quantity = float(position.get("quantity") or 0.0)
        if position.get("side") == "long":
            return (exit_price - entry) * quantity
        return (entry - exit_price) * quantity

    def _best_execution_price(self, ticker: Any | None, side: str) -> float:
        """Choose the best available execution price from a ticker snapshot."""

        if ticker is None:
            return 0.0
        if side == "long":
            return float(getattr(ticker, "ask_price", 0.0) or getattr(ticker, "last_price", 0.0) or getattr(ticker, "mark_price", 0.0) or 0.0)
        return float(getattr(ticker, "bid_price", 0.0) or getattr(ticker, "last_price", 0.0) or getattr(ticker, "mark_price", 0.0) or 0.0)

    def _primary_target_from_plan(self, target_plan: list[dict[str, Any]], side: str, entry_price: float) -> float | None:
        """Extract the furthest (TP4 = 2.0R) target price from the target plan."""

        result: float | None = None
        for item in target_plan:
            if not isinstance(item, dict):
                continue
            raw_price = item.get("price")
            if raw_price in (None, ""):
                continue
            price = float(raw_price)
            if side == "long" and price > entry_price:
                if result is None or price > result:
                    result = price  # 가장 먼 타겟 (최고가)
            if side == "short" and price < entry_price:
                if result is None or price < result:
                    result = price  # 가장 먼 타겟 (최저가)
        return result

    def _human_action_reason(self, action: str) -> str:
        """Translate internal exit action keys into Korean labels."""

        mapping = {
            "partial_tp1": "TP1 도달 (0.5R), 잔량 50% 익절",
            "partial_tp2": "TP2 도달 (1.0R), 잔량 50% 익절",
            "partial_tp3": "TP3 도달 (1.5R), 잔량 50% 익절",
            "final_target_exit": "TP4 도달 (2.0R), 전량 익절",
            "stop_out": "손절 (SL 도달)",
            "time_stop": "시간 제한 청산 (최대 보유시간 초과)",
            "move_stop": "손절 → BE 이동 (손실 제로화)",
            "manual_close": "수동 청산",
            "close_order_filled": "청산 주문 체결",
        }
        return mapping.get(action, action.replace("_", " "))

    def _exit_chart_label(self, action: str) -> str:
        """Return a short chart label for exit type."""

        mapping = {
            "stop_out": "손절 청산",
            "final_target_exit": "TP4 전량익절",
            "time_stop": "시간초과 청산",
            "manual_close": "수동 청산",
            "close_order_filled": "청산",
        }
        return mapping.get(action, "청산")

    def _update_stop_reason_cooldown(self, stop_reason: str) -> bool:
        """같은 stop_reason이 N회 연속 손절되면 쿨다운 등록. 쿨다운 신규 등록 시 True 반환."""
        if not stop_reason or stop_reason in ("unknown", ""):
            return False
        threshold = self.settings.risk.stop_reason_cooldown_threshold
        cooldown_min = self.settings.risk.stop_reason_cooldown_minutes

        # 최근 closed_trades에서 같은 stop_reason 연속 손절 횟수 계산
        try:
            closed = self.persistence.list_closed_trades(limit=threshold + 5)
            consecutive = 0
            for trade in closed:
                if trade.get("exit_reason") not in ("stop_out", "time_stop"):
                    break  # 연속 체인 끊김
                if trade.get("stop_reason") != stop_reason:
                    break  # 다른 이유면 체인 끊김
                consecutive += 1
                if consecutive >= threshold:
                    break
        except Exception:
            return False

        # 이번 손절 포함하면 threshold 도달
        if consecutive + 1 >= threshold:
            cooldown_until = datetime.now(UTC) + timedelta(minutes=cooldown_min)
            self._stop_reason_cooldowns[stop_reason] = cooldown_until
            self.logger.warning(
                "🚫 stop_reason 반복 손절 쿨다운 등록",
                extra={"extra_data": {
                    "stop_reason": stop_reason,
                    "consecutive": consecutive + 1,
                    "cooldown_until": cooldown_until.isoformat(),
                    "cooldown_minutes": cooldown_min,
                }},
            )
            return True
        return False

    def _is_stop_reason_cooled_down(self, stop_reason: str) -> bool:
        """해당 stop_reason이 현재 쿨다운 중이면 True 반환. 만료된 쿨다운은 자동 해제."""
        if not stop_reason:
            return False
        cooldown_until = self._stop_reason_cooldowns.get(stop_reason)
        if cooldown_until is None:
            return False
        if datetime.now(UTC) >= cooldown_until:
            del self._stop_reason_cooldowns[stop_reason]
            self.logger.info(
                "✅ stop_reason 쿨다운 해제",
                extra={"extra_data": {"stop_reason": stop_reason}},
            )
            return False
        return True

    def _build_stop_retrospective(self, position: dict[str, Any], exit_price: float) -> list[str]:
        """손절 회고: 왜 손절됐는지 분석하고 개선점을 제시."""

        lines: list[str] = []
        entry_price = float(position.get("entry_price") or 0.0)
        stop_price = float(position.get("stop_price") or 0.0)
        side = position.get("side", "long")
        stop_reason = position.get("stop_reason", "unknown")
        strategy = position.get("display_name") or position.get("chosen_strategy") or position.get("strategy", "")
        rationale = position.get("rationale") or {}
        entry_reason = rationale.get("entry_reason_title", "")
        hold_minutes = 0
        if position.get("entry_time"):
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                entry_time = datetime.fromisoformat(str(position["entry_time"]).replace("Z", "+00:00"))
                hold_minutes = int((now - entry_time).total_seconds() / 60)
            except Exception:
                pass

        # 1. SL 분석
        if stop_price > 0 and entry_price > 0:
            risk_pct = abs(entry_price - stop_price) / entry_price * 100
            lines.append(f"- SL 거리: {risk_pct:.2f}% ({stop_reason})")

        # 2. TP1 도달 여부
        tp1 = float(position.get("tp1_price") or 0.0)
        if tp1 > 0:
            if (side == "long" and exit_price < tp1) or (side == "short" and exit_price > tp1):
                lines.append("- TP1 미도달: 진입 타이밍/방향 재검토 필요")
            else:
                lines.append("- TP1 도달 후 손절: BE 이동이 늦었을 가능성")

        # 3. 보유 시간
        if hold_minutes > 0:
            lines.append(f"- 보유시간: {hold_minutes}분")
            if hold_minutes < 5:
                lines.append("- ⚡ 즉각 손절 → 진입 시점이 너무 이르거나 변동성 급등 구간")

        # 4. 전략별 패턴
        if entry_reason:
            lines.append(f"- 진입근거: {entry_reason}")

        # 5. 최근 손절 패턴 분석
        recent_losses = self._get_recent_stop_losses(strategy, limit=5)
        if len(recent_losses) >= 3:
            lines.append(f"- ⚠️ 최근 {len(recent_losses)}건 연속 손절 — {strategy} 전략 일시 중단 검토")

        return lines

    def _get_recent_stop_losses(self, strategy: str, limit: int = 5) -> list[dict[str, Any]]:
        """최근 손절 기록 조회 (연속 손절 패턴 감지)."""

        try:
            closed = self.persistence.list_closed_trades(limit=20)
            losses = []
            for trade in closed:
                if trade.get("exit_reason") != "stop_out":
                    break  # 연속 손절 체인만 카운트
                if strategy and trade.get("chosen_strategy") == strategy:
                    losses.append(trade)
                if len(losses) >= limit:
                    break
            return losses
        except Exception:
            return []

    async def _evict_stale_positions(self) -> bool:
        """횡보 판정 포지션 정리: 새 진입 슬롯 확보.

        stale_eviction_minutes 이상 보유 중이고 TP1도 미도달한 포지션을
        오래된 것부터 최대 1개 정리한다.
        """
        from datetime import datetime, timezone

        threshold_minutes = self.settings.risk.stale_eviction_minutes
        now = datetime.now(tz=timezone.utc)
        open_positions = self.state_store.state.open_positions

        # 오래된 순으로 정렬
        stale_candidates: list[tuple[float, str]] = []
        for symbol, pos in open_positions.items():
            entry_time_str = pos.get("entry_time") or pos.get("created_at")
            if not entry_time_str:
                continue
            try:
                entry_time = datetime.fromisoformat(str(entry_time_str).replace("Z", "+00:00"))
                hold_minutes = (now - entry_time).total_seconds() / 60
            except Exception:
                continue
            if hold_minutes < threshold_minutes:
                continue
            # TP1 미도달 포지션만 (TP1 이미 익절한 포지션은 유지)
            managed = self.sltp_manager._managed.get(symbol)
            if managed and managed.tp1_done:
                continue  # TP1 이미 도달 → 유지
            stale_candidates.append((hold_minutes, symbol))

        if not stale_candidates:
            return False

        # 가장 오래된 포지션 정리
        stale_candidates.sort(reverse=True)
        _, evict_symbol = stale_candidates[0]
        hold_min = stale_candidates[0][0]

        self.logger.info(
            "횡보 포지션 정리 (슬롯 확보)",
            extra={"extra_data": {"symbol": evict_symbol, "hold_minutes": round(hold_min, 1)}},
        )
        await self.telegram_bot.broadcast_admins(
            f"⏱ 횡보 포지션 정리: {evict_symbol}\n"
            f"- 보유시간: {round(hold_min)}분 (기준: {threshold_minutes}분)\n"
            f"- 사유: 새 진입 슬롯 확보 (max_concurrent_positions)"
        )
        result = await self._close_symbol_internal(evict_symbol, enforce_permission=False, source="stale_eviction")
        return "closed" in result.lower() or "청산" in result

    async def _ensure_contracts_loaded(self) -> None:
        """Populate contract metadata if not already cached."""

        if self.contracts_by_symbol:
            return
        contracts = await self.exchange.get_all_contracts()
        self.contracts_by_symbol = {contract.symbol: contract for contract in contracts}

    async def _build_trade_chart(  # noqa: PLR0913
        self,
        *,
        contract: ContractConfig,
        event_label: str,
        side: str,
        entry_price: float,
        current_price: float,
        stop_price: float | None = None,
        tp1_price: float | None = None,
        tp2_price: float | None = None,
        tp3_price: float | None = None,
        tp4_price: float | None = None,
        final_target_price: float | None = None,
        quantity: float | None = None,
        entry_notional_usdt: float | None = None,
        remaining_notional_usdt: float | None = None,
        leverage: float | None = None,
        notes: list[str] | None = None,
        indicators: list[str] | None = None,
        timeframe: str | None = None,
        strategy_name: str | None = None,
        current_regime: str | None = None,
        conflict_resolution_summary: str | None = None,
        stop_reason: str | None = None,
        target_reasons: list[str] | None = None,
        entry_reason_title: str | None = None,
        entry_reason_lines: list[str] | None = None,
        chart_levels: list[dict[str, Any]] | None = None,
        chart_zones: list[dict[str, Any]] | None = None,
        chart_marker: dict[str, Any] | None = None,
        rr_to_tp1: float | None = None,
        rr_to_tp2: float | None = None,
        candle_limit: int = 180,
        wide_view: bool = False,
    ) -> Path | None:
        """Render a chart snapshot and return the local file path."""

        chart_timeframe = timeframe or self.settings.timeframes.confirm
        try:
            frame = await self.klines.get_dataframe(
                contract.product_type,
                contract.symbol,
                chart_timeframe,
                limit=candle_limit,
                ttl_seconds=0,
            )
            if frame.empty:
                return None
            spec = TradeChartSpec(
                symbol=contract.symbol,
                product_type=contract.product_type.value,
                timeframe=chart_timeframe,
                event_label=event_label,
                mode=self.settings.mode.value,
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                stop_price=stop_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                tp3_price=tp3_price,
                tp4_price=tp4_price,
                final_target_price=final_target_price,
                quantity=quantity,
                entry_notional_usdt=entry_notional_usdt,
                remaining_notional_usdt=remaining_notional_usdt,
                leverage=leverage,
                indicators=indicators or ["EMA20", "EMA50", "VWAP", "RSI14", "ADX14", "ATR14"],
                notes=notes or [],
                strategy_name=strategy_name,
                current_regime=current_regime or "",
                conflict_resolution_summary=conflict_resolution_summary or "",
                stop_reason=stop_reason or "",
                target_reasons=target_reasons or [],
                entry_reason_title=entry_reason_title or "",
                entry_reason_lines=entry_reason_lines or [],
                chart_levels=chart_levels or [],
                chart_zones=chart_zones or [],
                chart_marker=chart_marker or {},
                rr_to_tp1=rr_to_tp1,
                rr_to_tp2=rr_to_tp2,
                wide_view=wide_view,
            )
            return render_trade_chart(frame, spec, output_dir=ROOT_DIR / "data" / "chart_snapshots")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Trade chart render failed",
                extra={"extra_data": {"symbol": contract.symbol, "event_label": event_label, "error": str(exc)}},
            )
            return None

    async def _broadcast_trade_chart(
        self,
        *,
        contract: ContractConfig,
        event_label: str,
        side: str,
        entry_price: float,
        current_price: float,
        stop_price: float | None = None,
        tp1_price: float | None = None,
        tp2_price: float | None = None,
        tp3_price: float | None = None,
        tp4_price: float | None = None,
        final_target_price: float | None = None,
        quantity: float | None = None,
        entry_notional_usdt: float | None = None,
        remaining_notional_usdt: float | None = None,
        leverage: float | None = None,
        notes: list[str] | None = None,
        strategy_name: str | None = None,
        current_regime: str | None = None,
        conflict_resolution_summary: str | None = None,
        stop_reason: str | None = None,
        target_reasons: list[str] | None = None,
        entry_reason_title: str | None = None,
        entry_reason_lines: list[str] | None = None,
        chart_levels: list[dict[str, Any]] | None = None,
        chart_zones: list[dict[str, Any]] | None = None,
        chart_marker: dict[str, Any] | None = None,
        rr_to_tp1: float | None = None,
        rr_to_tp2: float | None = None,
    ) -> None:
        """Render and send a trade chart to Telegram admins."""

        chart_path = await self._build_trade_chart(
            contract=contract,
            event_label=event_label,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            tp4_price=tp4_price,
            final_target_price=final_target_price,
            quantity=quantity,
            entry_notional_usdt=entry_notional_usdt,
            remaining_notional_usdt=remaining_notional_usdt,
            leverage=leverage,
            notes=notes,
            strategy_name=strategy_name,
            current_regime=current_regime,
            conflict_resolution_summary=conflict_resolution_summary,
            stop_reason=stop_reason,
            target_reasons=target_reasons,
            entry_reason_title=entry_reason_title,
            entry_reason_lines=entry_reason_lines,
            chart_levels=chart_levels,
            chart_zones=chart_zones,
            chart_marker=chart_marker,
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
        )
        if chart_path is None:
            return
        await self.telegram_bot.broadcast_admins_photo(
            chart_path,
            caption=f"{contract.symbol} {event_label} 차트",
        )

    async def _finalize_market_entry(
        self,
        *,
        contract: ContractConfig,
        fill_payload: dict[str, Any],
        chart_label: str = "진입",
        chart_notes: list[str] | None = None,
    ) -> None:
        """Apply entry fill bookkeeping and send text/photo alerts."""

        self.logger.info(
            "진입 완료 처리 시작",
            extra={"extra_data": {
                "symbol": contract.symbol,
                "client_order_id": fill_payload.get("client_order_id"),
                "exchange_order_id": fill_payload.get("exchange_order_id"),
                "side": fill_payload.get("side"),
                "filled_quantity": fill_payload.get("filled_quantity"),
                "avg_fill_price": fill_payload.get("avg_fill_price"),
                "leverage": fill_payload.get("leverage"),
                "strategy": fill_payload.get("strategy"),
                "chart_label": chart_label,
            }},
        )
        self.fill_handler.on_entry_filled(fill_payload)
        self.sltp_manager.register_payload(fill_payload)
        self.state_store.remove_order(fill_payload["client_order_id"])
        filled_quantity = float(fill_payload.get("filled_quantity", fill_payload.get("quantity", 0.0)) or 0.0)
        fill_price = float(fill_payload.get("avg_fill_price", fill_payload.get("price", 0.0)) or 0.0)
        remaining_quantity = float(fill_payload.get("remaining_quantity", filled_quantity) or 0.0)
        final_target_price = self._primary_target_from_plan(
            fill_payload.get("target_plan") or [],
            str(fill_payload.get("side", "long")),
            fill_price,
        )
        await self.telegram_bot.broadcast_admins(
            format_entry_fill_alert(
                {
                    **fill_payload,
                    "symbol": contract.symbol,
                    "final_target_price": final_target_price,
                    "entry_notional_usdt": fill_price * filled_quantity,
                    "remaining_notional_usdt": fill_price * remaining_quantity,
                }
            )
        )
        await self._broadcast_trade_chart(
            contract=contract,
            event_label=chart_label,
            side=str(fill_payload.get("side", "long")),
            entry_price=fill_price,
            current_price=fill_price,
            stop_price=float(fill_payload.get("stop_price") or 0.0) or None,
            tp1_price=float(fill_payload.get("tp1_price") or 0.0) or None,
            tp2_price=float(fill_payload.get("tp2_price") or 0.0) or None,
            tp3_price=float(fill_payload.get("tp3_price") or 0.0) or None,
            tp4_price=float(fill_payload.get("tp4_price") or 0.0) or None,
            final_target_price=final_target_price,
            quantity=filled_quantity or None,
            entry_notional_usdt=fill_price * filled_quantity,
            remaining_notional_usdt=fill_price * remaining_quantity,
            leverage=float(fill_payload.get("leverage", 0.0) or 0.0) or None,
            notes=chart_notes,
            strategy_name=str(fill_payload.get("display_name") or fill_payload.get("chosen_strategy") or fill_payload.get("strategy", "")),
            current_regime=str(fill_payload.get("market_regime", "")),
            conflict_resolution_summary=str(fill_payload.get("conflict_resolution_decision", "")),
            stop_reason=str(fill_payload.get("stop_reason", "")),
            target_reasons=[item.get("reason", "") for item in (fill_payload.get("target_plan") or [])[:3] if isinstance(item, dict)],
            entry_reason_title=str((fill_payload.get("rationale") or {}).get("entry_reason_title", "")),
            entry_reason_lines=list((fill_payload.get("rationale") or {}).get("entry_reason_lines", [])),
            chart_levels=list((fill_payload.get("rationale") or {}).get("chart_levels", [])),
            chart_zones=list((fill_payload.get("rationale") or {}).get("chart_zones", [])),
            chart_marker=dict((fill_payload.get("rationale") or {}).get("chart_marker", {})),
            rr_to_tp1=float(fill_payload.get("rr_to_tp1") or 0.0) or None,
            rr_to_tp2=float(fill_payload.get("rr_to_tp2") or 0.0) or None,
        )

    async def _finalize_position_close(
        self,
        *,
        contract: ContractConfig,
        position: dict[str, Any],
        exit_price: float,
        exit_reason: str,
        chart_label: str = "청산",
        chart_notes: list[str] | None = None,
    ) -> float:
        """Apply close bookkeeping and send text/photo alerts."""

        realized_pnl = self._estimate_exit_pnl(position, exit_price)
        _stop_dist = abs(
            float(position.get("entry_price") or 0.0)
            - float(position.get("stop_price") or position.get("entry_price") or 0.0)
        )
        _qty = float(position.get("remaining_quantity") or position.get("quantity") or 0.0)
        risk_denominator = max(_stop_dist * max(_qty, 1e-9), 1e-9)
        self.fill_handler.on_position_closed(
            {
                "symbol": contract.symbol,
                "mode": self.exchange.mode.value,
                "side": position["side"],
                "price": exit_price,
                "avg_fill_price": exit_price,
                "realized_pnl": realized_pnl,
                "realized_pnl_usdt": realized_pnl,
                "pnl_r": realized_pnl / risk_denominator,
                "exit_reason": exit_reason,
                "tags": position.get("tags", []),
                "rationale": position.get("rationale", {}),
                "stop_reason": position.get("stop_reason", ""),
                "tp3_price": position.get("tp3_price"),
                "target_plan": position.get("target_plan", []),
                "rr_to_tp1": position.get("rr_to_tp1", 0.0),
                "rr_to_tp2": position.get("rr_to_tp2", 0.0),
            }
        )
        self.sltp_manager.remove(contract.symbol)
        # 상세 청산 알림
        entry_price = float(position.get("entry_price") or 0.0)
        stop_reason = position.get("stop_reason", "")
        strategy_name = position.get("display_name") or position.get("chosen_strategy") or position.get("strategy", "")
        side = position.get("side", "long")
        pnl_emoji = "+" if realized_pnl >= 0 else ""
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        if side == "short":
            pnl_pct = -pnl_pct
        close_msg_lines = [
            f"{'✅' if realized_pnl >= 0 else '❌'} 포지션 종료: {contract.symbol}",
            f"- 전략: {strategy_name}",
            f"- 방향: {'LONG' if side == 'long' else 'SHORT'}",
            f"- 사유: {self._human_action_reason(exit_reason)}",
            f"- 진입가: {entry_price:,.4f} → 청산가: {exit_price:,.4f}",
            f"- 손익: {pnl_emoji}{realized_pnl:,.2f} USDT ({pnl_pct:+.2f}%)",
        ]
        if stop_reason:
            close_msg_lines.append(f"- SL 기준: {stop_reason}")
        # 손절 시 회고 분석
        if exit_reason in ("stop_out", "time_stop") and realized_pnl < 0:
            retrospective = self._build_stop_retrospective(position, exit_price)
            if retrospective:
                close_msg_lines.append("")
                close_msg_lines.append("📝 손절 회고:")
                close_msg_lines.extend(retrospective)
            # 손절 회고 DB 저장
            try:
                hold_min = 0.0
                if position.get("entry_time"):
                    from datetime import datetime as _dt, timezone as _tz
                    _et = _dt.fromisoformat(str(position["entry_time"]).replace("Z", "+00:00"))
                    hold_min = (_dt.now(_tz.utc) - _et).total_seconds() / 60
                self.trade_journal.log_retrospective({
                    "symbol": symbol,
                    "strategy": str(position.get("display_name") or position.get("chosen_strategy") or position.get("strategy", "")),
                    "exit_reason": exit_reason,
                    "stop_reason": stop_reason,
                    "entry_price": float(position.get("entry_price") or 0.0),
                    "stop_price": float(position.get("stop_price") or 0.0),
                    "exit_price": exit_price,
                    "hold_minutes": round(hold_min, 1),
                    "pnl_r": round(realized_pnl / max(abs(float(position.get("entry_price") or 1.0) - float(position.get("stop_price") or 1.0)), 1e-9), 2),
                    "lessons": retrospective,
                })
            except Exception as _exc:  # noqa: BLE001
                self.logger.warning("손절 회고 DB 저장 실패", extra={"extra_data": {"error": str(_exc)}})
            # 같은 손절 이유 연속 N회 → 쿨다운 등록
            if stop_reason:
                newly_blocked = self._update_stop_reason_cooldown(stop_reason)
                if newly_blocked:
                    cooldown_until = self._stop_reason_cooldowns.get(stop_reason)
                    cooldown_str = cooldown_until.strftime("%H:%M") if cooldown_until else "?"
                    close_msg_lines.append("")
                    close_msg_lines.append(
                        f"🚫 [{stop_reason}] 패턴 {self.settings.risk.stop_reason_cooldown_threshold}회 연속 손절 → "
                        f"{self.settings.risk.stop_reason_cooldown_minutes}분 쿨다운 ({cooldown_str}까지)"
                    )
        await self.telegram_bot.broadcast_admins("\n".join(close_msg_lines))
        await self._broadcast_trade_chart(
            contract=contract,
            event_label=chart_label,
            side=str(position.get("side", "long")),
            entry_price=float(position.get("entry_price") or 0.0),
            current_price=exit_price,
            stop_price=float(position.get("stop_price") or 0.0) or None,
            tp1_price=float(position.get("tp1_price") or 0.0) or None,
            tp2_price=float(position.get("tp2_price") or 0.0) or None,
            tp3_price=float(position.get("tp3_price") or 0.0) or None,
            tp4_price=float(position.get("tp4_price") or 0.0) or None,
            final_target_price=self._primary_target_from_plan(
                position.get("target_plan") or [],
                str(position.get("side", "long")),
                float(position.get("entry_price") or 0.0),
            ),
            quantity=float(position.get("quantity") or 0.0) or None,
            entry_notional_usdt=float(position.get("entry_price") or 0.0) * float(position.get("quantity") or 0.0),
            remaining_notional_usdt=0.0,
            leverage=float(position.get("leverage", 0.0) or 0.0) or None,
            notes=chart_notes,
            strategy_name=str(position.get("display_name") or position.get("chosen_strategy") or position.get("strategy", "")),
            current_regime=str(position.get("market_regime", "")),
            conflict_resolution_summary=str(position.get("conflict_resolution_decision", "")),
            stop_reason=str(position.get("stop_reason", "")),
            target_reasons=[item.get("reason", "") for item in (position.get("target_plan") or [])[:3] if isinstance(item, dict)],
            entry_reason_title=str((position.get("rationale") or {}).get("entry_reason_title", "")),
            entry_reason_lines=list((position.get("rationale") or {}).get("entry_reason_lines", [])),
            chart_levels=list((position.get("rationale") or {}).get("chart_levels", [])),
            chart_zones=list((position.get("rationale") or {}).get("chart_zones", [])),
            chart_marker=dict((position.get("rationale") or {}).get("chart_marker", {})),
            rr_to_tp1=float(position.get("rr_to_tp1") or 0.0) or None,
            rr_to_tp2=float(position.get("rr_to_tp2") or 0.0) or None,
        )
        return realized_pnl

    async def _select_demo_roundtrip_contract(self, symbol: str | None = None) -> ContractConfig | None:
        """Choose a contract for the demo roundtrip command."""

        await self._ensure_contracts_loaded()
        if symbol:
            return self.contracts_by_symbol.get(symbol.upper())

        preferred_symbols = ["BTCUSDT", "ETHUSDT", "BTCPERP", "BTCUSD"]
        for candidate in preferred_symbols:
            contract = self.contracts_by_symbol.get(candidate)
            if contract is not None:
                return contract

        for candidate in self.state_store.state.active_universe:
            contract = self.contracts_by_symbol.get(candidate)
            if contract is not None:
                return contract

        ordered_contracts = sorted(
            self.contracts_by_symbol.values(),
            key=lambda item: (item.product_type != ProductType.USDT_FUTURES, item.symbol),
        )
        return ordered_contracts[0] if ordered_contracts else None

    def _log_event(self, title: str, message: str, *, level: str = "INFO") -> None:
        """Persist and log an event."""

        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.logger.log(getattr(logging, level.upper()), f"{title}: {message}")
        self.persistence.insert_event("system", level, title, message, {}, now_iso)
        self.state_store.set_last_event(title, message, level=level)

    async def request_shutdown(self, reason: str) -> None:
        """Signal the run loop to stop."""

        self._log_event("종료 요청", reason)
        self.stop_event.set()

    async def get_status_payload(self) -> dict[str, Any]:
        """CommandProvider implementation."""

        settings = self.settings_manager.load().to_runtime_dict()
        return {"settings": settings, "runtime": asdict(self.state_store.state), "balance": self.last_balance}

    async def get_positions_payload(self) -> list[dict[str, Any]]:
        """Return current positions."""

        payloads: list[dict[str, Any]] = []
        for position in self.state_store.state.open_positions.values():
            quantity = float(position.get("quantity") or 0.0)
            entry_price = float(position.get("entry_price") or 0.0)
            mark_price = float(position.get("mark_price") or 0.0)
            side = str(position.get("side") or "long").lower()
            entry_notional_usdt = entry_price * quantity
            current_notional_usdt = mark_price * quantity
            unrealized_pnl = float(position.get("unrealized_pnl") or 0.0)
            position_return_pct = (unrealized_pnl / entry_notional_usdt * 100.0) if entry_notional_usdt > 0 else 0.0

            # 목표 수익률: tp3 > tp2 > tp1 순으로 유효한 최종 목표가 사용
            tp3 = float(position.get("tp3_price") or 0.0)
            tp2 = float(position.get("tp2_price") or 0.0)
            tp1 = float(position.get("tp1_price") or 0.0)
            final_tp = tp3 if tp3 > 0 else (tp2 if tp2 > 0 else tp1)
            if entry_price > 0 and final_tp > 0:
                target_pnl_pct = ((final_tp - entry_price) / entry_price * 100.0) if side == "long" else ((entry_price - final_tp) / entry_price * 100.0)
            else:
                target_pnl_pct = 0.0

            payloads.append(
                {
                    **position,
                    "entry_notional_usdt": entry_notional_usdt,
                    "current_notional_usdt": current_notional_usdt,
                    "position_return_pct": position_return_pct,
                    "target_pnl_pct": target_pnl_pct,
                    "final_tp_price": final_tp,
                    "side_display": "롱" if side == "long" else "숏",
                }
            )
        return payloads

    async def get_position_chart(self, symbol: str, timeframe: str = "5m") -> str | None:
        """Render a chart snapshot for the given open position and return the file path."""

        position = self.state_store.state.open_positions.get(symbol)
        if not position:
            return None
        await self._ensure_contracts_loaded()
        contract = self.contracts_by_symbol.get(symbol)
        if contract is None:
            return None
        side = str(position.get("side") or "long").lower()
        entry_price = float(position.get("entry_price") or 0.0)
        mark_price = float(position.get("mark_price") or 0.0)
        if mark_price <= 0:
            mark_price = entry_price

        def _price(key: str) -> float | None:
            v = float(position.get(key) or 0.0)
            return v if v > 0 else None

        path = await self._build_trade_chart(
            contract=contract,
            event_label=f"포지션 ({timeframe})",
            side=side,
            entry_price=entry_price,
            current_price=mark_price,
            stop_price=_price("stop_price"),
            tp1_price=_price("tp1_price"),
            tp2_price=_price("tp2_price"),
            tp3_price=_price("tp3_price"),
            tp4_price=_price("tp4_price"),
            quantity=float(position.get("quantity") or 0) or None,
            leverage=float(position.get("leverage") or 1) or None,
            strategy_name=str(position.get("strategy") or ""),
            timeframe=timeframe,
            notes=[f"실시간 포지션 차트 ({timeframe})"],
            candle_limit=350,
            wide_view=True,
        )
        return str(path) if path else None

    async def get_balance_payload(self) -> dict[str, Any]:
        """Return cached balance."""

        await self._refresh_balance_cache()
        return self.last_balance

    async def get_pnl_payload(self) -> dict[str, Any]:
        """Return PnL summary."""

        perf = self.performance.summarize()
        balance = self.last_balance.get("balance", 1.0)
        return {
            "realized_pnl": perf.realized_pnl,
            "unrealized_pnl": self.last_balance.get("unrealized_pnl", 0.0),
            "roi_pct": (perf.realized_pnl / balance) * 100 if balance else 0.0,
        }

    async def get_watchlist_payload(self) -> list[str]:
        """Return active universe symbols."""

        return list(self.state_store.state.active_universe)

    async def get_bot_status_detailed(self) -> dict[str, Any]:
        """Return detailed bot status without AI calls."""

        runtime_state = self.state_store.state
        open_positions = list(runtime_state.open_positions.values())
        open_orders = runtime_state.open_orders
        settings = self.settings_manager.load()

        # 활성화된 전략 확인
        enabled_strategies = {}
        if settings.strategy and settings.strategy.enabled:
            enabled_strategies = {
                k: v for k, v in settings.strategy.enabled.items()
            }

        # 마지막 이벤트
        last_event = runtime_state.last_event or {}

        return {
            "mode": self.exchange.mode.value,
            "bot_status": "RUNNING" if not runtime_state.paused else "PAUSED",
            "paused": runtime_state.paused,
            "active_positions": open_positions,
            "open_orders": open_orders,
            "enabled_strategies": enabled_strategies,
            "risk_flags": runtime_state.risk_flags or [],
            "last_event_title": last_event.get("title", "-"),
            "active_universe_count": len(runtime_state.active_universe),
            "active_universe": list(runtime_state.active_universe)[:10],
        }

    async def get_recent_signals_payload(self) -> list[dict[str, Any]]:
        """Return recent signals."""

        return self.trade_journal.recent_signals()

    async def get_today_payload(self) -> dict[str, Any]:
        """Return today's summary."""

        perf = self.performance.summarize()
        return {
            "title": "오늘 요약",
            "trade_count": perf.trade_count,
            "win_rate": perf.win_rate,
            "realized_pnl": perf.realized_pnl,
            "profit_factor": perf.profit_factor,
            "expectancy": perf.expectancy,
        }

    async def get_journal_payload(self) -> list[dict[str, Any]]:
        """Return recent journal rows."""

        return self.trade_journal.recent_trades()

    async def get_why_payload(self, symbol: str) -> dict[str, Any]:
        """Return latest rationale for a symbol."""

        payload = self.trade_journal.why_symbol(symbol)
        explanation = await self.ai_summarizer.explain_why(symbol, payload)
        if explanation:
            return {
                **payload,
                **explanation.model_dump(mode="json"),
            }
        rule_summary = payload.get("rule_summary") or {}
        return {
            **payload,
            "summary_ko": rule_summary.get("summary_ko", "최근 기록이 부족합니다."),
            "bullet_points": rule_summary.get("bullet_points", []),
            "regime_commentary": "",
            "risk_commentary": rule_summary.get("rr_commentary", ""),
        }

    async def pause_trading(self) -> str:
        """Pause trading loops."""

        self.state_store.set_paused(True)
        await self.telegram_bot.broadcast_admins("전략 일시정지\n- 신규 진입 차단")
        return "신규 진입을 일시정지했습니다."

    async def resume_trading(self) -> str:
        """Resume trading loops."""

        self.state_store.set_paused(False)
        await self.telegram_bot.broadcast_admins("전략 재개\n- 신규 진입 허용")
        return "신규 진입을 재개했습니다."

    async def switch_mode(self) -> str:
        """Show current mode and safe switching guidance."""

        return (
            f"현재 모드: {self.settings.mode.value}\n"
            "모드 전환은 Streamlit Settings 페이지와 .env의 LIVE_TRADING_ENABLED=true, "
            "그리고 live_streamlit_confirmed=true 를 함께 맞춘 뒤 /reload 로 반영하세요."
        )

    async def get_risk_payload(self) -> dict[str, Any]:
        """Return current risk settings and flags."""

        return {
            "risk_per_trade": self.settings.risk.risk_per_trade,
            "max_daily_loss_r": self.settings.risk.max_daily_loss_r,
            "max_concurrent_positions": self.settings.risk.max_concurrent_positions,
            "risk_flags": self.state_store.state.risk_flags,
        }

    async def get_settings_payload(self) -> dict[str, Any]:
        """Return current settings snapshot."""

        return self.settings.to_runtime_dict()

    async def _close_symbol_internal(self, symbol: str, *, enforce_permission: bool, source: str) -> str:
        """Close a single symbol through a market reduce-only order."""

        if enforce_permission and not self.settings.telegram.allow_close_commands:
            return "close 명령은 현재 비활성화되어 있습니다."
        position = self.state_store.state.open_positions.get(symbol)
        if not position:
            return f"{symbol} 포지션이 없습니다."
        await self._ensure_contracts_loaded()
        contract = self.contracts_by_symbol.get(symbol)
        if contract is None:
            return f"{symbol} 계약 정보를 찾지 못했습니다."
        try:
            await self.order_manager.close_position_market(
                symbol=symbol,
                product_type=contract.product_type,
                margin_coin=contract.margin_coin,
                side=position["side"],
                quantity=float(position["quantity"]),
            )
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            self.logger.error("close_position_market failed", extra={"extra_data": {"symbol": symbol, "error": err_str}})
            # ✅ Exchange에 포지션이 없으면 (22002 / No position to close) local state도 정리
            if "22002" in err_str or "No position to close" in err_str:
                self.state_store.remove_position(symbol)
                self.sltp_manager.remove(symbol)
                return f"✅ {symbol} local 포지션 정리 완료 (거래소에 이미 없음)"
            return f"❌ {symbol} 청산 실패: {exc}"
        return f"✅ {source} 요청으로 {symbol} 청산 주문 전송"

    async def close_symbol(self, symbol: str) -> str:
        """Close one symbol if enabled."""

        return await self._close_symbol_internal(symbol, enforce_permission=True, source="telegram")

    async def _close_all_positions_internal(self, *, enforce_permission: bool, source: str) -> str:
        """Close all current positions and pending orders."""

        if enforce_permission and not self.settings.telegram.allow_close_commands:
            return "closeall 명령은 현재 비활성화되어 있습니다."
        messages: list[str] = []
        # Close all tracked positions
        for symbol in list(self.state_store.state.open_positions.keys()):
            messages.append(
                await self._close_symbol_internal(
                    symbol,
                    enforce_permission=False,
                    source=source,
                )
            )
        # ✅ Cancel all open orders too
        cancelled = []
        for client_oid, order in list(self.state_store.state.open_orders.items()):
            order_symbol = str(order.get("symbol") or "")
            try:
                contract = self.contracts_by_symbol.get(order_symbol)
                if contract:
                    await self.exchange.rest.cancel_order(
                        contract.product_type,
                        order_symbol,
                        contract.margin_coin,
                        client_order_id=client_oid,
                    )
            except Exception:  # noqa: BLE001
                pass  # Already filled or not on exchange — just clean local state
            finally:
                self.state_store.remove_order(client_oid)
                cancelled.append(order_symbol)
        if cancelled:
            messages.append(f"🗑️ 미체결 주문 {len(cancelled)}개 정리: {', '.join(dict.fromkeys(cancelled))}")
        return "\n".join(messages) if messages else "열린 포지션/주문이 없습니다."

    async def close_all(self) -> str:
        """Close all positions if enabled."""

        return await self._close_all_positions_internal(enforce_permission=True, source="telegram")

    async def get_news_payload(self) -> list[dict[str, Any]]:
        """Return recent analyzed news."""

        rows = self.persistence.fetchall(
            """
            SELECT n.title, a.summary_ko, a.impact_level, a.direction_bias, a.should_block_new_entries
            FROM news_items n
            JOIN ai_news_analysis a ON a.news_hash = n.news_hash
            ORDER BY a.id DESC LIMIT 20
            """
        )
        return rows

    async def get_events_payload(self) -> list[dict[str, Any]]:
        """Return recent events."""

        return self.persistence.fetchall("SELECT title, message, created_at FROM bot_events ORDER BY id DESC LIMIT 20")

    async def reload_settings(self) -> str:
        """Reload settings from disk immediately."""

        await self._reload_if_changed()
        return "설정을 다시 읽었습니다."

    async def demo_roundtrip(self, symbol: str | None = None) -> str:
        """Open a tiny demo long position and close it shortly after."""

        if self.settings.mode != TradingMode.DEMO:
            return "demo_roundtrip 명령은 DEMO 모드에서만 사용할 수 있습니다."

        contract = await self._select_demo_roundtrip_contract(symbol)
        if contract is None:
            return "진입 가능한 데모 심볼을 찾지 못했습니다."
        if contract.symbol in self.state_store.state.open_positions:
            return f"{contract.symbol} 에 이미 열린 포지션이 있어 demo_roundtrip을 건너뜁니다."

        await self._refresh_balance_cache()
        ticker = self.latest_tickers.get(contract.symbol) or await self.exchange.rest.get_ticker(contract.product_type, contract.symbol)
        if ticker is None:
            return f"{contract.symbol} 현재가를 가져오지 못했습니다."

        frame = await self.klines.get_dataframe(
            contract.product_type,
            contract.symbol,
            self.settings.timeframes.confirm,
            limit=180,
            ttl_seconds=0,
        )
        if frame.empty:
            return f"{contract.symbol} 차트 데이터를 가져오지 못했습니다."

        entry_price = float(ticker.ask_price or ticker.last_price or ticker.mark_price)
        if entry_price <= 0:
            return f"{contract.symbol} 진입가가 비정상입니다."

        atr_value = float(atr(frame, 14).iloc[-1]) if not frame.empty else 0.0
        if atr_value <= 0:
            atr_value = entry_price * 0.003
        stop_buffer = max(atr_value * 1.2, entry_price * 0.003)
        stop_price = round_to_step(max(entry_price - stop_buffer, 0.0), contract.price_step or 0.0001, mode="down")
        tp1_price = round_to_step(entry_price + stop_buffer, contract.price_step or 0.0001, mode="up")
        tp2_price = round_to_step(entry_price + stop_buffer * 2, contract.price_step or 0.0001, mode="up")

        quantity_result = self.risk_engine.position_sizer.size_from_risk(
            account_equity=max(self.last_balance.get("balance", 0.0), 100.0),
            risk_fraction=min(self.settings.risk.risk_per_trade, 0.0005),
            entry_price=entry_price,
            stop_price=stop_price,
            leverage=min(self.settings.execution.default_leverage, 2.0),
            contract=contract,
        )
        if not quantity_result.valid:
            return f"수량 계산 실패: {quantity_result.reason}"

        signal = StrategySignal(
            symbol=contract.symbol,
            product_type=contract.product_type.value,
            strategy=StrategyName.MANUAL_DEMO,
            side=Side.LONG,
            entry_price=entry_price,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            score=1.0,
            confidence=1.0,
            expected_r=2.0,
            fees_r=0.0,
            slippage_r=0.0,
            tags=["manual_demo", "roundtrip", "telegram_command"],
            rationale={
                "trigger": "telegram_demo_roundtrip",
                "symbol": contract.symbol,
                "notes": "demo 전용 강제 진입 후 자동 청산",
            },
        )
        approved = ApprovedTrade(
            signal=signal,
            quantity=quantity_result.quantity,
            leverage=min(self.settings.execution.default_leverage, 2.0),
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            risk_amount=quantity_result.risk_amount,
            notes=["manual_demo_roundtrip", "force_market_entry"],
        )
        order = OrderRequest(
            symbol=contract.symbol,
            product_type=contract.product_type,
            margin_coin=contract.margin_coin,
            side="buy",
            trade_side="open",
            order_type=OrderType.MARKET,
            quantity=quantity_result.quantity,
            preset_take_profit_price=tp2_price,
            preset_stop_loss_price=stop_price,
            dry_run_reason="telegram_demo_roundtrip",
        )

        try:
            result = await self.order_manager.submit_entry(approved, contract, order, ["demo_roundtrip"])
        except Exception as exc:  # noqa: BLE001
            return f"demo_roundtrip 진입 실패: {exc}"

        fill_payload = {
            **self.state_store.state.open_orders.get(result.client_order_id, {}),
            "client_order_id": result.client_order_id,
            "exchange_order_id": result.exchange_order_id,
            "symbol": contract.symbol,
            "mode": self.exchange.mode.value,
            "side": "long",
            "order_type": order.order_type.value,
            "price": entry_price,
            "avg_fill_price": entry_price,
            "quantity": quantity_result.quantity,
            "filled_quantity": quantity_result.quantity,
            "stop_price": stop_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "strategy": StrategyName.MANUAL_DEMO.value,
            "signal_id": signal.signal_id,
            "tags": signal.tags,
            "rationale": signal.rationale,
            "leverage": approved.leverage,
        }
        await self._finalize_market_entry(
            contract=contract,
            fill_payload=fill_payload,
            chart_label="데모 진입",
            chart_notes=[
                "데모 강제 라운드트립",
                "보조지표: EMA20, EMA50, VWAP, RSI14, ADX14, ATR14",
            ],
        )

        await asyncio.sleep(2.0)
        position = self.state_store.state.open_positions.get(contract.symbol)
        if position is None:
            return f"{contract.symbol} 진입은 완료됐지만 열린 포지션을 찾지 못했습니다."

        try:
            await self.order_manager.close_position_market(
                symbol=contract.symbol,
                product_type=contract.product_type,
                margin_coin=contract.margin_coin,
                side=str(position["side"]),
                quantity=float(position["quantity"]),
            )
        except Exception as exc:  # noqa: BLE001
            return f"{contract.symbol} 진입 완료. 자동 청산 주문 실패: {exc}"

        latest_ticker = self.latest_tickers.get(contract.symbol) or await self.exchange.rest.get_ticker(contract.product_type, contract.symbol)
        exit_price = float(latest_ticker.bid_price if latest_ticker else entry_price)
        realized_pnl = await self._finalize_position_close(
            contract=contract,
            position=position,
            exit_price=exit_price,
            exit_reason="demo_roundtrip",
            chart_label="데모 청산",
            chart_notes=[
                "데모 강제 라운드트립 종료",
                "보조지표: EMA20, EMA50, VWAP, RSI14, ADX14, ATR14",
            ],
        )
        return (
            f"demo_roundtrip 완료\n"
            f"- 심볼: {contract.symbol}\n"
            f"- 진입가: {entry_price:,.4f}\n"
            f"- 청산가: {exit_price:,.4f}\n"
            f"- 손익: {realized_pnl:,.4f} USDT"
        )

    # ------------------------------------------------------------------
    # AI 강제 스캔 — 시장이 조용할 때 AI 가 직접 기회를 찾아 진입
    # ------------------------------------------------------------------

    async def ai_scan(self) -> str:
        """AI 가 모든 차트를 스캔해 수익 기회를 찾아 진입.

        흐름:
        1. 유니버스 상위 심볼 데이터 수집 (가격, ATR, 레짐, 최근 캔들)
        2. OpenAI 에 구조화된 JSON 요청 (어떤 심볼에 어느 방향으로 진입할지)
        3. AI 신뢰도 ≥ 0.6 이면 즉시 시장가 진입
        4. 결과를 텔레그램으로 반환
        """
        import json as _json

        if not self.settings.secrets.openai_api_key:
            return "❌ OpenAI API 키가 설정되지 않았습니다."

        await self._refresh_balance_cache()
        equity = self.last_balance.get("balance", 0.0)
        if equity <= 0:
            return "❌ 잔고를 가져올 수 없습니다."

        # 유니버스가 비어있으면 즉시 갱신 (거래소에서 실시간 심볼 수신)
        if not self.state_store.state.active_universe:
            try:
                await self.universe_manager.refresh()
            except Exception:
                pass

        universe = list(self.state_store.state.active_universe)[:5]
        if not universe:
            return "❌ 활성 유니버스가 비어있습니다. 잠시 후 다시 시도하세요."

        # 열려있는 심볼은 제외
        open_symbols = set(self.state_store.state.open_positions.keys())

        market_data: list[dict] = []
        for sym in universe:
            if sym in open_symbols:
                continue
            try:
                # ✅ 실제 거래 가능 여부 사전 검증 (40034 심볼 사전 제거)
                _contracts_check = await self.exchange.rest.get_contracts(
                    ProductType.USDT_FUTURES, sym
                )
                if not _contracts_check:
                    continue

                ticker = self.latest_tickers.get(sym)
                if ticker is None:
                    ticker = await self.exchange.rest.get_ticker(
                        ProductType.USDT_FUTURES, sym
                    )
                if ticker is None:
                    continue

                frames_ai = await self.klines.get_multi_timeframe(
                    ProductType.USDT_FUTURES, sym,
                    timeframes=["5m", "15m"],
                    limit=60,
                )
                frame_5m = frames_ai.get("5m")
                frame_15m = frames_ai.get("15m")
                if frame_5m is None or frame_5m.empty or len(frame_5m) < 10:
                    continue
                if frame_15m is None or frame_15m.empty:
                    frame_15m = frame_5m

                from market.indicators import atr as _atr
                atr_val = float(_atr(frame_5m).iloc[-1])
                last_close = float(frame_5m["close"].iloc[-1])
                change_24h = (last_close - float(frame_5m["close"].iloc[-min(len(frame_5m), 24)])) / max(float(frame_5m["close"].iloc[-min(len(frame_5m), 24)]), 1e-9) * 100

                # 최근 5 캔들 요약 (OHLCV)
                recent = frame_5m.tail(5)
                candles = [
                    {
                        "o": round(float(r["open"]), 4),
                        "h": round(float(r["high"]), 4),
                        "l": round(float(r["low"]), 4),
                        "c": round(float(r["close"]), 4),
                        "v": round(float(r["volume"]), 2),
                    }
                    for _, r in recent.iterrows()
                ]

                # 레짐 감지
                try:
                    regime_snap = self.regime_classifier.classify(frame_15m, frame_15m)
                    regime = regime_snap.regime.value
                except Exception:
                    regime = "unknown"

                market_data.append({
                    "symbol": sym,
                    "price": round(last_close, 4),
                    "atr": round(atr_val, 4),
                    "change_24h_pct": round(change_24h, 2),
                    "regime": regime,
                    "candles_5m": candles,
                })
            except Exception:  # noqa: BLE001
                continue

        if not market_data:
            return "❌ 분석할 심볼 데이터를 가져오지 못했습니다."

        available_symbols = [md.get("symbol") for md in market_data]

        # AI 프롬프트
        prompt = f"""당신은 전문 크립토 선물 트레이더입니다.
아래 {len(market_data)}개 심볼의 5분 캔들 데이터, ATR, 레짐 정보를 보고
단 1개 심볼에 진입하여 최소 1% 이상 수익 실현이 가능한 트레이드를 결정하세요.

[계정]
- 총 자산: {equity:.2f} USDT
- 현재 오픈 포지션: {len(open_symbols)}개

[시장 데이터]
{_json.dumps(market_data, ensure_ascii=False, indent=2)}

[응답 형식 — 반드시 아래 JSON 만 반환]
{{
  "symbol": "{available_symbols[0]}",
  "side": "long",
  "entry_price": 70000.0,
  "stop_price": 69300.0,
  "tp1_price": 71000.0,
  "confidence": 0.75,
  "reason": "간단한 한국어 이유 (2줄 이내)"
}}

⚠️ 반드시 지켜야할 규칙:
- symbol: 반드시 위 시장 데이터의 심볼 중 하나만 선택 ({', '.join(available_symbols)})
- side: "long" 또는 "short" 만 허용
- confidence: 0.0 ~ 1.0 (0.6 이상이면 진입 실행)
- stop: ATR 의 1.5배 이상 여유를 두세요
- tp1: 최소 stop 거리의 1.5배 이상 수익이 되어야 합니다
- 응답: JSON 만 반환하고 다른 텍스트는 없어야 합니다"""

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.settings.secrets.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            decision = _json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return f"❌ AI 분석 실패: {exc}"

        sym = str(decision.get("symbol", "")).upper()
        side_str = str(decision.get("side", "long")).lower()
        entry_price = float(decision.get("entry_price", 0))
        stop_price = float(decision.get("stop_price", 0))
        tp1_price = float(decision.get("tp1_price", 0))
        confidence = float(decision.get("confidence", 0))
        reason = str(decision.get("reason", ""))

        # ✅ 심볼 유효성 검사: market_data에 있는 심볼만 사용
        available_symbols = [md.get("symbol") for md in market_data]
        if sym not in available_symbols:
            # Fallback: 첫 번째 유효한 심볼로 자동 변경
            if available_symbols:
                old_sym = sym
                sym = available_symbols[0]
                reason = f"{reason} (⚠️ {old_sym} 거래 불가 → {sym}로 변경)"
            else:
                return "❌ 유효한 분석 심볼이 없습니다."

        result_lines = [
            f"🤖 AI 스캔 결과",
            f"심볼: {sym} / 방향: {'📈 롱' if side_str == 'long' else '📉 숏'}",
            f"진입가: {entry_price:,.4f} | 손절: {stop_price:,.4f} | TP1: {tp1_price:,.4f}",
            f"신뢰도: {confidence:.0%}",
            f"이유: {reason}",
        ]

        if confidence < 0.6:
            result_lines.append(f"\n⚠️  신뢰도 {confidence:.0%} 미달 (기준: 60%) — 진입 건너뜀")
            return "\n".join(result_lines)

        # 진입 실행 (DEMO 모드 전용)
        if self.settings.mode != TradingMode.DEMO:
            result_lines.append("\n⚠️  DEMO 모드에서만 AI 스캔 자동 진입이 허용됩니다.")
            return "\n".join(result_lines)

        try:
            contracts = await self.exchange.rest.get_contracts(
                ProductType.USDT_FUTURES, sym
            )
            if not contracts:
                result_lines.append(f"\n❌ {sym} 컨트랙트 정보 없음")
                result_lines.append(f"   (이 심볼은 더 이상 거래소에서 지원되지 않을 수 있습니다)")
                return "\n".join(result_lines)
            contract = contracts[0]

            # 현재 시장가 (market order 체결가 추정)
            ticker_now = self.latest_tickers.get(sym) or await self.exchange.rest.get_ticker(
                ProductType.USDT_FUTURES, sym
            )
            actual_price = float(
                (ticker_now.last_price or ticker_now.mark_price) if ticker_now else entry_price
            ) or entry_price

            # AI 제안 가격 검증: stop/tp 없으면 ATR 기반으로 보정
            if not stop_price or stop_price <= 0:
                from market.indicators import atr as _atr_fn
                _frames_now = await self.klines.get_multi_timeframe(
                    ProductType.USDT_FUTURES, sym, timeframes=["5m"], limit=30
                )
                _f5 = _frames_now.get("5m")
                _atr_now = float(_atr_fn(_f5).iloc[-1]) if _f5 is not None and not _f5.empty else actual_price * 0.005
                stop_price = actual_price - _atr_now * 1.5 if side_str == "long" else actual_price + _atr_now * 1.5
            if not tp1_price or tp1_price <= 0:
                risk_dist = abs(actual_price - stop_price)
                tp1_price = actual_price + risk_dist * 2.0 if side_str == "long" else actual_price - risk_dist * 2.0

            # 손절/목표 방향 검증
            if side_str == "long":
                if stop_price >= actual_price:
                    stop_price = actual_price * 0.985
                if tp1_price <= actual_price:
                    tp1_price = actual_price * 1.02
            else:
                if stop_price <= actual_price:
                    stop_price = actual_price * 1.015
                if tp1_price >= actual_price:
                    tp1_price = actual_price * 0.98

            # 지정가 주문용 좋은 진입가 찾기: 현재가 근처에서 ATR 기반으로 계산
            # 롱: 현재가 - 0.5*ATR (지지대 근처)
            # 숏: 현재가 + 0.5*ATR (저항대 근처)
            from market.indicators import atr as _atr_fn
            if "f5" not in locals():
                _frames_now = await self.klines.get_multi_timeframe(
                    ProductType.USDT_FUTURES, sym, timeframes=["5m"], limit=30
                )
                _f5 = _frames_now.get("5m")
            _atr_val = float(_atr_fn(_f5).iloc[-1]) if _f5 is not None and not _f5.empty else actual_price * 0.005

            # 좋은 진입가: 현재가와 AI 제안가 사이에서 ATR 기반으로 선택
            if side_str == "long":
                # 지지대 찾기: AI 제안가와 현재가 사이에서, 현재가에 더 가까운 지점
                good_entry = actual_price - _atr_val * 0.3  # 현재가보다 약간 아래
                good_entry = min(good_entry, entry_price)  # AI 제안가보다 낮게
                good_entry = max(good_entry, actual_price * 0.99)  # 현재가의 1% 이내
            else:
                # 저항대 찾기
                good_entry = actual_price + _atr_val * 0.3
                good_entry = max(good_entry, entry_price)
                good_entry = min(good_entry, actual_price * 1.01)

            # ✅ 가격을 컨트랙트 최소 단위(priceEndStep)에 맞춰 반올림
            _pp = contract.price_precision  # 소수점 자릿수
            good_entry = round(good_entry, _pp)
            stop_price = round(stop_price, _pp)
            tp1_price = round(tp1_price, _pp)

            # RR 계산 (지정가 기준)
            raw_risk = abs(good_entry - stop_price)
            rr_to_tp1 = round(abs(tp1_price - good_entry) / raw_risk, 2) if raw_risk > 0 else 0.0

            # target_plan 구성
            target_plan = [{"price": tp1_price, "quantity_pct": 1.0, "reason": "AI 스캔 목표가"}]

            # 진입 수량 계산 (지정가 기준)
            qty_result = self.risk_engine.position_sizer.size_from_equity_share(
                account_equity=equity,
                equity_share_count=self.settings.risk.equity_share_count,
                entry_price=good_entry,
                leverage=self.settings.execution.default_leverage,
                contract=contract,
            )
            if not qty_result.valid:
                result_lines.append(f"\n❌ 수량 계산 실패: {qty_result.reason}")
                return "\n".join(result_lines)

            import uuid as _uuid
            client_oid = f"aiscan_{_uuid.uuid4().hex[:10]}"
            # ✅ 지정가 주문 (LIMIT)
            order = OrderRequest(
                symbol=sym,
                product_type=ProductType.USDT_FUTURES,
                margin_coin="USDT",
                side="buy" if side_str == "long" else "sell",
                trade_side="open",
                order_type=OrderType.LIMIT,
                price=good_entry,
                quantity=qty_result.quantity,
                client_order_id=client_oid,
                margin_mode="isolated",
                preset_stop_loss_price=stop_price,
                preset_take_profit_price=tp1_price,
            )
            order_result = await self.exchange.rest.place_order(order)

            # AI 결정에서 regime 조회 (market_data에서 해당 심볼 찾기)
            sym_regime = next(
                (md.get("regime", "trend") for md in market_data if md.get("symbol") == sym),
                "trend",
            )

            now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
            order_payload: dict[str, Any] = {
                "client_order_id": order_result.client_order_id or client_oid,
                "exchange_order_id": order_result.exchange_order_id,
                "symbol": sym,
                "mode": self.settings.mode.value,
                "side": side_str,
                "order_type": OrderType.LIMIT.value,
                "status": "new",
                "price": good_entry,
                "quantity": qty_result.quantity,
                "filled_quantity": 0.0,  # ✅ 아직 미체결
                "stop_price": round(stop_price, 6),
                "tp1_price": round(tp1_price, 6),
                "tp2_price": None,
                "tp3_price": None,
                "target_plan": target_plan,
                "rr_to_tp1": rr_to_tp1,
                "rr_to_tp2": 0.0,
                "rr_to_best_target": rr_to_tp1,
                "ev_metrics": {},
                "strategy": "ai_scan",
                "display_name": "AI 스캔",
                "chosen_strategy": "ai_scan",
                "signal_id": client_oid,
                "tags": ["ai_scan"],
                "rationale": {
                    "entry_reason_title": f"AI 스캔 지정가 진입",
                    "entry_reason_lines": [f"{reason}\n→ 지정가: {good_entry:,.4f}"],
                    "chart_levels": [
                        {"label": "AI 제안가", "price": entry_price, "color": "#fbbf24", "linestyle": "--"},
                        {"label": "지정가 주문", "price": good_entry, "color": "#10b981", "linestyle": "-"},
                    ],
                    "chart_zones": [],
                    "chart_marker": {"price": good_entry, "label": "진입 대기중"},
                },
                "stop_reason": "AI 제안 손절가",
                "leverage": self.settings.execution.default_leverage,
                "market_regime": sym_regime,
                "candidate_strategies": [],
                "rejected_strategies": [],
                "rejection_reasons": {},
                "overlap_score": 0.0,
                "conflict_resolution_decision": "ai_scan_limit_order",
                "reduce_only": False,
                "reason": "ai_scan_limit",
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            # 상태 저장
            self.persistence.save_order(order_payload)
            self.state_store.update_order(order_payload["client_order_id"], order_payload)

            # ✅ 지정가 주문 결과 + 차트 전송 (체결 대기)
            result_lines.append(
                f"\n✅ AI 스캔 지정가 주문 접수!\n"
                f"지정가: {good_entry:,.4f} (AI 제안: {entry_price:,.4f})\n"
                f"수량: {qty_result.quantity:.4f} {sym.replace('USDT','')}\n"
                f"손절: {stop_price:,.4f} | TP1: {tp1_price:,.4f}\n"
                f"RR: {rr_to_tp1:.2f}\n\n"
                f"⏳ 지정가 체결 대기 중... (Timeout: {self.settings.execution.maker_timeout_seconds}초)"
            )

            # 지정가 주문 차트 전송 (스캔 결과 + 진입가 표시)
            await self._broadcast_trade_chart(
                contract=contract,
                event_label="AI 스캔 - 지정가 주문",
                side=side_str,
                entry_price=good_entry,
                current_price=actual_price,
                stop_price=float(stop_price),
                tp1_price=float(tp1_price),
                tp2_price=None,
                tp3_price=None,
                tp4_price=None,
                final_target_price=tp1_price,
                quantity=qty_result.quantity,
                entry_notional_usdt=good_entry * qty_result.quantity,
                remaining_notional_usdt=good_entry * qty_result.quantity,
                leverage=self.settings.execution.default_leverage,
                strategy_name="AI 스캔",
                current_regime=sym_regime,
                stop_reason="AI 제안",
                target_reasons=["AI 스캔 분석"],
                entry_reason_title=f"AI 스캔: {reason[:40]}",
                entry_reason_lines=[reason],
                chart_levels=[
                    {"label": "AI 제안가", "price": float(entry_price), "color": "#fbbf24", "linestyle": "--"},
                    {"label": "지정가 주문", "price": float(good_entry), "color": "#10b981", "linestyle": "-"},
                ],
                notes=[
                    f"신뢰도: {confidence:.0%}",
                    f"AI 제안: {entry_price:,.4f}",
                    f"지정가: {good_entry:,.4f}",
                    f"레짐: {sym_regime}",
                ],
            )
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            if "40034" in err_str:
                # 이 심볼은 거래 불가 → 다음 유효 심볼로 대체
                fallback = next((s for s in available_symbols if s != sym), None)
                result_lines.append(f"\n⚠️ {sym} 거래 불가 (40034) → 건너뜀")
                if fallback:
                    result_lines.append(f"   다음 심볼 {fallback}로 재스캔하려면 /ai_scan 을 다시 실행하세요.")
            else:
                result_lines.append(f"\n❌ 진입 실패: {exc}")

        return "\n".join(result_lines)


async def main() -> None:
    """Application entrypoint."""

    app = TradingApplication()
    try:
        await app.run_forever()
    finally:
        if not app.stop_event.is_set():
            await app.request_shutdown("main 종료")
