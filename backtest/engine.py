"""CLI backtest runner reusing the live strategy stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.enums import ProductType, SignalStatus, TradingMode
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from market.market_regime import MarketRegimeClassifier
from market.sr_levels import SupportResistanceDetector
from risk.risk_engine import RiskEngine
from strategy.base import SignalContext
from strategy.break_retest import BreakRetestStrategy
from strategy.ev_filter import ExpectedValueFilter
from strategy.liquidity_raid import LiquidityRaidStrategy
from strategy.rr_filter import evaluate_trade_viability
from strategy.signal_score import SignalScorer

from .reports import build_report, write_report
from .simulator import TradeSimulator


@dataclass(slots=True)
class BacktestResult:
    """Backtest output container."""

    trades: list[Any]
    report_path: Path


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with timestamp/open/high/low/close/volume."""

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        df[column] = df[column].astype(float)
    if "quote_volume" not in df.columns:
        df["quote_volume"] = df["close"] * df["volume"]
    return df.set_index("timestamp").sort_index()


def resample_frame(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data."""

    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
            }
        )
        .dropna()
    )


class BacktestEngine:
    """Drive a single-symbol backtest using the same strategy logic."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()
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
        self.scorer = SignalScorer(self.settings.strategy.trend_weight, self.settings.strategy.raid_weight)
        self.ev_filter = ExpectedValueFilter(self.settings.ev)
        self.risk_engine = RiskEngine(self.settings)
        self.contract = ContractConfig(
            symbol="BTCUSDT",
            product_type=ProductType.USDT_FUTURES,
            base_coin="BTC",
            quote_coin="USDT",
            margin_coin="USDT",
            min_order_size=0.001,
            size_step=0.001,
            price_step=0.1,
        )
        self.simulator = TradeSimulator(
            fee_bps=self.settings.ev.estimated_maker_fee_bps,
            slippage_bps=self.settings.ev.slippage_bps,
            tp1_partial_close_pct=self.settings.risk.tp1_partial_close_pct,
            tp2_partial_close_pct=self.settings.risk.tp2_partial_close_pct,
            move_sl_to_be=self.settings.risk.move_sl_to_be_after_tp1,
            be_offset_r=self.settings.risk.be_offset_r,
            max_hold_minutes=self.settings.risk.max_position_hold_minutes,
        )

    def run(self, *, data_path: Path, symbol: str, output_path: Path) -> BacktestResult:
        """Run the backtest."""

        df_1m = load_ohlcv_csv(data_path)
        df_3m = resample_frame(df_1m, "3min")
        df_5m = resample_frame(df_1m, "5min")
        df_15m = resample_frame(df_1m, "15min")
        df_1h = resample_frame(df_1m, "1h")
        account_equity = 10_000.0
        self.contract.symbol = symbol

        for timestamp in df_3m.index[120:]:
            candle = df_3m.loc[timestamp]
            self.simulator.on_bar(candle)
            if self.simulator.active_trade is not None:
                continue
            frames = {
                "3m": df_3m.loc[:timestamp].tail(250),
                "5m": df_5m.loc[:timestamp].tail(250),
                "15m": df_15m.loc[:timestamp].tail(250),
                "1H": df_1h.loc[:timestamp].tail(250),
            }
            levels = self.level_detector.build_levels(frames["3m"], frames["15m"])
            regime = self.regime_classifier.classify(frames["15m"], frames["1H"])
            last_close = float(frames["3m"]["close"].iloc[-1])
            ticker = {
                "last_price": last_close,
                "bid_price": last_close * 0.9998,
                "ask_price": last_close * 1.0002,
                "spread_bps": 4.0,
                "change_24h": float((last_close / frames["3m"]["close"].iloc[0]) - 1),
            }
            context = SignalContext(
                symbol=symbol,
                product_type=ProductType.USDT_FUTURES.value,
                frames=frames,
                levels=levels.levels,
                level_tags=levels.tags,
                regime=regime.regime,
                regime_notes=regime.notes,
                ticker=ticker,
                orderbook={"spread_bps": 4.0, "top5_notional": 500_000},
                historical_stats=self._historical_stats(),
            )

            candidates = []
            for strategy in [self.break_retest, self.liquidity_raid]:
                signal = strategy.evaluate(context)
                if signal:
                    risk_distance = max(abs(signal.entry_price - signal.stop_price), 1e-9)
                    signal.fees_r = (signal.entry_price * (self.settings.ev.estimated_maker_fee_bps * 2 / 10_000)) / risk_distance
                    signal.slippage_r = (signal.entry_price * (self.settings.ev.slippage_bps / 10_000)) / risk_distance
                    viability = evaluate_trade_viability(
                        context,
                        signal.stop_price,
                        signal.target_plan,
                        signal.fees_r,
                        signal.slippage_r,
                        entry_price=signal.entry_price,
                        side=signal.side,
                        strategy_name=signal.strategy.value,
                        min_rr_to_tp1_break_retest=self.settings.ev.min_rr_to_tp1_break_retest,
                        preferred_rr_to_tp2_break_retest=self.settings.ev.preferred_rr_to_tp2_break_retest,
                        min_rr_to_tp1_liquidity_raid=self.settings.ev.min_rr_to_tp1_liquidity_raid,
                        preferred_rr_to_tp2_liquidity_raid=self.settings.ev.preferred_rr_to_tp2_liquidity_raid,
                        reject_trade_if_targets_are_inside_range_middle=self.settings.strategy.reject_trade_if_targets_are_inside_range_middle,
                    )
                    signal.rr_to_tp1 = viability.rr_to_tp1
                    signal.rr_to_tp2 = viability.rr_to_tp2
                    signal.rr_to_best_target = viability.rr_to_best_target
                    signal.expected_r = viability.rr_to_tp2 or viability.rr_to_best_target or signal.expected_r
                    signal.ev_metrics = {**signal.ev_metrics, **viability.to_dict()}
                    if not viability.approved:
                        signal.trade_rejected_reason = viability.reject_reason
                        continue
                    signal.score = self.scorer.score(signal, regime.regime).final_score
                    candidates.append(signal)
            if not candidates:
                continue
            signal = max(candidates, key=lambda item: item.score)
            ev = self.ev_filter.evaluate(
                signal,
                historical_win_rate=context.historical_stats["win_rate"],
                historical_avg_win_r=context.historical_stats["avg_win_r"],
                historical_avg_loss_r=context.historical_stats["avg_loss_r"],
                spread_bps=ticker["spread_bps"],
                funding_minutes_away=None,
                news_penalty=0.0,
            )
            if not ev.approved:
                continue
            approved = self.risk_engine.approve(
                signal=signal,
                contract=self.contract,
                account_equity=account_equity,
                open_positions={},
                runtime_metrics={"daily_loss_r": 0.0, "consecutive_losses": 0, "daily_order_count": 0, "paused": False, "account_drawdown_pct": 0.0, "unrealized_loss_pct": 0.0},
                atr_value=signal.risk_per_unit,
                news_blocked=False,
                funding_blocked=False,
            )
            if approved is None:
                continue
            self.simulator.open_trade(approved, timestamp.to_pydatetime())

        self.simulator.finalize(df_3m.iloc[-1])
        report_content = build_report(self.simulator.closed_trades)
        report_path = write_report(output_path, report_content)
        return BacktestResult(trades=self.simulator.closed_trades, report_path=report_path)

    def _historical_stats(self) -> dict[str, float]:
        """Return rolling stats from simulated closed trades."""

        if not self.simulator.closed_trades:
            return {"win_rate": 0.45, "avg_win_r": 1.3, "avg_loss_r": 1.0}
        wins = [trade.pnl_r for trade in self.simulator.closed_trades if trade.pnl_r > 0]
        losses = [abs(trade.pnl_r) for trade in self.simulator.closed_trades if trade.pnl_r <= 0]
        trades = self.simulator.closed_trades
        return {
            "win_rate": len(wins) / len(trades),
            "avg_win_r": sum(wins) / len(wins) if wins else 1.3,
            "avg_loss_r": sum(losses) / len(losses) if losses else 1.0,
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the Bitget futures backtest.")
    parser.add_argument("--data", required=True, help="CSV path with 1m OHLCV data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for the report")
    parser.add_argument("--output", default="data/backtest/report.md", help="Markdown report output path")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    engine = BacktestEngine()
    result = engine.run(
        data_path=Path(args.data),
        symbol=args.symbol.upper(),
        output_path=Path(args.output),
    )
    print(f"Backtest complete: {result.report_path}")


if __name__ == "__main__":
    main()
