# CLAUDE.md

## Project Overview
Bitget futures automated trading bot. Python 3.12+, asyncio-based.

## Entry Points
- `python run_bot.py` — Headless trading bot
- `streamlit run run_streamlit.py` — Dashboard (port 8501)
- `pytest tests/` — Run tests

## Architecture
- `app/main.py` — TradingApplication orchestrator (5 async loops: universe, signal, news, order_sync, health)
- `core/` — Settings (Pydantic), persistence (SQLite), state store, logging, enums, utils
- `strategy/` — 4 strategies (break_retest, liquidity_raid, momentum_pullback, session_breakout) + scoring/filtering/routing
- `risk/` — RiskEngine, position sizing, trade guards, stops
- `execution/` — OrderManager, SLTPManager, FillHandler, OrderRouter
- `exchange/` — Bitget REST (tenacity retries) + WS (auto-reconnect), demo/live switching
- `market/` — Indicators, klines cache, regime classifier, orderbook, S/R levels, universe manager
- `ai/` — OpenAI Responses API client with structured outputs + caching
- `news/` — RSS collector, AI analyzer, impact filter, economic calendar
- `telegram_bot/` — Bot service (long-polling), commands, formatters
- `journal/` — Trade journal, performance analytics (Sharpe, expectancy, drawdown)
- `backtest/` — Engine, simulator, reports
- `dashboard/` — Streamlit multi-page app (overview, positions, symbols, news, journal, settings)

## Critical Rules
1. **Safe-by-default**: sandbox/demo mode on. No live trading without explicit `BOT_ALLOW_LIVE_TRADING=true` + config confirmation
2. **Never silently swallow exceptions**: log with reason codes. Use specific exception types, not bare `except Exception`
3. **Every trade decision must write structured logs** to SQLite persistence
4. **Fill verification mandatory**: never update local state (remaining_quantity, position) before confirming exchange fill
5. **Atomic state writes**: use tmp+rename for JSON files, transactions for multi-table SQLite ops
6. **No hardcoded thresholds in strategy code**: all tunable values must come from settings.json
7. **Test coverage required** for any risk/execution path change
8. **No .env in git**: secrets only via environment variables or .env (gitignored)

## Config
- `config/defaults.json` — Default settings
- `bot_settings.json` — Runtime overrides (auto-generated via dashboard)
- `.env` — Secrets (NEVER commit). See `.env.example` for required keys

## Key Dependencies
httpx, websockets, pandas, pydantic, pydantic-settings, openai, streamlit, tenacity, orjson, numpy

## Project Conventions
- Korean comments and UI text throughout
- Monolithic `aa.py` and `bot/` have been removed — all logic lives in modular packages
- Trading styles: 스캘핑 (Scalping), 단타 (Daytrading), 스윙 (Swing)
- 3 trading modes: DEMO (sandbox), LIVE (production), backtest
