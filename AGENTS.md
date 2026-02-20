# Project instructions (must follow)
- Primary goal: improve trading system expectancy and robustness; do NOT optimize for “more trades at any cost”.
- Do not make profitability guarantees; implement tooling to measure expectancy via backtests.
- Keep changes safe-by-default: sandbox mode on, no live trading without explicit config.
- Refactor incrementally: extract modules but preserve current behavior behind a feature flag when possible.
- Never silently swallow exceptions: log with throttle; always include reason codes for trade skips.
- Every strategy/risk rule must write structured logs to trade_details JSON.
- Add unit tests for risk/TP/SL conversion logic and trade validation.
- Keep dependencies minimal; prefer stdlib + existing deps.
- Do not depend on Telegram; remove or fully disable Telegram features.
