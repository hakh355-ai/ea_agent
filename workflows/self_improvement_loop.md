# Self-Improvement Loop

## Objective
Continuously improve the EA's trading performance through 4 automatic stages.
All stages run on a schedule — no manual intervention needed after initial setup.

## Architecture Overview

```
Every 5 seconds (per symbol):
  MT5 → Bridge → [Regime Detection] → [Kimi Signal] → Trade

Every hour (automatic):
  Stufe 2: Check if 50+ new trades → analyze patterns → update learned_rules

Every Sunday 02:00 UTC (automatic):
  Stufe 1: Evolve Kimi's system prompt → update trading_prompt

Every Sunday 03:00 UTC (automatic):
  Stufe 4: Discover new strategies → backtest → promote to active pool
```

## Stufe 1 — Prompt Evolution (Sundays 02:00 UTC)

**What it does:** Treats Kimi's system prompt as a genetic organism. Generates variants,
scores each using GPT-4o as judge (weighing recent losing trades), evolves the best.

**How to run manually:**
```
python tools/evolve_trading_prompt.py --generations 3 --size 6
```

**Result:** `strategy_params.json → trading_prompt` updated. Bridge auto-reloads.

**When to run manually:** After a bad week (> 5 consecutive losses).

## Stufe 2 — Trade Feedback Loop (hourly, triggers at 50 fills)

**What it does:** Sends recent trade history (with entry context: RSI, regime, sentiment, confidence)
to GPT-4o. GPT-4o identifies patterns in wins vs. losses and writes specific rules.

**How to run manually:**
```
python tools/analyze_trade_patterns.py --min-trades 10
```

**Result:** `strategy_params.json → learned_rules` updated. Kimi reads these rules on every trade.

## Stufe 3 — Market Regime Detection (real-time, every signal)

**What it does:** Classifies market as `trending / ranging / volatile / quiet` using ADX + Bollinger Band width.
Applies different confidence thresholds and SL/TP multipliers per regime.

**Configuration:** Edit regime-specific params in `.tmp/strategy_params.json → regimes`.

**Regime definitions:**
- `trending`: ADX > 25, normal BB width → follow momentum, wider SL, larger TP
- `volatile`: ADX > 25, wide BB → high confidence required (0.80), very wide SL
- `ranging`: ADX < 20, normal BB → mean reversion, tighter TP
- `quiet`: ADX < 20, very tight BB → market compression, wait for breakout

## Stufe 4 — Strategy Discovery (Sundays 03:00 UTC)

**What it does:** Asks GPT-4o to propose new, distinct trading strategies using available indicators.
Backtests each against `trade_log.jsonl`. Promotes those with Sharpe >= 0.5 to "active" pool.

**How to run manually:**
```
python tools/discover_strategy.py --candidates 3
python tools/backtest_strategy.py --threshold 0.5
```

**Result:** `strategy_params.json → strategy_pool` updated with active/rejected strategies.
Active strategies are available for the trading agent to reference.

## Self-Improvement Metrics to Watch

| Metric | Target | How to check |
|--------|--------|-------------|
| Weekly Sharpe ratio | > 0.5 | Calculate from `.tmp/trade_log.jsonl` |
| Win rate | > 55% | `analyze_trade_patterns.py` output |
| Avg PnL per trade | > 0 | `analyze_trade_patterns.py` output |
| Learned rules count | 3-5 | `strategy_params.json → learned_rules` |
| Active strategies | ≥ 1 | `strategy_params.json → strategy_pool` |
| Prompt evolution score | > 0.7 | `evolve_trading_prompt.py` output |

## Edge Cases
| Problem | Response |
|---------|----------|
| < 50 trades for analysis | Stufe 2 skips and waits — no harm done |
| All strategies rejected in backtest | Normal if trade log is too small; retry after 2 weeks |
| Prompt evolution produces worse prompt | Seed (current prompt) is always included in generation 0; worst case: no change |
| API rate limit during discovery | Run discovery off-peak or reduce `--candidates` to 1 |
