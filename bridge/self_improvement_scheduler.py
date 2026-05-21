"""
Background scheduler — triggers all 4 self-improvement stages automatically.
Started by bridge/server.py on startup.

Schedule:
  Stufe 2 (trade analysis)   : every hour if 50+ new fills since last run
  Stufe 1 (prompt evolution) : every Sunday 02:00 UTC
  Stufe 4 (strategy discovery + backtest): every Sunday 03:00 UTC
"""
import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_last_fill_count = 0


def _check_trade_analysis():
    global _last_fill_count
    try:
        from tools.trade_logger import count_trades
        current = count_trades()
        if current - _last_fill_count >= 50:
            logger.info(f"Stufe 2: {current - _last_fill_count} new fills — running trade analysis...")
            from tools.analyze_trade_patterns import analyze
            analyze(min_trades=50)
            _last_fill_count = current
            # Reload params in bridge so new rules take effect immediately
            from bridge.state import reload_strategy_params
            reload_strategy_params()
            logger.info("Stufe 2: learned rules updated and reloaded.")
    except Exception as e:
        logger.error(f"Stufe 2 scheduler error: {e}")


def _run_prompt_evolution():
    logger.info("Stufe 1: Weekly prompt evolution starting...")
    try:
        from tools.evolve_trading_prompt import run_prompt_evolution
        run_prompt_evolution(generations=3, size=6)
        from bridge.state import reload_strategy_params
        reload_strategy_params()
        logger.info("Stufe 1: Trading prompt evolved and reloaded.")
    except Exception as e:
        logger.error(f"Stufe 1 scheduler error: {e}")


def _run_strategy_discovery():
    logger.info("Stufe 4: Weekly strategy discovery and backtest starting...")
    try:
        from tools.discover_strategy import discover
        from tools.backtest_strategy import run_backtest
        discover(n_candidates=2)
        run_backtest()
        from bridge.state import reload_strategy_params
        reload_strategy_params()
        logger.info("Stufe 4: Strategy pool updated and reloaded.")
    except Exception as e:
        logger.error(f"Stufe 4 scheduler error: {e}")


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Europe/Berlin")

    # Stufe 2: hourly check
    scheduler.add_job(_check_trade_analysis, "interval", hours=1, id="trade_analysis")

    # Stufe 1: Saturday 22:00 Berlin time
    scheduler.add_job(_run_prompt_evolution, "cron",
                      day_of_week="sat", hour=22, minute=0, id="prompt_evolution")

    # Stufe 4: Saturday 23:00 Berlin time (after prompt evolution finishes)
    scheduler.add_job(_run_strategy_discovery, "cron",
                      day_of_week="sat", hour=23, minute=0, id="strategy_discovery")

    scheduler.start()
    logger.info(
        "Self-improvement scheduler started: "
        "trade analysis (hourly), "
        "prompt evolution (Sat 22:00 Berlin), "
        "strategy discovery (Sat 23:00 Berlin)"
    )
    return scheduler
