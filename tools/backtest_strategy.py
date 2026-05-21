"""
Stufe 4b: Backtest candidate strategies against trade_log.jsonl.
Promotes strategies with Sharpe >= threshold to "active", rejects others.
Usage: python tools/backtest_strategy.py [--threshold 0.5]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from tools.trade_logger import read_recent_trades

load_dotenv()


def _sharpe(pnls: list[float]) -> float:
    arr = np.array(pnls, dtype=float)
    if arr.std() < 1e-9:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def evaluate(strategy: dict, trades: list[dict]) -> dict:
    best_regimes    = set(strategy.get("best_regimes", ["trending", "ranging", "volatile", "quiet"]))
    best_instruments = set(strategy.get("best_instruments", []))
    min_conf        = float(strategy.get("exit_rules", {}).get("min_confidence", 0.65))

    signals = {}
    for t in trades:
        if t["type"] == "signal":
            signals[t.get("symbol", "")] = t

    pnls = []
    for close in [t for t in trades if t["type"] == "close"]:
        symbol = close.get("symbol", "")
        if best_instruments and symbol not in best_instruments:
            continue
        sig = signals.get(symbol, {})
        if sig.get("regime") not in best_regimes:
            continue
        if float(sig.get("confidence", 0)) < min_conf:
            continue
        pnls.append(float(close.get("pnl", 0)))

    if len(pnls) < 5:
        return {"sharpe": -1.0, "win_rate": 0.0, "trades": len(pnls), "avg_pnl": 0.0}

    arr = np.array(pnls)
    return {
        "sharpe":   round(_sharpe(pnls), 3),
        "win_rate": round(float((arr > 0).mean()), 3),
        "trades":   len(pnls),
        "avg_pnl":  round(float(arr.mean()), 2),
    }


def run_backtest(promote_threshold: float = 0.5):
    params_path = Path(os.getenv("STRATEGY_PARAMS_PATH", ".tmp/strategy_params.json"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    pool = params.get("strategy_pool", [])

    candidates = [s for s in pool if s.get("status") == "candidate"]
    if not candidates:
        print("No candidate strategies. Run discover_strategy.py first.")
        return

    trades = read_recent_trades(days=30)
    print(f"Backtesting {len(candidates)} candidate(s) on {len(trades)} log entries...")

    for s in candidates:
        metrics = evaluate(s, trades)
        s["metrics"] = metrics
        promoted = metrics["sharpe"] >= promote_threshold and metrics["trades"] >= 5
        s["status"] = "active" if promoted else "rejected"
        badge = "PROMOTED" if promoted else "Rejected"
        print(
            f"  {s['strategy_name']}: Sharpe={metrics['sharpe']:.3f} "
            f"WinRate={metrics['win_rate']:.1%} Trades={metrics['trades']} → {badge}"
        )

    params["strategy_pool"] = pool
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print("\nPool updated in strategy_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_backtest(args.threshold)
