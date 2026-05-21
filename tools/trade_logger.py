"""
Append-only trade log (JSONL). Records signals, fills, and closes.
Used by analyze_trade_patterns.py, backtest_strategy.py, and run_strategy_evolution.py.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_PATH = Path(os.getenv("TRADE_LOG_PATH", ".tmp/trade_log.jsonl"))


def _append(entry: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_signal(symbol: str, action: str, confidence: float, indicators: dict,
               news: dict, regime: str, lot_size: float, sl_price: float,
               tp_price: float, request_id: str):
    _append({
        "type": "signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "lot_size": lot_size,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "regime": regime,
        "rsi": indicators.get("rsi"),
        "atr": indicators.get("atr"),
        "adx": indicators.get("adx"),
        "macd_hist": indicators.get("macd_hist"),
        "bb_width": indicators.get("bb_width"),
        "news_sentiment": news.get("sentiment_score"),
        "news_flags": news.get("risk_flags", []),
    })


def log_fill(ticket: int, symbol: str, action: str, fill_price: float,
             lots: float, request_id: str):
    _append({
        "type": "fill",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "ticket": ticket,
        "symbol": symbol,
        "action": action,
        "fill_price": fill_price,
        "lots": lots,
    })


def log_close(ticket: int, symbol: str, close_price: float, pnl: float, outcome: str):
    _append({
        "type": "close",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket": ticket,
        "symbol": symbol,
        "close_price": close_price,
        "pnl": pnl,
        "outcome": outcome,  # "tp_hit" | "sl_hit" | "manual"
    })


def read_recent_trades(days: int = 7) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    trades = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    trades.append(entry)
            except Exception:
                pass
    return trades


def count_trades() -> int:
    if not LOG_PATH.exists():
        return 0
    count = 0
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if '"type": "fill"' in line:
                count += 1
    return count
