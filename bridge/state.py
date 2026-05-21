"""
Application state: open positions, daily PnL, news cache, strategy params.
Positions are persisted to disk on every change so bridge restarts don't lose state.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

_POSITIONS_PATH = Path(".tmp/positions.json")


_SMT_PAIRS = {"EURUSD": "GBPUSD", "GBPUSD": "EURUSD"}


@dataclass
class AppState:
    open_positions: dict = field(default_factory=dict)
    daily_realized_pnl: float = 0.0
    daily_reset_date: str = ""
    news_cache: dict = field(default_factory=dict)
    strategy_params: dict = field(default_factory=dict)
    consecutive_losses: int = 0
    symbol_losses_today: dict = field(default_factory=dict)  # {symbol: loss_count} resets daily
    smt_data: dict = field(default_factory=dict)  # {symbol: {structure, bos, last_hh, last_ll, timestamp}}
    pending_signals: dict = field(default_factory=dict)  # {request_id: signal data for Telegram fill notify}
    last_known_balance: float = 10000.0  # updated on every /signal request
    last_ticks: dict = field(default_factory=dict)  # {symbol: {bid, ask}} updated on every /signal


_state = AppState()
_position_lock = asyncio.Lock()   # prevents race condition on simultaneous signal requests


def get_state() -> AppState:
    return _state


def update_tick(symbol: str, bid: float, ask: float):
    _state.last_ticks[symbol] = {"bid": bid, "ask": ask}


def reset_daily_if_needed():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _state.daily_reset_date != today:
        _state.daily_realized_pnl = 0.0
        _state.daily_reset_date = today
        _state.symbol_losses_today = {}  # reset per-symbol loss counter daily


def record_symbol_loss(symbol: str):
    _state.symbol_losses_today[symbol] = _state.symbol_losses_today.get(symbol, 0) + 1


def get_symbol_losses_today(symbol: str) -> int:
    return _state.symbol_losses_today.get(symbol, 0)


def get_open_position_count() -> int:
    return len(_state.open_positions)


def _persist_positions():
    """Save open_positions to disk so bridge restarts don't lose state."""
    _POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _POSITIONS_PATH.write_text(
        json.dumps(_state.open_positions, indent=2), encoding="utf-8"
    )


def _load_positions():
    """Load persisted positions from disk on startup."""
    if _POSITIONS_PATH.exists():
        try:
            _state.open_positions = json.loads(
                _POSITIONS_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            _state.open_positions = {}


async def reserve_symbol(symbol: str, action: str) -> bool:
    """Atomically mark symbol as pending. Returns True if reserved, False if already exists.
    Uses asyncio lock to prevent race conditions when two EAs send simultaneous requests."""
    async with _position_lock:
        if symbol in _state.open_positions:
            return False
        _state.open_positions[symbol] = {
            "ticket": 0,
            "action": action,
            "fill_price": 0.0,
            "lots": 0.0,
            "open_time": time.time(),
            "pending": True,
        }
        _persist_positions()
        return True


def record_fill(ticket: int, symbol: str, action: str, fill_price: float, lots: float,
                sl: float = 0.0, tp: float = 0.0, confidence: float = 0.0, regime: str = ""):
    if action in ("buy", "sell"):
        _state.open_positions[symbol] = {
            "ticket":     ticket,
            "action":     action,
            "fill_price": fill_price,
            "lots":       lots,
            "sl":         sl,
            "tp":         tp,
            "confidence": confidence,
            "regime":     regime,
            "open_time":  time.time(),
        }
    elif action == "close" and symbol in _state.open_positions:
        del _state.open_positions[symbol]
    _persist_positions()


def sync_positions(broker_positions: dict):
    """Overwrite state with real broker positions (called on EA startup via /sync_positions)."""
    _state.open_positions = broker_positions
    _persist_positions()


def record_close_pnl(pnl: float, symbol: str = ""):
    _state.daily_realized_pnl += pnl
    if pnl < 0:
        _state.consecutive_losses += 1
        if symbol:
            _state.symbol_losses_today[symbol] = _state.symbol_losses_today.get(symbol, 0) + 1
    else:
        _state.consecutive_losses = 0


def get_cached_news(cache_key: str, ttl_seconds: int = 300) -> Optional[dict]:
    if cache_key in _state.news_cache:
        ts, result = _state.news_cache[cache_key]
        if time.time() - ts < ttl_seconds:
            return result
    return None


def set_news_cache(cache_key: str, result: dict):
    _state.news_cache[cache_key] = (time.time(), result)


def store_pending_signal(request_id: str, symbol: str, confidence: float,
                         regime: str, sl_price: float, tp_price: float,
                         sl_pips: float, tp_pips: float):
    _state.pending_signals[request_id] = {
        "symbol":     symbol,
        "confidence": confidence,
        "regime":     regime,
        "sl_price":   sl_price,
        "tp_price":   tp_price,
        "sl_pips":    sl_pips,
        "tp_pips":    tp_pips,
        "timestamp":  time.time(),
    }
    # Cleanup entries older than 2 minutes
    cutoff = time.time() - 120
    _state.pending_signals = {k: v for k, v in _state.pending_signals.items()
                               if v["timestamp"] > cutoff}


def get_pending_signal(request_id: str) -> dict:
    return _state.pending_signals.pop(request_id, {})


def update_symbol_structure(symbol: str, structure: str, bos: str,
                            last_hh: float, last_ll: float):
    _state.smt_data[symbol] = {
        "structure": structure,
        "bos":       bos,
        "last_hh":   last_hh,
        "last_ll":   last_ll,
        "timestamp": time.time(),
    }


def get_smt_signal(symbol: str) -> str:
    """Detect SMT divergence between correlated pairs (EURUSD↔GBPUSD)."""
    corr = _SMT_PAIRS.get(symbol)
    if not corr:
        return "none"
    my   = _state.smt_data.get(symbol, {})
    them = _state.smt_data.get(corr, {})
    if not my or not them:
        return "none"
    now = time.time()
    if now - my.get("timestamp", 0) > 600 or now - them.get("timestamp", 0) > 600:
        return "none"  # stale data

    my_bos      = my.get("bos", "none")
    them_bos    = them.get("bos", "none")
    my_struct   = my.get("structure", "")
    them_struct = them.get("structure", "")

    # Strong SMT: I made a BOS but correlated pair did not → fake breakout
    if my_bos == "bearish_bos" and them_bos != "bearish_bos":
        return "smt_bullish"   # my bearish break is fake → buy
    if my_bos == "bullish_bos" and them_bos != "bullish_bos":
        return "smt_bearish"   # my bullish break is fake → sell

    # Weak SMT: structural divergence without BOS
    if my_struct == "downtrend" and them_struct == "uptrend":
        return "smt_weak_bullish"
    if my_struct == "uptrend" and them_struct == "downtrend":
        return "smt_weak_bearish"

    return "none"


def load_strategy_params() -> dict:
    path = Path(os.getenv("STRATEGY_PARAMS_PATH", ".tmp/strategy_params.json"))
    if path.exists():
        _state.strategy_params = json.loads(path.read_text(encoding="utf-8"))
    _load_positions()  # restore persisted positions on every startup
    return _state.strategy_params


def reload_strategy_params() -> dict:
    return load_strategy_params()


def get_strategy_params() -> dict:
    if not _state.strategy_params:
        load_strategy_params()
    return _state.strategy_params
