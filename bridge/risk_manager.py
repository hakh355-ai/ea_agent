"""
Risk gate and position sizing.
All functions are pure (no side effects) — state is passed in as arguments.
"""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ── Correlation map ───────────────────────────────────────────────────────────
# "same"    → block if correlated pair open in same direction
# "inverse" → block if correlated pair open in opposite direction
_CORRELATIONS = {
    "EURUSD": [("GBPUSD", "same")],
    "GBPUSD": [("EURUSD", "same")],
    "XAUUSD": [],
    "GER40":  [],
    "BTCUSD": [],
}

# Symbols that trade 24/7 (no session filter)
_CRYPTO = {"BTCUSD"}


def _check_session(timestamp_utc: str, symbol: str,
                   kill_zones_only: bool = False) -> tuple[bool, str]:
    """
    Session filter with optional TJR Kill Zone mode.

    Normal mode (kill_zones_only=False):
      Active 08:00-22:00 UTC on weekdays. Crypto 24/7.

    Kill Zone mode (kill_zones_only=True):
      Only trade during TJR Kill Zones (New York time):
        AM Kill Zone: 09:50-10:10 EST = 14:50-15:10 UTC (winter) / 13:50-14:10 UTC (summer)
        PM Kill Zone: 13:50-14:10 EST = 18:50-19:10 UTC (winter) / 17:50-18:10 UTC (summer)
      Allow ±30min buffer to cover EST/EDT seasonal shift.
      Crypto trades 24/7 regardless.
    """
    if symbol in _CRYPTO:
        return True, ""
    try:
        ts      = timestamp_utc[:19].replace("T", " ").replace(".", "-", 2)
        dt_utc  = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        dt_utc  = dt_utc.replace(tzinfo=timezone.utc)
        dt_de   = dt_utc.astimezone(ZoneInfo("Europe/Berlin"))

        weekday = dt_de.weekday()
        hour    = dt_de.hour
        minute  = dt_de.minute
        t_min   = hour * 60 + minute   # minutes since midnight Berlin time

        if weekday >= 5:
            return False, "weekend (market closed)"

        if kill_zones_only:
            # AM Kill Zone: 13:50-15:10 UTC (covers EST & EDT)
            t_min_utc = dt_utc.hour * 60 + dt_utc.minute
            am_start, am_end = 13 * 60 + 50, 15 * 60 + 10
            # PM Kill Zone: 17:50-19:10 UTC (covers EST & EDT)
            pm_start, pm_end = 17 * 60 + 50, 19 * 60 + 10

            in_am = am_start <= t_min_utc <= am_end
            in_pm = pm_start <= t_min_utc <= pm_end
            if not (in_am or in_pm):
                return False, f"outside_killzone (Berlin {hour:02d}:{minute:02d}, AM=13:50-15:10 UTC, PM=17:50-19:10 UTC)"
        else:
            if hour < 8 or hour >= 22:
                return False, f"outside_session (Berlin {hour:02d}:{minute:02d}, active 08-22 Berliner Zeit)"

    except Exception:
        pass
    return True, ""


def _check_correlation(symbol: str, action: str,
                       open_positions: dict) -> tuple[bool, str]:
    """
    Prevent opening a new trade that doubles up on an existing correlated position.
    Example: already long EURUSD → block long GBPUSD (same direction, same USD exposure).
    """
    for corr_sym, corr_type in _CORRELATIONS.get(symbol, []):
        if corr_sym not in open_positions:
            continue
        existing = open_positions[corr_sym].get("action", "")
        if corr_type == "same" and existing == action:
            return False, f"correlation ({symbol}↔{corr_sym} same direction)"
        if corr_type == "inverse" and existing != action and existing != "":
            return False, f"correlation ({symbol}↔{corr_sym} inverse conflict)"
    return True, ""


def pre_check(account: dict, symbol: str, news_flags: list[str],
              open_positions: dict, daily_realized_pnl: float,
              params: dict, timestamp_utc: str = "", action: str = "") -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Checks: session, position count, symbol duplicate, correlation,
            realized drawdown, equity drawdown, news blackout.
    """
    max_pos             = int(params.get("max_positions", int(os.getenv("MAX_CONCURRENT_POSITIONS", 3))))
    daily_limit         = float(params.get("daily_drawdown_limit", float(os.getenv("DAILY_DRAWDOWN_LIMIT", 0.05))))
    blackout_mins       = int(params.get("news_blackout_minutes", int(os.getenv("NEWS_BLACKOUT_MINUTES", 30))))
    kill_zones_only     = bool(params.get("kill_zones_only", False))

    balance = float(account.get("balance", 10000))
    equity  = float(account.get("equity", balance))

    # Daily profit target: percentage-based takes priority over fixed euro amount
    profit_target_pct   = float(params.get("daily_profit_target_pct", 0.0))
    fixed_profit_target = float(params.get("daily_profit_target", 0.0))
    if profit_target_pct > 0 and balance > 0:
        daily_profit_target = balance * profit_target_pct
    else:
        daily_profit_target = fixed_profit_target

    # 1. Session filter
    if timestamp_utc:
        ok, reason = _check_session(timestamp_utc, symbol, kill_zones_only)
        if not ok:
            return False, reason

    # 2. Max concurrent positions
    if len(open_positions) >= max_pos:
        return False, f"max_positions ({max_pos} reached)"

    # 3. Same symbol already open
    if symbol in open_positions:
        return False, f"position_exists ({symbol})"

    # 4. Correlation filter
    if action:
        ok, reason = _check_correlation(symbol, action, open_positions)
        if not ok:
            return False, reason

    # 5. Daily profit target reached → stop trading for today
    if daily_profit_target > 0 and daily_realized_pnl >= daily_profit_target:
        return False, f"daily_profit_target reached ({daily_realized_pnl:.2f} >= {daily_profit_target:.2f})"

    # 6. Daily realized drawdown
    if balance > 0 and abs(daily_realized_pnl) / balance >= daily_limit:
        return False, f"daily_drawdown ({daily_realized_pnl:.2f} / {balance:.2f})"

    # 6. Equity drawdown (floating losses)
    if balance > 0 and (balance - equity) / balance >= daily_limit:
        return False, f"equity_drawdown ({balance - equity:.2f})"

    # 7. News blackout (skip for crypto — NFP/CPI don't directly affect BTC/ETH)
    if symbol not in _CRYPTO:
        for flag in news_flags:
            flag_lower = flag.lower()
            if "in" in flag_lower and "min" in flag_lower:
                try:
                    part = flag_lower.split("in")[1].split("min")[0].strip()
                    mins = int("".join(c for c in part if c.isdigit()))
                    if mins <= blackout_mins:
                        return False, f"news_blackout ({flag})"
                except Exception:
                    pass

    return True, ""


def calc_lot_size(balance: float, sl_pips: float, risk_pct: float,
                  pip_value: float, symbol: str, params: dict,
                  consecutive_losses: int = 0) -> float:
    min_sl = params.get("min_sl_pips", {}).get(symbol, 10)
    sl_pips = max(float(sl_pips), float(min_sl))

    if sl_pips <= 0 or pip_value <= 0:
        return 0.01

    risk_amount = balance * risk_pct
    lots = risk_amount / (sl_pips * pip_value)

    # Losing streak protection: reduce lot size by 50% after 3+ consecutive losses
    if consecutive_losses >= 3:
        lots *= 0.5

    return max(0.01, min(100.0, round(lots, 2)))


def calc_sl_tp(atr: float, atr_multiplier: float, tp_sl_ratio: float,
               action: str, current_price: float) -> tuple[float, float]:
    sl_distance = atr * atr_multiplier
    tp_distance = sl_distance * tp_sl_ratio

    if action == "buy":
        sl = current_price - sl_distance
        tp = current_price + tp_distance
    else:
        sl = current_price + sl_distance
        tp = current_price - tp_distance

    return round(sl, 6), round(tp, 6)
