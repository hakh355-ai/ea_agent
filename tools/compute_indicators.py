"""
Compute technical indicators from OHLC bars.

Sources:
  Candlestick patterns : Steve Nison, "Japanese Candlestick Charting Techniques" (1991)
  Market structure     : Richard Wyckoff Method (1930s) + ICT concepts
  Support/Resistance   : Al Brooks, "Reading Price Charts Bar by Bar" (2009)
  Fibonacci levels     : Standard ratios derived from Fibonacci sequence (0.236, 0.382, 0.5, 0.618, 0.786)
  Chart patterns       : Thomas Bulkowski, "Encyclopedia of Chart Patterns" (2000)
  Standard indicators  : ta library (RSI, ATR, EMA, MACD, ADX, Bollinger Bands)

Usage: python tools/compute_indicators.py --bars <json_file>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ta


# ── Candlestick Patterns (Steve Nison) ───────────────────────────────────────

def _candlestick_patterns(df: pd.DataFrame, atr: float) -> dict:
    """
    Detects single and multi-candle patterns from the last 3 bars.

    Definitions follow Steve Nison, "Japanese Candlestick Charting Techniques":
      Doji          p.31-36  — open ≈ close, indecision
      Hammer        p.37-42  — small body top, long lower shadow ≥ 2x body
      Shooting Star p.72-73  — small body bottom, long upper shadow ≥ 2x body
      Engulfing     p.43-50  — current body completely engulfs previous body
      Pin Bar       Price action standard — body < 25% range, long wick rejection
      Morning Star  p.57-61  — large bearish → small body → large bullish (3-bar reversal)
      Evening Star  p.57-61  — large bullish → small body → large bearish (3-bar reversal)
    """
    if len(df) < 3:
        return {"patterns": [], "pattern_bias": "neutral"}

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # Last 3 candles (index -3, -2, -1)
    o1, h1, l1, c1 = o[-3], h[-3], l[-3], c[-3]  # 3 bars ago
    o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]  # 2 bars ago
    o3, h3, l3, c3 = o[-1], h[-1], l[-1], c[-1]  # current (last) bar

    def body(op, cl):   return abs(cl - op)
    def rng(hi, lo):    return hi - lo if hi - lo > 1e-10 else 1e-10
    def upper_wick(op, cl, hi): return hi - max(op, cl)
    def lower_wick(op, cl, lo): return min(op, cl) - lo
    def is_bull(op, cl): return cl > op
    def is_bear(op, cl): return cl < op

    body3  = body(o3, c3)
    rng3   = rng(h3, l3)
    body2  = body(o2, c2)
    rng2   = rng(h2, l2)
    body1  = body(o1, c1)

    patterns = []
    bullish_count = 0
    bearish_count = 0

    # ── Doji (Nison p.31-36) ─────────────────────────────────────────────────
    # Body < 5% of candle range → pure indecision
    if body3 < rng3 * 0.05:
        patterns.append("doji")

    # ── Hammer / Hanging Man (Nison p.37-42) ─────────────────────────────────
    # Small body in upper 33% of range, lower shadow ≥ 2x body, tiny upper wick
    low_wick3   = lower_wick(o3, c3, l3)
    up_wick3    = upper_wick(o3, c3, h3)
    body_top    = max(o3, c3)
    if (body3 < rng3 * 0.33
            and low_wick3 >= 2 * body3
            and up_wick3 <= body3 * 0.5
            and body_top >= l3 + rng3 * 0.6):
        patterns.append("hammer")
        bullish_count += 1

    # ── Shooting Star / Inverted Hammer (Nison p.72-73) ──────────────────────
    # Small body in lower 33% of range, upper shadow ≥ 2x body, tiny lower wick
    body_bot = min(o3, c3)
    if (body3 < rng3 * 0.33
            and up_wick3 >= 2 * body3
            and low_wick3 <= body3 * 0.5
            and body_bot <= l3 + rng3 * 0.4):
        patterns.append("shooting_star")
        bearish_count += 1

    # ── Pin Bar (Price action rejection candle) ───────────────────────────────
    # Body < 25% of range AND one wick > 60% of total range → strong rejection
    if body3 < rng3 * 0.25:
        if low_wick3 > rng3 * 0.6:
            patterns.append("pin_bar_bullish")
            bullish_count += 2
        elif up_wick3 > rng3 * 0.6:
            patterns.append("pin_bar_bearish")
            bearish_count += 2

    # ── Bullish Engulfing (Nison p.43-50) ────────────────────────────────────
    # Current bullish candle body completely engulfs previous bearish body
    if (is_bear(o2, c2) and is_bull(o3, c3)
            and o3 <= c2 and c3 >= o2
            and body3 > body2):
        patterns.append("bullish_engulfing")
        bullish_count += 2

    # ── Bearish Engulfing (Nison p.43-50) ────────────────────────────────────
    # Current bearish candle body completely engulfs previous bullish body
    if (is_bull(o2, c2) and is_bear(o3, c3)
            and o3 >= c2 and c3 <= o2
            and body3 > body2):
        patterns.append("bearish_engulfing")
        bearish_count += 2

    # ── Morning Star (Nison p.57-61) — 3-bar bullish reversal ───────────────
    # Large bearish | Small indecision body | Large bullish recovering into bar 1
    if (is_bear(o1, c1) and body1 > atr * 0.5
            and body2 < body1 * 0.4
            and is_bull(o3, c3) and body3 > atr * 0.5
            and c3 > (o1 + c1) / 2):
        patterns.append("morning_star")
        bullish_count += 3

    # ── Evening Star (Nison p.57-61) — 3-bar bearish reversal ───────────────
    # Large bullish | Small indecision body | Large bearish recovering into bar 1
    if (is_bull(o1, c1) and body1 > atr * 0.5
            and body2 < body1 * 0.4
            and is_bear(o3, c3) and body3 > atr * 0.5
            and c3 < (o1 + c1) / 2):
        patterns.append("evening_star")
        bearish_count += 3

    # ── Bias summary ─────────────────────────────────────────────────────────
    if bullish_count > bearish_count:
        bias = "bullish"
    elif bearish_count > bullish_count:
        bias = "bearish"
    else:
        bias = "neutral"

    return {"patterns": patterns, "pattern_bias": bias}


# ── Market Structure — Wyckoff / ICT ─────────────────────────────────────────

def _market_structure(df: pd.DataFrame) -> dict:
    """
    Identifies swing highs/lows and trend direction.

    Based on:
      - Wyckoff Method (Richard Wyckoff, 1930s): accumulation/distribution phases
      - ICT (Inner Circle Trader): Break of Structure (BOS), Change of Character (CHoCH)

    Swing detection: a pivot high is a bar whose high is higher than the 2 bars
    on each side (classic 5-bar pivot, widely used in institutional analysis).

    Market structure:
      HH + HL = uptrend   (Higher Highs + Higher Lows)
      LH + LL = downtrend (Lower Highs + Lower Lows)
      Mixed   = consolidation / no clear structure
    """
    if len(df) < 20:
        return {"structure": "unknown", "swing_highs": [], "swing_lows": [],
                "bos": "none", "last_hh": 0.0, "last_ll": 0.0}

    highs = df["high"].values
    lows  = df["low"].values

    swing_highs = []
    swing_lows  = []

    # 5-bar pivot: bar[i] is pivot high if high[i] > high[i±1] and high[i±2]
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_highs.append(float(highs[i]))
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2]
                and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swing_lows.append(float(lows[i]))

    # Need at least 2 of each to determine structure
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "insufficient_data", "swing_highs": swing_highs[-3:],
                "swing_lows": swing_lows[-3:], "bos": "none",
                "last_hh": swing_highs[-1] if swing_highs else 0.0,
                "last_ll": swing_lows[-1] if swing_lows else 0.0}

    last_hh = swing_highs[-1]
    prev_hh = swing_highs[-2]
    last_ll = swing_lows[-1]
    prev_ll = swing_lows[-2]

    higher_highs = last_hh > prev_hh   # HH: bullish structure
    higher_lows  = last_ll > prev_ll   # HL: bullish confirmation
    lower_highs  = last_hh < prev_hh   # LH: bearish structure
    lower_lows   = last_ll < prev_ll   # LL: bearish confirmation

    if higher_highs and higher_lows:
        structure = "uptrend"
    elif lower_highs and lower_lows:
        structure = "downtrend"
    elif higher_highs and lower_lows:
        structure = "expansion"        # volatility expansion
    elif lower_highs and higher_lows:
        structure = "consolidation"    # price squeezing
    else:
        structure = "ranging"

    # Break of Structure detection (ICT/SMC)
    # BOS is a REVERSAL signal — only valid against the current structure:
    #   downtrend → bullish BOS (reversal up)   | uptrend → bearish BOS (reversal down)
    #   ranging/other → both directions valid
    current_close = float(df["close"].iloc[-1])
    if structure in ("downtrend", "ranging", "consolidation", "expansion", "insufficient_data"):
        if any(current_close > sh for sh in swing_highs[-3:]):
            bos = "bullish_bos"
        else:
            bos = "none"
    elif structure in ("uptrend",):
        if any(current_close < sl for sl in swing_lows[-3:]):
            bos = "bearish_bos"
        else:
            bos = "none"
    else:
        bos = "none"

    return {
        "structure":   structure,
        "swing_highs": [round(x, 5) for x in swing_highs[-3:]],
        "swing_lows":  [round(x, 5) for x in swing_lows[-3:]],
        "bos":         bos,
        "last_hh":     round(last_hh, 5),
        "last_ll":     round(last_ll, 5),
    }


# ── Support & Resistance (Al Brooks) ─────────────────────────────────────────

def _support_resistance(df: pd.DataFrame, atr: float) -> dict:
    """
    Identifies key S&R levels from recent swing highs/lows.

    Based on Al Brooks, "Reading Price Charts Bar by Bar" (2009):
    Key levels are price areas where the market has reversed multiple times.
    Levels within 0.5x ATR of each other are clustered into one zone.

    Returns nearest resistance above price and nearest support below price.
    """
    if len(df) < 20 or atr < 1e-10:
        return {"resistance": 0.0, "support": 0.0, "key_levels": []}

    highs  = df["high"].values[-50:]
    lows   = df["low"].values[-50:]
    closes = df["close"].values

    current_price = float(closes[-1])
    cluster_dist  = atr * 0.5

    raw_levels = []
    # Collect pivot highs (resistance candidates)
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] \
                and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            raw_levels.append(float(highs[i]))
    # Collect pivot lows (support candidates)
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] \
                and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            raw_levels.append(float(lows[i]))

    # Cluster nearby levels (within 0.5 ATR)
    raw_levels.sort()
    clustered = []
    for lvl in raw_levels:
        if not clustered or abs(lvl - clustered[-1]) > cluster_dist:
            clustered.append(round(lvl, 5))
        else:
            # Merge: take midpoint
            clustered[-1] = round((clustered[-1] + lvl) / 2, 5)

    resistance_levels = sorted([l for l in clustered if l > current_price])
    support_levels    = sorted([l for l in clustered if l < current_price], reverse=True)

    return {
        "resistance": resistance_levels[0] if resistance_levels else 0.0,
        "support":    support_levels[0]    if support_levels    else 0.0,
        "key_levels": clustered[-8:],      # last 8 key levels
    }


# ── Fibonacci Retracement ─────────────────────────────────────────────────────

def _fibonacci(df: pd.DataFrame) -> dict:
    """
    Calculates Fibonacci retracement levels from the most recent significant swing.

    Standard Fibonacci ratios derived from the Fibonacci sequence (Leonardo Fibonacci, 1202):
      0.236, 0.382, 0.500, 0.618, 0.786

    Method: Find the highest high and lowest low in the last 50 bars.
    Determine swing direction from which end formed more recently.
    """
    if len(df) < 20:
        return {"fib_levels": {}, "nearest_fib": 0.0, "fib_direction": "unknown"}

    window    = df.iloc[-50:]
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())

    high_idx = window["high"].idxmax()
    low_idx  = window["low"].idxmin()

    # Direction: if low formed after high → upswing retracement
    #            if high formed after low → downswing retracement
    if low_idx > high_idx:
        # Price fell from high to low → retracing back up (bullish fib)
        direction = "bullish"
        diff = swing_high - swing_low
        levels = {
            "0.0":   round(swing_low, 5),
            "0.236": round(swing_low + diff * 0.236, 5),
            "0.382": round(swing_low + diff * 0.382, 5),
            "0.500": round(swing_low + diff * 0.500, 5),
            "0.618": round(swing_low + diff * 0.618, 5),
            "0.786": round(swing_low + diff * 0.786, 5),
            "0.790": round(swing_low + diff * 0.790, 5),
            "1.0":   round(swing_high, 5),
        }
    else:
        # Price rose from low to high → retracing back down (bearish fib)
        direction = "bearish"
        diff = swing_high - swing_low
        levels = {
            "0.0":   round(swing_high, 5),
            "0.236": round(swing_high - diff * 0.236, 5),
            "0.382": round(swing_high - diff * 0.382, 5),
            "0.500": round(swing_high - diff * 0.500, 5),
            "0.618": round(swing_high - diff * 0.618, 5),
            "0.786": round(swing_high - diff * 0.786, 5),
            "0.790": round(swing_high - diff * 0.790, 5),
            "1.0":   round(swing_low, 5),
        }

    # Find which Fib level price is closest to
    current = float(df["close"].iloc[-1])
    nearest = min(levels.values(), key=lambda x: abs(x - current))

    return {
        "fib_levels":   levels,
        "nearest_fib":  round(nearest, 5),
        "fib_direction": direction,
    }


# ── Equilibrium / Discount-Premium Zone (TJR / ICT) ─────────────────────────

def _equilibrium(df: pd.DataFrame) -> dict:
    """
    TJR/ICT Equilibrium: 50% of the most recent significant swing range.
    - Below 50% = Discount zone  → look for BUY setups
    - Above 50% = Premium zone   → look for SELL setups
    - At 50%    = Equilibrium    → highest probability entry area
    """
    if len(df) < 20:
        return {"eq_level": 0.0, "eq_zone": "unknown", "swing_high": 0.0, "swing_low": 0.0}

    window = df.iloc[-50:] if len(df) >= 50 else df
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())
    current    = float(df["close"].iloc[-1])

    if swing_high <= swing_low:
        return {"eq_level": 0.0, "eq_zone": "unknown", "swing_high": swing_high, "swing_low": swing_low}

    eq_level = round((swing_high + swing_low) / 2, 6)
    range_size = swing_high - swing_low
    tolerance = range_size * 0.05   # 5% of range = "at equilibrium"

    if current < eq_level - tolerance:
        eq_zone = "discount"     # below 50% → look for longs
    elif current > eq_level + tolerance:
        eq_zone = "premium"      # above 50% → look for shorts
    else:
        eq_zone = "equilibrium"  # at 50% → highest probability

    return {
        "eq_level":   eq_level,
        "eq_zone":    eq_zone,        # "discount" | "premium" | "equilibrium"
        "swing_high": round(swing_high, 6),
        "swing_low":  round(swing_low, 6),
    }


# ── VPA — Volume Price Analysis ───────────────────────────────────────────────

def _compute_vpa(df: pd.DataFrame, atr: float) -> str:
    """
    Volume Price Analysis — classifies last candle's volume/price relationship.

    absorption : High volume + small body → big player absorbing the move (reversal likely)
    breakout   : High volume + large body in trend direction → genuine breakout
    fakeout    : High volume + wick rejection (body < 30% of range) → false move
    weak_trend : Low volume + price moving → trend losing steam
    neutral    : No significant signal
    """
    if len(df) < 21:
        return "neutral"

    vol    = df["volume"]
    body   = (df["close"] - df["open"]).abs()
    rng    = df["high"] - df["low"]

    last_vol  = float(vol.iloc[-1])
    avg_vol   = float(vol.iloc[-21:-1].mean())
    last_body = float(body.iloc[-1])
    last_rng  = float(rng.iloc[-1])

    if avg_vol < 1e-9:
        return "neutral"

    vol_ratio  = last_vol / avg_vol
    body_ratio = last_body / last_rng if last_rng > 1e-9 else 0.5

    high_vol   = vol_ratio > 1.5
    low_vol    = vol_ratio < 0.6
    large_body = body_ratio > 0.6
    small_body = body_ratio < 0.3

    if high_vol and small_body:
        return "absorption"
    if high_vol and large_body:
        return "breakout"
    if high_vol and not large_body and not small_body:
        return "fakeout"
    if low_vol and last_body > atr * 0.3:
        return "weak_trend"
    return "neutral"


# ── SMC: Fair Value Gap (ICT) ─────────────────────────────────────────────────

def _fvg(df: pd.DataFrame, atr: float) -> dict:
    """
    Fair Value Gap: 3-candle imbalance where price may return to fill.
    Bullish FVG: high[i-1] < low[i+1]  →  gap above price = magnet for buys
    Bearish FVG: low[i-1] > high[i+1]  →  gap below price = magnet for sells
    """
    if len(df) < 3 or atr < 1e-10:
        return {"bullish_fvgs": [], "bearish_fvgs": []}

    h = df["high"].values
    l = df["low"].values
    current = float(df["close"].iloc[-1])
    fvgs_bull, fvgs_bear = [], []

    for i in range(1, len(df) - 1):
        if h[i-1] < l[i+1]:
            fvgs_bull.append({"top": round(float(l[i+1]), 6),
                              "bottom": round(float(h[i-1]), 6)})
        elif l[i-1] > h[i+1]:
            fvgs_bear.append({"top": round(float(l[i-1]), 6),
                              "bottom": round(float(h[i+1]), 6)})

    def _nearest(lst):
        if not lst:
            return []
        return sorted(lst, key=lambda x: abs((x["top"]+x["bottom"])/2 - current))[:2]

    return {"bullish_fvgs": _nearest(fvgs_bull),
            "bearish_fvgs": _nearest(fvgs_bear)}


# ── SMC: Inverse Fair Value Gap (IFVG) — ICT reversal signal ─────────────────

def _ifvg(fvg_data: dict, structure: dict) -> dict:
    """
    IFVG: an FVG forming AGAINST the current structure = reversal signal at key level.
    In downtrend: bullish FVG is IFVG (price trying to reverse up).
    In uptrend:   bearish FVG is IFVG (price trying to reverse down).
    """
    struct       = structure.get("structure", "unknown")
    bullish_fvgs = fvg_data.get("bullish_fvgs", [])
    bearish_fvgs = fvg_data.get("bearish_fvgs", [])

    if struct in ("downtrend",):
        return {"ifvg_bullish": bullish_fvgs[:2], "ifvg_bearish": []}
    elif struct in ("uptrend",):
        return {"ifvg_bullish": [], "ifvg_bearish": bearish_fvgs[:2]}
    else:
        return {"ifvg_bullish": bullish_fvgs[:1], "ifvg_bearish": bearish_fvgs[:1]}


# ── Session Highs/Lows (ICT — Asia / London / NY) ────────────────────────────

def _session_highs_lows(h1_bars: list, timestamp_utc: str) -> dict:
    """
    Compute session highs/lows server-side from H1 bars (no EA changes needed).
    Sessions (UTC): Asia 22:00–06:59 | London 07:00–11:59 | NY 12:00–16:59
    These are HTF draws on liquidity — price is magnetically drawn to them.
    """
    from datetime import datetime, timezone

    if not h1_bars:
        return {"asia_high": 0.0, "asia_low": 0.0,
                "london_high": 0.0, "london_low": 0.0,
                "ny_high": 0.0, "ny_low": 0.0, "session_draws": []}

    try:
        now_str  = str(timestamp_utc)[:19].replace("T", " ")
        now_utc  = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        today    = now_utc.date()
    except Exception:
        today = None

    asia_h, asia_l, lon_h, lon_l, ny_h, ny_l = [], [], [], [], [], []

    for bar in h1_bars[-50:]:
        t_str = str(bar.get("t", ""))[:19].replace("T", " ")
        try:
            bar_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if today and (today - bar_dt.date()).days > 1:
            continue

        bh = float(bar.get("h", 0))
        bl = float(bar.get("l", 0))
        hr = bar_dt.hour

        if hr >= 22 or hr <= 6:
            asia_h.append(bh); asia_l.append(bl)
        elif 7 <= hr <= 11:
            lon_h.append(bh);  lon_l.append(bl)
        elif 12 <= hr <= 16:
            ny_h.append(bh);   ny_l.append(bl)

    result = {
        "asia_high":   round(max(asia_h), 6) if asia_h else 0.0,
        "asia_low":    round(min(asia_l), 6) if asia_l else 0.0,
        "london_high": round(max(lon_h),  6) if lon_h  else 0.0,
        "london_low":  round(min(lon_l),  6) if lon_l  else 0.0,
        "ny_high":     round(max(ny_h),   6) if ny_h   else 0.0,
        "ny_low":      round(min(ny_l),   6) if ny_l   else 0.0,
    }
    result["session_draws"] = sorted({v for v in result.values() if v > 0.0})
    return result


# ── SMC: Balanced Price Range (BPR) ──────────────────────────────────────────

def _bpr(fvg_bull: list, fvg_bear: list) -> list:
    """
    Balanced Price Range: overlap zone between a bullish FVG and a bearish FVG.
    Overlap = 'fair price' — highest-probability reversal/entry zone.
    Stronger signal than either FVG alone.
    """
    zones = []
    for b in fvg_bull:
        for s in fvg_bear:
            overlap_bottom = max(b["bottom"], s["bottom"])
            overlap_top    = min(b["top"],    s["top"])
            if overlap_top > overlap_bottom:
                zones.append({
                    "top":    round(overlap_top, 6),
                    "bottom": round(overlap_bottom, 6),
                    "mid":    round((overlap_top + overlap_bottom) / 2, 6),
                })
    return zones[:2]


# ── SMC: Order Block (ICT) ────────────────────────────────────────────────────

def _order_blocks(df: pd.DataFrame, atr: float) -> dict:
    """
    Order Block: last opposing candle before an impulsive move (3+ candles).
    Bullish OB: last bearish candle before 3 consecutive bullish candles.
    Bearish OB: last bullish candle before 3 consecutive bearish candles.
    """
    if len(df) < 8 or atr < 1e-10:
        return {"bullish_ob": None, "bearish_ob": None}

    o = df["open"].values
    c = df["close"].values
    bull_ob, bear_ob = None, None

    for i in range(len(df) - 5, 1, -1):
        remaining = len(c) - i - 1
        if remaining < 3:
            continue
        if c[i] < o[i] and not bull_ob:
            if all(c[i+k] > o[i+k] for k in range(1, 4)):
                bull_ob = {"top":    round(float(max(o[i], c[i])), 6),
                           "bottom": round(float(min(o[i], c[i])), 6),
                           "bars_ago": len(df) - 1 - i}
        if c[i] > o[i] and not bear_ob:
            if all(c[i+k] < o[i+k] for k in range(1, 4)):
                bear_ob = {"top":    round(float(max(o[i], c[i])), 6),
                           "bottom": round(float(min(o[i], c[i])), 6),
                           "bars_ago": len(df) - 1 - i}
        if bull_ob and bear_ob:
            break

    return {"bullish_ob": bull_ob, "bearish_ob": bear_ob}


# ── SMC: Breaker Block (ICT) ──────────────────────────────────────────────────

def _breaker_blocks(df: pd.DataFrame, ob_bull: dict | None, ob_bear: dict | None) -> dict:
    """
    Breaker Block: Order Block violated by price closing through it.
    - Bullish OB breached (close < OB bottom) → Bearish Breaker (former support = now resistance)
    - Bearish OB breached (close > OB top)    → Bullish Breaker (former resistance = now support)
    Price often retests a Breaker from the opposite side — high-probability entry.
    """
    current = float(df["close"].iloc[-1])
    bull_breaker = None
    bear_breaker = None

    if ob_bull and current < ob_bull["bottom"]:
        bear_breaker = {
            "top":    ob_bull["top"],
            "bottom": ob_bull["bottom"],
            "note":   "former bullish OB — now bearish resistance",
        }
    if ob_bear and current > ob_bear["top"]:
        bull_breaker = {
            "top":    ob_bear["top"],
            "bottom": ob_bear["bottom"],
            "note":   "former bearish OB — now bullish support",
        }

    return {"bull_breaker": bull_breaker, "bear_breaker": bear_breaker}


# ── SMC: Equal Highs / Equal Lows (EQ) ───────────────────────────────────────

def _equal_hl(df: pd.DataFrame, atr: float) -> dict:
    """
    Equal Highs/Lows: swing levels at same price = liquidity pool.
    Price will likely sweep these to collect stop orders before reversing.
    """
    if len(df) < 20 or atr < 1e-10:
        return {"equal_highs": [], "equal_lows": []}

    h = df["high"].values
    l = df["low"].values
    tolerance = atr * 0.15

    swing_highs, swing_lows = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append(float(l[i]))

    def find_eq(levels):
        eq = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                if abs(levels[i] - levels[j]) <= tolerance:
                    lvl = round((levels[i] + levels[j]) / 2, 6)
                    if lvl not in eq:
                        eq.append(lvl)
        return eq

    return {"equal_highs": sorted(find_eq(swing_highs))[-3:],
            "equal_lows":  sorted(find_eq(swing_lows))[:3]}


# ── SMC: Liquidity Sweep ──────────────────────────────────────────────────────

def _liquidity_sweep(df: pd.DataFrame) -> dict:
    """
    Liquidity Sweep: price wicks past a recent swing high/low but closes back.
    Bullish sweep: wick below recent low, closed above → long opportunity.
    Bearish sweep: wick above recent high, closed below → short opportunity.
    """
    if len(df) < 15:
        return {"sweep": "none", "sweep_level": 0.0}

    lookback = df.iloc[-15:-3]
    if len(lookback) < 5:
        return {"sweep": "none", "sweep_level": 0.0}

    recent_high = float(lookback["high"].max())
    recent_low  = float(lookback["low"].min())
    last_h = float(df["high"].iloc[-1])
    last_l = float(df["low"].iloc[-1])
    last_c = float(df["close"].iloc[-1])

    if last_h > recent_high and last_c < recent_high:
        return {"sweep": "bearish", "sweep_level": round(recent_high, 6)}
    if last_l < recent_low and last_c > recent_low:
        return {"sweep": "bullish", "sweep_level": round(recent_low, 6)}

    return {"sweep": "none", "sweep_level": 0.0}


# ── SMC: Displacement Candle (ICT) ───────────────────────────────────────────

def _displacement_candle(df: pd.DataFrame, atr: float) -> dict:
    """
    Displacement candle: large impulsive candle proving Smart Money entered with force.
    Occurs after a liquidity sweep — confirms the manipulation phase is COMPLETE.

    Criteria:
      - Body > 1.5× ATR  (institutional size move)
      - Body > 70% of candle range (strong close, minimal wicks = conviction)
      - Checks last 2 candles (displacement might be 1 bar ago)

    displacement_size = body / ATR ratio (higher = stronger Smart Money move)
    """
    if len(df) < 3 or atr < 1e-10:
        return {"displacement": "none", "displacement_size": 0.0}

    best = {"displacement": "none", "displacement_size": 0.0}

    for idx in [-2, -1]:
        bar  = df.iloc[idx]
        o    = float(bar["open"])
        c    = float(bar["close"])
        h    = float(bar["high"])
        lo   = float(bar["low"])
        body = abs(c - o)
        rng  = h - lo
        if rng < 1e-10:
            continue

        body_ratio = body / rng
        size_ratio = body / atr

        if size_ratio >= 1.5 and body_ratio >= 0.70:
            direction = "bullish" if c > o else "bearish"
            if size_ratio > best["displacement_size"]:
                best = {"displacement": direction, "displacement_size": round(size_ratio, 2)}

    return best


# ── SMC: Change of Character (CHoCH) ─────────────────────────────────────────

def _choch(structure: dict, df: pd.DataFrame) -> str:
    """
    Change of Character: first sign price is reversing structure.
    Weaker signal than BOS — triggers when price approaches the opposing swing.
    CHoCH confirmed = same as BOS but named differently for prompt clarity.
    """
    bos = structure.get("bos", "none")
    if bos != "none":
        return bos.replace("bos", "choch_confirmed")

    current   = float(df["close"].iloc[-1])
    last_hh   = structure.get("last_hh", 0.0)
    last_ll   = structure.get("last_ll", 0.0)
    struct    = structure.get("structure", "")

    if struct == "downtrend" and last_hh > 0 and current > last_hh * 0.999:
        return "potential_bullish_choch"
    if struct == "uptrend" and last_ll > 0 and current < last_ll * 1.001:
        return "potential_bearish_choch"

    return "none"


# ── Main compute function ─────────────────────────────────────────────────────

def compute(ohlc_bars: list[dict]) -> dict:
    if len(ohlc_bars) < 50:
        return {}

    df = pd.DataFrame(ohlc_bars)
    df.rename(columns={"o": "open", "h": "high", "l": "low",
                        "c": "close", "v": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── Standard indicators ───────────────────────────────────────────────────
    rsi  = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    atr  = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]

    macd_obj    = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd        = macd_obj.macd().iloc[-1]
    macd_signal = macd_obj.macd_signal().iloc[-1]
    macd_hist   = macd_obj.macd_diff().iloc[-1]

    adx_obj  = ta.trend.ADXIndicator(high, low, close, window=14)
    adx      = adx_obj.adx().iloc[-1]
    adx_pos  = adx_obj.adx_pos().iloc[-1]   # DI+ (bullish pressure)
    adx_neg  = adx_obj.adx_neg().iloc[-1]   # DI- (bearish pressure)

    bb_obj   = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_width = bb_obj.bollinger_wband().iloc[-1]
    bb_upper = bb_obj.bollinger_hband().iloc[-1]
    bb_lower = bb_obj.bollinger_lband().iloc[-1]

    last_close = float(close.iloc[-1])

    def safe(v):
        v = float(v)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v

    atr_safe = safe(atr)

    # ── Advanced analysis ─────────────────────────────────────────────────────
    candles    = _candlestick_patterns(df, atr_safe)
    structure  = _market_structure(df)
    sr         = _support_resistance(df, atr_safe)
    fib        = _fibonacci(df)
    vpa_signal = _compute_vpa(df, atr_safe)

    # ── SMC Analysis (ICT) ───────────────────────────────────────────────────
    fvg_data     = _fvg(df, atr_safe)
    ifvg_data    = _ifvg(fvg_data, structure)
    ob_data      = _order_blocks(df, atr_safe)
    bpr_data     = _bpr(fvg_data["bullish_fvgs"], fvg_data["bearish_fvgs"])
    breaker_data = _breaker_blocks(df, ob_data["bullish_ob"], ob_data["bearish_ob"])
    eq_data      = _equal_hl(df, atr_safe)
    sweep        = _liquidity_sweep(df)
    choch        = _choch(structure, df)
    eqm          = _equilibrium(df)
    displacement = _displacement_candle(df, atr_safe)

    # ADX direction: DI+ > DI- means bulls dominate
    adx_direction = "bullish" if safe(adx_pos) > safe(adx_neg) else "bearish"

    return {
        # Standard indicators
        "rsi":            round(safe(rsi), 2),
        "atr":            round(atr_safe, 6),
        "ema20":          round(safe(ema20), 6),
        "ema50":          round(safe(ema50), 6),
        "macd":           round(safe(macd), 7),
        "macd_signal":    round(safe(macd_signal), 7),
        "macd_hist":      round(safe(macd_hist), 7),
        "adx":            round(safe(adx), 2),
        "adx_direction":  adx_direction,
        "bb_width":       round(safe(bb_width), 5),
        "bb_upper":       round(safe(bb_upper), 6),
        "bb_lower":       round(safe(bb_lower), 6),
        "price_vs_ema20": round((last_close - safe(ema20)) / safe(ema20) * 100, 4) if safe(ema20) else 0.0,
        "price_vs_ema50": round((last_close - safe(ema50)) / safe(ema50) * 100, 4) if safe(ema50) else 0.0,
        "vpa_signal":     vpa_signal,

        # Candlestick patterns (Nison)
        "candle_patterns": candles["patterns"],
        "pattern_bias":    candles["pattern_bias"],

        # Market structure (Wyckoff/ICT)
        "market_structure": structure["structure"],
        "swing_highs":      structure["swing_highs"],
        "swing_lows":       structure["swing_lows"],
        "break_of_structure": structure["bos"],
        "last_hh":          structure["last_hh"],
        "last_ll":          structure["last_ll"],

        # Support & Resistance (Al Brooks)
        "resistance":    sr["resistance"],
        "support":       sr["support"],
        "key_levels":    sr["key_levels"],

        # Fibonacci
        "fib_levels":    fib["fib_levels"],
        "nearest_fib":   fib["nearest_fib"],
        "fib_direction": fib["fib_direction"],

        # SMC (ICT)
        "fvg_bullish":       fvg_data["bullish_fvgs"],
        "fvg_bearish":       fvg_data["bearish_fvgs"],
        "ifvg_bullish":      ifvg_data["ifvg_bullish"],
        "ifvg_bearish":      ifvg_data["ifvg_bearish"],
        "ob_bullish":        ob_data["bullish_ob"],
        "ob_bearish":        ob_data["bearish_ob"],
        "equal_highs":       eq_data["equal_highs"],
        "equal_lows":        eq_data["equal_lows"],
        "liquidity_sweep":   sweep["sweep"],
        "sweep_level":       sweep["sweep_level"],
        "choch":             choch,

        # Equilibrium / Discount-Premium (TJR)
        "eq_level":          eqm["eq_level"],
        "eq_zone":           eqm["eq_zone"],
        "eq_swing_high":     eqm["swing_high"],
        "eq_swing_low":      eqm["swing_low"],

        # Breaker Blocks & BPR (ICT/TJR)
        "bpr_zones":         bpr_data,
        "bull_breaker":      breaker_data["bull_breaker"],
        "bear_breaker":      breaker_data["bear_breaker"],

        # Displacement candle (ICT — Smart Money confirmation)
        "displacement":      displacement["displacement"],
        "displacement_size": displacement["displacement_size"],
    }


def compute_daily(ohlc_bars: list[dict]) -> dict:
    """
    Lightweight analysis for Daily/HTF bars (min 3 bars required).
    Returns: bias, PDH/PDL, daily open, FVG, OB.
    """
    if len(ohlc_bars) < 3:
        return {}

    df = pd.DataFrame(ohlc_bars)
    df.rename(columns={"o": "open", "h": "high", "l": "low",
                        "c": "close", "v": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    prev = df.iloc[-2]
    today = df.iloc[-1]

    atr = float(ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=min(14, len(df))
    ).average_true_range().iloc[-1]) if len(df) >= 3 else 0.0

    def safe(v):
        v = float(v)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v

    atr = safe(atr)

    # Daily bias: is today's close above or below today's open?
    daily_bias = "bullish" if float(today["close"]) >= float(today["open"]) else "bearish"

    # Broader bias from last 5 bars
    if len(df) >= 5:
        closes = df["close"].values[-5:]
        if closes[-1] > closes[0]:
            htf_bias = "bullish"
        elif closes[-1] < closes[0]:
            htf_bias = "bearish"
        else:
            htf_bias = "neutral"
    else:
        htf_bias = daily_bias

    fvg  = _fvg(df, atr) if atr > 0 else {"bullish_fvgs": [], "bearish_fvgs": []}
    ob   = _order_blocks(df, atr) if len(df) >= 8 and atr > 0 else {"bullish_ob": None, "bearish_ob": None}
    bpr  = _bpr(fvg["bullish_fvgs"], fvg["bearish_fvgs"])
    eq   = _equal_hl(df, atr) if atr > 0 else {"equal_highs": [], "equal_lows": []}
    sweep = _liquidity_sweep(df)
    disp  = _displacement_candle(df, atr) if atr > 0 else {"displacement": "none", "displacement_size": 0.0}

    return {
        "daily_bias":    daily_bias,
        "htf_bias":      htf_bias,
        "daily_open":    round(float(today["open"]), 5),
        "daily_high":    round(float(today["high"]), 5),
        "daily_low":     round(float(today["low"]), 5),
        "pdh":           round(float(prev["high"]), 5),
        "pdl":           round(float(prev["low"]), 5),
        "fvg_bullish":   fvg["bullish_fvgs"],
        "fvg_bearish":   fvg["bearish_fvgs"],
        "ob_bullish":    ob["bullish_ob"],
        "ob_bearish":    ob["bearish_ob"],
        "bpr_zones":         bpr,
        "equal_highs":       eq["equal_highs"],
        "equal_lows":        eq["equal_lows"],
        "liquidity_sweep":   sweep["sweep"],
        "sweep_level":       sweep["sweep_level"],
        "displacement":      disp["displacement"],
        "displacement_size": disp["displacement_size"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", help="Path to JSON file with OHLC bars list")
    args = parser.parse_args()

    if args.bars:
        bars = json.loads(Path(args.bars).read_text())
    else:
        print("Provide --bars <path.json>")
        sys.exit(1)

    result = compute(bars)
    print(json.dumps(result, indent=2))
