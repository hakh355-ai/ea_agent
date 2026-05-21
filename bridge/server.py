"""
FastAPI bridge server — the single entry point for MT5 WebRequest calls.
Start with: uvicorn bridge.server:app --host 127.0.0.1 --port 5000
"""
import logging
import math
import os
import time
from typing import Optional, List

from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bridge")

app = FastAPI(title="EA AI Bridge", version="2.0")


def _find_nearest_htf_draw(current_price: float, htf_draws: dict,
                            pip_size: float, proximity_pips: float = 5.0) -> dict:
    """Return which HTF liquidity draw price is nearest to, and whether it's within proximity."""
    proximity_dist = proximity_pips * pip_size
    nearest_name, nearest_dist = None, float("inf")
    for name, level in htf_draws.items():
        if isinstance(level, (int, float)) and level > 0.0:
            dist = abs(current_price - level)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_name = name
    nearest_pips = (nearest_dist / pip_size) if pip_size > 0 else 0.0
    return {
        "key_level_hit":           nearest_dist <= proximity_dist,
        "nearest_draw":            nearest_name or "none",
        "nearest_draw_dist_pips":  round(nearest_pips, 1),
    }


# ── Request / Response models ─────────────────────────────────────────────────

class Account(BaseModel):
    balance: float
    equity: float
    open_positions: int
    daily_pnl: float = 0.0


class Tick(BaseModel):
    bid: float
    ask: float
    spread_points: int


class OpenPosition(BaseModel):
    ticket: int
    type: str
    lots: float
    open_price: float
    sl: float
    tp: float
    pnl_float: float


class MarketDataRequest(BaseModel):
    request_id: str
    symbol: str
    timestamp_utc: str
    account: Account
    ohlc: dict           # {"M5": [...], "H1": [...], "H4": [...]}
    current_tick: Tick
    open_position: Optional[OpenPosition] = None


class ConfirmRequest(BaseModel):
    request_id: str
    ticket: int
    symbol: str
    action: str
    fill_price: float
    lots: float
    sl: float = 0.0
    tp: float = 0.0


class CloseRequest(BaseModel):
    ticket: int
    symbol: str
    close_price: float
    pnl: float
    outcome: str         # "tp_hit" | "sl_hit" | "manual"


class SignalResponse(BaseModel):
    request_id: str
    action: str          # "buy" | "sell" | "hold" | "blocked"
    confidence: float = 0.0
    lot_size: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    tp1_price: float = 0.0   # partial close at 50% of TP distance
    sl_pips: float = 0.0     # SL distance in pips (EA uses this to recalculate at fill time)
    tp_pips: float = 0.0     # TP distance in pips
    reason: str = ""
    news_sentiment: float = 0.0
    risk_flags: list = []
    blocked_reason: Optional[str] = None


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    import bridge.state as state
    state.load_strategy_params()
    from bridge.self_improvement_scheduler import start_scheduler
    start_scheduler()
    from bridge.telegram_bot import start_bot, send
    start_bot()
    await send(
        f"🤖 <b>EA Bridge gestartet</b>\n"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Befehle: /status /pause /resume /risk /report"
    )
    logger.info("Bridge ready. Strategy params loaded. Scheduler running. Telegram active.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import bridge.state as state
    return {
        "status": "ok",
        "positions_open": state.get_open_position_count(),
        "timestamp": time.time(),
    }


@app.post("/signal", response_model=SignalResponse)
async def signal(req: MarketDataRequest):
    import bridge.state as state
    import bridge.risk_manager as risk
    from bridge.regime_router import detect_regime, get_regime_params
    from agents.news_agent import get_sentiment
    from agents.trading_agent import get_signal
    from tools.compute_indicators import compute, compute_daily
    from tools.trade_logger import log_signal
    from bridge.telegram_bot import is_paused

    if is_paused():
        logger.info(f"BLOCKED {req.symbol}: EA paused via Telegram")
        return SignalResponse(request_id=req.request_id, action="blocked",
                              blocked_reason="EA paused via /pause command")

    state.reset_daily_if_needed()
    params = state.get_strategy_params()
    s = state.get_state()

    # ── 0. Per-symbol loss limit (max 2 losses per symbol per day) ──────────
    state.reset_daily_if_needed()
    symbol_losses = state.get_symbol_losses_today(req.symbol)
    max_symbol_losses = int(params.get("max_losses_per_symbol_per_day", 2))
    if symbol_losses >= max_symbol_losses:
        logger.info(f"BLOCKED {req.symbol}: {symbol_losses} losses today (limit={max_symbol_losses})")
        return SignalResponse(request_id=req.request_id, action="blocked",
                              blocked_reason=f"max_losses_per_symbol ({symbol_losses} today)")

    # ── 0b. Spread check (per-symbol) ───────────────────────────────────────
    _spread_by_sym = params.get("max_spread_points_by_symbol", {})
    _max_spread = int(_spread_by_sym.get(req.symbol, params.get("max_spread_points", 30)))
    if req.current_tick.spread_points > _max_spread:
        logger.info(f"BLOCKED {req.symbol}: spread {req.current_tick.spread_points} > max {_max_spread}")
        return SignalResponse(request_id=req.request_id, action="blocked",
                              blocked_reason=f"Spread too wide: {req.current_tick.spread_points} points (max={_max_spread})")

    # ── 1. Fast risk gate (no API calls) ─────────────────────────────────────
    allowed, blocked = risk.pre_check(
        req.account.model_dump(), req.symbol,
        [],  # no news flags — news is info only, does not block
        s.open_positions, s.daily_realized_pnl, params,
        timestamp_utc=req.timestamp_utc,
    )
    if not allowed:
        logger.info(f"BLOCKED {req.symbol}: {blocked}")
        return SignalResponse(request_id=req.request_id, action="blocked",
                              blocked_reason=blocked)

    # ── 3. Compute indicators ────────────────────────────────────────────────
    m1_bars    = req.ohlc.get("M1", [])
    m5_bars    = req.ohlc.get("M5", [])
    m15_bars   = req.ohlc.get("M15", [])
    h1_bars    = req.ohlc.get("H1", [])
    h4_bars    = req.ohlc.get("H4", [])
    daily_bars = req.ohlc.get("Daily", [])

    logger.info(f"{req.symbol} bars: M1={len(m1_bars)} M5={len(m5_bars)} M15={len(m15_bars)} H1={len(h1_bars)} H4={len(h4_bars)} D={len(daily_bars)}")
    indicators = compute(m5_bars) if len(m5_bars) >= 50 else {}
    if not indicators:
        logger.warning(f"{req.symbol}: not enough bars ({len(m5_bars)}), holding")
        return SignalResponse(request_id=req.request_id, action="hold",
                              reason="Insufficient bars for indicators")

    # SMC multi-timeframe analysis
    smc = {
        "daily": compute_daily(daily_bars) if len(daily_bars) >= 3 else {},
        "h4":    compute(h4_bars)  if len(h4_bars)  >= 20 else compute_daily(h4_bars)  if len(h4_bars)  >= 3 else {},
        "h1":    compute(h1_bars)  if len(h1_bars)  >= 50 else compute_daily(h1_bars)  if len(h1_bars)  >= 3 else {},
        "m15":   compute(m15_bars) if len(m15_bars) >= 50 else compute_daily(m15_bars) if len(m15_bars) >= 3 else {},
    }

    # Fallback: wenn pivot-Erkennung scheitert (starker Trend, wenig Swings),
    # einfachen Trend aus H4/H1-Schlusskursen berechnen
    for tf, bars in (("h4", h4_bars), ("h1", h1_bars)):
        if smc[tf].get("market_structure") in ("insufficient_data", "unknown", None, ""):
            if len(bars) >= 5:
                closes = [b["c"] for b in bars[-10:]]
                if closes[-1] > closes[0] * 1.0001:
                    smc[tf]["market_structure"] = "uptrend"
                elif closes[-1] < closes[0] * 0.9999:
                    smc[tf]["market_structure"] = "downtrend"
                else:
                    smc[tf]["market_structure"] = "consolidation"

    # ── 3b. Session highs/lows + HTF draws on liquidity ─────────────────────
    from tools.compute_indicators import _session_highs_lows
    session_hl = _session_highs_lows(h1_bars, req.timestamp_utc)

    # Robust H1/H4 high/low: use swing-detected last_hh/ll, fall back to
    # rolling bar max/min if swing detection returned 0.0 (insufficient pivots).
    def _bar_key(b, k1, k2, default=0.0):
        return b.get(k1, b.get(k2, default))

    h1_high = smc["h1"].get("last_hh", 0.0)
    h1_low  = smc["h1"].get("last_ll", 0.0)
    if h1_high == 0.0 and len(h1_bars) >= 5:
        h1_high = max(_bar_key(b, "h", "high", 0.0) for b in h1_bars[-20:])
    if h1_low == 0.0 and len(h1_bars) >= 5:
        h1_low = min(_bar_key(b, "l", "low", float("inf")) for b in h1_bars[-20:])
        if h1_low == float("inf"):
            h1_low = 0.0

    h4_high = smc["h4"].get("last_hh", 0.0)
    h4_low  = smc["h4"].get("last_ll", 0.0)
    if h4_high == 0.0 and len(h4_bars) >= 5:
        h4_high = max(_bar_key(b, "h", "high", 0.0) for b in h4_bars[-20:])
    if h4_low == 0.0 and len(h4_bars) >= 5:
        h4_low = min(_bar_key(b, "l", "low", float("inf")) for b in h4_bars[-20:])
        if h4_low == float("inf"):
            h4_low = 0.0

    # All draws shown to KI as context (H4 included)
    htf_draws = {
        "h1_high": h1_high,
        "h1_low":  h1_low,
        "h4_high": h4_high,
        "h4_low":  h4_low,
        "pdh":     smc["daily"].get("pdh", 0.0),
        "pdl":     smc["daily"].get("pdl", 0.0),
        "asia_high":   session_hl["asia_high"],
        "asia_low":    session_hl["asia_low"],
        "london_high": session_hl["london_high"],
        "london_low":  session_hl["london_low"],
        "ny_high":     session_hl["ny_high"],
        "ny_low":      session_hl["ny_low"],
    }

    # Direction draws: H4 is data only — only H1, PDH/PDL, session H/L set direction
    direction_draws = {
        "h1_high": htf_draws["h1_high"],
        "h1_low":  htf_draws["h1_low"],
        "pdh":     htf_draws["pdh"],
        "pdl":     htf_draws["pdl"],
        "asia_high":   htf_draws["asia_high"],
        "asia_low":    htf_draws["asia_low"],
        "london_high": htf_draws["london_high"],
        "london_low":  htf_draws["london_low"],
        "ny_high":     htf_draws["ny_high"],
        "ny_low":      htf_draws["ny_low"],
    }

    pip_size = float(params.get("pip_sizes", {}).get(req.symbol, 0.0001))
    _prox_by_sym = params.get("key_level_proximity_pips_by_symbol", {})
    proximity_pips = float(_prox_by_sym.get(req.symbol, params.get("key_level_proximity_pips", 5.0)))
    current_mid = (req.current_tick.bid + req.current_tick.ask) / 2

    key_level_info = _find_nearest_htf_draw(current_mid, direction_draws, pip_size, proximity_pips)
    logger.info(
        f"{req.symbol} key_level_hit={key_level_info['key_level_hit']} "
        f"nearest={key_level_info['nearest_draw']} ({key_level_info['nearest_draw_dist_pips']}p) "
        f"threshold={proximity_pips}p"
    )

    # Early exit: no key level hit — skip all API calls
    if not key_level_info["key_level_hit"]:
        _nearest = key_level_info["nearest_draw"]
        _dist    = key_level_info["nearest_draw_dist_pips"]
        logger.info(f"{req.symbol} HOLD (pre-KI): no HTF key level — nearest={_nearest} ({_dist}p)")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=0.0,
                              reason=f"No HTF key level hit (nearest={_nearest}, {_dist}p away) — wait for level")

    # ── 2. News sentiment (info only for KI — no trade blocking) ───────────────
    news = await get_sentiment([req.symbol])

    # ── 4. Regime detection (Stufe 3) ────────────────────────────────────────
    regime = detect_regime(indicators)
    regime_params = get_regime_params(regime, params)
    logger.info(
        f"{req.symbol} | regime={regime} adx={indicators.get('adx')} "
        f"bb_w={indicators.get('bb_width')} sentiment={news.get('sentiment_score', 0):.2f}"
    )

    # ── 4b. SMT Divergence (EURUSD↔GBPUSD correlation) ──────────────────────
    state.update_symbol_structure(
        req.symbol,
        indicators.get("market_structure", "unknown"),
        indicators.get("break_of_structure", "none"),
        indicators.get("last_hh", 0.0),
        indicators.get("last_ll", 0.0),
    )
    smt_signal = state.get_smt_signal(req.symbol)
    if smt_signal != "none":
        logger.info(f"{req.symbol} SMT signal: {smt_signal}")

    # ── 5. Trading agent ─────────────────────────────────────────────────────
    sig = await get_signal(req.symbol, req.ohlc, indicators, news,
                           regime, regime_params, params, smc=smc,
                           smt_signal=smt_signal, m1_bars=m1_bars,
                           htf_draws=htf_draws, key_level_info=key_level_info)
    action = sig.get("action", "hold")
    confidence = float(sig.get("confidence", 0.0))

    # Boost confidence slightly when displacement candle aligns with trade direction
    m5_disp = indicators.get("displacement", "none")
    if (action == "buy" and m5_disp == "bullish") or (action == "sell" and m5_disp == "bearish"):
        confidence = min(0.95, confidence + 0.03)
        logger.info(f"{req.symbol} displacement {m5_disp} aligned → conf boosted to {confidence:.2f}")

    if action == "hold":
        logger.info(f"{req.symbol} HOLD | conf={confidence:.2f} reason: {sig.get('reasoning', '')[:80]}")
        return SignalResponse(
            request_id=req.request_id, action="hold", confidence=confidence,
            reason=sig.get("reasoning", ""),
            news_sentiment=news.get("sentiment_score", 0.0),
            risk_flags=news.get("risk_flags", []),
        )

    # ── 5b. Key level direction filter ──────────────────────────────────────────
    # Direction comes from WHICH type of HTF level was hit:
    #   hit a HIGH (h1_high, h4_high, asia_high, london_high, ny_high, pdh) → bearish → only SELL
    #   hit a LOW  (h1_low,  h4_low,  asia_low,  london_low,  ny_low,  pdl) → bullish → only BUY
    nearest_draw = key_level_info.get("nearest_draw", "none")

    _is_high_level = any(nearest_draw.endswith(s) for s in ("_high", "pdh"))
    _is_low_level  = any(nearest_draw.endswith(s) for s in ("_low",  "pdl"))

    if _is_high_level and action == "buy":
        logger.info(f"{req.symbol} BLOCKED: buy at HTF high ({nearest_draw}) — level hit = bearish bias")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason=f"HTF high hit ({nearest_draw}) — bearish bias, only sells allowed",
                              news_sentiment=news.get("sentiment_score", 0.0))

    if _is_low_level and action == "sell":
        logger.info(f"{req.symbol} BLOCKED: sell at HTF low ({nearest_draw}) — level hit = bullish bias")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason=f"HTF low hit ({nearest_draw}) — bullish bias, only buys allowed",
                              news_sentiment=news.get("sentiment_score", 0.0))

    # M5 structure logged as context only — M5 is highest priority via IFVG/BOS checks below
    m5_struct = indicators.get("market_structure", "unknown")
    logger.info(f"{req.symbol} M5 structure: {m5_struct} (context only)")

    # ── 5b3. STEP 3: LTF Reversal signal required (BOS or IFVG in bias direction) ──
    m5_bos       = indicators.get("break_of_structure", "none")
    m5_ifvg_bull = indicators.get("ifvg_bullish", [])
    m5_ifvg_bear = indicators.get("ifvg_bearish", [])

    has_bull_reversal = (m5_bos == "bullish_bos") or bool(m5_ifvg_bull)
    has_bear_reversal = (m5_bos == "bearish_bos") or bool(m5_ifvg_bear)

    if action == "buy" and not has_bull_reversal:
        logger.info(f"{req.symbol} BLOCKED step3: no bullish BOS or IFVG on M5 — wait for reversal signal")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason="Step 3 failed: no bullish reversal signal on M5 (need BOS or IFVG)",
                              news_sentiment=news.get("sentiment_score", 0.0))

    if action == "sell" and not has_bear_reversal:
        logger.info(f"{req.symbol} BLOCKED step3: no bearish BOS or IFVG on M5 — wait for reversal signal")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason="Step 3 failed: no bearish reversal signal on M5 (need BOS or IFVG)",
                              news_sentiment=news.get("sentiment_score", 0.0))

    # ── 5b4. STEP 4: LTF Continuation required (FVG or OB in bias direction) ────
    m5_fvg_bull  = indicators.get("fvg_bullish", [])
    m5_fvg_bear  = indicators.get("fvg_bearish", [])
    m5_ob_bull   = indicators.get("ob_bullish")
    m5_ob_bear   = indicators.get("ob_bearish")
    m5_bull_brk  = indicators.get("bull_breaker")
    m5_bear_brk  = indicators.get("bear_breaker")
    m5_eq_zone   = indicators.get("eq_zone", "unknown")

    has_bull_continuation = bool(m5_fvg_bull) or bool(m5_ob_bull) or bool(m5_bull_brk) or m5_eq_zone == "discount"
    has_bear_continuation = bool(m5_fvg_bear) or bool(m5_ob_bear) or bool(m5_bear_brk) or m5_eq_zone == "premium"

    if action == "buy" and not has_bull_continuation:
        logger.info(f"{req.symbol} BLOCKED step4: no bullish continuation on M5 (need FVG/OB/Breaker/discount)")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason="Step 4 failed: no bullish continuation on M5 (need FVG, OB, Breaker, or discount EQ)",
                              news_sentiment=news.get("sentiment_score", 0.0))

    if action == "sell" and not has_bear_continuation:
        logger.info(f"{req.symbol} BLOCKED step4: no bearish continuation on M5 (need FVG/OB/Breaker/premium)")
        return SignalResponse(request_id=req.request_id, action="hold", confidence=confidence,
                              reason="Step 4 failed: no bearish continuation on M5 (need FVG, OB, Breaker, or premium EQ)",
                              news_sentiment=news.get("sentiment_score", 0.0))

    # ── 5c. Multi-Timeframe Confirmation ────────────────────────────────────
    # Require H1 and H4 trend to agree with the signal direction.
    # Simple check: recent closes vs older closes on each timeframe.
    def _mtf_bullish(bars: list) -> Optional[bool]:
        if len(bars) < 10:
            return None
        closes = [b.get("c", b.get("close", 0)) for b in bars]
        recent = sum(closes[-5:]) / 5
        older  = sum(closes[:5]) / 5
        if recent > older * 1.0003:  return True
        if recent < older * 0.9997:  return False
        return None   # neutral

    h1_bull = _mtf_bullish(req.ohlc.get("H1", []))
    # H1 direction is priority info only — does not block trades
    # bullish H1 = BUY preferred, bearish H1 = SELL preferred
    if h1_bull is True:
        logger.info(f"{req.symbol} H1 priority: bullish (BUY preferred)")
    elif h1_bull is False:
        logger.info(f"{req.symbol} H1 priority: bearish (SELL preferred)")
    else:
        logger.info(f"{req.symbol} H1 priority: neutral")

    # ── 5c. Sweep direction filter — only trade post-sweep direction ────────
    # If Smart Money has just swept liquidity, only allow the reversal trade.
    # Trading INTO a sweep = trading against Smart Money = blocked.
    m5_sweep  = indicators.get("liquidity_sweep", "none")
    m15_sweep = smc.get("m15", {}).get("liquidity_sweep", "none")
    active_sweep = m5_sweep if m5_sweep != "none" else m15_sweep

    if active_sweep == "bullish" and action == "sell":
        logger.info(f"{req.symbol} sweep filter: bullish sweep → sell blocked (Smart Money buying)")
        return SignalResponse(request_id=req.request_id, action="hold",
                              confidence=confidence,
                              reason="Bullish sweep active — Smart Money buying, sell blocked",
                              news_sentiment=news.get("sentiment_score", 0.0))
    if active_sweep == "bearish" and action == "buy":
        logger.info(f"{req.symbol} sweep filter: bearish sweep → buy blocked (Smart Money selling)")
        return SignalResponse(request_id=req.request_id, action="hold",
                              confidence=confidence,
                              reason="Bearish sweep active — Smart Money selling, buy blocked",
                              news_sentiment=news.get("sentiment_score", 0.0))

    # ── 5e. Correlation filter (post-signal) ────────────────────────────────
    corr_ok, corr_reason = risk._check_correlation(req.symbol, action, s.open_positions)
    if not corr_ok:
        logger.info(f"BLOCKED {req.symbol}: {corr_reason}")
        return SignalResponse(request_id=req.request_id, action="blocked",
                              blocked_reason=corr_reason)

    # ── 6. Position sizing ───────────────────────────────────────────────────
    current_price = req.current_tick.bid if action == "sell" else req.current_tick.ask
    tp_sl_ratio = float(regime_params["tp_sl_ratio"])

    fixed_sl = float(params.get("fixed_sl_pips", 0))
    fixed_tp = float(params.get("fixed_tp_pips", 0))
    fixed_lot = float(params.get("fixed_lot_size", 0))

    if fixed_sl > 0 and fixed_tp > 0:
        sl_dist = fixed_sl * pip_size
        tp_dist = fixed_tp * pip_size
        if action == "buy":
            sl_price = round(current_price - sl_dist, 6)
            tp_price = round(current_price + tp_dist, 6)
        else:
            sl_price = round(current_price + sl_dist, 6)
            tp_price = round(current_price - tp_dist, 6)
        sl_pips = fixed_sl
        tp_pips_val = fixed_tp
    else:
        # Try OB distal line as structural SL (S/D zone: SL outside the zone)
        ob_bull = indicators.get("ob_bullish")
        ob_bear = indicators.get("ob_bearish")
        sl_buffer = 2.0 * pip_size   # 2 pips outside distal line to avoid stop-hunts
        atr_sl_pips = indicators["atr"] * float(regime_params["sl_atr_multiplier"]) / pip_size
        sl_from_ob = False

        if action == "buy" and ob_bull and ob_bull.get("bottom", 0) > 0:
            ob_distal = float(ob_bull["bottom"])
            if ob_distal < current_price:                       # distal below entry = valid
                candidate_sl = round(ob_distal - sl_buffer, 6)
                candidate_pips = (current_price - candidate_sl) / pip_size
                if 0 < atr_sl_pips and candidate_pips <= atr_sl_pips * 2.5:  # cap: 2.5× ATR
                    sl_price    = candidate_sl
                    sl_dist     = current_price - sl_price
                    tp_dist     = sl_dist * tp_sl_ratio
                    tp_price    = round(current_price + tp_dist, 6)
                    sl_pips     = round(sl_dist / pip_size, 1)
                    tp_pips_val = round(tp_dist / pip_size, 1)
                    sl_from_ob  = True

        elif action == "sell" and ob_bear and ob_bear.get("top", 0) > 0:
            ob_distal = float(ob_bear["top"])
            if ob_distal > current_price:                       # distal above entry = valid
                candidate_sl = round(ob_distal + sl_buffer, 6)
                candidate_pips = (candidate_sl - current_price) / pip_size
                if 0 < atr_sl_pips and candidate_pips <= atr_sl_pips * 2.5:  # cap: 2.5× ATR
                    sl_price    = candidate_sl
                    sl_dist     = sl_price - current_price
                    tp_dist     = sl_dist * tp_sl_ratio
                    tp_price    = round(current_price - tp_dist, 6)
                    sl_pips     = round(sl_dist / pip_size, 1)
                    tp_pips_val = round(tp_dist / pip_size, 1)
                    sl_from_ob  = True

        if not sl_from_ob:
            sl_price, tp_price = risk.calc_sl_tp(
                atr=indicators["atr"],
                atr_multiplier=float(regime_params["sl_atr_multiplier"]),
                tp_sl_ratio=tp_sl_ratio,
                action=action,
                current_price=current_price,
            )
            sl_pips     = round(abs(current_price - sl_price) / pip_size, 1)
            tp_pips_val = round(abs(current_price - tp_price) / pip_size, 1)

        logger.info(f"{req.symbol} SL: {'OB distal line' if sl_from_ob else 'ATR-based'} | sl_pips={sl_pips}")

        # Enforce minimum SL on the actual price
        min_sl_p = float(params.get("min_sl_pips", {}).get(req.symbol, 3))
        if sl_pips < min_sl_p:
            sl_dist     = min_sl_p * pip_size
            tp_dist     = sl_dist * tp_sl_ratio
            if action == "buy":
                sl_price = round(current_price - sl_dist, 6)
                tp_price = round(current_price + tp_dist, 6)
            else:
                sl_price = round(current_price + sl_dist, 6)
                tp_price = round(current_price - tp_dist, 6)
            sl_pips     = min_sl_p
            tp_pips_val = round(tp_dist / pip_size, 1)

    # TP1: partial close at 50% of TP distance (for Multiple TPs in EA)
    if action == "buy":
        tp1_price = round(current_price + (tp_price - current_price) * 0.5, 6)
    else:
        tp1_price = round(current_price - (current_price - tp_price) * 0.5, 6)

    if fixed_lot > 0:
        lot_size = fixed_lot
        if s.consecutive_losses >= 3:
            lot_size = max(0.01, round(lot_size * 0.5, 2))
    else:
        pip_value = float(params.get("pip_values", {}).get(req.symbol, 10.0))
        risk_pct = float(params.get("default_risk_pct", float(os.getenv("DEFAULT_RISK_PCT", 0.02))))
        lot_size = risk.calc_lot_size(
            req.account.balance, sl_pips, risk_pct, pip_value, req.symbol, params,
            consecutive_losses=s.consecutive_losses,
        )
    max_lot = float(params.get("max_lot_size", 1.0))
    if lot_size > max_lot:
        logger.warning(f"{req.symbol}: lot {lot_size} capped to max_lot_size {max_lot}")
        lot_size = max_lot
    if s.consecutive_losses >= 3:
        logger.info(f"{req.symbol}: losing streak={s.consecutive_losses} → lot reduced to {lot_size}")

    # ── 7. Log ───────────────────────────────────────────────────────────────
    log_signal(req.symbol, action, confidence, indicators, news,
               regime, lot_size, sl_price, tp_price, req.request_id)

    logger.info(
        f"{req.symbol} → {action.upper()} | conf={confidence:.2f} lots={lot_size} "
        f"sl={sl_price} tp={tp_price} regime={regime}"
    )

    # Atomically reserve symbol — if another request already reserved it, return hold
    reserved = await state.reserve_symbol(req.symbol, action)
    if not reserved:
        logger.info(f"{req.symbol}: symbol already reserved (concurrent request) → hold")
        return SignalResponse(request_id=req.request_id, action="hold",
                              confidence=confidence, reason="symbol reserved by concurrent request")

    # Store signal data so /confirm can send accurate Telegram notification
    state.store_pending_signal(
        req.request_id, req.symbol, confidence, regime,
        sl_price, tp_price, sl_pips, tp_pips_val
    )

    return SignalResponse(
        request_id=req.request_id,
        action=action,
        confidence=confidence,
        lot_size=lot_size,
        sl_price=sl_price,
        tp_price=tp_price,
        tp1_price=tp1_price,
        sl_pips=sl_pips,
        tp_pips=tp_pips_val,
        reason=sig.get("reasoning", ""),
        news_sentiment=news.get("sentiment_score", 0.0),
        risk_flags=news.get("risk_flags", []),
    )


@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    import bridge.state as state
    from tools.trade_logger import log_fill
    from bridge.telegram_bot import notify_fill

    # Retrieve stored signal data (confidence, regime, sl, tp, pips)
    sig = state.get_pending_signal(req.request_id)

    sl = req.sl if req.sl else sig.get("sl_price", 0.0)
    tp = req.tp if req.tp else sig.get("tp_price", 0.0)
    state.record_fill(
        req.ticket, req.symbol, req.action, req.fill_price, req.lots,
        sl=sl, tp=tp,
        confidence=sig.get("confidence", 0.0), regime=sig.get("regime", ""),
    )
    log_fill(req.ticket, req.symbol, req.action, req.fill_price, req.lots, req.request_id)
    logger.info(f"Fill confirmed: {req.action.upper()} {req.symbol} "
                f"ticket={req.ticket} @ {req.fill_price} lots={req.lots} "
                f"conf={sig.get('confidence', 0):.0%} regime={sig.get('regime', '?')}")

    await notify_fill(
        req.symbol, req.action, req.lots, req.fill_price,
        sig.get("confidence", 0.0), sig.get("regime", "unknown"),
        sig.get("sl_price", 0.0), sig.get("tp_price", 0.0),
        sig.get("sl_pips", 0.0), sig.get("tp_pips", 0.0),
    )
    return {"status": "ok"}


@app.post("/trade_close")
async def close_trade(req: CloseRequest):
    import bridge.state as state
    from tools.trade_logger import log_close
    from bridge.telegram_bot import notify_close, notify_drawdown_warning
    state.record_close_pnl(req.pnl, symbol=req.symbol)
    if req.symbol in state.get_state().open_positions:
        del state.get_state().open_positions[req.symbol]
    log_close(req.ticket, req.symbol, req.close_price, req.pnl, req.outcome)
    logger.info(f"Close: {req.symbol} ticket={req.ticket} pnl={req.pnl:.2f} outcome={req.outcome}")
    await notify_close(req.symbol, req.pnl, req.outcome, req.close_price)
    # Drawdown warning at 3%
    s = state.get_state()
    balance = 10000.0  # approximate; real balance comes from next MT5 tick
    if balance > 0 and abs(s.daily_realized_pnl) / balance >= 0.03:
        await notify_drawdown_warning(abs(s.daily_realized_pnl) / balance, balance)
    return {"status": "ok"}


@app.post("/sync_positions")
async def sync_positions(data: dict):
    """
    Called by AI_EA on OnInit() — sends all currently open broker positions.
    Bridge overwrites its in-memory state to match reality after any restart.
    """
    import bridge.state as state
    broker_positions = data.get("positions", {})
    state.sync_positions(broker_positions)
    logger.info(f"Position sync received: {len(broker_positions)} open position(s) → {list(broker_positions.keys())}")
    return {"status": "ok", "synced": len(broker_positions)}


@app.post("/reload_params")
async def reload_params():
    import bridge.state as state
    state.reload_strategy_params()
    logger.info("Strategy params reloaded from disk.")
    return {"status": "ok"}
