"""
Trading Agent (Claude Sonnet 4.6).
Receives OHLC bars, indicators, news sentiment, and regime params.
Returns a buy/sell/hold decision with confidence, SL pips, TP pips.
"""
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Module-level singleton — reuses the underlying httpx connection pool
_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _ohlc_table(bars: list[dict], limit: int = 20) -> str:
    rows = bars[-limit:]
    lines = ["Time             | Open     | High     | Low      | Close    | Vol"]
    lines.append("-" * 68)
    for b in rows:
        t = str(b.get("t", "?"))[:16]
        lines.append(
            f"{t:<16} | {b.get('o',0):>8.5f} | {b.get('h',0):>8.5f} | "
            f"{b.get('l',0):>8.5f} | {b.get('c',0):>8.5f} | {int(b.get('v',0))}"
        )
    return "\n".join(lines)


def _build_prompt(symbol: str, ohlc: dict, indicators: dict, news: dict,
                  regime: str, regime_params: dict, learned_rules: list[str],
                  smc: dict = None, smt_signal: str = "none",
                  m1_bars: list = None,
                  htf_draws: dict = None, key_level_info: dict = None) -> str:
    """ICT 6-step daily bias methodology: HTF draws → key level hit → LTF reversal → LTF continuation → entry"""
    m1_data = m1_bars or ohlc.get("M1", [])
    m1  = _ohlc_table(m1_data, 15)

    # M1 momentum: direction of last 15 candles
    m1_momentum = "unknown"
    if len(m1_data) >= 5:
        closes = [b.get("c", b.get("close", 0)) for b in m1_data[-15:]]
        recent = sum(closes[-5:]) / 5
        older  = sum(closes[:5]) / 5
        if recent > older * 1.0002:
            m1_momentum = "bullish"
        elif recent < older * 0.9998:
            m1_momentum = "bearish"
        else:
            m1_momentum = "neutral"

    m5 = _ohlc_table(ohlc.get("M5", []), 15)

    rules       = "\n".join(f"  - {r}" for r in learned_rules) if learned_rules else "  None yet."
    patterns    = indicators.get("candle_patterns", [])
    pattern_str = ", ".join(patterns) if patterns else "none"
    smc         = smc or {}
    draws       = htf_draws or {}
    kl          = key_level_info or {}

    # Key level info
    kl_hit       = kl.get("key_level_hit", False)
    kl_nearest   = kl.get("nearest_draw", "none")
    kl_dist      = kl.get("nearest_draw_dist_pips", 999)

    # H4 context
    h4           = smc.get("h4", {})
    h4_struct    = h4.get("market_structure") or h4.get("htf_bias") or h4.get("daily_bias") or "unknown"
    h4_bos       = h4.get("break_of_structure", "none")
    h4_choch     = h4.get("choch", "none")
    h4_eq_zone   = h4.get("eq_zone", "unknown")

    # H1 context
    h1           = smc.get("h1", {})
    h1_struct    = h1.get("market_structure", "unknown")
    h1_bos       = h1.get("break_of_structure", "none")
    h1_choch     = h1.get("choch", "none")
    h1_eq_zone   = h1.get("eq_zone", "unknown")
    h1_eq_level  = h1.get("eq_level", 0)

    # Daily context
    daily        = smc.get("daily", {})
    d_ob_bull    = daily.get("ob_bullish")
    d_ob_bear    = daily.get("ob_bearish")
    d_fvg_bull   = daily.get("fvg_bullish", [])
    d_fvg_bear   = daily.get("fvg_bearish", [])

    # M5 SMC (entry)
    m5_fvg_bull  = indicators.get("fvg_bullish", [])
    m5_fvg_bear  = indicators.get("fvg_bearish", [])
    m5_ifvg_bull = indicators.get("ifvg_bullish", [])
    m5_ifvg_bear = indicators.get("ifvg_bearish", [])
    m5_ob_bull   = indicators.get("ob_bullish")   # {"top": X, "bottom": Y} — distal=bottom for buy
    m5_ob_bear   = indicators.get("ob_bearish")   # {"top": X, "bottom": Y} — distal=top for sell
    m5_sweep     = indicators.get("liquidity_sweep", "none")
    m5_choch     = indicators.get("choch", "none")
    m5_bos       = indicators.get("break_of_structure", "none")
    m5_bull_brk  = indicators.get("bull_breaker")
    m5_bear_brk  = indicators.get("bear_breaker")
    m5_bpr       = indicators.get("bpr_zones", [])
    m5_disp      = indicators.get("displacement", "none")
    m5_disp_size = indicators.get("displacement_size", 0.0)
    m5_eq_level  = indicators.get("eq_level", 0)
    m5_eq_zone   = indicators.get("eq_zone", "unknown")
    m5_eq_high   = indicators.get("eq_swing_high", 0)
    m5_eq_low    = indicators.get("eq_swing_low", 0)

    # Fibonacci (incl. 0.79)
    fib_levels   = indicators.get("fib_levels", {})
    fib_079      = fib_levels.get("0.790", 0.0)
    fib_dir      = indicators.get("fib_direction", "unknown")
    nearest_fib  = indicators.get("nearest_fib", 0.0)

    m5_last = ohlc.get("M5", [{}])[-1] if ohlc.get("M5") else {}
    current_price = float(m5_last.get("c", m5_last.get("close", 0)))

    return f"""=== ICT DAILY BIAS METHODOLOGY: {symbol} ===
Current Price : {current_price}
Market Regime : {regime}  (ADX={indicators.get('adx','?')} [{indicators.get('adx_direction','?')}])
Min Confidence: {regime_params.get('min_confidence_threshold', 0.5)}
News          : sentiment={news.get('sentiment_score', 0.0)}  flags={', '.join(news.get('risk_flags', [])) or 'none'}
SMT Signal    : {smt_signal}

━━━ STEP 1: HTF DRAWS ON LIQUIDITY ━━━
These are the price MAGNETS. Price will eventually be drawn to every one of these levels.
H4 High (swing): {draws.get('h4_high') or 'N/A'}     H4 Low: {draws.get('h4_low') or 'N/A'}
H1 High (swing): {draws.get('h1_high') or 'N/A'}     H1 Low: {draws.get('h1_low') or 'N/A'}
PDH / PDL      : {draws.get('pdh') or 'N/A'} / {draws.get('pdl') or 'N/A'}
Asia H/L       : {draws.get('asia_high') or 'N/A'} / {draws.get('asia_low') or 'N/A'}
London H/L     : {draws.get('london_high') or 'N/A'} / {draws.get('london_low') or 'N/A'}
NY H/L         : {draws.get('ny_high') or 'N/A'} / {draws.get('ny_low') or 'N/A'}
H4 Structure   : {h4_struct.upper()}  BOS: {h4_bos}  CHoCH: {h4_choch}  ← DATA ONLY (context)
H4 EQ Zone     : {h4_eq_zone.upper()}  ← DISCOUNT=buy side, PREMIUM=sell side
H1 Priority    : {h1_struct.upper()}  BOS: {h1_bos}  ← PRIORITY (bullish=BUY preferred, bearish=SELL preferred, both directions allowed)
Daily OB       : bull={d_ob_bull or 'none'}  bear={d_ob_bear or 'none'}
Daily FVG      : bull={d_fvg_bull or 'none'}  bear={d_fvg_bear or 'none'}

━━━ STEP 2: IS PRICE AT A KEY LEVEL? ━━━
KEY LEVEL HIT  : {'YES ✓' if kl_hit else 'NO — price is NOT at a HTF draw'}
Nearest Draw   : {kl_nearest}  ({kl_dist} pips away)
→ If NO key level hit → output HOLD. Do NOT trade between levels. Patience is the strategy.
→ Only proceed to Step 3 if KEY LEVEL HIT = YES.

━━━ STEP 3: LTF REVERSAL SIGNALS (M1/M5) ━━━
Look for at least ONE of these three reversal confluences AT the key level:
  [1] IFVG (FVG forming AGAINST current structure = reversal signal):
      M5 IFVG Bull : {m5_ifvg_bull or 'none'}  ← bullish reversal (in downtrend = price reversing up)
      M5 IFVG Bear : {m5_ifvg_bear or 'none'}  ← bearish reversal (in uptrend = price reversing down)
  [2] BOS / CHoCH on M5 (structure breaking at the key level):
      M5 BOS       : {m5_bos}
      M5 CHoCH     : {m5_choch}
      M1 Momentum  : {m1_momentum.upper()}  ← direction of last 15 minutes
  [3] 79% Fibonacci Extension (price near the 0.79 fib = deep retracement = reversal zone):
      Fib Direction: {fib_dir}  |  0.79 Level: {fib_079}  |  Nearest Fib: {nearest_fib}
      → Price within 3 pips of 0.790 = strong reversal confluence

━━━ STEP 4: LTF CONTINUATION CONFLUENCE ━━━
After reversal signal, wait for price to SHOW continuation in the new direction:
H1 Structure   : {h1_struct}  BOS: {h1_bos}  CHoCH: {h1_choch}
H1 EQ Zone     : {h1_eq_zone.upper()} at {h1_eq_level}
M5 FVG Bull    : {m5_fvg_bull or 'none'}     M5 FVG Bear : {m5_fvg_bear or 'none'}
M5 OB Bull     : {m5_ob_bull or 'none'}      M5 OB Bear  : {m5_ob_bear or 'none'}
Bull Breaker   : {m5_bull_brk or 'none'}     Bear Breaker: {m5_bear_brk or 'none'}
M5 BPR Zones   : {m5_bpr or 'none'}
M5 EQ Zone     : {m5_eq_zone.upper()} at {m5_eq_level}  (range: {m5_eq_low}→{m5_eq_high})
Displacement   : {m5_disp} ({m5_disp_size}× ATR)
Sweep          : {m5_sweep}
Pattern        : {pattern_str}  [{indicators.get('pattern_bias','neutral')}]
RSI / MACD     : {indicators.get('rsi','?')} / hist={indicators.get('macd_hist','?')}

=== M5 Bars (last 15) ===
{m5}

=== M1 Bars (last 15) ===
{m1}

━━━ STEP 5: ENTRY RULES (ALL must be satisfied) ━━━
  A. KEY LEVEL HIT = YES — direction is set by WHICH level was hit:
     → HTF LOW hit  (h1_low, h4_low, asia_low, london_low, ny_low, pdl) = BULLISH bias → BUY only
     → HTF HIGH hit (h1_high, h4_high, asia_high, london_high, ny_high, pdh) = BEARISH bias → SELL only
     → No key level hit = price in middle of range = NO TRADE → HOLD
  B. At least 1 reversal confluence from Step 3 (IFVG OR BOS/CHoCH OR 0.79 fib)
  C. At least 1 continuation confluence from Step 4 (FVG, OB, Breaker, BPR, or Equilibrium)
  D. M1 momentum not opposing (not bearish for buy, not bullish for sell)

BUY  : HTF LOW hit → bullish bias → M5 was bearish INTO the low → bullish reversal (IFVG/BOS breaking the downswing) → LTF continuation (FVG/OB/EQ) → enter
SELL : HTF HIGH hit → bearish bias → M5 was bullish INTO the high → bearish reversal (IFVG/BOS breaking the upswing) → LTF continuation (FVG/OB/EQ) → enter
HOLD : No key level hit OR no reversal signal OR no continuation confluence

Confidence scoring:
  0.50 base if Key Level Hit + 1 reversal signal + 1 continuation signal
  +0.05 for each additional confluence (IFVG, BOS/CHoCH, 0.79 fib, FVG, OB, Breaker, BPR)
  +0.05 for displacement candle confirming direction
  +0.05 for SMT divergence alignment ({smt_signal})
  +0.05 for H4 EQ zone alignment (discount for buy, premium for sell)
  max 0.95

━━━ STEP 6: SL / TP PLACEMENT ━━━
SL = Order Block DISTAL LINE (outside the zone) — NOT ATR-based:
  If BUY  → SL below demand OB bottom (M5 OB bull bottom = {m5_ob_bull.get('bottom', 'N/A') if m5_ob_bull else 'N/A'})
  If SELL → SL above supply OB top    (M5 OB bear top    = {m5_ob_bear.get('top', 'N/A') if m5_ob_bear else 'N/A'})
TP = opposite HTF draw on liquidity (price magnet):
  If BUY  → TP toward H1/H4 high, PDH, London/NY session high
  If SELL → TP toward H1/H4 low,  PDL, London/NY session low
Set sl_pips to the distance from entry to the OB distal line (bridge will use OB-based SL automatically).

=== Learned Rules ===
{rules}

Respond with ONLY this JSON:
{{
  "action": "buy" | "sell" | "hold",
  "confidence": <0.0-1.0>,
  "sl_pips": <positive integer>,
  "tp_pips": <positive integer>,
  "reasoning": "<HTF draw hit> + <reversal signal> + <continuation confluence> + <direction>"
}}"""


async def get_signal(symbol: str, ohlc: dict, indicators: dict, news: dict,
                     regime: str, regime_params: dict, strategy_params: dict,
                     smc: dict = None, smt_signal: str = "none",
                     m1_bars: list = None,
                     htf_draws: dict = None, key_level_info: dict = None) -> dict:
    # Validate price data before calling LLM
    m5 = ohlc.get("M5", [])
    if not m5 or float(m5[-1].get("c", m5[-1].get("close", 0))) == 0:
        return {"action": "hold", "confidence": 0.0, "sl_pips": 30,
                "tp_pips": 60, "reasoning": "No valid M5 price data"}

    learned_rules = strategy_params.get("learned_rules", [])
    system_prompt = strategy_params.get(
        "trading_prompt",
        "You are an expert ICT/SMC trader. Follow the 6-step daily bias methodology exactly. Respond with precise JSON only."
    )

    user_prompt = _build_prompt(symbol, ohlc, indicators, news, regime, regime_params,
                                learned_rules, smc, smt_signal, m1_bars=m1_bars,
                                htf_draws=htf_draws, key_level_info=key_level_info)

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0.2,
            system=[{"type": "text", "text": system_prompt,
                      "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            raw = m.group(1).strip() if m else re.sub(r"^json\s*", "", raw.split("```")[1].strip())

        result = json.loads(raw)

    except Exception as e:
        print(f"[trading_agent] Claude error: {e}", file=sys.stderr)
        return {"action": "hold", "confidence": 0.0, "sl_pips": 30,
                "tp_pips": 60, "reasoning": f"Agent error: {e}"}

    # Enforce minimum confidence threshold for regime
    min_conf = float(regime_params.get("min_confidence_threshold", 0.65))
    if float(result.get("confidence", 0)) < min_conf:
        result["action"] = "hold"
        result["reasoning"] = (
            f"Confidence {result.get('confidence', 0):.2f} below "
            f"regime threshold {min_conf}"
        )

    return result
