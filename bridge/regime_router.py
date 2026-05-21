"""
Market regime detection (Stufe 3).
Classifies market as trending / ranging / volatile / quiet
based on ADX and Bollinger Band width, then returns regime-specific strategy params.
"""


def detect_regime(indicators: dict) -> str:
    """
    trending  → ADX > 25
    volatile  → ADX > 25 AND bb_width > 0.04 (strong trend with wide bands)
    ranging   → ADX < 20 AND bb_width between 0.01–0.05
    quiet     → ADX < 20 AND bb_width < 0.01 (compression / low volatility)
    """
    adx = float(indicators.get("adx", 20))
    bb_width = float(indicators.get("bb_width", 0.02))

    if adx > 25:
        if bb_width > 0.04:
            return "volatile"
        return "trending"

    if bb_width < 0.01:
        return "quiet"
    if bb_width > 0.05:
        return "volatile"

    return "ranging"


def get_regime_params(regime: str, strategy_params: dict) -> dict:
    """Merge base params with regime-specific overrides."""
    base = {
        "min_confidence_threshold": strategy_params.get("min_confidence_threshold", 0.65),
        "sl_atr_multiplier":        strategy_params.get("sl_atr_multiplier", 1.5),
        "tp_sl_ratio":              strategy_params.get("tp_sl_ratio", 2.0),
    }
    overrides = strategy_params.get("regimes", {}).get(regime, {})
    merged = {**base, **{k: v for k, v in overrides.items() if k in base}}
    return merged
