"""
Telegram Bot — trade notifications and remote control.

Notifications (outgoing):
  • Every fill: symbol, action, lots, price, confidence, regime
  • Every close: symbol, PnL, outcome (TP/SL hit)
  • Drawdown alert: when daily drawdown exceeds 3%
  • Weekly report: win rate, total PnL, best/worst trade (Sunday 04:00 UTC)

Remote commands (incoming):
  /status    — current positions, today's PnL, bridge uptime
  /pause     — stop accepting new signals (existing positions unaffected)
  /resume    — re-enable signal processing
  /risk <n>  — change risk % per trade (e.g. /risk 1.5)
  /report    — trigger weekly report on demand

Setup:
  1. Create a bot via @BotFather → copy token to TELEGRAM_BOT_TOKEN
  2. Get your chat ID: message @userinfobot → copy to TELEGRAM_CHAT_ID
  3. Both values go in .env
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

logger = logging.getLogger(__name__)

_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
_paused     = False
_start_time = time.time()

# Shared bot instance (send-only, used from bridge/server.py)
_bot: Bot | None = None


def _get_bot() -> Bot | None:
    global _bot
    if not _BOT_TOKEN:
        return None
    if _bot is None:
        _bot = Bot(token=_BOT_TOKEN)
    return _bot


async def send(text: str):
    """Fire-and-forget Telegram message. Silently skips if token not set."""
    bot = _get_bot()
    if bot is None or not _CHAT_ID:
        return
    try:
        await bot.send_message(chat_id=_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ── Notification helpers ──────────────────────────────────────────────────────

def _fmt_price(price: float, symbol: str) -> str:
    """Format price with correct decimal places per symbol."""
    if symbol in ("XAUUSD",):
        return f"{price:.2f}"
    if symbol in ("US500", "GER40"):
        return f"{price:.1f}"
    if symbol in ("BTCUSD",):
        return f"{price:.0f}"
    if "JPY" in symbol:
        return f"{price:.3f}"
    return f"{price:.5f}"


async def notify_fill(symbol: str, action: str, lots: float, price: float,
                      confidence: float, regime: str, sl: float, tp: float,
                      sl_pips: float = 0.0, tp_pips: float = 0.0):
    icon  = "🟢" if action == "buy" else "🔴"
    p     = _fmt_price(price, symbol)
    sl_s  = _fmt_price(sl, symbol) if sl else "—"
    tp_s  = _fmt_price(tp, symbol) if tp else "—"
    pips  = f"  ({sl_pips:.0f}p SL / {tp_pips:.0f}p TP)" if sl_pips else ""
    conf  = f"{confidence:.0%}" if confidence else "—"
    await send(
        f"{icon} <b>{action.upper()} {symbol}</b>\n"
        f"Entry: {p}  Lots: {lots:.2f}\n"
        f"SL: {sl_s}  TP: {tp_s}{pips}\n"
        f"Confidence: {conf}  Regime: {regime}"
    )


async def notify_close(symbol: str, pnl: float, outcome: str, close_price: float):
    icon  = "✅" if pnl >= 0 else "❌"
    label = {"tp_hit": "TP hit", "sl_hit": "SL hit",
             "manual": "Manual", "global_tp": "Global TP"}.get(outcome, outcome)
    sign  = "+" if pnl >= 0 else ""
    await send(
        f"{icon} <b>CLOSE {symbol}</b> — {label}\n"
        f"PnL: <b>{sign}{pnl:.2f} €</b>  @{_fmt_price(close_price, symbol)}"
    )


async def notify_drawdown_warning(current_pct: float, balance: float):
    await send(
        f"⚠️ <b>Drawdown Warnung</b>\n"
        f"Tages-Drawdown: <b>{current_pct:.1%}</b> von {balance:.0f} €\n"
        f"Limit: 5% — Vorsicht!"
    )


async def send_weekly_report():
    from tools.trade_logger import read_recent_trades
    trades = read_recent_trades(days=7)
    closes = [t for t in trades if t["type"] == "close"]
    if not closes:
        await send("📊 <b>Weekly Report</b>\nNo closed trades this week.")
        return

    wins   = [t for t in closes if t.get("pnl", 0) > 0]
    losses = [t for t in closes if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in closes)
    win_rate  = len(wins) / len(closes) if closes else 0
    best  = max(closes, key=lambda t: t.get("pnl", 0))
    worst = min(closes, key=lambda t: t.get("pnl", 0))

    await send(
        f"📊 <b>Weekly Report</b>\n"
        f"Trades: {len(closes)}  Wins: {len(wins)}  Losses: {len(losses)}\n"
        f"Win rate: <b>{win_rate:.1%}</b>\n"
        f"Total PnL: <b>{'+'if total_pnl>=0 else ''}{total_pnl:.2f} €</b>\n"
        f"Best:  +{best.get('pnl',0):.2f} ({best.get('symbol','')})\n"
        f"Worst: {worst.get('pnl',0):.2f} ({worst.get('symbol','')})"
    )


# ── Remote command handlers ───────────────────────────────────────────────────

def is_paused() -> bool:
    return _paused


async def _cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    import bridge.state as state
    s      = state.get_state()
    params = state.get_strategy_params()
    uptime_h   = (time.time() - _start_time) / 3600
    fixed_lot  = params.get("fixed_lot_size", 0)
    daily_tgt  = params.get("daily_profit_target", 0)
    sl_pips    = params.get("fixed_sl_pips", 0)
    tp_pips    = params.get("fixed_tp_pips", 0)

    pos_lines = []
    for sym, pos in s.open_positions.items():
        lots   = pos.get("lots", 0)
        entry  = pos.get("fill_price", 0)
        action = pos.get("action", "?").upper()
        conf   = pos.get("confidence", 0)
        regime = pos.get("regime", "")
        if lots > 0:
            line = f"  • {sym}: {action} {lots:.2f}L @ {_fmt_price(entry, sym)}"
            if conf:
                line += f"  [{conf:.0%}]"
            pos_lines.append(line)
        else:
            pos_lines.append(f"  • {sym}: pending...")

    positions = "\n".join(pos_lines) or "  None"
    pnl_sign  = "+" if s.daily_realized_pnl >= 0 else ""

    await update.message.reply_text(
        f"📡 <b>Bridge Status</b>\n"
        f"Uptime: {uptime_h:.1f}h  "
        f"{'⏸ Paused' if _paused else '▶️ Active'}\n\n"
        f"Lots: {fixed_lot}  SL: {sl_pips}p  TP: {tp_pips}p\n"
        f"Daily PnL: <b>{pnl_sign}{s.daily_realized_pnl:.2f} €</b>  "
        f"(Ziel: {daily_tgt:.0f} €)\n"
        f"Losing streak: {s.consecutive_losses}\n\n"
        f"Offene Positionen ({len(s.open_positions)}):\n{positions}",
        parse_mode="HTML"
    )


async def _cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _paused
    _paused = True
    await update.message.reply_text("⏸ EA <b>paused</b> — no new trades until /resume", parse_mode="HTML")
    logger.info("EA paused via Telegram")


async def _cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _paused
    _paused = False
    await update.message.reply_text("▶️ EA <b>resumed</b>", parse_mode="HTML")
    logger.info("EA resumed via Telegram")


async def _cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    import bridge.state as state
    from pathlib import Path
    import json

    args = ctx.args
    if not args:
        await update.message.reply_text("Usage: /risk 2.0  (sets risk to 2% per trade)")
        return
    try:
        new_risk = float(args[0]) / 100.0
        assert 0.001 <= new_risk <= 0.50, "Risk must be between 0.1% and 50%"
    except Exception as e:
        await update.message.reply_text(f"Invalid value: {e}")
        return

    params_path = Path(os.getenv("STRATEGY_PARAMS_PATH", ".tmp/strategy_params.json"))
    params = json.loads(params_path.read_text())
    params["default_risk_pct"] = new_risk
    params_path.write_text(json.dumps(params, indent=2))
    state.reload_strategy_params()

    await update.message.reply_text(
        f"✅ Risk updated to <b>{new_risk:.1%}</b> per trade", parse_mode="HTML"
    )
    logger.info(f"Risk changed to {new_risk:.1%} via Telegram")


async def _cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await send_weekly_report()


# ── Start polling (runs in a background thread) ───────────────────────────────

def start_bot():
    """Start the Telegram command listener. Call once at bridge startup."""
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.info("Telegram not configured (missing BOT_TOKEN or CHAT_ID). Skipping.")
        return

    app = Application.builder().token(_BOT_TOKEN).build()
    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("pause",  _cmd_pause))
    app.add_handler(CommandHandler("resume", _cmd_resume))
    app.add_handler(CommandHandler("risk",   _cmd_risk))
    app.add_handler(CommandHandler("report", _cmd_report))

    # Run in background thread (non-blocking)
    import threading
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(app.run_polling(close_loop=False))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("Telegram bot started (polling for commands)")
