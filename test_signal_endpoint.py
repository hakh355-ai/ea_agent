"""End-to-end test of POST /signal — run: python test_signal_endpoint.py"""
import httpx, time

# Build uptrending bars for BTCUSD
def make_uptrend(n, start, step):
    bars = []
    for i in range(n):
        o = round(start + i * step, 2)
        c = round(o + step * 0.8, 2)
        h = round(c + step * 0.5, 2)
        l = round(o - step * 0.3, 2)
        bars.append({"t": f"2026-05-10 {8 + i//12:02d}:{(i%12)*5:02d}:00",
                     "o": o, "h": h, "l": l, "c": c, "v": 500 + i})
    return bars

m5_bars = make_uptrend(100, 100000, 10)
h1_bars = make_uptrend(30, 99000, 50)
h4_bars = make_uptrend(10, 97000, 200)

current_price = m5_bars[-1]["c"]

payload = {
    "request_id": "BTCUSD_test_001",
    "symbol": "BTCUSD",
    "timestamp_utc": "2026-05-10T14:00:00Z",   # Sonntag OK — Crypto 24/7
    "account": {"balance": 10000.0, "equity": 10000.0, "open_positions": 0, "daily_pnl": 0.0},
    "ohlc": {"M5": m5_bars, "H1": h1_bars, "H4": h4_bars},
    "current_tick": {"bid": current_price - 5, "ask": current_price + 5, "spread_points": 10},
    "open_position": None,
}

print(f"Sending POST /signal for BTCUSD (Crypto 24/7)...")
print(f"Aktueller Preis: ${current_price:,.2f}")
t0 = time.time()
r = httpx.post("http://127.0.0.1:5000/signal", json=payload, timeout=60)
elapsed = time.time() - t0

print(f"\nStatus: {r.status_code}  Zeit: {elapsed:.1f}s")
data = r.json()
print(f"action         : {data.get('action')}")
print(f"confidence     : {data.get('confidence')}")
print(f"lot_size       : {data.get('lot_size')}  (erwartet: 0.05)")
print(f"sl_price       : {data.get('sl_price')}  (erwartet: ~${current_price - 10:,.2f} = 10 Pips tiefer)")
print(f"tp_price       : {data.get('tp_price')}  (erwartet: ~${current_price + 30:,.2f} = 30 Pips höher)")
print(f"blocked_reason : {data.get('blocked_reason')}")
print(f"reason         : {str(data.get('reason', ''))[:150]}")
