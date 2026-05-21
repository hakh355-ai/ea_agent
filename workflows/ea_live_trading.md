# EA Live Trading

## Objective
Run the AI_EA on a Vantage MT5 account, fully supervised by the Python bridge.
All trades are decided by Claude Sonnet 4.6 (trading agent) after news screening by GPT-4o.

## Pre-Start Checklist
- [ ] API keys in `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NEWSAPI_KEY`
- [ ] Python dependencies installed: `pip install -r requirements.txt`
- [ ] MT5 installed and connected to Vantage
- [ ] MT5 → Tools → Options → Expert Advisors → WebRequest URLs → add `http://127.0.0.1:5000`
- [ ] `LiveTrade = false` for first test run (dry run mode)

## Steps

### 1. Start the bridge
```
cd EA_Agent
uvicorn bridge.server:app --host 127.0.0.1 --port 5000
```
Verify: open `http://127.0.0.1:5000/health` in browser → should return `{"status":"ok"}`

### 2. Compile and attach the EA
- Open MT5 → MetaEditor → open `mt5_ea/AI_EA.mq5` → Compile (F7)
- Attach `AI_EA` to any chart (e.g. EURUSD H1)
- Set `LiveTrade = false` for dry run, `true` for live

### 3. Monitor
- **MT5 Journal tab**: shows every signal received and trade decision
- **Bridge console**: shows regime detection, sentiment scores, and blocked reasons
- **`.tmp/trade_log.jsonl`**: append-only log of all signals and fills

### 4. Go live
- Stop EA, set `LiveTrade = true`, restart EA
- Watch the first 5 trades manually to confirm correct execution

## Edge Cases
| Problem | Response |
|---------|----------|
| Bridge unreachable (error 4014) | Add URL to MT5 allowed list; EA logs error, no trades placed |
| Claude API timeout | `trading_agent.py` catches exception → returns `hold` |
| News fetch fails | `news_agent.py` returns neutral sentiment (0.0), trading continues with caution |
| All signals blocked | Normal when: drawdown limit hit, 3 positions open, or news blackout active |
| Wide spread | EA's `MaxSpreadPoints` guard skips the trade |

## Outputs
- Live trades on Vantage MT5 account
- `.tmp/trade_log.jsonl` — full decision audit trail
- Bridge console logs — real-time monitoring
