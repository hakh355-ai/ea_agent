//+------------------------------------------------------------------+
//| AI_EA.mq5 — AI-powered Expert Advisor for Vantage Broker        |
//| Sends market data to Python bridge, executes AI signals          |
//+------------------------------------------------------------------+
#property copyright "EA Agent"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//── Inputs ───────────────────────────────────────────────────────────────────
input string   BridgeURL        = "http://127.0.0.1:5000";  // Bridge server URL
input bool     LiveTrade        = false;                     // Enable live execution
input int      PollIntervalSec  = 5;                         // Seconds between requests
input int      M1Bars           = 30;                        // M1 bars to send
input int      M5Bars           = 100;                       // M5 bars to send
input int      H1Bars           = 50;                        // H1 bars to send
input int      H4Bars           = 50;                        // H4 bars to send
input int      M15Bars          = 50;                        // M15 bars to send
input int      DailyBars        = 10;                        // Daily bars to send
input int      MaxSpreadPoints  = 999999;                    // Max spread (bridge handles per-symbol check)
input double   ProfitTargetEUR      = 20.0;                 // Close single trade at this profit (EUR) [0.05 lots: 2:1 R:R]
input double   GlobalProfitEUR     = 0.0;                   // Close ALL trades at total profit (0=disabled, bridge handles daily 50 EUR)

//── Symbols ───────────────────────────────────────────────────────────────────
string Symbols[] = {"EURUSD","GBPUSD","XAUUSD","GER40","BTCUSD"};
int    SymbolCount = 5;

//── State ─────────────────────────────────────────────────────────────────────
CTrade         Trade;
CPositionInfo  PosInfo;
datetime       LastSent[];      // last send time per symbol
datetime       LastCandle[];    // last M1 candle time per symbol (candle-close filter)
double         TP1Prices[];     // partial close level per symbol (50% of TP distance)
bool           PartialClosed[]; // whether partial close already executed for current position
bool           HadPosition[];   // tracks whether each symbol had an open position last tick
double         InitSlDist[];    // initial SL distance at order placement (for correct trailing)
int            RequestSeq = 0;
datetime       LastSyncTime = 0;   // for periodic position sync

//+------------------------------------------------------------------+
int OnInit()
{
   ArrayResize(LastSent, SymbolCount);
   ArrayInitialize(LastSent, 0);
   ArrayResize(LastCandle, SymbolCount);
   ArrayInitialize(LastCandle, 0);
   ArrayResize(TP1Prices, SymbolCount);
   ArrayInitialize(TP1Prices, 0.0);
   ArrayResize(PartialClosed, SymbolCount);
   ArrayInitialize(PartialClosed, false);
   ArrayResize(HadPosition, SymbolCount);
   ArrayInitialize(HadPosition, false);
   ArrayResize(InitSlDist, SymbolCount);
   ArrayInitialize(InitSlDist, 0.0);

   // Add all symbols to Market Watch
   for(int i = 0; i < SymbolCount; i++)
      SymbolSelect(Symbols[i], true);

   // Verify WebRequest is allowed (will fail silently if URL not whitelisted)
   Print("AI_EA v2.0 initialized. LiveTrade=", LiveTrade,
         " Bridge=", BridgeURL);

   if(!LiveTrade)
      Print("WARNING: LiveTrade=false — signals will be logged but no orders placed.");

   // Sync open positions with bridge so state is correct after any restart
   SyncPositionsWithBridge();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime now = TimeCurrent();

   // Detect manual/external closes and notify bridge to clear stale state
   DetectManualCloses();

   // Periodic sync every 10 minutes: ensures bridge state matches reality even if /close fails
   if(TimeCurrent() - LastSyncTime > 600)
   {
      SyncPositionsWithBridge();
      LastSyncTime = TimeCurrent();
   }

   // Close ALL positions if portfolio hits global profit target
   CheckGlobalProfit();

   // Partial close at TP1 (50% of TP), then move SL to breakeven
   CheckPartialTP();

   // Close single positions that hit individual profit/loss targets
   CheckProfitTarget();

   // Trail open positions every tick
   TrailPositions();

   for(int i = 0; i < SymbolCount; i++)
   {
      // Candle-close filter: only analyse when a new M1 candle has opened
      datetime candleTime = iTime(Symbols[i], PERIOD_M1, 0);
      if(candleTime == LastCandle[i]) continue;   // same candle, skip
      LastCandle[i] = candleTime;

      // Rate limit: respect PollIntervalSec minimum between requests
      if((int)(now - LastSent[i]) < PollIntervalSec) continue;
      LastSent[i] = now;

      ProcessSymbol(Symbols[i]);
   }
}

//+------------------------------------------------------------------+
// Detect Manual / External Closes
// Fires once when a position transitions from open → closed.
// Reads actual PnL and close reason from MT5 deal history.
//+------------------------------------------------------------------+
void DetectManualCloses()
{
   for(int i = 0; i < SymbolCount; i++)
   {
      bool hasNow = false;
      for(int j = 0; j < PositionsTotal(); j++)
         if(PosInfo.SelectByIndex(j) && PosInfo.Symbol() == Symbols[i])
            { hasNow = true; break; }

      // Transition: had position → now gone (SL/TP hit, manual close, etc.)
      if(HadPosition[i] && !hasNow)
      {
         PrintFormat("[%s] External close detected → reading deal history", Symbols[i]);

         // Reset partial close state
         TP1Prices[i]    = 0.0;
         PartialClosed[i] = false;
         InitSlDist[i]   = 0.0;

         // Read actual PnL + close reason from deal history (last 5 minutes)
         double closePnl   = 0.0;
         double closePrice = 0.0;
         long   closeTkt   = 0;
         string outcome    = "manual";

         if(HistorySelect(TimeCurrent() - 600, TimeCurrent() + 1))
         {
            for(int d = HistoryDealsTotal() - 1; d >= 0; d--)
            {
               ulong dealTkt = HistoryDealGetTicket(d);
               if(HistoryDealGetString(dealTkt, DEAL_SYMBOL) == Symbols[i] &&
                  HistoryDealGetInteger(dealTkt, DEAL_ENTRY) == DEAL_ENTRY_OUT)
               {
                  closeTkt   = (long)dealTkt;
                  closePrice = HistoryDealGetDouble(dealTkt, DEAL_PRICE);
                  closePnl   = HistoryDealGetDouble(dealTkt, DEAL_PROFIT)
                             + HistoryDealGetDouble(dealTkt, DEAL_COMMISSION)
                             + HistoryDealGetDouble(dealTkt, DEAL_SWAP);
                  ENUM_DEAL_REASON reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(dealTkt, DEAL_REASON);
                  if(reason == DEAL_REASON_SL)      outcome = "sl_hit";
                  else if(reason == DEAL_REASON_TP) outcome = "tp_hit";
                  else                              outcome = "manual";
                  break;
               }
            }
         }

         PrintFormat("[%s] Close: outcome=%s pnl=%.2f price=%.5f",
                     Symbols[i], outcome, closePnl, closePrice);

         string payload = StringFormat(
            "{\"ticket\":%I64d,\"symbol\":\"%s\",\"close_price\":%.6f,\"pnl\":%.2f,\"outcome\":\"%s\"}",
            closeTkt, Symbols[i], closePrice, closePnl, outcome
         );
         SendRequest("/trade_close",payload);
      }

      HadPosition[i] = hasNow;
   }
}

//+------------------------------------------------------------------+
// Partial Close at TP1 (Multiple Take-Profits)
// When price reaches 50% of TP distance:
//   1. Close half the position
//   2. Move SL to breakeven so remaining position is risk-free
//+------------------------------------------------------------------+
void CheckPartialTP()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PosInfo.SelectByIndex(i)) continue;

      string sym = PosInfo.Symbol();
      int idx = -1;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { idx = k; break; }
      if(idx < 0 || PartialClosed[idx] || TP1Prices[idx] <= 0.0) continue;

      double openPrice = PosInfo.PriceOpen();
      bool   tp1Hit   = false;

      if(PosInfo.PositionType() == POSITION_TYPE_BUY)
      {
         if(SymbolInfoDouble(sym, SYMBOL_BID) >= TP1Prices[idx]) tp1Hit = true;
      }
      else
      {
         if(SymbolInfoDouble(sym, SYMBOL_ASK) <= TP1Prices[idx]) tp1Hit = true;
      }

      if(!tp1Hit) continue;

      double totalLots = PosInfo.Volume();
      double halfLots  = MathFloor(totalLots * 50.0) / 100.0;
      if(halfLots < 0.01) halfLots = 0.01;
      if(halfLots >= totalLots) continue;  // position too small to split

      ulong  ticket   = PosInfo.Ticket();
      double keepTP   = PosInfo.TakeProfit();
      int    digits   = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      double point    = SymbolInfoDouble(sym, SYMBOL_POINT);
      double bePrice  = (PosInfo.PositionType() == POSITION_TYPE_BUY)
                      ? NormalizeDouble(openPrice + point * 2, digits)
                      : NormalizeDouble(openPrice - point * 2, digits);

      PrintFormat("[%s] TP1 %.5f hit → partial close %.2f lots, SL → breakeven %.5f",
                  sym, TP1Prices[idx], halfLots, bePrice);

      if(LiveTrade)
      {
         Trade.PositionClosePartial(ticket, halfLots);
         Trade.PositionModify(ticket, bePrice, keepTP);
      }
      else
         PrintFormat("[%s] DRY-RUN: partial close %.2f lots @ TP1=%.5f, SL→%.5f",
                     sym, halfLots, TP1Prices[idx], bePrice);

      PartialClosed[idx] = true;
   }
}

//+------------------------------------------------------------------+
// Profit Target Close
// Closes any position that has reached ProfitTargetEUR floating profit.
// After closing, the EA will re-enter on the next 5-second poll cycle.
//+------------------------------------------------------------------+
void CheckProfitTarget()
{
   if(ProfitTargetEUR <= 0) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PosInfo.SelectByIndex(i)) continue;

      string sym = PosInfo.Symbol();
      bool isOurs = false;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { isOurs = true; break; }
      if(!isOurs) continue;

      double profit = PosInfo.Profit() + PosInfo.Commission() + PosInfo.Swap();

      if(profit >= ProfitTargetEUR)
      {
         PrintFormat("[%s] Profit target hit: %.2f EUR >= %.2f EUR → closing",
                     sym, profit, ProfitTargetEUR);

         long   ticket     = (long)PosInfo.Ticket();
         double closePrice = PosInfo.PriceCurrent();

         if(LiveTrade)
            Trade.PositionClose(sym);

         for(int m = 0; m < SymbolCount; m++)
            if(Symbols[m] == sym) { TP1Prices[m] = 0.0; PartialClosed[m] = false; InitSlDist[m] = 0.0; break; }

         string closePayload = StringFormat(
            "{\"ticket\":%I64d,\"symbol\":\"%s\",\"close_price\":%.6f,\"pnl\":%.2f,\"outcome\":\"tp_hit\"}",
            ticket, sym, closePrice, profit
         );
         SendRequest("/trade_close",closePayload);
      }
   }
}

//+------------------------------------------------------------------+
// Global Portfolio Profit Target
// If total floating P&L of ALL our positions >= GlobalProfitEUR → close all.
// After closing the EA continues its normal polling cycle and re-enters on new signals.
//+------------------------------------------------------------------+
void CheckGlobalProfit()
{
   if(GlobalProfitEUR <= 0) return;

   // Sum floating P&L across all our positions
   double totalProfit = 0.0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!PosInfo.SelectByIndex(i)) continue;
      string sym = PosInfo.Symbol();
      bool isOurs = false;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { isOurs = true; break; }
      if(!isOurs) continue;
      totalProfit += PosInfo.Profit() + PosInfo.Commission() + PosInfo.Swap();
   }

   if(totalProfit < GlobalProfitEUR) return;

   PrintFormat("[GLOBAL] Portfolio profit %.2f EUR >= %.2f EUR → closing ALL positions",
               totalProfit, GlobalProfitEUR);

   // Close all our positions
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PosInfo.SelectByIndex(i)) continue;
      string sym = PosInfo.Symbol();
      bool isOurs = false;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { isOurs = true; break; }
      if(!isOurs) continue;

      double profit     = PosInfo.Profit() + PosInfo.Commission() + PosInfo.Swap();
      long   ticket     = (long)PosInfo.Ticket();
      double closePrice = PosInfo.PriceCurrent();

      if(LiveTrade)
         Trade.PositionClose(sym);

      for(int m = 0; m < SymbolCount; m++)
         if(Symbols[m] == sym) { TP1Prices[m] = 0.0; PartialClosed[m] = false; InitSlDist[m] = 0.0; break; }

      string closePayload = StringFormat(
         "{\"ticket\":%I64d,\"symbol\":\"%s\",\"close_price\":%.6f,\"pnl\":%.2f,\"outcome\":\"global_tp\"}",
         ticket, sym, closePrice, profit
      );
      SendRequest("/trade_close",closePayload);
   }
}

//+------------------------------------------------------------------+
// Trailing Stop Loss
// Phase 1: When profit >= 1x SL distance → move SL to breakeven
// Phase 2: When profit >= 2x SL distance → trail SL at 1x SL distance behind price
//+------------------------------------------------------------------+
void TrailPositions()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!PosInfo.SelectByIndex(i)) continue;

      string sym = PosInfo.Symbol();
      bool isOurs = false;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { isOurs = true; break; }
      if(!isOurs) continue;

      double openPrice  = PosInfo.PriceOpen();
      double currentSL  = PosInfo.StopLoss();
      double currentTP  = PosInfo.TakeProfit();
      double point      = SymbolInfoDouble(sym, SYMBOL_POINT);
      double digits     = (double)SymbolInfoInteger(sym, SYMBOL_DIGITS);

      // SL distance = original SL distance stored at order placement
      int symIdx = -1;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { symIdx = k; break; }
      double slDist = (symIdx >= 0 && InitSlDist[symIdx] > 0)
                      ? InitSlDist[symIdx]
                      : MathAbs(currentSL - openPrice);
      if(slDist < point * 5) continue;  // Too small, skip

      double newSL = currentSL;

      if(PosInfo.PositionType() == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(sym, SYMBOL_BID);
         double profit_dist = bid - openPrice;

         if(profit_dist >= slDist * 2.0)
            newSL = NormalizeDouble(bid - slDist, (int)digits);       // Phase 2: trail
         else if(profit_dist >= slDist && currentSL < openPrice)
            newSL = NormalizeDouble(openPrice + point * 2, (int)digits); // Phase 1: breakeven

         if(newSL > currentSL && newSL < bid)
         {
            if(LiveTrade)
               Trade.PositionModify(PosInfo.Ticket(), newSL, currentTP);
            else
               PrintFormat("[%s] TRAIL BUY: SL %.5f → %.5f", sym, currentSL, newSL);
         }
      }
      else if(PosInfo.PositionType() == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
         double profit_dist = openPrice - ask;

         if(profit_dist >= slDist * 2.0)
            newSL = NormalizeDouble(ask + slDist, (int)digits);          // Phase 2: trail
         else if(profit_dist >= slDist && (currentSL > openPrice || currentSL == 0))
            newSL = NormalizeDouble(openPrice - point * 2, (int)digits); // Phase 1: breakeven

         if((newSL < currentSL || currentSL == 0) && newSL > ask)
         {
            if(LiveTrade)
               Trade.PositionModify(PosInfo.Ticket(), newSL, currentTP);
            else
               PrintFormat("[%s] TRAIL SELL: SL %.5f → %.5f", sym, currentSL, newSL);
         }
      }
   }
}

//+------------------------------------------------------------------+
void ProcessSymbol(string symbol)
{
   // Skip immediately if position already open for this symbol
   for(int i = 0; i < PositionsTotal(); i++)
      if(PosInfo.SelectByIndex(i) && PosInfo.Symbol() == symbol) return;

   // Build payload
   string payload = BuildPayload(symbol);
   if(payload == "") return;

   // Send to bridge
   string response = SendRequest("/signal", payload);
   if(response == "") return;

   // Parse response
   string action        = JsonGetString(response, "action");
   double confidence    = JsonGetDouble(response, "confidence");
   double lotSize       = JsonGetDouble(response, "lot_size");
   double slPips        = JsonGetDouble(response, "sl_pips");
   double tpPips        = JsonGetDouble(response, "tp_pips");
   string reason        = JsonGetString(response, "reason");
   string blockedReason = JsonGetString(response, "blocked_reason");
   string requestId     = JsonGetString(response, "request_id");

   PrintFormat("[%s] action=%s conf=%.2f lots=%.2f sl=%.0fpips tp=%.0fpips | %s",
               symbol, action, confidence, lotSize, slPips, tpPips, reason);

   if(action == "blocked")
   {
      PrintFormat("[%s] Blocked: %s", symbol, blockedReason);
      return;
   }

   if(action == "hold") return;

   // Execute trade — SL/TP recalculated from live price inside ExecuteSignal
   if(action == "buy" || action == "sell")
   {
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == symbol) { PartialClosed[k] = false; break; }
      ExecuteSignal(symbol, action, lotSize, slPips, tpPips, confidence, requestId);
   }
   else if(action == "close")
      ClosePosition(symbol, requestId);
}

//+------------------------------------------------------------------+
string BuildPayload(string symbol)
{
   MqlRates m1rates[], m5rates[], m15rates[], h1rates[], h4rates[], drates[];
   ArraySetAsSeries(m1rates,  true);
   ArraySetAsSeries(m5rates,  true);
   ArraySetAsSeries(m15rates, true);
   ArraySetAsSeries(h1rates,  true);
   ArraySetAsSeries(h4rates,  true);
   ArraySetAsSeries(drates,   true);

   int m1count    = CopyRates(symbol, PERIOD_M1,  0, M1Bars,    m1rates);
   int m5count    = CopyRates(symbol, PERIOD_M5,  0, M5Bars,    m5rates);
   int m15count   = CopyRates(symbol, PERIOD_M15, 0, M15Bars,   m15rates);
   int h1count    = CopyRates(symbol, PERIOD_H1,  0, H1Bars,    h1rates);
   int h4count    = CopyRates(symbol, PERIOD_H4,  0, H4Bars,    h4rates);
   int dcount     = CopyRates(symbol, PERIOD_D1,  0, DailyBars, drates);

   if(m5count < 50)
   {
      Print("[", symbol, "] Not enough M5 bars: ", m5count);
      return("");
   }

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return("");

   double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   int    openPos  = 0;

   // Count our open positions
   for(int j = 0; j < PositionsTotal(); j++)
      if(PosInfo.SelectByIndex(j)) openPos++;

   // Build open position JSON for this symbol
   string openPosJson = "null";
   if(PositionSelect(symbol))
   {
      openPosJson = StringFormat(
         "{\"ticket\":%I64d,\"type\":\"%s\",\"lots\":%.2f,\"open_price\":%.5f,"
         "\"sl\":%.5f,\"tp\":%.5f,\"pnl_float\":%.2f}",
         (long)PositionGetInteger(POSITION_TICKET),
         PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell",
         PositionGetDouble(POSITION_VOLUME),
         PositionGetDouble(POSITION_PRICE_OPEN),
         PositionGetDouble(POSITION_SL),
         PositionGetDouble(POSITION_TP),
         PositionGetDouble(POSITION_PROFIT)
      );
   }

   RequestSeq++;
   string reqId = symbol + "_" + IntegerToString(TimeCurrent()) + "_" + IntegerToString(RequestSeq);

   string payload = "{";
   payload += "\"request_id\":\"" + reqId + "\",";
   payload += "\"symbol\":\"" + symbol + "\",";
   payload += "\"timestamp_utc\":\"" + TimeToString(TimeGMT(), TIME_DATE|TIME_SECONDS) + "\",";
   payload += "\"account\":{";
   payload +=   "\"balance\":" + DoubleToString(balance, 2) + ",";
   payload +=   "\"equity\":"  + DoubleToString(equity, 2) + ",";
   payload +=   "\"open_positions\":" + IntegerToString(openPos) + ",";
   payload +=   "\"daily_pnl\":" + DoubleToString(equity - balance, 2);
   payload += "},";
   payload += "\"current_tick\":{";
   payload +=   "\"bid\":"           + DoubleToString(tick.bid, 6) + ",";
   payload +=   "\"ask\":"           + DoubleToString(tick.ask, 6) + ",";
   payload +=   "\"spread_points\":" + IntegerToString((int)((tick.ask - tick.bid) / SymbolInfoDouble(symbol, SYMBOL_POINT)));
   payload += "},";
   payload += "\"open_position\":" + openPosJson + ",";

   // M1 bars (last 30 candles for short-term momentum)
   payload += "\"ohlc\":{\"M1\":[";
   int m1limit = MathMin(m1count, M1Bars);
   for(int k = m1limit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(m1rates[k].time, TIME_DATE|TIME_SECONDS),
                  m1rates[k].open, m1rates[k].high, m1rates[k].low, m1rates[k].close,
                  (int)m1rates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   // M5 bars
   payload += "],\"M5\":[";
   int m5limit = MathMin(m5count, M5Bars);
   for(int k = m5limit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(m5rates[k].time, TIME_DATE|TIME_SECONDS),
                  m5rates[k].open, m5rates[k].high, m5rates[k].low, m5rates[k].close,
                  (int)m5rates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   payload += "],\"H1\":[";
   int h1limit = MathMin(h1count, H1Bars);
   for(int k = h1limit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(h1rates[k].time, TIME_DATE|TIME_SECONDS),
                  h1rates[k].open, h1rates[k].high, h1rates[k].low, h1rates[k].close,
                  (int)h1rates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   payload += "],\"H4\":[";
   int h4limit = MathMin(h4count, H4Bars);
   for(int k = h4limit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(h4rates[k].time, TIME_DATE|TIME_SECONDS),
                  h4rates[k].open, h4rates[k].high, h4rates[k].low, h4rates[k].close,
                  (int)h4rates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   payload += "],\"M15\":[";
   int m15limit = MathMin(m15count, M15Bars);
   for(int k = m15limit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(m15rates[k].time, TIME_DATE|TIME_SECONDS),
                  m15rates[k].open, m15rates[k].high, m15rates[k].low, m15rates[k].close,
                  (int)m15rates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   payload += "],\"Daily\":[";
   int dlimit = MathMin(dcount, DailyBars);
   for(int k = dlimit - 1; k >= 0; k--)
   {
      payload += StringFormat("{\"t\":\"%s\",\"o\":%.6f,\"h\":%.6f,\"l\":%.6f,\"c\":%.6f,\"v\":%d}",
                  TimeToString(drates[k].time, TIME_DATE|TIME_SECONDS),
                  drates[k].open, drates[k].high, drates[k].low, drates[k].close,
                  (int)drates[k].tick_volume);
      if(k > 0) payload += ",";
   }
   payload += "]}}";

   return(payload);
}

//+------------------------------------------------------------------+
void ExecuteSignal(string symbol, string action, double lots,
                   double slPips, double tpPips, double confidence, string requestId)
{
   // Guard: check ALL positions for this symbol
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PosInfo.SelectByIndex(i) && PosInfo.Symbol() == symbol)
      {
         PrintFormat("[%s] Position already open (ticket=%d), skipping.", symbol, (int)PosInfo.Ticket());
         return;
      }
   }

   // Guard: spread check
   MqlTick tick;
   SymbolInfoTick(symbol, tick);
   int spread = (int)((tick.ask - tick.bid) / SymbolInfoDouble(symbol, SYMBOL_POINT));
   if(spread > MaxSpreadPoints)
   {
      PrintFormat("[%s] Spread too wide: %d points. Skipping.", symbol, spread);
      return;
   }

   // Guard: minimum lot
   if(lots < 0.01) lots = 0.01;

   // Fallback pip values if bridge didn't send them
   if(slPips <= 0.0) slPips = 15.0;
   if(tpPips <= 0.0) tpPips = 30.0;

   // Enforce broker minimum stop level
   double point      = SymbolInfoDouble(symbol, SYMBOL_POINT);
   long   stopsLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double pipSize    = GetPipSize(symbol);
   double minDist    = stopsLevel * point;           // broker minimum in price
   double minPips    = (pipSize > 0) ? minDist / pipSize : 0;
   if(slPips < minPips + 2) slPips = minPips + 2;   // +2 pip buffer
   if(tpPips < slPips * 1.5) tpPips = slPips * 1.5;

   // Recalculate SL/TP/TP1 from CURRENT tick — eliminates latency offset
   double slDist    = slPips * pipSize;
   double tpDist    = tpPips * pipSize;
   double tp1Dist   = (tpPips * 0.5) * pipSize;
   int    digits    = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double entryPrice, sl, tp, tp1;

   if(action == "buy")
   {
      entryPrice = tick.ask;
      sl  = NormalizeDouble(entryPrice - slDist,  digits);
      tp  = NormalizeDouble(entryPrice + tpDist,  digits);
      tp1 = NormalizeDouble(entryPrice + tp1Dist, digits);
   }
   else
   {
      entryPrice = tick.bid;
      sl  = NormalizeDouble(entryPrice + slDist,  digits);
      tp  = NormalizeDouble(entryPrice - tpDist,  digits);
      tp1 = NormalizeDouble(entryPrice - tp1Dist, digits);
   }

   // Store TP1 for partial close
   for(int k = 0; k < SymbolCount; k++)
      if(Symbols[k] == symbol) { TP1Prices[k] = tp1; break; }

   PrintFormat("[%s] %s entry=%.5f sl=%.5f(%.0fpips) tp1=%.5f tp=%.5f(%.0fpips)",
               symbol, action, entryPrice, sl, slPips, tp1, tp, tpPips);

   string comment = StringFormat("AI|%.2f|%s", confidence, requestId);
   bool   result  = false;

   if(LiveTrade)
   {
      if(action == "buy")
         result = Trade.Buy(lots, symbol, 0, sl, tp, comment);
      else
         result = Trade.Sell(lots, symbol, 0, sl, tp, comment);
      if(result)
         for(int m = 0; m < SymbolCount; m++)
            if(Symbols[m] == symbol) { InitSlDist[m] = slDist; break; }
   }
   else
   {
      PrintFormat("[%s] DRY-RUN: would %s %.2f lots sl=%.5f tp=%.5f",
                  symbol, action, lots, sl, tp);
      result = true;
   }

   if(result && LiveTrade)
   {
      double fillPrice = (action == "buy") ? tick.ask : tick.bid;
      // Use position ticket (available immediately); ResultDeal() returns 0 on STP brokers
      long posTicket = PositionSelect(symbol) ? (long)PositionGetInteger(POSITION_TICKET) : 0;
      string fillPayload = StringFormat(
         "{\"request_id\":\"%s\",\"ticket\":%I64d,\"symbol\":\"%s\","
         "\"action\":\"%s\",\"fill_price\":%.6f,\"lots\":%.2f,"
         "\"sl\":%.6f,\"tp\":%.6f}",
         requestId, posTicket, symbol, action,
         fillPrice, lots, sl, tp
      );
      SendRequest("/confirm", fillPayload);
   }
   else if(!result && LiveTrade)
   {
      PrintFormat("[%s] Order failed: %d %s", symbol,
                  (int)Trade.ResultRetcode(), Trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
void ClosePosition(string symbol, string requestId)
{
   if(!PositionSelect(symbol)) return;

   double pnl    = PositionGetDouble(POSITION_PROFIT);
   long   ticket = (long)PositionGetInteger(POSITION_TICKET);
   double closePrice = PositionGetDouble(POSITION_PRICE_CURRENT);

   if(LiveTrade)
      Trade.PositionClose(symbol);

   for(int k = 0; k < SymbolCount; k++)
      if(Symbols[k] == symbol) { TP1Prices[k] = 0.0; PartialClosed[k] = false; InitSlDist[k] = 0.0; break; }

   string closePayload = StringFormat(
      "{\"ticket\":%I64d,\"symbol\":\"%s\",\"close_price\":%.6f,\"pnl\":%.2f,\"outcome\":\"manual\"}",
      ticket, symbol, closePrice, pnl
   );
   SendRequest("/trade_close",closePayload);
}

//+------------------------------------------------------------------+
string SendRequest(string endpoint, string jsonBody)
{
   string headers = "Content-Type: application/json\r\n";
   char   data[], result[];
   string responseHeaders;

   StringToCharArray(jsonBody, data, 0, StringLen(jsonBody));

   int code = WebRequest("POST", BridgeURL + endpoint, headers,
                         5000, data, result, responseHeaders);

   if(code == -1)
   {
      int err = GetLastError();
      if(err == 4014)
         Print("ERROR: Add '", BridgeURL, "' to MT5 → Tools → Options → Expert Advisors → Allowed URLs");
      else
         PrintFormat("WebRequest error %d for %s", err, endpoint);
      return("");
   }

   if(code != 200)
   {
      PrintFormat("Bridge HTTP %d for %s", code, endpoint);
      return("");
   }

   return(CharArrayToString(result));
}

//── Pip size per symbol (must match bridge strategy_params.json pip_sizes) ────
double GetPipSize(string symbol)
{
   if(symbol == "USDJPY") return 0.01;
   if(symbol == "XAUUSD") return 1.0;
   if(symbol == "SP500")  return 5.0;
   if(symbol == "GER40")  return 5.0;
   if(symbol == "BTCUSD") return 50.0;
   return 0.0001;   // EURUSD, GBPUSD, and other 4-decimal pairs
}

//── JSON helpers (minimal string-based extraction) ───────────────────────────

string JsonGetString(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start < 0) return("");
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return("");
   return(StringSubstr(json, start, end - start));
}

double JsonGetDouble(string json, string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return(0.0);
   start += StringLen(search);
   // Skip optional quote (in case value is string)
   if(StringSubstr(json, start, 1) == "\"") start++;
   int end = start;
   while(end < StringLen(json))
   {
      string ch = StringSubstr(json, end, 1);
      if(ch == "," || ch == "}" || ch == "\"" || ch == "]") break;
      end++;
   }
   return(StringToDouble(StringSubstr(json, start, end - start)));
}

//+------------------------------------------------------------------+
void SyncPositionsWithBridge()
{
   // Build JSON of all currently open positions and send to /sync_positions
   string posJson = "{\"positions\":{";
   bool first = true;

   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!PosInfo.SelectByIndex(i)) continue;
      string sym = PosInfo.Symbol();

      // Only our symbols
      int symIdx = -1;
      for(int k = 0; k < SymbolCount; k++)
         if(Symbols[k] == sym) { symIdx = k; break; }
      if(symIdx < 0) continue;

      // Repopulate InitSlDist so trailing survives EA restarts
      double sl = PosInfo.StopLoss();
      if(sl > 0.0 && InitSlDist[symIdx] == 0.0)
         InitSlDist[symIdx] = MathAbs(sl - PosInfo.PriceOpen());

      if(!first) posJson += ",";
      first = false;

      posJson += "\"" + sym + "\":{";
      posJson += "\"ticket\":"     + IntegerToString(PosInfo.Ticket()) + ",";
      posJson += "\"action\":\""   + (PosInfo.PositionType() == POSITION_TYPE_BUY ? "buy" : "sell") + "\",";
      posJson += "\"lots\":"       + DoubleToString(PosInfo.Volume(), 2) + ",";
      posJson += "\"fill_price\":" + DoubleToString(PosInfo.PriceOpen(), 6) + ",";
      posJson += "\"open_time\":"  + IntegerToString(PosInfo.Time());
      posJson += "}";
   }
   posJson += "}}";

   string response = SendRequest("/sync_positions", posJson);
   if(response != "")
      Print("Position sync sent to bridge: ", posJson);
   else
      Print("Position sync failed (bridge not running yet — will retry on next tick)");
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("AI_EA deinitialized. Reason: ", reason);
}
//+------------------------------------------------------------------+
