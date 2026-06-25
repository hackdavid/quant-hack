#!/usr/bin/env python3
"""Multi-Agent LLM Trading System with Specialized Personas.

Agents communicate in real-time to make trading decisions:
  1. TrendOracle - Identifies market trend (bull/bear/ranging)
  2. EntryScout - Finds optimal entry points
  3. RiskGuard - Manages position size, SL/TP
  4. ProfitKeeper - Monitors open trades and exits
  5. Orchestrator - Final decision maker

Usage:
    .venv/Scripts/python.exe scripts/multi_agent_trader.py
        --transformer-run models/transformer/20260623T132957Z
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER"
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import structlog
from dotenv import load_dotenv

load_dotenv()

from intraday.trader.mt5_wrapper import MT5TradingWrapper

log = structlog.get_logger(__name__)

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 400.0        # Max loss per trade
MAX_TP = 200.0        # Profit target
MAX_HOLD_SECONDS = 600  # 10 minutes
TRAIL_ACTIVATE = 150.0  # Trailing stop activates
TRAIL_DROP = 100.0     # Close if drops $100 from peak

# ── Agent System ───────────────────────────────────────────────────────────
class AgentState:
    """Shared state that all agents can read/write."""
    def __init__(self):
        self.data: dict[str, Any] = {}
        self.messages: list[dict] = []  # Agent communication log
        self.last_update = time.time()

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.last_update = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def log(self, agent: str, message: str, decision: str = ""):
        self.messages.append({
            "agent": agent,
            "message": message,
            "decision": decision,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })


class TrendOracle:
    """Analyzes market trend using multiple timeframes."""
    def __init__(self, state: AgentState):
        self.state = state
        self.name = "TrendOracle"

    def analyze(self, candles: list[dict]) -> dict:
        """Determine trend: bull, bear, or ranging."""
        if len(candles) < 10:
            return {"trend": "unknown", "confidence": 0.0}

        # EMA 5 and 10
        closes = [c["close"] for c in candles]
        ema5 = sum(closes[-5:]) / 5
        ema10 = sum(closes[-10:]) / 10

        # Recent highs/lows
        recent_high = max(c["high"] for c in candles[-5:])
        recent_low = min(c["low"] for c in candles[-5:])
        prev_high = max(c["high"] for c in candles[-10:-5])
        prev_low = min(c["low"] for c in candles[-10:-5])

        # Higher highs / lower lows
        higher_highs = recent_high > prev_high
        lower_lows = recent_low < prev_low

        # Volume trend
        volumes = [c["volume"] for c in candles[-5:]]
        avg_vol = sum(volumes) / len(volumes)
        prev_volumes = [c["volume"] for c in candles[-10:-5]]
        prev_avg_vol = sum(prev_volumes) / len(prev_volumes)
        volume_increasing = avg_vol > prev_avg_vol * 1.2

        # Determine trend
        if ema5 > ema10 and higher_highs and not lower_lows:
            trend = "bull"
            confidence = 0.75 if volume_increasing else 0.60
        elif ema5 < ema10 and lower_lows and not higher_highs:
            trend = "bear"
            confidence = 0.75 if volume_increasing else 0.60
        else:
            trend = "ranging"
            confidence = 0.50

        self.state.log(self.name, f"EMA5={ema5:.0f} EMA10={ema10:.0f} HH={higher_highs} LL={lower_lows}", trend)
        self.state.set("trend", {"trend": trend, "confidence": confidence, "ema5": ema5, "ema10": ema10})
        return self.state.get("trend")


class EntryScout:
    """Finds optimal entry points based on price action."""
    def __init__(self, state: AgentState):
        self.state = state
        self.name = "EntryScout"

    def find_entry(self, candles: list[dict]) -> dict:
        """Find entry signal: BUY, SELL, or WAIT."""
        trend = self.state.get("trend", {"trend": "unknown", "confidence": 0.0})
        trend_dir = trend.get("trend", "unknown")
        trend_conf = trend.get("confidence", 0.0)

        if trend_dir == "unknown":
            return {"signal": "HOLD", "confidence": 0.0, "reason": "No trend"}

        if len(candles) < 5:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Not enough data"}

        # Get last 3 candles
        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # Price momentum
        momentum = last["close"] - prev["close"]
        momentum_pct = abs(momentum) / prev["close"] * 100

        # Volume confirmation
        vol_increase = last["volume"] > prev["volume"] * 1.1

        # Candle pattern
        is_bullish = last["close"] > last["open"]
        is_bearish = last["close"] < last["open"]

        # Support/Resistance levels
        recent_lows = [c["low"] for c in candles[-10:]]
        recent_highs = [c["high"] for c in candles[-10:]]
        support = min(recent_lows)
        resistance = max(recent_highs)
        current = last["close"]

        near_support = (current - support) / (resistance - support) < 0.3
        near_resistance = (current - support) / (resistance - support) > 0.7

        # Decision logic
        signal = "HOLD"
        confidence = 0.0
        reason = ""

        if trend_dir == "bull":
            if is_bullish and momentum > 0 and near_support:
                signal = "BUY"
                confidence = 0.70 + (0.10 if vol_increase else 0.0)
                reason = "Bullish trend + bounce from support"
            elif is_bullish and momentum > 0 and momentum_pct > 0.05:
                signal = "BUY"
                confidence = 0.65
                reason = "Bullish momentum continuation"
            else:
                reason = "No clear bullish entry"
        elif trend_dir == "bear":
            if is_bearish and momentum < 0 and near_resistance:
                signal = "SELL"
                confidence = 0.70 + (0.10 if vol_increase else 0.0)
                reason = "Bearish trend + rejection at resistance"
            elif is_bearish and momentum < 0 and momentum_pct > 0.05:
                signal = "SELL"
                confidence = 0.65
                reason = "Bearish momentum continuation"
            else:
                reason = "No clear bearish entry"
        else:
            reason = "Ranging market - no directional entry"

        self.state.log(self.name, f"Pattern={is_bullish}/{is_bearish} Momentum={momentum:.0f} Near S/R={near_support}/{near_resistance}", signal)
        self.state.set("entry", {"signal": signal, "confidence": confidence, "reason": reason})
        return self.state.get("entry")


class RiskGuard:
    """Manages risk and calculates position parameters."""
    def __init__(self, state: AgentState):
        self.state = state
        self.name = "RiskGuard"

    def evaluate(self, entry_price: float, signal: str, account: dict) -> dict:
        """Calculate safe position size and SL/TP."""
        balance = account.get("balance", 1000000.0)
        equity = account.get("equity", balance)

        # Risk 1% of equity per trade
        max_risk = equity * 0.01
        risk_per_trade = min(max_risk, MAX_SL)

        # Calculate SL/TP in price terms
        sl_price = risk_per_trade / LOT_SIZE
        tp_price = MAX_TP / LOT_SIZE

        if signal == "BUY":
            sl = entry_price - sl_price
            tp = entry_price + tp_price
        else:
            sl = entry_price + sl_price
            tp = entry_price - tp_price

        # Check if SL/TP are valid (minimum distance)
        min_distance = 0.05
        if abs(sl - entry_price) < min_distance:
            sl = entry_price - min_distance if signal == "BUY" else entry_price + min_distance
        if abs(tp - entry_price) < min_distance:
            tp = entry_price + min_distance if signal == "SELL" else entry_price - min_distance

        self.state.log(self.name, f"Risk=${risk_per_trade:.0f} SL={sl:.2f} TP={tp:.2f}", "APPROVED")
        self.state.set("risk", {
            "sl": sl,
            "tp": tp,
            "max_risk": risk_per_trade,
            "approved": True,
        })
        return self.state.get("risk")


class ProfitKeeper:
    """Monitors open trades and manages exits."""
    def __init__(self, state: AgentState):
        self.state = state
        self.name = "ProfitKeeper"
        self.max_profit = 0.0
        self.start_time = None

    def monitor(self, position: dict, candles: list[dict]) -> dict:
        """Decide whether to close or hold the trade."""
        profit = position.get("profit", 0.0)
        current_price = position.get("current_price", 0.0)
        side = position.get("side", "")

        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time

        # Track max profit
        if profit > self.max_profit:
            self.max_profit = profit

        self.state.set("max_profit", self.max_profit)
        self.state.set("elapsed", elapsed)

        # Check hard limits
        if profit >= MAX_TP:
            self.state.log(self.name, f"Profit ${profit:.2f} >= ${MAX_TP:.0f} target", "CLOSE")
            return {"action": "CLOSE", "reason": "PROFIT_TARGET", "profit": profit}

        if profit < -MAX_SL:
            self.state.log(self.name, f"Loss ${profit:.2f} < -${MAX_SL:.0f} limit", "CLOSE")
            return {"action": "CLOSE", "reason": "STOP_LOSS", "profit": profit}

        if elapsed >= MAX_HOLD_SECONDS:
            self.state.log(self.name, f"Max hold time {elapsed:.0f}s reached", "CLOSE")
            return {"action": "CLOSE", "reason": "TIMEOUT", "profit": profit}

        # Trailing stop
        if self.max_profit > TRAIL_ACTIVATE and profit <= self.max_profit - TRAIL_DROP:
            self.state.log(self.name, f"Trailing: ${profit:.2f} dropped from ${self.max_profit:.2f}", "CLOSE")
            return {"action": "CLOSE", "reason": "TRAILING_STOP", "profit": profit}

        # Breakeven stop: once profit > $50, move SL to breakeven
        if self.max_profit > 50 and profit <= 0:
            self.state.log(self.name, f"Breakeven: profit ${profit:.2f} dropped from ${self.max_profit:.2f}", "CLOSE")
            return {"action": "CLOSE", "reason": "BREAKEVEN", "profit": profit}

        # Analyze price action for early exit
        if len(candles) >= 3:
            last = candles[-1]
            prev = candles[-2]
            # Reversal candle against position
            if side == "buy" and last["close"] < last["open"] and last["close"] < prev["close"]:
                if profit > 0 and profit < self.max_profit - 30:
                    self.state.log(self.name, f"Bearish reversal at ${profit:.2f}", "CLOSE")
                    return {"action": "CLOSE", "reason": "REVERSAL", "profit": profit}
            elif side == "sell" and last["close"] > last["open"] and last["close"] > prev["close"]:
                if profit > 0 and profit < self.max_profit - 30:
                    self.state.log(self.name, f"Bullish reversal at ${profit:.2f}", "CLOSE")
                    return {"action": "CLOSE", "reason": "REVERSAL", "profit": profit}

        self.state.log(self.name, f"PnL=${profit:.2f} max=${self.max_profit:.2f} elapsed={elapsed:.0f}s", "HOLD")
        return {"action": "HOLD", "reason": "MONITOR", "profit": profit}

    def reset(self):
        self.max_profit = 0.0
        self.start_time = None


class Orchestrator:
    """Makes final trading decisions based on all agent inputs."""
    def __init__(self, state: AgentState):
        self.state = state
        self.name = "Orchestrator"

    def decide_entry(self) -> dict:
        """Decide whether to enter a trade."""
        trend = self.state.get("trend", {})
        entry = self.state.get("entry", {})
        risk = self.state.get("risk", {})

        trend_dir = trend.get("trend", "unknown")
        trend_conf = trend.get("confidence", 0.0)
        signal = entry.get("signal", "HOLD")
        entry_conf = entry.get("confidence", 0.0)
        risk_approved = risk.get("approved", False)

        # All agents must agree
        if trend_dir == "unknown":
            self.state.log(self.name, "Trend unclear", "HOLD")
            return {"action": "HOLD", "reason": "No trend"}

        if signal == "HOLD":
            self.state.log(self.name, "No entry signal", "HOLD")
            return {"action": "HOLD", "reason": "No signal"}

        if trend_conf < 0.60:
            self.state.log(self.name, f"Trend confidence {trend_conf:.2f} too low", "HOLD")
            return {"action": "HOLD", "reason": "Low trend confidence"}

        if entry_conf < 0.60:
            self.state.log(self.name, f"Entry confidence {entry_conf:.2f} too low", "HOLD")
            return {"action": "HOLD", "reason": "Low entry confidence"}

        if not risk_approved:
            self.state.log(self.name, "Risk not approved", "HOLD")
            return {"action": "HOLD", "reason": "Risk blocked"}

        # Trend-direction alignment
        if trend_dir == "bull" and signal == "SELL":
            self.state.log(self.name, "Trend is bull but signal is SELL", "HOLD")
            return {"action": "HOLD", "reason": "Trend mismatch"}
        if trend_dir == "bear" and signal == "BUY":
            self.state.log(self.name, "Trend is bear but signal is BUY", "HOLD")
            return {"action": "HOLD", "reason": "Trend mismatch"}

        self.state.log(self.name, f"All agents agree: {signal}", signal)
        return {"action": signal, "reason": "All agents agree", "confidence": min(trend_conf, entry_conf)}

    def decide_exit(self, monitor_result: dict) -> dict:
        """Decide whether to exit."""
        action = monitor_result.get("action", "HOLD")
        reason = monitor_result.get("reason", "")
        profit = monitor_result.get("profit", 0.0)

        if action == "CLOSE":
            self.state.log(self.name, f"ProfitKeeper says CLOSE: {reason}", "CLOSE")
            return {"action": "CLOSE", "reason": reason, "profit": profit}

        self.state.log(self.name, "Holding position", "HOLD")
        return {"action": "HOLD", "reason": "Hold", "profit": profit}


# ── Main Bot ───────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, limit: int = 50) -> list[dict]:
    """Fetch 1m candles."""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = httpx.get(url, params={"symbol": symbol.upper(), "interval": "1m", "limit": limit}, timeout=30.0)
    r.raise_for_status()
    candles = []
    for row in r.json():
        candles.append({
            "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
            "close": float(row[4]), "volume": float(row[5]),
            "trades": int(row[8]), "taker_buy_pct": (float(row[9]) / float(row[5]) * 100) if float(row[5]) > 0 else 50.0,
        })
    return candles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt5-account", type=int, required=True)
    parser.add_argument("--mt5-password", type=str, required=True)
    parser.add_argument("--mt5-server", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-AGENT TRADING SYSTEM")
    print("=" * 60)
    print("Agents: TrendOracle | EntryScout | RiskGuard | ProfitKeeper")
    print(f"Settings: TP=${MAX_TP:.0f} SL=${MAX_SL:.0f} Trailing=${TRAIL_DROP:.0f}")
    print("=" * 60)

    # Connect to MT5
    wrapper = MT5TradingWrapper(
        account_id=args.mt5_account,
        password=args.mt5_password,
        server=args.mt5_server,
        magic=999999,
    )
    if not wrapper.connect():
        print("[red]Failed to connect to MT5[/red]")
        sys.exit(1)

    print("[green]MT5 Connected[/green]")
    print("[cyan]Press Ctrl+C to stop[/cyan]\n")

    # Initialize agent system
    state = AgentState()
    trend_oracle = TrendOracle(state)
    entry_scout = EntryScout(state)
    risk_guard = RiskGuard(state)
    profit_keeper = ProfitKeeper(state)
    orchestrator = Orchestrator(state)

    try:
        while True:
            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # ── NO TRADE: Entry analysis ──
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- ENTRY ANALYSIS ---")
                candles = fetch_candles(SYMBOL, limit=50)
                account = wrapper.state_snapshot()

                # Run agents
                trend = trend_oracle.analyze(candles)
                entry = entry_scout.find_entry(candles)
                risk = risk_guard.evaluate(candles[-1]["close"], entry.get("signal", "HOLD"), account)

                # Orchestrator decides
                decision = orchestrator.decide_entry()
                signal = decision.get("action", "HOLD")
                reason = decision.get("reason", "")
                confidence = decision.get("confidence", 0.0)

                print(f"  Trend: {trend['trend']} (conf={trend['confidence']:.2f})")
                print(f"  Entry: {entry['signal']} (conf={entry['confidence']:.2f}) - {entry['reason']}")
                print(f"  Risk: SL={risk['sl']:.2f} TP={risk['tp']:.2f}")
                print(f"  DECISION: {signal} (conf={confidence:.2f}) - {reason}")

                # Show agent communication
                for msg in state.messages:
                    print(f"    [{msg['timestamp']}] {msg['agent']}: {msg['message']} -> {msg['decision']}")

                if signal in ("BUY", "SELL"):
                    desired_side = "buy" if signal == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    result = wrapper.market_order(SYMBOL, desired_side, LOT_SIZE, risk['sl'], risk['tp'])
                    if result.success:
                        print(f"  [green]Trade opened: {signal} at {price:.2f}[/green]")
                        print(f"  SL: {risk['sl']:.2f} | TP: {risk['tp']:.2f}")
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]HOLD — waiting 60s[/yellow]")
                    time.sleep(60)
                    continue

            else:
                # ── TRADE OPEN: Monitor ──
                position = positions[0]
                ticket = position.ticket
                side = position.side
                profit = position.profit
                open_price = position.open_price
                current_price = position.current_price
                volume = position.volume

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring #{ticket} | {side} | PnL=${profit:.2f}")

                candles = fetch_candles(SYMBOL, limit=20)

                # Run ProfitKeeper
                monitor_result = profit_keeper.monitor({
                    "side": side,
                    "profit": profit,
                    "open_price": open_price,
                    "current_price": current_price,
                    "volume": volume,
                }, candles)

                # Orchestrator decides exit
                exit_decision = orchestrator.decide_exit(monitor_result)
                action = exit_decision.get("action", "HOLD")
                reason = exit_decision.get("reason", "")

                if action == "CLOSE":
                    print(f"  [cyan]CLOSE: {reason} at ${profit:.2f}[/cyan]")
                    wrapper.close_position(ticket)
                    profit_keeper.reset()
                    time.sleep(60)
                    continue
                else:
                    print(f"  [green]HOLD: PnL=${profit:.2f} max=${profit_keeper.max_profit:.2f}[/green]")

                time.sleep(10)

    except KeyboardInterrupt:
        print("\n[yellow]Stopping...[/yellow]")
        positions = wrapper.get_positions(SYMBOL)
        for p in positions:
            wrapper.close_position(p.ticket)
        wrapper.shutdown()
        print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
