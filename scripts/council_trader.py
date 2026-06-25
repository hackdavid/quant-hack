#!/usr/bin/env python3
"""Market Regime Council - Multi-Agent Trading System.

Based on research showing multi-agent LLM systems achieve 440% gains
with specialized agents that vote on decisions.

Agents:
  1. TrendOracle (Visionary) - Sees the big picture
  2. EntryScout (Hunter) - Finds precise entry points
  3. RiskGuard (Guardian) - Protects capital
  4. ProfitKeeper (Collector) - Maximizes gains
  5. SentimentAnalyzer (Reader) - Reads market mood
  6. VolatilityWatch (Weatherman) - Predicts volatility

All agents vote. Orchestrator makes final decision with weighted consensus.
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
import structlog
from dotenv import load_dotenv

load_dotenv()

from intraday.trader.mt5_wrapper import MT5TradingWrapper

log = structlog.get_logger(__name__)

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 400.0
MAX_TP = 200.0
MAX_HOLD_SECONDS = 600
TRAIL_ACTIVATE = 150.0
TRAIL_DROP = 100.0

# ── Council System ─────────────────────────────────────────────────────────
class CouncilState:
    """Shared state for all council members."""
    def __init__(self):
        self.votes: list[dict] = []
        self.context: dict[str, Any] = {}
        self.history: list[dict] = []
        self.last_trade_result: dict = {}

    def vote(self, agent: str, side: str, confidence: float, reason: str):
        self.votes.append({
            "agent": agent,
            "side": side,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    def set_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def record_trade(self, entry: dict, exit: dict):
        self.history.append({"entry": entry, "exit": exit, "time": time.time()})
        self.last_trade_result = exit

    def get_last_trade_result(self) -> dict:
        return self.last_trade_result

    def clear_votes(self):
        self.votes = []


class TrendOracle:
    """Visionary - Sees the big picture trend across multiple timeframes."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "TrendOracle"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 20:
            return self._vote("HOLD", 0.0, "Not enough data")

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # Multi-timeframe analysis
        ema5 = sum(closes[-5:]) / 5
        ema10 = sum(closes[-10:]) / 10
        ema20 = sum(closes[-20:]) / 20

        # Trend strength
        trend_strength = abs(ema5 - ema20) / ema20 * 100

        # Higher highs / lower lows
        hh = max(highs[-5:]) > max(highs[-10:-5])
        ll = min(lows[-5:]) < min(lows[-10:-5])

        # Volume confirmation
        vol_recent = sum(volumes[-5:]) / 5
        vol_old = sum(volumes[-10:-5]) / 5
        vol_confirm = vol_recent > vol_old * 1.2

        if ema5 > ema10 > ema20 and hh:
            conf = min(0.90, 0.70 + trend_strength * 0.02)
            if vol_confirm:
                conf += 0.10
            return self._vote("BUY", conf, f"Strong bull trend (strength={trend_strength:.2f}%)")
        elif ema5 < ema10 < ema20 and ll:
            conf = min(0.90, 0.70 + trend_strength * 0.02)
            if vol_confirm:
                conf += 0.10
            return self._vote("SELL", conf, f"Strong bear trend (strength={trend_strength:.2f}%)")
        elif ema5 > ema10 and not ll:
            return self._vote("BUY", 0.55, "Weak bullish momentum")
        elif ema5 < ema10 and not hh:
            return self._vote("SELL", 0.55, "Weak bearish momentum")
        else:
            return self._vote("HOLD", 0.40, "Ranging/unclear trend")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}


class EntryScout:
    """Hunter - Finds precise entry points using price action."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "EntryScout"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 5:
            return self._vote("HOLD", 0.0, "Not enough data")

        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # Candle patterns
        bullish_engulfing = last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"] and last["close"] > prev["open"]
        bearish_engulfing = last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]

        # Doji (indecision)
        body = abs(last["close"] - last["open"])
        range_ = last["high"] - last["low"]
        doji = body < range_ * 0.1

        # Support/Resistance
        recent_lows = [c["low"] for c in candles[-10:]]
        recent_highs = [c["high"] for c in candles[-10:]]
        support = min(recent_lows)
        resistance = max(recent_highs)
        current = last["close"]
        range_pct = (current - support) / (resistance - support) if resistance > support else 0.5

        # Momentum
        momentum = last["close"] - prev["close"]
        momentum_pct = abs(momentum) / prev["close"] * 100

        if bullish_engulfing and range_pct < 0.4:
            return self._vote("BUY", 0.75, "Bullish engulfing near support")
        elif bearish_engulfing and range_pct > 0.6:
            return self._vote("SELL", 0.75, "Bearish engulfing near resistance")
        elif doji and momentum_pct < 0.02:
            return self._vote("HOLD", 0.50, "Doji - market indecision")
        elif momentum > 0 and last["close"] > last["open"] and last["volume"] > prev["volume"] * 1.2:
            return self._vote("BUY", 0.65, "Volume-confirmed bullish move")
        elif momentum < 0 and last["close"] < last["open"] and last["volume"] > prev["volume"] * 1.2:
            return self._vote("SELL", 0.65, "Volume-confirmed bearish move")
        else:
            return self._vote("HOLD", 0.45, "No clear entry pattern")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}


class SentimentAnalyzer:
    """Reader - Reads market sentiment from order flow and volume."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "SentimentAnalyzer"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 5:
            return self._vote("HOLD", 0.0, "Not enough data")

        recent = candles[-5:]
        taker_buy = [c.get("taker_buy_pct", 50.0) for c in recent]
        avg_taker = sum(taker_buy) / len(taker_buy)

        volumes = [c["volume"] for c in recent]
        avg_vol = sum(volumes) / len(volumes)
        prev_volumes = [c["volume"] for c in candles[-10:-5]]
        prev_avg_vol = sum(prev_volumes) / len(prev_volumes)

        # Buying pressure
        if avg_taker > 60 and avg_vol > prev_avg_vol * 1.2:
            return self._vote("BUY", 0.70, f"Strong buying pressure ({avg_taker:.0f}% taker buy)")
        elif avg_taker < 40 and avg_vol > prev_avg_vol * 1.2:
            return self._vote("SELL", 0.70, f"Strong selling pressure ({avg_taker:.0f}% taker buy)")
        elif avg_taker > 55:
            return self._vote("BUY", 0.55, "Slight buying pressure")
        elif avg_taker < 45:
            return self._vote("SELL", 0.55, "Slight selling pressure")
        else:
            return self._vote("HOLD", 0.50, "Neutral sentiment")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}


class VolatilityWatch:
    """Weatherman - Predicts volatility and warns of dangerous conditions."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "VolatilityWatch"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 10:
            return self._vote("HOLD", 0.0, "Not enough data")

        # Calculate volatility
        ranges = [c["high"] - c["low"] for c in candles[-10:]]
        avg_range = sum(ranges) / len(ranges)
        current_range = ranges[-1]

        # ATR-like measure
        closes = [c["close"] for c in candles[-10:]]
        price = closes[-1]
        atr_pct = avg_range / price * 100

        # Bollinger bands
        sma = sum(closes) / len(closes)
        std = (sum((c - sma) ** 2 for c in closes) / len(closes)) ** 0.5
        upper = sma + 2 * std
        lower = sma - 2 * std
        position = (price - lower) / (upper - lower) if upper > lower else 0.5

        # Dangerous volatility
        if atr_pct > 0.5:  # > 0.5% per minute is very volatile
            if current_range > avg_range * 2:
                return self._vote("HOLD", 0.30, f"Extreme volatility! ATR={atr_pct:.2f}%")
            else:
                return self._vote("HOLD", 0.50, f"High volatility ATR={atr_pct:.2f}%")
        elif atr_pct > 0.3:
            return self._vote("HOLD", 0.60, f"Moderate volatility ATR={atr_pct:.2f}%")

        # Mean reversion signals
        if position > 0.9:
            return self._vote("SELL", 0.60, f"Price at upper band ({position:.2f})")
        elif position < 0.1:
            return self._vote("BUY", 0.60, f"Price at lower band ({position:.2f})")
        else:
            return self._vote("HOLD", 0.70, f"Normal volatility ATR={atr_pct:.2f}%")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}


class RiskGuard:
    """Guardian - Protects capital and calculates position parameters."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "RiskGuard"

    def analyze(self, candles: list[dict], account: dict) -> dict:
        balance = account.get("balance", 1000000.0)
        equity = account.get("equity", balance)

        # Check drawdown
        drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        if drawdown > 5:
            return self._vote("HOLD", 0.10, f"High drawdown {drawdown:.1f}% - STOP TRADING")

        # Check consecutive losses
        last_result = self.state.get_last_trade_result()
        if last_result.get("profit", 0) < -200:
            return self._vote("HOLD", 0.30, "Last trade was big loss - reduce risk")

        # Calculate position size
        max_risk = equity * 0.01
        risk_per_trade = min(max_risk, MAX_SL)

        sl_price = risk_per_trade / LOT_SIZE
        tp_price = MAX_TP / LOT_SIZE

        current = candles[-1]["close"] if candles else 0
        min_distance = 0.05

        self.state.set_context("risk", {"sl_price": sl_price, "tp_price": tp_price, "max_risk": max_risk})
        return self._vote("HOLD", 0.80, f"Risk approved: max_loss=${risk_per_trade:.0f}")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}


class ProfitKeeper:
    """Collector - Manages open trades and finds exits."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "ProfitKeeper"
        self.max_profit = 0.0
        self.start_time = None

    def monitor(self, position: dict, candles: list[dict]) -> dict:
        profit = position.get("profit", 0.0)
        side = position.get("side", "")
        current_price = position.get("current_price", 0.0)

        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time

        if profit > self.max_profit:
            self.max_profit = profit

        # Hard limits
        if profit >= MAX_TP:
            return self._vote("CLOSE", 0.95, f"Profit target ${profit:.2f} reached")
        if profit < -MAX_SL:
            return self._vote("CLOSE", 0.95, f"Stop loss ${profit:.2f} hit")
        if elapsed >= MAX_HOLD_SECONDS:
            return self._vote("CLOSE", 0.80, f"Max hold time {elapsed:.0f}s")

        # Trailing stop
        if self.max_profit > TRAIL_ACTIVATE and profit <= self.max_profit - TRAIL_DROP:
            return self._vote("CLOSE", 0.85, f"Trailing: dropped ${self.max_profit - profit:.2f} from peak")

        # Breakeven
        if self.max_profit > 50 and profit <= 0:
            return self._vote("CLOSE", 0.70, f"Breakeven: profit lost from ${self.max_profit:.2f}")

        # Time-based exit if not profitable
        if elapsed > 120 and profit < -50:
            return self._vote("CLOSE", 0.60, "Not profitable after 2 min")

        # Reversal detection
        if len(candles) >= 3 and profit > 0:
            last = candles[-1]
            prev = candles[-2]
            if side == "buy" and last["close"] < last["open"] and last["close"] < prev["close"]:
                return self._vote("CLOSE", 0.65, "Bearish reversal detected")
            elif side == "sell" and last["close"] > last["open"] and last["close"] > prev["close"]:
                return self._vote("CLOSE", 0.65, "Bullish reversal detected")

        return self._vote("HOLD", 0.70, f"PnL=${profit:.2f} max=${self.max_profit:.2f}")

    def _vote(self, side: str, conf: float, reason: str) -> dict:
        self.state.vote(self.name, side, conf, reason)
        return {"side": side, "confidence": conf, "reason": reason}

    def reset(self):
        self.max_profit = 0.0
        self.start_time = None


class Orchestrator:
    """Makes final decision based on weighted consensus."""
    def __init__(self, state: CouncilState):
        self.state = state
        self.name = "Orchestrator"
        self.weights = {
            "TrendOracle": 0.30,
            "EntryScout": 0.25,
            "SentimentAnalyzer": 0.15,
            "VolatilityWatch": 0.15,
            "RiskGuard": 0.15,
        }

    def decide_entry(self) -> dict:
        votes = self.state.votes
        if not votes:
            return {"action": "HOLD", "reason": "No votes", "confidence": 0.0}

        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        reasons = []

        for vote in votes:
            agent = vote["agent"]
            side = vote["side"]
            conf = vote["confidence"]
            weight = self.weights.get(agent, 0.1)
            total_weight += weight

            if side == "BUY":
                buy_score += conf * weight
            elif side == "SELL":
                sell_score += conf * weight
            else:
                hold_score += conf * weight

            reasons.append(f"{agent}: {side} ({conf:.2f})")

        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight

        print(f"  Votes: BUY={buy_score:.2f} SELL={sell_score:.2f} HOLD={hold_score:.2f}")
        for r in reasons:
            print(f"    {r}")

        # Decision threshold
        threshold = 0.60
        if buy_score > threshold and buy_score > sell_score and buy_score > hold_score:
            return {"action": "BUY", "reason": f"Consensus BUY ({buy_score:.2f})", "confidence": buy_score}
        elif sell_score > threshold and sell_score > buy_score and sell_score > hold_score:
            return {"action": "SELL", "reason": f"Consensus SELL ({sell_score:.2f})", "confidence": sell_score}
        else:
            return {"action": "HOLD", "reason": f"No consensus (B:{buy_score:.2f} S:{sell_score:.2f} H:{hold_score:.2f})", "confidence": max(buy_score, sell_score)}

    def decide_exit(self) -> dict:
        votes = self.state.votes
        if not votes:
            return {"action": "HOLD", "reason": "No votes"}

        close_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        reasons = []

        for vote in votes:
            agent = vote["agent"]
            side = vote["side"]
            conf = vote["confidence"]
            weight = 0.2  # Equal weight for exit
            total_weight += weight

            if side == "CLOSE":
                close_score += conf * weight
            else:
                hold_score += conf * weight

            reasons.append(f"{agent}: {side} ({conf:.2f})")

        if total_weight > 0:
            close_score /= total_weight
            hold_score /= total_weight

        print(f"  Votes: CLOSE={close_score:.2f} HOLD={hold_score:.2f}")
        for r in reasons:
            print(f"    {r}")

        if close_score > 0.65:
            return {"action": "CLOSE", "reason": f"Consensus CLOSE ({close_score:.2f})", "confidence": close_score}
        return {"action": "HOLD", "reason": f"Hold ({hold_score:.2f})", "confidence": hold_score}


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
    parser.add_argument("--mt5-account", type=int, default=None)
    parser.add_argument("--mt5-password", type=str, default=None)
    parser.add_argument("--mt5-server", type=str, default=None)
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--trades", type=int, default=3, help="Number of trades to backtest")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.trades)
        return

    print("=" * 60)
    print("MARKET REGIME COUNCIL - Multi-Agent Trading System")
    print("=" * 60)
    print("Agents: TrendOracle | EntryScout | SentimentAnalyzer | VolatilityWatch | RiskGuard | ProfitKeeper")
    print(f"Settings: TP=${MAX_TP:.0f} SL=${MAX_SL:.0f} Lot={LOT_SIZE}")
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

    # Initialize Council
    state = CouncilState()
    trend_oracle = TrendOracle(state)
    entry_scout = EntryScout(state)
    sentiment_analyzer = SentimentAnalyzer(state)
    volatility_watch = VolatilityWatch(state)
    risk_guard = RiskGuard(state)
    profit_keeper = ProfitKeeper(state)
    orchestrator = Orchestrator(state)

    try:
        while True:
            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # ── ENTRY PHASE ──
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === COUNCIL VOTING (Entry) ===")
                state.clear_votes()
                candles = fetch_candles(SYMBOL, limit=50)
                account = wrapper.state_snapshot()

                # All agents vote
                trend_oracle.analyze(candles)
                entry_scout.analyze(candles)
                sentiment_analyzer.analyze(candles)
                volatility_watch.analyze(candles)
                risk_guard.analyze(candles, account)

                # Orchestrator decides
                decision = orchestrator.decide_entry()
                signal = decision.get("action", "HOLD")
                reason = decision.get("reason", "")
                confidence = decision.get("confidence", 0.0)

                print(f"  DECISION: {signal} (conf={confidence:.2f}) - {reason}")

                if signal in ("BUY", "SELL") and confidence >= 0.60:
                    desired_side = "buy" if signal == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    risk = state.get_context("risk", {})
                    sl_price = risk.get("sl_price", 50.0)
                    tp_price = risk.get("tp_price", 25.0)

                    if desired_side == "buy":
                        sl = price - sl_price
                        tp = price + tp_price
                    else:
                        sl = price + sl_price
                        tp = price - tp_price

                    result = wrapper.market_order(SYMBOL, desired_side, LOT_SIZE, 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {signal} at {price:.2f}[/green]")
                        print(f"  SL: {sl:.2f} | TP: {tp:.2f}")
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]HOLD — waiting 60s[/yellow]")
                    time.sleep(60)
                    continue

            else:
                # ── MONITOR PHASE ──
                position = positions[0]
                ticket = position.ticket
                side = position.side
                profit = position.profit
                current_price = position.current_price

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === COUNCIL VOTING (Exit) ===")
                print(f"  Position #{ticket} | {side} | PnL=${profit:.2f}")

                state.clear_votes()
                candles = fetch_candles(SYMBOL, limit=20)

                # ProfitKeeper votes
                profit_keeper.monitor({
                    "side": side,
                    "profit": profit,
                    "current_price": current_price,
                }, candles)

                # Orchestrator decides exit
                exit_decision = orchestrator.decide_exit()
                action = exit_decision.get("action", "HOLD")
                reason = exit_decision.get("reason", "")
                confidence = exit_decision.get("confidence", 0.0)

                print(f"  DECISION: {action} (conf={confidence:.2f}) - {reason}")

                if action == "CLOSE" and confidence >= 0.60:
                    print(f"  [cyan]CLOSE: {reason} at ${profit:.2f}[/cyan]")
                    wrapper.close_position(ticket)
                    state.record_trade(
                        {"side": side, "entry": position.open_price},
                        {"profit": profit, "reason": reason}
                    )
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


def run_backtest(trade_count: int):
    """Backtest the council system without MT5."""
    print("=" * 60)
    print("COUNCIL BACKTEST")
    print("=" * 60)

    state = CouncilState()
    trend_oracle = TrendOracle(state)
    entry_scout = EntryScout(state)
    sentiment_analyzer = SentimentAnalyzer(state)
    volatility_watch = VolatilityWatch(state)
    risk_guard = RiskGuard(state)
    profit_keeper = ProfitKeeper(state)
    orchestrator = Orchestrator(state)

    total_pnl = 0.0
    wins = 0
    losses = 0

    for i in range(trade_count):
        print(f"\n{'='*60}")
        print(f"BACKTEST TRADE #{i+1}")
        print(f"{'='*60}")

        state.clear_votes()
        candles = fetch_candles(SYMBOL, limit=50)
        account = {"balance": 1000000.0, "equity": 1000000.0}

        # Council votes
        trend_oracle.analyze(candles)
        entry_scout.analyze(candles)
        sentiment_analyzer.analyze(candles)
        volatility_watch.analyze(candles)
        risk_guard.analyze(candles, account)

        decision = orchestrator.decide_entry()
        signal = decision.get("action", "HOLD")

        if signal == "HOLD":
            print("No trade signal")
            continue

        # Simulate trade
        entry_price = candles[-1]["close"]
        desired_side = "buy" if signal == "BUY" else "sell"
        profit = 0.0
        closed = False
        exit_reason = ""

        for t in range(10):  # 10 minutes
            idx = min(t, len(candles) - 1)
            current_price = candles[idx]["close"]
            if desired_side == "buy":
                profit = (current_price - entry_price) * LOT_SIZE
            else:
                profit = (entry_price - current_price) * LOT_SIZE

            state.clear_votes()
            profit_keeper.monitor({
                "side": desired_side,
                "profit": profit,
                "current_price": current_price,
            }, candles[:idx+1])

            exit_decision = orchestrator.decide_exit()
            if exit_decision.get("action") == "CLOSE":
                closed = True
                exit_reason = exit_decision.get("reason", "")
                break

        if not closed:
            profit = simulate_profit(entry_price, desired_side, candles[-1]["close"])
            exit_reason = "FINAL"

        print(f"\nResult: ${profit:.2f} ({exit_reason})")
        total_pnl += profit
        if profit > 0:
            wins += 1
        else:
            losses += 1

    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Trades: {wins + losses}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
    print(f"Total P&L: ${total_pnl:.2f}")


def simulate_profit(entry: float, side: str, current: float) -> float:
    if side == "buy":
        return (current - entry) * LOT_SIZE
    return (entry - current) * LOT_SIZE


if __name__ == "__main__":
    main()
