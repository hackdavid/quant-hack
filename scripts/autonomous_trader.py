#!/usr/bin/env python3
"""Autonomous trading loop: Binance WebSocket data → LLM review → MT5 execution.

Architecture:
    Binance WS (5-min bars) → Feature buffer → RL/Rule signal → LLM risk review
        → Risk validation → MT5 execution → Trade logger

Usage:
    # Paper mode (no real orders)
    uv run python scripts/autonomous_trader.py \
        --mt5-account 10408 --mt5-password "..." --mt5-server "..." \
        --symbol BTCUSDT --capital 1000000 --mock-execution

    # Live mode (Windows only, real MT5 orders)
    uv run python scripts/autonomous_trader.py \
        --mt5-account 10408 --mt5-password "..." --mt5-server "..." \
        --symbol BTCUSDT --capital 1000000

    # With LLM review (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
    export OPENAI_API_KEY="sk-..."
    uv run python scripts/autonomous_trader.py --mt5-account ... --use-llm

The LLM receives a structured prompt with market context, risk state, and the RL signal.
It returns a fixed JSON schema:
    {
      "action": "BUY | SELL | HOLD",
      "confidence": 0.0-1.0,
      "reason": "...",
      "position_size": 0.0-1.0,
      "sl_price": float,
      "tp_price": float,
      "risk_approved": true
    }

Logs are written to `logs/autonomous_trader/trade_log_YYYY-MM-DD.jsonl`.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import structlog
import typer
import websockets
from rich import print as rprint

# Lazy imports for optional components
try:
    from intraday.trader.mt5_wrapper import MT5TradingWrapper
except ImportError:
    MT5TradingWrapper = None

log = structlog.get_logger(__name__)

app = typer.Typer()

# ── Constants ──────────────────────────────────────────────────────────────────

BINANCE_WS = "wss://fstream.binance.com/ws"
BAR_MS = 5 * 60 * 1000

LOG_DIR = Path("logs/autonomous_trader")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Risk constants from user spec
MAX_RISK_PER_TRADE_PCT = 0.5
MAX_DAILY_LOSS_PCT = 2.0
MAX_WEEKLY_LOSS_PCT = 5.0
SOFT_DD_PCT = 8.0
HARD_DD_PCT = 12.0
MAX_EXPOSURE_PCT = 25.0
MIN_TRADES_PER_DAY = 5
TARGET_TRADES_PER_DAY = 10
REJECT_CONFIDENCE = 0.65
MIN_RISK_REWARD = 2.0
PREFERRED_LEVERAGE = 1
ABS_MAX_LEVERAGE = 5


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeLog:
    ts: str
    ts_ms: int
    symbol: str
    action: str
    confidence: float
    reason: str
    position_size: float
    sl_price: float
    tp_price: float
    risk_approved: bool
    account_balance: float
    equity: float
    open_pl: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    trade_count_today: int
    bar_close: float
    bar_rsi: float
    bar_vol: float


@dataclass
class LLMReview:
    action: str
    confidence: float
    reason: str
    position_size: float
    sl_price: float
    tp_price: float
    risk_approved: bool


# ── TradeLogger ───────────────────────────────────────────────────────────────

class TradeLogger:
    """Append-only JSONL logger with last-N retrieval for LLM context."""

    def __init__(self, log_dir: Path = LOG_DIR) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.now(timezone.utc).date().isoformat()
        self._path = self.log_dir / f"trade_log_{self._today}.jsonl"
        self._entries: list[dict] = []

    def _rotate(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._today:
            self._today = today
            self._path = self.log_dir / f"trade_log_{self._today}.jsonl"
            self._entries = []

    def write(self, entry: TradeLog) -> None:
        self._rotate()
        d = asdict(entry)
        with open(self._path, "a") as f:
            f.write(json.dumps(d, default=str) + "\n")
        self._entries.append(d)
        log.info("trade_logged", action=entry.action, confidence=entry.confidence)

    def last_n(self, n: int = 50) -> list[dict]:
        """Return last N log entries for LLM context."""
        self._rotate()
        # Read from disk + memory
        all_entries: list[dict] = []
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))
        return all_entries[-n:] if len(all_entries) >= n else all_entries

    def summary(self) -> dict:
        """Return summary stats for LLM context."""
        entries = self.last_n(9999)
        if not entries:
            return {"trades_today": 0, "win_rate": 0.0, "avg_return": 0.0}
        returns = [e.get("open_pl", 0.0) for e in entries]
        wins = sum(1 for r in returns if r > 0)
        return {
            "trades_today": len(entries),
            "win_rate": round(wins / len(entries), 3),
            "avg_return": round(float(np.mean(returns)), 2),
            "max_drawdown_pct": max((e.get("max_drawdown_pct", 0.0) for e in entries), default=0.0),
        }


# ── Binance WebSocket Feed ───────────────────────────────────────────────────

class BinanceFeed:
    """Async WebSocket feed for Binance USDT-M futures klines."""

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m") -> None:
        self.symbol = symbol.lower()
        self.interval = interval
        self._buffer: deque[Bar] = deque(maxlen=200)
        self._running = False
        self._last_close: float = 0.0

    @property
    def buffer(self) -> list[Bar]:
        return list(self._buffer)

    def latest_df(self) -> pl.DataFrame | None:
        """Return last 128 bars as a Polars DataFrame."""
        if len(self._buffer) < 128:
            return None
        bars = list(self._buffer)[-128:]
        return pl.DataFrame({
            "bar_time_ms": [b.ts_ms for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        })

    async def run(self) -> None:
        stream = f"{self.symbol}@kline_{self.interval}"
        url = f"{BINANCE_WS}/{stream}"
        log.info("binance_ws_connect", url=url)
        self._running = True
        async with websockets.connect(url, ping_interval=20) as ws:
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("e") != "kline":
                    continue
                k = msg["k"]
                if k["x"]:  # bar is closed
                    bar = Bar(
                        ts_ms=int(k["t"]),
                        open=float(k["o"]),
                        high=float(k["h"]),
                        low=float(k["l"]),
                        close=float(k["c"]),
                        volume=float(k["v"]),
                    )
                    self._buffer.append(bar)
                    self._last_close = bar.close
                    log.info("bar_closed", ts_ms=bar.ts_ms, close=bar.close, volume=bar.volume)

    def stop(self) -> None:
        self._running = False


# ── MT5 Executor ────────────────────────────────────────────────────────────

class MT5Executor:
    """MT5 execution layer with mock mode for Linux."""

    def __init__(
        self,
        account: int,
        password: str,
        server: str,
        mock: bool = False,
        magic: int = 999999,
    ) -> None:
        self.account = account
        self.password = password
        self.server = server
        self.mock = mock
        self.magic = magic
        self._mt5: Any | None = None
        self._initial_capital: float = 0.0
        self._peak_equity: float = 0.0
        self._daily_start: float = 0.0
        self._trade_count_today: int = 0
        self._last_trade_date: str = ""

    def connect(self) -> bool:
        if self.mock:
            log.info("mt5_mock_mode", capital=self._initial_capital)
            return True
        if MT5TradingWrapper is None:
            log.error("mt5_not_available")
            return False
        self._mt5 = MT5TradingWrapper(
            account_id=self.account,
            password=self.password,
            server=self.server,
            magic=self.magic,
        )
        return self._mt5.connect()

    def shutdown(self) -> None:
        if self._mt5 and not self.mock:
            self._mt5.shutdown()

    def state(self) -> dict:
        if self.mock:
            return {
                "balance": self._initial_capital,
                "equity": self._initial_capital,
                "profit": 0.0,
                "margin": 0.0,
                "free_margin": self._initial_capital,
            }
        if self._mt5 is None:
            return {}
        s = self._mt5.account_state()
        return s.to_dict() if s else {}

    def get_positions(self, symbol: str) -> list[dict]:
        if self.mock:
            return []
        if self._mt5 is None:
            return []
        return [p.to_dict() for p in self._mt5.get_positions(symbol)]

    def position_count(self, symbol: str) -> int:
        return len(self.get_positions(symbol))

    def place_order(self, symbol: str, side: str, volume: float, sl: float, tp: float) -> dict:
        if self.mock:
            log.info("mock_order", symbol=symbol, side=side, volume=volume, sl=sl, tp=tp)
            return {"success": True, "ticket": 123456, "price": 0.0, "comment": "mock"}
        if self._mt5 is None:
            return {"success": False, "comment": "MT5 not connected"}
        result = self._mt5.market_order(symbol, side, volume=volume, sl=sl, tp=tp)
        return result.to_dict()

    def close_all(self, symbol: str) -> list[dict]:
        if self.mock:
            log.info("mock_close_all", symbol=symbol)
            return []
        if self._mt5 is None:
            return []
        return [r.to_dict() for r in self._mt5.close_all_positions(symbol)]

    def update_stats(self, capital: float) -> None:
        self._initial_capital = capital
        self._peak_equity = capital
        self._daily_start = capital

    def update_drawdown(self, equity: float) -> float:
        self._peak_equity = max(self._peak_equity, equity)
        return (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0

    def update_daily_trade_count(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._last_trade_date:
            self._last_trade_date = today
            self._trade_count_today = 0
        return self._trade_count_today

    def increment_trade_count(self) -> None:
        self.update_daily_trade_count()
        self._trade_count_today += 1


# ── LLM Review Agent ─────────────────────────────────────────────────────────

class LLMReviewAgent:
    """Calls an LLM to validate the trade signal. Falls back to rule-based if no API key."""

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm
        self._client: Any | None = None
        if use_llm:
            self._client = self._init_client()

    def _init_client(self) -> Any:
        """Try OpenAI, then Anthropic, then fallback."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                return openai.OpenAI(api_key=openai_key)
            except ImportError:
                pass
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=anthropic_key)
            except ImportError:
                pass
        log.warning("no_llm_api_key")
        return None

    def review(
        self,
        signal: str,
        confidence: float,
        bar: Bar,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
    ) -> LLMReview:
        """Get LLM review or fallback to rule-based."""
        if self._client is None or not self.use_llm:
            return self._rule_based(signal, confidence, bar, risk_state)

        prompt = self._build_prompt(signal, confidence, bar, positions, account, risk_state, recent_logs)
        try:
            return self._call_llm(prompt)
        except Exception as exc:
            log.error("llm_call_failed", error=str(exc))
            return self._rule_based(signal, confidence, bar, risk_state)

    def _build_prompt(
        self,
        signal: str,
        confidence: float,
        bar: Bar,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
    ) -> str:
        log_summary = TradeLogger().summary()
        recent_actions = "\n".join(
            f"- {e['ts']}: {e['action']} (conf={e['confidence']}, reason={e['reason']})"
            for e in recent_logs[-10:]
        )

        return f"""You are a quantitative risk committee for a live trading competition.

COMPETITION RULES:
- Initial capital: $1,000,000 | Max leverage: 30 | Stop-out: 30%
- Max risk per trade: 0.5% | Max daily loss: 2% | Max weekly loss: 5%
- Soft drawdown limit: 8% | Hard drawdown limit: 12%
- Max total exposure: 25% | Preferred leverage: 1 | Absolute max: 5
- Min trades per day: 5 | Target: 10
- NEVER martingale, NEVER average losers, NEVER remove stop loss
- Mandatory SL/TP using ATR method | Minimum risk/reward: 2.0

CURRENT STATE:
- Signal: {signal} | Confidence: {confidence}
- Bar close: {bar.close} | Volume: {bar.volume}
- Account balance: {account.get('balance', 0)} | Equity: {account.get('equity', 0)}
- Open positions: {len(positions)}
- Daily trades so far: {log_summary['trades_today']}
- Current drawdown: {risk_state.get('drawdown_pct', 0)}%
- Daily PnL: {risk_state.get('daily_pnl_pct', 0)}%

RECENT TRADES:
{recent_actions}

YOUR TASK:
Review the signal and return ONLY a JSON object with this exact schema:
{{
  "action": "BUY or SELL or HOLD",
  "confidence": 0.0-1.0,
  "reason": "One sentence explaining your decision",
  "position_size": 0.0-1.0,
  "sl_price": float,
  "tp_price": float,
  "risk_approved": true or false
}}

Rules for position_size:
- confidence < 0.60 → size = 0.0
- 0.60-0.75 → size = 0.25
- 0.75-0.85 → size = 0.50
- > 0.85 → size = 1.00

If risk rules are violated, set risk_approved=false and action=HOLD.
"""

    def _call_llm(self, prompt: str) -> LLMReview:
        """Call OpenAI or Anthropic."""
        if self._client is None:
            return self._rule_based("HOLD", 0.0, Bar(0, 0, 0, 0, 0, 0), {})

        # Try OpenAI
        if hasattr(self._client, "chat"):
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading risk officer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            text = resp.choices[0].message.content
        # Try Anthropic
        elif hasattr(self._client, "messages"):
            resp = self._client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        else:
            return self._rule_based("HOLD", 0.0, Bar(0, 0, 0, 0, 0, 0), {})

        # Extract JSON
        try:
            # Find JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text.strip())
        except Exception:
            log.warning("llm_json_parse_failed", text=text[:200])
            return self._rule_based("HOLD", 0.0, Bar(0, 0, 0, 0, 0, 0), {})

        return LLMReview(
            action=data.get("action", "HOLD").upper(),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", "LLM fallback"),
            position_size=float(data.get("position_size", 0.0)),
            sl_price=float(data.get("sl_price", 0.0)),
            tp_price=float(data.get("tp_price", 0.0)),
            risk_approved=bool(data.get("risk_approved", False)),
        )

    def _rule_based(
        self,
        signal: str,
        confidence: float,
        bar: Bar,
        risk_state: dict,
    ) -> LLMReview:
        """Fallback when LLM is unavailable."""
        dd = risk_state.get("drawdown_pct", 0)
        daily_pnl = risk_state.get("daily_pnl_pct", 0)
        trades = risk_state.get("trade_count_today", 0)

        # Hard stops
        if dd >= HARD_DD_PCT:
            return LLMReview("HOLD", 0.0, f"Hard drawdown limit hit ({dd:.1f}%)", 0.0, 0.0, 0.0, False)
        if daily_pnl <= -MAX_DAILY_LOSS_PCT:
            return LLMReview("HOLD", 0.0, f"Daily loss limit hit ({daily_pnl:.1f}%)", 0.0, 0.0, 0.0, False)
        if confidence < REJECT_CONFIDENCE:
            return LLMReview("HOLD", confidence, f"Confidence {confidence:.2f} below {REJECT_CONFIDENCE}", 0.0, 0.0, 0.0, False)

        # Position sizing by confidence
        if confidence < 0.60:
            size = 0.0
        elif confidence < 0.75:
            size = 0.25
        elif confidence < 0.85:
            size = 0.50
        else:
            size = 1.00

        # ATR-based SL/TP
        atr = self._estimate_atr(bar)
        sl = bar.close - atr * 2 if signal == "BUY" else bar.close + atr * 2
        tp = bar.close + atr * 4 if signal == "BUY" else bar.close - atr * 4

        # Risk/reward check
        risk = abs(bar.close - sl)
        reward = abs(tp - bar.close)
        rr = reward / risk if risk > 0 else 0
        if rr < MIN_RISK_REWARD:
            return LLMReview("HOLD", confidence, f"R/R {rr:.1f} below {MIN_RISK_REWARD}", 0.0, 0.0, 0.0, False)

        return LLMReview(
            action=signal.upper(),
            confidence=confidence,
            reason=f"Rule-based: {signal} conf={confidence:.2f} size={size} R/R={rr:.1f}",
            position_size=size,
            sl_price=round(sl, 2),
            tp_price=round(tp, 2),
            risk_approved=True,
        )

    @staticmethod
    def _estimate_atr(bar: Bar) -> float:
        """Simple ATR estimate from current bar."""
        return bar.high - bar.low if bar.high > bar.low else bar.close * 0.001


# ── Risk Manager ───────────────────────────────────────────────────────────

class RiskManager:
    """Enforces all risk constraints before execution."""

    def __init__(self, capital: float) -> None:
        self.capital = capital
        self.initial_capital = capital
        self.peak_equity = capital
        self.daily_start = capital
        self.weekly_start = capital
        self.last_trade_date = ""
        self.last_trade_week = ""
        self.trade_count_today = 0
        self.total_exposure = 0.0

    def update(self, equity: float) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        week = datetime.now(timezone.utc).isocalendar().week
        if today != self.last_trade_date:
            self.daily_start = equity
            self.trade_count_today = 0
            self.last_trade_date = today
        if str(week) != self.last_trade_week:
            self.weekly_start = equity
            self.last_trade_week = str(week)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        daily_pnl = (equity - self.daily_start) / self.daily_start * 100 if self.daily_start > 0 else 0.0
        weekly_pnl = (equity - self.weekly_start) / self.weekly_start * 100 if self.weekly_start > 0 else 0.0
        return {
            "drawdown_pct": drawdown * 100,
            "daily_pnl_pct": daily_pnl,
            "weekly_pnl_pct": weekly_pnl,
            "trade_count_today": self.trade_count_today,
            "total_exposure_pct": self.total_exposure / self.capital * 100,
        }

    def can_trade(self, review: LLMReview, risk_state: dict) -> tuple[bool, str]:
        """Return (ok, reason) after checking all risk rules."""
        if not review.risk_approved:
            return False, "LLM rejected risk"
        if risk_state["drawdown_pct"] >= HARD_DD_PCT:
            return False, f"Hard drawdown {risk_state['drawdown_pct']:.1f}%"
        if risk_state["daily_pnl_pct"] <= -MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss {risk_state['daily_pnl_pct']:.1f}%"
        if risk_state["weekly_pnl_pct"] <= -MAX_WEEKLY_LOSS_PCT:
            return False, f"Weekly loss {risk_state['weekly_pnl_pct']:.1f}%"
        if review.confidence < REJECT_CONFIDENCE:
            return False, f"Confidence {review.confidence:.2f} < {REJECT_CONFIDENCE}"
        if risk_state["total_exposure_pct"] + review.position_size * 100 > MAX_EXPOSURE_PCT:
            return False, f"Exposure would exceed {MAX_EXPOSURE_PCT}%"
        return True, "ok"

    def record_trade(self, notional: float) -> None:
        self.trade_count_today += 1
        self.total_exposure += notional


# ── Main Trading Loop ───────────────────────────────────────────────────────

class AutonomousTrader:
    """Main orchestrator: data → signal → LLM review → risk → execute → log."""

    def __init__(
        self,
        symbol: str,
        capital: float,
        mt5_account: int,
        mt5_password: str,
        mt5_server: str,
        mock_execution: bool = False,
        use_llm: bool = False,
        interval: int = 5,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.feed = BinanceFeed(symbol=symbol)
        self.executor = MT5Executor(
            account=mt5_account,
            password=mt5_password,
            server=mt5_server,
            mock=mock_execution,
        )
        self.llm = LLMReviewAgent(use_llm=use_llm)
        self.risk = RiskManager(capital=capital)
        self.logger = TradeLogger()
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        rprint("[green]Starting autonomous trader...[/green]")
        rprint(f"  Symbol: {self.symbol}")
        rprint(f"  Capital: {self.capital:,.2f}")
        rprint(f"  Mock execution: {self.executor.mock}")
        rprint(f"  LLM review: {self.llm.use_llm}")

        if not self.executor.connect():
            rprint("[red]Failed to connect to MT5[/red]")
            return

        self.executor.update_stats(self.capital)
        self._running = True

        # Start Binance feed in background
        self._task = asyncio.create_task(self.feed.run())
        rprint("[yellow]Waiting for 128 bars to warm up...[/yellow]")

        # Main loop
        while self._running:
            await asyncio.sleep(self.interval * 60)
            if not self._running:
                break
            try:
                await self._on_bar()
            except Exception as exc:
                log.error("main_loop_error", error=str(exc))
                traceback.print_exc()

    def stop(self) -> None:
        self._running = False
        self.feed.stop()
        if self._task:
            self._task.cancel()
        self.executor.shutdown()
        rprint("[red]Trader stopped.[/red]")

    async def _on_bar(self) -> None:
        if len(self.feed.buffer) < 128:
            log.info("warming_up", bars=len(self.feed.buffer))
            return

        bar = self.feed.buffer[-1]
        account = self.executor.state()
        equity = account.get("equity", self.capital)
        balance = account.get("balance", self.capital)

        risk_state = self.risk.update(equity)
        positions = self.executor.get_positions(self.symbol)
        recent_logs = self.logger.last_n(20)

        # Generate signal (simple rule-based for now; swap with RL later)
        signal, confidence = self._generate_signal(bar)

        # LLM review
        review = self.llm.review(
            signal=signal,
            confidence=confidence,
            bar=bar,
            positions=positions,
            account=account,
            risk_state=risk_state,
            recent_logs=recent_logs,
        )

        rprint(f"\n[bold]Signal: {signal} ({confidence:.2f})[/bold]")
        rprint(f"  LLM action: {review.action} (conf={review.confidence:.2f})")
        rprint(f"  Reason: {review.reason}")
        rprint(f"  Size: {review.position_size} | SL: {review.sl_price} | TP: {review.tp_price}")

        # Risk validation
        ok, reason = self.risk.can_trade(review, risk_state)
        if not ok:
            rprint(f"[red]Risk block: {reason}[/red]")
            self._log(bar, review, account, risk_state, reason)
            return

        # Execute
        if review.action in ("BUY", "SELL") and review.position_size > 0:
            side = "buy" if review.action == "BUY" else "sell"
            # Calculate lot size based on position_size % of capital
            lot_size = self._calculate_lot_size(review.position_size, balance)
            result = self.executor.place_order(
                symbol=self.symbol,
                side=side,
                volume=lot_size,
                sl=review.sl_price,
                tp=review.tp_price,
            )
            rprint(f"[green]Order: {result}[/green]")
            self.executor.increment_trade_count()
            notional = lot_size * bar.close
            self.risk.record_trade(notional)
            self._log(bar, review, account, risk_state, f"executed: {result}")
        else:
            self._log(bar, review, account, risk_state, "hold")
            rprint("[yellow]HOLD[/yellow]")

    def _generate_signal(self, bar: Bar) -> tuple[str, float]:
        """Simple momentum signal. Replace with RL or transformer inference."""
        closes = [b.close for b in self.feed.buffer][-20:]
        if len(closes) < 20:
            return "HOLD", 0.5
        ma_fast = np.mean(closes[-5:])
        ma_slow = np.mean(closes[-20:])
        if ma_fast > ma_slow * 1.001:
            return "BUY", 0.72
        elif ma_fast < ma_slow * 0.999:
            return "SELL", 0.72
        return "HOLD", 0.5

    def _calculate_lot_size(self, position_size_pct: float, balance: float) -> float:
        """Convert position size % to lot size."""
        notional = balance * position_size_pct
        # Assuming BTC ~ $90k, 0.01 lot = 0.01 BTC
        lot_size = notional / self.feed._last_close if self.feed._last_close > 0 else 0.01
        return round(lot_size, 4)

    def _log(self, bar: Bar, review: LLMReview, account: dict, risk_state: dict, exec_note: str) -> None:
        entry = TradeLog(
            ts=datetime.now(timezone.utc).isoformat(),
            ts_ms=int(time.time() * 1000),
            symbol=self.symbol,
            action=review.action,
            confidence=review.confidence,
            reason=review.reason + " | " + exec_note,
            position_size=review.position_size,
            sl_price=review.sl_price,
            tp_price=review.tp_price,
            risk_approved=review.risk_approved,
            account_balance=account.get("balance", 0.0),
            equity=account.get("equity", 0.0),
            open_pl=account.get("profit", 0.0),
            daily_pnl_pct=risk_state.get("daily_pnl_pct", 0.0),
            max_drawdown_pct=risk_state.get("drawdown_pct", 0.0),
            trade_count_today=risk_state.get("trade_count_today", 0),
            bar_close=bar.close,
            bar_rsi=0.0,
            bar_vol=bar.volume,
        )
        self.logger.write(entry)


# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    capital: float = typer.Option(1_000_000.0, help="Initial capital"),
    mt5_account: int = typer.Option(..., help="MT5 account number"),
    mt5_password: str = typer.Option(..., help="MT5 account password"),
    mt5_server: str = typer.Option(..., help="MT5 server"),
    mock_execution: bool = typer.Option(False, help="Simulate MT5 (no real orders)"),
    use_llm: bool = typer.Option(False, help="Use LLM review (requires API key)"),
    interval: int = typer.Option(5, help="Minutes between decisions"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    trader = AutonomousTrader(
        symbol=symbol,
        capital=capital,
        mt5_account=mt5_account,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        mock_execution=mock_execution,
        use_llm=use_llm,
        interval=interval,
    )

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        trader.stop()


if __name__ == "__main__":
    app()
