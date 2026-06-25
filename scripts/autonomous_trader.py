#!/usr/bin/env python3
"""Autonomous trading loop: 1m-Primary LLM -> MT5 execution.

Production-ready paper trading for Windows MT5 demo accounts.
Uses Binance WebSocket for live 1m + 5m klines.
1m bars drive LLM decisions every minute.
5m bars feed the V6 pipeline for trend context.

Features:
    - 1m-primary decision engine (LLM analyzes every 1m bar)
    - Dynamic profit monitor (scales with lot size)
    - 2-minute post-trade cooldown with observation
    - Competition rule alignment (Sharpe, drawdown, trade frequency)
    - WebSocket auto-reconnect
    - No repeated MT5 login

Architecture:
    Binance WS (1m+5m) -> 1m Buffer (LLM) + 5m Pipeline (context)
        -> LLM review (1m chart + 5m indicators) -> Risk check
        -> MT5 execute -> Dynamic profit monitor -> Log

Usage (Windows — real MT5 demo orders):
    uv run python scripts/autonomous_trader.py \
        --transformer-run models/transformer/20260623T132957Z \
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER" \
        --use-llm --paper-mode

Press Ctrl+C to stop gracefully.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import structlog
import typer
import websockets
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

from intraday.agents.forecast import ForecastAgent
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import RegimeAgent
from intraday.agents.risk import RiskAgent
from intraday.agents.stay_out import StayOutDetector
from intraday.aggregator.decision import Decision, DecisionEngine
from intraday.aggregator.features import build_aggregator_row
from intraday.aggregator.meta_learner import MetaLearner
from intraday.features.calculator import FeatureCalculator, KlineBar
from intraday.features.schema import FEATURE_ROW_SCHEMA
from intraday.forecast.output import ForecastOutput
from intraday.llm.review import LLMReviewAgent, LLMReview

log = structlog.get_logger(__name__)
app = typer.Typer()

BINANCE_WS = "wss://stream.binance.com:9443/stream"
LOG_DIR = Path("logs/autonomous_trader")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Risk constants — aligned with competition rules
MAX_DAILY_LOSS_PCT = 2.0          # Stop trading at -2% daily
MAX_WEEKLY_LOSS_PCT = 5.0         # Stop trading at -5% weekly
SOFT_DD_PCT = 5.0                 # Soft drawdown warning at 5%
HARD_DD_PCT = 8.0                 # Hard stop at 8% (well below 30% margin call)
MAX_EXPOSURE_PCT = 50.0           # Allow up to 50% for 8 lot sizing
REJECT_CONFIDENCE = 0.65          # Minimum LLM confidence to trade
MIN_RISK_REWARD = 2.0             # Minimum R:R for any trade
MIN_TRADES_PER_DAY = 8            # Need 8+ trades for meaningful Sharpe
TARGET_TRADES_PER_DAY = 15        # Target 15 trades/day for volume
COOLDOWN_SECONDS = 60.0           # 60-second cooldown after trade close
MIN_TRADE_INTERVAL_SECONDS = 60.0 # Minimum 60 seconds between new trades
TARGET_LOT_SIZE = 8.0             # Fixed lot size for all trades
DYNAMIC_PROFIT_TICKS = 50.0       # Not used — profit cap is fixed below


@dataclass
class Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    taker_buy_vol: float
    trade_count: int


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
    pipeline_summary: str
    llm_output: str


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------

class TradeLogger:
    """Append-only JSONL logger with last-N retrieval for LLM context."""

    def __init__(self, log_dir: Path = LOG_DIR) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.now(timezone.utc).date().isoformat()
        self._path = self.log_dir / f"trade_log_{self._today}.jsonl"

    def _rotate(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._today:
            self._today = today
            self._path = self.log_dir / f"trade_log_{self._today}.jsonl"

    def write(self, entry: TradeLog) -> None:
        self._rotate()
        d = asdict(entry)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, default=str) + "\n")
        log.info("trade_logged", action=entry.action, confidence=entry.confidence)

    def last_n(self, n: int = 50) -> list[dict]:
        self._rotate()
        all_entries: list[dict] = []
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))
        return all_entries[-n:] if len(all_entries) >= n else all_entries


# ---------------------------------------------------------------------------
# BinanceFeed — event-driven feature calculator
# ---------------------------------------------------------------------------

class BinanceFeed:
    """Consumes Binance 1m + 5m klines.
    1m bars drive LLM decisions (every minute).
    5m bars feed the V6 pipeline (every 5 minutes).
    """

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol.upper()
        self._sym_lower = symbol.lower()
        self._calc = FeatureCalculator(symbol=self.symbol, live_mode=True)
        # 5m feature rows for the pipeline
        self._rows: deque[Any] = deque(maxlen=200)
        # 1m raw candles for LLM chart analysis
        self._raw_1m: deque[dict] = deque(maxlen=120)
        self._running = False
        self._last_close: float = 0.0
        self._bar_ready_1m = asyncio.Event()
        self._bar_ready_5m = asyncio.Event()
        self._last_bar_ts: int = 0
        self._last_raw_bar: dict | None = None

    @property
    def buffer(self) -> list[Any]:
        return list(self._rows)

    @property
    def raw_1m_buffer(self) -> list[dict]:
        return list(self._raw_1m)

    def to_df(self) -> pl.DataFrame | None:
        if len(self._rows) < 128:
            return None
        return pl.DataFrame(
            [r.model_dump() for r in self._rows],
            schema=FEATURE_ROW_SCHEMA,
        ).fill_null(0)

    def _make_kline_bar(self, k: dict, interval: str) -> KlineBar:
        return KlineBar(
            open_time_ms=int(k["t"]),
            close_time_ms=int(k["T"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            trade_count=int(k["n"]),
            taker_buy_volume=float(k["V"]),
            interval=interval,
        )

    def _on_bar(self, bar: KlineBar) -> None:
        """Feed a bar into the calculator and store the result."""
        self._last_raw_bar = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "trade_count": bar.trade_count,
            "taker_buy_volume": bar.taker_buy_volume,
        }
        if bar.interval == "1m":
            self._raw_1m.append(self._last_raw_bar)
            self._last_close = bar.close
            self._last_bar_ts = bar.open_time_ms
            self._bar_ready_1m.set()
        elif bar.interval == "5m":
            row = self._calc.dispatch(bar)
            if row is not None:
                self._rows.append(row)
                self._last_close = row.close
                self._last_bar_ts = row.bar_time_ms
            elif bar.close > 0:
                self._last_close = bar.close
                self._last_bar_ts = bar.open_time_ms
            self._bar_ready_5m.set()

    def load_historical(self, bars_5m: int = 128) -> int:
        """Load historical bars from Binance Vision API.
        We load 1m bars for feature calculation and 5m bars for alignment.
        """
        loaded = 0
        # Load 1m bars (need 5 * bars_5m 1m bars to cover the same window)
        m1_limit = bars_5m * 5 + 60  # extra buffer for rolling windows
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": "1m",
            "limit": m1_limit,
        }
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}
            resp = httpx.get(url, params=params, timeout=30.0, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for row in data:
                bar = KlineBar(
                    open_time_ms=int(row[0]),
                    close_time_ms=int(row[6]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    trade_count=int(row[8]),
                    taker_buy_volume=float(row[9]),
                    interval="1m",
                )
                self._calc.dispatch(bar)
                # Also build raw dict for LLM chart buffer
                self._last_raw_bar = {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "trade_count": bar.trade_count,
                    "taker_buy_volume": bar.taker_buy_volume,
                }
            # Seed LLM buffer with last 40 historical 1m bars
            for row in data[-40:]:
                raw_bar = {
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "trade_count": int(row[8]),
                    "taker_buy_volume": float(row[9]),
                }
                self._raw_1m.append(raw_bar)
            log.info("historical_1m_loaded", bars=len(data), symbol=self.symbol, llm_seeded=min(len(data), 40))
        except Exception as exc:
            log.error("historical_1m_load_failed", error=str(exc))
            return 0

        # Load 5m bars
        params = {
            "symbol": self.symbol,
            "interval": "5m",
            "limit": bars_5m,
        }
        try:
            resp = httpx.get(url, params=params, timeout=30.0, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for row in data:
                bar = KlineBar(
                    open_time_ms=int(row[0]),
                    close_time_ms=int(row[6]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    trade_count=int(row[8]),
                    taker_buy_volume=float(row[9]),
                    interval="5m",
                )
                self._on_bar(bar)
                loaded += 1
            log.info("historical_5m_loaded", bars=loaded, symbol=self.symbol)
            return loaded
        except Exception as exc:
            log.error("historical_5m_load_failed", error=str(exc))
            return 0

    async def run(self) -> None:
        """Connect to Binance combined stream (1m + 5m klines) with reconnect."""
        streams = f"{self._sym_lower}@kline_1m/{self._sym_lower}@kline_5m"
        url = f"{BINANCE_WS}?streams={streams}"
        self._running = True
        reconnect_count = 0
        while self._running:
            try:
                log.info("binance_ws_connect", url=url, reconnect_count=reconnect_count)
                reconnect_count += 1
                async with websockets.connect(url, ping_interval=20, close_timeout=10) as ws:
                    log.info("binance_ws_connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        # Combined stream wraps messages in {stream: ..., data: ...}
                        data = msg.get("data", msg)
                        if data.get("e") != "kline":
                            continue
                        k = data["k"]
                        if not k.get("x"):
                            continue  # bar not closed yet
                        interval = k["i"]
                        bar = self._make_kline_bar(k, interval)
                        self._on_bar(bar)
            except websockets.exceptions.ConnectionClosed as exc:
                log.warning("binance_ws_connection_closed", code=exc.code, reason=exc.reason)
            except websockets.exceptions.WebSocketException as exc:
                log.warning("binance_ws_exception", error=str(exc))
            except Exception as exc:
                log.error("binance_ws_error", error=str(exc))
            if self._running:
                log.info("binance_ws_reconnect", wait_seconds=5)
                await asyncio.sleep(5)

    def stop(self) -> None:
        self._running = False

    async def wait_for_1m_bar(self, timeout: float = 600.0) -> bool:
        """Block until a new 1m bar closes."""
        try:
            await asyncio.wait_for(self._bar_ready_1m.wait(), timeout=timeout)
            self._bar_ready_1m.clear()
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_5m_bar(self, timeout: float = 600.0) -> bool:
        """Block until a new 5m bar closes."""
        try:
            await asyncio.wait_for(self._bar_ready_5m.wait(), timeout=timeout)
            self._bar_ready_5m.clear()
            return True
        except asyncio.TimeoutError:
            return False


# ---------------------------------------------------------------------------
# MT5 Executor
# ---------------------------------------------------------------------------

class MT5Executor:
    def __init__(self, account: int, password: str, server: str, magic: int = 999999) -> None:
        self.account = account
        self.password = password
        self.server = server
        self.magic = magic
        self._mt5: Any | None = None
        self._initial_capital: float = 0.0

    def connect(self) -> bool:
        try:
            from intraday.trader.mt5_wrapper import MT5TradingWrapper
            self._mt5 = MT5TradingWrapper(
                account_id=self.account,
                password=self.password,
                server=self.server,
                magic=self.magic,
            )
            return self._mt5.connect()
        except ImportError:
            log.error("mt5_not_available")
            return False

    def shutdown(self) -> None:
        if self._mt5:
            self._mt5.shutdown()

    def state(self) -> dict:
        if self._mt5 is None:
            return {}
        s = self._mt5.account_state()
        return s.to_dict() if s else {}

    def get_positions(self, symbol: str) -> list[dict]:
        if self._mt5 is None:
            return []
        return [p.to_dict() for p in self._mt5.get_positions(symbol)]

    def place_order(self, symbol: str, side: str, volume: float, sl: float, tp: float) -> dict:
        if self._mt5 is None:
            return {"success": False, "comment": "MT5 not connected"}
        result = self._mt5.market_order(symbol, side, volume=volume, sl=sl, tp=tp)
        return result.to_dict()

    def close_position(self, ticket: int) -> dict:
        if self._mt5 is None:
            return {"success": False, "comment": "MT5 not connected"}
        result = self._mt5.close_position(ticket)
        return result.to_dict()

    def close_all(self, symbol: str) -> list[dict]:
        if self._mt5 is None:
            return []
        return [r.to_dict() for r in self._mt5.close_all_positions(symbol)]


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manages MT5 positions to avoid double-entry or conflicting trades."""

    def __init__(self, executor: MT5Executor) -> None:
        self.executor = executor

    def get_net_side(self, symbol: str) -> str | None:
        """Return 'long', 'short', or None if flat."""
        positions = self.executor.get_positions(symbol)
        net = 0.0
        for p in positions:
            net += p["volume"] if p["side"] == "long" else -p["volume"]
        if net > 0.001:
            return "long"
        if net < -0.001:
            return "short"
        return None

    def get_position_tickets(self, symbol: str, side: str | None = None) -> list[int]:
        positions = self.executor.get_positions(symbol)
        tickets = []
        for p in positions:
            if side is None or p["side"] == side:
                tickets.append(p["ticket"])
        return tickets

    def ensure_side(self, symbol: str, desired_side: str, volume: float, sl: float, tp: float) -> dict:
        """Close opposite positions, then open if not already on desired side.
        Returns the order result or a no-op dict.
        """
        current = self.get_net_side(symbol)
        if current == desired_side:
            return {"success": True, "comment": f"Already {desired_side}", "ticket": None}

        if current is not None:
            # Close opposite positions
            results = self.executor.close_all(symbol)
            log.info("positions_closed", symbol=symbol, prev_side=current, results=results)

        result = self.executor.place_order(symbol, desired_side, volume, sl, tp)
        log.info("position_opened", symbol=symbol, side=desired_side, volume=volume, result=result)
        return result


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(self, capital: float) -> None:
        self.capital = capital
        self.initial_capital = capital
        self.peak_equity = capital
        self.daily_start = capital
        self.weekly_start = capital
        self.last_trade_date = ""
        self.last_trade_week = ""
        self.trade_count_today = 0

    def update(self, equity: float, positions: list[dict] | None = None) -> dict:
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
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        daily_pnl = (equity - self.daily_start) / self.daily_start * 100 if self.daily_start > 0 else 0.0
        weekly_pnl = (equity - self.weekly_start) / self.weekly_start * 100 if self.weekly_start > 0 else 0.0
        # Calculate actual exposure from MT5 positions
        total_exposure = 0.0
        if positions:
            for p in positions:
                total_exposure += p.get("volume", 0.0) * p.get("current_price", p.get("open_price", 0.0))
        return {
            "drawdown_pct": dd * 100,
            "daily_pnl_pct": daily_pnl,
            "weekly_pnl_pct": weekly_pnl,
            "trade_count_today": self.trade_count_today,
            "total_exposure_pct": total_exposure / self.capital * 100,
        }

    def can_trade(self, review: LLMReview, risk_state: dict) -> tuple[bool, str]:
        if not review.risk_approved:
            return False, "LLM rejected"
        if risk_state["drawdown_pct"] >= HARD_DD_PCT:
            return False, f"Hard DD {risk_state['drawdown_pct']:.1f}%"
        if risk_state["daily_pnl_pct"] <= -MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss {risk_state['daily_pnl_pct']:.1f}%"
        if risk_state["weekly_pnl_pct"] <= -MAX_WEEKLY_LOSS_PCT:
            return False, f"Weekly loss {risk_state['weekly_pnl_pct']:.1f}%"
        if review.confidence < REJECT_CONFIDENCE:
            return False, f"Confidence {review.confidence:.2f} < {REJECT_CONFIDENCE}"
        if review.position_size * 100 > MAX_EXPOSURE_PCT:
            return False, f"Exposure > {MAX_EXPOSURE_PCT}%"
        return True, "ok"

    def record_trade(self) -> None:
        self.trade_count_today += 1

    def competition_status(self, risk_state: dict) -> dict:
        """Return competition-relevant metrics for LLM prompt."""
        return {
            "sharpe_estimate": self._estimate_sharpe(risk_state),
            "drawdown_status": "CRITICAL" if risk_state["drawdown_pct"] >= SOFT_DD_PCT else "OK",
            "daily_loss_status": "STOP" if risk_state["daily_pnl_pct"] <= -MAX_DAILY_LOSS_PCT else "OK",
            "trade_count_status": "LOW" if risk_state["trade_count_today"] < MIN_TRADES_PER_DAY else "OK",
            "target_trades_remaining": max(0, TARGET_TRADES_PER_DAY - risk_state["trade_count_today"]),
        }

    def _estimate_sharpe(self, risk_state: dict) -> float:
        """Rough Sharpe estimate from daily PnL and drawdown."""
        daily_pnl = risk_state.get("daily_pnl_pct", 0.0)
        dd = risk_state.get("drawdown_pct", 0.1)
        if dd < 0.1:
            dd = 0.1
        return daily_pnl / dd if dd > 0 else 0.0


# ---------------------------------------------------------------------------
# Main Trader
# ---------------------------------------------------------------------------

class AutonomousTrader:
    def __init__(
        self,
        symbol: str,
        capital: float,
        transformer_run: Path,
        data_dir: Path,
        mt5_account: int,
        mt5_password: str,
        mt5_server: str,
        use_llm: bool = False,
        llm_debug: bool = False,
        forecast_confidence: float = 0.04,
        meta_threshold: float | None = None,
        regime_fallback: bool = False,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.data_dir = data_dir
        self.forecast_confidence = forecast_confidence
        self.meta_threshold = meta_threshold
        self.regime_fallback = regime_fallback
        self.feed = BinanceFeed(symbol=symbol)
        self.executor = MT5Executor(
            account=mt5_account,
            password=mt5_password,
            server=mt5_server,
        )
        self.pos_mgr = PositionManager(self.executor)
        self.llm = LLMReviewAgent(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL"),
            debug=llm_debug,
        ) if use_llm else LLMReviewAgent(api_key="")
        self.risk = RiskManager(capital=capital)
        self.logger = TradeLogger()
        self._running = False
        self._task: asyncio.Task | None = None
        self._task_5m: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # 1m-primary state
        self._in_cooldown = False
        self._cooldown_end_ts = 0.0
        self._last_trade_close_ts = 0.0
        self._last_trade_open_ts = 0.0
        self._last_trade_close_reason: str | None = None
        self._closed_tickets: set[int] = set()
        self._pipeline_cache: dict | None = None
        self._pipeline_cache_ts = 0

        # Pipeline components
        rprint("[yellow]Loading pipeline...[/yellow]")
        self.forecast_agent = ForecastAgent(run_dir=transformer_run, device="cpu")
        self.orderflow_agent = OrderflowAgent()
        self.regime_agent = RegimeAgent.load(data_dir / "models" / "regime.pkl")
        self.risk_agent = RiskAgent()
        self.stay_out = StayOutDetector()
        self.meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
        # Use configurable threshold (paper mode overrides default)
        thresh = meta_threshold if meta_threshold is not None else self.meta_learner._threshold
        self.decision_engine = DecisionEngine(
            meta_learner=self.meta_learner,
            threshold=thresh,
        )
        rprint(f"[green]OK Pipeline loaded[/green]  (meta_threshold={thresh:.4f}, forecast_confidence={forecast_confidence:.4f})")

    async def start(self) -> None:
        rprint("[green]Starting autonomous trader...[/green]")
        rprint(f"  Symbol: {self.symbol}")
        rprint(f"  Capital: {self.capital:,.2f}")
        rprint(f"  LLM: {bool(self.llm.api_key)}")
        rprint("  Execution: Real MT5 demo orders")
        rprint("  [cyan]1m-Primary Mode: ENABLED[/cyan]")
        rprint("  [cyan]Dynamic Profit Monitor: ENABLED[/cyan]")
        rprint("  [cyan]Lot Size: 8.0[/cyan]")
        rprint("  [cyan]Profit Cap: $200 | Stop Loss: $100[/cyan]")
        rprint("  [cyan]Post-Trade Cooldown: 60 seconds[/cyan]")

        if not self.executor.connect():
            rprint("[red]Failed to connect to MT5[/red]")
            return

        self._running = True

        # Load historical bars (5m for pipeline, 1m for LLM context)
        loaded = self.feed.load_historical(bars_5m=128)
        rprint(f"[green]Loaded {loaded} historical 5m bars[/green]")
        if loaded >= 128:
            rprint("[green]Ready to trade immediately — no warm-up needed[/green]")
        else:
            rprint("[yellow]Warm-up needed — waiting for 128 bars...[/yellow]")

        # Start WebSocket feed
        self._task = asyncio.create_task(self.feed.run())
        # Start profit monitor in background
        self._profit_monitor_task = asyncio.create_task(self._profit_monitor_loop())
        # Start 5m pipeline updater in background
        self._task_5m = asyncio.create_task(self._pipeline_5m_loop())

        # Main loop — triggers every 1m bar
        while self._running and not self._shutdown_event.is_set():
            try:
                got_1m = await self.feed.wait_for_1m_bar(timeout=600.0)
                if not got_1m:
                    log.warning("1m_bar_timeout", seconds=600)
                    continue
                if not self._running:
                    break
                await self._on_1m_bar()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("main_loop_error", error=str(exc))
                traceback.print_exc()

        rprint("[yellow]Main loop exited. Shutting down...[/yellow]")
        self.executor.shutdown()

    def stop(self) -> None:
        self._running = False
        self._shutdown_event.set()
        self.feed.stop()
        if self._task:
            self._task.cancel()
        if self._task_5m:
            self._task_5m.cancel()
        if hasattr(self, '_profit_monitor_task') and self._profit_monitor_task:
            self._profit_monitor_task.cancel()
        rprint("[red]Trader stopped.[/red]")

    async def _profit_monitor_loop(self) -> None:
        """Monitor open positions every 1 second.

        Rules:
        - Close at $200 profit (profit cap)
        - Close at $100 loss (stop loss)
        - No trailing stop, no minimum profit lock
        """
        CHECK_INTERVAL = 1.0
        PROFIT_CAP = 200.0        # Close immediately when profit hits $200
        EMERGENCY_STOP = -100.0   # Close when loss hits $100

        max_profit = 0.0
        last_profit = 0.0
        log.info("profit_monitor_started", interval=CHECK_INTERVAL, mode="profit_cap_sl")

        while self._running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                if not self._running:
                    break

                # Get current positions
                positions = self.executor.get_positions(self.symbol)
                if not positions:
                    max_profit = 0.0
                    last_profit = 0.0
                    self._closed_tickets.clear()
                    continue

                for p in positions:
                    profit = p.get('profit', 0.0)
                    ticket = p.get('ticket', 0)
                    if ticket in self._closed_tickets:
                        continue
                    side = p.get('side', 'unknown')
                    current_price = p.get('current_price', 0.0)
                    open_price = p.get('open_price', 0.0)
                    volume = p.get('volume', 0.0)

                    # Track max profit (peak)
                    if profit > max_profit:
                        max_profit = profit

                    # 1. Profit cap: close at $200
                    if profit >= PROFIT_CAP:
                        rprint(f"\n[bold green]PROFIT CAP: ${profit:.2f} >= ${PROFIT_CAP:.2f} (lot={volume:.2f}) — closing![/bold green]")
                        rprint(f"  Ticket: {ticket}, Side: {side}, Open: {open_price:.2f}, Current: {current_price:.2f}")
                        result = self.executor.close_position(ticket)
                        rprint(f"  [green]Closed: {result}[/green]")
                        log.info("profit_cap_closed", ticket=ticket, profit=profit, cap=PROFIT_CAP, volume=volume)
                        self._last_trade_close_reason = f"profit_cap: ${profit:.2f}"
                        self._start_cooldown()
                        self._closed_tickets.add(ticket)
                        max_profit = 0.0
                        last_profit = 0.0
                        continue

                    # 2. Stop loss: close at $100 loss
                    if profit < EMERGENCY_STOP:
                        rprint(f"\n[bold red]STOP LOSS: PnL=${profit:.2f} (lot={volume:.2f}) — closing![/bold red]")
                        rprint(f"  Ticket: {ticket}, Side: {side}, Open: {open_price:.2f}, Current: {current_price:.2f}")
                        result = self.executor.close_position(ticket)
                        rprint(f"  [red]Closed: {result}[/red]")
                        log.info("stop_loss_closed", ticket=ticket, profit=profit, volume=volume)
                        self._last_trade_close_reason = f"stop_loss: ${profit:.2f}"
                        self._start_cooldown()
                        self._closed_tickets.add(ticket)
                        max_profit = 0.0
                        last_profit = 0.0
                        continue

                    # Print profit updates when PnL changes by $1+
                    if abs(profit - last_profit) > 1.0:
                        color = "green" if profit > 0 else "red" if profit < 0 else "yellow"
                        rprint(f"[bold {color}]Position {ticket}: PnL=${profit:.2f} (max=${max_profit:.2f}, cap=${PROFIT_CAP:.2f}) | Open={open_price:.2f} | Current={current_price:.2f}[/bold {color}]")
                        last_profit = profit

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("profit_monitor_error", error=str(exc))

    def _start_cooldown(self) -> None:
        """Start 2-minute observation cooldown after trade close."""
        self._in_cooldown = True
        self._cooldown_end_ts = time.time() + COOLDOWN_SECONDS
        self._last_trade_close_ts = time.time()
        log.info("cooldown_started", duration=COOLDOWN_SECONDS, end_ts=self._cooldown_end_ts)

    def _check_cooldown(self) -> bool:
        """Return True if cooldown is active."""
        if not self._in_cooldown:
            return False
        if time.time() >= self._cooldown_end_ts:
            self._in_cooldown = False
            log.info("cooldown_ended")
            return False
        return True

    def _cooldown_seconds_remaining(self) -> float:
        """Return seconds remaining in cooldown."""
        if not self._in_cooldown:
            return 0.0
        return max(0.0, self._cooldown_end_ts - time.time())

    async def _pipeline_5m_loop(self) -> None:
        """Background loop: update the V6 pipeline every 5m bar."""
        while self._running and not self._shutdown_event.is_set():
            try:
                got_5m = await self.feed.wait_for_5m_bar(timeout=600.0)
                if not got_5m or not self._running:
                    continue
                df = self.feed.to_df()
                if df is None or len(df) < 128:
                    continue
                try:
                    pipeline = self._run_pipeline(df)
                    self._pipeline_cache = pipeline
                    self._pipeline_cache_ts = int(time.time() * 1000)
                    log.info("pipeline_5m_updated", ts=self._pipeline_cache_ts)
                except Exception as exc:
                    log.error("pipeline_5m_error", error=str(exc))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("pipeline_5m_loop_error", error=str(exc))

    async def _on_1m_bar(self) -> None:
        """Called every 1m bar close. This is the primary decision engine."""
        # Check cooldown
        if self._check_cooldown():
            remaining = self._cooldown_seconds_remaining()
            log.info("cooldown_active", remaining_seconds=remaining)
            return

        # Minimum trade interval: don't trade too frequently even if no cooldown
        if self._last_trade_open_ts > 0:
            elapsed = time.time() - self._last_trade_open_ts
            if elapsed < MIN_TRADE_INTERVAL_SECONDS:
                log.info("min_trade_interval_active", elapsed_seconds=elapsed, min_seconds=MIN_TRADE_INTERVAL_SECONDS)
                return

        # Need 1m candles for LLM chart
        raw_1m = self.feed.raw_1m_buffer
        if len(raw_1m) < 40:
            log.info("1m_warming_up", bars=len(raw_1m))
            return

        # Get latest 1m candle
        latest_1m = raw_1m[-1]
        candle_close = latest_1m.get("close", 0.0)

        # Extract recent 1m candles for LLM (last 20 x 1m)
        recent_1m_candles = self._extract_recent_1m_candles(raw_1m)

        # Extract 5m indicators from pipeline cache (or compute if missing)
        indicators = self._get_indicators()

        # Account / risk state
        account = self.executor.state()
        equity = account.get("equity", self.capital)
        positions = self.executor.get_positions(self.symbol)
        risk_state = self.risk.update(equity, positions)
        recent_logs = self.logger.last_n(20)

        # Competition status for LLM
        comp_status = self.risk.competition_status(risk_state)

        # Build competition rules for LLM
        competition_rules = self._build_competition_rules(comp_status, risk_state)

        # Primary decision: LLM analyzes 1m chart directly
        review = self.llm.analyze_chart(
            candles=recent_1m_candles,
            indicators=indicators,
            positions=positions,
            account=account,
            risk_state=risk_state,
            recent_logs=recent_logs,
            competition_rules=competition_rules,
            last_trade_reason=self._last_trade_close_reason,
        )

        # Fallback: if LLM failed/timed out, use pipeline signal
        if review.confidence == 0.0 and not review.risk_approved and self._pipeline_cache:
            pipe_decision = self._pipeline_cache.get("decision", {})
            pipe_side = pipe_decision.get("side", "flat")
            pipe_conf = pipe_decision.get("confidence", 0.0)
            if pipe_side in ("long", "short") and pipe_conf >= 0.40:
                action = "BUY" if pipe_side == "long" else "SELL"
                # Compute ATR-based SL/TP
                atr = candle_close * 0.001
                sl = candle_close - atr * 2 if action == "BUY" else candle_close + atr * 2
                tp = candle_close + atr * 4 if action == "BUY" else candle_close - atr * 4
                size = 0.25 if pipe_conf < 0.75 else 0.50
                review = LLMReview(
                    action=action,
                    confidence=round(pipe_conf, 2),
                    reason=f"LLM timeout fallback: pipeline {pipe_side} conf={pipe_conf:.2f}",
                    position_size=size,
                    sl_price=round(sl, 2),
                    tp_price=round(tp, 2),
                    risk_approved=True,
                )
                rprint(f"[yellow]LLM timed out — using pipeline fallback: {action} (conf={pipe_conf:.2f})[/yellow]")

        # Regime filter: Only trade WITH the 5m trend direction
        if self._pipeline_cache and review.action in ("BUY", "SELL"):
            regime = self._pipeline_cache.get("regime", {}).get("regime", "unknown")
            pipe_side = self._pipeline_cache.get("decision", {}).get("side", "flat")
            # Use pipeline decision if available, otherwise fall back to regime
            trend_direction = pipe_side if pipe_side != "flat" else regime
            if trend_direction in ("long", "bull") and review.action == "SELL":
                rprint("[yellow]Regime filter: bullish trend, overriding SELL to HOLD[/yellow]")
                review = LLMReview("HOLD", 0.0, "Regime filter: bullish trend", 0.0, 0.0, 0.0, False)
            elif trend_direction in ("short", "bear") and review.action == "BUY":
                rprint("[yellow]Regime filter: bearish trend, overriding BUY to HOLD[/yellow]")
                review = LLMReview("HOLD", 0.0, "Regime filter: bearish trend", 0.0, 0.0, 0.0, False)

        rprint(f"\n[bold]LLM Decision: {review.action} (conf={review.confidence:.2f})[/bold]")
        rprint(f"  Reason: {review.reason}")
        rprint(f"  SL={review.sl_price:.2f} TP={review.tp_price:.2f} Price={candle_close:.2f}")

        # Clamp position_size to max exposure limit
        max_position_size = MAX_EXPOSURE_PCT / 100.0
        if review.position_size > max_position_size:
            rprint(f"[yellow]Position size clamped: {review.position_size:.2f} -> {max_position_size:.2f} (max exposure)[/yellow]")
            review.position_size = max_position_size

        # Validate SL/TP before risk check
        if review.action in ("BUY", "SELL") and review.position_size > 0:
            side = "buy" if review.action == "BUY" else "sell"
            sl_ok, tp_ok, sl_fixed, tp_fixed = self._validate_stops(
                side, candle_close, review.sl_price, review.tp_price
            )
            if not sl_ok:
                rprint(f"[yellow]SL adjusted: {review.sl_price:.2f} -> {sl_fixed:.2f}[/yellow]")
                review.sl_price = sl_fixed
            if not tp_ok:
                rprint(f"[yellow]TP adjusted: {review.tp_price:.2f} -> {tp_fixed:.2f}[/yellow]")
                review.tp_price = tp_fixed

        # Risk validation
        ok, reason = self.risk.can_trade(review, risk_state)
        if not ok:
            rprint(f"[red]Risk block: {reason}[/red]")
            self._log(candle_close, review, account, risk_state, self._pipeline_cache or {}, reason)
            return

        # Execute
        if review.action in ("BUY", "SELL") and review.position_size > 0:
            side = "buy" if review.action == "BUY" else "sell"
            lot_size = self._calculate_lot_size(
                review.position_size,
                account.get("balance", self.capital),
                candle_close,
            )
            result = self.pos_mgr.ensure_side(
                self.symbol, side, lot_size,
                review.sl_price, review.tp_price,
            )
            rprint(f"[green]Order: {result}[/green]")
            self.risk.record_trade()
            self._last_trade_open_ts = time.time()
            self._log(candle_close, review, account, risk_state, self._pipeline_cache or {}, f"executed: {result}")
        else:
            self._log(candle_close, review, account, risk_state, self._pipeline_cache or {}, "hold")
            rprint("[yellow]HOLD[/yellow]")

    def _build_competition_rules(self, comp_status: dict, risk_state: dict) -> dict:
        """Build competition rules context for LLM prompt."""
        return {
            "competition_context": {
                "name": "AI Quant Trading Competition",
                "initial_capital": 1000000,
                "max_leverage": 30,
                "stop_out_level_percent": 30,
                "primary_asset": "BTCUSD",
                "objective": "Maximize risk-adjusted returns while maintaining low drawdown and sufficient trading activity.",
            },
            "leaderboard_priorities": [
                "Preserve capital",
                "Maintain high Sharpe Ratio",
                "Control Maximum Drawdown",
                "Generate positive returns",
                "Meet trading volume requirements",
            ],
            "risk_constraints": {
                "max_risk_per_trade_percent": 0.5,
                "max_daily_loss_percent": MAX_DAILY_LOSS_PCT,
                "max_weekly_loss_percent": MAX_WEEKLY_LOSS_PCT,
                "soft_drawdown_limit_percent": SOFT_DD_PCT,
                "hard_drawdown_limit_percent": HARD_DD_PCT,
                "max_total_exposure_percent": MAX_EXPOSURE_PCT,
                "preferred_leverage": 1,
                "absolute_max_leverage": 5,
            },
            "trade_frequency": {
                "goal": "Generate enough volume for Sharpe and drawdown calculations",
                "minimum_trades_per_day": MIN_TRADES_PER_DAY,
                "target_trades_per_day": TARGET_TRADES_PER_DAY,
                "avoid_overtrading": True,
            },
            "position_sizing": {
                "confidence_below_0_60": 0.0,
                "confidence_0_60_to_0_75": 0.25,
                "confidence_0_75_to_0_85": 0.5,
                "confidence_above_0_85": 1.0,
                "reduce_size_during_high_volatility": True,
            },
            "trade_filtering": {
                "reject_if_confidence_below": REJECT_CONFIDENCE,
                "reject_if_risk_reward_below": MIN_RISK_REWARD,
                "reject_if_spread_abnormally_high": True,
                "reject_if_drawdown_limit_hit": True,
            },
            "take_profit_and_stop_loss": {
                "mandatory": True,
                "stop_loss_method": "ATR",
                "take_profit_method": "ATR",
                "minimum_risk_reward": MIN_RISK_REWARD,
            },
            "capital_preservation_rules": [
                "Never martingale",
                "Never average losing positions",
                "Never use max leverage",
                "Never remove stop loss",
                "Never risk account survival for single trade",
            ],
            "current_status": {
                "sharpe_estimate": round(comp_status.get("sharpe_estimate", 0.0), 2),
                "drawdown_pct": round(risk_state.get("drawdown_pct", 0.0), 2),
                "daily_pnl_pct": round(risk_state.get("daily_pnl_pct", 0.0), 2),
                "trade_count_today": risk_state.get("trade_count_today", 0),
                "target_trades_remaining": comp_status.get("target_trades_remaining", 0),
            },
        }

    def _get_indicators(self) -> dict:
        """Get indicators from 5m pipeline or compute from 1m buffer."""
        df = self.feed.to_df()
        if df is None or len(df) < 2:
            return {}
        last_row = df.row(len(df) - 1, named=True)
        return self._extract_indicators(last_row)

    def _extract_recent_1m_candles(self, raw_1m: list[dict]) -> list[dict]:
        """Extract last 40 x 1m candles from raw buffer for LLM chart analysis."""
        candles = []
        n = min(40, len(raw_1m))
        for i in range(n):
            row = raw_1m[len(raw_1m) - n + i]
            taker_vol = row.get("taker_buy_volume", 0.0)
            vol = row.get("volume", 0.0)
            taker_pct = (taker_vol / vol * 100) if vol > 0 else 50.0
            candles.append({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
                "trades": row.get("trade_count", 0),
                "taker_buy_pct": taker_pct,
            })
        return candles

    def _extract_indicators(self, row: dict) -> dict:
        """Extract key technical indicators from the latest 5m row."""
        return {
            "realized_vol_30m": row.get("realized_vol_30m", None),
            "taker_buy_ratio_5m": row.get("taker_buy_ratio_5m", None),
            "rsi_5m": row.get("rsi_5m", None),
            "ema_5m": row.get("ema_5m", None),
            "ema_20m": row.get("ema_20m", None),
            "macd_line": row.get("macd_line", None),
            "macd_signal": row.get("macd_signal", None),
            "macd_histogram": row.get("macd_histogram", None),
            "bollinger_upper": row.get("bollinger_upper", None),
            "bollinger_mid": row.get("bollinger_mid", None),
            "bollinger_lower": row.get("bollinger_lower", None),
            "atr_5m": row.get("atr_5m", None),
            "vpin_5m": row.get("vpin_5m", None),
            "hawkes_5m": row.get("hawkes_5m", None),
            "funding_rate": row.get("funding_rate", None),
            "spread_bps": row.get("spread_bps", None),
            "skew_5m": row.get("skew_5m", None),
            "kurt_5m": row.get("kurt_5m", None),
        }

    def _run_pipeline(self, df: pl.DataFrame) -> dict:
        """Run the full V6 pipeline on the latest bar."""
        i = len(df) - 1
        row = df.row(i, named=True)
        ts_ms = int(row["bar_time_ms"])
        feat_window = df.slice(max(0, i - 127), min(128, i + 1))

        forecast_opinion = self.forecast_agent.predict(feat_window)
        opinions = {
            "orderflow": self.orderflow_agent.predict(feat_window),
            "regime": self.regime_agent.predict(feat_window),
            "risk": self.risk_agent.predict(feat_window),
            "stay_out": self.stay_out.predict(feat_window),
        }

        fc_payload = forecast_opinion.payload
        prob_up = fc_payload.get("forecast_prob_up", 0.5)
        p_up = max(0.0, min(1.0, prob_up))
        p_down = 1.0 - p_up
        forecast = ForecastOutput(
            ts_ms=ts_ms,
            horizon_minutes=15,
            p_bins=[p_down, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_up],
            p_up_05sigma=p_up,
            p_down_05sigma=p_down,
            expected_move_sigma=(p_up - 0.5) * 2.0,
            confidence=abs(p_up - 0.5) * 2.0,
            meta_act=p_up > 0.5 + self.forecast_confidence / 2 or p_up < 0.5 - self.forecast_confidence / 2,
            meta_p_correct=abs(p_up - 0.5) * 2.0,
            model_version="forecast_agent",
            inference_ms=forecast_opinion.inference_ms,
        )

        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        # Pull actual market context from the latest bar
        realized_vol = row.get("realized_vol_30m", 0.0) or 0.0
        spread_bps = row.get("spread_bps", 0.0) or 0.0
        funding_rate = row.get("funding_rate", 0.0) or 0.0

        agg_row = build_aggregator_row(
            forecast=forecast,
            opinions=opinions,
            spread_bps=spread_bps,
            realized_vol_30m=realized_vol,
            funding_rate=funding_rate,
            hour_utc=dt.hour,
            minute_of_hour=dt.minute,
            day_of_week=dt.weekday(),
        )

        decision = self.decision_engine.decide(agg_row, forecast)

        # ── Regime fallback: ALWAYS use regime + orderflow when regime is clear ──
        if self.regime_fallback:
            rg = opinions["regime"].payload.get("regime", "unknown")
            of_bias = opinions["orderflow"].payload.get("flow_bias", 0.0)
            # Bull regime + positive flow = long
            if rg == "bull" and of_bias > 0:
                decision = Decision(
                    ts_ms=ts_ms,
                    side="long",
                    confidence=0.70,
                    horizon_minutes=15,
                    reason=f"regime_fallback: {rg} + flow_bias={of_bias}",
                )
            # Bear regime + negative flow = short
            elif rg == "bear" and of_bias < 0:
                decision = Decision(
                    ts_ms=ts_ms,
                    side="short",
                    confidence=0.70,
                    horizon_minutes=15,
                    reason=f"regime_fallback: {rg} + flow_bias={of_bias}",
                )

        return {
            "forecast": {
                "p_up": round(p_up, 4),
                "confidence": round(forecast.confidence, 4),
                "expected_move": round(forecast.expected_move_sigma, 4),
            },
            "orderflow": {
                "flow_bias": opinions["orderflow"].payload.get("flow_bias", 0.0),
                "vpin": opinions["orderflow"].payload.get("vpin", 0.0),
            },
            "regime": {
                "regime": opinions["regime"].payload.get("regime", "unknown"),
                "vol_regime": opinions["regime"].payload.get("vol_regime", "normal"),
            },
            "risk": {
                "allow_trade": opinions["risk"].payload.get("allow_trade", True),
                "risk_multiplier": opinions["risk"].payload.get("risk_multiplier", 1.0),
            },
            "stay_out": {
                "mode": opinions["stay_out"].payload.get("mode", "normal"),
            },
            "decision": {
                "side": decision.side,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        }

    def _calculate_lot_size(self, position_size_pct: float, balance: float, price: float) -> float:
        return TARGET_LOT_SIZE

    def _validate_stops(self, side: str, price: float, sl: float, tp: float) -> tuple[bool, bool, float, float]:
        """Validate and fix SL/TP for MT5.
        Returns (sl_ok, tp_ok, fixed_sl, fixed_tp).
        Minimum stop distance: 50 points = $0.05 for BTCUSD (point=0.001).
        """
        MIN_STOP_DISTANCE = 0.05  # $0.05 for BTCUSD
        sl_ok = True
        tp_ok = True
        fixed_sl = sl
        fixed_tp = tp

        if side == "buy":
            if sl >= price - MIN_STOP_DISTANCE:
                fixed_sl = round(price - max(MIN_STOP_DISTANCE * 2, price - sl + MIN_STOP_DISTANCE), 2)
                sl_ok = False
            if tp <= price + MIN_STOP_DISTANCE:
                fixed_tp = round(price + max(MIN_STOP_DISTANCE * 2, tp - price + MIN_STOP_DISTANCE), 2)
                tp_ok = False
        else:  # sell
            if sl <= price + MIN_STOP_DISTANCE:
                fixed_sl = round(price + max(MIN_STOP_DISTANCE * 2, sl - price + MIN_STOP_DISTANCE), 2)
                sl_ok = False
            if tp >= price - MIN_STOP_DISTANCE:
                fixed_tp = round(price - max(MIN_STOP_DISTANCE * 2, price - tp + MIN_STOP_DISTANCE), 2)
                tp_ok = False

        return sl_ok, tp_ok, fixed_sl, fixed_tp

    def _log(self, candle_close: float, review: LLMReview, account: dict, risk_state: dict, pipeline: dict, exec_note: str) -> None:
        summary = json.dumps(pipeline.get("decision", {}))
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
            bar_close=candle_close,
            pipeline_summary=summary,
            llm_output=json.dumps(review.to_dict()),
        )
        self.logger.write(entry)


@app.command()
def main(
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    capital: float = typer.Option(1_000_000.0, help="Initial capital"),
    transformer_run: Path = typer.Option(..., help="Path to transformer run dir"),
    data_dir: Path = typer.Option(Path("data"), help="Data root directory"),
    mt5_account: int = typer.Option(..., help="MT5 account number"),
    mt5_password: str = typer.Option(..., help="MT5 account password"),
    mt5_server: str = typer.Option(..., help="MT5 server"),
    use_llm: bool = typer.Option(False, help="Use LLM review (requires LLM_TOKEN)"),
    llm_debug: bool = typer.Option(False, help="Log full LLM prompt to logger"),
    paper_mode: bool = typer.Option(False, help="Paper mode: lower thresholds so model can trade (demo)"),
    forecast_confidence: float = typer.Option(0.04, help="Min forecast confidence to activate (0.02 for paper)"),
    meta_threshold: float = typer.Option(None, help="Override meta-learner threshold (0.05 for paper)"),
    regime_fallback: bool = typer.Option(False, help="Use regime+orderflow as fallback when transformer is uncertain"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    # Paper mode overrides
    if paper_mode:
        fc = 0.02
        mt = 0.05
        rprint("[yellow]PAPER MODE ENABLED: thresholds lowered for demo trading[/yellow]")
        rprint(f"  forecast_confidence={fc}  meta_threshold={mt}")
    else:
        fc = forecast_confidence
        mt = meta_threshold

    trader = AutonomousTrader(
        symbol=symbol,
        capital=capital,
        transformer_run=transformer_run,
        data_dir=data_dir,
        mt5_account=mt5_account,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        use_llm=use_llm,
        llm_debug=llm_debug,
        forecast_confidence=fc,
        meta_threshold=mt,
        regime_fallback=regime_fallback,
    )

    # Signal handling for graceful shutdown on Windows
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, trader.stop)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler in some cases

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        trader.stop()
        sys.exit(0)


if __name__ == "__main__":
    app()
