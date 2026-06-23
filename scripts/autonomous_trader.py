#!/usr/bin/env python3
"""Autonomous trading loop: Pipeline → LLM review → MT5 execution.

Architecture:
    Binance WS (5m bars) → Feature buffer → V6 Pipeline (agents + aggregator)
        → LLM review (Fireworks Kimi K2.6) → Risk check → MT5 execute → Log

The LLM receives:
    - Full pipeline output (forecast, orderflow, regime, risk, stay_out)
    - Candle data (OHLCV + computed indicators)
    - Account state (balance, equity, drawdown)
    - Recent trade history

Returns fixed JSON schema:
    {"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "...",
     "position_size": 0.0-1.0, "sl_price": float, "tp_price": float,
     "risk_approved": true}

Usage:
    export LLM_TOKEN="fw_..."
    export LLM_BASE_URL="https://api.fireworks.ai/inference"
    export LLM_MODEL="accounts/fireworks/routers/kimi-k2p6-turbo"

    # Mock mode (Linux, no real orders)
    uv run python scripts/autonomous_trader.py \
        --mt5-account 10408 --mt5-password "..." --mt5-server "..." \
        --transformer-run models/transformer/20260623T132957Z \
        --mock-execution --use-llm --interval 5

    # Live (Windows, real MT5)
    uv run python scripts/autonomous_trader.py \
        --mt5-account 10408 --mt5-password "..." --mt5-server "..." \
        --transformer-run models/transformer/20260623T132957Z \
        --use-llm --interval 5
"""
from __future__ import annotations

import asyncio
import json
import os
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

from intraday.agents.forecast import ForecastAgent
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import RegimeAgent
from intraday.agents.risk import RiskAgent
from intraday.agents.stay_out import StayOutDetector
from intraday.aggregator.decision import DecisionEngine
from intraday.aggregator.features import build_aggregator_row
from intraday.aggregator.meta_learner import MetaLearner
from intraday.forecast.output import ForecastOutput
from intraday.llm.review import LLMReviewAgent, LLMReview

log = structlog.get_logger(__name__)
app = typer.Typer()

BINANCE_WS = "wss://fstream.binance.com/ws"
LOG_DIR = Path("logs/autonomous_trader")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Risk constants
MAX_DAILY_LOSS_PCT = 2.0
MAX_WEEKLY_LOSS_PCT = 5.0
HARD_DD_PCT = 12.0
MAX_EXPOSURE_PCT = 25.0
REJECT_CONFIDENCE = 0.65
MIN_RISK_REWARD = 2.0


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


# ── TradeLogger ───────────────────────────────────────────────────────────────

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
        with open(self._path, "a") as f:
            f.write(json.dumps(d, default=str) + "\n")
        log.info("trade_logged", action=entry.action, confidence=entry.confidence)

    def last_n(self, n: int = 50) -> list[dict]:
        self._rotate()
        all_entries: list[dict] = []
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))
        return all_entries[-n:] if len(all_entries) >= n else all_entries


# ── Binance Feed ─────────────────────────────────────────────────────────────

class BinanceFeed:
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m") -> None:
        self.symbol = symbol.lower()
        self.interval = interval
        self._buffer: deque[Candle] = deque(maxlen=200)
        self._running = False
        self._last_close: float = 0.0

    @property
    def buffer(self) -> list[Candle]:
        return list(self._buffer)

    def to_df(self) -> pl.DataFrame | None:
        if len(self._buffer) < 128:
            return None
        candles = list(self._buffer)
        return pl.DataFrame({
            "bar_time_ms": [c.ts_ms for c in candles],
            "close": [c.close for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "volume": [c.volume for c in candles],
            "quote_volume": [c.quote_volume for c in candles],
            "taker_buy_vol": [c.taker_buy_vol for c in candles],
            "trade_count": [c.trade_count for c in candles],
        })

    def load_historical(self, limit: int = 128) -> int:
        """Load historical bars from Binance Vision API (public data, no auth needed).
        Uses data-api.binance.vision which is accessible from all regions.
        """
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {
            "symbol": self.symbol.upper(),
            "interval": self.interval,
            "limit": limit,
        }
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}
            resp = httpx.get(url, params=params, timeout=30.0, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            loaded = 0
            for row in data:
                candle = Candle(
                    ts_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[7]),
                    taker_buy_vol=float(row[9]),
                    trade_count=int(row[8]),
                )
                self._buffer.append(candle)
                self._last_close = candle.close
                loaded += 1
            log.info("historical_loaded_vision", bars=loaded, symbol=self.symbol)
            return loaded
        except Exception as exc:
            log.error("historical_load_failed", error=str(exc))
            return 0

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
                if k["x"]:
                    candle = Candle(
                        ts_ms=int(k["t"]),
                        open=float(k["o"]),
                        high=float(k["h"]),
                        low=float(k["l"]),
                        close=float(k["c"]),
                        volume=float(k["v"]),
                        quote_volume=float(k["q"]),
                        taker_buy_vol=float(k["V"]),
                        trade_count=int(k["n"]),
                    )
                    self._buffer.append(candle)
                    self._last_close = candle.close
                    log.info("bar_closed", ts_ms=candle.ts_ms, close=candle.close, volume=candle.volume)

    def stop(self) -> None:
        self._running = False


# ── MT5 Executor ─────────────────────────────────────────────────────────────

class MT5Executor:
    def __init__(self, account: int, password: str, server: str, mock: bool = False, magic: int = 999999) -> None:
        self.account = account
        self.password = password
        self.server = server
        self.mock = mock
        self.magic = magic
        self._mt5: Any | None = None
        self._initial_capital: float = 0.0

    def connect(self) -> bool:
        if self.mock:
            log.info("mt5_mock_mode")
            return True
        try:
            from intraday.trader.mt5_wrapper import MT5TradingWrapper
            self._mt5 = MT5TradingWrapper(account_id=self.account, password=self.password, server=self.server, magic=self.magic)
            return self._mt5.connect()
        except ImportError:
            log.error("mt5_not_available")
            return False

    def shutdown(self) -> None:
        if self._mt5 and not self.mock:
            self._mt5.shutdown()

    def state(self) -> dict:
        if self.mock:
            return {"balance": self._initial_capital, "equity": self._initial_capital, "profit": 0.0, "margin": 0.0, "free_margin": self._initial_capital}
        if self._mt5 is None:
            return {}
        s = self._mt5.account_state()
        return s.to_dict() if s else {}

    def get_positions(self, symbol: str) -> list[dict]:
        if self.mock or self._mt5 is None:
            return []
        return [p.to_dict() for p in self._mt5.get_positions(symbol)]

    def place_order(self, symbol: str, side: str, volume: float, sl: float, tp: float) -> dict:
        if self.mock:
            log.info("mock_order", symbol=symbol, side=side, volume=volume, sl=sl, tp=tp)
            return {"success": True, "ticket": 123456, "price": 0.0, "comment": "mock"}
        if self._mt5 is None:
            return {"success": False, "comment": "MT5 not connected"}
        result = self._mt5.market_order(symbol, side, volume=volume, sl=sl, tp=tp)
        return result.to_dict()

    def close_all(self, symbol: str) -> list[dict]:
        if self.mock or self._mt5 is None:
            return []
        return [r.to_dict() for r in self._mt5.close_all_positions(symbol)]


# ── Risk Manager ─────────────────────────────────────────────────────────────

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
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        daily_pnl = (equity - self.daily_start) / self.daily_start * 100 if self.daily_start > 0 else 0.0
        weekly_pnl = (equity - self.weekly_start) / self.weekly_start * 100 if self.weekly_start > 0 else 0.0
        return {
            "drawdown_pct": dd * 100,
            "daily_pnl_pct": daily_pnl,
            "weekly_pnl_pct": weekly_pnl,
            "trade_count_today": self.trade_count_today,
            "total_exposure_pct": self.total_exposure / self.capital * 100,
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
        if risk_state["total_exposure_pct"] + review.position_size * 100 > MAX_EXPOSURE_PCT:
            return False, f"Exposure > {MAX_EXPOSURE_PCT}%"
        return True, "ok"

    def record_trade(self, notional: float) -> None:
        self.trade_count_today += 1
        self.total_exposure += notional


# ── Main Trader ───────────────────────────────────────────────────────────────

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
        mock_execution: bool = False,
        use_llm: bool = False,
        interval: int = 5,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.data_dir = data_dir
        self.feed = BinanceFeed(symbol=symbol)
        self.executor = MT5Executor(account=mt5_account, password=mt5_password, server=mt5_server, mock=mock_execution)
        self.llm = LLMReviewAgent(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL"),
        ) if use_llm else LLMReviewAgent()
        self.risk = RiskManager(capital=capital)
        self.logger = TradeLogger()
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

        # Pipeline components
        rprint("[yellow]Loading pipeline...[/yellow]")
        self.forecast_agent = ForecastAgent(run_dir=transformer_run)
        self.orderflow_agent = OrderflowAgent()
        self.regime_agent = RegimeAgent.load(data_dir / "models" / "regime.pkl")
        self.risk_agent = RiskAgent()
        self.stay_out = StayOutDetector()
        self.meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
        self.decision_engine = DecisionEngine(meta_learner=self.meta_learner, threshold=self.meta_learner._threshold)
        rprint("[green]✓ Pipeline loaded[/green]")

    async def start(self) -> None:
        rprint("[green]Starting autonomous trader...[/green]")
        rprint(f"  Symbol: {self.symbol}")
        rprint(f"  Capital: {self.capital:,.2f}")
        rprint(f"  Mock: {self.executor.mock} | LLM: {bool(self.llm.api_key)}")

        if not self.executor.connect():
            rprint("[red]Failed to connect to MT5[/red]")
            return

        self._running = True
        # Load historical bars first so we don't wait 10 hours
        loaded = self.feed.load_historical(limit=128)
        rprint(f"[green]Loaded {loaded} historical bars[/green]")
        self._task = asyncio.create_task(self.feed.run())
        if loaded >= 128:
            rprint("[green]Ready to trade immediately — no warm-up needed[/green]")

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

        candle = self.feed.buffer[-1]
        df = self.feed.to_df()
        if df is None:
            return

        # Run pipeline
        try:
            pipeline = self._run_pipeline(df)
        except Exception as exc:
            log.error("pipeline_error", error=str(exc))
            return

        # Account / risk state
        account = self.executor.state()
        equity = account.get("equity", self.capital)
        risk_state = self.risk.update(equity)
        positions = self.executor.get_positions(self.symbol)
        recent_logs = self.logger.last_n(20)

        # Signal from pipeline
        signal = "HOLD"
        confidence = 0.5
        if pipeline.get("decision"):
            side = pipeline["decision"].get("side", "flat")
            if side == "long":
                signal = "BUY"
                confidence = pipeline["decision"].get("confidence", 0.5)
            elif side == "short":
                signal = "SELL"
                confidence = pipeline["decision"].get("confidence", 0.5)

        # LLM review
        review = self.llm.review(
            signal=signal,
            confidence=confidence,
            bar={"close": candle.close, "volume": candle.volume, "high": candle.high, "low": candle.low},
            positions=positions,
            account=account,
            risk_state=risk_state,
            recent_logs=recent_logs,
        )

        rprint(f"\n[bold]Signal: {signal} ({confidence:.2f})[/bold]")
        rprint(f"  LLM: {review.action} (conf={review.confidence:.2f})")
        rprint(f"  Reason: {review.reason}")

        # Risk validation
        ok, reason = self.risk.can_trade(review, risk_state)
        if not ok:
            rprint(f"[red]Risk block: {reason}[/red]")
            self._log(candle, review, account, risk_state, pipeline, reason)
            return

        # Execute
        if review.action in ("BUY", "SELL") and review.position_size > 0:
            side = "buy" if review.action == "BUY" else "sell"
            lot_size = self._calculate_lot_size(review.position_size, account.get("balance", self.capital))
            result = self.executor.place_order(self.symbol, side, lot_size, review.sl_price, review.tp_price)
            rprint(f"[green]Order: {result}[/green]")
            self.risk.record_trade(lot_size * candle.close)
            self._log(candle, review, account, risk_state, pipeline, f"executed: {result}")
        else:
            self._log(candle, review, account, risk_state, pipeline, "hold")
            rprint("[yellow]HOLD[/yellow]")

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
            meta_act=p_up > 0.52 or p_up < 0.48,
            meta_p_correct=abs(p_up - 0.5) * 2.0,
            model_version="forecast_agent",
            inference_ms=forecast_opinion.inference_ms,
        )

        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        agg_row = build_aggregator_row(
            forecast=forecast,
            opinions=opinions,
            spread_bps=0.0,
            realized_vol_30m=0.0,
            funding_rate=0.0,
            hour_utc=dt.hour,
            minute_of_hour=dt.minute,
            day_of_week=dt.weekday(),
        )

        decision = self.decision_engine.decide(agg_row, forecast)

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

    def _calculate_lot_size(self, position_size_pct: float, balance: float) -> float:
        notional = balance * position_size_pct
        return round(notional / self.feed._last_close, 4) if self.feed._last_close > 0 else 0.01

    def _log(self, candle: Candle, review: LLMReview, account: dict, risk_state: dict, pipeline: dict, exec_note: str) -> None:
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
            bar_close=candle.close,
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
    mock_execution: bool = typer.Option(False, help="Simulate MT5 (no real orders)"),
    use_llm: bool = typer.Option(False, help="Use LLM review (requires LLM_TOKEN)"),
    interval: int = typer.Option(5, help="Minutes between decisions"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    trader = AutonomousTrader(
        symbol=symbol,
        capital=capital,
        transformer_run=transformer_run,
        data_dir=data_dir,
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
