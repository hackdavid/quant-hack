"""Main trading loop: WebSocket bar feed → features → signal → risk → execution.

Runs every 5-min bar close. Works in paper mode (no real orders) and live mode.

Architecture:
    Binance kline WS  →  rolling feature buffer  →  SignalCombiner
                                                           ↓
                                                      RiskAgent
                                                           ↓
                                                        Exchange
                                                           ↓
                                                     trade_log.jsonl

Usage (see scripts/run_paper_trade.py and scripts/run_live_trade.py):
    asyncio.run(TradingLoop(cfg).run())
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import numpy as np
import polars as pl
import websockets

from intraday.features.schema import ALL_FEATURES
from intraday.risk.agent import RiskAgent
from intraday.signal.combiner import SignalCombiner
from intraday.trader.exchange import Exchange


BINANCE_WS = "wss://fstream.binance.com/ws"   # USDT-M futures WebSocket
BAR_MS     = 5 * 60 * 1000                    # 5-min in ms


@dataclass
class BarRecord:
    bar_time_ms: int
    open: float; high: float; low: float; close: float
    volume: float; quote_volume: float
    taker_buy_vol: float; taker_sell_vol: float
    trade_count: int


@dataclass
class TradeLogEntry:
    ts:            str
    bar_time_ms:   int
    close_price:   float
    prob_up:       float
    signal:        int       # -1, 0, +1
    size_usd:      float
    position_btc:  float
    equity:        float
    daily_pnl_pct: float
    components:    dict


class TradingLoop:
    """
    Args:
        combiner:          SignalCombiner instance
        risk:              RiskAgent instance
        exchange:          Exchange instance (paper or live)
        initial_capital:   starting USD equity
        threshold:         min blended prob to enter a trade (default 0.55)
        symbol:            Binance symbol (default BTCUSDT)
        log_dir:           directory to write trade_log.jsonl
        buffer_size:       rolling window size for feature buffer (must >= seq_len + lags)
    """

    def __init__(
        self,
        combiner:        SignalCombiner,
        risk:            RiskAgent,
        exchange:        Exchange,
        initial_capital: float = 10_000.0,
        threshold:       float = 0.55,
        symbol:          str   = "BTCUSDT",
        log_dir:         str   = "logs/trader",
        buffer_size:     int   = 400,
    ) -> None:
        self.combiner  = combiner
        self.risk      = risk
        self.exchange  = exchange
        self.capital   = initial_capital
        self.threshold = threshold
        self.symbol    = symbol.lower()
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: Deque[BarRecord]     = deque(maxlen=buffer_size)
        self._position_btc: float          = 0.0
        self._equity: float                = initial_capital
        self._day_start_equity: float      = initial_capital
        self._log_path: Path               = self.log_dir / f"trade_log_{int(time.time())}.jsonl"

        self.risk.update_equity(initial_capital)
        print(f"  TradingLoop ready | threshold={threshold} | log={self._log_path}")

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect to Binance kline WebSocket and run until interrupted."""
        stream = f"{self.symbol}@kline_5m"
        url    = f"{BINANCE_WS}/{stream}"
        print(f"  Connecting to {url}")
        async with websockets.connect(url, ping_interval=20) as ws:
            print("  Connected. Waiting for bar closes...")
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("e") != "kline":
                    continue
                k   = msg["k"]
                bar = BarRecord(
                    bar_time_ms  = int(k["t"]),
                    open         = float(k["o"]),
                    high         = float(k["h"]),
                    low          = float(k["l"]),
                    close        = float(k["c"]),
                    volume       = float(k["v"]),
                    quote_volume = float(k["q"]),
                    taker_buy_vol  = float(k["V"]),
                    taker_sell_vol = float(k["v"]) - float(k["V"]),
                    trade_count  = int(k["n"]),
                )
                if k["x"]:   # bar is closed
                    await self._on_bar_close(bar)

    # ── Bar-close handler ──────────────────────────────────────────────────────

    async def _on_bar_close(self, bar: BarRecord) -> None:
        self._buffer.append(bar)
        self.exchange.set_paper_price(bar.close, self._equity)
        self._update_equity(bar.close)
        self.risk.update_equity(self._equity)

        ts_str = datetime.now(timezone.utc).isoformat()
        daily_pnl = (self._equity - self._day_start_equity) / self._day_start_equity

        print(f"\n[{ts_str}] bar={bar.bar_time_ms} close={bar.close:.2f}"
              f"  pos={self._position_btc:+.4f}  equity={self._equity:.2f}"
              f"  daily_pnl={daily_pnl*100:+.2f}%")

        # Need enough history for feature computation
        if len(self._buffer) < 50:
            print(f"  Warming up ({len(self._buffer)}/50 bars)")
            return

        # Build feature DataFrame from buffer
        try:
            df = self._build_feature_df()
        except Exception as e:
            print(f"  Feature error: {e}")
            return

        # Check risk limits
        can_trade, reason = self.risk.check_can_trade(self._equity)
        if not can_trade:
            print(f"  No trade: {reason}")
            return

        # Generate signal
        try:
            prob, components = self.combiner.predict_from_df(df)
        except Exception as e:
            print(f"  Prediction error: {e}")
            return

        direction = (1 if prob > self.threshold else (-1 if prob < 1 - self.threshold else 0))
        print(f"  prob={prob:.3f}  signal={direction:+d}  {components}")

        # Size and execute
        realized_vol = self._realized_vol()
        size_usd = self.risk.size_position(prob, self._equity, realized_vol)
        await self._execute(direction, size_usd, bar.close)

        # Log
        entry = TradeLogEntry(
            ts=ts_str, bar_time_ms=bar.bar_time_ms, close_price=bar.close,
            prob_up=prob, signal=direction, size_usd=size_usd,
            position_btc=self._position_btc, equity=self._equity,
            daily_pnl_pct=daily_pnl * 100, components=components,
        )
        with open(self._log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    # ── Execution ──────────────────────────────────────────────────────────────

    async def _execute(self, target_direction: int, size_usd: float, price: float) -> None:
        current_sign = 1 if self._position_btc > 0 else (-1 if self._position_btc < 0 else 0)
        if target_direction == current_sign:
            return   # already in the right direction

        # Close existing position
        if self._position_btc != 0:
            side = "sell" if self._position_btc > 0 else "buy"
            fill = self.exchange.place_order(side, abs(self._position_btc))
            self._equity -= fill.fee_usdt
            self._position_btc = 0.0
            print(f"  Closed: {side} {fill.qty:.4f} BTC @ {fill.price:.2f}"
                  f"  fee={fill.fee_usdt:.4f} USDT")

        # Open new position
        if target_direction != 0 and abs(size_usd) > 10:
            qty_btc = abs(size_usd) / price
            side    = "buy" if target_direction > 0 else "sell"
            fill    = self.exchange.place_order(side, qty_btc)
            self._equity      -= fill.fee_usdt
            self._position_btc = fill.qty if side == "buy" else -fill.qty
            print(f"  Opened: {side} {fill.qty:.4f} BTC @ {fill.price:.2f}"
                  f"  fee={fill.fee_usdt:.4f} USDT")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _update_equity(self, price: float) -> None:
        if self._position_btc != 0:
            self._equity += self._position_btc * (price - self._last_price)
        self._last_price = price

    def _realized_vol(self) -> float:
        if len(self._buffer) < 30:
            return 0.002
        closes = np.array([b.close for b in list(self._buffer)[-30:]])
        log_rets = np.diff(np.log(closes))
        return float(np.std(log_rets))

    def _build_feature_df(self) -> pl.DataFrame:
        """Convert buffer to a polars DataFrame with approximate features."""
        rows = list(self._buffer)
        closes       = [b.close        for b in rows]
        volumes      = [b.volume       for b in rows]
        quote_vols   = [b.quote_volume for b in rows]
        taker_buys   = [b.taker_buy_vol for b in rows]
        trade_counts = [b.trade_count  for b in rows]
        timestamps   = [b.bar_time_ms  for b in rows]
        n = len(rows)

        close_arr = np.array(closes, dtype=np.float64)
        log_rets  = np.concatenate([[0], np.diff(np.log(close_arr + 1e-10))])

        def rolling_std(arr, w):
            out = np.full(len(arr), np.nan)
            for i in range(w, len(arr)):
                out[i] = float(np.std(arr[i-w:i], ddof=1))
            return out

        taker_buy  = np.array(taker_buys, dtype=np.float64)
        vol_arr    = np.array(volumes, dtype=np.float64)
        taker_ratio = np.where(vol_arr > 0, taker_buy / vol_arr, 0.5)

        realized_vol = rolling_std(log_rets, 6)

        data = {
            "bar_time_ms":        timestamps,
            "close":              closes,
            "log_ret_1m":         log_rets.tolist(),
            "log_ret_5m":         log_rets.tolist(),
            "log_ret_15m":        [np.nanmean(log_rets[max(0,i-3):i+1]) for i in range(n)],
            "log_ret_60m":        [np.nanmean(log_rets[max(0,i-12):i+1]) for i in range(n)],
            "realized_vol_30m":   realized_vol.tolist(),
            "rsi_14":             self._rsi(close_arr, 14).tolist(),
            "vol_5m":             volumes,
            "taker_buy_ratio_5m": taker_ratio.tolist(),
            "trade_count_5m":     trade_counts,
            "avg_trade_size_5m":  [v/max(t,1) for v,t in zip(volumes, trade_counts)],
            "depth_imbalance_1pct": [0.0] * n,   # needs order book WS
            "vpin_50":              [0.0] * n,   # needs tick data
            "vpin_bucket_imbalance":[0.0] * n,
            "hawkes_buy_intensity": [0.0] * n,
            "hawkes_sell_intensity":[0.0] * n,
            "hawkes_net":           [0.0] * n,
            "oi_btc":               [0.0] * n,   # needs OI WS stream
            "oi_change_1h":         [0.0] * n,
            "ls_count_ratio":       [0.0] * n,
            "taker_ls_vol_ratio":   [0.0] * n,
        }
        # Fill any missing ALL_FEATURES columns with 0
        for col in ALL_FEATURES:
            if col not in data:
                data[col] = [0.0] * n

        return pl.DataFrame(data).sort("bar_time_ms")

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.full(len(close), np.nan)
        avg_l = np.full(len(close), np.nan)
        if len(close) > period:
            avg_g[period] = gain[1:period+1].mean()
            avg_l[period] = loss[1:period+1].mean()
            for i in range(period+1, len(close)):
                avg_g[i] = (avg_g[i-1] * (period-1) + gain[i]) / period
                avg_l[i] = (avg_l[i-1] * (period-1) + loss[i]) / period
        rs  = np.where(avg_l > 0, avg_g / avg_l, 100.0)
        rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, nan=50.0)
