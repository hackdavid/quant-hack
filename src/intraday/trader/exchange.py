"""Binance USDT-M Futures exchange wrapper (paper + live).

Paper mode: all orders are simulated locally, no API calls made.
Live mode:  orders placed via ccxt (requires BINANCE_API_KEY + BINANCE_API_SECRET env vars).

Usage:
    # Paper
    ex = Exchange(symbol="BTCUSDT", paper=True)
    ex.set_paper_price(43500.0)
    fill = ex.place_order("buy", 0.01)   # returns fill dict

    # Live
    ex = Exchange(symbol="BTCUSDT", paper=False)
    fill = ex.place_order("buy", 0.01)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class Fill:
    side:       str      # "buy" | "sell"
    qty:        float    # BTC
    price:      float    # USDT
    fee_usdt:   float
    timestamp:  int      # ms
    order_id:   str
    simulated:  bool


class Exchange:
    TAKER_FEE = 0.0004   # 0.04%

    def __init__(
        self,
        symbol:   str   = "BTCUSDT",
        paper:    bool  = True,
        leverage: int   = 1,
    ) -> None:
        self.symbol   = symbol
        self.paper    = paper
        self.leverage = leverage

        self._paper_price: float = 0.0
        self._paper_position: float = 0.0    # BTC
        self._paper_equity:   float = 0.0    # starts uninitialised
        self._paper_order_id: int   = 0

        if not paper:
            self._init_live()

    # ── Paper trading ──────────────────────────────────────────────────────────

    def set_paper_price(self, price: float, equity: float | None = None) -> None:
        self._paper_price = price
        if equity is not None:
            self._paper_equity = equity

    def place_order(self, side: str, qty_btc: float) -> Fill:
        """side: 'buy' (long/cover) or 'sell' (short/close)."""
        if self.paper:
            return self._paper_fill(side, qty_btc)
        return self._live_fill(side, qty_btc)

    def cancel_all(self) -> None:
        if not self.paper:
            try:
                self._exchange.cancel_all_orders(self.symbol + "/USDT:USDT")
            except Exception as e:
                print(f"  cancel_all error: {e}")

    def get_position(self) -> float:
        """Current net position in BTC (+ long, - short)."""
        if self.paper:
            return self._paper_position
        try:
            positions = self._exchange.fetch_positions([self.symbol + "/USDT:USDT"])
            for p in positions:
                if p["symbol"] == self.symbol + "/USDT:USDT":
                    return float(p.get("contracts", 0)) * (1 if p["side"] == "long" else -1)
        except Exception:
            pass
        return 0.0

    def get_price(self) -> float:
        """Last traded price."""
        if self.paper:
            return self._paper_price
        try:
            ticker = self._exchange.fetch_ticker(self.symbol + "/USDT:USDT")
            return float(ticker["last"])
        except Exception:
            return 0.0

    # ── Internal: paper ────────────────────────────────────────────────────────

    def _paper_fill(self, side: str, qty_btc: float) -> Fill:
        price    = self._paper_price
        fee_usdt = price * abs(qty_btc) * self.TAKER_FEE
        if side == "buy":
            self._paper_position += qty_btc
        else:
            self._paper_position -= qty_btc
        self._paper_order_id += 1
        return Fill(
            side=side, qty=abs(qty_btc), price=price, fee_usdt=fee_usdt,
            timestamp=int(time.time() * 1000),
            order_id=f"paper_{self._paper_order_id}",
            simulated=True,
        )

    # ── Internal: live ─────────────────────────────────────────────────────────

    def _init_live(self) -> None:
        import ccxt
        api_key    = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if not api_key or not api_secret:
            raise EnvironmentError(
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )
        self._exchange = ccxt.binanceusdm({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {"defaultType": "future"},
        })
        self._exchange.set_leverage(self.leverage, self.symbol + "/USDT:USDT")
        print(f"  Live exchange connected: {self.symbol} leverage={self.leverage}x")

    def _live_fill(self, side: str, qty_btc: float) -> Fill:
        ccxt_side = "buy" if side == "buy" else "sell"
        order = self._exchange.create_market_order(
            self.symbol + "/USDT:USDT",
            ccxt_side,
            abs(qty_btc),
        )
        avg_price = float(order.get("average") or order.get("price") or 0)
        fee_usdt  = avg_price * abs(qty_btc) * self.TAKER_FEE
        return Fill(
            side=side, qty=abs(qty_btc), price=avg_price, fee_usdt=fee_usdt,
            timestamp=int(order.get("timestamp") or time.time() * 1000),
            order_id=str(order.get("id", "")),
            simulated=False,
        )
