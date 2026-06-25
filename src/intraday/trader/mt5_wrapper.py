"""MetaTrader 5 wrapper for algorithmic trading.

Provides a clean Pythonic interface over the official `MetaTrader5` library
for order execution, position management, and account state. Designed to be
used alongside a separate data feed (e.g., Binance WebSocket) where the AI
model is trained, with MT5 acting as the execution venue.

Usage:
    from intraday.trader.mt5_wrapper import MT5TradingWrapper

    mt5 = MT5TradingWrapper(account_id=123456, password="...", server="...")
    mt5.connect()
    mt5.market_order("BTCUSD", "buy", volume=0.01)
    print(mt5.account_state())
    mt5.shutdown()

⚠️  MT5 is Windows-only natively. On Linux/Mac run MT5 via Wine or a VPS.
⚠️  MT5 offers BTC *CFDs* (not Binance perpetuals). Symbol mapping is required.

Symbol mapping (Binance → MT5):
    BTCUSDT  → BTCUSD
    ETHUSDT  → ETHUSD
    XRPUSDT  → XRPUSD
    SOLUSDT  → SOLUSD
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — MetaTrader5 is Windows-only and may not be installed
# ---------------------------------------------------------------------------

_mt5: Any | None = None


def _load_mt5() -> Any:
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5_mod
            _mt5 = mt5_mod
        except ImportError as exc:
            raise ImportError(
                "MetaTrader5 not installed. Run: pip install MetaTrader5"
            ) from exc
    return _mt5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MT5_SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "XRPUSDT": "XRPUSD",
    "SOLUSDT": "SOLUSD",
    "ADAUSDT": "ADAUSD",
    "BNBUSDT": "BNBUSD",
    "DOGEUSDT": "DOGEUSD",
    "AVAXUSDT": "AVAXUSD",
}

# Reverse lookup for reporting
BINANCE_SYMBOL_MAP: dict[str, str] = {v: k for k, v in MT5_SYMBOL_MAP.items()}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class AccountState:
    balance: float
    equity: float
    profit: float
    margin: float
    free_margin: float
    margin_level: float

    def to_dict(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "profit": round(self.profit, 2),
            "margin": round(self.margin, 2),
            "free_margin": round(self.free_margin, 2),
            "margin_level": round(self.margin_level, 2),
        }


@dataclass
class Position:
    ticket: int
    symbol: str
    side: str
    volume: float
    open_price: float
    current_price: float
    profit: float
    sl: float
    tp: float
    swap: float
    time_ms: int

    def to_dict(self) -> dict:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "volume": round(self.volume, 4),
            "open_price": round(self.open_price, 5),
            "current_price": round(self.current_price, 5),
            "profit": round(self.profit, 2),
            "sl": round(self.sl, 5),
            "tp": round(self.tp, 5),
            "swap": round(self.swap, 2),
            "time_ms": self.time_ms,
        }


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    price: float | None
    volume: float | None
    comment: str
    retcode: int

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "ticket": self.ticket,
            "price": round(self.price, 5) if self.price else None,
            "volume": round(self.volume, 4) if self.volume else None,
            "comment": self.comment,
            "retcode": self.retcode,
        }


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class MT5TradingWrapper:
    """Production wrapper around the MetaTrader5 terminal.

    Args:
        account_id: MT5 account number
        password: MT5 account password
        server: MT5 broker server name (e.g., "XMGlobal-MT5")
        path: Optional path to terminal64.exe (for multiple terminals)
        magic: Unique EA identifier to tag your trades
    """

    def __init__(
        self,
        account_id: int,
        password: str,
        server: str,
        path: str | None = None,
        magic: int = 123456,
    ) -> None:
        self.account_id = account_id
        self.password = password
        self.server = server
        self.path = path
        self.magic = magic
        self._connected = False
        self._mt5 = _load_mt5()

    # ── Connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 and log in. Returns True on success.

        Avoids re-login if already connected to prevent MT5 disabling AutoTrading.
        """
        # ── Strategy 1: terminal already running and connected ──
        try:
            # If terminal_info() works, MT5 is already initialized
            tinfo = self._mt5.terminal_info()
            if tinfo is not None:
                info = self._mt5.account_info()
                if info is not None and info.login == self.account_id:
                    self._connected = True
                    log.info(
                        "mt5_already_connected",
                        server=self.server,
                        account=self.account_id,
                        balance=info.balance,
                    )
                    return True
        except Exception:
            pass  # Not initialized yet, proceed with full connect

        # ── Strategy 2: full connect (first run) ──
        init_kwargs: dict[str, Any] = {}
        if self.path:
            init_kwargs["path"] = self.path

        if not self._mt5.initialize(**init_kwargs):
            err = self._mt5.last_error()
            log.error("mt5_init_failed", error=err)
            return False

        # After initialize(), check if we're already logged in
        # This avoids re-login which disables AutoTrading
        try:
            info = self._mt5.account_info()
            if info is not None and info.login == self.account_id:
                self._connected = True
                log.info(
                    "mt5_connected_no_login",
                    server=self.server,
                    account=self.account_id,
                    balance=info.balance,
                    reason="already_logged_in",
                )
                return True
        except Exception:
            pass

        # Only login if not already logged in
        login_ok = self._mt5.login(
            login=self.account_id,
            password=self.password,
            server=self.server,
        )
        if not login_ok:
            err = self._mt5.last_error()
            log.error("mt5_login_failed", account=self.account_id, error=err)
            self._mt5.shutdown()
            return False

        self._connected = True
        info = self._mt5.account_info()
        log.info(
            "mt5_connected",
            server=self.server,
            account=self.account_id,
            balance=info.balance if info else None,
        )
        return True

    def shutdown(self) -> None:
        """Close the MT5 terminal connection."""
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
            log.info("mt5_shutdown")

    def __enter__(self) -> MT5TradingWrapper:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    # ── Account state ─────────────────────────────────────────────────────

    def account_state(self) -> AccountState | None:
        """Return current account metrics."""
        if not self._connected:
            log.warning("mt5_not_connected")
            return None
        info = self._mt5.account_info()
        if info is None:
            return None
        return AccountState(
            balance=info.balance,
            equity=info.equity,
            profit=info.profit,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level,
        )

    # ── Symbol helpers ────────────────────────────────────────────────────

    @staticmethod
    def to_mt5_symbol(binance_symbol: str) -> str:
        """Map Binance-style symbol to MT5 symbol."""
        return MT5_SYMBOL_MAP.get(binance_symbol.upper(), binance_symbol.upper())

    @staticmethod
    def to_binance_symbol(mt5_symbol: str) -> str:
        """Map MT5 symbol back to Binance-style."""
        return BINANCE_SYMBOL_MAP.get(mt5_symbol.upper(), mt5_symbol.upper())

    # ── Positions ───────────────────────────────────────────────────────────

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Return list of open positions (optionally filtered by symbol)."""
        if not self._connected:
            log.warning("mt5_not_connected")
            return []

        mt5_sym = self.to_mt5_symbol(symbol) if symbol else None
        raw = self._mt5.positions_get(symbol=mt5_sym) if mt5_sym else self._mt5.positions_get()
        if raw is None:
            return []

        positions: list[Position] = []
        for p in raw:
            positions.append(
                Position(
                    ticket=p.ticket,
                    symbol=self.to_binance_symbol(p.symbol),
                    side="long" if p.type == self._mt5.ORDER_TYPE_BUY else "short",
                    volume=p.volume,
                    open_price=p.price_open,
                    current_price=p.price_current,
                    profit=p.profit,
                    sl=p.sl,
                    tp=p.tp,
                    swap=p.swap,
                    time_ms=int(p.time * 1000),
                )
            )
        return positions

    def position_count(self, symbol: str | None = None) -> int:
        """Return number of open positions."""
        return len(self.get_positions(symbol))

    def total_profit(self, symbol: str | None = None) -> float:
        """Sum of profit for all open positions."""
        return sum(p.profit for p in self.get_positions(symbol))

    def get_current_price(self, symbol: str, side: str) -> float | None:
        """Get current tick price for a symbol."""
        if not self._connected:
            return None
        mt5_sym = self.to_mt5_symbol(symbol)
        tick = self._mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return None
        return float(tick.ask if side.lower() == "buy" else tick.bid)

    def get_min_stop_distance(self, symbol: str) -> float:
        """Get minimum stop distance for a symbol (price units)."""
        if not self._connected:
            return 0.03
        mt5_sym = self.to_mt5_symbol(symbol)
        info = self._mt5.symbol_info(mt5_sym)
        if info is None:
            return 0.03
        point = info.point
        trade_stops_level = info.trade_stops_level
        return max(trade_stops_level * point, 30 * point, 0.03)

    # ── Market orders ─────────────────────────────────────────────────────

    def market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "AI market order",
    ) -> OrderResult:
        """Send a market order (BUY or SELL).

        Args:
            symbol: Binance-style symbol (e.g., BTCUSDT)
            side: "buy" or "sell"
            volume: Lot size (e.g., 0.01)
            sl: Stop-loss price (0 = none)
            tp: Take-profit price (0 = none)
        """
        if not self._connected:
            return OrderResult(False, None, None, None, "Not connected", -1)

        mt5_sym = self.to_mt5_symbol(symbol)
        tick = self._mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return OrderResult(False, None, None, None, f"Symbol {mt5_sym} not found", -1)

        order_type = (
            self._mt5.ORDER_TYPE_BUY if side.lower() == "buy" else self._mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if order_type == self._mt5.ORDER_TYPE_BUY else tick.bid
        info = self._mt5.symbol_info(mt5_sym)
        point = info.point if info else 0.001
        trade_stops_level = info.trade_stops_level if info else 30
        # Use broker's actual trade_stops_level, not hardcoded 30
        min_distance = max(trade_stops_level * point, 30 * point, 0.03)
        log.info("mt5_symbol_info", symbol=mt5_sym, point=point, trade_stops_level=trade_stops_level, min_distance=min_distance)

        # Fix SL/TP against the ACTUAL tick price (not candle close)
        fixed_sl = sl
        fixed_tp = tp
        if side.lower() == "buy":
            if sl > 0 and sl >= price - min_distance:
                fixed_sl = round(price - min_distance * 2, 3)
                log.info("sl_adjusted_tick", symbol=symbol, price=price, old_sl=sl, new_sl=fixed_sl, min_distance=min_distance)
            if tp > 0 and tp <= price + min_distance:
                fixed_tp = round(price + min_distance * 2, 3)
                log.info("tp_adjusted_tick", symbol=symbol, price=price, old_tp=tp, new_tp=fixed_tp, min_distance=min_distance)
        else:
            if sl > 0 and sl <= price + min_distance:
                fixed_sl = round(price + min_distance * 2, 3)
                log.info("sl_adjusted_tick", symbol=symbol, price=price, old_sl=sl, new_sl=fixed_sl, min_distance=min_distance)
            if tp > 0 and tp >= price - min_distance:
                fixed_tp = round(price - min_distance * 2, 3)
                log.info("tp_adjusted_tick", symbol=symbol, price=price, old_tp=tp, new_tp=fixed_tp, min_distance=min_distance)

        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(fixed_sl),
            "tp": float(fixed_tp),
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        result = self._mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, None, None, "order_send returned None", -1)

        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            log.error("market_order_failed", symbol=symbol, side=side, retcode=result.retcode, comment=result.comment)
            return OrderResult(False, None, None, None, result.comment, result.retcode)

        log.info("market_order_filled", symbol=symbol, side=side, volume=volume, ticket=result.order, price=result.price)
        return OrderResult(True, result.order, result.price, volume, result.comment, result.retcode)

    # ── Pending orders ────────────────────────────────────────────────────

    def place_limit(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "AI limit order",
    ) -> OrderResult:
        """Place a limit order."""
        if not self._connected:
            return OrderResult(False, None, None, None, "Not connected", -1)

        mt5_sym = self.to_mt5_symbol(symbol)
        order_type = (
            self._mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == "buy" else self._mt5.ORDER_TYPE_SELL_LIMIT
        )

        request = {
            "action": self._mt5.TRADE_ACTION_PENDING,
            "symbol": mt5_sym,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.magic,
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            log.error("limit_order_failed", symbol=symbol, retcode=getattr(result, "retcode", -1))
            return OrderResult(False, None, None, None, getattr(result, "comment", "unknown"), getattr(result, "retcode", -1))

        return OrderResult(True, result.order, result.price, volume, result.comment, result.retcode)

    def modify_order(
        self,
        order_id: int,
        new_price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        """Modify an existing pending order (SL, TP, or trigger price)."""
        if not self._connected:
            return OrderResult(False, None, None, None, "Not connected", -1)

        request: dict[str, Any] = {
            "action": self._mt5.TRADE_ACTION_MODIFY,
            "order": order_id,
        }
        if new_price is not None:
            request["price"] = float(new_price)
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, None, None, getattr(result, "comment", "unknown"), getattr(result, "retcode", -1))

        return OrderResult(True, order_id, None, None, result.comment, result.retcode)

    def cancel_pending(self, order_id: int) -> bool:
        """Cancel a pending order by ticket."""
        if not self._connected:
            return False

        request = {
            "action": self._mt5.TRADE_ACTION_REMOVE,
            "order": order_id,
        }
        result = self._mt5.order_send(request)
        return result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE

    # ── Close positions ────────────────────────────────────────────────────

    def close_position(self, ticket: int) -> OrderResult:
        """Close an open position by ticket."""
        if not self._connected:
            return OrderResult(False, None, None, None, "Not connected", -1)

        positions = self._mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(False, None, None, None, f"Position {ticket} not found", -1)

        pos = positions[0]
        mt5_sym = pos.symbol
        close_type = (
            self._mt5.ORDER_TYPE_SELL if pos.type == self._mt5.ORDER_TYPE_BUY else self._mt5.ORDER_TYPE_BUY
        )
        tick = self._mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return OrderResult(False, None, None, None, f"Tick not found for {mt5_sym}", -1)

        price = tick.bid if close_type == self._mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": "AI close",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, None, None, getattr(result, "comment", "unknown"), getattr(result, "retcode", -1))

        log.info("position_closed", ticket=ticket, symbol=mt5_sym, price=result.price)
        return OrderResult(True, result.order, result.price, pos.volume, result.comment, result.retcode)

    def close_all_positions(self, symbol: str | None = None) -> list[OrderResult]:
        """Close all open positions (optionally filtered by symbol)."""
        results: list[OrderResult] = []
        for pos in self.get_positions(symbol):
            results.append(self.close_position(pos.ticket))
        return results

    # ── Live tick stream ──────────────────────────────────────────────────

    async def stream_ticks(
        self,
        symbols: list[str],
        callback: Callable[[dict], Any],
        interval: float = 0.1,
    ) -> None:
        """Asynchronous tick poller — pushes price updates to callback.

        Args:
            symbols: List of Binance-style symbols (e.g., ["BTCUSDT"])
            callback: Async or sync callable receiving tick dict
            interval: Poll interval in seconds (default 0.1 = 100ms)
        """
        if not self._connected:
            raise RuntimeError("MT5 not connected")

        mt5_syms = [self.to_mt5_symbol(s) for s in symbols]
        for s in mt5_syms:
            self._mt5.symbol_select(s, True)

        log.info("tick_stream_started", symbols=symbols, interval=interval)
        last_ts: dict[str, int] = {s: 0 for s in mt5_syms}

        while self._connected:
            for s in mt5_syms:
                tick = self._mt5.symbol_info_tick(s)
                if tick and tick.time_msc != last_ts[s]:
                    last_ts[s] = tick.time_msc
                    payload = {
                        "symbol": self.to_binance_symbol(s),
                        "mt5_symbol": s,
                        "timestamp_ms": tick.time_msc,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "last": tick.last,
                        "volume": tick.volume,
                        "account": self.account_state().to_dict() if self.account_state() else None,
                    }
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(payload)
                        else:
                            callback(payload)
                    except Exception as exc:
                        log.error("tick_callback_error", error=str(exc))
            await asyncio.sleep(interval)

    # ── Helper: dump state for LLM / logging ─────────────────────────────

    def state_snapshot(self, symbol: str | None = None) -> dict:
        """Return a JSON-serializable snapshot of account + positions."""
        return {
            "account": self.account_state().to_dict() if self.account_state() else None,
            "positions": [p.to_dict() for p in self.get_positions(symbol)],
            "timestamp_ms": int(time.time() * 1000),
        }


__all__ = [
    "MT5TradingWrapper",
    "AccountState",
    "Position",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "MT5_SYMBOL_MAP",
    "BINANCE_SYMBOL_MAP",
]
