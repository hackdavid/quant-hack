"""Core simulator loop, order models, fill models, and result types.

SimulatorLoop drives event-by-event simulation with latency-delayed order processing.
StrategyContext provides a read-only, future-data-free view to strategies.
"""

from __future__ import annotations

import copy
import heapq
import math
import uuid
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, Field

from intraday.sim.book import LocalOrderBook
from intraday.sim.events import DepthEvent, FundingEvent, MarkEvent, TradeEvent

if TYPE_CHECKING:
    from intraday.sim.account import Account
    from intraday.sim.events import Event
    from intraday.sim.latency import LatencyModel
    from intraday.sim.matching import MatchingEngine
    from intraday.sim.strategies.base import Strategy

log = structlog.get_logger(__name__)


class OrderRequest(BaseModel):
    side: Literal["buy", "sell"]
    qty_base: float
    type: Literal["market", "limit", "post_only", "ioc"]
    limit_price: float | None = None
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"
    reduce_only: bool = False
    client_order_id: str


class Fill(BaseModel):
    ts_ms: int
    order_id: str
    side: Literal["buy", "sell"]
    qty_base: float
    price: float
    is_maker: bool
    fee_quote: float


class RunResult(BaseModel):
    run_id: str
    start_ms: int
    end_ms: int
    n_events: int
    n_orders: int
    n_fills: int
    fill_rate: float
    gross_pnl_quote: float
    net_pnl_quote: float
    fees_paid_quote: float
    funding_paid_quote: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    avg_slippage_bps: float


class StrategyContext:
    """Read-only view for strategies. Never exposes future data."""

    def __init__(
        self,
        book: LocalOrderBook,
        account: "Account",
        ts_ms: int,
        mark_price: float = 0.0,
        feature_snapshot: dict | None = None,
    ) -> None:
        self._book = book
        self._account = copy.copy(account)
        self._ts_ms = ts_ms
        self._mark_price = mark_price
        self._feature_snapshot = feature_snapshot

    @property
    def book(self) -> LocalOrderBook:
        return self._book

    @property
    def account(self) -> "Account":
        return self._account

    @property
    def ts_ms(self) -> int:
        return self._ts_ms

    @property
    def mark_price(self) -> float:
        return self._mark_price

    @property
    def feature_snapshot(self) -> dict | None:
        return self._feature_snapshot


def _compute_sharpe(equity_deltas: list[float]) -> float:
    if len(equity_deltas) < 2:
        return 0.0
    n = len(equity_deltas)
    mean = sum(equity_deltas) / n
    variance = sum((x - mean) ** 2 for x in equity_deltas) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0.0:
        return 0.0
    return mean / std * math.sqrt(252 * 288)  # annualized for 5-min bars


def _compute_sortino(equity_deltas: list[float]) -> float:
    if len(equity_deltas) < 2:
        return 0.0
    n = len(equity_deltas)
    mean = sum(equity_deltas) / n
    downside = [min(x, 0.0) for x in equity_deltas]
    down_var = sum(x ** 2 for x in downside) / max(n - 1, 1)
    down_std = math.sqrt(down_var) if down_var > 0 else 0.0
    if down_std == 0.0:
        return 0.0
    return mean / down_std * math.sqrt(252 * 288)


def _compute_calmar(net_pnl: float, start_equity: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct == 0.0 or start_equity == 0.0:
        return 0.0
    annual_return = (net_pnl / start_equity)
    return annual_return / (max_drawdown_pct / 100.0)


class SimulatorLoop:
    def __init__(
        self,
        *,
        events: list["Event"],
        book: LocalOrderBook,
        matching: "MatchingEngine",
        account: "Account",
        strategy: "Strategy",
        latency: "LatencyModel",
        run_id: str,
        seed: int = 42,
    ) -> None:
        self._events = sorted(events, key=lambda e: e.ts_ms)
        self._book = book
        self._matching = matching
        self._account = account
        self._strategy = strategy
        self._latency = latency
        self._run_id = run_id
        self._seed = seed

    def run(self) -> RunResult:
        from intraday.sim.account import Account
        from intraday.sim.costs import apply_funding, fee_for_fill

        events = self._events
        if not events:
            raise ValueError("No events to simulate")

        start_ms = events[0].ts_ms
        end_ms = events[-1].ts_ms
        start_equity = self._account.cash_quote

        all_fills: list[Fill] = []
        equity_curve: list[tuple[int, float]] = []
        pending_orders: list[tuple[float, OrderRequest]] = []  # (deliver_ts_ms, req)

        n_orders = 0
        n_events = len(events)
        peak_equity = start_equity
        max_drawdown_pct = 0.0
        mark_price = 0.0
        slippage_bps_list: list[float] = []

        prev_equity = start_equity
        equity_deltas: list[float] = []

        for event in events:
            ts_ms = event.ts_ms

            # Deliver any orders whose latency has elapsed
            still_pending: list[tuple[float, OrderRequest]] = []
            for deliver_ts, req in pending_orders:
                if deliver_ts <= ts_ms:
                    fills = self._process_order(req, ts_ms, mark_price)
                    for fill in fills:
                        fee = fee_for_fill(fill)
                        fill = Fill(
                            ts_ms=fill.ts_ms,
                            order_id=fill.order_id,
                            side=fill.side,
                            qty_base=fill.qty_base,
                            price=fill.price,
                            is_maker=fill.is_maker,
                            fee_quote=fee,
                        )
                        self._account.update_on_fill(fill, fee)
                        all_fills.append(fill)
                        self._strategy.on_fill(fill, self._make_ctx(ts_ms, mark_price))
                        log.info(
                            "order.filled",
                            run_id=self._run_id,
                            order_id=fill.order_id,
                            side=fill.side,
                            qty=fill.qty_base,
                            price=fill.price,
                            fee=fee,
                        )
                else:
                    still_pending.append((deliver_ts, req))
            pending_orders = still_pending

            # Update book from market events
            if isinstance(event, DepthEvent):
                if event.is_snapshot:
                    self._book.apply_snapshot(event.bids, event.asks)
                else:
                    depth_fills = self._matching.on_depth_event(event, self._book)
                    self._book.apply_diff(event.bids, event.asks)
                    for fill in depth_fills:
                        fee = fee_for_fill(fill)
                        fill = Fill(
                            ts_ms=fill.ts_ms,
                            order_id=fill.order_id,
                            side=fill.side,
                            qty_base=fill.qty_base,
                            price=fill.price,
                            is_maker=fill.is_maker,
                            fee_quote=fee,
                        )
                        self._account.update_on_fill(fill, fee)
                        all_fills.append(fill)
                        self._strategy.on_fill(fill, self._make_ctx(ts_ms, mark_price))

            elif isinstance(event, TradeEvent):
                trade_fills = self._matching.on_trade_event(event, self._book)
                for fill in trade_fills:
                    fee = fee_for_fill(fill)
                    fill = Fill(
                        ts_ms=fill.ts_ms,
                        order_id=fill.order_id,
                        side=fill.side,
                        qty_base=fill.qty_base,
                        price=fill.price,
                        is_maker=fill.is_maker,
                        fee_quote=fee,
                    )
                    self._account.update_on_fill(fill, fee)
                    all_fills.append(fill)
                    self._strategy.on_fill(fill, self._make_ctx(ts_ms, mark_price))

            elif isinstance(event, FundingEvent):
                mark_price = event.mark_price
                payment = apply_funding(self._account, event)
                if payment != 0.0:
                    log.info(
                        "funding.applied",
                        run_id=self._run_id,
                        ts_ms=ts_ms,
                        rate=event.funding_rate,
                        payment=payment,
                    )

            elif isinstance(event, MarkEvent):
                mark_price = event.mark_price

            # Update mark price from bar close if no explicit mark
            if hasattr(event, "close") and mark_price == 0.0:
                mark_price = event.close  # type: ignore[attr-defined]
            elif hasattr(event, "close"):
                mark_price = event.close  # type: ignore[attr-defined]

            # Give strategy a view and collect new order requests
            ctx = self._make_ctx(ts_ms, mark_price)
            new_requests = self._strategy.on_event(event, ctx)

            for req in new_requests:
                latency_ms = self._latency.sample_ms()
                deliver_ts = ts_ms + latency_ms
                pending_orders.append((deliver_ts, req))
                n_orders += 1
                log.info(
                    "order.submitted",
                    run_id=self._run_id,
                    client_order_id=req.client_order_id,
                    side=req.side,
                    qty=req.qty_base,
                    type=req.type,
                    deliver_at_ms=deliver_ts,
                )

            # Track equity and drawdown
            current_equity = self._account.equity(mark_price if mark_price > 0 else self._account.avg_entry_price)
            equity_curve.append((ts_ms, current_equity))
            delta = current_equity - prev_equity
            equity_deltas.append(delta)
            prev_equity = current_equity

            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = self._account.drawdown_pct(mark_price if mark_price > 0 else self._account.avg_entry_price, peak_equity)
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

            if hasattr(event, "kind") and event.kind == "bar":
                log.debug(
                    "data.bar_received",
                    run_id=self._run_id,
                    ts_ms=ts_ms,
                    equity=current_equity,
                )

        # Process any remaining pending orders at end_ms
        for deliver_ts, req in pending_orders:
            fills = self._process_order(req, end_ms, mark_price)
            for fill in fills:
                fee = fee_for_fill(fill)
                fill = Fill(
                    ts_ms=fill.ts_ms,
                    order_id=fill.order_id,
                    side=fill.side,
                    qty_base=fill.qty_base,
                    price=fill.price,
                    is_maker=fill.is_maker,
                    fee_quote=fee,
                )
                self._account.update_on_fill(fill, fee)
                all_fills.append(fill)

        # Compute slippage stats
        for fill in all_fills:
            if not fill.is_maker:
                mid = self._book.mid_price()
                if mid > 0:
                    slip = abs(fill.price - mid) / mid * 10_000
                    slippage_bps_list.append(slip)

        final_equity = self._account.equity(mark_price if mark_price > 0 else start_equity)
        gross_pnl = self._account.realized_pnl_quote + self._account.unrealized_pnl(
            mark_price if mark_price > 0 else start_equity
        )
        net_pnl = final_equity - start_equity

        log.info(
            "pnl.update",
            run_id=self._run_id,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=self._account.fee_paid_quote,
            funding=self._account.funding_paid_quote,
            max_dd_pct=max_drawdown_pct,
        )

        return RunResult(
            run_id=self._run_id,
            start_ms=start_ms,
            end_ms=end_ms,
            n_events=n_events,
            n_orders=n_orders,
            n_fills=len(all_fills),
            fill_rate=len(all_fills) / max(n_orders, 1),
            gross_pnl_quote=gross_pnl,
            net_pnl_quote=net_pnl,
            fees_paid_quote=self._account.fee_paid_quote,
            funding_paid_quote=self._account.funding_paid_quote,
            max_drawdown_pct=max_drawdown_pct,
            sharpe=_compute_sharpe(equity_deltas),
            sortino=_compute_sortino(equity_deltas),
            calmar=_compute_calmar(net_pnl, start_equity, max_drawdown_pct),
            avg_slippage_bps=sum(slippage_bps_list) / max(len(slippage_bps_list), 1),
        )

    def _make_ctx(self, ts_ms: int, mark_price: float) -> StrategyContext:
        return StrategyContext(
            book=self._book,
            account=self._account,
            ts_ms=ts_ms,
            mark_price=mark_price,
        )

    def _process_order(self, req: OrderRequest, ts_ms: int, mark_price: float = 0.0) -> list[Fill]:
        """Route order to matching engine based on type."""
        if req.type in ("market", "ioc"):
            fills = self._matching.on_market_order(req, self._book, ts_ms)
            if not fills and mark_price > 0:
                # Book has no levels (common when depth data is aggregated/missing).
                # Fall back to filling at mark ± half-spread (taker cost already accounted for in fees).
                import uuid
                HALF_SPREAD_BPS = 2.5
                if req.side == "buy":
                    fill_price = mark_price * (1 + HALF_SPREAD_BPS / 10_000)
                else:
                    fill_price = mark_price * (1 - HALF_SPREAD_BPS / 10_000)
                fills = [Fill(
                    ts_ms=ts_ms,
                    order_id=str(uuid.uuid4()),
                    side=req.side,
                    qty_base=req.qty_base,
                    price=fill_price,
                    is_maker=False,
                    fee_quote=0.0,
                )]
                log.debug("matching.markprice_fallback", side=req.side, price=fill_price, qty=req.qty_base)
            return fills
        elif req.type in ("limit", "post_only"):
            self._matching.place_order(req, self._book, ts_ms)
            return []
        return []


__all__ = ["OrderRequest", "Fill", "RunResult", "StrategyContext", "SimulatorLoop"]
