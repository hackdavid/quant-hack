"""FeatureCalculator — stateful, event-driven feature computation.

IDENTICAL code path for:
  Batch mode   — replay sorted historical Parquet events day by day
  Live mode    — consume Binance futures WebSocket events in real-time

Usage (batch):
    calc = FeatureCalculator()
    for event in sorted_events:          # AggTrade | DepthBands | MetricsUpdate | KlineBar
        calc.dispatch(event)
    rows = calc.flush()                  # remaining rows at end of day

Usage (live, per WS message):
    calc.dispatch(trade_from_ws_msg)
    # at each 5m kline close:
    row = calc.dispatch(kline_5m_bar)    # returns FeatureRow or None
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

from intraday.features.hawkes import HawkesCalculator
from intraday.features.schema import FeatureRow
from intraday.features.vpin import VPINCalculator


# ---------------------------------------------------------------------------
# Input event types — lightweight dataclasses, no validation overhead
# ---------------------------------------------------------------------------

@dataclass
class AggTrade:
    time_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool   # True = sell-aggressor (taker sold)


@dataclass
class DepthBands:
    """Depth snapshot in %-band format.

    Populated from:
      Historical  — bookDepth Parquet (already wide-format)
      Live        — depth20 WS msg → binance_bulk.depth_bands_from_top20()
    """
    snapshot_time_ms: int
    bid_02pct: float = 0.0; bid_1pct: float = 0.0
    bid_2pct: float = 0.0;  bid_3pct: float = 0.0
    bid_4pct: float = 0.0;  bid_5pct: float = 0.0
    ask_02pct: float = 0.0; ask_1pct: float = 0.0
    ask_2pct: float = 0.0;  ask_3pct: float = 0.0
    ask_4pct: float = 0.0;  ask_5pct: float = 0.0


@dataclass
class MetricsUpdate:
    create_time_ms: int
    oi_btc: float
    oi_usd: float
    ls_count_ratio: float
    taker_ls_vol_ratio: float
    top_ls_count: float = 0.0
    top_ls_value: float = 0.0
    funding_rate: float = 0.0


@dataclass
class KlineBar:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    taker_buy_volume: float
    interval: str = "1m"   # "1m" or "5m"


# Union type for dispatch
Event = Union[AggTrade, DepthBands, MetricsUpdate, KlineBar]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_ret(prev: float, curr: float) -> float:
    if prev > 0 and curr > 0:
        return math.log(curr / prev)
    return 0.0


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# FeatureCalculator
# ---------------------------------------------------------------------------

class FeatureCalculator:
    """Pure state machine — feed events, get FeatureRows at 5m bar closes."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        vpin_bucket_btc: float = 100.0,
        vpin_window: int = 50,
        hawkes_alpha: float = 1.0,
        hawkes_beta: float = 10.0,
        hawkes_mu: float = 6.0,
        direction_threshold: float = 0.0005,
    ) -> None:
        self.symbol = symbol
        self._thresh = direction_threshold

        # Price history
        self._klines_1m: deque[KlineBar] = deque(maxlen=60)
        self._klines_5m: deque[KlineBar] = deque(maxlen=15)

        # Trade buffer for current 5m bar (holds KlineBar volume from klines,
        # but we still need the tick stream for VPIN / Hawkes)
        self._bar_trade_count: int = 0

        # Depth
        self._depth_prev: Optional[DepthBands] = None
        self._depth_latest: Optional[DepthBands] = None
        self._ofi_accum: list[float] = []

        # Metrics
        self._metrics_latest: Optional[MetricsUpdate] = None
        self._metrics_history: deque[MetricsUpdate] = deque(maxlen=13)

        # VPIN — state persists across bars and day boundaries
        self._vpin = VPINCalculator(bucket_btc=vpin_bucket_btc, window=vpin_window)

        # Hawkes — state persists across bars and day boundaries
        self._hawkes = HawkesCalculator(alpha=hawkes_alpha, beta=hawkes_beta, mu=hawkes_mu)

        # Pending rows waiting for forward-target attachment.
        # No maxlen — we manually popleft() when a row is ready.
        self._pending: deque[FeatureRow] = deque()

    # ── Public dispatch ───────────────────────────────────────────────────

    def dispatch(self, event: Event) -> Optional[FeatureRow]:
        """Route any event to the correct handler.

        Returns a completed FeatureRow when a 5m bar closes and the row
        that's 12 bars old (i.e., it now has its 60m forward target filled).
        Returns None otherwise.
        """
        if isinstance(event, AggTrade):
            self._on_trade(event)
        elif isinstance(event, DepthBands):
            self._on_depth(event)
        elif isinstance(event, MetricsUpdate):
            self._on_metrics(event)
        elif isinstance(event, KlineBar):
            if event.interval == "1m":
                self._on_kline_1m(event)
            elif event.interval == "5m":
                return self._on_kline_5m(event)
        return None

    def flush(self) -> list[FeatureRow]:
        """End-of-day flush: attach available targets and return all pending rows."""
        self._attach_targets()
        rows = list(self._pending)
        self._pending.clear()
        return rows

    # ── Internal handlers ─────────────────────────────────────────────────

    def _on_trade(self, t: AggTrade) -> None:
        self._vpin.update(t.quantity, t.is_buyer_maker)
        self._hawkes.update(t.time_ms, t.is_buyer_maker)
        self._bar_trade_count += 1

    def _on_depth(self, d: DepthBands) -> None:
        if self._depth_latest is not None:
            db = d.bid_02pct - self._depth_latest.bid_02pct
            da = d.ask_02pct - self._depth_latest.ask_02pct
            self._ofi_accum.append(db - da)
        self._depth_prev = self._depth_latest
        self._depth_latest = d

    def _on_metrics(self, m: MetricsUpdate) -> None:
        self._metrics_latest = m
        self._metrics_history.append(m)

    def _on_kline_1m(self, bar: KlineBar) -> None:
        self._klines_1m.append(bar)

    def _on_kline_5m(self, bar: KlineBar) -> Optional[FeatureRow]:
        # Decay Hawkes to bar close time before sampling
        self._hawkes.decay_to(bar.close_time_ms)

        row = self._build_row(bar)
        self._klines_5m.append(bar)
        self._ofi_accum.clear()
        self._bar_trade_count = 0

        if row is None:
            return None

        self._pending.append(row)
        self._attach_targets()

        # Once the oldest row has its 60m forward return (needs 12 future bars),
        # pop and return it — each row is returned exactly once.
        if len(self._pending) >= 13:
            return self._pending.popleft()
        return None

    # ── Feature computation ───────────────────────────────────────────────

    def _build_row(self, bar: KlineBar) -> Optional[FeatureRow]:
        closes_1m = [k.close for k in self._klines_1m]
        if not closes_1m:
            return None

        closes_5m = [k.close for k in self._klines_5m]
        close = bar.close

        # ── Price features ────────────────────────────────────────────────
        log_ret_1m  = _log_ret(closes_1m[-1], close)
        log_ret_5m  = _log_ret(closes_5m[-1], close) if closes_5m else 0.0
        log_ret_15m = _log_ret(closes_5m[-3],  close) if len(closes_5m) >= 3  else None
        log_ret_60m = _log_ret(closes_5m[-12], close) if len(closes_5m) >= 12 else None

        realized_vol_30m: Optional[float] = None
        n = min(30, len(closes_1m))
        if n >= 2:
            rets = [_log_ret(closes_1m[i-1], closes_1m[i]) for i in range(-n+1, 0)]
            rets.append(log_ret_1m)
            mean = sum(rets) / len(rets)
            var  = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            realized_vol_30m = math.sqrt(var)

        rsi_14 = _rsi(closes_5m + [close]) if len(closes_5m) >= 14 else None

        # ── Volume / taker features ───────────────────────────────────────
        vol_5m    = bar.volume
        tbv       = bar.taker_buy_volume
        tbr       = tbv / vol_5m if vol_5m > 0 else 0.5
        tc        = bar.trade_count if bar.trade_count > 0 else max(self._bar_trade_count, 1)
        avg_size  = vol_5m / tc if tc > 0 else 0.0

        # ── Depth features ────────────────────────────────────────────────
        d_imb_02 = d_imb_1 = bid_02 = ask_02 = ofi = None
        if self._depth_latest is not None:
            d = self._depth_latest
            b02, a02 = d.bid_02pct, d.ask_02pct
            b1,  a1  = d.bid_1pct,  d.ask_1pct
            if b02 + a02 > 0:
                d_imb_02 = (b02 - a02) / (b02 + a02)
            if b1 + a1 > 0:
                d_imb_1 = (b1 - a1) / (b1 + a1)
            bid_02, ask_02 = b02, a02
        if self._ofi_accum:
            ofi = sum(self._ofi_accum)

        # ── VPIN ──────────────────────────────────────────────────────────
        vpin_val   = self._vpin.vpin()
        vpin_bkt   = self._vpin.current_bucket_imbalance()

        # ── Hawkes ────────────────────────────────────────────────────────
        h_buy  = self._hawkes.buy_intensity
        h_sell = self._hawkes.sell_intensity
        h_net  = self._hawkes.net

        # ── Market structure ──────────────────────────────────────────────
        oi_btc = oi_chg = ls_r = taker_lr = fund = None
        if self._metrics_latest is not None:
            m = self._metrics_latest
            oi_btc  = m.oi_btc
            ls_r    = m.ls_count_ratio
            taker_lr = m.taker_ls_vol_ratio
            fund    = m.funding_rate
            if len(self._metrics_history) >= 12:
                old = self._metrics_history[-12].oi_btc
                if old > 0:
                    oi_chg = (oi_btc - old) / old

        return FeatureRow(
            bar_time_ms=bar.open_time_ms,
            symbol=self.symbol,
            close=close,
            log_ret_1m=log_ret_1m,
            log_ret_5m=log_ret_5m,
            log_ret_15m=log_ret_15m,
            log_ret_60m=log_ret_60m,
            realized_vol_30m=realized_vol_30m,
            rsi_14=rsi_14,
            vol_5m=vol_5m,
            taker_buy_ratio_5m=tbr,
            trade_count_5m=tc,
            avg_trade_size_5m=avg_size,
            depth_imbalance_02pct=d_imb_02,
            depth_imbalance_1pct=d_imb_1,
            bid_depth_02pct=bid_02,
            ask_depth_02pct=ask_02,
            ofi_5m=ofi,
            vpin_50=vpin_val,
            vpin_bucket_imbalance=vpin_bkt,
            hawkes_buy_intensity=h_buy,
            hawkes_sell_intensity=h_sell,
            hawkes_net=h_net,
            oi_btc=oi_btc,
            oi_change_1h=oi_chg,
            ls_count_ratio=ls_r,
            taker_ls_vol_ratio=taker_lr,
            funding_rate=fund,
        )

    def _attach_targets(self) -> None:
        rows = list(self._pending)
        n = len(rows)
        for i, r in enumerate(rows):
            if r.fwd_ret_5m is None and i + 1 < n:
                ret = _log_ret(r.close, rows[i + 1].close)
                r.fwd_ret_5m = ret
                r.fwd_direction_5m = (
                    1 if ret >  self._thresh else
                   -1 if ret < -self._thresh else 0
                )
            if r.fwd_ret_15m is None and i + 3 < n:
                r.fwd_ret_15m = _log_ret(r.close, rows[i + 3].close)
            if r.fwd_ret_60m is None and i + 12 < n:
                r.fwd_ret_60m = _log_ret(r.close, rows[i + 12].close)
