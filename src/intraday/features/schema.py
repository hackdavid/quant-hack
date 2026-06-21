"""Canonical feature row — one per 5-minute bar.

CONTRACT between:
  TransformationPipeline  (produces rows from any data source)
  ML model training       (consumes rows via LazyFeatureStore)
  Live paper trading      (produces rows in real-time from WS events)

Adding/removing fields requires updating the pipeline AND the model input layer.
"""

from typing import Optional

from pydantic import BaseModel, Field


class FeatureRow(BaseModel):
    """All features for one 5-minute bar close."""

    # ── Identity ──────────────────────────────────────────────────────────
    bar_time_ms: int = Field(description="Bar open time in ms UTC")
    symbol: str = "BTCUSDT"

    # ── Price ─────────────────────────────────────────────────────────────
    close: float
    log_ret_1m: float   = Field(description="Log return of the last 1m bar")
    log_ret_5m: float   = Field(description="Log return of this 5m bar")
    log_ret_15m: Optional[float] = None
    log_ret_60m: Optional[float] = None
    realized_vol_30m: Optional[float] = Field(None, description="Std-dev of 1m log-returns over last 30 bars")
    rsi_14:      Optional[float] = Field(None, description="RSI(14) on 5m close prices")

    # ── Volume / Taker flow ───────────────────────────────────────────────
    vol_5m:             float = Field(description="Total BTC volume in this 5m bar")
    taker_buy_ratio_5m: float = Field(description="Taker-buy / total volume [0, 1]")
    trade_count_5m:     int   = Field(description="aggTrade count in this 5m bar")
    avg_trade_size_5m:  float = Field(description="Average aggTrade size (BTC)")

    # ── Order book depth (%-band format, aligned between hist & live) ─────
    depth_imbalance_02pct: Optional[float] = Field(None, description="(bid−ask)/(bid+ask) at ±0.2% band")
    depth_imbalance_1pct:  Optional[float] = Field(None, description="(bid−ask)/(bid+ask) at ±1% band")
    bid_depth_02pct:       Optional[float] = Field(None, description="Cumulative bid depth within 0.2% of mid (BTC)")
    ask_depth_02pct:       Optional[float] = Field(None, description="Cumulative ask depth within 0.2% of mid (BTC)")
    ofi_5m:                Optional[float] = Field(None, description="Sum of signed depth-changes at ±0.2% band over 5m bar")

    # ── VPIN ──────────────────────────────────────────────────────────────
    vpin_50:               Optional[float] = Field(None, description="VPIN over last 50 volume buckets (~65min)")
    vpin_bucket_imbalance: Optional[float] = Field(None, description="Buy-volume fraction in current open bucket [0,1]")

    # ── Hawkes intensities ────────────────────────────────────────────────
    hawkes_buy_intensity:  Optional[float] = Field(None, description="λ_buy(t) at bar close (trades/s)")
    hawkes_sell_intensity: Optional[float] = Field(None, description="λ_sell(t) at bar close (trades/s)")
    hawkes_net:            Optional[float] = Field(None, description="(λ_buy−λ_sell)/(λ_buy+λ_sell) ∈ [−1,1]")

    # ── Market structure ──────────────────────────────────────────────────
    oi_btc:           Optional[float] = Field(None, description="Open interest in BTC")
    oi_change_1h:     Optional[float] = Field(None, description="Fractional OI change vs 60m ago")
    ls_count_ratio:   Optional[float] = Field(None, description="Long/short account count ratio")
    taker_ls_vol_ratio: Optional[float] = Field(None, description="Taker buy/sell volume ratio")
    funding_rate:     Optional[float] = Field(None, description="Latest funding rate")

    # ── Forward targets (filled post-hoc; None for the last few bars) ─────
    fwd_ret_5m:       Optional[float] = Field(None, description="Log return of the NEXT 5m bar")
    fwd_ret_15m:      Optional[float] = None
    fwd_ret_60m:      Optional[float] = None
    fwd_direction_5m: Optional[int]   = Field(None, description="1=up / −1=down / 0=flat for next 5m bar")


# ── Column groups (for model input selection) ──────────────────────────────

PRICE_FEATURES   = ["log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_60m",
                    "realized_vol_30m", "rsi_14"]
VOLUME_FEATURES  = ["vol_5m", "taker_buy_ratio_5m", "trade_count_5m", "avg_trade_size_5m"]
DEPTH_FEATURES   = ["depth_imbalance_02pct", "depth_imbalance_1pct",
                    "bid_depth_02pct", "ask_depth_02pct", "ofi_5m"]
VPIN_FEATURES    = ["vpin_50", "vpin_bucket_imbalance"]
HAWKES_FEATURES  = ["hawkes_buy_intensity", "hawkes_sell_intensity", "hawkes_net"]
MARKET_FEATURES  = ["oi_btc", "oi_change_1h", "ls_count_ratio",
                    "taker_ls_vol_ratio", "funding_rate"]

ALL_FEATURES = (PRICE_FEATURES + VOLUME_FEATURES + DEPTH_FEATURES +
                VPIN_FEATURES + HAWKES_FEATURES + MARKET_FEATURES)
TARGET_COLS  = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "fwd_direction_5m"]
