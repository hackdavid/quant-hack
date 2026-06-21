"""Feature engineering — identical pipeline for training and live trading."""

from intraday.features.calculator import (
    AggTrade,
    DepthBands,
    FeatureCalculator,
    KlineBar,
    MetricsUpdate,
)
from intraday.features.hawkes import HawkesCalculator
from intraday.features.pipeline import TransformationPipeline
from intraday.features.schema import (
    ALL_FEATURES,
    DEPTH_FEATURES,
    HAWKES_FEATURES,
    MARKET_FEATURES,
    PRICE_FEATURES,
    TARGET_COLS,
    VOLUME_FEATURES,
    VPIN_FEATURES,
    FeatureRow,
)
from intraday.features.store import LazyFeatureStore
from intraday.features.vpin import VPINCalculator

__all__ = [
    # Pipeline
    "TransformationPipeline",
    # Calculator + event types
    "FeatureCalculator",
    "AggTrade", "DepthBands", "KlineBar", "MetricsUpdate",
    # Sub-calculators (also usable standalone in live trading)
    "VPINCalculator",
    "HawkesCalculator",
    # Schema
    "FeatureRow",
    "ALL_FEATURES", "TARGET_COLS",
    "PRICE_FEATURES", "VOLUME_FEATURES", "DEPTH_FEATURES",
    "VPIN_FEATURES", "HAWKES_FEATURES", "MARKET_FEATURES",
    # Store
    "LazyFeatureStore",
]
