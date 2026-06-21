"""Data acquisition and management."""

from intraday.data.binance_bulk import BulkKind, depth_bands_from_top20, download_bulk
from intraday.data.capture import CaptureConfig, capture_live
from intraday.data.checkpoint import Checkpoint, CheckpointEntry, get_checkpoint_path
from intraday.data.schemas import (
    Depth,
    DepthLevel,
    FundingRate,
    Kline,
    Liquidation,
    MarkPrice,
    OpenInterest,
    Trade,
)

__all__ = [
    # Schemas
    "Kline",
    "Trade",
    "Depth",
    "DepthLevel",
    "FundingRate",
    "OpenInterest",
    "MarkPrice",
    "Liquidation",
    # Bulk download (data.binance.vision)
    "BulkKind",
    "download_bulk",
    "depth_bands_from_top20",
    # Live capture
    "CaptureConfig",
    "capture_live",
    # Checkpoint
    "Checkpoint",
    "CheckpointEntry",
    "get_checkpoint_path",
]
