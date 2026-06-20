"""Test checkpoint tracking system."""

import json
from pathlib import Path

import pytest

from intraday.data.checkpoint import Checkpoint, CheckpointEntry


def test_checkpoint_empty() -> None:
    """Test empty checkpoint creation."""
    cp = Checkpoint()
    assert len(cp.entries) == 0
    assert cp.summary() == "No checkpoints found."


def test_checkpoint_update() -> None:
    """Test checkpoint update."""
    cp = Checkpoint()

    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000000000,
        end_time_ms=1700000300000,
        num_records=100,
        file_path="data/raw/binance/klines_5m/BTCUSDT/2024/2024-01.parquet",
    )

    assert len(cp.entries) == 1
    key = "binance/klines_5m/BTCUSDT"
    assert key in cp.entries

    entry = cp.entries[key]
    assert entry.symbol == "BTCUSDT"
    assert entry.venue == "binance"
    assert entry.kind == "klines_5m"
    assert entry.num_records == 100
    assert entry.num_files == 1


def test_checkpoint_extend_existing() -> None:
    """Test extending existing checkpoint."""
    cp = Checkpoint()

    # First batch
    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000000000,
        end_time_ms=1700000300000,
        num_records=100,
        file_path="file1.parquet",
    )

    # Second batch (extends time range)
    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000300001,
        end_time_ms=1700000600000,
        num_records=100,
        file_path="file2.parquet",
    )

    key = "binance/klines_5m/BTCUSDT"
    entry = cp.entries[key]

    assert entry.num_records == 200
    assert entry.num_files == 2
    assert entry.start_time_ms == 1700000000000  # Unchanged
    assert entry.end_time_ms == 1700000600000  # Extended


def test_checkpoint_get_next_start_time() -> None:
    """Test next start time calculation."""
    cp = Checkpoint()

    # No checkpoint → use requested start
    start = cp.get_next_start_time_ms("BTCUSDT", "binance", "klines_5m", 1700000000000)
    assert start == 1700000000000

    # No checkpoint, no requested → start from epoch
    start = cp.get_next_start_time_ms("BTCUSDT", "binance", "klines_5m", None)
    assert start == 0

    # Add checkpoint
    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000000000,
        end_time_ms=1700000300000,
        num_records=100,
        file_path="file.parquet",
    )

    # Checkpoint exists, no requested → continue from checkpoint
    start = cp.get_next_start_time_ms("BTCUSDT", "binance", "klines_5m", None)
    assert start == 1700000300001  # checkpoint end + 1

    # Checkpoint exists, requested provided → use requested
    start = cp.get_next_start_time_ms("BTCUSDT", "binance", "klines_5m", 1800000000000)
    assert start == 1800000000000


def test_checkpoint_has_data() -> None:
    """Test has_data check."""
    cp = Checkpoint()

    # No data
    assert not cp.has_data("BTCUSDT", "binance", "klines_5m")

    # Add data
    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000000000,
        end_time_ms=1700100000000,
        num_records=1000,
        file_path="file.parquet",
    )

    # Has data (no range specified)
    assert cp.has_data("BTCUSDT", "binance", "klines_5m")

    # Range fully covered
    assert cp.has_data(
        "BTCUSDT",
        "binance",
        "klines_5m",
        start_ms=1700010000000,
        end_ms=1700090000000,
    )

    # Range not covered (start before)
    assert not cp.has_data(
        "BTCUSDT",
        "binance",
        "klines_5m",
        start_ms=1699990000000,
        end_ms=1700050000000,
    )

    # Range not covered (end after)
    assert not cp.has_data(
        "BTCUSDT",
        "binance",
        "klines_5m",
        start_ms=1700050000000,
        end_ms=1700110000000,
    )


def test_checkpoint_save_load(tmp_path: Path) -> None:
    """Test checkpoint save/load."""
    cp = Checkpoint()

    cp.update(
        symbol="BTCUSDT",
        venue="binance",
        kind="klines_5m",
        start_time_ms=1700000000000,
        end_time_ms=1700000300000,
        num_records=100,
        file_path="file.parquet",
    )

    # Save
    path = tmp_path / "checkpoint.json"
    cp.save(path)

    assert path.exists()

    # Load
    cp2 = Checkpoint.load(path)
    assert len(cp2.entries) == 1
    assert "binance/klines_5m/BTCUSDT" in cp2.entries

    # Verify content
    entry = cp2.entries["binance/klines_5m/BTCUSDT"]
    assert entry.num_records == 100


def test_checkpoint_load_nonexistent(tmp_path: Path) -> None:
    """Test loading non-existent checkpoint returns empty."""
    path = tmp_path / "nonexistent.json"
    cp = Checkpoint.load(path)

    assert len(cp.entries) == 0
