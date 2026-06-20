"""Checkpoint tracking for incremental data downloads.

Tracks:
- What data has been downloaded (date ranges per symbol/venue/kind)
- Last successful timestamp for resumption
- Supports pagination (start from checkpoint, start+N, custom range)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CheckpointEntry(BaseModel):
    """Single checkpoint entry for a data stream."""

    symbol: str
    venue: Literal["binance", "coinbase", "deribit"]
    kind: Literal[
        "klines_1m",
        "klines_5m",
        "klines_15m",
        "klines_1h",
        "trades",
        "depth",
        "funding",
        "open_interest",
        "mark_price",
        "liquidations",
    ]

    # Time range covered
    start_time_ms: int = Field(description="Earliest timestamp covered")
    end_time_ms: int = Field(description="Latest timestamp covered")
    last_updated_ms: int = Field(description="When checkpoint was last updated")

    # Metadata
    num_records: int = Field(default=0, description="Total records in this range")
    num_files: int = Field(default=0, description="Number of parquet files")
    file_paths: list[str] = Field(default_factory=list, description="List of file paths")

    @property
    def start_time(self) -> datetime:
        return datetime.fromtimestamp(self.start_time_ms / 1000, tz=timezone.utc)

    @property
    def end_time(self) -> datetime:
        return datetime.fromtimestamp(self.end_time_ms / 1000, tz=timezone.utc)

    @property
    def last_updated(self) -> datetime:
        return datetime.fromtimestamp(self.last_updated_ms / 1000, tz=timezone.utc)

    @property
    def key(self) -> str:
        """Unique key for this checkpoint."""
        return f"{self.venue}/{self.kind}/{self.symbol}"


class Checkpoint(BaseModel):
    """Checkpoint manager for all data streams."""

    entries: dict[str, CheckpointEntry] = Field(
        default_factory=dict, description="Checkpoints keyed by venue/kind/symbol"
    )

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from disk."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)
            return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def get(
        self, symbol: str, venue: str, kind: str
    ) -> CheckpointEntry | None:
        """Get checkpoint for a specific stream."""
        key = f"{venue}/{kind}/{symbol}"
        return self.entries.get(key)

    def update(
        self,
        symbol: str,
        venue: str,
        kind: str,
        start_time_ms: int,
        end_time_ms: int,
        num_records: int,
        file_path: str,
    ) -> None:
        """Update checkpoint with new data range."""
        key = f"{venue}/{kind}/{symbol}"
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if key in self.entries:
            # Extend existing checkpoint
            entry = self.entries[key]
            entry.start_time_ms = min(entry.start_time_ms, start_time_ms)
            entry.end_time_ms = max(entry.end_time_ms, end_time_ms)
            entry.num_records += num_records
            entry.num_files += 1
            if file_path not in entry.file_paths:
                entry.file_paths.append(file_path)
            entry.last_updated_ms = now_ms
        else:
            # Create new checkpoint
            self.entries[key] = CheckpointEntry(
                symbol=symbol,
                venue=venue,
                kind=kind,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                last_updated_ms=now_ms,
                num_records=num_records,
                num_files=1,
                file_paths=[file_path],
            )

    def get_next_start_time_ms(
        self, symbol: str, venue: str, kind: str, requested_start_ms: int | None = None
    ) -> int:
        """Get next start time for download.

        Logic:
        - If no checkpoint exists: use requested_start_ms
        - If checkpoint exists and requested_start_ms is None: continue from checkpoint end
        - If checkpoint exists and requested_start_ms provided: use requested_start_ms
        """
        entry = self.get(symbol, venue, kind)

        if entry is None:
            # No checkpoint, start from requested or epoch
            return requested_start_ms if requested_start_ms else 0

        if requested_start_ms is not None:
            # User explicitly requested a start time (e.g., start+N or custom range)
            return requested_start_ms

        # Continue from last checkpoint (pagination)
        return entry.end_time_ms + 1

    def has_data(
        self,
        symbol: str,
        venue: str,
        kind: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> bool:
        """Check if we already have data for a time range."""
        entry = self.get(symbol, venue, kind)
        if entry is None:
            return False

        if start_ms is None and end_ms is None:
            return True  # Has some data

        # Check if requested range is covered
        if start_ms is not None and start_ms < entry.start_time_ms:
            return False
        if end_ms is not None and end_ms > entry.end_time_ms:
            return False

        return True

    def summary(self) -> str:
        """Human-readable summary of checkpoints."""
        if not self.entries:
            return "No checkpoints found."

        lines = ["Checkpoint Summary", "=" * 80]
        for key, entry in sorted(self.entries.items()):
            lines.append(
                f"{entry.venue}/{entry.kind}/{entry.symbol}: "
                f"{entry.start_time.isoformat()} → {entry.end_time.isoformat()} "
                f"({entry.num_records:,} records, {entry.num_files} files)"
            )
        return "\n".join(lines)


def get_checkpoint_path(data_dir: Path) -> Path:
    """Get path to checkpoint file."""
    return data_dir / "checkpoints" / "data_checkpoint.json"
