"""LazyFeatureStore — memory-efficient, tick-by-tick access to feature Parquet files.

Reads daily Parquet files one at a time — never loads the full dataset into RAM.

Iteration patterns:

  Bar-by-bar (backtesting / paper trading replay):
      store = LazyFeatureStore(Path("data/features/BTCUSDT"))
      for bar in store.iter_bars():
          signal = model.predict(bar)

  Day-by-day (daily batch processing):
      for day_df in store.iter_days():
          # day_df is a Polars DataFrame, 288 rows, sorted by bar_time_ms
          daily_sharpe = backtest_day(day_df)

  Mini-batch (ML training):
      for batch in store.iter_batches(batch_size=512):
          X = batch.select(ALL_FEATURES).to_numpy()
          y = batch["fwd_direction_5m"].to_numpy()

  Sequence batches (LSTM / Transformer training):
      for X_seq, y in store.iter_sequences(seq_len=60, batch_size=32):
          model.train_step(X_seq, y)
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import polars as pl

from intraday.features.schema import ALL_FEATURES, TARGET_COLS


class LazyFeatureStore:
    """Lazy reader over a directory of daily feature Parquet files."""

    def __init__(
        self,
        features_dir: Path,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        require_targets: bool = True,
    ) -> None:
        self._dir = Path(features_dir)
        if not self._dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self._dir}")

        # Discover and filter files by date range
        all_files = sorted(self._dir.glob("*.parquet"))
        self._files: list[Path] = []
        for f in all_files:
            try:
                d = date.fromisoformat(f.stem)
            except ValueError:
                continue
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            self._files.append(f)

        self._require_targets = require_targets

    # ── Properties ────────────────────────────────────────────────────────

    def days(self) -> list[date]:
        return [date.fromisoformat(f.stem) for f in self._files]

    def n_days(self) -> int:
        return len(self._files)

    def total_rows(self) -> int:
        return sum(
            len(pl.scan_parquet(f).select("bar_time_ms").collect())
            for f in self._files
        )

    def feature_columns(self) -> list[str]:
        return ALL_FEATURES

    def target_columns(self) -> list[str]:
        return TARGET_COLS

    # ── Iteration ─────────────────────────────────────────────────────────

    def iter_days(self) -> Iterator[pl.DataFrame]:
        """Yield one Polars DataFrame per calendar day, sorted by bar_time_ms.

        Loads one file at a time — O(1) peak memory regardless of date range.
        """
        for f in self._files:
            df = pl.read_parquet(f).sort("bar_time_ms")
            if self._require_targets:
                df = df.filter(pl.col("fwd_ret_5m").is_not_null())
            if len(df) > 0:
                yield df

    def iter_bars(self) -> Iterator[dict]:
        """Yield one bar dict at a time, day-by-day, sorted by timestamp.

        Use this for tick-by-tick backtesting and paper trading replay.

        Example:
            for bar in store.iter_bars():
                features = {col: bar[col] for col in ALL_FEATURES}
                model.step(bar["bar_time_ms"], features, bar["fwd_direction_5m"])
        """
        for day_df in self.iter_days():
            for row in day_df.iter_rows(named=True):
                yield row

    def iter_batches(self, batch_size: int = 512) -> Iterator[pl.DataFrame]:
        """Yield DataFrames of `batch_size` rows for mini-batch training.

        Rows are ordered chronologically across all days.
        Buffer holds at most 2 × batch_size rows at once.
        """
        buffer: list[dict] = []
        for day_df in self.iter_days():
            buffer.extend(day_df.iter_rows(named=True))
            while len(buffer) >= batch_size:
                yield pl.DataFrame(buffer[:batch_size])
                buffer = buffer[batch_size:]
        if buffer:
            yield pl.DataFrame(buffer)

    def iter_sequences(
        self,
        seq_len: int = 60,
        step: int = 1,
        batch_size: int = 32,
    ) -> Iterator[tuple[list[list[dict]], list[Optional[int]]]]:
        """Yield (X_sequences, y_labels) pairs for sequence model training.

        X_sequences: list of `batch_size` sequences, each `seq_len` bar dicts
        y_labels:    list of `batch_size` fwd_direction_5m values

        Memory: maintains a rolling window of seq_len + batch_size rows max.

        Example (PyTorch):
            for X_seqs, y_labels in store.iter_sequences(seq_len=60, batch_size=32):
                X = torch.tensor([[
                    [bar[col] or 0.0 for col in ALL_FEATURES]
                    for bar in seq
                ] for seq in X_seqs])
                y = torch.tensor(y_labels)
                loss = model(X, y)
        """
        window: list[dict] = []
        needed = seq_len + batch_size * step

        for day_df in self.iter_days():
            window.extend(day_df.iter_rows(named=True))

            while len(window) >= needed:
                X_batch: list[list[dict]] = []
                y_batch: list[Optional[int]] = []

                for b in range(batch_size):
                    start = b * step
                    end   = start + seq_len
                    if end >= len(window):
                        break
                    X_batch.append(window[start:end])
                    y_batch.append(window[end].get("fwd_direction_5m"))

                if X_batch:
                    yield X_batch, y_batch

                window = window[batch_size * step:]

        # Final partial batch
        while len(window) >= seq_len + 1:
            X_batch, y_batch = [], []
            for b in range(min(batch_size, len(window) - seq_len)):
                start = b * step
                end   = start + seq_len
                if end >= len(window):
                    break
                X_batch.append(window[start:end])
                y_batch.append(window[end].get("fwd_direction_5m"))
            if X_batch:
                yield X_batch, y_batch
            window = window[batch_size * step:]
