"""Purged k-fold cross-validation — Lopez de Prado (2018) ch. 7.

Prevents information leakage caused by overlapping labels.
"""

from __future__ import annotations

import numpy as np


def purged_kfold(
    timestamps_ms: np.ndarray,
    label_first_touch_ms: np.ndarray,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold cross-validation.

    Splits the time-ordered sample set into ``n_splits`` consecutive folds and
    returns ``(train_idx, test_idx)`` pairs.  For each fold:

    1. **Purge**: any train sample whose ``label_first_touch_ms`` falls *inside*
       the test-fold time window is removed from the training set (its label
       overlaps with the test period → data leakage).
    2. **Embargo**: an additional gap of ``embargo_pct * total_span`` is applied
       *after* the test fold's end timestamp.  Any train sample that starts
       inside this embargo window is also excluded.

    Args:
        timestamps_ms:        Shape (N,).  Bar open timestamps in ms UTC,
                              **sorted ascending**.
        label_first_touch_ms: Shape (N,).  Timestamp of first barrier touch
                              (or horizon end) for each sample.
        n_splits:             Number of folds.
        embargo_pct:          Fraction of the total time span used as embargo
                              gap after each test fold.

    Returns:
        List of ``n_splits`` ``(train_idx, val_idx)`` tuples; each element is
        an ``int64`` numpy array of indices into the original array.
    """
    n = len(timestamps_ms)
    if n != len(label_first_touch_ms):
        raise ValueError(
            "timestamps_ms and label_first_touch_ms must have the same length"
        )

    t_start = timestamps_ms[0]
    t_end = timestamps_ms[-1]
    total_span = float(t_end - t_start)
    embargo_ms = int(total_span * embargo_pct)

    # Divide indices into n_splits consecutive folds
    fold_indices = np.array_split(np.arange(n), n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_i, test_raw in enumerate(fold_indices):
        if len(test_raw) == 0:
            continue

        test_t_start = timestamps_ms[test_raw[0]]
        test_t_end = timestamps_ms[test_raw[-1]]
        embargo_end = test_t_end + embargo_ms

        # Candidate train indices: everything NOT in the test fold
        all_idx = np.arange(n)
        # Boolean mask: True if index is in test fold
        in_test = np.zeros(n, dtype=bool)
        in_test[test_raw] = True

        # Start with all non-test indices
        train_mask = ~in_test

        # Purge: remove train samples whose label overlaps the test window
        # A train sample at index j is purged if:
        #   label_first_touch_ms[j] >= test_t_start   (its label reaches into test period)
        #   AND timestamps_ms[j] < test_t_start         (it starts before test)
        purge_mask = (
            (~in_test)
            & (label_first_touch_ms >= test_t_start)
            & (timestamps_ms < test_t_start)
        )
        train_mask = train_mask & ~purge_mask

        # Embargo: remove train samples that start inside embargo window
        # (samples that come just after the test fold and might be contaminated)
        embargo_mask = (
            (~in_test)
            & (timestamps_ms > test_t_end)
            & (timestamps_ms <= embargo_end)
        )
        train_mask = train_mask & ~embargo_mask

        train_idx = all_idx[train_mask]
        test_idx = test_raw.astype(np.int64)
        train_idx = train_idx.astype(np.int64)

        splits.append((train_idx, test_idx))

    return splits
