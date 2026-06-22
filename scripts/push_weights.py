#!/usr/bin/env python3
"""Push forecast model weights to a Hugging Face dataset repo.

Usage:
    HF_TOKEN=hf_xxx uv run python scripts/push_weights.py
    uv run python scripts/push_weights.py --token hf_xxx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import HfHubHTTPError

REPO_ID     = "ibrahimdaud/binance-btcusdt"
REPO_TYPE   = "dataset"
RUN_DIR     = Path("models/forecast/20260621T212013Z")
DEST_PREFIX = "models/forecast/run_v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    token = args.token.strip()
    if not token:
        sys.exit(
            "No token found.\n"
            "Set it with:  export HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "Then re-run:  uv run python scripts/push_weights.py"
        )

    if not RUN_DIR.exists():
        sys.exit(f"Checkpoint directory not found: {RUN_DIR}")

    files = list(RUN_DIR.iterdir())
    total_mb = sum(f.stat().st_size for f in files) / 1024**2
    print(f"Uploading {len(files)} files  ({total_mb:.1f} MB)")
    print(f"  from : {RUN_DIR}")
    print(f"  to   : {REPO_ID}/{DEST_PREFIX}\n")

    api = HfApi(token=token)

    try:
        # Verify repo exists and we have write access
        api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    except HfHubHTTPError as e:
        sys.exit(f"Cannot access repo {REPO_ID}: {e}")

    upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(RUN_DIR),
        path_in_repo=DEST_PREFIX,
        token=token,
        commit_message="Add Kronos LoRA forecast weights (run_v3, 7 epochs, binary labels, seq=512, rank=32)",
    )

    print(f"\nDone — https://huggingface.co/datasets/{REPO_ID}/tree/main/{DEST_PREFIX}")


if __name__ == "__main__":
    main()
