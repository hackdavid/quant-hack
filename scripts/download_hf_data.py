"""Download 2023+ features + klines_1m from HuggingFace dataset in parallel."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "ibrahimdaud/binance-btcusdt"
START_DATE = "2023-01-01"
DATA_ROOT = Path("/home/ubuntu/quant-hack/data")
MAX_WORKERS = 16

def get_file_list():
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    targets = []
    for f in files:
        if f.startswith("features/BTCUSDT/") or f.startswith("raw/klines_1m/BTCUSDT/"):
            date_str = f.split("/")[-1].replace(".parquet", "")
            if date_str >= START_DATE:
                targets.append(f)
    return targets

def download_one(repo_file: str) -> tuple[str, bool]:
    # Map HF path → local path
    if repo_file.startswith("features/"):
        local_path = DATA_ROOT / repo_file  # data/features/BTCUSDT/YYYY-MM-DD.parquet
    else:
        # raw/klines_1m/BTCUSDT/... → data/raw/binance/klines_1m/BTCUSDT/...
        parts = repo_file.split("/")  # ['raw', 'klines_1m', 'BTCUSDT', 'date.parquet']
        local_path = DATA_ROOT / "raw" / "binance" / parts[1] / parts[2] / parts[3]

    if local_path.exists():
        return repo_file, False  # already exists

    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_file,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
    )
    # hf_hub_download puts it at DATA_ROOT/repo_file, move if needed
    hf_path = DATA_ROOT / repo_file
    if hf_path.exists() and hf_path != local_path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_path.rename(local_path)
    return repo_file, True


def main():
    print(f"Listing files from {REPO_ID}...")
    files = get_file_list()
    print(f"Found {len(files)} files from {START_DATE}+ (features + klines_1m)")

    already = sum(1 for f in files if _local_exists(f))
    to_download = [f for f in files if not _local_exists(f)]
    print(f"Already downloaded: {already}  |  To download: {len(to_download)}")

    if not to_download:
        print("All files already present!")
        return

    ok = err = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, f): f for f in to_download}
        with tqdm(total=len(to_download), unit="file") as bar:
            for fut in as_completed(futures):
                try:
                    _, downloaded = fut.result()
                    if downloaded:
                        ok += 1
                except Exception as e:
                    err += 1
                    tqdm.write(f"ERROR: {futures[fut]}: {e}")
                bar.update(1)
                bar.set_postfix(ok=ok, err=err)

    print(f"\nDone. Downloaded {ok} files. Errors: {err}")


def _local_exists(repo_file: str) -> bool:
    if repo_file.startswith("features/"):
        return (DATA_ROOT / repo_file).exists()
    parts = repo_file.split("/")
    return (DATA_ROOT / "raw" / "binance" / parts[1] / parts[2] / parts[3]).exists()


if __name__ == "__main__":
    main()
