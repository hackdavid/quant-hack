"""Parallel download using ThreadPoolExecutor + hf_hub_download, 2023+ only."""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

REPO_ID    = "ibrahimdaud/binance-btcusdt"
DATA_ROOT  = Path("/home/ubuntu/quant-hack/data")
START      = "2023-01-01"
MAX_WORKERS = 16

print("Listing repo files...")
all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))

# Build work list: (repo_path, local_path)
todo = []
for f in all_files:
    date_str = f.split("/")[-1].replace(".parquet", "")
    if date_str < START:
        continue
    if f.startswith("features/BTCUSDT/"):
        local = DATA_ROOT / f
    elif f.startswith("raw/klines_1m/BTCUSDT/"):
        parts = f.split("/")  # ['raw', 'klines_1m', 'BTCUSDT', 'date.parquet']
        local = DATA_ROOT / "raw" / "binance" / parts[1] / parts[2] / parts[3]
    else:
        continue
    if not local.exists():
        todo.append((f, local))

print(f"Files to download: {len(todo)}  (workers={MAX_WORKERS})")
if not todo:
    print("Nothing to download!")
else:
    def download_one(item):
        repo_file, local_path = item
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            repo_type="dataset",
            local_dir=str(DATA_ROOT),
        )
        # hf_hub_download drops the file at DATA_ROOT/repo_file; move if dest differs
        src = Path(tmp)
        if src != local_path and src.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(local_path))
        return repo_file

    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, item): item for item in todo}
        with tqdm(total=len(todo), unit="file") as bar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    errors.append((futures[fut][0], str(e)))
                bar.update(1)

    if errors:
        print(f"\n{len(errors)} errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")

feat_total = len(list((DATA_ROOT / "features" / "BTCUSDT").glob("*.parquet")))
klin_total = len(list((DATA_ROOT / "raw" / "binance" / "klines_1m" / "BTCUSDT").glob("*.parquet")))
print(f"\nFinal count: features={feat_total}  klines={klin_total}")
