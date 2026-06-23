#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/quant-hack

echo "=========================================="
echo "  Waiting for data download to complete..."
echo "=========================================="

while pgrep -f download_fast.py > /dev/null 2>&1; do
    FEAT=$(ls data/features/BTCUSDT/*.parquet 2>/dev/null | wc -l)
    KLIN=$(ls data/raw/binance/klines_1m/BTCUSDT/*.parquet 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] features=$FEAT  klines=$KLIN"
    sleep 15
done

FEAT=$(ls data/features/BTCUSDT/*.parquet 2>/dev/null | wc -l)
KLIN=$(ls data/raw/binance/klines_1m/BTCUSDT/*.parquet 2>/dev/null | wc -l)
echo ""
echo "Download complete: features=$FEAT  klines=$KLIN"

if [ "$FEAT" -lt 100 ] || [ "$KLIN" -lt 100 ]; then
    echo "ERROR: Not enough data files. Aborting."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Starting Forecast Training"
echo "  A100 80GB | 14 CPU | 98GB RAM"
echo "  epochs=10, batch=512, grad_accum=2 -> eff_batch=1024"
echo "  LoRA top-4 Kronos layers"
echo "  Train: 2023-01-01 -> 2026-02-28"
echo "  Val:   2026-03-01 -> 2026-05-31"
echo "=========================================="
echo ""

uv run intraday forecast train \
    --train-start 2023-01-01 \
    --train-end   2025-03-31 \
    --val-start   2025-04-01 \
    --val-end     2025-09-30 \
    --epochs 10 \
    --batch-size 512 \
    --grad-accum 2 \
    --unfreeze-top-k 4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lr-lora 5e-5 \
    --lr-head 2e-4 \
    --warmup-steps 500 \
    --device cuda \
    --data-dir data \
    --symbol BTCUSDT \
    --kronos-checkpoint models/kronos-base \
    --tokenizer-checkpoint models/kronos-tokenizer \
    --log-every 50

echo ""
echo "Training complete!"
