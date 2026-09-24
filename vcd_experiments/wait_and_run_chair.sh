#!/bin/bash
# =============================================================================
# Script: wait_and_run_chair.sh
# Purpose: Wait until GPU has >= 15GB VRAM free, then run Qwen2-VL CHAIR (16-bit, max_tokens=128)
# Target: NVIDIA RTX 6000
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[Info] Working directory: $(pwd)"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
TARGET_MB=15360      # 15 GB (15 * 1024 MiB)
INTERVAL=10          # Check interval in seconds

echo "================================================================="
echo " Monitoring GPU ${GPU_ID} - Waiting for >= ${TARGET_MB} MiB (15 GB) free VRAM..."
echo "================================================================="

while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' \r\n')
    
    if [ -n "$FREE_MEM" ] && [ "$FREE_MEM" -ge "$TARGET_MB" ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> GPU READY!"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID} currently has: ${FREE_MEM} MiB free (Required: >= ${TARGET_MB} MiB)."
        echo "================================================================="
        echo ">>> Starting Qwen2-VL CHAIR benchmark (16-bit, max_new_tokens=128)..."
        echo "================================================================="
        break
    fi

    echo "[$(date '+%H:%M:%S')] GPU ${GPU_ID}: ${FREE_MEM} MiB free / ${TARGET_MB} MiB. Retrying in ${INTERVAL}s..."
    sleep "$INTERVAL"
done

# Run CHAIR benchmark (both baseline and VCD)
bash run_chair.sh "$@"
