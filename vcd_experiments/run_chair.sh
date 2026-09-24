#!/bin/bash
# =============================================================================
# CHAIR Benchmark for Qwen2-VL with VCD (16-bit, max_new_tokens=128)
# Target Environment: NVIDIA RTX 6000 (Ada Generation / RTX A6000 / Turing)
#
# Usage:
#   bash run_chair.sh                         # Run both Baseline and VCD (batch_size=4)
#   BATCH_SIZE=8 bash run_chair.sh            # Run with batch_size=8
#   bash run_chair.sh baseline                # Run Baseline only
#   bash run_chair.sh vcd                     # Run VCD only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[CHAIR] Working directory: $(pwd)"

MODE="${1:-all}"
SEED=2027
MAX_TOKENS=128
BATCH_SIZE="${BATCH_SIZE:-4}"  # Accelerated batch generation on RTX 6000 (e.g. 4 or 8)
DTYPE="bf16"                   # 16-bit precision (bf16 for RTX 6000 Ada / A6000; fp16 on Turing)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Shift if first argument was mode, so remaining flags can be passed to python
if [ "$#" -gt 0 ] && { [ "$1" = "all" ] || [ "$1" = "baseline" ] || [ "$1" = "vcd" ]; }; then
    shift
fi

if [ "$MODE" = "baseline" ] || [ "$MODE" = "all" ]; then
    echo "================================================================="
    echo " CHAIR | Model: Qwen2-VL-7B | Baseline (no VCD) | Seed: $SEED"
    echo " Precision: 16-bit ($DTYPE) | Batch Size: $BATCH_SIZE | Max Tokens: $MAX_TOKENS"
    echo "================================================================="
    python benchmarks/chair/run_chair.py \
        --model qwen2vl \
        --no_vcd \
        --seed "$SEED" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_TOKENS" \
        --dtype "$DTYPE" \
        "$@"
fi

if [ "$MODE" = "vcd" ] || [ "$MODE" = "all" ]; then
    echo "================================================================="
    echo " CHAIR | Model: Qwen2-VL-7B | With VCD | Seed: $SEED"
    echo " Precision: 16-bit ($DTYPE) | Batch Size: $BATCH_SIZE | Max Tokens: $MAX_TOKENS"
    echo "================================================================="
    python benchmarks/chair/run_chair.py \
        --model qwen2vl \
        --use_vcd \
        --seed "$SEED" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_TOKENS" \
        --dtype "$DTYPE" \
        "$@"
fi

echo ""
echo "===== CHAIR benchmark completed successfully! ====="
echo "Results and metrics saved in results/ directory."
