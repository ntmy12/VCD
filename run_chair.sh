#!/bin/bash
# =============================================================================
# CHAIR Benchmark for Qwen2-VL with VCD (16-bit, max_new_tokens=128)
# Root wrapper script
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/vcd_experiments"
bash run_chair.sh "$@"
