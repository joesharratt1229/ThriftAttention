#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./cuda-13-0-profile.sh

PYTHON="${PYTHON:-python3}"

"$CUDA_HOME/bin/ncu" \
    --set full \
    --kernel-name-base function \
    --kernel-name nvfp4_sm100_attention_kernel \
    --launch-count 1 \
    --force-overwrite \
    --export fp4_attention_sm100 \
    "$PYTHON" run_fp4_attention.py "$@"
