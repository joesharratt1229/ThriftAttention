#!/usr/bin/env bash
# sweep_nll.sh plus the error probe. probe goes last: its summary reads the fp16/fp4/
# block_mean rows.
set -euo pipefail

GPUS=${GPUS:-$(nvidia-smi --list-gpus | wc -l)}
export NCCL_P2P_DISABLE=1

for length in 4096 8192 16384 32768 65536 131072; do
    /venv/main/bin/torchrun --nproc_per_node="$GPUS" examples/long_context_quality/run_nll_per_token.py \
        --length "$length" --methods fp16,fp4,block_mean,probe --budgets 0.05,0.10,0.25 --num-docs 100
done
