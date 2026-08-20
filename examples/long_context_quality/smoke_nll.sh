#!/usr/bin/env bash
# Does every attention path still run? 4096 tokens, 2 docs -- wiring only, the NLLs are
# on far too little text to compare methods.
set -euo pipefail

GPUS=${GPUS:-$(nvidia-smi --list-gpus | wc -l)}
export NCCL_P2P_DISABLE=1

/venv/main/bin/torchrun --nproc_per_node="$GPUS" examples/long_context_quality/run_nll_per_token.py \
    --length 16384 --methods fp16,fp4,block_mean,probe \
    --budgets 0.05,0.10,0.25 --num-docs 20 --output results/nll_smoke
