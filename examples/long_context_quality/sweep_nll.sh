#!/usr/bin/env bash
# ,quest_drop,block_mean_drop
set -euo pipefail

# Tensor parallel across every visible GPU. 131072 does not fit on one 96 GiB card:
# a 36B model is 67 GiB of bf16 weights and the MLP alone peaks at 21.5 GiB of
# activations at that length. Sharding heads + the MLP intermediate halves both.
GPUS=${GPUS:-$(nvidia-smi --list-gpus | wc -l)}
export NCCL_P2P_DISABLE=1

for length in 4096 8192 16384 32768 65536 131072; do
    /venv/main/bin/torchrun --nproc_per_node="$GPUS" examples/long_context_quality/run_nll_per_token.py \
        --length "$length" --methods fp16,fp4,block_mean --budgets 0.05,0.10,0.25 --num-docs 100
done

