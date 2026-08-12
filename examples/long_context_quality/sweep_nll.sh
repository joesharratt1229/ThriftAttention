#!/usr/bin/env bash
# ,quest_drop,block_mean_drop

for length in 16384 32768 65536; do
    python3 examples/long_context_quality/run_nll_per_token.py \
        --length "$length" --methods fp16,fp4,block_mean --budgets 0.05,0.10,0.25 --num-docs 100
done
