#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./cuda-13-0-profile.sh

PYTHON="${PYTHON:-python3}"

# FA4's forward kernel is JIT-emitted by the CuTe DSL, so its symbol carries the
# Python class plus a hash of the argument layouts:
#   kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor...
# Match on the stable class-name substring rather than the full mangled name.
"$CUDA_HOME/bin/ncu" \
    --set full \
    --kernel-name-base function \
    --kernel-name regex:FlashAttentionForwardSm100 \
    --launch-count 1 \
    --force-overwrite \
    --export fa4_attention_sm100 \
    "$PYTHON" run_fa4_attention.py "$@"
