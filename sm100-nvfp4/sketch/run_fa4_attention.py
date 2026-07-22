#!/usr/bin/env python3
"""Single forward pass of the FlashAttention-4 baseline (flash_attn.cute).

Mirrors run_fp4_attention.py, but there is no extension to build: FA4 is pure
Python over the CuTe DSL.  Inputs are created in (B, H, S, D) like the NVFP4
kernel expects and transposed to the (B, S, H, D) layout flash_attn_func wants,
so the printed stats are directly comparable with run_fp4_attention.py.

Needs `pip install nvidia-cutlass-dsl` and a flash-attn install with
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE (see modal_bench.py's image notes).
"""
from __future__ import annotations

import argparse
import math

import torch


def load_fa4():
    try:
        # See bench_attention.py: the skip-CUDA-build install has no compiled
        # FA2 extension, so stub it before touching the flash_attn package.
        import sys
        import types

        sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
        from flash_attn.cute import flash_attn_func
    except Exception as exc:  # noqa: BLE001 - surface any import failure as one message
        raise RuntimeError(
            f"flash_attn.cute unavailable ({exc!r}); install nvidia-cutlass-dsl and "
            "FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation flash-attn"
        ) from exc
    return flash_attn_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--q-len", type=int, default=4096)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    head_dim = 128
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.q_heads, args.q_len, head_dim, device=device, dtype=dtype).contiguous()
    k = torch.randn(args.batch, args.kv_heads, args.kv_len, head_dim, device=device, dtype=dtype).contiguous()
    v = torch.randn(args.batch, args.kv_heads, args.kv_len, head_dim, device=device, dtype=dtype).contiguous()

    flash_attn_func = load_fa4()
    # FA4 takes (B, S, H, D).
    result = flash_attn_func(q.transpose(1, 2).contiguous(),
                             k.transpose(1, 2).contiguous(),
                             v.transpose(1, 2).contiguous(),
                             softmax_scale=scale, causal=args.causal)
    torch.cuda.synchronize()

    # Newer flash_attn.cute returns (out, lse); older returns just out.
    out = result[0] if isinstance(result, (tuple, list)) else result
    out = out.transpose(1, 2)  # back to (B, H, S, D) to match the NVFP4 kernel
    print(f"out shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print(f"out finite={torch.isfinite(out.float()).all().item()} mean={out.float().mean().item():.6f} std={out.float().std().item():.6f}")


if __name__ == "__main__":
    main()
