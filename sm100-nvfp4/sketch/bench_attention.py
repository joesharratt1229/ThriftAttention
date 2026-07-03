#!/usr/bin/env python3
"""Kernel-only benchmark: NVFP4 SM100 attention vs FlashAttention-4 (CuTe DSL).

Both kernels run non-causal attention, head_dim=128, bf16 in/out.
NVFP4 quantisation/packing happens once outside the timed region; the timed
call is attention_only(), which launches just nvfp4_sm100_attention_kernel.
"""
from __future__ import annotations

import argparse
import math

import torch

from run_fp4_attention import build_extension
from flash_attn.cute import flash_attn_func

HEAD_DIM = 128


def time_fn(fn, warmup: int, iters: int) -> float:
    """Return average milliseconds per call, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_interleaved(fn_a, fn_b, warmup: int, iters: int, chunks: int = 10) -> tuple[float, float]:
    """Time two kernels in alternating chunks so both see the same average
    thermal/power state (back-to-back blocks bias against whichever runs
    second once the GPU throttles under sustained load)."""
    for _ in range(warmup):
        fn_a()
        fn_b()
    chunk = max(1, iters // chunks)
    total_a = total_b = 0.0
    n = 0
    for _ in range(chunks):
        for fn, acc in ((fn_a, "a"), (fn_b, "b")):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(chunk):
                fn()
            e.record()
            torch.cuda.synchronize()
            if acc == "a":
                total_a += s.elapsed_time(e)
            else:
                total_b += s.elapsed_time(e)
        n += chunk
    return total_a / n, total_b / n


def bench_shape(ext, batch: int, heads: int, q_len: int, kv_len: int,
                warmup: int, iters: int) -> dict:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(HEAD_DIM)

    torch.manual_seed(0)
    q = torch.randn(batch, heads, q_len, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(batch, heads, kv_len, HEAD_DIM, device=device, dtype=dtype)

    # Quantise once (untimed), keep fp4 payloads + scale atoms for the timed loop.
    pre = ext.quantise_and_attention(q, k, v, scale)
    args = (pre["q_fp4"], pre["k_fp4"], pre["v_t_fp4"],
            pre["q_sf_atoms"], pre["k_sf_atoms"], pre["v_sf_atoms"])

    # FA4 wants (B, S, H, D).
    q4 = q.transpose(1, 2).contiguous()
    k4 = k.transpose(1, 2).contiguous()
    v4 = v.transpose(1, 2).contiguous()

    nvfp4_ms, fa4_ms = time_interleaved(
        lambda: ext.attention_only(*args, scale),
        lambda: flash_attn_func(q4, k4, v4, softmax_scale=scale, causal=False),
        warmup, iters)

    # Output agreement (fp4 quantisation bounds the achievable match).
    out_nvfp4 = ext.attention_only(*args, scale)
    out_fa4 = flash_attn_func(q4, k4, v4, softmax_scale=scale, causal=False)[0]
    out_fa4 = out_fa4.transpose(1, 2)
    cos = torch.nn.functional.cosine_similarity(
        out_nvfp4.float().flatten(), out_fa4.float().flatten(), dim=0).item()

    flops = 4.0 * batch * heads * q_len * kv_len * HEAD_DIM
    return {
        "nvfp4_ms": nvfp4_ms,
        "fa4_ms": fa4_ms,
        "nvfp4_tflops": flops / (nvfp4_ms * 1e-3) / 1e12,
        "fa4_tflops": flops / (fa4_ms * 1e-3) / 1e12,
        "cos": cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seqlens", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    ext = build_extension(verbose=False)
    print(f"GPU: {torch.cuda.get_device_name()}  batch={args.batch} heads={args.heads} "
          f"head_dim={HEAD_DIM} non-causal, q_len=kv_len")
    header = (f"{'seqlen':>7} | {'nvfp4 ms':>9} {'nvfp4 TF/s':>10} | "
              f"{'fa4 ms':>9} {'fa4 TF/s':>10} | {'speedup':>7} {'cos':>6}")
    print(header)
    print("-" * len(header))
    for s in args.seqlens:
        r = bench_shape(ext, args.batch, args.heads, s, s, args.warmup, args.iters)
        print(f"{s:>7} | {r['nvfp4_ms']:>9.3f} {r['nvfp4_tflops']:>10.1f} | "
              f"{r['fa4_ms']:>9.3f} {r['fa4_tflops']:>10.1f} | "
              f"{r['fa4_ms'] / r['nvfp4_ms']:>6.2f}x {r['cos']:>6.3f}")


if __name__ == "__main__":
    main()
