#!/usr/bin/env python3
"""Compare tiled NVFP4 Thrift exp paths at a fixed FP16 selection budget."""
from __future__ import annotations

import argparse

from compare_exp_approx import (
    cosine, get_extension, get_quant_format, make_qkv, make_sdpa_fn,
    measure_kernels, parse_dtype, parse_int_list, torch, validate_args,
)
from thriftattention.selection.block_mean import select_block_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Thrift exp and exp_approx with identical selected FP16 blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seq_lens", nargs="+", help="KV lengths, separated by spaces or commas.")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--fraction", "--budget", type=float, default=None,
                        help="Target fraction of eligible block pairs computed in FP16 (default: 0.05).")
    budget.add_argument("--top-k", type=int, help="Fixed number of FP16 KV blocks per query block.")
    parser.add_argument("--q-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--skip-error", action="store_true", help="Skip cosine and nonfinite checks.")
    args = parser.parse_args()
    if args.fraction is None and args.top_k is None:
        args.fraction = 0.05
    if args.fraction is not None and not 0 <= args.fraction <= 1:
        parser.error("--fraction must be in [0, 1]")
    if args.top_k is not None and args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if min(args.batch_size, args.heads, args.kv_heads) < 1:
        parser.error("batch size and head counts must be positive")
    args.seq_lens = parse_int_list(args.seq_lens)
    args.torch_dtype = parse_dtype(args.dtype)
    validate_args(args)
    if max(args.seq_lens) > 131072:
        parser.error("Thrift supports at most 2048 KV blocks (131072 tokens)")
    if args.causal and args.q_len is not None and args.fraction is not None:
        if any(args.q_len != n for n in args.seq_lens):
            parser.error("causal --fraction assumes equal Q/KV lengths; use --top-k for rectangular attention")
    return args


def make_thrift_fns(args, q, k, v):
    is_bf16 = q.dtype == torch.bfloat16
    # Select once and share the exact tensor between the two implementations.
    selected = select_block_pairs(q, k, causal=args.causal, top_k=args.top_k,
                                  fraction=args.fraction, is_bf16=is_bf16)
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
    ext = get_extension()
    fn = ext.thrift_attention_causal_nvfp4_packed if args.causal else ext.thrift_attention_noncausal_nvfp4_packed

    def run_exp():
        return fn(q, k, v, selected, *packed, is_bf16, False)

    def run_approx():
        return fn(q, k, v, selected, *packed, is_bf16, True)

    q_blocks, kv_blocks = q.size(2) // 64, k.size(2) // 64
    eligible = sum(min(i + 1, kv_blocks) for i in range(q_blocks)) if args.causal else q_blocks * kv_blocks
    actual_pct = 100 * (selected >= 0).sum().item() / (q.size(0) * q.size(1) * eligible)
    return run_exp, run_approx, selected.size(-1), actual_pct


def compare_cosines(run_sdpa, run_exp, run_approx):
    sdpa, exp, approx = (fn().float() for fn in (run_sdpa, run_exp, run_approx))
    for name, out in (("sdpa_fp16", sdpa), ("exp", exp), ("exp_approx", approx)):
        nan, inf = torch.isnan(out).sum().item(), torch.isinf(out).sum().item()
        if nan or inf:
            print(f"nonfinite: {name} nan={nan} inf={inf}")
    return cosine(exp, approx), cosine(exp, sdpa), cosine(approx, sdpa)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must be a CUDA device")
    if device.index is not None:
        torch.cuda.set_device(device)
    budget = f"top-k={args.top_k}" if args.top_k is not None else f"fraction={args.fraction:g}"
    print(f"Thrift NVFP4, {budget}, block-mean selection shared by both paths.")
    print("exp_approx changes the FP4 pass; selected FP16/BF16 blocks keep ordinary exponentials.")
    print("Median CUDA timings include packed attention, mask setup, and FP16 finalize;")
    print("selection, QKV quantization, and SDPA FP16 conversion are excluded.")
    print("Speedups = SDPA FP16 time / Thrift time; fp16_pct is the actual selected block-pair percentage.")
    columns = (("seq", 6), ("q_len", 6), ("top_k", 5), ("fp16_pct", 8),
               ("sdpa_fp16_ms", 12), ("exp_ms", 9), ("exp_approx_ms", 13),
               ("exp_speedup", 11), ("approx_speedup", 14),
               ("exp_app_cos", 11), ("exp_sdpa_cos", 12), ("approx_sdpa_cos", 15))
    print("  ".join(f"{name:>{width}}" for name, width in columns))
    print("  ".join("-" * width for _, width in columns))
    for seq_len in args.seq_lens:
        q, k, v = make_qkv(args, seq_len)
        run_exp, run_approx, top_k, actual_pct = make_thrift_fns(args, q, k, v)
        run_sdpa = make_sdpa_fn(args, q, k, v)
        if args.skip_error:
            cos_cols = f"{'-':>11}  {'-':>12}  {'-':>15}"
        else:
            exp_app, exp_sdpa, app_sdpa = compare_cosines(run_sdpa, run_exp, run_approx)
            cos_cols = f"{exp_app:>11.5f}  {exp_sdpa:>12.5f}  {app_sdpa:>15.5f}"
        sdpa_stats, exp_stats, approx_stats = measure_kernels(
            run_sdpa, run_exp, run_approx, warmup=args.warmup, repeat=args.repeat)
        sdpa_ms, exp_ms, approx_ms = (s.median_ms for s in (sdpa_stats, exp_stats, approx_stats))
        print(f"{seq_len:>6}  {q.size(2):>6}  {top_k:>5}  {actual_pct:>8.3f}  "
              f"{sdpa_ms:>12.3f}  {exp_ms:>9.3f}  {approx_ms:>13.3f}  "
              f"{sdpa_ms / exp_ms:>10.3f}x  {sdpa_ms / approx_ms:>13.3f}x  {cos_cols}", flush=True)
        del run_sdpa, run_exp, run_approx, q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
