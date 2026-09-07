#!/usr/bin/env python3
"""Profile attention kernels and the full attention pipeline against FP16 SDPA.

Defaults: lengths 4096..131072, both exp modes for FP4 and Thrift at
5/10/25%, and one FP16 SDPA baseline.
Kernel timings exclude quantization/selection; end-to-end timings call the
public attention API and include both. This profiles attention, not a model.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from compare_exp_approx import (
    TensorFn, cosine, get_extension, get_quant_format, make_qkv, make_sdpa_fn,
    measure_kernels, parse_dtype, parse_int_list, torch, validate_args,
)
from thriftattention import AttentionConfig, attention
from thriftattention.selection.block_mean import select_block_pairs

DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


@dataclass(frozen=True)
class Case:
    name: str
    fraction: float
    top_k: int
    actual_pct: float
    exp_approx: bool | None
    kernel: TensorFn
    e2e: TensorFn

    @property
    def exp_mode(self) -> str:
        if self.exp_approx is None:
            return "-"
        return "exp_approx" if self.exp_approx else "exp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("seq_lens", nargs="*", help="Lengths separated by spaces or commas; default: 4096 through 131072.")
    parser.add_argument("--fractions", nargs="+", type=float, default=[.05, .10, .25],
                        help="Thrift FP16 budget fractions of eligible block pairs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16",
                        help="Input/high-precision Thrift dtype; SDPA reference is always FP16.")
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, help="CSV path; default: a timestamped file in benchmarks/results.")
    parser.set_defaults(q_len=None)
    args = parser.parse_args()
    args.seq_lens = parse_int_list(args.seq_lens) if args.seq_lens else DEFAULT_LENGTHS.copy()
    args.torch_dtype = parse_dtype(args.dtype)
    if min(args.batch_size, args.heads, args.kv_heads) < 1:
        parser.error("batch size and head counts must be positive")
    if any(not 0 <= fraction <= 1 for fraction in args.fractions):
        parser.error("--fractions must be in [0, 1]")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("--fractions must not contain duplicates")
    if max(args.seq_lens) > 131072:
        parser.error("Thrift supports at most 131072 tokens (2048 blocks)")
    validate_args(args)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        args.output = Path(__file__).resolve().parent / "results" / f"e2e_profiling_{stamp}.csv"
    return args


def make_cases(args, q, k, v) -> list[Case]:
    is_bf16 = q.dtype == torch.bfloat16
    ext = get_extension()
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
    sdpa = make_sdpa_fn(args, q, k, v)
    fp4_fn = ext.fp4_attention_causal_nvfp4_packed if args.causal else ext.fp4_attention_noncausal_nvfp4_packed
    thrift_fn = ext.thrift_attention_causal_nvfp4_packed if args.causal else ext.thrift_attention_noncausal_nvfp4_packed
    common = dict(causal=args.causal, implementation="tiled", quant_format="nvfp4")
    cases = [Case("fp16", 1.0, q.size(2) // 64, 100.0, None, sdpa, sdpa)]
    for exp_approx in (False, True):
        config = AttentionConfig(method="fp4", exp_approx=exp_approx, **common)
        cases.append(Case(
            "fp4", 0.0, 0, 0.0, exp_approx,
            lambda exp_approx=exp_approx: fp4_fn(*packed, is_bf16, exp_approx),
            lambda config=config: attention(q, k, v, config=config),
        ))
    blocks = q.size(2) // 64
    eligible_pairs = blocks * (blocks + 1) // 2 if args.causal else blocks * blocks
    for fraction in args.fractions:
        # Both modes share these selected blocks and the same packed QKV.
        selected = select_block_pairs(q, k, causal=args.causal, fraction=fraction, is_bf16=is_bf16)
        actual_pct = 100 * (selected >= 0).sum().item() / (q.size(0) * q.size(1) * eligible_pairs)
        for exp_approx in (False, True):
            config = AttentionConfig(method="thrift", fraction=fraction, exp_approx=exp_approx, **common)
            # Bind each budget and mode to avoid late-bound loop closures.
            def kernel(selected=selected, exp_approx=exp_approx):
                return thrift_fn(q, k, v, selected, *packed, is_bf16, exp_approx)

            def e2e(config=config):
                return attention(q, k, v, config=config)

            cases.append(Case(f"thrift_{100 * fraction:g}%", fraction, selected.size(-1),
                              actual_pct, exp_approx, kernel, e2e))
    return cases


def output_metrics(case: Case, reference: torch.Tensor) -> tuple[float, int, int]:
    output = case.e2e().float()
    nan = int(torch.isnan(output).sum().item())
    inf = int(torch.isinf(output).sum().item())
    return cosine(output, reference), nan, inf


def print_header() -> None:
    print("Median CUDA-event timings. Speedups = FP16 SDPA time / method time for each timing scope.")
    print("kernel_ms: packed attention calls (Thrift includes mask setup and both precision passes).")
    print("e2e_ms: public attention API, including QKV quantization and Thrift block selection.")
    print("SDPA uses FP16 inputs prepared before timing; data generation and cosine checks are untimed.")
    print("fp16_pct is the actual selected block-pair budget after rounding to whole blocks.")
    columns = (("seq", 6), ("method", 12), ("exp_mode", 10), ("top_k", 5), ("fp16_pct", 8),
               ("kernel_ms", 10), ("kernel_speedup", 14), ("e2e_ms", 10),
               ("e2e_speedup", 11), ("cos_fp16", 10))
    print("  ".join(f"{name:>{width}}" for name, width in columns))
    print("  ".join("-" * width for _, width in columns))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must be a CUDA device")
    if device.index is not None:
        torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seq_len", "method", "exp_mode", "target_fraction", "actual_fp16_pct", "top_k",
              "kernel_ms", "kernel_speedup", "e2e_ms", "e2e_speedup", "cos_fp16", "nan", "inf",
              "batch_size", "q_heads", "kv_heads", "head_dim", "dtype", "causal", "exp_approx",
              "input_scale", "warmup", "repeat", "seed", "gpu"]
    # Flush each row, preserving completed lengths if a later run is interrupted.
    with args.output.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        output_file.flush()
        print(f"GPU: {gpu}; dtype={args.dtype}; causal={args.causal}; exp modes=exp, exp_approx")
        print(f"CSV: {args.output.resolve()}", flush=True)
        print_header()
        for seq_len in args.seq_lens:
            q, k, v = make_qkv(args, seq_len)
            cases = make_cases(args, q, k, v)
            reference = cases[0].e2e().float()
            metrics = [output_metrics(case, reference) for case in cases]
            del reference
            kernel_stats = measure_kernels(*(case.kernel for case in cases), warmup=args.warmup, repeat=args.repeat)
            e2e_stats = measure_kernels(*(case.e2e for case in cases), warmup=args.warmup, repeat=args.repeat)
            for case, kernel, e2e, (similarity, nan, inf) in zip(cases, kernel_stats, e2e_stats, metrics):
                kernel_speedup = kernel_stats[0].median_ms / kernel.median_ms
                e2e_speedup = e2e_stats[0].median_ms / e2e.median_ms
                writer.writerow(dict(
                    seq_len=seq_len, method=case.name, exp_mode=case.exp_mode, target_fraction=case.fraction,
                    actual_fp16_pct=case.actual_pct, top_k=case.top_k,
                    kernel_ms=kernel.median_ms, kernel_speedup=kernel_speedup,
                    e2e_ms=e2e.median_ms, e2e_speedup=e2e_speedup, cos_fp16=similarity, nan=nan, inf=inf,
                    batch_size=args.batch_size, q_heads=args.heads, kv_heads=args.kv_heads,
                    head_dim=args.head_dim, dtype=args.dtype, causal=args.causal, exp_approx=case.exp_approx,
                    input_scale=args.input_scale, warmup=args.warmup, repeat=args.repeat, seed=args.seed, gpu=gpu,
                ))
                output_file.flush()
                if nan or inf:
                    print(f"nonfinite: seq={seq_len} method={case.name} exp_mode={case.exp_mode} nan={nan} inf={inf}")
                print(f"{seq_len:>6}  {case.name:>12}  {case.exp_mode:>10}  {case.top_k:>5}  {case.actual_pct:>8.3f}  "
                      f"{kernel.median_ms:>10.3f}  {kernel_speedup:>13.3f}x  "
                      f"{e2e.median_ms:>10.3f}  {e2e_speedup:>10.3f}x  {similarity:>10.5f}", flush=True)
            del case, cases, q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
