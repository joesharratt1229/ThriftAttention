#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

import torch

from thriftattention._extension import get_extension
from thriftattention.quant.formats import get_quant_format


TensorFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    mean_abs: float
    max_rel: float
    cosine: float
    exact_nan: int
    exact_inf: int
    approx_nan: int
    approx_inf: int


def parse_int_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(int(part))
    if not items:
        raise argparse.ArgumentTypeError("at least one sequence length is required")
    return items


def parse_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError("dtype must be fp16 or bf16")


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(times_ms: list[float]) -> TimingStats:
    return TimingStats(
        mean_ms=statistics.fmean(times_ms),
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p90_ms=percentile(times_ms, 0.90),
    )


def make_qkv(args: argparse.Namespace, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len = seq_len if args.q_len is None else args.q_len
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed + seq_len + q_len)

    q_shape = (args.batch_size, args.heads, q_len, args.head_dim)
    kv_shape = (args.batch_size, args.kv_heads, seq_len, args.head_dim)
    q = torch.randn(q_shape, device=args.device, dtype=args.torch_dtype, generator=generator)
    k = torch.randn(kv_shape, device=args.device, dtype=args.torch_dtype, generator=generator)
    v = torch.randn(kv_shape, device=args.device, dtype=args.torch_dtype, generator=generator)

    if args.input_scale != 1.0:
        q = q * args.input_scale
        k = k * args.input_scale
        v = v * args.input_scale

    return q.contiguous(), k.contiguous(), v.contiguous()


def make_kernel_fns(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[TensorFn, TensorFn]:
    is_bf16 = q.dtype == torch.bfloat16
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
    ext = get_extension()
    fn = (
        ext.fp4_attention_causal_nvfp4_packed
        if args.causal
        else ext.fp4_attention_noncausal_nvfp4_packed
    )

    def run_exp() -> torch.Tensor:
        return fn(*packed, is_bf16, False)

    def run_exp_approx() -> torch.Tensor:
        return fn(*packed, is_bf16, True)

    return run_exp, run_exp_approx


def measure_one(fn: TensorFn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def measure_pair(
    run_exp: TensorFn,
    run_exp_approx: TensorFn,
    *,
    warmup: int,
    repeat: int,
) -> tuple[TimingStats, TimingStats]:
    for _ in range(warmup):
        run_exp()
        run_exp_approx()
    torch.cuda.synchronize()

    exp_times: list[float] = []
    approx_times: list[float] = []
    for i in range(repeat):
        if i % 2 == 0:
            exp_times.append(measure_one(run_exp))
            approx_times.append(measure_one(run_exp_approx))
        else:
            approx_times.append(measure_one(run_exp_approx))
            exp_times.append(measure_one(run_exp))

    return summarize(exp_times), summarize(approx_times)


def compare_outputs(run_exp: TensorFn, run_exp_approx: TensorFn) -> ErrorStats:
    out_exp = run_exp()
    out_approx = run_exp_approx()
    torch.cuda.synchronize()

    exact = out_exp.float()
    approx = out_approx.float()
    diff = (exact - approx).abs()
    denom = exact.abs().clamp_min(1.0e-6)
    cosine = torch.nn.functional.cosine_similarity(
        exact.flatten(), approx.flatten(), dim=0
    )
    return ErrorStats(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float((diff / denom).max().item()),
        cosine=float(cosine.item()),
        exact_nan=int(torch.isnan(exact).sum().item()),
        exact_inf=int(torch.isinf(exact).sum().item()),
        approx_nan=int(torch.isnan(approx).sum().item()),
        approx_inf=int(torch.isinf(approx).sum().item()),
    )


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.head_dim not in (64, 128):
        raise SystemExit("--head-dim must be 64 or 128")
    if args.heads % args.kv_heads != 0:
        raise SystemExit("--heads must be divisible by --kv-heads")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    for seq_len in args.seq_lens:
        q_len = seq_len if args.q_len is None else args.q_len
        if seq_len <= 0 or q_len <= 0:
            raise SystemExit("sequence lengths and --q-len must be positive")
        if seq_len % 64 != 0 or q_len % 64 != 0:
            raise SystemExit("this benchmark uses tiled fp4 attention; kv length and q length must be divisible by 64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NVFP4 FP4 attention using __expf against exp_approx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seq_lens", nargs="+", help="KV sequence lengths, as spaces or comma-separated values.")
    parser.add_argument("--q-len", type=int, default=None, help="Query length. Defaults to each KV sequence length.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--skip-error", action="store_true", help="Skip output-difference metrics.")
    args = parser.parse_args()
    args.seq_lens = parse_int_list(args.seq_lens)
    args.torch_dtype = parse_dtype(args.dtype)
    return args


def print_header() -> None:
    print(
        "seq  q_len  exp_ms  exp_approx_ms  speedup  "
        "max_abs  mean_abs  max_rel  cosine"
    )
    print(
        "---  -----  ------  -------------  -------  "
        "-------  --------  -------  ------"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    print_header()
    for seq_len in args.seq_lens:
        q, k, v = make_qkv(args, seq_len)
        run_exp, run_exp_approx = make_kernel_fns(args, q, k, v)
        errors = None if args.skip_error else compare_outputs(run_exp, run_exp_approx)
        exp_stats, approx_stats = measure_pair(
            run_exp,
            run_exp_approx,
            warmup=args.warmup,
            repeat=args.repeat,
        )

        speedup = exp_stats.median_ms / approx_stats.median_ms
        q_len = q.shape[2]
        if errors is None:
            error_cols = "-        -         -        -"
        else:
            error_cols = (
                f"{errors.max_abs:.3e}  {errors.mean_abs:.3e}  "
                f"{errors.max_rel:.3e}  {errors.cosine:.5f}"
            )
            if any(
                count
                for count in (
                    errors.exact_nan,
                    errors.exact_inf,
                    errors.approx_nan,
                    errors.approx_inf,
                )
            ):
                print(
                    "nonfinite: "
                    f"exp nan={errors.exact_nan} inf={errors.exact_inf}; "
                    f"exp_approx nan={errors.approx_nan} inf={errors.approx_inf}"
                )

        print(
            f"{seq_len:<3}  {q_len:<5}  "
            f"{exp_stats.median_ms:>6.3f}  "
            f"{approx_stats.median_ms:>13.3f}  "
            f"{speedup:>7.3f}  "
            f"{error_cols}"
        )

        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
