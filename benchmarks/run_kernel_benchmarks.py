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

import thriftattention as ta


TensorFn = Callable[[], torch.Tensor]


class MissingDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    coverage: float | None
    top_k: int | None
    fn: TensorFn


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    seq_len: int
    name: str
    coverage: float | None
    top_k: int | None
    status: str
    stats: TimingStats | None = None
    note: str = ""


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


def parse_fraction(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        fraction = float(value[:-1]) / 100.0
    else:
        fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise argparse.ArgumentTypeError("coverage values must be in [0, 1], or percentages")
    return fraction


def parse_fraction_list(value: str) -> list[float]:
    fractions = [parse_fraction(part) for part in value.split(",") if part.strip()]
    if not fractions:
        raise argparse.ArgumentTypeError("at least one coverage value is required")
    return fractions


def parse_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError("dtype must be fp16 or bf16")


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
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


def format_coverage(coverage: float | None) -> str:
    if coverage is None:
        return "-"
    return f"{coverage * 100:.2f}%"


def query_len(args: argparse.Namespace, seq_len: int) -> int:
    return seq_len if args.q_len is None else args.q_len


def attention_implementation(q_len: int) -> str:
    return "single_query" if q_len == 1 else "tiled"


def make_qkv(args: argparse.Namespace, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=args.device)
    q_len = query_len(args, seq_len)
    generator.manual_seed(args.seed + seq_len + q_len)

    shape_q = (args.batch_size, args.heads, q_len, args.head_dim)
    shape_kv = (args.batch_size, args.kv_heads, seq_len, args.head_dim)
    q = torch.randn(shape_q, device=args.device, dtype=args.torch_dtype, generator=generator)
    k = torch.randn(shape_kv, device=args.device, dtype=args.torch_dtype, generator=generator)
    v = torch.randn(shape_kv, device=args.device, dtype=args.torch_dtype, generator=generator)
    return q, k, v


def require_flash_attn_func() -> Callable[..., torch.Tensor]:
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        raise MissingDependency(
            "flash-attn is not importable; install it or use --fp16-backend torch"
        ) from exc
    return flash_attn_func


def missing_dependency_fn(note: str) -> TensorFn:
    def run() -> torch.Tensor:
        raise MissingDependency(note)

    return run


def make_flash_attn_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> TensorFn:
    flash_attn_func = require_flash_attn_func()
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    def run() -> torch.Tensor:
        return flash_attn_func(q_bshd, k_bshd, v_bshd, dropout_p=0.0, causal=causal)

    return run


def require_sageattn3() -> Callable[..., torch.Tensor]:
    try:
        from sageattn3 import sageattn3_blackwell
    except Exception as exc:
        raise MissingDependency("sageattn3 is not importable; install SageAttention/sageattention3_blackwell") from exc
    return sageattn3_blackwell


def make_sage_fp4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> TensorFn:
    sageattn3_blackwell = require_sageattn3()

    def run() -> torch.Tensor:
        return sageattn3_blackwell(q, k, v, is_causal=causal, per_block_mean=False)

    return run


def selected_top_k(seq_len: int, coverage: float, *, causal: bool, block_size: int = 64) -> int:
    return ta.resolve_top_k(seq_len // block_size, causal=causal, fraction=coverage)


def build_specs(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: int,
) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    impl = attention_implementation(q.shape[2])

    if not args.skip_fp16:
        if args.fp16_backend == "flash-attn":
            fn = make_flash_attn_fp16(q, k, v, causal=args.causal)
            specs.append(
                BenchmarkSpec(
                    f"{args.dtype_name}_flash_attn",
                    None,
                    None,
                    fn,
                )
            )
    if not args.skip_fp4:
        specs.append(
            BenchmarkSpec(
                "ta_fp4_attention",
                None,
                None,
                lambda: ta.attention(
                    q,
                    k,
                    v,
                    config=ta.AttentionConfig(
                        method="fp4",
                        causal=args.causal,
                        implementation=impl,
                    ),
                ),
            )
        )
    if not args.skip_thrift:
        for coverage in args.coverages:
            specs.append(
                BenchmarkSpec(
                    f"thrift_{coverage * 100:g}pct",
                    coverage,
                    selected_top_k(seq_len, coverage, causal=args.causal and q.shape[2] != 1),
                    lambda coverage=coverage: ta.attention(
                        q,
                        k,
                        v,
                        config=ta.AttentionConfig(
                            method="thrift",
                            causal=args.causal,
                            fraction=coverage,
                            implementation=impl,
                        ),
                    ),
                )
            )

    if not args.skip_fp4:
        try:
            fn = make_sage_fp4(q, k, v, causal=args.causal)
        except MissingDependency as exc:
            fn = missing_dependency_fn(str(exc))
        specs.append(
            BenchmarkSpec(
                "fp4_sageattn3",
                None,
                None,
                fn,
            )
        )

    return specs


def measure_cuda_events(fn: TensorFn, *, warmup: int, repeat: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return TimingStats(
        mean_ms=statistics.fmean(times_ms),
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p90_ms=percentile(times_ms, 0.90),
    )


def profile_cuda_kernels(name: str, fn: TensorFn, *, iters: int, row_limit: int) -> None:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception as exc:
        print(f"\nProfiler unavailable for {name}: {exc}")
        return

    print(f"\nProfiler: {name}")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for _ in range(iters):
            with record_function(name):
                fn()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))


def run_spec(args: argparse.Namespace, seq_len: int, spec: BenchmarkSpec) -> BenchmarkResult:
    try:
        stats = measure_cuda_events(spec.fn, warmup=args.warmup, repeat=args.repeat)
        if args.profile:
            profile_cuda_kernels(spec.name, spec.fn, iters=args.profile_iters, row_limit=args.profile_top)
        return BenchmarkResult(seq_len, spec.name, spec.coverage, spec.top_k, "ok", stats=stats)
    except MissingDependency as exc:
        return BenchmarkResult(seq_len, spec.name, spec.coverage, spec.top_k, "skip", note=str(exc))
    except Exception as exc:
        if args.keep_going:
            return BenchmarkResult(seq_len, spec.name, spec.coverage, spec.top_k, "error", note=repr(exc))
        raise


def print_results(results: list[BenchmarkResult]) -> None:
    headers = [
        "seq",
        "benchmark",
        "coverage",
        "top_k",
        "status",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "min_ms",
        "max_ms",
        "note",
    ]
    rows: list[list[str]] = []
    for result in results:
        stats = result.stats
        rows.append(
            [
                str(result.seq_len),
                result.name,
                format_coverage(result.coverage),
                "-" if result.top_k is None else str(result.top_k),
                result.status,
                "-" if stats is None else f"{stats.mean_ms:.3f}",
                "-" if stats is None else f"{stats.median_ms:.3f}",
                "-" if stats is None else f"{stats.p90_ms:.3f}",
                "-" if stats is None else f"{stats.min_ms:.3f}",
                "-" if stats is None else f"{stats.max_ms:.3f}",
                result.note,
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print()
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def validate_args(args: argparse.Namespace) -> None:
    if args.head_dim not in (64, 128):
        raise SystemExit("--head-dim must be 64 or 128")
    if args.heads % args.kv_heads != 0:
        raise SystemExit("--heads must be divisible by --kv-heads")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for these benchmarks")
    for seq_len in args.seq_lens:
        q_len = query_len(args, seq_len)
        if seq_len <= 0:
            raise SystemExit("sequence lengths must be positive")
        if q_len <= 0:
            raise SystemExit("--q-len must be positive")
        if q_len != 1 and q_len % 64 != 0 and not args.skip_thrift:
            raise SystemExit("query lengths must be 1 or divisible by 64 for ThriftAttention")
        if q_len != 1 and q_len % 64 != 0 and not args.skip_fp4:
            raise SystemExit("query lengths must be 1 or divisible by 64 for FP4 paths")
        if seq_len % 64 != 0 and not args.skip_thrift:
            raise SystemExit("sequence lengths must be divisible by 64 for ThriftAttention")
        if seq_len % 64 != 0 and not args.skip_fp4:
            raise SystemExit("sequence lengths must be divisible by 64 for FP4 paths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark high-precision flash attention, ThriftAttention coverages, and SageAttention3 FP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seq_lens", nargs="+", help="Sequence lengths, as spaces or comma-separated values.")
    parser.add_argument(
        "--coverages",
        type=parse_fraction_list,
        default=parse_fraction_list("1%,5%,10%"),
        help="Comma-separated ThriftAttention coverages. Values may be fractions or percentages.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--q-len", type=int, default=None, help="Query length. Defaults to each sequence length; use 1 for decode.")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16", help="Input dtype.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--fp16-backend", choices=("flash-attn", "torch"), default="flash-attn")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-thrift", action="store_true")
    parser.add_argument("--skip-fp4", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Print a torch.profiler CUDA table for each benchmark.")
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--profile-top", type=int, default=20)
    parser.add_argument("--keep-going", action="store_true", help="Report benchmark errors instead of stopping.")
    args = parser.parse_args()
    args.seq_lens = parse_int_list(args.seq_lens)
    args.torch_dtype = parse_dtype(args.dtype)
    args.dtype_name = args.dtype
    return args


def main() -> None:
    args = parse_args()
    validate_args(args)

    all_results: list[BenchmarkResult] = []
    for seq_len in args.seq_lens:
        q, k, v = make_qkv(args, seq_len)
        q_len = q.shape[2]
        impl = attention_implementation(q_len)
        print(
            f"Running q={q_len}, kv={seq_len}, implementation={impl}, batch={args.batch_size}, "
            f"heads={args.heads}, kv_heads={args.kv_heads}, dim={args.head_dim}, dtype={args.dtype_name}"
        )
        specs = build_specs(args, q, k, v, seq_len)

        for spec in specs:
            result = run_spec(args, seq_len, spec)
            all_results.append(result)
            status = result.status.upper()
            suffix = f": {result.note}" if result.note else ""
            print(f"  {status} {spec.name}{suffix}")

        del q, k, v
        torch.cuda.empty_cache()

    print_results(all_results)


if __name__ == "__main__":
    main()
