#!/usr/bin/env python3
from __future__ import annotations

import argparse

import run_kernel_benchmarks as base
import thriftattention as ta


_ORIGINAL_VALIDATE_ARGS = base.validate_args


def _requires_rebuilt_exp_approx_extension() -> None:
    from thriftattention._extension import get_extension

    ext = get_extension()
    fn = ext.fp4_attention_noncausal_nvfp4_packed
    doc = getattr(fn, "__doc__", "") or ""
    if "exp_approx" not in doc:
        raise RuntimeError(
            "compiled thriftattention._C does not expose exp_approx yet; rebuild with "
            "`python3 -m pip install -e . --no-build-isolation`"
        )


def build_specs(
    args: argparse.Namespace,
    q,
    k,
    v,
    seq_len: int,
) -> list[base.BenchmarkSpec]:
    specs: list[base.BenchmarkSpec] = []
    impl = base.attention_implementation(q.shape[2])

    if not args.skip_fp16:
        if args.fp16_backend == "flash-attn":
            fn = base.make_flash_attn_fp16(q, k, v, causal=args.causal)
            specs.append(base.BenchmarkSpec(f"{args.dtype_name}_flash_attn", None, None, fn))

    if not args.skip_fp4:
        _requires_rebuilt_exp_approx_extension()
        specs.append(
            base.BenchmarkSpec(
                f"ta_{args.quant_format}_fp4_attention_exact",
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
                        quant_format=args.quant_format,
                    ),
                ),
            )
        )
        specs.append(
            base.BenchmarkSpec(
                f"ta_{args.quant_format}_fp4_attention_exp_approx",
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
                        quant_format=args.quant_format,
                        exp_approx="codebook",
                    ),
                ),
            )
        )

    if not args.skip_thrift:
        for coverage in args.coverages:
            specs.append(
                base.BenchmarkSpec(
                    f"thrift_{args.quant_format}_{coverage * 100:g}pct",
                    coverage,
                    base.selected_top_k(seq_len, coverage, causal=args.causal and q.shape[2] != 1),
                    lambda coverage=coverage: ta.attention(
                        q,
                        k,
                        v,
                        config=ta.AttentionConfig(
                            method="thrift",
                            causal=args.causal,
                            fraction=coverage,
                            implementation=impl,
                            quant_format=args.quant_format,
                        ),
                    ),
                )
            )

    if not args.skip_fp4:
        try:
            fn = base.make_sage_fp4(q, k, v, causal=args.causal)
        except base.MissingDependency as exc:
            fn = base.missing_dependency_fn(str(exc))
        specs.append(base.BenchmarkSpec("fp4_sageattn3", None, None, fn))

    return specs


def validate_args(args: argparse.Namespace) -> None:
    _ORIGINAL_VALIDATE_ARGS(args)
    if args.skip_fp4:
        return
    for seq_len in args.seq_lens:
        if base.query_len(args, seq_len) == 1:
            raise SystemExit("exp-approx fp4 benchmark only supports tiled prefill; do not use --q-len 1")


base.build_specs = build_specs
base.validate_args = validate_args


if __name__ == "__main__":
    base.main()
