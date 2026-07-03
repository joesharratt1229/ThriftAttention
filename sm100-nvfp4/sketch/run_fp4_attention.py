#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def build_extension(verbose: bool = True):
    build_dir = ROOT / ".torch_extensions" / "fp4_attention_sm100_ext"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="fp4_attention_sm100_ext",
        sources=[
            str(ROOT / "fp4_attention_extension.cpp"),
            str(ROOT / "fp4_attention_sm100.cu"),
            str(ROOT / "quantise_nvfp4.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--quiet-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    head_dim = 128

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.q_heads, args.q_len, head_dim, device=device, dtype=dtype).contiguous()
    k = torch.randn(args.batch, args.kv_heads, args.kv_len, head_dim, device=device, dtype=dtype).contiguous()
    v = torch.randn(args.batch, args.kv_heads, args.kv_len, head_dim, device=device, dtype=dtype).contiguous()

    ext = build_extension(verbose=not args.quiet_build)
    result = ext.quantise_and_attention(q, k, v, 1.0 / math.sqrt(head_dim))
    torch.cuda.synchronize()

    out = result["out"]
    print(f"out shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print(f"out finite={torch.isfinite(out.float()).all().item()} mean={out.float().mean().item():.6f} std={out.float().std().item():.6f}")
    #print(f"q_fp4 shape={tuple(result['q_fp4'].shape)} q_sf_atoms shape={tuple(result['q_sf_atoms'].shape)}")
    #print(f"k_fp4 shape={tuple(result['k_fp4'].shape)} k_sf_atoms shape={tuple(result['k_sf_atoms'].shape)}")
    #print(f"v_t_fp4 shape={tuple(result['v_t_fp4'].shape)} v_sf_atoms shape={tuple(result['v_sf_atoms'].shape)}")


if __name__ == "__main__":
    main()
