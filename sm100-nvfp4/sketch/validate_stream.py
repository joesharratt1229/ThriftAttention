#!/usr/bin/env python3
"""Bit-exact A/B validation of the streamed-softmax restructure.

Builds two extensions from the same .cpp/quantise sources:
  base: fp4_attention_sm100_pre_stream_snapshot.cu (pre-restructure kernel)
  new:  fp4_attention_sm100.cu (chunk-pipelined S stream)

The restructure changes no arithmetic (convert is block-relative; only the
load/convert schedule moved), so torch.equal must hold on every shape.
Also runs the standing hang matrix (RESTRUCTURE_PLAN section E), a
determinism check, and cos vs fp32 SDPA.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
HEAD_DIM = 128

CUDA_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--relocatable-device-code=false",
    "-lineinfo",
]


def build(name: str, kernel_cu: Path):
    build_dir = ROOT / ".torch_extensions" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "lock").unlink(missing_ok=True)
    # torch's extension loader keys the module on the source file names, so
    # the snapshot kernel must be compiled under the canonical file name.
    src_dir = Path(tempfile.mkdtemp(prefix=f"{name}_src_"))
    for f in ("fp4_attention_extension.cpp", "quantise_nvfp4.cu"):
        shutil.copy(ROOT / f, src_dir / f)
    shutil.copy(kernel_cu, src_dir / "fp4_attention_sm100.cu")
    return load(
        name=name,
        sources=[str(src_dir / "fp4_attention_extension.cpp"),
                 str(src_dir / "fp4_attention_sm100.cu"),
                 str(src_dir / "quantise_nvfp4.cu")],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_args(ext, batch, heads, kv_heads, q_len, kv_len):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, heads, q_len, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(batch, kv_heads, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn(batch, kv_heads, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    pre = ext.quantise_and_attention(q, k, v, scale)
    args = (pre["q_fp4"], pre["k_fp4"], pre["v_t_fp4"],
            pre["q_sf_atoms"], pre["k_sf_atoms"], pre["v_sf_atoms"])
    return args, scale, (q, k, v)


# (batch, heads, kv_heads, q, kv) -- the standing hang/validation matrix.
SHAPES = [
    (1, 16, 16, 128, 128),      # minimum shape
    (1, 16, 16, 512, 512),
    (1, 16, 16, 512, 4096),
    (1, 16, 16, 512, 384),      # odd kv_iters
    (1, 16, 16, 2048, 128),     # kv_iters=1, multi-tile
    (1, 16, 16, 2048, 1152),    # multi-tile + odd
    (1, 8, 2, 2048, 1152),      # GQA 8:2
    (1, 16, 16, 4096, 4096),
    (1, 16, 16, 8192, 8192),
]


def main() -> None:
    snap = ROOT / "fp4_attention_sm100_pre_stream_snapshot.cu"
    assert snap.exists(), "snapshot kernel missing"
    print("building base (pre-stream snapshot)...", flush=True)
    base = build("fp4attn_base_snap", snap)
    print("building new (streamed)...", flush=True)
    new = build("fp4attn_stream", ROOT / "fp4_attention_sm100.cu")

    ok = True
    for batch, heads, kv_heads, q_len, kv_len in SHAPES:
        args, scale, (q, k, v) = make_args(base, batch, heads, kv_heads, q_len, kv_len)
        out_base = base.attention_only(*args, scale)
        outs_new = [new.attention_only(*args, scale) for _ in range(5)]
        torch.cuda.synchronize()
        biteq = all(torch.equal(out_base, o) for o in outs_new)
        det = all(torch.equal(outs_new[0], o) for o in outs_new[1:])
        line = (f"b{batch} h{heads}/{kv_heads} q{q_len} kv{kv_len}: "
                f"biteq={biteq} det(x5)={det}")
        if q_len == kv_len and q_len >= 4096 and heads == kv_heads:
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), scale=scale)
            cos = torch.nn.functional.cosine_similarity(
                outs_new[0].float().flatten(), ref.flatten(), dim=0).item()
            line += f" cos={cos:.4f}"
        print(line, flush=True)
        ok &= biteq and det
    print("PASS" if ok else "FAIL", flush=True)
    if not ok:
        raise SystemExit(1)

    # Interleaved A/B timing (same thermal state for both kernels).
    print("\nperf (interleaved, batch=1 h16):", flush=True)
    for q_len in (2048, 4096, 8192, 16384):
        args, scale, _ = make_args(base, 1, 16, 16, q_len, q_len)
        fa = lambda: base.attention_only(*args, scale)
        fb = lambda: new.attention_only(*args, scale)
        for f in (fa, fb):
            for _ in range(10):
                f()
        chunk, total_a, total_b, n = 10, 0.0, 0.0, 0
        for _ in range(10):
            for which, f in (("a", fa), ("b", fb)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                s.record()
                for _ in range(chunk):
                    f()
                e.record()
                torch.cuda.synchronize()
                if which == "a":
                    total_a += s.elapsed_time(e)
                else:
                    total_b += s.elapsed_time(e)
            n += chunk
        print(f"  q=kv={q_len}: base {total_a / n:.3f} ms | stream {total_b / n:.3f} ms "
              f"| {total_a / total_b:.3f}x", flush=True)


if __name__ == "__main__":
    main()
