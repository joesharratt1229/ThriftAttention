from __future__ import annotations

import argparse
import os
import pwd
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

sudo_user = os.environ.get("SUDO_USER")
if sudo_user:
    try:
        sudo_home = Path(pwd.getpwnam(sudo_user).pw_dir)
    except KeyError:
        sudo_home = None
    if sudo_home is not None:
        user_site = (
            sudo_home
            / ".local"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if user_site.exists():
            sys.path.append(str(user_site))

import torch
from thriftattention import _C


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def cuda_profiler_start() -> None:
    result = torch.cuda.cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with error code {result}")


def cuda_profiler_stop() -> None:
    result = torch.cuda.cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with error code {result}")


def _mxfp4_quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_packed, q_scale = _C.mxfp4_quantize(q.contiguous(), is_bf16)
    k_packed, k_scale = _C.mxfp4_quantize_permuted(k.contiguous(), is_bf16)
    v_packed_t, v_scale_t = _C.mxfp4_quantize_transposed(v.contiguous(), is_bf16)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--exp-approx", type=bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    dtype = torch.float16
    device = torch.device("cuda")
    q = torch.randn(args.batch, args.q_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    packed = _mxfp4_quantize_qkv(q, k, v, is_bf16=False)

    label = "mxfp4 approx exp" if args.exp_approx else "mxfp4 exact exp"

    for _ in range(args.warmup):
        _C.fp4_attention_noncausal_mxfp4_packed(*packed, is_bf16=False, exp_approx=args.exp_approx)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    cuda_profiler_start()
    x = torch.randn(128, 128)
    
    
    try:
        with nvtx_range(label):
            #torch.exp(x)
            #_C.fp4_attention_noncausal_mxfp4_packed(*packed, is_bf16=False, exp_approx=args.exp_approx)
        torch.cuda.synchronize()
    finally:
        cuda_profiler_stop()


if __name__ == "__main__":
    main()
