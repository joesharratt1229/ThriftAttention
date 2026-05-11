from __future__ import annotations

import torch


def require_cuda_half(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.float16:
        raise ValueError(f"{name} must have dtype torch.float16")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 4D [batch, heads, seq, head_dim]")


def require_supported_head_dim(head_dim: int) -> None:
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")


def check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    require_cuda_half("q", q)
    require_cuda_half("k", k)
    require_cuda_half("v", v)
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of KV heads")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("q heads must be divisible by KV heads")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("q, k, and v must have the same head_dim")
    require_supported_head_dim(q.shape[3])


def require_causal(causal: bool) -> None:
    if not causal:
        raise NotImplementedError(
            "non-causal kernels are planned but are not part of this migration yet"
        )


def require_block_aligned(name: str, seq_len: int, block_size: int) -> None:
    if seq_len % block_size != 0:
        raise ValueError(f"{name} sequence length must be divisible by {block_size}")
