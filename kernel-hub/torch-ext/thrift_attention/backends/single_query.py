from __future__ import annotations

import torch


def _use_single_query(q: torch.Tensor, implementation: str) -> bool:
    if implementation == "auto":
        return q.shape[2] == 1
    if implementation == "tiled":
        return False
    if implementation == "single_query":
        if q.shape[2] != 1:
            raise ValueError("implementation='single_query' requires q sequence length 1")
        return True
    raise ValueError("implementation must be 'auto', 'tiled', or 'single_query'")


def _group_single_query(q: torch.Tensor, kv_heads: int) -> torch.Tensor:
    batch, q_heads, q_len, head_dim = q.shape
    if q_len != 1:
        raise ValueError("single-query attention expects q sequence length 1")
    groups = q_heads // kv_heads
    if groups > 16:
        raise NotImplementedError("single-query kernels support at most 16 Q heads per KV head")
    return q.reshape(batch, kv_heads, groups, head_dim).contiguous()


def _ungroup_single_query(out: torch.Tensor, q_heads: int) -> torch.Tensor:
    batch, _, _, head_dim = out.shape
    return out.reshape(batch, q_heads, 1, head_dim).contiguous()
