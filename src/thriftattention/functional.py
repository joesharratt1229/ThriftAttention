from __future__ import annotations

import torch

from ._checks import check_qkv, require_block_aligned
from ._extension import get_extension
from .quantization import (
    nvfp4_quantize,
    nvfp4_quantize_permuted,
    nvfp4_quantize_transposed,
)
from .selection import select_block_pairs, select_key_blocks


def _quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_packed, q_scale = nvfp4_quantize(q)
    k_packed, k_scale = nvfp4_quantize_permuted(k)
    v_packed_t, v_scale_t = nvfp4_quantize_transposed(v)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


def _quantize_single_query_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_packed, q_scale = nvfp4_quantize(q)
    k_packed, k_scale = nvfp4_quantize(k)
    v_packed_t, v_scale_t = nvfp4_quantize_transposed(v)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


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


def fp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    _implementation: str = "auto",
) -> torch.Tensor:
    """Run the pure NVFP4 attention baseline."""
    check_qkv(q, k, v)
    if _use_single_query(q, _implementation):
        require_block_aligned("k", k.shape[2], 64)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_grouped = _group_single_query(q, k.shape[1])
        packed = _quantize_single_query_qkv(q_grouped, k, v)
        out = get_extension().fp4_attention_single_query_nvfp4_packed(*packed)
        return _ungroup_single_query(out, q.shape[1])

    require_block_aligned("q", q.shape[2], 64)
    require_block_aligned("k", k.shape[2], 64)
    packed = _quantize_qkv(q.contiguous(), k.contiguous(), v.contiguous())
    ext = get_extension()
    if causal:
        return ext.fp4_attention_causal_nvfp4_packed(*packed)
    return ext.fp4_attention_noncausal_nvfp4_packed(*packed)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    selector: str = "block_mean",
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
    _implementation: str = "auto",
) -> torch.Tensor:
    """Run ThriftAttention with separate block selection and NVFP4 quantization."""
    check_qkv(q, k, v)
    if block_size != 64:
        raise NotImplementedError("the current CUDA attention kernel uses 64-token KV blocks")
    if _use_single_query(q, _implementation):
        require_block_aligned("k", k.shape[2], block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_grouped = _group_single_query(q, k.shape[1])
        selected_blocks = select_key_blocks(
            q,
            k,
            method=selector,
            top_k=top_k,
            fraction=fraction,
            block_size=block_size,
        )
        packed = _quantize_single_query_qkv(q_grouped, k, v)
        out = get_extension().thrift_attention_single_query_nvfp4_packed(
            q_grouped,
            k,
            v,
            selected_blocks,
            *packed,
        )
        return _ungroup_single_query(out, q.shape[1])

    require_block_aligned("q", q.shape[2], 64)
    require_block_aligned("k", k.shape[2], block_size)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    selected_blocks = select_block_pairs(
        q,
        k,
        causal=causal,
        method=selector,
        top_k=top_k,
        fraction=fraction,
        block_size=block_size,
    )
    packed = _quantize_qkv(q, k, v)
    ext = get_extension()
    fn = (
        ext.thrift_attention_causal_nvfp4_packed
        if causal
        else ext.thrift_attention_noncausal_nvfp4_packed
    )
    return fn(
        q,
        k,
        v,
        selected_blocks,
        *packed,
    )
