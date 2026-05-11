from __future__ import annotations

import torch

from ._checks import check_qkv, require_block_aligned, require_causal
from ._extension import get_extension
from .quantization import nvfp4_quantize, nvfp4_quantize_transposed
from .selection import select_blocks


def _quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_packed, q_scale = nvfp4_quantize(q)
    k_packed, k_scale = nvfp4_quantize(k)
    v_packed_t, v_scale_t = nvfp4_quantize_transposed(v)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


def fp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Run the pure NVFP4 causal attention baseline."""
    require_causal(causal)
    check_qkv(q, k, v)
    require_block_aligned("q", q.shape[2], 64)
    require_block_aligned("k", k.shape[2], 64)
    packed = _quantize_qkv(q.contiguous(), k.contiguous(), v.contiguous())
    return get_extension().fp4_attention_causal_nvfp4_packed(*packed)


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
) -> torch.Tensor:
    """Run ThriftAttention with separate block selection and NVFP4 quantization."""
    require_causal(causal)
    check_qkv(q, k, v)
    if block_size != 64:
        raise NotImplementedError("the current CUDA attention kernel uses 64-token KV blocks")
    require_block_aligned("q", q.shape[2], 64)
    require_block_aligned("k", k.shape[2], block_size)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    selected_blocks = select_blocks(
        q,
        k,
        causal=causal,
        method=selector,
        top_k=top_k,
        fraction=fraction,
        block_size=block_size,
    )
    packed = _quantize_qkv(q, k, v)
    return get_extension().thrift_attention_causal_nvfp4_packed(
        q,
        k,
        v,
        selected_blocks,
        *packed,
    )
