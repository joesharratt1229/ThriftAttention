from __future__ import annotations

import torch

from thriftattention.backends.registry import select_backend
from thriftattention.config import AttentionConfig
from thriftattention.quant.formats import get_quant_format
from thriftattention.selection import get_selection_policy
from thriftattention.selection.base import SelectionConfig


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    config: AttentionConfig | None = None,
) -> torch.Tensor:
    if config is None:
        config = AttentionConfig()

    head_dim = q.shape[-1]
    is_bf16 = q.dtype == torch.bfloat16
    quant_format = get_quant_format(config.quant_format)
    backend = select_backend(
        config,
        quant_format,
        head_dim=head_dim,
        device=q.device,
    )

    selection = None
    if config.method == "thrift":
        policy = get_selection_policy(config.selection)
        # hd256 tiled kernels use 128 Q/KV blocks; all other paths use config.block_size
        is_tiled = q.shape[2] > 1
        block_size = 128 if (head_dim == 256 and is_tiled) else config.block_size
        selection = policy.select(
            q,
            k,
            config=SelectionConfig(
                name=config.selection,
                fraction=config.fraction,
                top_k=config.top_k,
                block_size=block_size,
            ),
            causal=config.causal,
            is_bf16=is_bf16,
        )

    return backend.attention(
        q,
        k,
        v,
        selection=selection,
        quant_format=quant_format,
        config=config,
        is_bf16=is_bf16,
    )
