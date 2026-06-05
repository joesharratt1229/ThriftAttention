"""kernels-community entrypoint for ThriftAttention.

Loaded by `kernels.get_kernel(...)` when a `transformers` model is created
with `attn_implementation="hrsh-venket/thrift-attention"`. Exposes a single
`forward` callable matching the `AttentionInterface` signature.

To override the default attention config (selection fraction, method, etc.),
mutate `default_config` after loading the kernel:

    from kernels import get_kernel
    ta = get_kernel("hrsh-venket/thrift-attention")
    ta.default_config = ta.AttentionConfig(fraction=0.25)
"""

from __future__ import annotations

from typing import Any

import torch

from .config import AttentionConfig
from .functional import attention as _thrift_attention
from .transformers_cache import (
    cached_decode_attention,
    cached_prefill_attention,
    get_active_thrift_attention_cache,
)

__all__ = ["AttentionConfig", "default_config", "forward"]

default_config: AttentionConfig = AttentionConfig()


def forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if attention_mask is not None:
        raise RuntimeError("ThriftAttention does not accept explicit attention masks")
    if kwargs.get("output_attentions", False):
        raise RuntimeError("ThriftAttention does not return attention weights")

    cache = get_active_thrift_attention_cache()
    if cache is not None:
        layer_idx = int(getattr(module, "layer_idx"))
        if layer_idx < len(cache.layers) and cache.get_seq_length(layer_idx) > 0:
            fn = cached_decode_attention if query.shape[2] == 1 else cached_prefill_attention
            out = fn(query, cache, layer_idx, default_config)
            return out.transpose(1, 2).contiguous(), None

    out = _thrift_attention(query, key, value, config=default_config)
    return out.transpose(1, 2).contiguous(), None
