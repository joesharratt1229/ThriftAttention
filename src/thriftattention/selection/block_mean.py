from __future__ import annotations

import math

import torch

from thriftattention._checks import check_qkv, require_block_aligned
from thriftattention._extension import get_extension
from thriftattention.selection.base import SelectionConfig


def resolve_top_k(
    num_blocks: int,
    *,
    causal: bool = True,
    top_k: int | None = None,
    fraction: float | None = None,
) -> int:
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        return min(top_k, num_blocks)
    if fraction is None:
        fraction = 0.05
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if num_blocks <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return num_blocks

    if not causal:
        return max(1, min(num_blocks, round(fraction * num_blocks)))

    b = -(2 * num_blocks + 1)
    c = fraction * num_blocks * (num_blocks + 1)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return num_blocks
    top_k_float = (-b - math.sqrt(discriminant)) / 2.0
    return max(1, min(num_blocks, round(top_k_float)))


def block_means(x: torch.Tensor, *, block_size: int = 64) -> torch.Tensor:
    require_block_aligned("x", x.shape[2], block_size)
    batch, heads, seq_len, head_dim = x.shape
    return (
        x.reshape(batch, heads, seq_len // block_size, block_size, head_dim)
        .float()
        .mean(dim=3)
        .to(torch.float16)
        .contiguous()
    )


def _expand_kv_heads(q: torch.Tensor, k_mean: torch.Tensor) -> torch.Tensor:
    if q.shape[1] == k_mean.shape[1]:
        return k_mean
    groups = q.shape[1] // k_mean.shape[1]
    return k_mean.repeat_interleave(groups, dim=1).contiguous()


def select_block_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
) -> torch.Tensor:
    """Return selected KV blocks for each query block."""
    check_qkv(q, k, k)
    require_block_aligned("q", q.shape[2], block_size)
    require_block_aligned("k", k.shape[2], block_size)

    num_kv_blocks = k.shape[2] // block_size
    selected_count = resolve_top_k(
        num_kv_blocks,
        causal=causal,
        top_k=top_k,
        fraction=fraction,
    )
    if selected_count == 0:
        return torch.empty(
            q.shape[0] * q.shape[1],
            q.shape[2] // block_size,
            0,
            device=q.device,
            dtype=torch.int32,
        )

    q_mean = block_means(q, block_size=block_size)
    k_mean = _expand_kv_heads(q, block_means(k, block_size=block_size))
    if num_kv_blocks <= 2048:
        return get_extension().block_mean_topk(q_mean, k_mean, selected_count, causal)

    scores = (
        q_mean.reshape(q.shape[0] * q.shape[1], q_mean.shape[2], q.shape[3]).float()
        @ k_mean.reshape(q.shape[0] * q.shape[1], num_kv_blocks, q.shape[3])
        .float()
        .transpose(-1, -2)
    )
    if causal:
        mask = torch.triu(
            torch.ones(q_mean.shape[2], num_kv_blocks, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    indices = scores.topk(selected_count, dim=-1).indices.to(torch.int32)
    if causal:
        valid_counts = torch.arange(1, q_mean.shape[2] + 1, device=q.device).clamp(max=num_kv_blocks)
        ranks = torch.arange(selected_count, device=q.device)
        indices.masked_fill_(ranks.view(1, 1, -1) >= valid_counts.view(1, -1, 1), -1)
    return indices.contiguous()


def select_key_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
) -> torch.Tensor:
    """Return selected KV blocks for single-query attention."""
    check_qkv(q, k, k)
    if q.shape[2] != 1:
        raise ValueError("single-query block selection expects q sequence length 1")
    require_block_aligned("k", k.shape[2], block_size)

    num_kv_blocks = k.shape[2] // block_size
    selected_count = resolve_top_k(
        num_kv_blocks,
        causal=False,
        top_k=top_k,
        fraction=fraction,
    )
    batch, q_heads, _, head_dim = q.shape
    kv_heads = k.shape[1]
    groups = q_heads // kv_heads
    if selected_count == 0:
        return torch.empty(
            batch * kv_heads,
            0,
            device=q.device,
            dtype=torch.int32,
        )

    q_grouped = q.reshape(batch, kv_heads, groups, head_dim)
    k_mean = block_means(k, block_size=block_size)
    scores = (q_grouped.float().unsqueeze(3) * k_mean.float().unsqueeze(2)).sum(dim=-1)
    scores = scores.amax(dim=2).reshape(batch * kv_heads, num_kv_blocks)
    return scores.topk(selected_count, dim=-1).indices.to(torch.int32).contiguous()


class BlockMeanSelectionPolicy:
    name = "block_mean"

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        config: SelectionConfig,
        causal: bool,
    ) -> torch.Tensor:
        if config.name != self.name:
            raise ValueError(f"selection config {config.name!r} does not match block_mean policy")
        if q.shape[2] == 1:
            return select_key_blocks(
                q,
                k,
                top_k=config.top_k,
                fraction=config.fraction,
                block_size=config.block_size,
            )
        return select_block_pairs(
            q,
            k,
            causal=causal,
            top_k=config.top_k,
            fraction=config.fraction,
            block_size=config.block_size,
        )
