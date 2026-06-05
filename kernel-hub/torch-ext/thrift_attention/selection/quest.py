from __future__ import annotations

import torch

from thrift_attention._checks import check_qkv, require_block_aligned
from thrift_attention._extension import get_extension
from thrift_attention.selection.base import SelectionConfig
from thrift_attention.selection.block_mean import resolve_top_k


def block_minmax(
    x: torch.Tensor,
    *,
    block_size: int = 64,
    is_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_block_aligned("x", x.shape[2], block_size)
    batch, heads, seq_len, head_dim = x.shape
    out_dtype = torch.bfloat16 if is_bf16 else torch.float16
    blocks = x.reshape(batch, heads, seq_len // block_size, block_size, head_dim).float()
    return (
        blocks.amin(dim=3).to(out_dtype).contiguous(),
        blocks.amax(dim=3).to(out_dtype).contiguous(),
    )


def _expand_kv_heads(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if q.shape[1] == x.shape[1]:
        return x
    groups = q.shape[1] // x.shape[1]
    return x.repeat_interleave(groups, dim=1).contiguous()


def _quest_scores(q: torch.Tensor, k_min: torch.Tensor, k_max: torch.Tensor) -> torch.Tensor:
    qf = q.float().unsqueeze(3)
    kmin = k_min.float().unsqueeze(2)
    kmax = k_max.float().unsqueeze(2)
    return (qf * torch.where(qf >= 0.0, kmax, kmin)).sum(dim=-1)


def select_quest_block_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
    is_bf16: bool = False,
) -> torch.Tensor:
    """Return selected KV blocks using QUEST min/max block scores."""
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

    q_mean = (
        q.reshape(q.shape[0], q.shape[1], q.shape[2] // block_size, block_size, q.shape[3])
        .float()
        .mean(dim=3)
        .to(torch.bfloat16 if is_bf16 else torch.float16)
        .contiguous()
    )
    k_min, k_max = block_minmax(k, block_size=block_size, is_bf16=is_bf16)
    k_min = _expand_kv_heads(q, k_min)
    k_max = _expand_kv_heads(q, k_max)

    if q.is_cuda and num_kv_blocks <= 2048 and q.shape[3] in (64, 128):
        return get_extension().quest_block_topk(q_mean, k_min, k_max, selected_count, causal, is_bf16)

    scores = _quest_scores(q_mean, k_min, k_max)
    if causal:
        mask = torch.triu(
            torch.ones(q_mean.shape[2], num_kv_blocks, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    scores = scores.reshape(q.shape[0] * q.shape[1], q_mean.shape[2], num_kv_blocks)
    indices = scores.topk(selected_count, dim=-1).indices.to(torch.int32)
    if causal:
        valid_counts = torch.arange(1, q_mean.shape[2] + 1, device=q.device).clamp(max=num_kv_blocks)
        ranks = torch.arange(selected_count, device=q.device)
        indices.masked_fill_(ranks.view(1, 1, -1) >= valid_counts.view(1, -1, 1), -1)
    return indices.contiguous()


def select_quest_key_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
    is_bf16: bool = False,
) -> torch.Tensor:
    """Return selected KV blocks for single-query QUEST attention."""
    check_qkv(q, k, k)
    if q.shape[2] != 1:
        raise ValueError("single-query QUEST selection expects q sequence length 1")
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

    q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()
    k_min, k_max = block_minmax(k, block_size=block_size, is_bf16=is_bf16)

    if q.is_cuda and num_kv_blocks <= 2048 and head_dim in (64, 128):
        return get_extension().single_query_quest_topk(
            q_grouped,
            k_min,
            k_max,
            selected_count,
            num_kv_blocks,
            is_bf16,
        )

    qf = q_grouped.float().unsqueeze(3)
    kmin = k_min.float().unsqueeze(2)
    kmax = k_max.float().unsqueeze(2)
    scores = (qf * torch.where(qf >= 0.0, kmax, kmin)).sum(dim=-1)
    scores = scores.amax(dim=2).reshape(batch * kv_heads, num_kv_blocks)
    return scores.topk(selected_count, dim=-1).indices.to(torch.int32).contiguous()


class QuestSelectionPolicy:
    name = "quest"

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        config: SelectionConfig,
        causal: bool,
        is_bf16: bool,
    ) -> torch.Tensor:
        if config.name != self.name:
            raise ValueError(f"selection config {config.name!r} does not match QUEST policy")
        if q.shape[2] == 1:
            return select_quest_key_blocks(
                q,
                k,
                top_k=config.top_k,
                fraction=config.fraction,
                block_size=config.block_size,
                is_bf16=is_bf16,
            )
        return select_quest_block_pairs(
            q,
            k,
            causal=causal,
            top_k=config.top_k,
            fraction=config.fraction,
            block_size=config.block_size,
            is_bf16=is_bf16,
        )
