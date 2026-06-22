from __future__ import annotations

import torch

from thriftattention._checks import check_qkv, require_block_aligned
from thriftattention._extension import get_extension
from thriftattention.selection.base import SelectionConfig
from thriftattention.selection.block_mean import resolve_top_k


def local_block_indices(
    flat_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    selected_count: int,
    *,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    if selected_count == 0:
        return torch.empty(flat_heads, num_q_blocks, 0, device=device, dtype=torch.int32)

    ranks = torch.arange(selected_count, device=device, dtype=torch.int32)
    q_blocks = torch.arange(num_q_blocks, device=device, dtype=torch.int32)
    if causal:
        ends = q_blocks.clamp(max=num_kv_blocks - 1)
        valid_counts = (ends + 1).clamp(max=selected_count)
        starts = (ends - selected_count + 1).clamp(min=0)
        indices = starts[:, None] + ranks[None, :]
        indices = indices.masked_fill(ranks.view(1, -1) >= valid_counts.view(-1, 1), -1)
    else:
        centers = q_blocks.clamp(max=num_kv_blocks - 1)
        max_start = max(num_kv_blocks - selected_count, 0)
        starts = (centers - selected_count // 2).clamp(min=0, max=max_start)
        indices = starts[:, None] + ranks[None, :]
    return indices.unsqueeze(0).expand(flat_heads, -1, -1).contiguous()


def local_decode_indices(
    flat_heads: int,
    num_kv_blocks: int,
    selected_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if selected_count == 0:
        return torch.empty(flat_heads, 0, device=device, dtype=torch.int32)
    row = torch.arange(
        num_kv_blocks - selected_count,
        num_kv_blocks,
        device=device,
        dtype=torch.int32,
    )
    return row.unsqueeze(0).expand(flat_heads, -1).contiguous()


def select_local_block_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
    is_bf16: bool = False,
) -> torch.Tensor:
    del is_bf16
    check_qkv(q, k, k)
    require_block_aligned("q", q.shape[2], block_size)
    require_block_aligned("k", k.shape[2], block_size)

    flat_heads = q.shape[0] * q.shape[1]
    num_q_blocks = q.shape[2] // block_size
    num_kv_blocks = k.shape[2] // block_size
    selected_count = resolve_top_k(
        num_kv_blocks,
        causal=causal,
        top_k=top_k,
        fraction=fraction,
    )
    if q.is_cuda and selected_count > 0:
        return get_extension().local_block_topk(
            q,
            num_kv_blocks,
            selected_count,
            causal,
            block_size,
        )

    return local_block_indices(
        flat_heads,
        num_q_blocks,
        num_kv_blocks,
        selected_count,
        causal=causal,
        device=q.device,
    )


def select_local_key_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    top_k: int | None = None,
    fraction: float | None = 0.05,
    block_size: int = 64,
    is_bf16: bool = False,
) -> torch.Tensor:
    del is_bf16
    check_qkv(q, k, k)
    if q.shape[2] != 1:
        raise ValueError("single-query local selection expects q sequence length 1")
    require_block_aligned("k", k.shape[2], block_size)

    num_kv_blocks = k.shape[2] // block_size
    selected_count = resolve_top_k(
        num_kv_blocks,
        causal=False,
        top_k=top_k,
        fraction=fraction,
    )
    batch = q.shape[0]
    q_heads = q.shape[1]
    head_dim = q.shape[3]
    kv_heads = k.shape[1]
    groups = q_heads // kv_heads
    flat_heads = batch * kv_heads

    if q.is_cuda and selected_count > 0:
        q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()
        return get_extension().single_query_local_topk(
            q_grouped,
            selected_count,
            num_kv_blocks,
        )

    return local_decode_indices(
        flat_heads,
        num_kv_blocks,
        selected_count,
        device=q.device,
    )


class LocalSelectionPolicy:
    name = "local"

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
            raise ValueError(f"selection config {config.name!r} does not match local policy")
        if q.shape[2] == 1:
            return select_local_key_blocks(
                q,
                k,
                top_k=config.top_k,
                fraction=config.fraction,
                block_size=config.block_size,
                is_bf16=is_bf16,
            )
        return select_local_block_pairs(
            q,
            k,
            causal=causal,
            top_k=config.top_k,
            fraction=config.fraction,
            block_size=config.block_size,
            is_bf16=is_bf16,
        )
