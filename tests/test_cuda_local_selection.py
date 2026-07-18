from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
extension = pytest.importorskip("thriftattention._C")


def expected_local_block_topk(
    flat_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    topk_count: int,
    *,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for q_block_id in range(num_q_blocks):
        if causal:
            end = min(q_block_id, num_kv_blocks - 1)
            valid_count = min(end + 1, topk_count)
            start = max(0, end - topk_count + 1)
            row = [start + rank if rank < valid_count else -1 for rank in range(topk_count)]
        else:
            center = min(q_block_id, num_kv_blocks - 1)
            max_start = max(num_kv_blocks - topk_count, 0)
            start = min(max(center - topk_count // 2, 0), max_start)
            row = [start + rank for rank in range(topk_count)]
        rows.append(row)
    return torch.tensor(rows, device=device, dtype=torch.int32).unsqueeze(0).expand(
        flat_heads,
        -1,
        -1,
    ).contiguous()


def expected_single_query_local_topk(
    flat_heads: int,
    num_kv_blocks: int,
    topk_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    row = list(range(num_kv_blocks - topk_count, num_kv_blocks))
    return torch.tensor(row, device=device, dtype=torch.int32).unsqueeze(0).expand(
        flat_heads,
        -1,
    ).contiguous()


def require_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


@pytest.mark.parametrize(
    ("batch", "heads", "num_q_blocks", "num_kv_blocks", "topk_count", "causal", "head_dim"),
    [
        (1, 2, 5, 6, 3, True, 64),
        (2, 3, 5, 3, 3, True, 64),
        (1, 4, 7, 9, 1, True, 128),
        (1, 1, 2, 4, 0, True, 64),
        (1, 2, 5, 6, 3, False, 64),
        (2, 1, 4, 4, 4, False, 128),
    ],
)
def test_local_block_topk_matches_reference(
    batch: int,
    heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    topk_count: int,
    causal: bool,
    head_dim: int,
) -> None:
    require_sm120_cuda()
    device = torch.device("cuda")
    q = torch.empty(batch, heads, num_q_blocks * 64, head_dim, device=device, dtype=torch.float16)

    actual = extension.local_block_topk(q, num_kv_blocks, topk_count, causal)
    expected = expected_local_block_topk(
        batch * heads,
        num_q_blocks,
        num_kv_blocks,
        topk_count,
        causal=causal,
        device=device,
    )

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("batch", "kv_heads", "groups", "num_kv_blocks", "topk_count", "head_dim"),
    [
        (1, 2, 3, 7, 3, 64),
        (2, 3, 4, 5, 5, 128),
        (1, 1, 1, 1, 1, 64),
        (1, 2, 3, 4, 0, 64),
    ],
)
def test_single_query_local_topk_matches_reference(
    batch: int,
    kv_heads: int,
    groups: int,
    num_kv_blocks: int,
    topk_count: int,
    head_dim: int,
) -> None:
    require_sm120_cuda()
    device = torch.device("cuda")
    q_grouped = torch.empty(batch, kv_heads, groups, head_dim, device=device, dtype=torch.float16)

    actual = extension.single_query_local_topk(q_grouped, topk_count, num_kv_blocks)
    expected = expected_single_query_local_topk(
        batch * kv_heads,
        num_kv_blocks,
        topk_count,
        device=device,
    )

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
