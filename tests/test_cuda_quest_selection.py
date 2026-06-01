from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_C = pytest.importorskip("thriftattention._C")


DTYPES = (torch.float16, torch.bfloat16)


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


def _quest_block_reference(
    q_mean: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    top_k: int,
    *,
    causal: bool,
) -> torch.Tensor:
    qf = q_mean.float().unsqueeze(3)
    scores = (qf * torch.where(qf >= 0.0, k_max.float().unsqueeze(2), k_min.float().unsqueeze(2))).sum(dim=-1)
    if causal:
        _, _, num_q_blocks, _ = q_mean.shape
        num_kv_blocks = k_min.shape[2]
        mask = torch.triu(
            torch.ones(num_q_blocks, num_kv_blocks, device=q_mean.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    indices = scores.reshape(q_mean.shape[0] * q_mean.shape[1], q_mean.shape[2], k_min.shape[2]).topk(
        top_k,
        dim=-1,
    ).indices.to(torch.int32)
    if causal:
        valid_counts = torch.arange(1, q_mean.shape[2] + 1, device=q_mean.device).clamp(max=k_min.shape[2])
        ranks = torch.arange(top_k, device=q_mean.device)
        indices.masked_fill_(ranks.view(1, 1, -1) >= valid_counts.view(1, -1, 1), -1)
    return indices.contiguous()


def _quest_decode_reference(
    q_grouped: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    top_k: int,
    num_kv_blocks: int,
) -> torch.Tensor:
    qf = q_grouped.float().unsqueeze(3)
    scores = (qf * torch.where(qf >= 0.0, k_max[:, :, :num_kv_blocks].float().unsqueeze(2), k_min[:, :, :num_kv_blocks].float().unsqueeze(2))).sum(dim=-1)
    scores = scores.amax(dim=2).reshape(q_grouped.shape[0] * q_grouped.shape[1], num_kv_blocks)
    return scores.topk(top_k, dim=-1).indices.to(torch.int32).contiguous()


@pytest.mark.parametrize("dtype", DTYPES)
def test_quest_block_topk_matches_torch_reference(dtype: torch.dtype) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(11)
    device = torch.device("cuda")
    q_mean = torch.randn(1, 2, 4, 64, device=device, dtype=dtype).contiguous()
    k_tokens = torch.randn(1, 2, 5, 3, 64, device=device, dtype=dtype)
    k_min = k_tokens.amin(dim=3).contiguous()
    k_max = k_tokens.amax(dim=3).contiguous()

    actual = _C.quest_block_topk(q_mean, k_min, k_max, 2, True, dtype == torch.bfloat16)
    expected = _quest_block_reference(q_mean, k_min, k_max, 2, causal=True)

    torch.cuda.synchronize()
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("dtype", DTYPES)
def test_single_query_quest_topk_matches_torch_reference(dtype: torch.dtype) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(17)
    device = torch.device("cuda")
    q_grouped = torch.randn(1, 2, 3, 64, device=device, dtype=dtype).contiguous()
    k_tokens = torch.randn(1, 2, 7, 4, 64, device=device, dtype=dtype)
    k_min = k_tokens.amin(dim=3).contiguous()
    k_max = k_tokens.amax(dim=3).contiguous()

    actual = _C.single_query_quest_topk(q_grouped, k_min, k_max, 3, 7, dtype == torch.bfloat16)
    expected = _quest_decode_reference(q_grouped, k_min, k_max, 3, 7)

    torch.cuda.synchronize()
    assert actual.tolist() == expected.tolist()
