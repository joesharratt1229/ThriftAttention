from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@pytest.mark.parametrize("fmt", ("nvfp4", "mxfp4"))
@pytest.mark.parametrize("causal", (False, True))
def test_hd256_thrift_with_every_block_selected_matches_sdpa(fmt: str, causal: bool) -> None:
    """Selecting every 128-token KV block leaves nothing to the FP4 pass, so the output is the 16-bit attention."""
    _requires_sm120_cuda()
    torch.manual_seed(0)
    seq_len, head_dim = 4096, 256
    q = torch.randn(1, 2, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scale = getattr(_C, f"{fmt}_quantize")(q, True)
    k_packed, k_scale = getattr(_C, f"{fmt}_quantize_permuted")(k, True)
    v_packed_t, v_scale_t = getattr(_C, f"{fmt}_quantize_transposed")(v, True)
    blocks = seq_len // 128
    selected = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, 1, blocks).expand(2, blocks, blocks)
    kernel = getattr(_C, f"thrift_attention_{'causal' if causal else 'noncausal'}_{fmt}_packed")
    out = kernel(q, k, v, selected.contiguous(), q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, True)
    ref = F.scaled_dot_product_attention(q.float(), k.repeat_interleave(2, dim=1).float(),
                                         v.repeat_interleave(2, dim=1).float(), is_causal=causal)
    torch.cuda.synchronize()
    # read in 64-token units, the FP4 pass added half of the selected blocks a second time (cosine 0.95 non-causal)
    assert _cosine(out, ref) > 0.999
