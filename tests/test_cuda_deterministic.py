from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_C = pytest.importorskip("thriftattention._C")


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


@pytest.mark.parametrize("fmt", ("nvfp4", "mxfp4"))
@pytest.mark.parametrize("head_dim", (128, 256))
@pytest.mark.parametrize("causal", (False, True))
def test_tiled_kernels_give_the_same_bits_on_every_run(fmt: str, head_dim: int, causal: bool) -> None:
    """A warp that copies the next K/V tile into shared memory while other warps still read the current one makes
    the output change from run to run; every run of the same input must give the same bits."""
    _requires_sm120_cuda()
    torch.manual_seed(0)
    q = torch.randn(4, 8, 4096, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 2, 4096, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(4, 2, 4096, head_dim, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scale = getattr(_C, f"{fmt}_quantize")(q, True)
    k_packed, k_scale = getattr(_C, f"{fmt}_quantize_permuted")(k, True)
    v_packed_t, v_scale_t = getattr(_C, f"{fmt}_quantize_transposed")(v, True)
    attention = getattr(_C, f"fp4_attention_{'causal' if causal else 'noncausal'}_{fmt}_packed")
    args = (q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, True)
    first = attention(*args)
    for _ in range(10):
        assert torch.equal(attention(*args).view(torch.int16), first.view(torch.int16))
