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


def _requires_exp_approx_binding(fn) -> None:
    doc = getattr(fn, "__doc__", "") or ""
    if "exp_approx" not in doc:
        pytest.skip("compiled thriftattention._C extension must be rebuilt for exp_approx bindings")


def _quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    quant_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if quant_format == "mxfp4":
        q_packed, q_scale = _C.mxfp4_quantize(q, False)
        k_packed, k_scale = _C.mxfp4_quantize_permuted(k, False)
        v_packed_t, v_scale_t = _C.mxfp4_quantize_transposed(v, False)
    else:
        q_packed, q_scale = _C.nvfp4_quantize(q, False)
        k_packed, k_scale = _C.nvfp4_quantize_permuted(k, False)
        v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(v, False)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


@pytest.mark.parametrize("quant_format", ("nvfp4", "mxfp4"))
def test_tiled_fp4_exp_approx_matches_sdpa(quant_format: str) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, q_heads, kv_heads, q_len, kv_len, head_dim = 1, 2, 1, 64, 4096, 64
    groups = q_heads // kv_heads

    q = (torch.randn(batch, q_heads, q_len, head_dim, device=device, dtype=torch.float16) * 0.25).contiguous()
    k = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=torch.float16) * 0.25).contiguous()
    v = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=torch.float16) * 0.25).contiguous()

    packed = _quantize_qkv(q, k, v, quant_format=quant_format)
    fn = (
        _C.fp4_attention_noncausal_mxfp4_packed
        if quant_format == "mxfp4"
        else _C.fp4_attention_noncausal_nvfp4_packed
    )
    _requires_exp_approx_binding(fn)
    fp4_out = fn(*packed, is_bf16=False, exp_approx=True)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

    torch.cuda.synchronize()
    assert fp4_out.dtype == torch.float16
    assert torch.isfinite(fp4_out).all()
    assert _cosine(fp4_out, ref) > 0.98
