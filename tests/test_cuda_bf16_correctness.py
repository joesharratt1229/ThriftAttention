from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")


CONTEXT_LENGTHS = (4096, 8192, 32768, 131072)
DTYPES = (torch.float16, torch.bfloat16)


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_bf16: bool,
    permute_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_packed, q_scale = _C.nvfp4_quantize(q, is_bf16)
    if permute_k:
        k_packed, k_scale = _C.nvfp4_quantize_permuted(k, is_bf16)
    else:
        k_packed, k_scale = _C.nvfp4_quantize(k, is_bf16)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(v, is_bf16)
    return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
def test_single_query_attention_cosine_4k_to_131k(dtype: torch.dtype, kv_len: int) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")
    is_bf16 = dtype == torch.bfloat16
    batch, q_heads, kv_heads, head_dim = 1, 4, 2, 64
    groups = q_heads // kv_heads

    q = (torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=dtype) * 0.25).contiguous()
    k = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype) * 0.25).contiguous()
    v = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype) * 0.25).contiguous()
    q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()

    packed = _quantize_qkv(q_grouped, k, v, is_bf16=is_bf16, permute_k=False)
    fp4_out = _C.fp4_attention_single_query_nvfp4_packed(*packed, is_bf16)
    fp4_out = fp4_out.reshape(batch, q_heads, 1, head_dim)

    num_kv_blocks = kv_len // 64
    k_mean = k.reshape(batch, kv_heads, num_kv_blocks, 64, head_dim).float().mean(dim=3).to(dtype)
    selected = _C.single_query_key_mean_topk(
        q_grouped,
        k_mean.contiguous(),
        num_kv_blocks,
        num_kv_blocks,
        is_bf16,
    )
    thrift_out = _C.thrift_attention_single_query_nvfp4_packed(
        q_grouped,
        k,
        v,
        selected,
        *packed,
        is_bf16,
    ).reshape(batch, q_heads, 1, head_dim)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

    torch.cuda.synchronize()
    assert _cosine(fp4_out, ref) > 0.95
    assert _cosine(thrift_out, ref) > 0.95


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
def test_tiled_attention_cosine(dtype: torch.dtype, kv_len: int) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(1)
    device = torch.device("cuda")
    is_bf16 = dtype == torch.bfloat16
    q_len = kv_len
    batch, q_heads, kv_heads, head_dim = 1, 2, 1, 64
    groups = q_heads // kv_heads

    q = (torch.randn(batch, q_heads, q_len, head_dim, device=device, dtype=dtype) * 0.25).contiguous()
    k = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype) * 0.25).contiguous()
    v = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype) * 0.25).contiguous()

    packed = _quantize_qkv(q, k, v, is_bf16=is_bf16, permute_k=True)
    fp4_out = _C.fp4_attention_causal_nvfp4_packed(*packed, is_bf16)

    num_q_blocks = q_len // 64
    num_kv_blocks = kv_len // 64
    selected = (
        torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
        .view(1, 1, num_kv_blocks)
        .expand(batch * q_heads, num_q_blocks, num_kv_blocks)
        .contiguous()
    )
    thrift_out = _C.thrift_attention_causal_nvfp4_packed(q, k, v, selected, *packed, is_bf16)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=True)

    torch.cuda.synchronize()
    assert _cosine(fp4_out, ref) > 0.95
    assert _cosine(thrift_out, ref) > 0.95


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_mean_topk_accepts_high_precision_dtype(dtype: torch.dtype) -> None:
    _requires_sm120_cuda()
    device = torch.device("cuda")
    is_bf16 = dtype == torch.bfloat16
    q_mean = torch.zeros(1, 1, 1, 64, device=device, dtype=dtype)
    k_mean = torch.zeros(1, 1, 8, 64, device=device, dtype=dtype)
    q_mean[:, :, :, 0] = 1
    k_mean[0, 0, :, 0] = torch.arange(8, device=device, dtype=dtype)

    selected = _C.block_mean_topk(q_mean.contiguous(), k_mean.contiguous(), 3, False, is_bf16)
    torch.cuda.synchronize()

    expected = torch.tensor([[[7, 6, 5]]], device=device, dtype=torch.int32)
    assert torch.equal(selected, expected)
