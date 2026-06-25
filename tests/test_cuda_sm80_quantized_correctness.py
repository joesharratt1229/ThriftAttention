import math

import torch
import pytest

from thriftattention._extension import get_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def quantize_per_32(x: torch.Tensor):
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-6) / 127.0
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape_as(x).contiguous(), scale.contiguous()



def pack_signed_int4(x: torch.Tensor):
    even = x[..., 0::2].to(torch.int16) & 0xF
    odd = x[..., 1::2].to(torch.int16) & 0xF
    return (even | (odd << 4)).to(torch.uint8).contiguous()


def unpack_signed_int4(x: torch.Tensor):
    low = (x.to(torch.int16) & 0xF)
    high = ((x.to(torch.int16) >> 4) & 0xF)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(*x.shape[:-1], x.shape[-1] * 2, device=x.device, dtype=torch.int16)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out.to(torch.int8)


def quantize_int4_per_64(x: torch.Tensor):
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 64, 64)
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-6) / 7.0
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    return pack_signed_int4(q.reshape_as(x)), scale.contiguous()


def dequantize_int4_per_64(x: torch.Tensor, scale: torch.Tensor):
    unpacked = unpack_signed_int4(x).float()
    grouped = unpacked.reshape(*unpacked.shape[:-1], unpacked.shape[-1] // 64, 64)
    return (grouped * scale.unsqueeze(-1)).reshape_as(unpacked)


def dequantize_per_32(x: torch.Tensor, scale: torch.Tensor):
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    return (grouped * scale.unsqueeze(-1)).reshape_as(x)



@pytest.mark.parametrize("head_dim", [64, 128])
def test_sm80_int4_quantize_matches_python_reference(head_dim: int):
    ext = get_extension()

    x = torch.randn(2, 17, 3, head_dim, device="cuda", dtype=torch.float16)
    packed, scale = ext.sm80_int4_quantize(x.contiguous(), False)
    _, expected_scale = quantize_int4_per_64(x)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    dequantized = dequantize_int4_per_64(packed, scale)
    max_allowed_error = scale.repeat_interleave(64, dim=-1) * 0.5 + 1.0e-3
    assert torch.all((dequantized - x.float()).abs() <= max_allowed_error)


def test_sm80_mma_m16n8k32_s8_all_ones():
    ext = get_extension()

    a = torch.ones((16, 32), device="cuda", dtype=torch.int8)
    b = torch.ones((32, 8), device="cuda", dtype=torch.int8)

    out = ext.sm80_mma_m16n8k32_s8_test(a, b)
    expected = (a.cpu().int() @ b.cpu().int()).to(out.device)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)

def test_sm80_mma_m16n8k32_s8_random():
    ext = get_extension()

    a = torch.randint(-128, 128, (16, 32), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (32, 8), device="cuda", dtype=torch.int8)

    out = ext.sm80_mma_m16n8k32_s8_test(a, b)
    expected = (a.cpu().int() @ b.cpu().int()).to(out.device)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)



def test_sm80_mma_m16n8k64_s4_random():
    ext = get_extension()

    a = torch.randint(-8, 8, (16, 64), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (64, 8), device="cuda", dtype=torch.int8)

    out = ext.sm80_mma_m16n8k64_s4_test(a, b)
    expected = (a.cpu().int() @ b.cpu().int()).to(out.device)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_sm80_mma_int8_scores_matches_scalar_reference(head_dim: int):
    ext = get_extension()

    q = torch.randint(-128, 128, (16, head_dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-128, 128, (8, head_dim), device="cuda", dtype=torch.int8)
    scale_dim = head_dim // 32
    s_q = (torch.rand((16, scale_dim), device="cuda", dtype=torch.float32) * 0.05 + 0.001).contiguous()
    s_k = (torch.rand((8, scale_dim), device="cuda", dtype=torch.float32) * 0.05 + 0.001).contiguous()

    out = ext.sm80_mma_int8_scores_test(q, k, s_q, s_k)

    expected = torch.zeros((16, 8), device="cuda", dtype=torch.float32)
    for group in range(scale_dim):
        start = group * 32
        stop = start + 32
        partial = (q[:, start:stop].cpu().int() @ k[:, start:stop].cpu().int().T).to(out.device).float()
        expected += partial * s_q[:, group, None] * s_k[None, :, group]
    expected *= 1.0 / math.sqrt(head_dim)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,q_len,kv_len,head_dim,test_case", [
        (1, 1, 1, 64, 64, 64, "matching lengths"),
        (1, 1, 1, 64, 128, 64, "different lengths"),
        (1, 8, 4, 64, 64, 64, "q_heads > kv_heads"),
        (4, 1, 1, 64, 64, 64, "larger batch size"),
        (1, 1, 1, 64, 64, 128, "headd dim 128"),
        (1, 1, 1, 65, 67, 128, "indivisible length"),
    ]
)
def test_sm80_int8_attention_noncausal_matches_torch(
    bs: int,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    test_case: str,
):
    ext = get_extension()

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)

    q_i8, s_q = quantize_per_32(q)
    k_i8, s_k = quantize_per_32(k)
    v_i8, s_v = quantize_per_32(v)

    out = ext.sm80_int8_attention_noncausal(
        q_i8, k_i8, v_i8,
        s_q, s_k, s_v,
        False,
    )

    q_ref = dequantize_per_32(q_i8, s_q).transpose(1, 2)
    k_ref = dequantize_per_32(k_i8, s_k).transpose(1, 2)
    v_ref = dequantize_per_32(v_i8, s_v).transpose(1, 2)

    repeat = q_heads // kv_heads
    k_ref = k_ref.repeat_interleave(repeat, dim=1)
    v_ref = v_ref.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2), test_case


@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,q_len,kv_len,head_dim,test_case", [
        (1, 1, 1, 64, 64, 64, "matching lengths"),
        (1, 1, 1, 64, 128, 64, "different lengths"),
        (1, 8, 4, 64, 64, 64, "q_heads > kv_heads"),
        (4, 1, 1, 64, 64, 64, "larger batch size"),
        (1, 1, 1, 64, 64, 128, "headd dim 128"),
        (1, 1, 1, 256, 256, 128, "long kv_len"),
        (1, 1, 1, 65, 67, 128, "indivisible length"),
    ]
)
def test_sm80_int8_attention_causal_matches_torch(
    bs: int,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    test_case: str,
):
    ext = get_extension()

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)

    q_i8, s_q = quantize_per_32(q)
    k_i8, s_k = quantize_per_32(k)
    v_i8, s_v = quantize_per_32(v)

    out = ext.sm80_int8_attention_causal(
        q_i8, k_i8, v_i8,
        s_q, s_k, s_v,
        False,
    )

    q_ref = dequantize_per_32(q_i8, s_q).transpose(1, 2)
    k_ref = dequantize_per_32(k_i8, s_k).transpose(1, 2)
    v_ref = dequantize_per_32(v_i8, s_v).transpose(1, 2)

    repeat = q_heads // kv_heads
    k_ref = k_ref.repeat_interleave(repeat, dim=1)
    v_ref = v_ref.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones_like(scores)).bool()
    scores = torch.where(mask, torch.tril(scores), -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_ref).transpose(1, 2).half()
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2), test_case

@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,q_len,kv_len,head_dim,test_case", [
        (1, 1, 1, 64, 64, 64, "matching lengths"),
        (1, 1, 1, 64, 128, 64, "different lengths"),
        (1, 8, 4, 64, 64, 64, "q_heads > kv_heads"),
        (1, 1, 1, 65, 67, 128, "indivisible length"),
    ]
)
def test_sm80_int4_attention_noncausal_matches_torch(
    bs: int,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    test_case: str,
):
    ext = get_extension()

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)

    q_i4, s_q = ext.sm80_int4_quantize(q.contiguous(), False)
    k_i4, s_k = ext.sm80_int4_quantize(k.contiguous(), False)
    v_i4, s_v = ext.sm80_int4_quantize(v.contiguous(), False)

    out = ext.sm80_int4_attention_noncausal(
        q_i4, k_i4, v_i4,
        s_q, s_k, s_v,
        False,
    )

    q_ref = dequantize_int4_per_64(q_i4, s_q).transpose(1, 2)
    k_ref = dequantize_int4_per_64(k_i4, s_k).transpose(1, 2)
    v_ref = dequantize_int4_per_64(v_i4, s_v).transpose(1, 2)

    repeat = q_heads // kv_heads
    k_ref = k_ref.repeat_interleave(repeat, dim=1)
    v_ref = v_ref.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2), test_case


@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,q_len,kv_len,head_dim,test_case", [
        (1, 1, 1, 64, 64, 64, "matching lengths"),
        (1, 1, 1, 64, 128, 64, "different lengths"),
        (1, 8, 4, 64, 64, 64, "q_heads > kv_heads"),
        (1, 1, 1, 65, 67, 128, "indivisible length"),
    ]
)
def test_sm80_int4_attention_causal_matches_torch(
    bs: int,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    test_case: str,
):
    ext = get_extension()

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)

    q_i4, s_q = ext.sm80_int4_quantize(q.contiguous(), False)
    k_i4, s_k = ext.sm80_int4_quantize(k.contiguous(), False)
    v_i4, s_v = ext.sm80_int4_quantize(v.contiguous(), False)

    out = ext.sm80_int4_attention_causal(
        q_i4, k_i4, v_i4,
        s_q, s_k, s_v,
        False,
    )

    q_ref = dequantize_int4_per_64(q_i4, s_q).transpose(1, 2)
    k_ref = dequantize_int4_per_64(k_i4, s_k).transpose(1, 2)
    v_ref = dequantize_int4_per_64(v_i4, s_v).transpose(1, 2)

    repeat = q_heads // kv_heads
    k_ref = k_ref.repeat_interleave(repeat, dim=1)
    v_ref = v_ref.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones_like(scores)).bool()
    scores = torch.where(mask, torch.tril(scores), -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2), test_case



def test_sm80_int8_attention_noncausal_supports_kv_capacity_stride():
    ext = get_extension()
    bs, q_len, kv_len, kv_capacity = 2, 64, 67, 96
    q_heads, kv_heads, head_dim = 2, 1, 64

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(bs, kv_capacity, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    k = k_cache[:, :kv_len]
    v = v_cache[:, :kv_len]

    q_i8, s_q = quantize_per_32(q)
    k_i8_cache, s_k_cache = quantize_per_32(k_cache)
    v_i8_cache, s_v_cache = quantize_per_32(v_cache)
    k_i8 = k_i8_cache[:, :kv_len]
    v_i8 = v_i8_cache[:, :kv_len]
    s_k = s_k_cache[:, :kv_len]
    s_v = s_v_cache[:, :kv_len]

    out = ext.sm80_int8_attention_noncausal(q_i8, k_i8, v_i8, s_q, s_k, s_v, False)

    q_ref = dequantize_per_32(q_i8, s_q).transpose(1, 2)
    k_ref = dequantize_per_32(k_i8, s_k).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    v_ref = dequantize_per_32(v_i8, s_v).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_sm80_int8_attention_noncausal_bf16_output_matches_torch():
    ext = get_extension()
    bs, q_len, kv_len, q_heads, kv_heads, head_dim = 1, 64, 64, 2, 1, 64

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    q_i8, s_q = quantize_per_32(q)
    k_i8, s_k = quantize_per_32(k)
    v_i8, s_v = quantize_per_32(v)

    out = ext.sm80_int8_attention_noncausal(q_i8, k_i8, v_i8, s_q, s_k, s_v, True)

    q_ref = dequantize_per_32(q_i8, s_q).transpose(1, 2)
    k_ref = dequantize_per_32(k_i8, s_k).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    v_ref = dequantize_per_32(v_i8, s_v).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_ref).transpose(1, 2).bfloat16()

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_sm80_int4_attention_noncausal_bf16_output_matches_torch():
    ext = get_extension()
    bs, q_len, kv_len, q_heads, kv_heads, head_dim = 1, 64, 64, 2, 1, 64

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    q_i4, s_q = ext.sm80_int4_quantize(q.contiguous(), True)
    k_i4, s_k = ext.sm80_int4_quantize(k.contiguous(), True)
    v_i4, s_v = ext.sm80_int4_quantize(v.contiguous(), True)

    out = ext.sm80_int4_attention_noncausal(q_i4, k_i4, v_i4, s_q, s_k, s_v, True)

    q_ref = dequantize_int4_per_64(q_i4, s_q).transpose(1, 2)
    k_ref = dequantize_int4_per_64(k_i4, s_k).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    v_ref = dequantize_int4_per_64(v_i4, s_v).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_ref).transpose(1, 2).bfloat16()

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_sm80_int4_quantize_bf16_matches_error_bound():
    ext = get_extension()
    x = torch.randn(2, 17, 3, 128, device="cuda", dtype=torch.bfloat16)
    packed, scale = ext.sm80_int4_quantize(x.contiguous(), True)
    dequantized = dequantize_int4_per_64(packed, scale)
    max_allowed_error = scale.repeat_interleave(64, dim=-1) * 0.5 + 1.0e-3

    assert torch.all((dequantized - x.float()).abs() <= max_allowed_error)


def test_sm80_int4_attention_noncausal_supports_kv_capacity_stride():
    ext = get_extension()
    bs, q_len, kv_len, kv_capacity = 2, 64, 67, 96
    q_heads, kv_heads, head_dim = 2, 1, 64

    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(bs, kv_capacity, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)

    q_i4, s_q = ext.sm80_int4_quantize(q.contiguous(), False)
    k_i4_cache, s_k_cache = ext.sm80_int4_quantize(k_cache.contiguous(), False)
    v_i4_cache, s_v_cache = ext.sm80_int4_quantize(v_cache.contiguous(), False)
    k_i4 = k_i4_cache[:, :kv_len]
    v_i4 = v_i4_cache[:, :kv_len]
    s_k = s_k_cache[:, :kv_len]
    s_v = s_v_cache[:, :kv_len]

    out = ext.sm80_int4_attention_noncausal(q_i4, k_i4, v_i4, s_q, s_k, s_v, False)

    q_ref = dequantize_int4_per_64(q_i4, s_q).transpose(1, 2)
    k_ref = dequantize_int4_per_64(k_i4, s_k).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    v_ref = dequantize_int4_per_64(v_i4, s_v).transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
