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


def dequantize_per_32(x: torch.Tensor, scale: torch.Tensor):
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    return (grouped * scale.unsqueeze(-1)).reshape_as(x)


def test_sm80_mma_m16n8k32_s8_all_ones():
    ext = get_extension()

    a = torch.ones((16, 32), device="cuda", dtype=torch.int8)
    b = torch.ones((32, 8), device="cuda", dtype=torch.int8)

    out = ext.sm80_mma_m16n8k32_s8_test(a, b)
    expected = (a.cpu().int() @ b.cpu().int()).to(out.device)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,q_len,kv_len,head_dim,test_case", [
        (1, 1, 1, 64, 64, 64, "matching lengths"),
        (1, 1, 1, 64, 128, 64, "different lengths"),
        (1, 8, 4, 64, 64, 64, "q_heads > kv_heads"),
        (4, 1, 1, 64, 64, 64, "larger batch size"),
        (1, 1, 1, 64, 64, 128, "headd dim 128"),
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