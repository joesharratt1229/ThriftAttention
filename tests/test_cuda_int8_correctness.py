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


def test_sm80_int8_attention_noncausal_matches_torch():
    ext = get_extension()

    bs = 1
    q_heads = 1
    kv_heads = 1
    q_len = 64
    kv_len = 64
    head_dim = 64

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

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v_ref).transpose(1, 2).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
