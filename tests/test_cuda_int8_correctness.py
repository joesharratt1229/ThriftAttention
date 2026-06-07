import torch
import pytest

from thriftattention._extension import get_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)

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

    q = torch.randn(bs, q_heads, q_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bs, kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

    out = ext.int8_attention_noncausal(q, k, v, False)
    
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (head_dim ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probs, v.float()).half()

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)