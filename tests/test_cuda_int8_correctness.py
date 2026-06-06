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
    expected = a.int() @ b.int()

    torch.testing.assert_close(out, expected, rtol=0, atol=0)