from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
_C = pytest.importorskip("thriftattention._C")

from test_cuda_varlen import LENGTHS, _pack_padded, _padded, _random_sequences, _requires_sm120_cuda  # noqa: E402
from test_cuda_varlen_fp4_out import E2M1_VALUES  # noqa: E402

# per sequence of LENGTHS (1, 37, 64, 100, 150, 257), the first query row that is computed; the keys before it are
# those a prefix cache holds: none, all but the last, block ends and starts
Q_START = (0, 36, 63, 65, 128, 200)


def _starts() -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = [n - s for n, s in zip(LENGTHS, Q_START)]
    q_start = torch.tensor(Q_START, dtype=torch.int32, device="cuda")
    out_start = torch.tensor([0, *itertools.accumulate(rows)][:-1], dtype=torch.int32, device="cuda")
    return q_start, out_start, sum(rows)


def _full_args(qs: list[torch.Tensor], ks: list[torch.Tensor], vs: list[torch.Tensor]) -> tuple:
    """The packed inputs and lengths of fp4_attention_varlen_nvfp4_packed, without causal and is_bf16."""
    is_bf16 = qs[0].dtype == torch.bfloat16
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], dtype=torch.int32, device="cuda")
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, LENGTHS))], dtype=torch.int32, device="cuda")
    q_packed, q_scale = _C.nvfp4_quantize(_pack_padded(qs), is_bf16)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(_pack_padded(ks), is_bf16)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), is_bf16)
    return (q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu, cu_padded, max(LENGTHS), sum(LENGTHS))


def _cached_args(full_args: tuple, causal: bool, is_bf16: bool) -> tuple:
    """The positional arguments of fp4_attention_varlen_nvfp4_cached (max_seqlen: the most rows a sequence
    computes)."""
    q_start, out_start, out_tokens = _starts()
    max_rows = max(n - first for n, first in zip(LENGTHS, Q_START))
    return (*full_args[:-2], max_rows, out_tokens, causal, is_bf16, None, q_start, out_start)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_cached_rows_are_those_of_the_full_kernel(dtype: torch.dtype, head_dim: int, causal: bool) -> None:
    _requires_sm120_cuda()
    is_bf16 = dtype == torch.bfloat16
    args = _full_args(*_random_sequences(dtype, head_dim))
    full = _C.fp4_attention_varlen_nvfp4_packed(*args, causal, is_bf16, None)
    (cached,) = _C.fp4_attention_varlen_nvfp4_cached(*_cached_args(args, causal, is_bf16))
    torch.cuda.synchronize()

    starts = itertools.accumulate(LENGTHS, initial=0)
    expected = torch.cat([full[start + first:start + n] for start, n, first in zip(starts, LENGTHS, Q_START)])
    assert torch.equal(cached, expected)


def _exact_sequences(heads: int, head_dim: int) -> list[torch.Tensor]:
    """Random E2M1 values times 1/8 with a +-6/8 in every 16, which NVFP4 holds exactly (each block scale is 1/8)."""
    grid = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device="cuda") / 8
    sequences = []
    for n in LENGTHS:
        x = grid[torch.randint(len(grid), (heads, n, head_dim), device="cuda")]
        x[..., ::16] = 6 / 8
        sequences.append(x.to(torch.bfloat16))
    return sequences


@pytest.mark.parametrize("causal", (False, True))
def test_lse_is_the_log_sum_exp_of_the_scaled_logits(causal: bool) -> None:
    _requires_sm120_cuda()
    torch.manual_seed(0)
    head_dim = 128
    qs, ks = _exact_sequences(8, head_dim), _exact_sequences(2, head_dim)
    vs = [torch.randn(2, n, head_dim, device="cuda").to(torch.bfloat16) for n in LENGTHS]
    _, lse = _C.fp4_attention_varlen_nvfp4_cached(*_cached_args(_full_args(qs, ks, vs), causal, True), return_lse=True)
    torch.cuda.synchronize()

    expected = []
    for q, k, first in zip(qs, ks, Q_START):
        logits = q.double() @ k.double().repeat_interleave(4, dim=0).transpose(1, 2) * head_dim ** -0.5
        if causal:
            n = q.shape[1]
            logits = logits.masked_fill(torch.ones(n, n, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        expected.append(logits[:, first:].logsumexp(dim=-1))
    assert lse.shape == (8, _starts()[2])
    assert torch.allclose(lse.double(), torch.cat(expected, dim=1), atol=1e-4, rtol=0)


def test_given_buffers_are_written_in_place() -> None:
    _requires_sm120_cuda()
    cached_args = _cached_args(_full_args(*_random_sequences(torch.bfloat16, 128)), True, True)
    out, lse = _C.fp4_attention_varlen_nvfp4_cached(*cached_args, return_lse=True)
    out_buffer, lse_buffer = torch.full_like(out, float("nan")), torch.full_like(lse, float("nan"))
    returned = _C.fp4_attention_varlen_nvfp4_cached(*cached_args, out=out_buffer, lse=lse_buffer)
    torch.cuda.synchronize()
    assert [t.data_ptr() for t in returned] == [out_buffer.data_ptr(), lse_buffer.data_ptr()]
    assert torch.equal(out_buffer, out) and torch.equal(lse_buffer, lse)

    global_scale = torch.tensor([448 * 6 / out.float().abs().max().item()], device="cuda")
    values, scales = _C.fp4_attention_varlen_nvfp4_cached(*cached_args, out_global_scale=global_scale)
    value_buffer = torch.zeros_like(values)
    scale_buffer = torch.full(scales.shape, 0xFF, dtype=torch.uint8, device="cuda")  # the scale bytes, as uint8
    returned = _C.fp4_attention_varlen_nvfp4_cached(
        *cached_args, out_global_scale=global_scale, out=value_buffer, out_scales=scale_buffer)
    torch.cuda.synchronize()
    assert [t.data_ptr() for t in returned] == [value_buffer.data_ptr(), scale_buffer.data_ptr()]
    assert torch.equal(value_buffer, values)
    assert torch.equal(scale_buffer, scales.view(torch.uint8))  # padded scale rows are zeroed in the buffer too


def test_a_buffer_the_call_would_not_write_is_rejected() -> None:
    _requires_sm120_cuda()
    cached_args = _cached_args(_full_args(*_random_sequences(torch.bfloat16, 128)), True, True)
    with pytest.raises(RuntimeError, match="out_scales"):
        _C.fp4_attention_varlen_nvfp4_cached(*cached_args, out_scales=torch.zeros(128, 32, dtype=torch.uint8,
                                                                                   device="cuda"))
