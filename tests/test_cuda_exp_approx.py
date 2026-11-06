"""The exp_approx option of the tiled NVFP4 kernels (power-of-two softmax weights, P @ ones row sums)."""
from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")

from thriftattention.quant.formats import get_quant_format  # noqa: E402
from thriftattention.selection.block_mean import select_block_pairs  # noqa: E402


DTYPES = (torch.float16, torch.bfloat16)
HEAD_DIMS = (64, 128)
LN_2 = 0.6931471805599453
LOG2_E = 1.4426950408889634


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _relative_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _sdpa(q, k, v, causal):
    groups = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(
        q.float(), k.repeat_interleave(groups, 1).float(), v.repeat_interleave(groups, 1).float(), is_causal=causal
    )


def _dense(packed, is_bf16, causal, exp_approx):
    fn = _C.fp4_attention_causal_nvfp4_packed if causal else _C.fp4_attention_noncausal_nvfp4_packed
    return fn(*packed, is_bf16, exp_approx)


def _thrift(q, k, v, selected, packed, is_bf16, causal, exp_approx):
    fn = _C.thrift_attention_causal_nvfp4_packed if causal else _C.thrift_attention_noncausal_nvfp4_packed
    return fn(q, k, v, selected, *packed, is_bf16, exp_approx)


def _random_qkv(dtype, head_dim, seq_len=2048, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, 4, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, 2, seq_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------------------------------------------
# A reference in logical key order that quantizes P exactly as the kernels do, for both exp paths.
# ---------------------------------------------------------------------------------------------------------------

def _dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
    return levels[codes & 7] * torch.where(codes < 8, 1, -1) * scales.float().repeat_interleave(16, -1)


def _fp4_round(x: torch.Tensor) -> torch.Tensor:
    # At a midpoint, choose the even E2M1 code, as cvt.rn does.
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    candidates = levels[torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)]
    return candidates[(x.unsqueeze(-1) - candidates).abs().argmin(-1)]


def _reference(q, k, v, packed, selected, causal, exp_approx):
    """Thrift attention with the FP4 pass over the blocks not in `selected` and the FP16 pass over the rest."""
    q4, k4, vt4 = (_dequantize(packed[i], packed[i + 3]) for i in range(3))
    n = k.shape[2]
    physical = torch.arange(n, device=q.device)
    x = physical % 32
    logical = physical // 32 * 32 + x // 8 * 2 + (x % 8) // 2 * 8 + x % 2
    k4 = k4[:, :, logical.argsort()]
    v4 = vt4.transpose(-2, -1)[:, :, :n]
    groups = q.shape[1] // k.shape[1]
    k4, v4, k, v = (t.repeat_interleave(groups, 1).float() for t in (k4, v4, k, v))
    result = torch.empty_like(q)
    for qi in range(q.shape[2] // 64):
        query = slice(qi * 64, (qi + 1) * 64)
        shape = (*q.shape[:2], 64, 1)
        m = torch.full(shape, -126.0 if exp_approx else torch.finfo(torch.float32).min, device=q.device)
        ell = torch.zeros_like(m)
        out = torch.zeros((*shape[:-1], q.shape[-1]), device=q.device)
        valid_blocks = list(range(qi + 1 if causal else n // 64))
        low_blocks = [j for j in valid_blocks if j not in selected]
        high_blocks = [j for j in selected if j in valid_blocks]
        for high, blocks in ((False, low_blocks), (True, high_blocks)):
            approx = exp_approx and not high
            for j in blocks:
                keys = slice(j * 64, (j + 1) * 64)
                scores = ((q[:, :, query].float() if high else q4[:, :, query])
                          @ (k[:, :, keys] if high else k4[:, :, keys]).transpose(-2, -1))
                scores *= q.shape[-1] ** -.5 * (LOG2_E if approx else 1.0)
                if causal:
                    qpos = torch.arange(qi * 64, (qi + 1) * 64, device=q.device)
                    kpos = torch.arange(j * 64, (j + 1) * 64, device=q.device)
                    scores.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
                tile_max = scores.amax(-1, keepdim=True)
                if approx:
                    tile_max = (tile_max + .5).round()
                new_m = torch.maximum(m, tile_max)
                rescale = (m - new_m).exp2() if approx else (m - new_m).exp()
                out *= rescale
                ell *= rescale
                m = new_m
                p = (scores - m).exp()
                if not approx:
                    ell += p.sum(-1, keepdim=True)
                if high:
                    weights = (2688 * p).to(q.dtype).float()
                elif approx:
                    blocks_s = scores.unflatten(-1, (4, 16))
                    b = (blocks_s.amax(-1, keepdim=True) + .5).round().clamp_min(-126)
                    sf = (448 * (b - m.unsqueeze(-1)).exp2()).to(torch.float8_e4m3fn).float()
                    weights = (_fp4_round(4 * (blocks_s - b).round().exp2()) * sf).flatten(-2)
                    ell += weights.sum(-1, keepdim=True) / 1792
                else:
                    blocks_s = scores.unflatten(-1, (4, 16))
                    b = blocks_s.amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).min)
                    sf = (448 * (b - m.unsqueeze(-1)).exp()).to(torch.float8_e4m3fn).float()
                    weights = (_fp4_round(6 * (blocks_s - b).exp()) * sf).flatten(-2)
                out += weights @ (v[:, :, keys] if high else v4[:, :, keys])
            if not high:
                # The FP4 pass hands the FP16 pass a normalized FP16/BF16 partial, a natural-log row maximum,
                # and an unscaled denominator.
                partial = (out / ((1792 if exp_approx else 2688) * ell.clamp_min(1e-30))).to(q.dtype).float()
                if exp_approx:
                    m = torch.where(ell > 0, m * LN_2, torch.finfo(torch.float32).min)
                out = partial * (2688 * ell)
        result[:, :, query] = out / (2688 * ell)
    return result


def _structured_qkv(dtype, head_dim):
    """Exactly representable scores with distinct group maxima, weak tails, and a rising row maximum."""
    torch.manual_seed(41)
    q = torch.zeros((2, 2, 192, head_dim), dtype=dtype, device="cuda")
    q_values = torch.tensor([0, .5, 1, 2, 4, 6, -2], device="cuda", dtype=dtype)
    q[..., :16] = q_values[torch.arange(192, device="cuda") % 7][None, None, :, None]
    k = torch.zeros((2, 1, 192, head_dim), dtype=dtype, device="cuda")
    k_values = torch.tensor([1.5, .5, 1, 2, .5, 3, 1, 4, 1, 6, 2, .5], device="cuda", dtype=dtype)
    k[..., :16] = (k_values.repeat_interleave(16) * .125)[None, None, :, None]
    k[..., 16:32] = .75
    v = torch.randn_like(k)
    return q, k, v


@pytest.mark.parametrize("exp_approx", (False, True))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tiled_kernels_match_the_quantized_reference(dtype, head_dim, causal, exp_approx):
    _requires_sm120_cuda()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        is_bf16 = dtype == torch.bfloat16
        q, k, v = _structured_qkv(dtype, head_dim)
        packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
        tolerance = dict(atol=.002, rtol=.01) if is_bf16 else dict(atol=.0003, rtol=.002)
        for chosen in ([], [0], [1], [2], [0, 1, 2]):
            # A -1 sentinel keeps the FP4 pass in the thrift kernel; an empty selection dispatches the dense kernel.
            selected = torch.tensor(chosen or [-1], device="cuda", dtype=torch.int32)
            selected = selected.view(1, 1, -1).expand(4, 3, -1).contiguous()
            actual = _thrift(q, k, v, selected, packed, is_bf16, causal, exp_approx)
            expected = _reference(q, k, v, packed, chosen, causal, exp_approx)
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual.float(), expected.float(), **tolerance)
            if not chosen:
                dense = _dense(packed, is_bf16, causal, exp_approx)
                torch.testing.assert_close(dense.float(), expected.float(), **tolerance)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp_approx_tracks_sdpa_on_random_inputs(dtype, head_dim, causal):
    _requires_sm120_cuda()
    is_bf16 = dtype == torch.bfloat16
    q, k, v = _random_qkv(dtype, head_dim)
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
    reference = _sdpa(q, k, v, causal)
    exact = _dense(packed, is_bf16, causal, False)
    approx = _dense(packed, is_bf16, causal, True)
    selected = select_block_pairs(q, k, causal=causal, fraction=0.1, is_bf16=is_bf16)
    thrift_exact = _thrift(q, k, v, selected, packed, is_bf16, causal, False)
    thrift_approx = _thrift(q, k, v, selected, packed, is_bf16, causal, True)
    torch.cuda.synchronize()

    for out in (approx, thrift_approx):
        assert torch.isfinite(out).all().item()
    # Measured on an RTX PRO 6000 (non-causal is the lower of the two): exp_approx 0.953-0.964 against SDPA
    # and 0.964-0.972 against the exact path; thrift exp_approx 0.958-0.986 and 0.968-0.988. The exact path
    # sits at about 0.985 (dense) and 0.995 (thrift) against SDPA.
    assert _cosine(approx, reference) > 0.93
    assert _cosine(approx, exact) > 0.94
    assert _cosine(thrift_approx, reference) > 0.94
    assert _cosine(thrift_approx, thrift_exact) > 0.95


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp_approx_keeps_probabilities_far_below_the_row_max(dtype, causal):
    """Key 0 takes most of every row, as an attention sink does, and V[0] is zero: the output comes only from
    keys whose probabilities are far below the row maximum, so the per-group scales must carry them."""
    _requires_sm120_cuda()
    torch.manual_seed(0)
    is_bf16 = dtype == torch.bfloat16
    shared = torch.ones(128, device="cuda")
    q = 0.5 * torch.randn(1, 2, 256, 128, device="cuda") + shared
    k = 0.5 * torch.randn(1, 1, 256, 128, device="cuda")
    v = torch.randn(1, 1, 256, 128, device="cuda") + 1
    k[:, :, 0] = 0.75 * shared
    v[:, :, 0] = 0
    q, k, v = (x.to(dtype).contiguous() for x in (q, k, v))
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=is_bf16)
    out = _dense(packed, is_bf16, causal, True)
    ref = _sdpa(q, k, v, causal)
    torch.cuda.synchronize()
    rows = slice(64, None) if causal else slice(None)
    assert _relative_l2(out[:, :, rows], ref[:, :, rows]) < 0.3


def test_exp_approx_is_deterministic():
    _requires_sm120_cuda()
    q, k, v = _random_qkv(torch.bfloat16, 128, seq_len=4096)
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=True)
    first = _dense(packed, True, True, True)
    for _ in range(5):
        assert torch.equal(_dense(packed, True, True, True).view(torch.int16), first.view(torch.int16))


def test_thrift_exp_approx_matches_exact_when_every_block_is_selected():
    _requires_sm120_cuda()
    q, k, v = _random_qkv(torch.float16, 128, seq_len=1024)
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=False)
    blocks = q.shape[2] // 64
    selected = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, 1, -1).expand(4, blocks, -1).contiguous()
    exact = _thrift(q, k, v, selected, packed, False, True, False)
    approx = _thrift(q, k, v, selected, packed, False, True, True)
    assert torch.equal(exact.view(torch.int16), approx.view(torch.int16))


def test_exp_approx_rejects_head_dim_256():
    _requires_sm120_cuda()
    q, k, v = _random_qkv(torch.float16, 256, seq_len=256)
    packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=False)
    with pytest.raises(RuntimeError, match="head_dim 256"):
        _dense(packed, False, True, True)
    selected = torch.zeros(4, 2, 1, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="head_dim 256"):
        _thrift(q, k, v, selected, packed, False, True, True)
