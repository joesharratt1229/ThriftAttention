from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")


LENGTHS = (1, 37, 64, 100, 150, 257)
DTYPES = (torch.float16, torch.bfloat16)


def _requires_sm120_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")


def _relative_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _padded(length: int) -> int:
    return -(-length // 64) * 64


def _pack_padded(sequences: list[torch.Tensor]) -> torch.Tensor:
    """[heads, length, dim] per sequence -> [1, heads, padded tokens, dim], each sequence at a multiple of 64."""
    heads, dim = sequences[0].shape[0], sequences[0].shape[2]
    total = sum(_padded(s.shape[1]) for s in sequences)
    out = torch.zeros(1, heads, total, dim, dtype=sequences[0].dtype, device=sequences[0].device)
    start = 0
    for s in sequences:
        out[0, :, start:start + s.shape[1]] = s
        start += _padded(s.shape[1])
    return out.contiguous()


def _varlen(qs, ks, vs, causal: bool, key_bias: torch.Tensor | None = None) -> torch.Tensor:
    is_bf16 = qs[0].dtype == torch.bfloat16
    lengths = [q.shape[1] for q in qs]
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, lengths))], dtype=torch.int32, device="cuda")
    q_packed, q_scale = _C.nvfp4_quantize(_pack_padded(qs), is_bf16)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(_pack_padded(ks), is_bf16)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), is_bf16)
    return _C.fp4_attention_varlen_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu, cu_padded,
        max(lengths), sum(lengths), causal, is_bf16, key_bias)


def _random_sequences(dtype: torch.dtype, head_dim: int, q_heads: int = 8, kv_heads: int = 2,
                      offset: float = 0.0) -> tuple[list[torch.Tensor], ...]:
    torch.manual_seed(0)
    q_shift = offset * torch.randn(q_heads, 1, head_dim, device="cuda")
    k_shift = offset * torch.randn(kv_heads, 1, head_dim, device="cuda")
    qs = [(torch.randn(q_heads, n, head_dim, device="cuda") + q_shift).to(dtype) for n in LENGTHS]
    ks = [(torch.randn(kv_heads, n, head_dim, device="cuda") + k_shift).to(dtype) for n in LENGTHS]
    vs = [torch.randn(kv_heads, n, head_dim, device="cuda").to(dtype) for n in LENGTHS]
    return qs, ks, vs


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """[heads, length, dim] -> [length, heads, dim]"""
    out = F.scaled_dot_product_attention(q.float()[None], k.float()[None], v.float()[None], is_causal=causal,
                                         enable_gqa=True)
    return out[0].transpose(0, 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_varlen_matches_sdpa_per_sequence(dtype: torch.dtype, head_dim: int, causal: bool) -> None:
    _requires_sm120_cuda()
    qs, ks, vs = _random_sequences(dtype, head_dim)
    out = _varlen(qs, ks, vs, causal)
    torch.cuda.synchronize()

    assert out.shape == (sum(LENGTHS), 8, head_dim)
    start = 0
    for q, k, v in zip(qs, ks, vs):
        n = q.shape[1]
        # FP4 Q, K, P and V leave a relative L2 of about 0.2 on random inputs; a sequence that read or wrote
        # another sequence's tokens would be far off
        assert _relative_l2(out[start:start + n], _sdpa(q, k, v, causal)) < 0.3, n
        start += n


@pytest.mark.parametrize("dtype", DTYPES)
def test_varlen_equals_the_fixed_length_kernel_on_whole_blocks(dtype: torch.dtype) -> None:
    _requires_sm120_cuda()
    is_bf16 = dtype == torch.bfloat16
    torch.manual_seed(1)
    q = torch.randn(1, 8, 128, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, 128, 128, device="cuda", dtype=dtype)
    v = torch.randn(1, 2, 128, 128, device="cuda", dtype=dtype)
    q_packed, q_scale = _C.nvfp4_quantize(q, is_bf16)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(k, is_bf16)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(v, is_bf16)
    fixed = _C.fp4_attention_causal_nvfp4_packed(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, is_bf16)
    varlen = _varlen([q[0]], [k[0]], [v[0]], causal=True)
    torch.cuda.synchronize()
    assert torch.equal(varlen, fixed[0].transpose(0, 1))


@pytest.mark.parametrize("causal", (False, True))
def test_key_bias_restores_attention_after_fixed_smoothing(causal: bool) -> None:
    """Subtracting fixed vectors c_q from every query and c_k from every key changes the logits of a row only by
    c_q . (k_j - c_k) plus a constant, so adding that per-key bias back gives the original attention."""
    _requires_sm120_cuda()
    dtype = torch.bfloat16
    qs, ks, vs = _random_sequences(dtype, 128, offset=4.0)  # a large offset shared by all tokens, as in LLMs
    c_q = torch.cat(qs, dim=1).float().mean(dim=1, keepdim=True)  # [q heads, 1, dim]
    c_k = torch.cat(ks, dim=1).float().mean(dim=1, keepdim=True)  # [kv heads, 1, dim]
    groups = qs[0].shape[0] // ks[0].shape[0]
    q_smooth = [(q.float() - c_q).to(dtype) for q in qs]
    k_smooth = [(k.float() - c_k).to(dtype) for k in ks]
    # bias of every key for every query head: c_q . (k - c_k), in the padded token positions
    bias = torch.einsum("hd,htd->ht", c_q[:, 0], _pack_padded(k_smooth)[0].float().repeat_interleave(groups, 0))

    plain = _varlen(qs, ks, vs, causal)
    smoothed = _varlen(q_smooth, k_smooth, vs, causal, key_bias=bias.contiguous())
    ref = torch.cat([_sdpa(q, k, v, causal) for q, k, v in zip(qs, ks, vs)])
    torch.cuda.synchronize()
    assert _relative_l2(smoothed, ref) < 0.3
    assert _relative_l2(smoothed, ref) < _relative_l2(plain, ref)


def _packed_with_random_padding(sequences: list[torch.Tensor]) -> torch.Tensor:
    """_pack_padded with random values instead of zeros between the sequences."""
    out = _pack_padded(sequences)
    start = 0
    for s in sequences:
        n = s.shape[1]
        out[0, :, start + n:start + _padded(n)] = torch.randn_like(out[0, :, start + n:start + _padded(n)])
        start += _padded(n)
    return out


@pytest.mark.parametrize("causal", (False, True))
def test_each_sequence_alone_equals_its_rows_in_the_batch(causal: bool) -> None:
    _requires_sm120_cuda()
    qs, ks, vs = _random_sequences(torch.bfloat16, 128)
    batch = _varlen(qs, ks, vs, causal)
    torch.cuda.synchronize()
    start = 0
    for q, k, v in zip(qs, ks, vs):
        n = q.shape[1]
        assert torch.equal(_varlen([q], [k], [v], causal), batch[start:start + n]), n
        start += n


@pytest.mark.parametrize("causal", (False, True))
def test_q_and_k_padding_is_not_read(causal: bool) -> None:
    _requires_sm120_cuda()
    qs, ks, vs = _random_sequences(torch.bfloat16, 128)
    lengths = [q.shape[1] for q in qs]
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, lengths))], dtype=torch.int32, device="cuda")
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), True)

    outputs = []
    for pack in (_pack_padded, _packed_with_random_padding):
        q_packed, q_scale = _C.nvfp4_quantize(pack(qs), True)
        k_packed, k_scale = _C.nvfp4_quantize_permuted(pack(ks), True)
        outputs.append(_C.fp4_attention_varlen_nvfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu, cu_padded,
            max(lengths), sum(lengths), causal, True, None))
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], outputs[1])


def test_invalid_inputs_are_rejected() -> None:
    _requires_sm120_cuda()
    qs, ks, vs = _random_sequences(torch.bfloat16, 128)
    with pytest.raises(RuntimeError, match="key_bias"):
        _varlen(qs, ks, vs, True, key_bias=torch.zeros(8, 64, device="cuda"))
    lengths = [q.shape[1] for q in qs]
    q_packed, q_scale = _C.nvfp4_quantize(_pack_padded(qs), True)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(_pack_padded(ks), True)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), True)
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda")  # int64
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, lengths))], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="int32"):
        _C.fp4_attention_varlen_nvfp4_packed(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu,
                                             cu_padded, max(lengths), sum(lengths), True, True, None)
