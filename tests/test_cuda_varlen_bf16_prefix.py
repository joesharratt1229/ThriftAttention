"""The bfloat16 prefix of the cached varlen kernel: keys 0 .. prefix_len - 1 of each sequence are read in bfloat16
from a paged cache and continue the online softmax of the sequence's FP4 keys."""
from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
_C = pytest.importorskip("thriftattention._C")

from test_cuda_varlen import _pack_padded, _padded, _relative_l2, _requires_sm120_cuda  # noqa: E402
from test_cuda_varlen_fp4_out import _dequantize, _reference_scales, _unswizzle  # noqa: E402

# (prefix length, sequence length): no prefix, one key, block ends and starts, whole blocks, and one new token
SEQUENCES = ((0, 37), (1, 40), (31, 100), (32, 64), (33, 150), (76, 300), (93, 257), (64, 65))
Q_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 128
GROUPS = Q_HEADS // KV_HEADS


def _sequences(smooth: bool = True) -> dict:
    """Random sequences with a sink at each first token, a large key along the mean query as in LLMs, and the fixed
    smoothing vectors c_q and c_k that FP4 attention callers subtract (zero without smoothing)."""
    torch.manual_seed(0)
    total = sum(n for _, n in SEQUENCES)
    q = (torch.randn(Q_HEADS, total, HEAD_DIM, device="cuda") * 1.5 + 0.3).bfloat16()
    k = (torch.randn(KV_HEADS, total, HEAD_DIM, device="cuda") * 2 - 0.5).bfloat16()
    v = torch.randn(KV_HEADS, total, HEAD_DIM, device="cuda").bfloat16()
    starts = list(itertools.accumulate((n for _, n in SEQUENCES), initial=0))[:-1]
    for s in starts:
        k[:, s] = (q[:, s:s + 32].float().mean(dim=1).view(KV_HEADS, GROUPS, HEAD_DIM).mean(dim=1) * 6).bfloat16()
    scale = 0.3 if smooth else 0.0
    c_q = torch.randn(Q_HEADS, HEAD_DIM, device="cuda") * scale
    c_k = torch.randn(KV_HEADS, HEAD_DIM, device="cuda") * scale
    return {"q": q, "k": k, "v": v, "starts": starts, "c_q": c_q, "c_k": c_k}


def _fp4_inputs(x: dict, firsts: list[int]) -> tuple[tuple, torch.Tensor, list[int]]:
    """The packed FP4 operands of the tokens from firsts[i] on in sequence i, smoothed (q - c_q, k - c_k) with the
    key bias c_q . (k - c_k) that restores the logits, and the number of tokens of each."""
    tokens = [torch.arange(s + first, s + n, device="cuda")
              for (_, n), s, first in zip(SEQUENCES, x["starts"], firsts)]
    lengths = [len(t) for t in tokens]
    q_smooth = [(x["q"][:, t].float() - x["c_q"][:, None]).bfloat16() for t in tokens]
    k_smooth = [(x["k"][:, t].float() - x["c_k"][:, None]).bfloat16() for t in tokens]
    vs = [x["v"][:, t] for t in tokens]
    bias = torch.einsum("hd,htd->ht", x["c_q"], _pack_padded(k_smooth)[0].float().repeat_interleave(GROUPS, 0))
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, lengths))], dtype=torch.int32, device="cuda")
    q_packed, q_scale = _C.nvfp4_quantize(_pack_padded(q_smooth), True)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(_pack_padded(k_smooth), True)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), True)
    args = (q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu, cu_padded, max(lengths))
    return args, bias.contiguous(), lengths


def _paged_prefixes(x: dict, block_size: int = 16, fill: float | None = None, hnd: bool = False) -> dict:
    """The prefix keys and values of every sequence in a paged bfloat16 cache whose blocks are shuffled; the other
    slots hold random values, or fill. With hnd, the cache is a [blocks, block_size, heads, dim] view of a
    [blocks, heads, block_size, dim] tensor."""
    prefixes = [p for p, _ in SEQUENCES]
    blocks = [max(1, -(-p // block_size)) for p in prefixes]
    order = torch.randperm(sum(blocks) + 7).tolist()
    shape = (len(order), block_size, KV_HEADS, HEAD_DIM)
    key_cache = torch.randn(shape, device="cuda").bfloat16()
    value_cache = torch.randn(shape, device="cuda").bfloat16()
    if fill is not None:
        key_cache.fill_(fill)
        value_cache.fill_(fill)
    block_table = torch.zeros(len(SEQUENCES), max(blocks), dtype=torch.int32)
    for i, (p, s) in enumerate(zip(prefixes, x["starts"])):
        for b in range(blocks[i]):
            block = order.pop()
            block_table[i, b] = block
            first, last = s + b * block_size, s + min(p, (b + 1) * block_size)
            key_cache[block, :last - first] = x["k"][:, first:last].transpose(0, 1)
            value_cache[block, :last - first] = x["v"][:, first:last].transpose(0, 1)
    if hnd:
        key_cache = key_cache.transpose(1, 2).contiguous().transpose(1, 2)
        value_cache = value_cache.transpose(1, 2).contiguous().transpose(1, 2)
    return {"key_cache": key_cache, "value_cache": value_cache, "block_table": block_table.cuda(),
            "prefix_len": torch.tensor(prefixes, dtype=torch.int32, device="cuda")}


def _with_prefix(x: dict, causal: bool = True, key_mean: bool = True, cache: dict | None = None,
                 **outputs) -> list[torch.Tensor]:
    """Rows after each prefix: their FP4 keys from the packed inputs, the prefix from the paged cache."""
    args, bias, lengths = _fp4_inputs(x, [p for p, _ in SEQUENCES])
    tokens = torch.cat([torch.arange(s + p, s + n, device="cuda") for (p, n), s in zip(SEQUENCES, x["starts"])])
    q_bf16 = x["q"][:, tokens].transpose(0, 1).contiguous()  # the rows' unsmoothed queries
    zeros = torch.zeros(len(SEQUENCES), dtype=torch.int32, device="cuda")
    return _C.fp4_attention_varlen_nvfp4_cached(
        *args, sum(lengths), causal, True, bias, zeros, args[6][:-1], q_bf16=q_bf16,
        key_mean=x["c_k"].contiguous() if key_mean else None, **(cache or _paged_prefixes(x)), **outputs)


def _without_prefix(x: dict, firsts: list[int]) -> torch.Tensor:
    """Rows after each prefix with every key in FP4: whole sequences, computed from row firsts[i] = prefix on."""
    args, bias, lengths = _fp4_inputs(x, firsts)
    rows = [n - p for p, n in SEQUENCES]
    q_start = torch.tensor([p - first for (p, _), first in zip(SEQUENCES, firsts)], dtype=torch.int32, device="cuda")
    out_start = torch.tensor([0, *itertools.accumulate(rows)][:-1], dtype=torch.int32, device="cuda")
    (out,) = _C.fp4_attention_varlen_nvfp4_cached(*args[:-1], max(rows), sum(rows), True, True, bias, q_start,
                                                  out_start)
    return out


def _reference(x: dict) -> torch.Tensor:
    """float64 causal attention of the rows after each prefix over all keys of their sequence."""
    rows = []
    for (p, n), s in zip(SEQUENCES, x["starts"]):
        q = x["q"][:, s + p:s + n].double()
        k = x["k"][:, s:s + n].double().repeat_interleave(GROUPS, 0)
        v = x["v"][:, s:s + n].double().repeat_interleave(GROUPS, 0)
        logits = q @ k.transpose(1, 2) * HEAD_DIM ** -0.5
        future = torch.arange(n, device="cuda")[None, :] > torch.arange(p, n, device="cuda")[:, None]
        rows.append((logits.masked_fill(future, float("-inf")).softmax(dim=-1) @ v).transpose(0, 1))
    return torch.cat(rows)


def _rows(select) -> torch.Tensor:
    """The output rows of the sequences for which select(prefix, length) holds."""
    return torch.cat([torch.full((n - p,), bool(select(p, n))) for p, n in SEQUENCES]).cuda()


@pytest.mark.parametrize("block_size", (16, 32))
def test_bf16_prefix_is_closer_to_float64_than_fp4_keys(block_size: int) -> None:
    _requires_sm120_cuda()
    x = _sequences()
    (out,) = _with_prefix(x, cache=_paged_prefixes(x, block_size))
    fp4 = _without_prefix(x, [0] * len(SEQUENCES))
    reference = _reference(x)
    torch.cuda.synchronize()
    rows = _rows(lambda p, n: p > 0)
    error, fp4_error = _relative_l2(out[rows], reference[rows]), _relative_l2(fp4[rows], reference[rows])
    # measured: 0.13 against 0.20
    assert error < 0.8 * fp4_error, (error, fp4_error)
    # one new token after 64 prefix keys: every key but its own is exact, so a wrong frame or scale shows at once
    single = _rows(lambda p, n: n - p == 1 and p == 64)
    assert _relative_l2(out[single], reference[single]) < 0.01


def test_sequences_without_a_prefix_are_unchanged() -> None:
    _requires_sm120_cuda()
    x = _sequences()
    (out,) = _with_prefix(x)
    plain = _without_prefix(x, [p for p, _ in SEQUENCES])
    torch.cuda.synchronize()
    rows = _rows(lambda p, n: p == 0)
    assert torch.equal(out[rows], plain[rows])


def test_cache_slots_past_the_prefix_are_not_read() -> None:
    _requires_sm120_cuda()
    x = _sequences()
    (out,) = _with_prefix(x)  # random values in the other slots
    (nan_filled,) = _with_prefix(x, cache=_paged_prefixes(x, fill=float("nan")))
    torch.cuda.synchronize()
    assert torch.equal(out, nan_filled)


def test_hnd_cache_views_give_the_same_output() -> None:
    _requires_sm120_cuda()
    x = _sequences()
    torch.manual_seed(1)
    (nhd,) = _with_prefix(x, cache=_paged_prefixes(x))
    torch.manual_seed(1)
    (hnd,) = _with_prefix(x, cache=_paged_prefixes(x, hnd=True))
    torch.cuda.synchronize()
    assert torch.equal(nhd, hnd)


def test_no_key_mean_is_a_zero_key_mean() -> None:
    _requires_sm120_cuda()
    x = _sequences(smooth=False)
    torch.manual_seed(1)
    (zero_mean,) = _with_prefix(x)
    torch.manual_seed(1)
    (no_mean,) = _with_prefix(x, key_mean=False)
    torch.cuda.synchronize()
    assert torch.equal(zero_mean, no_mean)


def test_fp4_out_with_a_prefix_is_the_nvfp4_quantization_of_its_16_bit_output() -> None:
    _requires_sm120_cuda()
    x = _sequences()
    torch.manual_seed(1)
    (out,) = _with_prefix(x)
    out = out.flatten(1)
    global_scale = 448 * 6 / out.float().abs().max().item()
    torch.manual_seed(1)
    values, scales = _with_prefix(x, out_global_scale=torch.tensor([global_scale], device="cuda"))
    torch.cuda.synchronize()
    row_scales = _unswizzle(scales, out.shape[0], out.shape[1] // 16)
    assert torch.equal(row_scales.view(torch.uint8), _reference_scales(out, global_scale))
    # each value is within one unit (half the widest gap between E2M1 values) of the 16-bit output
    unit = row_scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1) / global_scale
    assert ((_dequantize(values, row_scales, global_scale) - out.float()).abs() <= unit + 1e-6).all()


def test_unsupported_calls_are_rejected() -> None:
    _requires_sm120_cuda()
    x = _sequences()
    with pytest.raises(RuntimeError, match="causal bfloat16"):
        _with_prefix(x, causal=False)
    args, bias, lengths = _fp4_inputs(x, [p for p, _ in SEQUENCES])
    zeros = torch.zeros(len(SEQUENCES), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="only used with prefix_len"):
        _C.fp4_attention_varlen_nvfp4_cached(*args, sum(lengths), True, True, bias, zeros, args[6][:-1],
                                             key_mean=x["c_k"].contiguous())
