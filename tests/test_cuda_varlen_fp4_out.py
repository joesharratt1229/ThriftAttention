from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
_C = pytest.importorskip("thriftattention._C")

from test_cuda_varlen import LENGTHS, _pack_padded, _padded, _random_sequences, _requires_sm120_cuda  # noqa: E402

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _inputs(dtype: torch.dtype, head_dim: int) -> tuple:
    qs, ks, vs = _random_sequences(dtype, head_dim)
    is_bf16 = dtype == torch.bfloat16
    lengths = [q.shape[1] for q in qs]
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    cu_padded = torch.tensor([0, *itertools.accumulate(map(_padded, lengths))], dtype=torch.int32, device="cuda")
    q_packed, q_scale = _C.nvfp4_quantize(_pack_padded(qs), is_bf16)
    k_packed, k_scale = _C.nvfp4_quantize_permuted(_pack_padded(ks), is_bf16)
    v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(_pack_padded(vs), is_bf16)
    return (q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu, cu_padded,
            max(lengths), sum(lengths)), is_bf16


def _unswizzle(scales: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """The swizzled 128x4 scale layout of the block-scaled GEMMs -> [rows, groups]."""
    k_tiles = scales.shape[1] // 4
    m = torch.arange(rows, device=scales.device)[:, None]
    k = torch.arange(groups, device=scales.device)[None, :]
    index = ((m // 128) * k_tiles + k // 4) * 512 + (m % 32) * 16 + ((m // 32) % 4) * 4 + k % 4
    return scales.flatten()[index]


def _dequantize(packed: torch.Tensor, scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=packed.device)
    values = torch.stack([table[(packed & 0xF).long()], table[(packed >> 4).long()]], dim=-1).flatten(-2)
    groups = scales.view(torch.float8_e4m3fn).float() / global_scale
    return values * groups.repeat_interleave(16, dim=-1)


def _reference_scales(out: torch.Tensor, global_scale: float) -> torch.Tensor:
    """e4m3(global_scale * amax / 6) of each 16 values, the scale of vLLM's scaled_fp4_quant."""
    amax = out.float().unflatten(-1, (-1, 16)).abs().amax(dim=-1)
    sixth = torch.tensor(1 / 6, dtype=torch.float32)
    return (global_scale * (amax * sixth)).clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_fp4_out_is_the_nvfp4_quantization_of_the_16_bit_output(dtype: torch.dtype, head_dim: int,
                                                                  causal: bool) -> None:
    _requires_sm120_cuda()
    args, is_bf16 = _inputs(dtype, head_dim)
    total = args[-1]
    out = _C.fp4_attention_varlen_nvfp4_packed(*args, causal, is_bf16, None).flatten(1)
    amax = out.float().abs().max().item()
    global_scale = 448 * 6 / amax  # vLLM's input_global_scale_inv for a calibrated amax
    packed, scales = _C.fp4_attention_varlen_nvfp4_packed_fp4_out(
        *args, causal, is_bf16, None, torch.tensor([global_scale], device="cuda"))
    torch.cuda.synchronize()

    columns = out.shape[1]
    assert packed.shape == (total, columns // 2) and packed.dtype == torch.uint8
    assert scales.shape == (-(-total // 128) * 128, -(-columns // 64) * 4) and scales.dtype == torch.float8_e4m3fn
    row_scales = _unswizzle(scales, total, columns // 16)
    assert torch.equal(row_scales.view(torch.uint8), _reference_scales(out, global_scale))
    # the GEMM reads the scale rows up to a multiple of 128; those past the output are zero
    padded_rows = _unswizzle(scales, scales.shape[0], columns // 16)[total:]
    assert torch.count_nonzero(padded_rows.view(torch.uint8)) == 0

    # each value is the nearest E2M1 value of value / unit: the error is at most half the widest gap (2) in units
    restored = _dequantize(packed, row_scales, global_scale)
    unit = row_scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1) / global_scale
    assert ((restored - out.float()).abs() <= unit * 1.0 + 1e-6).all()
    nearest = torch.tensor(E2M1_VALUES, device="cuda")
    scaled = (out.float().abs() / unit).clamp(max=6)
    best = (scaled[..., None] - nearest).abs().min(dim=-1).values
    chosen = (restored.abs() / unit - scaled).abs()
    assert ((chosen - best).abs() < 1e-3).float().mean().item() > 0.999



def test_fp4_out_equals_vllm_scaled_fp4_quant_of_the_16_bit_output() -> None:
    _requires_sm120_cuda()
    ops = pytest.importorskip("vllm._custom_ops")
    args, is_bf16 = _inputs(torch.bfloat16, 128)
    out = _C.fp4_attention_varlen_nvfp4_packed(*args, True, is_bf16, None).flatten(1)
    global_scale = torch.tensor([448 * 6 / out.float().abs().max().item()], device="cuda")
    packed, scales = _C.fp4_attention_varlen_nvfp4_packed_fp4_out(*args, True, is_bf16, None, global_scale)
    ref_packed, ref_scales = ops.scaled_fp4_quant(out, global_scale, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    assert torch.equal(packed, ref_packed)
    assert scales.shape == ref_scales.shape and torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))
