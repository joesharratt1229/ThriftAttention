from __future__ import annotations

import torch

from ._checks import require_cuda_half, require_supported_head_dim
from ._extension import get_extension

# E2M1 magnitudes indexed by the low 3 bits of a nibble; bit 3 is the sign.
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _check_dtype_flag(name: str, tensor: torch.Tensor, *, is_bf16: bool) -> None:
    expected = torch.bfloat16 if is_bf16 else torch.float16
    if tensor.dtype != expected:
        raise ValueError(f"{name} dtype must match is_bf16={is_bf16}")


def nvfp4_quantize(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous FP16/BF16 tensor to packed NVFP4 and FP8 scales."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize(x.contiguous(), is_bf16))


def nvfp4_quantize_permuted(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K to packed NVFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_permuted(x.contiguous(), is_bf16))


def nvfp4_quantize_transposed(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to the transposed packed NVFP4 layout consumed by attention kernels."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_transposed(x.contiguous(), is_bf16))


def nvfp4_quantize_transposed_permuted(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to transposed packed NVFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_transposed_permuted(x.contiguous(), is_bf16))


def nvfp4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    is_bf16: bool = False,
) -> torch.Tensor:
    """Expand `nvfp4_quantize` output back to FP16/BF16 (a pure-PyTorch reference).

    The result is exact: an E2M1 value times an E4M3 scale carries at most six
    significand bits, so it is representable in both half formats.
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be the uint8 tensor returned by nvfp4_quantize")
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError("scale must be the float8_e4m3fn tensor returned by nvfp4_quantize")
    if packed.shape[:-1] != scale.shape[:-1] or packed.shape[-1] != scale.shape[-1] * 8:
        raise ValueError("packed/scale shapes do not match a head_dim/2 vs head_dim/16 pair")

    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=packed.device)
    lut = torch.cat([lut, -lut])  # nibble bit 3 is the sign
    values = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.float32, device=packed.device)
    values[..., 0::2] = lut[(packed & 0x0F).long()]  # even element in the low nibble
    values[..., 1::2] = lut[(packed >> 4).long()]
    values *= scale.float().repeat_interleave(16, dim=-1)
    return values.to(torch.bfloat16 if is_bf16 else torch.float16)


def mxfp4_quantize(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous FP16/BF16 tensor to packed MXFP4 and E8M0 scales."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().mxfp4_quantize(x.contiguous(), is_bf16))


def mxfp4_quantize_permuted(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K to packed MXFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().mxfp4_quantize_permuted(x.contiguous(), is_bf16))


def mxfp4_quantize_transposed(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to the transposed packed MXFP4 layout consumed by attention kernels."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().mxfp4_quantize_transposed(x.contiguous(), is_bf16))


def mxfp4_quantize_transposed_permuted(x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to transposed packed MXFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    _check_dtype_flag("x", x, is_bf16=is_bf16)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().mxfp4_quantize_transposed_permuted(x.contiguous(), is_bf16))
