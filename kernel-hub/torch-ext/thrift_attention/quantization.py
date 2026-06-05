from __future__ import annotations

import torch

from ._checks import require_cuda_half, require_supported_head_dim
from ._extension import get_extension


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
