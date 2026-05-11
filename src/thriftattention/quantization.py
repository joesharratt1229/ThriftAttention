from __future__ import annotations

import torch

from ._checks import require_cuda_half, require_supported_head_dim
from ._extension import get_extension


def nvfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous FP16 tensor to packed NVFP4 and FP8 scales."""
    require_cuda_half("x", x)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize(x.contiguous()))


def nvfp4_quantize_permuted(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K to packed NVFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_permuted(x.contiguous()))


def nvfp4_quantize_transposed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to the transposed packed NVFP4 layout consumed by attention kernels."""
    require_cuda_half("x", x)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_transposed(x.contiguous()))


def nvfp4_quantize_transposed_permuted(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize V to transposed packed NVFP4 with Sage-style sequence permutation."""
    require_cuda_half("x", x)
    require_supported_head_dim(x.shape[-1])
    return tuple(get_extension().nvfp4_quantize_transposed_permuted(x.contiguous()))
