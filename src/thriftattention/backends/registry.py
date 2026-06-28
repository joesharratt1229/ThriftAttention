from __future__ import annotations

import torch

from thriftattention._checks import require_supported_head_dim
from thriftattention.backends.base import AttentionBackend
from thriftattention.backends.sm120 import SM120_NVFP4_BACKEND
from thriftattention.config import AttentionConfig
from thriftattention.quant.formats import QuantFormat


def select_backend(
    config: AttentionConfig,
    quant_format: QuantFormat,
    *,
    head_dim: int,
    device: torch.device | None = None,
) -> AttentionBackend:
    require_supported_head_dim(head_dim)
    if config.method not in ("thrift", "fp4"):
        raise NotImplementedError(f"attention method {config.method!r} is not implemented")
    if quant_format.name not in ("nvfp4", "mxfp4"):
        raise NotImplementedError(f"quant format {quant_format.name!r} is not implemented")
    if config.backend == "auto":
        return _select_auto_backend(device)
    if config.backend == "sm120":
        return SM120_NVFP4_BACKEND
    raise NotImplementedError(f"backend {config.backend!r} is not implemented")


def _select_auto_backend(device: torch.device | None) -> AttentionBackend:
    # Future SM100 support should branch here based on device capability.
    return SM120_NVFP4_BACKEND
