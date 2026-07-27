from __future__ import annotations

import torch

from thriftattention._checks import require_supported_head_dim
from thriftattention.backends.base import AttentionBackend
from thriftattention.backends.sm100 import SM100_NVFP4_BACKEND, require_sm100_available
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
        return select_auto_backend(config, quant_format, device)
    if config.backend == "sm100":
        if config.method != "fp4":
            raise NotImplementedError("SM100 backend currently supports only pure FP4 attention")
        if quant_format.name != "nvfp4":
            raise NotImplementedError("SM100 backend currently supports only quant_format='nvfp4'")
        require_sm100_available(device)
        return SM100_NVFP4_BACKEND
    if config.backend == "sm120":
        return SM120_NVFP4_BACKEND
    raise NotImplementedError(f"backend {config.backend!r} is not implemented")


def select_auto_backend(
    config: AttentionConfig,
    quant_format: QuantFormat,
    device: torch.device | None,
) -> AttentionBackend:
    if device is not None and device.type == "cuda":
        if torch.cuda.get_device_capability(device) == (10, 0):
            if config.method == "fp4" and quant_format.name == "nvfp4":
                require_sm100_available(device)
                return SM100_NVFP4_BACKEND
            raise NotImplementedError(
                "auto backend on SM100 currently supports only method='fp4' with quant_format='nvfp4'"
            )
    return SM120_NVFP4_BACKEND
