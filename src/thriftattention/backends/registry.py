from __future__ import annotations

import torch

from thriftattention._checks import require_supported_head_dim
from thriftattention.backends.base import AttentionBackend
from thriftattention.backends.sm80 import SM80_BACKEND
from thriftattention.backends.sm120 import SM120_NVFP4_BACKEND
from thriftattention.config import AttentionConfig
from thriftattention.quant.formats import QuantFormat


BACKENDS_BY_NAME = {
    "sm80": SM80_BACKEND,
    "sm120": SM120_NVFP4_BACKEND,
}
BACKENDS_BY_CAPABILITY_MAJOR = {
    8: SM80_BACKEND,
    12: SM120_NVFP4_BACKEND,
}


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
    try:
        return BACKENDS_BY_NAME[config.backend]
    except KeyError:
        raise NotImplementedError(f"backend {config.backend!r} is not implemented") from None


def _select_auto_backend(device: torch.device | None) -> AttentionBackend:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("automatic backend selection requires a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("automatic backend selection requires a CUDA device")

    capability = torch.cuda.get_device_capability(device)
    try:
        return BACKENDS_BY_CAPABILITY_MAJOR[capability[0]]
    except KeyError:
        raise NotImplementedError(
            f"no ThriftAttention backend is registered for CUDA capability "
            f"{capability[0]}.{capability[1]}"
        ) from None
