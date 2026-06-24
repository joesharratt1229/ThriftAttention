from __future__ import annotations

from importlib import import_module
from typing import Any

_hub_extension: Any = None


def get_extension() -> Any:
    if _hub_extension is not None:
        return _hub_extension
    try:
        # Loading _C triggers TORCH_LIBRARY registration of all ops under torch.ops._C
        import_module("thriftattention._C")
    except ImportError as exc:
        raise ImportError(
            "The ThriftAttention CUDA extension is not built. Install the package "
            "with `pip install -e .` from the repository root on a CUDA-enabled "
            "machine, or load via the HF Hub kernel "
            "(`attn_implementation='Hrsh-Venket/thrift-attention'`)."
        ) from exc
    import torch
    return torch.ops._C
