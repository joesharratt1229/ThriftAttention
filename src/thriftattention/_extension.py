from __future__ import annotations

from importlib import import_module
from types import ModuleType


def get_extension() -> ModuleType:
    try:
        return import_module("thriftattention._C")
    except ImportError as exc:
        raise ImportError(
            "The ThriftAttention CUDA extension is not built. Install the package "
            "with `pip install -e .` from the repository root on a CUDA-enabled "
            "machine."
        ) from exc
