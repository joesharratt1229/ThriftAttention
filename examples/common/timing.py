from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def sync_cuda(device: str | None = None) -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.synchronize(device)


def timed_call(fn: Callable[[], T], *, device: str | None = None) -> tuple[T, float]:
    sync_cuda(device)
    start = time.perf_counter()
    value = fn()
    sync_cuda(device)
    return value, time.perf_counter() - start


def cuda_memory_mb(device: str | None = None) -> float | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available() or (device is not None and not str(device).startswith("cuda")):
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)
