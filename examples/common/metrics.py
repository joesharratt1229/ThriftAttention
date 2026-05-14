from __future__ import annotations

import math
from typing import Any


def mae(reference: Any, candidate: Any) -> float:
    ref, cand = _as_float_tensors(reference, candidate)
    return float((cand - ref).abs().mean().item())


def rmse(reference: Any, candidate: Any) -> float:
    ref, cand = _as_float_tensors(reference, candidate)
    return float(((cand - ref) ** 2).mean().sqrt().item())


def psnr(reference: Any, candidate: Any, *, max_value: float = 1.0) -> float:
    value = rmse(reference, candidate)
    if value == 0.0:
        return float("inf")
    return float(20.0 * math.log10(float(max_value) / value))


def _as_float_tensors(reference: Any, candidate: Any):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("metrics helpers require PyTorch") from exc
    ref = torch.as_tensor(reference, dtype=torch.float64)
    cand = torch.as_tensor(candidate, dtype=torch.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: reference {tuple(ref.shape)} vs candidate {tuple(cand.shape)}")
    return ref, cand
