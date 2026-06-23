from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..config import SelectionMethod


@dataclass(frozen=True)
class SelectionConfig:
    name: SelectionMethod = "block_mean"
    fraction: float | None = 0.05
    top_k: int | None = None
    block_size: int = 64


class SelectionPolicy(Protocol):
    name: str

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        config: SelectionConfig,
        causal: bool,
        is_bf16: bool,
    ) -> torch.Tensor:
        ...
