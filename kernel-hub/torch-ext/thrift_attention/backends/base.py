from __future__ import annotations

from typing import Protocol

import torch

from ..config import AttentionConfig
from ..quant.formats import QuantFormat


class AttentionBackend(Protocol):
    name: str

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        selection: torch.Tensor | None,
        quant_format: QuantFormat,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        ...
