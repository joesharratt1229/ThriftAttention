from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AttentionMode = Literal["thrift", "fp4"]
SelectionMethod = Literal["block_mean"]


@dataclass(frozen=True)
class AttentionConfig:
    mode: AttentionMode = "thrift"
    causal: bool = True
    selector: SelectionMethod = "block_mean"
    fp16_fraction: float = 0.05
    top_k: int | None = None
    block_size: int = 64
    backend: str = "auto"
