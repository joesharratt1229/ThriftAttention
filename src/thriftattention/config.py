from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AttentionMode = Literal["thrift", "fp4"]
SelectionMethod = Literal["block_mean"]
PatchBackend = Literal["auto", "hf"]
FallbackBackend = Literal["error"]


@dataclass(frozen=True)
class AttentionConfig:
    mode: AttentionMode = "thrift"
    causal: bool = True
    selector: SelectionMethod = "block_mean"
    fp16_fraction: float = 0.05
    top_k: int | None = None
    block_size: int = 64
    backend: PatchBackend = "auto"
    fallback: FallbackBackend = "error"
    patch_generation: bool = True
