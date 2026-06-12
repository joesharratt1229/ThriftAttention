from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AttentionMethod = Literal["thrift", "fp4"]
SelectionMethod = Literal["block_mean", "quest"]
QuantFormatName = Literal["nvfp4", "mxfp4"]
AttentionBackendName = Literal["auto", "sm120"]
AttentionImplementation = Literal["auto", "tiled", "single_query"]
FallbackBackend = Literal["error"]


@dataclass(frozen=True)
class AttentionConfig:
    method: AttentionMethod = "thrift"
    causal: bool = True
    selection: SelectionMethod = "block_mean"
    fraction: float | None = 0.05
    top_k: int | None = None
    block_size: int = 64
    quant_format: QuantFormatName = "nvfp4"
    backend: AttentionBackendName = "auto"
    implementation: AttentionImplementation = "auto"
    fallback: FallbackBackend = "error"
    exp_approx: bool = False
