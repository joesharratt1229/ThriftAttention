from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from thriftattention.config import QuantFormatName
from thriftattention.quantization import (
    nvfp4_quantize,
    nvfp4_quantize_permuted,
    nvfp4_quantize_transposed,
)

PackedQKV = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class QuantFormat(Protocol):
    name: str

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def quantize_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PackedQKV:
        ...

    def quantize_single_query_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> PackedQKV:
        ...


@dataclass(frozen=True)
class Nvfp4QuantFormat:
    name: QuantFormatName = "nvfp4"

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_quantize(x)

    def quantize_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> PackedQKV:
        q_packed, q_scale = nvfp4_quantize(q)
        k_packed, k_scale = nvfp4_quantize_permuted(k)
        v_packed_t, v_scale_t = nvfp4_quantize_transposed(v)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t

    def quantize_single_query_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> PackedQKV:
        q_packed, q_scale = nvfp4_quantize(q)
        k_packed, k_scale = nvfp4_quantize(k)
        v_packed_t, v_scale_t = nvfp4_quantize_transposed(v)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


_NVFP4 = Nvfp4QuantFormat()


def get_quant_format(name: QuantFormatName) -> QuantFormat:
    if name == "nvfp4":
        return _NVFP4
    raise NotImplementedError(f"quant format {name!r} is not implemented")
