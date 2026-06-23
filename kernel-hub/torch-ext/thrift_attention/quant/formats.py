from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..config import QuantFormatName
from ..quantization import (
    mxfp4_quantize,
    mxfp4_quantize_permuted,
    mxfp4_quantize_transposed,
    nvfp4_quantize,
    nvfp4_quantize_permuted,
    nvfp4_quantize_transposed,
)

PackedQKV = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class QuantFormat(Protocol):
    name: str

    def quantize(self, x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def quantize_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        ...

    def quantize_single_query_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        ...


@dataclass(frozen=True)
class Nvfp4QuantFormat:
    name: QuantFormatName = "nvfp4"

    def quantize(self, x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_quantize(x, is_bf16=is_bf16)

    def quantize_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        q_packed, q_scale = nvfp4_quantize(q, is_bf16=is_bf16)
        k_packed, k_scale = nvfp4_quantize_permuted(k, is_bf16=is_bf16)
        v_packed_t, v_scale_t = nvfp4_quantize_transposed(v, is_bf16=is_bf16)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t

    def quantize_single_query_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        q_packed, q_scale = nvfp4_quantize(q, is_bf16=is_bf16)
        k_packed, k_scale = nvfp4_quantize(k, is_bf16=is_bf16)
        v_packed_t, v_scale_t = nvfp4_quantize_transposed(v, is_bf16=is_bf16)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


@dataclass(frozen=True)
class Mxfp4QuantFormat:
    name: QuantFormatName = "mxfp4"

    def quantize(self, x: torch.Tensor, *, is_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return mxfp4_quantize(x, is_bf16=is_bf16)

    def quantize_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        q_packed, q_scale = mxfp4_quantize(q, is_bf16=is_bf16)
        k_packed, k_scale = mxfp4_quantize_permuted(k, is_bf16=is_bf16)
        v_packed_t, v_scale_t = mxfp4_quantize_transposed(v, is_bf16=is_bf16)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t

    def quantize_single_query_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_bf16: bool = False,
    ) -> PackedQKV:
        q_packed, q_scale = mxfp4_quantize(q, is_bf16=is_bf16)
        k_packed, k_scale = mxfp4_quantize(k, is_bf16=is_bf16)
        v_packed_t, v_scale_t = mxfp4_quantize_transposed(v, is_bf16=is_bf16)
        return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


_NVFP4 = Nvfp4QuantFormat()
_MXFP4 = Mxfp4QuantFormat()


def get_quant_format(name: QuantFormatName) -> QuantFormat:
    if name == "nvfp4":
        return _NVFP4
    if name == "mxfp4":
        return _MXFP4
    raise NotImplementedError(f"quant format {name!r} is not implemented")
