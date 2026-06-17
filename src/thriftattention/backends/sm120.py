from __future__ import annotations

import torch

from thriftattention.backends.single_query import (
    _group_single_query,
    _ungroup_single_query,
    _use_single_query,
)
from thriftattention._checks import check_qkv, require_block_aligned
from thriftattention._extension import get_extension
from thriftattention.config import AttentionConfig
from thriftattention.quant.formats import QuantFormat



class Sm120Nvfp4Backend:
    name = "sm120"

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
        check_qkv(q, k, v)
        if config.block_size != 64:
            raise NotImplementedError("the current SM120 attention kernels use 64-token KV blocks")
        if quant_format.name not in ("nvfp4", "mxfp4"):
            raise NotImplementedError(f"SM120 backend does not support quant format {quant_format.name!r}")

        if _use_single_query(q, config.implementation):
            return self._single_query_attention(q, k, v, selection, quant_format, config, is_bf16)
        return self._tiled_attention(q, k, v, selection, quant_format, config, is_bf16)

    def _single_query_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selection: torch.Tensor | None,
        quant_format: QuantFormat,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        require_block_aligned("k", k.shape[2], config.block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_grouped = _group_single_query(q, k.shape[1])
        packed = quant_format.quantize_single_query_qkv(q_grouped, k, v, is_bf16=is_bf16)
        ext = get_extension()

        if config.method == "fp4":
            fn = (
                ext.fp4_attention_single_query_mxfp4_packed
                if quant_format.name == "mxfp4"
                else ext.fp4_attention_single_query_nvfp4_packed
            )
            out = fn(*packed, is_bf16)
        elif config.method == "thrift":
            if selection is None:
                raise ValueError("thrift attention requires a selection tensor")
            fn = (
                ext.thrift_attention_single_query_mxfp4_packed
                if quant_format.name == "mxfp4"
                else ext.thrift_attention_single_query_nvfp4_packed
            )
            out = fn(
                q_grouped,
                k,
                v,
                selection,
                *packed,
                is_bf16,
            )
        else:
            raise ValueError(f"unsupported attention method {config.method!r}")
        return _ungroup_single_query(out, q.shape[1])

    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selection: torch.Tensor | None,
        quant_format: QuantFormat,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        require_block_aligned("q", q.shape[2], 64)
        require_block_aligned("k", k.shape[2], config.block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        packed = quant_format.quantize_qkv(q, k, v, is_bf16=is_bf16)
        ext = get_extension()

        if config.method == "fp4":
            if quant_format.name == "mxfp4":
                fn = (
                    ext.fp4_attention_causal_mxfp4_packed
                    if config.causal
                    else ext.fp4_attention_noncausal_mxfp4_packed
                )
            else:
                fn = (
                    ext.fp4_attention_causal_nvfp4_packed
                    if config.causal
                    else ext.fp4_attention_noncausal_nvfp4_packed
                )
                return fn(*packed, is_bf16, config.exp_approx, config.microblock_p)
            return fn(*packed, is_bf16)
        if config.method == "thrift":
            if selection is None:
                raise ValueError("thrift attention requires a selection tensor")
            if quant_format.name == "mxfp4":
                fn = (
                    ext.thrift_attention_causal_mxfp4_packed
                    if config.causal
                    else ext.thrift_attention_noncausal_mxfp4_packed
                )
            else:
                fn = (
                    ext.thrift_attention_causal_nvfp4_packed
                    if config.causal
                    else ext.thrift_attention_noncausal_nvfp4_packed
                )
            return fn(q, k, v, selection, *packed, is_bf16)
        raise ValueError(f"unsupported attention method {config.method!r}")


SM120_NVFP4_BACKEND = Sm120Nvfp4Backend()
