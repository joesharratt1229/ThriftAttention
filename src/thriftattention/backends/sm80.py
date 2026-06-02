from __future__ import annotations

import torch

from thriftattention._checks import check_qkv, require_block_aligned
from thriftattention._extension import get_extension
from thriftattention.backends.single_query import (
    _group_single_query,
    _ungroup_single_query,
    _use_single_query,
)
from thriftattention.config import AttentionConfig
from thriftattention.quant.formats import QuantFormat


class Sm80Int8Backend:
    """Ampere backend for dense or selected-block INT8 attention."""

    name = "sm80"

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
            raise NotImplementedError("the SM80 attention kernels use 64-token KV blocks")
        if config.method not in ("fp4", "thrift"):
            raise NotImplementedError(
                f"SM80 backend does not support attention method {config.method!r}; "
                "use 'fp4' for full INT8 or 'thrift' for selected-block INT8"
            )

        # SM80 kernels consume FP16/BF16 inputs and perform their own INT8
        # conversion. The SM120 FP4 packing format is intentionally irrelevant.
        del quant_format

        if _use_single_query(q, config.implementation):
            return self._single_query_attention(q, k, v, selection, config, is_bf16)
        return self._tiled_attention(q, k, v, selection, config, is_bf16)

    def _single_query_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selection: torch.Tensor | None,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        require_block_aligned("k", k.shape[2], config.block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_grouped = _group_single_query(q, k.shape[1])
        ext = get_extension()

        if config.method == "fp4":
            out = ext.int8_attention_single_query(q_grouped, k, v, is_bf16)
        else:
            if selection is None:
                raise ValueError("thrift attention requires a selection tensor")
            out = ext.thrift_attention_single_query_int8(
                q_grouped,
                k,
                v,
                selection,
                is_bf16,
            )
        return _ungroup_single_query(out, q.shape[1])

    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selection: torch.Tensor | None,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        require_block_aligned("q", q.shape[2], 64)
        require_block_aligned("k", k.shape[2], config.block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ext = get_extension()

        if config.method == "fp4":
            fn = ext.int8_attention_causal if config.causal else ext.int8_attention_noncausal
            return fn(q, k, v, is_bf16)

        if selection is None:
            raise ValueError("thrift attention requires a selection tensor")
        fn = (
            ext.thrift_attention_causal_int8
            if config.causal
            else ext.thrift_attention_noncausal_int8
        )
        return fn(q, k, v, selection, is_bf16)


SM80_BACKEND = Sm80Int8Backend()
