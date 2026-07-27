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


SM100_EXTENSION_SYMBOLS = (
    "nvfp4_quantize",
    "nvfp4_quantize_permuted",
    "nvfp4_quantize_transposed",
    "sm100_fp4_attention_causal_nvfp4_packed",
    "sm100_fp4_attention_noncausal_nvfp4_packed",
    "sm100_fp4_attention_single_query_nvfp4_packed",
)


def require_sm100_available(device: torch.device | None) -> None:
    if device is None or device.type != "cuda":
        raise NotImplementedError("SM100 backend requires CUDA tensors on an SM100 device")
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise NotImplementedError(f"SM100 backend requires compute capability 10.0, got {capability}")
    ext = get_extension()
    missing = [name for name in SM100_EXTENSION_SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise ImportError(
            "SM100 FP4 extension symbols are not available. Rebuild ThriftAttention with "
            "TORCH_CUDA_ARCH_LIST=10.0a and THRIFTATTENTION_PTXAS_GPU_NAME=sm_100a. "
            f"Missing: {', '.join(missing)}"
        )


class Sm100Nvfp4Backend:
    name = "sm100"

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
        if config.method != "fp4":
            raise NotImplementedError("SM100 backend currently supports only pure FP4 attention")
        if quant_format.name != "nvfp4":
            raise NotImplementedError(f"SM100 backend does not support quant format {quant_format.name!r}")
        if selection is not None:
            raise ValueError("SM100 pure FP4 attention does not consume a selection tensor")
        if config.block_size != 64:
            raise NotImplementedError("the current SM100 FP4 kernels use 64-token KV blocks")

        if _use_single_query(q, config.implementation):
            return self.single_query_attention(q, k, v, quant_format, is_bf16)
        return self.tiled_attention(q, k, v, quant_format, config, is_bf16)

    def single_query_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        quant_format: QuantFormat,
        is_bf16: bool,
    ) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_grouped = _group_single_query(q, k.shape[1])
        packed = quant_format.quantize_single_query_qkv(q_grouped, k, v, is_bf16=is_bf16)
        out = get_extension().sm100_fp4_attention_single_query_nvfp4_packed(*packed, is_bf16)
        return _ungroup_single_query(out, q.shape[1])

    def tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        quant_format: QuantFormat,
        config: AttentionConfig,
        is_bf16: bool,
    ) -> torch.Tensor:
        if config.causal and q.shape[2] != k.shape[2]:
            raise NotImplementedError("SM100 causal tiled attention requires q_len == kv_len")
        if config.causal:
            require_block_aligned("q", q.shape[2], 64)
        require_block_aligned("k", k.shape[2], config.block_size)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        packed = quant_format.quantize_qkv(q, k, v, is_bf16=is_bf16)
        ext = get_extension()
        fn = (
            ext.sm100_fp4_attention_causal_nvfp4_packed
            if config.causal
            else ext.sm100_fp4_attention_noncausal_nvfp4_packed
        )
        return fn(*packed, is_bf16)


SM100_NVFP4_BACKEND = Sm100Nvfp4Backend()
