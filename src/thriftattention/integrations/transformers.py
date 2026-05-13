from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..config import AttentionConfig
from ..functional import attention as thrift_attention
from ..functional import fp4_attention
from ..selection import resolve_top_k


BACKEND_NAME = "thriftattention"
_CONFIG_ATTR = "_thriftattention_config"
_ORIGINAL_ATTN_IMPL_ATTR = "_thriftattention_original_attn_implementation"
_PATCHED_ATTR = "_thriftattention_patched"


@dataclass(frozen=True)
class HFPatchHandle:
    model: object
    original_attn_implementation: object
    backend_name: str = BACKEND_NAME

    def unpatch(self) -> object:
        return unpatch_hf_model(self.model)


def register_thriftattention_backend(config: AttentionConfig | None = None) -> str:
    registry = _get_attention_registry()
    _register_once(registry, BACKEND_NAME, thriftattention_forward)
    _register_attention_mask()
    return BACKEND_NAME


def patch_hf_model(model: object, config: AttentionConfig) -> object:
    set_attn_implementation = _attn_implementation_setter(model)
    register_thriftattention_backend(config)
    original = _current_attn_implementation(model)
    if not hasattr(model, _ORIGINAL_ATTN_IMPL_ATTR):
        setattr(model, _ORIGINAL_ATTN_IMPL_ATTR, original)

    for module in _iter_modules(model):
        setattr(module, _CONFIG_ATTR, config)

    set_attn_implementation(BACKEND_NAME)
    setattr(model, _PATCHED_ATTR, True)
    return model


def unpatch_hf_model(model: object) -> object:
    original = getattr(model, _ORIGINAL_ATTN_IMPL_ATTR, None)
    if hasattr(model, _ORIGINAL_ATTN_IMPL_ATTR):
        _set_attn_implementation(model, original)
        delattr(model, _ORIGINAL_ATTN_IMPL_ATTR)

    for module in _iter_modules(model):
        if hasattr(module, _CONFIG_ATTR):
            delattr(module, _CONFIG_ATTR)

    if hasattr(model, _PATCHED_ATTR):
        delattr(model, _PATCHED_ATTR)
    return model


def thriftattention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    config = _module_config(module)
    rejection = _fast_path_rejection_reason(module, query, key, value, attention_mask, config, kwargs)
    if rejection is not None:
        return _fallback_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            config=config,
            reason=rejection,
            **kwargs,
        )

    if config.mode == "thrift":
        output = thrift_attention(
            query,
            key,
            value,
            causal=config.causal,
            selector=config.selector,
            top_k=config.top_k,
            fraction=config.fp16_fraction,
            block_size=config.block_size,
        )
    elif config.mode == "fp4":
        output = fp4_attention(query, key, value, causal=config.causal)
    else:
        raise ValueError(f"unsupported ThriftAttention mode {config.mode!r}")

    return output.transpose(1, 2).contiguous(), None


def _get_attention_registry() -> Any:
    try:
        from transformers import AttentionInterface
    except Exception:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as AttentionInterface
        except Exception as exc:
            raise ImportError(
                "backend='hf' requires Hugging Face Transformers with AttentionInterface support"
            ) from exc
    return AttentionInterface


def _register_attention_mask() -> None:
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
    except Exception:
        return
    _register_once(AttentionMaskInterface, BACKEND_NAME, sdpa_mask)


def _register_once(registry: Any, name: str, fn: Callable[..., Any]) -> None:
    try:
        registry.register(name, fn)
    except ValueError as exc:
        if "already" not in str(exc).lower():
            raise


def _iter_modules(model: object) -> list[object]:
    modules = getattr(model, "modules", None)
    if callable(modules):
        return list(modules())
    return [model]


def _current_attn_implementation(model: object) -> object:
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", None)


def _set_attn_implementation(model: object, name: object) -> None:
    _attn_implementation_setter(model)(name)


def _attn_implementation_setter(model: object) -> Callable[[object], None]:
    setter = getattr(model, "set_attn_implementation", None)
    if not callable(setter):
        raise TypeError(
            "backend='hf' requires a Transformers model with set_attn_implementation()"
        )
    return setter


def _module_config(module: torch.nn.Module) -> AttentionConfig:
    config = getattr(module, _CONFIG_ATTR, None)
    if config is None:
        raise RuntimeError(
            "ThriftAttention HF backend was called on an unpatched module. "
            "Call thriftattention.patch_model(model, backend='hf') first."
        )
    return config


def _fast_path_rejection_reason(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: AttentionConfig,
    kwargs: dict[str, Any],
) -> str | None:
    if getattr(module, "training", False):
        return "ThriftAttention inference patch does not run while the attention module is in training mode"
    if kwargs.get("output_attentions", False):
        return "ThriftAttention does not return attention weights"

    if attention_mask is not None:
        return "ThriftAttention fast path does not support explicit attention masks"

    is_causal = kwargs.get("is_causal", getattr(module, "is_causal", config.causal))
    if is_causal is None:
        is_causal = config.causal
    if bool(is_causal) != config.causal:
        mode = "causal" if config.causal else "non-causal"
        requested = "causal" if bool(is_causal) else "non-causal"
        return f"ThriftAttention is configured for {mode} attention but this call requested {requested}"

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if tensor.ndim != 4:
            return f"{name} must be shaped [batch, heads, seq, head_dim]"
        if not tensor.is_cuda:
            return "ThriftAttention fast path requires CUDA tensors"
        if tensor.dtype != torch.float16:
            return "ThriftAttention fast path requires float16 query/key/value tensors"

    head_dim = query.shape[-1]
    if head_dim not in (64, 128):
        return f"ThriftAttention supports head_dim 64 or 128, got {head_dim}"

    expected_scaling = head_dim**-0.5
    scaling = kwargs.get("scaling", expected_scaling)
    if not _is_expected_scaling(scaling, expected_scaling):
        return "ThriftAttention kernels use the default 1/sqrt(head_dim) scaling"

    if query.shape[2] % config.block_size != 0:
        return f"query length must be divisible by {config.block_size}"
    if key.shape[2] % config.block_size != 0:
        return f"key/value length must be divisible by {config.block_size}"

    if config.mode == "thrift":
        kv_blocks = key.shape[2] // config.block_size
        selected = resolve_top_k(
            kv_blocks,
            causal=config.causal,
            top_k=config.top_k,
            fraction=config.fp16_fraction,
        )
        if selected > 0 and kv_blocks > 2048:
            return "ThriftAttention currently supports at most 2048 selected KV blocks"

    return None


def _is_expected_scaling(scaling: object, expected: float) -> bool:
    if isinstance(scaling, torch.Tensor):
        if scaling.numel() != 1:
            return False
        try:
            scaling = float(scaling.detach().cpu().item())
        except Exception:
            return False
    try:
        return math.isclose(float(scaling), expected, rel_tol=1e-5, abs_tol=1e-7)
    except Exception:
        return False


def _fallback_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    config: AttentionConfig,
    reason: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raise RuntimeError(f"ThriftAttention fast path rejected this attention call: {reason}")
