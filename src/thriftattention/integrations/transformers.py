from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..config import AttentionConfig
from ..functional import attention as thrift_attention
from ..functional import fp4_attention
from ..selection import resolve_top_k
from .transformers_cache import (
    ThriftAttentionCache,
    cached_decode_attention,
    cached_prefill_attention,
    get_active_thriftattention_cache,
    is_thriftattention_cache,
    use_thriftattention_cache,
)


BACKEND_NAME = "thriftattention"
_CONFIG_ATTR = "_thriftattention_config"
_GENERATION_CONFIG_ATTR = "_thriftattention_generation_config"
_ORIGINAL_ATTN_IMPL_ATTR = "_thriftattention_original_attn_implementation"
_ORIGINAL_GENERATE_ATTR = "_thriftattention_original_generate"
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
    if config.patch_generation:
        _patch_generate(model, config)
    setattr(model, _PATCHED_ATTR, True)
    return model


def unpatch_hf_model(model: object) -> object:
    if hasattr(model, _ORIGINAL_GENERATE_ATTR):
        setattr(model, "generate", getattr(model, _ORIGINAL_GENERATE_ATTR))
        delattr(model, _ORIGINAL_GENERATE_ATTR)
    if hasattr(model, _GENERATION_CONFIG_ATTR):
        delattr(model, _GENERATION_CONFIG_ATTR)

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
    cached_decode_rejection = _cached_decode_rejection_reason(
        module,
        query,
        key,
        value,
        attention_mask,
        config,
        kwargs,
    )
    if cached_decode_rejection is None:
        cache = get_active_thriftattention_cache()
        layer_idx = getattr(module, "layer_idx")
        assert cache is not None
        output = cached_decode_attention(query, cache, int(layer_idx), config)
        return output.transpose(1, 2).contiguous(), None

    cached_prefill_rejection = _cached_prefill_rejection_reason(
        module,
        query,
        key,
        value,
        attention_mask,
        config,
        kwargs,
    )
    if cached_prefill_rejection is None:
        cache = get_active_thriftattention_cache()
        layer_idx = getattr(module, "layer_idx")
        assert cache is not None
        output = cached_prefill_attention(query, cache, int(layer_idx), config)
        return output.transpose(1, 2).contiguous(), None

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


def _patch_generate(model: object, config: AttentionConfig) -> None:
    setattr(model, _GENERATION_CONFIG_ATTR, config)
    if hasattr(model, _ORIGINAL_GENERATE_ATTR):
        return
    original_generate = getattr(model, "generate", None)
    if not callable(original_generate):
        return

    def thriftattention_generate(*args: Any, **kwargs: Any) -> Any:
        active_config = getattr(model, _GENERATION_CONFIG_ATTR, config)
        return _generate_with_thriftattention_cache(
            model,
            original_generate,
            active_config,
            *args,
            **kwargs,
        )

    setattr(model, _ORIGINAL_GENERATE_ATTR, original_generate)
    setattr(model, "generate", thriftattention_generate)


def _generate_with_thriftattention_cache(
    model: object,
    original_generate: Callable[..., Any],
    config: AttentionConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    explicit_cache = kwargs.get("past_key_values", None)
    if is_thriftattention_cache(explicit_cache):
        with use_thriftattention_cache(explicit_cache):
            return original_generate(*args, **kwargs)

    if not _should_inject_generation_cache(model, kwargs):
        return original_generate(*args, **kwargs)

    cache = ThriftAttentionCache.from_model(
        model,
        config=config,
        max_cache_len=_infer_generation_cache_length(model, args, kwargs),
    )
    patched_kwargs = dict(kwargs)
    patched_kwargs["past_key_values"] = cache
    patched_kwargs.setdefault("use_cache", True)
    with use_thriftattention_cache(cache):
        return original_generate(*args, **patched_kwargs)


def _should_inject_generation_cache(model: object, kwargs: dict[str, Any]) -> bool:
    if getattr(model, "training", False):
        return False
    model_config = getattr(model, "config", None)
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        return False
    if kwargs.get("use_cache", True) is False:
        return False
    if kwargs.get("past_key_values", None) is not None:
        return False
    if kwargs.get("cache_implementation", None) is not None:
        return False
    return True


def _infer_generation_cache_length(
    model: object,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> int | None:
    max_length = kwargs.get("max_length", None)
    if max_length is not None:
        return int(max_length)

    prompt_len = _infer_prompt_length(args, kwargs)
    max_new_tokens = kwargs.get("max_new_tokens", None)
    if prompt_len is not None and max_new_tokens is not None:
        return int(prompt_len + max_new_tokens)

    generation_config = kwargs.get("generation_config", getattr(model, "generation_config", None))
    max_length = getattr(generation_config, "max_length", None)
    if max_length is not None:
        return int(max_length)

    model_config = getattr(model, "config", None)
    max_positions = getattr(model_config, "max_position_embeddings", None)
    if max_positions is not None:
        return int(max_positions)
    return None


def _infer_prompt_length(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    inputs = kwargs.get("input_ids", None)
    if inputs is None and args:
        inputs = args[0]
    if isinstance(inputs, torch.Tensor) and inputs.ndim >= 2:
        return int(inputs.shape[1])
    inputs_embeds = kwargs.get("inputs_embeds", None)
    if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.ndim >= 2:
        return int(inputs_embeds.shape[1])
    return None


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

    if query.shape[2] != 1 and query.shape[2] % config.block_size != 0:
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
        if query.shape[2] != 1 and selected > 0 and kv_blocks > 2048:
            return "ThriftAttention currently supports at most 2048 selected KV blocks"

    return None


def _cached_decode_rejection_reason(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: AttentionConfig,
    kwargs: dict[str, Any],
) -> str | None:
    cache = get_active_thriftattention_cache()
    if cache is None:
        return "no active ThriftAttention generation cache"
    if not config.causal:
        return "cached decode only supports causal attention"
    if getattr(module, "training", False):
        return "cached decode does not run while the attention module is in training mode"
    if kwargs.get("output_attentions", False):
        return "cached decode does not return attention weights"
    if attention_mask is not None:
        return "cached decode does not support explicit attention masks"

    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        return "cached decode requires attention modules to expose layer_idx"
    try:
        layer_idx = int(layer_idx)
    except Exception:
        return "cached decode requires an integer layer_idx"
    if layer_idx >= len(cache.layers) or cache.get_seq_length(layer_idx) == 0:
        return "cached decode cache layer has not been populated"

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if tensor.ndim != 4:
            return f"{name} must be shaped [batch, heads, seq, head_dim]"
        if not tensor.is_cuda:
            return "cached decode requires CUDA tensors"
        if tensor.dtype != torch.float16:
            return "cached decode requires float16 query/key/value tensors"

    if query.shape[2] != 1:
        return "cached decode requires query sequence length 1"
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return "query/key/value batch dimensions must match"
    if key.shape[1] != value.shape[1]:
        return "key/value head dimensions must match"
    if query.shape[1] % key.shape[1] != 0:
        return "query heads must be divisible by KV heads"
    if key.shape[3] != query.shape[3] or value.shape[3] != query.shape[3]:
        return "query/key/value head_dim must match"
    if query.shape[3] not in (64, 128):
        return f"ThriftAttention supports head_dim 64 or 128, got {query.shape[3]}"

    groups = query.shape[1] // key.shape[1]
    if groups > 16:
        return "single-query kernels support at most 16 Q heads per KV head"

    expected_scaling = query.shape[3] ** -0.5
    scaling = kwargs.get("scaling", expected_scaling)
    if not _is_expected_scaling(scaling, expected_scaling):
        return "ThriftAttention kernels use the default 1/sqrt(head_dim) scaling"

    if config.block_size != 64:
        return "cached decode currently requires 64-token KV blocks"
    return None


def _cached_prefill_rejection_reason(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: AttentionConfig,
    kwargs: dict[str, Any],
) -> str | None:
    cache = get_active_thriftattention_cache()
    if cache is None:
        return "no active ThriftAttention generation cache"
    if query.shape[2] == 1:
        return "cached prefill requires query sequence length greater than 1"

    rejection = _fast_path_rejection_reason(
        module,
        query,
        key,
        value,
        attention_mask,
        config,
        kwargs,
    )
    if rejection is not None:
        return rejection

    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        return "cached prefill requires attention modules to expose layer_idx"
    try:
        layer_idx = int(layer_idx)
    except Exception:
        return "cached prefill requires an integer layer_idx"
    if layer_idx >= len(cache.layers) or cache.get_seq_length(layer_idx) == 0:
        return "cached prefill cache layer has not been populated"
    if cache.get_seq_length(layer_idx) != key.shape[2]:
        return "cached prefill cache length does not match attention key length"
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
