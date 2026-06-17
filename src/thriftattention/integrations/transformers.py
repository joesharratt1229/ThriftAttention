from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from ..config import (
    AttentionBackendName,
    AttentionConfig,
    AttentionImplementation,
    AttentionMethod,
    FallbackBackend,
    QuantFormatName,
    SelectionMethod,
)
from ..functional import attention as thrift_attention
from ..selection import resolve_top_k
from .transformers_cache import (
    ThriftAttentionCache,
    cached_decode_attention,
    cached_prefill_attention,
    get_active_thriftattention_cache,
    use_thriftattention_cache,
)


DEFAULT_TRANSFORMERS_ATTENTION_NAME = "thrift_attention"
_REGISTERED_CONFIGS: dict[str, AttentionConfig] = {}


@dataclass(frozen=True)
class TransformersAttentionConfig:
    name: str = DEFAULT_TRANSFORMERS_ATTENTION_NAME
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
    microblock_p: bool = False

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            method=_validate_choice("method", self.method, ("thrift", "fp4")),
            causal=bool(self.causal),
            selection=_validate_choice("selection", self.selection, ("block_mean",)),
            fraction=_validate_fraction(self.fraction),
            top_k=_validate_top_k(self.top_k),
            block_size=_validate_block_size(self.block_size),
            quant_format=_validate_choice("quant_format", self.quant_format, ("nvfp4",)),
            backend=_validate_choice("backend", self.backend, ("auto", "sm120")),
            implementation=_validate_choice(
                "implementation",
                self.implementation,
                ("auto", "tiled", "single_query"),
            ),
            fallback=_validate_choice("fallback", self.fallback, ("error",)),
            exp_approx=bool(self.exp_approx),
            microblock_p=bool(self.microblock_p),
        )


@dataclass(frozen=True)
class TransformersCacheInputs:
    input_ids: torch.Tensor
    cache_position: torch.Tensor
    past_key_values: ThriftAttentionCache
    prompt_length: int
    padded_length: int
    padding: int
    config: AttentionConfig

    def activate(self) -> Any:
        return use_thriftattention_cache(self.past_key_values)

    def trim_padding(self) -> None:
        if self.padding:
            self.past_key_values.crop(self.prompt_length)


def register_transformers_attention(config: TransformersAttentionConfig | None = None) -> str:
    config = config or TransformersAttentionConfig()
    name = _validate_attention_name(config.name)
    _register_transformers_attention_impl(name, config.attention_config())
    return name


def get_registered_transformers_attention_config(name: str) -> AttentionConfig | None:
    return _REGISTERED_CONFIGS.get(name)


def prepare_transformers_generation_cache(
    model: object,
    input_ids: torch.Tensor | list[int] | tuple[int, ...],
    *,
    config: AttentionConfig | TransformersAttentionConfig | None = None,
    max_new_tokens: int = 0,
    device: torch.device | str | None = None,
    pad_to_block: bool = True,
) -> TransformersCacheInputs:
    active_config = _coerce_attention_config(config) if config is not None else _model_config(model)
    prompt_length = _input_ids_length(input_ids)
    active_config = _generation_cache_config(active_config, prompt_length)

    if isinstance(input_ids, torch.Tensor):
        encoded = input_ids.to(device=device) if device is not None else input_ids
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        elif encoded.ndim != 2:
            raise ValueError("input_ids tensor must be rank 1 or rank 2")
    else:
        encoded = torch.tensor([list(input_ids)], dtype=torch.long, device=device)

    padding = (-prompt_length) % active_config.block_size if pad_to_block else 0
    if padding:
        pad = torch.zeros(
            encoded.shape[0],
            padding,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        encoded = torch.cat([encoded, pad], dim=1)

    cache_position = torch.arange(encoded.shape[1], device=encoded.device, dtype=torch.long)
    past = ThriftAttentionCache.from_model(
        model,
        config=active_config,
        max_cache_len=encoded.shape[1] + int(max_new_tokens),
    )
    past.prefill_real_seq_len = prompt_length
    return TransformersCacheInputs(
        input_ids=encoded,
        cache_position=cache_position,
        past_key_values=past,
        prompt_length=prompt_length,
        padded_length=int(encoded.shape[1]),
        padding=padding,
        config=active_config,
    )


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

    output = thrift_attention(query, key, value, config=config)
    return output.transpose(1, 2).contiguous(), None


def _register_transformers_attention_impl(name: str, config: AttentionConfig) -> None:
    registry = _get_attention_registry()
    _REGISTERED_CONFIGS[name] = config
    _register_once(registry, name, thriftattention_forward)
    _register_attention_mask(name)


def _get_attention_registry() -> Any:
    try:
        from transformers import AttentionInterface
    except Exception:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as AttentionInterface
        except Exception as exc:
            raise ImportError(
                "register_transformers_attention() requires Hugging Face Transformers "
                "with AttentionInterface support"
            ) from exc
    return AttentionInterface


def _register_attention_mask(name: str = DEFAULT_TRANSFORMERS_ATTENTION_NAME) -> None:
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
    except Exception:
        return
    _register_once(AttentionMaskInterface, name, sdpa_mask)


def _register_once(registry: Any, name: str, fn: Callable[..., Any]) -> None:
    try:
        registry.register(name, fn)
    except ValueError as exc:
        if "already" not in str(exc).lower():
            raise


def _module_config(module: torch.nn.Module) -> AttentionConfig:
    cache = get_active_thriftattention_cache()
    if cache is not None:
        return cache.config

    name = _module_attn_implementation(module)
    if name is not None and name in _REGISTERED_CONFIGS:
        return _REGISTERED_CONFIGS[name]

    raise RuntimeError(
        "ThriftAttention was called for an unregistered Transformers attention "
        "implementation. Call register_transformers_attention(...) before loading "
        "or setting a model with attn_implementation=<registered name>."
    )


def _model_config(model: object) -> AttentionConfig:
    name = _module_attn_implementation(model)
    if name is not None and name in _REGISTERED_CONFIGS:
        return _REGISTERED_CONFIGS[name]
    raise RuntimeError(
        "could not infer a registered ThriftAttention config from model.config; "
        "pass config= explicitly"
    )


def _module_attn_implementation(module: object) -> str | None:
    config = getattr(module, "config", None)
    name = getattr(config, "_attn_implementation", None)
    return name if isinstance(name, str) else None


def _coerce_attention_config(config: AttentionConfig | TransformersAttentionConfig) -> AttentionConfig:
    if isinstance(config, TransformersAttentionConfig):
        return config.attention_config()
    if isinstance(config, AttentionConfig):
        return config
    raise TypeError("config must be an AttentionConfig or TransformersAttentionConfig")


def _generation_cache_config(config: AttentionConfig, prompt_length: int) -> AttentionConfig:
    if config.method != "thrift" or config.top_k is not None:
        return config
    blocks = max(prompt_length // config.block_size, 1)
    top_k = resolve_top_k(blocks, causal=True, fraction=config.fraction)
    return replace(config, top_k=top_k)


def _input_ids_length(input_ids: torch.Tensor | list[int] | tuple[int, ...]) -> int:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
        if input_ids.ndim == 2:
            return int(input_ids.shape[1])
        raise ValueError("input_ids tensor must be rank 1 or rank 2")
    return len(input_ids)


def _validate_attention_name(name: str) -> str:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError(
            "attention implementation name must be a non-empty string without "
            "surrounding whitespace"
        )
    return name


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        formatted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {formatted}, got {value!r}")
    return value


def _validate_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {value!r}")
    return value


def _validate_top_k(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"top_k must be non-negative, got {value!r}")
    return value


def _validate_block_size(value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"block_size must be positive, got {value!r}")
    return value


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
        return "ThriftAttention registered attention does not run while the attention module is in training mode"
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
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            return "ThriftAttention fast path requires float16 or bfloat16 query/key/value tensors"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "query/key/value tensors must have the same dtype"

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

    if config.method == "thrift":
        kv_blocks = key.shape[2] // config.block_size
        selected = resolve_top_k(
            kv_blocks,
            causal=config.causal,
            top_k=config.top_k,
            fraction=config.fraction,
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
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            return "cached decode requires float16 or bfloat16 query/key/value tensors"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "query/key/value tensors must have the same dtype"

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
