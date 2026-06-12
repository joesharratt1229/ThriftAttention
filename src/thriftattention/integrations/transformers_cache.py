from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from ..config import AttentionConfig
from ..backends.single_query import _group_single_query, _ungroup_single_query
from ..quant.formats import get_quant_format
from ..quantization import nvfp4_quantize, nvfp4_quantize_transposed
from ..selection import resolve_top_k
from .._extension import get_extension


_ACTIVE_CACHE: ContextVar["ThriftAttentionCache | None"] = ContextVar(
    "thriftattention_active_cache",
    default=None,
)


def get_active_thriftattention_cache() -> "ThriftAttentionCache | None":
    return _ACTIVE_CACHE.get()


@contextmanager
def use_thriftattention_cache(cache: "ThriftAttentionCache") -> Iterator["ThriftAttentionCache"]:
    token = _ACTIVE_CACHE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE_CACHE.reset(token)


def is_thriftattention_cache(value: object) -> bool:
    return isinstance(value, ThriftAttentionCache)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _cache_position_start(cache_kwargs: dict[str, Any] | None, fallback: int) -> int:
    if not cache_kwargs or "cache_position" not in cache_kwargs:
        return fallback
    position = cache_kwargs["cache_position"]
    if position is None:
        return fallback
    if isinstance(position, torch.Tensor):
        if position.numel() == 0:
            return fallback
        flat = position.reshape(-1)
        if flat.numel() > 1:
            expected = torch.arange(
                int(flat[0].item()),
                int(flat[0].item()) + flat.numel(),
                device=flat.device,
                dtype=flat.dtype,
            )
            if not torch.equal(flat, expected):
                raise ValueError("ThriftAttentionCache requires contiguous cache_position values")
        return int(flat[0].item())
    return int(position)


@dataclass
class ThriftAttentionCacheLayer:
    max_cache_len: int | None = None

    k_fp16: torch.Tensor | None = None
    v_fp16: torch.Tensor | None = None
    k_packed: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None
    v_packed_t: torch.Tensor | None = None
    v_scale_t: torch.Tensor | None = None
    k_mean: torch.Tensor | None = None
    selected_blocks: torch.Tensor | None = None
    selection_local_scores: torch.Tensor | None = None
    selection_local_indices: torch.Tensor | None = None
    selection_done_counts: torch.Tensor | None = None
    seq_len: int = 0
    capacity: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("KV cache tensors must be [batch, heads, seq, head_dim]")
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")
        if key_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("KV cache tensors must be float16 or bfloat16")
        if key_states.dtype != value_states.dtype:
            raise ValueError("key_states and value_states must have the same dtype")

        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        start = _cache_position_start(cache_kwargs, self.seq_len)
        token_count = key_states.shape[2]
        end = start + token_count
        if self.max_cache_len is not None and end > self.max_cache_len:
            raise ValueError(
                f"ThriftAttentionCache length {end} exceeds max_cache_len={self.max_cache_len}"
            )

        self._ensure_capacity(key_states, end)
        assert self.k_fp16 is not None
        assert self.v_fp16 is not None

        self.k_fp16[:, :, start:end] = key_states
        self.v_fp16[:, :, start:end] = value_states
        self.seq_len = max(self.seq_len, end)
        if token_count:
            self._quantize_k_range(key_states, start, end)
            self._quantize_v_range(start, end)
            self._refresh_k_means(start, end)
        return self.key_view(), self.value_view()

    def key_view(self) -> torch.Tensor:
        if self.k_fp16 is None:
            raise RuntimeError("ThriftAttention cache layer has not been initialized")
        return self.k_fp16[:, :, : self.seq_len]

    def value_view(self) -> torch.Tensor:
        if self.v_fp16 is None:
            raise RuntimeError("ThriftAttention cache layer has not been initialized")
        return self.v_fp16[:, :, : self.seq_len]

    def packed_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.k_packed is None or self.k_scale is None:
            raise RuntimeError("ThriftAttention packed K cache has not been initialized")
        if self.v_packed_t is None or self.v_scale_t is None:
            raise RuntimeError("ThriftAttention packed V cache has not been initialized")
        # Physical packed views are rounded for the CUDA block loops. Logical
        # length remains seq_len; top-k selection below only sees complete
        # 64-token blocks, matching the decode workaround used in experiments.
        kernel_len = max(64, _round_up(self.seq_len, 64))
        v_len = _round_up(kernel_len, 128)
        return (
            self.k_packed[:, :, :kernel_len],
            self.v_packed_t[:, :, :, : v_len // 2],
            self.k_scale[:, :, :kernel_len],
            self.v_scale_t[:, :, :, : v_len // 16],
        )

    def select_key_blocks(
        self,
        q_grouped: torch.Tensor,
        *,
        top_k: int | None,
        fraction: float | None,
        block_size: int,
    ) -> torch.Tensor:
        # The trailing partial block stays in the packed FP4 path. Only complete
        # blocks are candidates for FP16 promotion.
        complete_blocks = self.seq_len // block_size
        selected_count = resolve_top_k(
            complete_blocks,
            causal=False,
            top_k=top_k,
            fraction=fraction,
        )
        batch, kv_heads, groups, _ = q_grouped.shape
        if selected_count == 0:
            return torch.empty(
                batch * kv_heads,
                0,
                device=q_grouped.device,
                dtype=torch.int32,
            )
        if self.k_mean is None:
            raise RuntimeError("ThriftAttention K block means have not been initialized")

        is_bf16 = q_grouped.dtype == torch.bfloat16
        if q_grouped.dtype != self.k_mean.dtype:
            raise ValueError("q_grouped and cached K means must have the same dtype")
        ext = get_extension()
        if complete_blocks <= 2048 and hasattr(ext, "single_query_key_mean_topk_into"):
            topk, local_scores, local_indices, done_counts = self._ensure_selection_workspace(
                q_grouped,
                complete_blocks,
                selected_count,
            )
            return ext.single_query_key_mean_topk_into(
                q_grouped,
                self.k_mean,
                topk,
                local_scores,
                local_indices,
                done_counts,
                complete_blocks,
                is_bf16,
            )
        if complete_blocks <= 2048 and hasattr(ext, "single_query_key_mean_topk"):
            return ext.single_query_key_mean_topk(
                q_grouped,
                self.k_mean,
                selected_count,
                complete_blocks,
                is_bf16,
            )

        k_mean = self.k_mean[:, :, :complete_blocks]
        scores = torch.matmul(q_grouped.float(), k_mean.float().transpose(-1, -2))
        scores = scores.amax(dim=2).reshape(batch * kv_heads, complete_blocks)
        return scores.topk(selected_count, dim=-1).indices.to(torch.int32).contiguous()

    def reorder(self, beam_idx: torch.Tensor) -> None:
        self._index_select_batch(beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._index_select_batch(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for name in self._tensor_names():
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.repeat_interleave(repeats, dim=0))

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(self.seq_len + max_length, 0)
        new_len = min(self.seq_len, max_length)
        if new_len < self.seq_len:
            self._zero_physical_tail(new_len)
        self.seq_len = new_len

    def reset(self) -> None:
        self.seq_len = 0

    def _zero_physical_tail(self, logical_len: int) -> None:
        if self.k_fp16 is None or self.v_fp16 is None:
            return
        key_end = min(max(64, _round_up(logical_len, 64)), self.k_fp16.shape[2])
        value_end = min(_round_up(key_end, 128), self.v_fp16.shape[2])
        if logical_len < key_end:
            self.k_fp16[:, :, logical_len:key_end].zero_()
            if self.k_packed is not None and self.k_scale is not None:
                self._quantize_k_range(
                    self.k_fp16[:, :, logical_len:key_end].contiguous(),
                    logical_len,
                    key_end,
                )
        if logical_len < value_end:
            self.v_fp16[:, :, logical_len:value_end].zero_()
            if self.v_packed_t is not None and self.v_scale_t is not None:
                self._quantize_v_range(logical_len, value_end)

    def _ensure_capacity(self, like: torch.Tensor, required: int) -> None:
        if self.k_fp16 is not None:
            if like.shape[0] != self.k_fp16.shape[0] or like.shape[1] != self.k_fp16.shape[1]:
                raise ValueError("ThriftAttentionCache batch/head shape changed unexpectedly")
            if like.dtype != self.k_fp16.dtype:
                raise ValueError("ThriftAttentionCache dtype changed unexpectedly")
            if required <= self.capacity:
                return

        batch, heads, _, head_dim = like.shape
        if self.max_cache_len is not None and self.capacity == 0:
            grown = self.max_cache_len
        else:
            grown = max(required, 64 if self.capacity == 0 else self.capacity * 2)
        if self.max_cache_len is not None:
            grown = min(max(grown, required), self.max_cache_len)
        new_capacity = _round_up(grown, 128)
        if self.max_cache_len is not None:
            new_capacity = min(new_capacity, _round_up(self.max_cache_len, 128))
        if new_capacity < required:
            raise ValueError("could not grow ThriftAttentionCache to required length")

        half_opts = dict(device=like.device, dtype=like.dtype)
        u8_opts = dict(device=like.device, dtype=torch.uint8)
        f8_opts = dict(device=like.device, dtype=torch.float8_e4m3fn)

        new_k = torch.zeros(batch, heads, new_capacity, head_dim, **half_opts)
        new_v = torch.zeros_like(new_k)
        new_k_packed = torch.zeros(batch, heads, new_capacity, head_dim // 2, **u8_opts)
        new_k_scale = torch.zeros(batch, heads, new_capacity, head_dim // 16, **f8_opts)
        new_v_packed_t = torch.zeros(batch, heads, head_dim, new_capacity // 2, **u8_opts)
        new_v_scale_t = torch.zeros(batch, heads, head_dim, new_capacity // 16, **f8_opts)
        new_k_mean = torch.zeros(batch, heads, new_capacity // 64, head_dim, **half_opts)

        if self.k_fp16 is not None:
            old = self.capacity
            new_k[:, :, :old] = self.k_fp16
            new_v[:, :, :old] = self.v_fp16
            new_k_packed[:, :, :old] = self.k_packed
            new_k_scale[:, :, :old] = self.k_scale
            new_v_packed_t[:, :, :, : old // 2] = self.v_packed_t
            new_v_scale_t[:, :, :, : old // 16] = self.v_scale_t
            new_k_mean[:, :, : old // 64] = self.k_mean

        self.k_fp16 = new_k
        self.v_fp16 = new_v
        self.k_packed = new_k_packed
        self.k_scale = new_k_scale
        self.v_packed_t = new_v_packed_t
        self.v_scale_t = new_v_scale_t
        self.k_mean = new_k_mean
        self.capacity = new_capacity

    def _quantize_k_range(self, key_states: torch.Tensor, start: int, end: int) -> None:
        assert self.k_packed is not None
        assert self.k_scale is not None
        k_packed, k_scale = nvfp4_quantize(
            key_states,
            is_bf16=key_states.dtype == torch.bfloat16,
        )
        self.k_packed[:, :, start:end] = k_packed
        self.k_scale[:, :, start:end] = k_scale

    def _quantize_v_range(self, start: int, end: int) -> None:
        assert self.v_fp16 is not None
        assert self.v_packed_t is not None
        assert self.v_scale_t is not None
        begin = (start // 16) * 16
        finish = _round_up(end, 16)
        v_packed_t, v_scale_t = nvfp4_quantize_transposed(
            self.v_fp16[:, :, begin:finish].contiguous(),
            is_bf16=self.v_fp16.dtype == torch.bfloat16,
        )
        self.v_packed_t[:, :, :, begin // 2 : finish // 2] = v_packed_t[
            :, :, :, : (finish - begin) // 2
        ]
        self.v_scale_t[:, :, :, begin // 16 : finish // 16] = v_scale_t[
            :, :, :, : (finish - begin) // 16
        ]

    def _refresh_k_means(self, start: int, end: int) -> None:
        assert self.k_fp16 is not None
        assert self.k_mean is not None
        first = start // 64
        last = min(end // 64, self.seq_len // 64)
        if last <= first:
            return
        block_start = first * 64
        block_end = last * 64
        batch, heads, _, head_dim = self.k_fp16.shape
        self.k_mean[:, :, first:last] = (
            self.k_fp16[:, :, block_start:block_end]
            .reshape(batch, heads, last - first, 64, head_dim)
            .float()
            .mean(dim=3)
            .to(self.k_mean.dtype)
        )

    def _ensure_selection_workspace(
        self,
        q_grouped: torch.Tensor,
        complete_blocks: int,
        selected_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, kv_heads, _, _ = q_grouped.shape
        flat_heads = batch * kv_heads
        chunk_count = _round_up(complete_blocks, 64) // 64
        local_count = min(selected_count, 64)
        topk_shape = (flat_heads, selected_count)
        local_shape = (flat_heads, chunk_count, local_count)
        done_shape = (flat_heads,)

        if (
            self.selected_blocks is None
            or tuple(self.selected_blocks.shape) != topk_shape
            or self.selected_blocks.device != q_grouped.device
        ):
            self.selected_blocks = torch.empty(
                topk_shape,
                device=q_grouped.device,
                dtype=torch.int32,
            )
        if (
            self.selection_local_scores is None
            or tuple(self.selection_local_scores.shape) != local_shape
            or self.selection_local_scores.device != q_grouped.device
        ):
            self.selection_local_scores = torch.empty(
                local_shape,
                device=q_grouped.device,
                dtype=torch.float32,
            )
        if (
            self.selection_local_indices is None
            or tuple(self.selection_local_indices.shape) != local_shape
            or self.selection_local_indices.device != q_grouped.device
        ):
            self.selection_local_indices = torch.empty(
                local_shape,
                device=q_grouped.device,
                dtype=torch.int32,
            )
        if (
            self.selection_done_counts is None
            or tuple(self.selection_done_counts.shape) != done_shape
            or self.selection_done_counts.device != q_grouped.device
        ):
            self.selection_done_counts = torch.empty(
                done_shape,
                device=q_grouped.device,
                dtype=torch.int32,
            )

        return (
            self.selected_blocks,
            self.selection_local_scores,
            self.selection_local_indices,
            self.selection_done_counts,
        )

    def _index_select_batch(self, indices: torch.Tensor) -> None:
        for name in self._tensor_names():
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, indices.to(tensor.device)))

    @staticmethod
    def _tensor_names() -> tuple[str, ...]:
        return (
            "k_fp16",
            "v_fp16",
            "k_packed",
            "k_scale",
            "v_packed_t",
            "v_scale_t",
            "k_mean",
        )


class ThriftAttentionCache:
    """HF-style cache with FP16 and packed FP4 state for ThriftAttention decode."""

    def __init__(
        self,
        *,
        config: AttentionConfig | None = None,
        max_cache_len: int | None = None,
    ) -> None:
        # Transformers' Cache base class constructor is not stable across
        # versions. This class implements the cache protocol directly, so avoid
        # calling the base __init__ and keep construction version-agnostic.
        self.config = config or AttentionConfig()
        self._max_cache_len = max_cache_len
        self.layers: list[ThriftAttentionCacheLayer] = []
        self._seen_tokens = 0
        self.prefill_real_seq_len: int | None = None

    @classmethod
    def from_model(
        cls,
        model: object,
        *,
        config: AttentionConfig | None = None,
        max_cache_len: int | None = None,
    ) -> "ThriftAttentionCache":
        if max_cache_len is None:
            model_config = getattr(model, "config", None)
            max_cache_len = getattr(model_config, "max_position_embeddings", None)
        return cls(config=config, max_cache_len=max_cache_len)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def max_cache_len(self) -> int | None:
        return self._max_cache_len

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        return [False for _ in self.layers]

    @property
    def max_batch_size(self) -> int | None:
        for layer in self.layers:
            if layer.k_fp16 is not None:
                return int(layer.k_fp16.shape[0])
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int | None = None,
        cache_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx is None:
            raise ValueError("ThriftAttentionCache.update requires layer_idx")
        layer = self.layer(layer_idx)
        key, value = layer.update(key_states, value_states, cache_kwargs)
        if layer_idx == 0:
            self._seen_tokens = layer.seq_len
        return key, value

    def layer(self, layer_idx: int) -> ThriftAttentionCacheLayer:
        while len(self.layers) <= layer_idx:
            self.layers.append(ThriftAttentionCacheLayer(max_cache_len=self._max_cache_len))
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].seq_len

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].seq_len + query_length, 0

    def get_max_cache_shape(self) -> int | None:
        return self._max_cache_len

    def get_max_length(self) -> int | None:
        return self._max_cache_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        previous = self.get_seq_length(layer_idx)
        if self._max_cache_len is None:
            return previous
        if previous + new_seq_length <= self._max_cache_len:
            return previous
        return max(self._max_cache_len - new_seq_length, 0)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer in self.layers:
            layer.reorder(beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)
        self._seen_tokens = self.get_seq_length(0)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((layer.key_view(), layer.value_view()) for layer in self.layers)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layer(layer_idx)
        return layer.key_view(), layer.value_view()

    def __len__(self) -> int:
        return len(self.layers)


def cached_decode_attention(
    query: torch.Tensor,
    cache: ThriftAttentionCache,
    layer_idx: int,
    config: AttentionConfig,
) -> torch.Tensor:
    if config.quant_format != "nvfp4":
        raise NotImplementedError("generation cache currently supports quant_format='nvfp4' only")
    layer = cache.layer(layer_idx)
    if layer.seq_len == 0:
        raise RuntimeError("ThriftAttention decode cache is empty")

    q = query.contiguous()
    is_bf16 = q.dtype == torch.bfloat16
    kv_heads = layer.key_view().shape[1]
    q_grouped = _group_single_query(q, kv_heads)
    quant_format = get_quant_format(config.quant_format)
    q_packed, q_scale = quant_format.quantize(q_grouped, is_bf16=is_bf16)
    k_packed, v_packed_t, k_scale, v_scale_t = layer.packed_views()

    if config.method == "fp4":
        out = get_extension().fp4_attention_single_query_nvfp4_packed(
            q_packed,
            k_packed,
            v_packed_t,
            q_scale,
            k_scale,
            v_scale_t,
            is_bf16,
        )
        return _ungroup_single_query(out, q.shape[1])

    if config.method != "thrift":
        raise ValueError(f"unsupported ThriftAttention method {config.method!r}")

    selected_blocks = layer.select_key_blocks(
        q_grouped,
        top_k=config.top_k,
        fraction=config.fraction,
        block_size=config.block_size,
    )
    out = get_extension().thrift_attention_single_query_nvfp4_packed(
        q_grouped,
        layer.key_view(),
        layer.value_view(),
        selected_blocks,
        q_packed,
        k_packed,
        v_packed_t,
        q_scale,
        k_scale,
        v_scale_t,
        is_bf16,
    )
    return _ungroup_single_query(out, q.shape[1])


def cached_prefill_attention(
    query: torch.Tensor,
    cache: ThriftAttentionCache,
    layer_idx: int,
    config: AttentionConfig,
) -> torch.Tensor:
    if config.quant_format != "nvfp4":
        raise NotImplementedError("generation cache currently supports quant_format='nvfp4' only")
    layer = cache.layer(layer_idx)
    q = query.contiguous()
    is_bf16 = q.dtype == torch.bfloat16
    k_fp16 = layer.key_view().contiguous()
    v_fp16 = layer.value_view().contiguous()
    quant_format = get_quant_format(config.quant_format)
    q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t = quant_format.quantize_qkv(
        q,
        k_fp16,
        v_fp16,
        is_bf16=is_bf16,
    )
    ext = get_extension()

    if config.method == "fp4":
        fn = (
            ext.fp4_attention_causal_nvfp4_packed
            if config.causal
            else ext.fp4_attention_noncausal_nvfp4_packed
        )
        return fn(
            q_packed,
            k_packed,
            v_packed_t,
            q_scale,
            k_scale,
            v_scale_t,
            is_bf16,
            config.exp_approx,
        )

    if config.method != "thrift":
        raise ValueError(f"unsupported ThriftAttention method {config.method!r}")

    selected_blocks = _select_block_pairs_from_cached_means(
        q,
        layer,
        config,
        real_q_len=cache.prefill_real_seq_len,
        is_bf16=is_bf16,
    )
    fn = (
        ext.thrift_attention_causal_nvfp4_packed
        if config.causal
        else ext.thrift_attention_noncausal_nvfp4_packed
    )
    return fn(
        q,
        k_fp16,
        v_fp16,
        selected_blocks,
        q_packed,
        k_packed,
        v_packed_t,
        q_scale,
        k_scale,
        v_scale_t,
        is_bf16,
    )


def _select_block_pairs_from_cached_means(
    q: torch.Tensor,
    layer: ThriftAttentionCacheLayer,
    config: AttentionConfig,
    real_q_len: int | None = None,
    is_bf16: bool = False,
) -> torch.Tensor:
    if layer.k_mean is None:
        raise RuntimeError("ThriftAttention K block means have not been initialized")
    block_size = config.block_size
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = layer.key_view().shape[1]
    num_q_blocks = q_len // block_size
    num_kv_blocks = layer.seq_len // block_size
    selected_count = resolve_top_k(
        num_kv_blocks,
        causal=config.causal,
        top_k=config.top_k,
        fraction=config.fraction,
    )
    if selected_count == 0:
        return torch.empty(
            batch * q_heads,
            num_q_blocks,
            0,
            device=q.device,
            dtype=torch.int32,
        )

    if real_q_len is None or real_q_len >= q_len:
        q_mean = (
            q.reshape(batch, q_heads, num_q_blocks, block_size, head_dim)
            .float()
            .mean(dim=3)
            .to(torch.bfloat16 if is_bf16 else torch.float16)
            .contiguous()
        )
    else:
        real_q_len = max(0, int(real_q_len))
        q_blocks = q.reshape(batch, q_heads, num_q_blocks, block_size, head_dim).float()
        counts = torch.full((num_q_blocks,), block_size, device=q.device, dtype=torch.float32)
        full_blocks = real_q_len // block_size
        tail = real_q_len % block_size
        if full_blocks < num_q_blocks:
            counts[full_blocks:] = 0
            if tail:
                counts[full_blocks] = tail
                q_blocks[:, :, full_blocks, tail:] = 0
                q_blocks[:, :, full_blocks + 1 :] = 0
            else:
                q_blocks[:, :, full_blocks:] = 0
        q_mean = (
            q_blocks.sum(dim=3) / counts.clamp_min(1).view(1, 1, -1, 1)
        ).to(torch.bfloat16 if is_bf16 else torch.float16).contiguous()
    k_mean = layer.k_mean[:, :, :num_kv_blocks]
    if q_heads != kv_heads:
        k_mean = k_mean.repeat_interleave(q_heads // kv_heads, dim=1).contiguous()

    if num_kv_blocks <= 2048:
        return get_extension().block_mean_topk(q_mean, k_mean, selected_count, config.causal, is_bf16)

    scores = (
        q_mean.reshape(batch * q_heads, num_q_blocks, head_dim).float()
        @ k_mean.reshape(batch * q_heads, num_kv_blocks, head_dim).float().transpose(-1, -2)
    )
    if config.causal:
        mask = torch.triu(
            torch.ones(num_q_blocks, num_kv_blocks, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    indices = scores.topk(selected_count, dim=-1).indices.to(torch.int32)
    if config.causal:
        valid_counts = torch.arange(1, num_q_blocks + 1, device=q.device).clamp(max=num_kv_blocks)
        ranks = torch.arange(selected_count, device=q.device)
        indices.masked_fill_(ranks.view(1, 1, -1) >= valid_counts.view(1, -1, 1), -1)
    return indices.contiguous()
