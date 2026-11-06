from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from thriftattention.config import AttentionConfig


def _stub_attention_registry(monkeypatch, hf):
    registered = {}

    class Registry:
        @staticmethod
        def register(name, fn):
            registered[name] = fn

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda name=None: None)
    return registered


def test_register_transformers_attention_registers_custom_name(monkeypatch):
    from thriftattention.integrations import transformers as hf

    registered = {}
    masks = {}

    class Registry:
        @staticmethod
        def register(name, fn):
            registered[name] = fn

    def register_mask(name=None):
        masks[name] = True

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", register_mask)

    name = hf.register_transformers_attention(
        hf.TransformersAttentionConfig(name="unit_thrift_attention", method="fp4")
    )

    assert name == "unit_thrift_attention"
    assert registered["unit_thrift_attention"] is hf.thriftattention_forward
    assert masks["unit_thrift_attention"] is True
    assert hf.get_registered_transformers_attention_config("unit_thrift_attention").method == "fp4"


def test_registered_attention_config_is_used_without_model_mutation(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_registered"))

    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_registered"),
        training=False,
        is_causal=True,
        num_key_value_groups=1,
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_unregistered_attention_reports_registration_error():
    from thriftattention.integrations.transformers import thriftattention_forward

    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="missing"))
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="register_transformers_attention"):
        thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_prepare_transformers_generation_cache_uses_registered_model_config(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(
        hf.TransformersAttentionConfig(name="unit_cache", fraction=0.05)
    )

    child = SimpleNamespace()

    class Model:
        config = SimpleNamespace(_attn_implementation="unit_cache", max_position_embeddings=1024)

        def modules(self):
            return [self, child]

    model = Model()
    prepared = hf.prepare_transformers_generation_cache(
        model,
        [1, 2, 3],
        max_new_tokens=5,
        device="cpu",
    )

    assert prepared.input_ids.shape == (1, 64)
    assert prepared.cache_position.tolist() == list(range(64))
    assert prepared.prompt_length == 3
    assert prepared.padding == 61
    assert prepared.past_key_values.max_cache_len == 69
    assert prepared.past_key_values.prefill_real_seq_len == 3
    assert prepared.config.top_k == 1
    assert vars(child) == {}


def test_active_cache_config_overrides_registered_config(monkeypatch):
    from thriftattention.integrations import transformers as hf
    from thriftattention.integrations.transformers_cache import (
        ThriftAttentionCache,
        use_thriftattention_cache,
    )

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_cache_override", method="fp4"))
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="unit_cache_override"))
    cache = ThriftAttentionCache(config=AttentionConfig(method="thrift", top_k=7))

    with use_thriftattention_cache(cache):
        resolved = hf._module_config(module)

    assert resolved.method == "thrift"
    assert resolved.top_k == 7


def test_cached_decode_local_selection_uses_local_extension(monkeypatch):
    from thriftattention.integrations import transformers_cache as cache_mod

    calls = []

    class Extension:
        @staticmethod
        def single_query_local_topk(q_grouped, selected_count, num_kv_blocks):
            calls.append((tuple(q_grouped.shape), selected_count, num_kv_blocks))
            return torch.tensor([[2, 3]], dtype=torch.int32)

    monkeypatch.setattr(cache_mod, "get_extension", lambda: Extension)
    q_grouped = torch.empty(1, 1, 1, 64)
    layer = SimpleNamespace(seq_len=256)

    selected = cache_mod.select_cached_decode_blocks(
        q_grouped,
        layer,
        AttentionConfig(selection="local", top_k=2),
    )

    assert selected.tolist() == [[2, 3]]
    assert calls == [((1, 1, 1, 64), 2, 4)]


def test_cached_prefill_local_selection_uses_local_policy(monkeypatch):
    from thriftattention.integrations import transformers_cache as cache_mod
    from thriftattention.selection import local

    calls = []

    class Extension:
        @staticmethod
        def local_block_topk(q, num_kv_blocks, selected_count, causal, block_size):
            calls.append((tuple(q.shape), num_kv_blocks, selected_count, causal, block_size))
            return torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32)

    monkeypatch.setattr(local, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(local, "get_extension", lambda: Extension)
    q = torch.empty(1, 1, 128, 64, dtype=torch.float16)
    k = torch.empty(1, 1, 256, 64, dtype=torch.float16)
    layer = SimpleNamespace(key_view=lambda: k)

    selected = cache_mod.select_cached_prefill_blocks(
        q,
        k,
        layer,
        AttentionConfig(selection="local", top_k=2),
    )

    assert selected.tolist() == [[[0, 1], [1, 2]]]
    assert calls == [((1, 1, 128, 64), 4, 2, True, 64)]


def test_cache_crop_zeroes_rounded_physical_tail():
    from thriftattention.integrations.transformers_cache import ThriftAttentionCacheLayer

    layer = ThriftAttentionCacheLayer()
    layer.seq_len = 128
    layer.capacity = 128
    layer.k_fp16 = torch.ones(1, 1, 128, 64)
    layer.v_fp16 = torch.ones(1, 1, 128, 64)
    layer.k_packed = torch.empty(1)
    layer.k_scale = torch.empty(1)
    layer.v_packed_t = torch.empty(1)
    layer.v_scale_t = torch.empty(1)
    calls = []

    def quantize_k(key_states, start, end):
        calls.append(("k", tuple(key_states.shape), start, end))

    def quantize_v(start, end):
        calls.append(("v", start, end))

    layer._quantize_k_range = quantize_k
    layer._quantize_v_range = quantize_v

    layer.crop(65)

    assert layer.seq_len == 65
    assert torch.all(layer.k_fp16[:, :, :65] == 1)
    assert torch.all(layer.v_fp16[:, :, :65] == 1)
    assert torch.all(layer.k_fp16[:, :, 65:128] == 0)
    assert torch.all(layer.v_fp16[:, :, 65:128] == 0)
    assert calls == [("k", (1, 1, 63, 64), 65, 128), ("v", 65, 128)]


def test_adapter_fast_path_transposes_thrift_output(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_fast_thrift"))
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_fast_thrift"),
        training=False,
        is_causal=True,
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["config"].selection == "block_mean"
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None


def test_adapter_fast_path_passes_noncausal_mode(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(
        hf.TransformersAttentionConfig(name="unit_fast_noncausal", causal=False)
    )
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_fast_noncausal"),
        training=False,
        is_causal=False,
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["config"].causal is False
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None


def test_adapter_pads_unaligned_causal_prefill(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_unaligned_prefill"))
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_unaligned_prefill"),
        training=False,
        is_causal=True,
    )
    query = torch.randn(1, 2, 100, 64)
    key = torch.randn(1, 2, 100, 64)
    value = torch.randn(1, 2, 100, 64)
    seen = {}

    def fake_attention(q, k, v, **kwargs):
        seen["shapes"] = (q.shape, k.shape, v.shape)
        assert torch.equal(q[:, :, :100], query) and torch.count_nonzero(q[:, :, 100:]) == 0
        assert torch.equal(k[:, :, :100], key) and torch.count_nonzero(k[:, :, 100:]) == 0
        return torch.arange(q.numel(), dtype=torch.float32).reshape(q.shape)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert seen["shapes"] == ((1, 2, 128, 64),) * 3
    expected = torch.arange(1 * 2 * 128 * 64, dtype=torch.float32).reshape(1, 2, 128, 64)[:, :, :100]
    assert torch.equal(output, expected.transpose(1, 2))
    assert weights is None


def test_causal_prefill_padding_only_for_unmasked_causal_self_attention():
    from thriftattention.integrations import transformers as hf

    causal = AttentionConfig(method="fp4", causal=True, fraction=None)
    noncausal = AttentionConfig(method="fp4", causal=False, fraction=None)
    query = torch.randn(1, 2, 100, 64)
    key = torch.randn(1, 2, 100, 64)

    assert hf._causal_prefill_padding(query, key, None, causal) == 28
    assert hf._causal_prefill_padding(query[:, :, :64], key[:, :, :64], None, causal) == 0
    assert hf._causal_prefill_padding(query, key, None, noncausal) == 0
    assert hf._causal_prefill_padding(query, key, torch.ones(1, 1, 100, 100, dtype=torch.bool), causal) == 0
    assert hf._causal_prefill_padding(query[:, :, :1], key, None, causal) == 0
    assert hf._causal_prefill_padding(query, torch.randn(1, 2, 128, 64), None, causal) == 0
    hd256 = torch.randn(1, 2, 100, 256)
    assert hf._causal_prefill_padding(hd256, hd256, None, causal) == 28
    assert hf._causal_prefill_padding(hd256[:, :, :64], hd256[:, :, :64], None, causal) == 64


def test_causal_prefill_padding_leaves_an_active_cache_to_pad_to_block(monkeypatch):
    from thriftattention.integrations import transformers as hf

    monkeypatch.setattr(hf, "get_active_thriftattention_cache", lambda: object())
    query = torch.randn(1, 2, 100, 64)
    config = AttentionConfig(method="fp4", causal=True, fraction=None)

    assert hf._causal_prefill_padding(query, query, None, config) == 0


def test_rejected_unaligned_prefill_falls_back_on_the_unpadded_inputs(monkeypatch):
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_unaligned_fallback"))
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_unaligned_fallback"),
        training=False,
        is_causal=True,
    )
    query, key, value = (torch.randn(1, 2, 100, 64) for _ in range(3))
    seen = {}

    def fake_fallback(module, q, k, v, attention_mask, **kwargs):
        seen["inputs"] = (q, k, v)
        return q.transpose(1, 2), None

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: "rejected")
    monkeypatch.setattr(hf, "_fallback_attention", fake_fallback)

    hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert all(a is b for a, b in zip(seen["inputs"], (query, key, value)))


# thrift at head_dim 256 is left to the block-sparse kernels' own tests
@pytest.mark.parametrize(
    ("method", "head_dim"), [("fp4", 64), ("fp4", 128), ("fp4", 256), ("thrift", 64), ("thrift", 128)]
)
@pytest.mark.parametrize("length", [100, 192])
def test_unaligned_causal_prefill_matches_sdpa(monkeypatch, method, head_dim, length):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 CUDA device required")
    from thriftattention.integrations import transformers as hf

    _stub_attention_registry(monkeypatch, hf)
    name = f"unit_unaligned_{method}"
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name=name, method=method))
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation=name), training=False, is_causal=True)
    F = torch.nn.functional
    torch.manual_seed(0)
    shape = (1, 4, length, head_dim)
    query, key, value = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    output, _ = hf.thriftattention_forward(module, query, key, value, None, scaling=head_dim**-0.5)
    # the reference sees only the real tokens
    ref = F.scaled_dot_product_attention(query.float(), key.float(), value.float(), is_causal=True)

    assert output.shape == (1, length, 4, head_dim)
    cosine = F.cosine_similarity(output.transpose(1, 2).float().flatten(), ref.flatten(), dim=0)
    assert cosine.item() > 0.95
