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
