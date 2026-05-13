from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from thriftattention.config import AttentionConfig
from thriftattention.patch import patch_model, unpatch_model


def test_patch_model_rejects_old_selection_name_as_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        patch_model(object(), mode="topk")


def test_patch_model_validates_fraction_and_top_k():
    with pytest.raises(ValueError, match="fp16_fraction"):
        patch_model(object(), fp16_fraction=1.5)
    with pytest.raises(ValueError, match="top_k"):
        patch_model(object(), top_k=-1)
    with pytest.raises(ValueError, match="fallback"):
        patch_model(object(), fallback="eager")


def test_patch_model_sets_noncausal_config(monkeypatch):
    from thriftattention.integrations import transformers as hf

    model = SimpleNamespace()
    monkeypatch.setattr(hf, "patch_hf_model", lambda patched_model, config: config)

    config = patch_model(model, causal=False)

    assert config.causal is False


def test_hf_patch_sets_backend_and_tags_modules(monkeypatch):
    from thriftattention.integrations import transformers as hf

    registered = {}

    class Registry:
        @staticmethod
        def register(name, fn):
            registered[name] = fn

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda: None)

    child = SimpleNamespace()

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.calls = []

        def modules(self):
            return [self, child]

        def set_attn_implementation(self, name):
            self.calls.append(name)
            self.config._attn_implementation = name

    model = Model()
    patched = patch_model(model)

    assert patched is model
    assert "thriftattention" in registered
    assert model.config._attn_implementation == "thriftattention"
    assert model.calls == ["thriftattention"]
    assert child._thriftattention_config.fallback == "error"

    unpatch_model(model)
    assert model.config._attn_implementation == "sdpa"
    assert not hasattr(child, "_thriftattention_config")


def test_hf_patch_requires_public_set_attn_implementation(monkeypatch):
    from thriftattention.integrations import transformers as hf

    class Registry:
        @staticmethod
        def register(name, fn):
            pass

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda: None)

    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))

    with pytest.raises(TypeError, match="set_attn_implementation"):
        patch_model(model)

    assert not hasattr(model, "_thriftattention_original_attn_implementation")
    assert not hasattr(model, "_thriftattention_config")


def test_adapter_requires_patched_module_config():
    from thriftattention.integrations.transformers import thriftattention_forward

    module = SimpleNamespace(training=False, is_causal=True)
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="unpatched module"):
        thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_adapter_error_fallback_reports_rejection_reason():
    from thriftattention.integrations.transformers import thriftattention_forward

    module = SimpleNamespace(
        training=False,
        is_causal=True,
        num_key_value_groups=1,
        _thriftattention_config=AttentionConfig(backend="hf", fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_adapter_fast_path_transposes_thrift_output(monkeypatch):
    from thriftattention.integrations import transformers as hf

    module = SimpleNamespace(
        training=False,
        is_causal=True,
        _thriftattention_config=AttentionConfig(backend="hf", fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["selector"] == "block_mean"
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None


def test_adapter_fast_path_passes_noncausal_mode(monkeypatch):
    from thriftattention.integrations import transformers as hf

    module = SimpleNamespace(
        training=False,
        is_causal=False,
        _thriftattention_config=AttentionConfig(backend="hf", causal=False, fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["causal"] is False
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None
