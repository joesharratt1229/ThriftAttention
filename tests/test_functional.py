from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from thriftattention.config import AttentionConfig


class FakeQuantFormat:
    def __init__(self, name: str = "nvfp4"):
        self.name = name

    def quantize_qkv(self, *args, **kwargs):
        return "qp", "kp", "vp", "qs", "ks", "vs"

    def quantize_single_query_qkv(self, *args, **kwargs):
        return "qp", "kp", "vp", "qs", "ks", "vs"


def test_attention_orchestrates_fp4_without_selection(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 64, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)
    config = AttentionConfig(method="fp4", causal=False)
    quant_format = FakeQuantFormat()

    class Backend:
        def attention(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "out"

    def fail_selection(name):
        raise AssertionError(f"selection policy should not be requested for {name}")

    monkeypatch.setattr(functional, "get_quant_format", lambda name: quant_format)
    monkeypatch.setattr(functional, "get_selection_policy", fail_selection)
    monkeypatch.setattr(
        functional,
        "select_backend",
        lambda cfg, qfmt, *, head_dim, device=None: Backend(),
    )

    assert functional.attention(q, k, v, config=config) == "out"
    assert calls == [
        (
            (q, k, v),
            {"selection": None, "quant_format": quant_format, "config": config, "is_bf16": False},
        )
    ]


def test_attention_builds_selection_for_thrift(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 64, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)
    config = AttentionConfig(
        method="thrift",
        causal=False,
        selection="block_mean",
        fraction=0.25,
        top_k=3,
        block_size=64,
    )
    quant_format = FakeQuantFormat()

    class Policy:
        def select(self, q_arg, k_arg, *, config, causal, is_bf16):
            calls.append(("select", q_arg, k_arg, config, causal, is_bf16))
            return "selected"

    class Backend:
        def attention(self, *args, **kwargs):
            calls.append(("backend", args, kwargs))
            return "out"

    monkeypatch.setattr(functional, "get_quant_format", lambda name: quant_format)
    monkeypatch.setattr(functional, "get_selection_policy", lambda name: Policy())
    monkeypatch.setattr(
        functional,
        "select_backend",
        lambda cfg, qfmt, *, head_dim, device=None: Backend(),
    )

    assert functional.attention(q, k, v, config=config) == "out"

    _, q_arg, k_arg, selection_config, causal, is_bf16 = calls[0]
    assert q_arg is q
    assert k_arg is k
    assert selection_config.name == "block_mean"
    assert selection_config.fraction == 0.25
    assert selection_config.top_k == 3
    assert selection_config.block_size == 64
    assert causal is False
    assert is_bf16 is False
    assert calls[1] == (
        "backend",
        (q, k, v),
        {"selection": "selected", "quant_format": quant_format, "config": config, "is_bf16": False},
    )


def test_attention_routes_bf16_from_query_dtype(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)
    config = AttentionConfig(method="thrift", causal=False, selection="block_mean", top_k=1)
    quant_format = FakeQuantFormat()

    class Policy:
        def select(self, q_arg, k_arg, *, config, causal, is_bf16):
            calls.append(("select", is_bf16))
            return "selected"

    class Backend:
        def attention(self, *args, **kwargs):
            calls.append(("backend", kwargs["is_bf16"]))
            return "out"

    monkeypatch.setattr(functional, "get_quant_format", lambda name: quant_format)
    monkeypatch.setattr(functional, "get_selection_policy", lambda name: Policy())
    monkeypatch.setattr(
        functional,
        "select_backend",
        lambda cfg, qfmt, *, head_dim, device=None: Backend(),
    )

    assert functional.attention(q, k, v, config=config) == "out"
    assert calls == [("select", True), ("backend", True)]


def test_sm120_backend_dispatches_fp4_noncausal_extension(monkeypatch):
    from thriftattention.backends import sm120 as sm120_backend

    calls = []
    q = torch.empty(1, 1, 64, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_noncausal(*args):
        calls.append(("noncausal", args))
        return "out"

    ext = SimpleNamespace(
        fp4_attention_causal_nvfp4_packed=lambda *args: calls.append(("causal", args)),
        fp4_attention_noncausal_nvfp4_packed=fake_noncausal,
    )
    monkeypatch.setattr(sm120_backend, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "get_extension", lambda: ext)

    out = sm120_backend.Sm120Nvfp4Backend().attention(
        q,
        k,
        v,
        selection=None,
        quant_format=FakeQuantFormat(),
        config=AttentionConfig(method="fp4", causal=False, exp_approx=True),
        is_bf16=False,
    )

    assert out == "out"
    assert calls == [("noncausal", ("qp", "kp", "vp", "qs", "ks", "vs", False, True))]


def test_sm120_backend_dispatches_fp4_mxfp4_extension(monkeypatch):
    from thriftattention.backends import sm120 as sm120_backend

    calls = []
    q = torch.empty(1, 1, 64, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_causal(*args):
        calls.append(("mxfp4_causal", args))
        return "out"

    ext = SimpleNamespace(
        fp4_attention_causal_mxfp4_packed=fake_causal,
        fp4_attention_noncausal_mxfp4_packed=lambda *args: calls.append(("noncausal", args)),
    )
    monkeypatch.setattr(sm120_backend, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "get_extension", lambda: ext)

    out = sm120_backend.Sm120Nvfp4Backend().attention(
        q,
        k,
        v,
        selection=None,
        quant_format=FakeQuantFormat("mxfp4"),
        config=AttentionConfig(method="fp4", causal=True, implementation="tiled"),
        is_bf16=False,
    )

    assert out == "out"
    assert calls == [("mxfp4_causal", ("qp", "kp", "vp", "qs", "ks", "vs", False))]


def test_sm120_backend_dispatches_mxfp4_single_query_extension(monkeypatch):
    from thriftattention.backends import sm120 as sm120_backend

    calls = []
    q = torch.empty(1, 1, 1, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_single_query(*args):
        calls.append(("mxfp4_single_query", args))
        return torch.empty(1, 1, 1, 64)

    ext = SimpleNamespace(fp4_attention_single_query_mxfp4_packed=fake_single_query)
    monkeypatch.setattr(sm120_backend, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "get_extension", lambda: ext)

    out = sm120_backend.Sm120Nvfp4Backend().attention(
        q,
        k,
        v,
        selection=None,
        quant_format=FakeQuantFormat("mxfp4"),
        config=AttentionConfig(method="fp4", implementation="auto"),
        is_bf16=False,
    )

    assert out.shape == q.shape
    assert calls == [("mxfp4_single_query", ("qp", "kp", "vp", "qs", "ks", "vs", False))]


def test_sm120_backend_dispatches_thrift_single_query_extension(monkeypatch):
    from thriftattention.backends import sm120 as sm120_backend

    calls = []
    q = torch.empty(1, 1, 1, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_single_query(*args):
        calls.append(("single_query", args))
        return torch.empty(1, 1, 1, 64)

    ext = SimpleNamespace(thrift_attention_single_query_nvfp4_packed=fake_single_query)
    monkeypatch.setattr(sm120_backend, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "get_extension", lambda: ext)

    out = sm120_backend.Sm120Nvfp4Backend().attention(
        q,
        k,
        v,
        selection="selected",
        quant_format=FakeQuantFormat(),
        config=AttentionConfig(method="thrift"),
        is_bf16=False,
    )

    assert out.shape == q.shape
    assert calls[0][0] == "single_query"
    assert calls[0][1][3:] == ("selected", "qp", "kp", "vp", "qs", "ks", "vs", False)


def test_sm120_backend_forwards_bf16_to_quantizer_and_extension(monkeypatch):
    from thriftattention.backends import sm120 as sm120_backend

    calls = []
    q = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 64, 64, dtype=torch.bfloat16)

    class QuantFormat:
        name = "nvfp4"

        def quantize_qkv(self, *args, **kwargs):
            calls.append(("quantize_qkv", kwargs["is_bf16"]))
            return "qp", "kp", "vp", "qs", "ks", "vs"

    def fake_noncausal(*args):
        calls.append(("noncausal", args[-2], args[-1]))
        return "out"

    ext = SimpleNamespace(fp4_attention_noncausal_nvfp4_packed=fake_noncausal)
    monkeypatch.setattr(sm120_backend, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(sm120_backend, "get_extension", lambda: ext)

    out = sm120_backend.Sm120Nvfp4Backend().attention(
        q,
        k,
        v,
        selection=None,
        quant_format=QuantFormat(),
        config=AttentionConfig(method="fp4", causal=False),
        is_bf16=True,
    )

    assert out == "out"
    assert calls == [("quantize_qkv", True), ("noncausal", True, False)]
