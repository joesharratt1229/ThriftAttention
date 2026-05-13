from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_fp4_attention_dispatches_noncausal_extension(monkeypatch):
    from thriftattention import functional

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
    monkeypatch.setattr(functional, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "_quantize_qkv", lambda *args: ("qp", "kp", "vp", "qs", "ks", "vs"))
    monkeypatch.setattr(functional, "get_extension", lambda: ext)

    assert functional.fp4_attention(q, k, v, causal=False) == "out"
    assert calls == [("noncausal", ("qp", "kp", "vp", "qs", "ks", "vs"))]


def test_fp4_attention_dispatches_single_query_extension(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 1, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_single_query(*args):
        calls.append(("single_query", args))
        return torch.empty(1, 1, 1, 64)

    ext = SimpleNamespace(fp4_attention_single_query_nvfp4_packed=fake_single_query)
    monkeypatch.setattr(functional, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        functional,
        "_quantize_single_query_qkv",
        lambda *args: ("qp", "kp", "vp", "qs", "ks", "vs"),
    )
    monkeypatch.setattr(functional, "get_extension", lambda: ext)

    out = functional.fp4_attention(q, k, v)

    assert out.shape == q.shape
    assert calls == [("single_query", ("qp", "kp", "vp", "qs", "ks", "vs"))]


def test_attention_dispatches_noncausal_extension(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 64, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_noncausal(*args):
        calls.append(("noncausal", args))
        return "out"

    ext = SimpleNamespace(
        thrift_attention_causal_nvfp4_packed=lambda *args: calls.append(("causal", args)),
        thrift_attention_noncausal_nvfp4_packed=fake_noncausal,
    )
    monkeypatch.setattr(functional, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "select_block_pairs", lambda *args, **kwargs: "selected")
    monkeypatch.setattr(functional, "_quantize_qkv", lambda *args: ("qp", "kp", "vp", "qs", "ks", "vs"))
    monkeypatch.setattr(functional, "get_extension", lambda: ext)

    assert functional.attention(q, k, v, causal=False) == "out"
    assert calls == [
        (
            "noncausal",
            (q, k, v, "selected", "qp", "kp", "vp", "qs", "ks", "vs"),
        )
    ]


def test_attention_dispatches_single_query_extension(monkeypatch):
    from thriftattention import functional

    calls = []
    q = torch.empty(1, 1, 1, 64)
    k = torch.empty(1, 1, 64, 64)
    v = torch.empty(1, 1, 64, 64)

    def fake_single_query(*args):
        calls.append(("single_query", args))
        return torch.empty(1, 1, 1, 64)

    ext = SimpleNamespace(thrift_attention_single_query_nvfp4_packed=fake_single_query)
    monkeypatch.setattr(functional, "check_qkv", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "require_block_aligned", lambda *args, **kwargs: None)
    monkeypatch.setattr(functional, "select_key_blocks", lambda *args, **kwargs: "selected")
    monkeypatch.setattr(
        functional,
        "_quantize_single_query_qkv",
        lambda *args: ("qp", "kp", "vp", "qs", "ks", "vs"),
    )
    monkeypatch.setattr(functional, "get_extension", lambda: ext)

    out = functional.attention(q, k, v)

    assert out.shape == q.shape
    assert calls[0][0] == "single_query"
    assert calls[0][1][3:] == ("selected", "qp", "kp", "vp", "qs", "ks", "vs")
