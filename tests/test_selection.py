import pytest

pytest.importorskip("torch")

from thriftattention.selection import resolve_top_k


def test_resolve_top_k_clamps_explicit_value():
    assert resolve_top_k(8, top_k=99) == 8


def test_resolve_top_k_uses_fraction():
    assert resolve_top_k(2048, fraction=0.05) == 52
    assert resolve_top_k(2048, fraction=0.10) == 105
    assert resolve_top_k(2048, fraction=0.25) == 274


def test_resolve_top_k_rejects_negative_value():
    with pytest.raises(ValueError, match="top_k"):
        resolve_top_k(8, top_k=-1)


def test_resolve_top_k_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        resolve_top_k(8, fraction=1.5)
