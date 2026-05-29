import pytest

torch = pytest.importorskip("torch")

from thriftattention.selection import resolve_top_k


def test_resolve_top_k_clamps_explicit_value():
    assert resolve_top_k(8, top_k=99) == 8


def test_resolve_top_k_uses_fraction():
    assert resolve_top_k(2048, fraction=0.05) == 52
    assert resolve_top_k(2048, fraction=0.10) == 105
    assert resolve_top_k(2048, fraction=0.25) == 274


def test_resolve_top_k_uses_dense_fraction_for_noncausal():
    assert resolve_top_k(2048, causal=False, fraction=0.05) == 102
    assert resolve_top_k(2048, causal=False, fraction=0.10) == 205
    assert resolve_top_k(2048, causal=False, fraction=0.25) == 512


def test_resolve_top_k_rejects_negative_value():
    with pytest.raises(ValueError, match="top_k"):
        resolve_top_k(8, top_k=-1)


def test_resolve_top_k_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        resolve_top_k(8, fraction=1.5)


def test_select_key_blocks_groups_gqa(monkeypatch):
    from thriftattention import selection
    from thriftattention.selection import block_mean

    monkeypatch.setattr(block_mean, "check_qkv", lambda *args, **kwargs: None)

    q = torch.tensor(
        [[[[10.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 10.0]]]],
        dtype=torch.float16,
    )
    k = torch.tensor(
        [
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            ]
        ],
        dtype=torch.float16,
    )

    selected = selection.select_key_blocks(q, k, top_k=1, block_size=2)

    assert selected.tolist() == [[0], [0]]
