def test_public_imports():
    import pytest

    pytest.importorskip("torch")

    import thriftattention as ta

    assert callable(ta.attention)
    assert callable(ta.fp4_attention)
    assert callable(ta.select_blocks)
