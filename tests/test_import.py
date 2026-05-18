def test_public_imports():
    import pytest

    pytest.importorskip("torch")

    import thriftattention as ta

    assert callable(ta.attention)
    assert ta.AttentionConfig().method == "thrift"
    assert callable(ta.get_quant_format)
    assert callable(ta.get_selection_policy)
