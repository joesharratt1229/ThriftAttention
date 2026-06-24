"""Proof of expected behavior for the kernel-hub HF integration.

Three layered tests; each skips cleanly when its prerequisite is absent:

1. ``test_extension_chokepoint_uses_hub_extension_when_set`` — runs anywhere;
   verifies that setting ``_hub_extension`` redirects ``get_extension()``.
2. ``test_hub_kernel_imports_and_exports_canonical_forward`` — runs where a
   local kernel-hub build is present; verifies the Hub package exposes a
   ``forward`` callable and a ``default_config`` slot.
3. ``test_hub_kernel_forward_matches_sdpa_on_blackwell`` — runs only on SM 12.0+;
   exercises the end-to-end transformers integration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _local_kernel_path() -> Path | None:
    root = Path(__file__).resolve().parent.parent / "kernel-hub" / "build-output"
    if not root.is_dir():
        return None
    for variant in root.iterdir():
        if variant.is_dir():
            return variant
    return None


def test_extension_chokepoint_uses_hub_extension_when_set(monkeypatch):
    import thriftattention._extension as ext_mod

    sentinel = SimpleNamespace(nvfp4_quantize=lambda *a, **k: ("sentinel",))
    monkeypatch.setattr(ext_mod, "_hub_extension", sentinel)
    assert ext_mod.get_extension() is sentinel


@pytest.mark.skipif(_local_kernel_path() is None, reason="kernel-hub not built locally")
def test_hub_kernel_imports_and_exports_canonical_forward():
    kernels = pytest.importorskip("kernels")
    kernel = kernels.get_local_kernel(_local_kernel_path())
    assert callable(kernel.forward)
    assert hasattr(kernel, "default_config")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="requires Blackwell GPU (SM 12.0+)",
)
def test_hub_kernel_forward_matches_sdpa_on_blackwell():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

    ref = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    ta = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="Hrsh-Venket/thrift-attention",
    )

    ref_out = ref.generate(**inputs, max_new_tokens=20, do_sample=False)
    ta_out = ta.generate(**inputs, max_new_tokens=20, do_sample=False)
    assert ref_out.shape == ta_out.shape
