from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "run_kernel_benchmarks.py"
module_spec = importlib.util.spec_from_file_location("thriftattention_benchmark", benchmark_path)
benchmark = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = benchmark
module_spec.loader.exec_module(benchmark)


def test_expected_sm_parser_accepts_blackwell_forms():
    assert benchmark.parse_expected_sm("10") == (10, 0)
    assert benchmark.parse_expected_sm("10.0") == (10, 0)
    assert benchmark.parse_expected_sm("100") == (10, 0)
    assert benchmark.parse_expected_sm("sm_100") == (10, 0)
    assert benchmark.parse_expected_sm("compute_100") == (10, 0)
    assert benchmark.parse_expected_sm("sm120") == (12, 0)


def test_expected_sm_parser_rejects_invalid_value():
    with pytest.raises(benchmark.argparse.ArgumentTypeError):
        benchmark.parse_expected_sm("0")


def test_build_specs_keeps_optional_comparison_targets_skippable(monkeypatch):
    def missing():
        raise benchmark.MissingDependency("not installed")

    monkeypatch.setattr(benchmark, "require_flash_attn_func", missing)
    monkeypatch.setattr(benchmark, "require_flash_attn4_func", missing)
    monkeypatch.setattr(benchmark, "require_sageattn3", missing)
    args = SimpleNamespace(
        causal=True,
        coverages=[0.05],
        dtype_name="fp16",
        fp16_backend="both",
        quant_format="nvfp4",
        skip_fa4=False,
        skip_fp16=False,
        skip_fp4=False,
        skip_thrift=False,
    )
    q = torch.empty(1, 2, 64, 8)
    k = torch.empty(1, 2, 64, 8)
    v = torch.empty(1, 2, 64, 8)

    specs = benchmark.build_specs(args, q, k, v, seq_len=64)
    by_name = {spec.name: spec for spec in specs}

    assert set(by_name) == {
        "fp16_torch_sdpa",
        "fp16_flash_attn",
        "flash_attn4",
        "ta_nvfp4_fp4_attention",
        "thrift_nvfp4_5pct",
        "fp4_sageattn3",
    }
    for name in ("fp16_flash_attn", "flash_attn4", "fp4_sageattn3"):
        with pytest.raises(benchmark.MissingDependency):
            by_name[name].fn()


def test_torch_sdpa_backend_handles_grouped_query_attention():
    q = torch.zeros(1, 4, 3, 8)
    k = torch.zeros(1, 2, 3, 8)
    v = torch.zeros(1, 2, 3, 8)

    out = benchmark.make_torch_sdpa(q, k, v, causal=False)()

    assert out.shape == q.shape
