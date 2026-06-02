from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thriftattention.backends.registry import select_backend
from thriftattention.config import AttentionConfig


class FakeQuantFormat:
    name = "nvfp4"


@pytest.mark.parametrize(
    ("capability", "expected_backend"),
    [
        ((8, 0), "sm80"),
        ((8, 6), "sm80"),
        ((12, 0), "sm120"),
        ((12, 1), "sm120"),
    ],
)
def test_auto_backend_selects_registered_architecture_family(
    monkeypatch,
    capability,
    expected_backend,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)

    backend = select_backend(
        AttentionConfig(backend="auto"),
        FakeQuantFormat(),
        head_dim=64,
        device=torch.device("cuda"),
    )

    assert backend.name == expected_backend


def test_auto_backend_does_not_route_other_architectures_to_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    with pytest.raises(NotImplementedError, match=r"CUDA capability 9\.0"):
        select_backend(
            AttentionConfig(backend="auto"),
            FakeQuantFormat(),
            head_dim=64,
            device=torch.device("cuda"),
        )


def test_auto_backend_rejects_non_cuda_device():
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        select_backend(
            AttentionConfig(backend="auto"),
            FakeQuantFormat(),
            head_dim=64,
            device=torch.device("cpu"),
        )


def test_explicit_sm80_backend_can_be_selected():
    backend = select_backend(
        AttentionConfig(backend="sm80"),
        FakeQuantFormat(),
        head_dim=64,
    )

    assert backend.name == "sm80"
