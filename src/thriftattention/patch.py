from __future__ import annotations

from .config import AttentionConfig


def patch_model(
    model: object,
    *,
    backend: str = "hf",
    mode: str = "thrift",
    causal: bool = True,
    selector: str = "block_mean",
    fp16_fraction: float = 0.05,
    top_k: int | None = None,
    fallback: str = "error",
    patch_generation: bool = True,
) -> object:
    """Patch a supported model framework to use ThriftAttention where possible."""
    config = AttentionConfig(
        mode=_validate_choice("mode", mode, ("thrift", "fp4")),
        causal=bool(causal),
        selector=_validate_choice("selector", selector, ("block_mean",)),
        fp16_fraction=_validate_fraction(fp16_fraction),
        top_k=_validate_top_k(top_k),
        backend=_validate_choice("backend", backend, ("auto", "hf")),
        fallback=_validate_choice("fallback", fallback, ("error",)),
        patch_generation=bool(patch_generation),
    )

    if config.backend in ("auto", "hf"):
        from .integrations.transformers import patch_hf_model

        return patch_hf_model(model, config)

    raise ValueError(f"unsupported patch backend {backend!r}")


def unpatch_model(model: object, *, backend: str = "hf") -> object:
    backend = _validate_choice("backend", backend, ("auto", "hf"))
    if backend in ("auto", "hf"):
        from .integrations.transformers import unpatch_hf_model

        return unpatch_hf_model(model)
    raise ValueError(f"unsupported patch backend {backend!r}")


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        formatted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {formatted}, got {value!r}")
    return value


def _validate_fraction(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"fp16_fraction must be in [0, 1], got {value!r}")
    return value


def _validate_top_k(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"top_k must be non-negative, got {value!r}")
    return value
