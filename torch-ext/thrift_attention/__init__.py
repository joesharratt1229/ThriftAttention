"""Thin HF Hub kernel wrapper for ThriftAttention.

Delegates all logic to the canonical ``thriftattention`` pip package. Users
must have ``thriftattention`` installed; the compiled ``.so`` comes from the
Hub. All ``thriftattention`` imports are deferred to the first ``forward()``
call so the in-build ``get-kernel`` load check does not require
``thriftattention`` in the Nix sandbox.

To override the default config:

    from kernels import get_kernel
    from thriftattention.config import AttentionConfig
    ta = get_kernel("Hrsh-Venket/thrift-attention")
    ta.default_config = AttentionConfig(fraction=0.25)

To use a fully custom config under your own attn_implementation name, call
``thriftattention.integrations.transformers.register_transformers_attention(...)``
before loading the model.
"""

from . import _ops as _ops_module

default_config = None


def forward(module, query, key, value, attention_mask, **kwargs):
    from thriftattention import _extension as _ext_mod
    from thriftattention.config import AttentionConfig
    from thriftattention.integrations.transformers import (
        _REGISTERED_CONFIGS,
        thriftattention_forward,
    )

    global default_config
    if default_config is None:
        default_config = AttentionConfig()

    if _ext_mod._hub_extension is None:
        _ext_mod._hub_extension = _ops_module.ops

    impl_name = getattr(getattr(module, "config", None), "_attn_implementation", None)
    if impl_name and impl_name not in _REGISTERED_CONFIGS:
        _REGISTERED_CONFIGS[impl_name] = default_config

    return thriftattention_forward(module, query, key, value, attention_mask, **kwargs)


__all__ = ["forward", "default_config"]
