from __future__ import annotations

from types import ModuleType

from ._ops import ops


def get_extension() -> ModuleType:
    return ops
