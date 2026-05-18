from __future__ import annotations

from thriftattention.config import SelectionMethod

from .base import SelectionConfig, SelectionPolicy
from .block_mean import (
    BlockMeanSelectionPolicy,
    block_means,
    resolve_top_k,
    select_block_pairs,
    select_key_blocks,
)

_POLICIES: dict[str, SelectionPolicy] = {
    "block_mean": BlockMeanSelectionPolicy(),
}


def get_selection_policy(name: SelectionMethod) -> SelectionPolicy:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        raise NotImplementedError(f"selection policy {name!r} is not implemented") from exc


__all__ = [
    "SelectionConfig",
    "SelectionPolicy",
    "BlockMeanSelectionPolicy",
    "block_means",
    "get_selection_policy",
    "resolve_top_k",
    "select_block_pairs",
    "select_key_blocks",
]
