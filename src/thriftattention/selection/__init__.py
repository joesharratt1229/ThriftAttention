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
from .local import (
    LocalSelectionPolicy,
    select_local_block_pairs,
    select_local_key_blocks,
)
from .quest import (
    QuestSelectionPolicy,
    block_minmax,
    select_quest_block_pairs,
    select_quest_key_blocks,
)

_POLICIES: dict[str, SelectionPolicy] = {
    "block_mean": BlockMeanSelectionPolicy(),
    "local": LocalSelectionPolicy(),
    "quest": QuestSelectionPolicy(),
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
    "LocalSelectionPolicy",
    "QuestSelectionPolicy",
    "block_minmax",
    "block_means",
    "get_selection_policy",
    "resolve_top_k",
    "select_block_pairs",
    "select_key_blocks",
    "select_local_block_pairs",
    "select_local_key_blocks",
    "select_quest_block_pairs",
    "select_quest_key_blocks",
]
