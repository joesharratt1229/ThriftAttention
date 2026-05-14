from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
