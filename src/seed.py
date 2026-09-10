from __future__ import annotations

import os
import random

SEED = 42


def seed_everything(seed: int = SEED, deterministic_torch: bool = False) -> int:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    if deterministic_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def env_snapshot() -> dict[str, str]:
    keys = [
        "PYTHONHASHSEED",
        "PYTHONUTF8",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_OFFLINE",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}
