"""
    - Deterministic / reproducibility hardening (per-fold seeding, deterministic flags)

"""

import os
import random
import numpy as np
import torch

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k in list(os.environ.keys()):
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def truthy(x) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "on")


def fold_seed(base_seed: int, fold: int) -> int:
    # stable mapping base_seed + fold -> unique fold seed
    return int(base_seed) + 10_000 * int(fold)


def set_reproducible(seed: int, deterministic: bool = True, strict: bool = False):
    """
    Best-effort determinism:
      - deterministic=True: sets cudnn flags + disables TF32 to reduce drift
      - strict=True: may error/warn on nondeterministic ops (often hit with sparse/scatter)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuBLAS determinism (best set before CUDA context creation)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # reduce numeric drift
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if strict:
            try:
                torch.use_deterministic_algorithms(True)
                torch.set_deterministic_debug_mode("warn")  # set "error" if you want hard fail
            except Exception:
                pass
