"""Seeding and determinism shared by Tasks 2 and 3.

Two runs with the same config previously gave target accuracies 5-7 points
apart because cuDNN kernels were non-deterministic. Calling
``set_all_seeds(seed, deterministic=True)`` before building models and
loaders makes repeated runs reproducible on the same GPU/software stack.
"""
import os
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA is initialised

import torch  # noqa: E402


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
