from typing import Tuple
import numpy as np


def vec_to_params(x: np.ndarray, PARAM_KEYS: Tuple[str]) -> dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    if x.size != len(PARAM_KEYS):
        raise ValueError(f"Expected {len(PARAM_KEYS)} params, got {x.size}")
    return dict(zip(PARAM_KEYS, map(float, x)))


def params_to_vec(p: dict, PARAM_KEYS: Tuple[str]) -> np.ndarray:
    return np.array([p[k] for k in PARAM_KEYS], dtype=float)