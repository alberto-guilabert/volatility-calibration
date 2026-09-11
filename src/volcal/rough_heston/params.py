"""Immutable parameters for the power-law-kernel rough-Heston model."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar, Tuple

import numpy as np

from volcal.utils.params import params_to_vec, vec_to_params


PARAMETER_ORDER = ("H", "kappa", "theta", "sigma", "v0", "rho")


@dataclass(frozen=True)
class RoughHestonParams:
    """Parameters with alpha = H + 1/2; sigma is the variance vol-of-vol.

    At H = 1/2 these are exactly the classical Heston parameters:
    dV = kappa * (theta - V) dt + sigma * sqrt(V) dW.
    No Feller restriction is imposed. Vector conversion always uses
    PARAMETER_ORDER, independently of mapping or field insertion order.
    """

    H: float
    kappa: float
    theta: float
    sigma: float
    v0: float
    rho: float

    parameter_order: ClassVar[Tuple[str, ...]] = PARAMETER_ORDER

    def __post_init__(self) -> None:
        for key in PARAMETER_ORDER:
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{key} must be a finite real number")
            if not isfinite(value):
                raise ValueError(f"{key} must be finite")
            object.__setattr__(self, key, float(value))
        if not 0.0 < self.H <= 0.5:
            raise ValueError("H must satisfy 0 < H <= 0.5")
        for key in ("kappa", "sigma"):
            if getattr(self, key) <= 0.0:
                raise ValueError(f"{key} must be > 0")
        for key in ("theta", "v0"):
            if getattr(self, key) < 0.0:
                raise ValueError(f"{key} must be >= 0")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must satisfy -1 <= rho <= 1")

    @property
    def alpha(self) -> float:
        return self.H + 0.5

    def to_vector(self) -> np.ndarray:
        """Return a new float64 vector in PARAMETER_ORDER."""
        return params_to_vec(
            {key: getattr(self, key) for key in PARAMETER_ORDER}, PARAMETER_ORDER
        )

    @classmethod
    def from_vector(cls, values) -> "RoughHestonParams":
        """Construct from a real one-dimensional vector in PARAMETER_ORDER."""
        array = np.asarray(values)
        if array.shape != (len(PARAMETER_ORDER),) or array.dtype.kind not in "iuf":
            raise ValueError(f"parameters must be a real vector in {PARAMETER_ORDER}")
        return cls(**vec_to_params(array, PARAMETER_ORDER))
