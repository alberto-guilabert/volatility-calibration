"""Explicit calibration controls; bounds always follow PARAMETER_ORDER."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from ..params import RoughHestonParams


@dataclass(frozen=True)
class ObjectiveConfig:
    convention: str = 'vega_normalized_price_mse'
    vega_floor: float = 1.0  # price per unit (decimal) volatility
    negative_price_tolerance: float = 1e-8

    def __post_init__(self):
        if self.convention not in ('vega_normalized_price_mse', 'price_mse'):
            raise ValueError('unknown objective convention')
        for name in ('vega_floor', 'negative_price_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.vega_floor == 0:
            raise ValueError('vega_floor must be positive')


@dataclass(frozen=True)
class CalibrationConfig:
    bounds: tuple = ((.03, .45), (.3, 3.), (.01, .10), (.1, .8), (.01, .10), (-.9, -.1))
    fixed_H: object = None
    de_popsize: int = 8
    de_maxiter: int = 30
    de_tol: float = 1e-6
    seed: int = 42
    de_polish: bool = False
    lbfgsb_maxiter: int = 200
    lbfgsb_ftol: float = 1e-14
    lbfgsb_gtol: float = 1e-8
    objective: ObjectiveConfig = ObjectiveConfig()
    failure_penalty: float = 1e6

    def __post_init__(self):
        a = np.asarray(self.bounds, dtype=float)
        if a.shape != (6, 2) or not np.all(np.isfinite(a)) or np.any(a[:, 0] >= a[:, 1]):
            raise ValueError('bounds must be six finite increasing pairs in PARAMETER_ORDER')
        RoughHestonParams.from_vector(a[:, 0])
        RoughHestonParams.from_vector(a[:, 1])
        object.__setattr__(self, 'bounds', tuple(map(tuple, a.tolist())))
        if self.fixed_H is not None:
            if isinstance(self.fixed_H, (bool, np.bool_)) or not np.isfinite(self.fixed_H) or not a[0, 0] <= self.fixed_H <= a[0, 1]:
                raise ValueError('fixed_H must lie within the H bounds')
        for name in ('de_popsize', 'de_maxiter', 'seed', 'lbfgsb_maxiter'):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < (1 if name == 'de_popsize' else 0):
                raise ValueError(f'invalid {name}')
        for name in ('de_tol', 'lbfgsb_ftol', 'lbfgsb_gtol', 'failure_penalty'):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0 or (name == 'failure_penalty' and v == 0):
                raise ValueError(f'invalid {name}')
        if not isinstance(self.de_polish, bool) or not isinstance(self.objective, ObjectiveConfig):
            raise TypeError('invalid polish or objective configuration')

    @property
    def active_bounds(self):
        return self.bounds if self.fixed_H is None else self.bounds[1:]

    def parameters(self, vector):
        a = np.asarray(vector, dtype=float)
        if a.shape != (len(self.active_bounds),):
            raise ValueError('incorrect active parameter vector shape')
        return RoughHestonParams.from_vector(a if self.fixed_H is None else np.r_[self.fixed_H, a])
