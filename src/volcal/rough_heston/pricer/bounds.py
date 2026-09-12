"""Explicit contour and truncation settings; no estimated error bounds."""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class SinhConfig:
    """Integrate y in [0, spacing*nodes] using conjugate symmetry.

    Defaults use a horizontal sinh contour inside the stock moment strip.
    Nonzero omega bends the contour: callers must establish analytic
    continuation and tail decay outside that strip. Only the intercept is
    checked here; this is not a moment-domain or accuracy certificate.
    nodes counts intervals, so nodes+1 CF evaluations are performed.
    """
    omega1: float = -0.5
    omega: float = 0.0
    b: float = 1.0
    spacing: float = 0.025
    nodes: int = 240

    def __post_init__(self):
        for name in ('omega1', 'omega', 'b', 'spacing'):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f'{name} must be a finite real number')
        if self.b <= 0 or self.spacing <= 0:
            raise ValueError('b and spacing must be positive')
        if not -np.pi/2 < self.omega < np.pi/2:
            raise ValueError('omega must be strictly between -pi/2 and pi/2')
        intercept = self.omega1 + self.b*np.sin(self.omega)
        if not -1 < intercept < 0:
            raise ValueError('contour intercept must lie strictly inside (-1, 0)')
        if (isinstance(self.nodes, (bool, np.bool_))
                or not isinstance(self.nodes, Integral) or self.nodes < 1):
            raise ValueError('nodes must be a positive integer')
        if not np.isfinite(self.spacing*self.nodes) or self.spacing*self.nodes > 700:
            raise ValueError('contour extent is too large for float64 sinh')
