"""Rough-Heston CF selection feeding a single Fourier pricer."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..characteristic_function import rough_heston_cf_adams, rough_heston_cf_pade
from ..params import RoughHestonParams
from .bounds import SinhConfig
from .sinh import vanilla_price_from_cf


@dataclass(frozen=True)
class AdamsConfig:
    time_steps: int = 1000
    picard_iterations: int = 2

    def __post_init__(self):
        for name in ('time_steps', 'picard_iterations'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class PadeConfig:
    order: int = 4

    def __post_init__(self):
        if (isinstance(self.order, (bool, np.bool_)) or not isinstance(self.order, Integral)
                or self.order not in (2, 3, 4, 5)):
            raise ValueError('order must be 2, 3, 4 or 5')


@dataclass(frozen=True)
class RoughHestonPricer:
    """Adams is the default numerical reference / correctness-first method.

    Explicit Padé mode is a faster approximation with quantified error and
    known parameter/wing limitations; its default order is 4. Numerical
    resolution remains explicit. Methods and orders never switch automatically,
    and raw numerical failures and invalid prices remain visible.
    """
    cf_method: str = 'adams'
    cf_config: object = None
    integration_config: SinhConfig = SinhConfig()

    def __post_init__(self):
        if self.cf_method not in ('adams', 'pade'):
            raise ValueError('cf_method must be adams or pade')
        config_type = AdamsConfig if self.cf_method == 'adams' else PadeConfig
        if self.cf_config is None:
            object.__setattr__(self, 'cf_config', config_type())
        elif not isinstance(self.cf_config, config_type):
            raise TypeError(f'{self.cf_method} requires {config_type.__name__}')
        if not isinstance(self.integration_config, SinhConfig):
            raise TypeError('integration_config must be SinhConfig')

    def vanilla_price(self, *, T, K, option_params, rough_heston_params,
                      option_type='call', F=None):
        if not isinstance(rough_heston_params, RoughHestonParams):
            raise TypeError('rough_heston_params must be RoughHestonParams')
        def cf(u):
            if self.cf_method == 'adams':
                return rough_heston_cf_adams(
                    u, T, rough_heston_params, time_steps=self.cf_config.time_steps,
                    picard_iterations=self.cf_config.picard_iterations)
            return rough_heston_cf_pade(u, T, rough_heston_params, order=self.cf_config.order)
        return vanilla_price_from_cf(cf, T=T, K=K, option_params=option_params,
                                     option_type=option_type, F=F,
                                     integration_config=self.integration_config)
