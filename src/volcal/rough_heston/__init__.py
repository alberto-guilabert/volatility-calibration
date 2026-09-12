"""Rough Heston with kernel t**(H-1/2)/Gamma(H+1/2).

The CF is E[exp(i*u*log(S_T/F_T))], not the log-spot CF.
See characteristic_function.riccati for the mathematical contract.
"""

from .params import PARAMETER_ORDER, RoughHestonParams
from .characteristic_function import rough_heston_cf_adams, rough_heston_cf_pade

__all__ = ["PARAMETER_ORDER", "RoughHestonParams", "rough_heston_cf_adams",
           "rough_heston_cf_pade"]
