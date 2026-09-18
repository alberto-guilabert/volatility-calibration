"""Shared Phase-5/6 sparse synthetic surface (12 parity-linked quotes)."""
import numpy as np
from ..params import RoughHestonParams
from .loss import PreparedQuotes
from volcal.utils.black_scholes import iv_solver

TRUE_PARAMS = RoughHestonParams(.2, 1.5, .04, .4, .04, -.7)


def synthetic_quotes(pricer, params=TRUE_PARAMS):
    rows = []
    for t, r, q in [(.25, .03, .01), (.75, .04, .015)]:
        strikes = np.array([90., 100., 110., 90., 100., 110.])
        types = np.array(['call'] * 3 + ['put'] * 3)
        prices = pricer.vanilla_price(T=t, K=strikes, option_params=(100., r, q),
            option_type=types, rough_heston_params=params)
        for k, typ, price in zip(strikes, types, prices):
            rows.append((t, k, 100., r, q, typ, price,
                         iv_solver(price, t, k, (100., r, q), typ)))
    return PreparedQuotes(*zip(*rows))

