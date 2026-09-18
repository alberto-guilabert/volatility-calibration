"""Prepared quote contract and fixed-resolution objective (no refinement)."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from volcal.utils.black_scholes import vega
from ..pricer import RoughHestonPricer
from .config import CalibrationConfig


@dataclass(frozen=True)
class PreparedQuotes:
    """Equal-length, nonempty 1-D columns; no scalar broadcasting.

    T is years, r/q continuously compounded, price in spot currency. Vega is
    price per unit decimal volatility. Supplied vega takes precedence over IV.
    IV/vega are optional for price MSE; normalized loss requires one of them.
    Every row has its own spot/r/q. Inputs are copied into immutable tuples.
    """
    T: tuple
    K: tuple
    S0: tuple
    r: tuple
    q: tuple
    option_type: tuple
    market_price: tuple
    market_iv: object = None
    market_vega: object = None

    def __post_init__(self):
        size = None
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None and name in ('market_iv', 'market_vega'):
                continue
            a = np.asarray(value)
            if a.ndim != 1 or not a.size or (size is not None and a.size != size):
                raise ValueError('quote columns must be nonempty equal-length 1-D arrays')
            size = a.size
            if name == 'option_type':
                if not np.all((a == 'call') | (a == 'put')):
                    raise ValueError('option_type must contain call or put')
            else:
                if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
                    raise ValueError(f'{name} must be finite real values')
                a = a.astype(float)
                if name in ('T', 'K', 'S0', 'market_iv') and np.any(a <= 0):
                    raise ValueError(f'{name} must be positive')
                if name in ('market_price', 'market_vega') and np.any(a < 0):
                    raise ValueError(f'{name} must be nonnegative')
            object.__setattr__(self, name, tuple(a.tolist()))

    def groups(self):
        groups = {}
        for i, key in enumerate(zip(self.T, self.S0, self.r, self.q)):
            groups.setdefault(key, []).append(i)
        return tuple((key, tuple(indices)) for key, indices in groups.items())


@dataclass(frozen=True)
class NumericalFailure:
    category: str
    parameters: tuple
    message: str


class InvalidPrices(FloatingPointError):
    pass


def price_quotes(pricer, quotes, params, negative_tolerance=1e-8):
    prices = np.empty(len(quotes.T))
    for (t, s, r, q), indices in quotes.groups():
        idx = np.asarray(indices)
        p = np.asarray(pricer.vanilla_price(T=t, K=np.asarray(quotes.K)[idx],
            option_params=(s, r, q), option_type=np.asarray(quotes.option_type)[idx],
            rough_heston_params=params), dtype=float)
        if p.shape != idx.shape:
            raise InvalidPrices('inconsistent model price shape')
        if not np.all(np.isfinite(p)):
            raise InvalidPrices('nonfinite model prices')
        if np.any(p < -negative_tolerance):
            raise InvalidPrices('material negative model prices')
        prices[idx] = p
    return prices


class CalibrationObjective:
    """Stateful evaluation ledger. Failures have no residuals or model prices.

    Only expected numerical/configuration exceptions are penalized; programming
    errors propagate. Best valid evaluation is tracked independently of penalties.
    """
    def __init__(self, quotes, pricer, config=CalibrationConfig()):
        if not isinstance(quotes, PreparedQuotes) or not isinstance(pricer, RoughHestonPricer) or not isinstance(config, CalibrationConfig):
            raise TypeError('expected prepared quotes, explicit pricer and calibration config')
        self.quotes, self.pricer, self.config = quotes, pricer, config
        self.scale = np.ones(len(quotes.T))
        if config.objective.convention == 'vega_normalized_price_mse':
            if quotes.market_vega is not None:
                raw = np.asarray(quotes.market_vega)
            elif quotes.market_iv is not None:
                raw = vega(np.asarray(quotes.market_iv), np.asarray(quotes.T), np.asarray(quotes.K),
                           (np.asarray(quotes.S0), np.asarray(quotes.r), np.asarray(quotes.q)))
            else:
                raise ValueError('normalized objective requires market_vega or market_iv')
            if not np.all(np.isfinite(raw)):
                raise ValueError('nonfinite market vega')
            self.scale = np.maximum(raw, config.objective.vega_floor)
        self.evaluations = 0
        self.seconds = 0.
        self.failures = []
        self.best_params = None
        self.best_loss = np.inf

    def loss(self, prices):
        p = np.asarray(prices)
        if p.shape != self.scale.shape or not np.all(np.isfinite(p)):
            raise InvalidPrices('invalid price vector in loss')
        with np.errstate(over='raise', invalid='raise'):
            value = float(np.mean(((np.asarray(self.quotes.market_price) - p) / self.scale)**2))
        if not np.isfinite(value):
            raise InvalidPrices('nonfinite objective')
        return value

    def __call__(self, vector):
        return self._evaluate(vector, False)

    def residuals(self, vector):
        """Residual vector whose mean square is this exact scalar objective.

        Uses the same pricing, failure policy and evaluation ledger. Invalid
        evaluations return constant residuals with mean square failure_penalty.
        """
        return self._evaluate(vector, True)

    def _evaluate(self, vector, residuals):
        start = perf_counter()
        self.evaluations += 1
        params = None
        try:
            params = self.config.parameters(vector)
            prices = price_quotes(self.pricer, self.quotes, params,
                                  self.config.objective.negative_price_tolerance)
            value = self.loss(prices)
            if value < self.best_loss:
                self.best_loss, self.best_params = value, params
            return ((prices - np.asarray(self.quotes.market_price)) / self.scale
                    if residuals else value)
        except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
            message = str(exc)
            category = ('invalid_prices' if isinstance(exc, InvalidPrices) else
                        'invalid_configuration' if isinstance(exc, ValueError) else 'numerical_pricing')
            full = tuple(params.to_vector()) if params is not None else tuple(
                np.r_[self.config.fixed_H, vector] if self.config.fixed_H is not None else vector)
            self.failures.append(NumericalFailure(category, full, f'{type(exc).__name__}: {message}'))
            return (np.full(len(self.scale), np.sqrt(self.config.failure_penalty))
                    if residuals else self.config.failure_penalty)
        finally:
            self.seconds += perf_counter() - start
