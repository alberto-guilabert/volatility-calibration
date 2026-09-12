"""Bounded, coordinate-wise empirical stabilization, never an error bound.

Spacing is divided while nodes are multiplied, preserving the y extent.
Truncation adds intervals at fixed spacing. Final validation probes every
coordinate independently from the final anchor, without adopting probe prices.
Padé approximation and its fixed time quadrature errors are not estimated.
"""
from dataclasses import dataclass, replace
from numbers import Integral
import warnings
import numpy as np

from .main import RoughHestonPricer
from .diagnostics import (finite_values, price_bounds, put_call_parity,
                          strike_monotonicity, strike_convexity)


def _count(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class AdamsRefinementConfig:
    """Initial counts come from the immutable input pricer."""
    max_time_steps: int = 6400
    time_step_multiplier: int = 2
    max_picard_iterations: int = 8
    picard_increment: int = 1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _count(name, getattr(self, name), 2 if name == 'time_step_multiplier' else 1)


@dataclass(frozen=True)
class FourierRefinementConfig:
    max_nodes: int = 4096
    spacing_divisor: int = 2
    truncation_increment: int = 40

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _count(name, getattr(self, name), 2 if name == 'spacing_divisor' else 1)


@dataclass(frozen=True)
class RefinementConfig:
    atol: float = 1e-6
    rtol: float = 1e-6
    sanity_atol: float = 1e-8
    max_iterations_per_stage: int = 8
    adams: AdamsRefinementConfig = AdamsRefinementConfig()
    fourier: FourierRefinementConfig = FourierRefinementConfig()

    def __post_init__(self):
        for name in ('atol', 'rtol', 'sanity_atol'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.atol == self.rtol == 0:
            raise ValueError('at least one convergence tolerance must be positive')
        _count('max_iterations_per_stage', self.max_iterations_per_stage)
        if not isinstance(self.adams, AdamsRefinementConfig) or not isinstance(self.fourier, FourierRefinementConfig):
            raise TypeError('invalid refinement policy')


@dataclass(frozen=True)
class RefinementStep:
    stage: str
    configuration: RoughHestonPricer
    prices: tuple
    absolute_change: tuple = ()
    relative_change: tuple = ()
    passed: bool = False
    diagnostics: tuple = ()
    warnings: tuple = ()
    failure: str = ''


@dataclass(frozen=True)
class RefinementResult:
    final_prices: tuple
    converged: bool
    cf_method: str
    pade_order: object
    initial_configuration: RoughHestonPricer
    final_configuration: RoughHestonPricer
    policy: RefinementConfig
    history: tuple
    termination_reason: str
    warnings: tuple
    failures: tuple


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.termination_reason + ': ' + '; '.join(result.failures))


def price_change(old, new, *, atol, rtol):
    """Per-price scale=max(abs(old),abs(new)); relative change is zero for two zeros.

    Every price must satisfy abs(new-old) <= atol + rtol*scale. The absolute
    term protects near-zero prices. This measures stabilization, not true error.
    """
    for value in (atol, rtol):
        if not np.isfinite(value) or value < 0:
            raise ValueError('tolerances must be finite and nonnegative')
    a, b = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if a.shape != b.shape or not a.size:
        raise ValueError('price shapes must match and be nonempty')
    delta = abs(b-a)
    scale = np.maximum(abs(a), abs(b))
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale != 0)
    passed = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and
                  np.all(delta <= atol+rtol*scale))
    return tuple(delta.reshape(-1)), tuple(relative.reshape(-1)), passed


def _next(pricer, stage, policy):
    cf, f = pricer.cf_config, pricer.integration_config
    if stage == 'adams_time':
        n = cf.time_steps*policy.adams.time_step_multiplier
        if n > policy.adams.max_time_steps:
            return None, 'Adams time-step cap reached'
        cf = replace(cf, time_steps=n)
    elif stage == 'picard':
        n = cf.picard_iterations+policy.adams.picard_increment
        if n > policy.adams.max_picard_iterations:
            return None, 'Picard cap reached'
        cf = replace(cf, picard_iterations=n)
    else:
        n = (f.nodes*policy.fourier.spacing_divisor if stage == 'fourier_spacing'
             else f.nodes+policy.fourier.truncation_increment)
        if n > policy.fourier.max_nodes:
            return None, 'integration node cap reached'
        spacing = f.spacing/policy.fourier.spacing_divisor if stage == 'fourier_spacing' else f.spacing
        if spacing == 0 or spacing*n > 700:
            return None, 'Fourier float64 resolution limit reached'
        f = replace(f, nodes=n, spacing=spacing)
    return replace(pricer, cf_config=cf, integration_config=f), ''


def refine_prices(pricer, *, T, K, option_params, rough_heston_params,
                  option_type='call', config=RefinementConfig(), raise_on_failure=False):
    """Return immutable snapshots (flat price tuples, including scalar batches).

    Each stage needs a successful comparison; reaching a cap is not convergence.
    Financial checks run on calls and puts together in one CF evaluation. Parity
    is a reconstruction check, not independent evidence of CF accuracy. Tiny
    negatives within sanity_atol remain unchanged and are recorded as warnings.
    Invalid caller inputs raise normally; numerical failures retain their text
    in history/result, or in ConvergenceError.result when requested.
    """
    if not isinstance(pricer, RoughHestonPricer) or not isinstance(config, RefinementConfig):
        raise TypeError('expected RoughHestonPricer and RefinementConfig')
    from .diagnostics import no_arbitrage_bounds
    no_arbitrage_bounds(T=T, K=K, option_params=option_params, option_type=option_type)
    from ..params import RoughHestonParams
    if not isinstance(rough_heston_params, RoughHestonParams):
        raise TypeError('rough_heston_params must be RoughHestonParams')
    k = np.atleast_1d(np.asarray(K, dtype=float))
    types = np.broadcast_to(option_type, k.shape)
    history, notices, failures = [], [], []
    current, prices = pricer, ()
    stages = (['adams_time', 'picard'] if pricer.cf_method == 'adams' else []) + ['fourier_spacing', 'fourier_truncation']
    if pricer.cf_method == 'pade':
        notices.append('Fourier stabilization does not measure Padé approximation or fixed CF time-quadrature error.')

    def finish(reason, converged=False):
        result = RefinementResult(prices, converged, pricer.cf_method,
            pricer.cf_config.order if pricer.cf_method == 'pade' else None,
            pricer, current, config, tuple(history), reason, tuple(notices), tuple(failures))
        if raise_on_failure and not converged:
            raise ConvergenceError(result)
        return result

    def evaluate(candidate, stage, previous=()):
        ds, notes, values, failure = [], [], (), ''
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                both = np.asarray(candidate.vanilla_price(T=T, K=np.concatenate([k, k]),
                    option_params=option_params, rough_heston_params=rough_heston_params,
                    option_type=np.array(['call']*len(k)+['put']*len(k))))
                if both.shape != (2*len(k),):
                    raise FloatingPointError('unexpected price output shape')
                calls, puts = np.split(both, 2)
                values = tuple(np.where(types == 'call', calls, puts))
                ds.append(finite_values(both))
                for label, a in [('call', calls), ('put', puts)]:
                    ds.append(price_bounds(a, T=T, K=k, option_params=option_params,
                                           option_type=label, atol=config.sanity_atol))
                    if np.any((a < 0) & (a >= -config.sanity_atol)):
                        notes.append(f'tiny negative {label} price retained')
                    unique, indices = np.unique(k, return_index=True)
                    ds.append(strike_monotonicity(unique, a[indices], option_type=label, atol=config.sanity_atol))
                    ds.append(strike_convexity(unique, a[indices], atol=config.sanity_atol))
                ds.append(put_call_parity(calls, puts, T=T, K=k, option_params=option_params, atol=config.sanity_atol))
                if any(not d.passed for d in ds):
                    failure = 'financial/nonfinite diagnostics failed: ' + ', '.join(d.name for d in ds if not d.passed)
            except FloatingPointError as exc:
                failure = f'{type(exc).__name__}: {exc}'
            notes.extend(str(w.message) for w in caught)
        delta, relative, passed = price_change(previous, values, atol=config.atol, rtol=config.rtol) if previous and values else ((), (), False)
        history.append(RefinementStep(stage, candidate, values, delta, relative,
                                      passed and not failure, tuple(ds), tuple(notes), failure))
        notices.extend(notes)
        if failure:
            failures.append(failure)
        return values, passed and not failure, failure

    if current.integration_config.nodes > config.fourier.max_nodes:
        return finish('initial configuration exceeds integration node cap')
    if current.cf_method == 'adams':
        if current.cf_config.time_steps > config.adams.max_time_steps:
            return finish('initial configuration exceeds Adams time-step cap')
        if current.cf_config.picard_iterations > config.adams.max_picard_iterations:
            return finish('initial configuration exceeds Picard cap')
    prices, _, failure = evaluate(current, 'initial')
    if failure:
        return finish('initial evaluation failed')
    for stage in stages:
        for _ in range(config.max_iterations_per_stage):
            candidate, reason = _next(current, stage, config)
            if candidate is None:
                return finish(f'{stage}: {reason}')
            values, passed, failure = evaluate(candidate, stage, prices)
            if failure:
                return finish(f'{stage}: evaluation failed')
            current, prices = candidate, values
            if passed:
                break
        else:
            return finish(f'{stage}: non-convergence within iteration limit')
    # Fixed-anchor probes avoid claiming earlier convergence on a changed grid.
    for stage in stages:
        candidate, reason = _next(current, stage, config)
        if candidate is None:
            return finish(f'validation/{stage}: {reason}')
        _, passed, failure = evaluate(candidate, 'validation/'+stage, prices)
        if failure or not passed:
            return finish(f'validation/{stage}: ' + ('evaluation failed' if failure else 'non-convergence'))
    return finish('all stages and final independent probes stabilized', True)
