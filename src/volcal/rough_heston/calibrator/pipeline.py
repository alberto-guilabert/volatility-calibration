"""Differential Evolution -> L-BFGS-B using one explicit fixed pricer."""
from time import perf_counter
import numpy as np
from scipy.optimize import differential_evolution, minimize
from ..params import RoughHestonParams
from ..pricer import RoughHestonPricer, RefinementConfig, refine_prices
from ..pricer.diagnostics import (finite_values, price_bounds, put_call_parity,
                                  strike_monotonicity, strike_convexity)
from .config import CalibrationConfig
from .loss import CalibrationObjective, price_quotes, quote_iv
from .result import CalibrationResult, CalibrationError, OptimizerStatus, RepricingResult


def _status(result):
    return OptimizerStatus(bool(result.success), getattr(result, 'status', None),
        str(result.message), float(result.fun), int(result.nfev), int(result.nit))


def _reprice(objective, params, pricer, calibration_prices=None):
    quotes = objective.quotes
    try:
        p = price_quotes(pricer, quotes, params, objective.config.objective.negative_price_tolerance)
        ds = [(tuple(range(len(p))), finite_values(p))]
        for key, indices in quotes.groups():
            t, s, r, q = key[:4]
            forward = quotes.forward_kwargs(indices[0])
            idx = np.asarray(indices)
            k, types = np.asarray(quotes.K)[idx], np.asarray(quotes.option_type)[idx]
            ds.append((indices, price_bounds(p[idx], T=t, K=k, option_params=(s, r, q),
                option_type=types, atol=objective.config.objective.negative_price_tolerance, **forward)))
            for label in ('call', 'put'):
                selected = idx[types == label]
                if not len(selected):
                    continue
                unique, first = np.unique(np.asarray(quotes.K)[selected], return_index=True)
                selected = selected[first]
                ds.append((tuple(selected), strike_monotonicity(unique, p[selected], option_type=label)))
                ds.append((tuple(selected), strike_convexity(unique, p[selected])))
            for strike in np.unique(k):
                calls, puts = idx[(k == strike) & (types == 'call')], idx[(k == strike) & (types == 'put')]
                if len(calls) and len(puts):
                    ds.append(((int(calls[0]), int(puts[0])), put_call_parity(
                        p[calls[:1]], p[puts[:1]], T=t, K=[strike], option_params=(s, r, q), **forward)))
        iv_errors = []
        for i, value in enumerate(p):
            try:
                market_iv = quotes.market_iv[i] if quotes.market_iv is not None else quote_iv(quotes, i, quotes.market_price[i])
                model_iv = quote_iv(quotes, i, value)
                if np.isfinite(market_iv) and np.isfinite(model_iv):
                    iv_errors.append(abs(model_iv - market_iv))
            except (ValueError, FloatingPointError, OverflowError):
                pass  # Explicit count below identifies how many IV comparisons exist.
        return RepricingResult(pricer, tuple(p), objective.loss(p), tuple(ds), '',
            float(max(abs(p - quotes.market_price))), max(iv_errors) if iv_errors else None,
            len(iv_errors), None if calibration_prices is None else float(max(abs(p - calibration_prices))))
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return RepricingResult(pricer, failure=f'{type(exc).__name__}: {exc}')


def calibrate(quotes, pricer, config=CalibrationConfig(), *, reference_params=None,
              refinement_config=None, adams_validation_pricer=None):
    """Fit prepared quotes; optional checks run only after both optimizers.

    Bounds have six rows in H,kappa,theta,sigma,v0,rho order even with fixed H.
    A missing reference uses the active bounds midpoint. Final parameters are
    the best *valid* evaluation encountered, so failed/poor local termination
    cannot overwrite a better fit. Both raw optimizer outcomes are preserved.
    Runtime includes final validation; objective counts exclude validation.
    Caller is responsible for validating the fixed numerical grid beforehand.
    """
    start = perf_counter()
    objective = CalibrationObjective(quotes, pricer, config)
    if refinement_config is not None and not isinstance(refinement_config, RefinementConfig):
        raise TypeError('refinement_config must be RefinementConfig')
    if adams_validation_pricer is not None:
        if not isinstance(adams_validation_pricer, RoughHestonPricer) or adams_validation_pricer.cf_method != 'adams':
            raise ValueError('independent validation requires an explicit Adams pricer')
    if reference_params is None:
        x0 = np.mean(config.active_bounds, axis=1)
    else:
        if not isinstance(reference_params, RoughHestonParams):
            raise TypeError('reference_params must be RoughHestonParams')
        if config.fixed_H is not None and reference_params.H != config.fixed_H:
            raise ValueError('reference H differs from fixed_H')
        x0 = reference_params.to_vector()[0 if config.fixed_H is None else 1:]
        bounds = np.asarray(config.active_bounds)
        if np.any(x0 < bounds[:, 0]) or np.any(x0 > bounds[:, 1]):
            raise ValueError('reference parameters outside bounds')
    initial = objective(x0)
    initial = initial if not objective.failures else None
    de = differential_evolution(objective, config.active_bounds, popsize=config.de_popsize,
        maxiter=config.de_maxiter, tol=config.de_tol, seed=config.seed,
        polish=config.de_polish, workers=1, updating='immediate', x0=x0)
    local_start = (de.x if objective.best_params is None else
                   objective.best_params.to_vector()[0 if config.fixed_H is None else 1:])
    lb = minimize(objective, local_start, method='L-BFGS-B', bounds=config.active_bounds,
        options=dict(maxiter=config.lbfgsb_maxiter, ftol=config.lbfgsb_ftol, gtol=config.lbfgsb_gtol))
    if objective.best_params is None:
        raise CalibrationError(objective.failures, objective.evaluations)
    params = objective.best_params
    final = _reprice(objective, params, pricer)
    refinements = []
    if refinement_config is not None:
        for key, indices in quotes.groups():
            t, s, r, q = key[:4]
            idx = np.asarray(indices)
            refinements.append((indices, refine_prices(pricer, T=t, K=np.asarray(quotes.K)[idx],
                option_params=(s, r, q), option_type=np.asarray(quotes.option_type)[idx],
                rough_heston_params=params, config=refinement_config,
                **quotes.forward_kwargs(indices[0]))))
    adams = None if adams_validation_pricer is None else _reprice(
        objective, params, adams_validation_pricer, np.asarray(final.prices) if final.prices else None)
    return CalibrationResult(params, initial, float(de.fun), objective.best_loss,
        perf_counter() - start, _status(de), _status(lb), objective.evaluations,
        len(objective.failures), tuple(objective.failures), config.objective.convention,
        pricer.cf_method, pricer, config, config.fixed_H, objective.seconds / objective.evaluations,
        final, tuple(refinements), adams, 'best_valid_evaluation')
