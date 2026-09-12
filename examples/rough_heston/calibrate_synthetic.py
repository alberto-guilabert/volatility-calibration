"""Test. No market ingestion or plotting."""
import argparse
from dataclasses import asdict
import json
import numpy as np
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import (RoughHestonPricer, PadeConfig, AdamsConfig,
    SinhConfig, RefinementConfig, FourierRefinementConfig, refine_prices)
from volcal.rough_heston.calibrator import PreparedQuotes, CalibrationConfig, calibrate
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


def summary(result, truth=TRUE_PARAMS):
    out = dict(params=asdict(result.params), true_params=asdict(truth), fixed_H=result.fixed_H,
        initial_loss=result.initial_loss, de_loss=result.de_loss, final_loss=result.final_loss,
        runtime_seconds=result.runtime, mean_objective_seconds=result.mean_objective_seconds,
        total_evaluations=result.total_objective_evaluations,
        failures=result.failed_numerical_evaluations, de=asdict(result.de), lbfgsb=asdict(result.lbfgsb),
        max_price_error=result.final_repricing.max_market_price_error,
        max_iv_error=result.final_repricing.max_market_iv_error,
        financial_diagnostics_passed=result.final_repricing.passed,
        failure_examples=[asdict(f) for f in result.failures[:3]])
    out['refinement'] = [dict(indices=indices, converged=r.converged, reason=r.termination_reason,
        max_price_change=float(max(abs(np.asarray(r.final_prices)-np.asarray(result.final_repricing.prices)[list(indices)]))) if r.final_prices else None,
        final_configuration=asdict(r.final_configuration), warnings=r.warnings, failures=r.failures)
        for indices, r in result.refinement]
    if result.adams_validation is not None:
        a = result.adams_validation
        out['adams'] = dict(loss=a.loss, max_market_price_error=a.max_market_price_error,
            max_market_iv_error=a.max_market_iv_error, iv_comparisons=a.iv_comparisons,
            max_pade_adams_difference=a.max_calibration_price_difference,
            financial_diagnostics_passed=a.passed, failure=a.failure)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--cross', action='store_true')
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    grid = SinhConfig(spacing=.05, nodes=120)
    pade = RoughHestonPricer('pade', PadeConfig(4), grid)
    adams = RoughHestonPricer('adams', AdamsConfig(1600), grid)
    quotes = synthetic_quotes(adams if args.cross else pade)
    # Validate the chosen fixed grid once, before either optimizer is invoked.
    preflight = []
    for (t, s, r, q), indices in quotes.groups():
        check = refine_prices(pade, T=t, K=np.asarray(quotes.K)[list(indices)],
            option_params=(s, r, q), option_type=np.asarray(quotes.option_type)[list(indices)],
            rough_heston_params=TRUE_PARAMS,
            config=RefinementConfig(atol=2e-6, rtol=1e-6,
                fourier=FourierRefinementConfig(max_nodes=2400, truncation_increment=20)))
        change = float(max(abs(np.asarray(check.final_prices)-check.history[0].prices)))
        if not check.converged or change > 2e-5:
            raise RuntimeError('fixed pricing grid failed synthetic preflight validation')
        preflight.append(dict(T=t, converged=check.converged, max_price_change=change))
    reference_change = None
    if args.cross:
        finer = synthetic_quotes(RoughHestonPricer('adams', AdamsConfig(3200), grid))
        reference_change = float(max(abs(np.asarray(finer.market_price)-quotes.market_price)))
    policy = RefinementConfig(atol=2e-6, rtol=1e-6,
        fourier=FourierRefinementConfig(max_nodes=2400, truncation_increment=20)) if args.refine else None
    results = []
    for fixed in (TRUE_PARAMS.H, None):
        config = CalibrationConfig(fixed_H=fixed, de_popsize=3 if args.quick else 8,
            de_maxiter=2 if args.quick else 40, lbfgsb_maxiter=100 if args.quick else 300)
        result = calibrate(quotes, pade, config, refinement_config=policy,
                           adams_validation_pricer=adams if args.cross else None)
        results.append(summary(result))
    print(json.dumps(dict(fixed_grid_preflight=preflight,
                         adams_reference_1600_to_3200_max_change=reference_change,
                         results=results), indent=2))


if __name__ == '__main__':
    main()
