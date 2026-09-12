# Phase 5: prepared synthetic rough-Heston calibration

## Scope and public API

This phase adds only `volcal.rough_heston.calibrator`, its tests, and the
synthetic example. Adams, Padé, sinh inversion and refinement mathematics are
unchanged. There is no market ingestion, DataLoader dependency or plotting.

```python
from volcal.rough_heston.calibrator import (
    PreparedQuotes, CalibrationConfig, ObjectiveConfig, calibrate,
)
from volcal.rough_heston.pricer import RoughHestonPricer, PadeConfig, SinhConfig

# Each column is a nonempty one-dimensional sequence of equal length.
quotes = PreparedQuotes(
    T=[.5, .5], K=[100., 110.], S0=[100., 100.],
    r=[.03, .04], q=[.01, .02], option_type=['call', 'put'],
    market_price=[6., 12.], market_iv=[.2, .25],
)
pricer = RoughHestonPricer('pade', PadeConfig(4),
                         SinhConfig(spacing=.05, nodes=120))
config = CalibrationConfig(fixed_H=.2, seed=42)
# Validate your numerical configuration for your intended surface before fitting.
result = calibrate(quotes, pricer, config)
```

`CalibrationObjective` is also public for inspecting the fixed-resolution loss
and failure ledger. `CalibrationResult`, `RepricingResult`, `OptimizerStatus`,
`NumericalFailure`, and `CalibrationError` expose immutable result snapshots.
No optimization runs at import time.

## Quote contract

T is strictly positive in years; K and S0 are strictly positive; r and q are
finite continuously compounded rates. Each row specifies its own financial
inputs. Batches group by **all four** `(T, S0, r, q)` values, never maturity
alone. Option types are exactly `call` or `put`; market prices are finite and
nonnegative, in spot currency. Expired options, scalar broadcasting, ragged
columns, missing fields and nonfinite inputs are rejected. Columns are copied
into tuples, so caller array mutations cannot change a calibration.

Optional `market_iv` is positive annualized decimal volatility. Optional
`market_vega` is nonnegative price sensitivity to one unit decimal volatility
(not one percentage point). Supplied vega takes precedence; otherwise vega is
computed once using `volcal.utils.black_scholes.vega` and supplied market IV.
The normalized objective requires at least one of these columns; price MSE
requires neither. Prepared prices and supplied IV/vega are caller inputs, not
silently recalculated or forced to agree. Market no-arbitrage consistency is
not an input validation requirement; model financial diagnostics run post-fit.

## Objective and failure handling

The default is

```
effective_vega = maximum(market_vega, vega_floor)
loss = mean(((market_price - model_price) / effective_vega)**2)
```

This is vega-normalized price MSE, approximately IV-space error, **not exact
implied-volatility least squares**. The default floor is 1.0 price/unit-vol.
`ObjectiveConfig(convention='price_mse')` uses mean squared price error.
Neither mode clips model prices. Tiny negatives within the explicit default
`negative_price_tolerance=1e-8` are retained. Material negatives, unexpected
price shapes and nonfinite prices fail the evaluation. Expected pricing,
linear algebra and numerical configuration exceptions return the configured
finite penalty (default `1e6`). Programming errors such as AttributeError
propagate.

Every failed objective call records a category, the six-parameter vector and
exception text. The immutable result stores all such records and their count.
The objective never creates prices or residuals from penalties. A separate
best-valid ledger prevents even an undersized penalty from becoming a model
fit. If every call fails, `CalibrationError` carries the failure ledger and
number of evaluations. A penalty should still be chosen well above plausible
valid losses to guide the optimizer effectively.

## Parameters and optimization

Bounds always have six rows in explicit `H, kappa, theta, sigma, v0, rho` order.
Defaults are `(0.03,0.45), (0.3,3), (0.01,0.10), (0.1,0.8), (0.01,0.10),
(-0.9,-0.1)`. Both endpoints must be valid model parameters. Fixed-H mode
retains the six-row bounds contract and removes H from the optimizer vector;
fixed H must lie within its bounds. The returned parameters always have type
`RoughHestonParams`.

DE uses `seed=42`, `popsize=8` (SciPy's population multiplier), `maxiter=30`,
`tol=1e-6`, one worker and immediate updating. DE polish defaults to false;
if enabled its evaluations are included in the DE count. Explicit L-BFGS-B
then starts at the best valid evaluation, with `maxiter=200`, `ftol=1e-14`,
`gtol=1e-8`. All these settings, bounds, objective controls and penalty are
configurable. The default reference is the bounds midpoint and is inserted
into the DE population; `reference_params` can supply a bounded alternative.
A failed reference has `initial_loss=None`.

The result selects the best valid evaluation from the reference and both
optimization stages. It preserves DE and L-BFGS-B success flags, raw status
(where SciPy provides it), messages, raw losses, iterations and evaluation
counts independently of the selected final loss. Low loss does not imply
optimizer convergence. Objective counts include the single reference call;
runtime includes all requested post-fit validation. Average objective time
excludes post-fit checks and includes failed evaluations.

## Fixed pricing and post-fit validation

An explicit immutable `RoughHestonPricer` is required. Adams is supported;
Padé 4 is used in the practical experiment for speed and remains an
approximation. The objective calls only this fixed pricer. It never refines,
changes CF method/order, falls back, or clips prices.

Same-configuration final repricing always runs, followed by finite-value and
price-bound checks, strike monotonicity/convexity within each financial group
and option type, and parity where both types share a strike. Diagnostics retain
original quote indices. Actual IV inversions are only post-fit reporting, using
the shared solver; unbracketable IVs are omitted with an explicit comparison
count. Maximum market price error, maximum IV error where available, and
validation failure text are separate from the optimization loss. Failed final
repricing returns no prices and no repricing loss.

`refinement_config=RefinementConfig(...)` requests Phase-4 refinement for each
financial group after optimization. Results include convergence, final grids,
price changes, warnings, history and failures. Refinement is empirical numerical
stabilization; for Padé it does not estimate approximation bias or fixed CF
quadrature error. `adams_validation_pricer=<explicit Adams pricer>` independently
reprices the fitted parameters and reports its objective, market price/IV
errors and maximum difference from calibration prices. Neither validation
replaces the optimum or its fixed-configuration final loss.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONPATH=src \
  python examples/rough_heston/calibrate_synthetic.py --quick --refine
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONPATH=src \
  python examples/rough_heston/calibrate_synthetic.py --quick --cross --refine
# Larger optional benchmark: remove --quick (DE popsize 8, 40 iterations;
# L-BFGS-B 300 iterations). This is never run by normal pytest.
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 python -m pytest tests/rough_heston/test_calibration_*.py -q -s
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 python -m pytest tests/rough_heston -q
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 python -m pytest -q
git diff --check
```

The example uses twelve quotes: strikes 90/100/110, calls and puts, maturities
0.25/0.75, spot 100, rates 0.03/0.04 and dividends 0.01/0.015. True parameters
are `(0.2, 1.5, 0.04, 0.4, 0.04, -0.7)`. The fixed sinh grid is spacing 0.05,
120 nodes. Before optimization the example validates this grid at the known
synthetic parameters, then uses it unchanged. Cross-method mode generates
Adams-1600 quotes and also compares with Adams-3200. CI uses two DE iterations
and small deterministic problems, with repricing-based assertions rather than
six-parameter recovery requirements.

## Recorded validation (2026-09-12)

### Same-method synthetic CI fits

The CI surface uses the same 12-quote layout, with rates `0.02+0.02*T` and
dividends `0.01+0.005*T`. These runs used DE popsize 3/maxiter 2, L-BFGS-B
maxiter 100, seed 42 and the broad default bounds.

| Quantity | Fixed H = 0.2 | Free H |
|---|---:|---:|
| Initial midpoint loss | 1.65268e-3 | 1.73324e-3 |
| Final normalized loss | 9.94986e-16 | 3.95678e-11 |
| Maximum price error | 9.44867e-7 | 4.10041e-4 |
| DE evaluations | 45 | 54 |
| L-BFGS-B evaluations | 360 | 378 |
| Total objective evaluations | 406 | 433 |
| Runtime, including refinement | 46.18 s | 62.95 s |
| Failed evaluations | 0 | 0 |

Both passed financial diagnostics and both maturity refinements converged.
Fixed-H fitted parameters were approximately
`(0.2, 1.4999805, 0.04000003, 0.39999705, 0.03999996, -0.70000113)`.
Free-H fitted parameters were approximately
`(0.04107592, 0.46643454, 0.03884494, 0.25045351, 0.04026135, -0.70265014)`.
The latter is a very accurate vanilla fit with materially different H/kappa/
sigma: a practical indication of weak identification in this small surface.
Calls and puts at identical strikes are linked by parity and do not provide
12 independent shape observations. No assertion forces parameter recovery.

### Adams market -> Padé 4 calibration -> Adams validation

Recorded using `--quick --cross --refine`, with the final 120-node grid.
The Adams market surface changed by at most `4.61241e-7` when time resolution
increased from 1600 to 3200. The Padé fixed-grid preflight converged with
maximum changes `2.15384e-9` and `1.42109e-14` at the two maturities.

| Quantity | Fixed H = 0.2 | Free H |
|---|---:|---:|
| Initial normalized loss | 1.62311e-3 | 1.70164e-3 |
| DE loss | 1.75568e-4 | 6.64771e-5 |
| Padé final calibration loss | 9.24500e-13 | 7.55769e-11 |
| Adams repricing loss | 5.08369e-10 | 1.67268e-8 |
| Padé maximum market price error | 6.34727e-5 | 5.68835e-4 |
| Adams maximum market price error | 1.08444e-3 | 2.79129e-3 |
| Padé maximum market IV error | 1.89433e-6 | 1.69767e-5 |
| Adams maximum market IV error | 3.55718e-5 | 2.87408e-4 |
| Maximum Padé/Adams price difference | 1.05311e-3 | 2.77298e-3 |
| Mean objective evaluation time | 0.14383 s | 0.11326 s |
| DE / L-BFGS-B evaluations | 45 / 348 | 54 / 336 |
| Total objective evaluations | 394 | 391 |
| Total calibration runtime | 59.78 s | 46.45 s |
| Failed objective evaluations | 0 | 0 |

IV errors are in decimal volatility; all 12 IV comparisons succeeded in each
validation. Both methods' financial diagnostics passed. Both DE runs exhausted
the deliberately small iteration budget (`success=False`); both L-BFGS-B runs
reported projected-gradient convergence (`status=0`, `success=True`). Their
statuses are retained despite the low losses. Timings are local wall-clock
measurements with other validation work running, not universal performance
claims; quote generation and preflight are outside calibration runtime.

Fixed-H cross-method parameters were
`(0.2, 1.49893736, 0.03999826, 0.40021569, 0.04000478, -0.70008097)`;
free-H parameters were
`(0.04144138, 0.46757093, 0.03883896, 0.25075069, 0.04026302, -0.70297091)`.
The independent repricing makes Padé approximation bias visible; no fit or
method was automatically replaced.

All four final Padé refinement groups converged to spacing 0.025/nodes 260,
with independent final probes. Maximum changes from calibration prices were
`2.16602e-9` / `2.84217e-14` (fixed H) and `1.77458e-9` / `7.10543e-14`
(free H). This Fourier stabilization is much smaller than the cross-method
price discrepancy and does not remove Padé approximation bias.

A smaller automated cross-method check uses one maturity and six call/put
quotes, DE popsize 2/maxiter 1 and L-BFGS-B maxiter 25. Recorded Padé loss was
`1.61185e-15`, Adams loss `4.14531e-10`, maximum Adams price error `7.09489e-4`,
maximum IV error `3.33078e-5`, and runtime 11.08 s. Exact parameter recovery is
not asserted.

### Numerical failures and coverage

The successful synthetic fits above had zero failed evaluations. Separate
controlled tests verify nonfinite CF/prices, material negatives, wrong output
shape, invalid configuration, a real Padé-5 singularity at H=1/6, and failure
ledger counts/vectors/messages. All-failed optimization raises with every call
recorded, while a deliberately undersized penalty still cannot produce fitted
prices. Tiny negative prices remain unchanged. Other tests cover quote shape
and immutability, full financial grouping, parameter order, fixed-H bounds,
vega computation/floor/precedence, deterministic seeding, method preservation,
Adams support, optimizer status, immutable results, failed final repricing and
post-validation sequencing.

An additional check at the fitted cross-method parameters increased Adams time
steps from 1600 to 3200 while keeping the Fourier grid fixed. Maximum price
changes at maturities 0.25/0.75 were `4.61724e-7 / 2.61392e-7` (fixed H) and
`1.90202e-6 / 2.28709e-6` (free H). These empirical resolution changes are much
smaller than the measured Padé/Adams differences, supporting interpretation of
the discrepancy as predominantly approximation bias for these fits. They are
not rigorous error bounds.
