# Phase 4: explicit numerical refinement and diagnostics

Only the refinement/diagnostic layer and its exports/tests were added. The
Adams and Padé mathematics, Fourier kernel and default CF method are unchanged.
No calibration, market-data handling, fallback, order switching or price clipping.

## API

All APIs below are exported from `volcal.rough_heston.pricer`.

- `refine_prices(pricer, *, T, K, option_params, rough_heston_params,
  option_type='call', config=RefinementConfig(), raise_on_failure=False)`.
- Immutable policies: `RefinementConfig`, `AdamsRefinementConfig`,
  `FourierRefinementConfig`. Initial resolution is supplied by the immutable
  `RoughHestonPricer`, `AdamsConfig` and `SinhConfig` snapshot. There is no second
  conflicting source of initial numerical settings.
- Immutable `RefinementResult` and `RefinementStep`; prices, changes, warnings,
  diagnostics and histories are tuples. Scalar requests return one-element
  price tuples. Results retain the policy and initial/final pricer snapshots.
- `ConvergenceError.result` retains the entire failed result.
- Pure diagnostics: `finite_values`, `no_arbitrage_bounds`, `price_bounds`,
  `put_call_parity`, `strike_monotonicity`, `strike_convexity`.
  `Diagnostic` contains residuals, violation indices and a `passed` property.
- `price_change` exposes the convergence comparison independently.

```python
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import (
    RoughHestonPricer, AdamsConfig, SinhConfig, RefinementConfig,
    FourierRefinementConfig, refine_prices,
)

result = refine_prices(
    RoughHestonPricer('adams', AdamsConfig(400),
                      SinhConfig(spacing=.05, nodes=100)),
    T=.5, K=[80, 100, 120], option_params=(100, .04, .015),
    rough_heston_params=RoughHestonParams(.2, 1.5, .04, .4, .04, -.7),
    config=RefinementConfig(atol=2e-6, rtol=1e-6,
        fourier=FourierRefinementConfig(max_nodes=2400,
                                        truncation_increment=20)),
)
assert result.converged
```

## Convergence semantics

Every requested price must satisfy
`abs(new-old) <= atol + rtol*max(abs(old), abs(new))`.
Relative changes use the same scale, with zero for two zero prices. The absolute
term protects near-zero prices; for example 0 to 1e-10 passes atol=1e-9 even
though its relative change is 1. These are empirical stabilization tests, not
rigorous error estimates or guarantees of accuracy relative to the true model.

Adams stages are time resolution, Picard count, Fourier spacing, then truncation.
Padé stages are only spacing and truncation; order remains fixed. Halving spacing
also doubles the interval count to preserve integration extent. Truncation adds
intervals at fixed spacing. Each stage must make at least one comparison.

Final validation probes each dimension independently from the final configuration.
Probe configurations and prices are recorded but do not replace the final anchor.
If a probe fails, the result does not claim convergence. This is deliberately
conservative: callers may explicitly choose a new initial configuration/policy.
Padé approximation error and fixed internal time-quadrature error remain outside
this layer's convergence claim. A warning states this on every Padé result.

Financial checks evaluate both calls and puts together, using one shared CF batch.
Parity is a reconstruction consistency check, not independent CF validation.
Monotonicity and convexity checks sort unique strikes separately for calls/puts;
standalone curve functions require strictly increasing strikes. Convexity uses
successive secant slopes, supporting irregular strike spacing; its tolerance is
in slope units. Other financial tolerances are in price units.

## Representative histories

Synthetic parameters and policy are those in the example above. Padé uses the
same initial Fourier configuration and order 4. Maximum absolute price changes:

| Stage | Adams | Padé 4 |
|---|---:|---:|
| Time 400 → 800 | 5.506e-6 | — |
| Time 800 → 1600 | 1.704e-6 | — |
| Picard 2 → 3 | 8.819e-8 | — |
| Spacing .05 → .025, extent 5 | 1.433e-7 | 1.461e-7 |
| Nodes 200 → 220, spacing .025 | 1.467e-5 | 1.462e-5 |
| Nodes 220 → 240 | 1.551e-8 | 1.478e-8 |
| Final time probe 3200 | 5.332e-7 | — |
| Final Picard probe 4 | 1.007e-9 | — |
| Final spacing probe .0125, nodes 480 | 7.106e-14 | 4.263e-14 |
| Final truncation probe nodes 260 | 8.669e-13 | 8.384e-13 |

Both converge. Final calls, rounded to eight decimals:
Adams `(21.54490289, 5.78234078, 0.18506606)`;
Padé `(21.54452266, 5.78282388, 0.18493212)`.
The difference between these methods illustrates why Fourier stabilization
cannot establish Padé approximation accuracy.

## Failure behavior

Nonfinite CF exceptions, nonfinite prices, Padé rational-pole errors and the
known order-5 singularity at H=1/6 retain their failure details. Failed numerical
attempts appear in history; the final configuration/prices remain the last
accepted anchor (or the raw initial prices if initial financial checks fail).
An exception before any prices exist leaves the initial price tuple empty.
Invalid caller inputs still raise validation errors normally.

The known order-4 short-maturity wing (H=.05, T=.05,
K=100*exp(.025*.05)*1.2, other parameters as above) produces a negative call
between -3e-6 and zero. Default sanity tolerance rejects it via bounds diagnostics;
the negative value is retained, not repaired. Negative prices within
`sanity_atol` are retained and explicitly warned about. They may pass numerical
stabilization, subject to the other financial checks. Material negatives, bounds
violations and parity violations prohibit convergence.

Time-step, Picard and integration-node caps, including caps needed by final
probes, return `converged=False` with a stage-specific reason. Iteration exhaustion
also returns failure. No parameter is silently clamped to its cap. Failed results
can instead raise `ConvergenceError` with `raise_on_failure=True`. All loops are
bounded and no recursive retries are used.

## Performance and validation

Local measurements with `OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2`, after JIT
warmup, for three strikes: refinement approximately 11.8 s Adams / 0.68 s Padé.
Separate single-price timings at the final configurations were approximately
1.01 s / 0.12 s; a batch of the five diagnostic checks averaged 0.43 ms / 0.42 ms
over 1000 repetitions. These are smoke measurements, not benchmark guarantees;
other test processes were active during some measurements. The extra CF solves,
especially the 3200-step final Adams probe, dominate refinement cost.

Tests cover real convergence, decreasing Adams differences, independent Fourier
stages, caps/iteration exhaustion, NaN/inf prices and CFs, singularity preservation,
known negative wings, tiny negatives, near-zero comparison, immutability, fixed CF
selection, absence of recursion, parity and standalone financial diagnostics.

Final validation: focused tests 29 passed; all rough-Heston tests 478 passed in
85.77 s; full pytest 478 passed in 86.39 s. `git diff --check` and whitespace
checks of new files were clean. Six files changed/created, 704 added lines,
no deletions. Nothing staged or committed.
