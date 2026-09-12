# Phase 3: shared sinh Fourier pricing

Baseline: clean commit `47e9b0b814fdb0ad99eb1c11716a591635258008`;
385 tests passed in 15.84 seconds before changes. No validated CF or parameter
mathematics changed. All data below are synthetic, with spot 100.

## Public interface

```python
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import (
    RoughHestonPricer, AdamsConfig, PadeConfig, SinhConfig,
    vanilla_price_from_cf,
)

params = RoughHestonParams(H=.2, kappa=1.5, theta=.04,
                          sigma=.4, v0=.04, rho=-.7)
pricer = RoughHestonPricer(
    cf_method="pade", cf_config=PadeConfig(order=4),
    integration_config=SinhConfig(omega1=-.5, omega=0, b=1,
                                  spacing=.025, nodes=240),
)
prices = pricer.vanilla_price(
    T=.5, K=[80, 100, 120], option_params=(100, .04, .015),
    rough_heston_params=params, option_type=["put", "call", "call"],
)
# Explicit alternative: RoughHestonPricer("adams", AdamsConfig(time_steps=1600))
```

`RoughHestonPricer()` defaults to Adams with 1000 time steps and two Picard
corrections: the numerical reference / correctness-first method.
`RoughHestonPricer(cf_method="pade")` explicitly selects the faster approximation,
with order 4 as the default within Padé mode. Padé has quantified approximation
error and known parameter/wing limitations, including the stabilized negative
wing documented below. Methods and Padé orders never switch automatically;
raw numerical failures and invalid prices remain visible. Scalar strike returns float; a 1-D strike vector returns
an array. Option labels are scalar or match strike shape. Expiry returns
intrinsic without CF evaluation, including for Padé 5 at its singular alpha.
Positive-maturity singularities propagate. No file I/O, plotting, market data,
refinement, fallback, calibration, price clipping, or timing instrumentation
is present in the pricing implementation.

`vanilla_price_from_cf(cf, ...)` accepts a callable `cf(u)` bound to maturity.
This permits independent validation and uses exactly the same inversion as
both rough-Heston methods. One vector CF evaluation is shared by all strikes.

## Fourier derivation and contour contract

Write X=log(S_T/F_T), m=log(F_T/K). The covered-call payoff is
K min(exp(X+m),1). On -1<Im(u)<0 the Fourier transform of min(exp(x),1) is
1/(u(u+i)): integrating exp((1-iu)x) below zero and exp(-iux) above zero
and adding gives this expression. Thus its discounted expectation is

    K exp(-rT)/(2 pi) integral exp(i u m) phi(u,T)/(u(u+i)) du.

The mapping is u=i omega1+b sinh(y+i omega), with full Jacobian
b cosh(y+i omega). Conjugate symmetry reduces the full contour to the real
part of its positive half with prefactor 1/pi. Both finite trapezoidal
endpoints have half weight. Calls subtract the covered call from
S0 exp(-qT); puts subtract it from K exp(-rT). No extra carry phase enters
CF evaluation. Tests compare the half integral against a full contour and
check the Jacobian by finite differences.

`bounds.py` supplies explicit immutable settings, not error estimates.
The default omega=0 keeps the infinite contour inside the stock moment strip.
Nonzero omega requires analytic continuation and tail decay beyond that strip;
only the intercept, orientation, and finite arithmetic configuration are
validated. Finite CF values do not certify that continuation or accuracy.
`nodes` counts positive-half intervals, with extent spacing*nodes.
Spacing experiments hold extent fixed; truncation experiments hold spacing
fixed. Default settings are a starting resolution, not a universal guarantee.

## Independent Black–Scholes validation

Injected phi(u)=exp(-sigma² T (u²+i u)/2), sigma=.25, r=.04, q=.015.
Calls and puts: K=50,80,100,120,200; T=.001,.02,.5,3; omega=-.015,0,.015.
With spacing=.005, nodes=2000, maximum absolute error against the existing
`volcal.utils.black_scholes.price` was **1.279e-13**.

Independent spacing experiment (T=.5, sigma=.2, K=80,100,120, r=.03,q=.01):

| spacing (extent 6) | max absolute BS error |
|---|---:|
| .4 | .0584136 |
| .2 | 1.56322e-5 |
| .1 | 1.19194e-12 |
| .05 | 4.08562e-14 |

At spacing=.05, extents 3,4,5,6 gave errors .276010, 3.86877e-5,
4.08562e-14, 4.08562e-14 respectively.

## Financial properties and rough-Heston Fourier convergence

For H=.2, kappa=1.5, theta=v0=.04, sigma=.4, rho=-.7, T=.5,
r=.04, q=.015 and 61 equally spaced strikes from 50 to 200:

| method | max parity error | minimum lower-bound margin | largest call increment | minimum second difference |
|---|---:|---:|---:|---:|
| Adams 1000 | 0 | 5.94126e-8 | -2.63245e-8 | 1.22562e-8 |
| Padé 4 | 0 | 9.01536e-7 | -2.68203e-7 | 6.78677e-8 |

Upper bounds passed as well. Spot/strike scaling by three passed at 3e-12
absolute tolerance. Mixed calls/puts and scalar/vector equivalence passed.
Expiry and T=.001, K=50,100,200 passed with explicitly finer spacing=.005,
extent=9. The reasonable tested wing range is K/S0 in [.5,2], not all strikes.

Holding Adams at 1600 time steps, spacing .2,.1,.05,.025 (extent 6)
gave successive maximum price changes 1.55724e-5, 5.01075e-11, 2.41585e-13.
Padé 4 gave 1.55724e-5, 5.67866e-11, 2.27374e-13.
At fixed spacing=.025, extents 4,5,6,6.5 gave successive changes
.0226608, 1.46650e-5, 8.81073e-13 for Adams, and
.0225573, 1.46110e-5, 8.38440e-13 for Padé 4.
These experiments change Fourier resolution only.

## CF time resolution and approximation errors

All rows use kappa=1.5, theta=v0=.04, r=.04, q=.015, K/F=.8,1,1.2.
Fourier spacing=.02; extent=7 for row A and 5.5 for B–D.
Adams reference has 3200 steps. Its changes from 1600 steps, at fixed Fourier
nodes, are given separately:

| row | H | T | rho | sigma | Adams 1600→3200 max change |
|---|---:|---:|---:|---:|---:|
| A | .05 | .05 | -.7 | .4 | 2.58497e-7 |
| B | .2 | .5 | -.7 | .4 | 4.46421e-7 |
| C | .49 | 2 | .7 | .8 | 3.24425e-7 |
| D | .5 | 1 | 0 | .1 | 3.21090e-9 |

Maximum absolute price errors versus that reference:

| row | Padé 2 | Padé 3 | Padé 4 | Padé 5 |
|---|---:|---:|---:|---:|
| A | .0169606 | .000135083 | .00123641 | .00126849 |
| B | .0404360 | .00687417 | .000576609 | .000188515 |
| C | .249458 | .0918557 | .0362805 | .0148758 |
| D | .00354381 | .000971699 | .000244160 | .0000556749 |

Maximum absolute IV errors in decimal volatility (multiply by 10,000 for
volatility basis points). Only reference BS vega >1 is included, excluding
ill-conditioned near-intrinsic wings. IV is obtained using the existing solver.

| row | Padé 2 | Padé 3 | Padé 4 | Padé 5 |
|---|---:|---:|---:|---:|
| A | .00190306 | 3.31177e-7 | .000138731 | .000142331 |
| B | .00189175 | .000255667 | .0000414263 | .0000346393 |
| C | .00459150 | .00169047 | .000667661 | .000273751 |
| D | .0000906213 | .0000248480 | .00000624357 | .00000142370 |

These are empirical approximation budgets, not universal accuracy claims.
Order is not monotonically related to accuracy. Adams is the default pricing
method; order 4 remains the default only within explicitly selected Padé mode.

## Performance after correctness checks

41 batched strikes 80–120, T=.5, parameters from row B, default Fourier grid
(241 CF evaluations). Median of three warm calls, including CF evaluation,
node generation and pricing; Numba compilation excluded. Local machine timing
is descriptive and not a test threshold. Reference: Adams 3200 on the same grid.

| method | median milliseconds | max price error |
|---|---:|---:|
| Adams 1600 | 210.47 | 6.36387e-7 |
| Padé 2 | 80.44 | .0444780 |
| Padé 3 | 90.79 | .00705840 |
| Padé 4 | 102.72 | .000863457 |
| Padé 5 | 144.56 | .000470358 |

## Legacy discrepancies and failure regions

- Retained the covered-call transform, normalized-forward phase, sinh map,
  Jacobian and symmetry reduction. Replaced strike-specific legacy orchestration
  with one contour and one CF batch shared across strikes.
- Legacy Adams assigned half weight only at zero. The new finite-interval
  trapezoid assigns half weight at the truncation endpoint too.
- Rejected the legacy `zT` asymptotic drift formula: its v0*rho term lacks
  the sigma divisor present in the accompanying decay coefficient and in
  the classical-Heston limit. No replacement asymptotic estimate is needed.
- Rejected absolute-value/floor treatment of the directional decay coefficient:
  negative decay must not be converted into positive decay. The unverified
  Hbound=100 and formulas based on log(Hbound/eps) are not rigorous bounds.
- `if T > 0*0.25` makes the positive short-maturity branch unreachable.
  Neither branch, fixed-point truncation iterations, arbitrary floors nor
  any legacy refinement/retry orchestration was migrated.
- An initial BS probe at T=.001, spacing=.025, extent=8 had 1.17e-4 wing
  error even on the horizontal contour. Using omega=±.15 for both strike
  directions caused catastrophic cancellation (errors around 1e34). A smaller
  angle and finer spacing stabilized it; no automatic correction was added.
- Adams with H=.1, sigma=.3, T=2 and extent=6 failed at 400 and 800 time
  steps (large-frequency path overflow). At 1600 it produced finite prices.
  A regression requires the under-resolved failure to propagate, never switch.
- Padé 4 at row A's K/F=1.2 gives **-1.64123e-6**; spacing=.01 and extent=7.5
  gives **-1.64173e-6**. Adams gives +4.52737e-6. This finite approximation
  violates the zero lower bound. A regression explicitly asserts the raw
  negative value and its Fourier stabilization. Financial-property success
  above does not extend to this region; no clipping conceals the violation.
- Padé 5 at alpha=2/3 raises the existing informative error for T>0.
  Invalid contours, input shapes/types, nonfinite CF values, wrong CF output
  shapes and nonfinite quadrature are covered by assertion-based tests.

## Files and scope

Created `src/volcal/rough_heston/pricer/{__init__,main,sinh,bounds}.py`,
`tests/rough_heston/test_{sinh_inversion,pricer,pricer_adams_vs_pade}.py`, and
this report. No existing CF, calibration, market-data or legacy files changed.
No staging or commits. Phase 3 only.

Final validation: all rough-Heston tests **449 passed in 39.97s**;
full pytest suite **449 passed in 40.05s**. `git diff --check` passed, and
additional `git diff --no-index --check` checks covered every new untracked
file. Eight new Phase-3 files; no tracked-file modifications.
