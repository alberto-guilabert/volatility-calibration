# Phase 7A: market ingestion and authoritative forwards

Phase 7A prepares the data, pricing and diagnostic stack for real-market
rough-Heston calibration. It does **not** run or tune a real-market calibration,
choose a market calibration grid, or complete Phase 7. The implementation uses
the same ingestion and adaptation functions for the two `Mid` sheets, with
explicit schema configurations.

## Workbook schemas

Both workbooks have metadata columns in this order:

```text
Act Date | Spot | Expiry | Exp Date | Risk Free | Impl (Yld) | ImplFwd
```

Valuation date and spot occur in the first data row and are shared by the sheet.
The dates are stored as native Excel dates. All quote cells in the populated
Mid tables are positive and present. Blank formatted rows/columns are ignored;
populated cells beneath missing headers are errors.

| Property | SPX Mid | GOOGL Mid |
|---|---|---|
| Workbook | `data/spx/SPX_17_10_25.xlsx` | `data/googl/GOOGL_16_12_25.xlsx` |
| Populated table | A1:R8 | A1:X10 |
| Valuation date | 2025-10-17 | 2025-12-16 |
| Spot | 6543.93 | 307.62 |
| Axis | Percentage forward moneyness, K/F | Absolute strike |
| Quote nodes | 11 per expiry | 17 per expiry |
| IV storage | Percentage points; divide by 100 | Decimal; preserve |
| Rate storage | Percentage points; divide by 100 | Decimal; preserve |
| Dividend-yield storage | Percentage points; divide by 100 | Decimal; preserve |
| Model day count | ACT/365F | ACT/360 |
| Quotes / expiries | 77 / 7 | 153 / 9 |

SPX quote headers:

```text
80%, 85%, 90%, 95%, 97.5%, 100%, 102.5%, 105%, 110%, 115%, 120%
```

For example, IV `38.12`, rate `4.005`, and yield `0.57` become decimal
`0.3812`, `0.04005`, and `0.0057`. Each strike is the header percentage times
the **supplied** maturity forward.

GOOGL quote headers (displayed without binary floating-point noise):

```text
156.860, 188.232, 219.604, 235.290, 250.976, 266.662,
282.348, 298.034, 313.720, 329.406, 345.092, 360.778,
376.464, 392.150, 407.836, 439.208, 470.580
```

The loader preserves the actual stored numeric headers, without rounding them
to this display. IV `0.5423128762478799` and rate `0.0413` are already decimals.
The rate cell's Excel percentage formatting does not change its stored value.
The apparent `313.72` reference grid is not used to construct or modify strikes.
Forward moneyness is computed as the stored strike divided by supplied forward.

Different header representations and units require different schema settings;
there is no ticker-specific pricing logic and no magnitude-based unit guessing.
`SPX_MID_SCHEMA` and `GOOGL_MID_SCHEMA` in `market_data/schema.py` specify these
settings explicitly. Both expose a configurable `day_count`; supported choices
are `ACT/365F` and `ACT/360`.

Bid/Ask sheets are outside Phase 7A. In particular, GOOGL Mid and Bid/Ask have
different dates, curves and grids and must not be matched by tenor labels or
treated as corresponding bounds. No Bid/Ask sheets are used by the smoke check.

## Authoritative inputs and time conventions

The source meanings are preserved:

```text
Risk Free  -> r(T)
Impl (Yld) -> q(T)
ImplFwd    -> F(T)
```

Rates and yields are interpreted as continuously compounded annual inputs.
Maturity uses actual date differences divided by the configured year basis:

```text
SPX:   T = (expiration date - valuation date).days / 365
GOOGL: T = (expiration date - valuation date).days / 360
```

Tenor labels are provenance, not numerical maturity. Dates are not shifted to
weekdays. The selected convention is retained in the schema and every prepared
observation. Source rates, yields, spot, forwards and IVs are never re-estimated
or rescaled to enforce carry consistency (apart from explicit percent-to-decimal
unit conversion).

**The source IV annualization convention is not independently documented.**
These are the selected Phase-7 model-time conventions. GOOGL Mid forwards match
ACT/360 carry to machine precision; this is evidence about its carry inputs,
not independent proof of its IV annualization. SPX is close to ACT/365F carry,
with a larger residual at 18M.

The legacy calendar-days/252 calculation is not inherited: relative to /365 it
increases maturity by 44.84%, and relative to /360 by 42.86%. It is not an actual
business-day count. Existing Heston preprocessing and its defaults are unchanged.

## API and provenance

```python
from pathlib import Path
from volcal.market_data.preprocessing import DataLoader
from volcal.market_data.schema import SPX_MID_SCHEMA, GOOGL_MID_SCHEMA
from volcal.market_data.adapter import prepare_quotes

path = Path("data/spx/SPX_17_10_25.xlsx")
source = DataLoader(path.parent, path.name).load_surface(SPX_MID_SCHEMA, "Mid")
market = prepare_quotes(source)
quotes = market.quotes  # PreparedQuotes; F always supplied by this adapter
observations = market.observations
rejections = market.rejected
```

Use the same calls with the GOOGL path and `GOOGL_MID_SCHEMA`. The old
`DataLoader.load_iv_table()` retains its original output and semantics; the new
schema-aware method does not route through `heston/calibrator/refine_data.py`.
`openpyxl` is declared as a runtime dependency for the new Excel reader.

Every accepted observation records workbook path, sheet, Excel row, original
header, raw source row, valuation date, expiry label, expiration date, day count,
T, S0, r, q, F, K, K/F (`moneyness`), decimal IV, option type, price and vega.
The adapter adds signed, absolute and relative forward residuals and their
diagnostic status. `market.source` retains the complete schema configuration.

Individual invalid source quotes/rows produce `RejectedObservation` records,
including source coordinates, raw values and reason. Missing IV, invalid
forward, bad axis header, nonpositive maturity and invalid calculated prices
are not silently repaired. Ambiguous shared metadata fails the load. If no
usable quotes remain, `MarketDataError.rejected` carries the rejection records.
A rates-forward diagnostic overflow is flagged without discarding an otherwise
usable authoritative-forward quote. Workbook formulas are not evaluated or
silently replaced with cached values by this API.

## Pricing and PreparedQuotes propagation

`PreparedQuotes.F` is optional, finite, positive, one-dimensional and aligned
with the other arrays. Inputs are copied into immutable tuples. It is appended
after the existing optional fields to preserve positional callers. Legacy
groups retain `(T, S0, r, q)` keys; explicit-forward groups add F to the key, so
different supplied forwards cannot share a pricing batch.

The adapter selects puts for K < F and calls for K >= F. It computes prices
and vegas with the separate forward API in `volcal.utils.black`:

\[
D=e^{-rT},\quad
d_1=\frac{\log(F/K)+\tfrac12\sigma^2T}{\sigma\sqrt T},\quad
d_2=d_1-\sigma\sqrt T,
\]

\[
C=D[F\Phi(d_1)-K\Phi(d_2)],\qquad
P=D[K\Phi(-d_2)-F\Phi(-d_1)],
\]

\[
\mathrm{Vega}=DF\phi(d_1)\sqrt T.
\]

Volatility is decimal; vega is price sensitivity per **unit decimal volatility**.
`black.price` and `black.vega` broadcast numeric arrays. `black.iv_solver` is a
scalar bracketed inversion with explicit bounds/bracketing failures. Zero-time
or zero-volatility pricing returns discounted intrinsic; IV inversion requires
positive time. The existing `black_scholes` spot/r/q utilities are unchanged.

Both rough-Heston methods receive the same explicit F through
`RoughHestonPricer.vanilla_price(..., F=...)`. Shared Fourier inversion uses
`log(F/K)`, discounted forward `D*F`, and discounted strike `D*K`. The normalized
CF, Riccati, Padé approximation and Adams solvers are unchanged. No effective q
or replacement spot is manufactured. S0 and q remain source/diagnostic inputs.

The explicit-F path is also used by:

- Calibration quote batching and price residuals.
- Vega fallback when PreparedQuotes contains IV but no supplied vega.
- Post-fit bounds, parity, IV errors and profile financial metrics.
- Optional Adams repricing and every refinement stage/final probe.

Parity is `C - P = D*(F-K)`; call bounds are `[max(D*(F-K), 0), D*F]`, and put
bounds are `[max(D*(K-F), 0), D*K]`. At T=0 an explicit F is used for intrinsic
value; the absent-F path continues to use spot. At positive T, absent-F callers
retain the existing spot/r/q expressions and conventions.

## Forward residual diagnostics

For diagnostics only:

\[
F_{\mathrm{rates}}=S_0e^{(r-q)T},\quad
\Delta=F_{\mathrm{rates}}-F,\quad
\delta=\Delta/F.
\]

The tables display rates/yields as decimals and relative residuals as percent.
The adapter stores the relative residual as a fraction. Figures below are
rounded for display; ingestion preserves the stored precision. Absolute
residual is the magnitude of the signed delta. No residual changes supplied F.

### SPX Mid — ACT/365F

| Expiry | Expiration | T | r | q | Supplied F | Rates F | Delta | Relative (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2M | 2025-12-17 | 0.167123288 | 0.04005 | 0.00570 | 6581.580000000 | 6581.604676101 | +0.024676101 | +0.0003749 |
| 3M | 2026-01-17 | 0.252054795 | 0.03916 | 0.00244 | 6604.730000000 | 6604.778184722 | +0.048184722 | +0.0007295 |
| 6M | 2026-04-17 | 0.498630137 | 0.03707 | 0.00400 | 6652.630000000 | 6652.732024500 | +0.102024500 | +0.0015336 |
| 9M | 2026-07-17 | 0.747945205 | 0.03550 | 0.00411 | 6699.240000000 | 6699.386147488 | +0.146147488 | +0.0021816 |
| 1Y | 2026-10-17 | 1.000000000 | 0.03415 | 0.00422 | 6742.530000000 | 6742.750329331 | +0.220329331 | +0.0032678 |
| 18M | 2027-04-17 | 1.498630137 | 0.03241 | 0.00319 | 6829.110000000 | 6836.855303898 | +7.745303898 | +0.1134160 |
| 2Y | 2027-10-17 | 2.000000000 | 0.03156 | 0.00384 | 6916.580000000 | 6916.970622203 | +0.390622203 | +0.0056476 |

### GOOGL Mid — ACT/360

q is zero at every maturity. All signed, absolute and relative residuals were
zero in the smoke check (agreement to floating-point precision).

| Expiry | Expiration | T | r | Supplied F = Rates F | Delta | Relative (%) |
|---|---|---:|---:|---:|---:|---:|
| 1M | 2026-01-08 | 0.063888889 | 0.04130 | 308.432761359 | 0 | 0 |
| 3M | 2026-03-08 | 0.227777778 | 0.04041 | 310.464559581 | 0 | 0 |
| 6M | 2026-06-08 | 0.483333333 | 0.03860 | 313.413035131 | 0 | 0 |
| 9M | 2026-09-08 | 0.738888889 | 0.03642 | 316.010546603 | 0 | 0 |
| 1Y | 2026-12-08 | 0.991666667 | 0.03779 | 319.366816887 | 0 | 0 |
| 2Y | 2027-12-08 | 2.005555556 | 0.03613 | 330.737820108 | 0 | 0 |
| 3Y | 2028-12-10 | 3.027777778 | 0.03596 | 343.004698016 | 0 | 0 |
| 4Y | 2029-12-09 | 4.038888889 | 0.03625 | 356.122572813 | 0 | 0 |
| 5Y | 2030-12-08 | 5.050000000 | 0.03673 | 370.314170580 | 0 | 0 |

For context, ACT/365F would give GOOGL Mid a maximum absolute residual of
0.939740930 (0.2537686% relative, at 5Y). Legacy /252 would give maximum
absolute residuals of 174.501746227 for SPX and 30.639533436 for GOOGL Mid.
The selected convention is explicit and is not fitted to each observation.

## Validation evidence

The new tests in `tests/market_data/test_adapter.py` and
`tests/rough_heston/test_explicit_forward.py` cover both actual Mid sheets,
exact stored strikes and units, source provenance, rejection reporting,
configurable day counts, independent metadata checks, and legacy loader output.
Pricing tests cover forward validation, grouping, Black/rough-Heston parity,
legacy equivalence, supplied-vega precedence, fallback vega, IV round trips,
profile metrics, refinement and optional Adams propagation.

The pipeline propagation test replaces optimizers with fixed synthetic stubs;
it does not optimize a market surface. Actual Padé and Adams prices and repricing
are exercised separately on small synthetic quote batches.

For a numerical check with T=0.5, S0=100, r=0.03, q=0.01, K=[90,100,110],
sigma=0.25 and rough parameters (H,kappa,theta,sigma,v0,rho) =
(0.2,1.5,0.04,0.4,0.04,-0.7):

| Check | Maximum absolute difference |
|---|---:|
| Black vs legacy price, consistent F | 1.954e-14 |
| Black vs legacy vega, consistent F | 2.487e-14 |
| Padé explicit vs legacy price, consistent F | 2.842e-14 |
| Adams explicit vs legacy price, consistent F | 2.842e-14 |
| Explicit-F parity, each rough method | 7.106e-15 |
| Explicit-F prices after changing source spot/q, each method | 0 |

The deliberately inconsistent-forward check keeps S0=100, r=0.03, q=0.01 but
sets F to 105 or 115, versus rates-derived F approximately 101.005. At K=100:

| Method | Legacy call | Call at F=105 | Call at F=115 |
|---|---:|---:|---:|
| Padé | 5.64895606 | 8.41265715 | 16.55669872 |
| Adams | 5.64845244 | 8.41244886 | 16.55705365 |

Changing source spot/q to 230/-0.2 with F=115 fixed leaves these prices exactly
unchanged. This establishes that the explicit path does not silently reconstruct
carry from spot and q. The assertions use tight absolute tolerances; they do not
claim Padé and Adams are identical approximations.

Reproduce the ingestion smoke check without calibration:

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 .venv/bin/python examples/rough_heston/inspect_market_data.py
```

| Output | SPX Mid | GOOGL Mid |
|---|---:|---:|
| Quotes | 77 | 153 |
| Expiries | 7 | 9 |
| Rejected observations | 0 | 0 |
| Minimum T | 0.167123287671 | 0.063888888889 |
| Maximum T | 2 | 5.05 |
| Minimum K/F | 0.8 | 0.423586274742 |
| Maximum K/F | 1.2 | 1.525713409712 |
| First maturity S0 | 6543.93 | 307.62 |
| First maturity r | 0.04005 | 0.0413 |
| First maturity q | 0.0057 | 0 |
| First maturity F | 6581.58 | 308.43276135902113 |
| Maximum absolute forward residual | 7.74530389765641 | 0 |
| Maximum absolute relative residual | 0.00113416007322424 | 0 |

Validation commands:

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 .venv/bin/python -m pytest -q
git diff --check
```

Final validation: **582 tests passed in 196.00 seconds**, including all existing
rough-Heston tests and 41 new Phase-7A tests. `git diff --check` passed. The
legacy profile IV-solver call path remains intact for absent-F quotes, including
its existing missing-IV-coverage regression test. No real-market calibration
was run.

## Remaining work before market calibration

- No real-market optimizer run, parameter tuning, fitted surface or calibration
  quality claim is part of Phase 7A.
- Numerical grids, parameter bounds and Padé/Adams approximation errors need
  assessment over the actual market maturity/strike ranges, especially GOOGL's
  wide wings and 5.05 model-year horizon. Synthetic tests do not establish this.
- Prices are European Black conversions of source IV, not independent traded
  option-price observations. Source IV annualization and exercise/IV construction
  conventions are not independently documented by these workbooks.
- Forward discrepancies remain visible, particularly SPX 18M. No curve fit,
  implied-q substitution, interpolation, or adjustment has been performed.
- Bid/Ask alignment, liquidity selection, quote weighting and market fit reporting
  remain outside this phase. No bid/ask accuracy claims can be drawn from Mid-only
  ingestion.
- Legacy Heston preprocessing still has its previous behavior; Phase 7A bypasses
  it rather than modifying it. `benchmarks/` is unrelated and untouched.
