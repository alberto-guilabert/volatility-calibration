# Heston Pricer & Calibration Toolkit

Vectorized Heston (1993) option pricer and full calibration pipeline for implied volatility surfaces, implemented as a reusable Python package using a modern `src/` layout.

The project is designed for robust calibration, numerical stability, and practical use in equity derivatives research.

---

## Features

- Vectorized Heston vanilla pricer supporting many strikes per maturity with a single characteristic function evaluation.
- Gauss–Laguerre quadrature (P1/P2 formulation) with numerical stability enhancements.
- NEW: Heston vanilla pricing via sinh-acceleration, enabling faster and more stable numerical integration for oscillatory Fourier integrals.
- Black–Scholes utilities: price, vega, and implied volatility solver.
- Robust implied volatility inversion using Brent’s method with dynamic bracketing.
- Calibration workflow based on Differential Evolution followed by L-BFGS-B local refinement.
- Vega-weighted loss function, reducing the impact of noisy deep OTM options.
- Visual validation tools for bid/ask and model-implied volatility smiles.

---

## Repository Structure

heston-model-calibration/
├── pyproject.toml
├── README.md
├── data/
│   └── spx/
│       └── SPX_17_10_25.xlsx
├── examples/
└── src/
    └── heston_model_calibration/
        ├── __init__.py
        ├── calibration/
        │   ├── __init__.py
        │   └── heston_calibration.py
        ├── pricing/
        │   ├── __init__.py
        │   ├── black_scholes.py
        │   ├── heston.py
        │   └── heston_sinh.py
        └── utils/
            ├── __init__.py
            └── market_data_preprocessing.py

The project follows a src-based package layout to prevent accidental imports from the working directory and to enforce correct installation via pip.

---

## Installation

From the repository root:

pip install -e .

This installs the package in editable mode and makes `heston_model_calibration` importable from any location.

---

## Example Usage

from heston_model_calibration.pricing.heston import heston_price
from heston_model_calibration.pricing.heston_sinh import heston_price_sinh

Both Gauss–Laguerre and sinh-accelerated pricing methods are available for vanilla options.

---

## Data

The data/spx/ directory contains an example implied volatility dataset (SPX options) used for calibration and validation.

In practical workflows, this data is expected to be replaced or loaded dynamically from external sources.

---

## References

Heston, S. L. (1993).
A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.
The Review of Financial Studies, 6(2), 327–343.
https://doi.org/10.1093/rfs/6.2.327

Gatheral, J. (2006).
The Volatility Surface: A Practitioner’s Guide.
Wiley Finance Series.

Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
The Little Heston Trap.
Wilmott Magazine, January, 83–92.

Ortiz Ramírez, A., Venegas Martínez, F., & Martínez Palacios, M. T. V. (2021).
Parameter calibration of stochastic volatility Heston’s model: constrained optimization vs. differential evolution.
Accounting and Management, 67(1), 309.
https://doi.org/10.22201/fca.24488410e.2022.2789
