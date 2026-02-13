# Heston Pricer & Calibration Toolkit

Vectorized **Heston (1993)** option pricer and full **calibration pipeline** for implied volatility surfaces, implemented as a reusable Python package using a modern `src/` layout.

The project is designed for **robust calibration**, **numerical stability**, and **practical use** in equity derivatives research.

---

## Features

- **Vectorized Heston vanilla pricer** supporting many strikes per maturity with a single characteristic function evaluation.
- **Gauss–Laguerre quadrature** (P1/P2 formulation) with numerical stability enhancements.
- **NEW: Heston vanilla pricing via sinh-acceleration**, enabling faster and more stable numerical integration for oscillatory Fourier integrals.
- **Black–Scholes utilities**: price, vega, and implied volatility solver.
- **Robust implied volatility inversion** using Brent’s method with dynamic bracketing.
- **Calibration workflow** based on Differential Evolution followed by L-BFGS-B local refinement.
- **Vega-weighted loss function**, reducing the impact of noisy deep OTM options.
- **Visual validation tools** for bid/ask and model-implied volatility smiles.

---

## Repository Structure

```text
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
