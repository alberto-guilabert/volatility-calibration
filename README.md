# Heston Pricer & Calibration Toolkit

Vectorized **Heston (1993)** option pricer and full **calibration pipeline** for implied volatility surfaces.  
This repository includes:

- A **Gauss–Laguerre Heston pricer** (P1/P2 formulation) with numerical stability enhancements.  
- **Black–Scholes utilities** (price, vega, implied volatility solver).  
- **Excel/CSV adapter** for converting Bloomberg-style IV tables into a long-format DataFrame.  
- A complete **calibration workflow** using Differential Evolution + L-BFGS-B, with bid/ask and smile visualization.

> ⚠️ For research and educational purposes only. Validate before production use.

---

## ✨ Features

- **Fast and vectorized**: prices many strikes per maturity with a single CF evaluation.  
- **Stable characteristic function** implementation (Heston trap mitigations).  
- **Robust implied volatility solver** via Brent’s method with dynamic bracketing.  
- **Vega-weighted loss** prevents OTM noise from dominating the calibration.  
- **Visual validation**: generates bid/ask + model IV smiles for each maturity.  

---

## 📁 Repository Structure

```bash
.
├── black_scholes_calculator.py   # Black–Scholes price, vega, and IV solver
├── convert_df.py                 # Excel/CSV adapter for Bloomberg-style IV tables
├── heston_vanilla_pricer.py      # Vectorized Heston pricer (Gauss–Laguerre)
├── heston_calibration.py         # Main calibration script (DE + L-BFGS-B)
└── SPX_17_10_25.xlsx             # Example dataset
```

## 📚 References

- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.*  
  **The Review of Financial Studies**, 6(2), 327–343.  
  [doi:10.1093/rfs/6.2.327](https://doi.org/10.1093/rfs/6.2.327)

- Gatheral, J. (2006). *The Volatility Surface: A Practitioner’s Guide.*  
  Wiley Finance Series.

- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The Little Heston Trap.*  
  **Wilmott Magazine**, January, 83–92.

- Ortiz Ramírez, A., Venegas Martínez, F., & Martínez Palacios, M. T. V. (2021).  
  *Parameter calibration of stochastic volatility Heston’s model: constrained optimization vs. differential evolution.*  
  **Contaduría y Administración**, 67(1), 309.  
  [doi:10.22201/fca.24488410e.2022.2789](https://doi.org/10.22201/fca.24488410e.2022.2789)
