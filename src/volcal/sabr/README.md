# SABR model

Implementation of the SABR stochastic volatility model.

### Pricing engines
- Hagan et al. (2002) asymptotic expansion

### Calibration
- Global + local optimization on implied-volatility surfaces

### Conventions
- Forward-based formulation
- Continuous dividend yield

---

## Module structure
```text
sabr/
├── calibrator/
└── pricer/
    └── hagan/
        └── price.py
```

---

## References
Hagan, P., Kumar, D., Lesniewski, A., and Woodward, D. (2002). 
Managing Smile Risk, Wilmott Magazine, September, 84-108.
https://doi.org/10.1002/wilm.10290

