from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from volcal.utils.params import vec_to_params


@dataclass(frozen=True, slots=True)
class LossConfig:
    param_keys: Sequence[str] = ("v0", "kappa", "theta", "sigma", "rho")
    vega_floor: float = 1e-12


def loss_function(
    x: ArrayLike,
    *,
    pricer: Any,
    S0: float,
    df: pd.DataFrame,
    cfg: LossConfig = LossConfig(),
) -> float:
    heston_params = vec_to_params(x, cfg.param_keys)

    T = df["To expiry"].to_numpy(dtype=np.float64)
    K = df["Strike"].to_numpy(dtype=np.float64)
    r = df["Risk Free"].to_numpy(dtype=np.float64)
    q = df["Impl (Yld)"].to_numpy(dtype=np.float64)
    market = df["Market price"].to_numpy(dtype=np.float64)
    vega = np.maximum(df["Vega"].to_numpy(dtype=np.float64), cfg.vega_floor)
    opt_type = df["Option type"].to_numpy(dtype=str)

    model = np.empty_like(market)
    for Tj in np.unique(T):
        idx = (T == Tj)
        model[idx] = pricer.vanilla_price(
            T=float(Tj),
            K=K[idx],
            option_params=(float(S0), float(r[idx][0]), float(q[idx][0])),
            heston_params=heston_params,
            option_type=opt_type[idx],
        )

    err = (market - model) / vega
    return float(np.mean(err * err))
