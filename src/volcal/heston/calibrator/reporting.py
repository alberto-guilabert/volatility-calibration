from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence
import numpy as np


def _line(char: str = "=", n: int = 94) -> str:
    return char * n

def _fmt_sci(x: float) -> str:
    return f"{x: .6e}"

def _fmt_elapsed(t0: datetime, t1: datetime | None = None) -> str:
    t1 = datetime.now() if t1 is None else t1
    sec = (t1 - t0).total_seconds()
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"

def _fmt_params_table(params: Mapping[str, float], keys: Sequence[str]) -> str:
    # Table: key | value
    rows = []
    for k in keys:
        v = float(params[k])
        rows.append(f"  {k:<8}  {v: .10g}")
    return "\n".join(rows)

def feller_ok(params: Mapping[str, float]) -> bool:
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(params["sigma"])
    return bool(2.0 * kappa * theta > sigma * sigma)


def print_calibration_summary(
    *,
    title: str,
    method: str,
    keys: Sequence[str],
    x_init: np.ndarray,
    x_de: np.ndarray | None,
    x_opt: np.ndarray,
    params_opt: Mapping[str, float],
    loss_init: float | None,
    loss_de: float | None,
    loss_opt: float,
    t0: datetime,
    notes: str | None = None,
) -> None:
    print()
    print(_line("="))
    print(f"{title}")
    print(_line("="))

    print(f"Method        : {method}")
    print(f"Elapsed       : {_fmt_elapsed(t0)}")

    if loss_init is not None:
        print(f"Loss (init)   : {_fmt_sci(loss_init)}")
    if loss_de is not None:
        print(f"Loss (DE)     : {_fmt_sci(loss_de)}")
    print(f"Loss (opt)    : {_fmt_sci(loss_opt)}")

    print(_line("-"))
    print("Parameters (optimal)")
    print(_line("-"))
    print(_fmt_params_table(params_opt, keys))

    # deltas vs init
    try:
        dx = np.asarray(x_opt, dtype=float) - np.asarray(x_init, dtype=float)
        print(_line("-"))
        print("Δ vs init (x_opt - x_init)")
        print(_line("-"))
        for k, d in zip(keys, dx):
            print(f"  {k:<8}  {d: .6e}")
    except Exception:
        pass

    # optional: show DE -> opt delta
    if x_de is not None:
        try:
            dx2 = np.asarray(x_opt, dtype=float) - np.asarray(x_de, dtype=float)
            print(_line("-"))
            print("Δ vs DE (x_opt - x_de)")
            print(_line("-"))
            for k, d in zip(keys, dx2):
                print(f"  {k:<8}  {d: .6e}")
        except Exception:
            pass

    # checks
    print(_line("-"))
    print("Sanity checks")
    print(_line("-"))
    if all(k in params_opt for k in ("kappa", "theta", "sigma")):
        print(f"  Feller      : {feller_ok(params_opt)}  (2*kappa*theta > sigma^2)")
    else:
        print("  Feller      : n/a")

    if notes:
        print(_line("-"))
        print("Notes")
        print(_line("-"))
        print(notes)

    print(_line("="))
