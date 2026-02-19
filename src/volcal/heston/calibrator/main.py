import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from datetime import datetime
from pathlib import Path

# Project modules
from volcal.market_data.preprocessing import DataLoader
from volcal.utils import black_scholes as bs
from volcal.heston.pricer.main import HestonPricer
from .refine_data import refine_data
from volcal.utils.params import vec_to_params, params_to_vec
from .loss import loss_function, LossConfig
from .callbacks import make_de_callback, make_lbfgs_callback, DEEarlyStop
from .reporting import print_calibration_summary


############################## IMPORT DATA #############################
# Volatility file & folder:
# - 'book_name' must exist inside 'folder_name'.
# - preprocessing.load_iv_table returns: Spot (S0), valuation date (act_date), and adapted DataFrame.
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "spx"
BOOK_NAME = 'SPX_17_10_25.xlsx'

loader = DataLoader(DATA_DIR, BOOK_NAME)
S0, act_date, df = loader.load_iv_table('Mid')
option_args = (S0, df['Risk Free'], df['Impl (Yld)'])

# Refine input data
df = refine_data(S0, act_date, df, mn_low=0.8, mn_high=1.2, vega=True, trading_days=252, market_quote="otm")

# Build the pricer engine
pricer = HestonPricer("laguerre").config(N=185)

# Configure the loss function
PARAM_KEYS = ("v0", "kappa", "theta", "sigma", "rho")
cfg = LossConfig(
    param_keys=PARAM_KEYS,
    vega_floor=1e-12
)


########################################################################
######################### INITIAL HESTON PARAMS ########################
# Heuristic seed:
# - v0  ~ (ATM IV of the closest-to-expiry slice)^2.
# - theta ~ v0 (neutral long-run variance).
# - kappa, sigma, rho in typical equity ranges.
# - lambda = 0 (risk-neutral pricing).
print(df['Strike'] - S0)
iv_atm_guess = float(df.loc[(df['Strike'] - S0).abs().idxmin(), 'IV'])
v0_init = max(1e-6, iv_atm_guess**2)
theta_init = v0_init
kappa_init = 3.0
sigma_init = 0.5
rho_init = -0.5
x0_init = np.array([v0_init, kappa_init, theta_init, sigma_init, rho_init], dtype=float)
print("Initial params dict:", vec_to_params(x0_init, cfg.param_keys))

# Parameter bounds
bounds = [
    (1e-4, 1),    # v0   >= 0
    (1e-4, 15),   # kappa >= 0
    (1e-4, 1),    # theta >= 0
    (1e-4, 2),    # sigma >= 0
    (-0.9, 0.1),  # rho   in (-1, 1)
]
########################################################################


########################################################################
############################## CHECKS ##################################
# Put/Call parity under Heston with the initial seed:
# seed_params = vec_to_params(x0)

# heston_calls = df.apply(
#     lambda row: hp.vanilla_price(
#         T=row['To expiry'],
#         K=row['Strike'],
#         option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
#         heston_params=seed_params,
#         option_type="call",
#         N=185,
#     ),
#     axis=1
# )


# heston_puts = df.apply(
#     lambda row: hp.vanilla_price(
#         T=row['To expiry'],
#         K=row['Strike'],
#         option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
#         heston_params=seed_params,
#         option_type="put",
#         N=185,
#     ),
#     axis=1
# )

# heston_puts_via_parity = (
#     heston_calls
#     + df['Strike'] * np.exp(-df['Risk Free'] * df['To expiry'])
#     - S0 * np.exp(-df['Impl (Yld)'] * df['To expiry'])
# )
# diffs = heston_puts - heston_puts_via_parity
# print("\nHeston — Put/Call parity (seed): {}.   Max|diff|: {}".format(
#     np.max(diffs) < 1e-10, np.max(np.abs(diffs))
# ))
########################################################################


# Early-stopping hyperparameters for Differential Evolution (DE)
TOL = 1e-12
ATOL = 1e-6
PATIENCE_CB = 8 # consecutive DE iters without improvement
MIN_REL_IMPROV = 1e-5
MIN_ABS_IMPROV = 1e-6
MAX_SECONDS_CB = None # optional time 
max_iter = 100

########################################################################
############################ CALIBRATION ###############################
x0 = x0_init
print(f"\nInitial params: [v0, kappa, theta, sigma, rho, lambda]")
print(f"Initial params: {np.round(x0, 3)}")
print(f"\nStarting Differential Evolution (global)")
print(f"==========================================================================================")
t0 = datetime.now()

obj = lambda x: loss_function(x, pricer=pricer, S0=float(S0), df=df, cfg=cfg)
v2p = lambda x: vec_to_params(x, cfg.param_keys)

cb_de = make_de_callback(
    obj,
    v2p,
    early=DEEarlyStop(
        patience=PATIENCE_CB,
        min_rel_improv=MIN_REL_IMPROV,
        min_abs_improv=MIN_ABS_IMPROV,
        max_seconds=MAX_SECONDS_CB,
    ),
    tag="DE",
)


loss_init = float(obj(x0))
result_de = differential_evolution(
    obj,
    bounds=bounds,
    strategy="best1bin",
    popsize=10,
    maxiter=100,
    tol=TOL,
    atol=ATOL,
    mutation=(0.3, 0.8),
    recombination=0.9,
    polish=False,
    seed=7,
    updating="immediate",
    workers=1,
    callback=cb_de,
)

print("\nGlobal method completed.")
x0 = result_de.x
loss_de = float(result_de.fun)

print("\nStarting L-BFGS-B (local refinement)")
print("==========================================================================================")

cb_lbfgs, loss_history = make_lbfgs_callback(obj, v2p, tag="LBFGS")

result = minimize(
    obj,                     # <-- usa el mismo wrapper
    x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": max_iter, "ftol": 1e-12},
    callback=cb_lbfgs,
)

loss_opt = float(result.fun)
optimal_vec = result.x
optimal_params = vec_to_params(optimal_vec, cfg.param_keys)

print_calibration_summary(
    title="Heston calibration summary",
    method=f"{pricer.method} | DE -> L-BFGS-B",
    keys=cfg.param_keys,
    x_init=np.asarray(x0_init, dtype=float),   # guarda x0_init antes de DE
    x_de=np.asarray(result_de.x, dtype=float),
    x_opt=np.asarray(optimal_vec, dtype=float),
    params_opt=optimal_params,
    loss_init=loss_init,
    loss_de=loss_de,
    loss_opt=loss_opt,
    t0=t0,
    notes=None,
)
########################################################################


########################################################################
######################### CALIBRATION ANALYSIS #########################
# df['Heston price'] = df.apply(
#     lambda row: hp.vanilla_price(
#         T=row['To expiry'],
#         K=row['Strike'],
#         option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
#         heston_params=optimal_params,
#         option_type=row['Option type'],
#         N=128
#     ),
#     axis=1
# )

# df['Heston IV'] = df.apply(
#     lambda row: bs.iv_solver(
#         mkt_price=row['Heston price'],
#         T=row['To expiry'],
#         K=row['Strike'],
#         option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
#         option_type=row['Option type']
#     ),
#     axis=1
# )

# # Bid/Ask sheets (if available)
# _, _, df_bid = loader.load_iv_table('Bid')
# df_bid = df_bid[df_bid["IV"] > 0]
# print(df_bid)

# _, _, df_ask = loader.load_iv_table('Ask')
# df_ask = df_ask[df_ask["IV"] > 0]

# # Align moneyness & expiries with main set
# df_bid = df_bid[df_bid['Moneyness'].isin(df['Moneyness'])]
# df_ask = df_ask[df_ask['Moneyness'].isin(df['Moneyness'])]
# df_bid = df_bid[df_bid['Expiry'].isin(df['Expiry'])]
# df_ask = df_ask[df_ask['Expiry'].isin(df['Expiry'])]

# # Recompute time-to-expiry on bid/ask
# df_bid['Exp Date'] = pd.to_datetime(df_bid['Exp Date'], dayfirst=True, errors='coerce')
# maturities = df_bid['Exp Date'].unique()
# df_bid['To expiry'] = (df_bid['Exp Date'] - pd.to_datetime(act_date)).dt.days / 252

# df_ask['Exp Date'] = pd.to_datetime(df_ask['Exp Date'], dayfirst=True, errors='coerce')
# maturities = df_ask['Exp Date'].unique()
# df_ask['To expiry'] = (df_ask['Exp Date'] - pd.to_datetime(act_date)).dt.days / 252

# # Clean reindex
# df = df.reset_index(drop=True)

# # Plot smiles per maturity
# import matplotlib.pyplot as plt
# mn_grid = np.linspace(0.8, 1.2, 300)
# for Tj in sorted(df['To expiry'].unique()):
#     # Market subsets
#     aux = df[df['To expiry'] == Tj]
#     aux_bid = df_bid[df_bid['To expiry'] == Tj]
#     aux_ask = df_ask[df_ask['To expiry'] == Tj]

#     fwdj = aux['ImplFwd'].iloc[0]
#     K_grid = mn_grid * fwdj
#     rj = aux['Risk Free'].iloc[0]
#     qj = aux['Impl (Yld)'].iloc[0]
#     opt_type_grid = np.where(mn_grid >= 1, "call", "put")

#     # Model prices on strike grid
#     prices_heston = [
#         hp.vanilla_price(
#             T=Tj, K=Kk,
#             option_params=(S0, rj, qj),
#             heston_params=optimal_params,
#             option_type=otype
#         )
#         for Kk, otype in zip(K_grid, opt_type_grid)
#     ]

#     # Convert model prices to IV on the same grid
#     ivs_heston = []
#     for Kk, Pk, otype in zip(K_grid, prices_heston, opt_type_grid):
#         try:
#             ivs_heston.append(
#                 bs.iv_solver(
#                     mkt_price=Pk, T=Tj, K=Kk,
#                     option_params=(S0, rj, qj),
#                     option_type=otype
#                 )
#             )
#         except Exception:
#             ivs_heston.append(np.nan)
#     ivs_heston = np.array(ivs_heston)

#     # Market data aligned in forward moneyness
#     mn_mkt = aux['Strike'].to_numpy() / fwdj
#     iv_mkt = aux['IV'].to_numpy()
#     iv_mkt_bid = aux_bid['IV'].to_numpy()
#     iv_mkt_ask = aux_ask['IV'].to_numpy()

#     # Plot smile (IV)
#     plt.figure()
#     plt.title(f"IV Smile — {aux['Expiry'].iloc[0]}")
#     plt.plot(100*mn_mkt, 100*iv_mkt_bid, 'o', color='blue', label='Market bid IV')
#     plt.plot(100*mn_mkt, 100*iv_mkt_ask, 'o', color='red', label='Market ask IV')
#     plt.plot(100*mn_grid, 100*ivs_heston, color='green', label='Heston IV')
#     plt.xlabel("Moneyness K/F (%)")
#     plt.ylabel("IV (%)")
#     plt.legend()
#     plt.tight_layout()

# plt.show()