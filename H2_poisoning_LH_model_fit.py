# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =========================================================
# SETTINGS
# =========================================================
ROOT = r"C:/Users/HH/LH Fitting/Only NH3 and H2 cooling"
CATALYST_FOLDERS = ["La", "Ce", "CoNi"]
FILE_PATTERNS = ["*.dat", "*.txt", "*.asc", "*.csv"]

OUT_FOLDER_NAME = "Fit_results_alternating_globalfit_then_mn"

# =========================================================
# MANUAL FEED ASSIGNMENT
# =========================================================
MANUAL_FEED_RULES = {
    # La
    "La 450.dat":  {"all": 0},
    "La 460.dat":  {"all": 0},
    "La 470.dat":  {"all": 0},

    # Ce
    "Ce 450.dat":  {"all": 0},
    "Ce 460.dat":  {"all": 0},
    "Ce 470.dat":  {"all": 0},
    "Ce 480.dat":  {"all": 0},

    # CoNi
    "CoNi 457.dat": {"all": 0},
    "CoNi 468.dat": {"all": 0},
    "CoNi 479.dat": {"all": 0},
    "CoNi 488.dat": {"all": 0},
    "CoNi 498.dat": {"all": 0},
}
DEFAULT_FEED_FLAG_IF_MISSING = 0

# =========================================================
# INITIAL VALUES FOR ALTERNATING FIT
# =========================================================
M_INIT = 0.10
N_INIT = 4.00

# Grid for m, n update step
M_GRID = np.round(np.arange(0.00, 0.201, 0.01), 2)
N_GRID = np.round(np.arange(0.50, 4.001, 0.25), 2)

# optional local fine grid around best coarse point
USE_FINE_GRID = True
M_FINE_HALF_WIDTH = 0.03
N_FINE_HALF_WIDTH = 0.50
M_FINE_STEP = 0.005
N_FINE_STEP = 0.10

# alternating iterations
MAX_ALT_ITER = 6
REL_SCORE_TOL = 1e-4

# =========================================================
# OPTIMIZER SETTINGS
# =========================================================
LOSS_GLOBAL = "soft_l1"
F_SCALE_GLOBAL = 1.0
MAX_NFEV_GLOBAL = 100000
N_STARTS_GLOBAL = 30
RNG_SEED = 123

USE_LOG_RESIDUAL = True

# =========================================================
# SOFT PRIORS / PENALTIES
# =========================================================
# Soft prior for Ea
EA_SOFT_LOW = 70e3
EA_SOFT_HIGH = 170e3
EA_RANGE_SIGMA = 15e3
EA_RANGE_MULTIPLIER = 6

# Mild mean-pooling for Ea among catalysts
EA_POOL_SIGMA = 35e3
EA_POOL_MULTIPLIER = 2

# Soft prior for dH to avoid runaway values
DH_SOFT_LOW = -300e3
DH_SOFT_HIGH = 50e3
DH_RANGE_SIGMA = 30e3
DH_RANGE_MULTIPLIER = 4

# =========================================================
# CONSTANTS / BOUNDS
# =========================================================
R = 8.31446261815324
EPS = 1e-16

# param block per catalyst = [lnA, Ea, lnKH0, dH]
LB_BLOCK = np.array([
    -100.0,   # lnA
    1e3,      # Ea (J/mol)
    -80.0,    # lnKH0
    -400e3    # dH (J/mol)
], dtype=float)

UB_BLOCK = np.array([
    100.0,    # lnA
    400e3,    # Ea (J/mol)
    80.0,     # lnKH0
    200e3     # dH (J/mol)
], dtype=float)

PARITY_DPI = 300
ERROR_BAND_FRAC = 0.20

CAT_ORDER = ["La", "Ce", "CoNi"]
CAT_TO_IDX = {c: i for i, c in enumerate(CAT_ORDER)}

# =========================================================
# HELPERS
# =========================================================
def parse_temp_c(filename: str):
    name = os.path.basename(filename)
    name = name.replace("º", "°").replace("Â°", "°")

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*)?C", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m2 = re.search(r"(\d+(?:\.\d+)?)", name)
    if m2:
        return float(m2.group(1))

    return None


def read_ascii_3cols_with_rowid(path: str):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    parsed_idx = -1
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue

        parts = re.split(r"[,\s;]+", s)
        if len(parts) < 3:
            continue

        try:
            pnh3 = float(parts[0])
            ph2 = float(parts[1])
            rate = float(parts[2])
            parsed_idx += 1
        except ValueError:
            continue

        rows.append([parsed_idx, pnh3, ph2, rate])

    if not rows:
        raise ValueError(f"Cannot read numeric data from file: {path}")

    data = np.array(rows, dtype=float)
    row_id = data[:, 0].astype(int)
    PNH3 = data[:, 1]
    PH2 = data[:, 2]
    r = data[:, 3]

    mask = (
        np.isfinite(PNH3) &
        np.isfinite(PH2) &
        np.isfinite(r) &
        (PNH3 > 0) &
        (PH2 >= 0) &
        (r > 0)
    )

    row_id = row_id[mask]
    PNH3 = PNH3[mask]
    PH2 = PH2[mask]
    r = r[mask]

    if len(r) == 0:
        raise ValueError(f"No valid points left after filtering: {path}")

    return row_id, PNH3, PH2, r


def list_data_files(folder):
    files = []
    for pat in FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))


def assign_feed_flag_manual(file_basename, row_id, n_points):
    rule = MANUAL_FEED_RULES.get(file_basename, None)

    if rule is None:
        return np.full(n_points, DEFAULT_FEED_FLAG_IF_MISSING, dtype=int)

    if "all" in rule:
        return np.full(n_points, int(rule["all"]), dtype=int)

    flag = np.full(n_points, DEFAULT_FEED_FLAG_IF_MISSING, dtype=int)

    if "rows_feed" in rule:
        flag[:] = 0
        rows_feed = set(int(x) for x in rule["rows_feed"])
        for i, rid in enumerate(row_id):
            if int(rid) in rows_feed:
                flag[i] = 1
        return flag

    if "rows_nofeed" in rule:
        flag[:] = 1
        rows_nofeed = set(int(x) for x in rule["rows_nofeed"])
        for i, rid in enumerate(row_id):
            if int(rid) in rows_nofeed:
                flag[i] = 0
        return flag

    return np.full(n_points, DEFAULT_FEED_FLAG_IF_MISSING, dtype=int)


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ape = np.abs(err) / np.maximum(np.abs(y_true), EPS) * 100.0

    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": float(np.mean(ape)),
        "MedianAPE_%": float(np.median(ape)),
        "MaxAPE_%": float(np.max(ape)),
        "R2": r2
    }


def parameter_se_from_result(res):
    try:
        f = res.fun
        J = res.jac
        n = f.size
        p = res.x.size
        dof = max(n - p, 1)
        s2 = float(np.sum(f**2)) / dof
        cov = s2 * np.linalg.pinv(J.T @ J)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return se
    except Exception:
        return np.full_like(res.x, np.nan, dtype=float)


def block_slice(cat_idx):
    i0 = 4 * cat_idx
    i1 = i0 + 4
    return slice(i0, i1)


def unpack_block(params, cat_idx):
    sl = block_slice(cat_idx)
    lnA, Ea, lnKH0, dH = params[sl]
    return lnA, Ea, lnKH0, dH


def build_bounds_joint():
    lb = np.tile(LB_BLOCK, len(CAT_ORDER))
    ub = np.tile(UB_BLOCK, len(CAT_ORDER))
    return lb, ub


def k_of_T(lnA, Ea, T_K):
    return np.exp(np.clip(lnA - Ea / (R * T_K), -700, 700))


def KH_of_T(lnKH0, dH, T_K):
    return np.exp(np.clip(lnKH0 - dH / (R * T_K), -700, 700))


def initial_guess_joint(df):
    parts = []
    for cat in CAT_ORDER:
        sub = df[df["catalyst"] == cat].copy()
        if sub.empty:
            raise ValueError(f"No data found for catalyst {cat}")

        T_K = sub["T_K"].to_numpy(dtype=float)
        PH2 = sub["PH2"].to_numpy(dtype=float)
        r = sub["r_exp"].to_numpy(dtype=float)

        tmid = np.median(T_K)
        lnA_guess = np.log(max(np.median(r), 1e-20)) + 100e3 / (R * tmid)
        Ea_guess = 100e3

        ph2_pos = PH2[PH2 > 0]
        if len(ph2_pos) > 0:
            KH_guess = 1.0 / max(np.median(ph2_pos), 1e-12)
        else:
            KH_guess = 1.0

        lnKH0_guess = np.log(np.clip(KH_guess, 1e-30, 1e30))
        dH_guess = -90e3

        block = np.array([lnA_guess, Ea_guess, lnKH0_guess, dH_guess], dtype=float)
        block = np.clip(block, LB_BLOCK, UB_BLOCK)
        parts.append(block)

    return np.concatenate(parts)


def build_fine_grid(center, half_width, step, lower, upper, decimals=3):
    a = max(lower, center - half_width)
    b = min(upper, center + half_width)
    arr = np.arange(a, b + 0.5 * step, step)
    return np.unique(np.round(arr, decimals))


# =========================================================
# DATA LOADING
# =========================================================
def load_all_data():
    all_rows = []

    for cat in CATALYST_FOLDERS:
        folder = os.path.join(ROOT, cat)
        files = list_data_files(folder)
        if not files:
            print(f"No files found in {folder}")
            continue

        for fp in files:
            fname = os.path.basename(fp)
            tC = parse_temp_c(fname)
            if tC is None:
                raise ValueError(f"Cannot parse T from filename: {fname}")

            row_id, PNH3, PH2, r = read_ascii_3cols_with_rowid(fp)
            feed_flag = assign_feed_flag_manual(fname, row_id, len(r))
            T_K = np.full(len(r), tC + 273.15, dtype=float)

            for i in range(len(r)):
                all_rows.append({
                    "catalyst": cat,
                    "cat_idx": CAT_TO_IDX[cat],
                    "file": fname,
                    "row_id": int(row_id[i]),
                    "T_C": tC,
                    "T_K": T_K[i],
                    "PNH3": PNH3[i],
                    "PH2": PH2[i],
                    "feed_flag": int(feed_flag[i]),
                    "r_exp": r[i]
                })

    if not all_rows:
        raise ValueError("No valid data found in any catalyst folder.")

    return pd.DataFrame(all_rows).sort_values(
        ["catalyst", "T_C", "file", "row_id"]
    ).reset_index(drop=True)


# =========================================================
# MODEL (NO LAMBDA)
# =========================================================
def model_joint(params, cat_idx_arr, T_K, PNH3, PH2, m_fixed, n_fixed):
    r_pred = np.empty_like(T_K, dtype=float)

    for ci in range(len(CAT_ORDER)):
        mask = (cat_idx_arr == ci)
        if not np.any(mask):
            continue

        lnA, Ea, lnKH0, dH = unpack_block(params, ci)

        ln_k = lnA - Ea / (R * T_K[mask])
        ln_KH = lnKH0 - dH / (R * T_K[mask])

        k = np.exp(np.clip(ln_k, -700, 700))
        KH = np.exp(np.clip(ln_KH, -700, 700))

        inhib = np.sqrt(np.maximum(KH * PH2[mask], 0.0))
        denom = (1.0 + inhib) ** n_fixed

        r_pred[mask] = k * np.power(np.maximum(PNH3[mask], EPS), m_fixed) / denom

    return np.maximum(r_pred, EPS)


def bound_penalty_residual(value, low, high, sigma, multiplier):
    low_violation = max(0.0, low - value)
    high_violation = max(0.0, value - high)

    res_low = low_violation / sigma
    res_high = high_violation / sigma

    pen = []
    for _ in range(multiplier):
        pen.append(res_low)
        pen.append(res_high)
    return np.array(pen, dtype=float)


def ea_mean_pool_penalty_residual(Ea_La, Ea_Ce, Ea_CoNi):
    Ea_mean = (Ea_La + Ea_Ce + Ea_CoNi) / 3.0
    pen = np.array([
        (Ea_La - Ea_mean) / EA_POOL_SIGMA,
        (Ea_Ce - Ea_mean) / EA_POOL_SIGMA,
        (Ea_CoNi - Ea_mean) / EA_POOL_SIGMA
    ], dtype=float)

    if EA_POOL_MULTIPLIER <= 1:
        return pen

    out = []
    for _ in range(EA_POOL_MULTIPLIER):
        out.extend(pen.tolist())
    return np.array(out, dtype=float)


def residual_joint(params, cat_idx_arr, T_K, PNH3, PH2, r_obs, m_fixed, n_fixed):
    r_pred = model_joint(params, cat_idx_arr, T_K, PNH3, PH2, m_fixed, n_fixed)

    if USE_LOG_RESIDUAL:
        data_res = np.log(r_pred + EPS) - np.log(r_obs + EPS)
    else:
        floor = 0.05 * np.median(np.abs(r_obs)) + EPS
        data_res = (r_pred - r_obs) / (np.abs(r_obs) + floor)

    Ea_La = params[block_slice(CAT_TO_IDX["La"])][1]
    Ea_Ce = params[block_slice(CAT_TO_IDX["Ce"])][1]
    Ea_CoNi = params[block_slice(CAT_TO_IDX["CoNi"])][1]

    dH_La = params[block_slice(CAT_TO_IDX["La"])][3]
    dH_Ce = params[block_slice(CAT_TO_IDX["Ce"])][3]
    dH_CoNi = params[block_slice(CAT_TO_IDX["CoNi"])][3]

    pen_ea = np.concatenate([
        bound_penalty_residual(Ea_La, EA_SOFT_LOW, EA_SOFT_HIGH, EA_RANGE_SIGMA, EA_RANGE_MULTIPLIER),
        bound_penalty_residual(Ea_Ce, EA_SOFT_LOW, EA_SOFT_HIGH, EA_RANGE_SIGMA, EA_RANGE_MULTIPLIER),
        bound_penalty_residual(Ea_CoNi, EA_SOFT_LOW, EA_SOFT_HIGH, EA_RANGE_SIGMA, EA_RANGE_MULTIPLIER)
    ])

    pen_dh = np.concatenate([
        bound_penalty_residual(dH_La, DH_SOFT_LOW, DH_SOFT_HIGH, DH_RANGE_SIGMA, DH_RANGE_MULTIPLIER),
        bound_penalty_residual(dH_Ce, DH_SOFT_LOW, DH_SOFT_HIGH, DH_RANGE_SIGMA, DH_RANGE_MULTIPLIER),
        bound_penalty_residual(dH_CoNi, DH_SOFT_LOW, DH_SOFT_HIGH, DH_RANGE_SIGMA, DH_RANGE_MULTIPLIER)
    ])

    pen_pool = ea_mean_pool_penalty_residual(Ea_La, Ea_Ce, Ea_CoNi)

    return np.concatenate([data_res, pen_ea, pen_dh, pen_pool])


# =========================================================
# FIT CORE
# =========================================================
def fit_joint_global(df, m_fixed, n_fixed, x0_base=None):
    lb, ub = build_bounds_joint()

    cat_idx_arr = df["cat_idx"].to_numpy(dtype=int)
    T_K = df["T_K"].to_numpy(dtype=float)
    PNH3 = df["PNH3"].to_numpy(dtype=float)
    PH2 = df["PH2"].to_numpy(dtype=float)
    r_obs = df["r_exp"].to_numpy(dtype=float)

    base = initial_guess_joint(df) if x0_base is None else np.clip(np.array(x0_base, dtype=float), lb, ub)
    rng = np.random.default_rng(RNG_SEED)

    best_res = None
    best_score = np.inf

    for i in range(N_STARTS_GLOBAL):
        x0 = base.copy()

        if i != 0:
            for ci in range(len(CAT_ORDER)):
                sl = block_slice(ci)
                x0[sl][0] += rng.uniform(-2.5, 2.5)
                x0[sl][1] += rng.uniform(-20e3, 20e3)
                x0[sl][2] += rng.uniform(-2.5, 2.5)
                x0[sl][3] += rng.uniform(-25e3, 25e3)
            x0 = np.clip(x0, lb, ub)

        res = least_squares(
            residual_joint,
            x0=x0,
            bounds=(lb, ub),
            args=(cat_idx_arr, T_K, PNH3, PH2, r_obs, m_fixed, n_fixed),
            method="trf",
            loss=LOSS_GLOBAL,
            f_scale=F_SCALE_GLOBAL,
            x_scale="jac",
            max_nfev=MAX_NFEV_GLOBAL
        )

        score = float(np.sum(res.fun**2))
        if score < best_score:
            best_score = score
            best_res = res

    se = parameter_se_from_result(best_res)
    return best_res, se, best_score


def evaluate_mn_given_params(df, params, m_val, n_val):
    r_calc = model_joint(
        params,
        df["cat_idx"].to_numpy(dtype=int),
        df["T_K"].to_numpy(dtype=float),
        df["PNH3"].to_numpy(dtype=float),
        df["PH2"].to_numpy(dtype=float),
        m_fixed=m_val,
        n_fixed=n_val
    )

    if USE_LOG_RESIDUAL:
        data_res = np.log(r_calc + EPS) - np.log(df["r_exp"].to_numpy(dtype=float) + EPS)
        score = float(np.sum(data_res ** 2))
    else:
        err = r_calc - df["r_exp"].to_numpy(dtype=float)
        score = float(np.sum(err ** 2))

    metrics = calc_metrics(df["r_exp"].to_numpy(dtype=float), r_calc)
    return score, metrics, r_calc


def grid_search_mn_given_params(df, params, m_grid, n_grid, verbose=False):
    rows = []
    best = None

    for m_val in m_grid:
        for n_val in n_grid:
            score, metrics, _ = evaluate_mn_given_params(df, params, m_val, n_val)
            rows.append({
                "m": m_val,
                "n": n_val,
                "score": score,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE_%": metrics["MAPE_%"],
                "R2": metrics["R2"]
            })

            if (best is None) or (score < best["score"]):
                best = {
                    "m": m_val,
                    "n": n_val,
                    "score": score,
                    "metrics": metrics
                }

            if verbose:
                print(f"Grid m={m_val:.4f}, n={n_val:.4f}, score={score:.6g}")

    return best, pd.DataFrame(rows).sort_values(["score", "m", "n"]).reset_index(drop=True)


# =========================================================
# PLOTS
# =========================================================
def parity_plot_loglog(title, r_exp, r_pred, out_png, band_frac=0.20):
    mask = (
        np.isfinite(r_exp) & np.isfinite(r_pred) &
        (r_exp > 0) & (r_pred > 0)
    )

    x = np.asarray(r_exp)[mask]
    y = np.asarray(r_pred)[mask]

    mn = max(min(np.min(x), np.min(y)), EPS)
    mx = max(np.max(x), np.max(y))
    xx = np.logspace(np.log10(mn), np.log10(mx), 200)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.plot(xx, xx, label="y=x")
    plt.plot(xx, (1.0 + band_frac) * xx, "--", label=f"+{int(band_frac*100)}%")
    plt.plot(xx, (1.0 - band_frac) * xx, "--", label=f"-{int(band_frac*100)}%")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r_exp")
    plt.ylabel("r_calc")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=PARITY_DPI)
    plt.close()


def residual_plot(x, y, xlabel, title, out_png):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    plt.axhline(0, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("relative error (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def arrhenius_plot(invT, lnval, a, b, title, ylabel, out_png):
    xx = np.linspace(np.min(invT), np.max(invT), 200)
    yy = a + b * xx

    plt.figure(figsize=(6, 4.5))
    plt.scatter(invT, lnval)
    plt.plot(xx, yy)
    plt.xlabel("1/T (1/K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def grid_heatmap(grid_df, out_png, value_col="score"):
    pivot = grid_df.pivot(index="n", columns="m", values=value_col)
    xvals = pivot.columns.to_numpy(dtype=float)
    yvals = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(8, 5.5))
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()]
    )
    plt.colorbar(im, label=value_col)
    plt.xlabel("m")
    plt.ylabel("n")
    plt.title(f"Grid search map: {value_col}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def alt_history_plot(hist_df, out_png, value_col="score"):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(hist_df["iteration"], hist_df[value_col], marker="o")
    plt.xlabel("iteration")
    plt.ylabel(value_col)
    plt.title(f"Alternating fit history: {value_col}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# =========================================================
# SAVE FINAL OUTPUTS
# =========================================================
def save_final_outputs(df, params, se, score, out_root, m_final, n_final):
    r_calc = model_joint(
        params,
        df["cat_idx"].to_numpy(dtype=int),
        df["T_K"].to_numpy(dtype=float),
        df["PNH3"].to_numpy(dtype=float),
        df["PH2"].to_numpy(dtype=float),
        m_fixed=m_final,
        n_fixed=n_final
    )

    df_out = df.copy()
    df_out["r_calc"] = r_calc
    df_out["rel_error_%"] = (df_out["r_calc"] - df_out["r_exp"]) / np.maximum(df_out["r_exp"], EPS) * 100.0
    df_out.to_csv(os.path.join(out_root, "ALL_CATALYSTS_final_predictions.csv"), index=False)

    param_rows = []
    summary_rows = []
    temp_rows_all = []

    for cat in CAT_ORDER:
        ci = CAT_TO_IDX[cat]
        sl = block_slice(ci)
        lnA, Ea, lnKH0, dH = params[sl]
        SE_lnA, SE_Ea, SE_lnKH0, SE_dH = se[sl]

        sub = df_out[df_out["catalyst"] == cat].copy()
        metrics = calc_metrics(sub["r_exp"].to_numpy(), sub["r_calc"].to_numpy())

        param_rows.append({
            "catalyst": cat,
            "m_fixed_final": m_final,
            "n_fixed_final": n_final,
            "lnA": lnA,
            "SE_lnA": SE_lnA,
            "Ea_J_per_mol": Ea,
            "SE_Ea_J_per_mol": SE_Ea,
            "Ea_kJ_per_mol": Ea / 1000.0,
            "SE_Ea_kJ_per_mol": SE_Ea / 1000.0,
            "lnKH0": lnKH0,
            "SE_lnKH0": SE_lnKH0,
            "dH_J_per_mol": dH,
            "SE_dH_J_per_mol": SE_dH,
            "dH_kJ_per_mol": dH / 1000.0,
            "SE_dH_kJ_per_mol": SE_dH / 1000.0,
            "objective_score_final": score,
            **metrics
        })

        summary_rows.append({
            "catalyst": cat,
            "n_temperatures": sub["T_K"].nunique(),
            "total_points": len(sub),
            "m_final": m_final,
            "n_final": n_final,
            "Ea_kJ_per_mol": Ea / 1000.0,
            "SE_Ea_kJ_per_mol": SE_Ea / 1000.0,
            "dH_kJ_per_mol": dH / 1000.0,
            "SE_dH_kJ_per_mol": SE_dH / 1000.0,
            "overall_RMSE": metrics["RMSE"],
            "overall_MAE": metrics["MAE"],
            "overall_MAPE_%": metrics["MAPE_%"],
            "overall_R2": metrics["R2"]
        })

        temp_unique = (
            sub[["T_C", "T_K"]]
            .drop_duplicates()
            .sort_values("T_C")
            .reset_index(drop=True)
        )

        temp_fit_rows = []
        for _, row in temp_unique.iterrows():
            T_C = float(row["T_C"])
            T_K = float(row["T_K"])

            mask = np.isclose(sub["T_K"].to_numpy(), T_K)
            subT = sub.loc[mask].copy()

            kT = k_of_T(lnA, Ea, T_K)
            KHT = KH_of_T(lnKH0, dH, T_K)
            sub_metrics = calc_metrics(subT["r_exp"].to_numpy(), subT["r_calc"].to_numpy())

            temp_fit_rows.append({
                "catalyst": cat,
                "T_C": T_C,
                "T_K": T_K,
                "n_points": len(subT),
                "n_feed_points": int(np.sum(subT["feed_flag"])),
                "ln_k_global": np.log(np.maximum(kT, EPS)),
                "k(T)_global": kT,
                "ln_KH_global": np.log(np.maximum(KHT, EPS)),
                "KH(T)_global": KHT,
                **sub_metrics
            })

        temp_df = pd.DataFrame(temp_fit_rows).sort_values("T_C").reset_index(drop=True)
        temp_df["m_final"] = m_final
        temp_df["n_final"] = n_final
        temp_df["lnA_global"] = lnA
        temp_df["Ea_kJ_per_mol_global"] = Ea / 1000.0
        temp_df["lnKH0_global"] = lnKH0
        temp_df["dH_kJ_per_mol_global"] = dH / 1000.0
        temp_rows_all.append(temp_df)

        cat_out = os.path.join(out_root, cat)
        os.makedirs(cat_out, exist_ok=True)

        sub.to_csv(os.path.join(cat_out, f"{cat}_predictions_final.csv"), index=False)
        temp_df.to_csv(os.path.join(cat_out, f"{cat}_k_KH_by_temperature_final.csv"), index=False)

        parity_plot_loglog(
            title=f"Parity plot - {cat} (alternating final fit)",
            r_exp=sub["r_exp"].to_numpy(),
            r_pred=sub["r_calc"].to_numpy(),
            out_png=os.path.join(cat_out, f"parity_{cat}.png"),
            band_frac=ERROR_BAND_FRAC
        )

        residual_plot(
            sub["PH2"].to_numpy(),
            sub["rel_error_%"].to_numpy(),
            xlabel="PH2",
            title=f"Residual vs PH2 - {cat}",
            out_png=os.path.join(cat_out, f"{cat}_residual_vs_PH2.png")
        )

        residual_plot(
            sub["PNH3"].to_numpy(),
            sub["rel_error_%"].to_numpy(),
            xlabel="PNH3",
            title=f"Residual vs PNH3 - {cat}",
            out_png=os.path.join(cat_out, f"{cat}_residual_vs_PNH3.png")
        )

        residual_plot(
            sub["T_C"].to_numpy(),
            sub["rel_error_%"].to_numpy(),
            xlabel="T (C)",
            title=f"Residual vs T - {cat}",
            out_png=os.path.join(cat_out, f"{cat}_residual_vs_T.png")
        )

        invT = 1.0 / temp_df["T_K"].to_numpy()
        lnk = np.log(np.maximum(temp_df["k(T)_global"].to_numpy(), EPS))
        lnKH = np.log(np.maximum(temp_df["KH(T)_global"].to_numpy(), EPS))

        a_k = lnA
        b_k = -Ea / R
        a_h = lnKH0
        b_h = -dH / R

        arrhenius_plot(
            invT, lnk, a_k, b_k,
            title=f"Arrhenius plot - {cat}",
            ylabel="ln k",
            out_png=os.path.join(cat_out, f"{cat}_Arrhenius_lnk_vs_1overT.png")
        )

        arrhenius_plot(
            invT, lnKH, a_h, b_h,
            title=f"van't Hoff plot - {cat}",
            ylabel="ln KH",
            out_png=os.path.join(cat_out, f"{cat}_VantHoff_lnKH_vs_1overT.png")
        )

    pd.DataFrame(param_rows).to_csv(os.path.join(out_root, "ALL_CATALYSTS_final_parameters.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_root, "ALL_CATALYSTS_final_summary.csv"), index=False)
    pd.concat(temp_rows_all, ignore_index=True).to_csv(
        os.path.join(out_root, "ALL_CATALYSTS_k_KH_by_temperature_final.csv"),
        index=False
    )

    parity_plot_loglog(
        title="Parity plot - all catalysts (alternating final fit)",
        r_exp=df_out["r_exp"].to_numpy(),
        r_pred=df_out["r_calc"].to_numpy(),
        out_png=os.path.join(out_root, "parity_ALL_catalysts.png"),
        band_frac=ERROR_BAND_FRAC
    )

    residual_plot(
        df_out["PH2"].to_numpy(),
        df_out["rel_error_%"].to_numpy(),
        xlabel="PH2",
        title="Residual vs PH2 - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_PH2.png")
    )

    residual_plot(
        df_out["PNH3"].to_numpy(),
        df_out["rel_error_%"].to_numpy(),
        xlabel="PNH3",
        title="Residual vs PNH3 - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_PNH3.png")
    )

    residual_plot(
        df_out["T_C"].to_numpy(),
        df_out["rel_error_%"].to_numpy(),
        xlabel="T (C)",
        title="Residual vs T - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_T.png")
    )


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 100)
    print("ALTERNATING OPTIMIZATION WITHOUT LAMBDA")
    print("Step A: fix m,n and run global fit")
    print("Step B: fix global parameters and optimize m,n by grid search")
    print("Repeat until m,n stop changing or improvement is negligible")
    print("=" * 100)

    out_root = os.path.join(ROOT, OUT_FOLDER_NAME)
    os.makedirs(out_root, exist_ok=True)

    df = load_all_data()

    m_current = M_INIT
    n_current = N_INIT
    x0_current = None
    prev_score = np.inf

    history_rows = []
    best_overall = None

    for it in range(1, MAX_ALT_ITER + 1):
        print("\n" + "=" * 100)
        print(f"ALTERNATING ITERATION {it}")
        print("=" * 100)
        print(f"Starting with m = {m_current:.4f}, n = {n_current:.4f}")

        iter_out = os.path.join(out_root, f"iter_{it:02d}")
        os.makedirs(iter_out, exist_ok=True)

        # -------------------------------------------------
        # Step A: global fit at current m,n
        # -------------------------------------------------
        print("\nGlobal fit...")
        res_fit, se_fit, score_fit = fit_joint_global(
            df=df,
            m_fixed=m_current,
            n_fixed=n_current,
            x0_base=x0_current
        )

        r_calc_fit = model_joint(
            res_fit.x,
            df["cat_idx"].to_numpy(dtype=int),
            df["T_K"].to_numpy(dtype=float),
            df["PNH3"].to_numpy(dtype=float),
            df["PH2"].to_numpy(dtype=float),
            m_fixed=m_current,
            n_fixed=n_current
        )
        metrics_fit = calc_metrics(df["r_exp"].to_numpy(dtype=float), r_calc_fit)

        pd.DataFrame([{
            "iteration": it,
            "step": "global_fit",
            "m_used": m_current,
            "n_used": n_current,
            "score": score_fit,
            "RMSE": metrics_fit["RMSE"],
            "MAE": metrics_fit["MAE"],
            "MAPE_%": metrics_fit["MAPE_%"],
            "R2": metrics_fit["R2"]
        }]).to_csv(os.path.join(iter_out, "stepA_global_fit_summary.csv"), index=False)

        print(f"Global fit score = {score_fit:.6g}")
        print(f"Global fit MAPE  = {metrics_fit['MAPE_%']:.6g} %")

        # -------------------------------------------------
        # Step B: optimize m,n with params fixed
        # -------------------------------------------------
        print("\nCoarse grid search for m,n...")
        best_coarse, coarse_df = grid_search_mn_given_params(
            df=df,
            params=res_fit.x,
            m_grid=M_GRID,
            n_grid=N_GRID,
            verbose=False
        )
        coarse_df.to_csv(os.path.join(iter_out, "stepB_coarse_grid_m_n.csv"), index=False)
        grid_heatmap(coarse_df, os.path.join(iter_out, "stepB_coarse_score_heatmap.png"), value_col="score")
        grid_heatmap(coarse_df, os.path.join(iter_out, "stepB_coarse_MAPE_heatmap.png"), value_col="MAPE_%")

        m_best = float(best_coarse["m"])
        n_best = float(best_coarse["n"])
        score_best = float(best_coarse["score"])
        metrics_best = best_coarse["metrics"]

        print(f"Best coarse m = {m_best:.4f}")
        print(f"Best coarse n = {n_best:.4f}")
        print(f"Best coarse score = {score_best:.6g}")

        if USE_FINE_GRID:
            print("\nFine grid search around best coarse point...")
            m_fine = build_fine_grid(
                center=m_best,
                half_width=M_FINE_HALF_WIDTH,
                step=M_FINE_STEP,
                lower=0.0,
                upper=0.2,
                decimals=3
            )
            n_fine = build_fine_grid(
                center=n_best,
                half_width=N_FINE_HALF_WIDTH,
                step=N_FINE_STEP,
                lower=0.5,
                upper=4.0,
                decimals=3
            )

            best_fine, fine_df = grid_search_mn_given_params(
                df=df,
                params=res_fit.x,
                m_grid=m_fine,
                n_grid=n_fine,
                verbose=False
            )
            fine_df.to_csv(os.path.join(iter_out, "stepB_fine_grid_m_n.csv"), index=False)
            grid_heatmap(fine_df, os.path.join(iter_out, "stepB_fine_score_heatmap.png"), value_col="score")
            grid_heatmap(fine_df, os.path.join(iter_out, "stepB_fine_MAPE_heatmap.png"), value_col="MAPE_%")

            if best_fine["score"] < score_best:
                m_best = float(best_fine["m"])
                n_best = float(best_fine["n"])
                score_best = float(best_fine["score"])
                metrics_best = best_fine["metrics"]

                print(f"Best fine m = {m_best:.4f}")
                print(f"Best fine n = {n_best:.4f}")
                print(f"Best fine score = {score_best:.6g}")

        pd.DataFrame([{
            "iteration": it,
            "step": "mn_grid",
            "m_best": m_best,
            "n_best": n_best,
            "score": score_best,
            "RMSE": metrics_best["RMSE"],
            "MAE": metrics_best["MAE"],
            "MAPE_%": metrics_best["MAPE_%"],
            "R2": metrics_best["R2"]
        }]).to_csv(os.path.join(iter_out, "stepB_best_mn_summary.csv"), index=False)

        history_rows.append({
            "iteration": it,
            "m_before": m_current,
            "n_before": n_current,
            "score_global_fit": score_fit,
            "RMSE_global_fit": metrics_fit["RMSE"],
            "MAPE_global_fit_%": metrics_fit["MAPE_%"],
            "m_after": m_best,
            "n_after": n_best,
            "score_grid": score_best,
            "RMSE_grid": metrics_best["RMSE"],
            "MAPE_grid_%": metrics_best["MAPE_%"]
        })

        # current iteration result tracked by global fit result
        if (best_overall is None) or (score_fit < best_overall["score"]):
            best_overall = {
                "iteration": it,
                "m": m_current,
                "n": n_current,
                "score": score_fit,
                "res": res_fit,
                "se": se_fit,
                "metrics": metrics_fit
            }

        # stopping logic
        rel_improve = np.inf
        if np.isfinite(prev_score) and prev_score > 0:
            rel_improve = (prev_score - score_fit) / prev_score

        same_mn = (abs(m_best - m_current) < 1e-12) and (abs(n_best - n_current) < 1e-12)

        print("\nIteration summary:")
        print(f"  score_global_fit = {score_fit:.6g}")
        print(f"  m update: {m_current:.4f} -> {m_best:.4f}")
        print(f"  n update: {n_current:.4f} -> {n_best:.4f}")
        print(f"  relative improvement = {rel_improve:.6g}")

        # prepare next iteration
        prev_score = score_fit
        x0_current = res_fit.x.copy()
        m_current = m_best
        n_current = n_best

        # save predictions from current global fit
        df_iter = df.copy()
        df_iter["r_calc"] = r_calc_fit
        df_iter["rel_error_%"] = (df_iter["r_calc"] - df_iter["r_exp"]) / np.maximum(df_iter["r_exp"], EPS) * 100.0
        df_iter.to_csv(os.path.join(iter_out, "global_fit_predictions.csv"), index=False)

        if same_mn:
            print("Stopping: m and n do not change anymore.")
            break

        if np.isfinite(rel_improve) and (rel_improve >= 0) and (rel_improve < REL_SCORE_TOL):
            print("Stopping: score improvement is below tolerance.")
            break

    # -----------------------------------------------------
    # final refit once more with final m,n
    # -----------------------------------------------------
    print("\n" + "=" * 100)
    print("FINAL RE-FIT WITH CONVERGED m,n")
    print("=" * 100)

    res_final, se_final, score_final = fit_joint_global(
        df=df,
        m_fixed=m_current,
        n_fixed=n_current,
        x0_base=x0_current
    )

    r_calc_final = model_joint(
        res_final.x,
        df["cat_idx"].to_numpy(dtype=int),
        df["T_K"].to_numpy(dtype=float),
        df["PNH3"].to_numpy(dtype=float),
        df["PH2"].to_numpy(dtype=float),
        m_fixed=m_current,
        n_fixed=n_current
    )
    metrics_final = calc_metrics(df["r_exp"].to_numpy(dtype=float), r_calc_final)

    if (best_overall is None) or (score_final < best_overall["score"]):
        best_overall = {
            "iteration": len(history_rows) + 1,
            "m": m_current,
            "n": n_current,
            "score": score_final,
            "res": res_final,
            "se": se_final,
            "metrics": metrics_final
        }

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(out_root, "alternating_history.csv"), index=False)
    if len(hist_df) > 0:
        alt_history_plot(hist_df, os.path.join(out_root, "alternating_history_score.png"), value_col="score_global_fit")
        alt_history_plot(hist_df, os.path.join(out_root, "alternating_history_MAPE.png"), value_col="MAPE_global_fit_%")

    pd.DataFrame([{
        "m_final": m_current,
        "n_final": n_current,
        "score_final": score_final,
        "RMSE_final": metrics_final["RMSE"],
        "MAE_final": metrics_final["MAE"],
        "MAPE_final_%": metrics_final["MAPE_%"],
        "R2_final": metrics_final["R2"]
    }]).to_csv(os.path.join(out_root, "final_overall_metrics.csv"), index=False)

    save_final_outputs(
        df=df,
        params=res_final.x,
        se=se_final,
        score=score_final,
        out_root=out_root,
        m_final=m_current,
        n_final=n_current
    )

    print("\n" + "=" * 100)
    print("FINAL RESULT")
    print(f"m_final     = {m_current:.6g}")
    print(f"n_final     = {n_current:.6g}")
    print(f"score_final = {score_final:.6g}")
    print(f"RMSE_final  = {metrics_final['RMSE']:.6g}")
    print(f"MAPE_final  = {metrics_final['MAPE_%']:.6g} %")
    print("=" * 100)
    print(f"Saved results to:\n{out_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()