# -*- coding: utf-8 -*-
"""
Semi-empirical H2-poisoning LH fit for NH3 decomposition.

Rate expression:
    r = k(T) * P_NH3**n / (1 + sqrt(K_H2(T) * P_H2))**4

where
    k(T)    = exp(lnA - Ea/(R*T))
    K_H2(T) = exp(lnKH0 - dH/(R*T))

A single common NH3 exponent n is selected by grid search from 0.00 to 0.30
with a step of 0.01 using all catalysts simultaneously. For each trial n, all
catalyst-specific kinetic parameters are refitted by global nonlinear regression.
"""

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

OUT_FOLDER_NAME = "Fit_results_common_empirical_n_0_030_H2exp4"

# fixed H2 inhibition exponent in denominator
H2_INHIBITION_EXPONENT = 4.0

# grid for NH3 empirical exponent n
N_NH3_GRID = np.round(np.arange(0.00, 0.301, 0.01), 2)

# If True: one common n for all catalysts.
# If False: each catalyst is fitted independently and can have its own n.
USE_COMMON_N_FOR_ALL_CATALYSTS = True

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
# OPTIMIZER SETTINGS
# =========================================================
LOSS_GLOBAL = "soft_l1"
F_SCALE_GLOBAL = 1.0
MAX_NFEV_GLOBAL = 100000
N_STARTS_GLOBAL = 30
RNG_SEED = 123
USE_LOG_RESIDUAL = True

# Soft priors / penalties
EA_SOFT_LOW = 70e3
EA_SOFT_HIGH = 170e3
EA_RANGE_SIGMA = 15e3
EA_RANGE_MULTIPLIER = 6

EA_POOL_SIGMA = 35e3
EA_POOL_MULTIPLIER = 2

DH_SOFT_LOW = -300e3
DH_SOFT_HIGH = 50e3
DH_RANGE_SIGMA = 30e3
DH_RANGE_MULTIPLIER = 4

# =========================================================
# CONSTANTS / BOUNDS
# =========================================================
R = 8.31446261815324
EPS = 1e-16

# parameter block per catalyst = [lnA, Ea, lnKH0, dH]
LB_BLOCK = np.array([-100.0, 1e3, -80.0, -400e3], dtype=float)
UB_BLOCK = np.array([100.0, 400e3, 80.0, 200e3], dtype=float)

PARITY_DPI = 300
ERROR_BAND_FRAC = 0.20

CAT_ORDER = ["La", "Ce", "CoNi"]
CAT_TO_IDX = {c: i for i, c in enumerate(CAT_ORDER)}

# =========================================================
# HELPERS
# =========================================================
def parse_temp_c(filename: str):
    name = os.path.basename(filename).replace("º", "°").replace("Â°", "°")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*)?C", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m2.group(1)) if m2 else None


def read_ascii_3cols_with_rowid(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    parsed_idx = -1
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
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

    mask = np.isfinite(PNH3) & np.isfinite(PH2) & np.isfinite(r) & (PNH3 > 0) & (PH2 >= 0) & (r > 0)
    row_id, PNH3, PH2, r = row_id[mask], PNH3[mask], PH2[mask], r[mask]
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
    elif "rows_nofeed" in rule:
        flag[:] = 1
        rows_nofeed = set(int(x) for x in rule["rows_nofeed"])
        for i, rid in enumerate(row_id):
            if int(rid) in rows_nofeed:
                flag[i] = 0
    return flag


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
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": float(np.mean(ape)),
            "MedianAPE_%": float(np.median(ape)), "MaxAPE_%": float(np.max(ape)), "R2": r2}


def parameter_se_from_result(res):
    try:
        f, J = res.fun, res.jac
        dof = max(f.size - res.x.size, 1)
        s2 = float(np.sum(f**2)) / dof
        cov = s2 * np.linalg.pinv(J.T @ J)
        return np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        return np.full_like(res.x, np.nan, dtype=float)


def block_slice(cat_idx):
    return slice(4 * cat_idx, 4 * cat_idx + 4)


def unpack_block(params, cat_idx):
    return params[block_slice(cat_idx)]


def build_bounds_joint(n_cats):
    return np.tile(LB_BLOCK, n_cats), np.tile(UB_BLOCK, n_cats)


def k_of_T(lnA, Ea, T_K):
    return np.exp(np.clip(lnA - Ea / (R * T_K), -700, 700))


def KH_of_T(lnKH0, dH, T_K):
    return np.exp(np.clip(lnKH0 - dH / (R * T_K), -700, 700))


def initial_guess_joint(df, cat_order):
    parts = []
    for cat in cat_order:
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
        KH_guess = 1.0 / max(np.median(ph2_pos), 1e-12) if len(ph2_pos) else 1.0
        lnKH0_guess = np.log(np.clip(KH_guess, 1e-30, 1e30))
        dH_guess = -90e3
        parts.append(np.clip(np.array([lnA_guess, Ea_guess, lnKH0_guess, dH_guess]), LB_BLOCK, UB_BLOCK))
    return np.concatenate(parts)

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
            for i in range(len(r)):
                all_rows.append({"catalyst": cat, "file": fname, "row_id": int(row_id[i]),
                                 "T_C": float(tC), "T_K": float(tC + 273.15),
                                 "PNH3": float(PNH3[i]), "PH2": float(PH2[i]),
                                 "feed_flag": int(feed_flag[i]), "r_exp": float(r[i])})
    if not all_rows:
        raise ValueError("No valid data found in any catalyst folder.")
    df = pd.DataFrame(all_rows).sort_values(["catalyst", "T_C", "file", "row_id"]).reset_index(drop=True)
    return df

# =========================================================
# MODEL
# =========================================================
def model_joint(params, cat_idx_arr, T_K, PNH3, PH2, n_nh3, h2_exp=H2_INHIBITION_EXPONENT):
    r_pred = np.empty_like(T_K, dtype=float)
    n_cats = int(np.max(cat_idx_arr)) + 1
    for ci in range(n_cats):
        mask = (cat_idx_arr == ci)
        if not np.any(mask):
            continue
        lnA, Ea, lnKH0, dH = unpack_block(params, ci)
        k = k_of_T(lnA, Ea, T_K[mask])
        KH = KH_of_T(lnKH0, dH, T_K[mask])
        inhib = np.sqrt(np.maximum(KH * PH2[mask], 0.0))
        denom = (1.0 + inhib) ** h2_exp
        r_pred[mask] = k * np.power(np.maximum(PNH3[mask], EPS), n_nh3) / denom
    return np.maximum(r_pred, EPS)


def theta_H_from_KH_PH2(KH, PH2):
    s = np.sqrt(np.maximum(KH * PH2, 0.0))
    return s / (1.0 + s)


def bound_penalty_residual(value, low, high, sigma, multiplier):
    low_violation = max(0.0, low - value)
    high_violation = max(0.0, value - high)
    res = []
    for _ in range(multiplier):
        res.extend([low_violation / sigma, high_violation / sigma])
    return np.array(res, dtype=float)


def ea_mean_pool_penalty_residual(params, n_cats):
    if n_cats < 2 or EA_POOL_MULTIPLIER <= 0:
        return np.array([], dtype=float)
    Eas = np.array([params[block_slice(ci)][1] for ci in range(n_cats)], dtype=float)
    pen = (Eas - np.mean(Eas)) / EA_POOL_SIGMA
    return np.tile(pen, EA_POOL_MULTIPLIER)


def residual_joint(params, cat_idx_arr, T_K, PNH3, PH2, r_obs, n_nh3):
    r_pred = model_joint(params, cat_idx_arr, T_K, PNH3, PH2, n_nh3)
    if USE_LOG_RESIDUAL:
        data_res = np.log(r_pred + EPS) - np.log(r_obs + EPS)
    else:
        floor = 0.05 * np.median(np.abs(r_obs)) + EPS
        data_res = (r_pred - r_obs) / (np.abs(r_obs) + floor)

    n_cats = int(np.max(cat_idx_arr)) + 1
    pen_ea = []
    pen_dh = []
    for ci in range(n_cats):
        lnA, Ea, lnKH0, dH = unpack_block(params, ci)
        pen_ea.append(bound_penalty_residual(Ea, EA_SOFT_LOW, EA_SOFT_HIGH, EA_RANGE_SIGMA, EA_RANGE_MULTIPLIER))
        pen_dh.append(bound_penalty_residual(dH, DH_SOFT_LOW, DH_SOFT_HIGH, DH_RANGE_SIGMA, DH_RANGE_MULTIPLIER))
    pen_pool = ea_mean_pool_penalty_residual(params, n_cats)
    return np.concatenate([data_res] + pen_ea + pen_dh + [pen_pool])

# =========================================================
# FIT CORE
# =========================================================
def fit_joint_global(df, cat_order, n_nh3, x0_base=None):
    local_cat_to_idx = {c: i for i, c in enumerate(cat_order)}
    cat_idx_arr = df["catalyst"].map(local_cat_to_idx).to_numpy(dtype=int)
    T_K = df["T_K"].to_numpy(dtype=float)
    PNH3 = df["PNH3"].to_numpy(dtype=float)
    PH2 = df["PH2"].to_numpy(dtype=float)
    r_obs = df["r_exp"].to_numpy(dtype=float)

    lb, ub = build_bounds_joint(len(cat_order))
    base = initial_guess_joint(df, cat_order) if x0_base is None else np.clip(np.array(x0_base, dtype=float), lb, ub)
    rng = np.random.default_rng(RNG_SEED)

    best_res, best_score = None, np.inf
    for i in range(N_STARTS_GLOBAL):
        x0 = base.copy()
        if i != 0:
            for ci in range(len(cat_order)):
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
            args=(cat_idx_arr, T_K, PNH3, PH2, r_obs, n_nh3),
            method="trf",
            loss=LOSS_GLOBAL,
            f_scale=F_SCALE_GLOBAL,
            x_scale="jac",
            max_nfev=MAX_NFEV_GLOBAL,
        )
        score = float(np.sum(res.fun**2))
        if score < best_score:
            best_score, best_res = score, res

    se = parameter_se_from_result(best_res)
    r_calc = model_joint(best_res.x, cat_idx_arr, T_K, PNH3, PH2, n_nh3)
    metrics = calc_metrics(r_obs, r_calc)
    return best_res, se, best_score, metrics, r_calc, cat_idx_arr


def grid_search_n(df, cat_order, out_folder):
    rows = []
    best = None
    x0 = None
    for n_val in N_NH3_GRID:
        print(f"  fitting n = {n_val:.2f}")
        res, se, score, metrics, r_calc, cat_idx_arr = fit_joint_global(df, cat_order, n_val, x0_base=x0)
        # warm start next grid point
        x0 = res.x.copy()
        row = {"n_NH3": n_val, "score": score, **metrics}
        rows.append(row)
        if best is None or score < best["score"]:
            best = {"n_NH3": n_val, "score": score, "res": res, "se": se, "metrics": metrics,
                    "r_calc": r_calc, "cat_idx_arr": cat_idx_arr}
    grid_df = pd.DataFrame(rows).sort_values("n_NH3").reset_index(drop=True)
    grid_df.to_csv(os.path.join(out_folder, "n_grid_search.csv"), index=False)
    return best, grid_df

# =========================================================
# PLOTS
# =========================================================
def parity_plot_loglog(title, r_exp, r_pred, out_png, band_frac=0.20):
    mask = np.isfinite(r_exp) & np.isfinite(r_pred) & (r_exp > 0) & (r_pred > 0)
    x, y = np.asarray(r_exp)[mask], np.asarray(r_pred)[mask]
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


def plot_grid_n(grid_df, out_png):
    plt.figure(figsize=(6, 4))
    plt.plot(grid_df["n_NH3"], grid_df["score"], marker="o")
    plt.xlabel("NH3 exponent n")
    plt.ylabel("objective score")
    plt.title("Grid search for empirical NH3 exponent")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs(df, cat_order, best, grid_df, out_root, label):
    n_final = float(best["n_NH3"])
    params = best["res"].x
    se = best["se"]
    r_calc = best["r_calc"]
    cat_idx_arr = best["cat_idx_arr"]

    df_out = df.copy()
    df_out["cat_idx_local"] = cat_idx_arr
    df_out["n_NH3_final"] = n_final
    df_out["h2_inhibition_exponent_fixed"] = H2_INHIBITION_EXPONENT
    df_out["r_calc"] = r_calc
    df_out["rel_error_%"] = (df_out["r_calc"] - df_out["r_exp"]) / np.maximum(df_out["r_exp"], EPS) * 100.0

    # calculate KH and theta_H for every point
    KH_list = []
    theta_list = []
    k_list = []
    for _, row in df_out.iterrows():
        ci = int(row["cat_idx_local"])
        lnA, Ea, lnKH0, dH = unpack_block(params, ci)
        kT = k_of_T(lnA, Ea, row["T_K"])
        KHT = KH_of_T(lnKH0, dH, row["T_K"])
        k_list.append(kT)
        KH_list.append(KHT)
        theta_list.append(theta_H_from_KH_PH2(KHT, row["PH2"]))
    df_out["k_T"] = k_list
    df_out["KH2_T"] = KH_list
    df_out["theta_H"] = theta_list

    df_out.to_csv(os.path.join(out_root, f"{label}_predictions.csv"), index=False)

    param_rows = []
    temp_rows = []
    for cat in cat_order:
        ci = cat_order.index(cat)
        lnA, Ea, lnKH0, dH = unpack_block(params, ci)
        SE_lnA, SE_Ea, SE_lnKH0, SE_dH = se[block_slice(ci)]
        sub = df_out[df_out["catalyst"] == cat].copy()
        metrics = calc_metrics(sub["r_exp"], sub["r_calc"])
        param_rows.append({"catalyst": cat, "n_NH3_final": n_final,
                           "h2_inhibition_exponent_fixed": H2_INHIBITION_EXPONENT,
                           "lnA": lnA, "SE_lnA": SE_lnA,
                           "Ea_J_per_mol": Ea, "SE_Ea_J_per_mol": SE_Ea,
                           "Ea_kJ_per_mol": Ea/1000.0, "SE_Ea_kJ_per_mol": SE_Ea/1000.0,
                           "lnKH0": lnKH0, "SE_lnKH0": SE_lnKH0,
                           "dH_J_per_mol": dH, "SE_dH_J_per_mol": SE_dH,
                           "dH_kJ_per_mol": dH/1000.0, "SE_dH_kJ_per_mol": SE_dH/1000.0,
                           "score": best["score"], **metrics})

        for T_C, subT in sub.groupby("T_C"):
            T_K = float(subT["T_K"].iloc[0])
            kT = k_of_T(lnA, Ea, T_K)
            KHT = KH_of_T(lnKH0, dH, T_K)
            temp_rows.append({"catalyst": cat, "T_C": T_C, "T_K": T_K,
                              "n_NH3_final": n_final,
                              "k_T": kT, "ln_k_T": np.log(max(kT, EPS)),
                              "KH2_T": KHT, "ln_KH2_T": np.log(max(KHT, EPS)),
                              "theta_H_mean": float(subT["theta_H"].mean()),
                              "theta_H_min": float(subT["theta_H"].min()),
                              "theta_H_max": float(subT["theta_H"].max()),
                              **calc_metrics(subT["r_exp"], subT["r_calc"])})

        cat_out = os.path.join(out_root, cat)
        os.makedirs(cat_out, exist_ok=True)
        sub.to_csv(os.path.join(cat_out, f"{cat}_predictions.csv"), index=False)
        parity_plot_loglog(f"Parity plot - {cat}", sub["r_exp"], sub["r_calc"], os.path.join(cat_out, f"parity_{cat}.png"), ERROR_BAND_FRAC)
        residual_plot(sub["PH2"], sub["rel_error_%"], "PH2", f"Residual vs PH2 - {cat}", os.path.join(cat_out, f"{cat}_residual_vs_PH2.png"))
        residual_plot(sub["PNH3"], sub["rel_error_%"], "PNH3", f"Residual vs PNH3 - {cat}", os.path.join(cat_out, f"{cat}_residual_vs_PNH3.png"))
        residual_plot(sub["T_C"], sub["rel_error_%"], "T (C)", f"Residual vs T - {cat}", os.path.join(cat_out, f"{cat}_residual_vs_T.png"))

    pd.DataFrame(param_rows).to_csv(os.path.join(out_root, f"{label}_parameters.csv"), index=False)
    pd.DataFrame(temp_rows).to_csv(os.path.join(out_root, f"{label}_k_KH2_thetaH_by_temperature.csv"), index=False)

    parity_plot_loglog(f"Parity plot - {label}", df_out["r_exp"], df_out["r_calc"], os.path.join(out_root, f"parity_{label}.png"), ERROR_BAND_FRAC)
    plot_grid_n(grid_df, os.path.join(out_root, f"{label}_n_grid_score.png"))

# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 100)
    print("SEMI-EMPIRICAL H2-POISONING LH MODEL")
    print("r = k(T) * PNH3^n / (1 + sqrt(KH2(T)*PH2))^4")
    print("one common n is selected by grid search from 0.00 to 0.30 with step 0.01")
    print("=" * 100)

    out_root = os.path.join(ROOT, OUT_FOLDER_NAME)
    os.makedirs(out_root, exist_ok=True)
    df = load_all_data()
    df.to_csv(os.path.join(out_root, "loaded_data.csv"), index=False)

    if USE_COMMON_N_FOR_ALL_CATALYSTS:
        print("\nFitting all catalysts together with one common n...")
        best, grid_df = grid_search_n(df, CAT_ORDER, out_root)
        save_outputs(df, CAT_ORDER, best, grid_df, out_root, label="ALL_common_n")
        print(f"\nBest common n = {best['n_NH3']:.2f}")
        print(f"Best score    = {best['score']:.6g}")
        print(f"MAPE          = {best['metrics']['MAPE_%']:.3f} %")
    else:
        summary = []
        for cat in CAT_ORDER:
            print("\n" + "=" * 100)
            print(f"Fitting catalyst {cat} with its own n")
            print("=" * 100)
            cat_out = os.path.join(out_root, cat)
            os.makedirs(cat_out, exist_ok=True)
            df_cat = df[df["catalyst"] == cat].copy().reset_index(drop=True)
            best, grid_df = grid_search_n(df_cat, [cat], cat_out)
            save_outputs(df_cat, [cat], best, grid_df, cat_out, label=cat)
            summary.append({"catalyst": cat, "n_NH3_final": best["n_NH3"], "score": best["score"], **best["metrics"]})
            print(f"Best n for {cat} = {best['n_NH3']:.2f}")
            print(f"MAPE             = {best['metrics']['MAPE_%']:.3f} %")
        pd.DataFrame(summary).to_csv(os.path.join(out_root, "summary_best_n_by_catalyst.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"Saved results to:\n{out_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
