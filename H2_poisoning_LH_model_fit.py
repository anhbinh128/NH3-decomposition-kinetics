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

OUT_FOLDER_NAME = "Fit_results_joint_global_Ea_meanpool"

# Fixed model parameters
M_FIXED = 0.1
N_FIXED = 4.0
LAMBDA_FEED_FIXED = 1.736

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
MAX_NFEV_GLOBAL = 150000
N_STARTS_GLOBAL = 60
RNG_SEED = 123

USE_LOG_RESIDUAL = True

# =========================================================
# Ea SOFT PRIOR (PHYSICAL RANGE)
# =========================================================
# Encourage each Ea to stay in 70-170 kJ/mol
EA_SOFT_LOW = 70e3
EA_SOFT_HIGH = 170e3
EA_RANGE_SIGMA = 15e3
EA_RANGE_MULTIPLIER = 6

# =========================================================
# Ea MEAN-POOLING PENALTY
# =========================================================
# Penalize deviations from mean Ea across catalysts
# Softer than pairwise pooling
EA_POOL_SIGMA = 35e3
EA_POOL_MULTIPLIER = 2

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
    400e3     # dH (J/mol)
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
    """
    Approximate SE from Jacobian.
    Because penalty residuals are included, SE is regularized/approximate.
    """
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


def k_of_T(lnA, Ea, T_K):
    return np.exp(np.clip(lnA - Ea / (R * T_K), -700, 700))


def KH_of_T(lnKH0, dH, T_K):
    return np.exp(np.clip(lnKH0 - dH / (R * T_K), -700, 700))


# =========================================================
# MODEL
# =========================================================
def model_joint(params, cat_idx_arr, T_K, PNH3, PH2, feed_flag):
    r_pred = np.empty_like(T_K, dtype=float)

    for cat_name, ci in CAT_TO_IDX.items():
        mask = (cat_idx_arr == ci)
        if not np.any(mask):
            continue

        lnA, Ea, lnKH0, dH = unpack_block(params, ci)

        ln_k = lnA - Ea / (R * T_K[mask])
        ln_KH = lnKH0 - dH / (R * T_K[mask])

        k = np.exp(np.clip(ln_k, -700, 700))
        KH = np.exp(np.clip(ln_KH, -700, 700))

        feed_factor = np.where(feed_flag[mask] > 0, LAMBDA_FEED_FIXED, 1.0)
        inhib = feed_factor * np.sqrt(np.maximum(KH * PH2[mask], 0.0))
        denom = (1.0 + inhib) ** N_FIXED

        r_pred[mask] = k * np.power(np.maximum(PNH3[mask], EPS), M_FIXED) / denom

    return np.maximum(r_pred, EPS)


def ea_range_penalty_residual(Ea):
    """
    Zero inside [EA_SOFT_LOW, EA_SOFT_HIGH], nonzero outside.
    """
    low_violation = max(0.0, EA_SOFT_LOW - Ea)
    high_violation = max(0.0, Ea - EA_SOFT_HIGH)

    res_low = low_violation / EA_RANGE_SIGMA
    res_high = high_violation / EA_RANGE_SIGMA

    pen = []
    for _ in range(EA_RANGE_MULTIPLIER):
        pen.append(res_low)
        pen.append(res_high)
    return np.array(pen, dtype=float)


def ea_mean_pool_penalty_residual(Ea_La, Ea_Ce, Ea_CoNi):
    """
    Penalize deviation from mean Ea.
    Softer than pairwise pooling.
    """
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


def residual_joint(params, cat_idx_arr, T_K, PNH3, PH2, feed_flag, r_obs):
    r_pred = model_joint(params, cat_idx_arr, T_K, PNH3, PH2, feed_flag)

    if USE_LOG_RESIDUAL:
        data_res = np.log(r_pred + EPS) - np.log(r_obs + EPS)
    else:
        floor = 0.05 * np.median(np.abs(r_obs)) + EPS
        data_res = (r_pred - r_obs) / (np.abs(r_obs) + floor)

    Ea_La = params[block_slice(CAT_TO_IDX["La"])][1]
    Ea_Ce = params[block_slice(CAT_TO_IDX["Ce"])][1]
    Ea_CoNi = params[block_slice(CAT_TO_IDX["CoNi"])][1]

    pen_range = np.concatenate([
        ea_range_penalty_residual(Ea_La),
        ea_range_penalty_residual(Ea_Ce),
        ea_range_penalty_residual(Ea_CoNi)
    ])

    pen_pool = ea_mean_pool_penalty_residual(Ea_La, Ea_Ce, Ea_CoNi)

    return np.concatenate([data_res, pen_range, pen_pool])


# =========================================================
# INITIAL GUESS / FIT
# =========================================================
def build_bounds_joint():
    lb = np.tile(LB_BLOCK, len(CAT_ORDER))
    ub = np.tile(UB_BLOCK, len(CAT_ORDER))
    return lb, ub


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
        lnA_guess = np.log(max(np.median(r), 1e-20)) + 110e3 / (R * tmid)
        Ea_guess = 110e3

        ph2_pos = PH2[PH2 > 0]
        if len(ph2_pos) > 0:
            KH_guess = 1.0 / max(np.median(ph2_pos), 1e-12)
        else:
            KH_guess = 1.0

        lnKH0_guess = np.log(np.clip(KH_guess, 1e-30, 1e30))
        dH_guess = 90e3

        block = np.array([lnA_guess, Ea_guess, lnKH0_guess, dH_guess], dtype=float)
        block = np.clip(block, LB_BLOCK, UB_BLOCK)
        parts.append(block)

    return np.concatenate(parts)


def fit_joint_global(df):
    lb, ub = build_bounds_joint()

    cat_idx_arr = df["cat_idx"].to_numpy(dtype=int)
    T_K = df["T_K"].to_numpy(dtype=float)
    PNH3 = df["PNH3"].to_numpy(dtype=float)
    PH2 = df["PH2"].to_numpy(dtype=float)
    feed_flag = df["feed_flag"].to_numpy(dtype=int)
    r_obs = df["r_exp"].to_numpy(dtype=float)

    base = initial_guess_joint(df)
    rng = np.random.default_rng(RNG_SEED)

    best_res = None
    best_score = np.inf

    for i in range(N_STARTS_GLOBAL):
        x0 = base.copy()

        if i != 0:
            for ci in range(len(CAT_ORDER)):
                sl = block_slice(ci)
                x0[sl][0] += rng.uniform(-3.0, 3.0)        # lnA
                x0[sl][1] += rng.uniform(-25e3, 25e3)      # Ea
                x0[sl][2] += rng.uniform(-3.0, 3.0)        # lnKH0
                x0[sl][3] += rng.uniform(-40e3, 40e3)      # dH

            x0 = np.clip(x0, lb, ub)

        res = least_squares(
            residual_joint,
            x0=x0,
            bounds=(lb, ub),
            args=(cat_idx_arr, T_K, PNH3, PH2, feed_flag, r_obs),
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


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 100)
    print("JOINT GLOBAL LH FIT WITH Ea MEAN-POOLING PENALTY")
    print(f"Model: r = k(T) * PNH3^{M_FIXED} / (1 + lambda_feed^Ifeed * sqrt(KH(T)*PH2))^{N_FIXED}")
    print(f"Fixed lambda_feed = {LAMBDA_FEED_FIXED}")
    print(f"Ea soft prior range = [{EA_SOFT_LOW/1000:.1f}, {EA_SOFT_HIGH/1000:.1f}] kJ/mol")
    print(f"Ea range sigma      = {EA_RANGE_SIGMA/1000:.1f} kJ/mol")
    print(f"Ea mean-pool sigma  = {EA_POOL_SIGMA/1000:.1f} kJ/mol")
    print(f"Ea mean-pool mult   = {EA_POOL_MULTIPLIER}")
    print("=" * 100)

    out_root = os.path.join(ROOT, OUT_FOLDER_NAME)
    os.makedirs(out_root, exist_ok=True)

    # -----------------------------------------------------
    # READ ALL DATA
    # -----------------------------------------------------
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

    df = pd.DataFrame(all_rows).sort_values(["catalyst", "T_C", "file", "row_id"]).reset_index(drop=True)

    # -----------------------------------------------------
    # JOINT FIT
    # -----------------------------------------------------
    res, se, score = fit_joint_global(df)
    params = res.x

    r_calc = model_joint(
        params,
        df["cat_idx"].to_numpy(dtype=int),
        df["T_K"].to_numpy(dtype=float),
        df["PNH3"].to_numpy(dtype=float),
        df["PH2"].to_numpy(dtype=float),
        df["feed_flag"].to_numpy(dtype=int)
    )

    df["r_calc"] = r_calc
    df["rel_error_%"] = (df["r_calc"] - df["r_exp"]) / np.maximum(df["r_exp"], EPS) * 100.0

    # Save overall predictions
    df.to_csv(os.path.join(out_root, "ALL_CATALYSTS_joint_predictions.csv"), index=False)

    # -----------------------------------------------------
    # PARAMETER TABLES
    # -----------------------------------------------------
    param_rows = []
    summary_rows = []
    temp_rows_all = []

    Ea_vals = {}

    for cat in CAT_ORDER:
        ci = CAT_TO_IDX[cat]
        sl = block_slice(ci)

        lnA, Ea, lnKH0, dH = params[sl]
        SE_lnA, SE_Ea, SE_lnKH0, SE_dH = se[sl]

        Ea_vals[cat] = Ea

        sub = df[df["catalyst"] == cat].copy()
        metrics = calc_metrics(sub["r_exp"].to_numpy(), sub["r_calc"].to_numpy())

        penalty_low = max(0.0, EA_SOFT_LOW - Ea) / EA_RANGE_SIGMA
        penalty_high = max(0.0, Ea - EA_SOFT_HIGH) / EA_RANGE_SIGMA
        penalty_active = (penalty_low > 0) or (penalty_high > 0)

        param_rows.append({
            "catalyst": cat,
            "lambda_feed_fixed": LAMBDA_FEED_FIXED,
            "m_fixed": M_FIXED,
            "n_fixed": N_FIXED,

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

            "objective_score_joint": score,
            "Ea_range_penalty_active": int(penalty_active),
            "Ea_penalty_low_residual": penalty_low,
            "Ea_penalty_high_residual": penalty_high,

            **metrics
        })

        summary_rows.append({
            "catalyst": cat,
            "n_temperatures": sub["T_K"].nunique(),
            "total_points": len(sub),

            "Ea_kJ_per_mol": Ea / 1000.0,
            "SE_Ea_kJ_per_mol": SE_Ea / 1000.0,
            "dH_kJ_per_mol": dH / 1000.0,
            "SE_dH_kJ_per_mol": SE_dH / 1000.0,

            "overall_RMSE": metrics["RMSE"],
            "overall_MAE": metrics["MAE"],
            "overall_MAPE_%": metrics["MAPE_%"],
            "overall_R2": metrics["R2"],
        })

        # Per-temperature derived k(T), KH(T)
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
        temp_df["lnA_global"] = lnA
        temp_df["Ea_kJ_per_mol_global"] = Ea / 1000.0
        temp_df["lnKH0_global"] = lnKH0
        temp_df["dH_kJ_per_mol_global"] = dH / 1000.0
        temp_rows_all.append(temp_df)

        cat_out = os.path.join(out_root, cat)
        os.makedirs(cat_out, exist_ok=True)

        sub.to_csv(os.path.join(cat_out, f"{cat}_predictions_from_joint_fit.csv"), index=False)
        temp_df.to_csv(os.path.join(cat_out, f"{cat}_k_KH_by_temperature_from_joint_fit.csv"), index=False)

        # Plots
        parity_plot_loglog(
            title=f"Parity plot - {cat} (joint Ea-meanpooled fit)",
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

        # Arrhenius and van't Hoff plots
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

        print("\n" + "-" * 80)
        print(f"{cat}")
        print(f"  lnA   = {lnA:.6g} ± {SE_lnA:.3g}")
        print(f"  Ea    = {Ea/1000.0:.6g} ± {SE_Ea/1000.0:.3g} kJ/mol")
        print(f"  lnKH0 = {lnKH0:.6g} ± {SE_lnKH0:.3g}")
        print(f"  dH    = {dH/1000.0:.6g} ± {SE_dH/1000.0:.3g} kJ/mol")
        print(f"  RMSE  = {metrics['RMSE']:.6g}")
        print(f"  MAPE  = {metrics['MAPE_%']:.6g} %")

    # -----------------------------------------------------
    # Mean-pooling diagnostics
    # -----------------------------------------------------
    Ea_mean = (Ea_vals["La"] + Ea_vals["Ce"] + Ea_vals["CoNi"]) / 3.0

    pool_diag = {
        "Ea_La_kJ_per_mol": Ea_vals["La"] / 1000.0,
        "Ea_Ce_kJ_per_mol": Ea_vals["Ce"] / 1000.0,
        "Ea_CoNi_kJ_per_mol": Ea_vals["CoNi"] / 1000.0,
        "Ea_mean_kJ_per_mol": Ea_mean / 1000.0,
        "Ea_La_minus_mean_kJ_per_mol": (Ea_vals["La"] - Ea_mean) / 1000.0,
        "Ea_Ce_minus_mean_kJ_per_mol": (Ea_vals["Ce"] - Ea_mean) / 1000.0,
        "Ea_CoNi_minus_mean_kJ_per_mol": (Ea_vals["CoNi"] - Ea_mean) / 1000.0,
        "Ea_mean_pool_sigma_kJ_per_mol": EA_POOL_SIGMA / 1000.0,
        "Ea_range_sigma_kJ_per_mol": EA_RANGE_SIGMA / 1000.0,
        "objective_score_joint": score
    }

    # -----------------------------------------------------
    # SAVE TABLES
    # -----------------------------------------------------
    param_df = pd.DataFrame(param_rows)
    summary_df = pd.DataFrame(summary_rows)
    temp_all_df = pd.concat(temp_rows_all, ignore_index=True)

    param_df.to_csv(os.path.join(out_root, "ALL_CATALYSTS_joint_fit_parameters.csv"), index=False)
    summary_df.to_csv(os.path.join(out_root, "ALL_CATALYSTS_joint_fit_summary.csv"), index=False)
    temp_all_df.to_csv(os.path.join(out_root, "ALL_CATALYSTS_k_KH_by_temperature_from_joint_fit.csv"), index=False)
    pd.DataFrame([pool_diag]).to_csv(os.path.join(out_root, "JOINT_Ea_meanpool_diagnostics.csv"), index=False)

    # Overall parity across all catalysts
    parity_plot_loglog(
        title="Parity plot - all catalysts (joint Ea-meanpooled fit)",
        r_exp=df["r_exp"].to_numpy(),
        r_pred=df["r_calc"].to_numpy(),
        out_png=os.path.join(out_root, "parity_ALL_catalysts.png"),
        band_frac=ERROR_BAND_FRAC
    )

    residual_plot(
        df["PH2"].to_numpy(),
        df["rel_error_%"].to_numpy(),
        xlabel="PH2",
        title="Residual vs PH2 - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_PH2.png")
    )

    residual_plot(
        df["PNH3"].to_numpy(),
        df["rel_error_%"].to_numpy(),
        xlabel="PNH3",
        title="Residual vs PNH3 - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_PNH3.png")
    )

    residual_plot(
        df["T_C"].to_numpy(),
        df["rel_error_%"].to_numpy(),
        xlabel="T (C)",
        title="Residual vs T - all catalysts",
        out_png=os.path.join(out_root, "ALL_residual_vs_T.png")
    )

    print("\n" + "=" * 100)
    print("JOINT Ea VALUES AFTER MEAN-POOLING:")
    print(f"  Ea(La)    = {Ea_vals['La']/1000.0:.4f} kJ/mol")
    print(f"  Ea(Ce)    = {Ea_vals['Ce']/1000.0:.4f} kJ/mol")
    print(f"  Ea(CoNi)  = {Ea_vals['CoNi']/1000.0:.4f} kJ/mol")
    print(f"  Ea(mean)  = {Ea_mean/1000.0:.4f} kJ/mol")
    print(f"  La - mean    = {(Ea_vals['La'] - Ea_mean)/1000.0:.4f} kJ/mol")
    print(f"  Ce - mean    = {(Ea_vals['Ce'] - Ea_mean)/1000.0:.4f} kJ/mol")
    print(f"  CoNi - mean  = {(Ea_vals['CoNi'] - Ea_mean)/1000.0:.4f} kJ/mol")
    print("=" * 100)

    print(f"Saved results to:\n{out_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()