# ============================================================
# ELECTROSTATIC ΔΔG MODELING PIPELINE
# ============================================================
#
# Features:
#   1. Physics-based electrostatic descriptors
#   2. Distance-dependent dielectric screening
#   3. LOOCV evaluation
#   4. Feature subset search
#   6. Rank-2 free-energy model
#   7. Correlation analysis
#   8. Feature importance analysis
#
# ============================================================

import os
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from Bio.PDB import PDBParser

from scipy.stats import pearsonr

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

OUTDIR = "results_rc_final"
os.makedirs(OUTDIR, exist_ok=True)

PDB_FILE = "WT_amber.pdb"

# ============================================================
# ELECTROSTATIC CONSTANTS
# ============================================================

COULOMB = 332.0
KAPPA = 0.10

EPS_IN = 4.0
EPS_OUT = 80.0

LAMBDA = 7.0

# Effective Ca2+ charge
CA_CHARGE = 2.0

# Effective Born radius
R_EFF = 2.0

# ============================================================
# EXPERIMENTAL DATA
# ============================================================

exp_data = {
    "WT": -6.56,
    "Y5H": -6.44,
    "A8V": -6.66,
    "F20Q": -6.97,
    "A23Q": -7.59,
    "L29Q": -6.89,
    "A31S": -6.93,
    "S37G": -6.67,
    "E40A": -5.87,
    "V44Q": -8.19,
    "M45Q": -7.46,
    "L48Q": -7.63,
    "Q50R": -7.31,
    "L57Q": -5.99,
    "E59D": -6.23,
    "I61Q": -5.39,
    "D67A": -5.09,
    "D73A": -5.09,
    "D73N": -5.69,
    "D75Y": -6.14,
    "V79Q": -6.63,
    "M81Q": -7.28,
    "C84Y": -7.37
}

mutations = [
    ("Y5H", 5),
    ("A8V", 8),
    ("F20Q", 20),
    ("A23Q", 23),
    ("L29Q", 29),
    ("A31S", 31),
    ("S37G", 37),
    ("E40A", 40),
    ("V44Q", 44),
    ("M45Q", 45),
    ("L48Q", 48),
    ("Q50R", 50),
    ("L57Q", 57),
    ("E59D", 59),
    ("I61Q", 61),
    ("D67A", 67),
    ("D73A", 73),
    ("D73N", 73),
    ("D75Y", 75),
    ("V79Q", 79),
    ("M81Q", 81),
    ("C84Y", 84)
]

# ============================================================
# CHARGE DEFINITIONS
# ============================================================

charge = {
    "D": -1,
    "E": -1,
    "K":  1,
    "R":  1,
    "H":  0.3
}

aa_map = {
    "ASP": "D",
    "GLU": "E",
    "LYS": "K",
    "ARG": "R",
    "HIS": "H"
}

# ============================================================
# UTILITY
# ============================================================

def banner(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ============================================================
# STRUCTURE PARSING
# ============================================================

banner("LOADING STRUCTURE")

parser = PDBParser(QUIET=True)
model = parser.get_structure("WT", PDB_FILE)[0]

residues = [
    (res.id[1], res, res["CA"].coord)
    for chain in model
    for res in chain
    if "CA" in res
]

print(f"[INFO] Total CA residues: {len(residues)}")

# ============================================================
# CALCIUM POSITION
# ============================================================

ca_pos = None

for atom in model.get_atoms():

    if atom.get_parent().get_resname().strip() == "CA":
        ca_pos = atom.coord
        break

if ca_pos is None:
    raise ValueError("Calcium ion not found.")

# ============================================================
# ELECTROSTATIC PHYSICS
# ============================================================

def dielectric(r: float) -> float:
    """
    Distance-dependent dielectric function.
    """
    return EPS_OUT - (EPS_OUT - EPS_IN) * np.exp(-r / LAMBDA)

# ============================================================

def screened_coulomb(q: float, r: float) -> float:
    """
    Screened Coulomb interaction.
    """
    return (
        COULOMB
        * q
        * np.exp(-KAPPA * r)
        / (dielectric(r) * r)
    )

# ============================================================

@lru_cache(maxsize=None)
def potential_cached(
    x: float,
    y: float,
    z: float,
    exclude_index: int
) -> float:

    coord_i = np.array([x, y, z])

    phi = 0.0

    for rid_j, res_j, coord_j in residues:

        if rid_j == exclude_index:
            continue

        aa = aa_map.get(res_j.get_resname(), None)

        if aa is None:
            continue

        q = charge.get(aa, 0)

        if q == 0:
            continue

        r = np.linalg.norm(coord_i - coord_j)

        if r < 1e-6:
            continue

        phi += screened_coulomb(q, r)

    return phi

# ============================================================

def potential(coord: np.ndarray, exclude_index: int) -> float:

    x, y, z = np.round(coord, 3)

    return potential_cached(x, y, z, exclude_index)

# ============================================================

def compute_field_features(
    ca_position: np.ndarray,
    coord_i: np.ndarray,
    mut_resi: int,
    dq: float
):

    E_vec = np.zeros(3)

    for rid_j, res_j, coord_j in residues:

        aa = aa_map.get(res_j.get_resname(), None)

        if aa is None:
            continue

        q = charge.get(aa, 0)

        if rid_j == mut_resi:
            q += dq

        if q == 0:
            continue

        r_vec = coord_j - ca_position
        r = np.linalg.norm(r_vec)

        if r < 1e-6:
            continue

        pref = (
            COULOMB
            * q
            * np.exp(-KAPPA * r)
            / dielectric(r)
        )

        E_vec += pref * r_vec / (r ** 3)

    E_mag = np.linalg.norm(E_vec)

    axis = coord_i - ca_position

    u_axis = axis / (np.linalg.norm(axis) + 1e-8)

    E_parallel = np.dot(E_vec, u_axis)

    E_perp = np.sqrt(
        max(E_mag**2 - E_parallel**2, 0.0)
    )

    return [E_mag, E_parallel, E_perp]

# ============================================================
# FEATURE ENGINEERING
# ============================================================

banner("FEATURE ENGINEERING")

rows = []

for mut, resi in tqdm(mutations):

    coord_i = next(
        coord
        for rid, _, coord in residues
        if rid == resi
    )

    wt = mut[0]
    mt = mut[-1]

    dq = charge.get(mt, 0) - charge.get(wt, 0)

    # --------------------------------------------------------
    # Electrostatic potential
    # --------------------------------------------------------

    phi = potential(coord_i, resi)

    # --------------------------------------------------------
    # Ca2+ interaction
    # --------------------------------------------------------

    r_ca = np.linalg.norm(coord_i - ca_pos)

    E_ca = (
        COULOMB
        * dq
        * CA_CHARGE
        * np.exp(-KAPPA * r_ca)
        / (dielectric(r_ca) * r_ca)
    )

    # --------------------------------------------------------
    # Born term
    # --------------------------------------------------------

    born = (
        (dq ** 2)
        * (1 / EPS_IN - 1 / EPS_OUT)
        / (2 * R_EFF)
    )

    # --------------------------------------------------------
    # Electric field features
    # --------------------------------------------------------

    E_mag, E_parallel, E_perp = compute_field_features(
        ca_pos,
        coord_i,
        resi,
        dq
    )

    rows.append([
        dq * phi,
        E_ca,
        born,
        phi ** 2,
        E_mag,
        E_parallel,
        E_perp,
        exp_data[mut] - exp_data["WT"],
        mut,
        resi
    ])

# ============================================================
# DATAFRAME
# ============================================================

columns = [
    "dq_phi",
    "E_ca",
    "born",
    "phi_sq",
    "E_mag",
    "E_parallel",
    "E_perp",
    "ddG_exp",
    "mutant",
    "resi"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv(
    f"{OUTDIR}/raw_features.csv",
    index=False
)

features = [
    "dq_phi",
    "E_ca",
    "born",
    "phi_sq",
    "E_mag",
    "E_parallel",
    "E_perp"
]

X_full = df[features].values
y_full = df["ddG_exp"].values

# ============================================================
# FEATURE CORRELATION
# ============================================================

banner("FEATURE CORRELATION")

corr = df[features].corr()

print(corr.round(2))

corr.to_csv(
    f"{OUTDIR}/feature_correlation.csv"
)

# ============================================================
# PREPROCESSING
# ============================================================

def whiten(X: np.ndarray):

    if X.shape[1] == 1:

        std = np.std(X) + 1e-8

        return X / std, np.eye(1)

    lw = LedoitWolf().fit(X)

    cov = lw.covariance_

    U, S, _ = np.linalg.svd(cov)

    W = (
        U
        @ np.diag(1.0 / np.sqrt(S + 1e-6))
        @ U.T
    )

    return X @ W, W

# ============================================================

def preprocess_train_test(Xtr, Xte):

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr)

    Xte_scaled = scaler.transform(Xte)

    Xtr_w, W = whiten(Xtr_scaled)

    Xte_w = Xte_scaled @ W

    return Xtr_w, Xte_w

# ============================================================
# LOOCV REGRESSION
# ============================================================

def compute_loocv_metrics(X, y):

    loo = LeaveOneOut()

    preds = np.zeros(len(y))

    for tr, te in loo.split(X):

        Xtr_w, Xte_w = preprocess_train_test(
            X[tr],
            X[te]
        )

        model = LinearRegression()

        model.fit(Xtr_w, y[tr])

        preds[te] = model.predict(Xte_w)

    rmse = np.sqrt(
        mean_squared_error(y, preds)
    )

    r2 = r2_score(y, preds)

    rp, p = pearsonr(y, preds)

    stderr = np.std(y - preds)

    return rmse, r2, rp, p, stderr

# ============================================================
# FEATURE SUBSET SEARCH
# ============================================================

banner("FEATURE SUBSET SEARCH")

subset_results = []

for k in tqdm(
    range(1, len(features) + 1),
    desc="Subset size"
):

    for subset in itertools.combinations(features, k):

        idx = [features.index(f) for f in subset]

        rmse, r2, rp, p, stderr = compute_loocv_metrics(
            X_full[:, idx],
            y_full
        )

        subset_results.append([
            subset,
            rmse,
            r2,
            rp,
            stderr
        ])

# ============================================================

df_rank = pd.DataFrame(
    subset_results,
    columns=[
        "subset",
        "rmse",
        "r2",
        "rp",
        "stderr"
    ]
)

df_rank = df_rank.sort_values("rmse")

df_rank.to_csv(
    f"{OUTDIR}/subset_ranking.csv",
    index=False
)

best_subset = df_rank.iloc[0]["subset"]

print("\n[INFO] Best subset:")
print(best_subset)

# ============================================================
# BEST SUBSET MODEL
# ============================================================

banner("BEST SUBSET MODEL")

idx_best = [
    features.index(f)
    for f in best_subset
]

X_best = X_full[:, idx_best]

rmse_best, r2_best, rp_best, p_best, stderr_best = (
    compute_loocv_metrics(X_best, y_full)
)

print(f"RMSE   = {rmse_best:.2f}")
print(f"R²     = {r2_best:.2f}")
print(f"Rₚ     = {rp_best:.2f}")
print(f"StdErr = {stderr_best:.2f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_best)

importance_model = LinearRegression()

importance_model.fit(X_scaled, y_full)

importance = np.abs(importance_model.coef_)

importance /= np.sum(importance)

df_imp = pd.DataFrame({
    "feature": best_subset,
    "importance": importance
})

df_imp = df_imp.sort_values(
    "importance",
    ascending=False
)

print("\n[INFO] Feature importance:")
print(df_imp.round(2))

df_imp.to_csv(
    f"{OUTDIR}/feature_importance.csv",
    index=False
)




# ============================================================
# BEST SUBSET STABILITY ANALYSIS
# ============================================================

banner("BEST SUBSET STABILITY")

loo = LeaveOneOut()

coef_all = []

for tr, te in loo.split(X_best):

    Xtr_w, Xte_w = preprocess_train_test(
        X_best[tr],
        X_best[te]
    )

    stability_model = LinearRegression()

    stability_model.fit(
        Xtr_w,
        y_full[tr]
    )

    coef = stability_model.coef_

    # --------------------------------------------------------
    # normalize for directional consistency
    # --------------------------------------------------------

    coef /= (
        np.linalg.norm(coef)
        + 1e-8
    )

    coef_all.append(coef)

coef_all = np.array(coef_all)

coef_mean = np.mean(coef_all, axis=0)
coef_std = np.std(coef_all, axis=0)

# ------------------------------------------------------------
# sign consistency
# ------------------------------------------------------------

sign_consistency = []

for i in range(coef_all.shape[1]):

    pos = np.sum(coef_all[:, i] > 0)
    neg = np.sum(coef_all[:, i] < 0)

    frac = max(pos, neg) / len(coef_all)

    sign_consistency.append(frac)

# ------------------------------------------------------------
# summary dataframe
# ------------------------------------------------------------

df_stability = pd.DataFrame({

    "feature": list(best_subset),

    "coef_mean": coef_mean,

    "coef_std": coef_std,

    "stability_ratio": np.abs(coef_mean) / (coef_std + 1e-8),

    "sign_consistency": sign_consistency
})

df_stability = df_stability.sort_values(
    "stability_ratio",
    ascending=False
)

print(df_stability.round(2))

df_stability.to_csv(
    f"{OUTDIR}/best_subset_stability.csv",
    index=False
)
# ============================================================
# SAVE BEST DATASET
# ============================================================

df_best = df[
    list(best_subset)
    + ["ddG_exp", "mutant", "resi"]
]

df_best.to_csv(
    f"{OUTDIR}/best_features_dataset.csv",
    index=False
)

# ============================================================
# FULL MODEL PERFORMANCE
# ============================================================

banner("FULL FEATURE MODEL")

rmse_full, r2_full, rp_full, p_full, stderr_full = (
    compute_loocv_metrics(X_full, y_full)
)

print(f"RMSE   = {rmse_full:.2f}")
print(f"R²     = {r2_full:.2f}")
print(f"Rₚ     = {rp_full:.2f}")
print(f"StdErr = {stderr_full:.2f}")


# ============================================================
# FINAL RC MODEL
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_full)

X_w, W = whiten(X_scaled)

direction_model = LinearRegression(
    fit_intercept=True
)

direction_model.fit(X_w, y_full)

w_final = direction_model.coef_

w_final /= (
    np.linalg.norm(w_final)
    + 1e-8
)

xi = X_w @ w_final

energy_model = LinearRegression()

energy_model.fit(
    xi.reshape(-1, 1),
    y_full
)

a = energy_model.coef_[0]
b = energy_model.intercept_


# ============================================================
# RANK-2 MODEL
# ============================================================

banner("RANK-2 MODEL")

rank2_features = [
    "E_ca",
    "phi_sq"
]

X_rank2 = df[rank2_features].values

rmse_rank2, r2_rank2, rp_rank2, p_rank2, stderr_rank2 = (
    compute_loocv_metrics(
        X_rank2,
        y_full
    )
)

print(f"RMSE   = {rmse_rank2:.2f}")
print(f"R²     = {r2_rank2:.2f}")
print(f"Rₚ     = {rp_rank2:.2f}")
print(f"StdErr = {stderr_rank2:.2f}")

# ============================================================
# FINAL RANK-2 FIT
# ============================================================

scaler = StandardScaler()

X_rank2_scaled = scaler.fit_transform(X_rank2)

rank2_model = LinearRegression()

rank2_model.fit(
    X_rank2_scaled,
    y_full
)

coef = rank2_model.coef_
intercept = rank2_model.intercept_

# ============================================================
# RANK-2 IMPORTANCE
# ============================================================

importance = np.abs(coef)

importance /= np.sum(importance)

df_rank2_imp = pd.DataFrame({
    "feature": rank2_features,
    "importance": importance
})

df_rank2_imp = df_rank2_imp.sort_values(
    "importance",
    ascending=False
)

df_rank2_imp.to_csv(
    f"{OUTDIR}/rank2_importance.csv",
    index=False
)

# ============================================================
# RANK-2 PREDICTIONS
# ============================================================

loo = LeaveOneOut()

preds_rank2 = np.zeros(len(y_full))

for tr, te in loo.split(X_rank2):

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(X_rank2[tr])

    Xte = scaler.transform(X_rank2[te])

    model = LinearRegression()

    model.fit(Xtr, y_full[tr])

    preds_rank2[te] = model.predict(Xte)

df_rank2 = df.copy()

df_rank2["ddG_pred_rank2"] = preds_rank2

df_rank2.to_csv(
    f"{OUTDIR}/rank2_predictions.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

banner("FINAL MODEL SUMMARY")

summary = pd.DataFrame({

    "Model": [
        "Full",
        "BestSubset",
        "Rank2"
    ],

    "RMSE": [
        rmse_full,
        rmse_best,
        rmse_rank2
    ],

    "R2": [
        r2_full,
        r2_best,
        r2_rank2
    ],

    "Rp": [
        rp_full,
        rp_best,
        rp_rank2
    ],

    
})

print(summary.round(2))

summary.to_csv(
    f"{OUTDIR}/model_summary.csv",
    index=False
)

# ============================================================
# RC EQUATION
# ============================================================

banner("FINAL RC EQUATION")

print(
    f"ΔΔG = {a:.2f} ξ {b:+.2f}"
)

    
# ============================================================
# RC STABILITY
# ============================================================

banner("RC STABILITY")

loo = LeaveOneOut()

w_all = []

for tr, te in loo.split(X_full):

    # --------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(X_full[tr])

    lw = LedoitWolf().fit(Xtr_scaled)

    cov = lw.covariance_

    U, S, _ = np.linalg.svd(cov)

    W = (
        U
        @ np.diag(1.0 / np.sqrt(S + 1e-6))
        @ U.T
    )

    Xtr_w = Xtr_scaled @ W

    # --------------------------------------------------------
    # fit RC direction
    # --------------------------------------------------------

    model = LinearRegression()

    model.fit(
        Xtr_w,
        y_full[tr]
    )

    w = model.coef_

    # normalize direction
    w /= (
        np.linalg.norm(w)
        + 1e-8
    )

    w_all.append(w)

# ------------------------------------------------------------
# aggregate
# ------------------------------------------------------------

w_all = np.array(w_all)

w_mean = np.mean(w_all, axis=0)

w_std = np.std(w_all, axis=0)

rc_stability = pd.DataFrame({

    "feature": features,

    "mean_weight": w_mean,

    "std_weight": w_std,

    "stability_ratio":
        np.abs(w_mean) / (w_std + 1e-8),

    "sign_consistency": [

        max(
            np.mean(w_all[:, i] > 0),
            np.mean(w_all[:, i] < 0)
        )

        for i in range(len(features))
    ]
})

rc_stability = rc_stability.sort_values(
    "stability_ratio",
    ascending=False
)

print(rc_stability.round(2))




# ============================================================
# RC PARAMETER STABILITY
# ============================================================

banner("RC PARAMETER STABILITY")

loo = LeaveOneOut()

a_all = []
b_all = []

for tr, te in loo.split(X_full):

    # --------------------------------------------------------
    # scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(
        X_full[tr]
    )

    Xte_scaled = scaler.transform(
        X_full[te]
    )

    # --------------------------------------------------------
    # whitening
    # --------------------------------------------------------

    lw = LedoitWolf().fit(Xtr_scaled)

    cov = lw.covariance_

    U, S, _ = np.linalg.svd(cov)

    W = (
        U
        @ np.diag(
            1.0 / np.sqrt(S + 1e-6)
        )
        @ U.T
    )

    Xtr_w = Xtr_scaled @ W

    Xte_w = Xte_scaled @ W

    # --------------------------------------------------------
    # RC direction
    # --------------------------------------------------------

    rc_model = LinearRegression()

    rc_model.fit(
        Xtr_w,
        y_full[tr]
    )

    w = rc_model.coef_

    w /= (
        np.linalg.norm(w)
        + 1e-8
    )

    # --------------------------------------------------------
    # RC coordinate
    # --------------------------------------------------------

    xi_tr = Xtr_w @ w

    # --------------------------------------------------------
    # free-energy fit
    # --------------------------------------------------------

    energy_model = LinearRegression()

    energy_model.fit(
        xi_tr.reshape(-1, 1),
        y_full[tr]
    )

    a_all.append(
        energy_model.coef_[0]
    )

    b_all.append(
        energy_model.intercept_
    )

# ============================================================
# SUMMARY
# ============================================================

a_all = np.array(a_all)
b_all = np.array(b_all)

summary_rc = pd.DataFrame({

    "parameter": ["a", "b"],

    "mean": [
        np.mean(a_all),
        np.mean(b_all)
    ],

    "std": [
        np.std(a_all),
        np.std(b_all)
    ],

    "stability_ratio": [

        np.abs(np.mean(a_all))
        / (np.std(a_all) + 1e-8),

        np.abs(np.mean(b_all))
        / (np.std(b_all) + 1e-8)
    ],

    "sign_consistency": [

        max(
            np.mean(a_all > 0),
            np.mean(a_all < 0)
        ),

        max(
            np.mean(b_all > 0),
            np.mean(b_all < 0)
        )
    ]
})

print(summary_rc.round(3))

summary_rc.to_csv(
    f"{OUTDIR}/rc_parameter_stability.csv",
    index=False
)
# ============================================================
# COMPLETE
# ============================================================

banner("PIPELINE COMPLETE")

print(f"[INFO] Outputs saved → {OUTDIR}")