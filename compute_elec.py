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

OUTDIR = "ele_results"
os.makedirs(OUTDIR, exist_ok=True)

PDB_FILE = "WT_amber.pdb"

# ============================================================
# ELECTROSTATIC CONSTANTS
# ============================================================

COULOMB = 332.0
KAPPA = 0.05

EPS_IN = 10.0
EPS_OUT = 80.0

LAMBDA = 4.0

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
structure = parser.get_structure("WT", PDB_FILE)[0]
residues = [
    (res.id[1], res, res["CA"].coord)
    for chain in structure
    for res in chain
    if "CA" in res
]

print(f"[INFO] Total CA residues: {len(residues)}")

# ============================================================
# CALCIUM POSITION
# ============================================================

ca_pos = None

for atom in structure.get_atoms():
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
# NESTED LOOCV FEATURE SELECTION
# ============================================================

banner("NESTED LOOCV FEATURE SELECTION")

outer_loo = LeaveOneOut()

outer_preds = np.zeros(len(y_full))

selected_subsets = []

selected_features_counter = {
    f: 0 for f in features
}

all_outer_results = []

# ============================================================
# OUTER LOOP
# ============================================================

for outer_fold, (tr_outer, te_outer) in enumerate(
    outer_loo.split(X_full),
    start=1
):

    print(
        f"\n[OUTER FOLD {outer_fold:02d}/{len(y_full)}]"
    )

    X_train = X_full[tr_outer]
    y_train = y_full[tr_outer]

    X_test = X_full[te_outer]
    y_test = y_full[te_outer]

    # --------------------------------------------------------
    # INNER FEATURE SEARCH
    # --------------------------------------------------------

    inner_results = []

    for k in range(1, len(features) + 1):

        for subset in itertools.combinations(features, k):

            idx = [
                features.index(f)
                for f in subset
            ]

            rmse, r2, rp, p, stderr = (
                compute_loocv_metrics(
                    X_train[:, idx],
                    y_train
                )
            )

            inner_results.append([
                subset,
                rmse,
                r2,
                rp,
                stderr
            ])

    # --------------------------------------------------------
    # SELECT BEST INNER SUBSET
    # --------------------------------------------------------

    df_inner = pd.DataFrame(
        inner_results,
        columns=[
            "subset",
            "rmse",
            "r2",
            "rp",
            "stderr"
        ]
    )

    df_inner = df_inner.sort_values(
        "rmse"
    )

    best_subset = df_inner.iloc[0]["subset"]

    selected_subsets.append(best_subset)

    print(
        f"[INFO] Best subset: {best_subset}"
    )

    # --------------------------------------------------------
    # FEATURE FREQUENCY
    # --------------------------------------------------------

    for f in best_subset:

        selected_features_counter[f] += 1

    # --------------------------------------------------------
    # TRAIN FINAL MODEL ON OUTER TRAINING SET
    # --------------------------------------------------------

    idx_best = [
        features.index(f)
        for f in best_subset
    ]

    Xtr_best = X_train[:, idx_best]

    Xte_best = X_test[:, idx_best]

    Xtr_w, Xte_w = preprocess_train_test(
        Xtr_best,
        Xte_best
    )

    model = LinearRegression()

    model.fit(
        Xtr_w,
        y_train
    )

    pred = model.predict(Xte_w)[0]

    outer_preds[te_outer] = pred

    all_outer_results.append({

        "fold": outer_fold,

        "test_index": int(te_outer[0]),

        "true_ddG": float(y_test[0]),

        "pred_ddG": float(pred),

        "best_subset": ",".join(best_subset)
    })

# ============================================================
# FINAL NESTED-CV PERFORMANCE
# ============================================================

rmse_nested = np.sqrt(
    mean_squared_error(
        y_full,
        outer_preds
    )
)

r2_nested = r2_score(
    y_full,
    outer_preds
)

rp_nested, rp_p_nested = pearsonr(
    y_full,
    outer_preds
)

stderr_nested = np.std(
    y_full - outer_preds
)

banner("NESTED LOOCV PERFORMANCE")

print(f"RMSE   = {rmse_nested:.2f}")
print(f"R²     = {r2_nested:.2f}")
print(f"Rₚ     = {rp_nested:.2f}")
print(f"StdErr = {stderr_nested:.2f}")

# ============================================================
# SAVE OUTER PREDICTIONS
# ============================================================

df_nested_preds = pd.DataFrame({

    "mutant": df["mutant"],

    "resi": df["resi"],

    "ddG_exp": y_full,

    "ddG_pred_nested": outer_preds
})

df_nested_preds.to_csv(
    f"{OUTDIR}/nested_loocv_predictions.csv",
    index=False
)

# ============================================================
# FEATURE SELECTION STABILITY
# ============================================================

banner("FEATURE SELECTION STABILITY")

selection_freq = pd.DataFrame({

    "feature": list(
        selected_features_counter.keys()
    ),

    "selection_count": list(
        selected_features_counter.values()
    )
})

selection_freq["selection_fraction"] = (

    selection_freq["selection_count"]
    / len(y_full)
)

selection_freq = selection_freq.sort_values(
    "selection_fraction",
    ascending=False
)

print(selection_freq.round(2))

selection_freq.to_csv(
    f"{OUTDIR}/nested_feature_frequency.csv",
    index=False
)

# ============================================================
# CONSENSUS SUBSET
# ============================================================

subset_strings = [
    ",".join(sorted(s))
    for s in selected_subsets
]

subset_counts = pd.Series(
    subset_strings
).value_counts()

print("\n[INFO] Most common subsets:")
print(subset_counts.head())

subset_counts.to_csv(
    f"{OUTDIR}/nested_subset_counts.csv"
)

# ============================================================
# BEST CONSENSUS SUBSET
# ============================================================

consensus_subset = tuple(
    subset_counts.index[0].split(",")
)

print("\n[INFO] Consensus subset:")
print(consensus_subset)


# ============================================================
# DEFINE CONSENSUS FEATURE MATRIX
# ============================================================

best_subset = consensus_subset

idx_best = [
    features.index(f)
    for f in best_subset
]

X_best = X_full[:, idx_best]


# ============================================================
# CONSENSUS SUBSET PERFORMANCE
# ============================================================

(
    rmse_best,
    r2_best,
    rp_best,
    p_best,
    stderr_best
) = compute_loocv_metrics(
    X_best,
    y_full
)

banner("CONSENSUS SUBSET MODEL")

print(f"RMSE   = {rmse_best:.2f}")
print(f"R²     = {r2_best:.2f}")
print(f"Rₚ     = {rp_best:.2f}")
print(f"StdErr = {stderr_best:.2f}")

# ============================================================
# CONSENSUS MODEL
# ============================================================

idx_consensus = [
    features.index(f)
    for f in consensus_subset
]

X_consensus = X_full[:, idx_consensus]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_consensus
)

X_w, W = whiten(X_scaled)

consensus_model = LinearRegression()

consensus_model.fit(
    X_w,
    y_full
)

coef_consensus = consensus_model.coef_

coef_consensus /= (
    np.linalg.norm(coef_consensus)
    + 1e-8
)

df_consensus = pd.DataFrame({

    "feature": consensus_subset,

    "weight": coef_consensus
})

print("\n[INFO] Consensus model weights:")
print(df_consensus.round(3))

df_consensus.to_csv(
    f"{OUTDIR}/consensus_model_weights.csv",
    index=False
)

# ============================================================
# SUMMARY TABLE
# ============================================================

summary_nested = pd.DataFrame({

    "Metric": [
        "RMSE",
        "R2",
        "Rp",
        "StdErr"
    ],

    "Nested_LOOCV": [
        rmse_nested,
        r2_nested,
        rp_nested,
        stderr_nested
    ]
})

summary_nested.to_csv(
    f"{OUTDIR}/nested_summary.csv",
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
# ΔΔG MODEL USING MINIMAL SUBSET ONLY
# ============================================================

banner("ΔG MODEL")

# ------------------------------------------------------------
# minimal subset matrix
# ------------------------------------------------------------

idx_best = [
    features.index(f)
    for f in best_subset
]

X_min = X_full[:, idx_best]

# ------------------------------------------------------------
# preprocessing
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_min)

X_w, W = whiten(X_scaled)

# ------------------------------------------------------------
# reaction coordinate
# ------------------------------------------------------------

rc_model = LinearRegression()

rc_model.fit(X_w, y_full)

w_rc = rc_model.coef_

w_rc /= (
    np.linalg.norm(w_rc)
    + 1e-8
)

phi = X_w @ w_rc


rp_xi, p_xi = pearsonr(phi, y_full)

print(f"Rp(xi, ddG) = {rp_xi:.2f}")

# ------------------------------------------------------------
# ΔΔG model
# ------------------------------------------------------------



loo = LeaveOneOut()
pred = np.zeros(len(y_full))

for tr, te in loo.split(phi):

    model = LinearRegression()

    model.fit(
        phi[tr].reshape(-1, 1),
        y_full[tr]
    )

    pred[te] = model.predict(
        phi[te].reshape(-1, 1)
    )
# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================

n = len(y_full)
p = 2

rss = np.sum((y_full - pred) ** 2)

sigma2 = rss / (n - p)

X_design = np.column_stack([
    phi,
    np.ones(n)
])

cov_beta = sigma2 * np.linalg.inv(
    X_design.T @ X_design
)

se_a = np.sqrt(cov_beta[0, 0])
se_b = np.sqrt(cov_beta[1, 1])

t_a = a / se_a
t_b = b / se_b

from scipy.stats import t

dfree = n - p

p_a = 2 * (
    1 - t.cdf(np.abs(t_a), df=dfree)
)

p_b = 2 * (
    1 - t.cdf(np.abs(t_b), df=dfree)
)

# ------------------------------------------------------------
# confidence intervals
# ------------------------------------------------------------

tcrit = t.ppf(0.975, df=dfree)

ci_a = (
    a - tcrit * se_a,
    a + tcrit * se_a
)

ci_b = (
    b - tcrit * se_b,
    b + tcrit * se_b
)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

rmse_phi = np.sqrt(
    mean_squared_error(y_full, pred)
)

r2_phi = r2_score(y_full, pred)

rp_phi, rp_p = pearsonr(
    y_full,
    pred
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("FREE-ENERGY EQUATION")
print("--------------------------------")

print(
    f"ΔΔG = ({a:.2f})φ {b:+.2f}"
)


print("\nPARAMETER SIGNIFICANCE")
print("----------------------")

summary_stats = pd.DataFrame({

    "parameter": ["a", "b"],

    "estimate": [a, b],

    "std_error": [se_a, se_b],

    "t_stat": [t_a, t_b],

    "p_value": [p_a, p_b],

    "CI_low": [
        ci_a[0],
        ci_b[0]
    ],

    "CI_high": [
        ci_a[1],
        ci_b[1]
    ]
})

print(summary_stats.round(4))

summary_stats.to_csv(
    f"{OUTDIR}/dG_significance.csv",
    index=False
)


# ============================================================
# PARAMETER SENSITIVITY ANALYSIS
# ============================================================

banner("PARAMETER SENSITIVITY ANALYSIS")

# ------------------------------------------------------------
# parameter grids
# ------------------------------------------------------------

KAPPA_GRID = [
    0.05,
    0.08,
    0.10,
    0.12,
    0.15
]

EPS_IN_GRID = [
    2.0,
    4.0,
    6.0,
    8.0,
    10.0
]

LAMBDA_GRID = [
    4.0,
    5.0,
    7.0,
    9.0,
    12.0
]

R_EFF_GRID = [
    1.5,
    2.0,
    2.5,
    3.0
]

# ============================================================
# FEATURE GENERATION UNDER VARIABLE PHYSICS
# ============================================================

def dielectric_scan(r, eps_in, lambda_val):

    return (
        EPS_OUT
        - (EPS_OUT - eps_in)
        * np.exp(-r / lambda_val)
    )

# ------------------------------------------------------------

def screened_coulomb_scan(
    q,
    r,
    kappa_val,
    eps_in,
    lambda_val
):

    return (
        COULOMB
        * q
        * np.exp(-kappa_val * r)
        / (
            dielectric_scan(
                r,
                eps_in,
                lambda_val
            ) * r
        )
    )

# ------------------------------------------------------------

def potential_scan(
    coord_i,
    exclude_index,
    kappa_val,
    eps_in,
    lambda_val
):

    phi = 0.0

    for rid_j, res_j, coord_j in residues:

        if rid_j == exclude_index:
            continue

        aa = aa_map.get(
            res_j.get_resname(),
            None
        )

        if aa is None:
            continue

        q = charge.get(aa, 0)

        if q == 0:
            continue

        r = np.linalg.norm(
            coord_i - coord_j
        )

        if r < 1e-6:
            continue

        phi += screened_coulomb_scan(
            q,
            r,
            kappa_val,
            eps_in,
            lambda_val
        )

    return phi

# ------------------------------------------------------------

def generate_feature_matrix(
    kappa_val,
    eps_in,
    lambda_val,
    r_eff_val
):

    rows_scan = []

    for mut, resi in mutations:

        coord_i = next(
            coord
            for rid, _, coord in residues
            if rid == resi
        )

        wt = mut[0]
        mt = mut[-1]

        dq = (
            charge.get(mt, 0)
            - charge.get(wt, 0)
        )

        # ----------------------------------------------------
        # electrostatic potential
        # ----------------------------------------------------

        phi = potential_scan(
            coord_i,
            resi,
            kappa_val,
            eps_in,
            lambda_val
        )

        # ----------------------------------------------------
        # calcium interaction
        # ----------------------------------------------------

        r_ca = np.linalg.norm(
            coord_i - ca_pos
        )

        e_ca = (
            COULOMB
            * dq
            * CA_CHARGE
            * np.exp(-kappa_val * r_ca)
            / (
                dielectric_scan(
                    r_ca,
                    eps_in,
                    lambda_val
                ) * r_ca
            )
        )

        # ----------------------------------------------------
        # born term
        # ----------------------------------------------------

        born = (
            (dq ** 2)
            * (
                1 / eps_in
                - 1 / EPS_OUT
            )
            / (
                2 * r_eff_val
            )
        )

        rows_scan.append([

            dq * phi,

            e_ca,

            born,

            phi ** 2
        ])

    return np.array(rows_scan)

# ============================================================
# PARAMETER SCAN
# ============================================================

scan_results = []

parameter_combinations = list(
    itertools.product(

        KAPPA_GRID,

        EPS_IN_GRID,

        LAMBDA_GRID,

        R_EFF_GRID
    )
)

print(
    f"[INFO] Total parameter combinations: "
    f"{len(parameter_combinations)}"
)

for (
    kappa_val,
    eps_in,
    lambda_val,
    r_eff_val
) in tqdm(
    parameter_combinations,
    desc="Parameter scan"
):

    # --------------------------------------------------------
    # generate feature matrix
    # --------------------------------------------------------

    X_scan = generate_feature_matrix(

        kappa_val,

        eps_in,

        lambda_val,

        r_eff_val
    )

    # --------------------------------------------------------
    # evaluate reduced physics model
    # --------------------------------------------------------

    rmse_scan, r2_scan, rp_scan, _, stderr_scan = (
        compute_loocv_metrics(
            X_scan,
            y_full
        )
    )

    scan_results.append({

        "kappa": kappa_val,

        "eps_in": eps_in,

        "lambda": lambda_val,

        "r_eff": r_eff_val,

        "rmse": rmse_scan,

        "r2": r2_scan,

        "rp": rp_scan,

        "stderr": stderr_scan
    })

# ============================================================
# RESULTS DATAFRAME
# ============================================================

df_scan = pd.DataFrame(
    scan_results
)

df_scan = df_scan.sort_values(
    "rmse"
)

# ============================================================
# SAVE RESULTS
# ============================================================

df_scan.to_csv(
    f"{OUTDIR}/parameter_scan.csv",
    index=False
)

# ============================================================
# BEST PARAMETER SETS
# ============================================================

banner("BEST PARAMETER SETS")

print(
    df_scan.head(10).round(3)
)

# ============================================================
# PARAMETER ROBUSTNESS
# ============================================================

banner("PARAMETER ROBUSTNESS")

robustness = {

    "RMSE_mean":
        df_scan["rmse"].mean(),

    "RMSE_std":
        df_scan["rmse"].std(),

    "Rp_mean":
        df_scan["rp"].mean(),

    "Rp_std":
        df_scan["rp"].std(),

    "R2_mean":
        df_scan["r2"].mean(),

    "R2_std":
        df_scan["r2"].std()
}

for k, v in robustness.items():

    print(f"{k}: {v:.3f}")

# ============================================================
# TOP PARAMETER FREQUENCY
# ============================================================

banner("TOP PARAMETER FREQUENCY")

top_fraction = int(
    0.10 * len(df_scan)
)

df_top = df_scan.head(top_fraction)

for param in [
    "kappa",
    "eps_in",
    "lambda",
    "r_eff"
]:

    print(f"\n[{param}]")

    freq = (
        df_top[param]
        .value_counts(normalize=True)
        .sort_index()
    )

    print(freq.round(3))

# ============================================================
# SUMMARY EXPORT
# ============================================================

robustness_df = pd.DataFrame([robustness])

robustness_df.to_csv(
    f"{OUTDIR}/parameter_robustness.csv",
    index=False
)

print("\n[INFO] Parameter scan complete.")

# ============================================================
# ELECTROSTATIC RESIDUE INTERPRETATION
# ============================================================

def classify_electrostatic_sector(
    e_ca,
    phi_sq,
    e_perp,
    ddg,
    distance_to_ca
):

    # --------------------------------------------
    # coordination electrostatic disruption
    # --------------------------------------------

    if (
        distance_to_ca < 8.0
        and ddg > 0.5
    ):

        return "coordination electrostatic disruption"

    # --------------------------------------------
    # directional field frustration
    # --------------------------------------------

    if (
        abs(e_perp) > np.percentile(
            np.abs(df["E_perp"]),
            75
        )
        and ddg > 0
    ):

        return "directional field frustration"

    # --------------------------------------------
    # dielectric compensation
    # --------------------------------------------

    if (
        ddg < -0.5
        and phi_sq > np.median(df["phi_sq"])
    ):

        return "dielectric compensation"

    # --------------------------------------------
    # weak perturbation
    # --------------------------------------------

    return "weak electrostatic perturbation"


# ============================================================

def generate_electrostatic_interpretation_table():

    print(
        "\n[INFO] Generating electrostatic "
        "interpretation table..."
    )

    # --------------------------------------------------------
    # reaction coordinate projection
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_best
    )

    X_w, W = whiten(X_scaled)

    phi_rc = X_w @ w_rc

    rows = []

    for i, row in df.iterrows():

        mutant = row["mutant"]

        resi = row["resi"]

        coord_i = next(
            coord
            for rid, _, coord in residues
            if rid == resi
        )

        distance_to_ca = np.linalg.norm(
            coord_i - ca_pos
        )

        sector = classify_electrostatic_sector(

            row["E_ca"],

            row["phi_sq"],

            row["E_perp"],

            row["ddG_exp"],

            distance_to_ca
        )

        # ----------------------------------------------------
        # INTERPRETATION TEXT
        # ----------------------------------------------------

        if sector == (
            "coordination electrostatic disruption"
        ):

            interpretation = (
                "Disruption of Ca-coupled "
                "electrostatic stabilization "
                "through coordination-shell "
                "field perturbation"
            )

        elif sector == (
            "directional field frustration"
        ):

            interpretation = (
                "Strong anisotropic field "
                "redistribution associated "
                "with electrostatic frustration"
            )

        elif sector == (
            "dielectric compensation"
        ):

            interpretation = (
                "Favorable dielectric "
                "reorganization and "
                "electrostatic compensation"
            )

        else:

            interpretation = (
                "Weak electrostatic perturbation "
                "with limited field reorganization"
            )

        rows.append({

            "mutant": mutant,

            "ddG_exp": row["ddG_exp"],

            "RC_projection": phi_rc[i],

            "E_ca": row["E_ca"],

            "phi_sq": row["phi_sq"],

            "E_parallel": row["E_parallel"],

            "E_perp": row["E_perp"],

            "dq_phi": row["dq_phi"],

            "distance_to_Ca": distance_to_ca,

            "electrostatic_sector": sector,

            "interpretation": interpretation
        })

    df_out = pd.DataFrame(rows)

    # --------------------------------------------------------
    # rank ordering
    # --------------------------------------------------------

    df_out = df_out.sort_values(
        by="RC_projection",
        ascending=False
    )

    # --------------------------------------------------------
    # RC extremeness
    # --------------------------------------------------------

    q_low = df_out["RC_projection"].quantile(0.25)

    q_high = df_out["RC_projection"].quantile(0.75)

    df_out["extreme_class"] = "middle"

    df_out.loc[
        df_out["RC_projection"] >= q_high,
        "extreme_class"
    ] = "electrostatic_destabilizing_extreme"

    df_out.loc[
        df_out["RC_projection"] <= q_low,
        "extreme_class"
    ] = "electrostatic_stabilizing_extreme"

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    df_out.to_csv(

        f"{OUTDIR}/electrostatic_residue_interpretation.csv",

        index=False
    )

    print(
        "[INFO] Written: "
        "electrostatic_residue_interpretation.csv"
    )

    return df_out

generate_electrostatic_interpretation_table()
# ============================================================
# ATOM-WISE ELECTROSTATIC COMPARISON (MINIMAL, FAST)
# ============================================================

banner("ATOM-WISE FEATURE GENERATION (MINIMAL MODEL)")

# ------------------------------------------------------------
# simple atom-wise charge model (heuristic, no force-field)
# ------------------------------------------------------------

def atom_charge(atom):
    name = atom.get_name().strip()
    element = atom.element.strip() if atom.element else ""

    # backbone
    if name == "N":
        return +0.3
    if name == "O":
        return -0.5

    # side-chain heuristics
    if element == "O":
        return -0.5
    if element == "N":
        return +0.3
    if element == "S":
        return -0.2

    return 0.0



# ------------------------------------------------------------
# atom list (heavy atoms only)
# ------------------------------------------------------------

atoms = [
    atom for atom in structure.get_atoms()
    if atom.element != "H"
]

print(f"[INFO] Total heavy atoms: {len(atoms)}")


# ------------------------------------------------------------
# atom-wise potential
# ------------------------------------------------------------

def potential_atomwise(coord_i, exclude_resi):

    phi = 0.0

    for atom_j in atoms:

        res_j = atom_j.get_parent()
        rid_j = res_j.id[1]

        if rid_j == exclude_resi:
            continue

        q = atom_charge(atom_j)

        if q == 0:
            continue

        r = np.linalg.norm(coord_i - atom_j.coord)

        if r < 1e-6:
            continue

        phi += (
            COULOMB
            * q
            * np.exp(-KAPPA * r)
            / (dielectric(r) * r)
        )

    return phi


# ------------------------------------------------------------
# atom-wise feature generation
# ------------------------------------------------------------

rows_atom = []

for mut, resi in tqdm(mutations, desc="Atom-wise features"):

    coord_i = next(
        coord
        for rid, _, coord in residues
        if rid == resi
    )

    wt = mut[0]
    mt = mut[-1]

    dq = charge.get(mt, 0) - charge.get(wt, 0)

    # atom-wise electrostatic potential
    phi_atom = potential_atomwise(coord_i, resi)

    # Ca interaction (same definition)
    r_ca = np.linalg.norm(coord_i - ca_pos)

    E_ca_atom = (
        COULOMB
        * dq
        * CA_CHARGE
        * np.exp(-KAPPA * r_ca)
        / (dielectric(r_ca) * r_ca)
    )

    # Born term (same)
    born_atom = (
        (dq ** 2)
        * (1 / EPS_IN - 1 / EPS_OUT)
        / (2 * R_EFF)
    )

    rows_atom.append([
        dq * phi_atom,
        E_ca_atom,
        born_atom,
        phi_atom ** 2,
        exp_data[mut] - exp_data["WT"]
    ])


# ------------------------------------------------------------
# dataframe
# ------------------------------------------------------------

df_atom = pd.DataFrame(rows_atom, columns=[
    "dq_phi",
    "E_ca",
    "born",
    "phi_sq",
    "ddG_exp"
])

X_atom = df_atom[["dq_phi", "E_ca", "born", "phi_sq"]].values
y_atom = df_atom["ddG_exp"].values


# ------------------------------------------------------------
# evaluation (same pipeline)
# ------------------------------------------------------------

rmse_atom, r2_atom, rp_atom, _, stderr_atom = compute_loocv_metrics(
    X_atom,
    y_atom
)

banner("ATOM-WISE MODEL PERFORMANCE")

print(f"RMSE   = {rmse_atom:.2f}")
print(f"R²     = {r2_atom:.2f}")
print(f"Rₚ     = {rp_atom:.2f}")
print(f"StdErr = {stderr_atom:.2f}")


# ------------------------------------------------------------
# comparison table (LaTeX-ready)
# ------------------------------------------------------------

comparison_df = pd.DataFrame({

    "Model": ["Residue-level", "Atom-wise"],

    "RMSE": [rmse_full, rmse_atom],

    "R2": [r2_full, r2_atom],

    "Rp": [rp_full, rp_atom]
})

print("\nCOMPARISON TABLE")
print(comparison_df.round(2))

comparison_df.to_csv(
    f"{OUTDIR}/residue_vs_atomwise_comparison.csv",
    index=False
)

banner("PIPELINE COMPLETE")



