# ==============================================================
# SELF-CONTAINED REACTION COORDINATE PIPELINE
# ==============================================================
#
# Features
# --------
# 1. WT-only ANM
# 2. Fixed optimal ANM parameters
# 3. Reaction-coordinate construction
# 4. Nested LOOCV
# 5. Publication-grade metrics
# 6. RC coefficients
# 7. Saves RC table
#
# ==============================================================

import re
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from prody import *

from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau
)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

from sklearn.covariance import LedoitWolf
warnings.filterwarnings("ignore")

np.random.seed(42)

# ==============================================================
# OPTIMAL PARAMETERS
# ==============================================================

CONTACT_CUTOFF = 9.0
CORR_THRESHOLD = 0.05
N_MODES = 20
LOCAL_RADIUS = 8.0

# ==============================================================
# EXPERIMENTAL DATA
# ==============================================================

exp_data = {

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

# ==============================================================
# RC FEATURES
# ==============================================================

RC_FEATURES = [

    "comm_eff",
    "f_collective",
    "f_entropy",
    "prs_sens"

]

# ==============================================================
# UTILITIES
# ==============================================================

def extract_mut_index(name):

    return int(
        re.findall(r"\d+", name)[0]
    )

def regression_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    rp = pearsonr(
        y_true,
        y_pred
    )[0]

    rs = spearmanr(
        y_true,
        y_pred
    )[0]

    kt = kendalltau(
        y_true,
        y_pred
    )[0]

    return rmse, r2, rp, rs, kt


# ==============================================================
# WHITENING
# ==============================================================

def whiten_train_test(

    Xtr,
    Xte

):

    # ----------------------------------------------------------
    # STANDARDIZATION
    # ----------------------------------------------------------

    scaler = StandardScaler()

    Xtr_s = scaler.fit_transform(
        Xtr
    )

    Xte_s = scaler.transform(
        Xte
    )

    # ----------------------------------------------------------
    # SHRINKAGE COVARIANCE
    # ----------------------------------------------------------

    lw = LedoitWolf().fit(
        Xtr_s
    )

    cov = lw.covariance_

    # ----------------------------------------------------------
    # SVD WHITENING
    # ----------------------------------------------------------

    U, S, _ = np.linalg.svd(
        cov
    )

    W = (

        U
        @ np.diag(
            1.0 / np.sqrt(S + 1e-8)
        )
        @ U.T

    )

    # ----------------------------------------------------------
    # WHITEN
    # ----------------------------------------------------------

    Xtr_w = Xtr_s @ W

    Xte_w = Xte_s @ W

    return (

        Xtr_w,
        Xte_w,
        scaler,
        W

    )
# ==============================================================
# GRAPH
# ==============================================================

def build_graph(cc):

    G = nx.Graph()

    n = cc.shape[0]

    G.add_nodes_from(range(n))

    for i in range(n):

        for j in range(i + 1, n):

            corr = abs(cc[i, j])

            if corr > CORR_THRESHOLD:

                G.add_edge(
                    i,
                    j,
                    weight=corr
                )

    return G

# ==============================================================
# ORTHOGONAL FEATURES
# ==============================================================

def orthogonal_features(

    anm,
    mut_index,
    neighbors,
    cc

):

    eigvecs = anm.getEigvecs()

    eigvals = anm.getEigvals()

    n_res = eigvecs.shape[0] // 3

    vecs = eigvecs.reshape(
        (n_res, 3, -1)
    )

    # ----------------------------------------------------------
    # Collective motion
    # ----------------------------------------------------------

    low = vecs[
        mut_index,
        :,
        :5
    ]

    low_amp = np.linalg.norm(
        low,
        axis=0
    )

    f_collective = np.mean(
        low_amp
    )

    # ----------------------------------------------------------
    # Entropy
    # ----------------------------------------------------------

    power = (
        low_amp ** 2
    ) / (
        eigvals[:5] + 1e-8
    )

    denom = np.sum(power)

    if denom < 1e-12:

        f_entropy = 0.0

    else:

        power /= denom

        f_entropy = -np.sum(
            power * np.log(
                power + 1e-12
            )
        )

    # ----------------------------------------------------------
    # Correlation asymmetry
    # ----------------------------------------------------------

    local_corr = np.abs(
        cc[
            mut_index,
            neighbors
        ]
    )

    f_asym = np.std(
        local_corr
    )

    return (
        f_collective,
        f_entropy,
        f_asym
    )

# ==============================================================
# FEATURE EXTRACTION
# ==============================================================



def extract_features(

    structure,
    anm,
    cc,
    sens,
    eff,
    mutant

):
    ca_atoms = structure.select(
        "protein and name CA"
    )

    coords = ca_atoms.getCoords()

    resnums = ca_atoms.getResnums()

    # ----------------------------------------------------------
    # Calcium ion
    # ----------------------------------------------------------

    ion = structure.select(
        "resname CAL or element CA and not protein"
    )

    ca_coord = ion.getCoords()[0]

    dist_to_ca = np.linalg.norm(
        coords - ca_coord,
        axis=1
    )

    ca_index = np.argmin(
        dist_to_ca
    )

    # ----------------------------------------------------------
    # Mutation index
    # ----------------------------------------------------------

    mut_resnum = extract_mut_index(
        mutant
    )

    mut_index = np.where(
        resnums == mut_resnum
    )[0][0]

    # ----------------------------------------------------------
    # Local neighbors
    # ----------------------------------------------------------

    mut_coord = coords[mut_index]

    dist_mut = np.linalg.norm(
        coords - mut_coord,
        axis=1
    )

    neighbors = np.where(
        dist_mut < LOCAL_RADIUS
    )[0]

    # ----------------------------------------------------------
    # Graph
    # ----------------------------------------------------------

    G = build_graph(cc)

    # ----------------------------------------------------------
    # Communication efficiency
    # ----------------------------------------------------------

    try:

        path_len = nx.shortest_path_length(

            G,

            mut_index,

            ca_index,

            weight=lambda u, v, d:
            1 / d["weight"]

        )

        comm_eff = 1.0 / path_len

    except:

        comm_eff = 0.0

    # ----------------------------------------------------------
    # Orthogonal features
    # ----------------------------------------------------------

    (
        f_collective,
        f_entropy,
        f_asym

    ) = orthogonal_features(

        anm,
        mut_index,
        neighbors,
        cc

    )

    prs_sens = sens[mut_index]

    features = {

        "comm_eff": comm_eff,
        "f_collective": f_collective,
        "f_asym": f_asym,
        "prs_sens": prs_sens,
        "f_entropy": f_entropy

    }
    # ----------------------------------------------------------
    # Additional descriptors
    # ----------------------------------------------------------

    mut_ca_coupling = cc[
        mut_index,
        ca_index
    ]

    prs_eff = eff[mut_index]

    # ----------------------------------------------------------
    # Local motion
    # ----------------------------------------------------------

    eigvecs = anm.getEigvecs()

    n_res = eigvecs.shape[0] // 3

    vecs = eigvecs.reshape(
        (n_res, 3, -1)
    )

    high = vecs[
        mut_index,
        :,
        -5:
    ]

    high_amp = np.linalg.norm(
        high,
        axis=0
    )

    f_local = np.mean(
        high_amp
    )

    # ----------------------------------------------------------
    # Square fluctuations
    # ----------------------------------------------------------

    sq_fluct = calcSqFlucts(
        anm
    )[mut_index]

    # ----------------------------------------------------------
    # Store all descriptors
    # ----------------------------------------------------------

    features = {

        "comm_eff": comm_eff,

        "f_collective": f_collective,

        "f_asym": f_asym,

        "prs_sens": prs_sens,

        "f_entropy": f_entropy,

        "mut_ca_coupling": mut_ca_coupling,

        "prs_eff": prs_eff,

        "f_local": f_local,

        "sq_fluct": sq_fluct

    }

    return features

# ==============================================================
# BUILD WT ANM
# ==============================================================

print("\n================================================")
print("BUILDING WT ANM")
print("================================================")

structure = parsePDB(
    "WT_amber.pdb"
)

ca_atoms = structure.select(
    "protein and name CA"
)

anm = ANM("WT_ANM")

anm.buildHessian(
    ca_atoms,
    cutoff=CONTACT_CUTOFF
)

anm.calcModes(
    n_modes=N_MODES
)

print("[INFO] Computing correlations")

cc = calcCrossCorr(
    anm
)

print("[INFO] Computing PRS")

prs, eff, sens = calcPerturbResponse(
    anm
)

# ==============================================================
# BUILD DATASET
# ==============================================================

print("\n================================================")
print("EXTRACTING FEATURES")
print("================================================")

records = []
all_features = []

for mutant, dg in tqdm(
    exp_data.items()
):

    feats = extract_features(

        structure=structure,

        anm=anm,

        cc=cc,

        sens=sens,

        eff=eff,

        mutant=mutant

    )

    record = {

        "mutation": mutant,

        "dg": dg

    }

    record.update(feats)

    records.append(
        record
    )

    all_features.append(
        feats
    )

df = pd.DataFrame(
    records
)

# ==============================================================
# FEATURE MATRIX
# ==============================================================

X = df[
    RC_FEATURES
].values

WT_DG = -6.56  # example value

df["ddg"] = df["dg"] - WT_DG

y = df["ddg"].values
# ==============================================================
# NESTED LOOCV
# ==============================================================

print("\n================================================")
print("LOOCV WITH FIXED HYPERPARAMETERS")
print("================================================")

loo = LeaveOneOut()

pred_all = []

true_all = []

coef_all = []

for train_idx, test_idx in tqdm(

    loo.split(X),

    total=len(X),

    desc="Outer LOOCV"

):

    X_train = X[train_idx]

    y_train = y[train_idx]

    X_test = X[test_idx]

    y_test = y[test_idx]


    # ----------------------------------------------------------
    # WHITENING
    # ----------------------------------------------------------

    Xtr_w, Xte_w, scaler, W = whiten_train_test(

        X_train,
        X_test

    )

    # ----------------------------------------------------------
    # LINEAR REGRESSION
    # ----------------------------------------------------------

    model = LinearRegression()

    model.fit(

        Xtr_w,
        y_train

    )

    pred = model.predict(
        Xte_w
    )

    pred_all.append(
        pred[0]
    )

    true_all.append(
        y_test[0]
    )

    # ----------------------------------------------------------
    # Coefficients
    # ----------------------------------------------------------

    coef = model.coef_

    coef = coef / (
        np.linalg.norm(coef) + 1e-12
    )

    coef_all.append(
        coef
    )









# ==============================================================
# FINAL MODEL
# ==============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# --------------------------------------------------------------
# SHRINKAGE WHITENING
# --------------------------------------------------------------

lw = LedoitWolf().fit(
    X_scaled
)

cov = lw.covariance_

U, S, _ = np.linalg.svd(
    cov
)

W = (

    U
    @ np.diag(
        1.0 / np.sqrt(S + 1e-8)
    )
    @ U.T

)

# --------------------------------------------------------------
# WHITENED FEATURES
# --------------------------------------------------------------

X_white = X_scaled @ W

# --------------------------------------------------------------
# LINEAR REGRESSION
# --------------------------------------------------------------

final_model = LinearRegression()

final_model.fit(
    X_white,
    y
)

# --------------------------------------------------------------
# NORMALIZED RC VECTOR
# --------------------------------------------------------------

final_coef = final_model.coef_

final_coef = (

    final_coef
    / (
        np.linalg.norm(
            final_coef
        ) + 1e-12
    )

)

# --------------------------------------------------------------
# REACTION COORDINATE
# --------------------------------------------------------------

RC = np.dot(
    X_white,
    final_coef
)


# ==============================================================
# BACK-TRANSFORM RC VECTOR
# ==============================================================

final_coef_orig = np.linalg.pinv(W) @ final_coef
final_coef_orig = (

    final_coef_orig
    / (
        np.linalg.norm(
            final_coef_orig
        ) + 1e-12
    )

)

for feat, coef in zip(

    RC_FEATURES,
    final_coef_orig

):
    print(
        f"{feat:20s} : {coef:.2f}"
    )

# ==============================================================
# METRICS
# ==============================================================

pred_all = np.array(
    pred_all
)

true_all = np.array(
    true_all
)

rmse, r2, rp, rs, kt = regression_metrics(

    true_all,
    pred_all

)

# ==============================================================
# PRINT METRICS
# ==============================================================

print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

print(f"RMSE         = {rmse:.2f}")
print(f"R²           = {r2:.2f}")
print(f"Pearson r    = {rp:.2f}")
print(f"Spearman ρ   = {rs:.2f}")
print(f"Kendall τ    = {kt:.2f}")

# ==============================================================
# SAVE RESULTS
# ==============================================================

out = pd.DataFrame({

    "mutation": df["mutation"],

    "ddg_exp": y,

    "ddg_pred": pred_all,

    "RC": RC

})

for feat in RC_FEATURES:

    out[feat] = df[feat]

out.to_csv(
    "anm_reaction_coordinate.csv",
    index=False
)

# ==============================================================
# SAVE COEFFICIENTS
# ==============================================================

coef_df = pd.DataFrame({

    "feature": RC_FEATURES,

    "coefficient": final_coef_orig
})

coef_df.to_csv(
    "anm_rc_coefficients.csv",
    index=False
)

print("\n[INFO] Files saved")

print(
    " - reaction_coordinate.csv"
)

print(
    " - rc_coefficients.csv"
)



loo = LeaveOneOut()

dg_pred_rc = []
dg_true_rc = []

xi_all = []

for train_idx, test_idx in loo.split(X):

    # ----------------------------------------------------------
    # TRAIN / TEST
    # ----------------------------------------------------------

    Xtr = X[train_idx]
    Xte = X[test_idx]

    ytr = y[train_idx]
    yte = y[test_idx]

    # ----------------------------------------------------------
    # WHITENING
    # ----------------------------------------------------------

    Xtr_w, Xte_w, scaler, W_fold = whiten_train_test(

        Xtr,
        Xte

    )

    # ----------------------------------------------------------
    # TRAIN RC MODEL
    # ----------------------------------------------------------

    rc_train_model = LinearRegression()

    rc_train_model.fit(
        Xtr_w,
        ytr
    )

    # ----------------------------------------------------------
    # NORMALIZED RC VECTOR
    # ----------------------------------------------------------

    w_fold = rc_train_model.coef_

    w_fold = (

        w_fold
        / (
            np.linalg.norm(w_fold)
            + 1e-12
        )

    )

    # ----------------------------------------------------------
    # TRAIN RC
    # ----------------------------------------------------------

    xi_train = np.dot(
        Xtr_w,
        w_fold
    )

    xi_test = np.dot(
        Xte_w,
        w_fold
    )

    # ----------------------------------------------------------
    # RC -> ΔG REGRESSION
    # ----------------------------------------------------------

    dg_model = LinearRegression()

    dg_model.fit(
        xi_train.reshape(-1, 1),
        ytr
    )

    pred = dg_model.predict(
        xi_test.reshape(-1, 1)
    )[0]

    dg_pred_rc.append(pred)

    dg_true_rc.append(yte[0])

    xi_all.append(xi_test[0])

# ==============================================================
# FINAL METRICS
# ==============================================================

dg_pred_rc = np.array(dg_pred_rc)
dg_true_rc = np.array(dg_true_rc)

rmse_rc, r2_rc, rp_rc, rs_rc, kt_rc = regression_metrics(

    dg_true_rc,
    dg_pred_rc

)


# ==============================================================
# FINAL INTERPRETIVE RC MODEL
# ==============================================================

print("\n================================================")
print("FINAL INTERPRETIVE RC MODEL")
print("================================================")

# --------------------------------------------------------------
# Full-data interpretive model
# --------------------------------------------------------------

scaler_full = StandardScaler()

X_scaled = scaler_full.fit_transform(X)

lw_full = LedoitWolf().fit(X_scaled)

cov_full = lw_full.covariance_

U, S, _ = np.linalg.svd(cov_full)

W_full = (

    U
    @ np.diag(
        1.0 / np.sqrt(S + 1e-8)
    )
    @ U.T

)

X_white_full = X_scaled @ W_full

final_model = LinearRegression()

final_model.fit(
    X_white_full,
    y
)

final_coef = final_model.coef_

final_coef = (

    final_coef
    / (
        np.linalg.norm(final_coef)
        + 1e-12
    )

)

# --------------------------------------------------------------
# Back-transform
# --------------------------------------------------------------

final_coef_orig = np.linalg.pinv(W_full) @ final_coef

final_coef_orig = (

    final_coef_orig
    / (
        np.linalg.norm(final_coef_orig)
        + 1e-12
    )

)

# --------------------------------------------------------------
# Final RC equation
# --------------------------------------------------------------

print("\nREACTION COORDINATE:")

for feat, coef in zip(

    RC_FEATURES,
    final_coef_orig

):

    print(
        f"{feat:20s} : {coef:.2f}"
    )

# ==============================================================
# SAVE RC RESULTS
# ==============================================================

rc_df = pd.DataFrame({

    "mutation": df["mutation"],

    "dgd_exp": dg_true_rc,

    "dg_pred_rc": dg_pred_rc,

    "reaction_coordinate": xi_all

})

for feat in RC_FEATURES:

    rc_df[feat] = df[feat]

rc_df.to_csv(

    "anm_reaction_coordinate_rank1.csv",

    index=False

)

# ==============================================================
# SAVE COEFFICIENTS
# ==============================================================

coef_df = pd.DataFrame({

    "feature": RC_FEATURES,

    "coefficient": final_coef_orig

})

coef_df.to_csv(

    "anm_reaction_coordinate_coefficients.csv",

    index=False

)

print("\n[INFO] Files saved")

print(
    " - reaction_coordinate_rank1.csv"
)

print(
    " - reaction_coordinate_coefficients.csv"
)

# ==============================================================
# SAVE RAW FEATURE MATRIX
# ==============================================================

raw_feature_df = pd.DataFrame({

    "mutation": df["mutation"],

    "ddg_exp": y,

    "comm_eff": df["comm_eff"],

    "f_collective": df["f_collective"],

    "f_asym": df["f_asym"],

    "prs_sens": df["prs_sens"],

    "f_entropy": df["f_entropy"]

})

# ==============================================================
# SAVE
# ==============================================================

raw_feature_df.to_csv(
    "anm_raw_feature_matrix.csv",
    index=False
)

# ==============================================================
# PRINT
# ==============================================================

print("\n================================================")
print("RAW FEATURE MATRIX")
print("================================================")

print(
    raw_feature_df.head()
)

print("\n[INFO] File saved")

print(
    " - raw_feature_matrix.csv"
)


# ==============================================================
# SAVE ALL RAW FEATURES AT BEST PARAMETERS
# ==============================================================

all_feature_df = pd.DataFrame({

    "mutation": df["mutation"],

    "ddg_exp": y,

    # ----------------------------------------------------------
    # Core communication features
    # ----------------------------------------------------------

    "comm_eff": df["comm_eff"],

    "prs_sens": df["prs_sens"],

    "f_collective": df["f_collective"],

    "f_asym": df["f_asym"],

    "f_entropy": df["f_entropy"],

    # ----------------------------------------------------------
    # Additional descriptors
    # ----------------------------------------------------------

    "mut_ca_coupling": [
        feat["mut_ca_coupling"]
        for feat in all_features
    ],

    "prs_eff": [
        feat["prs_eff"]
        for feat in all_features
    ],

    "f_local": [
        feat["f_local"]
        for feat in all_features
    ],

    "sq_fluct": [
        feat["sq_fluct"]
        for feat in all_features
    ]

})

# ==============================================================
# SAVE ALL RAW FEATURES
# ==============================================================

all_feature_df = pd.DataFrame({

    "mutation": df["mutation"],

    "ddg_exp": y,

    "comm_eff": [
        feat["comm_eff"]
        for feat in all_features
    ],

    "f_collective": [
        feat["f_collective"]
        for feat in all_features
    ],

    "f_asym": [
        feat["f_asym"]
        for feat in all_features
    ],

    "prs_sens": [
        feat["prs_sens"]
        for feat in all_features
    ],

    "f_entropy": [
        feat["f_entropy"]
        for feat in all_features
    ],

    "mut_ca_coupling": [
        feat["mut_ca_coupling"]
        for feat in all_features
    ],

    "prs_eff": [
        feat["prs_eff"]
        for feat in all_features
    ],

    "f_local": [
        feat["f_local"]
        for feat in all_features
    ],

    "sq_fluct": [
        feat["sq_fluct"]
        for feat in all_features
    ]

})

# ==============================================================
# SAVE
# ==============================================================

all_feature_df.to_csv(

    "anm_all_raw_features_best_parameters.csv",

    index=False

)

# ==============================================================
# PRINT
# ==============================================================

print("\n================================================")
print("ALL RAW FEATURES")
print("================================================")

print(
    all_feature_df.head()
)

print("\n================================================")
print("FEATURE SUMMARY")
print("================================================")

print(
    all_feature_df.describe()
)

print("\n[INFO] File saved")

print(
    " - all_raw_features_best_parameters.csv"
)















# ==============================================================
# STATISTICAL SIGNIFICANCE + COEFFICIENT STABILITY
# ==============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.stats import (
    t,
    pearsonr
)

# ==============================================================
# INPUT
# ==============================================================

RC_FEATURES = [

    "comm_eff",
    "f_collective",
    "f_entropy",
    "prs_sens"

]

X = df[RC_FEATURES].values
y = df["ddg"].values


# ==============================================================
# STANDARDIZATION
# ==============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# ==============================================================
# SHRINKAGE WHITENING
# ==============================================================

lw = LedoitWolf().fit(
    X_scaled
)

cov = lw.covariance_

U, S, _ = np.linalg.svd(
    cov
)

W = (

    U
    @ np.diag(
        1.0 / np.sqrt(S + 1e-8)
    )
    @ U.T

)

X_white = X_scaled @ W

# ==============================================================
# FINAL LINEAR MODEL
# ==============================================================

final_model = LinearRegression()

final_model.fit(
    X_white,
    y
)



# ==============================================================
# NORMALIZED RC VECTOR
# ==============================================================

w = final_model.coef_

w = (

    w
    / (
        np.linalg.norm(w) + 1e-12
    )

)

# ==============================================================
# REACTION COORDINATE
# ==============================================================

xi = np.dot(
    X_white,
    w
)

# ==============================================================
# FREE ENERGY REGRESSION
# ==============================================================

rc_model = LinearRegression()

xi_reshape = xi.reshape(-1, 1)

rc_model.fit(
    xi_reshape,
    y
)

dg_pred = rc_model.predict(
    xi_reshape
)

# ==============================================================
# REGRESSION PARAMETERS
# ==============================================================

a = rc_model.coef_[0]

b = rc_model.intercept_

n = len(y)

rss = np.sum(
    (y - dg_pred) ** 2
)

x_mean = np.mean(xi)

ssx = np.sum(
    (xi - x_mean) ** 2
)

sigma2 = rss / (n - 2)

se_a = np.sqrt(
    sigma2 / ssx
)

t_a = a / (
    se_a + 1e-12
)

p_a = 2 * (
    1 - t.cdf(
        np.abs(t_a),
        df=n - 2
    )
)

# --------------------------------------------------------------
# Intercept significance
# --------------------------------------------------------------

se_b = np.sqrt(

    sigma2 * (
        1/n +
        x_mean**2 / ssx
    )

)

t_b = b / (
    se_b + 1e-12
)

p_b = 2 * (

    1 - t.cdf(
        np.abs(t_b),
        df=n - 2
    )

)

# ==============================================================
# MODEL CORRELATION
# ==============================================================

rp = pearsonr(
    y,
    dg_pred
)[0]


terms = []

for feat, coef in zip(

    RC_FEATURES,
    final_coef_orig

):

    terms.append(
        f"({coef:.2f})·{feat}"
    )

eq = " + ".join(terms)

print(f"\nξ = {eq}")
# ==============================================================
# PRINT FREE ENERGY MODEL
# ==============================================================

print("\n================================================")
print("FREE ENERGY MODEL")
print("================================================")

print(

    f"\nΔG_pred = "
    f"{a:.2f} · ξ "
    f"+ {b:.2f}"

)

# ==============================================================
# PRINT STATISTICAL SIGNIFICANCE
# ==============================================================

print("\n================================================")
print("FREE ENERGY REGRESSION SIGNIFICANCE")
print("================================================")

print(f"Slope (a)              = {a:.2f}")
print(f"Slope Std Error        = {se_a:.2f}")
print(f"Slope t-statistic      = {t_a:.2f}")
print(f"Slope p-value          = {p_a:.3e}")

print()

print(f"Intercept (b)          = {b:.2f}")
print(f"Intercept Std Error    = {se_b:.2f}")
print(f"Intercept t-statistic  = {t_b:.2f}")
print(f"Intercept p-value      = {p_b:.3e}")

print()

print(f"Pearson r              = {rp:.2f}")




# ==============================================================
# MINIMAL SUBSET RC STABILITY
# ADD THIS AFTER FINAL RC MODEL SECTION
# ==============================================================

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

# ==============================================================
# BANNER
# ==============================================================

def banner(txt):

    print("\n" + "=" * 60)
    print(txt)
    print("=" * 60)

# ==============================================================
# PREPROCESS
# ==============================================================



# ==============================================================
# MINIMAL FEATURE SUBSET
# ==============================================================

best_subset = RC_FEATURES

X_min = df[
    best_subset
].values

y_full = y.copy()

# ==============================================================
# RC STABILITY
# ==============================================================

banner(
    "MINIMAL SUBSET RC STABILITY"
)

loo = LeaveOneOut()

w_all = []

# --------------------------------------------------------------
# LOOCV COEFFICIENTS
# --------------------------------------------------------------

for tr, te in loo.split(X_min):

    # ----------------------------------------------------------
    # TRAIN / TEST
    # ----------------------------------------------------------

    Xtr_w, Xte_w, _, W_fold = whiten_train_test(

        X_min[tr],
        X_min[te]

    )

    # ----------------------------------------------------------
    # LINEAR MODEL
    # ----------------------------------------------------------

    model = LinearRegression()

    model.fit(

        Xtr_w,
        y_full[tr]

    )

    # ----------------------------------------------------------
    # NORMALIZED RC VECTOR
    # ----------------------------------------------------------

    w = model.coef_

    w = np.linalg.pinv(W_fold) @ w

    w /= (
        np.linalg.norm(w)
        + 1e-8
    )

    w_all.append(w)

# ==============================================================
# ARRAY
# ==============================================================

w_all = np.array(
    w_all
)

# ==============================================================
# COEFFICIENT STATISTICS
# ==============================================================

w_mean = np.mean(
    w_all,
    axis=0
)

w_std = np.std(

    w_all,
    axis=0,
    ddof=1

)

# ==============================================================
# SIGN CONSISTENCY
# ==============================================================

sign_consistency = []

for i in range(
    w_all.shape[1]
):

    pos = np.sum(
        w_all[:, i] > 0
    )

    neg = np.sum(
        w_all[:, i] < 0
    )

    frac = max(
        pos,
        neg
    ) / len(w_all)

    sign_consistency.append(
        frac
    )

# ==============================================================
# STABILITY METRIC
# ==============================================================

stability_ratio = (

    np.abs(w_mean)
    / (w_std + 1e-8)

)



# ==============================================================
# SAVE STABILITY RESULTS
# ==============================================================

stats_df = pd.DataFrame({

    "feature": RC_FEATURES,

    "coef_mean": w_mean,

    "coef_std": w_std,

    "stability_ratio": stability_ratio,

    "sign_consistency": sign_consistency

})

stats_df.to_csv(

    "anm_rc_coefficient_significance.csv",

    index=False

)

print("\n[INFO] File saved")

print(
    " - rc_coefficient_significance.csv"
)
# ==============================================================
# DATAFRAME
# ==============================================================

rc_stability = pd.DataFrame({

    "feature":
        list(best_subset),

    "mean_weight":
        w_mean,

    "std_weight":
        w_std,

    "stability_ratio":
        stability_ratio,

    "sign_consistency":
        sign_consistency

})

# ==============================================================
# SORT
# ==============================================================

rc_stability = rc_stability.sort_values(

    "stability_ratio",

    ascending=False

)

# ==============================================================
# PRINT
# ==============================================================

banner(
    "RC STABILITY RESULTS"
)

print(
    rc_stability.round(4)
)

# ==============================================================
# SAVE
# ==============================================================

rc_stability.to_csv(

    "anm_minimal_subset_rc_stability.csv",

    index=False

)

print("\n[INFO] File saved")

print(
    " - minimal_subset_rc_stability.csv"
)


# ==============================================================
# REACTION COORDINATE WITH UNCERTAINTY
# ==============================================================

banner(
    "REACTION COORDINATE WITH UNCERTAINTY"
)

eq_terms = []

for feat, mu, sd in zip(

    best_subset,
    w_mean,
    w_std

):

    eq_terms.append(
        f"({mu:.2f}±{sd:.2f})·{feat}"
    )

eq = " + ".join(eq_terms)

print(f"\nξ = {eq}")



# ==============================================================
# EXPLORATORY PARAMETER SENSITIVITY ANALYSIS
# Not used for predictive benchmarking
# ==============================================================
#
# Scans:
#   1. CONTACT_CUTOFF
#   2. CORR_THRESHOLD
#   3. N_MODES
#   4. LOCAL_RADIUS
#
# Evaluates:
#   RMSE
#   R²
#   Pearson r
#   Spearman ρ
#   Kendall τ
#
# Uses:
#   Nested LOOCV
#
# Saves:
#   parameter_scan_results.csv
#   best_parameter_set.csv
#
# ==============================================================

from itertools import product

# ==============================================================
# PARAMETER GRID
# ==============================================================

CONTACT_CUTOFF_GRID = [
    7.0,
    8.0,
    9.0,
    10.0,
    11.0
]

CORR_THRESHOLD_GRID = [
    0.05,
    0.10,
    0.15,
    0.20
]

N_MODES_GRID = [
    10,
    15,
    20,
    25,
    30
]

LOCAL_RADIUS_GRID = [
    8.0,
    10.0,
    12.0,
    14.0
]

# ==============================================================
# RESULTS
# ==============================================================

scan_results = []

# ==============================================================
# TOTAL COMBINATIONS
# ==============================================================

all_combinations = list(product(

    CONTACT_CUTOFF_GRID,
    CORR_THRESHOLD_GRID,
    N_MODES_GRID,
    LOCAL_RADIUS_GRID

))

print("\n================================================")
print("PARAMETER SCAN")
print("================================================")

print(
    "[WARNING] Exploratory scan only "
    "(not nested inside CV)"
)

print(f"[INFO] Total combinations = {len(all_combinations)}")

# ==============================================================
# LOOP
# ==============================================================

for (
    cutoff,
    corr_thr,
    n_modes,
    local_radius

) in tqdm(

    all_combinations,
    desc="Parameter Scan"

):

    try:

        # ------------------------------------------------------
        # UPDATE GLOBALS
        # ------------------------------------------------------

        CONTACT_CUTOFF = cutoff
        CORR_THRESHOLD = corr_thr
        N_MODES = n_modes
        LOCAL_RADIUS = local_radius

        # ------------------------------------------------------
        # BUILD ANM
        # ------------------------------------------------------

        anm_scan = ANM("SCAN_ANM")

        anm_scan.buildHessian(

            ca_atoms,
            cutoff=CONTACT_CUTOFF

        )

        anm_scan.calcModes(
            n_modes=N_MODES
        )

        # ------------------------------------------------------
        # CROSS CORRELATION
        # ------------------------------------------------------

        cc_scan = calcCrossCorr(
            anm_scan
        )

        # ------------------------------------------------------
        # PRS
        # ------------------------------------------------------

        prs_scan, eff_scan, sens_scan = calcPerturbResponse(
            anm_scan
        )

        # ------------------------------------------------------
        # FEATURE EXTRACTION
        # ------------------------------------------------------

        records_scan = []

        for mutant, dg in exp_data.items():

            feats = extract_features(

                structure=structure,

                anm=anm_scan,

                cc=cc_scan,

                sens=sens_scan,

                eff=eff_scan,

                mutant=mutant

            )

            record = {

                "mutation": mutant,
                "ddg": dg

            }

            record.update(feats)

            records_scan.append(
                record
            )

        df_scan = pd.DataFrame(
            records_scan
        )

        # ------------------------------------------------------
        # FEATURE MATRIX
        # ------------------------------------------------------

        X_scan = df_scan[
            RC_FEATURES
        ].values

        y_scan = df_scan[
            "ddg"
        ].values

        # ------------------------------------------------------
        # LOOCV
        # ------------------------------------------------------

        loo = LeaveOneOut()

        pred_all = []
        true_all = []

        for train_idx, test_idx in loo.split(X_scan):

            Xtr = X_scan[train_idx]
            Xte = X_scan[test_idx]

            ytr = y_scan[train_idx]
            yte = y_scan[test_idx]

            # --------------------------------------------------
            # STANDARDIZE
            # --------------------------------------------------

            Xtr_w, Xte_w, _, _ = whiten_train_test(

            Xtr,
            Xte

            )

            # --------------------------------------------------
            # MODEL
            # --------------------------------------------------

            model = LinearRegression()

            model.fit(
            Xtr_w,
            ytr
            )

            pred = model.predict(
            Xte_w
            )[0]

            pred_all.append(
                pred
            )

            true_all.append(
                yte[0]
            )

        # ------------------------------------------------------
        # METRICS
        # ------------------------------------------------------

        pred_all = np.array(
            pred_all
        )

        true_all = np.array(
            true_all
        )

        rmse, r2, rp, rs, kt = regression_metrics(

            true_all,
            pred_all

        )

        # ------------------------------------------------------
        # STORE
        # ------------------------------------------------------

        scan_results.append({

            "contact_cutoff":
                CONTACT_CUTOFF,

            "corr_threshold":
                CORR_THRESHOLD,

            "n_modes":
                N_MODES,

            "local_radius":
                LOCAL_RADIUS,

            "rmse":
                rmse,

            "r2":
                r2,

            "pearson_r":
                rp,

            "spearman_rho":
                rs,

            "kendall_tau":
                kt

        })

    except Exception as e:

        print("\n[WARNING] Failed combination")

        print(

            cutoff,
            corr_thr,
            n_modes,
            local_radius

        )

        print(str(e))

# ==============================================================
# RESULTS DATAFRAME
# ==============================================================

scan_df = pd.DataFrame(
    scan_results
)

# ==============================================================
# SORT
# ==============================================================

scan_df = scan_df.sort_values(

    by=[
        "r2",
        "pearson_r"
    ],

    ascending=False

)

# ==============================================================
# BEST PARAMETERS
# ==============================================================

best_row = scan_df.iloc[0]

print("\n================================================")
print("BEST PARAMETER SET")
print("================================================")

print(

    best_row[
        [

            "contact_cutoff",
            "corr_threshold",
            "n_modes",
            "local_radius"

        ]

    ]

)

print("\n================================================")
print("BEST PERFORMANCE")
print("================================================")

print(f"RMSE         = {best_row['rmse']:.2f}")
print(f"R²           = {best_row['r2']:.2f}")
print(f"Pearson r    = {best_row['pearson_r']:.2f}")
print(f"Spearman ρ   = {best_row['spearman_rho']:.2f}")
print(f"Kendall τ    = {best_row['kendall_tau']:.2f}")

# ==============================================================
# SAVE FULL RESULTS
# ==============================================================

scan_df.to_csv(

    "anm_parameter_scan_results.csv",

    index=False

)

# ==============================================================
# SAVE BEST PARAMETERS
# ==============================================================

best_df = pd.DataFrame([best_row])

best_df.to_csv(

    "anm_best_parameter_set.csv",

    index=False

)

# ==============================================================
# PRINT TOP 10
# ==============================================================

print("\n================================================")
print("TOP 10 PARAMETER SETS")
print("================================================")

print(

    scan_df.head(10).round(4)

)

print("\n[INFO] Files saved")

print(
    " - parameter_scan_results.csv"
)

print(
    " - best_parameter_set.csv"
)



# ==============================================================
# EXPLORATORY FEATURE SUBSET ANALYSIS
# Not used for predictive benchmarking
# LINEAR REGRESSION
# ==============================================================
#
# Evaluates all feature subsets:
#   size = 2 to 5
#
# Uses:
#   LOOCV
#
# Metrics:
#   RMSE
#   R²
#   Pearson r
#   Spearman ρ
#   Kendall τ
#
# Saves:
#   exhaustive_feature_search.csv
#   best_feature_combinations.csv
#
# ==============================================================

from itertools import combinations

from sklearn.linear_model import LinearRegression

# ==============================================================
# FEATURE POOL
# ==============================================================

ALL_FEATURES = [

    "comm_eff",

    "f_collective",

    "f_asym",

    "prs_sens",

    "f_entropy",

    "mut_ca_coupling",

    "prs_eff",

    "f_local",

    "sq_fluct"

]

# ==============================================================
# DATAFRAME CHECK
# ==============================================================

missing = [

    feat for feat in ALL_FEATURES
    if feat not in df.columns

]

if len(missing) > 0:

    raise ValueError(

        f"Missing features: {missing}"

    )

# ==============================================================
# TARGET
# ==============================================================

y_full = df["ddg"].values

# ==============================================================
# RESULTS
# ==============================================================

feature_results = []

# ==============================================================
# TOTAL COMBINATIONS
# ==============================================================

all_feature_sets = []

for k in range(2, 6):

    all_feature_sets.extend(

        list(
            combinations(
                ALL_FEATURES,
                k
            )
        )

    )

print("\n================================================")
print("EXHAUSTIVE FEATURE SEARCH")
print("================================================")

print(
    "\n[WARNING] Exploratory analysis only "
    "(not nested inside CV)"
)
print(

    f"[INFO] Total feature combinations = "
    f"{len(all_feature_sets)}"

)

# ==============================================================
# LOOP
# ==============================================================

for feat_set in tqdm(

    all_feature_sets,
    desc="Feature Search"

):

    feat_set = list(
        feat_set
    )

    # ----------------------------------------------------------
    # FEATURE MATRIX
    # ----------------------------------------------------------

    X_full = df[
        feat_set
    ].values

    # ----------------------------------------------------------
    # LOOCV
    # ----------------------------------------------------------

    loo = LeaveOneOut()

    pred_all = []
    true_all = []

    coef_all = []

    for tr, te in loo.split(X_full):

        Xtr = X_full[tr]
        Xte = X_full[te]

        ytr = y_full[tr]
        yte = y_full[te]

        # ------------------------------------------------------
        # STANDARDIZATION
        # ------------------------------------------------------

        Xtr_w, Xte_w, _, _ = whiten_train_test(

        Xtr,
        Xte
    
        )

        # ------------------------------------------------------
        # MODEL
        # ------------------------------------------------------

        model = LinearRegression()

        model.fit(
        Xtr_w,
        ytr
        )

        pred = model.predict(
            Xte_w
        )[0]
        

        pred_all.append(
            pred
        )

        true_all.append(
            yte[0]
        )

        # ------------------------------------------------------
        # NORMALIZED COEFFICIENTS
        # ------------------------------------------------------

        coef = model.coef_

        coef = coef / (

            np.linalg.norm(coef)
            + 1e-12

        )

        coef_all.append(
            coef
        )

    # ----------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------

    pred_all = np.array(
        pred_all
    )

    true_all = np.array(
        true_all
    )

    rmse, r2, rp, rs, kt = regression_metrics(

        true_all,
        pred_all

    )

    # ----------------------------------------------------------
    # COEFFICIENT STABILITY
    # ----------------------------------------------------------

    coef_all = np.array(
        coef_all
    )

    coef_mean = np.mean(
        coef_all,
        axis=0
    )

    coef_std = np.std(

        coef_all,
        axis=0,
        ddof=1

    )

    stability_ratio = np.mean(

        np.abs(coef_mean)
        / (coef_std + 1e-8)

    )

    # ----------------------------------------------------------
    # STORE
    # ----------------------------------------------------------

    feature_results.append({

        "n_features":
            len(feat_set),

        "features":
            ",".join(feat_set),

        "rmse":
            rmse,

        "r2":
            r2,

        "pearson_r":
            rp,

        "spearman_rho":
            rs,

        "kendall_tau":
            kt,

        "stability_ratio":
            stability_ratio

    })

# ==============================================================
# DATAFRAME
# ==============================================================

feature_df = pd.DataFrame(
    feature_results
)

# ==============================================================
# SORT
# ==============================================================

feature_df = feature_df.sort_values(

    by=[

        "r2",
        "pearson_r",
        "stability_ratio"

    ],

    ascending=False

)

# ==============================================================
# BEST MODEL
# ==============================================================

best_model = feature_df.iloc[0]

print("\n================================================")
print("BEST FEATURE COMBINATION")
print("================================================")

print(

    f"Features = "
    f"{best_model['features']}"

)

print(

    f"Number of features = "
    f"{best_model['n_features']}"

)



# ==============================================================
# TOP RESULTS
# ==============================================================

print("\n================================================")
print("TOP 20 FEATURE COMBINATIONS")
print("================================================")

print(

    feature_df
    .head(20)
    .round(4)

)

# ==============================================================
# SAVE FULL RESULTS
# ==============================================================

feature_df.to_csv(

    "anm_exhaustive_feature_search.csv",

    index=False

)

# ==============================================================
# SAVE TOP RESULTS
# ==============================================================

top_df = feature_df.head(20)

top_df.to_csv(

    "anm_best_feature_combinations.csv",

    index=False

)

# ==============================================================
# FEATURE FREQUENCY ANALYSIS
# ==============================================================

feature_frequency = []

top_models = feature_df.head(50)

for feat in ALL_FEATURES:

    count = np.sum(

        top_models["features"]
        .str.contains(feat)

    )

    frequency = count / len(top_models)

    feature_frequency.append({

        "feature":
            feat,

        "frequency":
            frequency

    })

# ==============================================================
# FEATURE IMPORTANCE TABLE
# ==============================================================

freq_df = pd.DataFrame(
    feature_frequency
)

freq_df = freq_df.sort_values(

    "frequency",
    ascending=False

)

print("\n================================================")
print("FEATURE OCCURRENCE IN TOP 50 MODELS")
print("================================================")

print(
    freq_df.round(4)
)

# ==============================================================
# SAVE
# ==============================================================

freq_df.to_csv(

    "anm_feature_occurrence_frequency.csv",

    index=False

)

print("\n[INFO] Files saved")

print(
    " - exhaustive_feature_search.csv"
)

print(
    " - best_feature_combinations.csv"
)

print(
    " - feature_occurrence_frequency.csv"
)

def plot_rc_vs_ddg(
    df,
    xi,
    dg_pred,
    mode="loocv",   # "loocv" or "fit"
    metrics=None,   # optional dict to pass precomputed metrics
    save_name="RC_vs_ddG"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    print(f"\n[INFO] Plotting RC vs ΔΔG ({mode.upper()})...")

    # ----------------------------------------------------------
    # DATA
    # ----------------------------------------------------------

    x = np.array(xi)
    y = df["ddG_exp"].values
    labels = df["mutant"].values

    # ----------------------------------------------------------
    # METRICS (CONSISTENT)
    # ----------------------------------------------------------

    if metrics is not None:
        rmse = metrics["RMSE"]
        rp_rc = metrics["Rp_RC"]
        rp_pred = metrics["Rp_pred"]

    else:
        # RC correlation
        rp_rc, _ = stats.pearsonr(x, y)

        # prediction correlation
        rp_pred, _ = stats.pearsonr(dg_pred, y)

        # RMSE (prediction)
        rmse = np.sqrt(np.mean((y - dg_pred) ** 2))

    # ----------------------------------------------------------
    # LINEAR FIT (visual guide only)
    # ----------------------------------------------------------

    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_fit = slope * x + intercept

    # ----------------------------------------------------------
    # COLORS
    # ----------------------------------------------------------

    pos_color = '#E74C3C'
    neg_color = '#1ABC9C'
    fit_color = '#2C3E50'

    colors = [pos_color if val > 0 else neg_color for val in y]

    # ----------------------------------------------------------
    # STYLE
    # ----------------------------------------------------------

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold"
    })

    plt.figure(figsize=(5, 4))

    # ----------------------------------------------------------
    # SCATTER
    # ----------------------------------------------------------

    plt.scatter(
        x, y,
        c=colors,
        edgecolors='black',
        linewidths=0.5,
        zorder=3
    )

    # ----------------------------------------------------------
    # FIT LINE
    # ----------------------------------------------------------

    label_txt = (
        f'RMSE={rmse:.2f}, '
        r'$R_p^{RC}$' + f'={rp_rc:.2f}, '
        r'$R_p^{pred}$' + f'={rp_pred:.2f}'
    )

    if mode == "loocv":
        label_txt += " (LOOCV)"
    else:
        label_txt += " (fit)"

    plt.plot(
        x, y_fit,
        color=fit_color,
        linewidth=2,
        zorder=2,
        label=label_txt
    )

    # ----------------------------------------------------------
    # QUADRANT LINES
    # ----------------------------------------------------------

    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.axvline(0, linestyle='--', linewidth=0.8)

    # ----------------------------------------------------------
    # ANNOTATIONS
    # ----------------------------------------------------------

    from adjustText import adjust_text

    texts = []
    for xi_i, yi_i, lab in zip(x, y, labels):
        texts.append(
            plt.text(xi_i, yi_i, lab, fontsize=7)
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', lw=0.5),
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        force_text=0.5
    )

    # ----------------------------------------------------------
    # AXES
    # ----------------------------------------------------------

    plt.xlabel(r'Reaction Coordinate ($\xi$)', fontsize=11)
    plt.ylabel(
        r'$\Delta\Delta G_{\mathrm{exp}}\ (\mathrm{kcal\ mol^{-1}})$',
        fontsize=11
    )

    # ----------------------------------------------------------
    # TITLE TAG
    # ----------------------------------------------------------

    plt.text(
        0.02, 0.98,
        'Dynamics-derived RC',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top'
    )

    # ----------------------------------------------------------
    # LEGEND
    # ----------------------------------------------------------

    plt.legend(frameon=False)

    plt.tight_layout()

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------

    plt.savefig(f"{save_name}_{mode}.pdf")
    plt.savefig(f"{save_name}_{mode}.png", dpi=300)

    plt.show()

df_plot = pd.DataFrame({

    "mutant": df["mutation"],

    "ddG_exp": y

})

plot_rc_vs_ddg(
    df_plot,
    xi_all,          # LOOCV ξ
    dg_pred_rc,      # LOOCV predictions
    mode="loocv"
)


# ==============================================================
# DYNAMICAL RESIDUE INTERPRETATION
# ==============================================================

def classify_dynamical_sector(
    comm_eff,
    f_collective,
    f_entropy,
    prs_sens,
    ddg
):

    # ----------------------------------------------------------
    # communication breakdown
    # ----------------------------------------------------------

    if (
        comm_eff < np.percentile(
            df["comm_eff"],
            25
        )
        and ddg > 0.5
    ):

        return "communication breakdown"

    # ----------------------------------------------------------
    # entropy redistribution
    # ----------------------------------------------------------

    if (
        f_entropy > np.percentile(
            df["f_entropy"],
            75
        )
        and ddg > 0
    ):

        return "entropy redistribution"

    # ----------------------------------------------------------
    # collective compensation
    # ----------------------------------------------------------

    if (
        comm_eff > np.percentile(
            df["comm_eff"],
            75
        )
        and ddg < -0.5
    ):

        return "collective dynamical compensation"

    # ----------------------------------------------------------
    # perturbation-sensitive
    # ----------------------------------------------------------

    if (
        prs_sens > np.percentile(
            df["prs_sens"],
            75
        )
    ):

        return "perturbation-sensitive regime"

    return "intermediate dynamical response"


# ==============================================================
# BUILD INTERPRETATION TABLE
# ==============================================================

def generate_dynamical_interpretation_table():

    print(
        "\n[INFO] Generating dynamical "
        "interpretation table..."
    )

    rows = []

    for i, row in rc_df.iterrows():

        sector = classify_dynamical_sector(

            row["comm_eff"],

            row["f_collective"],

            row["f_entropy"],

            row["prs_sens"],

            row["dgd_exp"]
        )

        # ------------------------------------------------------
        # interpretation text
        # ------------------------------------------------------

        if sector == "communication breakdown":

            interpretation = (
                "Reduced allosteric communication "
                "efficiency and impaired network "
                "propagation"
            )

        elif sector == "entropy redistribution":

            interpretation = (
                "Destabilization dominated by "
                "redistribution of collective "
                "fluctuation entropy"
            )

        elif sector == (
            "collective dynamical compensation"
        ):

            interpretation = (
                "Stabilizing cooperative dynamics "
                "with preserved collective "
                "communication"
            )

        elif sector == (
            "perturbation-sensitive regime"
        ):

            interpretation = (
                "Enhanced perturbation sensitivity "
                "within the dynamical interaction "
                "network"
            )

        else:

            interpretation = (
                "Intermediate dynamical response "
                "with mixed communication "
                "characteristics"
            )

        rows.append({

            "mutation":
                row["mutation"],

            "ddg_exp":
                row["dgd_exp"],

            "RC_projection":
                row["reaction_coordinate"],

            "comm_eff":
                row["comm_eff"],

            "f_collective":
                row["f_collective"],

            "f_entropy":
                row["f_entropy"],

            "prs_sens":
                row["prs_sens"],

            "dynamical_sector":
                sector,

            "interpretation":
                interpretation
        })

    df_dyn = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # RC ranking
    # ----------------------------------------------------------

    df_dyn = df_dyn.sort_values(
        by="RC_projection",
        ascending=False
    )

    # ----------------------------------------------------------
    # extremeness
    # ----------------------------------------------------------

    q_low = df_dyn["RC_projection"].quantile(0.25)

    q_high = df_dyn["RC_projection"].quantile(0.75)

    df_dyn["extreme_class"] = "middle"

    df_dyn.loc[
        df_dyn["RC_projection"] >= q_high,
        "extreme_class"
    ] = "dynamical_destabilizing_extreme"

    df_dyn.loc[
        df_dyn["RC_projection"] <= q_low,
        "extreme_class"
    ] = "dynamical_stabilizing_extreme"

    # ----------------------------------------------------------
    # save
    # ----------------------------------------------------------

    df_dyn.to_csv(

        "dynamical_residue_interpretation.csv",

        index=False
    )

    print(
        "[INFO] Written: "
        "dynamical_residue_interpretation.csv"
    )

    return df_dyn

generate_dynamical_interpretation_table()

