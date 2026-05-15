# Strict Stability-Aware Reaction Coordinate Pipeline


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================
STRICT STABILITY-AWARE REACTION COORDINATE PIPELINE
=====================================================================

Pipeline Contents
-----------------
1. Configuration
2. Utilities
3. Data Loading
4. Metrics
5. Whitening
6. Leakage-Free RC Optimization
7. Statistical Inference
8. Correlation Analysis
9. Manifold Geometry
10. Nonlinear Visualization
11. Model Benchmarking
12. Thermodynamic Regime Analysis
13. Hierarchical Thermodynamics
14. Export Utilities
15. Main Execution

Target
------
ΔΔG = ΔG_mut − ΔG_WT

Canonical Reaction Coordinate
-----------------------------
ξ = Xw

Thermodynamic Model
-------------------
ΔΔG = aξ + b
=====================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from tqdm import tqdm

from scipy.stats import (
    pearsonr,
    spearmanr,
    gaussian_kde,
    zscore,
    t
)

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    pairwise_distances
)

from sklearn.pipeline import Pipeline
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.mixture import GaussianMixture

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import griddata

from adjustText import adjust_text

import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay

# =====================================================================
# OPTIONAL MODULES
# =====================================================================

USE_UMAP = True
USE_XGBOOST = True

try:
    import umap
except:
    USE_UMAP = False

try:
    from xgboost import XGBRegressor
except:
    USE_XGBOOST = False

# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class Config:

    random_seed: int = 42

    max_features: int = 5

    ridge_alpha: float = 0.1

    whiten_eps: float = 1e-6

    top_models: int = 25

    high_corr_threshold: float = 0.70

    isomap_neighbors: int = 5

    output_dir: str = "outputs"


CFG = Config()

np.random.seed(CFG.random_seed)

# =====================================================================
# UTILITIES
# =====================================================================


def section(title):

    print("\n" + "=" * 70)

    print(title)

    print("=" * 70)



def save_dataframe(df, filename):

    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    print(f"[SAVED] {path}")



def save_figure(fig, filename):

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight"
    )

    print(f"[SAVED] {path}")


# =====================================================================
# DATA LOADING
# =====================================================================


def load_dataset(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    if "mutation" in df.columns:
        df = df.rename(columns={"mutation": "mutant"})

    if "dG_exp" in df.columns:
        df = df.rename(columns={"dG_exp": "y"})

    df["mutant"] = (
        df["mutant"]
        .astype(str)
        .str.strip()
    )

    df = df.drop_duplicates("mutant")

    return df



def merge_feature_domains(enm, elc, struct):

    common = (
        set(enm.mutant)
        & set(elc.mutant)
        & set(struct.mutant)
    )

    enm = enm[enm.mutant.isin(common)]
    elc = elc[elc.mutant.isin(common)]
    struct = struct[struct.mutant.isin(common)]

    df = (
        enm
        .merge(elc, on="mutant")
        .merge(struct, on="mutant")
    )

    return df
    

# =====================================================================
# METRICS
# =====================================================================


def compute_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    rp = pearsonr(
        y_true,
        y_pred
    )[0]

    rho = spearmanr(
        y_true,
        y_pred
    )[0]

    return {

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "Rp": rp,

        "Spearman": rho
    }


# =====================================================================
# WHITENING
# =====================================================================


def fit_whitening_matrix(X):

    lw = LedoitWolf()

    lw.fit(X)

    cov = lw.covariance_

    U, S, _ = np.linalg.svd(cov)

    S_inv = np.zeros_like(S)

    mask = S > CFG.whiten_eps

    S_inv[mask] = 1.0 / np.sqrt(S[mask])

    W = U @ np.diag(S_inv) @ U.T

    return W


# =====================================================================
# REACTION COORDINATE ENGINE
# =====================================================================


def scale_fold_data(Xtr, Xte):

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr)

    Xte_scaled = scaler.transform(Xte)

    return scaler, Xtr_scaled, Xte_scaled



def recover_fold_coefficients(
    scaler,
    W,
    ridge_model
):

    w_white = ridge_model.coef_

    scale_inv = np.diag(
        1.0 / scaler.scale_
    )

    w_fold = scale_inv @ W @ w_white

    w_fold = (
        w_fold /
        np.linalg.norm(w_fold)
    )

    return w_fold



def compute_directional_stability(coef_matrix):

    ref = coef_matrix.mean(axis=0)

    ref = ref / np.linalg.norm(ref)

    cosines = []

    for w in coef_matrix:

        w_norm = w / np.linalg.norm(w)

        cosines.append(
            np.dot(ref, w_norm)
        )

    return np.mean(cosines)



def evaluate_subset_strict(
    df,
    y,
    feature_list
):

    X = df[feature_list].values

    loo = LeaveOneOut()

    y_pred = np.zeros(len(y))

    coef_records = []

    for tr, te in loo.split(X):

        Xtr_raw = X[tr]
        Xte_raw = X[te]

        ytr = y[tr]

        scaler, Xtr_scaled, Xte_scaled = (
            scale_fold_data(
                Xtr_raw,
                Xte_raw
            )
        )

        W = fit_whitening_matrix(
            Xtr_scaled
        )

        Xtr_white = Xtr_scaled @ W
        Xte_white = Xte_scaled @ W

        ridge = Ridge(
            alpha=CFG.ridge_alpha
        )

        ridge.fit(
            Xtr_white,
            ytr
        )

        w_fold = recover_fold_coefficients(
            scaler,
            W,
            ridge
        )

        coef_records.append(w_fold)

        xi_tr = Xtr_raw @ w_fold
        xi_te = Xte_raw @ w_fold

        thermo = LinearRegression()

        thermo.fit(
            xi_tr.reshape(-1, 1),
            ytr
        )

        pred = thermo.predict(
            xi_te.reshape(-1, 1)
        )

        y_pred[te] = pred[0]

    metrics = compute_metrics(
        y,
        y_pred
    )

    ref = coef_records[0]

    aligned = []

    for w in coef_records:

        if np.dot(w, ref) < 0:
            w = -w

        aligned.append(w)

    coef_matrix = np.vstack(aligned)

    coef_mean = coef_matrix.mean(axis=0)

    coef_mean = (
        coef_mean /
        np.linalg.norm(coef_mean)
    )

    coef_std = coef_matrix.std(axis=0)

    stability = compute_directional_stability(
        coef_matrix
    )

    score = (
        metrics["Rp"]
        - 0.15 * metrics["RMSE"]
        + 0.10 * metrics["R2"]
        + 0.15 * stability
    )

    return {

        "features": feature_list,

        "metrics": metrics,

        "stability": stability,

        "score": score,

        "coef_mean": coef_mean,

        "coef_std": coef_std,

        "coef_matrix": coef_matrix,

        "y_pred": y_pred
    }


# =====================================================================
# EXHAUSTIVE SUBSET SEARCH
# =====================================================================


def exhaustive_subset_search(
    df,
    y,
    feature_names
):

    section("STRICT STABILITY-AWARE SUBSET SEARCH")

    all_results = []

    for k in range(2, CFG.max_features + 1):

        subsets = list(
            combinations(feature_names, k)
        )

        print(f"\nEvaluating {k}-feature subsets")

        for subset in tqdm(subsets):

            result = evaluate_subset_strict(
                df,
                y,
                list(subset)
            )

            all_results.append(result)

    all_results = sorted(
        all_results,
        key=lambda x: (
            -x["score"],
            -x["metrics"]["Rp"],
            x["metrics"]["RMSE"]
        )
    )

    return all_results


# =====================================================================
# PUBLICATION MODEL
# =====================================================================


def build_publication_model(
    df,
    y,
    best_result
):

    features = best_result["features"]

    coef_mean = best_result["coef_mean"]

    X = df[features].values

    xi = X @ coef_mean

    thermo = LinearRegression()

    thermo.fit(
        xi.reshape(-1, 1),
        y
    )

    a_global = thermo.coef_[0]

    b_global = thermo.intercept_

    X_design = sm.add_constant(xi)

    ols = sm.OLS(
        y,
        X_design
    ).fit()

    coef_df = pd.DataFrame({

        "feature": features,

        "coefficient": coef_mean,

        "coef_std": best_result["coef_std"]
    })

    coef_df["abs_coef"] = np.abs(
        coef_df["coefficient"]
    )

    coef_df = coef_df.sort_values(
        by="abs_coef",
        ascending=False
    )

    return {

        "features": features,

        "xi": xi,

        "a_global": a_global,

        "b_global": b_global,

        "ols": ols,

        "coef_df": coef_df
    }


# =====================================================================
# CORRELATION ANALYSIS
# =====================================================================


def run_correlation_analysis(
    df,
    feature_names
):

    section("FEATURE CORRELATION ANALYSIS")

    X_corr = df[feature_names].copy()

    pearson_corr = X_corr.corr(
        method="pearson"
    )

    spearman_corr = X_corr.corr(
        method="spearman"
    )

    save_dataframe(
        pearson_corr,
        "feature_correlations_pearson.csv"
    )

    save_dataframe(
        spearman_corr,
        "feature_correlations_spearman.csv"
    )

    return {

        "pearson": pearson_corr,

        "spearman": spearman_corr
    }


# =====================================================================
# MANIFOLD GEOMETRY
# =====================================================================


def compute_intrinsic_dimension(eigenvalues):

    participation_ratio = (

        np.sum(eigenvalues) ** 2

        /

        np.sum(eigenvalues ** 2)
    )

    p = eigenvalues / np.sum(eigenvalues)

    spectral_entropy = -np.sum(
        p * np.log(p + 1e-12)
    )

    entropy_dimension = np.exp(
        spectral_entropy
    )

    return participation_ratio, entropy_dimension



def run_manifold_geometry(
    df,
    manifold_features
):

    section("MANIFOLD GEOMETRY")

    X = df[manifold_features].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)

    eigenvalues = pca.explained_variance_

    explained_ratio = (
        pca.explained_variance_ratio_
    )

    participation_ratio, entropy_dimension = (
        compute_intrinsic_dimension(
            eigenvalues
        )
    )

    print(
        f"Participation Ratio : "
        f"{participation_ratio:.3f}"
    )

    print(
        f"Entropy Dimension   : "
        f"{entropy_dimension:.3f}"
    )

    isomap = Isomap(

        n_neighbors=CFG.isomap_neighbors,

        n_components=2
    )

    X_iso = isomap.fit_transform(
        X_scaled
    )

    return {

        "X_pca": X_pca,

        "X_iso": X_iso,

        "explained_ratio": explained_ratio,

        "participation_ratio": participation_ratio,

        "entropy_dimension": entropy_dimension
    }


# =====================================================================
# PLOTTING UTILITIES
# =====================================================================


def scatter_embedding(
    x,
    y,
    color,
    xlabel,
    ylabel,
    title,
    cbar_label,
    filename
):

    fig = plt.figure(figsize=(7, 6))

    plt.scatter(
        x,
        y,
        c=color,
        s=80,
        edgecolor="k"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    cb = plt.colorbar()

    cb.set_label(cbar_label)

    plt.tight_layout()

    save_figure(fig, filename)

    plt.close()


# =====================================================================
# MODEL BENCHMARKING
# =====================================================================


def evaluate_model_loocv(
    model,
    X,
    y
):

    loo = LeaveOneOut()

    y_pred = np.zeros(len(y))

    for tr, te in loo.split(X):

        Xtr = X[tr]
        Xte = X[te]

        ytr = y[tr]

        model.fit(Xtr, ytr)

        pred = model.predict(Xte)

        y_pred[te] = pred[0]

    return compute_metrics(y, y_pred), y_pred



def benchmark_models(
    df,
    model_features,
    y
):

    section("MODEL BENCHMARKING")

    X = df[model_features].values

    models = {

        "Linear":

            Pipeline([

                ('scaler', StandardScaler()),

                ('model', LinearRegression())
            ]),

        "Ridge":

            Pipeline([

                ('scaler', StandardScaler()),

                ('model', Ridge(alpha=1.0))
            ]),

        "Lasso":

            Pipeline([

                ('scaler', StandardScaler()),

                ('model', Lasso(alpha=0.01))
            ]),

        "ElasticNet":

            Pipeline([

                ('scaler', StandardScaler()),

                ('model', ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5
                ))
            ]),

        "Polynomial-2":

            Pipeline([

                ('scaler', StandardScaler()),

                ('poly', PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )),

                ('model', Ridge(alpha=1.0))
            ]),

        "SVR-RBF":

            Pipeline([

                ('scaler', StandardScaler()),

                ('model', SVR(
                    kernel='rbf',
                    C=5.0,
                    epsilon=0.1,
                    gamma='scale'
                ))
            ]),

        "RandomForest":

            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                random_state=42
            ),

        "GradientBoosting":

            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                random_state=42
            )
    }

    if USE_XGBOOST:

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )

    results = []

    for name, model in models.items():

        metrics, _ = evaluate_model_loocv(
            model,
            X,
            y
        )

        row = {
            "model": name,
            **metrics
        }

        results.append(row)

        print(
            f"{name:<20} "
            f"Rp={metrics['Rp']:.3f} "
            f"RMSE={metrics['RMSE']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Rp",
        ascending=False
    )

    save_dataframe(
        results_df,
        "model_comparison_results.csv"
    )

    return results_df


# =====================================================================
# THERMODYNAMIC REGIME ANALYSIS
# =====================================================================


def analyze_thermodynamic_regimes(
    embedding,
    xi,
    y
):

    section("THERMODYNAMIC REGIME ANALYSIS")

    bic_scores = []

    gmm_models = []

    k_range = range(2, 7)

    for k in k_range:

        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42
        )

        gmm.fit(embedding)

        bic = gmm.bic(embedding)

        bic_scores.append(bic)

        gmm_models.append(gmm)

        print(f"K={k} | BIC={bic:.2f}")

    best_idx = np.argmin(bic_scores)

    best_gmm = gmm_models[best_idx]

    regimes = best_gmm.predict(embedding)

    probabilities = best_gmm.predict_proba(
        embedding
    )

    confidence = probabilities.max(axis=1)

    regime_df = pd.DataFrame({

        "ISO1": embedding[:, 0],

        "ISO2": embedding[:, 1],

        "xi": xi,

        "ddG_exp": y,

        "regime": regimes,

        "confidence": confidence
    })

    save_dataframe(
        regime_df,
        "thermodynamic_regimes.csv"
    )

    return regime_df


# =====================================================================
# HIERARCHICAL THERMODYNAMICS
# =====================================================================


def hierarchical_thermodynamic_analysis(
    df,
    y
):

    section("HIERARCHICAL THERMODYNAMICS")

    hierarchy = {

        "Layer-1_Steric": [
            "volume_change",
            "steric_energy"
        ],

        "Layer-2_Electrostatic": [
            "volume_change",
            "steric_energy",
            "E_ca",
            "phi_sq"
        ],

        "Layer-3_Anisotropic": [
            "volume_change",
            "steric_energy",
            "E_ca",
            "phi_sq",
            "E_perp"
        ]
    }

    rows = []

    for layer, features in hierarchy.items():

        X = df[features].values

        loo = LeaveOneOut()

        y_pred = np.zeros(len(y))

        for tr, te in loo.split(X):

            Xtr = X[tr]
            Xte = X[te]

            ytr = y[tr]

            scaler = StandardScaler()

            Xtr_scaled = scaler.fit_transform(Xtr)

            Xte_scaled = scaler.transform(Xte)

            pca = PCA(n_components=1)

            xi_tr = pca.fit_transform(
                Xtr_scaled
            ).flatten()

            xi_te = pca.transform(
                Xte_scaled
            ).flatten()

            reg = LinearRegression()

            reg.fit(
                xi_tr.reshape(-1, 1),
                ytr
            )

            pred = reg.predict(
                xi_te.reshape(-1, 1)
            )

            y_pred[te] = pred[0]

        metrics = compute_metrics(y, y_pred)

        row = {
            "layer": layer,
            **metrics
        }

        rows.append(row)

        print(
            f"{layer:<30} "
            f"Rp={metrics['Rp']:.3f}"
        )

    hierarchy_df = pd.DataFrame(rows)

    save_dataframe(
        hierarchy_df,
        "hierarchical_thermodynamics.csv"
    )

    return hierarchy_df


# =====================================================================
# RESULTS TABLE
# =====================================================================


def build_results_table(all_results):

    rows = []

    for idx, result in enumerate(all_results):

        metrics = result["metrics"]

        rows.append({

            "rank": idx + 1,

            "n_features": len(result["features"]),

            "features": ",".join(result["features"]),

            "RMSE": round(metrics["RMSE"], 3),

            "MAE": round(metrics["MAE"], 3),

            "R2": round(metrics["R2"], 3),

            "Rp": round(metrics["Rp"], 3),

            "Spearman": round(metrics["Spearman"], 3),

            "stability": round(result["stability"], 3),

            "score": round(result["score"], 3)
        })

    results_df = pd.DataFrame(rows)

    return results_df


# =====================================================================
# PUBLICATION REPORTING
# =====================================================================


def print_top_models(results_df):

    section("TOP STABILITY-AWARE RC MODELS")

    for _, row in results_df.head(CFG.top_models).iterrows():

        print(
            f"[{int(row['rank'])}] "
            f"Rp={row['Rp']:.3f} | "
            f"RMSE={row['RMSE']:.3f} | "
            f"R2={row['R2']:.3f} | "
            f"MAE={row['MAE']:.3f} | "
            f"Stability={row['stability']:.3f} | "
            f"Score={row['score']:.3f}"
        )

        print(f"Features: {row['features']}")



def print_publication_model(
    best_result,
    publication
):

    section("FINAL PUBLICATION RC MODEL")

    metrics = best_result["metrics"]

    print("\nBest Features:")

    for feature in publication["features"]:

        print(f"  - {feature}")

    section("STRICT LEAKAGE-FREE PERFORMANCE")

    print(f"RMSE      : {metrics['RMSE']:.3f}")
    print(f"MAE       : {metrics['MAE']:.3f}")
    print(f"R2        : {metrics['R2']:.3f}")
    print(f"Rp        : {metrics['Rp']:.3f}")
    print(f"Spearman  : {metrics['Spearman']:.3f}")
    print(f"Stability : {best_result['stability']:.3f}")
    print(f"Score     : {best_result['score']:.3f}")

    section("FINAL THERMODYNAMIC MODEL")

    print(
        f"ΔΔG = ({publication['a_global']:.3f}) ξ "
        f"+ ({publication['b_global']:.3f})"
    )

    section("GLOBAL CANONICAL RC")

    print("\nξ =")

    for _, row in publication["coef_df"].iterrows():

        print(
            f"{row['coefficient']:+.3f} * "
            f"{row['feature']}"
        )

    section("OLS STATISTICS")

    ols = publication["ols"]

    print(f"R²        : {ols.rsquared:.3f}")
    print(f"Adj. R²   : {ols.rsquared_adj:.3f}")
    print(f"F-stat    : {ols.fvalue:.3f}")
    print(f"p-value   : {ols.f_pvalue:.3e}")
    print(f"AIC       : {ols.aic:.3f}")
    print(f"BIC       : {ols.bic:.3f}")



def compute_rc_uncertainty(
    publication,
    n_samples
):

    coef_df = publication["coef_df"].copy()

    coef_se = coef_df["coef_std"].values

    coef_t = (
        coef_df["coefficient"].values /
        (coef_se + 1e-12)
    )

    dof = (
        n_samples -
        len(coef_df) - 1
    )

    coef_p = 2 * (
        1 - t.cdf(
            np.abs(coef_t),
            df=dof
        )
    )

    tcrit = t.ppf(
        0.975,
        df=dof
    )

    coef_df["t_value"] = coef_t

    coef_df["p_value"] = coef_p

    coef_df["ci_low"] = (
        coef_df["coefficient"] -
        tcrit * coef_se
    )

    coef_df["ci_high"] = (
        coef_df["coefficient"] +
        tcrit * coef_se
    )

    return coef_df



def print_rc_uncertainty(
    coef_df,
    publication
):

    section("GLOBAL CANONICAL RC WITH UNCERTAINTY")

    print("\nξ =")

    for _, row in coef_df.iterrows():

        print(
            f"{row['coefficient']:+.3f} ± "
            f"{row['coef_std']:.3f} * "
            f"{row['feature']} "
            f"(p={row['p_value']:.3e})"
        )

    section("THERMODYNAMIC MODEL UNCERTAINTY")

    ols = publication["ols"]

    a_global = publication["a_global"]
    b_global = publication["b_global"]

    a_se = ols.bse[1]
    b_se = ols.bse[0]

    a_p = ols.pvalues[1]
    b_p = ols.pvalues[0]

    print(
        f"a_global = {a_global:.3f} ± {a_se:.3f} "
        f"| p = {a_p:.3e}"
    )

    print(
        f"b_global = {b_global:.3f} ± {b_se:.3f} "
        f"| p = {b_p:.3e}"
    )




# =====================================================================
# 3D THERMODYNAMIC SECTOR PROJECTION (CLEAN PIPELINE)
# Structural / Electrostatic / Dynamical RC Decomposition
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize, TwoSlopeNorm
from tqdm import tqdm

# =====================================================================
# CONFIG
# =====================================================================

FIGSIZE = (9, 8)

OUTPUT_FIG_3D = "figure_3D_sector_RC.png"
OUTPUT_FIG_3D_LABELED = "figure_3D_sector_RC_labeled.png"
OUTPUT_CSV = "sector_projection_coordinates.csv"

# =====================================================================
# FEATURE GROUPS
# =====================================================================

STRUCTURAL_FEATURES = ["volume_change", "steric_energy", "inv_distance"]
ELECTROSTATIC_FEATURES = ["E_ca", "phi_sq"]
DYNAMICAL_FEATURES = ["comm_eff", "f_collective", "prs_sens", "f_entropy"]

# =====================================================================
# CORE: BUILD SECTOR PROJECTION
# =====================================================================

def build_sector_projection(df, publication, normalize=True):

    coef_df = publication["coef_df"].copy()
    coef_map = dict(zip(coef_df["feature"], coef_df["coefficient"]))

    def compute_projection(feature_group):
        feats = [f for f in coef_map if f in feature_group]
        vec = np.zeros(len(df))
        for f in feats:
            vec += coef_map[f] * df[f].values
        return vec, feats

    xi_struct, struct_feats = compute_projection(STRUCTURAL_FEATURES)
    xi_elec, elec_feats = compute_projection(ELECTROSTATIC_FEATURES)
    xi_dyn, dyn_feats = compute_projection(DYNAMICAL_FEATURES)

    print("\n[FEATURE MAP]")
    print("Structural:", struct_feats)
    print("Electrostatic:", elec_feats)
    print("Dynamical:", dyn_feats)

    if normalize:
        coords = np.column_stack([xi_struct, xi_elec, xi_dyn])
        coords = StandardScaler().fit_transform(coords)
        xi_struct, xi_elec, xi_dyn = coords.T

    proj_df = pd.DataFrame({
        "mutant": df["mutant"].values,
        "xi_struct": xi_struct,
        "xi_elec": xi_elec,
        "xi_dyn": xi_dyn,
        "ddG_exp": df["ddG_exp"].values
    })

    return proj_df

# =====================================================================
# PLOTTING: BASE 3D SCATTER
# =====================================================================

def plot_3D(
    proj_df,
    filename="figure_3D_adjustText.png"
):
    """
    3D scatter + 2D adjustText labeling with arrows (all mutations)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.mplot3d import proj3d
    from adjustText import adjust_text
    from tqdm import tqdm

    # =========================================================
    # FIGURE
    # =========================================================
    plt.rcParams.update({

        "font.family": "serif",

        "mathtext.fontset": "stix",

        "font.size": 10,

        "axes.titleweight": "bold",

        "axes.labelweight": "bold"

    })
    

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = TwoSlopeNorm(
        vmin=proj_df["ddG_exp"].min(),
        vcenter=0.0,
        vmax=proj_df["ddG_exp"].max()
    )

    sc = ax.scatter(
        proj_df["xi_struct"],
        proj_df["xi_elec"],
        proj_df["xi_dyn"],
        c=proj_df["ddG_exp"],
        cmap="coolwarm",
        norm=norm,
        s=80,
        edgecolors='black',
        linewidths=0.5
    )

    #ax.set_xlabel(r"$\xi_{\mathrm{struct}}^{(\mathrm{PC1})}$",labelpad=10)
    #ax.set_ylabel(r"$\xi_{\mathrm{elec}}^{(\mathrm{PC1})}$", labelpad=10)
    #ax.set_zlabel(r"$\xi_{\mathrm{dyn}}^{(\mathrm{PC1})}$", labelpad=10)

    ax.set_xlabel(
    r"$\xi_{\mathrm{struct}}^{(\mathrm{PC1})}$",
    fontsize=16,
    fontweight='bold',
    labelpad=12
    )

    ax.set_ylabel(
    r"$\xi_{\mathrm{elec}}^{(\mathrm{PC1})}$",
    fontsize=16,
    fontweight='bold',
    labelpad=12
    )

    ax.set_zlabel(
    r"$\xi_{\mathrm{dyn}}^{(\mathrm{PC1})}$",
    fontsize=16,
    fontweight='bold',
    labelpad=14
    )
    ax.view_init(elev=30, azim=210)

    # =========================================================
    # PROJECT 3D → 2D
    # =========================================================

    fig.canvas.draw()

    points_2D = []
    for x, y, z in zip(
        proj_df["xi_struct"],
        proj_df["xi_elec"],
        proj_df["xi_dyn"]
    ):
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        points_2D.append((x2, y2))

    # =========================================================
    # CREATE TEXT OBJECTS
    # =========================================================

    texts = []

    for i, row in tqdm(proj_df.iterrows(), total=len(proj_df), desc="Creating labels"):

        x2, y2 = points_2D[i]

        txt = ax.text2D(
            x2, y2,
            str(row["mutant"]),
            fontsize=7,
            transform=ax.transData
        )

        texts.append(txt)

    # =========================================================
    # ADJUST TEXT (OVERLAP RESOLUTION)
    # =========================================================

    adjust_text(
        texts,
        x=[p[0] for p in points_2D],
        y=[p[1] for p in points_2D],
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        force_text=0.5,
        force_points=0.5,
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            lw=0.5
        )
    )

    # =========================================================
    # COLORBAR
    # =========================================================

    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(r"$\Delta\Delta G_{\mathrm{exp}}$ (kcal mol$^{-1}$)")


    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()
    plt.close()

    print(f"[SAVED] {filename}")    
# =====================================================================
# EXPORT
# =====================================================================

def export_coordinates(proj_df, filename=OUTPUT_CSV):
    proj_df.to_csv(filename, index=False)
    print(f"[SAVED] {filename}")

# =====================================================================
# DRIVER
# =====================================================================

def run_sector_projection_analysis(df, publication):

    print("\n" + "=" * 60)
    print("THERMODYNAMIC SECTOR PROJECTION")
    print("=" * 60)

    proj_df = build_sector_projection(df, publication, normalize=True)

    export_coordinates(proj_df)

    plot_3D(proj_df, OUTPUT_FIG_3D)
    #plot_3D(proj_df, OUTPUT_FIG_3D_LABELED, labeled=True, diverging=True)

    print("\n[PIPELINE COMPLETE]")
    return proj_df

# ===============================
# USAGE
# ===============================
# RC_feature_attribution(proj_df)


# =====================================================================
# HIERARCHICAL ENERGETIC MANIFOLD FIGURE
# =====================================================================

def generate_energetic_manifold_figure(
    df,
    y,
    f_struct,
    f_elc,
    f_enm,
        publication

):

    section("HIERARCHICAL ENERGETIC MANIFOLD")

    # ================================================================
    # IMPORTS
    # ================================================================

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.interpolate import (
        UnivariateSpline,
        griddata
    )

    from scipy.spatial import Delaunay

    from sklearn.preprocessing import (
        StandardScaler
    )

    from sklearn.cluster import KMeans

    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis
    )

    from sklearn.cross_decomposition import (
        PLSRegression
    )

    from sklearn.linear_model import (
        Ridge,
        LinearRegression
    )

    # ================================================================
    # REPRODUCIBILITY
    # ================================================================

    np.random.seed(0)

    # ================================================================
    # FEATURE MATRICES
    # ================================================================

    X_struct = df[f_struct].values
    X_elec   = df[f_elc].values
    X_dyn    = df[f_enm].values

    # ================================================================
    # SUPERVISED BLOCK PROJECTION
    # ================================================================

    def supervised_projection(X, y):

        scaler = StandardScaler()

        Xs = scaler.fit_transform(X)

        ridge = Ridge(alpha=0.1)

        ridge.fit(Xs, y)

        w = ridge.coef_

        norm = np.linalg.norm(w)

        if norm < 1e-12:

            return np.zeros((len(X), 1))

        w = w / norm

        rc = Xs @ w

        return rc.reshape(-1, 1)

    # ================================================================
    # BLOCK REACTION COORDINATES
    # ================================================================

    D_struct = supervised_projection(
        X_struct,
        y
    )

    D_elec = supervised_projection(
        X_elec,
        y
    )

    D_dyn = supervised_projection(
        X_dyn,
        y
    )

    # ================================================================
    # HIERARCHICAL BLOCK MATRIX
    # ================================================================

    X_blocks = np.concatenate(
        [
            D_struct,
            D_elec,
            D_dyn
        ],
        axis=1
    )

    # ================================================================
    # SCALE BLOCK SPACE
    # ================================================================

    scaler_global = StandardScaler()

    Xs = scaler_global.fit_transform(
        X_blocks
    )

    # ================================================================
    # GLOBAL THERMODYNAMIC RC
    # ================================================================

    coef = publication["coef_df"]["coefficient"].values
    features = publication["coef_df"]["feature"].values

    X_sel = df[features].values
    RC = X_sel @ coef

    # ================================================================
    # GLOBAL ENERGETIC MODEL
    # ================================================================

    lin = LinearRegression()

    lin.fit(
        RC.reshape(-1, 1),
        y
    )

    yhat = lin.predict(
        RC.reshape(-1, 1)
    )

    residual = y - yhat

    # ================================================================
    # REGIME DISCOVERY
    # ================================================================

    cluster_space = np.column_stack([
        RC,
        residual
    ])

    kmeans = KMeans(
        n_clusters=2,
        random_state=0,
        n_init=50
    )

    labels = kmeans.fit_predict(
        cluster_space
    )

    # ================================================================
    # STABLE LABEL ORDERING
    # ================================================================

    centers = []

    for lab in np.unique(labels):

        centers.append(
            residual[labels == lab].mean()
        )

    centers = np.array(centers)

    order = np.argsort(centers)

    labels = np.array([
        np.where(order == l)[0][0]
        for l in labels
    ])

    # ================================================================
    # RESIDUAL DISCRIMINANT COORDINATE
    # ================================================================

    lda = LinearDiscriminantAnalysis(
        n_components=1
    )

    RDC = lda.fit_transform(
        Xs,
        labels
    ).flatten()

    # ================================================================
    # FREE ENERGY MANIFOLD
    # ================================================================

    rc_grid = np.linspace(
        RC.min(),
        RC.max(),
        180
    )

    rdc_grid = np.linspace(
        RDC.min(),
        RDC.max(),
        180
    )

    RC_mesh, RDC_mesh = np.meshgrid(
        rc_grid,
        rdc_grid
    )

    G_surface = griddata(
        (RC, RDC),
        y,
        (RC_mesh, RDC_mesh),
        method="linear"
    )

    # ================================================================
    # REMOVE EXTRAPOLATION
    # ================================================================

    points = np.column_stack([
        RC,
        RDC
    ])

    tri = Delaunay(points)

    mask = tri.find_simplex(
        np.column_stack([
            RC_mesh.ravel(),
            RDC_mesh.ravel()
        ])
    ) < 0

    G_surface_flat = G_surface.ravel()

    G_surface_flat[mask] = np.nan

    G_surface = G_surface_flat.reshape(
        G_surface.shape
    )

    # ================================================================
    # FIGURE STYLE
    # ================================================================

    plt.rcParams.update({

        "font.family": "serif",

        "mathtext.fontset": "stix",

        "font.size": 10,

        "axes.titleweight": "bold",

        "axes.labelweight": "bold"

    })
    
    # ================================================================
    # FIGURE
    # ================================================================

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(9.8, 3.5)
    )

    fig.subplots_adjust(
        wspace=0.42
    )

    # ================================================================
    # PANEL A
    # HIERARCHICAL DESCRIPTOR COLLAPSE
    # ================================================================

    features = [
        "volume_change",
        "steric_energy",
        "E_ca",
        "phi_sq",
        "E_perp"
    ]

    X_desc = df[features].values

    scaler_desc = StandardScaler()

    X_desc_scaled = scaler_desc.fit_transform(
        X_desc
    )

    idx_rc = np.argsort(RC)

    RC_sorted = RC[idx_rc]

    X_sorted = X_desc_scaled[idx_rc]

    feature_labels = {

        "volume_change": r"$\Delta V$",

        "steric_energy": r"$E_{\mathrm{steric}}$",

        "E_ca": r"$E_{\mathrm{Ca}}$",

        "phi_sq": r"$\phi^2$",

        "E_perp": r"$E_{\perp}$"
    }

    colors = plt.cm.tab10.colors

    for i, feat in enumerate(features):

        c = colors[i]

        ax[0].plot(
            RC_sorted,
            X_sorted[:, i],
            linewidth=1.2,
            alpha=0.30,
            color=c
        )

        ax[0].scatter(
            RC_sorted,
            X_sorted[:, i],
            s=30,
            alpha=0.75,
            color=c,
            edgecolor="black",
            linewidth=0.25,
            label=feature_labels[feat]
        )

    ax[0].set_xlabel(
        r"Reaction coordinate $(\xi)$",
        fontweight="bold"
    )

    ax[0].set_ylabel(
        "Standardized descriptor value",
        fontweight="bold"
    )

    ax[0].set_title(
        "Descriptor collapse",
        fontweight="bold"
    )

    ax[0].legend(
        frameon=False,
        fontsize=8
    )

    # ================================================================
    # PANEL B
    # GLOBAL ENERGETIC COORDINATE
    # ================================================================


    # ------------------------------------------------
    # LINEAR MODEL
    # ------------------------------------------------

    loo = LeaveOneOut()

    y_pred_panel = np.zeros(len(y))

    for tr, te in loo.split(RC):

        model = LinearRegression()

        model.fit(
        RC[tr].reshape(-1, 1),
        y[tr]
        )

        y_pred_panel[te] = model.predict(
            RC[te].reshape(-1, 1)
        )[0]

    rp = pearsonr(y, y_pred_panel)[0]

    rmse = np.sqrt(
    mean_squared_error(
        y,
        y_pred_panel
        )
    )
    
    # ------------------------------------------------
    # REGRESSION LINE
    # ------------------------------------------------
    lin_panel = LinearRegression()

    lin_panel.fit(
    RC.reshape(-1, 1),y)
    grid_A = np.linspace(
        RC.min(),
        RC.max(),
        300)

    pred_A = lin_panel.predict(
        grid_A.reshape(-1, 1)
    )
    
    # ------------------------------------------------
    # SCATTER
    # ------------------------------------------------

    ax[1].scatter(
        RC,
        y,
        s=46,
        color="#8c510a",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.90
    )


    # ------------------------------------------------
    # FORCE-BASED ANNOTATIONS
    # ------------------------------------------------

    important = np.argsort(
        np.abs(y)
    )[-10:]
    idx_all = np.arange(len(y))
    texts = []
    for i in idx_all:
        texts.append(
            ax[2].text(
            RC[i],
            y[i],
            mutants[i],
            fontsize=9,
            weight='bold',
            color='black',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.2',
                alpha=0.85
            )
        )
    )

    adjust_text(

    texts,

    ax=ax[1],

    expand_points=(1.6, 1.8),

    expand_text=(1.5, 1.7),

    force_points=0.9,

    force_text=1.0,

    lim=500,

    arrowprops=dict(
        arrowstyle="-",
        lw=0.6,
        alpha=0.6
    )
    )

    # ------------------------------------------------
    # LINE
    # ------------------------------------------------

    ax[1].plot(
        grid_A,
        pred_A,
        linewidth=2.6,
        color="#bf812d"
    )

    # ------------------------------------------------
    # LABELS
    # ------------------------------------------------

    ax[1].set_xlabel(
        r"Reaction coordinate $(\xi)$",
        fontweight="bold"
    )

    ax[1].set_ylabel(
        r"$\Delta\Delta G$",
        fontweight="bold"
    )

    ax[1].set_title(
        "Energetic coordinate",
        fontweight="bold"
    )

    # ------------------------------------------------
    # METRIC BOX
    # ------------------------------------------------

    stats_text = (
        rf"$R_p = {rp:.2f}$" "\n"
        rf"$\mathrm{{RMSE}} = {rmse:.2f}$"
    )

    ax[1].text(
        0.05,
        0.95,
        stats_text,
        transform=ax[1].transAxes,
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            fc="white",
            ec="#6b6b6b",
            alpha=0.92
        )
    )

    # ================================================================
    # PANEL C
    # RESIDUAL ENERGETIC MANIFOLD
    # ================================================================


    cf = ax[2].contourf(
        RC_mesh,
        RDC_mesh,
        G_surface,
        levels=20,
        alpha=0.85
    )

    ax[2].contour(
        RC_mesh,
        RDC_mesh,
        G_surface,
        levels=12,
        linewidths=0.7,
        alpha=0.70,
        colors="black"
    )

    markers = ["o", "s", "^", "d"]

    for lab, marker in zip(
        np.unique(labels),
        markers
    ):

        m = labels == lab

        ax[2].scatter(
            RC[m],
            RDC[m],
            marker=marker,
            s=44,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
            zorder=5,
            label=f"Regime {lab+1}"
        )

    # ------------------------------------------------
    # FORCE-BASED MUTATION LABELS
    # ------------------------------------------------

    important = np.argsort(
        np.abs(y)
    )[-10:]

    texts = []

    bbox = dict(

        boxstyle="round,pad=0.25",

        fc="white",

        ec="none",

        alpha=0.80
    )

    for i in important:

        txt = ax[2].text(

            RC[i],
            RDC[i],

            df["mutant"].iloc[i],

            fontsize=7,

            alpha=0.95,

            zorder=10,

            bbox=bbox
        )

        texts.append(txt)

    adjust_text(

        texts,

        ax=ax[2],

        expand_points=(1.50, 1.70),

        expand_text=(1.40, 1.60),

        force_points=0.90,

        force_text=1.00,

        lim=500,

        arrowprops=dict(
            arrowstyle="-",
            lw=0.6,
            alpha=0.60
        )
    )

    # ------------------------------------------------
    # LEGEND
    # ------------------------------------------------

    ax[2].legend(
        frameon=False,
        fontsize=8,
        loc="upper right"
    )

    # ------------------------------------------------
    # LABELS
    # ------------------------------------------------

    ax[2].set_xlabel(
        r"Reaction coordinate $(\xi)$",
        fontweight="bold"
    )

    ax[2].set_ylabel(
        "RDC",
        fontweight="bold"
    )

    ax[2].set_title(
        "Residual energetic manifold",
        fontweight="bold",
        fontsize=10
    )
    # ================================================================
    # COLORBAR
    # ================================================================

    cbar = fig.colorbar(
    cf,
    ax=ax[2],
    fraction=0.050,
    pad=0.06
    )

    cbar.set_label(r"$\Delta\Delta G_{\mathrm{exp}}$ (kcal mol$^{-1}$)")

    # ================================================================
    # AXIS STYLE
    # ================================================================

    for a in ax:

        a.spines["top"].set_visible(False)

        a.spines["right"].set_visible(False)

    # ================================================================
    # DYNAMIC LIMITS
    # ================================================================

    pad = 0.5

    ax[2].set_xlim(
        RC.min() - pad,
        RC.max() + pad
    )

    ax[2].set_ylim(
        RDC.min() - pad,
        RDC.max() + pad
    )

    # ================================================================
    # SAVE
    # ================================================================

    plt.savefig(
        "hierarchical_energetic_manifold.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.show()

def plot_three_panel_figure(
    df,
    y,
    publication,
    best_result,
    feature_list,
    output_file="figure_3panel.png"
):
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.cluster import KMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from scipy.stats import pearsonr
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay
    from scipy.ndimage import gaussian_filter

    from adjustText import adjust_text

    # ============================
    # INPUTS
    # ============================
    RC = np.asarray(publication["xi"])
    a = publication["a_global"]
    b = publication["b_global"]

    y = np.asarray(y)
    y_pred = np.asarray(best_result["y_pred"])

    # ============================
    # METRICS
    # ============================
    rp = pearsonr(y, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # ============================
    # RESIDUAL
    # ============================
    residual = y - (a * RC + b)

    # ============================
    # FEATURE SPACE
    # ============================
    X = df[feature_list].values
    Xs = StandardScaler().fit_transform(X)

    # ============================
    # REGIME DISCOVERY
    # ============================
    cluster_space = np.column_stack([RC, residual])
    labels = KMeans(n_clusters=2, random_state=0, n_init=20).fit_predict(cluster_space)

    # ============================
    # RDC (NO SCALING)
    # ============================
    lda = LinearDiscriminantAnalysis(n_components=1)
    RDC = lda.fit_transform(Xs, labels).flatten()

    RC_plot = RC
    RDC_plot = RDC

    # ============================
    # GRID (ANISOTROPIC)
    # ============================
    rc_range = RC_plot.max() - RC_plot.min()
    rdc_range = RDC_plot.max() - RDC_plot.min()

    aspect = rdc_range / (rc_range + 1e-12)

    nx = 120
    ny = int(np.clip(nx * aspect, 80, 300))

    gx = np.linspace(RC_plot.min(), RC_plot.max(), nx)
    gy = np.linspace(RDC_plot.min(), RDC_plot.max(), ny)

    GX, GY = np.meshgrid(gx, gy)

    # ============================
    # INTERPOLATION
    # ============================
    GZ = griddata((RC_plot, RDC_plot), y, (GX, GY), method="linear")

    # smoothing
    GZ = gaussian_filter(GZ, sigma=1.2)

    # ============================
    # MASK
    # ============================
    tri = Delaunay(np.column_stack([RC_plot, RDC_plot]))

    mask = tri.find_simplex(
        np.column_stack([GX.ravel(), GY.ravel()])
    ) < 0

    GZ_flat = GZ.ravel()
    GZ_flat[mask] = np.nan
    GZ = GZ_flat.reshape(GZ.shape)

    # ============================
    # COLOR NORMALIZATION
    # ============================
    vmin = np.percentile(y, 5)
    vmax = np.percentile(y, 95)

    # ============================
    # PANEL A
    # ============================
    X_scaled = StandardScaler().fit_transform(X)

    idx = np.argsort(RC)
    RC_sorted = RC[idx]
    X_sorted = X_scaled[idx]

    #fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    #plt.subplots_adjust(wspace=0.4)
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({

    "font.family": "serif",
        
    "mathtext.fontset": "stix",

    "axes.labelweight": "bold",

    "axes.titleweight": "bold",

    "axes.labelsize": 8,   # axis labels
    "axes.titlesize": 10,   # titles
    })

    

    fig = plt.figure(figsize=(11, 3.6))

    gs = GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.05],  # last column = colorbar
    wspace=0.35
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])   # colorbar axis

    features = [
        "volume_change",
        "steric_energy",
        "E_ca",
        "phi_sq",
        "E_perp"
    ]

    feature_labels = {

        "volume_change": r"$\Delta V$",

        "steric_energy": r"$E_{\mathrm{steric}}$",

        "E_ca": r"$E_{\mathrm{Ca}}$",

        "phi_sq": r"$\phi^2$",

        "E_perp": r"$E_{\perp}$"
    }


    ax = [ax0, ax1, ax2]
    for i in range(X_sorted.shape[1]):
        ax[0].plot(RC_sorted, X_sorted[:, i], alpha=0.2)
        ax[0].scatter(RC_sorted, X_sorted[:, i], s=15)

    for i, feat in enumerate(features):

        ax[0].scatter(RC_sorted, X_sorted[:, i], label=feature_labels[feat])

    ax[0].set_xlabel(r"Reaction coordinate $(\xi)$")
    ax[0].set_ylabel("Scaled descriptors")
    ax[0].set_title("Descriptor collapse")

    ax[0].legend(
        frameon=False,
        fontsize=8
    )

    # ============================
    # PANEL B
    # ============================
    ax[1].scatter(RC, y, s=40, edgecolor="black",
                         
        color="#8c510a",
        linewidth=0.45,
        alpha=0.90

                 )

    xg = np.linspace(RC.min(), RC.max(), 200)
    ax[1].plot(xg, a * xg + b, linewidth=2,         color="#bf812d")

    ax[1].text(
    0.05, 0.95,
    rf"$R_p = {rp:.2f}$" "\n"
    rf"$\mathrm{{RMSE}} = {rmse:.2f}$" "\n"
    r"$\Delta\Delta G = 5.41\,\xi + 0.21$",
    transform=ax[1].transAxes,
    va="top"
    )
    ax[1].set_xlabel(r"Reaction coordinate $(\xi)$")
    ax[1].set_title("Thermodynamic projection")




    # ============================
    # PANEL C
    # ============================
    cf = ax[2].contourf(GX, GY, GZ, levels=30, vmin=vmin, vmax=vmax, alpha=0.9)

    ax[2].contour(GX, GY, GZ, levels=12, linewidths=0.6, colors="black")

    sc = ax[2].scatter(
    RC_plot,
    RDC_plot,
    s=45,
    edgecolor="black",
    linewidth=0.6,
    c=y,                      # ← FIX
    cmap="coolwarm",# solid orange circles
    vmin=vmin,
    vmax=vmax,
    zorder=2
    )

    # ============================
    # MUTANT LABELS
    # ============================
    if "mutant" in df.columns:
        mutants = df["mutant"].astype(str).values
    else:
        mutants = np.array([f"M{i}" for i in range(len(df))])
    import matplotlib.patheffects as pe

    # ============================
    # ANNOTATE ALL POINTS
    # ============================
    texts = []
    for i in range(len(RC_plot)):
        texts.append(
            ax[2].text(
            RC_plot[i],
            RDC_plot[i],
            mutants[i],
            fontsize=5,
            color='black',
            zorder=1,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="white")],
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.1',
                alpha= 0.7
                )
            )
        )

    # ============================
    # ADJUST TEXT
    # ============================
    adjust_text(
        texts,
        ax=ax[2],
        expand_points=(2, 2),
        expand_text=(2, 2),
        force_text=1,
        arrowprops=dict(
        arrowstyle='-',
        lw=0.6,
        color='navy'
        )
    )

    # ============================
    # AXES
    # ============================
    ax[2].set_xlabel(r"Reaction coordinate $(\xi)$")
    ax[2].set_ylabel(r"Residual discriminant coordinate (RDC)")
    ax[2].set_title("Residual energy surface")
    ax[2].set_xlim(-0.3, 0.3)
    ax[2].set_ylim(-3, 3)

    # ============================
    # COLORBAR
    # ============================
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(r"$\Delta\Delta G_{\mathrm{exp}}$")

    # ============================
    # SAVE
    # ============================
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[SAVED] {output_file}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from tqdm import tqdm


# =========================================
# 3D PLOT
# =========================================
def save_3D_RC_plot(proj_df, output_file):

    norm = TwoSlopeNorm(
        vmin=proj_df["ddG_exp"].min(),
        vcenter=0.0,
        vmax=proj_df["ddG_exp"].max()
    )

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        proj_df["xi_struct"],
        proj_df["xi_elec"],
        proj_df["xi_dyn"],
        c=proj_df["ddG_exp"],
        cmap="coolwarm",
        norm=norm,
        s=85,
        edgecolors='black',
        linewidths=0.5
    )

    for i in tqdm(range(len(proj_df)), desc="3D labels"):
        row = proj_df.iloc[i]
        if abs(row["ddG_exp"]) > 0.0:
            ax.text(
                row["xi_struct"],
                row["xi_elec"],
                row["xi_dyn"],
                str(row["mutant"]),
                fontsize=6
            )

    ax.set_xlabel("Structural RC")
    ax.set_ylabel("Electrostatic RC")
    ax.set_zlabel("Dynamical RC")

    ax.view_init(elev=30, azim=210)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(r"$\Delta\Delta G_{\mathrm{exp}}$ (kcal mol$^{-1}$)")


    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()



# =========================================
# ATTRIBUTION
# =========================================
def run_attribution(proj_df, output_file):

    X = proj_df[["xi_struct", "xi_elec", "xi_dyn"]].values
    y = proj_df["ddG_exp"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    y_pred = model.predict(Xs)
    R2 = r2_score(y, y_pred)

    beta = model.coef_

    # permutation
    perm = permutation_importance(
        model, Xs, y,
        n_repeats=50,
        random_state=42
    )

    imp = perm.importances_mean

    # plot
    labels = ["Struct", "Elec", "Dyn"]
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(x - 0.2, beta, width=0.2, label="β")
    ax.bar(x, imp, width=0.2, label="Perm")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Contribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()

    # LaTeX output
    print("\n===== RESULTS =====\n")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Feature & $\beta$ & Perm \\")
    print(r"\hline")

    for i, f in enumerate(labels):
        print(f"{f} & {beta[i]:.2f} & {imp[i]:.2f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(f"\nR^2 = {R2:.2f}")


# =========================================
# MASTER PIPELINE
# =========================================
def run_full_RC_pipeline(proj_df, prefix="Figure3"):

    required = [
        "xi_struct",
        "xi_elec",
        "xi_dyn",
        "ddG_exp",
        "mutant"
    ]

    for col in required:
        if col not in proj_df.columns:
            raise ValueError(f"Missing column: {col}")

    proj_df = proj_df.dropna().reset_index(drop=True)

    print("\n[STEP 1] 3D plot")
    save_3D_RC_plot(proj_df, f"{prefix}_3A_3D.png")


    print("\n[STEP 3] Attribution")
    run_attribution(proj_df, f"{prefix}_3D_contributions.png")

    print("\n[PIPELINE COMPLETE]")

import numpy as np
import pandas as pd


def build_RC_projections(
    df,
    struct_features,
    elec_features,
    dyn_features
):
    """
    Build reaction coordinate projections manually.

    Parameters
    ----------
    df : pd.DataFrame
    struct_features : list
    elec_features : list
    dyn_features : list
    """

    df = df.copy()

    # ---------- validation ----------
    for group, name in zip(
        [struct_features, elec_features, dyn_features],
        ["Structural", "Electrostatic", "Dynamical"]
    ):
        for f in group:
            if f not in df.columns:
                raise ValueError(f"{name} feature missing: {f}")

    # ---------- helper ----------
    def compute_projection(features):
        if len(features) == 0:
            return np.zeros(len(df))
        X = df[features].values
        return np.mean(X, axis=1)

    # ---------- projections ----------
    df["xi_struct"] = compute_projection(struct_features)
    df["xi_elec"]   = compute_projection(elec_features)
    df["xi_dyn"]    = compute_projection(dyn_features)

    # ---------- report ----------
    print("\n===== RC FEATURE SUMMARY =====\n")

    print("Structural:", struct_features)
    print("Electrostatic:", elec_features)
    print("Dynamical:", dyn_features)

    print("\nVariance:")
    print(f"xi_struct = {np.var(df['xi_struct']):.3f}")
    print(f"xi_elec   = {np.var(df['xi_elec']):.3f}")
    print(f"xi_dyn    = {np.var(df['xi_dyn']):.3f}")

    return df


def main():

    global OUTPUT_DIR

    OUTPUT_DIR = Path(CFG.output_dir)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    section("LOAD DATA")

    enm = load_dataset(
        "enm_all_raw_features_best_parameters.csv"
    )

    elc = load_dataset(
        "raw_features.csv"
    )

    struct = load_dataset(
        "structural_features_full.csv"
    )


    df = merge_feature_domains(
        enm,
        elc,
        struct,
    )
    print(df)
    df["ddG_exp"] = (
        df.filter(regex="ddG_exp")
        .iloc[:, 0]
    )

    y = df["ddG_exp"].values

    f_enm = [
        'comm_eff',
        'f_collective',
        'f_asym',
        'prs_sens',
        'f_entropy',
        'mut_ca_coupling',
        'prs_eff',
        'f_local',
        'sq_fluct'
    ]

    f_elc = [
        "E_ca",
        "phi_sq",
        "dq_phi",
        "born",
        "E_mag",
        "E_parallel",
        "E_perp"
    ]

    f_struct = [
        "inv_distance",
        "volume_change",
        "steric_energy",
        "local_density",
        "coordination_number",
        "backbone_angle"
    ]



    feature_names = (
        f_struct +
        f_elc +
        f_enm )

    #all_results = exhaustive_subset_search(df,y,feature_names)

    #results_df = build_results_table(all_results)

    #save_dataframe(results_df,"strict_best_subset_models.csv")

    #results_df.to_json(OUTPUT_DIR / "strict_best_subset_models.json",orient="records",indent=2)

    #print_top_models(results_df)

    #best_result = all_results[0]

    #section("BEST MODEL")

    #print(best_result["features"])

    #print(f"Rp={best_result['metrics']['Rp']:.3f}")
    section("USING FIXED FEATURE SET (NO SUBSET SEARCH)")

    # ---- choose features (same as manifold for consistency)
    selected_features = [
    'volume_change',
    'steric_energy',
    'E_ca',
    'phi_sq',
    'E_perp'
    ]

    print("Selected features:")
    print(selected_features)

    # ---- evaluate once using strict pipeline
    best_result = evaluate_subset_strict(
    df,
    y,
    selected_features
    )

    #print(f"Rp = {best_result['metrics']['Rp']:.3f}")

    publication = build_publication_model(df,y,best_result)

    #print_publication_model(best_result,publication)

    #coef_df = compute_rc_uncertainty(publication,len(y))

    #print_rc_uncertainty(coef_df,publication)

    #save_dataframe(coef_df,"canonical_rc_coefficients.csv")
    #run_correlation_analysis(df,feature_names)

    manifold_features = [
        'volume_change',
        'steric_energy',
        'E_ca',
        'phi_sq',
        'E_perp'
    ]

    #manifold = run_manifold_geometry(df,manifold_features)

    #scatter_embedding(manifold["X_pca"][:, 0],manifold["X_pca"][:, 1],y,"PC1","PC2","PCA Energy Manifold","ΔΔG","figure_pca_ddG.png")

    #benchmark_models(df,manifold_features,y)

    #analyze_thermodynamic_regimes(manifold["X_iso"],publication["xi"],y)

    #generate_energetic_manifold_figure(df,y,f_struct,f_elc,f_enm,publication)

    #hierarchical_thermodynamic_analysis(df,y)


    #proj_df = run_sector_projection_analysis(df,publication)
    struct_features = [
    "inv_distance",
    "volume_change",
    "steric_energy"
    ]

    elec_features = [
    "E_ca",
    "phi_sq"
    ]

    #dyn_features = ['comm_eff', 'f_collective', 'f_entropy', 'prs_sens']

    #proj_df = build_RC_projections(df,struct_features,elec_features,dyn_features)
    #run_full_RC_pipeline(proj_df){re
    plot_three_panel_figure(
    df,
    y,
    publication,
    best_result,
    selected_features,
    output_file="figure_3panel.png")
    section("PIPELINE COMPLETE")
    


if __name__ == "__main__":

    main()
