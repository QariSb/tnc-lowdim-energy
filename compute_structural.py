# Physics-Informed Structural Model Pipeline

#!/usr/bin/env python3
# ============================================================
# PHYSICS-INFORMED STRUCTURAL MODEL
# ============================================================

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from Bio.PDB import PDBParser
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-6
VOL_SCALE = 200.0
DEFAULT_STERIC_CUTOFF = 6.0
DEFAULT_COORD_CUTOFF = 6.0
ATOMWISE_CUTOFF = 6.0

STERIC_CUTOFFS = [6.0, 7.0, 8.0, 9.0, 10.0]
COORD_CUTOFFS = [4.0, 5.0, 6.0, 7.0]

BASE_FEATURES = [
    "inv_distance",
    "steric_energy",
    "volume_change"
]

AUGMENTED_FEATURES = [
    "local_density",
    "coordination_number",
    "radial_position",
    "backbone_angle"
]

# ============================================================
# INPUT
# ============================================================

PDB_FILE = "WT_amber.pdb"

MUTATIONS = [
    ("Y5H",5), ("A8V",8), ("F20Q",20), ("A23Q",23), ("L29Q",29),
    ("A31S",31), ("S37G",37), ("E40A",40), ("V44Q",44), ("M45Q",45),
    ("L48Q",48), ("Q50R",50), ("L57Q",57), ("E59D",59), ("I61Q",61),
    ("D67A",67), ("D73A",73), ("D73N",73), ("D75Y",75),
    ("V79Q",79), ("M81Q",81), ("C84Y",84)
]

EXP_DATA = {
    "WT":-6.56,"Y5H":-6.44,"A8V":-6.66,"F20Q":-6.97,"A23Q":-7.59,
    "L29Q":-6.89,"A31S":-6.93,"S37G":-6.67,"E40A":-5.87,
    "V44Q":-8.19,"M45Q":-7.46,"L48Q":-7.63,"Q50R":-7.31,
    "L57Q":-5.99,"E59D":-6.23,"I61Q":-5.39,"D67A":-5.09,
    "D73A":-5.09,"D73N":-5.69,"D75Y":-6.14,"V79Q":-6.63,
    "M81Q":-7.28,"C84Y":-7.37
}

AA_VOLUME = {
    "A":91.5,"V":141.7,"L":167.9,"I":168.8,"F":203.4,"Y":203.6,"W":237.6,
    "D":124.5,"E":155.1,"N":135.2,"Q":161.1,"S":99.1,"T":122.1,"H":167.3,
    "K":171.3,"M":170.8,"C":105.6,"G":66.4,"P":129.3
}

# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass
class StructuralContext:
    model: object
    residue_map: dict
    all_ca: list
    ca_dict: dict
    ref_point: np.ndarray
    coord_shell: list

# ============================================================
# UTILITIES
# ============================================================


def print_banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rp, _ = pearsonr(y_true, y_pred)

    return {
        "RMSE": rmse,
        "R2": r2,
        "Rp": rp
    }



def build_residue_map(model):
    residue_map = {}

    for chain in model:
        for residue in chain:
            residue_map[residue.id[1]] = residue

    return residue_map



def get_ca_atoms(model):
    ca_atoms = []

    for chain in model:
        for residue in chain:
            if "CA" in residue:
                ca_atoms.append((residue.id[1], residue["CA"].coord))

    return ca_atoms



def find_calcium(model):
    for atom in model.get_atoms():
        parent = atom.get_parent()

        if parent.id[0] != " ":
            if atom.element.strip().upper() == "CA":
                return atom.coord

    raise ValueError("Calcium ion not found.")



def build_coordination_shell(all_ca, ref_point, cutoff=DEFAULT_COORD_CUTOFF):
    shell = []

    for rid, coord in all_ca:
        if np.linalg.norm(coord - ref_point) < cutoff:
            shell.append((rid, coord))

    return shell



def inverse_distance(coord, ref_point):
    dist = np.linalg.norm(coord - ref_point)
    return 1.0 / (dist + EPS)



def volume_change(wt_aa, mut_aa):
    wt_vol = AA_VOLUME.get(wt_aa, 120)
    mut_vol = AA_VOLUME.get(mut_aa, 120)

    return (mut_vol - wt_vol) / VOL_SCALE



def residue_neighbor_count(res_coord, all_ca, resi, cutoff):
    return sum(
        np.linalg.norm(coord - res_coord) < cutoff
        for rid, coord in all_ca
        if rid != resi
    )



def atomwise_neighbor_count(model, res_coord, resi, cutoff):
    neighbors = 0

    for chain in model:
        for residue in chain:
            if residue.id[1] == resi:
                continue

            for atom in residue:
                if np.linalg.norm(atom.coord - res_coord) < cutoff:
                    neighbors += 1

    return neighbors



def coordination_strength(res_coord, coord_shell, resi):
    strength = 0.0

    for rid, coord in coord_shell:
        if rid == resi:
            continue

        d = np.linalg.norm(coord - res_coord)
        strength += 1.0 / ((d + EPS) ** 2)

    return strength



def run_loocv_linear(X, y):
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))

    for train, test in loo.split(X):
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])

        reg = LinearRegression()
        reg.fit(Xtr, y[train])

        predictions[test] = reg.predict(Xte)

    return predictions



def run_loocv_ridge(X, y, alpha=0.5):
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))

    for train, test in loo.split(X):
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])

        reg = Ridge(alpha=alpha)
        reg.fit(Xtr, y[train])

        predictions[test] = reg.predict(Xte)

    return predictions

# ============================================================
# STRUCTURE LOADING
# ============================================================


def load_structure(pdb_file):
    print("\n[INFO] Loading structure...")

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("WT", pdb_file)[0]

    all_ca = get_ca_atoms(model)
    residue_map = build_residue_map(model)
    ca_dict = {rid: coord for rid, coord in all_ca}

    ref_point = find_calcium(model)
    coord_shell = build_coordination_shell(all_ca, ref_point)

    print(f"[INFO] Total CA residues: {len(all_ca)}")
    print(f"[INFO] Coordination shell size: {len(coord_shell)}")

    return StructuralContext(
        model=model,
        residue_map=residue_map,
        all_ca=all_ca,
        ca_dict=ca_dict,
        ref_point=ref_point,
        coord_shell=coord_shell
    )

# ============================================================
# FEATURE GENERATION
# ============================================================


def compute_base_feature_table(context):
    print("\n[INFO] Computing structural features...")

    rows = []

    for mut, resi in tqdm(MUTATIONS, desc="Base features"):
        residue = context.residue_map[resi]

        wt_aa = mut[0]
        mut_aa = mut[-1]

        res_coord = residue["CA"].coord

        inv_dist = inverse_distance(res_coord, context.ref_point)

        vol_change = volume_change(wt_aa, mut_aa)

        neighbors = residue_neighbor_count(
            res_coord,
            context.all_ca,
            resi,
            DEFAULT_STERIC_CUTOFF
        )

        steric_energy = vol_change * (neighbors + 1)

        rows.append({
            "mutant": mut,
            "dG_exp": EXP_DATA[mut],
            "inv_distance": inv_dist,
            "steric_energy": steric_energy,
            "volume_change": vol_change,
        })

    df = pd.DataFrame(rows)

    wt_value = EXP_DATA["WT"]
    df["ddG_exp"] = df["dG_exp"] - wt_value

    return df

# ============================================================
# STANDARD MODEL
# ============================================================


def evaluate_base_model(df):
    print("\n[INFO] Running LOOCV...")

    X = df[BASE_FEATURES].values
    y = df["ddG_exp"].values

    predictions = run_loocv_linear(X, y)
    metrics = compute_metrics(y, predictions)

    print_banner("FINAL ΔΔG MODEL PERFORMANCE")

    print(f"RMSE = {metrics['RMSE']:.2f}")
    print(f"R²   = {metrics['R2']:.2f}")
    print(f"Rp   = {metrics['Rp']:.2f}")

    return predictions, metrics

# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================


def robustness_analysis(context):
    print("\n[INFO] Running robustness analysis...")

    neighbor_counts = []

    for _, resi in MUTATIONS:
        residue = context.residue_map[resi]
        res_coord = residue["CA"].coord

        n = residue_neighbor_count(
            res_coord,
            context.all_ca,
            resi,
            cutoff=8.0
        )

        neighbor_counts.append(n)

    print("\n[VARIABILITY]")
    print(
        f"Neighbor count: min={min(neighbor_counts)}, "
        f"max={max(neighbor_counts)}, "
        f"mean={np.mean(neighbor_counts):.2f}"
    )

    print(
        f"Coordination shell size: {len(context.coord_shell)} residues"
    )

    results = []

    print("\n[INFO] Cutoff sensitivity analysis...")

    for steric_cut in STERIC_CUTOFFS:
        for coord_cut in COORD_CUTOFFS:

            coord_shell = build_coordination_shell(
                context.all_ca,
                context.ref_point,
                cutoff=coord_cut
            )

            rows = []

            for mut, resi in MUTATIONS:
                residue = context.residue_map[resi]

                wt_aa = mut[0]
                mut_aa = mut[-1]

                res_coord = residue["CA"].coord

                inv_dist = inverse_distance(
                    res_coord,
                    context.ref_point
                )

                vol_change = volume_change(wt_aa, mut_aa)

                neighbors = residue_neighbor_count(
                    res_coord,
                    context.all_ca,
                    resi,
                    steric_cut
                )

                steric_energy = vol_change * (neighbors + 1)


                rows.append([
                    inv_dist,
                    steric_energy,
                    vol_change,
                    EXP_DATA[mut] - EXP_DATA["WT"]
                ])

            data = np.array(rows)

            X = data[:, :3]
            y = data[:, 3]

            preds = run_loocv_linear(X, y)
            metrics = compute_metrics(y, preds)

            results.append({
                "steric_cut": steric_cut,
                "coord_cut": coord_cut,
                "Rp": metrics["Rp"]
            })

    print("\n[CUTOFF ROBUSTNESS RESULTS]")

    for result in results:
        print(
            f"Steric={result['steric_cut']}Å | "
            f"Coord={result['coord_cut']}Å | "
            f"Rp={result['Rp']:.2f}"
        )

    rp_values = [r["Rp"] for r in results]

    print("\n[SUMMARY]")
    print(f"Rp range: {min(rp_values):.3f} – {max(rp_values):.3f}")
    print(
        f"ΔRp max deviation: "
        f"{max(rp_values) - min(rp_values):.3f}"
    )

    return pd.DataFrame(results)

# ============================================================
# STERIC COMPARISON
# ============================================================


def compute_steric_dataset(context, atomwise=False):
    rows = []

    label = "atom" if atomwise else "residue"

    for mut, resi in tqdm(MUTATIONS, desc=f"Features ({label})"):
        residue = context.residue_map[resi]

        wt_aa = mut[0]
        mut_aa = mut[-1]

        res_coord = residue["CA"].coord

        inv_dist = inverse_distance(res_coord, context.ref_point)
        vol_change = volume_change(wt_aa, mut_aa)

        if atomwise:
            neighbors = atomwise_neighbor_count(
                context.model,
                res_coord,
                resi,
                ATOMWISE_CUTOFF
            )
        else:
            neighbors = residue_neighbor_count(
                res_coord,
                context.all_ca,
                resi,
                ATOMWISE_CUTOFF
            )

        steric_energy = vol_change * (neighbors + 1)

        coord_strength = coordination_strength(
            res_coord,
            context.coord_shell,
            resi
        )

        rows.append([
            inv_dist,
            steric_energy,
            vol_change,
            coord_strength,
            EXP_DATA[mut] - EXP_DATA["WT"]
        ])

    return np.array(rows)



def compare_steric_models(context):
    print(
        "\n[INFO] Comparing residue-wise vs atom-wise steric perturbation..."
    )

    data_residue = compute_steric_dataset(context, atomwise=False)
    data_atom = compute_steric_dataset(context, atomwise=True)

    X_residue = data_residue[:, :4]
    X_atom = data_atom[:, :4]
    y = data_residue[:, 4]

    pred_residue = run_loocv_linear(X_residue, y)
    pred_atom = run_loocv_linear(X_atom, y)

    metrics_residue = compute_metrics(y, pred_residue)
    metrics_atom = compute_metrics(y, pred_atom)

    print_banner("STERIC REPRESENTATION COMPARISON")

    print(
        f"Residue-wise  → RMSE={metrics_residue['RMSE']:.2f}, "
        f"R²={metrics_residue['R2']:.2f}, "
        f"Rp={metrics_residue['Rp']:.2f}"
    )

    print(
        f"Atom-wise     → RMSE={metrics_atom['RMSE']:.2f}, "
        f"R²={metrics_atom['R2']:.2f}, "
        f"Rp={metrics_atom['Rp']:.2f}"
    )

    print("\n[DIFFERENCE]")

    print(
        f"ΔRp = "
        f"{abs(metrics_residue['Rp'] - metrics_atom['Rp']):.3f}"
    )

    print(
        f"ΔRMSE = "
        f"{abs(metrics_residue['RMSE'] - metrics_atom['RMSE']):.3f}"
    )




# ============================================================
# ALPHA VS BETA CARBON REPRESENTATION COMPARISON
# ============================================================

BETA_CUTOFF = 6.0


def get_reference_atom(residue):
    """
    Use CB atom when available.
    Fallback to CA for glycine or missing CB.
    """

    if "CB" in residue:
        return residue["CB"].coord

    return residue["CA"].coord



def build_atom_representation_dataset(context, use_beta=False):
    rows = []

    label = "CB" if use_beta else "CA"

    for mut, resi in tqdm(
        MUTATIONS,
        desc=f"{label} representation"
    ):

        residue = context.residue_map[resi]

        wt_aa = mut[0]
        mut_aa = mut[-1]

        if use_beta:
            res_coord = get_reference_atom(residue)
        else:
            res_coord = residue["CA"].coord

        inv_dist = inverse_distance(
            res_coord,
            context.ref_point
        )

        vol_change = volume_change(
            wt_aa,
            mut_aa
        )

        neighbors = residue_neighbor_count(
            res_coord,
            context.all_ca,
            resi,
            BETA_CUTOFF
        )

        steric_energy = vol_change * (neighbors + 1)

        coord_strength = coordination_strength(
            res_coord,
            context.coord_shell,
            resi
        )

        rows.append([
            inv_dist,
            steric_energy,
            vol_change,
            coord_strength,
            EXP_DATA[mut] - EXP_DATA["WT"]
        ])

    return np.array(rows)



def compare_alpha_beta_representations(context):

    print(
        "\n[INFO] Comparing alpha-carbon vs beta-carbon representations..."
    )

    data_ca = build_atom_representation_dataset(
        context,
        use_beta=False
    )

    data_cb = build_atom_representation_dataset(
        context,
        use_beta=True
    )

    X_ca = data_ca[:, :4]
    X_cb = data_cb[:, :4]

    y = data_ca[:, 4]

    pred_ca = run_loocv_linear(X_ca, y)
    pred_cb = run_loocv_linear(X_cb, y)

    metrics_ca = compute_metrics(y, pred_ca)
    metrics_cb = compute_metrics(y, pred_cb)

    print_banner("CA VS CB REPRESENTATION")

    print(
        f"Alpha-carbon (CA) → "
        f"RMSE={metrics_ca['RMSE']:.2f}, "
        f"R²={metrics_ca['R2']:.2f}, "
        f"Rp={metrics_ca['Rp']:.2f}"
    )

    print(
        f"Beta-carbon  (CB) → "
        f"RMSE={metrics_cb['RMSE']:.2f}, "
        f"R²={metrics_cb['R2']:.2f}, "
        f"Rp={metrics_cb['Rp']:.2f}"
    )

    print("\n[REPRESENTATION DIFFERENCE]")

    print(
        f"ΔRp   = "
        f"{abs(metrics_ca['Rp'] - metrics_cb['Rp']):.3f}"
    )

    print(
        f"ΔRMSE = "
        f"{abs(metrics_ca['RMSE'] - metrics_cb['RMSE']):.3f}"
    )

    if metrics_cb["Rp"] > metrics_ca["Rp"]:
        print(
            "\n[INTERPRETATION] "
            "CB representation captures side-chain packing more effectively."
        )
    else:
        print(
            "\n[INTERPRETATION] "
            "CA representation provides a more stable global structural signal."
        )




# ============================================================
# REACTION COORDINATE MODEL
# ============================================================


def run_reaction_coordinate_model(df):
    print("\n[INFO] Running LOOCV with reaction coordinate...")

    X = df[BASE_FEATURES].values
    y = df["ddG_exp"].values

    loo = LeaveOneOut()

    y_pred = np.zeros(len(y))
    phi_all = np.zeros(len(y))

    for train, test in tqdm(loo.split(X), total=len(y), desc="RC LOOCV"):
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])

        ytr = y[train]

        reg1 = LinearRegression()
        reg1.fit(Xtr, ytr)

        w = reg1.coef_
        w = w / np.linalg.norm(w)

        phi_tr = Xtr @ w
        phi_te = Xte @ w

        reg2 = LinearRegression()
        reg2.fit(phi_tr.reshape(-1, 1), ytr)

        y_pred[test] = reg2.predict(phi_te.reshape(-1, 1))
        phi_all[test] = phi_te

    metrics = compute_metrics(y, y_pred)

    print_banner("FINAL ΔΔG MODEL (RC CONSISTENT)")

    print(f"RMSE = {metrics['RMSE']:.2f}")
    print(f"R²   = {metrics['R2']:.2f}")
    print(f"Rp   = {metrics['Rp']:.2f}")

    return phi_all, y_pred, metrics

# ============================================================
# RC COEFFICIENT ANALYSIS
# ============================================================


def analyze_rc_coefficients(df):
    print(
        "\n[INFO] Estimating reaction coordinate coefficients..."
    )

    X = df[BASE_FEATURES].values
    y = df["ddG_exp"].values

    loo = LeaveOneOut()

    coefficients = []
    intercepts = []

    for train, test in loo.split(X):
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(X[train])
        ytr = y[train]

        reg = LinearRegression()
        reg.fit(Xtr, ytr)

        coefficients.append(reg.coef_)
        intercepts.append(reg.intercept_)

    coefficients = np.array(coefficients)


    # ============================================================
    # NORMALIZED RC EQUATION WITH UNCERTAINTY
    # ============================================================

    # --------------------------------------------
    # normalize each coefficient vector
    # --------------------------------------------

    coefficients_normalized = []

    for w in coefficients:

        w_norm = w / (norm(w) + EPS)

        coefficients_normalized.append(w_norm)

    coefficients_normalized = np.array(
        coefficients_normalized
    )

    # --------------------------------------------
    # mean and std of normalized weights
    # --------------------------------------------

    w_mean_norm = np.mean(
        coefficients_normalized,
        axis=0
    )

    w_std_norm = np.std(
        coefficients_normalized,
        axis=0
    )

    # --------------------------------------------
    # print normalized RC equation
    # --------------------------------------------

    print("\n[NORMALIZED REACTION COORDINATE]\n")

    feature_labels = [
        r"\tilde{d}^{-1}",
        r"\tilde{E}_{\mathrm{steric}}",
        r"\Delta \tilde{V}"
    ]

    terms = []

    for i, label in enumerate(feature_labels):

        coef = w_mean_norm[i]
        err = w_std_norm[i]

        sign = "+" if coef >= 0 else "-"

        term = (
            f"{sign} "
            f"({abs(coef):.3f} \\pm {err:.3f})"
            f"{label}"
        )

        terms.append(term)

    equation = " ".join(terms)

    if equation.startswith("+"):
        equation = equation[2:]

    print(
        r"$\xi = "
        + equation
        + r"$"
    )
    intercepts = np.array(intercepts)

    w_mean = np.mean(coefficients, axis=0)
    w_std = np.std(coefficients, axis=0)

    b_mean = np.mean(intercepts)
    b_std = np.std(intercepts)

    z_scores = w_mean / (w_std + EPS)

    cosine_similarities = []

    for w in coefficients:
        similarity = np.dot(w, w_mean) / (
            norm(w) * norm(w_mean)
        )

        cosine_similarities.append(similarity)

    cosine_similarities = np.array(cosine_similarities)

    print_banner("REACTION COORDINATE (LOOCV)")

    for i, feature in enumerate(BASE_FEATURES):
        print(
            f"{feature:15s}  "
            f"coef = {w_mean[i]: .4f} ± {w_std[i]:.4f} | "
            f"z ≈ {z_scores[i]:.2f}"
        )

    print(
        f"{'Intercept':15s}  "
        f"coef = {b_mean:.4f} ± {b_std:.4f}"
    )

    print("\n[STABILITY]")

    print(
        f"Mean cosine similarity: "
        f"{cosine_similarities.mean():.4f}"
    )

    print(
        f"Min cosine similarity : "
        f"{cosine_similarities.min():.4f}"
    )

    # ============================================================
    # LATEX-FORMATTED RC EQUATION
    # ============================================================

    print("\n[REACTION COORDINATE EQUATION]\n")

    feature_labels = [
        r"\tilde{d}^{-1}",
        r"\tilde{E}_{\mathrm{steric}}",
        r"\Delta \tilde{V}"
    ]

    terms = []

    for i, label in enumerate(feature_labels):

        coef = w_mean[i]
        err = w_std[i]

        sign = "+" if coef >= 0 else "-"

        term = (
            f"{sign} "
            f"({abs(coef):.3f} \\pm {err:.3f})"
            f"{label}"
        )

        terms.append(term)

    equation = " ".join(terms)

    if equation.startswith("+"):
        equation = equation[2:]

    print(
        r"$\xi = "
        + equation
        + r"$"
    )



    # ============================================================
    # NORMALIZED RC EQUATION
    # ============================================================

    # --------------------------------------------------------
    # NORMALIZE MEAN RC VECTOR
    # --------------------------------------------------------

    w_norm = w_mean / np.linalg.norm(w_mean)

    print("\n[NORMALIZED REACTION COORDINATE]\n")

    feature_labels = [
        r"\tilde{d}^{-1}",
        r"\tilde{E}_{\mathrm{steric}}",
        r"\Delta \tilde{V}"
    ]

    terms = []

    for i, label in enumerate(feature_labels):

        coef = w_norm[i]

        sign = "+" if coef >= 0 else "-"

        term = (
            f"{sign} "
            f"{abs(coef):.3f}"
            f"{label}"
        )

        terms.append(term)

    equation = " ".join(terms)

    if equation.startswith("+"):
        equation = equation[2:]

    print(
        r"$\xi = "
        + equation
        + r"$"
    )

    print(
        f"\n||w|| = "
        f"{np.linalg.norm(w_norm):.4f}"
    )

    return {
        "w_mean": w_mean,
        "w_std": w_std,
        "b_mean": b_mean,
        "b_std": b_std,
        "cosine_similarities": cosine_similarities
    }

# ============================================================
# AUGMENTED DESCRIPTORS
# ============================================================


def local_density(res_coord, all_ca, sigma=4.0):
    return sum(
        np.exp(-(np.linalg.norm(coord - res_coord) ** 2) / (2 * sigma ** 2))
        for _, coord in all_ca
        if np.linalg.norm(coord - res_coord) > 0
    )



def coordination_number(res_coord, all_ca):
    shells = [6.0, 8.0, 10.0]
    weights = [1.0, 0.5, 0.25]

    cn = 0.0

    for radius, weight in zip(shells, weights):
        cn += weight * sum(
            np.linalg.norm(coord - res_coord) < radius
            for _, coord in all_ca
        )

    return cn



def radial_position(res_coord, all_ca):
    coords = np.array([coord for _, coord in all_ca])
    centroid = np.mean(coords, axis=0)

    return np.linalg.norm(res_coord - centroid)



def backbone_angle(resi, ca_dict):
    if resi - 1 not in ca_dict:
        return 0.0

    if resi + 1 not in ca_dict:
        return 0.0

    p1 = ca_dict[resi - 1]
    p2 = ca_dict[resi]
    p3 = ca_dict[resi + 1]

    v1 = p1 - p2
    v2 = p3 - p2

    cos_theta = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + EPS
    )

    return np.arccos(np.clip(cos_theta, -1, 1))



def augment_feature_table(df, context):
    print("\n[INFO] Computing additional structural descriptors...")

    rows = []

    for _, resi in tqdm(MUTATIONS, desc="Augmented features"):
        residue = context.residue_map[resi]
        coord = residue["CA"].coord

        rows.append({
            "local_density": local_density(coord, context.all_ca),
            "coordination_number": coordination_number(
                coord,
                context.all_ca
            ),
            "radial_position": radial_position(
                coord,
                context.all_ca
            ),
            "backbone_angle": backbone_angle(
                resi,
                context.ca_dict
            )
        })

    df_aug = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1
    )

    return df_aug

# ============================================================
# FEATURE SEARCH
# ============================================================


def evaluate_feature_set(df, feature_list):
    X = df[feature_list].values
    y = df["ddG_exp"].values

    predictions = run_loocv_linear(X, y)

    return compute_metrics(y, predictions)
    


def feature_combination_search(df):
    print("\n[INFO] Running feature combination search...")

    combinations = []

    for k in range(len(AUGMENTED_FEATURES) + 1):
        for combo in itertools.combinations(AUGMENTED_FEATURES, k):
            combinations.append(list(combo))

    results = []

    for combo in tqdm(combinations, desc="Feature search"):
        feature_set = BASE_FEATURES + combo

        metrics = evaluate_feature_set(df, feature_set)

        results.append({
            "features": "+".join(feature_set),
            "n_features": len(feature_set),
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "Rp": metrics["Rp"]
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Rp", ascending=False)

    print_banner("TOP FEATURE COMBINATIONS")
    print(results_df.head(10))

    return results_df

# ============================================================
# RC WITHOUT OLS
# ============================================================


def rc_without_ols(df, w_mean):
    print("\n[INFO] Computing reaction coordinate without OLS...")

    X = df[BASE_FEATURES].values
    y = df["ddG_exp"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    phi = X_scaled @ w_mean

    reg = LinearRegression()
    reg.fit(phi.reshape(-1, 1), y)

    y_fit = reg.predict(phi.reshape(-1, 1))

    rp, _ = pearsonr(phi, y)

    print("\n[RC PERFORMANCE]")
    print(f"Rp (φ vs ΔΔG) = {rp:.3f}")

    return phi, y_fit

# ============================================================
# STRUCTURAL RC FIT
# ============================================================


def structural_rc_fit(phi_all, y):
    print("\n[INFO] Fitting Structural ΔG = aφ + b...")

    phi = phi_all.copy()
    phi = (phi - np.mean(phi)) / np.std(phi)

    X_design = sm.add_constant(phi)

    model = sm.OLS(y, X_design).fit()

    a = model.params[1]
    b = model.params[0]

    a_se = model.bse[1]
    b_se = model.bse[0]

    a_p = model.pvalues[1]
    b_p = model.pvalues[0]

    print_banner("STRUCTURAL ΔG = aφ + b")

    print(
        f"a (slope)     = {a:.4f} ± {a_se:.4f} | p = {a_p:.3e}"
    )

    print(
        f"b (intercept) = {b:.4f} ± {b_se:.4f} | p = {b_p:.3e}"
    )

    print("\nModel stats:")

    print(f"R² = {model.rsquared:.4f}")

    print(
        f"F-statistic = {model.fvalue:.4f}, "
        f"p = {model.f_pvalue:.3e}"
    )

    return model

# ============================================================
# OUTPUT
# ============================================================


def save_outputs(df_base, df_augmented, feature_results):
    df_base.to_csv("structural_features.csv", index=False)

    print(
        "\n[INFO] Base features written to: structural_features.csv"
    )

    df_augmented.to_csv("structural_features_full.csv", index=False)

    print(
        "[INFO] Full feature table written to: "
        "structural_features_full.csv"
    )

    feature_results.to_csv(
        "structural_feature_combination_search.csv",
        index=False
    )

    print(
        "[INFO] Feature search written to: "
        "structural_feature_combination_search.csv"
    )



# ============================================================
# STATISTICAL COMPARISON:
# FULL MULTIVARIATE vs RANK-1 RC MODEL
# ============================================================

from scipy.stats import ttest_rel, wilcoxon


def compare_full_vs_rc_model(df):
    print(
        "\n[INFO] Comparing full multivariate model "
        "vs rank-1 reaction-coordinate model..."
    )

    X = df[BASE_FEATURES].values
    y = df["ddG_exp"].values

    # --------------------------------------------------------
    # FULL MODEL
    # --------------------------------------------------------

    pred_full = run_loocv_linear(X, y)

    metrics_full = compute_metrics(y, pred_full)

    # --------------------------------------------------------
    # RC MODEL
    # --------------------------------------------------------

    _, pred_rc, metrics_rc = run_reaction_coordinate_model(df)

    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residual_full = np.abs(y - pred_full)
    residual_rc = np.abs(y - pred_rc)

    # --------------------------------------------------------
    # PAIRED TESTS
    # --------------------------------------------------------

    t_stat, t_p = ttest_rel(
        residual_full,
        residual_rc
    )

    try:
        w_stat, w_p = wilcoxon(
            residual_full,
            residual_rc
        )
    except:
        w_stat, w_p = np.nan, np.nan

    # --------------------------------------------------------
    # PERFORMANCE DIFFERENCE
    # --------------------------------------------------------

    delta_rp = metrics_rc["Rp"] - metrics_full["Rp"]
    delta_rmse = metrics_rc["RMSE"] - metrics_full["RMSE"]

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_banner(
        "FULL MODEL vs RC MODEL COMPARISON"
    )

    print(
        f"Full model  → "
        f"RMSE={metrics_full['RMSE']:.3f}, "
        f"R²={metrics_full['R2']:.3f}, "
        f"Rp={metrics_full['Rp']:.3f}"
    )

    print(
        f"RC model    → "
        f"RMSE={metrics_rc['RMSE']:.3f}, "
        f"R²={metrics_rc['R2']:.3f}, "
        f"Rp={metrics_rc['Rp']:.3f}"
    )

    print("\n[DIFFERENCE]")

    print(f"ΔRp   = {delta_rp:.4f}")
    print(f"ΔRMSE = {delta_rmse:.4f}")

    print("\n[PAIRED RESIDUAL TESTS]")

    print(
        f"Paired t-test      : "
        f"t = {t_stat:.4f}, p = {t_p:.4e}"
    )

    print(
        f"Wilcoxon signed-rank: "
        f"W = {w_stat:.4f}, p = {w_p:.4e}"
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print("\n[INTERPRETATION]")

    if t_p > 0.05:
        print(
            "No statistically significant difference "
            "between the full multivariate model and "
            "the rank-1 reaction-coordinate model."
        )

        print(
            "The dominant predictive signal is therefore "
            "effectively low-dimensional."
        )

    else:
        print(
            "The RC projection produces statistically "
            "different residuals relative to the full model."
        )

    return {
        "full_metrics": metrics_full,
        "rc_metrics": metrics_rc,
        "delta_rp": delta_rp,
        "delta_rmse": delta_rmse,
        "t_p": t_p,
        "w_p": w_p
    }

# ============================================================
# MAIN
# ============================================================


def main():
    context = load_structure(PDB_FILE)

    df_base = compute_base_feature_table(context)

    evaluate_base_model(df_base)

    robustness_analysis(context)

    compare_steric_models(context)

    compare_alpha_beta_representations(context)

    phi_all, _, _ = run_reaction_coordinate_model(df_base)

    rc_results = analyze_rc_coefficients(df_base)

    df_augmented = augment_feature_table(df_base, context)

    feature_results = feature_combination_search(df_augmented)

    rc_without_ols(df_base, rc_results["w_mean"])

    phi_all, _, _ = run_reaction_coordinate_model(df_base)

    compare_full_vs_rc_model(df_base)

    structural_rc_fit(
        phi_all,
        df_base["ddG_exp"].values
    )

    save_outputs(
        df_base,
        df_augmented,
        feature_results
    )

    print_banner("PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
