# ======================================================
# PHYSICS-INFORMED STRUCTURAL MODEL (ΔΔG VERSION)
# ======================================================

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# ======================================================
# INPUT
# ======================================================

PDB_FILE = "WT_amber.pdb"

mutations = [
("Y5H",5), ("A8V",8), ("F20Q",20), ("A23Q",23), ("L29Q",29),
("A31S",31), ("S37G",37), ("E40A",40), ("V44Q",44), ("M45Q",45),
("L48Q",48), ("Q50R",50), ("L57Q",57), ("E59D",59), ("I61Q",61),
("D67A",67), ("D73A",73), ("D73N",73), ("D75Y",75),
("V79Q",79), ("M81Q",81), ("C84Y",84)
]

exp_data = {
"WT":-6.56,"Y5H":-6.44,"A8V":-6.66,"F20Q":-6.97,"A23Q":-7.59,
"L29Q":-6.89,"A31S":-6.93,"S37G":-6.67,"E40A":-5.87,
"V44Q":-8.19,"M45Q":-7.46,"L48Q":-7.63,"Q50R":-7.31,
"L57Q":-5.99,"E59D":-6.23,"I61Q":-5.39,"D67A":-5.09,
"D73A":-5.09,"D73N":-5.69,"D75Y":-6.14,"V79Q":-6.63,
"M81Q":-7.28,"C84Y":-7.37
}

volume = {
"A":88,"V":140,"L":166,"I":168,"F":189,"Y":193,"W":227,
"D":111,"E":138,"N":114,"Q":143,"S":89,"T":116,"H":153,
"R":173,"K":168,"M":162,"C":108,"G":60,"P":112
}

# ======================================================
# LOAD STRUCTURE
# ======================================================

print("\n[INFO] Loading structure...")

parser = PDBParser(QUIET=True)
model = parser.get_structure("WT", PDB_FILE)[0]

# Collect CA atoms
all_ca = []
for chain in model:
    for res in chain:
        if "CA" in res:
            all_ca.append((res.id[1], res["CA"].coord))

print(f"[INFO] Total CA residues: {len(all_ca)}")

# ======================================================
# FIND CALCIUM ION
# ======================================================

print("[INFO] Searching for calcium ion...")

ref_point = None
for atom in model.get_atoms():
    parent = atom.get_parent()
    if parent.id[0] != " ":
        if atom.element.strip().upper() == "CA":
            ref_point = atom.coord
            break

if ref_point is None:
    raise ValueError("Calcium ion not found.")

print("[INFO] Calcium position:", ref_point)

# ======================================================
# COORDINATION SHELL
# ======================================================

ca_coord_residues = []
for rid, coord in all_ca:
    if np.linalg.norm(coord - ref_point) < 6.0:
        ca_coord_residues.append((rid, coord))

print(f"[INFO] Coordination shell size: {len(ca_coord_residues)}")

# ======================================================
# FEATURE GENERATION
# ======================================================

print("\n[INFO] Computing structural features...")

rows = []

for mut, resi in tqdm(mutations, desc="Processing mutations"):

    # locate residue
    res = None
    for chain in model:
        for r in chain:
            if r.id[1] == resi:
                res = r
                break
        if res:
            break

    if res is None:
        raise ValueError(f"Residue {resi} not found")

    wt_aa = mut[0]
    mut_aa = mut[-1]

    # distance-based feature
    vec = res["CA"].coord - ref_point
    dist = np.linalg.norm(vec)
    inv_dist = 1.0 / (dist + 1e-6)

    # volume change
    vol_change = (volume.get(mut_aa,120) - volume.get(wt_aa,120)) / 200.0

    # steric environment
    neighbors = sum(
        np.linalg.norm(coord - res["CA"].coord) < 8.0
        for rid, coord in all_ca if rid != resi
    )
    steric_energy = abs(vol_change) * (neighbors + 1)

    # coordination strength
    coord_strength = 0.0
    for rid_c, coord_c in ca_coord_residues:
        if rid_c != resi:
            d = np.linalg.norm(coord_c - res["CA"].coord)
            coord_strength += 1.0 / ((d + 1e-6) ** 2)

    rows.append({
        "mutant": mut,
        "dG_exp": exp_data[mut],
        "inv_distance": inv_dist,
        "steric_energy": steric_energy,
        "volume_change": vol_change,
        "coordination": coord_strength
    })

df = pd.DataFrame(rows)

# ======================================================
# CONVERT TO ΔΔG (RELATIVE TO WT)
# ======================================================

print("\n[INFO] Converting to ΔΔG (relative to WT)...")

wt_value = exp_data["WT"]
df["ddG_exp"] = df["dG_exp"] - wt_value

# ======================================================
# MODEL: LOOCV
# ======================================================

print("\n[INFO] Running LOOCV...")

features = ["inv_distance", "steric_energy", "volume_change", "coordination"]

X = df[features].values
y = df["ddG_exp"].values

loo = LeaveOneOut()
y_pred = np.zeros(len(y))

for train, test in tqdm(loo.split(X), total=len(y), desc="LOOCV"):

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train])
    Xte = scaler.transform(X[test])

    reg = LinearRegression()
    reg.fit(Xtr, y[train])
    y_pred[test] = reg.predict(Xte)

# ======================================================
# RESULTS
# ======================================================

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
rp, _ = pearsonr(y, y_pred)

print("\n==============================")
print(" FINAL ΔΔG MODEL PERFORMANCE")
print("==============================")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.2f}")
print(f"Rp   = {rp:.2f}")

# ======================================================
# SAVE OUTPUT
# ======================================================

df["ddG_pred"] = y_pred
df.to_csv("results_structural_ddG_model.csv", index=False)

print("\n[INFO] Results saved to results_structural_ddG_model.csv")
