#!/usr/bin/env python3
# ============================================================
# ELECTROSTATICS-ONLY ABSOLUTE ΔG MODEL (APBS)
# ============================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression

from scipy.stats import pearsonr

# ============================================================
# INPUT
# ============================================================

BASE_DIR = "."

exp_data = {
"WT":-6.56,
"Y5H":-6.44,"A8V":-6.66,"F20Q":-6.97,"A23Q":-7.59,
"L29Q":-6.89,"A31S":-6.93,"S37G":-6.67,"E40A":-5.87,
"V44Q":-8.19,"M45Q":-7.46,"L48Q":-7.63,"Q50R":-7.31,
"L57Q":-5.99,"E59D":-6.23,"I61Q":-5.39,"D67A":-5.09,
"D73A":-5.09,"D73N":-5.69,"D75Y":-6.14,"V79Q":-6.63,
"M81Q":-7.28,"C84Y":-7.37
}

systems = list(exp_data.keys())

# ============================================================
# Ca2+ POSITION
# ============================================================

def get_ca_position(pqr):

    with open(pqr) as f:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                p = line.split()
                if p[2] == "CA" and p[3] in ["CA","CAL"]:
                    return np.array([float(p[6]),float(p[7]),float(p[8])])

    raise RuntimeError("Ca2+ not found")

# ============================================================
# ELECTROSTATICS
# ============================================================

def compute(system):

    pqr = os.path.join(BASE_DIR, system, f"{system}.pqr")

    ca = get_ca_position(pqr)

    E = np.zeros(3)
    coord_vecs = []

    with open(pqr) as f:

        for line in f:

            if not line.startswith(("ATOM","HETATM")):
                continue

            p = line.split()

            pos = np.array([float(p[6]),float(p[7]),float(p[8])])
            q = float(p[9])

            r_vec = pos - ca
            r = np.linalg.norm(r_vec)

            if r < 1e-6:
                continue

            # electric field
            if r < 10:
                E += q * r_vec / (r**3)

            # coordination shell
            if p[2].startswith("O") and r < 2.8:
                coord_vecs.append(r_vec / r)

    # coordination axis
    if len(coord_vecs) > 0:
        axis = np.mean(coord_vecs, axis=0)
        axis /= (np.linalg.norm(axis) + 1e-8)
    else:
        axis = np.zeros(3)

    return E, axis, coord_vecs

# ============================================================
# BUILD DATA
# ============================================================

print("\n[INFO] Computing electrostatics...")

raw_E = []
axes = []
coord_vecs = []

for s in tqdm(systems):

    E, axis, vecs = compute(s)

    raw_E.append(E)
    axes.append(axis)
    coord_vecs.append(vecs)

raw_E = np.array(raw_E)
axes = np.array(axes)
coord_vecs = np.array(coord_vecs, dtype=object)

# ============================================================
# GLOBAL ELECTROSTATIC AXIS
# ============================================================

mean_E = np.mean(raw_E, axis=0)
_,_,Vt = np.linalg.svd(raw_E - mean_E)

principal_axis = Vt[0] / np.linalg.norm(Vt[0])

# ============================================================
# FEATURES (ABSOLUTE)
# ============================================================

X = []

for i in range(len(systems)):

    E = raw_E[i]
    axis = axes[i]

    E_mag = np.linalg.norm(E) + 1e-8

    # global decomposition
    E_parallel = np.dot(E, principal_axis)
    E_perp = np.linalg.norm(E - E_parallel*principal_axis)

    # local frame
    if np.linalg.norm(axis) > 0:

        E_radial = np.dot(E, axis)
        tangential = np.linalg.norm(E - E_radial*axis)

        funnel = E_radial / E_mag

    else:
        tangential = 0
        funnel = 0

    # frustration
    if len(coord_vecs[i]) > 0:

        cos2 = []
        for r_hat in coord_vecs[i]:
            c = np.dot(E, r_hat)/E_mag
            cos2.append(c*c)

        frustration = 1 - np.mean(cos2)

    else:
        frustration = 0

    X.append([E_perp, tangential, funnel, frustration])

X = np.array(X)

feature_names = ["E_perp","E_tangential","funnel","frustration"]

# ============================================================
# TARGET (ABSOLUTE ΔG)
# ============================================================

y = np.array([exp_data[s] for s in systems])

# ============================================================
# PHYSICS DESCRIPTOR
# ============================================================

descriptor = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

descriptor.fit(X, y)

scaler = descriptor.named_steps["scaler"]
ridge = descriptor.named_steps["ridge"]

weights = ridge.coef_

print("\n=== Descriptor Weights ===")
for n,w in zip(feature_names, weights):
    print(f"{n:15s} {w: .4f}")

X_scaled = scaler.transform(X)
D = X_scaled @ weights

# ============================================================
# NONLINEAR MODEL
# ============================================================

model = Pipeline([
    ("poly", PolynomialFeatures(2, include_bias=False)),
    ("linreg", LinearRegression())
])

# ============================================================
# LOOCV
# ============================================================

loo = LeaveOneOut()

pred, true = [], []

print("\n[INFO] LOOCV...")

for tr, te in tqdm(loo.split(D), total=len(D)):

    m = clone(model)

    Xtr = D[tr].reshape(-1,1)
    Xte = D[te].reshape(-1,1)

    m.fit(Xtr, y[tr])

    pred.append(m.predict(Xte)[0])
    true.append(y[te][0])

pred = np.array(pred)
true = np.array(true)

# ============================================================
# RESULTS
# ============================================================

rmse = np.sqrt(mean_squared_error(true,pred))
r2 = r2_score(true,pred)
rp = pearsonr(true,pred)[0]

print("\n==============================")
print(" ABSOLUTE ΔG MODEL")
print("==============================")
print(f"RMSE = {rmse:.2f}")
print(f"R2   = {r2:.2f}")
print(f"Rp   = {rp:.2f}")

# ============================================================
# SAVE
# ============================================================

pd.DataFrame({
    "system": systems,
    "dG_exp": true,
    "dG_pred": pred
}).to_csv("electrostatics_absolute_dG.csv", index=False)

print("\n[INFO] Saved: electrostatics_absolute_dG.csv")