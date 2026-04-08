#!/usr/bin/env python3
import numpy as np
import re
from prody import *
import networkx as nx
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr

# ============================================================
# PARAMETERS (UNCHANGED)
# ============================================================

CONTACT_CUTOFF = 10.0
CORR_THRESHOLD = 0.2
WT_NAME = "WT"
np.random.seed(42)

# ============================================================
# EXPERIMENTAL DATA (UNCHANGED)
# ============================================================

exp_data = {
"Y5H":-6.44,"A8V":-6.66,"F20Q":-6.97,"A23Q":-7.59,
"L29Q":-6.89,"A31S":-6.93,"S37G":-6.67,"E40A":-5.87,
"V44Q":-8.19,"M45Q":-7.46,"L48Q":-7.63,"Q50R":-7.31,
"L57Q":-5.99,"E59D":-6.23,"I61Q":-5.39,"D67A":-5.09,
"D73A":-5.09,"D73N":-5.69,"D75Y":-6.14,"V79Q":-6.63,
"M81Q":-7.28,"C84Y":-7.37
}

# ============================================================
# HELPERS (UNCHANGED LOGIC)
# ============================================================

def extract_mut_index(name):
    return int(re.findall(r'\d+', name)[0])

def dynamic_entropy(cc, index, neighbors):
    local = np.abs(cc[index, neighbors])
    if np.sum(local) == 0:
        return 0.0
    local = local / np.sum(local)
    return -np.sum(local * np.log(local + 1e-12))

def build_graph(cc):
    G = nx.Graph()
    N = cc.shape[0]

    for i in range(N):
        G.add_node(i)

    for i in range(N):
        for j in range(i+1, N):
            corr = abs(cc[i, j])
            if corr > CORR_THRESHOLD:
                G.add_edge(i, j, weight=corr)

    return G

def dynamic_control_centrality(anm, residue_index, N):
    eigvecs = anm.getEigvecs()
    eigvals = anm.getEigvals()

    vecs = eigvecs.reshape((N,3,-1))
    amp = np.linalg.norm(vecs[residue_index,:,:], axis=0)

    return np.sum((amp**2)/(eigvals+1e-8))

# ============================================================
# FEATURE EXTRACTION (UNCHANGED + SAFE GUARDS)
# ============================================================

def extract_features(system):

    print(f"[INFO] Processing {system}")

    pdb_file = f"{system}/{system}_amber.pdb"
    structure = parsePDB(pdb_file)

    ca_atoms = structure.select("protein and name CA")
    coords = ca_atoms.getCoords()

    # calcium detection
    ion = structure.select("resname CAL or element CA and not protein")
    ca_coord = ion.getCoords()[0]

    dist = np.linalg.norm(coords - ca_coord, axis=1)
    ca_index = np.argmin(dist)

    resnums = ca_atoms.getResnums()

    if system == WT_NAME:
        mut_index = ca_index
    else:
        mut_resnum = extract_mut_index(system)
        mut_index = np.where(resnums == mut_resnum)[0][0]

    # ---------------- ANM ----------------
    anm = ANM("ANM")
    anm.buildHessian(ca_atoms)
    anm.calcModes(n_modes=20)

    msf = calcSqFlucts(anm)
    cc = calcCrossCorr(anm)

    mut_stiffness = 1/(msf[mut_index] + 1e-6)
    mut_ca_coupling = cc[mut_index, ca_index]

    prs, eff, sens = calcPerturbResponse(anm)

    neighbors = np.where(dist < 8.0)[0]

    dyn_entropy = dynamic_entropy(cc, mut_index, neighbors)
    frustration = msf[mut_index] - np.mean(msf[neighbors])

    # ---------------- NETWORK ----------------
    G = build_graph(cc)

    betweenness = nx.betweenness_centrality(G)[mut_index]

    try:
        path_len = nx.shortest_path_length(
            G,
            mut_index,
            ca_index,
            weight=lambda u,v,d: 1/d['weight']
        )
        comm_eff = 1/(path_len+1)
    except:
        comm_eff = 0.0

    # ---------------- MODES ----------------
    eigvecs = anm.getEigvecs()[:,0:3]
    N = len(coords)

    amp = np.linalg.norm(eigvecs.reshape(N,3,-1), axis=1).mean(axis=1)
    mut_mode_amp = amp[mut_index]

    # ---------------- CONTROL ----------------
    control_score = dynamic_control_centrality(anm, mut_index, N)

    return np.array([
        mut_stiffness,
        msf[mut_index],
        mut_ca_coupling,
        eff[mut_index],
        sens[mut_index],
        dyn_entropy,
        frustration,
        betweenness,
        comm_eff,
        mut_mode_amp,
        control_score
    ])

# ============================================================
# BUILD DATASET (UNCHANGED HYBRID MODEL)
# ============================================================

print("\n[INFO] Computing WT baseline...")
wt_features = extract_features(WT_NAME)

X = []
y = []

print("\n[INFO] Processing mutants...")
for mut, val in tqdm(exp_data.items()):

    mut_feats = extract_features(mut)
    delta_feats = mut_feats - wt_features

    # HYBRID (UNCHANGED)
    hybrid_feats = np.concatenate([mut_feats, delta_feats])

    X.append(hybrid_feats)
    y.append(val)

X = np.array(X)
y = np.array(y)

# ============================================================
# MODEL (UNCHANGED)
# ============================================================

model = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_regression, k=8)),
    ("svr", SVR(kernel="linear", C=10, epsilon=0.1))
])

loo = LeaveOneOut()

pred = []
true = []

print("\n[INFO] Running LOOCV...")
for train, test in loo.split(X):

    model.fit(X[train], y[train])
    p = model.predict(X[test])

    pred.append(p[0])
    true.append(y[test][0])

pred = np.array(pred)
true = np.array(true)

# ============================================================
# RESULTS
# ============================================================
print("\nModel Performance:")
rmse = np.sqrt(mean_squared_error(true, pred))
r2 = r2_score(true, pred)
pearson = pearsonr(true, pred)[0]

print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.2f}")
print(f"Rₚ   = {pearson:.2f}")
