#!/usr/bin/env python3
# ============================================================
# PHYSICS-INFORMED INTERACTION FIELD RC PIPELINE
# ============================================================
#

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

from tqdm import tqdm

# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-6

PDB_FILE = "WT_amber.pdb"

HBOND_CUTOFF = 3.5
HYDRO_SIGMA = 5.0


# ============================================================
# MUTATIONS
# ============================================================

MUTATIONS = [
    ("Y5H",5), ("A8V",8), ("F20Q",20), ("A23Q",23),
    ("L29Q",29), ("A31S",31), ("S37G",37), ("E40A",40),
    ("V44Q",44), ("M45Q",45), ("L48Q",48), ("Q50R",50),
    ("L57Q",57), ("E59D",59), ("I61Q",61), ("D67A",67),
    ("D73A",73), ("D73N",73), ("D75Y",75), ("V79Q",79),
    ("M81Q",81), ("C84Y",84)
]

EXP_DATA = {
    "WT":-6.56,"Y5H":-6.44,"A8V":-6.66,"F20Q":-6.97,
    "A23Q":-7.59,"L29Q":-6.89,"A31S":-6.93,"S37G":-6.67,
    "E40A":-5.87,"V44Q":-8.19,"M45Q":-7.46,"L48Q":-7.63,
    "Q50R":-7.31,"L57Q":-5.99,"E59D":-6.23,"I61Q":-5.39,
    "D67A":-5.09,"D73A":-5.09,"D73N":-5.69,"D75Y":-6.14,
    "V79Q":-6.63,"M81Q":-7.28,"C84Y":-7.37
}

# ============================================================
# RESIDUE MAPPING
# ============================================================

THREE_TO_ONE = {

    "ALA":"A",
    "VAL":"V",
    "LEU":"L",
    "ILE":"I",
    "PHE":"F",
    "TYR":"Y",
    "TRP":"W",

    "ASP":"D",
    "GLU":"E",
    "ASN":"N",
    "GLN":"Q",

    "SER":"S",
    "THR":"T",

    "HIS":"H",
    "LYS":"K",
    "ARG":"R",

    "MET":"M",
    "CYS":"C",
    "GLY":"G",
    "PRO":"P"
}

# ============================================================
# HYDROPHOBICITY SCALE
# ============================================================

HYDRO = {

    "A":1.8,
    "V":4.2,
    "L":3.8,
    "I":4.5,
    "F":2.8,

    "Y":-1.3,
    "W":-0.9,

    "D":-3.5,
    "E":-3.5,
    "N":-3.5,
    "Q":-3.5,

    "S":-0.8,
    "T":-0.7,

    "H":-3.2,
    "K":-3.9,
    "R":-4.5,

    "M":1.9,
    "C":2.5,
    "G":-0.4,
    "P":-1.6
}

# ============================================================
# HBOND DEFINITIONS
# ============================================================

HBOND_DONORS = {
    "N", "NE", "NH1", "NH2",
    "NZ", "OG", "OG1", "OH"
}

HBOND_ACCEPTORS = {
    "O", "OD1", "OD2",
    "OE1", "OE2", "OG", "OG1"
}

# ============================================================
# FEATURE SET
# ============================================================

FEATURES = [

    # hydrophobic
    "hydrophobic_field",

    # HBOND
    "hbond_inverse_square",

    # coordination
    "inv_distance_sq",
]

# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class StructureContext:

    model: object
    residue_map: dict
    all_ca: list
    ref_point: np.ndarray

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    rp, _ = pearsonr(
        y_true,
        y_pred
    )

    return {
        "RMSE": rmse,
        "R2": r2,
        "Rp": rp
    }


# ============================================================
# STRUCTURE LOADING
# ============================================================

def load_structure(pdb_file):

    parser = PDBParser(QUIET=True)

    model = parser.get_structure(
        "WT",
        pdb_file
    )[0]

    residue_map = {}
    all_ca = []

    for chain in model:

        for residue in chain:

            residue_map[
                residue.id[1]
            ] = residue

            if "CA" in residue:

                all_ca.append(
                    (
                        residue.id[1],
                        residue["CA"].coord
                    )
                )

    calcium = None

    for atom in model.get_atoms():

        parent = atom.get_parent()

        if parent.id[0] != " ":

            if (
                atom.element
                .strip()
                .upper()
                == "CA"
            ):

                calcium = atom.coord
                break

    if calcium is None:
        raise ValueError(
            "Calcium ion not found."
        )

    return StructureContext(
        model=model,
        residue_map=residue_map,
        all_ca=all_ca,
        ref_point=calcium
    )

# ============================================================
# HYDROPHOBIC FIELD
# ============================================================

def hydrophobic_field(
    res_coord,
    residue_map
):

    value = 0.0

    for _, residue in residue_map.items():

        if "CA" not in residue:
            continue

        coord = residue["CA"].coord

        r = np.linalg.norm(
            coord - res_coord
        )

        if r < EPS:
            continue

        resname = residue.get_resname()

        aa = THREE_TO_ONE.get(
            resname,
            "A"
        )

        hydro = HYDRO.get(
            aa,
            0.0
        )

        value += (
            hydro
            *
            np.exp(
                -(r ** 2)
                /
                (2 * HYDRO_SIGMA ** 2)
            )
        )

    return value

# ============================================================
# HBOND FIELD
# ============================================================

def hbond_field(
    residue,
    model
):

    invsq = 0.0

    for atom_i in residue:

        name_i = (
            atom_i
            .get_name()
            .strip()
        )

        coord_i = atom_i.coord

        for chain in model:

            for residue_j in chain:

                if residue_j == residue:
                    continue

                for atom_j in residue_j:

                    name_j = (
                        atom_j
                        .get_name()
                        .strip()
                    )

                    coord_j = atom_j.coord

                    r = np.linalg.norm(
                        coord_i - coord_j
                    )

                    if r > HBOND_CUTOFF:
                        continue

                    valid_pair = (

                        (
                            name_i in HBOND_DONORS
                            and
                            name_j in HBOND_ACCEPTORS
                        )

                        or

                        (
                            name_i in HBOND_ACCEPTORS
                            and
                            name_j in HBOND_DONORS
                        )
                    )

                    if valid_pair:

                        invsq += (
                            1.0
                            /
                            ((r + EPS) ** 2)
                        )

    return invsq

# ============================================================
# COORDINATION FIELD
# ============================================================

def coordination_field(
    res_coord,
    ref_point,
    all_ca
):

    r = np.linalg.norm(
        res_coord - ref_point
    )

    inv_r2 = (
        1.0
        /
        ((r + EPS) ** 2)
    )


    for _, coord in all_ca:

        d = np.linalg.norm(
            coord - res_coord
        )

        if d < EPS:
            continue


    return (
        inv_r2    )

# ============================================================
# FEATURE TABLE
# ============================================================

def build_feature_table(context):

    rows = []

    for mut, resi in tqdm(
        MUTATIONS,
        desc="Feature extraction"
    ):

        residue = context.residue_map[resi]

        res_coord = residue["CA"].coord

        hydro = hydrophobic_field(
            res_coord,
            context.residue_map
        )

        hb = hbond_field(
            residue,
            context.model
        )

        
        inv_r2 = coordination_field(
            res_coord,
            context.ref_point,
            context.all_ca
        )

        rows.append({

            "mutant": mut,

            "ddG_exp":
                EXP_DATA[mut]
                -
                EXP_DATA["WT"],

            "hydrophobic_field": hydro,

            "hbond_inverse_square": hb,

            "inv_distance_sq": inv_r2,

       })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading structure...")

    context = load_structure(
        PDB_FILE
    )

    print(
        "\nComputing descriptors..."
    )

    df = build_feature_table(
        context
    )

    df.to_csv(
        "raw_interaction_features.csv",
        index=False
    )

    print(
        "\nSaved raw features:"
        " raw_interaction_features.csv"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()