# Low-Dimensional Energetic Landscape of Ca²⁺ Binding in Troponin C

## Overview

This repository contains the data, scripts, and analysis pipeline used to construct a **physics-informed low-dimensional reaction coordinate** describing mutation-induced changes in Ca²⁺ binding free energy in Troponin C.

The framework integrates **structural, electrostatic, and dynamical descriptors** into a unified representation that captures the dominant energetic variation across mutations.

---

## Key Results

* Structural model: ( R_p = 0.83 ), RMSE = 0.47
* Electrostatic model: ( R_p = 0.83 ), RMSE = 0.47
* Dynamical model: ( R_p = 0.85 ), RMSE = 0.44
* Unified model:
  [
  \Delta\Delta G = (5.41 \pm 0.43),\xi + (0.21 \pm 0.07)
  ]
  with ( R_p = 0.90 ), RMSE = 0.38

The results demonstrate that mutation-induced energetics can be described by a **single dominant reaction coordinate** emerging from multiscale coupling.

---


---

## Methodology

### 1. Descriptor Construction

* **Structural**: inverse distance, steric perturbation, volume change
* **Electrostatic**: screened Coulomb descriptors, dielectric response, field anisotropy
* **Dynamical**: ANM-based correlation, communication efficiency, spectral entropy

---

### 2. Reaction Coordinate

Descriptors are standardized and whitened:
[
\tilde{x} = \frac{x - \mu}{\sigma}
]

A supervised projection defines the reaction coordinate:
[
\xi = w^\top \tilde{x}
]

---

### 3. Free Energy Model

[
\Delta\Delta G = a\xi + b
]

---

### 4. Validation

* Leave-One-Out Cross-Validation (LOOCV)
* Leakage-free preprocessing within each fold
* Stability analysis across descriptor subsets

---

## Requirements

```bash
python >= 3.9
numpy
scipy
pandas
scikit-learn
biopython
prody
networkx
matplotlib
```



---

## Reproducibility

* All preprocessing steps are performed **within cross-validation folds**
* Descriptor standardization and whitening are recomputed per fold
* Random seeds are fixed where applicable

---

## Data

Experimental Ca²⁺ binding affinities were compiled from published literature and converted to free energies:
[
\Delta G = RT \ln K_d
]
[
\Delta\Delta G = \Delta G_{\text{mut}} - \Delta G_{\text{WT}}
]

---

## Citation

If you use this code or data, please cite:

> Basit, A. *Emergence of Low-Dimensional Energetic Landscape Governing Calcium Binding in Troponin C*. Biophysical Journal (submitted).

---

## Notes

* This framework is designed for **mechanistic interpretability**, not large-scale prediction.
* Dataset size is limited (N = 22); conclusions emphasize **internal consistency and physical insight**.

---

## Contact

Abdul Basit
School of Computational and Integrative Sciences
Jawaharlal Nehru University
Email: [meetabasit@gmail.com](mailto:meetabasit@gmail.com)

---
