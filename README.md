# Hybrid Flamingo–Bat Feature Selection for Resistance Exercise Transcriptomics

This repository packages the computational artifacts supplied in `model.zip` for the resistance-exercise skeletal-muscle feature-selection project.

## Included methods
- LASSO
- Flamingo Search Algorithm (FSA)
- Bat Algorithm (BA)
- Hybrid Flamingo–Bat

## Included classifiers
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## Included artifacts
- Selected-gene lists
- LASSO coefficients
- Model-performance CSV files
- Supplied Bat ROC image
- Consolidated performance table
- Gene-subset-size table
- Reconstructed summary plots
- Original uploaded archive

## Important reproducibility/provenance note
The supplied `model.zip` contains model outputs and selected-gene files, but it does **not** contain the original source implementation of the optimization algorithms, the raw expression matrix, the exact preprocessing pipeline, fold assignments, or complete hyperparameter grids.

Accordingly, the scripts in `scripts/` are validation and summary utilities reconstructed from the supplied outputs. They do not claim to reproduce the original optimization implementation exactly, and no missing data/results have been fabricated.

## Run validation
```bash
python scripts/validate_project.py
python scripts/make_summary.py
```

## What is needed for full algorithm-level reproducibility
1. Raw/count expression matrix or GEO download script.
2. Exact sample labels and cross-validation design.
3. Filtering/normalization parameters.
4. LASSO grid and training code.
5. Flamingo implementation and parameters.
6. Bat implementation and parameters.
7. Exact hybrid update equations.
8. Fitness-function weighting.
9. Random seeds and independent-run settings.
10. Classifier hyperparameter grids.
11. Original fold-level predictions.

## Scientific integrity
Values in this repository that originate from the supplied archive are preserved as supplied. Derived summary tables/plots are explicitly labeled as reconstructed.
