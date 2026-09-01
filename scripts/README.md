# Model development

`model_development.py` is the full Python implementation for LASSO, Flamingo-style binary wrapper selection, Bat Algorithm selection, Hybrid Flamingo-Bat selection, and RF/SVM/XGBoost evaluation.

The original uploaded archive contained outputs but not the lost historical source code/raw expression matrix. This is therefore a transparent reproducible implementation based on the manuscript methodology, not a claim to be the exact original code.

Example:
```bash
python src/model_development.py --expression data/raw/expression_matrix.csv --label-column label
```
