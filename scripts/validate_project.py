from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'
required=[
'Bat_selected_genes.csv','bat_selected_model_performance.csv',
'flamingo_selected_model_performance.csv','flamngo_selected_genes_names.csv',
'Bat_flamingo_selected_model_performance.csv','Bat_flamngo_selected_genes.xlsx',
'lasso_selected_genes.csv','lasso_selected_model_performance.csv']
missing=[f for f in required if not (RAW/f).exists()]
if missing: raise FileNotFoundError(', '.join(missing))
print('All supplied artifacts are present.')
for f in required: print('OK:',f)
print('Validation complete.')
