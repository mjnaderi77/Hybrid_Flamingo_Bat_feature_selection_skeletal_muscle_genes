from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=pd.read_csv(ROOT/'results/tables/consolidated_model_performance.csv')
p.sort_values(['F1 Score','Accuracy'],ascending=False).head(10).to_csv(
    ROOT/'results/tables/top_performance_rows.csv',index=False)
print(p.sort_values(['F1 Score','Accuracy'],ascending=False).head(10).to_string(index=False))
