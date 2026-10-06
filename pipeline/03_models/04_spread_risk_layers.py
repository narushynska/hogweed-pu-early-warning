"""
04_spread_risk_layers.py

Фаза III: PU-моделювання придатності середовища та ризику поширення.

Inputs (expected in data/):
  - suitability_predictions.parquet
  - spread_risk_layers.parquet
  - rivers_ua.parquet
  - patches.parquet
  - transport_ua.gpkg
  - apiaceae_background_ua.parquet
  - target_points_clustered.parquet
  - admin_ua.gpkg

Вихід: spread_risk_layers_final.parquet

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import pandas as pd
import numpy as np

suit = pd.read_parquet('data/suitability_predictions.parquet')
risk = pd.read_parquet('data/spread_risk_layers.parquet')

merged = risk.merge(suit[["cell_id","suitability_targetbg"]], on="cell_id", how="left")

connectivity = 0.5*merged.connectivity_river + 0.5*merged.connectivity_road

def minmax01(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return s*0
    return (s-lo)/(hi-lo)

merged["suitability_targetbg_norm"] = minmax01(merged.suitability_targetbg)
merged["connectivity_combined"] = connectivity
merged["near_term_risk_v2"] = minmax01(
    merged.suitability_targetbg_norm * merged.propagule_pressure * merged.connectivity_combined
)

final_layers = merged[["cell_id","suitability_targetbg_norm","propagule_pressure",
                        "connectivity_river","connectivity_road","connectivity_combined","near_term_risk_v2"]].rename(
    columns={"suitability_targetbg_norm":"suitability", "near_term_risk_v2":"near_term_risk"})
final_layers.to_parquet("spread_risk_layers_final.parquet", compression="zstd")