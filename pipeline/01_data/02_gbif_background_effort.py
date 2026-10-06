"""
02_gbif_background_effort.py

Фаза I: збір первинних геошарів та GBIF-даних.

Inputs (expected in data/):
  - apiaceae_gapfill.parquet
  - apiaceae_background_ua.parquet

Вихід: apiaceae_background_ua.parquet

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import pandas as pd

gap = pd.read_parquet('data/apiaceae_background_ua.parquet')
yearpass = pd.read_parquet('data/apiaceae_background_ua.parquet')
merged = pd.concat([yearpass, gap], ignore_index=True).drop_duplicates("key", keep="last").reset_index(drop=True)
merged.to_parquet("apiaceae_background_ua.parquet", index=False)