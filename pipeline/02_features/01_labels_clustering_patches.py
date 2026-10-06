"""
01_labels_clustering_patches.py

Фаза II: побудова міток/патчів, клімат-ряду та таблиці ознак на сітці.

Inputs (expected in data/):
  - heracleum_ua_clean.parquet
  - apiaceae_background_ua.parquet
  - plantae_effort_sample_ua.parquet
  - grid_1km_ua.parquet

Вихід: labels_cells.parquet

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
import json
import os
from pyproj import Transformer

her = pd.read_parquet('data/heracleum_ua_clean.parquet')
bg = pd.read_parquet('data/apiaceae_background_ua.parquet')
ef = pd.read_parquet('data/plantae_effort_sample_ua.parquet')
grid = pd.read_parquet('data/grid_1km_ua.parquet')

T = Transformer.from_crs("EPSG:4326","EPSG:3035",always_xy=True)

def add_cell(df, latc="decimalLatitude", lonc="decimalLongitude"):
    d = df.dropna(subset=[latc,lonc]).copy()
    x,y = T.transform(d[lonc].to_numpy(), d[latc].to_numpy())
    d["x3035"]=x; d["y3035"]=y
    d["row_"]=np.floor(y/1000).astype("int64"); d["col_"]=np.floor(x/1000).astype("int64")
    d["cell_id"]=d["row_"]*100000+d["col_"]
    return d

grid_ids = pd.Index(grid["cell_id"].to_numpy())
her_c = add_cell(her); bg_c = add_cell(bg); ef_c = add_cell(ef)
her_c = her_c[her_c.cell_id.isin(grid_ids)]; bg_c=bg_c[bg_c.cell_id.isin(grid_ids)]; ef_c=ef_c[ef_c.cell_id.isin(grid_ids)]
bg_c["is_cs"]=(bg_c["basisOfRecord"]=="HUMAN_OBSERVATION")
her_c["is_cs"]=her_c["is_citizen_science"].fillna(False).astype(bool)

tg  = her_c[her_c.label_role=="target"]
iv  = her_c[her_c.label_role=="invasive_other"]
hn  = her_c[her_c.label_role=="hard_negative"]

def agg_counts(d, name):
    return d.groupby("cell_id").size().rename(name)

parts = [agg_counts(tg,"n_target"), agg_counts(iv,"n_invasive_other"),
         agg_counts(hn,"n_hard_negative"), agg_counts(bg_c,"n_apiaceae_bg"),
         agg_counts(ef_c,"n_plant_effort")]
lab = pd.concat(parts, axis=1).fillna(0).astype("int64")

pool = pd.concat([her_c[["cell_id","year","is_cs"]], bg_c[["cell_id","year","is_cs"]]], ignore_index=True)
pool_y = pool.dropna(subset=["year"])
g = pool_y.groupby("cell_id")["year"]
lab = lab.join(pd.DataFrame({"first_year":g.min(),"last_year":g.max(),"n_years_observed":g.nunique()}))
lab = lab.join(pool.groupby("cell_id")["is_cs"].mean().rename("is_citizen_science_share"))
gt = tg.dropna(subset=["year"]).groupby("cell_id")["year"]
lab = lab.join(pd.DataFrame({"first_year_target":gt.min(),"last_year_target":gt.max(),
                             "n_years_target":gt.nunique()}))
lab = lab.reset_index()

n_verified_negative = int((her["occurrenceStatus"]=="ABSENT").sum())
assert n_verified_negative == 0, "Phase I established zero ABSENT records"
lab["label_state"] = np.where(lab.n_target>0, "positive", "unknown")

lab["ctrl_target_group_bg"] = (lab.n_apiaceae_bg>0) & (lab.n_target==0)
rng = np.random.default_rng(42)
N_RANDOM_BG = 20000
rand_ids = rng.choice(grid["cell_id"].to_numpy(), size=N_RANDOM_BG, replace=False)
rand_set = pd.Index(rand_ids)
lab["ctrl_random_bg"] = lab.cell_id.isin(rand_set)
lab["has_hard_negative"] = lab.n_hard_negative>0
lab["has_invasive_other"] = lab.n_invasive_other>0

missing = rand_set.difference(pd.Index(lab.cell_id.to_numpy()))
add = pd.DataFrame({"cell_id":missing.to_numpy()})
for c in ["n_target","n_invasive_other","n_hard_negative","n_apiaceae_bg","n_plant_effort"]: add[c]=0
for c in ["first_year","last_year","n_years_observed","is_citizen_science_share",
          "first_year_target","last_year_target","n_years_target"]: add[c]=np.nan
add["label_state"]="unknown"; add["ctrl_target_group_bg"]=False; add["ctrl_random_bg"]=True
add["has_hard_negative"]=False; add["has_invasive_other"]=False
labels_cells = pd.concat([lab, add], ignore_index=True)
labels_cells = labels_cells.merge(grid[["cell_id","oblast","oblast_code","lon","lat","x","y","dist_river_m","basin_id"]],
                                  on="cell_id", how="left")
assert labels_cells.cell_id.is_unique

cols = ["cell_id","label_state","n_target","n_invasive_other","n_hard_negative","n_apiaceae_bg",
        "n_plant_effort","first_year","last_year","n_years_observed","is_citizen_science_share",
        "first_year_target","last_year_target","n_years_target",
        "ctrl_target_group_bg","ctrl_random_bg","has_hard_negative","has_invasive_other",
        "oblast","oblast_code","lon","lat","x","y","dist_river_m","basin_id"]
labels_cells = labels_cells[cols]
labels_cells.to_parquet("labels_cells.parquet", index=False)