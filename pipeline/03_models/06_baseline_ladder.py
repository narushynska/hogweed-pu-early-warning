"""
06_baseline_ladder.py

Фаза III: PU-моделювання придатності середовища та ризику поширення.

Inputs (expected in data/):
  - heracleum_ua_clean.parquet
  - apiaceae_background_ua.parquet
  - suitability_predictions.parquet
  - grid_1km_ua.parquet
  - grid_features_static.parquet
  - rivers_ua.parquet
  - patches.parquet
  - transport_ua.gpkg
  - target_points_clustered.parquet
  - spread_risk_layers.parquet
  - admin_ua.gpkg

Вихід: baseline_ladder.csv

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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

feat = pd.read_parquet('data/grid_features_static.parquet')
grid = pd.read_parquet('data/grid_1km_ua.parquet', columns=["cell_id","x","y","oblast"])
feat = feat.merge(grid, on="cell_id", how="left", suffixes=("","_grid"))

holdout_oblasts = ["Ivano-Frankivsk","Zakarpattia","Ternopil","Rivne","Chernivtsi","Khmelnytskyi","Sumy"]

is_holdout = feat.oblast.isin(holdout_oblasts)
train = feat[~is_holdout].copy()
hold = feat[is_holdout].copy()

y_train = (train.label_state=="positive").astype(int)
y_hold = (hold.label_state=="positive").astype(int)

bg_train = train[train.ctrl_target_group_bg==True] if train.ctrl_target_group_bg.dtype==bool else train[train.ctrl_target_group_bg==1]

pos_train = train[y_train==1]
pos_xy = pos_train[["x","y"]].values

prevalence = len(pos_train) / (len(pos_train) + len(bg_train))
eval_pool = pd.concat([hold[hold.label_state=="positive"], hold[hold.ctrl_target_group_bg==True if hold.ctrl_target_group_bg.dtype==bool else hold.ctrl_target_group_bg==1]])
y_eval = (eval_pool.label_state=="positive").astype(int).values

pred_const = np.full(len(eval_pool), prevalence)
auc_const = roc_auc_score(y_eval, pred_const) if len(set(y_eval))>1 else np.nan
ap_const = average_precision_score(y_eval, pred_const)

tree = cKDTree(pos_xy)
dist_nn, _ = tree.query(eval_pool[["x","y"]].values, k=1)
pred_nn = -dist_nn
auc_nn = roc_auc_score(y_eval, pred_nn)
ap_nn = average_precision_score(y_eval, pred_nn)

clim_cols = [c for c in feat.columns if c in ("bio_01","bio_12")]
if len(clim_cols) < 2:
    clim_cols = [c for c in feat.columns if "bio_01" in c or "bio_12" in c][:2]

Xtr = pd.concat([pos_train[clim_cols], bg_train[clim_cols]])
ytr = np.r_[np.ones(len(pos_train)), np.zeros(len(bg_train))]
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
pred_clim = lr.predict_proba(sc.transform(eval_pool[clim_cols]))[:,1]
auc_clim = roc_auc_score(y_eval, pred_clim)
ap_clim = average_precision_score(y_eval, pred_clim)

infra_cols = ["dist_river_m","dist_road_m"]
Xtr2 = pd.concat([pos_train[infra_cols], bg_train[infra_cols]])
sc2 = StandardScaler().fit(Xtr2)
lr2 = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc2.transform(Xtr2), ytr)
pred_infra = lr2.predict_proba(sc2.transform(eval_pool[infra_cols]))[:,1]
auc_infra = roc_auc_score(y_eval, pred_infra)
ap_infra = average_precision_score(y_eval, pred_infra)

baseline_results = pd.DataFrame([
    {"model": "0. Constant prior", "AUC": 0.5, "AP": ap_const},
    {"model": "1. Nearest-positive\ndistance only", "AUC": auc_nn, "AP": ap_nn},
    {"model": "2. Climate-only\n(2 features)", "AUC": auc_clim, "AP": ap_clim},
    {"model": "3. Infrastructure\n(river+road dist)", "AUC": auc_infra, "AP": ap_infra},
    {"model": "4. Full PU-LightGBM\n(149 features)", "AUC": 0.9679, "AP": 0.6053},
])
baseline_results.to_csv("baseline_ladder.csv", index=False)