"""
01_pu_model_training.py

Фаза III: PU-моделювання придатності середовища та ризику поширення.

Inputs (expected in data/):
  - grid_features_static.parquet

Вихід: pu_model_targetbg.pkl

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 200)

df = pd.read_parquet('data/grid_features_static.parquet')

climate_10yr = [c for c in df.columns if c.endswith('_2015_2024')]
wind_feats = ['wind_sin_mean','wind_cos_mean','wind_speed_mean_2015_2024','wind_u_mean_2015_2024',
              'wind_v_mean_2015_2024','wind_resultant_dir','wind_resultant_mag','wind_constancy']
wind_feats = [c for c in wind_feats if c not in climate_10yr]
bio_feats = [f'bio_{i:02d}' for i in range(1,20)]
terrain_feats = ['elevation_m','slope_deg','roughness_m','tpi_m']
hydro_feats = ['dist_river_m','n_upstream_target_patches_50km','network_distance_to_nearest_infected_km',
               'ord_stra','has_upstream_infection_50km']
access_feats = ['dist_road_m','dist_railroad_m','dist_settlement_m']
effort_feats = [c for c in df.columns if c.startswith('n_apiaceae_bg_r') or c.startswith('n_plant_effort_r') or c.startswith('n_hard_negative_r')]

feature_cols = climate_10yr + wind_feats + bio_feats + terrain_feats + hydro_feats + access_feats + effort_feats
feature_cols = sorted(set(feature_cols))

holdout_oblasts = ['Ivano-Frankivsk','Zakarpattia','Ternopil','Rivne','Chernivtsi','Khmelnytskyi','Sumy']

y_positive = (df['label_state']=='positive')

mask_a = y_positive | (df['ctrl_target_group_bg'] & ~y_positive)
data_a = df.loc[mask_a, ['cell_id','oblast'] + feature_cols].copy()
data_a['y'] = y_positive.loc[mask_a].astype(int).values

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

def train_pu_model(data, feature_cols, holdout_oblasts, seed=42, params_override=None):
    train_mask = ~data['oblast'].isin(holdout_oblasts)
    hold_mask = data['oblast'].isin(holdout_oblasts)
    Xtr, ytr = data.loc[train_mask, feature_cols], data.loc[train_mask, 'y']
    Xho, yho = data.loc[hold_mask, feature_cols], data.loc[hold_mask, 'y']

    n_pos = ytr.sum(); n_bg = (ytr==0).sum()
    spw = n_bg / n_pos

    params = dict(
        objective='binary', n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, verbosity=-1
    )
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, ytr)

    # holdout metrics (raw)
    p_ho_raw = model.predict_proba(Xho)[:,1]
    auc = roc_auc_score(yho, p_ho_raw) if yho.nunique()>1 else np.nan
    ap = average_precision_score(yho, p_ho_raw) if yho.nunique()>1 else np.nan

    # Elkan-Noto c estimation via 5-fold OOF on TRAIN positives only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(Xtr))
    Xtr_arr = Xtr.reset_index(drop=True); ytr_arr = ytr.reset_index(drop=True)
    for tr_idx, val_idx in skf.split(Xtr_arr, ytr_arr):
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr_arr.iloc[tr_idx], ytr_arr.iloc[tr_idx])
        oof_pred[val_idx] = m.predict_proba(Xtr_arr.iloc[val_idx])[:,1]
    c_hat = oof_pred[ytr_arr.values==1].mean()
    c_hat = max(c_hat, 1e-3)  # avoid div by 0

    return dict(model=model, train_mask=train_mask, hold_mask=hold_mask,
                auc_raw=auc, ap_raw=ap, c_hat=c_hat, spw=spw,
                Xtr=Xtr, ytr=ytr, Xho=Xho, yho=yho, p_ho_raw=p_ho_raw)

res_a = train_pu_model(data_a, feature_cols, holdout_oblasts)

import joblib

joblib.dump({
    'model': res_a['model'],
    'feature_cols': feature_cols,
    'c_hat_elkan_noto': res_a['c_hat'],
    'scale_pos_weight': res_a['spw'],
    'holdout_oblasts': holdout_oblasts,
    'auc_holdout': res_a['auc_raw'],
    'ap_holdout': res_a['ap_raw'],
    'background_type': 'ctrl_target_group_bg',
}, 'pu_model_targetbg.pkl')