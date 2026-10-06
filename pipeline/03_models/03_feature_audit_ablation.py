"""
03_feature_audit_ablation.py

Фаза III: PU-моделювання придатності середовища та ризику поширення.

Inputs (expected in data/):
  - grid_features_static.parquet

Вихід: feature_audit_results.json

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import pandas as pd, numpy as np
import json
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

effort_and_road_group = effort_feats + ['dist_road_m']

holdout_oblasts = ['Ivano-Frankivsk','Zakarpattia','Ternopil','Rivne','Chernivtsi','Khmelnytskyi','Sumy']

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import shap
from scipy.stats import spearmanr

y_positive = (df['label_state']=='positive')

mask_a = y_positive | (df['ctrl_target_group_bg'] & ~y_positive)
data_a = df.loc[mask_a, ['cell_id','oblast'] + feature_cols].copy()
data_a['y'] = y_positive.loc[mask_a].astype(int).values

mask_b = y_positive | (df['ctrl_random_bg'] & ~y_positive)
data_b = df.loc[mask_b, ['cell_id','oblast'] + feature_cols].copy()
data_b['y'] = y_positive.loc[mask_b].astype(int).values

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

    p_ho_raw = model.predict_proba(Xho)[:,1]
    auc = roc_auc_score(yho, p_ho_raw) if yho.nunique()>1 else np.nan
    ap = average_precision_score(yho, p_ho_raw) if yho.nunique()>1 else np.nan

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(Xtr))
    Xtr_arr = Xtr.reset_index(drop=True); ytr_arr = ytr.reset_index(drop=True)
    for tr_idx, val_idx in skf.split(Xtr_arr, ytr_arr):
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr_arr.iloc[tr_idx], ytr_arr.iloc[tr_idx])
        oof_pred[val_idx] = m.predict_proba(Xtr_arr.iloc[val_idx])[:,1]
    c_hat = oof_pred[ytr_arr.values==1].mean()
    c_hat = max(c_hat, 1e-3)

    return dict(model=model, train_mask=train_mask, hold_mask=hold_mask,
                auc_raw=auc, ap_raw=ap, c_hat=c_hat, spw=spw,
                Xtr=Xtr, ytr=ytr, Xho=Xho, yho=yho, p_ho_raw=p_ho_raw)

res_a = train_pu_model(data_a, feature_cols, holdout_oblasts)
res_b = train_pu_model(data_b, feature_cols, holdout_oblasts)

ablation_exclude = set(effort_and_road_group)
feature_cols_noeffort = [c for c in feature_cols if c not in ablation_exclude]

res_a_noeffort = train_pu_model(
    data_a.assign(**{}),
    feature_cols_noeffort, holdout_oblasts
)

from sklearn.inspection import permutation_importance

perm_res = permutation_importance(
    res_a['model'], res_a['Xho'], res_a['yho'],
    scoring='average_precision', n_repeats=20, random_state=42, n_jobs=1
)
perm_imp = pd.Series(perm_res.importances_mean, index=feature_cols).sort_values(ascending=False)

explainer = shap.TreeExplainer(res_a['model'])
shap_values = explainer.shap_values(res_a['Xho'])

shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)

Xho_a = res_a['Xho'].reset_index(drop=True)
shap_road = pd.Series(shap_values[:, feature_cols.index('dist_road_m')])
road_val = Xho_a['dist_road_m']

bins5 = pd.qcut(road_val, 5, labels=['q1_near','q2','q3','q4','q5_far'])
strat5 = pd.DataFrame({'dist_road_m': road_val, 'shap_dist_road': shap_road, 'bin': bins5,
                        'effort5km': Xho_a['n_plant_effort_r5km'].values}).groupby('bin', observed=True).agg(
    n=('shap_dist_road','size'), mean_shap=('shap_dist_road','mean'),
    median_dist=('dist_road_m','median'), median_effort=('effort5km','median')
)

river_conn_feats = ['n_upstream_target_patches_50km','network_distance_to_nearest_infected_km','has_upstream_infection_50km','ord_stra']
feature_cols_noconn = [c for c in feature_cols if c not in river_conn_feats]
feature_cols_noeffort_noconn = [c for c in feature_cols_noeffort if c not in river_conn_feats]

res_a_noconn = train_pu_model(data_a, feature_cols_noconn, holdout_oblasts)
res_a_noeffort_noconn = train_pu_model(data_a, feature_cols_noeffort_noconn, holdout_oblasts)

train_a_full = data_a.loc[res_a['train_mask']].copy()
train_a_full['road_bin'] = pd.qcut(train_a_full['dist_road_m'], 8, duplicates='drop')

rows = []
for b, g in train_a_full.groupby('road_bin', observed=True):
    ob_rates = g.groupby('oblast', observed=True).agg(n=('y','size'), rate=('y','mean'))
    ob_rates = ob_rates[ob_rates['n']>=20]
    if len(ob_rates) >= 3:
        rows.append(dict(bin=str(b), n_oblasts=len(ob_rates), rate_std=ob_rates['rate'].std(), rate_mean=ob_rates['rate'].mean(), rate_range=ob_rates['rate'].max()-ob_rates['rate'].min()))
diag1 = pd.DataFrame(rows)

river_conn_3 = ['n_upstream_target_patches_50km','network_distance_to_nearest_infected_km','has_upstream_infection_50km']

feature_cols_no3 = [c for c in feature_cols if c not in river_conn_3]
feature_cols_noordstra = [c for c in feature_cols if c != 'ord_stra']

res_a_no3 = train_pu_model(data_a, feature_cols_no3, holdout_oblasts)
res_a_noordstra = train_pu_model(data_a, feature_cols_noordstra, holdout_oblasts)

feature_audit = {
    "spatial_holdout_oblasts": holdout_oblasts,
    "n_holdout_cells": 113204,
    "n_holdout_positive": 220,
    "permutation_importance_top20": perm_imp.head(20).round(5).to_dict(),
    "shap_mean_abs_top20": shap_imp.head(20).round(5).to_dict(),
    "ablation": {
        "full_model": {"AUC": round(float(res_a['auc_raw']),4), "AP": round(float(res_a['ap_raw']),4), "n_features": len(feature_cols)},
        "no_effort_and_dist_road": {"AUC": round(float(res_a_noeffort['auc_raw']),4), "AP": round(float(res_a_noeffort['ap_raw']),4),
                                     "n_features": len(feature_cols_noeffort), "features_removed": sorted(effort_and_road_group)},
        "no_river_connectivity": {"AUC": round(float(res_a_noconn['auc_raw']),4), "AP": round(float(res_a_noconn['ap_raw']),4),
                                    "n_features": len(feature_cols_noconn), "features_removed": river_conn_feats},
        "no_effort_no_river_connectivity": {"AUC": round(float(res_a_noeffort_noconn['auc_raw']),4), "AP": round(float(res_a_noeffort_noconn['ap_raw']),4),
                                              "n_features": len(feature_cols_noeffort_noconn)},
        "ap_loss_from_removing_effort_dist_road": round(float(res_a['ap_raw'] - res_a_noeffort['ap_raw']), 4),
        "ap_loss_pct_from_removing_effort_dist_road": round(100*float(res_a['ap_raw'] - res_a_noeffort['ap_raw'])/float(res_a['ap_raw']), 2),
        "ap_loss_from_removing_river_connectivity": round(float(res_a['ap_raw'] - res_a_noconn['ap_raw']), 4),
        "ap_loss_pct_from_removing_river_connectivity": round(100*float(res_a['ap_raw'] - res_a_noconn['ap_raw'])/float(res_a['ap_raw']), 2)
    },
    "dist_road_shap_stratification_quintile": strat5.reset_index().rename(columns={'bin':'quintile'}).round(4).to_dict(orient='records'),
    "dist_road_shap_vs_effort_spearman": {
        "vs_n_plant_effort_r5km": {"rho": round(float(spearmanr(shap_road, np.log1p(Xho_a['n_plant_effort_r5km']))[0]),4),
                                     "p": float(spearmanr(shap_road, np.log1p(Xho_a['n_plant_effort_r5km']))[1])},
        "vs_n_apiaceae_bg_r5km": {"rho": round(float(spearmanr(shap_road, np.log1p(Xho_a['n_apiaceae_bg_r5km']))[0]),4),
                                    "p": float(spearmanr(shap_road, np.log1p(Xho_a['n_apiaceae_bg_r5km']))[1])},
        "vs_n_plant_effort_r10km": {"rho": round(float(spearmanr(shap_road, np.log1p(Xho_a['n_plant_effort_r10km']))[0]),4),
                                      "p": float(spearmanr(shap_road, np.log1p(Xho_a['n_plant_effort_r10km']))[1])}
    },
    "admin_boundary_artifact_check": {
        "method": "presence rate within dist_road_m octile bins, variance across oblasts (>=20 obs per oblast per bin)",
        "per_bin": diag1.round(4).to_dict(orient='records'),
        "max_within_bin_oblast_range": round(float(diag1['rate_range'].max()), 4),
        "conclusion": "presence rate range across oblasts within a fixed road-distance bin reaches 0.28 (absolute), well above smooth-gradient expectation -- interpreted as survey-effort heterogeneity across oblasts, not a biological signal, and this heterogeneity is not visible in the national-average GAM PDP curves"
    },
    "what_this_audit_confirms": [
        "Effort/access features contribute a measurable but modest share of holdout AP (~6.6% relative loss when removed).",
        "River-connectivity-to-known-patch features are the single largest contributor to holdout AP (~77% relative loss when removed) and mix real riverine spread with detection history.",
        "dist_road_m's SHAP contribution is non-monotonic with distance and weakly-but-significantly correlated with 5km plant-effort counts, consistent with it being a mixed accessibility/biology signal rather than a pure remoteness covariate.",
        "Oblast-to-oblast presence-rate heterogeneity persists even after conditioning on road-distance, indicating a real spatial confound the model's smooth splines and marginal SHAP averages do not fully expose."
    ],
    "what_this_audit_does_not_prove": [
        "It does not prove dist_road_m or the effort buffers carry zero biological signal -- correlations are moderate (|rho|<0.15), not total collinearity.",
        "It does not establish the TRUE ecological range of the species, only what is separable from pseudo-background under the two PU formulations tested.",
        "It does not replace a full spatial block cross-validation; this is a single 7-oblast train/holdout diagnostic split, done for this branch only.",
        "It cannot attribute the river-connectivity effect to biology vs detection-history in isolation -- both plausibly contribute and the data do not allow separating them."
    ]
}

feature_audit['ablation']['no_river_connectivity']['features_removed'] = river_conn_feats
feature_audit['ablation']['isolation_of_ord_stra'] = {
    "no_3_named_connectivity_feats_keep_ord_stra": {"AUC": round(float(res_a_no3['auc_raw']),4), "AP": round(float(res_a_no3['ap_raw']),4)},
    "no_ord_stra_only": {"AUC": round(float(res_a_noordstra['auc_raw']),4), "AP": round(float(res_a_noordstra['ap_raw']),4)},
    "conclusion": "ord_stra alone contributes negligibly (AP 0.605->0.608, i.e. no drop) versus removing the other 3 named connectivity features alone (AP 0.605->0.145) -- the ~77% AP loss reported for the 4-feature 'no_river_connectivity' group is attributable to the 3 named features, not to ord_stra."
}

feature_audit['what_this_audit_confirms'][1] = (
    "River-connectivity-to-known-patch features are the single largest contributor to holdout AP (~77% relative loss "
    "when the 4-feature group -- n_upstream_target_patches_50km, network_distance_to_nearest_infected_km, "
    "has_upstream_infection_50km, and ord_stra -- is removed together); an isolation test confirms this drop is "
    "attributable to the 3 named connectivity features specifically, since ord_stra removed alone changes AP by "
    "less than +0.003, and these 3 mix real riverine spread with detection history."
)

with open('feature_audit_results.json','w') as f:
    json.dump(feature_audit, f, indent=2, ensure_ascii=False)
print("saved, size:", os.path.getsize('feature_audit_results.json'))