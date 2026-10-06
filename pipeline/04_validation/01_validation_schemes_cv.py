"""
01_validation_schemes_cv.py

Фаза IV: валідаційні схеми, бюджетні метрики, абстенція, доменний зсув.

Inputs (expected in data/):
  - grid_features_static.parquet
  - patches.parquet
  - labels_cells.parquet
  - target_points_clustered.parquet
  - suitability_predictions.parquet

Вихід: validation_scheme_results.csv

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
import host
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import lightgbm as lgb
from scipy.spatial import cKDTree

gfs = pd.read_parquet('data/grid_features_static.parquet')
tpc = pd.read_parquet('data/target_points_clustered.parquet')

feature_list = ["annual_precip_mean_2015_2024","annual_precip_std_2015_2024","bio_01","bio_02","bio_03","bio_04","bio_05","bio_06","bio_07","bio_08","bio_09","bio_10","bio_11","bio_12","bio_13","bio_14","bio_15","bio_16","bio_17","bio_18","bio_19","dist_railroad_m","dist_river_m","dist_road_m","dist_settlement_m","elevation_m","et0_annual_mean_2015_2024","et0_annual_std_2015_2024","frost_days_mean_2015_2024","frost_days_spring_mean_2015_2024","frost_days_spring_std_2015_2024","frost_days_std_2015_2024","gdd_base5_mar_jun_mean_2015_2024","gdd_base5_mar_jun_std_2015_2024","gdd_base5_mean_2015_2024","gdd_base5_std_2015_2024","has_upstream_infection_50km","n_apiaceae_bg_r10km","n_apiaceae_bg_r1km","n_apiaceae_bg_r25km","n_apiaceae_bg_r5km","n_hard_negative_r10km","n_hard_negative_r1km","n_hard_negative_r25km","n_hard_negative_r5km","n_plant_effort_r10km","n_plant_effort_r1km","n_plant_effort_r25km","n_plant_effort_r5km","n_upstream_target_patches_50km","network_distance_to_nearest_infected_km","ord_stra","rain_30d_preflower_mean_2015_2024","rain_30d_preflower_std_2015_2024","roughness_m","slope_deg","spring_precip_mean_2015_2024","spring_precip_std_2015_2024","summer_precip_mean_2015_2024","summer_precip_std_2015_2024","tmean_annual_mean_2015_2024","tmean_annual_std_2015_2024","tmean_spring_mean_2015_2024","tmean_spring_std_2015_2024","tpi_m","wind_constancy","wind_cos_mean","wind_resultant_dir","wind_resultant_mag","wind_sin_mean","wind_speed_mean_2015_2024","wind_u_mean_2015_2024","wind_v_mean_2015_2024"]

STATIC_FEATS_ALL = feature_list

tpc_cell_to_patch = tpc.drop_duplicates('cell_id').set_index('cell_id')['patch_id']

pos = gfs[gfs['label_state']=='positive'].copy()
bg = gfs[gfs['ctrl_target_group_bg']].copy()

pos['label'] = 1
bg['label'] = 0
pool = pd.concat([pos, bg], ignore_index=True)
pool['group'] = pool['cell_id'].map(tpc_cell_to_patch)
pool.loc[pool['group'].isna() & (pool['label']==0), 'group'] = 'bg_' + pool.loc[pool['group'].isna() & (pool['label']==0),'cell_id'].astype(str)
pool.loc[pool['group'].isna() & (pool['label']==1), 'group'] = 'pos_' + pool.loc[pool['group'].isna() & (pool['label']==1),'cell_id'].astype(str)

HP = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=-1
)

FEATS = STATIC_FEATS_ALL

def fit_predict(train_df, test_df, feats=FEATS):
    n_bg = (train_df['label']==0).sum()
    n_pos = (train_df['label']==1).sum()
    spw = n_bg / max(n_pos,1)
    model = lgb.LGBMClassifier(scale_pos_weight=spw, **HP)
    model.fit(train_df[feats], train_df['label'])
    pred = model.predict_proba(test_df[feats])[:,1]
    return pred, model

results = []

# Scheme (a): random split by cell, multiple seeds
aucs_a, aps_a = [], []
for seed in range(5):
    tr, te = train_test_split(pool, test_size=0.2, random_state=seed, stratify=pool['label'])
    pred, _ = fit_predict(tr, te)
    aucs_a.append(roc_auc_score(te['label'], pred))
    aps_a.append(average_precision_score(te['label'], pred))
results.append({'scheme':'a_random_split','auc_mean':np.mean(aucs_a),'auc_std':np.std(aucs_a),
                 'ap_mean':np.mean(aps_a),'ap_std':np.std(aps_a), 'n_test_mean': len(te), 'n_pos_test_mean': int(te['label'].sum())})

gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
aucs_b, aps_b, n_test_b, n_pos_b = [], [], [], []
for tr_idx, te_idx in gss.split(pool, pool['label'], groups=pool['group']):
    tr, te = pool.iloc[tr_idx], pool.iloc[te_idx]
    pred, _ = fit_predict(tr, te)
    aucs_b.append(roc_auc_score(te['label'], pred))
    aps_b.append(average_precision_score(te['label'], pred))
    n_test_b.append(len(te)); n_pos_b.append(int(te['label'].sum()))

results.append({'scheme':'b_group_by_patch','auc_mean':np.mean(aucs_b),'auc_std':np.std(aucs_b),
                 'ap_mean':np.mean(aps_b),'ap_std':np.std(aps_b),'n_test_mean':int(np.mean(n_test_b)),
                 'n_pos_test_mean':int(np.mean(n_pos_b))})

BLOCK_SIZE = 100_000
BUFFER = 50_000

def make_spatial_block_split(pool, block_size, buffer, seed, target_test_frac=0.20):
    p = pool.copy()
    p['block_x'] = (p['x'] // block_size).astype(int)
    p['block_y'] = (p['y'] // block_size).astype(int)
    p['block_id'] = p['block_x'].astype(str) + '_' + p['block_y'].astype(str)
    block_stats = p.groupby('block_id').agg(n=('label','size')).reset_index()
    blocks_shuf = block_stats.sample(frac=1, random_state=seed).reset_index(drop=True)
    cum_n = 0; test_blocks=[]
    total_n = len(p)
    for _, row in blocks_shuf.iterrows():
        if cum_n/total_n >= target_test_frac:
            break
        test_blocks.append(row['block_id']); cum_n += row['n']
    test_mask = p['block_id'].isin(set(test_blocks)).values
    train_coords = p.loc[~test_mask, ['x','y']].values
    test_coords = p.loc[test_mask, ['x','y']].values
    if len(test_coords)==0 or len(train_coords)==0:
        return None
    tree_test = cKDTree(test_coords)
    dist_to_test,_ = tree_test.query(train_coords, k=1)
    buffer_exclude = dist_to_test < buffer
    train_idx = p.index[~test_mask][~buffer_exclude]
    test_idx = p.index[test_mask]
    return p.loc[train_idx], p.loc[test_idx]

aucs_c, aps_c, n_test_c, n_pos_c = [], [], [], []
for seed in range(5):
    out = make_spatial_block_split(pool, BLOCK_SIZE, BUFFER, seed)
    if out is None: continue
    tr, te = out
    if te['label'].sum() < 5:
        continue
    pred, _ = fit_predict(tr, te)
    aucs_c.append(roc_auc_score(te['label'], pred))
    aps_c.append(average_precision_score(te['label'], pred))
    n_test_c.append(len(te)); n_pos_c.append(int(te['label'].sum()))

results.append({'scheme':'c_spatial_block_buffered','auc_mean':np.mean(aucs_c),'auc_std':np.std(aucs_c),
                 'ap_mean':np.mean(aps_c),'ap_std':np.std(aps_c),'n_test_mean':int(np.mean(n_test_c)),
                 'n_pos_test_mean':int(np.mean(n_pos_c)), 'block_size_km': BLOCK_SIZE/1000, 'buffer_km': BUFFER/1000})

holdout_oblasts = ["Ivano-Frankivsk","Zakarpattia","Ternopil","Rivne","Chernivtsi","Khmelnytskyi","Sumy"]
test_mask_d = pool['oblast'].isin(holdout_oblasts)
train_d = pool[~test_mask_d]
test_d = pool[test_mask_d]

pred_d, _ = fit_predict(train_d, test_d)
auc_d = roc_auc_score(test_d['label'], pred_d)
ap_d = average_precision_score(test_d['label'], pred_d)

results.append({'scheme':'d_leave_region_out_7oblast','auc_mean':auc_d,'auc_std':0.0,
                 'ap_mean':ap_d,'ap_std':0.0,'n_test_mean':len(test_d),'n_pos_test_mean':int(test_d['label'].sum())})

step1_df = pd.DataFrame(results)
step1_df = step1_df[['scheme','auc_mean','auc_std','ap_mean','ap_std','n_test_mean','n_pos_test_mean']]
step1_df.to_csv('validation_scheme_results.csv', index=False)