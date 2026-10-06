"""
01_field_batch_selection.py

Фаза V: пріоритизація польового батчу, карти.

Inputs (expected in data/):
  - suitability_predictions.parquet
  - grid_1km_ua.parquet
  - labels_cells.parquet
  - grid_features_static.parquet
  - abstention_results.json
  - pu_model_targetbg.pkl
  - validation_manifest.json
  - budget_metrics_results.json

Вихід: field_batch_500.csv

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import json
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# Load data
gfs = pd.read_parquet('data/grid_features_static.parquet')
sp = pd.read_parquet('data/suitability_predictions.parquet')
grid = pd.read_parquet('data/grid_1km_ua.parquet'

)

# Feature list used by suitability model
feature_list = ["annual_precip_mean_2015_2024","annual_precip_std_2015_2024","bio_01","bio_02","bio_03","bio_04","bio_05","bio_06","bio_07","bio_08","bio_09","bio_10","bio_11","bio_12","bio_13","bio_14","bio_15","bio_16","bio_17","bio_18","bio_19","dist_railroad_m","dist_river_m","dist_road_m","dist_settlement_m","elevation_m","et0_annual_mean_2015_2024","et0_annual_std_2015_2024","frost_days_mean_2015_2024","frost_days_spring_mean_2015_2024","frost_days_spring_std_2015_2024","frost_days_std_2015_2024","gdd_base5_mar_jun_mean_2015_2024","gdd_base5_mar_jun_std_2015_2024","gdd_base5_mean_2015_2024","gdd_base5_std_2015_2024","has_upstream_infection_50km","n_apiaceae_bg_r10km","n_apiaceae_bg_r1km","n_apiaceae_bg_r25km","n_apiaceae_bg_r5km","n_hard_negative_r10km","n_hard_negative_r1km","n_hard_negative_r25km","n_hard_negative_r5km","n_plant_effort_r10km","n_plant_effort_r1km","n_plant_effort_r25km","n_plant_effort_r5km","n_upstream_target_patches_50km","network_distance_to_nearest_infected_km","ord_stra","rain_30d_preflower_mean_2015_2024","rain_30d_preflower_std_2015_2024","roughness_m","slope_deg","spring_precip_mean_2015_2024","spring_precip_std_2015_2024","summer_precip_mean_2015_2024","summer_precip_std_2015_2024","tmean_annual_mean_2015_2024","tmean_annual_std_2015_2024","tmean_spring_mean_2015_2024","tmean_spring_std_2015_2024","tpi_m","wind_constancy","wind_cos_mean","wind_resultant_dir","wind_resultant_mag","wind_sin_mean","wind_speed_mean_2015_2024","wind_u_mean_2015_2024","wind_v_mean_2015_2024"]

FEATS_FOR_MAHAL = [f for f in feature_list if f not in ('has_upstream_infection_50km',)]

# Compute Mahalanobis distance
gfs_mahal = gfs[FEATS_FOR_MAHAL].copy()
gfs_mahal['network_distance_to_nearest_infected_km'] = gfs_mahal['network_distance_to_nearest_infected_km'].fillna(9999)
gfs_mahal['slope_deg'] = gfs_mahal['slope_deg'].fillna(gfs_mahal['slope_deg'].median())

X_full2 = gfs_mahal.values
mu_full2 = X_full2.mean(axis=0)
sigma_full2 = X_full2.std(axis=0)
sigma_full2[sigma_full2==0] = 1.0
X_full2_z = (X_full2 - mu_full2) / sigma_full2

pos_idx = gfs['label_state']=='positive'
X_pos2_z = X_full2_z[pos_idx.values]
pos_mean2_z = X_pos2_z.mean(axis=0)
pos_cov2_z = np.cov(X_pos2_z, rowvar=False)
reg2 = 1e-2*np.eye(pos_cov2_z.shape[0])
cov_inv2 = inv(pos_cov2_z + reg2)

diff2 = X_full2_z - pos_mean2_z
mahal_sq2 = np.einsum('ij,jk,ik->i', diff2, cov_inv2, diff2)
mahal_dist2 = np.sqrt(np.clip(mahal_sq2, 0, None))
gfs['mahal_dist'] = mahal_dist2

# Build uq_full
uq_full = gfs[['cell_id','oblast']].merge(sp[['cell_id','suitability_targetbg','suitability_targetbg_raw','ensemble_mean','ensemble_std']], on='cell_id')
uq_full['mahal_dist'] = gfs['mahal_dist'].values
uq_full['label_state'] = gfs['label_state'].values

# Budget threshold from step 4: holdout oblasts, rank 500 by raw score
holdout_oblasts = ["Ivano-Frankivsk","Zakarpattia","Ternopil","Rivne","Chernivtsi","Khmelnytskyi","Sumy"]
holdout_full = gfs.loc[gfs['oblast'].isin(holdout_oblasts), ['cell_id','oblast','label_state']].merge(
    sp[['cell_id','suitability_targetbg','suitability_targetbg_raw']], on='cell_id', how='left')
holdout_by_raw = holdout_full.sort_values('suitability_targetbg_raw', ascending=False).reset_index(drop=True)
threshold_budget_raw = holdout_by_raw.iloc[499]['suitability_targetbg_raw']

# OOD cutoff: 95th percentile of Mahalanobis distance among known positives
mahal_pos = gfs.loc[gfs['label_state']=='positive','mahal_dist']
ood_cutoff = np.percentile(mahal_pos.dropna(), 95)

# Evidence density
effort_25 = gfs.set_index('cell_id')['n_plant_effort_r25km']
uq_full['n_plant_effort_r25km'] = uq_full['cell_id'].map(effort_25)

low_effort_cutoff_25km = np.percentile(uq_full['n_plant_effort_r25km'], 25)

# Action categories
high_suit_thresh = threshold_budget_raw
uq_full['is_high_suitability'] = uq_full['suitability_targetbg_raw'] >= high_suit_thresh
uq_full['is_high_ood'] = uq_full['mahal_dist'] > ood_cutoff
uq_full['is_low_evidence_density'] = uq_full['n_plant_effort_r25km'] <= low_effort_cutoff_25km
uq_full['is_high_ood_combined'] = uq_full['is_high_ood'] | uq_full['is_low_evidence_density']

def categorize2(row):
    if not row['is_high_suitability']:
        return 'low_suitability_no_action'
    if row['is_high_ood_combined']:
        return 'high_suitability_high_uncertainty_EXPERT_REVIEW'
    else:
        return 'high_suitability_low_uncertainty_DIRECT_PRIORITY'

uq_full['action_category_v2'] = uq_full.apply(categorize2, axis=1)

# Build work dataframe
work = grid[['cell_id','x','y','lon','lat','oblast']].merge(
    sp[['cell_id','suitability_targetbg_raw','suitability_targetbg']], on='cell_id', how='left'
).merge(
    gfs[['cell_id','label_state','n_plant_effort_r1km','n_plant_effort_r5km','n_plant_effort_r10km','n_plant_effort_r25km','bio_01','bio_12']], on='cell_id', how='left'
).merge(
    uq_full[['cell_id','action_category_v2']], on='cell_id', how='left'
)

# Exploit pool
eligible = work[(work['label_state']!='positive') & (work['action_category_v2']!='high_suitability_high_uncertainty_EXPERT_REVIEW')].copy()

exploit_pool = eligible.sort_values('suitability_targetbg_raw', ascending=False).reset_index(drop=True)
exploit = exploit_pool.head(250).copy()
exploit['category'] = 'exploit'

threshold = threshold_budget_raw

exploit_ids = set(exploit['cell_id'])
expert_ids = set(work.loc[work['action_category_v2']=='high_suitability_high_uncertainty_EXPERT_REVIEW','cell_id'])

remaining = eligible[~eligible['cell_id'].isin(exploit_ids)].copy()

dist_to_thresh = (remaining['suitability_targetbg_raw'] - threshold).abs()
remaining['dist_to_thresh'] = dist_to_thresh

remaining['signed_diff'] = remaining['suitability_targetbg_raw'] - threshold

def farthest_point_sampling(coords, k, seed_idx=None, min_dist=None):
    n = coords.shape[0]
    selected = []
    if seed_idx is None:
        seed_idx = 0
    selected.append(seed_idx)
    dist = np.linalg.norm(coords - coords[seed_idx], axis=1)
    while len(selected) < k:
        next_idx = np.argmax(dist)
        if min_dist is not None and dist[next_idx] < min_dist:
            break
        selected.append(next_idx)
        new_dist = np.linalg.norm(coords - coords[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)
    return selected

band_width = 0.10
unc_pool = remaining[remaining['dist_to_thresh'] <= band_width].reset_index(drop=True)

coords = unc_pool[['x','y']].values
seed_idx = unc_pool['dist_to_thresh'].idxmin()
sel_idx = farthest_point_sampling(coords, k=150, seed_idx=seed_idx, min_dist=15000)

uncertainty = unc_pool.iloc[sel_idx].copy()
uncertainty['category'] = 'uncertainty'

used_ids = set(exploit['cell_id']) | set(uncertainty['cell_id']) | expert_ids
low_pool = work[(work['action_category_v2']=='low_suitability_no_action') & (~work['cell_id'].isin(used_ids))].copy()

unc_band_low = unc_pool['suitability_targetbg_raw'].min()

low_pool2 = low_pool[low_pool['suitability_targetbg_raw'] < unc_band_low].copy()
low_pool2['effort_r10km'] = low_pool2['n_plant_effort_r10km']

np.random.seed(42)

accessible_oblasts = [o for o in low_pool2['oblast'].unique() if o not in ('Crimea','Sevastopol')]

oblast_effort = work[~work['oblast'].isin(['Crimea','Sevastoport','Sevastopol'])].groupby('oblast')['n_plant_effort_r10km'].median()
oblast_effort = oblast_effort.reindex(accessible_oblasts)

inv_w = 1.0 / (oblast_effort + 5.0)
inv_w = inv_w / inv_w.sum()

def allocate_capped(weights, total, min_n=1, max_n=12):
    n_ob = len(weights)
    alloc = pd.Series(min_n, index=weights.index)
    remaining = total - alloc.sum()
    w = weights.copy()
    while remaining > 0:
        capacity = (max_n - alloc)
        active = capacity[capacity>0]
        if active.empty:
            break
        w_active = w.reindex(active.index)
        w_active = w_active / w_active.sum()
        add = (w_active * remaining).round().astype(int)
        add = np.minimum(add, capacity)
        add = add[add>0]
        if add.empty:
            idx = w_active.sort_values(ascending=False).index
            for i in idx:
                if remaining<=0: break
                if capacity[i]>0:
                    alloc[i]+=1
                    remaining-=1
            break
        alloc[add.index] += add
        remaining = total - alloc.sum()
    return alloc

alloc = allocate_capped(inv_w, 100, min_n=1, max_n=12)

np.random.seed(7)

def sample_oblast(df_ob, n, min_dist=8000, weight_col='n_plant_effort_r10km'):
    df_ob = df_ob.reset_index(drop=True)
    w = 1.0 / (df_ob[weight_col].clip(lower=0) + 1.0)
    w = w / w.sum()
    order = np.random.choice(df_ob.index, size=len(df_ob), replace=False, p=w.values)
    chosen = []
    chosen_coords = []
    for idx in order:
        if len(chosen) >= n:
            break
        pt = df_ob.loc[idx, ['x','y']].values.astype(float)
        if chosen_coords:
            dmin = np.min(np.linalg.norm(np.array(chosen_coords) - pt, axis=1))
            if dmin < min_dist:
                continue
        chosen.append(idx)
        chosen_coords.append(pt)
    return df_ob.loc[chosen]

exploration_parts = []
for ob, n in alloc.items():
    sub = low_pool2[low_pool2['oblast']==ob]
    if sub.empty or n==0:
        continue
    picked = sample_oblast(sub, n)
    exploration_parts.append(picked)

exploration = pd.concat(exploration_parts, ignore_index=True)
exploration['category'] = 'exploration'

exploit['category'] = 'exploit'
uncertainty['category'] = 'uncertainty'

def make_justification(row):
    if row['category']=='exploit':
        return f"Top-ranked raw suitability score ({row['suitability_targetbg_raw']:.4f}) in DIRECT_PRIORITY action category; model has high confidence and adequate local survey effort."
    elif row['category']=='uncertainty':
        return f"Raw suitability ({row['suitability_targetbg_raw']:.4f}) sits near the decision threshold (0.988); geographically diversified via farthest-point sampling (>=15km spacing) to sample the model's decision boundary across multiple regions/oblasts rather than one cluster."
    else:
        return f"Low suitability ({row['suitability_targetbg_raw']:.6f}) but drawn with weight inversely proportional to local botanical survey density (n_plant_effort_r10km={row['n_plant_effort_r10km']:.0f}) to probe under-surveyed 'blind spot' regions for potential false negatives."

cols = ['cell_id','lon','lat','oblast','category','suitability_targetbg_raw']
parts = []
for part in [exploit, uncertainty, exploration]:
    p = part[cols].copy()
    p['justification'] = part.apply(make_justification, axis=1)
    parts.append(p)

field_batch = pd.concat(parts, ignore_index=True)
field_batch = field_batch.rename(columns={'suitability_targetbg_raw':'suitability_raw'})

field_batch_out = field_batch[['cell_id','lon','lat','oblast','category','suitability_raw','justification']].copy()
field_batch_out = field_batch_out.sort_values(['category','suitability_raw'], ascending=[True, False]).reset_index(drop=True)
field_batch_out.to_csv('field_batch_500.csv', index=False)