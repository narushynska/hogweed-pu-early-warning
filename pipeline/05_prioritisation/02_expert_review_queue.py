"""
02_expert_review_queue.py

Фаза V: пріоритизація польового батчу, карти.

Inputs (expected in data/):
  - suitability_predictions.parquet
  - grid_1km_ua.parquet
  - grid_features_static.parquet
  - abstention_results.json
  - pu_model_targetbg.pkl
  - labels_cells.parquet
  - validation_manifest.json
  - budget_metrics_results.json

Вихід: expert_review_queue.csv

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

# Load data
gfs = pd.read_parquet('data/grid_features_static.parquet')
sp = pd.read_parquet('data/suitability_predictions.parquet')

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
grid = pd.read_parquet('data/grid_1km_ua.parquet')
work = grid[['cell_id','x','y','lon','lat','oblast']].merge(
    sp[['cell_id','suitability_targetbg_raw','suitability_targetbg']], on='cell_id', how='left'
).merge(
    gfs[['cell_id','label_state','n_plant_effort_r1km','n_plant_effort_r5km','n_plant_effort_r10km','n_plant_effort_r25km','bio_01','bio_12']], on='cell_id', how='left'
).merge(
    uq_full[['cell_id','action_category_v2']], on='cell_id', how='left'
)

# Build expert_review
expert_review = work[work['action_category_v2']=='high_suitability_high_uncertainty_EXPERT_REVIEW'].copy()

flags = uq_full[['cell_id','is_high_ood','is_low_evidence_density','n_plant_effort_r25km','mahal_dist']]
expert_review = expert_review.drop(columns=[c for c in ['is_high_ood','is_low_evidence_density','n_plant_effort_r25km','mahal_dist'] if c in expert_review.columns])
expert_review = expert_review.merge(flags, on='cell_id', how='left')

def er_reason(row):
    if row['is_high_ood'] and row['is_low_evidence_density']:
        flag = "both env-OOD (Mahalanobis > 95th pct of known positives) and low local survey effort (n_plant_effort_r25km <= 25th pct)"
    elif row['is_high_ood']:
        flag = "env-OOD (Mahalanobis > 95th pct of known positives) -- feature space far from training positives"
    else:
        flag = "low local survey effort (n_plant_effort_r25km <= 25th pct) -- possible genuine observation blind spot, not necessarily model uncertainty"
    return (f"High raw suitability ({row['suitability_targetbg_raw']:.4f}, >= budget threshold 0.988) but flagged {flag}. "
            f"Route to expert desk review (aerial imagery / local contact / historical record check) before committing a field visit -- "
            f"could be a true unsurveyed invasion front or a model extrapolation artifact outside its training domain.")

expert_review['justification'] = expert_review.apply(er_reason, axis=1)
expert_review_out = expert_review[['cell_id','lon','lat','oblast','suitability_targetbg_raw','is_high_ood','is_low_evidence_density','n_plant_effort_r25km','mahal_dist','justification']].rename(
    columns={'suitability_targetbg_raw':'suitability_raw'})

expert_review_final = expert_review_out.sort_values('suitability_raw', ascending=False).reset_index(drop=True)
expert_review_final.to_csv('expert_review_queue.csv', index=False)
print(expert_review_final.shape)
print(expert_review_final.columns.tolist())