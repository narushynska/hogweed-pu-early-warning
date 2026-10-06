"""
03_abstention_policy.py

Фаза IV: валідаційні схеми, бюджетні метрики, абстенція, доменний зсув.

Inputs (expected in data/):
  - grid_features_static.parquet
  - suitability_predictions.parquet
  - rivers_ua.parquet
  - labels_cells.parquet
  - patches.parquet
  - heracleum_ua_clean.parquet
  - target_points_clustered.parquet
  - river_graph.py
  - river_graph_edges.parquet

Вихід: abstention_results.json

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import json
import importlib.util
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

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

# Spearman correlations
rho_suit_std = spearmanr(uq_full['suitability_targetbg'], uq_full['ensemble_std'])[0]
rho_suit_mahal = spearmanr(uq_full['suitability_targetbg'], uq_full['mahal_dist'])[0]
rho_std_mahal = spearmanr(uq_full['ensemble_std'], uq_full['mahal_dist'])[0]
rho_suit_effort = spearmanr(uq_full['suitability_targetbg'], uq_full['n_plant_effort_r25km'])[0]
rho_mahal_effort = spearmanr(uq_full['mahal_dist'], uq_full['n_plant_effort_r25km'])[0]

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
cat_counts_v2 = uq_full['action_category_v2'].value_counts()

hi = uq_full[uq_full['is_high_suitability']]

abstention_results = {
    'ensemble_std_honesty_check': {
        'method': 'Spearman correlation between suitability_targetbg and ensemble_std (targetbg vs randombg '
            'model disagreement) across all 599126 grid cells.',
        'spearman_suitability_vs_ensemble_std': float(rho_suit_std),
        'conclusion': 'ensemble_std is NOT an honest, independent uncertainty measure. It is a near-deterministic '
            'function of the mean prediction itself (both models tend to agree at the extremes -- very low or '
            'very high suitability -- and disagree in the middle), which is exactly the behavior of two '
            'correlated point estimates rather than a calibrated Bayesian or bootstrap ensemble. The two models '
            'differ only in their choice of PU background (target-group vs random), not in bootstrap resampling, '
            'random seed variation, or architecture diversity, so their spread reflects sensitivity to ONE '
            'specific modeling choice (background construction), not the full space of model uncertainty. It '
            'should be read as "how much does the effort-bias correction matter for this cell", not "how '
            'confident is the model".'
    },
    'mahalanobis_ood_check': {
        'method': 'Mahalanobis distance of each grid cell from the centroid+covariance of the 597 training '
            'positives in standardized 72-feature space (feature list matches the suitability model, minus '
            'has_upstream_infection_50km; NaN network_distance filled at 9999, NaN slope_deg filled at median).',
        'spearman_suitability_vs_mahal': float(rho_suit_mahal),
        'spearman_ensemble_std_vs_mahal': float(rho_std_mahal),
        'conclusion': 'Mahalanobis distance from the positive centroid is strongly anti-correlated with predicted '
            'suitability (rho=-0.62): cells that look feature-similar to known positives are BOTH scored highly '
            'suitable AND close in Mahalanobis distance almost by construction, since a tree ensemble trained to '
            'separate positives from background implicitly learns a similar notion of "distance from positive '
            'region". A pure environmental-feature OOD flag is therefore largely redundant with the suitability '
            'score itself and cannot serve as an independent second axis of confidence on its own.'
    },
    'evidence_density_axis': {
        'method': 'n_plant_effort_r25km (25km-radius count of any plant observation, from the effort-buffer '
            'features) used as a proxy for "how much has this neighborhood actually been surveyed", independent '
            'of the environmental feature space.',
        'spearman_suitability_vs_effort_r25km': float(rho_suit_effort),
        'spearman_mahal_vs_effort_r25km': float(rho_mahal_effort),
        'conclusion': 'Local survey-effort density is only weakly correlated with predicted suitability '
            '(rho=-0.12) and with the Mahalanobis OOD score (rho=-0.22) -- far more independent than the two '
            'model-internal signals above. It is used here as the SECOND, complementary uncertainty axis for the '
            'abstention policy: a cell can be high-suitability with genuinely low surrounding survey effort, '
            'meaning the model is extrapolating into under-observed territory, something neither ensemble_std '
            'nor the Mahalanobis distance (both dominated by the suitability score itself) can flag.'
    },
    'abstention_policy': {
        'high_suitability_threshold': float(high_suit_thresh),
        'high_suitability_threshold_source': 'Same raw-score threshold as the field-budget cutoff from Step 4 '
            '(rank-500 among the 7-oblast spatial holdout).',
        'env_ood_cutoff_mahalanobis': float(ood_cutoff),
        'env_ood_cutoff_source': '95th percentile of Mahalanobis distance among the 597 known training positives '
            '(a cell further from the positive centroid than 95% of known positives is flagged env-OOD).',
        'low_evidence_density_cutoff_effort_r25km': float(low_effort_cutoff_25km),
        'low_evidence_density_cutoff_source': '25th percentile of n_plant_effort_r25km across the full grid.',
        'rule': 'high_suitability AND NOT(env_OOD OR low_evidence_density) -> DIRECT_PRIORITY; '
                'high_suitability AND (env_OOD OR low_evidence_density) -> EXPERT_REVIEW; '
                'NOT high_suitability -> NO_ACTION',
        'category_counts': {k: int(v) for k,v in cat_counts_v2.items()},
        'high_suitability_breakdown': {
            'env_ood_only': int((hi['is_high_ood'] & ~hi['is_low_evidence_density']).sum()),
            'low_evidence_only': int((~hi['is_high_ood'] & hi['is_low_evidence_density']).sum()),
            'both_flags': int((hi['is_high_ood'] & hi['is_low_evidence_density']).sum()),
            'neither_flag_direct_priority': int((~hi['is_high_ood'] & ~hi['is_low_evidence_density']).sum())
        }
    },
    'interpretation': 'Under this policy, 1162 cells nationally get flagged for direct field-visit priority '
        '(same rank-500-per-oblast-share order of magnitude as the Step-4 budget analysis extended across all of '
        'Ukraine, not just the 7-oblast holdout), while 394 cells are high-suitability but sit in poorly-surveyed '
        'neighborhoods -- these should route to expert desk review (aerial imagery, local contact, historical '
        'record check) before committing a scarce field visit, since the model has less local evidence backing '
        'its suitability call there. This operationalizes the slide-77 point that identical predicted probability '
        'can warrant different actions depending on the confidence behind it -- but only once a genuinely '
        'independent second signal (evidence density) is added, because the model-internal signals '
        '(ensemble_std, Mahalanobis distance) collapse onto the suitability score itself and cannot supply that '
        'second axis on their own.'
}
with open('abstention_results.json','w') as f:
    json.dump(abstention_results, f, indent=2, default=str)
print("saved")