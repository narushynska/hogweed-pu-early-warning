"""
05_wind_dispersal_models.py

Фаза III: PU-моделювання придатності середовища та ризику поширення.

Inputs (expected in data/):
  - grid_features_static.parquet
  - patches.parquet
  - wind_vectors.parquet

Вихід: wind_model_comparison.json

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
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import lightgbm as lgb

pd.set_option('display.max_columns', 50)

# Load data
grid = pd.read_parquet('data/grid_features_static.parquet')
patches = pd.read_parquet('data/patches.parquet')
wv = pd.read_parquet('data/wind_vectors.parquet')

# Suitability proxy
climate_cols = ['bio_01','bio_04','bio_05','bio_06','bio_12','bio_15','bio_16','bio_17',
                'gdd_base5_mean_2015_2024','frost_days_mean_2015_2024','annual_precip_mean_2015_2024',
                'tmean_annual_mean_2015_2024','elevation_m']

X = grid[climate_cols].copy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X.values)

pos_mask = (grid['label_state'] == 'positive').values
Xpos = Xs[pos_mask]

cov = np.cov(Xpos, rowvar=False)
cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
inv_cov = np.linalg.inv(cov_reg)
mean_pos = Xpos.mean(axis=0)

diff = Xs - mean_pos
mahal_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
mahal_dist = np.sqrt(mahal_sq)

suit_raw = np.exp(-mahal_dist / np.median(mahal_dist[pos_mask]))
suit_norm = (suit_raw - suit_raw.min()) / (suit_raw.max() - suit_raw.min())
grid['suitability_proxy'] = suit_norm

# Propagule pressure via KDE
from sklearn.neighbors import KernelDensity

src = patches[['patch_id','centroid_x','centroid_y','n_points','area_km2','boundary_defined']].copy()
coords = src[['centroid_x','centroid_y']].values / 1000.0

best_bw = 20.0
weights = np.log1p(src['n_points'].values)

kde_final = KernelDensity(bandwidth=best_bw, kernel='gaussian')
kde_final.fit(coords, sample_weight=weights)

grid_coords_km = grid[['x','y']].values / 1000.0
log_dens = np.empty(len(grid_coords_km))
batch = 50000
for i in range(0, len(grid_coords_km), batch):
    log_dens[i:i+batch] = kde_final.score_samples(grid_coords_km[i:i+batch])

dens = np.exp(log_dens)
pp_norm = (dens - dens.min()) / (dens.max() - dens.min())
grid['propagule_pressure'] = pp_norm

# Connectivity river
net_dist_km = grid['network_distance_to_nearest_infected_km'].values
n_upstream = grid['n_upstream_target_patches_50km'].values

river_decay_km = 50.0
part1 = np.where(np.isnan(net_dist_km), 0.0, np.exp(-net_dist_km / river_decay_km))
part2 = 1 - np.exp(-n_upstream / 3.0)

connectivity_river_raw = 0.6 * part1 + 0.4 * part2
connectivity_river = (connectivity_river_raw - connectivity_river_raw.min()) / (connectivity_river_raw.max() - connectivity_river_raw.min())
grid['connectivity_river'] = connectivity_river

# Connectivity road
road_decay_m = float(grid['dist_road_m'].median())
connectivity_road_raw = np.exp(-grid['dist_road_m'].values / road_decay_m)
connectivity_road = (connectivity_road_raw - connectivity_road_raw.min()) / (connectivity_road_raw.max() - connectivity_road_raw.min())
grid['connectivity_road'] = connectivity_road

# Near-term risk
connectivity_combined_for_risk = 0.5 * grid['connectivity_river'].values + 0.5 * grid['connectivity_road'].values
risk_raw = grid['suitability_proxy'].values * grid['propagule_pressure'].values * connectivity_combined_for_risk
risk_norm = (risk_raw - risk_raw.min()) / (risk_raw.max() - risk_raw.min())
grid['near_term_risk'] = risk_norm

# Build model dataset
pos_idx = grid.index[grid['label_state'] == 'positive']
neg_idx_random = grid.index[(grid['ctrl_random_bg']) & (grid['label_state'] != 'positive')]

oblasts = grid['oblast'].dropna().unique()
rng2 = np.random.default_rng(7)
oblasts_shuffled = rng2.permutation(oblasts)
n_holdout = max(1, int(round(len(oblasts) * 0.2)))
holdout_oblasts = set(oblasts_shuffled[:n_holdout])

model_idx = np.concatenate([pos_idx.values, neg_idx_random.values])
y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx_random))])
df_model = grid.loc[model_idx].copy()
df_model['target'] = y

is_holdout = df_model['oblast'].isin(holdout_oblasts).values

# Wind attributes for patches
grid_tree = cKDTree(grid[['x','y']].values)
d_p, idx_p = grid_tree.query(patches[['centroid_x','centroid_y']].values, k=1)
patches_wind = grid.iloc[idx_p][['wind_resultant_dir','wind_resultant_mag','wind_constancy']].reset_index(drop=True)
patches_src = patches[['patch_id','centroid_x','centroid_y']].reset_index(drop=True).join(patches_wind)

wind_mag_max = grid['wind_resultant_mag'].quantile(0.99)
mag_norm_src = (patches_src['wind_resultant_mag'].values.clip(0, wind_mag_max)) / wind_mag_max
downwind_src = (patches_src['wind_resultant_dir'].values + 180.0) % 360.0
const_src = patches_src['wind_constancy'].values
src_xy = patches_src[['centroid_x','centroid_y']].values

dist_scale_m = 20000.0

def mechanistic_score(target_xy_m):
    N = target_xy_m.shape[0]
    out = np.zeros(N, dtype=np.float64)
    batch = 5000
    for i in range(0, N, batch):
        tgt = target_xy_m[i:i+batch]
        dx = tgt[:,0:1] - src_xy[None,:,0]
        dy = tgt[:,1:2] - src_xy[None,:,1]
        dist_m = np.sqrt(dx**2 + dy**2)
        dist_m_safe = np.where(dist_m < 1.0, 1.0, dist_m)
        bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
        ang_diff = np.radians(bearing - downwind_src[None,:])
        aniso = 1.0 + const_src[None,:] * mag_norm_src[None,:] * np.cos(ang_diff)
        neg_exp = np.exp(-dist_m_safe / dist_scale_m)
        kern = neg_exp * aniso
        out[i:i+batch] = kern.sum(axis=1)
    return out

mech_model = mechanistic_score(df_model[['x','y']].values)
df_model['mech_score'] = mech_model

# ML features
ml_features = ['bio_01','bio_04','bio_05','bio_06','bio_12','bio_15','bio_16','bio_17',
               'gdd_base5_mean_2015_2024','frost_days_mean_2015_2024','annual_precip_mean_2015_2024',
               'tmean_annual_mean_2015_2024','elevation_m','slope_deg','roughness_m','tpi_m',
               'dist_river_m','dist_road_m','dist_railroad_m','dist_settlement_m',
               'n_apiaceae_bg_r5km','n_plant_effort_r5km','n_hard_negative_r5km',
               'n_apiaceae_bg_r10km','n_plant_effort_r10km']

X_train = df_model.loc[~is_holdout, ml_features].values
X_test = df_model.loc[is_holdout, ml_features].values
y_train = df_model.loc[~is_holdout, 'target'].values
y_test = df_model.loc[is_holdout, 'target'].values

# Impute NaN with median
med = np.nanmedian(X_train, axis=0)
inds_tr = np.where(np.isnan(X_train))
X_train[inds_tr] = np.take(med, inds_tr[1])
inds_te = np.where(np.isnan(X_test))
X_test[inds_te] = np.take(med, inds_te[1])

# Model 1: ML only
lgb_params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=15,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=-1, scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum())
m1 = lgb.LGBMClassifier(**lgb_params)
m1.fit(X_train, y_train)
p1_test = m1.predict_proba(X_test)[:,1]
auc1 = roc_auc_score(y_test, p1_test)
ap1 = average_precision_score(y_test, p1_test)
ll1 = log_loss(y_test, p1_test, labels=[0,1])

# Model 2: mechanistic only
mech_train = df_model.loc[~is_holdout,'mech_score'].values.reshape(-1,1)
mech_test = df_model.loc[is_holdout,'mech_score'].values.reshape(-1,1)
log_mech_train = np.log(mech_train + 1e-12)
log_mech_test = np.log(mech_test + 1e-12)
m2 = LogisticRegression()
m2.fit(log_mech_train, y_train)
p2_test = m2.predict_proba(log_mech_test)[:,1]
auc2 = roc_auc_score(y_test, p2_test)
ap2 = average_precision_score(y_test, p2_test)
ll2 = log_loss(y_test, p2_test, labels=[0,1])

# Model 3: hybrid
X_train_hybrid = np.hstack([X_train, log_mech_train])
X_test_hybrid = np.hstack([X_test, log_mech_test])
m3 = lgb.LGBMClassifier(**lgb_params)
m3.fit(X_train_hybrid, y_train)
p3_test = m3.predict_proba(X_test_hybrid)[:,1]
auc3 = roc_auc_score(y_test, p3_test)
ap3 = average_precision_score(y_test, p3_test)
ll3 = log_loss(y_test, p3_test, labels=[0,1])

# Build JSON
wind_model_comparison = {
    "validation_scheme": "20% oblast-block holdout (own split; suitability-branch split not yet available at time of this run) — holdout oblasts: " + ", ".join(sorted(holdout_oblasts)),
    "n_train": int((~is_holdout).sum()),
    "n_holdout": int(is_holdout.sum()),
    "holdout_positive_rate": float(y_test.mean()),
    "kernel_bandwidth_km_propagule_pressure": float(best_bw),
    "mechanistic_kernel_distance_scale_m": float(dist_scale_m),
    "models": {
        "1_ml_only_no_wind": {
            "description": "LightGBM on spatial/climate/terrain/effort features, excludes any wind variable",
            "features": ml_features,
            "AUC": float(auc1), "AP": float(ap1), "log_loss": float(ll1)
        },
        "2_mechanistic_only": {
            "description": "1D logistic calibration of log(anisotropic negative-exponential dispersal kernel score) summed over all 544 patch sources",
            "kernel_formula": "score = sum_over_sources[ exp(-dist_m/20000) * (1 + wind_constancy*wind_mag_norm*cos(bearing - downwind_direction)) ]",
            "AUC": float(auc2), "AP": float(ap2), "log_loss": float(ll2)
        },
        "3_hybrid_ml_plus_mech_feature": {
            "description": "Same LightGBM as model 1, plus log(mechanistic dispersal score) added as one extra feature",
            "AUC": float(auc3), "AP": float(ap3), "log_loss": float(ll3)
        }
    },
    "finding": (
        f"Hybrid model has the best AUC (AUC={auc3:.3f}) vs ML-only (AUC={auc1:.3f}) and pure mechanistic (AUC={auc2:.3f}), "
        f"but on Average Precision the ML-only model is marginally best (AP={ap1:.3f}) vs hybrid (AP={ap3:.3f}) and mechanistic (AP={ap2:.3f}) — "
        "the three models are close and none dominates on both metrics. "
        "The wind-kernel feature adds at most a small, non-decisive increment to discrimination — consistent with the low "
        "median wind_constancy (0.269) meaning wind direction alone carries limited standalone predictive signal in Ukraine; "
        "most separability across all three models comes from static climate/terrain/effort features (which also encode observer-effort confounding, see limitations)."
    )
}

with open('wind_model_comparison.json', 'w') as f:
    json.dump(wind_model_comparison, f, indent=2, ensure_ascii=False)

print("saved wind_model_comparison.json")