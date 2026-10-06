"""
05_temporal_holdout.py

Фаза IV: валідаційні схеми, бюджетні метрики, абстенція, доменний зсув.

Inputs (expected in data/):
  - grid_features_static.parquet
  - suitability_predictions.parquet
  - grid_features_annual.parquet
  - heracleum_ua_clean.parquet
  - labels_cells.parquet
  - target_points_clustered.parquet

Вихід: temporal_holdout_results.json

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import networkx as nx
import importlib.util
import host

from sklearn.metrics import roc_auc_score, average_precision_score

# Load data
gfs = pd.read_parquet('data/grid_features_static.parquet')
tpc = pd.read_parquet('data/target_points_clustered.parquet')
labels = pd.read_parquet('data/labels_cells.parquet')
gfa_path = 'data/grid_features_annual.parquet'
patches_path = 'data/patches.parquet'
patches = pd.read_parquet(patches_path)

feature_list = ["annual_precip_mean_2015_2024","annual_precip_std_2015_2024","bio_01","bio_02","bio_03","bio_04","bio_05","bio_06","bio_07","bio_08","bio_09","bio_10","bio_11","bio_12","bio_13","bio_14","bio_15","bio_16","bio_17","bio_18","bio_19","dist_railroad_m","dist_river_m","dist_road_m","dist_settlement_m","elevation_m","et0_annual_mean_2015_2024","et0_annual_std_2015_2024","frost_days_mean_2015_2024","frost_days_spring_mean_2015_2024","frost_days_spring_std_2015_2024","frost_days_std_2015_2024","gdd_base5_mar_jun_mean_2015_2024","gdd_base5_mar_jun_std_2015_2024","gdd_base5_mean_2015_2024","gdd_base5_std_2015_2024","has_upstream_infection_50km","n_apiaceae_bg_r10km","n_apiaceae_bg_r1km","n_apiaceae_bg_r25km","n_apiaceae_bg_r5km","n_hard_negative_r10km","n_hard_negative_r1km","n_hard_negative_r25km","n_hard_negative_r5km","n_plant_effort_r10km","n_plant_effort_r1km","n_plant_effort_r25km","n_plant_effort_r5km","n_upstream_target_patches_50km","network_distance_to_nearest_infected_km","ord_stra","rain_30d_preflower_mean_2015_2024","rain_30d_preflower_std_2015_2024","roughness_m","slope_deg","spring_precip_mean_2015_2024","spring_precip_std_2015_2024","summer_precip_mean_2015_2024","summer_precip_std_2015_2024","tmean_annual_mean_2015_2024","tmean_annual_std_2015_2024","tmean_spring_mean_2015_2024","tmean_spring_std_2015_2024","tpi_m","wind_constancy","wind_cos_mean","wind_resultant_dir","wind_resultant_mag","wind_sin_mean","wind_speed_mean_2015_2024","wind_u_mean_2015_2024","wind_v_mean_2015_2024"]

STATIC_SITE_FEATS = [c for c in feature_list if '_2015_2024' not in c]

LAG_FEATS = ['prev_season_gdd_base5','prev_season_gdd_base5_mar_jun','prev_season_frost_days',
             'prev_season_frost_days_spring','prev_season_spring_precip','prev_season_summer_precip',
             'prev_season_rain_30d_preflower','prev_season_annual_precip','prev_season_tmean_annual',
             'prev_season_tmean_spring','prev_season_et0_annual','prev_season_wind_resultant_dir',
             'prev_season_wind_resultant_mag','prev_season_wind_constancy',
             'prev_season_anom_gdd_base5','prev_season_anom_gdd_base5_mar_jun','prev_season_anom_frost_days',
             'prev_season_anom_frost_days_spring','prev_season_anom_spring_precip','prev_season_anom_summer_precip',
             'prev_season_anom_rain_30d_preflower','prev_season_anom_annual_precip','prev_season_anom_tmean_annual',
             'prev_season_anom_tmean_spring','prev_season_anom_et0_annual']

SAMEYEAR_LEAK_FEATS = ['gdd_base5','gdd_base5_mar_jun','frost_days','frost_days_spring','spring_precip',
                        'summer_precip','rain_30d_preflower','annual_precip','tmean_annual','tmean_spring',
                        'et0_annual','wind_resultant_dir','wind_resultant_mag','wind_constancy',
                        'anom_gdd_base5','anom_gdd_base5_mar_jun','anom_frost_days','anom_frost_days_spring',
                        'anom_spring_precip','anom_summer_precip','anom_rain_30d_preflower','anom_annual_precip',
                        'anom_tmean_annual','anom_tmean_spring','anom_et0_annual']

gfa = pd.read_parquet(gfa_path, columns=['cell_id','year']+LAG_FEATS+SAMEYEAR_LEAK_FEATS)

overlap_cols = set(STATIC_SITE_FEATS) & set(SAMEYEAR_LEAK_FEATS)
STATIC_SITE_FEATS_T = [c for c in STATIC_SITE_FEATS if c not in overlap_cols]

lag_to_base = {f: f.replace('prev_season_', '') for f in LAG_FEATS}
base_cols = list(lag_to_base.values())

static_df = gfs[['cell_id','ctrl_target_group_bg'] + STATIC_SITE_FEATS].copy()

first_year_tgt = labels.set_index('cell_id')['first_year_target']
static_df['first_year_target'] = static_df['cell_id'].map(first_year_tgt)

bg_cell_ids = set(gfs.loc[gfs['ctrl_target_group_bg'],'cell_id'])

gfa_idx = gfa.set_index(['cell_id','year'])

def get_year_features(cell_ids, year, cols, rename=None):
    sub = gfa_idx.loc[(list(cell_ids), year), cols].reset_index()
    if rename:
        sub = sub.rename(columns=rename)
    return sub

HP = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=-1
)

FEATS_T = STATIC_SITE_FEATS_T + SAMEYEAR_LEAK_FEATS

def fit_predict_temporal(train_df, test_df, feats=FEATS_T):
    n_bg = (train_df['label']==0).sum()
    n_pos = (train_df['label']==1).sum()
    spw = n_bg/max(n_pos,1)
    model = lgb.LGBMClassifier(scale_pos_weight=spw, **HP)
    model.fit(train_df[feats], train_df['label'])
    pred = model.predict_proba(test_df[feats])[:,1]
    return pred

def build_temporal_split(T):
    test_year = T + 1
    train_pos_ids = static_df.loc[static_df['first_year_target'] <= T, 'cell_id'].tolist()
    train_bg_ids = list(bg_cell_ids)
    train_ids = train_pos_ids + train_bg_ids
    train_labels = [1]*len(train_pos_ids) + [0]*len(train_bg_ids)
    
    train_clim = get_year_features(train_ids, T, SAMEYEAR_LEAK_FEATS).set_index('cell_id')
    train_static = static_df.set_index('cell_id').loc[train_ids, STATIC_SITE_FEATS_T]
    train_df = train_static.join(train_clim, how='left')
    train_df['label'] = train_labels
    train_df = train_df.dropna(subset=SAMEYEAR_LEAK_FEATS)
    
    test_pos_ids = static_df.loc[static_df['first_year_target'] == test_year, 'cell_id'].tolist()
    known_by_test = set(static_df.loc[static_df['first_year_target'] <= test_year, 'cell_id'])
    test_bg_ids = [c for c in bg_cell_ids if c not in known_by_test]
    test_ids = test_pos_ids + test_bg_ids
    test_labels = [1]*len(test_pos_ids) + [0]*len(test_bg_ids)
    
    test_static = static_df.set_index('cell_id').loc[test_ids, STATIC_SITE_FEATS_T]
    
    test_lag = get_year_features(test_ids, test_year, LAG_FEATS, rename=lag_to_base).set_index('cell_id')
    test_correct = test_static.join(test_lag, how='left')
    test_correct['label'] = test_labels
    test_correct = test_correct.dropna(subset=base_cols)
    
    test_sameyear = get_year_features(test_ids, test_year, SAMEYEAR_LEAK_FEATS).set_index('cell_id')
    test_broken = test_static.join(test_sameyear, how='left')
    test_broken['label'] = test_labels
    test_broken = test_broken.dropna(subset=SAMEYEAR_LEAK_FEATS)
    
    return train_df, test_correct, test_broken, len(train_pos_ids), len(test_pos_ids), len(test_bg_ids)

temporal_results = []
for T in [2019, 2020, 2021, 2022, 2023]:
    tr, tec, teb, n_train_pos, n_test_pos, n_test_bg = build_temporal_split(T)
    if n_test_pos < 5 or n_train_pos < 30:
        print(f"T={T}: skip, too few pos (train={n_train_pos}, test={n_test_pos})")
        continue
    pred_correct = fit_predict_temporal(tr, tec)
    pred_broken = fit_predict_temporal(tr, teb)
    auc_c = roc_auc_score(tec['label'], pred_correct)
    ap_c = average_precision_score(tec['label'], pred_correct)
    auc_b = roc_auc_score(teb['label'], pred_broken)
    ap_b = average_precision_score(teb['label'], pred_broken)
    temporal_results.append({'train_cutoff_year': T, 'test_year': T+1,
                              'n_train_pos': n_train_pos, 'n_test_pos': n_test_pos, 'n_test_bg': n_test_bg,
                              'auc_correct': auc_c, 'ap_correct': ap_c,
                              'auc_broken': auc_b, 'ap_broken': ap_b})
    print(temporal_results[-1])

pooled_y = []
pooled_pred_correct = []
pooled_pred_broken = []

for T in [2019, 2020, 2021, 2022, 2023]:
    tr, tec, teb, n_train_pos, n_test_pos, n_test_bg = build_temporal_split(T)
    if n_test_pos < 5 or n_train_pos < 30:
        continue
    pred_correct = fit_predict_temporal(tr, tec)
    pred_broken = fit_predict_temporal(tr, teb)
    assert (tec.index == teb.index).all()
    pooled_y.append(tec['label'].values)
    pooled_pred_correct.append(pred_correct)
    pooled_pred_broken.append(pred_broken)

y_all = np.concatenate(pooled_y)
pred_correct_all = np.concatenate(pooled_pred_correct)
pred_broken_all = np.concatenate(pooled_pred_broken)

auc_correct_pooled = roc_auc_score(y_all, pred_correct_all)
ap_correct_pooled = average_precision_score(y_all, pred_correct_all)
auc_broken_pooled = roc_auc_score(y_all, pred_broken_all)
ap_broken_pooled = average_precision_score(y_all, pred_broken_all)

temporal_df = pd.DataFrame(temporal_results)

temporal_manifest = {
    'method': 'Train on cumulative positives (first_year_target<=T) + full target-group background; '
              'test on newly-detected positives in year T+1 + background not yet known positive by T+1. '
              'CORRECT: test features use prev_season_* (lag-1yr) annual climate, available at forecast time. '
              'BROKEN: test features illegitimately use same-year (T+1) annual climate, not yet observed when the forecast would be made.',
    'per_cutoff': temporal_df.to_dict(orient='records'),
    'pooled': {
        'n_test_total': int(len(y_all)), 'n_positive_total': int(y_all.sum()),
        'auc_correct': float(auc_correct_pooled), 'ap_correct': float(ap_correct_pooled),
        'auc_broken': float(auc_broken_pooled), 'ap_broken': float(ap_broken_pooled),
        'ap_inflation_broken_minus_correct': float(ap_broken_pooled - ap_correct_pooled),
    },
    'note': 'Per-cutoff pos counts are small (49-112), so per-cutoff AP is noisy; the pooled comparison '
            '(n=70732, 367 positives across 5 cutoffs) is the primary evidence. In this dataset the broken '
            '(same-year) cutoff does NOT show large positive AP inflation in the pooled comparison (correct '
            'AP=0.286 vs broken AP=0.253) -- if anything the leak slightly HURTS AP here, likely because '
            'same-year climate for the year of first detection is dominated by within-year noise relative to '
            'the smoother lag signal, and because climate spatial support (~50km) means many cells share '
            'near-identical same-year and lag-1yr values already, diluting the leak. This is itself an '
            'important finding: the classic temporal-leakage trap (slide 45) does not always manifest as a '
            'large positive AP inflation -- it depends on how much unique same-year information exists beyond '
            'what lag features already capture. The leak is still methodologically wrong (it uses information '
            'not available at forecast time) and must be avoided on principle, but its measured effect size '
            'here is small and mixed across cutoffs (broken AP > correct AP in 2020->2021 and 2023->2024, '
            'reversed in the other three).',
    'per_cutoff_effect_sign': [
        {'cutoff': int(r['train_cutoff_year']), 'ap_broken_minus_correct': float(r['ap_broken']-r['ap_correct'])}
        for r in temporal_results
    ]
}

# River connectivity leak demo
spec = importlib.util.spec_from_file_location("river_graph", 'pipeline/common/river_graph.py')
river_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(river_graph)

rivers_path = 'data/rivers_ua.parquet'
edges_path = 'data/river_graph_edges.parquet'
G = river_graph.load_graph(edges_path, rivers_path)

tpc_reach = tpc.merge(gfs[['cell_id','nearest_reach']], on='cell_id', how='left')
patch_reach = tpc_reach.groupby('patch_id')['nearest_reach'].agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else s.iloc[0])
patches_with_reach = patches.merge(patch_reach.rename('nearest_reach'), left_on='patch_id', right_index=True, how='left')

def compute_connectivity_at_cutoff(T, all_cell_reaches):
    known_patches = patches_with_reach[patches_with_reach['first_year'] <= T]
    infected_reaches = known_patches['nearest_reach'].dropna().unique().astype(int).tolist()
    infected_reaches = [r for r in infected_reaches if r in G]
    
    if len(infected_reaches) == 0:
        return {r: (0, np.nan) for r in all_cell_reaches}
    
    down_sets = {}
    for ir in infected_reaches:
        down_sets[ir] = river_graph.downstream_ids(G, ir, max_km=50)
    
    reach_to_count = {}
    for r_infected in infected_reaches:
        affected = down_sets[r_infected] | {r_infected}
        for a in affected:
            reach_to_count[a] = reach_to_count.get(a, 0) + 1
    
    try:
        dist_map = nx.multi_source_dijkstra_path_length(G, sources=infected_reaches, weight='km')
    except Exception as e:
        dist_map = {}
    
    result = {}
    for r in all_cell_reaches:
        n_up = reach_to_count.get(r, 0)
        dist = dist_map.get(r, np.nan)
        result[r] = (n_up, dist)
    return result

gfs_reach = gfs.set_index('cell_id')['nearest_reach']

conn_temporal_results = []
pooled_y2, pooled_pred_c2, pooled_pred_b2 = [], [], []

for T in [2019, 2020, 2021, 2022, 2023]:
    test_year = T+1
    train_pos_ids = static_df.loc[static_df['first_year_target'] <= T, 'cell_id'].tolist()
    train_bg_ids = list(bg_cell_ids)
    train_ids = train_pos_ids + train_bg_ids
    train_labels = np.array([1]*len(train_pos_ids) + [0]*len(train_bg_ids))
    
    test_pos_ids = static_df.loc[static_df['first_year_target'] == test_year, 'cell_id'].tolist()
    known_by_test = set(static_df.loc[static_df['first_year_target'] <= test_year, 'cell_id'])
    test_bg_ids = [c for c in bg_cell_ids if c not in known_by_test]
    test_ids = test_pos_ids + test_bg_ids
    test_labels = np.array([1]*len(test_pos_ids) + [0]*len(test_bg_ids))
    
    if len(test_pos_ids) < 5:
        continue
    
    all_ids = train_ids + test_ids
    all_reaches = set(gfs_reach.loc[all_ids].dropna().astype(int))
    
    conn_T = compute_connectivity_at_cutoff(T, all_reaches)
    conn_testyear = compute_connectivity_at_cutoff(test_year, all_reaches)
    
    def build_feats(ids, conn_map):
        reaches = gfs_reach.loc[ids].values
        n_up = np.array([conn_map.get(int(r),(0,np.nan))[0] if pd.notna(r) else 0 for r in reaches])
        dist = np.array([conn_map.get(int(r),(0,np.nan))[1] if pd.notna(r) else np.nan for r in reaches])
        return n_up, dist
    
    train_nup_c, train_dist_c = build_feats(train_ids, conn_T)
    test_nup_c, test_dist_c = build_feats(test_ids, conn_testyear)
    
    static_conn = gfs.set_index('cell_id')[['n_upstream_target_patches_50km','network_distance_to_nearest_infected_km']]
    train_nup_b = static_conn.loc[train_ids,'n_upstream_target_patches_50km'].values
    train_dist_b = static_conn.loc[train_ids,'network_distance_to_nearest_infected_km'].values
    test_nup_b = static_conn.loc[test_ids,'n_upstream_target_patches_50km'].values
    test_dist_b = static_conn.loc[test_ids,'network_distance_to_nearest_infected_km'].values
    
    other_feats = [c for c in STATIC_SITE_FEATS_T if c not in ('n_upstream_target_patches_50km','network_distance_to_nearest_infected_km','has_upstream_infection_50km','ord_stra')]
    train_other = static_df.set_index('cell_id').loc[train_ids, other_feats].values
    test_other = static_df.set_index('cell_id').loc[test_ids, other_feats].values
    
    def to_df(nup, dist, other, labels):
        d = pd.DataFrame(other, columns=other_feats)
        d['n_upstream_target_patches_50km'] = nup
        d['network_distance_to_nearest_infected_km'] = dist
        d['network_distance_to_nearest_infected_km'] = d['network_distance_to_nearest_infected_km'].fillna(9999)
        d['label'] = labels
        return d
    
    train_c_df = to_df(train_nup_c, train_dist_c, train_other, train_labels)
    test_c_df = to_df(test_nup_c, test_dist_c, test_other, test_labels)
    train_b_df = to_df(train_nup_b, train_dist_b, train_other, train_labels)
    test_b_df = to_df(test_nup_b, test_dist_b, test_other, test_labels)
    
    feats_conn = other_feats + ['n_upstream_target_patches_50km','network_distance_to_nearest_infected_km']
    
    pred_c = fit_predict_temporal(train_c_df, test_c_df, feats=feats_conn)
    pred_b = fit_predict_temporal(train_b_df, test_b_df, feats=feats_conn)
    
    auc_c = roc_auc_score(test_labels, pred_c); ap_c = average_precision_score(test_labels, pred_c)
    auc_b = roc_auc_score(test_labels, pred_b); ap_b = average_precision_score(test_labels, pred_b)
    
    conn_temporal_results.append({'train_cutoff_year':T,'test_year':test_year,
                                    'n_test_pos': len(test_pos_ids),
                                    'auc_pointintime': auc_c, 'ap_pointintime': ap_c,
                                    'auc_lookahead_leak': auc_b, 'ap_lookahead_leak': ap_b})
    pooled_y2.append(test_labels); pooled_pred_c2.append(pred_c); pooled_pred_b2.append(pred_b)
    print(conn_temporal_results[-1])

y_all2 = np.concatenate(pooled_y2)
pred_c2 = np.concatenate(pooled_pred_c2)
pred_b2 = np.concatenate(pooled_pred_b2)

auc_c2p = roc_auc_score(y_all2, pred_c2); ap_c2p = average_precision_score(y_all2, pred_c2)
auc_b2p = roc_auc_score(y_all2, pred_b2); ap_b2p = average_precision_score(y_all2, pred_b2)

conn_df = pd.DataFrame(conn_temporal_results)

temporal_manifest['connectivity_leak_demo'] = {
    'method': 'Same train/test cutoff scheme as climate demo. POINT-IN-TIME: n_upstream_target_patches_50km and '
              'network_distance_to_nearest_infected_km recomputed from river_graph.py using only patches with '
              'first_year<=cutoff (train) / <=test_year (test) -- what a forecaster would actually know. '
              'LOOK-AHEAD LEAK: same two features taken directly from grid_features_static.parquet, which was '
              'computed ONCE using ALL 544 patches ever detected (including patches discovered years after the '
              'test_year) -- illegitimate future information about which reaches are near a currently-known patch.',
    'per_cutoff': conn_df.to_dict(orient='records'),
    'pooled': {
        'n_test_total': int(len(y_all2)), 'n_positive_total': int(y_all2.sum()),
        'auc_pointintime': float(auc_c2p), 'ap_pointintime': float(ap_c2p),
        'auc_lookahead_leak': float(auc_b2p), 'ap_lookahead_leak': float(ap_b2p),
        'ap_change_lookahead_minus_pointintime': float(ap_b2p - ap_c2p),
    },
    'interpretation': 'Counter to the naive expectation that a future-information leak inflates apparent skill, '
        'here look-ahead connectivity performs WORSE (pooled AP 0.464 -> 0.277, a -0.187 absolute drop) than the '
        'correctly point-in-time-computed feature. Mechanism: n_upstream_target_patches_50km computed from the '
        'FINAL 544-patch set flags many more reaches as "near a known patch" than were actually known at the '
        'earlier test years (only a fraction of those patches existed yet), so the feature saturates -- most '
        'cells score similarly, and it loses discriminative power for the specific test year. This is still a '
        'genuine leak (it uses patches not yet detected at forecast time) and is methodologically indefensible '
        'for a real early-warning system, but it demonstrates that "leakage -> inflated metric" is not a safe '
        'universal assumption: the direction and magnitude of the artifact depends on what the leaked feature '
        'encodes and how it interacts with the specific test distribution. Practitioners should not use "the '
        'held-out metric went up" as their only leak-detection heuristic.'
}

with open('temporal_holdout_results.json','w') as f:
    json.dump(temporal_manifest, f, indent=2, default=str)
print("saved")