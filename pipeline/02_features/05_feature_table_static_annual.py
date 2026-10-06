"""
05_feature_table_static_annual.py

Фаза II: побудова міток/патчів, клімат-ряду та таблиці ознак на сітці.

Inputs (expected in data/):
  - grid_1km_ua.parquet
  - worldclim_ua.tif
  - worldclim_bands.csv
  - terrain_ua.tif
  - rivers_ua.parquet
  - river_graph_edges.parquet
  - river_graph.py
  - admin_ua.gpkg
  - transport_ua.gpkg
  - heracleum_ua_clean.parquet
  - apiaceae_background_ua.parquet
  - plantae_effort_sample_ua.parquet
  - labels_cells.parquet
  - patches.parquet
  - climate_annual_features.parquet
  - climate_grid_lookup.parquet
  - wind_vectors.parquet
  - labels_patches_manifest.json
  - patch_dynamics.parquet
  - target_points_clustered.parquet
  - geolayers_manifest.json
  - climate_manifest.json
  - patch_year_effort.parquet

Вихід: grid_features_static.parquet

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import sys
import os
import gc
import importlib
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
import networkx as nx
from scipy.spatial import cKDTree
from pyproj import Transformer
from collections import Counter

# Load grid
grid = pd.read_parquet('data/grid_1km_ua.parquet')

# Load worldclim bands
wb = pd.read_csv('data/worldclim_bands.csv')

# Load climate annual features
climate_annual = pd.read_parquet('data/climate_annual_features.parquet')

# Load climate grid lookup
cl = pd.read_parquet('data/climate_grid_lookup.parquet')

# Load labels cells
labels = pd.read_parquet('data/labels_cells.parquet')

# Load patches
patches = pd.read_parquet('data/patches.parquet')

# Load target points
tp = pd.read_parquet('data/target_points_clustered.parquet')

# Load heracleum
her = pd.read_parquet('data/heracleum_ua_clean.parquet')

# Load apiaceae background
apiaceae = pd.read_parquet('data/apiaceae_background_ua.parquet')

# Load plant effort
effort = pd.read_parquet('data/plantae_effort_sample_ua.parquet')

# Load rivers
rivers = pd.read_parquet('data/rivers_ua.parquet')

# Load transport layers
roads = gpd.read_file('data/transport_ua.gpkg', layer='roads')
railroads = gpd.read_file('data/transport_ua.gpkg', layer='railroads')
places = gpd.read_file('data/transport_ua.gpkg', layer='populated_places')

# Load river graph module
with open('/tmp/river_graph.py', 'w') as f:
    f.write(open('data/river_graph.py').read())
sys.path.insert(0, '/tmp')
import river_graph
importlib.reload(river_graph)
from river_graph import load_graph, downstream_ids

# Build climate static features (2015-2024 window)
STATIC_YEARS = (2015, 2024)
window = climate_annual[(climate_annual.year >= STATIC_YEARS[0]) & (climate_annual.year <= STATIC_YEARS[1])].copy()

raw_clim_cols = ['gdd_base5', 'frost_days', 'annual_precip', 'tmean_annual', 'et0_annual',
                 'gdd_base5_mar_jun', 'frost_days_spring', 'spring_precip', 'summer_precip',
                 'rain_30d_preflower', 'tmean_spring']

agg = window.groupby('point_id')[raw_clim_cols].agg(['mean', 'std'])
agg.columns = [f"{c}_{stat}_2015_2024" for c, stat in agg.columns]
agg = agg.reset_index()

wind_agg = window.groupby('point_id').agg(
    wind_sin_mean_2015_2024=('wind_sin_mean', 'mean'),
    wind_cos_mean_2015_2024=('wind_cos_mean', 'mean'),
    wind_speed_mean_2015_2024=('wind_speed_mean', 'mean'),
    wind_u_mean_2015_2024=('wind_u_mean', 'mean'),
    wind_v_mean_2015_2024=('wind_v_mean', 'mean'),
).reset_index()
wind_agg['wind_resultant_dir_2015_2024'] = np.degrees(np.arctan2(wind_agg['wind_u_mean_2015_2024'], wind_agg['wind_v_mean_2015_2024'])) % 360
wind_agg['wind_resultant_mag_2015_2024'] = np.hypot(wind_agg['wind_u_mean_2015_2024'], wind_agg['wind_v_mean_2015_2024'])
wind_agg['wind_constancy_2015_2024'] = np.hypot(wind_agg['wind_sin_mean_2015_2024'], wind_agg['wind_cos_mean_2015_2024'])

offsets = climate_annual.groupby('point_id')[['delta_tmean_wc_minus_power', 'ratio_precip_wc_over_power', 'offset_imputed']].first().reset_index()

climate_static_point = agg.merge(wind_agg, on='point_id').merge(offsets, on='point_id')

# Join climate to grid
grid_clim = grid[['cell_id']].merge(cl[['cell_id', 'point_id', 'assign_dist_m']], on='cell_id', how='left')
grid_clim = grid_clim.merge(climate_static_point, on='point_id', how='left')

# Sample worldclim and terrain rasters
coords = list(zip(grid['lon'].values, grid['lat'].values))

with rasterio.open('data/worldclim_ua.tif') as src:
    wc_vals = np.array(list(src.sample(coords)))
    wc_nodata = src.nodata

with rasterio.open('data/terrain_ua.tif') as src:
    terrain_vals = np.array(list(src.sample(coords)))
    terr_nodata = src.nodata
    terr_desc = src.descriptions

wc_vals_f = np.where(wc_vals == wc_nodata, np.nan, wc_vals)
terrain_vals_f = np.where(terrain_vals == terr_nodata, np.nan, terrain_vals)

wc_names = wb['name'].tolist()
wc_df = pd.DataFrame(wc_vals_f, columns=wc_names)
wc_df.insert(0, 'cell_id', grid['cell_id'].values)

terrain_cols = ['elevation_m', 'slope_deg', 'roughness_m', 'tpi_m']
terrain_df = pd.DataFrame(terrain_vals_f, columns=terrain_cols)
terrain_df.insert(0, 'cell_id', grid['cell_id'].values)

# Build river connectivity features
G = load_graph('data/river_graph_edges.parquet', 'data/rivers_ua.parquet')

tp_reach = tp.merge(grid[['cell_id', 'nearest_reach']], on='cell_id', how='left')
patch_reaches = tp_reach.groupby('patch_id')['nearest_reach'].apply(lambda s: sorted(set(s.dropna().astype(int)))).reset_index()
patch_reaches.columns = ['patch_id', 'reach_ids']

infected_reaches_per_patch = {row.patch_id: row.reach_ids for row in patch_reaches.itertuples()}
all_infected_reaches = sorted(set(r for ids in infected_reaches_per_patch.values() for r in ids))

patch_downstream_cache = {}
count_upstream_patches = Counter()

for patch_id, reach_ids in infected_reaches_per_patch.items():
    combined_downstream = set()
    for r in reach_ids:
        if r not in G:
            continue
        if r not in patch_downstream_cache:
            patch_downstream_cache[r] = downstream_ids(G, r, max_km=50)
        combined_downstream |= patch_downstream_cache[r]
        combined_downstream.add(r)
    for r in combined_downstream:
        count_upstream_patches[r] += 1

sources = [r for r in all_infected_reaches if r in G]
dist_from_infected = nx.multi_source_dijkstra_path_length(G, sources, weight='km')

ord_stra_map = rivers.set_index('HYRIV_ID')['ORD_STRA'].to_dict()
unique_reaches = sorted(grid['nearest_reach'].unique())
river_conn_df = pd.DataFrame({'nearest_reach': unique_reaches})
river_conn_df['n_upstream_target_patches_50km'] = river_conn_df['nearest_reach'].map(count_upstream_patches).fillna(0).astype(int)
river_conn_df['network_distance_to_nearest_infected_km'] = river_conn_df['nearest_reach'].map(dist_from_infected)
river_conn_df['ord_stra'] = river_conn_df['nearest_reach'].map(ord_stra_map)

grid_river = grid[['cell_id', 'nearest_reach', 'dist_river_m']].merge(river_conn_df, on='nearest_reach', how='left')
grid_river['has_upstream_infection_50km'] = grid_river['n_upstream_target_patches_50km'] > 0

# Transport distances
grid_pts = gpd.GeoDataFrame(
    grid[['cell_id']],
    geometry=gpd.points_from_xy(grid['x'], grid['y']),
    crs="EPSG:3035"
)

def nearest_dist(points_gdf, target_gdf, colname):
    joined = gpd.sjoin_nearest(points_gdf, target_gdf[['geometry']], distance_col=colname, how='left')
    out = joined.groupby(joined.index)[colname].min()
    out = out.reindex(points_gdf.index)
    return out.values

d_road = nearest_dist(grid_pts, roads, 'dist_road_m')
d_rail = nearest_dist(grid_pts, railroads, 'dist_railroad_m')
d_place = nearest_dist(grid_pts, places, 'dist_settlement_m')

# Effort buffer counts
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

def proj_xy(df, lonc='decimalLongitude', latc='decimalLatitude'):
    x, y = transformer.transform(df[lonc].values, df[latc].values)
    return x, y

apiaceae_x, apiaceae_y = proj_xy(apiaceae)
effort_x, effort_y = proj_xy(effort)
hardneg = her[her.label_role == 'hard_negative']
hardneg_x, hardneg_y = proj_xy(hardneg)

grid_xy = np.column_stack([grid['x'].values, grid['y'].values])
radii_km = [1, 5, 10, 25]
radii_m = [r * 1000 for r in radii_km]

def buffer_counts_per_point(source_xy, grid_xy, radii_m):
    src_tree = cKDTree(source_xy)
    grid_tree = cKDTree(grid_xy)
    results = {}
    for r in radii_m:
        counts = np.zeros(len(grid_xy), dtype=np.int32)
        neighbors = grid_tree.query_ball_tree(src_tree, r=r)
        counts = np.array([len(n) for n in neighbors], dtype=np.int32)
        results[r] = counts
    return results

apiaceae_xy = np.column_stack([apiaceae_x, apiaceae_y])
res_apiaceae = buffer_counts_per_point(apiaceae_xy, grid_xy, radii_m)

hardneg_xy = np.column_stack([hardneg_x, hardneg_y])
res_hardneg = buffer_counts_per_point(hardneg_xy, grid_xy, radii_m)
gc.collect()

effort_xy = np.column_stack([effort_x, effort_y])
src_tree = cKDTree(effort_xy)

def buffer_counts_batched(src_tree, grid_xy, radii_m, batch_size=50000):
    n = len(grid_xy)
    results = {r: np.zeros(n, dtype=np.int32) for r in radii_m}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_tree = cKDTree(grid_xy[start:end])
        for r in radii_m:
            neighbors = batch_tree.query_ball_tree(src_tree, r=r)
            results[r][start:end] = [len(x) for x in neighbors]
    return results

res_effort = buffer_counts_batched(src_tree, grid_xy, radii_m, batch_size=50000)
gc.collect()

effort_df = pd.DataFrame({'cell_id': grid['cell_id'].values})
for r_km, r_m in zip(radii_km, radii_m):
    effort_df[f'n_apiaceae_bg_r{r_km}km'] = res_apiaceae[r_m]
    effort_df[f'n_plant_effort_r{r_km}km'] = res_effort[r_m]
    effort_df[f'n_hard_negative_r{r_km}km'] = res_hardneg[r_m]

del res_apiaceae, res_hardneg, res_effort, apiaceae_xy, hardneg_xy, effort_xy, src_tree
gc.collect()

# Labels join
labels_join_cols = ['cell_id', 'label_state', 'ctrl_target_group_bg', 'ctrl_random_bg',
                    'has_hard_negative', 'n_target', 'n_hard_negative', 'n_apiaceae_bg', 'n_plant_effort',
                    'first_year', 'last_year', 'n_years_observed', 'is_citizen_science_share']
labels_sub = labels[labels_join_cols].copy()

grid_labels = grid[['cell_id']].merge(labels_sub, on='cell_id', how='left')
grid_labels['label_state'] = grid_labels['label_state'].fillna('unknown')
for c in ['ctrl_target_group_bg', 'ctrl_random_bg', 'has_hard_negative']:
    grid_labels[c] = grid_labels[c].fillna(False)
for c in ['n_target', 'n_hard_negative', 'n_apiaceae_bg', 'n_plant_effort', 'n_years_observed']:
    grid_labels[c] = grid_labels[c].fillna(0).astype(int)

# Assemble static feature table
static = grid[['cell_id', 'row', 'col', 'x', 'y', 'lon', 'lat', 'oblast', 'oblast_code', 'basin_id']].copy()

static = static.merge(grid_clim.drop(columns=['assign_dist_m']).rename(columns={'point_id': 'climate_point_id'}), on='cell_id', how='left')
static['climate_assign_dist_m'] = grid_clim['assign_dist_m'].values

static = static.merge(wc_df, on='cell_id', how='left')
static = static.merge(terrain_df, on='cell_id', how='left')

static = static.merge(grid_river.drop(columns=['dist_river_m']), on='cell_id', how='left')
static['dist_river_m'] = grid['dist_river_m'].values

static['dist_road_m'] = d_road
static['dist_railroad_m'] = d_rail
static['dist_settlement_m'] = d_place

static = static.merge(effort_df, on='cell_id', how='left')
static = static.merge(grid_labels, on='cell_id', how='left')

static = static.rename(columns={
    'wind_sin_mean_2015_2024': 'wind_sin_mean',
    'wind_cos_mean_2015_2024': 'wind_cos_mean',
    'wind_resultant_dir_2015_2024': 'wind_resultant_dir',
    'wind_resultant_mag_2015_2024': 'wind_resultant_mag',
    'wind_constancy_2015_2024': 'wind_constancy',
})

# Fill nodata for worldclim/terrain
nodata_mask = static['bio_01'].isna()

with rasterio.open('data/worldclim_ua.tif') as src:
    band1 = src.read(1)
    transform = src.transform
    valid_mask = band1 != wc_nodata
    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    valid_xy = np.column_stack([xs, ys])

nodata_idx = np.where(nodata_mask.values)[0]
nodata_coords = np.column_stack([grid['lon'].values[nodata_idx], grid['lat'].values[nodata_idx]])

tree_valid = cKDTree(valid_xy)
dist, nn_idx = tree_valid.query(nodata_coords, k=1)
nn_rows, nn_cols = rows[nn_idx], cols[nn_idx]

with rasterio.open('data/worldclim_ua.tif') as src:
    full_wc = src.read()
with rasterio.open('data/terrain_ua.tif') as src:
    full_terr = src.read()

fill_wc = full_wc[:, nn_rows, nn_cols].T
fill_terr = full_terr[:, nn_rows, nn_cols].T

for i, name in enumerate(wc_names):
    static.loc[nodata_idx, name] = fill_wc[:, i]
for i, name in enumerate(terrain_cols):
    static.loc[nodata_idx, name] = fill_terr[:, i]

static['worldclim_terrain_nodata_filled'] = False
static.loc[nodata_idx, 'worldclim_terrain_nodata_filled'] = True

static = static.copy()

# Downcast types
for c in ['ctrl_target_group_bg', 'ctrl_random_bg', 'has_hard_negative']:
    static[c] = static[c].astype(bool)

int_cols_downcast = ['basin_id', 'nearest_reach', 'n_upstream_target_patches_50km', 'ord_stra',
                     'n_target', 'n_hard_negative', 'n_apiaceae_bg', 'n_plant_effort', 'n_years_observed']
for c in int_cols_downcast:
    static[c] = static[c].astype('int32')

float_cols = static.select_dtypes(include='float64').columns
static[float_cols] = static[float_cols].astype('float32')

static.to_parquet('grid_features_static.parquet', index=False)