"""
03_geolayers_grid_admin_transport.py

Фаза I: збір первинних геошарів та GBIF-даних.

Inputs (expected in data/):
  - HydroRIVERS_v10_eu_shp.zip
  - HydroRIVERS_v10_eu.shp
  - HydroRIVERS_v10_eu.shx
  - HydroRIVERS_v10_eu.dbf
  - HydroRIVERS_v10_eu.prj
  - HydroRIVERS_v10_eu.sbn
  - ne_10m_admin_0_countries.zip
  - ne_10m_admin_1_states_provinces.zip

Вихід: grid_1km_ua.parquet

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import os, zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely import points as shp_points
from shapely.prepared import prep

EPSG_M = 3035
CELL = 1000

# Extract Natural Earth admin layers
os.makedirs("work/ne", exist_ok=True)
for z_path, z_name in [
    ('data/ne_10m_admin_0_countries.zip', "ne_10m_admin_0_countries"),
    ('data/ne_10m_admin_1_states_provinces.zip', "ne_10m_admin_1_states_provinces"),
]:
    with zipfile.ZipFile(z_path) as zf:
        zf.extractall("work/ne")

c0 = gpd.read_file("work/ne/ne_10m_admin_0_countries.shp")
a1 = gpd.read_file("work/ne/ne_10m_admin_1_states_provinces.shp")

ua_a1 = a1[a1["iso_3166_2"].astype(str).str.startswith("UA-")].copy()
assert len(ua_a1) == 27, f"expected 27 UA admin-1 units, got {len(ua_a1)}"

NAME_FIX = {
    "Kiev": "Kyiv", "Kiev City": "Kyiv City", "Odessa": "Odesa",
    "Transcarpathia": "Zakarpattia", "L'viv": "Lviv", "Ternopil'": "Ternopil",
    "Khmel'nyts'kyy": "Khmelnytskyi", "Dnipropetrovs'k": "Dnipropetrovsk",
    "Donets'k": "Donetsk", "Luhans'k": "Luhansk", "Ivano-Frankivs'k": "Ivano-Frankivsk",
    "Vinnytsya": "Vinnytsia", "Zaporizhzhya": "Zaporizhzhia", "Mykolayiv": "Mykolaiv",
    "Chernivtsi": "Chernivtsi", "Kherson": "Kherson",
}
ua_a1["oblast"] = ua_a1["name"].map(lambda s: NAME_FIX.get(s, s))
ua_a1["oblast_code"] = ua_a1["iso_3166_2"]
obl = ua_a1[["oblast", "oblast_code", "type_en", "name_local", "geometry"]].rename(
    columns={"type_en": "unit_type", "name_local": "name_uk"}).reset_index(drop=True)
obl_m = obl.to_crs(EPSG_M)
obl_m["area_km2"] = obl_m.area / 1e6

border_m = gpd.GeoDataFrame(
    {"country": ["Ukraine"], "iso_a3": ["UKR"],
     "definition": ["de jure / internationally recognised, incl. Crimea AR + Sevastopol"]},
    geometry=[obl_m.geometry.union_all()], crs=EPSG_M)
border_m["area_km2"] = border_m.area / 1e6

# Load HydroRIVERS
buf50_m = border_m.geometry.iloc[0].buffer(50_000)
buf50_ll = gpd.GeoSeries([buf50_m], crs=EPSG_M).to_crs(4326).iloc[0]
border_ll = border_m.to_crs(4326)
ua_poly_ll = border_ll.geometry.iloc[0]

riv = gpd.read_file('data/HydroRIVERS_v10_eu.shp', bbox=buf50_ll.bounds)
riv = riv[riv.intersects(buf50_ll)].copy()

riv_m = riv.to_crs(EPSG_M).reset_index(drop=True)
for c in ["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "ORD_STRA", "ORD_CLAS", "ORD_FLOW", "ENDORHEIC", "HYBAS_L12"]:
    riv_m[c] = riv_m[c].astype("int64")
riv_m["in_ukraine"] = riv_m.intersects(border_m.geometry.iloc[0])
assert riv_m["HYRIV_ID"].is_unique, "HYRIV_ID not unique"

G = nx.DiGraph()
G.add_nodes_from(riv_m["HYRIV_ID"].tolist())
edges = riv_m.loc[riv_m["NEXT_DOWN"] != 0, ["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM"]]
G.add_edges_from((int(a), int(b), {"km": float(k)}) for a, b, k in edges.itertuples(index=False))

is_dag = nx.is_directed_acyclic_graph(G)

# Build 1 km grid
ua_geom = border_m.geometry.iloc[0]
xmin, ymin, xmax, ymax = ua_geom.bounds
ORIGIN_X, ORIGIN_Y = 0, 0
col0, col1 = int(np.floor((xmin - ORIGIN_X) / CELL)), int(np.ceil((xmax - ORIGIN_X) / CELL))
row0, row1 = int(np.floor((ymin - ORIGIN_Y) / CELL)), int(np.ceil((ymax - ORIGIN_Y) / CELL))
cols = np.arange(col0, col1)
rows = np.arange(row0, row1)

CC, RR = np.meshgrid(cols, rows)
CC, RR = CC.ravel(), RR.ravel()
cx = ORIGIN_X + (CC + 0.5) * CELL
cy = ORIGIN_Y + (RR + 0.5) * CELL

pts = shp_points(np.c_[cx, cy])
pg = prep(ua_geom)
inside = np.fromiter((pg.contains(p) for p in pts), dtype=bool, count=len(pts))

grid = gpd.GeoDataFrame({
    "row": RR[inside].astype("int32"), "col": CC[inside].astype("int32"),
    "x": cx[inside], "y": cy[inside]},
    geometry=list(np.asarray(pts)[inside]), crs=EPSG_M)
grid["cell_id"] = (grid["row"].astype("int64") * 100_000 + grid["col"].astype("int64"))
assert grid["cell_id"].is_unique, "cell_id collision"

ll = grid.to_crs(4326).geometry
grid["lon"] = ll.x.values
grid["lat"] = ll.y.values

# Oblast attribution
j = gpd.sjoin(grid[["geometry"]], obl_m[["oblast", "oblast_code", "geometry"]],
              how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]
grid["oblast"] = j["oblast"].values
grid["oblast_code"] = j["oblast_code"].values
miss = grid["oblast"].isna()
if miss.any():
    nn = gpd.sjoin_nearest(grid.loc[miss, ["geometry"]], obl_m[["oblast", "oblast_code", "geometry"]], how="left")
    nn = nn[~nn.index.duplicated(keep="first")]
    grid.loc[miss, "oblast"] = nn["oblast"].values
    grid.loc[miss, "oblast_code"] = nn["oblast_code"].values
assert grid["oblast"].notna().all(), "oblast assignment incomplete"

# Nearest river reach
rivers_out = riv_m[["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "LENGTH_KM", "DIST_DN_KM", "DIST_UP_KM",
                     "CATCH_SKM", "UPLAND_SKM", "ENDORHEIC", "DIS_AV_CMS", "ORD_STRA", "ORD_CLAS",
                     "ORD_FLOW", "HYBAS_L12", "in_ukraine", "geometry"]].copy()

nr = gpd.sjoin_nearest(grid[["geometry"]], rivers_out[["HYRIV_ID", "geometry"]],
                       how="left", distance_col="dist_river_m")
nr = nr[~nr.index.duplicated(keep="first")]
grid["nearest_reach"] = nr["HYRIV_ID"].astype("int64").values
grid["dist_river_m"] = nr["dist_river_m"].astype("float32").values

# Basin root
root_cache = {}

def root_of(n):
    path = []
    cur = n
    while cur not in root_cache:
        succ = list(G.successors(cur))
        if not succ:
            root_cache[cur] = cur
            break
        path.append(cur)
        cur = succ[0]
    r = root_cache[cur]
    for p in path:
        root_cache[p] = r
    return r

for n in riv_m["HYRIV_ID"].astype(int).tolist():
    root_of(n)
grid["basin_id"] = grid["nearest_reach"].map(root_cache).astype("int64")

# Save
grid_out = grid[["cell_id", "row", "col", "x", "y", "lon", "lat", "oblast", "oblast_code",
                 "nearest_reach", "dist_river_m", "basin_id", "geometry"]].copy()
grid_out["oblast"] = grid_out["oblast"].astype("category")
grid_out["oblast_code"] = grid_out["oblast_code"].astype("category")
grid_out.to_parquet("grid_1km_ua.parquet", index=False)
N_CELLS = len(grid_out)
assert 550_000 < N_CELLS < 650_000, f"grid size {N_CELLS} outside expected ~600k"
print("grid_1km_ua.parquet:", N_CELLS, "cells")