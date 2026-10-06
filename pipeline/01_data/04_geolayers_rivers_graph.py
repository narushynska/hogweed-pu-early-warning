"""
04_geolayers_rivers_graph.py

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

Вихід: rivers_ua.parquet

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
import numpy as np

EPSG_M = 3035

# Extract Natural Earth admin layers
os.makedirs("work/ne", exist_ok=True)
for z in ["ne_10m_admin_0_countries", "ne_10m_admin_1_states_provinces"]:
    with zipfile.ZipFile(f"data/{z}.zip") as zf:
        zf.extractall("work/ne")

# Build Ukraine de jure border from admin-1 units with UA-* ISO codes
a1 = gpd.read_file("work/ne/ne_10m_admin_1_states_provinces.shp")
ua_a1 = a1[a1["iso_3166_2"].astype(str).str.startswith("UA-")].copy()
assert len(ua_a1) == 27, f"expected 27 UA admin-1 units, got {len(ua_a1)}"
obl_m = ua_a1.to_crs(EPSG_M)
border_m = gpd.GeoDataFrame(
    {"country": ["Ukraine"], "iso_a3": ["UKR"]},
    geometry=[obl_m.geometry.union_all()], crs=EPSG_M)
border_ll = border_m.to_crs(4326)
ua_poly_ll = border_ll.geometry.iloc[0]

# Extract HydroRIVERS shapefile components
os.makedirs("work/hydro", exist_ok=True)
# Copy individual shapefile components into place
import shutil
for artifact_id, filename in [
    ("371c33dd-0313-46a0-9d53-4446fe83e045", "HydroRIVERS_v10_eu.shp"),
    ("65fef8b3-e56c-44d1-a40a-faf43951f54d", "HydroRIVERS_v10_eu.shx"),
    ("d8c85f35-478e-4187-905f-72220bbdd347", "HydroRIVERS_v10_eu.dbf"),
    ("9aa40d6d-4943-4ec0-ab5c-b168da9bbe14", "HydroRIVERS_v10_eu.prj"),
    ("9778f3a9-7090-4996-a06b-f2f9819b55da", "HydroRIVERS_v10_eu.sbn"),
]:
    os.makedirs("work/hydro/HydroRIVERS_v10_eu_shp", exist_ok=True)
    shutil.copy(f"data/hydro/{filename}", f"work/hydro/HydroRIVERS_v10_eu_shp/{filename}")

shp = "work/hydro/HydroRIVERS_v10_eu_shp/HydroRIVERS_v10_eu.shp"

# 50 km buffer in metric CRS to keep transboundary headwaters
buf50_m = border_m.geometry.iloc[0].buffer(50_000)
buf50_ll = gpd.GeoSeries([buf50_m], crs=EPSG_M).to_crs(4326).iloc[0]

riv = gpd.read_file(shp, bbox=buf50_ll.bounds)
riv = riv[riv.intersects(buf50_ll)].copy()

riv_m = riv.to_crs(EPSG_M).reset_index(drop=True)
for c in ["HYRIV_ID","NEXT_DOWN","MAIN_RIV","ORD_STRA","ORD_CLAS","ORD_FLOW","ENDORHEIC","HYBAS_L12"]:
    riv_m[c] = riv_m[c].astype("int64")
riv_m["in_ukraine"] = riv_m.intersects(border_m.geometry.iloc[0])
assert riv_m["HYRIV_ID"].is_unique, "HYRIV_ID not unique"

riv_cols = ["HYRIV_ID","NEXT_DOWN","MAIN_RIV","LENGTH_KM","DIST_DN_KM","DIST_UP_KM",
            "CATCH_SKM","UPLAND_SKM","ENDORHEIC","DIS_AV_CMS","ORD_STRA","ORD_CLAS",
            "ORD_FLOW","HYBAS_L12","in_ukraine","geometry"]
rivers_out = riv_m[riv_cols].copy()
rivers_out.to_parquet("rivers_ua.parquet", index=False)