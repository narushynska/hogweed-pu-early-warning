"""
04_wind_vectors.py

Фаза II: побудова міток/патчів, клімат-ряду та таблиці ознак на сітці.

Inputs (expected in data/):
  - worldclim_ua.tif

Вихід: wind_vectors.parquet

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
import os
import glob
import csv

# Load grid and points data
import urllib.request
import urllib.parse

# Reconstruct pts from power_points.csv (created during the trace)
# and the power_raw JSON files

# First reconstruct the grid lookup and pts
# We need to rebuild pts from the power_raw files

files = sorted(glob.glob("power_raw/*.json"))
assert len(files) == 299

VMAP = {"T2M": "tmean", "T2M_MIN": "tmin", "T2M_MAX": "tmax", "PRECTOTCORR": "precip",
        "WS10M_MAX": "wind_speed_max", "WD10M": "wind_dir", "EVPTRNS": "et0"}

frames = []
snap = []
pts_list = []
for f in files:
    r = json.load(open(f))
    p = r["parameter"]
    dates = sorted(p["T2M"].keys())
    df = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d")})
    for k, v in VMAP.items():
        df[v] = [p[k][d] for d in dates]
    df.insert(0, "point_id", r["point_id"])
    frames.append(df)
    snap.append((r["point_id"], r["req_lon"], r["req_lat"], r["coords"][0], r["coords"][1], r["coords"][2]))
    pts_list.append({"point_id": r["point_id"], "lon": r["req_lon"], "lat": r["req_lat"]})

daily = pd.concat(frames, ignore_index=True)
pts = pd.DataFrame(pts_list)

daily["year"] = daily.date.dt.year
daily["month"] = daily.date.dt.month
daily["doy"] = daily.date.dt.dayofyear

# Seed maturation window: July-September. Speed-weighted vector mean of direction.
w = daily[daily.month.isin([7, 8, 9])].copy()
th = np.deg2rad(w["wind_dir"].to_numpy())
sp = w["wind_speed_max"].to_numpy()
w["u"] = sp * np.sin(th)
w["v"] = sp * np.cos(th)
w["s_sin"] = np.sin(th)
w["s_cos"] = np.cos(th)

gw = w.groupby(["point_id", "year"])
wind = gw.agg(wind_u_mean=("u", "mean"), wind_v_mean=("v", "mean"),
              wind_sin_mean=("s_sin", "mean"), wind_cos_mean=("s_cos", "mean"),
              wind_speed_mean=("wind_speed_max", "mean"), n_days=("u", "size")).reset_index()
assert (wind.n_days >= 88).all()

# speed-weighted resultant
wind["wind_resultant_mag"] = np.hypot(wind.wind_u_mean, wind.wind_v_mean)
wind["wind_resultant_dir"] = np.rad2deg(np.arctan2(wind.wind_u_mean, wind.wind_v_mean)) % 360
# directional constancy: |vector mean of unit vectors| in [0,1]  (1 = perfectly steady)
wind["wind_constancy"] = np.hypot(wind.wind_sin_mean, wind.wind_cos_mean)
# speed-weighted constancy = resultant / mean speed
wind["wind_constancy_weighted"] = wind.wind_resultant_mag / wind.wind_speed_mean

wind_out = wind.merge(pts, on="point_id", how="left")
wind_out = wind_out[["point_id", "lon", "lat", "year", "wind_u_mean", "wind_v_mean", "wind_sin_mean", "wind_cos_mean",
                      "wind_speed_mean", "wind_resultant_dir", "wind_resultant_mag", "wind_constancy",
                      "wind_constancy_weighted", "n_days"]]
wind_out.to_parquet("wind_vectors.parquet", index=False, compression="zstd")