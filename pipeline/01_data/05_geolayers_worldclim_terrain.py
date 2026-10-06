"""
05_geolayers_worldclim_terrain.py

Фаза I: збір первинних геошарів та GBIF-даних.

Inputs: WorldClim 2.1 2.5-arc-minute GeoTIFFs downloaded manually from worldclim.org into work/wc/{bio,tavg,prec}/ (see README).

Вихід: worldclim_ua.tif

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

os.makedirs("out", exist_ok=True)
os.makedirs("work/wc/bio", exist_ok=True)
os.makedirs("work/wc/tavg", exist_ok=True)
os.makedirs("work/wc/prec", exist_ok=True)
os.makedirs("work/wc/tmin", exist_ok=True)
os.makedirs("work/wc/wind", exist_ok=True)

BB = (22.0, 44.2, 40.3, 52.5)

def band_paths():
    out = []
    for i in range(1, 20):
        out.append((f"bio_{i:02d}", f"work/wc/bio/wc2.1_2.5m_bio_{i}.tif"))
    for grp in ["tavg", "prec", "tmin", "wind"]:
        for m in range(1, 13):
            out.append((f"{grp}_{m:02d}", f"work/wc/{grp}/wc2.1_2.5m_{grp}_{m:02d}.tif"))
    return out

BP = band_paths()

with rasterio.open(BP[0][1]) as src:
    win = from_bounds(*BB, transform=src.transform).round_offsets().round_lengths()
    prof = src.profile.copy()
    tr = src.window_transform(win)
    H, W = int(win.height), int(win.width)

stack = np.empty((len(BP), H, W), dtype="float32")
for k, (nm, p) in enumerate(BP):
    with rasterio.open(p) as src:
        stack[k] = src.read(1, window=win)
NODATA = -3.4e38
stack[stack <= NODATA / 1.0001] = np.nan

BIO_DESC = {1: "Annual Mean Temperature", 2: "Mean Diurnal Range", 3: "Isothermality",
            4: "Temperature Seasonality", 5: "Max Temp of Warmest Month", 6: "Min Temp of Coldest Month",
            7: "Temp Annual Range", 8: "Mean Temp of Wettest Quarter", 9: "Mean Temp of Driest Quarter",
            10: "Mean Temp of Warmest Quarter", 11: "Mean Temp of Coldest Quarter", 12: "Annual Precipitation",
            13: "Precip of Wettest Month", 14: "Precip of Driest Month", 15: "Precip Seasonality (CV)",
            16: "Precip of Wettest Quarter", 17: "Precip of Driest Quarter", 18: "Precip of Warmest Quarter",
            19: "Precip of Coldest Quarter"}
UNITS = {"bio": None, "tavg": "degC", "prec": "mm", "tmin": "degC", "wind": "m s-1"}
BIO_UNITS = {1: "degC", 2: "degC", 3: "%", 4: "degC*100 (sd)", 5: "degC", 6: "degC", 7: "degC", 8: "degC",
             9: "degC", 10: "degC", 11: "degC", 12: "mm", 13: "mm", 14: "mm", 15: "%", 16: "mm", 17: "mm", 18: "mm", 19: "mm"}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

rows = []
for k, (nm, p) in enumerate(BP, start=1):
    if nm.startswith("bio"):
        i = int(nm.split("_")[1])
        desc, unit = BIO_DESC[i], BIO_UNITS[i]
    else:
        grp, m = nm.split("_")
        mi = int(m)
        lbl = {"tavg": "Mean monthly temperature", "prec": "Monthly precipitation",
               "tmin": "Monthly minimum temperature", "wind": "Monthly mean wind speed"}[grp]
        desc, unit = f"{lbl} — {MON[mi-1]}", UNITS[grp]
    rows.append({"band": k, "name": nm, "description": desc, "units": unit,
                 "source_file": os.path.basename(p)})
bands_df = pd.DataFrame(rows)

OUT_NODATA = -9999.0
arr = np.where(np.isfinite(stack), stack, OUT_NODATA).astype("float32")
prof_out = dict(driver="GTiff", height=H, width=W, count=len(BP), dtype="float32",
                crs="EPSG:4326", transform=tr, nodata=OUT_NODATA,
                compress="DEFLATE", predictor=2, tiled=True, blockxsize=256, blockysize=256)
with rasterio.open("out/worldclim_ua.tif", "w", **prof_out) as dst:
    dst.write(arr)
    for k, (nm, _) in enumerate(BP, start=1):
        dst.set_band_description(k, f"{nm}: {rows[k-1]['description']} [{rows[k-1]['units']}]")
    dst.update_tags(source="WorldClim 2.1 base, 2.5 arcmin", clip_bbox=str(BB), units_note="temp degC, precip mm, wind m/s")