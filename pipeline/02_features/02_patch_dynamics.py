"""
02_patch_dynamics.py

Фаза II: побудова міток/патчів, клімат-ряду та таблиці ознак на сітці.

Inputs (expected in data/):
  - heracleum_ua_clean.parquet
  - apiaceae_background_ua.parquet
  - plantae_effort_sample_ua.parquet
  - grid_1km_ua.parquet

Вихід: patch_dynamics.parquet

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
import geopandas as gpd
import json
import os
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point
from shapely import concave_hull, convex_hull
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr, linregress

her = pd.read_parquet('data/heracleum_ua_clean.parquet')
bg = pd.read_parquet('data/apiaceae_background_ua.parquet')
ef = pd.read_parquet('data/plantae_effort_sample_ua.parquet')
grid = pd.read_parquet('data/grid_1km_ua.parquet')

T = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

def add_cell(df, latc="decimalLatitude", lonc="decimalLongitude"):
    d = df.dropna(subset=[latc, lonc]).copy()
    x, y = T.transform(d[lonc].to_numpy(), d[latc].to_numpy())
    d["x3035"] = x; d["y3035"] = y
    d["row_"] = np.floor(y / 1000).astype("int64"); d["col_"] = np.floor(x / 1000).astype("int64")
    d["cell_id"] = d["row_"] * 100000 + d["col_"]
    return d

grid_ids = pd.Index(grid["cell_id"].to_numpy())
her_c = add_cell(her); bg_c = add_cell(bg); ef_c = add_cell(ef)
her_c = her_c[her_c.cell_id.isin(grid_ids)]
bg_c = bg_c[bg_c.cell_id.isin(grid_ids)]
ef_c = ef_c[ef_c.cell_id.isin(grid_ids)]

bg_c["is_cs"] = (bg_c["basisOfRecord"] == "HUMAN_OBSERVATION")
her_c["is_cs"] = her_c["is_citizen_science"].fillna(False).astype(bool)

tg = her_c[her_c.label_role == "target"]

EPS_F, MS_F = 500, 2
RATIO = 0.4
PT_BUF = 50.0

P = tg[["x3035", "y3035"]].to_numpy()
lab_f = DBSCAN(eps=EPS_F, min_samples=MS_F).fit_predict(P)
tgf = tg.copy(); tgf["cluster"] = lab_f

def build_patch(pts, pid, kind):
    mp = MultiPoint([Point(*p) for p in pts]); n = len(pts)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    cvx = convex_hull(mp)
    if n >= 4:
        cav = concave_hull(mp, ratio=RATIO)
        geom = cav.buffer(PT_BUF); ref = cvx.buffer(PT_BUF)
        bnd = True
    else:
        geom = mp.buffer(max(PT_BUF, 100.0)) if n > 1 else Point(cx, cy).buffer(PT_BUF)
        ref = cvx.buffer(max(PT_BUF, 100.0)) if n > 1 else geom
        bnd = False
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    area = geom.area / 1e6; per = geom.length / 1000
    return dict(patch_id=pid, patch_kind=kind, n_points=n, area_km2=area, perimeter_km=per,
        centroid_x=float(geom.centroid.x), centroid_y=float(geom.centroid.y),
        compactness=float(4 * np.pi * geom.area / (geom.length ** 2)) if geom.length > 0 else np.nan,
        pt_density_per_km2=float(n / area) if area > 0 else np.nan,
        convex_area_km2=float(ref.area / 1e6),
        area_uncertainty_km2=float(ref.area - geom.area) / 1e6,
        area_uncertainty_frac=float((ref.area - geom.area) / ref.area) if ref.area > 0 else np.nan,
        enclosing_radius_m=float(d.max()), mean_radius_m=float(d.mean()),
        boundary_defined=bnd, geometry=geom)

yrs = tgf.dropna(subset=["year"]).copy(); yrs["year"] = yrs["year"].astype(int)
YEARS = [y for y in range(2018, 2027)]

yearly = {}; ycells = {}
for y in YEARS:
    d = yrs[yrs.year == y].reset_index(drop=True)
    Py = d[["x3035", "y3035"]].to_numpy()
    lb = DBSCAN(eps=EPS_F, min_samples=MS_F).fit_predict(Py)
    recs = []; cellsets = {}
    for c in range(lb.max() + 1):
        pid = f"{y}_P{c:03d}"; m = lb == c
        r = build_patch(Py[m], pid, "dbscan_cluster"); r["year"] = y; r["cs_share"] = float(d.is_cs[m].mean()); recs.append(r)
        cellsets[pid] = set(d.cell_id[m])
    for j, i in enumerate(np.where(lb == -1)[0]):
        pid = f"{y}_S{j:03d}"
        r = build_patch(Py[[i]], pid, "noise_singleton"); r["year"] = y; r["cs_share"] = float(d.is_cs.iat[i]); recs.append(r)
        cellsets[pid] = {d.cell_id.iat[i]}
    gy = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:3035")
    gy["n_cells"] = gy.patch_id.map(lambda p: len(cellsets[p]))
    yearly[y] = gy; ycells[y] = cellsets

MATCH_DIST = 1000.0

def match_years(g1, g2):
    idx = g2.sindex
    pairs = []
    for i, geom in enumerate(g1.geometry.values):
        cand = list(idx.query(geom.buffer(MATCH_DIST)))
        for j in cand:
            g_ = g2.geometry.iat[j]
            inter = geom.intersection(g_).area; uni = geom.union(g_).area
            iou = inter / uni if uni > 0 else 0.0
            dc = np.hypot(geom.centroid.x - g_.centroid.x, geom.centroid.y - g_.centroid.y)
            if iou > 0 or dc <= MATCH_DIST:
                pairs.append((i, j, iou, dc))
    pairs.sort(key=lambda t: (-t[2], t[3]))
    u1 = set(); u2 = set(); out = []
    for i, j, iou, dc in pairs:
        if i in u1 or j in u2: continue
        u1.add(i); u2.add(j); out.append((i, j, iou, dc))
    return out

dyn = []
yl = sorted(yearly)
for y1, y2 in zip(yl[:-1], yl[1:]):
    g1, g2 = yearly[y1], yearly[y2]
    m = match_years(g1, g2)
    matched1 = {i for i, _, _, _ in m}; matched2 = {j for _, j, _, _ in m}
    for i, j, iou, dc in m:
        r1 = g1.iloc[i]; r2 = g2.iloc[j]
        c1 = g1.geometry.iat[i].centroid; c2 = g2.geometry.iat[j].centroid
        dx, dy = c2.x - c1.x, c2.y - c1.y
        az = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        cells1 = ycells[y1][r1.patch_id]; cells2 = ycells[y2][r2.patch_id]
        dyn.append(dict(year_from=y1, year_to=y2, patch_from=r1.patch_id, patch_to=r2.patch_id,
            status="matched", iou=iou, centroid_dist_m=float(dc),
            area_from=r1.area_km2, area_to=r2.area_km2,
            area_change_km2=r2.area_km2 - r1.area_km2,
            growth_rate=float((r2.area_km2 - r1.area_km2) / r1.area_km2),
            log_area_ratio=float(np.log(r2.area_km2 / r1.area_km2)),
            centroid_shift_m=float(np.hypot(dx, dy)), centroid_azimuth_deg=float(az),
            n_points_from=int(r1.n_points), n_points_to=int(r2.n_points),
            d_log_points=float(np.log(r2.n_points / r1.n_points)),
            n_cells_from=int(r1.n_cells), n_cells_to=int(r2.n_cells),
            new_occupied_cells=len(cells2 - cells1), lost_cells=len(cells1 - cells2),
            survived=True, boundary_defined_from=bool(r1.boundary_defined),
            boundary_defined_to=bool(r2.boundary_defined),
            cx_from=c1.x, cy_from=c1.y, cx_to=c2.x, cy_to=c2.y))
    for i in range(len(g1)):
        if i in matched1: continue
        r1 = g1.iloc[i]; c1 = g1.geometry.iat[i].centroid
        dyn.append(dict(year_from=y1, year_to=y2, patch_from=r1.patch_id, patch_to=None, status="disappeared",
            iou=0.0, centroid_dist_m=np.nan, area_from=r1.area_km2, area_to=np.nan,
            area_change_km2=np.nan, growth_rate=np.nan, log_area_ratio=np.nan,
            centroid_shift_m=np.nan, centroid_azimuth_deg=np.nan, n_points_from=int(r1.n_points),
            n_points_to=0, d_log_points=np.nan, n_cells_from=int(r1.n_cells), n_cells_to=0,
            new_occupied_cells=0, lost_cells=int(r1.n_cells), survived=False,
            boundary_defined_from=bool(r1.boundary_defined), boundary_defined_to=False,
            cx_from=c1.x, cy_from=c1.y, cx_to=np.nan, cy_to=np.nan))
    for j in range(len(g2)):
        if j in matched2: continue
        r2 = g2.iloc[j]; c2 = g2.geometry.iat[j].centroid
        dyn.append(dict(year_from=y1, year_to=y2, patch_from=None, patch_to=r2.patch_id, status="new",
            iou=0.0, centroid_dist_m=np.nan, area_from=np.nan, area_to=r2.area_km2,
            area_change_km2=np.nan, growth_rate=np.nan, log_area_ratio=np.nan,
            centroid_shift_m=np.nan, centroid_azimuth_deg=np.nan, n_points_from=0,
            n_points_to=int(r2.n_points), d_log_points=np.nan, n_cells_from=0, n_cells_to=int(r2.n_cells),
            new_occupied_cells=int(r2.n_cells), lost_cells=0, survived=False,
            boundary_defined_from=False, boundary_defined_to=bool(r2.boundary_defined),
            cx_from=np.nan, cy_from=np.nan, cx_to=c2.x, cy_to=c2.y))

dynamics = pd.DataFrame(dyn)

EFF_R = 5000.0
bg_y = bg_c.dropna(subset=["year"]).copy(); bg_y["year"] = bg_y["year"].astype(int)
ef_y = ef_c.dropna(subset=["year"]).copy(); ef_y["year"] = ef_y["year"].astype(int)
trees_bg = {y: cKDTree(d[["x3035", "y3035"]].to_numpy()) for y, d in bg_y.groupby("year") if y in YEARS}
trees_ef = {y: cKDTree(d[["x3035", "y3035"]].to_numpy()) for y, d in ef_y.groupby("year") if y in YEARS}

def effort_at(y, xs, ys):
    pts = np.c_[xs, ys]
    nb = np.array([len(trees_bg[y].query_ball_point(p, EFF_R)) for p in pts]) if y in trees_bg else np.zeros(len(pts), int)
    ne = np.array([len(trees_ef[y].query_ball_point(p, EFF_R)) for p in pts]) if y in trees_ef else np.zeros(len(pts), int)
    return nb, ne

py = []
for y, g in yearly.items():
    cx = g.geometry.centroid.x.to_numpy(); cy = g.geometry.centroid.y.to_numpy()
    nb, ne = effort_at(y, cx, cy)
    t = pd.DataFrame(dict(year=y, patch_id=g.patch_id.to_numpy(), n_points=g.n_points.to_numpy(),
                          area_km2=g.area_km2.to_numpy(), n_cells=g.n_cells.to_numpy(),
                          effort_apiaceae_5km=nb, effort_plants_5km=ne, cs_share=g.cs_share.to_numpy()))
    py.append(t)
patch_year = pd.concat(py, ignore_index=True)

m_matched = dynamics.status == "matched"
dm = dynamics[m_matched].copy()
eff_from = patch_year.set_index(["year", "patch_id"])
dm["eff_bg_from"] = [eff_from.loc[(r.year_from, r.patch_from), "effort_apiaceae_5km"] for r in dm.itertuples()]
dm["eff_bg_to"] = [eff_from.loc[(r.year_to, r.patch_to), "effort_apiaceae_5km"] for r in dm.itertuples()]
dm["eff_pl_from"] = [eff_from.loc[(r.year_from, r.patch_from), "effort_plants_5km"] for r in dm.itertuples()]
dm["eff_pl_to"] = [eff_from.loc[(r.year_to, r.patch_to), "effort_plants_5km"] for r in dm.itertuples()]
dm["d_log_effort_bg"] = np.log((dm.eff_bg_to + 1) / (dm.eff_bg_from + 1))
dm["d_log_effort_pl"] = np.log((dm.eff_pl_to + 1) / (dm.eff_pl_from + 1))

dyn_out = dynamics.merge(dm[["year_from", "patch_from", "patch_to", "eff_bg_from", "eff_bg_to", "eff_pl_from",
        "eff_pl_to", "d_log_effort_bg", "d_log_effort_pl"]], on=["year_from", "patch_from", "patch_to"], how="left")
dyn_out["eps_m"] = EPS_F; dyn_out["min_samples"] = MS_F; dyn_out["match_max_dist_m"] = MATCH_DIST
dyn_out.to_parquet("patch_dynamics.parquet", index=False)