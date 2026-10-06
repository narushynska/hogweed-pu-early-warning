"""
04_patches_map_relief.py

Фаза V: пріоритизація польового батчу, карти.

Inputs (expected in data/):
  - patches.parquet
  - target_points_clustered.parquet
  - admin_ua.gpkg
  - apiaceae_background_ua.parquet

Вихід: patches_map.png

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb
from shapely.geometry import MultiPoint
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

patches = pd.read_parquet('data/patches.parquet')
pts = pd.read_parquet('data/target_points_clustered.parquet')

patches["geometry"] = patches.geometry_wkb.apply(lambda b: wkb.loads(b) if b is not None else None)
gpatches = gpd.GeoDataFrame(patches, geometry="geometry", crs="EPSG:3035")

border = gpd.read_file('data/admin_ua.gpkg', layer="border")
oblasts = gpd.read_file('data/admin_ua.gpkg', layer="oblasts")

gpts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x3035, pts.y3035), crs="EPSG:3035")

defined = gpatches[gpatches.boundary_defined]
undefined_cl = gpatches[(~gpatches.boundary_defined) & (gpatches.patch_kind=="dbscan_cluster")]
singles = gpatches[gpatches.patch_kind=="noise_singleton"]

undefined_cl_pts = undefined_cl.copy()
undefined_cl_pts["geometry"] = gpd.points_from_xy(undefined_cl.centroid_x, undefined_cl.centroid_y)

focus = gpatches[gpatches.patch_id=="P0011"].iloc[0]
focus_pts = gpts[gpts.patch_id=="P0011"]

cx, cy = focus.centroid_x, focus.centroid_y
buf = 900

fig, axes = plt.subplots(1, 3, figsize=(16.5, 6))

ax = axes[0]
border.boundary.plot(ax=ax, color="black", linewidth=0.8)
oblasts.boundary.plot(ax=ax, color="#cccccc", linewidth=0.4)
singles.plot(ax=ax, marker='.', color="#b0b0b0", markersize=6, alpha=0.5)
gpd.GeoDataFrame(undefined_cl_pts, geometry="geometry", crs="EPSG:3035").plot(
    ax=ax, marker='o', color="#f39c12", markersize=14, alpha=0.7)
defined.plot(ax=ax, facecolor="#c0392b", edgecolor="#7b241c", linewidth=0.5, alpha=0.85)
ax.scatter([cx],[cy], s=180, facecolor='none', edgecolor='#1a5276', linewidth=1.8, zorder=5)
ax.set_title("a. Occurrence patches across Ukraine")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
legend_elems = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#c0392b", markeredgecolor="#7b241c", markersize=8, label=f"defined-boundary patch (n={defined.shape[0]})"),
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#f39c12", markersize=8, label=f"cluster, boundary undefined (n={undefined_cl.shape[0]})"),
    Line2D([0],[0], marker='.', color='none', markerfacecolor="#b0b0b0", markersize=10, label=f"noise singleton (n={singles.shape[0]})"),
    Line2D([0],[0], marker='o', color='none', markerfacecolor='none', markeredgecolor='#1a5276', markersize=10, label="zoomed patch (panel c)"),
]
ax.legend(handles=legend_elems, loc="lower left", frameon=False, fontsize=6.5)

ax2 = axes[1]
d = defined.sort_values("area_km2")
ax2.scatter(d.area_km2, d.area_uncertainty_frac, s=np.clip(d.n_points*3, 10, 200),
            c="#c0392b", alpha=0.6, edgecolor="#7b241c", linewidth=0.4)
ax2.scatter([focus.area_km2],[focus.area_uncertainty_frac], s=focus.n_points*3, facecolor='none',
            edgecolor='#1a5276', linewidth=1.8, zorder=5)
ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.set_xlabel("patch area (km², concave hull)")
ax2.set_ylabel("boundary uncertainty\n(concave-vs-convex area fraction)")
ax2.set_title(f"b. Boundary uncertainty vs. area (n={len(d)})")
ax2.margins(0.08)

ax3 = axes[2]
mp = MultiPoint(list(zip(focus_pts.x3035, focus_pts.y3035)))
convex = mp.convex_hull
gpd.GeoSeries([convex], crs="EPSG:3035").boundary.plot(ax=ax3, color="#7f8c8d", linestyle="--", linewidth=1.2)
gpd.GeoSeries([focus.geometry], crs="EPSG:3035").plot(ax=ax3, facecolor="#c0392b", edgecolor="#7b241c", alpha=0.35, linewidth=1.5)
ax3.scatter(focus_pts.x3035, focus_pts.y3035, s=14, color="#7b241c", alpha=0.8, zorder=5)
ax3.set_xlim(cx-buf, cx+buf); ax3.set_ylim(cy-buf, cy+buf)
ax3.set_aspect('equal')
ax3.set_xticks([]); ax3.set_yticks([])
for spine in ax3.spines.values(): spine.set_visible(False)
sbx0, sby0 = cx-buf+80, cy-buf+80
ax3.plot([sbx0, sbx0+200], [sby0, sby0], color='black', linewidth=2)
ax3.text(sbx0+100, sby0+40, "200 m", ha='center', fontsize=7)
ax3.set_title(f"c. Patch P0011 (Kyiv), n={focus.n_points} points,\narea={focus.area_km2:.2f} km², boundary unc.={focus.area_uncertainty_frac:.1%}", fontsize=8.5)
legend_elems3 = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#7b241c", markersize=6, label="GBIF occurrence point"),
    Line2D([0],[0], color="#7b241c", linewidth=3, alpha=0.35, label="concave hull (patch boundary)"),
    Line2D([0],[0], color="#7f8c8d", linestyle="--", linewidth=1.2, label="convex hull (for comparison)"),
]
ax3.legend(handles=legend_elems3, loc="upper left", frameon=False, fontsize=6.5)

fig.suptitle("DBSCAN patches (eps=500 m, min_samples=2): 176 clusters + 368 unclustered singletons from 1,855 records", fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig("patches_map.png", dpi=160, bbox_inches="tight")