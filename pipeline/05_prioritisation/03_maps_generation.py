"""
03_maps_generation.py

Фаза V: пріоритизація польового батчу, карти.

Inputs (expected in data/):
  - grid_1km_ua.parquet
  - admin_ua.gpkg
  - suitability_predictions.parquet
  - spread_risk_layers_final.parquet
  - abstention_results.json
  - patch_dynamics.parquet
  - patches.parquet
  - apiaceae_background_ua.parquet
  - heracleum_ua_clean.parquet
  - budget_metrics_results.json

Вихід: map_suitability.png

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

# figure-style helpers
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.stats import rankdata

grid = pd.read_parquet('data/grid_1km_ua.parquet')

border = gpd.read_file('data/admin_ua.gpkg', layer="border")
oblasts = gpd.read_file('data/admin_ua.gpkg', layer="oblasts")

suit = pd.read_parquet('data/suitability_predictions.parquet')

row_min, row_max = grid['row'].min(), grid['row'].max()
col_min, col_max = grid['col'].min(), grid['col'].max()
n_rows = row_max - row_min + 1
n_cols = col_max - col_min + 1

cell_size = 1000
x_min = grid['x'].min() - cell_size/2
x_max = grid['x'].max() + cell_size/2
y_min = grid['y'].min() - cell_size/2
y_max = grid['y'].max() + cell_size/2

EXTENT = [x_min, x_max, y_min, y_max]

grid_idx = grid.set_index('cell_id')[['row','col']]

def raster_from_series(cell_ids, values):
    df = pd.DataFrame({'cell_id': cell_ids, 'value': values}).set_index('cell_id')
    joined = grid_idx.join(df, how='left')
    arr = np.full((n_rows, n_cols), np.nan, dtype=float)
    r_idx = (joined['row'] - row_min).values.astype(int)
    c_idx = (joined['col'] - col_min).values.astype(int)
    valid = ~pd.isna(joined['value'].values)
    arr[r_idx[valid], c_idx[valid]] = joined['value'].values[valid]
    return arr

apply_figure_style()

def add_basemap(ax, oblasts_gdf=oblasts, border_gdf=border, oblast_fc='#f2f2f0', oblast_ec='#bbbbbb'):
    oblasts_gdf.boundary.plot(ax=ax, color=oblast_ec, linewidth=0.4, zorder=1)
    border_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=5)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def add_scalebar(ax, length_km=100, loc='lower right', pad=0.5):
    fontprops = fm.FontProperties(size=7)
    bar = AnchoredSizeBar(ax.transData, length_km*1000, f'{length_km} km', loc,
                           pad=pad, color='black', frameon=False,
                           size_vertical=(y_max-y_min)*0.003, fontproperties=fontprops)
    ax.add_artist(bar)

def add_north_arrow(ax, x=0.06, y=0.92):
    ax.annotate('N', xy=(x, y), xytext=(x, y-0.06), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2),
                ha='center', va='center', fontsize=8, fontweight='bold')

suit_vals = suit['suitability_targetbg'].values
pct_rank = rankdata(suit_vals, method='average') / len(suit_vals) * 100

suit_pct_arr = raster_from_series(suit['cell_id'].values, pct_rank)

plt.close('all')
fig, ax = plt.subplots(figsize=(9,7))

im = ax.imshow(suit_pct_arr, extent=EXTENT, origin='lower', cmap='viridis',
               vmin=0, vmax=100, interpolation='nearest', zorder=2)

add_basemap(ax)
ax.set_title('Придатність середовища для Heracleum sosnowskyi/mantegazzianum (перцентильний ранг)', loc='left', fontsize=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.7)
cbar.set_label('перцентиль suitability_targetbg (0–100)', fontsize=7)
cbar.ax.tick_params(labelsize=6)

add_scalebar(ax, 100)
add_north_arrow(ax)
fig.text(0.01, 0.01,
          'Джерело: suitability_predictions.parquet (LightGBM, PU/Elkan-Noto, target-group background). '
          'Валідована якість: AP=0.58–0.61 (spatial-block / leave-region-out holdout), НЕ random-split (AP=0.67, завищено). '
          'Шкала — перцентильний ранг клітинки серед усіх 599126 (right-skewed розподіл score: медіана 0.0006, макс. 1.0).',
          fontsize=6, color='#555555', wrap=True)

fig.savefig('map_suitability.png', dpi=300, bbox_inches='tight')