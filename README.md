# Positive-unlabelled learning for Sosnowsky's hogweed early warning in Ukraine

Code and result manifests accompanying the manuscript

> O. Narushynska, M. Arzubov, V. Antoniv, and S. Baluk, "Positive-unlabelled learning for Sosnowsky's hogweed early warning in Ukraine," submitted to *IAES International Journal of Artificial Intelligence (IJ-AI)*.

Department of Automated Control Systems, Lviv Polytechnic National University, Lviv, Ukraine.
Contact: Olga Narushynska, Olha.O.Narushynska@lpnu.ua

## What is here

```
environment.yml        conda environment (python 3.11, geopandas, rasterio, lightgbm, pygam, ...)
pipeline/
  common/              shared modules (GBIF client, directed river-network graph)
  01_data/             GBIF extraction, target-group background, effort sample, 1 km grid, rivers, WorldClim/terrain
  02_features/         labels, DBSCAN patches, patch dynamics, annual climate (NASA POWER), wind, feature tables
  03_models/           PU-LightGBM (Elkan-Noto), suitability, feature audit/ablation, spread risk, wind dispersal, baseline ladder
  04_validation/       four CV schemes, budget metrics, abstention policy, domain shift, temporal holdout
  05_prioritisation/   field-batch selection, expert-review queue, maps
results/               JSON manifests of every quantitative result reported in the paper
figures/               figures used in the manuscript
```

## Running

```
mamba env create -f environment.yml
conda activate hogweed
```

Each script is a self-contained step that reads its inputs from `data/` (relative to the repository root)
and writes outputs to the working directory. Run the scripts phase by phase in numeric order
(`01_data` -> `05_prioritisation`); the outputs of earlier steps are the inputs of later ones and
should be placed into `data/`. The pipeline is not a single end-to-end command.

Expected files in `data/` (produced by earlier steps or downloaded from the public sources below):
- `HydroRIVERS_v10_eu.shp`
- `admin_ua.gpkg`
- `apiaceae_background_ua.parquet`
- `climate_annual_features.parquet`
- `climate_grid_lookup.parquet`
- `grid_1km_ua.parquet`
- `grid_features_annual.parquet`
- `grid_features_static.parquet`
- `heracleum_ua_clean.parquet`
- `labels_cells.parquet`
- `ne_10m_admin_0_countries.zip`
- `ne_10m_admin_1_states_provinces.zip`
- `patches.parquet`
- `plantae_effort_sample_ua.parquet`
- `river_graph_edges.parquet`
- `rivers_ua.parquet`
- `spread_risk_layers.parquet`
- `suitability_predictions.parquet`
- `target_points_clustered.parquet`
- `terrain_ua.tif`
- `transport_ua.gpkg`
- `wind_vectors.parquet`
- `worldclim_bands.csv`
- `worldclim_ua.tif`

## Data sources

- Occurrence records: Global Biodiversity Information Facility (GBIF), https://www.gbif.org
- Climate: WorldClim 2.1 bioclimatic normals; NASA POWER (MERRA-2) annual series
- Rivers: HydroRIVERS v1.0 (HydroSHEDS)
- Administrative boundaries and transport: Natural Earth

All inputs are open data, but only the GBIF records are downloaded by the code
(`pipeline/common/gbif_client.py`, GBIF occurrence API). The other raw layers have to be downloaded
manually from the sources above and placed where the scripts expect them:

- WorldClim 2.1, 2.5 arc-min: `work/wc/bio/wc2.1_2.5m_bio_{1..19}.tif`, `work/wc/tavg/wc2.1_2.5m_tavg_{01..12}.tif`,
  `work/wc/prec/wc2.1_2.5m_prec_{01..12}.tif` (used by `01_data/05_geolayers_worldclim_terrain.py`)
- Natural Earth 1:10m: `data/ne_10m_admin_0_countries.zip`, `data/ne_10m_admin_1_states_provinces.zip`
  (extracted to `work/ne/` by `01_data/04_geolayers_rivers_graph.py`)
- HydroRIVERS v1.0, Europe: shapefile components `HydroRIVERS_v10_eu.{shp,shx,dbf,prj,sbn}` in `data/hydro/`
  and `data/HydroRIVERS_v10_eu.shp`
- NASA POWER daily point series (JSON, one file per grid point) in `power_raw/`, plus the point list
  `handoff/power_points.csv` (used by `02_features/03_climate_annual_series.py` and `04_wind_vectors.py`).
  The script that downloaded these JSON files from the POWER API is not included in this repository.

All intermediate products (gridded feature tables, suitability and risk layers, ~370 MB) are rebuilt by
the pipeline from these inputs and are therefore not stored here.

## License

The code is released under the MIT License (see `LICENSE`).
