"""
01_gbif_extraction.py

Фаза I: збір первинних геошарів та GBIF-даних.

Inputs (expected in data/):
  - gbif_client.py

Вихід: heracleum_ua_clean.parquet

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

import sys
import json
import importlib
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, '.')

# gbif_client.py входить до цього архіву як common/gbif_client.py (тонка обгортка
# над requests для api.gbif.org/v1/, НЕ пакет pygbif). Додай корінь common/ до sys.path
# перед запуском цього файлу, або скопіюй common/gbif_client.py у робочу теку.
sys.path.insert(0, '../common')

import gbif_client as G
importlib.reload(G)

TAXA = {
    "Heracleum sosnowskyi":    dict(taxonKey=3642949, role="target"),
    "Heracleum mantegazzianum":dict(taxonKey=3034825, role="invasive_other"),
    "Heracleum sibiricum":     dict(taxonKey=6027427, role="hard_negative"),
    "Heracleum sphondylium":   dict(taxonKey=3034830, role="hard_negative"),
    "Pastinaca sativa":        dict(taxonKey=8262702, role="hard_negative"),
    "Angelica sylvestris":     dict(taxonKey=5371818, role="hard_negative"),
}

# Fetch species data
frames = []
sp_meta = {}
for name, spec in TAXA.items():
    rows, meta = G.page_all(G.SPECIES_FIELDS, f"species::{name}", taxonKey=spec["taxonKey"])
    d = pd.DataFrame(rows)
    d["query_species"] = name
    d["query_taxonKey"] = spec["taxonKey"]
    d["role"] = spec["role"]
    frames.append(d)
    sp_meta[name] = meta

raw = pd.concat(frames, ignore_index=True)

# Resolve cross-query duplicates
raw["query_species_all"] = raw["key"].map(raw.groupby("key")["query_species"].apply(lambda s: "|".join(sorted(set(s)))))
prio = {"Heracleum sibiricum": 0, "Heracleum sphondylium": 1, "Heracleum sosnowskyi": 0,
        "Heracleum mantegazzianum": 0, "Pastinaca sativa": 0, "Angelica sylvestris": 0}
raw["_prio"] = raw["query_species"].map(prio)
raw_u = raw.sort_values("_prio").drop_duplicates("key", keep="first").drop(columns="_prio").reset_index(drop=True)

# Filtering
BBOX = dict(lon=(22.0, 40.3), lat=(44.2, 52.5))
df = raw_u.copy()

# F0: ABSENT records
df["occurrenceStatus"] = df["occurrenceStatus"].astype("string")
absent = df[df["occurrenceStatus"].str.upper() == "ABSENT"].copy()
df = df[df["occurrenceStatus"].str.upper() != "ABSENT"].copy()

# F1: outside Ukraine bbox
df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
inbox = (df.decimalLongitude.between(*BBOX["lon"]) & df.decimalLatitude.between(*BBOX["lat"]))
df = df[inbox].copy()

# F2: basisOfRecord fossil/living
df["basisOfRecord"] = df["basisOfRecord"].astype("string")
df = df[~df.basisOfRecord.isin(["FOSSIL_SPECIMEN","LIVING_SPECIMEN"])].copy()
df["is_preserved_specimen"] = df.basisOfRecord.eq("PRESERVED_SPECIMEN")

# F3: coordinate uncertainty > 1000 m
df["coordinateUncertaintyInMeters"] = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
df["uncertainty_missing"] = df["coordinateUncertaintyInMeters"].isna()
keep = df["uncertainty_missing"] | (df["coordinateUncertaintyInMeters"] <= 1000)
df = df[keep].copy()

# F4: duplicates
df["_lat5"] = df.decimalLatitude.round(5)
df["_lon5"] = df.decimalLongitude.round(5)
df["_ed"] = df["eventDate"].astype("string").fillna("NA")
df = df.drop_duplicates(subset=["speciesKey","_lat5","_lon5","_ed"], keep="first").copy()
df = df.drop(columns=["_lat5","_lon5","_ed"])

df["label_role"] = df["role"]
df["accepted_species_key"] = df["speciesKey"]

# Add derived columns
def has_media(s):
    return s.notna() & s.astype("string").str.len().gt(2)

for c in ["datasetName","institutionCode","identificationVerificationStatus","references",
          "mediaTypes","issues","recordedBy","identifiedBy","license","datasetKey"]:
    df[c] = df[c].astype("string")

df["has_photo_or_ref"] = has_media(df["mediaTypes"]) | df["references"].notna()
df["has_verification_status"] = df["identificationVerificationStatus"].notna()

ds_names = df["datasetName"].fillna("")
CS_PAT = r"iNaturalist|Observation\.org|Pl@ntNet|PlantNet|eBird|Iris|casual|research-grade"
df["is_citizen_science"] = ds_names.str.contains(CS_PAT, case=False, regex=True, na=False)

# Resolve missing dataset names
miss = df.loc[df["datasetName"].isna(), "datasetKey"].value_counts()
name_map = {}
for dk in miss.index.dropna():
    try:
        j = G._get(f"dataset/{dk}", {})
        name_map[dk] = j.get("title")
    except Exception as e:
        name_map[dk] = None

df["datasetName_resolved"] = df["datasetName"].fillna(df["datasetKey"].map(name_map)).fillna("<unresolved>")
df["is_citizen_science"] = df["datasetName_resolved"].str.contains(CS_PAT, case=False, regex=True, na=False)

# Write output
clean = df.copy()
clean.to_parquet("heracleum_ua_clean.parquet", index=False)