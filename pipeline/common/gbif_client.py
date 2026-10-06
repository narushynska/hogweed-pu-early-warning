"""Minimal GBIF occurrence/search paging client for the hogweed early-warning project.

All requests go to https://api.gbif.org/v1/ . No synthetic fallback: on HTTP error we raise.
"""
import json
import time
import requests

API = "https://api.gbif.org/v1"
UA_BBOX = dict(lon=(22.0, 40.3), lat=(44.2, 52.5))

# Fields kept for species-level records (Step 1 requirement)
SPECIES_FIELDS = [
    "key", "scientificName", "acceptedScientificName", "taxonKey", "acceptedTaxonKey",
    "speciesKey", "species", "genusKey", "familyKey", "kingdomKey", "taxonRank",
    "taxonomicStatus", "decimalLatitude", "decimalLongitude",
    "coordinateUncertaintyInMeters", "coordinatePrecision", "eventDate", "year", "month", "day",
    "basisOfRecord", "occurrenceStatus", "datasetKey", "datasetName", "institutionCode",
    "collectionCode", "publishingOrgKey", "publishingCountry",
    "identificationVerificationStatus", "issues", "recordedBy", "identifiedBy",
    "stateProvince", "locality", "references", "license", "protocol",
    "individualCount", "establishmentMeans", "occurrenceID", "catalogNumber", "mediaTypes",
]

# Lean field set for the large effort sample (Step 2b)
EFFORT_FIELDS = [
    "key", "decimalLatitude", "decimalLongitude", "year", "month",
    "basisOfRecord", "datasetKey", "speciesKey", "familyKey",
]

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})

MANIFEST = []  # appended request records; caller serialises


def _get(path, params, tries=5):
    """GET with bounded retry on transient 5xx / connection resets."""
    last = None
    for attempt in range(tries):
        try:
            r = _session.get(f"{API}/{path}", params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"GBIF {path} HTTP {r.status_code}: {r.text[:300]}")
        except requests.exceptions.RequestException as e:
            last = repr(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GBIF {path} failed after {tries} tries: {last}")


def count(**filters):
    """Record count only (limit=0)."""
    p = dict(filters)
    p.update(limit=0, offset=0)
    js = _get("occurrence/search", p)
    return js["count"]


def flatten(rec, fields):
    out = {}
    for f in fields:
        v = rec.get(f)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False) if v else None
        out[f] = v
    return out


def page_all(fields, label, page_size=300, max_offset=100000, **filters):
    """Page through occurrence/search collecting `fields`. Returns (rows, meta)."""
    base = dict(filters)
    base.setdefault("country", "UA")
    base.setdefault("hasCoordinate", "true")
    base.setdefault("hasGeospatialIssue", "false")

    total = count(**base)
    rows, offset, truncated = [], 0, False
    while offset < total:
        if offset > max_offset:
            truncated = True
            break
        p = dict(base)
        p.update(limit=page_size, offset=offset)
        js = _get("occurrence/search", p)
        results = js.get("results", [])
        if not results:
            break
        rows.extend(flatten(r, fields) for r in results)
        offset += len(results)
        if js.get("endOfRecords"):
            break
    meta = dict(label=label, params=base, api_count=total, retrieved=len(rows),
                truncated_at_offset_cap=truncated, page_size=page_size,
                downloaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    MANIFEST.append(meta)
    return rows, meta
