"""
river_graph.py

Спільний утилітарний модуль: побудова напрямленого графа річкової мережі (HydroRIVERS) та функції load_graph/upstream_ids/downstream_ids/network_distance_km, використовувані у 01_data/04_geolayers_rivers_graph.py, 02_features/05_feature_table_static_annual.py, 03_models/04_spread_risk_layers.py.

Inputs: none (utility module imported by other steps).

Вихід: river_graph.py

Середовище виконання: environment="hogweed" (geopandas 1.1.4, rasterio 1.5.1, rioxarray, shapely 2.1.2,
pyproj, scikit-learn 1.9, hdbscan, pygam, networkx, lightgbm, shap, libpysal, esda, pandas 3.0.5,
numpy 2.5.2, pyarrow, matplotlib, seaborn).

Reproducibility note: this file is the exact code that produced the output listed above.
Each pipeline step is self-contained: it reads its inputs from data/ (paths relative to the
repository root) and writes its outputs to the working directory. See README.md for the
execution order.
"""

content = '''"""Directed river-network graph helpers for the hogweed early-warning pipeline.

Built from HydroRIVERS v1.0 (Europe cut, clipped to Ukraine + 50 km buffer).
Topology convention (HydroRIVERS): each reach has HYRIV_ID and NEXT_DOWN,
where NEXT_DOWN is the HYRIV_ID of the reach immediately downstream, or 0
when the reach is a network outlet (river mouth, endorheic sink, or the
point where the network leaves the clipped extent).

The graph is a networkx.DiGraph with an edge HYRIV_ID -> NEXT_DOWN, i.e.
edges point DOWNSTREAM. Therefore:
  * descendants(G, n) == everything downstream of n
  * ancestors(G, n)   == everything upstream of n  (the catchment reaches)

Portable storage: no pickle. The graph lives in two parquet files
(`river_graph_edges.parquet` = edge list, `rivers_ua.parquet` = reach
attributes + geometry) and is rebuilt with `load_graph()`.

Usage
-----
    from river_graph import load_graph, upstream_ids, downstream_ids, network_distance_km
    G = load_graph("river_graph_edges.parquet", "rivers_ua.parquet")
    up = upstream_ids(G, 20345678)                 # reaches draining INTO it
    dn = downstream_ids(G, 20345678, max_km=50)    # reaches it drains into
    d  = network_distance_km(G, a, b)              # along-network km, direction-aware
"""

from __future__ import annotations

import networkx as nx

__all__ = [
    "load_graph",
    "upstream_ids",
    "downstream_ids",
    "network_distance_km",
    "outlet_of",
    "basin_root",
    "strahler_at_least",
]

# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def load_graph(edges_parquet="river_graph_edges.parquet",
               reaches_parquet=None):
    """Rebuild the DiGraph from the portable parquet edge list.

    Parameters
    ----------
    edges_parquet : path to edge list with columns
        [HYRIV_ID, NEXT_DOWN, LENGTH_KM, ORD_STRA, UPLAND_SKM, DIST_DN_KM]
        (LENGTH_KM of the *source* reach becomes the edge weight in km).
    reaches_parquet : optional path to the reach table; when given, every
        reach becomes a node even if it has no downstream edge, and node
        attributes (ORD_STRA, UPLAND_SKM, LENGTH_KM, DIST_DN_KM) are attached.

    Returns
    -------
    networkx.DiGraph with edges pointing downstream, edge attr \'km\'.
    """
    import pandas as pd

    G = nx.DiGraph()

    if reaches_parquet is not None:
        cols = ["HYRIV_ID", "NEXT_DOWN", "ORD_STRA",
                "UPLAND_SKM", "LENGTH_KM", "DIST_DN_KM"]
        r = pd.read_parquet(reaches_parquet, columns=cols)
        for rec in r.itertuples(index=False):
            G.add_node(int(rec.HYRIV_ID),
                       ORD_STRA=int(rec.ORD_STRA),
                       UPLAND_SKM=float(rec.UPLAND_SKM),
                       LENGTH_KM=float(rec.LENGTH_KM),
                       DIST_DN_KM=float(rec.DIST_DN_KM))

    e = pd.read_parquet(edges_parquet)
    for rec in e.itertuples(index=False):
        G.add_edge(int(rec.HYRIV_ID), int(rec.NEXT_DOWN),
                   km=float(rec.LENGTH_KM))
    return G


# --------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------

def upstream_ids(G, node, max_km=None):
    """Reach IDs upstream of `node` (its contributing network).

    max_km : if given, only reaches within that along-network distance
             (sum of reach lengths on the path) are returned.
    """
    node = int(node)
    if node not in G:
        return set()
    if max_km is None:
        return set(nx.ancestors(G, node))
    # distances on the reversed graph = upstream distances
    dist = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), node, cutoff=max_km, weight="km")
    return {n for n in dist if n != node}


def downstream_ids(G, node, max_km=None):
    """Reach IDs downstream of `node` (where its water — and propagules — go)."""
    node = int(node)
    if node not in G:
        return set()
    if max_km is None:
        return set(nx.descendants(G, node))
    dist = nx.single_source_dijkstra_path_length(
        G, node, cutoff=max_km, weight="km")
    return {n for n in dist if n != node}


def network_distance_km(G, a, b, undirected=False):
    """Along-network distance in km between two reaches.

    By default respects flow direction: returns the km from `a` down to `b`
    if `b` is downstream of `a`, else from `b` down to `a`, else None when
    the two are on different branches (no directed path either way).

    undirected=True treats the network as undirected, giving the distance
    through the common confluence — appropriate for "how far apart along
    water" rather than "can propagules drift from a to b".
    """
    a, b = int(a), int(b)
    if a not in G or b not in G:
        return None
    if a == b:
        return 0.0
    if undirected:
        try:
            return nx.dijkstra_path_length(G.to_undirected(as_view=True),
                                           a, b, weight="km")
        except nx.NetworkXNoPath:
            return None
    for src, dst in ((a, b), (b, a)):
        try:
            return nx.dijkstra_path_length(G, src, dst, weight="km")
        except nx.NetworkXNoPath:
            continue
    return None


# --------------------------------------------------------------------------
# basin identification (used for spatial CV blocks)
# --------------------------------------------------------------------------

def outlet_of(G, node):
    """Terminal node of the downstream path from `node`.

    Walks NEXT_DOWN until a node with no outgoing edge is reached. That
    terminal node is the basin identifier used for spatial-block CV.
    Returns `node` itself when it is already terminal.
    """
    node = int(node)
    if node not in G:
        return None
    seen = set()
    cur = node
    while True:
        succ = list(G.successors(cur))
        if not succ:
            return cur
        nxt = succ[0]
        if nxt in seen:          # defensive: should not happen in a DAG
            return cur
        seen.add(cur)
        cur = nxt


basin_root = outlet_of  # alias: the basin id of a reach


def strahler_at_least(G, nodes, order):
    """Filter an iterable of reach IDs to those with ORD_STRA >= order.

    Requires the graph to carry node attributes (load_graph with
    reaches_parquet). Useful for restricting hydrochoric-dispersal
    features to reaches large enough to carry seeds any distance.
    """
    out = set()
    for n in nodes:
        d = G.nodes.get(int(n), {})
        if d.get("ORD_STRA", -1) >= order:
            out.add(int(n))
    return out
'''

with open('river_graph.py', 'w') as f:
    f.write(content)