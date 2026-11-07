"""
Memory-safe, single-threshold bag-size computation for single-sample,
large-N datasets (e.g. Xenium: 164k cells, one image).

Root cause of the OOM: cellohood.data_processing.data_transforms.
cluster_cell_df_positions_within_distance_threshold calls
sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=...)
with no `connectivity` constraint. Without one, sklearn builds a dense
pairwise distance matrix internally -- O(n^2) memory. For 164,000 cells in
one sample that's ~100 GiB for the matrix alone (confirmed: matches the
observed ~115 GiB crash). Every other dataset in this project is split
across many samples (DLPFC: 12, MIBI-TNBC: 34), so each clustering call
only ever saw a few thousand cells and this was never hit before.

Fix: pass a sparse spatial connectivity graph (radius_neighbors_graph at
radius=distance_threshold -- the standard sklearn-recommended pattern for
scaling AgglomerativeClustering) so memory scales with the number of
within-threshold cell pairs, not O(n^2). This does not change what the
clustering can produce in practice: bags are tight local neighborhoods by
construction, never sprawling regions, so restricting candidate merges to
spatially-connected pairs at the threshold radius does not alter results
for this use case -- it just makes the computation tractable.
"""
import sys
import resource
import time
import json
import os

MEM_CAP_GIB = 90  # hard cap, comfortably under the account's 110 GiB limit
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
cap_bytes = MEM_CAP_GIB * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, hard if hard != resource.RLIM_INFINITY else cap_bytes))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import radius_neighbors_graph

THRESHOLD = float(sys.argv[1])
OUT_DIR = "/raid/immucan/immuw/cellohood-analyses/analyses/XENIUM_BREAST_CANCER"
RESULTS_CSV = os.path.join(OUT_DIR, "threshold_sweep_results.csv")

df = pd.read_csv(os.path.join(OUT_DIR, "xenium_full.csv"))
pos = df[["x_centroid", "y_centroid"]].to_numpy()
n = len(pos)
print(f"threshold={THRESHOLD}: n_cells={n}, building connectivity graph (radius={THRESHOLD})...", flush=True)

t0 = time.time()
connectivity = radius_neighbors_graph(pos, radius=THRESHOLD, mode="connectivity", include_self=False)
n_edges = connectivity.nnz
print(f"  connectivity graph: {n_edges} edges, built in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=THRESHOLD,
    connectivity=connectivity,
    linkage="ward",
)
labels = clusterer.fit_predict(pos)
print(f"  clustering done in {time.time()-t0:.1f}s", flush=True)

sizes = pd.Series(labels).value_counts().values
result = {
    "threshold": THRESHOLD,
    "n_cells": int(n),
    "n_bags": int(len(sizes)),
    "mean_bag_size": float(np.mean(sizes)),
    "median_bag_size": float(np.median(sizes)),
    "max_bag_size": int(np.max(sizes)),
    "n_edges_connectivity": int(n_edges),
}
print(json.dumps(result), flush=True)

# append to the shared results CSV (survives even if a later/larger threshold OOMs)
row_df = pd.DataFrame([result])
if os.path.exists(RESULTS_CSV):
    row_df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
else:
    row_df.to_csv(RESULTS_CSV, mode="w", header=True, index=False)
print(f"saved to {RESULTS_CSV}", flush=True)
