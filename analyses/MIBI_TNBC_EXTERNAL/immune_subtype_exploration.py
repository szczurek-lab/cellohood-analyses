import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, mannwhitneyu

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")

emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)

n_comp, k = 10, 10
decomposed = decomposed_full[:, :n_comp]
gmm = GaussianMixture(n_components=k, random_state=42)
clusters = gmm.fit_predict(decomposed)

cell_type_arr = df["cell_type"].values
sample_arr = df["sample"].values
all_cell_types = sorted(set(cell_type_arr))

# Full composition breakdown per cluster
print("=== Full cell-type composition per Cellohood cluster (k=10) ===")
comp_table = pd.DataFrame({"cluster": clusters, "cell_type": cell_type_arr})
full_comp = comp_table.groupby("cluster")["cell_type"].value_counts(normalize=True).unstack(fill_value=0)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print(full_comp.round(3))
print()

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
y = np.array([labels_df[p] for p in patients])

# For EACH individual cell type (not just the tumor/immune binary), test whether
# "fraction of tumor cells' assigned clusters containing that cell type" discriminates the label
print("=== Per-cell-type enrichment in tumor cells' neighborhoods: AUCROC vs mixed/compartmentalized ===")
results = []
for ct in all_cell_types:
    if ct in TUMOR_TYPES:
        continue
    is_ct = cell_type_arr == ct
    cluster_frac_ct = {c: is_ct[clusters == c].mean() for c in range(k)}
    scores = []
    for p in patients:
        mask = (sample_arr == p) & is_tumor
        tumor_clusters = clusters[mask]
        scores.append(np.mean([cluster_frac_ct[c] for c in tumor_clusters]))
    scores = np.array(scores)
    if scores.std() < 1e-8:
        continue
    auc = roc_auc_score(y, scores)
    n_cells_total = is_ct.sum()
    results.append({"cell_type": ct, "n_cells": n_cells_total, "auc": auc, "auc_best_direction": max(auc, 1-auc)})

res_df = pd.DataFrame(results).sort_values("auc_best_direction", ascending=False)
print(res_df.to_string(index=False))
