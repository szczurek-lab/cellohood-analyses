import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

DISTANCE_THRESHOLD = 70.0

with open("cell_type_columns.txt") as f:
    CELL_TYPES = [l.strip() for l in f if l.strip()]

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}

df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
bag_cello_df = cellohood.load_bag_cello_df(f"MIBI_CB_{DISTANCE_THRESHOLD}")
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")

emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)
variance_ratios = decomposer.explained_variance_ratio_.cumsum()
print("cumulative variance ratios (first 10):", variance_ratios[:10])

cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
sample_arr = df["sample"].values

for n_comp_try, k_try in [(2, 8), (5, 8), (5, 12), (10, 10)]:
    decomposed = decomposed_full[:, :n_comp_try]
    gmm = GaussianMixture(n_components=k_try, random_state=42)
    clusters = gmm.fit_predict(decomposed)

    cluster_frac_immune = {}
    for k in range(k_try):
        mask = clusters == k
        cluster_frac_immune[k] = is_immune[mask].mean() if mask.sum() > 0 else 0.0

    results = []
    for patient in sorted(set(sample_arr)):
        p_tumor_mask = (sample_arr == patient) & is_tumor
        n_tumor = p_tumor_mask.sum()
        if n_tumor < 5:
            continue
        tumor_clusters = clusters[p_tumor_mask]
        mixing_score = np.mean([cluster_frac_immune[c] for c in tumor_clusters])
        results.append({"sample": patient, "mixing_score": mixing_score, "n_tumor": n_tumor, "graph_label": labels_df[patient]})

    res_df = pd.DataFrame(results)
    auc = roc_auc_score(res_df["graph_label"], res_df["mixing_score"])
    print(f"n_comp={n_comp_try}, k={k_try}: AUCROC={auc:.3f}  (n_patients={len(res_df)}, score range [{res_df.mixing_score.min():.3f}, {res_df.mixing_score.max():.3f}])")
    if n_comp_try == 5 and k_try == 8:
        res_df.to_csv("mibi_mixing_scores_v2.csv", index=False)
