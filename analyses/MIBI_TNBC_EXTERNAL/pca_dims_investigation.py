import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellohood
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)
print("explained variance ratio, first 5 dims:", decomposer.explained_variance_ratio_[:5])
print("cumulative:", decomposer.explained_variance_ratio_[:5].cumsum())

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
sample_arr = df["sample"].values
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
y = np.array([labels_df[p] for p in patients])
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])

print("\n=== Per-dimension correlation with the trivial bulk-immune confound ===")
for dim in range(5):
    per_patient_mean = np.array([decomposed_full[sample_arr == p, dim].mean() for p in patients])
    r, _ = pearsonr(per_patient_mean, baseline_immune)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, per_patient_mean)
    print(f"PC{dim}: corr with bulk-immune-confound r={r:.3f} | AUC vs label={auc:.3f} (best-dir={max(auc,1-auc):.3f})")

print("\n=== Pairwise scatter plots (per-cell, subsampled for plotting) ===")
rng = np.random.default_rng(42)
sub_idx = rng.choice(len(decomposed_full), size=20000, replace=False)
label_per_cell = np.array([labels_df.get(s, np.nan) for s in sample_arr])

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
pairs = [(0,1), (0,2), (1,2)]
colors = {0: "#0072B2", 1: "#D55E00"}
for ax, (d1, d2) in zip(axes, pairs):
    for lab in [0, 1]:
        m = sub_idx[label_per_cell[sub_idx] == lab]
        ax.scatter(decomposed_full[m, d1], decomposed_full[m, d2], s=2, alpha=0.3, c=colors[lab], label=f"label={lab}")
    ax.set_xlabel(f"PC{d1}")
    ax.set_ylabel(f"PC{d2}")
    ax.set_title(f"PC{d1} vs PC{d2} (per-cell, colored by patient label)")
    ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig("pca_pairwise_dims012.png")
print("saved pca_pairwise_dims012.png")
