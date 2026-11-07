import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")
import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

np.random.seed(0)
N_PERM = 20000

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposed_full = PCA().fit_transform(emb)

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
sample_arr = df["sample"].values
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
y = np.array([labels_df[p] for p in patients])
n = len(y)
print(f"n={n}, n_positive={y.sum()}, n_negative={n-y.sum()}")

# Reconstruct Cellohood's raw mixing score at K=13 (the proper-pipeline config)
decomposed = decomposed_full[:, :2]
clusters = GaussianMixture(n_components=13, random_state=42).fit_predict(decomposed)
cluster_frac_immune = {c: is_immune[clusters == c].mean() for c in range(13)}
raw_scores = np.array([
    np.mean([cluster_frac_immune[c] for c in clusters[(sample_arr == p) & is_tumor]])
    for p in patients
])

def best_dir_auc(score, labels):
    a = roc_auc_score(labels, score)
    return max(a, 1 - a)

observed = best_dir_auc(raw_scores, y)
print(f"\nObserved best-direction AUC (Cellohood raw mixing score, K=13): {observed:.4f}")

# Null distribution: permute labels, recompute best-direction AUC each time
null_dist = np.empty(N_PERM)
for i in range(N_PERM):
    y_perm = np.random.permutation(y)
    null_dist[i] = best_dir_auc(raw_scores, y_perm)

p_value = (null_dist >= observed).mean()
print(f"Null distribution (sign-selection-corrected): mean={null_dist.mean():.4f}, std={null_dist.std():.4f}")
print(f"95th/99th percentile of null: {np.percentile(null_dist,95):.4f} / {np.percentile(null_dist,99):.4f}")
print(f"Permutation p-value (P(null >= observed)): {p_value:.4f}")

print("\n" + "="*70)
print("Same check for the RESIDUALIZED score (confound regressed out)")
print("="*70)
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])
coef = np.polyfit(baseline_immune, raw_scores, 1)
resid = raw_scores - np.polyval(coef, baseline_immune)

observed_resid = best_dir_auc(resid, y)
print(f"Observed best-direction AUC (residualized): {observed_resid:.4f}")

null_dist_resid = np.empty(N_PERM)
for i in range(N_PERM):
    y_perm = np.random.permutation(y)
    null_dist_resid[i] = best_dir_auc(resid, y_perm)

p_value_resid = (null_dist_resid >= observed_resid).mean()
print(f"Null distribution: mean={null_dist_resid.mean():.4f}, std={null_dist_resid.std():.4f}")
print(f"Permutation p-value: {p_value_resid:.4f}")
