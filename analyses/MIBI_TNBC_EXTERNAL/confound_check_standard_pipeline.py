import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")
import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)

# STANDARD Cellohood convention: 75%-variance PLDA component selection, K=8 fixed default
variance_ratios = decomposer.explained_variance_ratio_.cumsum()
n_comp = max(np.abs(variance_ratios - 0.75).argmin(), 2)
K = 8
print(f"STANDARD pipeline: n_comp (75% variance) = {n_comp}, K = {K}")

decomposed = decomposed_full[:, :n_comp]
gmm = GaussianMixture(n_components=K, random_state=42)
clusters = gmm.fit_predict(decomposed)

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
sample_arr = df["sample"].values
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
print(f"n patients = {len(patients)}/{len(patients_all)}")
y = np.array([labels_df[p] for p in patients])
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])
auc_confound = roc_auc_score(y, baseline_immune)
print(f"Confound alone (overall immune fraction, zero clustering): AUCROC={auc_confound:.3f} (best-dir={max(auc_confound,1-auc_confound):.3f})")

cluster_frac_immune = {c: is_immune[clusters == c].mean() for c in range(K)}
raw_scores, enriched_scores = [], []
for p in patients:
    mask_tumor = (sample_arr == p) & is_tumor
    tumor_clusters = clusters[mask_tumor]
    raw = np.mean([cluster_frac_immune[c] for c in tumor_clusters])
    enriched = np.mean([cluster_frac_immune[c] / max(baseline_immune[list(patients).index(p)], 1e-6) for c in tumor_clusters])
    raw_scores.append(raw)
    enriched_scores.append(enriched)
raw_scores, enriched_scores = np.array(raw_scores), np.array(enriched_scores)

auc_raw = roc_auc_score(y, raw_scores)
auc_enriched = roc_auc_score(y, enriched_scores)
r_p, _ = pearsonr(enriched_scores, 1.0/np.maximum(baseline_immune,1e-6))

print(f"STANDARD-pipeline Cellohood RAW mixing score:     AUCROC={auc_raw:.3f} (best-dir={max(auc_raw,1-auc_raw):.3f})")
print(f"STANDARD-pipeline Cellohood ENRICHED mixing score: AUCROC={auc_enriched:.3f} (best-dir={max(auc_enriched,1-auc_enriched):.3f})")
print(f"Correlation(enriched, 1/baseline alone): Pearson r={r_p:.3f}")
