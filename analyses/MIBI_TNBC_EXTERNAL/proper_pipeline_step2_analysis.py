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
K = 13          # smallest stability-selected cluster count at 75% variance
N_COMP = 2       # 75%-variance PLDA dimension (matches standard pipeline convention)

df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposed_full = PCA().fit_transform(emb)
decomposed = decomposed_full[:, :N_COMP]

clusters = GaussianMixture(n_components=K, random_state=42).fit_predict(decomposed)

cell_type_arr = df["cell_type"].values
sample_arr = df["sample"].values
all_cell_types = sorted(set(cell_type_arr))

print(f"=== PROPER PIPELINE: K={K} (stability-selected), N_COMP={N_COMP} (75% variance) ===")
print()
print("=== Full cell-type composition per cluster ===")
comp_table = pd.DataFrame({"cluster": clusters, "cell_type": cell_type_arr})
full_comp = comp_table.groupby("cluster")["cell_type"].value_counts(normalize=True).unstack(fill_value=0)
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
print(full_comp.round(3))
# Quantify separability: range (max-min) across clusters per cell type, and mean pairwise L1 distance between cluster profiles
ranges = (full_comp.max() - full_comp.min()).sort_values(ascending=False)
print("\nPer-cell-type range across clusters (max-min), sorted:")
print(ranges.round(3))
from scipy.spatial.distance import pdist
mean_pairwise_l1 = pdist(full_comp.values, metric='cityblock').mean()
print(f"\nMean pairwise L1 distance between cluster composition profiles: {mean_pairwise_l1:.3f}")
print()

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
print(f"n patients = {len(patients)}/{len(patients_all)}")
y = np.array([labels_df[p] for p in patients])
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])

auc_confound = roc_auc_score(y, baseline_immune)
print(f"\n=== Confound check: overall immune fraction alone, zero clustering ===")
print(f"AUCROC={auc_confound:.3f} (best-dir={max(auc_confound,1-auc_confound):.3f})  [expect ~0.085 raw / ~0.915 flipped]")

cluster_frac_immune = {c: is_immune[clusters == c].mean() for c in range(K)}
raw_scores, enriched_scores = [], []
for i, p in enumerate(patients):
    mask_tumor = (sample_arr == p) & is_tumor
    tumor_clusters = clusters[mask_tumor]
    raw = np.mean([cluster_frac_immune[c] for c in tumor_clusters])
    enriched = np.mean([cluster_frac_immune[c] / max(baseline_immune[i], 1e-6) for c in tumor_clusters])
    raw_scores.append(raw)
    enriched_scores.append(enriched)
raw_scores, enriched_scores = np.array(raw_scores), np.array(enriched_scores)

auc_raw = roc_auc_score(y, raw_scores)
auc_enriched = roc_auc_score(y, enriched_scores)
r_p, _ = pearsonr(enriched_scores, 1.0/np.maximum(baseline_immune,1e-6))
r_s, _ = spearmanr(enriched_scores, 1.0/np.maximum(baseline_immune,1e-6))

print(f"\n=== Core mixing-score results (PROPER pipeline, K={K}) ===")
print(f"RAW mixing score:      AUCROC={auc_raw:.3f} (best-dir={max(auc_raw,1-auc_raw):.3f})")
print(f"ENRICHED mixing score: AUCROC={auc_enriched:.3f} (best-dir={max(auc_enriched,1-auc_enriched):.3f})")
print(f"Correlation(enriched, 1/baseline alone): Pearson r={r_p:.3f}, Spearman r={r_s:.3f}")

# Residualized check
coef = np.polyfit(baseline_immune, raw_scores, 1)
resid = raw_scores - np.polyval(coef, baseline_immune)
auc_resid = roc_auc_score(y, resid)
print(f"Residualized raw score (confound regressed out): AUCROC={auc_resid:.3f} (best-dir={max(auc_resid,1-auc_resid):.3f})")

print(f"\n=== Immune subtype breakdown (PROPER pipeline, K={K}) ===")
for ct in ["Treg", "CD8T", "DC", "CD11c-high"]:
    is_ct = cell_type_arr == ct
    baseline_ct = np.array([is_ct[sample_arr == p].mean() for p in patients])
    auc_baseline_ct = roc_auc_score(y, baseline_ct)
    cluster_frac_ct = {c: is_ct[clusters == c].mean() for c in range(K)}
    scores = []
    for p in patients:
        mask = (sample_arr == p) & is_tumor
        tumor_clusters = clusters[mask]
        scores.append(np.mean([cluster_frac_ct[c] for c in tumor_clusters]))
    scores = np.array(scores)
    auc_neighborhood = roc_auc_score(y, scores) if scores.std() > 1e-8 else float('nan')
    print(f"{ct}: baseline-alone best-dir={max(auc_baseline_ct,1-auc_baseline_ct):.3f} | "
          f"neighborhood-enrichment best-dir={max(auc_neighborhood,1-auc_neighborhood):.3f}")

# Save results for the report
results = {
    "K": K, "N_COMP": N_COMP,
    "auc_confound_best_dir": float(max(auc_confound,1-auc_confound)),
    "auc_raw_best_dir": float(max(auc_raw,1-auc_raw)),
    "auc_enriched_best_dir": float(max(auc_enriched,1-auc_enriched)),
    "corr_enriched_vs_confound_pearson": float(r_p),
    "corr_enriched_vs_confound_spearman": float(r_s),
    "auc_residualized_best_dir": float(max(auc_resid,1-auc_resid)),
    "mean_pairwise_l1_cluster_profiles": float(mean_pairwise_l1),
}
import json
with open("proper_pipeline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved proper_pipeline_results.json")
