import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")

emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
sample_arr = df["sample"].values
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if ((sample_arr == p) & is_tumor).sum() >= 5]
print(f"Using {len(patients)}/{len(patients_all)} patients with >=5 tumor cells")

# --- Step 0: does the trivial per-patient overall immune fraction alone predict the label? (sanity check vs. the 0.085 found for CytoCommunity's cohort) ---
baseline_immune = {p: is_immune[sample_arr == p].mean() for p in patients}
y = np.array([labels_df[p] for p in patients])
baseline_vals = np.array([baseline_immune[p] for p in patients])
auc_confound_only = roc_auc_score(y, baseline_vals)
print(f"Confound check (overall per-patient immune fraction alone, zero Cellohood info): AUCROC={auc_confound_only:.3f}")

n_comp, k = 10, 10  # best config from the earlier linear-probe sweep
decomposed = decomposed_full[:, :n_comp]
gmm = GaussianMixture(n_components=k, random_state=42)
clusters = gmm.fit_predict(decomposed)

cluster_frac_immune = {c: is_immune[clusters == c].mean() for c in range(k)}

# --- Original (raw/absolute) mixing score ---
raw_scores = []
enriched_scores = []
for p in patients:
    mask_tumor = (sample_arr == p) & is_tumor
    tumor_clusters = clusters[mask_tumor]
    raw = np.mean([cluster_frac_immune[c] for c in tumor_clusters])
    enriched = np.mean([cluster_frac_immune[c] / max(baseline_immune[p], 1e-6) for c in tumor_clusters])
    raw_scores.append(raw)
    enriched_scores.append(enriched)

raw_scores = np.array(raw_scores)
enriched_scores = np.array(enriched_scores)

auc_raw = roc_auc_score(y, raw_scores)
auc_enriched = roc_auc_score(y, enriched_scores)
print(f"Cellohood RAW mixing score:        AUCROC={auc_raw:.3f}  (flipped: {1-auc_raw:.3f})")
print(f"Cellohood ENRICHMENT-corrected:    AUCROC={auc_enriched:.3f}  (flipped: {1-auc_enriched:.3f})")

# --- Critical control: does the DENOMINATOR ALONE (1/baseline_immune, zero Cellohood clustering info) already explain the enrichment score's performance? ---
inv_baseline = np.array([1.0 / max(baseline_immune[p], 1e-6) for p in patients])
auc_denom_only = roc_auc_score(y, inv_baseline)
print(f"\nCONTROL -- 1/baseline_immune ALONE (no Cellohood clustering at all): AUCROC={auc_denom_only:.3f} (flipped: {1-auc_denom_only:.3f})")

# Correlation between the full enrichment score and the denominator-only control
from scipy.stats import pearsonr, spearmanr
r_pearson, _ = pearsonr(enriched_scores, inv_baseline)
r_spearman, _ = spearmanr(enriched_scores, inv_baseline)
print(f"Correlation(enrichment score, 1/baseline alone): Pearson r={r_pearson:.3f}, Spearman r={r_spearman:.3f}")

# Does the numerator (cluster_frac_immune averaged over tumor cells, WITHOUT dividing) correlate with label at all once we control for baseline?
# Partial check: residualize enriched score's numerator against baseline, see if what's left still predicts label
numerator_only = raw_scores  # this is cluster_frac_immune averaged, i.e. the numerator of the ratio
r_num_denom, _ = pearsonr(numerator_only, baseline_vals)
print(f"Correlation(numerator=raw mixing score, baseline_immune): Pearson r={r_num_denom:.3f}")
