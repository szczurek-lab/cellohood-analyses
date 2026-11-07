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
TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}

labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]

# ---- 1. Bag-level (raw spatial neighborhood, within-patient) mixing score, from the cellohood bag_df ----
bag_cello_df = cellohood.load_bag_cello_df(f"MIBI_CB_{DISTANCE_THRESHOLD}")
cell_df = bag_cello_df.cell_df
is_tumor = cell_df["cell_type"].isin(TUMOR_TYPES).values
is_immune = cell_df["cell_type"].isin(IMMUNE_TYPES).values
sample_arr = cell_df["sample"].values
bag_col = bag_cello_df.cellohood_neighborhood_cluster_colname
bag_id = cell_df[bag_col].values

patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]

rows = {p: {} for p in patients}
for p in patients:
    p_mask = sample_arr == p
    p_bag_id = bag_id[p_mask]
    p_is_tumor = is_tumor[p_mask]
    p_is_immune = is_immune[p_mask]
    bag_immune_frac = pd.Series(p_is_immune).groupby(p_bag_id).mean().to_dict()
    tumor_bags = p_bag_id[p_is_tumor]
    exposures = np.array([bag_immune_frac[b] for b in tumor_bags])
    overall = p_is_immune.mean()
    rows[p]["true_label"] = int(labels_df[p])
    rows[p]["overall_immune_frac"] = overall
    rows[p]["cellohood_bag_raw_mixing"] = exposures.mean()
    rows[p]["cellohood_bag_enrichment"] = (exposures / overall).mean() if overall > 0 else np.nan

# ---- 2. Cohort-level GMM-cluster (pooled) mixing score, from cellohood cell predictions (MIBI_CP) ----
df_full = pd.read_csv("mibi_tnbc_full.csv")
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)

cell_type_arr2 = df_full["cell_type"].values
is_tumor2 = np.isin(cell_type_arr2, list(TUMOR_TYPES))
is_immune2 = np.isin(cell_type_arr2, list(IMMUNE_TYPES))
sample_arr2 = df_full["sample"].values

n_comp, k = 10, 10
decomposed = decomposed_full[:, :n_comp]
gmm = GaussianMixture(n_components=k, random_state=42)
clusters = gmm.fit_predict(decomposed)
cluster_frac_immune = {c: is_immune2[clusters == c].mean() for c in range(k)}

for p in patients:
    mask_tumor = (sample_arr2 == p) & is_tumor2
    tumor_clusters = clusters[mask_tumor]
    overall = rows[p]["overall_immune_frac"]
    raw = np.mean([cluster_frac_immune[c] for c in tumor_clusters])
    enriched = np.mean([cluster_frac_immune[c] / max(overall, 1e-6) for c in tumor_clusters])
    rows[p]["cellohood_gmm_raw_mixing"] = raw
    rows[p]["cellohood_gmm_enrichment"] = enriched

out = pd.DataFrame.from_dict(rows, orient="index")
out.index.name = "patient"
out = out.reset_index()
out.to_csv("/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/cellohood_per_patient_stats.csv", index=False)
print(out)
print()

y = out["true_label"].values
for col in ["overall_immune_frac", "cellohood_bag_raw_mixing", "cellohood_bag_enrichment",
            "cellohood_gmm_raw_mixing", "cellohood_gmm_enrichment"]:
    v = out[col].values
    auc = roc_auc_score(y, v)
    rp, _ = pearsonr(v, out["overall_immune_frac"].values)
    rs, _ = spearmanr(v, out["overall_immune_frac"].values)
    print(f"{col}: AUCROC={auc:.3f} (flip={1-auc:.3f}) | corr_with_confound Pearson={rp:.3f} Spearman={rs:.3f}")

print(f"\nSaved per-patient Cellohood stats for {len(out)} patients.")
