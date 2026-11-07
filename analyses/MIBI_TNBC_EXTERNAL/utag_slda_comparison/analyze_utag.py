import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

OUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/utag_out"
adata = ad.read_h5ad(os.path.join(OUT_DIR, "mibi_tnbc_utag_result.h5ad"))
print(adata)
utag_cols = [c for c in adata.obs.columns if "UTAG" in c]
print("UTAG label columns:", utag_cols)

obs = adata.obs
is_tumor = obs["is_tumor"].values
is_immune = obs["is_immune"].values
sample_arr = obs["Slide"].astype(str).values
label_all = obs.groupby("Slide")["graph_label"].first().to_dict()

patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
print(f"Using {len(patients)}/{len(patients_all)} patients with >=5 tumor cells")

results_rows = {p: {"true_label": label_all[p]} for p in patients}
baseline_immune = {}
for p in patients:
    pm = sample_arr == p
    baseline_immune[p] = is_immune[pm].mean()
    results_rows[p]["overall_immune_frac"] = baseline_immune[p]

y = np.array([results_rows[p]["true_label"] for p in patients])
conf = np.array([results_rows[p]["overall_immune_frac"] for p in patients])
auc_conf = roc_auc_score(y, conf)
print(f"\nConfound check (overall per-patient immune fraction alone): AUCROC={auc_conf:.3f} (flip={1-auc_conf:.3f})")

for col in utag_cols:
    cluster_labels = obs[col].values
    unique_clusters = pd.unique(cluster_labels)
    # global (cohort-pooled) immune fraction per cluster
    cluster_immune_frac = {}
    for c in unique_clusters:
        m = cluster_labels == c
        cluster_immune_frac[c] = is_immune[m].mean()

    raw_scores, enrich_scores = [], []
    for p in patients:
        pm = (sample_arr == p) & is_tumor
        tumor_clusters = cluster_labels[pm]
        exposures = np.array([cluster_immune_frac[c] for c in tumor_clusters])
        raw = exposures.mean()
        enr = (exposures / max(baseline_immune[p], 1e-6)).mean()
        raw_scores.append(raw)
        enrich_scores.append(enr)
        results_rows[p][f"utag_raw__{col}"] = raw
        results_rows[p][f"utag_enrich__{col}"] = enr

    raw_scores = np.array(raw_scores)
    enrich_scores = np.array(enrich_scores)
    n_clusters_obs = len(unique_clusters)

    auc_raw = roc_auc_score(y, raw_scores)
    auc_enr = roc_auc_score(y, enrich_scores)
    rp_raw, _ = pearsonr(raw_scores, conf)
    rs_raw, _ = spearmanr(raw_scores, conf)
    rp_enr, _ = pearsonr(enrich_scores, conf)
    rs_enr, _ = spearmanr(enrich_scores, conf)

    # Residualize raw score on confound
    coef = np.polyfit(conf, raw_scores, 1)
    resid_raw = raw_scores - np.polyval(coef, conf)
    auc_resid_raw = roc_auc_score(y, resid_raw)

    coef2 = np.polyfit(conf, enrich_scores, 1)
    resid_enr = enrich_scores - np.polyval(coef2, conf)
    auc_resid_enr = roc_auc_score(y, resid_enr)

    print(f"\n=== {col} (n_clusters={n_clusters_obs}) ===")
    print(f"  RAW:    AUCROC={auc_raw:.3f} (flip={1-auc_raw:.3f}) | corr_w_confound Pearson={rp_raw:.3f} Spearman={rs_raw:.3f} | residual AUCROC={auc_resid_raw:.3f} (flip={1-auc_resid_raw:.3f})")
    print(f"  ENRICH: AUCROC={auc_enr:.3f} (flip={1-auc_enr:.3f}) | corr_w_confound Pearson={rp_enr:.3f} Spearman={rs_enr:.3f} | residual AUCROC={auc_resid_enr:.3f} (flip={1-auc_resid_enr:.3f})")

out = pd.DataFrame.from_dict(results_rows, orient="index")
out.index.name = "patient"
out = out.reset_index()
out.to_csv(os.path.join(OUT_DIR, "utag_per_patient_stats.csv"), index=False)
print("\nSaved per-patient UTAG stats CSV.")
