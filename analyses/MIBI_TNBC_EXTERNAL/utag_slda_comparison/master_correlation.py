import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

SCRATCH = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad"

cello = pd.read_csv(os.path.join(SCRATCH, "cellohood_per_patient_stats.csv"))
cyto = pd.read_csv(os.path.join(SCRATCH, "cytocommunity_repro/CytoCommunity/Tutorial/Unsupervised_MIBI/unsupervised_cytocommunity_stat.csv"))
utag = pd.read_csv(os.path.join(SCRATCH, "utag_out/utag_per_patient_stats.csv"))
slda = pd.read_csv(os.path.join(SCRATCH, "slda_out/slda_per_patient_stats.csv"))

cyto = cyto.rename(columns={
    "tumor_immune_exposure": "cytocommunity_raw_mixing",
    "tumor_immune_enrichment_ratio": "cytocommunity_enrichment",
})

# sanity check: overall_immune_frac (the trivial confound) should match across all four independently-computed sources
merged_check = cello[["patient", "overall_immune_frac"]].merge(
    cyto[["patient", "overall_immune_frac"]], on="patient", suffixes=("_cello", "_cyto")
)
max_diff = (merged_check["overall_immune_frac_cello"] - merged_check["overall_immune_frac_cyto"]).abs().max()
print(f"Max diff in confound (overall_immune_frac) between Cellohood-pipeline and CytoCommunity-pipeline computations: {max_diff:.6g}")

utag_cols_to_use = [c for c in utag.columns if c.startswith("utag_raw__") or c.startswith("utag_enrich__")]
slda_cols_to_use = [c for c in slda.columns if c.startswith("slda_")]

df = cello[["patient", "true_label", "overall_immune_frac", "cellohood_bag_raw_mixing",
            "cellohood_bag_enrichment", "cellohood_gmm_raw_mixing", "cellohood_gmm_enrichment"]].copy()
df = df.merge(cyto[["patient", "cytocommunity_raw_mixing", "cytocommunity_enrichment"]], on="patient", how="inner")
df = df.merge(utag[["patient"] + utag_cols_to_use], on="patient", how="inner")
df = df.merge(slda[["patient"] + slda_cols_to_use], on="patient", how="inner")

print(f"\nFinal merged table: {df.shape[0]} patients, {df.shape[1]} columns")
df.to_csv(os.path.join(SCRATCH, "all_methods_per_patient_stats.csv"), index=False)

y = df["true_label"].values
conf = df["overall_immune_frac"].values

stat_cols = [c for c in df.columns if c not in ("patient", "true_label")]

print("\n=== Per-statistic AUCROC + confound correlation summary ===")
summary_rows = []
for c in stat_cols:
    v = df[c].values
    auc = roc_auc_score(y, v)
    rp, _ = pearsonr(v, conf)
    rs, _ = spearmanr(v, conf)
    coef = np.polyfit(conf, v, 1)
    resid = v - np.polyval(coef, conf)
    auc_resid = roc_auc_score(y, resid)
    summary_rows.append({
        "statistic": c, "AUCROC": auc, "AUCROC_flip": 1 - auc,
        "pearson_vs_confound": rp, "spearman_vs_confound": rs,
        "residual_AUCROC": auc_resid, "residual_AUCROC_flip": 1 - auc_resid,
    })
summary = pd.DataFrame(summary_rows)
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
summary.to_csv(os.path.join(SCRATCH, "method_confound_summary.csv"), index=False)

print("\n=== Full pairwise Pearson correlation matrix (all statistics incl. confound) ===")
pearson_mat = df[stat_cols].corr(method="pearson")
print(pearson_mat.round(3).to_string())
pearson_mat.to_csv(os.path.join(SCRATCH, "pearson_corr_matrix.csv"))

print("\n=== Full pairwise Spearman correlation matrix (all statistics incl. confound) ===")
spearman_mat = df[stat_cols].corr(method="spearman")
print(spearman_mat.round(3).to_string())
spearman_mat.to_csv(os.path.join(SCRATCH, "spearman_corr_matrix.csv"))

print("\nDone.")
