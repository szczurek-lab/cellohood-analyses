import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, fisher_exact
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)

DISTANCE_THRESHOLD = 70.0
K = 13
N_COMP = 2

# ---------------------------------------------------------------------------
# 1. Rebuild the proper-pipeline K=13 clustering (identical to
#    proper_pipeline_step2_analysis.py, random_state=42 -> deterministic)
# ---------------------------------------------------------------------------
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposed_full = PCA().fit_transform(emb)
decomposed = decomposed_full[:, :N_COMP]
clusters = GaussianMixture(n_components=K, random_state=42).fit_predict(decomposed)

cell_type_arr = df["cell_type"].values
sample_arr = df["sample"].values

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
print(f"n patients (cellohood pipeline, >=5 tumor cells) = {len(patients)}/{len(patients_all)}")

baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])
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

pid_to_int = lambda s: int(s.replace("patient", ""))
patient_ids = np.array([pid_to_int(p) for p in patients])

feat_df = pd.DataFrame({
    "sample": patients,
    "ID": patient_ids,
    "baseline_immune": baseline_immune,
    "raw_score": raw_scores,
    "enriched_score": enriched_scores,
    "mixed_label": [labels_df[p] for p in patients],
})

# ---------------------------------------------------------------------------
# 2. Load & merge the original Keren et al. clinical covariates
#    (recovered from github.com/aalokpatwa/rasp-mibi, the RASP-MIBI repo for
#    Patwa/Risom/Angelo/Keren/Rubin, Commun Biol 2021 — reanalysis of this
#    exact 41-patient cohort). ID numbering verified (below) to match
#    CytoCommunity's patientN IDs directly, 1:1.
# ---------------------------------------------------------------------------
clin_path = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/rasp-mibi/intermediate_data/covariate_rsf_data.csv"
clin = pd.read_csv(clin_path)

merged = feat_df.merge(clin, on="ID", how="left")

# ID-mapping sanity check: Architecture column (independently derived by a
# different group) must match our mixed/compartmentalized label for every
# patient that has it.
check = merged.dropna(subset=["Architecture"])
arch_bin = check["Architecture"].map({"Mixed": 1, "Compartmentalized": 0})
n_mismatch = (arch_bin != check["mixed_label"]).sum()
print(f"\n=== ID-mapping sanity check ===")
print(f"Patients with Architecture label from independent source: {len(check)}")
print(f"Mismatches vs our mixed/compartmentalized label: {n_mismatch}")
print(f"Patients missing clinical data entirely: {sorted(merged.loc[merged['Architecture'].isna(), 'sample'].tolist())}")

merged = merged.dropna(subset=["Architecture"]).copy()
N = len(merged)
print(f"\nFinal clinically-annotated cohort: N={N}")
print(f"Recurrence events: {merged['Recurrence'].sum()}, Survival(death) events: {merged['Survival'].sum()}")

# standardize continuous predictors
for col in ["baseline_immune", "raw_score", "enriched_score"]:
    merged[col + "_z"] = (merged[col] - merged[col].mean()) / merged[col].std()

print("\nCorrelation(enriched_score, baseline_immune) on this subcohort:",
      pearsonr(merged["enriched_score"], merged["baseline_immune"]))

# ---------------------------------------------------------------------------
# 3. Survival analysis helper: baseline-only vs baseline+neighborhood,
#    nested likelihood-ratio test, light L2 penalizer for stability at N~33.
# ---------------------------------------------------------------------------
def run_cox_comparison(data, duration_col, event_col, label):
    print(f"\n===== {label}: duration={duration_col}, event={event_col} =====")
    results = {}
    for name, cols in [
        ("baseline (bulk immune fraction) only", ["baseline_immune_z"]),
        ("neighborhood (K=13 enriched score) only", ["enriched_score_z"]),
        ("neighborhood RAW (no enrichment-ratio) only", ["raw_score_z"]),
        ("baseline + neighborhood (combined)", ["baseline_immune_z", "enriched_score_z"]),
    ]:
        sub = data[cols + [duration_col, event_col]].dropna()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub, duration_col=duration_col, event_col=event_col)
        conc = cph.concordance_index_
        ll = cph.log_likelihood_
        print(f"-- {name} (n={len(sub)}) --")
        print(cph.summary[["coef", "exp(coef)", "p"]])
        print(f"   concordance={conc:.3f}  log-likelihood={ll:.3f}")
        results[name] = {"cph": cph, "conc": conc, "ll": ll, "n": len(sub)}

    # Nested LR test: baseline-only (null) vs baseline+neighborhood (alt)
    ll_null = results["baseline (bulk immune fraction) only"]["ll"]
    ll_alt = results["baseline + neighborhood (combined)"]["ll"]
    from scipy.stats import chi2
    lr_stat = 2 * (ll_alt - ll_null)
    lr_p = chi2.sf(lr_stat, df=1)
    print(f"\nLR test (does enriched_score add beyond bulk immune fraction?): "
          f"LR stat={lr_stat:.3f}, p={lr_p:.4f}")
    return results, lr_p


rec_results, rec_lr_p = run_cox_comparison(merged, "Recurrence_time", "Recurrence", "RECURRENCE")
os_results, os_lr_p = run_cox_comparison(merged, "Survival_time", "Survival", "OVERALL SURVIVAL")

# ---------------------------------------------------------------------------
# 4. Categorical / continuous clinical variables: grade, stage, age
#    using the same AUCROC-based confound-check structure as section 10
#    (baseline-alone vs neighborhood-alone vs residualized).
# ---------------------------------------------------------------------------
def auc_confound_check(y, baseline, raw, enriched, label):
    print(f"\n===== {label} (n={len(y)}) =====")
    auc_b = roc_auc_score(y, baseline)
    auc_r = roc_auc_score(y, raw)
    auc_e = roc_auc_score(y, enriched)
    coef = np.polyfit(baseline, raw, 1)
    resid = raw - np.polyval(coef, baseline)
    auc_resid = roc_auc_score(y, resid)
    r_p, _ = pearsonr(enriched, baseline)
    print(f"Bulk-composition-alone (immune fraction):        AUCROC best-dir={max(auc_b,1-auc_b):.3f}")
    print(f"Neighborhood RAW score alone:                     AUCROC best-dir={max(auc_r,1-auc_r):.3f}")
    print(f"Neighborhood ENRICHED score alone:                AUCROC best-dir={max(auc_e,1-auc_e):.3f}")
    print(f"Residualized RAW (confound regressed out):        AUCROC best-dir={max(auc_resid,1-auc_resid):.3f}")
    print(f"Correlation(enriched, baseline):                  r={r_p:.3f}")
    return dict(auc_baseline=max(auc_b,1-auc_b), auc_raw=max(auc_r,1-auc_r),
                auc_enriched=max(auc_e,1-auc_e), auc_residualized=max(auc_resid,1-auc_resid),
                corr_enriched_baseline=r_p)

# Grade: binarize high (3,4) vs low (1,2); drop unknown code 9 (n=1, patient 28)
grade_df = merged[merged["grade"] != 9].copy()
grade_df["high_grade"] = (grade_df["grade"] >= 3).astype(int)
print(f"\nGrade distribution (excl. unknown code 9): \n{grade_df['grade'].value_counts().sort_index()}")
print(f"High-grade (>=3) n={grade_df['high_grade'].sum()}, Low-grade n={(1-grade_df['high_grade']).sum()}")
grade_results = auc_confound_check(
    grade_df["high_grade"].values, grade_df["baseline_immune"].values,
    grade_df["raw_score"].values, grade_df["enriched_score"].values,
    "GRADE (high >=3 vs low <3)",
)

# Stage: first digit of the 2-digit code as a coarse ordinal; binarize III/IV vs I/II
merged["stage_group"] = (merged["stage"] // 10).astype(int)
print(f"\nStage-group (first digit of 2-digit 'stage' field) distribution:\n{merged['stage_group'].value_counts().sort_index()}")
merged["high_stage"] = (merged["stage_group"] >= 3).astype(int)
print(f"High-stage (III/IV) n={merged['high_stage'].sum()}, Low-stage (I/II) n={(1-merged['high_stage']).sum()}")
stage_results = auc_confound_check(
    merged["high_stage"].values, merged["baseline_immune"].values,
    merged["raw_score"].values, merged["enriched_score"].values,
    "STAGE (III/IV vs I/II, coarse decode -- caveat in report)",
)

# Age: continuous -- partial correlation style check
print(f"\n===== AGE (continuous, n={len(merged)}) =====")
r_base, p_base = pearsonr(merged["age"], merged["baseline_immune"])
r_raw, p_raw = pearsonr(merged["age"], merged["raw_score"])
r_enr, p_enr = pearsonr(merged["age"], merged["enriched_score"])
coef = np.polyfit(merged["baseline_immune"], merged["raw_score"], 1)
resid = merged["raw_score"] - np.polyval(coef, merged["baseline_immune"])
r_resid, p_resid = pearsonr(merged["age"], resid)
print(f"corr(age, baseline_immune):      r={r_base:.3f}, p={p_base:.3f}")
print(f"corr(age, raw_score):            r={r_raw:.3f}, p={p_raw:.3f}")
print(f"corr(age, enriched_score):       r={r_enr:.3f}, p={p_enr:.3f}")
print(f"corr(age, residualized raw):     r={r_resid:.3f}, p={p_resid:.3f}")

# ---------------------------------------------------------------------------
# 5. Save a compact results summary
# ---------------------------------------------------------------------------
import json
summary = {
    "N_clinical_cohort": int(N),
    "id_mapping_mismatches": int(n_mismatch),
    "recurrence_events": int(merged["Recurrence"].sum()),
    "survival_death_events": int(merged["Survival"].sum()),
    "recurrence_LR_test_p_enriched_beyond_baseline": float(rec_lr_p),
    "os_LR_test_p_enriched_beyond_baseline": float(os_lr_p),
    "recurrence_concordance": {k: v["conc"] for k, v in rec_results.items()},
    "os_concordance": {k: v["conc"] for k, v in os_results.items()},
    "grade": grade_results,
    "stage": stage_results,
    "age_corr": {"baseline": r_base, "raw": r_raw, "enriched": r_enr, "residualized": r_resid},
}
with open("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL/clinical_richer_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
merged.to_csv("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL/clinical_richer_merged.csv", index=False)
print("\nSaved clinical_richer_results.json and clinical_richer_merged.csv")
