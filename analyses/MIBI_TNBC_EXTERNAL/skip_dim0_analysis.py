import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")
import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from lifelines import CoxPHFitter
from scipy.stats import chi2

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
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])

clin = pd.read_csv("keren_clinical_covariates_rasp_mibi.csv")
id_col = [c for c in clin.columns if 'patient' in c.lower() or c.lower() in ('id','sample')][0]
clin['patient_str'] = clin[id_col].apply(lambda x: f"patient{x}")
clin = clin.set_index('patient_str')
patient_to_idx = {p: i for i, p in enumerate(patients)}

def run_config(name, dims, K):
    print(f"\n{'='*70}\n{name}: dims={dims}, K={K}\n{'='*70}")
    decomposed = decomposed_full[:, dims]
    clusters = GaussianMixture(n_components=K, random_state=42).fit_predict(decomposed)

    cluster_frac_immune = {c: is_immune[clusters == c].mean() for c in range(K)}
    raw_scores, enriched_scores = [], []
    for i, p in enumerate(patients):
        mask = (sample_arr == p) & is_tumor
        tc = clusters[mask]
        raw_scores.append(np.mean([cluster_frac_immune[c] for c in tc]))
        enriched_scores.append(np.mean([cluster_frac_immune[c] / max(baseline_immune[i], 1e-6) for c in tc]))
    raw_scores, enriched_scores = np.array(raw_scores), np.array(enriched_scores)

    auc_raw = roc_auc_score(y, raw_scores)
    auc_enriched = roc_auc_score(y, enriched_scores)
    r_enrich_confound, _ = pearsonr(enriched_scores, 1.0/np.maximum(baseline_immune, 1e-6))
    r_raw_confound, _ = pearsonr(raw_scores, baseline_immune)
    print(f"Raw mixing score: AUC best-dir={max(auc_raw,1-auc_raw):.3f} | corr(raw, confound) r={r_raw_confound:.3f}")
    print(f"Enriched mixing score: AUC best-dir={max(auc_enriched,1-auc_enriched):.3f} | corr(enriched, confound) r={r_enrich_confound:.3f}")

    # residualize raw score against confound
    coef = np.polyfit(baseline_immune, raw_scores, 1)
    resid = raw_scores - np.polyval(coef, baseline_immune)
    auc_resid = roc_auc_score(y, resid)
    print(f"Residualized raw score (confound regressed out): AUC best-dir={max(auc_resid,1-auc_resid):.3f}")

    for ct in ["Treg", "CD8T", "DC", "CD11c-high"]:
        is_ct = cell_type_arr == ct
        base_ct = np.array([is_ct[sample_arr == p].mean() for p in patients])
        auc_base_ct = roc_auc_score(y, base_ct)
        cf = {c: is_ct[clusters == c].mean() for c in range(K)}
        nb_scores = np.array([np.mean([cf[c] for c in clusters[(sample_arr==p)&is_tumor]]) for p in patients])
        auc_nb = roc_auc_score(y, nb_scores)
        print(f"  {ct}: baseline best-dir={max(auc_base_ct,1-auc_base_ct):.3f} | neighborhood best-dir={max(auc_nb,1-auc_nb):.3f}")

    enriched_z = (enriched_scores - enriched_scores.mean()) / enriched_scores.std()
    baseline_z = (baseline_immune - baseline_immune.mean()) / baseline_immune.std()
    for outcome_time, outcome_event, label in [("Recurrence_time", "Recurrence", "Recurrence"), ("Survival_time", "Survival", "Overall survival")]:
        if outcome_time not in clin.columns:
            continue
        rows = []
        for p in patients:
            if p not in clin.index:
                continue
            t, e = clin.loc[p, outcome_time], clin.loc[p, outcome_event]
            if pd.isna(t) or pd.isna(e):
                continue
            i = patient_to_idx[p]
            rows.append({"T": t, "E": e, "baseline_z": baseline_z[i], "enriched_z": enriched_z[i]})
        cdf = pd.DataFrame(rows)
        if len(cdf) < 10 or cdf["E"].sum() < 3:
            continue
        cph_base = CoxPHFitter(penalizer=0.1).fit(cdf[["T","E","baseline_z"]], duration_col="T", event_col="E")
        cph_full = CoxPHFitter(penalizer=0.1).fit(cdf[["T","E","baseline_z","enriched_z"]], duration_col="T", event_col="E")
        lr_p = chi2.sf(2*(cph_full.log_likelihood_ - cph_base.log_likelihood_), df=1)
        print(f"  {label} (n={len(cdf)}, events={int(cdf['E'].sum())}): baseline C={cph_base.concordance_index_:.3f}, full C={cph_full.concordance_index_:.3f}, LR p={lr_p:.3f}")

# Confound alone reference
auc_confound = roc_auc_score(y, baseline_immune)
print(f"[Reference] Confound alone: AUC best-dir={max(auc_confound,1-auc_confound):.3f}")

run_config("dims [0,1,2] (original, for reference)", [0,1,2], 13)
run_config("dims [1,2] -- SKIP PC0 as requested", [1,2], 13)
