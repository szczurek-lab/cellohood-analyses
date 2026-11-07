import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

INPUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/cytocommunity_repro/CytoCommunity/Tutorial/Supervised/MIBI-TNBC_Input"
OUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/slda_out"

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}

topic_weights = pd.read_pickle(os.path.join(OUT_DIR, "topic_weights.pkl"))
print("raw index type:", type(topic_weights.index), topic_weights.index[:3])
topic_weights.index = pd.MultiIndex.from_tuples(list(topic_weights.index), names=["sample", "local_idx"])
print("topic_weights shape:", topic_weights.shape)
print(topic_weights.head())

patients_list = [l.strip() for l in open(os.path.join(INPUT_DIR, "ImageNameList.txt")) if l.strip()]

# Reload per-cell cell type + sample + label to align with topic_weights' (sample, cell_idx) MultiIndex
cell_types_all = []
sample_all = []
label_all = {}
for p in patients_list:
    ct = pd.read_csv(os.path.join(INPUT_DIR, f"{p}_CellTypeLabel.txt"), header=None, names=["cell_type"])["cell_type"].values
    lab = int(open(os.path.join(INPUT_DIR, f"{p}_GraphLabel.txt")).read().strip())
    cell_types_all.append(ct)
    sample_all.append(np.full(len(ct), p))
    label_all[p] = lab
cell_types_all = np.concatenate(cell_types_all)
sample_all = np.concatenate(sample_all)

is_tumor = np.isin(cell_types_all, list(TUMOR_TYPES))
is_immune = np.isin(cell_types_all, list(IMMUNE_TYPES))

# Build a per-cell dataframe matching topic_weights' index order (sample, local_cell_idx)
# topic_weights index is a MultiIndex (sample, cell_id) where cell_id matches the row order in the
# per-sample dataframe passed into feat.featurize_samples (0..n-1 since we didn't filter anchors).
cell_idx_within_sample = np.concatenate([np.arange((sample_all == p).sum()) for p in patients_list])
cell_meta = pd.DataFrame({
    "sample": sample_all,
    "local_idx": cell_idx_within_sample,
    "cell_type": cell_types_all,
    "is_tumor": is_tumor,
    "is_immune": is_immune,
})
cell_meta.index = pd.MultiIndex.from_arrays([cell_meta["sample"], cell_meta["local_idx"]])

# Align: topic_weights may have dropped some cells (blank neighborhoods removed). Inner-join on index.
aligned = cell_meta.join(topic_weights, how="inner")
print(f"Aligned {len(aligned)} / {len(cell_meta)} cells to topic weights (some cells with empty neighborhoods may be dropped).")

topic_cols = [c for c in topic_weights.columns if c.startswith("Topic-")]
n_topics = len(topic_cols)

# Hard topic assignment = argmax
aligned["hard_topic"] = aligned[topic_cols].values.argmax(axis=1)

sample_arr = aligned["sample"].values
is_tumor_a = aligned["is_tumor"].values
is_immune_a = aligned["is_immune"].values

patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor_a & (sample_arr == p)).sum() >= 5]
print(f"Using {len(patients)}/{len(patients_all)} patients with >=5 tumor cells (post topic-weight alignment)")

baseline_immune = {p: is_immune_a[sample_arr == p].mean() for p in patients}
y = np.array([label_all[p] for p in patients])
conf = np.array([baseline_immune[p] for p in patients])
auc_conf = roc_auc_score(y, conf)
print(f"\nConfound check (overall per-patient immune fraction alone): AUCROC={auc_conf:.3f} (flip={1-auc_conf:.3f})")

results_rows = {p: {"true_label": label_all[p], "overall_immune_frac": baseline_immune[p]} for p in patients}

# --- Hard-topic (argmax) global cluster immune-fraction statistic, analogous to UTAG/Cellohood ---
hard_topic = aligned["hard_topic"].values
global_topic_immune_frac = {t: is_immune_a[hard_topic == t].mean() for t in range(n_topics)}

raw_scores, enrich_scores = [], []
for p in patients:
    pm = (sample_arr == p) & is_tumor_a
    tumor_topics = hard_topic[pm]
    exposures = np.array([global_topic_immune_frac[t] for t in tumor_topics])
    raw = exposures.mean()
    enr = (exposures / max(baseline_immune[p], 1e-6)).mean()
    raw_scores.append(raw)
    enrich_scores.append(enr)
    results_rows[p]["slda_hard_raw_mixing"] = raw
    results_rows[p]["slda_hard_enrichment"] = enr
raw_scores = np.array(raw_scores)
enrich_scores = np.array(enrich_scores)

def report(name, stat):
    auc = roc_auc_score(y, stat)
    rp, _ = pearsonr(stat, conf)
    rs, _ = spearmanr(stat, conf)
    coef = np.polyfit(conf, stat, 1)
    resid = stat - np.polyval(coef, conf)
    auc_resid = roc_auc_score(y, resid)
    print(f"{name}: AUCROC={auc:.3f} (flip={1-auc:.3f}) | corr_w_confound Pearson={rp:.3f} Spearman={rs:.3f} | residual AUCROC={auc_resid:.3f} (flip={1-auc_resid:.3f})")
    return resid

print(f"\n=== Hard-topic (argmax, n_topics={n_topics}) global-pooled immune-fraction stat ===")
report("RAW", raw_scores)
report("ENRICH", enrich_scores)

# --- Soft (probability-weighted) version: each cell contributes to ALL topics by its weight ---
# global topic immune-content: weighted immune fraction of each topic across the WHOLE cohort
W = aligned[topic_cols].values  # (n_cells, n_topics)
topic_mass = W.sum(axis=0)  # total soft-mass per topic
topic_immune_mass = (W * is_immune_a[:, None]).sum(axis=0)
soft_topic_immune_frac = topic_immune_mass / np.maximum(topic_mass, 1e-12)

raw_soft, enrich_soft = [], []
for p in patients:
    pm = (sample_arr == p) & is_tumor_a
    w_tumor = W[pm]  # (n_tumor_p, n_topics)
    # weighted average immune-content of topics this patient's tumor cells belong to
    numer = (w_tumor * soft_topic_immune_frac[None, :]).sum(axis=1)
    denom = w_tumor.sum(axis=1)
    per_cell_exposure = numer / np.maximum(denom, 1e-12)
    raw = per_cell_exposure.mean()
    enr = (per_cell_exposure / max(baseline_immune[p], 1e-6)).mean()
    raw_soft.append(raw)
    enrich_soft.append(enr)
    results_rows[p]["slda_soft_raw_mixing"] = raw
    results_rows[p]["slda_soft_enrichment"] = enr
raw_soft = np.array(raw_soft)
enrich_soft = np.array(enrich_soft)

print(f"\n=== Soft (topic-weighted) global-pooled immune-fraction stat ===")
report("RAW", raw_soft)
report("ENRICH", enrich_soft)

out = pd.DataFrame.from_dict(results_rows, orient="index")
out.index.name = "patient"
out = out.reset_index()
out.to_csv(os.path.join(OUT_DIR, "slda_per_patient_stats.csv"), index=False)
print("\nSaved per-patient Spatial-LDA stats CSV.")
