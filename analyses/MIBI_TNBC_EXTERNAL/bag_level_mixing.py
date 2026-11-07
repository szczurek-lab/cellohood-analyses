import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

DISTANCE_THRESHOLD = 70.0
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
bag_cello_df = cellohood.load_bag_cello_df(f"MIBI_CB_{DISTANCE_THRESHOLD}")
cell_df = bag_cello_df.cell_df

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}
is_tumor = cell_df["cell_type"].isin(TUMOR_TYPES).values
is_immune = cell_df["cell_type"].isin(IMMUNE_TYPES).values
sample_arr = cell_df["sample"].values
bag_col = bag_cello_df.cellohood_neighborhood_cluster_colname
bag_id = cell_df[bag_col].values

# Per-bag immune fraction, computed WITHIN EACH PATIENT'S OWN IMAGE (not pooled across the cohort) --
# mirrors CytoCommunity's (the less-confounded) approach, using raw spatial bag membership instead
# of a learned/pooled cluster.
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
print(f"Using {len(patients)}/{len(patients_all)} patients with >=5 tumor cells")

y = np.array([labels_df[p] for p in patients])
baseline_immune = np.array([is_immune[sample_arr == p].mean() for p in patients])

direct_bag_mixing = []
direct_bag_mixing_enriched = []
for p in patients:
    p_mask = sample_arr == p
    p_bag_id = bag_id[p_mask]
    p_is_tumor = is_tumor[p_mask]
    p_is_immune = is_immune[p_mask]

    # per-bag immune fraction, WITHIN this patient only
    bag_immune_frac = pd.Series(p_is_immune).groupby(p_bag_id).mean().to_dict()
    tumor_bags = p_bag_id[p_is_tumor]
    exposures = np.array([bag_immune_frac[b] for b in tumor_bags])
    stat = exposures.mean()
    direct_bag_mixing.append(stat)

    overall = p_is_immune.mean()
    enriched = (exposures / overall).mean() if overall > 0 else np.nan
    direct_bag_mixing_enriched.append(enriched)

direct_bag_mixing = np.array(direct_bag_mixing)
direct_bag_mixing_enriched = np.array(direct_bag_mixing_enriched)

for name, stat in [("raw bag-level local immune exposure", direct_bag_mixing),
                     ("enrichment-corrected bag-level (within-patient)", direct_bag_mixing_enriched)]:
    auc = roc_auc_score(y, stat)
    r_p, _ = pearsonr(stat, baseline_immune)
    r_s, _ = spearmanr(stat, baseline_immune)
    print(f"{name}: AUCROC={auc:.3f} (flipped={1-auc:.3f}) | corr with baseline: Pearson={r_p:.3f} Spearman={r_s:.3f}")

print()
# Residualize: regress raw stat on baseline_immune, check if residual still predicts label
from numpy.polynomial import polynomial as P
coef = np.polyfit(baseline_immune, direct_bag_mixing, 1)
pred = np.polyval(coef, baseline_immune)
resid = direct_bag_mixing - pred
auc_resid = roc_auc_score(y, resid)
print(f"Residual (raw bag-mixing after regressing out baseline_immune) AUCROC={auc_resid:.3f} (flipped={1-auc_resid:.3f})")
