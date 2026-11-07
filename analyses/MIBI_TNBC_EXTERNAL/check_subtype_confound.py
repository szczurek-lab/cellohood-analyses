import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")
import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

DISTANCE_THRESHOLD = 70.0
df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")
emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposed_full = PCA().fit_transform(emb)
n_comp, k = 10, 10
clusters = GaussianMixture(n_components=k, random_state=42).fit_predict(decomposed_full[:, :n_comp])

cell_type_arr = df["cell_type"].values
sample_arr = df["sample"].values
TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
patients_all = sorted(set(sample_arr))
patients = [p for p in patients_all if (is_tumor & (sample_arr == p)).sum() >= 5]
y = np.array([labels_df[p] for p in patients])

for ct in ["Treg", "CD8T", "DC", "CD11c-high"]:
    is_ct = cell_type_arr == ct
    baseline_ct = np.array([is_ct[sample_arr == p].mean() for p in patients])
    auc_baseline = roc_auc_score(y, baseline_ct)

    cluster_frac_ct = {c: is_ct[clusters == c].mean() for c in range(k)}
    neighborhood_scores = []
    for p in patients:
        mask = (sample_arr == p) & is_tumor
        tumor_clusters = clusters[mask]
        neighborhood_scores.append(np.mean([cluster_frac_ct[c] for c in tumor_clusters]))
    neighborhood_scores = np.array(neighborhood_scores)
    auc_neighborhood = roc_auc_score(y, neighborhood_scores)
    r, _ = pearsonr(neighborhood_scores, baseline_ct)

    print(f"{ct}: baseline-alone AUC={auc_baseline:.3f} (best-dir={max(auc_baseline,1-auc_baseline):.3f}) | "
          f"neighborhood-enrichment AUC={auc_neighborhood:.3f} (best-dir={max(auc_neighborhood,1-auc_neighborhood):.3f}) | "
          f"corr(neighborhood, baseline)={r:.3f}")
