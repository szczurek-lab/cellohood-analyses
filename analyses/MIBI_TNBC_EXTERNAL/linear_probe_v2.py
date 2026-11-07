import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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
patients = [p for p in patients_all if ((sample_arr == p) & is_tumor).sum() >= 5 and ((sample_arr == p) & is_immune).sum() >= 5]
print(f"Using {len(patients)}/{len(patients_all)} patients with >=5 tumor and >=5 immune cells")
y = np.array([labels_df[p] for p in patients])

def loo_auc(X):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000)
    proba = cross_val_predict(clf, Xs, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
    return roc_auc_score(y, proba)

results = {}

# Feature set A: mean tumor-cell embedding, top-2/3 PCA components (low-dim, continuous)
for n_comp in [1, 2, 3]:
    feat = np.array([decomposed_full[(sample_arr == p) & is_tumor, :n_comp].mean(axis=0) if ((sample_arr == p) & is_tumor).sum() > 0 else np.zeros(n_comp) for p in patients])
    results[f"mean_tumor_emb_pca{n_comp}"] = loo_auc(feat)

# Feature set B: mean immune-cell embedding, top-2/3 PCA components
for n_comp in [1, 2, 3]:
    feat = np.array([decomposed_full[(sample_arr == p) & is_immune, :n_comp].mean(axis=0) for p in patients])
    results[f"mean_immune_emb_pca{n_comp}"] = loo_auc(feat)

# Feature set C: mean tumor embedding minus mean immune embedding (a "distance between compartments" feature), top components
for n_comp in [1, 2, 3]:
    feat = []
    for p in patients:
        t = decomposed_full[(sample_arr == p) & is_tumor, :n_comp].mean(axis=0)
        i = decomposed_full[(sample_arr == p) & is_immune, :n_comp].mean(axis=0)
        feat.append(t - i)
    results[f"tumor_minus_immune_pca{n_comp}"] = loo_auc(np.array(feat))

# Feature set D: the original unsupervised heuristic score alone, fed through logistic regression (sanity check, should match raw AUCROC closely)
heuristic = pd.read_csv("mibi_mixing_scores_v2.csv").set_index("sample")["mixing_score"] if os.path.exists("mibi_mixing_scores_v2.csv") else None
if heuristic is not None:
    feat = np.array([[heuristic[p]] for p in patients if p in heuristic.index])
    y_h = np.array([labels_df[p] for p in patients if p in heuristic.index])
    Xs = StandardScaler().fit_transform(feat)
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000)
    proba = cross_val_predict(clf, Xs, y_h, cv=LeaveOneOut(), method='predict_proba')[:, 1]
    results["heuristic_via_logreg"] = roc_auc_score(y_h, proba)

for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{k}: LOO-CV AUCROC={v:.3f}")

print()
print("Reference: Cellohood unsupervised heuristic = 0.67-0.72")
print("Reference: CytoCommunity supervised, reproduced = 0.783")
print("Reference: CytoCommunity supervised, as published = 0.891")
