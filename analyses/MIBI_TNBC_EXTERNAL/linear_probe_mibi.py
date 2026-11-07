import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DISTANCE_THRESHOLD = 70.0

df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")

emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)

sample_arr = df["sample"].values
patients = sorted(set(sample_arr))
y = np.array([labels_df[p] for p in patients])

def eval_config(n_comp, k, feature_mode):
    decomposed = decomposed_full[:, :n_comp]
    gmm = GaussianMixture(n_components=k, random_state=42)
    clusters = gmm.fit_predict(decomposed)

    features = []
    for p in patients:
        mask = sample_arr == p
        p_clusters = clusters[mask]
        composition = np.array([(p_clusters == c).mean() for c in range(k)])
        features.append(composition)
    X = np.array(features)
    if feature_mode == "raw":
        Xs = X
    else:
        Xs = StandardScaler().fit_transform(X)

    loo = LeaveOneOut()
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000)
    proba = cross_val_predict(clf, Xs, y, cv=loo, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, proba)
    return auc, X

print("=== Linear probe: per-patient cluster-composition histogram -> logistic regression, Leave-One-Out CV ===")
for n_comp, k in [(2, 8), (5, 8), (5, 12), (10, 10), (5, 2), (10, 2)]:
    auc, X = eval_config(n_comp, k, "std")
    print(f"n_comp={n_comp}, k={k}: LOO-CV AUCROC={auc:.3f}")

print()
print("Reference: CytoCommunity supervised = 0.891, SPACE-GM supervised = 0.823")
print("Reference: Cellohood unsupervised heuristic (from before) = 0.67-0.72")
