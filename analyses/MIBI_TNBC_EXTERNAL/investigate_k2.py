import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

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

for n_comp in [5, 10]:
    decomposed = decomposed_full[:, :n_comp]
    gmm = GaussianMixture(n_components=2, random_state=42)
    clusters = gmm.fit_predict(decomposed)

    frac_cluster0 = []
    for p in patients:
        mask = sample_arr == p
        frac_cluster0.append((clusters[mask] == 0).mean())
    frac_cluster0 = np.array(frac_cluster0)

    auc_raw = roc_auc_score(y, frac_cluster0)
    print(f"n_comp={n_comp}: AUCROC(frac_cluster0 predicts label=1)={auc_raw:.3f}")
    order = np.argsort(frac_cluster0)
    print("sorted (patient, frac_cluster0, label):")
    for i in order:
        print(f"  {patients[i]:12s} frac0={frac_cluster0[i]:.4f}  label={y[i]}")
    print()
