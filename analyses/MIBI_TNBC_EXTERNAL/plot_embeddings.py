import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import umap

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
label_names = np.where(y == 1, "mixed", "compartmentalized")

def build_composition_features(n_comp, k):
    decomposed = decomposed_full[:, :n_comp]
    gmm = GaussianMixture(n_components=k, random_state=42)
    clusters = gmm.fit_predict(decomposed)
    features = []
    for p in patients:
        mask = sample_arr == p
        p_clusters = clusters[mask]
        composition = np.array([(p_clusters == c).mean() for c in range(k)])
        features.append(composition)
    return np.array(features)

X_best = build_composition_features(10, 10)   # best LOO-CV config from before

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

colors = {"mixed": "#d62728", "compartmentalized": "#1f77b4"}

tsne = TSNE(n_components=2, perplexity=8, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_best)
for cls in ["mixed", "compartmentalized"]:
    mask = label_names == cls
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[cls], label=cls, s=60, edgecolors='white', linewidths=0.5)
axes[0].set_title("t-SNE (per-patient TCN composition, n_comp=10, k=10)")
axes[0].legend()

reducer = umap.UMAP(n_components=2, n_neighbors=8, min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_best)
for cls in ["mixed", "compartmentalized"]:
    mask = label_names == cls
    axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[cls], label=cls, s=60, edgecolors='white', linewidths=0.5)
axes[1].set_title("UMAP (per-patient TCN composition, n_comp=10, k=10)")
axes[1].legend()

plt.tight_layout()
plt.savefig("mibi_patient_embedding.png")
print("saved mibi_patient_embedding.png")
