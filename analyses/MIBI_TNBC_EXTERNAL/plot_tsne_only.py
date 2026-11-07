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

n_comp, k = 10, 10
decomposed = decomposed_full[:, :n_comp]
gmm = GaussianMixture(n_components=k, random_state=42)
clusters = gmm.fit_predict(decomposed)
X = np.array([
    [(clusters[sample_arr == p] == c).mean() for c in range(k)]
    for p in patients
])

colors = {"mixed": "#d62728", "compartmentalized": "#1f77b4"}

tsne = TSNE(n_components=2, perplexity=8, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
for cls in ["mixed", "compartmentalized"]:
    mask = label_names == cls
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[cls], label=cls, s=90, edgecolors='white', linewidths=0.7)
for i, p in enumerate(patients):
    ax.annotate(p.replace("patient", "p"), (X_tsne[i, 0], X_tsne[i, 1]), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords='offset points')
ax.set_title("t-SNE: per-patient TCN composition (n_comp=10, k=10)\ncolored by Keren et al. mixed/compartmentalized label")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend()
plt.tight_layout()
plt.savefig("mibi_tsne_only.png")
print("saved mibi_tsne_only.png")
