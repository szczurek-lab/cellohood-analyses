import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/DLPFC_ABLATION")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

THRESHOLD = 5.0
DEFAULT_HP = (128, 256, 64, 4)  # ls, ils, lat, nh

bag_cello_df = cellohood.load_bag_cello_df(f"DLPFC_CB_{THRESHOLD}")
n_cells = bag_cello_df.cell_df.shape[0]
print(f"n_cells = {n_cells}")

def get_embedding(threshold, run, hp):
    ls, ils, lat, nh = hp
    cp_path = f"DLPFC_CP_{threshold}_{run}_{ls}_{ils}_{lat}_{nh}"
    pred = cellohood.load_cello_prediction(cp_path)
    emb = pred.cell_predictions[pred.cello_columns].values
    return emb

def cluster(emb, n_comp, k, seed):
    decomposer = PCA()
    dec_full = decomposer.fit_transform(emb)
    dec = dec_full[:, :n_comp]
    gmm = GaussianMixture(n_components=k, random_state=seed)
    return gmm.fit_predict(dec), dec_full, decomposer

# shared base embedding for seed + bootstrap stability
base_emb = get_embedding(THRESHOLD, 0, DEFAULT_HP)
dec_full_base = PCA().fit_transform(base_emb)

K = 7  # matches the 7-cluster convention used elsewhere (NB_OF_CLUSTERS in DLPFC scripts)
NCOMP_STANDARD = 8  # approx matches observed nb_of_standard_components at 75% var at this threshold; will reconfirm below
variance_ratios = PCA().fit(base_emb).explained_variance_ratio_.cumsum()
ncomp_standard = max(int(np.abs(variance_ratios - 0.75).argmin()), 2)
ncomp_coarse = max(int(np.abs(variance_ratios - 0.50).argmin()), 1)
print(f"ncomp_standard(75% var)={ncomp_standard}, ncomp_coarse(50% var)={ncomp_coarse}")

results = {}

# ============================================================
# Angle 1: seed stability (same embedding, 5 GMM seeds)
# ============================================================
print("\n=== Angle 1: GMM seed stability (fixed embedding, threshold=5.0, default hp) ===")
seed_labels = []
for seed in range(5):
    dec = dec_full_base[:, :ncomp_standard]
    gmm = GaussianMixture(n_components=K, random_state=42 + seed)
    labels = gmm.fit_predict(dec)
    seed_labels.append(labels)

seed_aris, seed_nmis = [], []
for i in range(5):
    for j in range(i+1, 5):
        a = ARI(seed_labels[i], seed_labels[j])
        n = NMI(seed_labels[i], seed_labels[j])
        seed_aris.append(a); seed_nmis.append(n)
print(f"pairwise ARI: mean={np.mean(seed_aris):.3f} std={np.std(seed_aris):.3f} (n_pairs={len(seed_aris)})")
print(f"pairwise NMI: mean={np.mean(seed_nmis):.3f} std={np.std(seed_nmis):.3f}")
results['seed_stability'] = {'ari_mean': np.mean(seed_aris), 'ari_std': np.std(seed_aris),
                               'nmi_mean': np.mean(seed_nmis), 'nmi_std': np.std(seed_nmis)}

# ============================================================
# Angle 2: bootstrap/subsample stability
# ============================================================
print("\n=== Angle 2: subsample stability (80% subsample x5, fixed embedding/seed) ===")
rng = np.random.RandomState(0)
full_dec = dec_full_base[:, :ncomp_standard]
full_labels = GaussianMixture(n_components=K, random_state=42).fit_predict(full_dec)

sub_aris, sub_nmis = [], []
for trial in range(5):
    idx = rng.choice(n_cells, size=int(0.8 * n_cells), replace=False)
    idx.sort()
    sub_dec = full_dec[idx]
    gmm = GaussianMixture(n_components=K, random_state=42 + trial)
    sub_labels = gmm.fit_predict(sub_dec)
    ref_labels = full_labels[idx]
    a = ARI(ref_labels, sub_labels)
    n = NMI(ref_labels, sub_labels)
    sub_aris.append(a); sub_nmis.append(n)
print(f"subsample vs full-data (restricted) ARI: mean={np.mean(sub_aris):.3f} std={np.std(sub_aris):.3f}")
print(f"subsample vs full-data (restricted) NMI: mean={np.mean(sub_nmis):.3f} std={np.std(sub_nmis):.3f}")
results['subsample_stability'] = {'ari_mean': np.mean(sub_aris), 'ari_std': np.std(sub_aris),
                                    'nmi_mean': np.mean(sub_nmis), 'nmi_std': np.std(sub_nmis)}

# ============================================================
# Angle 3: hyperparameter-config stability (different models, same cells, same k/seed)
# ============================================================
print("\n=== Angle 3: hyperparameter-config stability (5 diverse configs, threshold=5.0, run=0) ===")
hp_configs = [
    (128, 256, 64, 4),   # default
    (64, 128, 32, 4),    # small
    (256, 1024, 256, 8), # large
    (128, 512, 128, 8),  # mid
    (64, 256, 32, 8),    # small/wide-heads
]
hp_labels_list = []
hp_names = []
for hp in hp_configs:
    try:
        emb = get_embedding(THRESHOLD, 0, hp)
    except Exception as e:
        print(f"  skip {hp}: {e}")
        continue
    dec_full = PCA().fit_transform(emb)
    vr = PCA().fit(emb).explained_variance_ratio_.cumsum()
    nc = max(int(np.abs(vr - 0.75).argmin()), 2)
    labels = GaussianMixture(n_components=K, random_state=42).fit_predict(dec_full[:, :nc])
    hp_labels_list.append(labels)
    hp_names.append("_".join(map(str, hp)))

hp_aris, hp_nmis = [], []
for i in range(len(hp_labels_list)):
    for j in range(i+1, len(hp_labels_list)):
        a = ARI(hp_labels_list[i], hp_labels_list[j])
        n = NMI(hp_labels_list[i], hp_labels_list[j])
        hp_aris.append(a); hp_nmis.append(n)
        print(f"  {hp_names[i]} vs {hp_names[j]}: ARI={a:.3f} NMI={n:.3f}")
print(f"pairwise ARI across configs: mean={np.mean(hp_aris):.3f} std={np.std(hp_aris):.3f}")
print(f"pairwise NMI across configs: mean={np.mean(hp_nmis):.3f} std={np.std(hp_nmis):.3f}")
results['hp_config_stability'] = {'ari_mean': np.mean(hp_aris), 'ari_std': np.std(hp_aris),
                                    'nmi_mean': np.mean(hp_nmis), 'nmi_std': np.std(hp_nmis),
                                    'n_configs': len(hp_labels_list)}

# ============================================================
# Angle 4: resolution stability (coarse vs standard -- is coarse a clean coarsening of standard?)
# ============================================================
print("\n=== Angle 4: resolution stability (coarse 50%var vs standard 75%var, same embedding) ===")
K_STANDARD = 7
K_COARSE = 4  # matches convention seen elsewhere for coarser resolution
dec_standard = dec_full_base[:, :ncomp_standard]
dec_coarse = dec_full_base[:, :ncomp_coarse]
labels_standard = GaussianMixture(n_components=K_STANDARD, random_state=42).fit_predict(dec_standard)
labels_coarse = GaussianMixture(n_components=K_COARSE, random_state=42).fit_predict(dec_coarse)

# crosstab purity: for each coarse cluster, what fraction of its cells share the single most common standard cluster?
ct = pd.crosstab(labels_coarse, labels_standard)
purities = ct.max(axis=1) / ct.sum(axis=1)
print("Coarse-cluster -> standard-cluster crosstab:")
print(ct)
print(f"Per-coarse-cluster purity (fraction mapping to single dominant standard cluster): {purities.values.round(3)}")
print(f"Mean purity = {purities.mean():.3f}")
nmi_res = NMI(labels_coarse, labels_standard)
print(f"NMI(coarse, standard) = {nmi_res:.3f}")
results['resolution_stability'] = {'mean_purity': purities.mean(), 'nmi': nmi_res,
                                     'purities': purities.values.tolist()}

import json
with open("/raid/immucan/immuw/cellohood-analyses/analyses/ABLATION_SUMMARY/cluster_stability_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved results JSON.")
