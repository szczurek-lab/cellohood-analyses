import os, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import numpy as np
import pandas as pd
import cellohood
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

DISTANCE_THRESHOLD = 70.0
IMAGE_ID_COLUMN = "sample"
POS_COL_NAMES = ["pos_x", "pos_y"]

with open("cell_type_columns.txt") as f:
    CELL_TYPES = [l.strip() for l in f if l.strip()]

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {"B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK", "Neutrophil", "Treg", "CD11c-high"}

df = pd.read_csv("mibi_tnbc_full.csv")
labels_df = pd.read_csv("mibi_tnbc_labels.csv").set_index("sample")["graph_label"]

FULL_CB_PATH = f"MIBI_CB_{DISTANCE_THRESHOLD}"
SAVING_PATH = f"MIBI_CM_{DISTANCE_THRESHOLD}"
CP_PATH = f"MIBI_CP_{DISTANCE_THRESHOLD}"

print("Building/loading bag_cello_df...", flush=True)
if os.path.exists(FULL_CB_PATH):
    bag_cello_df = cellohood.load_bag_cello_df(FULL_CB_PATH)
else:
    bag_cello_df = cellohood.transform_df_to_bag_cello_df(
        df_=df,
        image_id_column=IMAGE_ID_COLUMN,
        pos_col_names=POS_COL_NAMES,
        distance_threshold=DISTANCE_THRESHOLD,
        marker_col_names=CELL_TYPES,
    )
    cellohood.save_bag_cello_df(bag_cello_df=bag_cello_df, path=FULL_CB_PATH, exist_ok=True)
print("bag_cello_df cells:", bag_cello_df.cell_df.shape[0], flush=True)

train_bag_cello_df, test_bag_cello_df = cellohood.split_bag_cello_df_by(
    bag_cello_df, by=IMAGE_ID_COLUMN, split_percentage=1.0,
)
standardized = cellohood.standardize_bag_cello_df(
    train_bag_cello_df=train_bag_cello_df,
    test_bag_cello_df=test_bag_cello_df,
    marker_col_names=CELL_TYPES,
)

if os.path.exists(SAVING_PATH):
    print("Loading cached model...", flush=True)
    cello_train_result = cellohood.load_cello_train_result(SAVING_PATH)
else:
    print("Training Cellohood (unsupervised, Transformer)...", flush=True)
    cello_train_result = cellohood.train(
        standardized_cello_dfs=standardized,
        epoch_nb=100,
        use_cuda=True,
        log_mem=True,
        log_time=True,
    )
    cellohood.save_cello_train_result(cello_train_result, SAVING_PATH)

if os.path.exists(CP_PATH):
    print("Loading cached prediction...", flush=True)
    smoothed = cellohood.load_cello_prediction(CP_PATH)
else:
    print("Running prediction + smoothing...", flush=True)
    cello_prediction = cellohood.run_prediction_on_cello_df(cello_train_result, bag_cello_df)
    cache_path = "smooth_cache_mibi"
    if os.path.exists(cache_path):
        subprocess.run(["rm", "-r", cache_path])
    smoothed = cellohood.smooth_cellohood_prediction(
        cello_prediction, distance_threshold=DISTANCE_THRESHOLD, cache_path=cache_path,
    )
    cellohood.save_cello_prediction(cello_prediction=smoothed, path=CP_PATH)

print("smoothed cell_predictions shape:", smoothed.cell_predictions.shape, flush=True)
print("bag_cello_df.cell_df shape:", bag_cello_df.cell_df.shape, flush=True)

# Guard against the row-count mismatch bug found in the DLPFC sweep
n_pred = smoothed.cell_predictions.shape[0]
n_cells = bag_cello_df.cell_df.shape[0]
if n_pred != n_cells:
    raise RuntimeError(f"Row mismatch: smoothed predictions have {n_pred} rows, bag_cello_df has {n_cells}. Aborting rather than silently misaligning.")

emb = smoothed.cell_predictions[smoothed.cello_columns].values
decomposer = PCA()
decomposed_full = decomposer.fit_transform(emb)
variance_ratios = decomposer.explained_variance_ratio_.cumsum()
n_comp = max(np.abs(variance_ratios - 0.75).argmin(), 2)
print("n_comp (75% variance):", n_comp, flush=True)
decomposed = decomposed_full[:, :n_comp]

K = 8
gmm = GaussianMixture(n_components=K, random_state=42)
clusters = gmm.fit_predict(decomposed)

cell_type_arr = df["cell_type"].values
is_tumor = np.isin(cell_type_arr, list(TUMOR_TYPES))
is_immune = np.isin(cell_type_arr, list(IMMUNE_TYPES))

cluster_summary = []
for k in range(K):
    mask = clusters == k
    n = mask.sum()
    frac_tumor = is_tumor[mask].mean() if n > 0 else 0
    frac_immune = is_immune[mask].mean() if n > 0 else 0
    cluster_summary.append({"cluster": k, "n": n, "frac_tumor": frac_tumor, "frac_immune": frac_immune})
cluster_summary_df = pd.DataFrame(cluster_summary)
print(cluster_summary_df, flush=True)

pure_tumor_clusters = set(cluster_summary_df[cluster_summary_df.frac_tumor > 0.5]["cluster"])
print("Pure-tumor clusters:", pure_tumor_clusters, flush=True)

sample_arr = df["sample"].values
results = []
for patient in sorted(set(sample_arr)):
    p_mask = sample_arr == patient
    p_tumor_mask = p_mask & is_tumor
    n_tumor = p_tumor_mask.sum()
    if n_tumor == 0:
        continue
    tumor_clusters = clusters[p_tumor_mask]
    frac_tumor_in_pure_tumor_cluster = np.isin(tumor_clusters, list(pure_tumor_clusters)).mean()
    mixing_score = 1.0 - frac_tumor_in_pure_tumor_cluster
    results.append({"sample": patient, "mixing_score": mixing_score, "n_tumor": n_tumor, "graph_label": labels_df[patient]})

res_df = pd.DataFrame(results)
res_df.to_csv("mibi_mixing_scores.csv", index=False)
print(res_df.sort_values("mixing_score"), flush=True)

auc = roc_auc_score(res_df["graph_label"], res_df["mixing_score"])
print(f"\nAUCROC (Cellohood mixing_score vs mixed/compartmentalized label): {auc:.3f}", flush=True)
print("Reference: CytoCommunity supervised = 0.891, SPACE-GM supervised = 0.823", flush=True)
