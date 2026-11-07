import os, sys, time, pickle
import numpy as np
import pandas as pd

import spatial_lda.featurization as feat
import spatial_lda.model as spmodel

INPUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/cytocommunity_repro/CytoCommunity/Tutorial/Supervised/MIBI-TNBC_Input"
OUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/slda_out"
os.makedirs(OUT_DIR, exist_ok=True)

RADIUS = 70.0
N_TOPICS = 10
N_PARALLEL = 16

patients = [l.strip() for l in open(os.path.join(INPUT_DIR, "ImageNameList.txt")) if l.strip()]

sample_dfs = {}
label_all = {}
for p in patients:
    ct = pd.read_csv(os.path.join(INPUT_DIR, f"{p}_CellTypeLabel.txt"), header=None, names=["cell_type"])["cell_type"].values
    xy = pd.read_csv(os.path.join(INPUT_DIR, f"{p}_Coordinates.txt"), header=None, sep="\t", names=["x", "y"]).values
    lab = int(open(os.path.join(INPUT_DIR, f"{p}_GraphLabel.txt")).read().strip())
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": ct})
    df["is_anchor"] = True
    sample_dfs[p] = df
    label_all[p] = lab

print(f"Loaded {len(sample_dfs)} patients, total cells = {sum(len(d) for d in sample_dfs.values())}")

t0 = time.time()
sample_features = feat.featurize_samples(
    sample_dfs,
    feat.neighborhood_to_cluster,
    radius=RADIUS,
    is_anchor_col="is_anchor",
    x_col="x",
    y_col="y",
    include_anchors=True,
    n_processes=N_PARALLEL,
)
print(f"Featurization done in {time.time()-t0:.1f}s, shape={sample_features.shape}")
print(sample_features.head())

sample_features.to_pickle(os.path.join(OUT_DIR, "sample_features.pkl"))

t0 = time.time()
difference_matrices = feat.make_merged_difference_matrices(
    sample_features, sample_dfs, x_col="x", y_col="y"
)
print(f"Difference matrices done in {time.time()-t0:.1f}s")

with open(os.path.join(OUT_DIR, "difference_matrices.pkl"), "wb") as f:
    pickle.dump(difference_matrices, f)

t0 = time.time()
lda = spmodel.train(
    sample_features,
    difference_matrices,
    n_topics=N_TOPICS,
    n_iters=3,
    n_parallel_processes=N_PARALLEL,
    verbosity=1,
)
print(f"Training done in {time.time()-t0:.1f}s")

with open(os.path.join(OUT_DIR, "slda_model.pkl"), "wb") as f:
    pickle.dump(lda, f)

lda.topic_weights.to_pickle(os.path.join(OUT_DIR, "topic_weights.pkl"))
print("Saved topic_weights, shape:", lda.topic_weights.shape)
print(lda.topic_weights.head())
print("Done.")
