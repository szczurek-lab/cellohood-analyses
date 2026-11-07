import os, sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

INPUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/cytocommunity_repro/CytoCommunity/Tutorial/Supervised/MIBI-TNBC_Input"
OUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/utag_out"
os.makedirs(OUT_DIR, exist_ok=True)

TUMOR_TYPES = {"Tumor/EGFR", "Tumor/Keratin"}
IMMUNE_TYPES = {
    "B", "CD4T", "CD8T", "DC", "MF_1", "MF_2", "MF/Glia", "NK",
    "Neutrophil", "Treg", "CD11c-high",
}

patients = [l.strip() for l in open(os.path.join(INPUT_DIR, "ImageNameList.txt")) if l.strip()]

cell_types_all = []
coords_all = []
sample_all = []
label_all = {}

for p in patients:
    ct = pd.read_csv(os.path.join(INPUT_DIR, f"{p}_CellTypeLabel.txt"), header=None, names=["cell_type"])["cell_type"].values
    xy = pd.read_csv(os.path.join(INPUT_DIR, f"{p}_Coordinates.txt"), header=None, sep="\t", names=["x", "y"]).values
    lab = int(open(os.path.join(INPUT_DIR, f"{p}_GraphLabel.txt")).read().strip())
    assert len(ct) == len(xy), f"{p}: mismatch {len(ct)} vs {len(xy)}"
    cell_types_all.append(ct)
    coords_all.append(xy)
    sample_all.append(np.full(len(ct), p))
    label_all[p] = lab

cell_types_all = np.concatenate(cell_types_all)
coords_all = np.concatenate(coords_all, axis=0)
sample_all = np.concatenate(sample_all)

all_types = sorted(set(cell_types_all))
print(f"n_cells={len(cell_types_all)}, n_patients={len(patients)}, n_cell_types={len(all_types)}")
print("cell types:", all_types)

# One-hot feature matrix
onehot = pd.get_dummies(pd.Categorical(cell_types_all, categories=all_types)).values.astype(np.float32)

obs = pd.DataFrame({
    "Slide": pd.Categorical(sample_all),
    "cell_type": pd.Categorical(cell_types_all, categories=all_types),
})
obs["graph_label"] = obs["Slide"].map(label_all).astype(int)
obs["is_tumor"] = obs["cell_type"].isin(TUMOR_TYPES)
obs["is_immune"] = obs["cell_type"].isin(IMMUNE_TYPES)

adata = ad.AnnData(X=onehot, obs=obs)
adata.var_names = all_types
adata.obsm["spatial"] = coords_all.astype(np.float32)

adata.write_h5ad(os.path.join(OUT_DIR, "mibi_tnbc_onehot.h5ad"))
print("Saved base AnnData:", adata.shape)
print(adata)
