import os
import numpy as np
import anndata as ad
from utag import utag

OUT_DIR = "/tmp/claude-1019/-raid-immucan-immuw-cellohood-notebooks/536ba41e-40a1-456f-bd32-287b07c10dc2/scratchpad/utag_out"
adata = ad.read_h5ad(os.path.join(OUT_DIR, "mibi_tnbc_onehot.h5ad"))
print(adata)

channels = list(adata.var_names)

result = utag(
    adata,
    channels_to_use=channels,
    slide_key="Slide",
    save_key="UTAG",
    max_dist=70.0,
    normalization_mode="l1_norm",
    pca_kwargs={"n_comps": 10},
    apply_clustering=True,
    clustering_method=["leiden"],
    resolutions=[0.3, 1.0],
    parallel=True,
    processes=8,
)
print(result)
result.write_h5ad(os.path.join(OUT_DIR, "mibi_tnbc_utag_result.h5ad"))
print("saved")
for col in result.obs.columns:
    if "UTAG" in col:
        print(col, result.obs[col].value_counts())
