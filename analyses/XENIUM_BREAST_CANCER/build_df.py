import os
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/XENIUM_BREAST_CANCER")
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

f = h5py.File('matrix.h5', 'r')
names = np.array([x.decode() for x in f['matrix/features/name'][:]])
ftypes = np.array([x.decode() for x in f['matrix/features/feature_type'][:]])
barcodes = np.array([x.decode() for x in f['matrix/barcodes'][:]])
shape = f['matrix/shape'][:]  # [n_features, n_cells]
data = f['matrix/data'][:]
indices = f['matrix/indices'][:]
indptr = f['matrix/indptr'][:]
mat = sp.csc_matrix((data, indices, indptr), shape=tuple(shape))  # features x cells
mat = mat.T.tocsr()  # cells x features
print("full matrix shape (cells x features):", mat.shape)

gene_mask = ftypes == 'Gene Expression'
print("n genes:", gene_mask.sum(), "n dropped (control probes):", (~gene_mask).sum())
gene_names = names[gene_mask]
mat_genes = mat[:, gene_mask]

expr_df = pd.DataFrame(mat_genes.toarray(), columns=[f"g_{g}" for g in gene_names])
expr_df['cell_id'] = barcodes.astype(int)

cells = pd.read_csv('cells.csv.gz')
cells = cells.rename(columns={'cell_id': 'cell_id'})
print("cells shape:", cells.shape)

df = expr_df.merge(cells, on='cell_id', how='inner')
print("merged shape:", df.shape)
df['sample'] = 'XeniumBreastRep1'

# basic QC: drop cells with very low total gene counts (keep consistent with transcript_counts already in file)
before = len(df)
df = df[df['transcript_counts'] >= 10].copy()
print(f"QC filter (transcript_counts>=10): {before} -> {len(df)}")

marker_cols = [c for c in df.columns if c.startswith('g_')]
print("n marker cols:", len(marker_cols))

df.to_csv('xenium_full.csv', index=False)
with open('marker_columns.txt', 'w') as fh:
    fh.write("\n".join(marker_cols))
print("saved xenium_full.csv and marker_columns.txt")
