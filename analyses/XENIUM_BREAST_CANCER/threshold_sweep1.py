import pandas as pd
import numpy as np
import cellohood

df = pd.read_csv("/raid/immucan/immuw/cellohood-analyses/analyses/XENIUM_BREAST_CANCER/xenium_full.csv")
with open("/raid/immucan/immuw/cellohood-analyses/analyses/XENIUM_BREAST_CANCER/marker_columns.txt") as f:
    markers = [l.strip() for l in f if l.strip()]

thresholds = [10, 20, 30, 40, 60, 80, 100, 130, 160, 200]
result = cellohood.select_bag_distance_threshold(
    df_=df,
    image_id_column="sample",
    pos_col_names=["x_centroid", "y_centroid"],
    thresholds=thresholds,
    marker_col_names=markers,
)
print("selected threshold:", result.selected_threshold)
for t in thresholds:
    sizes = result.threshold_to_sizes[t]
    print(f"threshold={t}: n_bags={len(sizes)}, mean_size={np.mean(sizes):.2f}")
