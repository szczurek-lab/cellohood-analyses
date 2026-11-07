import os
import matplotlib
matplotlib.use("Agg")
os.chdir("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL")

import cellohood
import joblib

DISTANCE_THRESHOLD = 70.0
smoothed = cellohood.load_cello_prediction(f"MIBI_CP_{DISTANCE_THRESHOLD}")

print("Running get_latent_directions (principled multi-resolution cluster selection)...", flush=True)
ld_result = cellohood.get_latent_directions(
    df=smoothed.bag_predictions,
    columns=smoothed.cello_columns,
    explained_variance_levels=(0.5, 0.75, 1.0),
    min_nb_of_clusters=5,
    max_nb_of_clusters=30,
    nb_of_tries_per_cluster_nb=5,
    take_every_example_for_training=100,
    clusters_step=1,
)
print("Done. Summary:", flush=True)
cellohood.summarize_latent_direction_result(ld_result)

joblib.dump(ld_result, "mibi_latent_direction_result.pickle")
print("Saved mibi_latent_direction_result.pickle", flush=True)
