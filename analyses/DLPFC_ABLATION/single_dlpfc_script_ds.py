import os
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from ccs import ccs
import cellohood

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


def main(distance_threshold: float, run_nb: int):

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ZENODO_PATH = "/raid/immucan/immuw/cellohood_artifacts/DLPFC"
    DATA_PATH = os.path.join(ZENODO_PATH, "dlpfc_scvi.csv")

    data_df = pd.read_csv(DATA_PATH)

    MARKER_COL_NAMES = [
        "SCVI5:0",
        "SCVI5:1",
        "SCVI5:2",
        "SCVI5:3",
        "SCVI5:4",
    ]

    POS_COL_NAMES = ["pos_x", "pos_y"]
    IMAGE_ID_COLUMN = "sample"
    SAMPLE_COL_NAME = "sample"

    FULL_CELLO_BAG_DF_PATH = f"DLPFC_CB_{distance_threshold}"
    LOAD_CELLO_BAG_DF = os.path.exists(FULL_CELLO_BAG_DF_PATH)

    CELLO_PREDICTION_PATH = f"DLPFC_CP_DS_{distance_threshold}_{run_nb}"
    LOAD_CELLO_PREDICTION = os.path.exists(CELLO_PREDICTION_PATH)

    SAVING_PATH = f"DLPFC_CELLOHOOD_MODEL_DS_{distance_threshold}_{run_nb}"
    TRAIN_MODEL = not os.path.exists(SAVING_PATH)


    # -------------------------
    # Load / compute bag cello
    # -------------------------
    if not LOAD_CELLO_BAG_DF:
        bag_cello_df = cellohood.transform_df_to_bag_cello_df(
            df_=data_df,
            image_id_column=IMAGE_ID_COLUMN,
            pos_col_names=POS_COL_NAMES,
            distance_threshold=distance_threshold,
            marker_col_names=MARKER_COL_NAMES,
        )

        cellohood.save_bag_cello_df(
            bag_cello_df=bag_cello_df,
            path=FULL_CELLO_BAG_DF_PATH,
            exist_ok=True,
        )

    else:
        bag_cello_df = cellohood.load_bag_cello_df(
            path=FULL_CELLO_BAG_DF_PATH
        )

    # -------------------------
    # Split + standardize
    # -------------------------
    train_bag_cello_df, test_bag_cello_df = cellohood.split_bag_cello_df_by(
        bag_cello_df,
        by=SAMPLE_COL_NAME,
        split_percentage=1.0,
    )

    standardized_cello_dfs = cellohood.standardize_bag_cello_df(
        train_bag_cello_df=train_bag_cello_df,
        test_bag_cello_df=test_bag_cello_df,
        marker_col_names=MARKER_COL_NAMES,
    )

    # -------------------------
    # Load / train model
    # -------------------------
    if TRAIN_MODEL:
        cello_train_result = cellohood.train(
            standardized_cello_dfs=standardized_cello_dfs,
            epoch_nb=100,
            use_cuda=True,
            model_type='DeepSet',
        )

        cellohood.save_cello_train_result(
            cello_train_result,
            SAVING_PATH,
        )

    elif not LOAD_CELLO_PREDICTION:
        cello_train_result = cellohood.load_cello_train_result(
            SAVING_PATH
        )

    # -------------------------
    # Prediction
    # -------------------------
    if not LOAD_CELLO_PREDICTION:

        cello_prediction = cellohood.run_prediction_on_cello_df(
            cello_train_result,
            bag_cello_df,
        )

        smoothed_df_result = cellohood.smooth_cellohood_prediction(
            cello_prediction,
            distance_threshold=distance_threshold,
            cache_path=f'new_cache_ds_{distance_threshold}_run_{run_nb}',
        )

        cellohood.save_cello_prediction(
            cello_prediction=smoothed_df_result,
            path=CELLO_PREDICTION_PATH,
        )

    else:
        smoothed_df_result = cellohood.load_cello_prediction(
            path=CELLO_PREDICTION_PATH
        )

    # -------------------------
    # PCA
    # -------------------------
    decomposer = PCA(whiten=True)

    plda_decomposed_full = decomposer.fit_transform(
        smoothed_df_result.cell_predictions[
            smoothed_df_result.cello_columns
            ].values
    )
    variance_ratios = decomposer.explained_variance_ratio_.cumsum()
    nb_of_standard_components = np.abs(variance_ratios - 0.75).argmin()
    print(f'Nb of standard components: {nb_of_standard_components}')
    plda_decomposed = plda_decomposed_full[:, :nb_of_standard_components]

    layers_ = sorted(list(set(bag_cello_df.cell_df['layer'])))
    fig = plt.figure(figsize=(3, 3), dpi=300)
    plt.subplot(2, 2, 1)
    plt.scatter(
            plda_decomposed_full[:, 0],
            plda_decomposed_full[:, 1],
            c=bag_cello_df.cell_df['layer'].apply(lambda x: layers_.index(x)),
            alpha=0.5,
            s=1,
    )
    plt.subplot(2, 2, 2)
    plt.scatter(
            plda_decomposed_full[:, 0],
            plda_decomposed_full[:, 2],
            c=bag_cello_df.cell_df['layer'].apply(lambda x: layers_.index(x)),
            alpha=0.5,
            s=1,
    )
    plt.subplot(2, 2, 3)
    plt.scatter(
            plda_decomposed_full[:, 1],
            plda_decomposed_full[:, 2],
            c=bag_cello_df.cell_df['layer'].apply(lambda x: layers_.index(x)),
            alpha=0.5,
            s=1,
    )
    plt.subplot(2, 2, 4)
    plt.title(nb_of_standard_components)
    plt.scatter(
            plda_decomposed_full[:, 0],
            plda_decomposed_full[:, 3],
            c=bag_cello_df.cell_df['layer'].apply(lambda x: layers_.index(x)),
            alpha=0.5,
            s=1,
    )
    plt.savefig(f'plot_{distance_threshold}_ds_run_{run_nb}.png')

    # -------------------------
    # ARI computation
    # -------------------------
    layers_ = sorted(list(set(bag_cello_df.cell_df["layer"])))
    samples_ = list(set(bag_cello_df.cell_df["sample"]))

    sample_to_aris = defaultdict(list)
    clusterers = []

    TEST_SAMPLES = [
            151507,
            151672,
            151673
            ]

    NON_TEST_SAMPLES = [
            sample 
            for sample in sorted(bag_cello_df.cell_df['sample'].unique())
            if sample not in TEST_SAMPLES
            ]

    variables = list(range(nb_of_standard_components))

    training_ld = plda_decomposed_full[
            bag_cello_df.cell_df['sample'].apply(lambda x: x in TEST_SAMPLES),
            ][:, variables] 

    for k in range(10):

        gmm = GaussianMixture(n_components=7, random_state=42 + k)
        gmm.fit(training_ld)
        predictions = gmm.predict(plda_decomposed)   

        clusterers.append(gmm)

        for sample in samples_:

            sample_mask = bag_cello_df.cell_df["sample"] == sample
            sample_cell_df = bag_cello_df.cell_df[sample_mask]

            true = bag_cello_df.cell_df.loc[sample_mask, "layer"]
            pred = predictions[sample_mask]

            ari = adjusted_rand_score(true[sample_cell_df['layer'] != 'UNK'], pred[sample_cell_df['layer'] != 'UNK'])
            sample_to_aris[sample].append(ari)

    sample_aris_df = pd.DataFrame(sample_to_aris)
    sample_aris_df['nb_of_standard_components'] = nb_of_standard_components
    print(f'Nb of standard components: {nb_of_standard_components}, test ARI: {sample_aris_df[TEST_SAMPLES].median()}')

    # -------------------------
    # Save result
    # -------------------------
    output_path = f"sample_aris_df_threshold_ds_{distance_threshold}_run_{run_nb}.csv"
    sample_aris_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Distance threshold used in cellohood",
    )

    parser.add_argument(
        "--run_nb",
        type=int,
        required=True,
        help="Run identifier",
    )

    args = parser.parse_args()

    main(args.threshold, args.run_nb)
