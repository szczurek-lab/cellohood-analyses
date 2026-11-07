import os
import argparse
import time
import subprocess
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


def main(
        distance_threshold: float,
        run_nb: int,
        layer_size: int = 128,
        intermediate_layer_size: int = 256,
        latent_size: int = 64,
        num_heads: int = 4,
        continue_on_output_existing: bool = True,
    ):
    print(f'DT: {distance_threshold}, RN: {run_nb}, LS: {layer_size}, ILS: {intermediate_layer_size}, Lat S: {latent_size}, NH: {num_heads}')


    output_path = f"benchmark_sample_aris_df_threshold_{distance_threshold}_run_{run_nb}_{layer_size}_{intermediate_layer_size}_{latent_size}_{num_heads}_split_deepset.csv"
    if continue_on_output_existing:
        if os.path.exists(output_path):
            return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    ZENODO_PATH = "/raid/immucan/immuw/cellohood_artifacts/DLPFC"
    DATA_PATH = os.path.join(ZENODO_PATH, "dlpfc_scvi.csv")
    
    data_df = pd.read_csv(DATA_PATH)
    
    TRAIN_SAMPLES = [
            151507,
            151672,
            151673,
    ]

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

    FULL_TRAIN_CELLO_BAG_DF_PATH = f"DLPFC_CB_{distance_threshold}_TRAIN"
    FULL_TEST_CELLO_BAG_DF_PATH = f"DLPFC_CB_{distance_threshold}_TEST"
    FULL_CELLO_BAG_DF_PATH = f"DLPFC_CB_{distance_threshold}"
    LOAD_TRAIN_CELLO_BAG_DF = os.path.exists(FULL_TRAIN_CELLO_BAG_DF_PATH)
    LOAD_TEST_CELLO_BAG_DF = os.path.exists(FULL_TEST_CELLO_BAG_DF_PATH)
    LOAD_CELLO_BAG_DF = os.path.exists(FULL_TEST_CELLO_BAG_DF_PATH)

    CELLO_PREDICTION_PATH = f"DLPFC_CP_SPLIT_DS_{distance_threshold}_{run_nb}_{layer_size}_{intermediate_layer_size}_{latent_size}_{num_heads}"
    LOAD_CELLO_PREDICTION = os.path.exists(CELLO_PREDICTION_PATH)

    SAVING_PATH = f"DLPFC_CELLOHOOD_MODEL_SPLIT_DS_{distance_threshold}_{run_nb}_{layer_size}_{intermediate_layer_size}_{latent_size}_{num_heads}"
    TRAIN_MODEL = not os.path.exists(SAVING_PATH)


    # -------------------------
    # Load / compute bag cello
    # -------------------------
    train_data_df = data_df[data_df[IMAGE_ID_COLUMN].apply(lambda x: x in TRAIN_SAMPLES)]
    test_data_df = data_df[data_df[IMAGE_ID_COLUMN].apply(lambda x: x not in TRAIN_SAMPLES)]
    
    if not LOAD_TRAIN_CELLO_BAG_DF:    
        train_bag_cello_df = cellohood.transform_df_to_bag_cello_df(
            df_=train_data_df,
            image_id_column=IMAGE_ID_COLUMN,
            pos_col_names=POS_COL_NAMES,
            distance_threshold=distance_threshold,
            marker_col_names=MARKER_COL_NAMES,
        )
        
        cellohood.save_bag_cello_df(
            bag_cello_df=train_bag_cello_df,
            path=FULL_TRAIN_CELLO_BAG_DF_PATH,
            exist_ok=True,
        )
    else:
        train_bag_cello_df = cellohood.load_bag_cello_df(
            path=FULL_TRAIN_CELLO_BAG_DF_PATH
        )
        
    if not LOAD_TEST_CELLO_BAG_DF:
        test_bag_cello_df = cellohood.transform_df_to_bag_cello_df(
            df_=test_data_df,
            image_id_column=IMAGE_ID_COLUMN,
            pos_col_names=POS_COL_NAMES,
            distance_threshold=distance_threshold,
            marker_col_names=MARKER_COL_NAMES,
        )
        
        cellohood.save_bag_cello_df(
            bag_cello_df=test_bag_cello_df,
            path=FULL_TEST_CELLO_BAG_DF_PATH,
            exist_ok=True,
        )

    else:
        test_bag_cello_df = cellohood.load_bag_cello_df(
            path=FULL_TEST_CELLO_BAG_DF_PATH
        )
        
        
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
                epoch_nb=400,
                use_cuda=True,
                log_mem=True,
                log_time=True,
                num_heads=num_heads,
                layer_size=layer_size,
                intermediate_layer_size=intermediate_layer_size,
                latent_size=latent_size,
                model_type='DeepSet',
        )

        cellohood.save_cello_train_result(
            cello_train_result,
            SAVING_PATH,
        )

    else:
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
        cache_path = f'new_cache_ds_{distance_threshold}_run_{run_nb}_{layer_size}_{intermediate_layer_size}_{latent_size}_{num_heads}'
        if os.path.exists(cache_path):
            print('Removing cache')
            subprocess.run([f'rm -r {cache_path}'])

        smoothed_df_result = cellohood.smooth_cellohood_prediction(
            cello_prediction,
            distance_threshold=distance_threshold,
            cache_path=cache_path,
        )

        cellohood.save_cello_prediction(
            cello_prediction=smoothed_df_result,
            path=CELLO_PREDICTION_PATH,
        )

    else:
        smoothed_df_result = cellohood.load_cello_prediction(
            path=CELLO_PREDICTION_PATH
        )

    # cell_df rebuilt in the same order/truncation the model's predictions use,
    # so ground-truth ('layer', 'sample') stays row-aligned with smoothed_df_result
    # even when a neighborhood exceeds cello_train_result.max_neighborhood_size.
    retained_cell_df = cellohood.get_retained_cells_df_from_df(
        df_=bag_cello_df.cell_df,
        cellohood_cluster_colname=bag_cello_df.cellohood_neighborhood_cluster_colname,
        image_col_name=bag_cello_df.image_id_column,
        max_neighborhood_size=cello_train_result.max_neighborhood_size,
    )

    # -------------------------
    # PCA
    # -------------------------
    decomposer = PCA()

    plda_decomposed_full_train = decomposer.fit_transform(
        smoothed_df_result.cell_predictions[
            smoothed_df_result.cello_columns
            ].values[retained_cell_df['sample'].apply(lambda x: x in TRAIN_SAMPLES)]
    )
    
    plda_decomposed_full_test = decomposer.transform(
        smoothed_df_result.cell_predictions[
            smoothed_df_result.cello_columns
            ].values[retained_cell_df['sample'].apply(lambda x: x not in TRAIN_SAMPLES)]
    )
    
    plda_decomposed_full = decomposer.transform(
        smoothed_df_result.cell_predictions[
            smoothed_df_result.cello_columns
            ].values
    )
    
    variance_ratios = decomposer.explained_variance_ratio_.cumsum()
    nb_of_coarse_components = max(np.abs(variance_ratios - 0.5).argmin() + 1, 1)
    nb_of_standard_components = max(np.abs(variance_ratios - 0.75).argmin() + 1, 2)
    
    print(f'Nb of standard components: {nb_of_standard_components}')
    print(f'Nb of coarse components: {nb_of_coarse_components}')

    plda_decomposed_full_standard_train = plda_decomposed_full_train[:, :nb_of_standard_components]
    plda_decomposed_full_standard_test = plda_decomposed_full_test[:, :nb_of_standard_components]
    
    plda_decomposed_full_coarse_train = plda_decomposed_full_train[:, :nb_of_coarse_components]
    plda_decomposed_full_coarse_test = plda_decomposed_full_test[:, :nb_of_coarse_components]


    layers_ = sorted(list(set(retained_cell_df['layer'])))
    
    # -------------------------
    # ARI computation
    # -------------------------
    layers_ = sorted(list(set(retained_cell_df["layer"])))
    samples_ = list(set(retained_cell_df["sample"]))

    sample_to_aris = defaultdict(list)
    clusterers = []
    
    TEST_SAMPLES = [
        sample 
        for sample in sorted(bag_cello_df.cell_df['sample'].unique())
        if sample not in TRAIN_SAMPLES
    ]

    standard_variables = list(range(nb_of_standard_components))
   
    for k in range(10):

        gmm = GaussianMixture(n_components=7, random_state=42 + k)
        gmm.fit(plda_decomposed_full_standard_train)
        predictions = gmm.predict(plda_decomposed_full[:, :nb_of_standard_components])   

        clusterers.append(gmm)

        for sample in samples_:

            sample_mask = (retained_cell_df["sample"] == sample).values
            sample_cell_df = retained_cell_df[sample_mask]

            true = retained_cell_df.loc[sample_mask, "layer"]
            pred = predictions[sample_mask]

            ari = adjusted_rand_score(
                true[sample_cell_df['layer'] != 'UNK'], 
                pred[sample_cell_df['layer'] != 'UNK'],
            )
            sample_to_aris[sample].append(ari)
            
        sample_to_aris['PLDA'].append('standard')
        
    for k in range(10):

        gmm = GaussianMixture(n_components=7, random_state=42 + k)
        gmm.fit(plda_decomposed_full_coarse_train)
        predictions = gmm.predict(plda_decomposed_full[:, :nb_of_coarse_components])   

        clusterers.append(gmm)

        for sample in samples_:

            sample_mask = (retained_cell_df["sample"] == sample).values
            sample_cell_df = retained_cell_df[sample_mask]

            true = retained_cell_df.loc[sample_mask, "layer"]
            pred = predictions[sample_mask]

            ari = adjusted_rand_score(
                true[sample_cell_df['layer'] != 'UNK'], 
                pred[sample_cell_df['layer'] != 'UNK'],
            )
            sample_to_aris[sample].append(ari)
            
        sample_to_aris['PLDA'].append('coarse')

    sample_aris_df = pd.DataFrame(sample_to_aris)
    sample_aris_df['nb_of_standard_components'] = nb_of_standard_components
    print(sample_aris_df.groupby('PLDA')[TRAIN_SAMPLES + TEST_SAMPLES].mean())


    # -------------------------
    # Save result
    # -------------------------
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
    
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of attention heads",
    )

    parser.add_argument(
        "--latent_size",
        type=int,
        required=True,
        help="Size of the latent space",
    )

    parser.add_argument(
        "--layer_size",
        type=int,
        required=True,
        help="Size of the layer",
    )

    parser.add_argument(
        "--intermediate_layer_size",
        type=int,
        required=True,
        help="Size of the intermediate layer",
    )

    args = parser.parse_args()


    start_time = time.time()
    main(
            distance_threshold=args.threshold,
            run_nb=args.run_nb,
            num_heads=args.num_heads,
            latent_size=args.latent_size,
            layer_size=args.layer_size,
            intermediate_layer_size=args.intermediate_layer_size,
        )
    print(f'Took: {time.time() - start_time} seconds.')
