from typing import List, Optional
import warnings

from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import scipy
import statsmodels
import statsmodels.api as sm
import pandas
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, ndcg_score
from sklearn.preprocessing import StandardScaler

from immuw.clustering import clustering_utils as cimu


PANEL_1_MARKER_NAMES = [
    'MPO', 
    'HistoneH3', 
    'SMA', 
    'CD16', 
    'CD38',
    'HLADR', 
    'CD27', 
    'CD15', 
    'CD45RA', 
    'CD163', 
    'B2M', 
    'CD20', 
    'CD68',
    'Ido1', 
    'CD3', 
    'LAG3', 
    'CD11c', 
    'PD1', 
    'PDGFRb', 
    'CD7', 
    'GrzB',
    'PDL1', 
    'TCF7', 
    'CD45RO', 
    'FOXP3', 
    'ICOS', 
    'CD8a', 
    'CarbonicAnhydrase',
    'CD33', 
    'Ki67', 
    'VISTA', 
    'CD40', 
    'CD4', 
    'CD14', 
    'Ecad', 
    'CD303',
    'CD206', 
    'cleavedPARP', 
    'DNA1', 
    'DNA2'
]

PANEL_2_MARKER_NAMES = [
    'panCK', 
    'HistoneH3', 
    'SMA', 
    'CD7',
    'CD11b', 
    'Arg1', 
    'CD146', 
    'EGFR', 
    'CD45', 
    'CD31', 
    'MMP9', 
    'CD20',
    'CD204', 
    'p53', 
    'CD3', 
    'Lamp3', 
    'CD11c', 
    'PD1', 
    'CD73', 
    'Bcl2', 
    'GATA3',
    'CD155', 
    'CD10', 
    'NKG2A', 
    'FOXP3', 
    'CXCL13', 
    'CD8a', 
    'EOMES', 
    'CD137',
    'CD134', 
    'CD209',
    'CD56', 
    'Tbet', 
    'GITR', 
    'Ecad', 
    'Tim3', 
    'CXCL8',
    'CD66b', 
    'DNA1', 
    'DNA2', 
    'Ki67', 
    'Podoplanin', 
    'IgG', 
    'CD15'
]


CELL_SHAPE_COLUMNS = ['area', 'major_axis_length', 'minor_axis_length',
       'eccentricity']


PANEL_1_DATA_COLUMNS = PANEL_1_MARKER_NAMES + CELL_SHAPE_COLUMNS

PANEL_1_CELL_TYPES = [
    'B', 
    'BnT', 
    'CD4', 
    'CD8', 
    'DC', 
    'HLADR', 
    'MacCD163',
    'Mural', 
    'NK', 
    'Neutrophil', 
    'Treg', 
    'Tumor', 
    'pDC', 
    'plasma',
    'undefined'
]

PANEL_2_CELL_TYPES = [
    'B',
    'BnT', 
    'CD4',
    'CD8', 
    'DC', 
    'Fibro', 
    'Igg', 
    'MDSC', 
    'MacCD204',
    'MacCD204_CD209', 
    'MacCD209', 
    'NK', 
    'Neutrophil', 
    'PMN_MDSC', 
    'SMA',
    'Treg', 
    'Tumor', 
    'Vasculature', 
    'mregDC', 
    'undefined', 
    'vCAF',
]


PURE_TUMOR_INTERESTING_MARKERS = [
    'CD15',
    'PDL1', 
    'CarbonicAnhydrase', 
    'Ido1',
    'HLADR',
    'Ki67',
    'B2M',
    'cleavedPARP',
    'Ecad',
    'CD38',
    'CD27',
    'CD68',
    'FOXP3',
    'CD45RA',
    'CD163',
    'CD20',
    'PDGFRb',
    'CD45RO',
    'CD33',
    'VISTA',
    'CD40',
    'CD4',
    'CD14',
    'CD303',
    'HistoneH3',
    'PD1',
]


B_CELL_INTRESTING_MARKERS = [
    'CD20',
    'CD27',
    'CD38',
    'CD45RA',
    'CD40',
]

NEUTROPHIL_CELL_INTERESTING_MARKERS = [
    'MPO', 
    'CD16',
    'CD15',
    'VISTA', 
    'PD1', 
]

MACROPHAGE_CELL_INTERESTING_MARKERS = [
    'CD68', 
    'CD163', 
    'CD11c', 
    'CD206', 
    'CD14',
    'VISTA',
    'PD1', 
    'PDGFRb',
    'cleavedPARP',
]

TREG_CELL_INTERESTING_MARKERS = [
    'CD3', 
    'CD4', 
    'FOXP3', 
    'ICOS',
    'LAG3', 
    'PD1', 
]

CD4_CELL_INTERESTING_MARKERS = [
    'CD27', 
    'CD45RA', 
    'CD3', 
    'CD7', 
    'CD45RO', 
    'CD40', 
    'CD4',
    'ICOS',
    'LAG3',
    'VISTA',
]

CD8_CELL_INTERESTING_MARKERS = [
    'CD27', 
    'CD45RA', 
    'CD3', 
    'CD8a',
    'LAG3',
    'VISTA',
    'ICOS',
    'GrzB',
    'Ki67',
]

PLASMA_CELL_INTERESTING_MARKERS = [
    'CD27', 
    'CD38',
]


PURE_TUMOR_INTERESTING_MARKERS_INDICES = [
    PANEL_1_MARKER_NAMES.index(ptim) for ptim in PURE_TUMOR_INTERESTING_MARKERS
]

B_CELL_INTERESTING_MARKERS_INDICES = [
    PANEL_1_MARKER_NAMES.index(ptim) for ptim in B_CELL_INTRESTING_MARKERS
]

NEUTROPHIL_CELL_INTERESTING_MARKERS_INDICES = [
    PANEL_1_MARKER_NAMES.index(ptim) for ptim in NEUTROPHIL_CELL_INTERESTING_MARKERS
]

MACROPHAGE_CELL_INTERESTING_MARKERS_INDICES = [
    PANEL_1_MARKER_NAMES.index(ptim) for ptim in MACROPHAGE_CELL_INTERESTING_MARKERS
]


SHAPE_COLUMNS = ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity']


PANEL_1_DATA_COLUMNS = PANEL_1_MARKER_NAMES + SHAPE_COLUMNS

META_CLUSTER_NAMES = [
    'PURE_TUMOR',
    'MURAL',
    'NEUTROPHIL',
    'PLASMA',
    'MAC',
    'INFILTRATION',
    'TLS',
]

META_CLUSTER_COL_NAME = 'meta_cluster'
META_CLUSTER_PERMUTATION_COLNAME = 'meta_cluster_sorting_index'
PATIENT_COLNAME = 'patient'
SIMPLE_HISTOLOGY_COLNAME = 'simple_histology'
DEATH_COLNAME = 'death'
SIMPLE_STAGE_COLNAME = 'simple_stage'
BINARY_STAGE_COLNAME = 'binary_stage'
BIOPSY_COLNAME = 'biopsy'
BATCH_COLNAME = 'batch'
SURVIVAL_COLNAME = 'days_to_death'


META_CLUSTER_TO_COLOR = {
    'PURE_TUMOR': mcolors.CSS4_COLORS['maroon'],
    'MURAL': mcolors.CSS4_COLORS['lightcoral'],
    'NEUTROPHIL': mcolors.CSS4_COLORS['royalblue'],
    'PLASMA': mcolors.CSS4_COLORS['darkorchid'],
    'MAC': mcolors.CSS4_COLORS['darkorange'],
    'INFILTRATION': mcolors.CSS4_COLORS['seagreen'],
    'TLS': mcolors.CSS4_COLORS['limegreen'],
    'DC_ENRICHED': mcolors.CSS4_COLORS['orchid'],
}


PANEL_1_CELL_TYPE_TO_COLOR = {
    'B': mcolors.CSS4_COLORS['limegreen'],
    'BnT': mcolors.CSS4_COLORS['green'],
    'CD4': mcolors.CSS4_COLORS['teal'],
    'CD8': mcolors.CSS4_COLORS['yellowgreen'],
    'DC': mcolors.CSS4_COLORS['plum'], 
    'HLADR': mcolors.CSS4_COLORS['orangered'],
    'MacCD163': mcolors.CSS4_COLORS['darkorange'],
    'Mural': mcolors.CSS4_COLORS['crimson'],
    'NK': mcolors.CSS4_COLORS['lime'], 
    'Neutrophil': mcolors.CSS4_COLORS['royalblue'], 
    'Treg': mcolors.CSS4_COLORS['cyan'], 
    'Tumor': mcolors.CSS4_COLORS['maroon'], 
    'pDC': mcolors.CSS4_COLORS['magenta'],
    'plasma': mcolors.CSS4_COLORS['darkorchid'],
    'undefined': mcolors.CSS4_COLORS['silver'],
}


PANEL_2_CELL_TYPE_TO_COLOR = {
    'B': mcolors.CSS4_COLORS['limegreen'],
    'BnT': mcolors.CSS4_COLORS['green'], 
    'CD4': mcolors.CSS4_COLORS['teal'],
    'CD8': mcolors.CSS4_COLORS['yellowgreen'], 
    'DC': mcolors.CSS4_COLORS['plum'], 
    'Fibro': mcolors.CSS4_COLORS['beige'],  
    'Igg': mcolors.CSS4_COLORS['darkorchid'], 
    'MDSC': mcolors.CSS4_COLORS['orange'], 
    'MacCD204': mcolors.CSS4_COLORS['orangered'],
    'MacCD204_CD209': mcolors.CSS4_COLORS['peru'], 
    'MacCD209': mcolors.CSS4_COLORS['darkgoldenrod'], 
    'NK': mcolors.CSS4_COLORS['lime'], 
    'Neutrophil': mcolors.CSS4_COLORS['royalblue'], 
    'PMN_MDSC': mcolors.CSS4_COLORS['darkorange'],
    'SMA': mcolors.CSS4_COLORS['crimson'],
    'Treg': mcolors.CSS4_COLORS['cyan'], 
    'Tumor': mcolors.CSS4_COLORS['maroon'], 
    'Vasculature': mcolors.CSS4_COLORS['deeppink'], 
    'mregDC': mcolors.CSS4_COLORS['orchid'],
    'undefined': mcolors.CSS4_COLORS['silver'], 
    'vCAF': mcolors.CSS4_COLORS['honeydew'], 
}


SIMPLE_HISTOLOGY_COLORS = {
     'Adenocarcinoma': 'magenta',
     'Adenosquamous carcinoma': 'lime',
     'Other': 'thistle',
     'Squamous cell carcinoma': 'aqua',
}


SIMPLE_STAGE_COLORS = {
    -1: 'tab:olive',
    1: 'tab:green',
    2: 'tab:orange',
    3: 'tab:red',
    4: 'maroon',
}


SIMPLE_STAGE_STR_COLORS = {
    'Unknown': 'tab:olive',
    1: 'tab:green',
    2: 'tab:orange',
    3: 'tab:red',
    4: 'maroon',
}


NSCLC_2_COLORS_DICT = {
    'NSCLC': np.array([0.0, 0.0, 1.0]),
    'NSCLC2': np.array([0.0, 0.0, 0.0]),
}


SAMPLING_COLORS_DICT = {
    'BIOPSY': np.array([0.0, 1.0, 0.0]),
    'NON BIOPSY': np.array([0.0, 0.0, 0.0]),
}


BINARY_STAGE_COLORS_DICT = {
    'HIGH': np.array([1.0, 0.0, 0.0]),
    'LOW': np.array([0.0, 0.0, 0.0]),
}


DEATH_COLORS_DICT = {
    'ALIVE': np.array([1.0, 1.0, 1.0]),
    'DEAD': np.array([0.0, 0.0, 0.0]),
}


def get_sh_colors(shs):
    return np.array([SIMPLE_HISTOLOGY_COLORS[sh_] for sh_ in shs]) 


def get_death_colors(deaths):
    return np.array([
         np.array([1.0, 1.0, 1.0]) * (1 - float(death_flag))
         for death_flag in deaths
    ])


def get_simple_stage_colors(simple_stages):
    return np.array([SIMPLE_STAGE_COLORS[simple_stage_] for simple_stage_ in simple_stages])


def get_binary_stage_colors(binary_stages):
    return np.array([
         np.array([1.0, 0.0, 0.0]) * float(binary_stage_)
         for binary_stage_ in binary_stages
    ])


def get_biopsy_colors(biopsy_flags):
    return np.array([
         np.array([0.0, 1.0, 0.0]) * float(biopsy_flag_)
         for biopsy_flag_ in biopsy_flags
    ])


def get_nsclc2_flags(patients):
    return np.array(['NSCLC2' in patient_ for patient_ in patients])


def get_nsclc2_colors(nsclc2_flags):
    return np.array([
         np.array([0.0, 0.0, 1.0]) * (1 - float(nsclc2_flag_))
         for nsclc2_flag_ in nsclc2_flags
    ])


# FILE CONSTANTS:
CLUSTER_TO_CELL_TYPE_AND_METACLUSTER_DF_FILENAME = 'cluster_to_cell_types_and_metaclusters.csv'
CLUSTER_AND_CELL_TYPE_TO_AUC_DF_FILENAME = 'marker_cell_types_aucs.df'
CLUSTER_AND_CELL_TYPE_TO_MWU_PV_DF_FILENAME = 'marker_cell_types_mwu_pvalues.df'
PATIENT_TO_CLUSTER_DF_FILENAME = 'patient_to_cluster_data.df'



# MISC
def get_stars_from_pvalue(pvalue):
    result = ''
    if pvalue <= 0.05:
        result += '*'
    if pvalue <= 0.01:
        result += '*'
    if pvalue <= 0.001:
        result += '*'
    return result


def fdr(pvalues, alpha=0.5):
    n_features = len(pvalues)
    sv = np.sort(pvalues)
    selected = sv[
            sv <= float(alpha) / n_features * np.arange(1, n_features + 1)
        ]
    if selected.size == 0:
        return np.zeros_like(pvalues, dtype=bool)
    return pvalues <= selected.max()


def absolut_power(x, power):
    return np.sign(x) * np.abs(x) ** power


# VECTORIZERS
def default_vectorizer(x):
    return x.to_numpy().T


def hellinger_vectorizer(x):
    np_x = x.to_numpy().T
    standarizers = np.maximum(np_x.sum(axis=0, keepdims=True), 1e-6)
    return (np_x / standarizers) ** 0.5


def median_div_vectorizer(x):
    np_x = x.to_numpy().T
    standarizers = np.median(np_x, axis=0, keepdims=True)
    return (np_x / standarizers)


def median_scaling_vectorizer(x):
    np_x = x.to_numpy().T
    median = np.median(np_x, axis=0, keepdims=True)
    #mad = scipy.stats.median_abs_deviation(np_x, axis=0)
    return (np_x - median)


def hellinger_vectorizer_wth_down_clip(x, threshold):
    np_x = x.to_numpy().T
    clipped_np_x = np.array(np_x).copy()
    clipped_np_x[clipped_np_x < threshold] = 0 
    standarizers = np.maximum(clipped_np_x.sum(axis=0, keepdims=True), 1e-6)
    return (clipped_np_x / standarizers) ** 0.5


def z_score_vectorizer_wth_down_clip(x, threshold):
    np_x = x.to_numpy().T
    clipped_np_x = np.array(np_x).copy()
    clipped_np_x[clipped_np_x < threshold] = 0 
    mean = clipped_np_x.mean(axis=1, keepdims=True)
    std = np.maximum(clipped_np_x.std(axis=1, keepdims=True), 1e-6)
    return (clipped_np_x - mean) / std


def hellinger_z_score_vectorizer_wth_down_clip(x, threshold):
    np_x = x.to_numpy().T
    clipped_np_x = np.array(np_x).copy()
    clipped_np_x[clipped_np_x < threshold] = 0 
    standarizers = np.maximum(clipped_np_x.sum(axis=0, keepdims=True), 1e-6)
    hellinger = (clipped_np_x / standarizers) ** 0.5
    mean = hellinger.mean(axis=1, keepdims=True)
    std = np.maximum(hellinger.std(axis=1, keepdims=True), 1e-6)
    return (hellinger - mean) / std


def get_ap_hellinger_vectorizer_with_clip(threshold, power=0.1):
    def res_fun(x):
        return absolut_power(hellinger_vectorizer_wth_down_clip(x, threshold), power)
    return res_fun


def get_ap_z_score_vectorizer_with_clip(threshold, power=0.1):
    def res_fun(x):
        return absolut_power(z_score_vectorizer_wth_down_clip(x, threshold), power)
    return res_fun


def get_ap_median_vectorizer(power=0.1):
    def res_fun(x):
        return absolut_power(median_scaling_vectorizer(x), power)
    return res_fun


def get_ap_hellinger_z_score_vectorizer_with_clip(threshold, power=0.1):
    def res_fun(x):
        return absolut_power(hellinger_z_score_vectorizer_wth_down_clip(x, threshold), power)
    return res_fun


def get_hellinger_vectorizer_with_clip(threshold):
    def res_fun(x):
        return hellinger_vectorizer_wth_down_clip(x, threshold)
    return res_fun


def get_z_score_vectorizer_with_clip(threshold):
    def res_fun(x):
        return z_score_vectorizer_wth_down_clip(x, threshold)
    return res_fun


def get_hellinger_z_score_vectorizer_with_clip(threshold):
    def res_fun(x):
        return hellinger_z_score_vectorizer_wth_down_clip(x, threshold)
    return res_fun


def convert_to_strings(els_):
    return [str(el_) for el_ in els_]


def convert_to_ints(els_):
    return [int(el_) for el_ in els_]


def filter_clusters_below_percentage(clusters, percentage, verbose=True, set_value=-1):
    new_clusters = np.array(list(clusters))
    possible_clusters = set(clusters)
    for possible_cluster in possible_clusters:
        current_percentage = np.mean(clusters == possible_cluster)
        if current_percentage < percentage:
            print(f'Deleting cluster {possible_cluster} with percentage {current_percentage}.')
            new_clusters[new_clusters == possible_cluster] = set_value
    return new_clusters
            


# VISUALIZATION_TOOLS:
def visualize_cell_types_df(
    cell_types_df,
    cell_types_names: Optional[List[str]] = None,
    cluster_permutation: Optional[List[str]] = None,
    cluster_descriptions: Optional[List[str]] = None,
    cell_type_colors: Optional[List] = None,
    scale: float = 0.7,
    show_legend: bool = True,
):
    scale=0.7
    cell_types_names = cell_types_names if cell_types_names is not None else PANEL_1_CELL_TYPES
    cluster_permutation = cluster_permutation if cluster_permutation is not None else cell_types_df[META_CLUSTER_PERMUTATION_COLNAME]
    cell_type_colors = cell_type_colors if cell_type_colors is not None else [
           META_CLUSTER_TO_COLOR[meta_cluster] for meta_cluster in cell_types_df[META_CLUSTER_COL_NAME].iloc[cluster_permutation]
    ]
    
    meta_cluster_names_ordered = []
    for metacluster_name in cell_types_df[META_CLUSTER_COL_NAME].iloc[cluster_permutation]:
        if metacluster_name not in meta_cluster_names_ordered:
            meta_cluster_names_ordered.append(metacluster_name)
            
    cluster_descriptions = cluster_descriptions if cluster_descriptions is not None else cluster_permutation
    
    ct_cm = seaborn.clustermap(
        cell_types_df[cell_types_names].to_numpy()[cluster_permutation].T ** 0.5, 
        row_cluster=True, 
        col_cluster=False,
        cmap='magma',
        col_colors=cell_type_colors,
        xticklabels=cluster_descriptions,
        yticklabels=cell_types_names,
        figsize=(int(len(cluster_permutation) * scale) + 3, int(len(cell_types_names) * scale) + 3),
        cbar_pos=(1.08, 0.15, 0.05, 0.5)
    )

    row_permutation = ct_cm.data2d.index.to_numpy()
    col_permutation = ct_cm.data2d.columns.to_numpy()

    cell_type_col_permutation = col_permutation

    for (j,i), label in np.ndenumerate(ct_cm.data2d):
        ct_cm.ax_heatmap.text(0.5 + i, 0.5 + j, round(label ** 2, 2), fontsize=16.0, ha='center',va='center')

    #ct_cm.figure.suptitle('Cell type to cluster distribution', fontsize=28.0)
    _ = ct_cm.ax_heatmap.set_xticklabels(
        ct_cm.ax_heatmap.get_xmajorticklabels(), 
        fontsize=24,
    )
    _ = ct_cm.ax_heatmap.set_yticklabels(
        ct_cm.ax_heatmap.get_ymajorticklabels(), 
        fontsize=24,
    )
    y_ticks = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    _ = ct_cm.ax_cbar.set_yticks(y_ticks ** 0.5)
    _ = ct_cm.ax_cbar.set_yticklabels(y_ticks, fontsize=16.0)
    
    if show_legend:
        present_metaclusters = meta_cluster_names_ordered
        handles = [Patch(facecolor=META_CLUSTER_TO_COLOR[name]) for name in meta_cluster_names_ordered]
        plt.legend(
            handles, 
            [' '.join(el.split('_')) for el in meta_cluster_names_ordered], 
            title='Metaclusters',
            bbox_to_anchor=(0.88, 0.9), 
            bbox_transform=plt.gcf().transFigure, 
            loc=(0.5, 1.5),
            fontsize=24,
            title_fontsize=24,
            ncols=len(meta_cluster_names_ordered) // 2,
        )
    
    return ct_cm


def visualize_cell_types_aucs(
    cell_type_name,
    cell_type_aucs,
    cell_type_aucs_pvalues,
    cell_type_percentages,
    scale=0.7,
):
    current_permutation = (-cell_type_percentages).argsort()
    descriptions = [
        f'{perm_index}\n({round(cell_type_percentages[perm_index], 2)})'
        for perm_index in current_permutation
    ]
    marker_cm = seaborn.clustermap(
        (cell_type_aucs.to_numpy().T[:, current_permutation]) * 2 - 1.0, 
        row_cluster=False, 
        col_cluster=False,
        cmap='YlGnBu_r',
        yticklabels=PANEL_1_MARKER_NAMES,
        xticklabels=descriptions,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        figsize=(int(len(cell_type_aucs) * 2 * scale) + 3, int(len(PANEL_1_MARKER_NAMES) * scale) + 3),
        cbar_pos=(1.1, 0.15, 0.05, 0.5)
    )
    pvals = (cell_type_aucs_pvalues.to_numpy().T[:, current_permutation])

    for (j,i), label in np.ndenumerate(marker_cm.data2d):
        marker_cm.ax_heatmap.text(
            0.5 + i, 
            0.5 + j, 
            f'{round((label + 1) / 2, 2)}\n{get_stars_from_pvalue(pvals[j, i])}', 
            fontsize=16.0, 
            ha='center',
            va='center',
        )

    #marker_cm.ax_row_dendrogram.figure.set_size_inches((16, 20))
    marker_cm.ax_heatmap.set_title(f'{cell_type_name} - Marker to cluster distribution', fontsize=28.0)
    marker_cm.ax_col_dendrogram.set_visible(False)
    _ = marker_cm.ax_heatmap.set_xticklabels(
        marker_cm.ax_heatmap.get_xmajorticklabels(), 
        fontsize=24,
    )
    _ = marker_cm.ax_heatmap.set_yticklabels(
        marker_cm.ax_heatmap.get_ymajorticklabels(), 
        fontsize=24,
    )
    _ = marker_cm.ax_cbar.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    _ = marker_cm.ax_cbar.set_yticklabels([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], fontsize=16.0)
    return marker_cm


def perform_cluster_selection(
    patients_df,
    clusters_to_select,
    clusters_to_analyze=None,
    vectorization_function=None,
    nb_of_tries_per_cluster_nb=5,
    take_every_example_for_training=1,
    plot=True,
):
    vectorization_function = vectorization_function if vectorization_function is not None else lambda x: x.to_numpy().T
    vectors = vectorization_function(patients_df[clusters_to_select])
    if clusters_to_analyze is None:
        clusters_to_analyze = list(range(3, 30 + 1))
    pl_result = cimu.cellohood_cluster_selection(
        clusters_to_analyze=clusters_to_analyze,
        nb_of_tries_per_cluster_nb=nb_of_tries_per_cluster_nb,
        take_every_example_for_training=take_every_example_for_training,
        predictions=vectors.T,
    )
    if plot:
        cimu.plot_cluster_size_to_fmses(
            cluster_nb_to_fmses=pl_result.cluster_nb_to_fmses,
            fms_clusters=pl_result.fms_clusters,
            fmses=pl_result.fmses,
            best_clusters=pl_result.best_clusterings,
            best_clusters_fmses=pl_result.best_clusterings_fmses,
            save_fig=False,
            path=None,
    )
    pl_best_clusterers = cimu.get_best_clusters_from_cimu_result(pl_result)
    return pl_result, pl_best_clusterers 


def visualize_patient_clusters(
    patients_df,
    clusters_to_select,
    number_of_clusters: int = 10,
    scale: float = 0.3,
    clusterer_type: str = 'agg',
    vectorization_function = None,
    method: str = 'complete',
    metric: str = 'euclidean',
    row_cluster: bool = False,
    title=None,
    cluster_descriptions=None,
    cmap='magma',
    center=0.5 ** 0.5,
    cluster_order=True,
    row_colors=None,
    figsize=None,
):
    vectorization_function = vectorization_function if vectorization_function is not None else lambda x: x.to_numpy().T
    vectors = vectorization_function(patients_df[clusters_to_select])

    if clusterer_type == 'agg':
        clusterer = AgglomerativeClustering(
            n_clusters=number_of_clusters, 
            linkage=method,
            affinity=metric,
        )
        clusterer.fit(vectors.T)
    elif clusterer_type == 'kmeans':
        clusterer = KMeans(n_clusters=number_of_clusters, random_state=42)
        clusterer.fit(vectors.T)
    else:
        clusterer = clusterer_type

    clusters = clusterer.predict(vectors.T)
    clusters_colors = [
        mpl.colors.XKCD_COLORS[list(mpl.colors.XKCD_COLORS.keys())[cluster_ind]]
        for cluster_ind in clusters
    ]
    
    cluster_to_colors_map = {
        ind_: mpl.colors.XKCD_COLORS[list(mpl.colors.XKCD_COLORS.keys())[ind_]]
        for ind_ in range(number_of_clusters)
    }
    
    #batches = set(patients_df[BATCH_COLNAME])
    #batch_to_color = {batch: plt.cm.tab20(ind_) for ind_, batch in enumerate(batches)}
    
    cluster_descriptions = cluster_descriptions if cluster_descriptions is not None else clusters_to_select
    
    plot_vectors = vectors[:, clusters.argsort()] if cluster_order else vectors
    order_ = clusters.argsort() if cluster_order else np.arange(len(clusters.argsort()))
    figsize = figsize if figsize is not None else (int(len(patients_df) * scale) + 3, int(len(clusters_to_select) * scale) + 3)
    
    marker_cm = seaborn.clustermap(
        plot_vectors, 
        method='complete',
        metric=metric,
        col_cluster=not cluster_order, 
        row_cluster=row_cluster,
        center=center,
        cmap=cmap,
        col_colors=[
            np.array(clusters_colors)[order_],
            get_sh_colors(patients_df[SIMPLE_HISTOLOGY_COLNAME])[order_],
            get_death_colors(patients_df[DEATH_COLNAME])[order_],
            get_simple_stage_colors(patients_df[SIMPLE_STAGE_COLNAME])[order_],
            get_binary_stage_colors(patients_df[BINARY_STAGE_COLNAME])[order_],
            #get_biopsy_colors(patients_df[BIOPSY_COLNAME])[order_],
            #get_nsclc2_colors(get_nsclc2_flags(patients_df[PATIENT_COLNAME]))[order_],
            #np.array([batch_to_color[batch] for batch in patients_df[BATCH_COLNAME]])[order_],
        ],
        row_colors=row_colors,
        #colors_ratio=0.025,
        xticklabels=[''] * len(order_),# None,#patients_df[PATIENT_COLNAME].iloc[order_],
        yticklabels=cluster_descriptions,
        figsize=figsize,
        cbar_pos=(1.1, 0.15, 0.05, 0.5),
    )
    
    delta = -0.1
    
    handles = [Patch(facecolor=cluster_to_colors_map[name]) for name in cluster_to_colors_map]
    sh_legend = plt.legend(
        handles, 
        list(cluster_to_colors_map.keys()), 
        title='Patient\nClusters',
        bbox_to_anchor=(0.25, 1), 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
        ncols=len(list(cluster_to_colors_map.keys())),
    )
    plt.gca().add_artist(sh_legend)
    
    handles = [Patch(facecolor=SIMPLE_HISTOLOGY_COLORS[name]) for name in SIMPLE_HISTOLOGY_COLORS]
    sh_legend = plt.legend(
        handles, 
        list(SIMPLE_HISTOLOGY_COLORS.keys()), 
        title='Histology',
        bbox_to_anchor=(0.0, 1 + delta) , 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
    )
    plt.gca().add_artist(sh_legend)
    
    handles = [Patch(facecolor=DEATH_COLORS_DICT[name]) for name in DEATH_COLORS_DICT]
    ss_lengend = plt.legend(
        handles, 
        list(DEATH_COLORS_DICT.keys()), 
        title='Death',
        bbox_to_anchor=(0.0, 0.8 + delta), 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
    )
    plt.gca().add_artist(ss_lengend)
    
    handles = [Patch(facecolor=SIMPLE_STAGE_STR_COLORS[name]) for name in SIMPLE_STAGE_STR_COLORS]
    ss_lengend = plt.legend(
        handles, 
        list(SIMPLE_STAGE_STR_COLORS.keys()), 
        title='Simple stage',
        bbox_to_anchor=(0.0, 0.65 + delta), 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
    )
    plt.gca().add_artist(ss_lengend)
    
    handles = [Patch(facecolor=BINARY_STAGE_COLORS_DICT[name]) for name in BINARY_STAGE_COLORS_DICT]
    bs_lengend = plt.legend(
        handles, 
        list(BINARY_STAGE_COLORS_DICT.keys()), 
        title='Binary stage',
        bbox_to_anchor=(0.0, 0.4 + delta), 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
    )
    plt.gca().add_artist(bs_lengend)
    
    #handles = [Patch(facecolor=SAMPLING_COLORS_DICT[name]) for name in SAMPLING_COLORS_DICT]
    #sampling_lengend = plt.legend(
    #    handles, 
    #    list(SAMPLING_COLORS_DICT.keys()), 
    #    title='Sampling type',
    #    bbox_to_anchor=(0.0, 0.25 + delta), 
    #    bbox_transform=plt.gcf().transFigure, 
    #    loc='upper left',
    #    fontsize=48,
    #    title_fontsize=48,
    #)
    #plt.gca().add_artist(sampling_lengend)
    
    #handles = [Patch(facecolor=NSCLC_2_COLORS_DICT[name]) for name in NSCLC_2_COLORS_DICT]
    #nsclc_legend = plt.legend(
    #    handles, 
    #    list(NSCLC_2_COLORS_DICT.keys()), 
    #    title='Cohort name',
    #    bbox_to_anchor=(0.0, 0.1), 
    #    bbox_transform=plt.gcf().transFigure, 
    #    loc='upper left',
    #    fontsize=48,
    #    title_fontsize=48,
    #)
    #plt.gca().add_artist(nsclc_legend)
    
        
    #handles = [Patch(facecolor=batch_to_color[name]) for name in batch_to_color]
    #batch_legend = plt.legend(
    #    handles, 
    #    list(batch_to_color.keys()), 
    #    title='Batch',
    #    bbox_to_anchor=(1.1, 1), 
    #    bbox_transform=plt.gcf().transFigure, 
    #    loc='upper left',
    #    fontsize=48,
    #    title_fontsize=48,
    #)
    #plt.gca().add_artist(batch_legend)
    
    marker_cm.ax_heatmap.set_title('Meta clusters clustering', fontsize=28.0)
    _ = marker_cm.ax_heatmap.set_xticklabels(
        marker_cm.ax_heatmap.get_xmajorticklabels(), 
        fontsize=24,
    )
    _ = marker_cm.ax_heatmap.set_yticklabels(
        marker_cm.ax_heatmap.get_ymajorticklabels(), 
        fontsize=54,
        rotation=0,
    )
    marker_cm.ax_cbar.set_title('(Meta)cluster\nabundance', fontsize=64, y=-0.15)
    return marker_cm, vectors, clusters, clusterer


def visualize_patient_clusters_no_lung(
    patients_df,
    clusters_to_select,
    number_of_clusters: int = 10,
    scale: float = 0.3,
    vectorization_function = None,
    clusterer_type: str = 'agg',
    method: str = 'complete',
    metric: str = 'euclidean',
    row_cluster: bool = False,
    title=None,
    cluster_descriptions=None,
    cmap='magma',
    center=0.5 ** 0.5,
    cluster_order=True,
    row_colors=None,
):
    vectorization_function = vectorization_function if vectorization_function is not None else lambda x: x.to_numpy().T
    vectors = vectorization_function(patients_df[clusters_to_select])

    if clusterer_type == 'agg':
        clusterer = AgglomerativeClustering(
            n_clusters=number_of_clusters, 
            linkage=method,
            affinity=metric,
        )
    elif clusterer_type == 'kmeans':
        clusterer = KMeans(n_clusters=number_of_clusters, random_state=42)
    else:
        clusterer = clusterer_type
        

    clusters = clusterer.fit_predict(vectors.T)
    clusters_colors = [
        mpl.colors.XKCD_COLORS[list(mpl.colors.XKCD_COLORS.keys())[cluster_ind]]
        for cluster_ind in clusters
    ]
    
    cluster_to_colors_map = {
        ind_: mpl.colors.XKCD_COLORS[list(mpl.colors.XKCD_COLORS.keys())[ind_]]
        for ind_ in range(number_of_clusters)
    }
    
    cluster_descriptions = cluster_descriptions if cluster_descriptions is not None else clusters_to_select
    
    plot_vectors = vectors[:, clusters.argsort()] if cluster_order else vectors
    order_ = clusters.argsort() if cluster_order else np.arange(len(clusters.argsort()))
    
    marker_cm = seaborn.clustermap(
        plot_vectors, 
        method='complete',
        metric=metric,
        col_cluster=not cluster_order, 
        row_cluster=row_cluster,
        center=center,
        cmap=cmap,
        col_colors=[
            np.array(clusters_colors)[order_],
        ],
        row_colors=row_colors,
        #colors_ratio=0.025,
        xticklabels=patients_df[PATIENT_COLNAME].iloc[order_],
        yticklabels=cluster_descriptions,
        figsize=(int(len(patients_df) * scale) + 3, int(90 * scale) + 3),
        cbar_pos=(1.1, 0.15, 0.05, 0.5),
    )
    
    handles = [Patch(facecolor=cluster_to_colors_map[name]) for name in cluster_to_colors_map]
    sh_legend = plt.legend(
        handles, 
        list(cluster_to_colors_map.keys()), 
        title='Patient\nClusters',
        bbox_to_anchor=(1.05, 1), 
        bbox_transform=plt.gcf().transFigure, 
        loc='upper left',
        fontsize=48,
        title_fontsize=48,
    )
    plt.gca().add_artist(sh_legend)
    
    delta = -0.1
    
    marker_cm.ax_heatmap.set_title('Meta clusters clustering', fontsize=28.0)
    _ = marker_cm.ax_heatmap.set_xticklabels(
        marker_cm.ax_heatmap.get_xmajorticklabels(), 
        fontsize=24,
    )
    _ = marker_cm.ax_heatmap.set_yticklabels(
        marker_cm.ax_heatmap.get_ymajorticklabels(), 
        fontsize=54,
        rotation=0,
    )
    marker_cm.ax_cbar.set_title('(Meta)cluster\nabundance', fontsize=64, y=-0.15)
    return marker_cm, vectors, clusters, clusterer



def get_roc_curves_for_class(
    score_matrix, 
    trues, 
    title,
    labels=None, 
    filter_at=0.0,
    fdr_alpha=0.05,
    minimal_patient_percentage=0.0,
    plot_fdr_only=False,
    label_lambda=None,
    rotate_violin_x=True,
):
    if label_lambda is None:
        label_lambda = lambda x: x
    labels = labels if labels is not None else list(range(score_matrix.shape[0]))
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 2)
    rses, mws = [], []
    tprs, fprs = [], []
    good_labels, good_labels_indices = [], []
    pps = []
    for i, label in enumerate(labels):
        if filter_at < 0.0:
            fpr, tpr, _ = roc_curve(y_true=trues, y_score=score_matrix[i])
            rs = roc_auc_score(y_true=trues, y_score=score_matrix[i])
            mw = mannwhitneyu(score_matrix[i][trues], score_matrix[i][np.logical_not(trues)]).pvalue
            rses.append(rs)
            mws.append(mw)
            pps.append(1.0)
        else:
            try:
                if (score_matrix[i] > 0).mean() <= minimal_patient_percentage:
                    print(f'Too small percentage of patients {label}.')
                    continue
                fpr, tpr, _ = roc_curve(y_true=trues[score_matrix[i] > filter_at], y_score=score_matrix[i][score_matrix[i] > 0])
                rs = roc_auc_score(y_true=trues[score_matrix[i] > filter_at], y_score=score_matrix[i][score_matrix[i] > 0])
                mw = mannwhitneyu(
                    score_matrix[i][np.logical_and(trues, score_matrix[i] > filter_at)], 
                    score_matrix[i][np.logical_and(np.logical_not(trues), score_matrix[i] > filter_at)]).pvalue
                rses.append(rs)
                mws.append(mw)
                pps.append((score_matrix[i] > filter_at).mean())
            except:
                print(f'Failed for {label}')
                continue
        fprs.append(fpr)
        tprs.append(tpr)
        good_labels.append(label)
        good_labels_indices.append(i)
        
    fdr_decisions = fdr(mws, alpha=fdr_alpha)
    for ind_, (fpr, tpr, label, label_ind, rs, mw, fdr_decision, pp) in enumerate(
        zip(
            fprs, 
            tprs, 
            good_labels, 
            good_labels_indices,
            rses, 
            mws, 
            fdr_decisions,
            pps,
        )
    ):
        if plot_fdr_only and not fdr_decision:
            continue
        current_lw = 2.0 if fdr_decision else 1.0
        current_alpha = 1.0 if fdr_decision else 0.3
        current_label = f"{label} (AUC = {round(rs, 2)}, P% = {round(100 * pp)}%) " + get_stars_from_pvalue(mw) + ' '
        if fdr_decision:
            current_label += u'\u2713'
        plt.plot(
            fpr, 
            tpr, 
            label=current_label, 
            c=plt.get_cmap('gist_rainbow')(ind_ / len(good_labels)),
            linewidth=current_lw,
            alpha=current_alpha,
        )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot([0, 1], [0, 1], c='black', linewidth=3.0, linestyle='--')
    plt.title(title + '\n' + 'ROC - curves analysis')
    
    ax2 = plt.subplot(1, 2, 1)
    violin_colors = []
    violins = []
    xticklabels = []
    for ind_i, i in enumerate(good_labels_indices):
        if plot_fdr_only and not fdr_decisions[ind_i]:
            continue
        violins.extend([score_matrix[i][np.logical_and(trues, score_matrix[i] > filter_at)], 
                    score_matrix[i][np.logical_and(np.logical_not(trues), score_matrix[i] > filter_at)]])
        violin_colors.extend([plt.get_cmap('gist_rainbow')(ind_i / len(good_labels)),] * 2)
        current_xtick = f'{label_lambda(labels[i])}\n' + get_stars_from_pvalue(mws[ind_i]) + '\n'
        if fdr_decisions[ind_i]:
            current_xtick += u'\u2713'
        xticklabels.append(current_xtick)
    parts = ax2.violinplot(violins, showmedians=True, showextrema=False)
    ax2.set_xticks([ind + 0.5 for ind in range(1, 2 * len(xticklabels) + 1, 2)])
    ax2.set_xticklabels(xticklabels, rotation=90 if rotate_violin_x else 0)
    for pc_ind, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[pc_ind])
        pc.set_edgecolor('black')
        pc.set_alpha(1 if fdr_decisions[pc_ind // 2] or plot_fdr_only else 0.3)
    parts['cmedians'].set_facecolor('w')
    parts['cmedians'].set_edgecolor('black')
    ax2.set_title(title + '\n' + 'Violin plots analysis')
    return rses, mws, fdr_decisions, good_labels


def plot_single_cluster_analysis_against(
    patients_df,
    clusters_to_analyze,
    against,
    title,
    cell_distributions_and_markers_to_visualize,
    marker_distributions,
    vectorizer=None,
    filter_at=0.0,
    scale=0.7,
    print_corr=False,
    plot_fdr_only=False,
    label_lambda=None,
    fdr_alpha=0.05,
    marker_z_score=None,
    marker_col_colors=None,
    marker_c_bar_position=(1.05, 0.15, 0.05, 0.5),
    return_all=False,
):
    vectorization_function = vectorizer if vectorizer is not None else lambda x: x.to_numpy().T
    scores = vectorization_function(patients_df[clusters_to_analyze])
    
    if print_corr:
        corr_df = pandas.DataFrame(data=scores.T, columns=clusters_to_analyze)
        corr = corr_df.corr()
        corr.style.background_gradient(cmap='coolwarm')
    
        cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    
        plt.figure(figsize=(2 * scores.shape[0], 2 * scores.shape[0]))
        seaborn.heatmap(
            corr, 
            cmap=cmap, 
            annot=True,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
        )
    
    trues = patients_df[against].to_numpy() if isinstance(against, str) else against
    rses, mws, fdrs, gls = get_roc_curves_for_class(
        score_matrix=scores, 
        trues=trues, 
        title=title,
        labels=clusters_to_analyze, 
        filter_at=filter_at,
        plot_fdr_only=plot_fdr_only,
        label_lambda=label_lambda,
        fdr_alpha=fdr_alpha,
    )
    
    marker_col_colors_permuted = None
    marker_col_names = []
    if marker_col_colors is not None:
        marker_col_colors_permuted = []
        for col_color_name, col_color in marker_col_colors:
            marker_col_colors_permuted.append(
                [col_color[int(gls[ind_])] for ind_ in np.array(rses).argsort()]
            )
            marker_col_names.append(col_color_name)
            
    old_fdrs = fdrs.copy()
    old_gls = gls.copy()
    old_rses = rses.copy()
    old_mws = mws.copy()
    if plot_fdr_only:
        gls = [gl for gl, fdr_ in zip(gls, fdrs) if fdr_]
        rses = [rs for rs, fdr_ in zip(rses, fdrs) if fdr_]
        mws = [mw for mw, fdr_ in zip(mws, fdrs) if fdr_]
        fdrs = [fdr_ for fdr_ in fdrs if fdr_]
        
    
    for cell_type, markers in cell_distributions_and_markers_to_visualize:
        current_ct_distribution = marker_distributions[markers][marker_distributions['celltype'] == cell_type].to_numpy()
        if plot_fdr_only:
            current_ct_distribution = current_ct_distribution[convert_to_ints(old_gls)][old_fdrs]
        else:
            current_ct_distribution = current_ct_distribution[convert_to_ints(old_gls)]
            
        #corrs = np.array([scipy.stats.spearmanr(row, rses).statistic for row in current_ct_distribution.T]).reshape((-1,))
        corrs = np.array([np.dot(row, rses) for row in current_ct_distribution.T]).reshape((-1,))

        corrs_argsort = np.argsort(corrs)
            
        marker_cm = seaborn.clustermap(
            current_ct_distribution.T[:, np.array(rses).argsort()][corrs_argsort], 
            row_cluster=False, 
            col_cluster=False,
            cmap='YlGnBu_r',
            yticklabels=np.array(markers)[corrs_argsort],
            z_score=marker_z_score,
            col_colors=marker_col_colors_permuted,
            center=0.5,
            vmin=0.0,
            vmax=1.0,
            xticklabels=[
                f'{gls[ind_]}\n'
                f'({round(rses[ind_], 2)})\n'
                f'{get_stars_from_pvalue(mws[ind_])}\n' + (u'\u2713' if fdrs[ind_] else '')
            for ind_ in np.array(rses).argsort()],
            figsize=(int(3 * len(gls) * scale) + 3, int(len(markers) * scale) + 3),
            cbar_pos=marker_c_bar_position,
         )

        for (j,i), label in np.ndenumerate(marker_cm.data2d):
            marker_cm.ax_heatmap.text(0.5 + i, 0.5 + j, round(100 * label), fontsize=16.0, ha='center',va='center')
      
        marker_cm.ax_col_dendrogram.set_visible(False)
        _ = marker_cm.ax_heatmap.set_xticklabels(
            marker_cm.ax_heatmap.get_xmajorticklabels(), 
            fontsize=24,
            rotation=0,
        )
        _ = marker_cm.ax_heatmap.set_yticklabels(
            marker_cm.ax_heatmap.get_ymajorticklabels(), 
            fontsize=24,
            rotation=0,
        )
        if marker_col_colors is not None:
            marker_cm.ax_col_colors.set_yticks(
                ticks=[ind_ + 0.5 for ind_ in range(len(marker_col_names))], 
                labels=marker_col_names,
                fontsize=18.0,
            )
            marker_cm.ax_col_colors.set_title(f'{title}\n{cell_type} marker distribution', fontsize=28.0)
        else:
            marker_cm.ax_heatmap.set_title(f'{title}\n{cell_type} marker distribution', fontsize=28.0)
        _ = marker_cm.ax_cbar.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        _ = marker_cm.ax_cbar.set_yticklabels([0, 25, 50, 75, 100], fontsize=16.0)
    if return_all:
        return scores, old_fdrs, old_gls, old_rses, old_mws
    return scores


def plot_glm_cluster_analysis_against(
    patients_df,
    clusters_to_analyze,
    against,
    title,
    cell_distributions_and_markers_to_visualize,
    marker_distributions,
    vectorizer=None,
    scale=0.7,
    plot_valid_only=False,
):
    vectorization_function = vectorizer if vectorizer is not None else lambda x: x.to_numpy().T
    scores = vectorization_function(patients_df[clusters_to_analyze])
    
    
    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)

    
    trues = patients_df[[against]].to_numpy() if isinstance(against, str) else against.reshape((-1, 1))
    

    model = sm.GLM(
        trues,
        statsmodels.tools.add_constant(scores.T), 
        family=sm.families.Binomial()
    )
    results = model.fit()
    pvalues, coefficients = results.pvalues[1:], results.params[1:]
    coefficients = np.clip(coefficients, -10, 10)
    
    marker_cm = seaborn.clustermap(
            coefficients.reshape((1, -1))[:, np.array(coefficients).argsort()], 
            row_cluster=False, 
            col_cluster=False,
            cmap='Wistia',
            center=0.5,
            xticklabels=[
                f'{clusters_to_analyze[ind_]}\n'
                f'({round(coefficients[ind_], 2)})\n'
                f'{get_stars_from_pvalue(pvalues[ind_])}\n'
            for ind_ in np.array(coefficients).argsort()] ,
            figsize=(int(3 * len(coefficients) * scale) + 3, int(scale) + 3),
            cbar_pos=(1.2, 0.15, 0.05, 0.5)
         )
    
    marker_cm.ax_heatmap.set_title(title, fontsize=28.0)
    
    coeff_sort = np.array(coefficients).argsort()
    
    for (j,i), label in np.ndenumerate(marker_cm.data2d):
        marker_cm.ax_heatmap.text(
            0.5 + i, 
            0.5 + j,
            f'{round(coefficients[coeff_sort[i]], 2)}\n'
            f'{get_stars_from_pvalue(pvalues[coeff_sort[i]])}',
            fontsize=16.0, ha='center',va='center', 
            weight='bold' if len(get_stars_from_pvalue(pvalues[coeff_sort[i]])) > 0 else 'normal'
        )
   
    for cell_type, markers in cell_distributions_and_markers_to_visualize:
        current_ct_distribution = marker_distributions[markers][marker_distributions['celltype'] == cell_type].to_numpy()
        marker_cm = seaborn.clustermap(
            current_ct_distribution.T[:, convert_to_ints(clusters_to_analyze)][:, np.array(coefficients).argsort()], 
            row_cluster=False, 
            col_cluster=False,
            cmap='YlGnBu_r',
            yticklabels=markers,
            center=0.5,
            xticklabels=[
                f'{clusters_to_analyze[ind_]}\n'
                f'({round(coefficients[ind_], 2)})\n'
                f'{get_stars_from_pvalue(pvalues[ind_])}\n'
            for ind_ in np.array(coefficients).argsort()],
            figsize=(int(3 * len(coefficients) * scale) + 3, int(len(markers) * scale) + 3),
            cbar_pos=(1.2, 0.15, 0.05, 0.5)
         )

        for (j,i), label in np.ndenumerate(marker_cm.data2d):
            marker_cm.ax_heatmap.text(0.5 + i, 0.5 + j, round(100 * label), fontsize=16.0, ha='center',va='center')

        marker_cm.ax_heatmap.set_title(f'{title}\n{cell_type} marker distribution', fontsize=28.0)
        marker_cm.ax_col_dendrogram.set_visible(False)
        _ = marker_cm.ax_heatmap.set_xticklabels(
            marker_cm.ax_heatmap.get_xmajorticklabels(), 
            fontsize=24,
            rotation=0,
        )
        _ = marker_cm.ax_heatmap.set_yticklabels(
            marker_cm.ax_heatmap.get_ymajorticklabels(), 
            fontsize=24,
            rotation=0,
        )
        _ = marker_cm.ax_cbar.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        _ = marker_cm.ax_cbar.set_yticklabels([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], fontsize=16.0)
    return scores




def surivial_on_df(
    survival_df,
    penalizer=0.001,
    l1_ratio=0.9,
    non_data_columns = ('survival', 'died', 'simple_stage'),
    standardize: bool = True,
    cox_kwargs=None,
    fit_kwargs=None,
):
    DATA_COLUMNS = [
        col for col in survival_df.columns
        if col not in non_data_columns
    ]
    NON_SIMPLE_STAGE_COLUMNS = [
        col for col in survival_df.columns
        if col != 'simple_stage'
    ]
    full_stage_t_n_m_selector = (survival_df.simple_stage >= 0)
    full_stage_known_df = survival_df[full_stage_t_n_m_selector].copy()
    if cox_kwargs is None:
        cox_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    cphfs = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio, **cox_kwargs)
    if standardize:
        survival_data_scaler = StandardScaler()
        scaled_survival_data = survival_data_scaler.fit_transform(full_stage_known_df[DATA_COLUMNS])
        full_stage_known_df[DATA_COLUMNS] = scaled_survival_data
    cphfs.fit(
        df=full_stage_known_df[NON_SIMPLE_STAGE_COLUMNS], 
        duration_col='survival', 
        event_col='died',
        **fit_kwargs,
    )
    return cphfs


def get_cox_trajectory(
    survival_df,
    l2=1e-3,
    l1_steps=21,
    l1_start=-5,
    l1_end=0,
    plot=True,
    standardize=True,
    non_data_columns = ('survival', 'died', 'simple_stage'),
):
    cox_trajectory = []
    l1s = []
    for l1_value in np.logspace(start=l1_start, stop=l1_end, num=l1_steps):
        penalizer = l1_value + l2
        l1 = l1_value / (l1_value + l2)
        l1s.append(l1_value)
        cox_trajectory.append(surivial_on_df(
            survival_df, 
            penalizer=penalizer, 
            l1_ratio=l1, 
            standardize=standardize,
            non_data_columns=non_data_columns,
        ))
    return cox_trajectory, l1s


def _get_cox_color(row_name, clustering_name):
    if clustering_name in row_name:
        cluster_ind = int(row_name.split('_')[-1])
        return mpl.colors.XKCD_COLORS[list(mpl.colors.XKCD_COLORS.keys())[cluster_ind]]
    if row_name.endswith('0'):
        return 'b'
    return 'r'
    


def plot_cox_trajectory(cox_trajectory, clustering_name, l1s, baseline_hazard='simple_stage_1'):
    cox_params = pandas.concat([cox_.hazard_ratios_.T for cox_ in cox_trajectory], axis=1)
    if baseline_hazard is not None:
        ss1_hr = np.log(np.array([cox_trajectory_.hazard_ratios_[baseline_hazard] for cox_trajectory_ in cox_trajectory]))
    else:
        ss1_hr = 0
    lr_cis = np.concatenate([cox_trajectory_.confidence_intervals_.to_numpy() for cox_trajectory_ in cox_trajectory], axis=1)
    zero_in_cis = (lr_cis[:, ::2] - ss1_hr) * (lr_cis[:, 1::2] - ss1_hr) > 0
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ticks, labels = [], []
    for real_row_nb, (row_nb, row) in enumerate(cox_params.iterrows()):
        ax.plot(np.log(row.to_numpy()), label=row.name, c=_get_cox_color(row.name, clustering_name), alpha=0.7) 
        if zero_in_cis[real_row_nb].sum() > 0:
            ax.plot(
                np.arange(len(row.to_numpy()))[zero_in_cis[real_row_nb]],
                np.log(row.to_numpy()[zero_in_cis[real_row_nb]]),
                label=row.name, 
                c=_get_cox_color(row.name, clustering_name),
                marker='*',
                markersize=15,
            ) 
        ticks.append(np.log(row.to_numpy()[0]))
        labels.append(row.name)
    ax.set_yticks(ticks)
    ax.set_xticks(list(range(len(l1s))))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(["\nE".join("{:.2E}".format(l1_).split('E')) for l1_ in l1s])
    ax.set_title(f'L1 Lasso Feature Importance Plot\n{clustering_name}')
    ax.set_xlabel('L1 Lasso Ratio')
    ax.set_ylabel('Feature Names')
    ax.legend()
    
    
def plot_cox_trajectory_for_clustering(
    patients_df,
    clustering,
    clustering_name,
    standardize=True,
    filter_below_percentage=0.05,
    non_data_columns = ('survival', 'died', 'simple_stage'),
    additional_feature_columns=('is_adeno', 'Gender_male', 'Smoking', 'Smoking-Non', 'Smoking-Ex'),
):
    clustering = filter_clusters_below_percentage(clustering, filter_below_percentage)
    good_patients = patients_df[clustering >= 0]
    good_clustering = clustering[clustering >= 0]
    current_survival_df = pandas.DataFrame(
       {
            **{
                'died': good_patients[DEATH_COLNAME],
                'survival': good_patients[SURVIVAL_COLNAME], 
                'simple_stage': good_patients[SIMPLE_STAGE_COLNAME],
            },
           **{
               col: good_patients[col]
               for col in additional_feature_columns
           },
            **{
                f'simple_stage_{i}': good_patients[SIMPLE_STAGE_COLNAME] == i
                for i in range(1, 4 + 1) if sum(good_patients[SIMPLE_STAGE_COLNAME] == i) > 0
            },
            **{
                f'{clustering_name}_{i}': good_clustering == i
                for i in set(good_clustering)
            },
        }
    )
    with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        cox_trajectory, l1s = get_cox_trajectory(
            current_survival_df, 
            standardize=standardize, 
            non_data_columns=non_data_columns,
        )
    plot_cox_trajectory(cox_trajectory, clustering_name, l1s)
    return cox_trajectory
    

def visualize_kaplan_meier(
    df,
    clustering,
    clustering_name,
    cluster_to_visualize,
    show_with_pvalue_lower=None,
):
    kmf = KaplanMeierFitter()
    results = logrank_test(
        df.iloc[clustering == cluster_to_visualize].days_to_death,
        df.iloc[(clustering != cluster_to_visualize)].days_to_death,
        df.iloc[clustering == cluster_to_visualize].death,
        df.iloc[(clustering != cluster_to_visualize)].death,
    )
    if show_with_pvalue_lower is not None and results.p_value > show_with_pvalue_lower:
        return results

    results.print_summary()

    kmf.fit(
        df.iloc[clustering == cluster_to_visualize].days_to_death, 
        event_observed=df.iloc[clustering == cluster_to_visualize].death,
        label=f'{cluster_to_visualize}',
    )
    kmf.plot_survival_function(show_censors=True)
    kmf.fit(
        df.iloc[(clustering != cluster_to_visualize)].days_to_death, 
        event_observed=df.iloc[(clustering != cluster_to_visualize)].death,
        label='rest',
    )
    kmf.plot_survival_function(show_censors=True)
    plt.title(f'{clustering_name} - cluster: {cluster_to_visualize}\np value = {round(results.p_value, 3)}{get_stars_from_pvalue(results.p_value)}')
    plt.show()
    return results


def compute_hypergeometrics_for_df(
    df,
    column,
    clusters,
    values,
    add_prefix=True,
    filter_nas=True,
    filter_clusters=None,
    min_p_value=1e-5,
):
    filter_clusters = filter_clusters if filter_clusters is not None else []
    nb_of_clusters = len(set(clusters))
    cluster_set = [cl_ for cl_ in range(nb_of_clusters) if cl_ not in filter_clusters]
    values = values if values is not None else list(set(df[column]))
    if filter_nas:
        values = [
            value_ 
            for value_ in values 
            if not (isinstance(value_, float) and np.isnan(value_)) and value_ is not None
        ]
    results = np.zeros((len(cluster_set), len(values)))
    for val_nb, val in enumerate(values):
        for cluster_ind, cluster_nb in enumerate(cluster_set):
            cluster_count = (clusters == cluster_nb).sum()
            if cluster_count == 0:
                results[cluster_ind, val_nb] = 1
                continue
            val_count = (df[column] == val).sum()
            val_within_cluster_count = (np.logical_and(df[column] == val, clusters == cluster_nb)).sum() - 1
            total = df.shape[0]
            cdf = scipy.stats.hypergeom.cdf(
                k=val_within_cluster_count,
                M=total, n=val_count, N=cluster_count
            )
            results[cluster_ind, val_nb] = 1 - cdf
    results = np.maximum(results, min_p_value)
    if add_prefix:
        col_values = [f'{column}: {val}' for val in values]
    else:
        col_values = values
    result_df = pandas.DataFrame(data=results, columns=col_values, index=cluster_set)
    return result_df


def compute_hypergeometrics_for_df_old(
    df,
    column,
    clusters,
    values,
    add_prefix=True,
    filter_nas=True,
    filter_clusters=None,
    min_p_value=1e-5,
):
    filter_clusters = filter_clusters if filter_clusters is not None else []
    nb_of_clusters = len(set(clusters))
    cluster_set = [cl_ for cl_ in range(nb_of_clusters) if cl_ not in filter_clusters]
    values = values if values is not None else list(set(df[column]))
    if filter_nas:
        values = [
            value_ 
            for value_ in values 
            if not (isinstance(value_, float) and np.isnan(value_)) and value_ is not None
        ]
    results = np.zeros((len(cluster_set), len(values)))
    for val_nb, val in enumerate(values):
        for cluster_ind, cluster_nb in enumerate(cluster_set):
            cluster_count = (clusters == cluster_nb).sum()
            if cluster_count == 0:
                results[cluster_ind, val_nb] = 1
                continue
            val_count = (df[column] == val).sum()
            val_within_cluster_count = (np.logical_and(df[column] == val, clusters == cluster_nb)).sum()
            total = df.shape[0]
            cdf = scipy.stats.hypergeom.cdf(
                k=val_within_cluster_count,
                M=total,
                n=cluster_count,
                N=val_count
            )
            results[cluster_ind, val_nb] = 1 - cdf
    results = np.maximum(results, min_p_value)
    if add_prefix:
        col_values = [f'{column}: {val}' for val in values]
    else:
        col_values = values
    result_df = pandas.DataFrame(data=results, columns=col_values, index=cluster_set)
    return result_df



def apply_fdr_to_df(
    df,
    alpha=0.1,
    method='fdr_bh',
):
    pvals = df.to_numpy().copy()
    flatten_pvals = pvals.flatten()
    _, q_vals, _, _ = statsmodels.stats.multitest.multipletests(flatten_pvals, method=method, alpha=alpha)
    new_df = pandas.DataFrame(data=q_vals.reshape(df.shape), columns=df.columns, index=df.index)
    return new_df


def compute_hypergeometrics_q_for_df(
    df,
    columns,
    clusters,
    values = None,
    add_prefix=True,
    filter_nas=True,
    filter_clusters=None,
    min_p_value=1e-5,
    fdr_alpha=0.1,
    method='fdr_bh',
    filter_below_percentage=0.05,
    plot=True,
):
    clusters_to_filter = [cluster_ for cluster_ in clusters if np.mean(clusters == cluster_) < filter_below_percentage]
    dfs = [
        compute_hypergeometrics_for_df(
            df=df,
            column=column,
            clusters=clusters,
            values=values,
            add_prefix=add_prefix,
            filter_nas=filter_nas,
            filter_clusters=clusters_to_filter,
            min_p_value=min_p_value,
        )
        for column in columns
    ]
    full_df = pandas.concat(dfs, axis=1)
    q_vals = apply_fdr_to_df(full_df, alpha=fdr_alpha, method=method)
    if plot:
        q_vals_sig_series = q_vals.min() < fdr_alpha
        s_cols = [i[0] for i in q_vals_sig_series.items() if i[1]]
        seaborn.clustermap(
            q_vals[s_cols].apply(lambda x: np.log(x) / np.log(fdr_alpha)), 
            annot=True, 
            vmin=0.0, 
            fmt='.2f',
            figsize=(1.0 * q_vals[s_cols].shape[1], 1.0 * q_vals[s_cols].shape[0]),
        )
    return q_vals


def visualize_patient_clusters_with_survival_analysis_and_hg_no_lung(
    patients_df,
    clusters_to_select,
    hypergeometric_columns,
    clusterer_type: str = 'agg',
    number_of_clusters: int = 10,
    scale: float = 0.3,
    vectorization_function = None,
    method: str = 'complete',
    metric: str = 'euclidean',
    row_cluster: bool = False,
    title=None,
    cluster_descriptions=None,
    cmap='magma',
    center=0.5 ** 0.5,
    cluster_order=True,
    standardize_cox=True,
    filter_below_percentage=0.05,
    non_data_columns = ('survival', 'died', 'simple_stage'),
    show_with_pvalue_lower=0.06,
    row_colors=None,
    hypergeometric_filter_nas=True,
    hypergeometric_fdr_alpha=0.1,
):
    marker_cm, vectors, clusters, clusterer = visualize_patient_clusters_no_lung(
        patients_df=patients_df,
        clusters_to_select=clusters_to_select,
        number_of_clusters=number_of_clusters,
        scale=scale,
        clusterer_type=clusterer_type,
        vectorization_function=vectorization_function,
        method=method,
        metric=metric,
        row_cluster=row_cluster,
        title=title,
        cluster_descriptions=cluster_descriptions,
        cmap=cmap,
        center=center,
        cluster_order=cluster_order,
        row_colors=row_colors,
    )
    
    q_vals = compute_hypergeometrics_q_for_df(
        df=patients_df,
        columns=hypergeometric_columns,
        clusters=clusters,
        values = None,
        add_prefix=True,
        filter_nas=True,
        filter_clusters=hypergeometric_filter_nas,
        min_p_value=1e-5,
        fdr_alpha=hypergeometric_fdr_alpha,
        method='fdr_bh',
        filter_below_percentage=filter_below_percentage,
        plot=True,
    )
    return marker_cm, vectors, clusters, clusterer, q_vals


def visualize_patient_clusters_with_survival_analysis_and_hg(
    patients_df,
    clusters_to_select,
    hypergeometric_columns,
    number_of_clusters: int = 10,
    clusterer_type: str = 'agg',
    scale: float = 0.3,
    vectorization_function = None,
    method: str = 'complete',
    metric: str = 'euclidean',
    row_cluster: bool = False,
    title=None,
    cluster_descriptions=None,
    cmap='magma',
    center=0.5 ** 0.5,
    cluster_order=True,
    standardize_cox=True,
    filter_below_percentage=0.05,
    non_data_columns = ('survival', 'died', 'simple_stage'),
    show_with_pvalue_lower=0.06,
    row_colors=None,
    hypergeometric_filter_nas=True,
    hypergeometric_fdr_alpha=0.1,
):
    marker_cm, vectors, clusters, clusterer = visualize_patient_clusters_with_survival_analysis(
        patients_df=patients_df,
        clusters_to_select=clusters_to_select,
        number_of_clusters=number_of_clusters,
        clusterer_type=clusterer_type,
        scale=scale,
        vectorization_function=vectorization_function,
        method=method,
        metric=metric,
        row_cluster=row_cluster,
        title=title,
        cluster_descriptions=cluster_descriptions,
        cmap=cmap,
        center=center,
        cluster_order=cluster_order,
        standardize_cox=standardize_cox,
        filter_below_percentage=filter_below_percentage,
        non_data_columns=non_data_columns,
        show_with_pvalue_lower=show_with_pvalue_lower,
        row_colors=row_colors,
    )
    
    q_vals = compute_hypergeometrics_q_for_df(
        df=patients_df,
        columns=hypergeometric_columns,
        clusters=clusters,
        values = None,
        add_prefix=True,
        filter_nas=True,
        filter_clusters=hypergeometric_filter_nas,
        min_p_value=1e-5,
        fdr_alpha=hypergeometric_fdr_alpha,
        method='fdr_bh',
        filter_below_percentage=filter_below_percentage,
        plot=True,
    )
    return marker_cm, vectors, clusters, clusterer, q_vals


def visualize_patient_clusters_with_survival_analysis(
    patients_df,
    clusters_to_select,
    number_of_clusters: int = 10,
    clusterer_type: str = 'agg',
    scale: float = 0.3,
    vectorization_function = None,
    method: str = 'complete',
    metric: str = 'euclidean',
    row_cluster: bool = False,
    title=None,
    cluster_descriptions=None,
    cmap='magma',
    center=0.5 ** 0.5,
    cluster_order=True,
    standardize_cox=True,
    filter_below_percentage=0.05,
    non_data_columns = ('survival', 'died', 'simple_stage'),
    show_with_pvalue_lower=0.06,
    row_colors=None,
):
    marker_cm, vectors, clusters, clusterer = visualize_patient_clusters(
            patients_df=patients_df,
            clusters_to_select=clusters_to_select,
            clusterer_type=clusterer_type,
            number_of_clusters=number_of_clusters,
            scale=scale,
            vectorization_function=vectorization_function,
            method=method,
            metric=metric,
            row_cluster=row_cluster,
            title=title,
            cluster_descriptions=cluster_descriptions,
            cmap=cmap,
            center=center,
            cluster_order=cluster_order,
            row_colors=row_colors,
            )
    plt.show()

    plot_cox_trajectory_for_clustering(
            patients_df=patients_df,
            clustering=clusters,
            clustering_name=title,
            standardize=standardize_cox,
            filter_below_percentage=filter_below_percentage,
            non_data_columns=non_data_columns,
            )
    plt.show()

    nb_of_clusters = np.max(clusters) + 1
    for cluster_ind in range(nb_of_clusters):
        visualize_kaplan_meier(
                patients_df,
                clusters,
                clustering_name=title,
                cluster_to_visualize=cluster_ind,
                show_with_pvalue_lower=show_with_pvalue_lower,
                )

    return marker_cm, vectors, clusters, clusterer


def compute_triangle_areas(triangulation):
    """
    Compute areas of all triangles in a matplotlib triangulation
    
    Args:
        triangulation: matplotlib.tri.Triangulation object
        
    Returns:
        numpy array with areas of each triangle
    """
    # Get coordinates and triangles
    x = triangulation.x
    y = triangulation.y
    triangles = triangulation.triangles
    
    # Get vertices for each triangle
    v1 = np.column_stack((x[triangles[:,0]], y[triangles[:,0]]))
    v2 = np.column_stack((x[triangles[:,1]], y[triangles[:,1]]))
    v3 = np.column_stack((x[triangles[:,2]], y[triangles[:,2]]))
    
    # Compute areas using cross product
    # Area = 1/2 * |cross product|
    areas = 0.5 * np.abs(
        (v2[:,0] - v1[:,0]) * (v3[:,1] - v1[:,1]) -
        (v3[:,0] - v1[:,0]) * (v2[:,1] - v1[:,1])
    )
    
    return areas
