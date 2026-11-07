import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

np.random.seed(1)
N_PERM = 20000

df = pd.read_csv("/raid/immucan/immuw/cellohood-analyses/analyses/MIBI_TNBC_EXTERNAL/utag_slda_comparison/all_methods_per_patient_stats.csv")
print("columns:", df.columns.tolist())
print("n rows:", len(df))

raw_cols = {
    "Cellohood (bag)": "cellohood_bag_raw_mixing",
    "UTAG (res 0.3)": "utag_raw__UTAG_leiden_0.3",
    "Spatial-LDA (hard)": "slda_hard_raw_mixing",
}

confound = df["overall_immune_frac"].values

print("\n=== Individual confound correlations (reference) ===")
for name, col in raw_cols.items():
    r, p = pearsonr(df[col], confound)
    print(f"{name}: r_vs_confound={r:.3f}, p={p:.4f}")

print("\n=== Cross-method pairwise correlations: standard t-test p-value AND permutation-test p-value ===")
names = list(raw_cols.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a_name, b_name = names[i], names[j]
        a = df[raw_cols[a_name]].values
        b = df[raw_cols[b_name]].values

        r_obs, p_ttest = pearsonr(a, b)
        rs_obs, ps_ttest = spearmanr(a, b)

        # Permutation test: shuffle b relative to a, break patient pairing, rebuild null
        null_r = np.empty(N_PERM)
        null_rs = np.empty(N_PERM)
        for k in range(N_PERM):
            b_perm = np.random.permutation(b)
            null_r[k] = pearsonr(a, b_perm)[0]
            null_rs[k] = spearmanr(a, b_perm)[0]
        p_perm = (np.abs(null_r) >= abs(r_obs)).mean()
        p_perm_s = (np.abs(null_rs) >= abs(rs_obs)).mean()

        print(f"{a_name} vs {b_name}: Pearson r={r_obs:.3f} (t-test p={p_ttest:.4f}, perm p={p_perm:.4f}) | "
              f"Spearman rs={rs_obs:.3f} (t-test p={ps_ttest:.4f}, perm p={p_perm_s:.4f})")
