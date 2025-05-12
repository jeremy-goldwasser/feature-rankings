import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib
from statsmodels.stats.multitest import multipletests
import sys
from os.path import join
path_to_file = str(pathlib.Path().resolve())

import pickle
retro_path = join(path_to_file, "..", "Results", "Retrospective")
alphas = [0.05, 0.1, 0.2]

dir_path = join(path_to_file, "../../")

import matplotlib.colors as mcolors

method = "kernelshap" # "ss"
# Method = "KernelSHAP" # "Shapley Sampling"
dataset = "credit"
with open(join(retro_path, method+"_"+dataset), 'rb') as f:
    retro_results = pickle.load(f)
shap_vals = retro_results["shap_vals"]
shap_vars = retro_results["shap_vars"]

import numpy as np
from scipy.stats import t

all_pvals = []
for a in range(shap_vals.shape[0]):
    print(a)
    for b in range(shap_vals.shape[1]):
    # b = 0
        # Inputs
        means = shap_vals[a,b]
        variances = shap_vars[a,b]
        d = len(means)
        n = (2 * d + 2048) / d

        # Step 1: reorder by absolute value of means
        order = np.argsort(-np.abs(means))
        ordered_means = np.abs(means[order])
        variances = variances[order]

        # Step 2: paired one-sided t-tests
        results = np.zeros((d, d))  # p-values
        for i in range(d):
            for j in range(i + 1, d):
                # Test if |mean_i| > |mean_j| implies mean_i > mean_j
                # if abs(means[i]) > abs(means[j]):
                diff = ordered_means[i] - ordered_means[j]
                se = np.sqrt(variances[i] + variances[j])
                t_stat = diff / se
                p_val = 1 - t.cdf(t_stat, df=n-1)
                results[i, j] = p_val

        # results[i, j] holds p-value for test: mean_i > mean_j, given |mean_i| > |mean_j|
        all_pvals.append(results)
    
N_verified_holm_all = []
for alpha in alphas:
    N_verified_holm = []
    
    print(alpha)
    for a in range(shap_vals.shape[0]):
        print(a)
        for b in range(shap_vals.shape[1]):
            results = all_pvals[a*shap_vals.shape[1]+b]
            # b = 0
            # results = all_pvals[a]
            # Flatten the upper triangle of the results matrix
            pvals = results[np.triu_indices(d, k=1)]

            # Apply Holm's method
            rejects, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='holm')

            # Reshape back to a matrix form
            corrected_results = np.zeros_like(results)
            corrected_pvals = np.zeros_like(pvals)
            corrected_pvals[:] = pvals_corrected
            corrected_results[np.triu_indices(d, k=1)] = corrected_pvals

            # Upper triangular matrix
            p_vals = np.max(corrected_results, axis=1)
            ct = 0
            while p_vals[ct] < alpha:
                ct += 1
            # print("Number of features with p-value < 0.05:", ct)
            N_verified_holm.append(ct)

            # N_verified_holm = N_verified_holm*50
    N_verified_holm_all.append(N_verified_holm)

N_verified_holm_all = np.array(N_verified_holm_all)

# save the Holm-verified counts array
output_path = join(retro_path, f"holm_{method}_{dataset}.npy")
np.save(output_path, N_verified_holm_all)
print(f"Saved Holm results to {output_path}")
