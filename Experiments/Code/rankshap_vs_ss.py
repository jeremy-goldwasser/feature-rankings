#%%
import numpy as np
import sys
import pickle
import pathlib
import os
from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")

sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from retrospective import *
from top_k import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

dataset = "census"
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
d = len(mapping_dict)
model = train_model(X_train, y_train, model="nn", lime=False)
# model = train_logreg(X_train, y_train)
np.random.seed(42)
K = int(sys.argv[1])
print(f"K={K}")

skip_thresh = 0.5
alpha = 0.2
N_runs = 50
N_pts = 30

fwers = []
N_samples_fixed = 500
top_K_rankshap_all = []
top_K_ss_fixed_all = []
top_K_ss_adaptive_all = []
n_samples_rankshap_all = []
x_idx = 0
n_successful_pts = 0

output_dir = join(dir_path, "Experiments", "Results", "Top_K", "alpha_"+str(alpha))
if not os.path.exists(output_dir): os.makedirs(output_dir)
fname_rankshap = "rankshap_top_k" + str(K)
fname_ss_adaptive = "ss_top_k" + str(K) + "_n_adaptive"
fname_ss_fixed = "ss_top_k" + str(K) + "_n" + str(N_samples_fixed)
fname_rankshap_samples = "rankshap_n_samples_k" + str(K)
while n_successful_pts < N_pts:
    # print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    x_idx += 1
    top_K = []
    
    n_samples_all_runs = []
    top_K_rankshap = []
    count, N_successful_runs = 0, 0
    while N_successful_runs < N_runs:
        rankshap_vals, diffs, N, converged = rankshap(model, X_train, xloc, K=K, alpha=alpha, 
                                        mapping_dict=mapping_dict, max_n_perms=10000, n_equal=False,
                                        abs=True)
        
        # Only consider inputs on which RankSHAP is capable of K rejections.
        count += 1
        if converged: 
            N_successful_runs += 1
            est_top_K = get_ranking(rankshap_vals)[:K]
            top_K_rankshap.append(est_top_K)
            n_samples_all_runs.append([len(diffs_feat) for diffs_feat in diffs])
        if count >= 5 and N_successful_runs/count < skip_thresh:
            break
    if N_successful_runs < N_runs:
        continue
    
    # RankSHAP consistently converged on this input x. 
    n_successful_pts += 1
    top_K_rankshap_all.append(top_K_rankshap)
    avg_samples_per_feat = int(np.mean(n_samples_all_runs))
    n_samples_rankshap_all.append(n_samples_all_runs)
    print(f"Successful run {x_idx}, {n_successful_pts} attempts")
    print(f"RankSHAP, average number of samples per feature: {avg_samples_per_feat}")
    print(f"FWER, RankSHAP: {calc_fwer(top_K_rankshap, digits=3)}")

    # Run Shapley Sampling
    top_K_ss_adaptive, top_K_ss_fixed = [], []
    for i in range(N_runs):
        shap_vals_fixed = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=N_samples_fixed)
        est_top_K = get_ranking(shap_vals_fixed)[:K]
        top_K_ss_fixed.append(est_top_K)

        shap_vals_adaptive = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=avg_samples_per_feat)
        est_top_K = get_ranking(shap_vals_adaptive)[:K]
        top_K_ss_adaptive.append(est_top_K)
    
    top_K_ss_fixed_all.append(top_K_ss_fixed)
    top_K_ss_adaptive_all.append(top_K_ss_adaptive)
    
    print(f"FWER, Shapley Sampling (fixed N={N_samples_fixed}): {calc_fwer(top_K_ss_fixed, digits=3)}")
    print(f"FWER, Shapley Sampling (adaptive N={avg_samples_per_feat}): {calc_fwer(top_K_ss_adaptive, digits=3)}")
        
    with open(join(output_dir, fname_rankshap), "wb") as fp:
        pickle.dump(top_K_rankshap_all, fp)
    with open(join(output_dir, fname_ss_fixed), "wb") as fp:
        pickle.dump(top_K_ss_fixed_all, fp)
    with open(join(output_dir, fname_ss_adaptive), "wb") as fp:
        pickle.dump(top_K_ss_adaptive_all, fp)
    with open(join(output_dir, fname_rankshap_samples), "wb") as fp:
        pickle.dump(n_samples_rankshap_all, fp)
