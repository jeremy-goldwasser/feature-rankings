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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type= int, default=2)
parser.add_argument('--alpha', type=float, default=0.2)
args = parser.parse_args() 
K = args.k
alpha = args.alpha
# K = int(sys.argv[1])
print(f"K={K}, alpha={alpha}")

skip_thresh = 0.25
# alpha = 0.2
N_runs = 50
N_pts = 30

fwers = []
# N_samples_fixed = 500
top_K_rankshap_all = []
# top_K_ss_fixed_all = []
top_K_ss_adaptive_all = []
N_samples_rankshap_all = []
indices_used = []
x_idx = 0
N_successful_pts = 0

output_dir = join(dir_path, "Experiments", "Results", "Top_K", "rank", "alpha_"+str(alpha))
os.makedirs(output_dir, exist_ok=True)
fname = 'rankshap_vs_ss_k' + str(K)

successful_iters_all = []
while N_successful_pts < N_pts:
    xloc = X_test[x_idx]
    x_idx += 1
    top_K = []
    
    N_samples_all_runs = []
    top_K_rankshap = []
    # count, N_successful_runs = 0, 0
    # while N_successful_runs < N_runs:
    #     rankshap_vals, diffs, N, converged = rankshap(model, X_train, xloc, mapping_dict=mapping_dict, 
    #                                                   K=K, alpha=alpha, n_equal=True, guarantee='rank', 
    #                                                   max_n_perms=10000, abs=True)
        
    #     # Only consider inputs on which RankSHAP is capable of K rejections.
    #     count += 1
    #     if converged: 
    #         N_successful_runs += 1
    #         est_top_K = get_ranking(rankshap_vals, abs=True)[:K]
    #         top_K_rankshap.append(est_top_K)
    #         N_samples_all_runs.append([len(diffs_feat) for diffs_feat in diffs])
    #     if count >= 5 and N_successful_runs/count < skip_thresh:
    #         break
    # if N_successful_runs < N_runs:
    #     continue
    # count, N_successful_runs = 0, 0
    successful_iters = []
    for i in range(N_runs):
        rankshap_vals, diffs, N, converged = rankshap(model, X_train, xloc, mapping_dict=mapping_dict, 
                                                      K=K, alpha=alpha, n_equal=True, guarantee='rank', 
                                                      max_n_perms=10000, abs=True)
        
        # if converged: 
        #     N_successful_runs += 1
        #     est_top_K = get_ranking(rankshap_vals, abs=True)[:K]

        # Store Shapley estimates, top-K ranking, and number of samples
        est_top_K = get_ranking(rankshap_vals, abs=True)[:K]
        top_K_rankshap.append(est_top_K)
        N_samples_all_runs.append([len(diffs_feat) for diffs_feat in diffs])
        if converged:
            successful_iters.append(i)
        # Doesn't consider inputs on which RankSHAP is rarely capable of K rejections.
        if (i+1) >= 5 and len(successful_iters)/(i+1) < skip_thresh:
            break
    if len(successful_iters) < N_runs:
        continue
    
    # RankSHAP (presumably) controlled FWER on this input x. 
    indices_used.append(x_idx-1)
    N_successful_pts += 1
    top_K_rankshap_all.append(top_K_rankshap)
    avg_samples_per_feat = int(np.mean(N_samples_all_runs))
    N_samples_rankshap_all.append(N_samples_all_runs)
    print("#"*20)
    print(f"Successful run {N_successful_pts}, {x_idx} attempts")
    print(f"RankSHAP, average number of samples per feature: {avg_samples_per_feat}")
    print(f"FWER, RankSHAP: {calc_fwer(top_K_rankshap, digits=3, rejection_idx=successful_iters)}")

    # Run Shapley Sampling
    top_K_ss_adaptive = [] 
    # top_K_ss_fixed = []
    for i in range(N_runs):
        # shap_vals_fixed = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=N_samples_fixed)
        # est_top_K = get_ranking(shap_vals_fixed, abs=True)[:K]
        # top_K_ss_fixed.append(est_top_K)

        shap_vals_adaptive = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=avg_samples_per_feat)
        est_top_K = get_ranking(shap_vals_adaptive, abs=True)[:K]
        top_K_ss_adaptive.append(est_top_K)
    
    # top_K_ss_fixed_all.append(top_K_ss_fixed)
    top_K_ss_adaptive_all.append(top_K_ss_adaptive)
    successful_iters_all.append(successful_iters)
    
    # print(f"FWER, Shapley Sampling (fixed N={N_samples_fixed}): {calc_fwer(top_K_ss_fixed, digits=3)}")
    print(f"FWER, Shapley Sampling (adaptive N={avg_samples_per_feat}): {calc_fwer(top_K_ss_adaptive, digits=3)}\n")
    all_results = {'rankshap': top_K_rankshap_all, 'rankshap_rejection_idx': np.array(successful_iters_all),
                   'ss_adaptive': np.array(top_K_ss_adaptive_all), 
                #    'ss_fixed': top_K_ss_fixed_all, 
                   'rankshap_n_samples': np.array(N_samples_rankshap_all), 'x_indices': np.array(indices_used)}
    with open(join(output_dir, fname), "wb") as fp:
        pickle.dump(all_results, fp)
