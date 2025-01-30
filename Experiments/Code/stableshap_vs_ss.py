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
np.random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type= int, default=2)
parser.add_argument('--alpha', type=float, default=0.2)
args = parser.parse_args() 
K = args.k
alpha = args.alpha
print(f"K={K}, alpha={alpha}")

skip_thresh = 0.25
N_runs = 50
N_pts = 30

fwers = []
top_K_stableshap_all = []
top_K_ss_adaptive_all = []
N_samples_stableshap_all = []
indices_used = []
x_idx = 0
N_successful_pts = 0

output_dir = join(dir_path, "Experiments", "Results", "Top_K", "rank", "alpha_"+str(alpha))
os.makedirs(output_dir, exist_ok=True)
fname = 'stableshap_vs_ss_k' + str(K)

successful_iters_all = []
while N_successful_pts < N_pts:
    xloc = X_test[x_idx]
    x_idx += 1
    top_K = []
    
    N_samples_all_runs = []
    top_K_stableshap = []
    successful_iters = []
    for i in range(N_runs):
        stableshap_vals, diffs, N, converged = stableshap(model, X_train, xloc, mapping_dict=mapping_dict, 
                                                      K=K, alpha=alpha, n_equal=True, guarantee='rank', 
                                                      max_n_perms=10000, abs=True)

        # Store Shapley estimates, top-K ranking, and number of samples
        est_top_K = get_ranking(stableshap_vals, abs=True)[:K]
        top_K_stableshap.append(est_top_K)
        N_samples_all_runs.append([len(diffs_feat) for diffs_feat in diffs])
        if converged:
            successful_iters.append(i)
        # Doesn't consider inputs on which StableSHAP is rarely capable of K rejections.
        if (i+1) >= 5 and len(successful_iters)/(i+1) < skip_thresh:
            break
    if len(successful_iters) < N_runs:
        continue
    
    # StableSHAP (presumably) controlled FWER on this input x. 
    indices_used.append(x_idx-1)
    N_successful_pts += 1
    top_K_stableshap_all.append(top_K_stableshap)
    avg_samples_per_feat = int(np.mean(N_samples_all_runs))
    N_samples_stableshap_all.append(N_samples_all_runs)
    print("#"*20)
    print(f"Successful run {N_successful_pts}, {x_idx} attempts")
    print(f"FWER, StableSHAP: {calc_fwer(top_K_stableshap, digits=3, rejection_idx=successful_iters)}")

    # Run Shapley Sampling
    top_K_ss_adaptive = [] 
    for i in range(N_runs):
        shap_vals_adaptive = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=avg_samples_per_feat)
        est_top_K = get_ranking(shap_vals_adaptive, abs=True)[:K]
        top_K_ss_adaptive.append(est_top_K)
    
    top_K_ss_adaptive_all.append(top_K_ss_adaptive)
    successful_iters_all.append(successful_iters)
    
    print(f"FWER, Shapley Sampling (adaptive N={avg_samples_per_feat}): {calc_fwer(top_K_ss_adaptive, digits=3)}\n")
    all_results = {'stableshap': top_K_stableshap_all, 'stableshap_rejection_idx': np.array(successful_iters_all),
                   'ss_adaptive': np.array(top_K_ss_adaptive_all), 
                   'stableshap_n_samples': np.array(N_samples_stableshap_all), 'x_indices': np.array(indices_used)}
    with open(join(output_dir, fname), "wb") as fp:
        pickle.dump(all_results, fp)
