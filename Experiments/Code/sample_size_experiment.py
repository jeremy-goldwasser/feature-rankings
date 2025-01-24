import numpy as np
import sys
import pickle
import pathlib
import os
from os.path import join, exists
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")
from slime import lime_tabular

sys.path.append(join(dir_path, "HelperFiles"))
import helper
import helper_shapley_sampling
import top_k
import train_models
import load_data

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

N_pts = 30
K = 2
guarantee = "rank"
alpha = 0.1
results_dir = join(dir_path, "Experiments", "Results", "Top_K", guarantee, "alpha_"+str(alpha))
fname = "sample_size_comparison.npy"

# if exists(join(results_dir, fname)):
#     with open(join(results_dir, fname), 'rb') as f:
#         N_samples_all_datasets = list(np.load(f))
# else:
#     N_samples_all_datasets = []

N_samples_all_datasets = []
datasets = ["census", "bank", "brca", "credit", "breast_cancer"]

max_n_rankshap = 10000
for dataset in datasets:
    X_train, y_train, X_test, y_test, mapping_dict = load_data.load_data(join(dir_path, "Experiments", "Data"), dataset)
    N_max_pts = y_test.shape[0] if dataset!="brca" else y_train.shape[0]
    d = len(mapping_dict) if mapping_dict is not None else X_train.shape[1]

    model = train_models.train_model(X_train, y_train, "nn")

    N_successful_pts = 0
    x_idx = 0
    n_init_per_feature = 100
    n_init_total = 100*d
    N_samples = []
    max_n_sprtshap = min(max_n_rankshap*d, 20000)
    while N_successful_pts < N_pts and x_idx < N_max_pts:
        if x_idx > 0:
            print(x_idx, N_successful_pts)
        xloc = X_test[x_idx] if dataset!="brca" else X_train[x_idx]
        x_idx += 1
        sprtshap_vals, _, N_sprtshap, sprtshap_converged = top_k.sprtshap(model, X_train, xloc, K=K, mapping_dict=mapping_dict, 
                                                guarantee=guarantee,
                                                n_samples_per_perm=10, n_perms_btwn_tests=1000, 
                                                n_max=max_n_sprtshap, alpha=alpha, beta=0.2, abs=True,
                                                n_init=n_init_total)
        if not sprtshap_converged:
            continue
        rankshap_vals, _, N_rankshap, rankshap_converged = top_k.rankshap(model, X_train, xloc, mapping_dict=mapping_dict,
                                                K=K, alpha=alpha, guarantee=guarantee,
                                                max_n_perms=max_n_rankshap, 
                                                n_equal=True, n_samples_per_perm=10, 
                                                n_init=n_init_per_feature, abs=True)
        if not rankshap_converged:
            continue
        # print(rankshap_converged, sprtshap_converged)
        # if rankshap_converged and sprtshap_converged:
        N_samples.append([N_rankshap, N_sprtshap])
        N_successful_pts += 1
        
    N_samples_all_datasets.append(N_samples)
    
    with open(join(results_dir, fname), 'wb') as f:
        np.save(f, N_samples_all_datasets)
