import numpy as np
import sys
import pickle
import pathlib
import os
from os.path import join
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

max_n_rankshap = 10000
# max_n_sprtshap = 100000
# datasets = ["census", "bank", "brca", "credit", "breast_cancer"]
datasets = ["census", "bank", "credit", "breast_cancer"]

N_samples_all_datasets = []
for dataset in datasets:
    X_train, y_train, X_test, y_test, mapping_dict = load_data.load_data(join(dir_path, "Experiments", "Data"), dataset)
    N_test = y_test.shape[0]
    d = len(mapping_dict) if mapping_dict is not None else X_train.shape[1]

    model = train_models.train_model(X_train, y_train, "nn")

    N_successful_pts = 0
    x_idx = 0
    n_init_per_feature = 100
    n_init_total = 100*d
    N_samples = []
    max_n_sprtshap = max_n_rankshap*d
    while N_successful_pts < N_pts and x_idx < N_test:
        xloc = X_test[x_idx]
        rankshap_vals, _, N_rankshap, rankshap_converged = top_k.rankshap(model, X_train, xloc, mapping_dict=mapping_dict,
                                                K=K, alpha=alpha, guarantee=guarantee,
                                                max_n_perms=max_n_rankshap, 
                                                n_equal=True, n_samples_per_perm=10, 
                                                n_init=n_init_per_feature, abs=True)
        sprtshap_vals, _, N_sprtshap, sprtshap_converged = top_k.sprtshap(model, X_train, xloc, K=K, mapping_dict=mapping_dict, 
                                                guarantee=guarantee,
                                                n_samples_per_perm=10, n_perms_btwn_tests=1000, 
                                                n_max=max_n_sprtshap, alpha=alpha, beta=0.2, abs=True,
                                                n_init=n_init_total)
        print(rankshap_converged, sprtshap_converged)
        if rankshap_converged and sprtshap_converged:
            N_samples.append([N_rankshap, N_sprtshap])
            N_successful_pts += 1
            
        x_idx += 1
        print(x_idx, N_successful_pts)
    N_samples_all_datasets.append(N_samples)
    
    with open(join(results_dir, fname), 'wb') as f:
        np.save(f, N_samples_all_datasets)
