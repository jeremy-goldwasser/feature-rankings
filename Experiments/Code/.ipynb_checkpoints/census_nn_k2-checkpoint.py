#%%
import numpy as np
import sys
import pickle

dir_path = "/accounts/grad/jeremy_goldwasser/RankSHAP"
from os.path import join
sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from shapley_sampling2 import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

X_train, y_train, X_test, y_test, mapping_dict = load_data("census", join(dir_path, "Experiments", "Data"))

model = train_neural_net(X_train, y_train)

#%%
np.random.seed(1)
K = 2
N_runs = 500
N_pts = 10

value_diffs_all = []
for i in range(N_pts):
    print(i)
    xloc = X_test[i:(i+1)]
    value_diffs = []
    top_K = []
    while len(top_K) < N_runs:
        shap_vals, diffs_all_feats = shapley_sampling_adaptive(model, X_train, xloc, K=K, alpha=0.2, 
                                mapping_dict=mapping_dict, max_n_perms=20000, n_init=100, n_equal=False)
        if isinstance(shap_vals, np.ndarray):
            #### Compute empirical FWER #####
            est_top_K = get_ordering(shap_vals)[:K]
            top_K.append(est_top_K)
            value_diffs.append(diffs_all_feats)
            if len(top_K) % 20 == 0:
                top_K_arr = np.array(top_K)
                most_common_row = mode_rows(top_K_arr)
                ct = 0
                for idx in range(len(top_K)):
                    if np.array_equal(most_common_row, top_K_arr[idx]):
                        ct += 1
                print(len(top_K), np.round(1 - ct / len(top_K), 2)) # "FWER
    value_diffs_all.append(value_diffs)

    # Store results
    with open(join(dir_path, "Experiments", "Results", "diffs_census_nn_K2_10x"), "wb") as fp:
        pickle.dump(value_diffs_all, fp)

