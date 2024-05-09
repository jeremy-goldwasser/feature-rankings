#%%
import numpy as np
import sys
import pickle
import pathlib
from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")

sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from rankshap import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

dataset = "census"
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
d = len(mapping_dict)
model = train_neural_net(X_train, y_train)
# model = train_logreg(X_train, y_train)
np.random.seed(42)
K = int(sys.argv[1])

n_perms = int(sys.argv[2]) if len(sys.argv)==3 else None
N_runs = 50
N_pts = 30
fwers = []
top_K_ss = []

x_idx = 0
n_successful = 0
N_runs_rs = 3
# for x_idx in range(N_pts):
while n_successful < N_pts:
    # print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    x_idx += 1
    top_K = []

    if n_perms is None:
        diffs_arr = []
        conv_arr = []
        for i in range(N_runs_rs):
            _, diffs, converged = rankshap(model, X_train, xloc, K=K, alpha=0.2, 
                                    mapping_dict=mapping_dict, max_n_perms=10000, n_init=100, n_equal=False)
            diffs_arr.append(diffs)
            conv_arr.append(converged)
        if np.sum(conv_arr) < N_runs_rs/2:
            continue
        else:
            avg_length = int(np.mean([np.mean([len(diffs[j]) for j in range(d)]) for diffs in diffs_arr]))
            # print(avg_length)


    n_init = avg_length if n_perms is None else n_perms
    for i in range(N_runs):
        shap_vals = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=n_init)
        est_top_K = get_ranking(shap_vals)[:K]
        top_K.append(est_top_K)
    n_successful += 1
    print(x_idx, n_successful, n_init, calc_fwer(top_K))
    top_K_ss.append(top_K)
    
    tail = "_adaptive" if n_perms is None else str(n_perms)
    fname = "ss_ranks_k" + str(K) + "_n" + tail
    with open(join(dir_path, "Experiments", "Results", "alpha0.2", fname), "wb") as fp:
        pickle.dump(top_K_ss, fp)
