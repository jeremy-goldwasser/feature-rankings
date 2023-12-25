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

dataset = sys.argv[1]
K = int(sys.argv[2])
# fname = "shap_" + dataset + "_nn_K" + str(K) + "_10x"
fname = "shap_vals_k" + str(K)
fname2 = "shap_fwers_k" + str(K)
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
print(len(y_test))
model = train_neural_net(X_train, y_train)

#%%
np.random.seed(1)
N_runs = 250
N_pts = 30
x_idx = 0
skip_thresh = 0.2

shap_vals_all_pts = []
fwers = []

# for i in range(N_pts):
while len(fwers) < N_pts:
    print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    shap_vals_all = []
    top_K = []
    count = 0
    while len(top_K) < N_runs:
        count += 1
        shap_vals, _, converged = rankshap(model, X_train, xloc, K=K, alpha=0.2, 
                                mapping_dict=mapping_dict, max_n_perms=10000, 
                                n_init=100, n_equal=False)
        if converged:
            est_top_K = get_ranking(shap_vals)[:K]
            top_K.append(est_top_K)
            shap_vals_all.append(shap_vals)
            if len(top_K) % 50 == 0:
                print(len(top_K), calc_fwer(top_K))
        else:
            # print("failed to converge; " + str(len(top_K)))
            num_successes = len(top_K)
            if count > 10 and num_successes/count < skip_thresh:
                break
    shap_vals_all_pts.append(shap_vals_all)
    if len(top_K)==N_runs:
        fwer = calc_fwer(top_K)
        print("#"*20, len(top_K), fwer, "#"*20)
        fwers.append(fwer)
    x_idx += 1

    # Store results
    with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
        pickle.dump(shap_vals_all_pts, fp)
    with open(join(dir_path, "Experiments", "Results", fname2), "wb") as fp:
        pickle.dump(fwers, fp)


# %%
