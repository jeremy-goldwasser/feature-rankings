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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type= str, default="census")
parser.add_argument('--k', type= int, default=2)
args = parser.parse_args() 
dataset = args.dataset
K = args.k
# dataset = sys.argv[1]
# K = int(sys.argv[2])

fname = "shap_vals_K" + str(K)
fname2 = "shap_fwers_K" + str(K)
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
print(len(y_test))
model = train_neural_net(X_train, y_train)

#%%
np.random.seed(1)
N_runs = 250
N_pts = 30
x_idx = 0
skip_thresh = 0.5

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
            num_successes = len(top_K)
            if count > 10 and num_successes/count < skip_thresh:
                break
    if len(top_K)==N_runs:
        fwer = calc_fwer(top_K)
        fwers.append(fwer)
        shap_vals_all_pts.append(shap_vals_all)
        print("#"*20, len(fwers), fwer, "#"*20)
    x_idx += 1

    # Store results
    with open(join(dir_path, "Experiments", "Results", "alpha0.2", fname), "wb") as fp:
        pickle.dump(shap_vals_all_pts, fp)
    with open(join(dir_path, "Experiments", "Results", "alpha0.2", fname2), "wb") as fp:
        pickle.dump(fwers, fp)


# %%
