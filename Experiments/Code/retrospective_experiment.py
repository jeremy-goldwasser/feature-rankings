import numpy as np
import sys
import pickle
import pathlib
import os
path_to_file = str(pathlib.Path().resolve())
dir_path = os.path.join(path_to_file, "../../")

sys.path.append(os.path.join(dir_path, "HelperFiles"))
from helper import *
from top_k import *
from retrospective import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="shap")
parser.add_argument('--dataset', type=str, default="census")
parser.add_argument('--algo', type=str, default="nn")
parser.add_argument('--nruns', type=int, default=50) #250
parser.add_argument('--npts', type=int, default=30) #10

args = parser.parse_args() 
print(args)

method = args.method
dataset = args.dataset
algo = args.algo
N_runs = args.nruns
N_pts = args.npts

fname = method + "_" + dataset
# fname = method + "_" + dataset + "_shap_vals"
# fname2 = method + "_" + dataset + "_N_verified"
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(os.path.join(dir_path, "Experiments", "Data"), dataset)
model = train_model(X_train, y_train, algo, False)
N_test = y_test.shape[0]

np.random.seed(0)
x_idx = 0
alphas = [0.05, 0.1, 0.2]
d = len(mapping_dict) if mapping_dict is not None else X_train.shape[1]

results_dir = os.path.join(dir_path, "Experiments", "Results", "Retrospective")
if not os.path.exists(results_dir): os.makedirs(results_dir)
shap_vals_all = []
N_verified_all = []
shap_vars_all = []

N_samples = 2*d + 2048
# while len(fwers) < N_pts and x_idx < N_test:
for x_idx in range(N_pts):
    print(x_idx)
    xloc = X_test[x_idx]
    shap_vals_pt = []
    shap_vars_pt = []
    Ns = []
    for i in range(N_runs):
        if method=="ss":
            shap_vals, n_verified, shap_vars = shapley_sampling(model, X_train, xloc, n_perms=N_samples//d, 
                                                    n_samples_per_perm=10, mapping_dict=mapping_dict, 
                                                    alphas=alphas, abs=True)
        elif method=="kernelshap":
            shap_vals, n_verified, kshap_covs = kernelshap(model, X_train, xloc, n_perms=N_samples, 
                                                            n_samples_per_perm=10, mapping_dict=mapping_dict,
                                                            alphas=alphas, abs=True)
            shap_vars = np.diag(kshap_covs)
        else:
            print("Name must be ss or kernelshap.")
        if i==0: print(n_verified)
        shap_vals_pt.append(shap_vals)
        shap_vars_pt.append(shap_vars)
        Ns.append(n_verified)
    shap_vals_all.append(shap_vals_pt)
    shap_vars_all.append(shap_vars_pt)
    N_verified_all.append(Ns)
            
    # Store results
    retro_results = {"shap_vals": np.array(shap_vals_all), 
                     "N_verified": np.array(N_verified_all), 
                     "shap_vars": np.array(shap_vars_all)}
    with open(os.path.join(results_dir, fname), "wb") as fp:
        pickle.dump(retro_results, fp)
