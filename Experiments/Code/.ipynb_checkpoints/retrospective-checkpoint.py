import numpy as np
import sys
import pickle
import pathlib
import os
path_to_file = str(pathlib.Path().resolve())
dir_path = os.path.join(path_to_file, "../../")

sys.path.append(os.path.join(dir_path, "HelperFiles"))
from helper import *
from rankshap import *
from kernelshap import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="shap")
parser.add_argument('--dataset', type=str, default="census")
parser.add_argument('--algo', type=str, default="nn")
parser.add_argument('--nruns', type=int, default=250)
parser.add_argument('--npts', type=int, default=10)

args = parser.parse_args() 
print(args)

method = args.method
dataset = args.dataset
algo = args.algo
N_runs = args.nruns
N_pts = args.npts

fname = method + "_" + dataset + "_all_ranks"
fname2 = method + "_" + dataset + "_N_verified"
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(os.path.join(dir_path, "Experiments", "Data"), dataset)
model = train_model(X_train, y_train, algo, False)
N_test = y_test.shape[0]

np.random.seed(0)
x_idx = 0
alphas = [0.05, 0.1, 0.2]
d = len(mapping_dict) if mapping_dict is not None else X_train.shape[1]

top_K_all = []
results_path = os.path.join(dir_path, "Experiments", "Results", "Retrospective")
if not os.path.exists(results_path): os.makedirs(results_path)
N_verified_all = []
N_samples = 2*d + 2048
# while len(fwers) < N_pts and x_idx < N_test:
for x_idx in range(N_pts):
    print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    shap_vals_all = []
    top_K = []
    Ns = []
    for i in range(N_runs):
        if method=="ss":
            shap_vals, n_verified = shapley_sampling(model, X_train, xloc, n_perms=N_samples//d, 
                    n_samples_per_perm=10, mapping_dict=mapping_dict, 
                    alphas=alphas, abs=True)
        elif method=="kernelshap":
            shap_vals, n_verified = kernelshap(model, X_train, xloc, n_perms=N_samples, 
                    n_samples_per_perm=10, mapping_dict=mapping_dict,
                    alphas=alphas, abs=True)
        else:
            print("Name must be ss or kernelshap.")
        if i==0: print(n_verified)
        est_top_K = get_ranking(shap_vals, abs=True)
        Ns.append(n_verified)
        top_K.append(est_top_K)
    top_K_all.append(top_K)
    N_verified_all.append(Ns)
            
    # Store results
    with open(os.path.join(results_path, fname), "wb") as fp:
        pickle.dump(np.array(top_K_all), fp)
    with open(os.path.join(results_path, fname2), "wb") as fp:
        pickle.dump(np.array(N_verified_all), fp)
