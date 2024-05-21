import numpy as np
import sys
import pickle
import pathlib
import os
path_to_file = str(pathlib.Path().resolve())
dir_path = os.path.join(path_to_file, "../../")
from slime import lime_tabular

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
parser.add_argument('--k', type= int, default=3)
parser.add_argument('--algo', type=str, default="nn")
parser.add_argument('--nruns', type=int, default=100)
parser.add_argument('--npts', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.2)

args = parser.parse_args() 
print(args)

method = args.method
dataset = args.dataset
K = args.k
algo = args.algo
N_runs = args.nruns
N_pts = args.npts
alpha = args.alpha

fname = method + "_" + dataset + "_K" + str(K) + "_fwers"
fname2 = method + "_" + dataset + "_K" + str(K) + "_ranks"
fname3 = method + "_" + dataset + "_K" + str(K) + "_samples"
isLime = (method=="lime")
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(os.path.join(dir_path, "Experiments", "Data"), dataset)
model = train_model(X_train, y_train, algo, isLime)
N_test = y_test.shape[0]
max_n_rankshap = 10000
# max_n_kernelshap = 200000
max_n_kernelshap = 50000
max_n_lime = 100000 # 500000

np.random.seed(0)
x_idx = 0
skip_thresh = 0.75 # Skip if successful with frequency below skip_thresh 

top_K_all = []
fwers = {}
results_path = os.path.join(dir_path, "Experiments", "Results", "alpha"+str(alpha))
if not os.path.exists(results_path): os.makedirs(results_path)
if isLime:
    explainer = lime_tabular.LimeTabularExplainer(X_train, 
                                              discretize_continuous = False, 
                                              feature_selection = "lasso_path", 
                                              sample_around_instance = True)
    alpha_adj = alpha/K/2
N_samples = []
while len(fwers) < N_pts and x_idx < N_test:
    # Don't bother in situations that rarely converge
    if x_idx >= 10 and len(fwers)/x_idx < 0.1:
        print("Aborting. Too infrequently converging.")
        break
    print(x_idx)
    xloc = X_test[x_idx] if isLime else X_test[x_idx:(x_idx+1)]
    shap_vals_all = []
    top_K = []
    count = 0
    Ns = []
    while len(top_K) < N_runs:
        if method=="lime":
            try:
                if dataset=="credit" and x_idx==19 and K==5: 
                    tol = 1e-5 # Get closer to true algorithm
                else:
                    tol = 1e-4
                exp = explainer.slime(xloc, model, num_features = K, 
                                            num_samples = 1000, n_max = max_n_lime, 
                                            alpha = alpha_adj, tol=tol, return_none=True) # not 1e-4
                if exp is not None:
                    # est_top_K = extract_lime_feats(exp, K, mapping_dict)
                    # Don't see a good way to get back to feature space.
                    est_top_K = [pair[0] for pair in list(exp.local_exp.items())[0][1]]
                    converged = True
                else:
                    converged = False
            except:
                converged = False
        else:
            if method=="rankshap":
                shap_vals, _, N, converged = rankshap(model, X_train, xloc, K=K, alpha=alpha, 
                                        mapping_dict=mapping_dict, max_n_perms=max_n_rankshap, 
                                        n_samples_per_perm=10, n_init=100, n_equal=False)
            elif method=="kernelshap":
                # print(len(top_K), count)
                shap_vals, N, converged = kernelshap_top_k(model, X_train, xloc, K=K, mapping_dict=mapping_dict, 
                    n_samples_per_perm=10, n_perms_btwn_tests=500, n_max=max_n_kernelshap, 
                    alpha=alpha, beta=0.2, abs=True)
            else:
                print("Name must be lime, rankshap, or kernelshap.")
            if converged:
                est_top_K = get_ranking(shap_vals)[:K]
                Ns.append(N)
        if converged:
            top_K.append(est_top_K)
            
        count += 1
        num_successes = len(top_K)
        if not converged:
            if count >= 5 and num_successes/count < skip_thresh:
                break
        else:
            if num_successes % 25 == 0 and num_successes > 0 and num_successes!=N_runs:
                print(num_successes, calc_fwer(top_K))
            
    if len(top_K)==N_runs:
        fwer = calc_fwer(top_K)
        fwers[x_idx] = fwer
        top_K_all.append(top_K)
        N_samples.append(Ns)
        print("#"*20, len(fwers), fwer, " (idx ", x_idx, ") ", "#"*20)
        # Store results
        with open(os.path.join(results_path, fname), "wb") as fp:
            pickle.dump(fwers, fp)
        with open(os.path.join(results_path, fname2), "wb") as fp:
            pickle.dump(top_K_all, fp)
        if method != "lime":
            with open(os.path.join(results_path, fname3), "wb") as fp:
                pickle.dump(np.array(N_samples), fp)
    x_idx += 1
