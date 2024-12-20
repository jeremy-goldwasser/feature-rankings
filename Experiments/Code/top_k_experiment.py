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
from helper import *
from helper_shapley_sampling import *
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
parser.add_argument('--k', type= int, default=3)
parser.add_argument('--algo', type=str, default="nn")
parser.add_argument('--nruns', type=int, default=50) #100
parser.add_argument('--npts', type=int, default=30) #10
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--guarantee', type=str, default="ranks")

args = parser.parse_args() 
print(args)

method = args.method
dataset = args.dataset
K = args.k
algo = args.algo
N_runs = args.nruns
N_pts = args.npts
alpha = args.alpha

guarantee = args.guarantee

fname = method + "_" + dataset + "_K" + str(K)
# fname = method + "_" + dataset + "_K" + str(K) + "_fwers"
# fname2 = method + "_" + dataset + "_K" + str(K) + "_ranks"
# fname3 = method + "_" + dataset + "_K" + str(K) + "_samples"
indices_used = []
if method not in ["rankshap", "sprtshap", "lime"]:
    print("Method must be rankshap, sprtshap, or lime.")
    sys.exit()
isLime = (method=="lime")

if guarantee not in ["rank", "set"]:
    print("Guarantee must be rank or set.")
    sys.exit()

print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
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
fwers_all = []
results_dir = join(dir_path, "Experiments", "Results", "Top_K", guarantee, "alpha_"+str(alpha))
os.makedirs(results_dir, exist_ok=True)
if isLime:
    explainer = lime_tabular.LimeTabularExplainer(X_train, 
                                              discretize_continuous = False, 
                                              feature_selection = "lasso_path", 
                                              sample_around_instance = True)
    alpha_adj = alpha/K/2

shap_values_all = []
shap_vars_all = []
N_samples_all = []
N_successful_pts = 0
while N_successful_pts < N_pts and x_idx < N_test:
    # Don't bother in situations that rarely converge
    if x_idx >= 10 and len(fwers)/x_idx < 0.1:
        print("Aborting. Too infrequently converging.")
        break
    print(x_idx)
    xloc = X_test[x_idx]
    shap_vals_all = []
    top_K = []
    count = 0
    N_samples = []
    shap_vals_i, shap_vars_i = [], []
    while len(top_K) < N_runs:
        if isLime:
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
                shap_vals, diffs, N, converged = rankshap(model, X_train, xloc, mapping_dict=mapping_dict,
                                                      K=K, alpha=alpha, guarantee=guarantee,
                                                      max_n_perms=max_n_rankshap, 
                                                      n_equal=False, n_samples_per_perm=10, 
                                                      n_init=100, abs=True)
                shap_vars = diffs_to_shap_vars(diffs)
            else:
                shap_vals, shap_covs, N, converged = sprtshap(model, X_train, xloc, K=K, mapping_dict=mapping_dict, 
                                                      guarantee=guarantee,
                                                      n_samples_per_perm=10, n_perms_btwn_tests=1000, 
                                                      n_max=max_n_kernelshap, alpha=alpha, beta=0.2, abs=True)
                shap_vars = np.diag(shap_covs)

            if converged:
                est_top_K = get_ranking(shap_vals)[:K]
                if guarantee=="set":
                    est_top_K = np.sort(est_top_K)
                N_samples.append(N)
                shap_vals_i.append(shap_vals)
                shap_vars_i.append(shap_vars)
        if converged:
            top_K.append(est_top_K)
            
        count += 1
        N_successful_runs = len(top_K)
        if not converged:
            if count >= 5 and N_successful_runs/count < skip_thresh:
                break
        else:
            if N_successful_runs % 25 == 0 and N_successful_runs > 0 and N_successful_runs!=N_runs:
                print(N_successful_runs, calc_fwer(top_K, digits=3))
            
    if len(top_K)==N_runs: # Made it through
        N_successful_pts += 1
        fwer = calc_fwer(top_K, digits=3)
        fwers_all.append(fwer)
        indices_used.append(x_idx)
        top_K_all.append(top_K)
        
        print("#"*20, len(fwers), fwer, " (idx ", x_idx, ") ", "#"*20)

        # Store results
        top_K_results = {'fwers': np.array(fwers_all), 'ranks': np.array(top_K_all), 'x_indices': np.array(indices_used)}
        if not isLime:
            N_samples_all.append(N_samples)
            shap_vals_all.append(shap_vals_i)
            shap_vars_all.append(shap_vars_i)
            top_K_results['N_samples'] = np.array(N_samples)
            top_K_results['shap_vals'] = np.array(shap_vals_all)
            top_K_results['shap_vars'] = np.array(shap_vars_all)
        with open(join(results_dir, fname), "wb") as fp:
            pickle.dump(top_K_results, fp)
            
    x_idx += 1
