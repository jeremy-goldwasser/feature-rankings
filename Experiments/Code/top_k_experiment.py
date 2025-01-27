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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="rankshap")
parser.add_argument('--dataset', type=str, default="census")
parser.add_argument('--k', type= int, default=2)
parser.add_argument('--algo', type=str, default="nn")
parser.add_argument('--nruns', type=int, default=50)
parser.add_argument('--npts', type=int, default=30)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--guarantee', type=str, default="rank")

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
indices_used = []
if method not in ["rankshap", "sprtshap", "lime"]:
    print("Method must be rankshap, sprtshap, or lime.")
    sys.exit()
isLime = (method=="lime")

if guarantee not in ["rank", "set"]:
    print("Guarantee must be rank or set.")
    sys.exit()

print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data.load_data(join(dir_path, "Experiments", "Data"), dataset)
model = train_models.train_model(X_train, y_train, algo, isLime)
N_test = y_test.shape[0]
max_n_rankshap = 10000
max_n_kernelshap = 50000
max_n_lime = 100000

# np.random.seed(1)
x_idx = 0

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

shap_vals_all = []
shap_vars_all = []
N_samples_all = []
Ns_per_feature_all = []
N_successful_pts = 0
rejection_idx_all = []
while N_successful_pts < N_pts and x_idx < N_test:
    # Don't bother in situations that rarely converge
    if x_idx >= 10 and N_successful_pts/x_idx < 0.1:
        print("Aborting. Too infrequently converging.")
        break
    # print(x_idx)

    xloc = X_test[x_idx]
    top_K = []
    N_samples = []
    Ns_per_feature = []
    shap_vals_i, shap_vars_i = [], []
    N_successful_runs = 0
    # run_idx = 0
    rejection_idx = []
    # while N_successful_runs < N_runs:
    for run_idx in range(N_runs):
        if isLime:
            tol = 1e-4
            if dataset=="breast_cancer" and x_idx==3:
                tol = 1e-5
            exp, converged = explainer.slime(xloc, model, num_features = K, 
                                        num_samples = 1000, n_max = max_n_lime, 
                                        alpha = alpha_adj, tol=tol, return_none=True)
            est_top_K = [pair[0] for pair in list(exp.local_exp.items())[0][1]]
        else:
            if method=="rankshap":
                shap_vals, diffs, N, converged = top_k.rankshap(model, X_train, xloc, mapping_dict=mapping_dict,
                                                      K=K, alpha=alpha, guarantee=guarantee,
                                                      max_n_perms=max_n_rankshap, 
                                                      n_equal=True, n_samples_per_perm=10, 
                                                      n_init=100, abs=True)
                shap_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs)
                N_per_feature = [len(diffs_feat) for diffs_feat in diffs]
                Ns_per_feature.append(N_per_feature)
            else:
                shap_vals, shap_covs, N, converged = top_k.sprtshap(model, X_train, xloc, K=K, mapping_dict=mapping_dict, 
                                                      guarantee=guarantee,
                                                      n_samples_per_perm=10, n_perms_btwn_tests=1000, 
                                                      n_max=max_n_kernelshap, alpha=alpha, beta=0.2, abs=True)
                shap_vars = np.diag(shap_covs)

            shap_vals_i.append(shap_vals)
            shap_vars_i.append(shap_vars)
            N_samples.append(N)
            est_top_K = helper.get_ranking(shap_vals, abs=True)[:K]
            if guarantee=="set":
                est_top_K = np.sort(est_top_K)
        
        top_K.append(est_top_K)
        if converged:
            # Indicate successful run. Will be used to calculate FWER.
            N_successful_runs += 1
            rejection_idx.append(run_idx)
            # if N_successful_runs % 5 == 0 and N_successful_runs > 0 and N_successful_runs!=N_runs:
            #     print(N_successful_runs, helper.calc_fwer(top_K, digits=3, rejection_idx=rejection_idx))
        else:
            N_completed_runs = run_idx + 1
            if N_completed_runs >= 10 and N_successful_runs/N_completed_runs < 0.9: # 0.5
                print(f'Skipping. {N_successful_runs} convergences in {N_completed_runs} runs.')
                break

    # Made it through
    if N_runs==run_idx+1:
        N_successful_pts += 1
        fwer = helper.calc_fwer(top_K, digits=3, rejection_idx=rejection_idx)
        fwers_all.append(fwer)
        indices_used.append(x_idx)
        rejection_idx_all.append(rejection_idx)
        top_K_all.append(top_K)
        
        print(f'FWER {fwer} on pt {N_successful_pts} (idx {x_idx}). {N_successful_runs} convergences in {N_runs} runs')

        # Store results
        top_K_results = {'fwers': np.array(fwers_all), 
                         'top_K': np.array(top_K_all), 
                         'x_indices': np.array(indices_used),
                         'rejection_idx': rejection_idx_all
                         }
        if not isLime:
            N_samples_all.append(N_samples)
            shap_vals_all.append(shap_vals_i)
            shap_vars_all.append(shap_vars_i)
            top_K_results['N_samples'] = np.array(N_samples_all)
            top_K_results['shap_vals'] = np.array(shap_vals_all)
            top_K_results['shap_vars'] = np.array(shap_vars_all)
            if method=="rankshap":
                Ns_per_feature_all.append(Ns_per_feature)
                top_K_results['N_samples_per_feature'] = np.array(Ns_per_feature_all)
        with open(join(results_dir, fname), "wb") as fp:
            pickle.dump(top_K_results, fp)
            
    x_idx += 1
