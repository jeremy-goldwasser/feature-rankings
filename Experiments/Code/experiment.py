import numpy as np
import sys
import pickle
import pathlib
from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")
from slime import lime_tabular

sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from rankshap import *
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
isLime = (method=="lime")
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
model = train_model(X_train, y_train, algo, isLime)
N_test = y_test.shape[0]

np.random.seed(1)
x_idx = 0
skip_thresh = 0.5

fwers = {}
if isLime:
    explainer = lime_tabular.LimeTabularExplainer(X_train, 
                                              discretize_continuous = False, 
                                              feature_selection = "lasso_path", 
                                              sample_around_instance = True)
    alpha_adj = alpha/K/2
while len(fwers) < N_pts and x_idx < N_test:
    print(x_idx)
    xloc = X_test[x_idx] if isLime else X_test[x_idx:(x_idx+1)]
    shap_vals_all = []
    top_K = []
    count = 0
    while len(top_K) < N_runs:
        if method=="shap":
            shap_vals, _, converged = rankshap(model, X_train, xloc, K=K, alpha=alpha, 
                                    mapping_dict=mapping_dict, max_n_perms=10000, 
                                    n_init=100, n_equal=False)
            if converged:
                est_top_K = get_ranking(shap_vals)[:K]
                top_K.append(est_top_K)
        else:
            exp = explainer.slime(xloc, model, num_features = K, 
                            num_samples = 1000, n_max = 200000, 
                            alpha = alpha_adj, tol=1e-4, return_none=True)
            if exp is not None:
                tuples = exp.local_exp[1]
                est_top_K = [tuples[i][0] for i in range(K)]
                top_K.append(est_top_K)
                converged = True
        count += 1
        num_successes = len(top_K)
        if not converged:
            if count > 10 and num_successes/count < skip_thresh:# 10
                print("skipping")
                break
        else:
            if num_successes % 50 == 0 and num_successes > 0:
                print(num_successes, calc_fwer(top_K))
            
    if len(top_K)==N_runs:
        fwer = calc_fwer(top_K, Round=False)
        fwers[x_idx] = fwer
        print("#"*20, len(fwers), fwer, "#"*20)
    x_idx += 1

    # Store results
    with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
        pickle.dump(fwers, fp)

