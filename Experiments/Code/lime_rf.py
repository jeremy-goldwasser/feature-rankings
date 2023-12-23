import numpy as np
import sys
import pickle
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from slime import lime_tabular

from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")
sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

dataset = "breast_cancer"
K = int(sys.argv[1])
fname = "lime_" + dataset + "_rf_K" + str(K) + "_10x"
print(fname)

breast_cancer = load_breast_cancer()
data_path = join(dir_path, "Experiments", "Data")
# train, test, labels_train, labels_test = train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.80, random_state=1)
# save_data(data_path, dataset, train, labels_train, test, labels_test, mapping_dict=None)
train, labels_train, test, labels_test, _ = load_data(data_path, dataset)

rf = RandomForestClassifier()
rf.fit(train, labels_train)

explainer = lime_tabular.LimeTabularExplainer(train, 
                                              feature_names = breast_cancer.feature_names, 
                                              class_names = breast_cancer.target_names, 
                                              discretize_continuous = False, 
                                              feature_selection = "lasso_path", 
                                              sample_around_instance = True)


alpha = 0.20
skip_thresh = 0.2
n_trials = 500
n_pts = 10
alpha_adj = alpha/K/2
top_K_all = []
fwers = []
x_idx = 0
while len(fwers) < n_pts:
    print(x_idx)
    xloc = test[x_idx]
    top_K = []
    count = 0
    while len(top_K) < n_trials:
        count += 1
        exp = explainer.slime(xloc, rf.predict_proba, num_features = K, 
                                num_samples = 1000, n_max = 100000, 
                                alpha = alpha_adj, tol=1e-4, return_none=True)
        if exp is not None:
            tuples = exp.local_exp[1]
            feats = [tuples[i][0] for i in range(K)]
            top_K.append(feats)
            # if (len(top_K)%10==0): print(len(top_K), calc_fwer(top_K))
        else:
            # print("failed to converge; " + str(len(top_K)))
            num_successes = len(top_K)
            if count > 5 and num_successes/count < skip_thresh:
                break
    if len(top_K)==n_trials:
        fwer = calc_fwer(top_K)
        print("#"*20, x_idx, fwer, "#"*20)
        fwers.append(fwer)
        top_K_all.append(top_K)
    x_idx += 1

    # Store results
    with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
        pickle.dump(top_K_all, fp)