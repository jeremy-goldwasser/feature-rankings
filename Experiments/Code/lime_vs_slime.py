import numpy as np
import sys
import pickle
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from slime import lime_tabular

from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")
sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)
dataset = "breast_cancer"

breast_cancer = load_breast_cancer()
data_path = join(dir_path, "Experiments", "Data")
train, labels_train, test, labels_test, _ = load_data(data_path, dataset)

# rf = RandomForestClassifier()
# rf.fit(train, labels_train)
colnames = breast_cancer.feature_names
explainer = lime_tabular.LimeTabularExplainer(train, 
                                              feature_names = colnames, 
                                              class_names = breast_cancer.target_names, 
                                              discretize_continuous = False, 
                                              feature_selection = "lasso_path", 
                                              sample_around_instance = True)

############# Compute top-5 LIME selections across many reruns #############

i = 6
K = 5
xloc = test[i]
top_K_lime = []
for _ in range(200):
    exp = explainer.explain_instance(xloc, model, num_features = K, num_samples = 5000) # Default
    # exp.show_in_notebook(show_table = True)
    tuples = exp.local_exp[1]
    feats = [tuples[i][0] for i in range(5)]
    top_K_lime.append(feats)

print(calc_fwer(top_K_lime))
## Slow: takes ~45 minutes
top_K_slime = []
count = 0
N_runs = 50
while len(top_K_slime) < N_runs:
    count += 1
    exp = explainer.slime(xloc, model, num_features = K, 
                                num_samples = 1000, n_max = 500000, #1000000
                                alpha = 0.2/K/2, tol=1e-4, return_none=True) #5e-5; rf.predict_proba
    if exp is not None:
        tuples = exp.local_exp[1]
        feats = [tuples[i][0] for i in range(K)]
        top_K_slime.append(feats)
    
    print(len(top_K_slime), count)
    fname = "lime_vs_slime"
    with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
            pickle.dump([top_K_lime, top_K_slime], fp)

print(calc_fwer(top_K_slime))