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
fname = "shap_" + dataset + "_nn_K" + str(K) + "_10x"
print(fname)
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)

model = train_neural_net(X_train, y_train)

#%%
np.random.seed(1)
N_runs = 500
N_pts = 10

shap_vals_all_pts = []
for i in range(N_pts):
    print(i)
    xloc = X_test[i:(i+1)]
    shap_vals_all = []
    top_K = []
    while len(top_K) < N_runs:
        shap_vals, _, converged = rankshap(model, X_train, xloc, K=K, alpha=0.2, 
                                mapping_dict=mapping_dict, max_n_perms=10000, 
                                n_init=100, n_equal=False)
        if converged:
            est_top_K = get_ranking(shap_vals)[:K]
            top_K.append(est_top_K)
            shap_vals_all.append(shap_vals)
            if len(top_K) % 20 == 0: # 20
                print(len(top_K), calc_fwer(top_K))
        else:
            print("Failed to converge")
    shap_vals_all_pts.append(shap_vals_all)

    # Store results
    with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
        pickle.dump(shap_vals_all_pts, fp)

