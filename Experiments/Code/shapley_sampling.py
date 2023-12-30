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

dataset = "census"
X_train, y_train, X_test, y_test, mapping_dict = load_data(join(dir_path, "Experiments", "Data"), dataset)
model = train_neural_net(X_train, y_train)
# model = train_logreg(X_train, y_train)
np.random.seed(42)
# K = 5
K = int(sys.argv[1])
n_perms = int(sys.argv[2])
N_runs = 50
N_pts = 30

fwers = []
top_K_ss = []
shap_vals_all_pts = []
for x_idx in range(N_pts):
    print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    top_K = []
    for i in range(N_runs):
        shap_vals = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=n_perms)
        est_top_K = get_ranking(shap_vals)[:K]
        top_K.append(est_top_K)
    print(calc_fwer(top_K))
    top_K_ss.append(top_K)
fname = "ss_ranks_k"+str(K)+"_n"+str(n_perms)+"b"
with open(join(dir_path, "Experiments", "Results", fname), "wb") as fp:
    pickle.dump(top_K_ss, fp)
