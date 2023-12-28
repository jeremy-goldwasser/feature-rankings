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
print(len(y_test))
model = train_neural_net(X_train, y_train)
np.random.seed(1)
K = 5
N_runs = 50
N_pts = 30

fwers = []
top_5_ss = []
shap_vals_all_pts = []
for x_idx in range(N_pts):
    print(x_idx)
    xloc = X_test[x_idx:(x_idx+1)]
    top_K = []
    for i in range(N_runs):
        shap_vals = shapley_sampling(model, X_train, xloc, mapping_dict=mapping_dict, n_perms=500)
        est_top_K = get_ranking(shap_vals)[:K]
        top_K.append(est_top_K)
    print(calc_fwer(top_K))
    top_5_ss.append(top_K)

with open(join(dir_path, "Experiments", "Results", "ss_ranks_k5"), "wb") as fp:
    pickle.dump(top_5_ss, fp)
