#%%
import numpy as np
import sys

dir_path = "/Users/jeremygoldwasser/Desktop/RankSHAP"
from os.path import join
sys.path.append(join(dir_path, "HelperFiles"))
from helper import *
from shapley_sampling2 import *
from train_models import *
from load_data import *

import warnings
warnings.filterwarnings('ignore')

X_train, y_train, X_test, y_test, mapping_dict = load_data("census", join(dir_path, "Experiments", "Data"))
xloc = X_test[0:1]
# model, approx, true_shap_vals = train_logreg(X_train, y_train, xloc, mapping_dict)
model, approx, true_shap_vals = train_neural_net(X_train, y_train, xloc, mapping_dict)

#%%
np.random.seed(1)
K = 1
top_K = []
# for i in range(100):
while len(top_K) < 100:
    shap_vals2, diffs_all_feats2 = shapley_sampling_adaptive(model, X_train, xloc, K=K, alpha=0.2, 
                            mapping_dict=mapping_dict, max_n_perms=10000, n_init=30, scale_var=True,
                            multiplicative=False)
    if shap_vals2 != "NA":
        est_top_K = get_ordering(shap_vals2)[:K]
        top_K.append(est_top_K)
        if (len(top_K)+1) % 5 == 0:
            print(len(top_K)+1)



# %%
