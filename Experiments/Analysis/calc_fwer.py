import numpy as np
import pickle
import pathlib
from os.path import join
path_to_file = str(pathlib.Path().resolve())
dir_path = join(path_to_file, "../../")

import sys
sys.path.append(join(dir_path, "HelperFiles"))
from helper import *

K = int(sys.argv[1])
fname = "shap_census_nn_K" + str(K) + "_10x"
print("K={}.".format(K))
with open(join(dir_path, "Experiments", "Results", fname), "rb") as fp:
    shap_vals_all_pts = pickle.load(fp) 

N_pts = len(shap_vals_all_pts)
N_runs = len(shap_vals_all_pts[0])
fwers = []
for shap_vals_all in shap_vals_all_pts:
    top_K = []
    for shap_vals in shap_vals_all:
        est_top_K = get_ranking(shap_vals)[:K]
        top_K.append(est_top_K)

    #### Compute average #####
    fwer = calc_fwer(top_K)
    fwers.append(np.round(fwer*100, 1))
    print("{}%".format(round(fwer*100,1))) # FWER

with open(join(dir_path, "Experiments", "Results", "census_fwers_K_" + str(K)), "wb") as fp:
    pickle.dump(fwers, fp) 