import numpy as np
import pickle
from os.path import join
dir_path = "/accounts/grad/jeremy_goldwasser/RankSHAP"
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
        est_top_K = get_ordering(shap_vals)[:K]
        top_K.append(est_top_K)

    #### Compute average #####
    top_K = np.array(top_K)
    most_common_row = mode_rows(top_K)
    ct = 0
    for idx in range(N_runs):
        if np.array_equal(most_common_row, top_K[idx]):
            ct += 1
    # print("{}%".format(np.round((1 - ct / N_runs), 3)*100)) # FWER
    fwer = 1 - ct / N_runs
    fwers.append(np.round(fwer*100, 1))
    print("{}%".format(round(fwer*100,1))) # FWER

with open(join(dir_path, "Experiments", "Results", "census_fwers_K_" + str(K)), "wb") as fp:
    pickle.dump(fwers, fp) 