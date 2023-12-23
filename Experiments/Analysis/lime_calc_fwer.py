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
fname = "lime_breast_cancer_rf_K" + str(K) + "_10x"
print("K={}.".format(K))
with open(join(dir_path, "Experiments", "Results", fname), "rb") as fp:
    top_K_all = pickle.load(fp) 

N_pts = len(top_K_all)
N_runs = len(top_K_all[0])
fwers = []
for top_K in top_K_all:
    fwer = calc_fwer(top_K)
    fwers.append(fwer)
    print("{}%".format(round(fwer*100,1))) # FWER

with open(join(dir_path, "Experiments", "Results", "lime_fwers_K" + str(K)), "wb") as fp:
    pickle.dump(fwers, fp) 