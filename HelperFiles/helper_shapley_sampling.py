import numpy as np
from helper import *


def diffs_to_shap_vals(diffs_all_feats, abs=False):
    shap_ests = np.array([np.mean(diffs) for diffs in diffs_all_feats])
    if abs:
        shap_ests = np.abs(shap_ests)
    return shap_ests

def diffs_to_shap_vars(diffs_all_feats, var_of_mean=True):
    if var_of_mean:
        return np.array([np.var(diffs, ddof=1)/len(diffs) for diffs in diffs_all_feats])
    return np.array([np.var(diffs, ddof=1) for diffs in diffs_all_feats])


def query_values_marginal(X, xloc, S, j,  mapping_dict, n_samples_per_perm):
    '''
    Per Strumbelj and Kononenko, select S via permutation and draw n_samples_per_perm of (x_S, w_Sc).
    '''
    SandJ = np.append(S,j)
    n = X.shape[0]
    if mapping_dict is not None:
        d = len(mapping_dict)
        S = map_S(S, mapping_dict)
        SandJ = map_S(SandJ, mapping_dict)

    w_vals = []
    wj_vals = []
    for _ in range(n_samples_per_perm):
        # Sample "unknown" features from a dataset sample z
        z = X[np.random.choice(n, size=1),:]
        z1, z2 = np.copy(z), np.copy(z)
        z1[0][S] = xloc[0][S]
        w_vals.append(z1)
        z2[0][SandJ] = xloc[0][SandJ]
        wj_vals.append(z2)
    return w_vals, wj_vals

def compute_diffs_all_feats(model, X, xloc, M, mapping_dict=None, n_samples_per_perm=2):
    d = len(mapping_dict) if mapping_dict is not None else xloc.shape[1]
    diffs_all_feats = []
    for j in range(d):
        w_vals,wj_vals = [], []
        for _ in range(M):
            perm = np.random.permutation(d)
            j_idx = np.argwhere(perm==j).item()
            S = np.array(perm[:j_idx])
            
            tw_vals, twj_vals = query_values_marginal(X, xloc, S, j, mapping_dict, n_samples_per_perm)
            w_vals.append(tw_vals)
            wj_vals.append(twj_vals)
        w_vals = np.reshape(w_vals, [M*n_samples_per_perm, xloc.shape[1]])
        wj_vals = np.reshape(wj_vals, [M*n_samples_per_perm, xloc.shape[1]])
        
        diffs_all = model(wj_vals) - model(w_vals)
        diffs_avg = np.mean(np.reshape(diffs_all,[M,n_samples_per_perm]),axis=1) # length M
        diffs_all_feats.append(diffs_avg.tolist())
    return(diffs_all_feats)


