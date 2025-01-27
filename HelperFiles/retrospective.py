import numpy as np
from helper import *
from helper_kernelshap import *
from helper_shapley_sampling import *

############### SHAPLEY SAMPLING ###############
def shapley_sampling(model, X, xloc, n_perms, 
                     mapping_dict=None, n_samples_per_perm=10, 
                     alphas=None, abs=True):
    if xloc.ndim==1:
        xloc = xloc.reshape(1,-1)
    diffs_all_feats = compute_diffs_all_feats(model, X, xloc, n_perms, 
                                            mapping_dict=mapping_dict, 
                                            n_samples_per_perm=n_samples_per_perm)
    # d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    shap_vals = np.array(diffs_to_shap_vals(diffs_all_feats))
    shap_vars = diffs_to_shap_vars(diffs_all_feats)
    if alphas is None:
        return shap_vals, shap_vars
    else:
        if isinstance(alphas, list) and len(alphas)>1:
            n_verified = [find_num_verified(shap_vals, shap_vars, alpha=alpha, abs=abs) for alpha in alphas]
        else:
            if isinstance(alphas, list): alphas = alphas[0]
            n_verified = find_num_verified(shap_vals, shap_vars, alpha=alphas, abs=abs)
        return shap_vals, n_verified, shap_vars


############### KERNELSHAP ###############

def kernelshap(model, X, xloc, n_perms=500, n_samples_per_perm=10, mapping_dict=None,
            alphas=None, abs=True):
    if xloc.ndim==1:
        xloc = xloc.reshape(1,-1)
    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    coalitions, coalition_values, _ = compute_coalitions_values(model, X, xloc, 
                                                                    n_perms, n_samples_per_perm, 
                                                                    mapping_dict)
    kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
    kshap_covs = compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, 
                                        coalition_values, n_boot=250)
    # kshap_covs = compute_kshap_vars_ls(coalition_vars,coalitions)
    if alphas is None:
        return kshap_vals, kshap_covs
    else:
        kshap_vars = np.diag(kshap_covs)

        if isinstance(alphas, list) and len(alphas)>1:
            n_verified = [find_num_verified(kshap_vals, kshap_vars, alpha=alpha, abs=abs) for alpha in alphas]
        else:
            if isinstance(alphas, list): alphas = alphas[0]
            n_verified = find_num_verified(kshap_vals, kshap_vars, alpha=alphas, abs=abs)
        return kshap_vals, n_verified, kshap_covs
