import numpy as np
from scipy.stats import t
from helper import *
from math import comb


def compute_coalitions_values(model, X, xloc,
            n_perms, n_samples_per_perm, mapping_dict):
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    kernel_weights = [0]*(d+1)
    for subset_size in range(d+1):
        if subset_size > 0 and subset_size < d:
            kernel_weights[subset_size] = (d-1)/(comb(d,subset_size)*subset_size*(d-subset_size))
    subset_size_distr = np.array(kernel_weights) / np.sum(kernel_weights)
    coalitions = []
    W_vals = []
    for count in range(n_perms):
        subset_size = np.random.choice(np.arange(len(subset_size_distr)), p=subset_size_distr)
        # Randomly choose these features, then convert to binary vector z
        S = np.random.choice(d, subset_size, replace=False)
        z = np.zeros(d)
        z[S] = 1
        # For each z/S, compute list of length {# samples/perm} of X_{S^c}|X_S
        w_x_vals = coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict)
        coalitions.append(z)
        W_vals.append(w_x_vals)
    coalitions = np.array(coalitions).reshape((n_perms, d))        
    # Compute all conditional means, variances, and covariances
    coalition_values, coalition_vars = conditional_means_vars_kshap(model,W_vals,xloc,
                            n_samples_per_perm)
    return coalitions, coalition_values, coalition_vars

def coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict=None):
    # Of true num features, if mapping_dict not None
    if mapping_dict is not None:
        # Map from "original" low # of dimensions to high in order to impute 
        S_orig = S
        # High # of dimensions
        S = map_S(S_orig, mapping_dict)
    
    w_x_vals = []
    n = X.shape[0]
    for _ in range(n_samples_per_perm):
        w = X[np.random.choice(n, size=1),:]
        # Use features in S from xloc, and others (Sc) from random sample w
        w[0][S] = xloc[0][S]
        w_x_vals.append(w)
    
    return w_x_vals


def conditional_means_vars_kshap(model, W_vals,xloc, n_samples_per_perm):
    # Calculates means (value functions) and variances for each value function
    W_vals = np.reshape(W_vals, [-1*n_samples_per_perm, xloc.shape[1]])

    preds_given_S = model(W_vals)
    preds_given_S = np.reshape(preds_given_S,[-1,n_samples_per_perm])
    coalition_values = np.mean(preds_given_S,axis=1)

    preds_given_S_c = preds_given_S - coalition_values[:,None]
    coalition_vars = np.mean( preds_given_S_c**2, axis = 1)    
    return coalition_values, coalition_vars

def invert_matrix(A):
    try:
        A_inv = np.linalg.inv(A)
    except:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0]/new_cond_num
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s2), vh))
        A_inv = np.linalg.inv(A2)
    return A_inv


def kshap_equation(yloc, coalitions, coalition_values, avg_pred):
    '''
    Computes KernelSHAP estimates for all features. The equation is the solution to the 
    least squares problem of KernelSHAP. This inputs the dataset of M (z, v(z)).

    If multilevel, coalitions is binary 1s & 0s of the low-dim problem.
    '''
    
    # Compute v(1), the prediction made using all known features in xloc
    M, d = coalitions.shape # d low-dim if mapped
    avg_pred_vec = np.repeat(avg_pred, M)

    # A matrix and b vector in Covert and Lee
    A = np.matmul(coalitions.T, coalitions) / M
    b = np.matmul(coalitions.T, coalition_values - avg_pred_vec) / M

    # Covert & Lee Equation 7
    A_inv = invert_matrix(A)
    ones_vec = np.ones(d).reshape((d, 1))
    numerator = np.matmul(np.matmul(ones_vec.T, A_inv), b) - yloc + avg_pred
    denominator = np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)
    term = b.reshape((d, 1)) - ones_vec*(numerator / denominator)

    kshap_ests = np.matmul(A_inv, term).reshape(-1)
    return kshap_ests


################### KernelSHAP method ###################
def compute_kshap_vars_ls(var_values, coalitions):
    d = coalitions.shape[1]
    var_values = np.diagflat(var_values)
    ones_vec = np.ones(d).reshape((d, 1))
    A = coalitions.T @ coalitions
    try:
        A_inv = np.linalg.inv(A)
    except:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0]/new_cond_num
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s2), vh))

        A_inv = np.linalg.inv(A2)
    
    C = np.diag(np.ones(d)) - np.outer(ones_vec,ones_vec) @ A_inv/np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)

    AZ = A_inv @ C @ coalitions.T
    kshap_covmat_ls = AZ @ var_values @ AZ.T
    return kshap_covmat_ls

def compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, 
                    coalition_values, n_boot):
    """
    Returns n_boot sets of kernelSHAP values for each feature, 
    fitting kernelSHAP on both the true model and its approximation with each bootstrapped resampling.

    We can probably make a version of this function where you don't have to bootstrap the model,
    since we don't need this for computing CV-kSHAP estimates; we only need it to compute the
    variance reduction of CV-kSHAP over vanilla kSHAP. Low priority.
    """

    kshap_vals_boot = []
    M = coalition_values.shape[0]
    yloc = model(xloc)
    for _ in range(n_boot):
        idx = np.random.randint(M, size=M)
        z_boot = coalitions[idx]
        coalition_values_model_boot = coalition_values[idx]

        # compute the kernelSHAP estimates on these bootstrapped samples, fitting ls
        kshap_vals_boot.append(kshap_equation(yloc, z_boot, coalition_values_model_boot, avg_pred))

    kshap_vals_boot = np.stack(kshap_vals_boot, axis=0)
    # Compute empirical covariance matrix of each feature's KernelSHAP value, using bootstrapped samples.
    kshap_cov_boot = np.cov(np.array(kshap_vals_boot), rowvar=False)
    return kshap_cov_boot
