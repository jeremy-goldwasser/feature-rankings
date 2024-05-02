import numpy as np
from math import comb
# from scipy.stats import ttest_ind, t
from helper import *

############### Compute coalitions, conditional means and KernelSHAP estimates ###############

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
        w_x_vals = coalitions_kshap(X, xloc, z, n_samples_per_perm, mapping_dict)

        count += 1
        coalitions = np.append(coalitions, z).reshape((count, d))        
        W_vals.append(w_x_vals)
        if count==n_perms:
            # Compute all conditional means, variances, and covariances
            coalition_values, coalition_vars = conditional_means_vars_kshap(model,W_vals,xloc,
                                   n_samples_per_perm)
            return coalitions, coalition_values, coalition_vars

def coalitions_kshap(X, xloc, z, n_samples_per_perm, mapping_dict=None):
    # Of true num features, if mapping_dict not None
    d = z.shape[0] 
    if mapping_dict is None:
        S = np.nonzero(z)[0]
        Sc = np.where(~np.isin(np.arange(d), S))[0]
    else:
        # "original" low # of dimensions
        S_orig = np.nonzero(z)[0]
        Sc_orig = np.where(~np.isin(np.arange(d), S_orig))[0]
        # High # of dimensions (each binary level as a column)
        S = map_S(S_orig, mapping_dict)
        Sc = map_S(Sc_orig, mapping_dict)
    
    w_x_vals = []
    for _ in range(n_samples_per_perm):
        w = X[np.random.choice(X.shape[0], size=1),:]
        # Copy xloc, then replace its "unknown" features with those of random sample w
        w_x_s = np.copy(xloc)
        w_x_s[0][Sc] = w[0][Sc]
        w_x_vals.append(w_x_s)
    
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
    term = (b - (numerator / denominator)).reshape((d, 1))

    kshap_ests = np.matmul(A_inv, term).reshape(-1)
    return kshap_ests


################### KernelSHAP method ###################


def kernelshap(model, X, xloc, n_perms=500, n_samples_per_perm=10, mapping_dict=None):
    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    coalitions, coalition_values, _ = compute_coalitions_values(model, X, xloc, 
                                                                    n_perms, n_samples_per_perm, 
                                                                    mapping_dict)
    kshap_ests = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
    return kshap_ests
