import numpy as np
from math import comb
from scipy.stats import ttest_ind, t
from helper import *

############### Compute coalitions, conditional means and KernelSHAP estimates ###############

def compute_coalitions_values(model, X, xloc,
            n_perms_btwn_tests, n_samples_per_perm, mapping_dict):
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    kernel_weights = [0]*(d+1)
    for subset_size in range(d+1):
        if subset_size > 0 and subset_size < d:
            kernel_weights[subset_size] = (d-1)/(comb(d,subset_size)*subset_size*(d-subset_size))
    subset_size_distr = np.array(kernel_weights) / np.sum(kernel_weights)
    coalitions = []
    W_vals = []
    for count in range(n_perms_btwn_tests):
        subset_size = np.random.choice(np.arange(len(subset_size_distr)), p=subset_size_distr)
        # Randomly choose these features, then convert to binary vector
        S = np.random.choice(d, subset_size, replace=False)
        z = np.zeros(d)
        z[S] = 1
        w_x_vals = coalitions_kshap(X, xloc, z, n_samples_per_perm, mapping_dict)

        count += 1
        coalitions = np.append(coalitions, z).reshape((count, d))        
        W_vals.append(w_x_vals)
        if count==n_perms_btwn_tests:
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

def kshap_equation(yloc, coalitions, coalition_values, avg_pred, unbiased=False):
    '''
    Computes KernelSHAP estimates for all features. The equation is the solution to the 
    least squares problem of KernelSHAP. This inputs the dataset of M (z, v(z)).

    If multilevel, coalitions is binary 1s & 0s of the low-dim problem.
    '''
    
    # Compute v(1), the prediction made using all known features in xloc
    M, d = coalitions.shape # d low-dim if mapped
    avg_pred_vec = np.repeat(avg_pred, M)

    # A matrix and b vector in Covert and Lee
    if unbiased:
        A = get_A_mat(coalitions)
        b_all = coalitions.T * coalition_values - np.full((d, M), 0.5*avg_pred) # 0.5 is E[Z_i]
        b = np.mean(b_all, axis=1)
    else:
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
    

############ Compute variance of KernelSHAP estimates ############

def compute_kshap_vars_boot(y_pred, avg_pred, coalitions, 
                    coalition_values, n_boot):
    kshap_boot_all = []
    M = coalition_values.shape[0]
    for _ in range(n_boot):
        # sample M (z, v(z)) pairs with replacement. this will be our coalition dataset.
        idx = np.random.randint(M, size=M)
        z_boot = coalitions[idx]
        coalition_values_boot = coalition_values[idx]
        # compute the kernelSHAP estimates on these bootstrapped samples, fitting ls
        kshap_boot_all.append(kshap_equation(y_pred, z_boot, coalition_values_boot, avg_pred))

    kshap_boot = np.stack(kshap_boot_all, axis=0)
    kshap_vars_boot = np.cov(np.array(kshap_boot), rowvar=False)
    return np.diag(kshap_vars_boot)


def compute_kshap_vars_ls(var_values, coalitions):
    d = coalitions.shape[1]
    var_values = np.diagflat(var_values)
    ones_vec = np.ones(d).reshape((d, 1))
    A = coalitions.T @ coalitions
    A_inv = invert_matrix(A)
    
    C = np.diag(np.ones(d)) - np.outer(ones_vec,ones_vec) @ A_inv/np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)

    inv_ZT = A_inv @ C @ coalitions.T
    kshap_vars_ls = np.diagonal(inv_ZT @ var_values @ inv_ZT.T)
    return kshap_vars_ls

################# Testing Procedures #################

def my_t_test_kshap(shap1, shap2, var1, var2, n, alpha=0.05):
    # Welch's 2-sample t test for independent samples with unequal variances
    if shap1*shap2 < 0:
        shap2 *= -1
    # test_stat = (shap1 - shap2)/np.sqrt(var1/n + var2/n)
    # Variance is of SHAP values (i.e. mean(X)), so don't need /n
    test_stat = (shap1 - shap2)/np.sqrt(var1 + var2) 
    # df = (var1/n + var2/n)**2 / ((var1/n)**2/(n-1) + (var2/n)**2/(n-1))
    df = (var1 + var2)**2 / (var1**2/(n-1) + var2**2/(n-1))
    pval = t.cdf(-np.abs(test_stat),df=df) # Tail probability
    # print(test_stat)
    return "reject" if pval < alpha/2 else "fail to reject"


def my_t_test_kshap2(shap1, shap2, var1, var2, n, alpha=0.05):
    # Generate Gaussian data; permute and compute test statistics;
    # Take permuted p-value of original test statistic
    if shap1*shap2 < 0:
        shap2 *= -1
    test_stat = np.abs(shap1 - shap2)
    # n = 1000
    feat1_samples = np.random.normal(loc=shap1, scale=np.sqrt(var1), size=n)
    feat2_samples = np.random.normal(loc=shap2, scale=np.sqrt(var2), size=n)
    # Compute permuted test statistics on synthetic data
    both = np.concatenate((feat1_samples, feat2_samples))
    test_stats_perm = []
    for _ in range(1000):
        random_order = np.random.choice(n*2, size=n*2, replace=False)
        feat1_perm = both[random_order[:n]]
        feat2_perm = both[random_order[n:]]
        test_stat_perm = np.abs(np.mean(feat1_perm) - np.mean(feat2_perm))
        test_stats_perm.append(test_stat_perm)
    test_stats_perm = np.array(test_stats_perm)
    # Proportion of permuted test stats exceeding
    pval = np.mean(test_stats_perm >= test_stat)
    return "reject" if pval < alpha else "fail to reject"


def do_all_tests_pass_kshap(kshap_ests, kshap_vars, K, n, 
                        alpha=0.05, K_thru_rest=True, perm_test=False):
    # Stop when top-K SHAP ests are significantly in order, and K'th is bigger than rest
    order = get_ordering(kshap_ests)
    first_test_to_fail = 0
    while first_test_to_fail < K:
        feat1_idx = int(order[first_test_to_fail])
        feat2_idx = int(order[first_test_to_fail+1])
        if perm_test:
            test_result = my_t_test_kshap2(kshap_ests[feat1_idx], kshap_ests[feat2_idx], 
                                        kshap_vars[feat1_idx], kshap_vars[feat2_idx], 
                                        n, alpha=alpha)
        else:
            test_result = my_t_test_kshap(kshap_ests[feat1_idx], kshap_ests[feat2_idx], 
                                        kshap_vars[feat1_idx], kshap_vars[feat2_idx], 
                                        n, alpha=alpha)
        if test_result=="reject":
            first_test_to_fail += 1
        else:
            return False

    # Test stability of K vs K+2; K vs K+3; etc until K vs d
    # I'm not sure we can do this and have the Fithian FWER hold, but might as well
    if K_thru_rest:
        featK_idx = int(order[K-1])
        d = kshap_ests.shape[0]
        while first_test_to_fail < d-1:
            feat2_idx = int(order[first_test_to_fail+1])
            test_result = my_t_test_kshap2(kshap_ests[featK_idx], kshap_ests[feat2_idx], 
                                kshap_vars[featK_idx], kshap_vars[feat2_idx], 
                                n, alpha=alpha)
            if test_result=="reject":
                first_test_to_fail += 1
            else:
                return False
    return True

################## Functions for unbiased KernelSHAP ##################

def get_A_mat(coalitions):
    _, d = coalitions.shape
    A = np.eye(d)*0.5
    num, denom = 0, 0
    for k in range(2, d):
        num += (k-1)/(d-k)
    for k in range(1, d):
        denom += 1/(k*(d-k))
    A_ij = num/(d*(d-1)*denom)
    A[~np.eye(d,dtype=bool)] = A_ij
    return(A)

def compute_kshap_vars_unbiased(coalition_values, coalitions, avg_pred):
    M, d = coalitions.shape
    ones_vec = np.ones(d).reshape((d, 1))
    A = get_A_mat(coalitions)
    A_inv = invert_matrix(A)
    
    b_all = coalitions.T * coalition_values - np.full((d, M), 0.5*avg_pred)
    cov_b = np.cov(b_all)
    C = A_inv - (A_inv @ ones_vec @ ones_vec.T @ A_inv)/ (ones_vec.T @ A_inv @ ones_vec)

    kshap_vars = C @ cov_b @ C.T / M
    kshap_vars = np.diag(kshap_vars)
    return kshap_vars

################### KernelSHAP method ###################


def kernelshap(model, X, xloc, K=10, 
            n_perms_btwn_tests=500, n_samples_per_perm=10,
            mapping_dict=None, alpha=0.05, K_thru_rest=False,
            perm_test=False, n=None, var_method="ls", n_init=None,
            unbiased=False, max_n_perms=None):
    converged = False
    first = True
    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    while not converged:
        if first:
            n_first = n_init if n_init is not None else n_perms_btwn_tests
            coalitions, coalition_values, coalition_vars = compute_coalitions_values(model, X, xloc, 
                                                                    n_first, n_samples_per_perm, 
                                                                    mapping_dict)
            first = False
        else:
            coalitions_new, coalition_values_new, coalition_vars_new = compute_coalitions_values(model, X, xloc, 
                                                                    n_perms_btwn_tests, n_samples_per_perm, 
                                                                    mapping_dict)
            coalitions = np.concatenate((coalitions, coalitions_new))
            coalition_values = np.concatenate((coalition_values, coalition_values_new))
            coalition_vars = np.concatenate((coalition_vars, coalition_vars_new))
        kshap_ests = kshap_equation(y_pred, coalitions, coalition_values, avg_pred, unbiased=unbiased)
        if unbiased:
            kshap_vars = compute_kshap_vars_unbiased(coalition_values, coalitions, avg_pred)
        else:
            if var_method is "ls":
                kshap_vars = compute_kshap_vars_ls(coalition_vars, coalitions)
            else:
                kshap_vars = compute_kshap_vars_boot(y_pred, avg_pred, coalitions, 
                        coalition_values, n_boot=250)
        n_test = coalition_values.shape[0] if n is None else n
        converged = do_all_tests_pass_kshap(kshap_ests, kshap_vars, K, n=n_test,
                                            alpha=alpha, K_thru_rest=K_thru_rest, perm_test=perm_test)
        if max_n_perms is not None:
            if coalition_values.shape[0] >= max_n_perms:
                print("Reached max number of permutations.")
                converged = True

    return kshap_ests, kshap_vars, coalition_values.shape[0]


