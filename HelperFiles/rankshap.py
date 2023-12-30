import numpy as np
from scipy.stats import norm
from helper import *


def normal_test(feat1, feat2, alpha=0.05, n_equal=True, abs=True):
    if abs is True and np.mean(feat1)*np.mean(feat2) < 0:
        if isinstance(feat2, np.ndarray): feat2 = -feat2
        else: feat2 = [-feat2[j] for j in range(len(feat2))]
    diff_shap_vals = np.mean(feat1) - np.mean(feat2)
    n1, n2 = len(feat1), len(feat2)
    var1, var2 = np.var(feat1, ddof=1), np.var(feat2, ddof=1)
    var_factor = 2
    Z_obs = np.abs(diff_shap_vals)/np.sqrt(var_factor*(var1/n1 + var2/n2))
    Z_crit = norm.ppf(1 - alpha/2) # 1-a/2 quantile (upper tail) of standard normal
    if Z_obs > Z_crit:
        return "reject"
    else:
        if n_equal: # Same number of permutations for each feature
            n_to_run = (Z_crit/diff_shap_vals)**2 * 2*(var1 + var2)
            n_to_run = [n_to_run, n_to_run]
        else: # Scale number of permutations by the variance
            new_n1 = 4*var1*(Z_crit/diff_shap_vals)**2
            new_n2 = (var2/var1)*new_n1
            n_to_run = [new_n1, new_n2]                
        return "fail to reject", n_to_run


def do_all_tests_pass(diffs_all_feats, K, alpha=0.05, 
                    order=None, n_equal=True, abs=True):
    # Stop when top-K SHAP ests are significantly in order, and K'th is bigger than rest
    d = len(diffs_all_feats)
    if order is None:
        # shap_ests = np.mean(diffs_all_feats, axis=1)
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(d)]
        # Indices with biggest to smallest absolute SHAP value
        order = get_ranking(shap_ests, abs=abs)
    first_test_to_fail = 0
    # Test stability of 1 vs 2; 2 vs 3; etc until K vs K+1
    while first_test_to_fail < K:
        feat1 = diffs_all_feats[int(order[first_test_to_fail])]
        feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
        test_result = normal_test(feat1, feat2, alpha=alpha, 
                                n_equal=n_equal)
        if test_result=="reject":
            first_test_to_fail += 1
        else:
            return False

    return True


def find_first_test_to_fail(diffs_all_feats, K, alpha=0.05, 
                            order=None, n_equal=True, abs=True):
    d = len(diffs_all_feats)
    if order is None:
        # shap_ests = np.mean(diffs_all_feats, axis=1)
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(d)]
        # Indices with biggest to smallest absolute SHAP value
        order = get_ranking(shap_ests, abs=abs)
    first_test_to_fail = 0
    while first_test_to_fail < K:
        feat1 = diffs_all_feats[int(order[first_test_to_fail])]
        feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
        test_result = normal_test(feat1, feat2, alpha=alpha, 
                                n_equal=n_equal, abs=abs)
        if test_result=="reject":
            first_test_to_fail += 1
        else:
            return first_test_to_fail
    return -1


def query_values_marginal(X, xloc, S, j,  mapping_dict, n_samples_per_perm):
    '''
    Per Strumbelj and Kononenko, select S via permutation and draw n_samples_per_perm of (x_S, w_Sc).
    '''
    SandJ = np.append(S,j)
    n = X.shape[0]
    if mapping_dict is None:
        d = X.shape[1]
        Sc = np.where(~np.isin(np.arange(d), S))[0]
        Sjc = np.where(~np.isin(np.arange(d), SandJ))[0]
    else:
        d = len(mapping_dict)
        Sc_orig = np.where(~np.isin(np.arange(d), S))[0] # "original" low # of dimensions
        Sjc_orig = np.where(~np.isin(np.arange(d), SandJ))[0]
        Sc = map_S(Sc_orig, mapping_dict)
        Sjc = map_S(Sjc_orig, mapping_dict)

    w_vals = []
    wj_vals = []

    for _ in range(n_samples_per_perm):
        # Sample "unknown" features from a dataset sample z
        z = X[np.random.choice(n, size=1),:]
        w_x_s, w_x_s_j = np.copy(xloc), np.copy(xloc)
        w_x_s[0][Sc] = z[0][Sc]
        w_x_s_j[0][Sjc] = z[0][Sjc]

        w_vals.append(w_x_s)
        wj_vals.append(w_x_s_j)

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


def rankshap(model, X, xloc, K, alpha=0.10, mapping_dict=None, 
            n_samples_per_perm=2, n_init=100, max_n_perms=10000,  
            n_equal=True, buffer=1.1, abs=True):
    '''
    - model: Inputs a numpy array, outputs a scalar
    - X: N by D matrix of samples
    - xloc: 1 by D matrix with one sample, whose SHAP values are estimated
    - K: Number of features we want to rank correctly
    - alpha: Significance level
    - mapping_dict: Dictionary mapping categorical variables to corresponding binary columns of X and xloc
    - n_samples_per_perm: Number of samples of X_{S^c} with which to estimate v(S) = E[f(X) | x_S)]
    - n_init: Number of initial permutations for all features, before testing pairs for ranking
    - n_equal: Boolean, whether we want ambiguously ranked features to receive equal number of permutations, or scale by relative variance
    - buffer: Factor by which to increase estimate of necessary number of permutations. Should be ≥ 1.
    - abs: Whether we want to rank features by the absolute values of their Shapley values
    
    '''
    converged = False
    diffs_all_feats = compute_diffs_all_feats(model, X, xloc, n_init, 
                                            mapping_dict=mapping_dict, 
                                            n_samples_per_perm=n_samples_per_perm)
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    
    while not do_all_tests_pass(diffs_all_feats, K, alpha=alpha, n_equal=n_equal):
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(d)]
        order = get_ranking(shap_ests, abs=abs)
        first_test_to_fail = find_first_test_to_fail(diffs_all_feats, K, 
                                    order=order, alpha=alpha, n_equal=n_equal, abs=abs)
        index_pair = (int(order[first_test_to_fail]), int(order[first_test_to_fail+1]))
        diffs_pair = [diffs_all_feats[index_pair[0]], diffs_all_feats[index_pair[1]]]
        # Run until order of pair is stable
        test_result = normal_test(diffs_pair[0], diffs_pair[1], alpha=alpha, n_equal=n_equal, abs=abs)
        exceeded = False
        while test_result != "reject":
            # Run for suggested number of samples to be significant difference
            n_to_run = [max(int(buffer*n), n_init) for n in test_result[1]]
            if max(n_to_run) > max_n_perms:
                if not exceeded:
                    n_to_run = [max_n_perms, max_n_perms]
                    exceeded = True
                else:
                    # print("Hit max # perms without converging. Returning `NA`.")
                    # return "NA", "NA"
                    shap_vals = np.array([np.mean(diffs_all_feats[j]) for j in range(d)])
                    return shap_vals, diffs_all_feats, converged
            diffs_pair = []
            for i in range(2):
                j = index_pair[i]
                w_vals,wj_vals = [], []
                for _ in range(n_to_run[i]):
                    perm = np.random.permutation(d)
                    j_idx = np.argwhere(perm==j).item()
                    S = np.array(perm[:j_idx])
                    
                    tw_vals, twj_vals = query_values_marginal(X, xloc, S, j, mapping_dict, n_samples_per_perm)
                    w_vals.append(tw_vals)
                    wj_vals.append(twj_vals)
                w_vals = np.reshape(w_vals, [-1, xloc.shape[1]])
                wj_vals = np.reshape(wj_vals, [-1, xloc.shape[1]])
                
                diffs_all = model(wj_vals) - model(w_vals)
                diffs_avg = np.mean(np.reshape(diffs_all,[-1,n_samples_per_perm]),axis=1) # length M
                diffs_pair.append(diffs_avg)
            test_result = normal_test(diffs_pair[0], diffs_pair[1], alpha=alpha, n_equal=n_equal, abs=abs)
        # Replace with new samples
        diffs_all_feats[index_pair[0]] = diffs_pair[0]
        diffs_all_feats[index_pair[1]] = diffs_pair[1]
    shap_vals = np.array([np.mean(diffs_all_feats[j]) for j in range(d)])
    converged = True
    return shap_vals, diffs_all_feats, converged


def shapley_sampling(model, X, xloc, n_perms, mapping_dict=None, n_samples_per_perm=2):
    diffs_all_feats = compute_diffs_all_feats(model, X, xloc, n_perms, 
                                            mapping_dict=mapping_dict, 
                                            n_samples_per_perm=n_samples_per_perm)
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    shap_vals = np.array([np.mean(diffs_all_feats[j]) for j in range(d)])
    return shap_vals
