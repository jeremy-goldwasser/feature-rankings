import numpy as np
from scipy.stats import ttest_ind
from helper import *

def my_t_test(feat1, feat2, alpha=0.05):
    if np.mean(feat1)*np.mean(feat2) < 0:
        if isinstance(feat2, np.ndarray): feat2 = -feat2
        else: feat2 = [-feat2[j] for j in range(len(feat2))]
    pval = ttest_ind(feat1, feat2, equal_var=False, 
                    alternative='two-sided', random_state=1)[1]
    return "reject" if pval < alpha else "fail to reject"


def do_all_tests_pass(diffs_all_feats, K, alpha=0.05, order=None, K_thru_rest=True):
    # Stop when top-K SHAP ests are significantly in order, and K'th is bigger than rest
    d = len(diffs_all_feats)
    if order is None:
        # shap_ests = np.mean(diffs_all_feats, axis=1)
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(d)]
        # Indices with biggest to smallest absolute SHAP value
        order = get_ordering(shap_ests)
    first_test_to_fail = 0
    # Test stability of 1 vs 2; 2 vs 3; etc until K vs K+1
    while first_test_to_fail < K:
        feat1 = diffs_all_feats[int(order[first_test_to_fail])]
        feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
        test_result = my_t_test(feat1, feat2, alpha=alpha)
        if test_result=="reject":
            first_test_to_fail += 1
        else:
            return False
    
    if K_thru_rest:
        # Test stability of K vs K+2; K vs K+3; etc until K vs d
        # I'm not sure we can do this and have the Fithian FWER hold, but might as well
        featK = diffs_all_feats[int(order[K-1])]
        while first_test_to_fail < d-1:
            feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
            test_result = my_t_test(featK, feat2, alpha=alpha)
            if test_result=="reject":
                first_test_to_fail += 1
            else:
                return False
    return True


def find_first_test_to_fail(diffs_all_feats, K, alpha=0.05, order=None, K_thru_rest=True):
    d = len(diffs_all_feats)
    if order is None:
        # shap_ests = np.mean(diffs_all_feats, axis=1)
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(len(diffs_all_feats))]
        # Indices with biggest to smallest absolute SHAP value
        order = get_ordering(shap_ests)
    first_test_to_fail = 0
    while first_test_to_fail < K:
        feat1 = diffs_all_feats[int(order[first_test_to_fail])]
        feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
        test_result = my_t_test(feat1, feat2, alpha=alpha)
        if test_result=="reject":
            first_test_to_fail += 1
        else:
            return first_test_to_fail
    if K_thru_rest:
        # Test stability of K vs K+2; K vs K+3; etc until K vs d
        # I'm not sure we can do this and have the Fithian FWER hold, but might as well
        featK = diffs_all_feats[int(order[K-1])]
        while first_test_to_fail < d-1:
            feat2 = diffs_all_feats[int(order[first_test_to_fail+1])]
            test_result = my_t_test(featK, feat2, alpha=alpha)
            if test_result=="reject":
                first_test_to_fail += 1
            else:
                return first_test_to_fail
    return -1


def query_values_marginal(X, xloc, 
                            S, j,  mapping_dict, 
                            n_samples_per_perm):
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

def compute_diffs_all_feats(model, X, xloc, M, mapping_dict=None, n_samples_per_perm=2, as_np=True):
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
        if not as_np: diffs_avg = diffs_avg.tolist()
        diffs_all_feats.append(diffs_avg)
    if not as_np:
        return(diffs_all_feats)
    return np.array(diffs_all_feats)


def shapley_sampling_basic(model, X, xloc, K,
                            mapping_dict=None, alpha=0.05, 
                            n_perms_btwn_tests=50, n_samples_per_perm=2, 
                            K_thru_rest=False, n_init=None):

    n_first = n_init if n_init is not None else n_perms_btwn_tests
    diffs_all_feats = compute_diffs_all_feats(model, X, xloc, n_first, mapping_dict=mapping_dict, n_samples_per_perm=n_samples_per_perm)
    while not do_all_tests_pass(diffs_all_feats, K, alpha=alpha, K_thru_rest=K_thru_rest):
        diffs_all_feats_new = compute_diffs_all_feats(model, X, xloc, n_perms_btwn_tests, mapping_dict=mapping_dict, n_samples_per_perm=n_samples_per_perm)
        diffs_all_feats = np.concatenate((diffs_all_feats, np.array(diffs_all_feats_new)), axis=1)
    shap_vals = np.mean(diffs_all_feats, axis=1)
    return shap_vals, diffs_all_feats


def shapley_sampling_adaptive(model, X, xloc, K, 
                            mapping_dict=None, alpha=0.05, n_perms_btwn_tests=50, 
                            n_samples_per_perm=2, K_thru_rest=False, n_init=None):
    n_first = n_init if n_init is not None else n_perms_btwn_tests
    diffs_all_feats = compute_diffs_all_feats(model, X, xloc, n_first, 
                                            mapping_dict=mapping_dict, 
                                            n_samples_per_perm=n_samples_per_perm, as_np=False)
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    while not do_all_tests_pass(diffs_all_feats, K, alpha=alpha, K_thru_rest=K_thru_rest):
        shap_ests = [np.mean(diffs_all_feats[j]) for j in range(d)]
        order = get_ordering(shap_ests)
        first_test_to_fail = find_first_test_to_fail(diffs_all_feats, K, 
                                    order=order, K_thru_rest=K_thru_rest, alpha=alpha)
        index_pair = (int(order[first_test_to_fail]), int(order[first_test_to_fail+1]))
        diffs_pair = [diffs_all_feats[index_pair[0]], diffs_all_feats[index_pair[1]]]
        # Run until order of pair is stable
        while my_t_test(diffs_pair[0], diffs_pair[1], alpha=alpha)=="fail to reject":
            diffs_pair_new = []
            for j in index_pair:
                w_vals,wj_vals = [], []
                for _ in range(n_perms_btwn_tests):
                    perm = np.random.permutation(d)
                    j_idx = np.argwhere(perm==j).item()
                    S = np.array(perm[:j_idx])
                    
                    tw_vals, twj_vals = query_values_marginal(X, xloc, S, j, mapping_dict, n_samples_per_perm)
                    w_vals.append(tw_vals)
                    wj_vals.append(twj_vals)
                w_vals = np.reshape(w_vals, [n_perms_btwn_tests*n_samples_per_perm, xloc.shape[1]])
                wj_vals = np.reshape(wj_vals, [n_perms_btwn_tests*n_samples_per_perm, xloc.shape[1]])
                
                diffs_all = model(wj_vals) - model(w_vals)
                diffs_avg = np.mean(np.reshape(diffs_all,[n_perms_btwn_tests,n_samples_per_perm]),axis=1) # length M
                diffs_pair_new.append(diffs_avg)
            diffs_pair[0].extend(diffs_pair_new[0])
            diffs_pair[1].extend(diffs_pair_new[1])

        diffs_all_feats[index_pair[0]] = diffs_pair[0]
        diffs_all_feats[index_pair[1]] = diffs_pair[1]
    shap_vals = np.array([np.mean(diffs_all_feats[j]) for j in range(d)])
    return shap_vals, diffs_all_feats