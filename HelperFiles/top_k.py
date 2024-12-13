import numpy as np
from scipy.stats import t, nct, norm
# from helper import *
import helper
import helper_shapley_sampling
from helper_kernelshap import *
# from helper_shapley_sampling import *


################ RankSHAP (Shapley Smampling) ################

def find_num_verified_rankshap(diffs_all_feats, alpha=0.1, abs=True, 
                    n_equal=True, K=None):
    # k passed <=> passed through rank k vs k+1 <=> first failure at test idx k vs k+1
    shap_ests = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats, abs=abs)
    shap_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats)
    value_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats, var_of_mean=False)
    order = get_ranking(shap_ests, abs=abs)
    
    # Test stability of 1 vs 2; 2 vs 3; etc (d-1 total tests)
    num_verified = 0
    d = len(shap_ests)
    max_num_tests = d-1 if K is None else K
    while num_verified < max_num_tests:
        idx_to_test = order[num_verified:].astype(int)
        # Means are in sorted order (may be with absolute value)
        means_to_test = shap_ests[idx_to_test]
        vars_to_test = shap_vars[idx_to_test]
        value_vars_to_test = value_vars[idx_to_test]

        # Find index with biggest variance. 
        # Subsequent tests will necessarily have lower p-values, so they don't need to be tested.
        max_test_idx = np.argmax(vars_to_test[1:]) + 1
        ns_to_reject = []
        for j in range(1, max_test_idx+1): # max_test_idx iterations
            test_result, n_to_reject = helper.test_for_max(means_to_test, vars_to_test, j, alpha,
                                                           compute_sample_size=True, n_equal=n_equal,
                                                           value_vars=value_vars_to_test)
            if test_result=="reject": # Significant P-value
                break
            
            ns_to_reject.append(n_to_reject)
        if test_result=="reject":
            # One of the tests passed. Move on.
            num_verified += 1
        else:
            # All of the tests failed to reject.
            # Identify features with fewest features needed to reject null
            n_totals = np.sum(ns_to_reject, axis=1)
            close_idx_among_tested = np.argmin(n_totals)
            n_samples_to_reject = ns_to_reject[close_idx_among_tested]
            close_idx = order[num_verified+close_idx_among_tested+1]

            break
    if num_verified == d-1:
        num_verified += 1
    if num_verified >= K:
        return num_verified, None, None
    return num_verified, close_idx, n_samples_to_reject



def rankshap(model, X, xloc, K, alpha=0.1, mapping_dict=None, 
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
    if xloc.ndim==1:
        xloc = xloc.reshape(1,-1)
    converged = False
    diffs_all_feats = helper_shapley_sampling.compute_diffs_all_feats(model, X, xloc, n_init, 
                                            mapping_dict=mapping_dict, 
                                            n_samples_per_perm=n_samples_per_perm)
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    N_total = n_init*d
    while True:
        shap_ests = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats, abs=abs)
        shap_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats)
        # value_vars = diffs_to_shap_vars(diffs_all_feats, var_of_mean=False)
        num_verified, close_idx, n_samples_to_reject = find_num_verified_rankshap(diffs_all_feats, alpha=alpha, 
                                                                                  abs=abs, n_equal=n_equal, K=K)
        if num_verified >= K:
            break
        order = helper.get_ranking(shap_ests, abs=abs)
        # Number of tests passed = index of test with first failure
        failure_idx = order[num_verified]

        # print(num_verified)
        # print(order)
        # print(failure_idx, close_idx)
        exceeded = False
        # Run until order of pair is stable
        ordered_pair = [failure_idx, close_idx]
        while True:
            # Run for suggested number of samples to be significant difference
            n_to_run = [int(buffer*n) for n in n_samples_to_reject]
            # print("n to run:", np.array(n_to_run).astype(int))
            # Suggested runtime exceeds computational maximum
            if max(n_to_run) > max_n_perms:
                # Run 1x for that maximum
                if not exceeded:
                    # Could make it balanced if unequal variance
                    n_to_run = [max_n_perms, max_n_perms]
                    exceeded = True
                else: 
                    # Didn't stabilize with max budget. Return unconverged results.
                    shap_ests = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats)
                    return shap_ests, diffs_all_feats, N_total, converged
            diffs_pair = []
            for i, idx_to_resample in enumerate(ordered_pair):
                w_vals, wj_vals = [], []
                num_samples = n_to_run[i]
                for _ in range(num_samples):
                    # Generate new permutations and thus new x_{S^c | S}
                    perm = np.random.permutation(d)
                    j_idx = np.argwhere(perm==idx_to_resample).item()
                    S = np.array(perm[:j_idx])
                    
                    tw_vals, twj_vals = helper_shapley_sampling.query_values_marginal(X, xloc, S, idx_to_resample, mapping_dict, n_samples_per_perm)
                    w_vals.append(tw_vals)
                    wj_vals.append(twj_vals)
                N_total += num_samples
                w_vals = np.reshape(w_vals, [-1, xloc.shape[1]])
                wj_vals = np.reshape(wj_vals, [-1, xloc.shape[1]])
                
                diffs_all = model(wj_vals) - model(w_vals)
                diffs_avg = np.mean(np.reshape(diffs_all,[-1,n_samples_per_perm]),axis=1) # length M
                diffs_pair.append(diffs_avg)
            # Replace with new samples
            diffs_all_feats[ordered_pair[0]] = diffs_pair[0]
            diffs_all_feats[ordered_pair[1]] = diffs_pair[1]

            # Test for stability between two features, whichever is bigger now
            shap_ests = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats, abs=abs)
            shap_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats)
            value_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats, var_of_mean=False)

            # Establish new order, having resampled
            new_order = helper.get_ranking(shap_ests, abs=abs)
            larger_resampled_idx = failure_idx if shap_ests[failure_idx] >= shap_ests[close_idx] else close_idx
            smaller_resampled_idx = close_idx if larger_resampled_idx==failure_idx else failure_idx
            ordered_pair = [larger_resampled_idx, smaller_resampled_idx]

            idx_to_test = new_order[np.where(new_order == larger_resampled_idx)[0][0]:]
            means_to_test = shap_ests[idx_to_test]
            vars_to_test = shap_vars[idx_to_test]
            value_vars_to_test = value_vars[idx_to_test]
            # Identify index (>= 1) of element in "order" that failure_idx = order[num_verified] is being compared against
            close_idx_among_tested = np.where(idx_to_test == smaller_resampled_idx)[0][0]

            # Perform the test
            test_result, n_samples_to_reject = helper.test_for_max(means_to_test, vars_to_test, 
                                                                   close_idx_among_tested, alpha, 
                                                                   compute_sample_size=True, n_equal=n_equal,
                                                                   value_vars=value_vars_to_test)
            if test_result == "reject":
                break
            

    shap_vals = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats)
    converged = True
    return shap_vals, diffs_all_feats, N_total, converged


############### SPRT-SHAP (KernelSHAP) ###############

    
def find_num_verified_sprtshap(shap_ests, shap_vars, alpha=0.1, beta=0.2,
                               abs=True, K=None):
    # Had used alpha/2 but I don't think this is right!
    acceptNullThresh = beta/(1-alpha)
    rejectNullThresh = (1-beta)/(alpha)
    order = get_ranking(shap_ests, abs=abs)
    if abs: shap_ests = np.abs(shap_ests)
    num_verified = 0
    d = len(shap_ests)
    max_num_tests = d-1 if K is None else K
    while num_verified < max_num_tests:
        # Perform test on index "num_verified"
        idx_to_test = order[num_verified:].astype(int)
        means_to_test = shap_ests[idx_to_test]
        vars_to_test = shap_vars[idx_to_test]

        # Find index with biggest variance. 
        # Subsequent tests will necessarily have lower p-values, so they don't need to be tested.
        max_test_idx = np.argmax(vars_to_test[1:]) + 1
        p_vals = []
        # Identify index with the highest p-value (i.e. likelihood) under the null.
        for j in range(1, max_test_idx+1):
            _, p_value = helper.test_for_max(means_to_test, vars_to_test, j, alpha, return_p_val=True)
            p_vals.append(p_value)
        null_p_val = np.max(p_vals)
        
        j_max = np.argmax(p_vals) + 1    
        
        # Test 0 vs j under alternate hypothesis - true mean > 0.
        x1, xj = means_to_test[0], means_to_test[j_max]
        s1, sj = vars_to_test[0], vars_to_test[j_max]
        Delta = x1 - xj
        mu_alt_1j = (sj*x1+s1*xj+s1*Delta) / (s1+sj)
        s_1j = s1**2 / (s1+sj)
        num_stat = (x1 - mu_alt_1j)/np.sqrt(s_1j)
        num = 1-norm.cdf(num_stat)
        highest_unseen_idx = 1 if j_max > 1 else 2
        denom_stat = (means_to_test[highest_unseen_idx] - mu_alt_1j)/np.sqrt(s_1j)
        denom = 1-norm.cdf(denom_stat)
        alt_p_val = num/denom

        # Accept null, reject null, or continue sampling depending on likelihood ratio
        LR = alt_p_val / null_p_val
        # print(num_verified)
        # print(LR)
        if LR > rejectNullThresh:
            num_verified += 1
            
        else: 
            # Should never have LR < acceptNullThresh
            return num_verified
    return num_verified



def sprtshap(model, X, xloc, K, mapping_dict=None, 
                n_samples_per_perm=5, n_perms_btwn_tests=100, n_max=100000, 
                alpha=0.1, beta=0.2, abs=True):
    if xloc.ndim==1:
        xloc = xloc.reshape(1,-1)
    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    N = 0
    num_verified = 0
    
    # coalitions, coalition_values, coalition_vars = [], [], []
    while N < n_max and num_verified < K:
        coalitions_t, coalition_values_t, coalition_vars_t = compute_coalitions_values(model, X, xloc, 
                                                                        n_perms_btwn_tests, n_samples_per_perm, 
                                                                        mapping_dict)
        N += n_perms_btwn_tests
        if N > n_perms_btwn_tests:
            # Append onto existing counts
            coalitions = np.concatenate((coalitions, coalitions_t)) # z vectors
            coalition_values = np.concatenate((coalition_values, coalition_values_t)) # E[f(X)|z]
            coalition_vars = np.concatenate((coalition_vars, coalition_vars_t)) # Var[f(X)|z]
        else:
            coalitions, coalition_values, coalition_vars = coalitions_t, coalition_values_t, coalition_vars_t

        # Obtain KernelSHAP values and their covariances
        kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
        # kshap_covs = compute_kshap_vars_ls(coalition_vars,coalitions)
        kshap_covs = compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, 
                    coalition_values, n_boot=250)
        kshap_vars = np.diag(kshap_covs)

        # Find the number of verified rankings
        num_verified = find_num_verified_sprtshap(kshap_vals, kshap_vars, alpha=alpha, 
                                                  beta=beta, abs=abs, K=K)

    if num_verified >= K:
        converged = True
        return kshap_vals, kshap_covs, N, converged
    else:
        # Hit max number of iterations without converging
        converged = False
        return kshap_vals, kshap_covs, N, converged

