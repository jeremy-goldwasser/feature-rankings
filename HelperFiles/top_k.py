import numpy as np
from scipy.stats import t, nct, norm
import helper
import helper_shapley_sampling
import helper_kernelshap


################ RankSHAP (Shapley Smampling) ################


def rankshap(model, X, xloc, K, alpha=0.1, mapping_dict=None, guarantee="rank",
            n_samples_per_perm=10, n_init=100, max_n_perms=10000,  
            n_equal=True, buffer=1.1, abs=True):
    '''
    - model: Inputs a numpy array, outputs a scalar
    - X: N by D matrix of samples
    - xloc: 1 by D matrix with one sample, whose SHAP values are estimated
    - K: Number of features we want to rank correctly
    - alpha: Significance level
    - mapping_dict: Dictionary mapping categorical variables to corresponding binary columns of X and xloc
    - guarantee: "rank" or "set". "rank" tests for order within top K, while "set" merely tests for belonging
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
        value_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats, var_of_mean=False)
        if guarantee=="rank":
            num_verified, pair_idx, n_to_reject_pair = helper.find_num_verified(shap_ests, shap_vars, alpha=alpha, 
                                                                        abs=abs, compute_sample_size=True, 
                                                                        K=K, n_equal=n_equal, value_vars=value_vars)
            if num_verified >= K:
                break
        else:
            test_result, pair_idx, n_to_reject_pair = helper.test_top_k_set(shap_ests, shap_vars, 
                                                                        K, alpha, abs=abs, 
                                                                        compute_sample_size=True, 
                                                                        n_equal=n_equal, value_vars=value_vars)
            if test_result == "reject":
                break

        # order = helper.get_ranking(shap_ests, abs=abs)
        # Number of tests passed = index of test with first failure
        # failure_idx = order[num_verified]

        exceeded = False
        # Run until order of pair is stable
        failure_idx, close_idx = pair_idx
        # pair_idx = [failure_idx, close_idx]
        while True:
            n_to_run = [int(buffer*n) for n in n_to_reject_pair]
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
            for i, idx_to_resample in enumerate(pair_idx):
                w_vals, wj_vals = [], []
                num_samples = max(n_to_run[i], n_init)
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
            diffs_all_feats[pair_idx[0]] = diffs_pair[0]
            diffs_all_feats[pair_idx[1]] = diffs_pair[1]

            # Test for stability between two features, whichever is bigger now
            shap_ests = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats, abs=abs)
            shap_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats)
            value_vars = helper_shapley_sampling.diffs_to_shap_vars(diffs_all_feats, var_of_mean=False)

            # Establish new order, having resampled
            new_order = helper.get_ranking(shap_ests, abs=abs)
            larger_resampled_idx = failure_idx if shap_ests[failure_idx] >= shap_ests[close_idx] else close_idx
            smaller_resampled_idx = close_idx if larger_resampled_idx==failure_idx else failure_idx
            pair_idx = [larger_resampled_idx, smaller_resampled_idx]

            idx_to_test = new_order[np.where(new_order == larger_resampled_idx)[0][0]:]
            means_to_test = shap_ests[idx_to_test]
            vars_to_test = shap_vars[idx_to_test]
            value_vars_to_test = value_vars[idx_to_test]
            # Identify index (>= 1) of element in "order" that failure_idx = order[num_verified] is being compared against
            close_idx_among_tested = np.where(idx_to_test == smaller_resampled_idx)[0][0]

            # Perform the test
            test_result, n_to_reject_pair = helper.test_for_max(means_to_test, vars_to_test, 
                                                                   close_idx_among_tested, alpha, 
                                                                   compute_sample_size=True, n_equal=n_equal,
                                                                   value_vars=value_vars_to_test)
            if test_result == "reject":
                break

    shap_vals = helper_shapley_sampling.diffs_to_shap_vals(diffs_all_feats)
    converged = True
    return shap_vals, diffs_all_feats, N_total, converged


############### SPRT-SHAP (KernelSHAP) ###############

def perform_SPRT(sorted_data, vars, alpha, beta):
    def calc_LR(x1, s1, xj, sj, min_x1):
        Delta = x1 - xj
        mu_alt_1j = (sj*x1 + s1*xj + s1*Delta) / (s1+sj)
        s_1j = s1**2 / (s1+sj)
        alt_num = norm.pdf(x1, mu_alt_1j, np.sqrt(s_1j))

        alt_denom = 1 - norm.cdf(min_x1, mu_alt_1j, np.sqrt(s_1j))
        alt_likelihood = alt_num / alt_denom

        mu_1j = (sj*x1 + s1*xj) / (s1+sj)
        null_num = norm.pdf(x1, mu_1j, np.sqrt(s_1j))
        null_denom = 1 - norm.cdf(min_x1, mu_1j, np.sqrt(s_1j))
        null_likelihood = null_num / null_denom

        LR_1j = alt_likelihood / null_likelihood
        return LR_1j
    
    LRs = []
    x1, s1 = sorted_data[0], vars[0]
    for j in range(1, len(sorted_data)): # d-K comparisons
        xj, sj = sorted_data[j], vars[j]
        min_x1 = sorted_data[1] if j>1 else sorted_data[2]
        LR_1j = calc_LR(x1, s1, xj, sj, min_x1)
        LRs.append(LR_1j)
    LR = np.min(LRs)
    
    rejectNullThresh = (1-beta)/(alpha)
    if LR > rejectNullThresh:
        return "reject"
    else:    
        return "fail to reject"


def find_num_verified_sprtshap(shap_ests, shap_vars, alpha=0.1, beta=0.2,
                               abs=True, K=None):
    order = helper.get_ranking(shap_ests, abs=abs)
    if abs: shap_ests = np.abs(shap_ests)
    num_verified = 0
    d = len(shap_ests)
    max_num_tests = d-1 if K is None else K
    while num_verified < max_num_tests:
        # Perform test on index "num_verified"
        idx_to_test = order[num_verified:].astype(int)
        relevant_ests = shap_ests[idx_to_test]
        relevant_vars = shap_vars[idx_to_test]

        SPRT_result = perform_SPRT(relevant_ests, relevant_vars, alpha, beta)
        if SPRT_result == "reject":
            num_verified += 1
        else: 
            # Should never have LR < acceptNullThresh; 
            # Either way, return number of previously verified rankings
            return num_verified
    return num_verified

    
def top_K_set_sprtshap(ests, vars, alpha=0.1, beta=0.2, abs=True, K=None):
    order = helper.get_ranking(ests, abs=abs)
    if abs:
        ests = np.abs(ests)
    top_K_ests = ests[order[:K]]
    top_K_vars = vars[order[:K]]
    bottom_means = ests[order[K:]]
    bottom_vars = vars[order[K:]]
            
    for i in range(K):
        # Pick a top-ranked feature, starting with hardest
        top_K_idx = K - i - 1
        x1, s1 = top_K_ests[top_K_idx], top_K_vars[top_K_idx]
        relevant_ests = np.hstack((x1, bottom_means))
        relevant_vars = np.hstack((s1, bottom_vars))

        # Test whether it is significantly higher than the lower-ranked ones
        SPRT_result = perform_SPRT(relevant_ests, relevant_vars, alpha, beta)
        
        # If so, move on to the next top-ranked feature; 
        # otherwise, return that not all features K rejected
        if SPRT_result == "fail to reject":
            return "fail to reject"
        
    return "reject"



def sprtshap(model, X, xloc, K, 
             mapping_dict=None, guarantee="rank",
             n_samples_per_perm=10, 
             n_perms_btwn_tests=100, n_max=100000, 
             alpha=0.1, beta=0.2, abs=True):
    if xloc.ndim==1:
        xloc = xloc.reshape(1,-1)
    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    N = 0
    num_verified = 0
    
    # coalitions, coalition_values, coalition_vars = [], [], []
    while N < n_max:# and num_verified < K:
        coalitions_t, coalition_values_t, coalition_vars_t = helper_kernelshap.compute_coalitions_values(model, X, xloc, 
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
        kshap_vals = helper_kernelshap.kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
        # kshap_covs = compute_kshap_vars_ls(coalition_vars,coalitions)
        kshap_covs = helper_kernelshap.compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, 
                    coalition_values, n_boot=250)
        kshap_vars = np.diag(kshap_covs)

        # Find the number of verified rankings
        if guarantee=="rank":
            num_verified = find_num_verified_sprtshap(kshap_vals, kshap_vars, alpha=alpha, 
                                                    beta=beta, abs=abs, K=K)
            if num_verified >= K:
                converged = True
                return kshap_vals, kshap_covs, N, converged
            elif num_verified < 0:
                converged = False
                return kshap_vals, kshap_covs, N, converged
            else:
                continue
        else:
            test_result = top_K_set_sprtshap(kshap_vals, kshap_vars, alpha=alpha, 
                                             beta=beta, abs=abs, K=K)
            if test_result == "reject":
                converged = True
                return kshap_vals, kshap_covs, N, converged

    # Hit max number of iterations without converging
    converged = False
    return kshap_vals, kshap_covs, N, converged

