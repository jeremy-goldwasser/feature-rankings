import numpy as np
from scipy.stats import t
from scipy.stats import norm

############### Estimation ###############
def map_S(S, mapping_dict):
    '''
    Maps a subset of feature indices to the corresponding subset of columns.
    mapping_dict contains feature indices as keys and their corresponding columns as values.
    '''

    S_cols_list = [mapping_dict[i] for i in S]
    S_cols = sorted([item for sublist in S_cols_list for item in sublist])
    return S_cols


def get_ranking(shap_vals, abs=True):
    # Indices with biggest to smallest (absolute) SHAP value
    if abs:
        return np.argsort(np.abs(shap_vals))[::-1]
    else:
        return np.argsort(shap_vals)[::-1]

############### Top-K Analysis ###############

def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def calc_fwer(top_K, digits=None, rejection_idx=None):
    '''
    Calculates the family-wise error rate (FWER) of the top-K rankings: {# false rejections}/{# total trials}.
    top_k is a np.array of shape (N_runs, K)
    rejection_idx is a subset of the N_runs iterations in which the test rejected
    '''
    most_common_row = mode_rows(top_K)
    relevant_top_K = np.array(top_K)
    if rejection_idx is not None:
        # Rows where test rejected
        relevant_top_K = relevant_top_K[rejection_idx]
    num_false_rejections = np.sum(np.all(relevant_top_K!=most_common_row,axis=1)).item()
    num_total_trials = len(top_K)
    fwer = num_false_rejections/num_total_trials
    if digits:
        return np.round(fwer, digits)
    return fwer
    
def shap_vals_to_ranks(shap_vals, abs=True):
    N_pts, N_runs, d = shap_vals.shape
    shap_ranks = np.array([get_ranking(shap_vals[i,j,:], abs=abs) for i in range(N_pts) for j in range(N_runs)]).reshape(shap_vals.shape)
    return shap_ranks

############### Retrospective Analysis ###############

def calc_retro_fwer(GTranks, rankings, N_verified, digits=3):
    '''
    Calculates FWER of observed rankings for a single alpha on a given sample. 
    Different iterations may have verified different numbers of ranks.
    '''
    # N_runs, _ = rankings.shape
    # nStable = np.sum(N_verified > 0)
    # if nStable <= thresh*N_runs: # Majority unverified
    #    # FWER will not be computed on iterations that verified too infrequently.
    #     # print("Majority unverified.")
    #     return np.nan
    # prop_stable = 0
    # # Number of runs with at least one stable rank
    # for runIdx in range(N_runs):
    #     nVerif = N_verified[runIdx]
    #     if nVerif > 0:
    #         stableRanks = rankings[runIdx,:nVerif]
    #         was_stable = np.array_equal(stableRanks, GTranks[:nVerif])
    #         prop_stable += was_stable
    #     else:
    #         prop_stable += 1
    # # prop_stable /= nStable
    # prop_stable /= N_runs
    prop_stable = np.mean([
        np.array_equal(rankings[i, :nVerif], GTranks[:nVerif]) if nVerif > 0 else True
        for i, nVerif in enumerate(N_verified)
    ])
    fwer = 1 - prop_stable
    if digits:
        return np.round(fwer, digits)
    return fwer


def calc_all_retro_fwers(N_verified_all, ranks, avgRanks, digits=3):
    N_pts, _, N_alphas = N_verified_all.shape
    fwers_all = [
        [
            calc_retro_fwer(avgRanks[ptIdx], ranks[ptIdx], N_verified_all[ptIdx,:,alphaIdx], digits=digits)
            for ptIdx in range(N_pts)
        ]
        for alphaIdx in range(N_alphas)
    ]
    fwers_all = np.array(fwers_all)
    # fwers_all = np.where(np.array(fwers_all, dtype=object) == None, np.nan, fwers_all)
    return fwers_all

    
############### Top-K Ranking ###############

def test_for_max(means, vars_of_means, j, alpha, 
                 compute_sample_size=False, n_equal=True, 
                 value_vars=None, return_p_val=False):
    # Assumes means are sorted in decreasing order.
    # Only use 2 means & variances considered, as well as the second-biggest mean.
    # If calculating sample size, use those two variances.
    x1, xj = means[0], means[j]
    s1, sj = vars_of_means[0], vars_of_means[j]
    mu_1j = (x1*sj + xj*s1)/(s1+sj)
    s_1j = (s1**2)/(s1+sj)
    num_stat = (x1 - mu_1j)/np.sqrt(s_1j)
    num = 1-norm.cdf(num_stat)
    # Compute denominator. Don't need to break up, just nice to see.
    if j>=2 and means[1] > mu_1j:
        denom_stat = (means[1] - mu_1j)/np.sqrt(s_1j)
        denom = 1-norm.cdf(denom_stat)
    else:
        denom = 0.5

    p_val = num/denom
    # print("j, p-value:", j, np.round(p_val,4))
    result = "reject" if p_val < alpha or np.isnan(p_val) else "fail to reject"
    if not compute_sample_size:
        if return_p_val:
            return result, p_val
        return result
    else:
        if p_val < alpha:
            if return_p_val:
                return result, p_val, None
            return result, None
        else:
            Z_crit = norm.ppf(1-alpha/2)
            value_var_1, value_var_j = value_vars[0], value_vars[j]
            if n_equal:
                n_to_run = (Z_crit/(x1-xj))**2 * (value_var_1 + value_var_j)
                n_to_reject_pair = np.ceil([n_to_run, n_to_run])
            else:
                n_to_run_1 = (Z_crit/(x1-xj))**2 * (2*value_var_1)
                n_to_run_j = (Z_crit/(x1-xj))**2 * (2*value_var_j)
                n_to_reject_pair = np.ceil([n_to_run_1, n_to_run_j])
            if return_p_val:
                return result, p_val, n_to_reject_pair
            return result, n_to_reject_pair


# def find_num_verified(shap_ests, shap_vars, alpha=0.1, abs=True):
#     # k passed <=> passed through rank k vs k+1 <=> first failure at test idx k vs k+1
#     d = len(shap_ests)
#     order = get_ranking(shap_ests, abs=abs)
#     if abs:
#         shap_ests = np.abs(shap_ests)
#     num_verified = 0
#     # Test stability of 1 vs 2; 2 vs 3; etc (d-1 total tests)
#     max_num_tests = d-1
#     while num_verified < max_num_tests:
#         idx_to_test = order[num_verified:].astype(int)
#         means_to_test = shap_ests[idx_to_test]
#         vars_to_test = shap_vars[idx_to_test]

#         # Find index with biggest variance. 
#         max_test_idx = np.argmax(vars_to_test[1:]) + 1
#         # print(np.round(np.sqrt(vars_to_test)*1000))
#         # Subsequent tests will necessarily have lower p-values, so they don't need to be tested.
#         # print(num_verified)
#         for j in range(1, max_test_idx+1):
#             test_result = test_for_max(means_to_test, vars_to_test, j, alpha)
#             if test_result=="fail to reject":
#                 break
#         # print(test_result)
#         # Reject null if all tests reject (max p-value < alpha)
#         if test_result=="reject":
#             num_verified += 1
#         else:
#             break
#     if num_verified == d-1:
#         num_verified += 1
#     return num_verified 

def find_num_verified(shap_ests, shap_vars, alpha=0.1, abs=True, 
                      compute_sample_size=False,
                      K=None, n_equal=True, value_vars=None):
    # k passed <=> passed through rank k vs k+1 <=> first failure at test idx k vs k+1
    d = len(shap_ests)
    order = get_ranking(shap_ests, abs=abs)
    if abs:
        shap_ests = np.abs(shap_ests)
    num_verified = 0
    # Test stability of 1 vs 2; 2 vs 3; etc (d-1 total tests)
    max_num_tests = d-1
    while num_verified < max_num_tests:
        idx_to_test = order[num_verified:].astype(int)
        means_to_test = shap_ests[idx_to_test]
        vars_to_test = shap_vars[idx_to_test]

        # Find index with biggest variance. 
        max_test_idx = np.argmax(vars_to_test[1:]) + 1
        reject = True
        # Subsequent tests will necessarily have lower p-values, so they don't need to be tested.
        # print(num_verified)
        ns_to_reject_all = []
        for j in range(1, max_test_idx+1):
            result = test_for_max(means_to_test, vars_to_test, j, alpha,
                                       compute_sample_size=compute_sample_size, 
                                       n_equal=n_equal, value_vars=value_vars)
            test_result, n_to_reject_pair = result if compute_sample_size else (result, None)
            if test_result=="fail to reject":
                reject = False
                if compute_sample_size:
                    ns_to_reject_all.append(n_to_reject_pair)
                else:
                    break
        
        if reject:
            # Reject null if all tests reject (max p-value < alpha)
            num_verified += 1
        else:
            # Fail to reject - cut off
            if compute_sample_size:
                n_totals = np.sum(ns_to_reject_all, axis=1)
                close_idx_among_tested = np.argmin(n_totals)
                n_to_reject_pair = ns_to_reject_all[close_idx_among_tested]

                close_idx = order[num_verified+close_idx_among_tested+1]
                failure_idx = order[num_verified]
                pair_idx = [failure_idx, close_idx]
            break
    if num_verified == d-1:
        num_verified += 1
    if compute_sample_size:
        if num_verified >= K:
            return num_verified, None, None
        return num_verified, pair_idx, n_to_reject_pair

    return num_verified 

############### Top-K Set ###############

def test_top_k_set(means, vars_of_means, K, alpha, abs=True,
                   compute_sample_size=False, n_equal=True, value_vars=None):
    # Split into top K and bottom D-K
    d = len(means)
    order = get_ranking(means, abs=abs)
    if abs:
        means = np.abs(means)
    top_K_means = means[order[:K]]
    top_K_vars = vars_of_means[order[:K]]
    bottom_means = means[order[K:]]
    bottom_vars = vars_of_means[order[K:]]
    
    if value_vars is not None:
        top_K_value_vars = value_vars[order[:K]]
        bottom_value_vars = value_vars[order[K:]]
        
    reject = True
    ns_to_reject_all = []
    pair_ranks = []
    p_vals = []
    for i in range(K):
        # Start with lower-ranked indices. Makes rejection faster if not computing sample sizes.
        top_K_idx = K - i - 1
        relevant_means = np.hstack((top_K_means[top_K_idx], bottom_means))
        relevant_vars = np.hstack((top_K_vars[top_K_idx], bottom_vars))
        relevant_value_vars = np.hstack((top_K_value_vars[top_K_idx], bottom_value_vars)) if value_vars is not None else None
        for j in range(K, d):
            bottom_D_minus_K_idx = j - K + 1
            result = test_for_max(relevant_means,
                                relevant_vars,
                                bottom_D_minus_K_idx, 
                                alpha, 
                                compute_sample_size=compute_sample_size, 
                                n_equal=n_equal, 
                                value_vars=relevant_value_vars,
                                return_p_val=compute_sample_size)
            test_result, p_val, n_to_reject_pair = result if compute_sample_size else (result, None, None)
            if test_result == "fail to reject":
                reject = False
                if compute_sample_size:
                    ns_to_reject_all.append(n_to_reject_pair)
                    # top_K_idx is in K, K-1, .., 1; j is in K, K+1, .., d-1
                    pair_ranks.append((top_K_idx, j)) 
                    p_vals.append(p_val)
                else:
                    break
        if not reject and not compute_sample_size:
            break
    
    
    if compute_sample_size:
        if reject:
            return "reject", None, None
        # WRONG: Identify the pair of indices that failed to reject with fewest samples to reject
        # n_totals = np.sum(np.array(ns_to_reject_all), axis=1)
        # best_idx = np.argmin(n_totals)

        # RIGHT: Identify the pair of indices that failed to reject with largest p-value
            # WEIRD: Getting a lot of nan p-values. Now lumping them into rejections.

        best_idx = np.nanargmax(p_vals)
        n_to_reject_pair = ns_to_reject_all[best_idx]
        pair_rank = pair_ranks[best_idx]
        pair_idx = [order[pair_rank[0]].item(), order[pair_rank[1]].item()]

        return "fail to reject", pair_idx, n_to_reject_pair
    return "reject" if reject else "fail to reject"