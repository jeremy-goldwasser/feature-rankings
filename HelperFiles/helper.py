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

############### General Analysis ###############

def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def calc_fwer(top_K, digits=None):
    most_common_row = mode_rows(top_K)
    fwer = 1 - np.mean(np.all(np.array(top_K)==most_common_row,axis=1))
    if digits:
        return np.round(fwer, digits)
    return fwer
    
def shap_vals_to_ranks(shap_vals, abs=True):
    N_pts, N_runs, d = shap_vals.shape
    shap_ranks = np.array([get_ranking(shap_vals[i,j,:], abs=abs) for i in range(N_pts) for j in range(N_runs)]).reshape(shap_vals.shape)
    return shap_ranks

############### Retrospective Analysis ###############

def calc_retro_fwer(GTranks, rankings, nVerified, alphaIdx):
    nStable = np.sum(nVerified[:,alphaIdx] > 0)
    N_runs, _ = rankings.shape
    if nStable <= 0.05*N_runs: # Majority unverified
        return None
    prop_stable = 0
    # Number of runs with at least one stable rank
    for runIdx in range(N_runs):
        nVerif = nVerified[runIdx,alphaIdx]
        if nVerif > 0:
            stableRanks = rankings[runIdx,:nVerif]
            was_stable = np.array_equal(stableRanks, GTranks[:nVerif])
            prop_stable += was_stable
    prop_stable /= nStable
    fwer = 1 - prop_stable
    return round(fwer,3)


def calc_all_retro_fwers(verif, ranks, avgRanks):
    fwers_all = []
    N_pts, N_runs, N_alphas = verif.shape
    for alphaIdx in range(N_alphas):
        fwers = []
        for ptIdx in range(N_pts):
            GTranks = avgRanks[ptIdx]
            fwer = calc_retro_fwer(GTranks, ranks[ptIdx], verif[ptIdx], alphaIdx)
            fwers.append(fwer)
        fwers_all.append(fwers)
    return np.array(fwers_all)

    
####### TESTING #######

def test_for_max(means, vars_of_means, j, alpha, 
                 compute_sample_size=False, n_equal=True, value_vars=None,
                 return_p_val=False):
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
    result = "reject" if p_val < alpha else "fail to reject"
    if not compute_sample_size:
        if return_p_val:
            return result, p_val
        return result
    else:
        if p_val < alpha:
            return result, None
        else:
            Z_crit = norm.ppf(1-alpha/2)
            value_var_1, value_var_j = value_vars[0], value_vars[j]
            if n_equal:
                n_to_run = (Z_crit/(x1-xj))**2 * (value_var_1 + value_var_j)
                List = [n_to_run, n_to_run]
                return result, List
            else:
                n_to_run_1 = (Z_crit/(x1-xj))**2 * (2*value_var_1)
                n_to_run_j = (Z_crit/(x1-xj))**2 * (2*value_var_j)
                return result, [n_to_run_1, n_to_run_j]


def find_num_verified(shap_ests, shap_vars, alpha=0.1, abs=True):
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
        # print(np.round(np.sqrt(vars_to_test)*1000))
        # Subsequent tests will necessarily have lower p-values, so they don't need to be tested.
        # print(num_verified)
        for j in range(1, max_test_idx+1):
            test_result = test_for_max(means_to_test, vars_to_test, j, alpha)
            if test_result=="fail to reject":
                break
        # print(test_result)
        # Reject null if all tests reject (max p-value < alpha)
        if test_result=="reject":
            num_verified += 1
        else:
            break
    if num_verified == d-1:
        num_verified += 1
    return num_verified 

