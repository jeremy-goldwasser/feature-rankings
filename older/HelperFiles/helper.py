import numpy as np

def correct_cov(cov_mat,Kr = 10000):
    u, s, vh = np.linalg.svd(cov_mat, full_matrices=True)
    if np.max(s)/np.min(s) < Kr:
        cov2 = cov_mat
    else:
        s_max = s[0]
        min_acceptable = s_max/Kr
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        cov2 = np.matmul(u, np.matmul(np.diag(s2), vh))
        
    return (cov2+cov2.T)/2

def logreg_gradient(y_pred, BETA):
    '''
    Computes the gradient of a logistic regression function fmodel with parameters BETA at xloc.
    '''
    d = BETA.shape[0]
    return np.array(y_pred*(1-y_pred)*BETA).reshape((d, 1))


def logreg_hessian(y_pred, BETA):
    '''
    Computes the hessian of a logistic regression function fmodel with parameters BETA at xloc.
    '''
    d = BETA.shape[0]
    beta_2d = np.array(BETA).reshape((d, -1))
    BBT = np.dot(beta_2d, beta_2d.T)
    return y_pred*(1-y_pred)*(1-2*y_pred)*BBT


def f_second_order_approx(y_pred, xnew, xloc, gradient, hessian):
    '''
    Second order approximation to model at xnew, a point around xloc. 
    Relevant when assuming feature independence for both SHAP and kernelSHAP.
    '''
    if xnew.ndim==1:
        xnew = xnew.reshape((1,xnew.shape[0]))
    
    n, d = xnew.shape
    if n==1:
        deltaX = np.array(xnew - xloc).reshape((d, -1))
        second_order_approx = y_pred + np.dot(deltaX.T, gradient) + 0.5*np.dot(np.dot(deltaX.T, hessian), deltaX) 
        return second_order_approx.item()
    else:
        second_order_approx = np.zeros(n)
        for i in range(n):
            deltaX_i = np.array(xnew[i,:] - xloc).reshape((d, -1))
            second_order_approx[i] = y_pred + np.dot(deltaX_i.T, gradient) + 0.5*np.dot(np.dot(deltaX_i.T, hessian), deltaX_i)
        return second_order_approx


def compute_shap_vals_quadratic(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict=None):
    '''
    Computes exact Shapley value of control variate in independent features case (second-order approximation).
    '''
    def compute_jth_shap_val(xloc, feature_means, cov_mat, j, gradient, hessian):
        d = xloc.shape[1]
        mean_j = feature_means[j]
        xloc_j = xloc[0,j]
        linear_term = gradient[j]*(xloc_j - mean_j)
        mean_term = -0.5*(mean_j - xloc_j) * np.sum([(feature_means[k]-xloc[0,k])*hessian[j,k] for k in range(d)])
        var_term = -0.5*np.sum([cov_mat[j,k]*hessian[j,k] for k in range(d)])
        # old_var_term = -0.5*cov_mat[j,j]*hessian[j,j]
        jth_shap_val = linear_term + mean_term + var_term
        return jth_shap_val

    d_total = xloc.shape[1]
    shap_vals = np.array([compute_jth_shap_val(xloc, feature_means, cov_mat, j, gradient, hessian) for j in range(d_total)])

    if mapping_dict is None:
        return shap_vals.reshape(-1)
    else:
        # Account for multilevel features
        true_shap_vals = []
        d = len(mapping_dict)
        for i in range(d):
            relevant_cols = mapping_dict[i]
            if len(relevant_cols)==1: # Not a column of a multilevel feature
                true_shap_vals.append(shap_vals[relevant_cols].item())
            else:
                true_shap_vals.append(np.sum(shap_vals[relevant_cols]))
        true_shap_vals = np.array(true_shap_vals).reshape(-1)

        return true_shap_vals
        

def map_S(S, mapping_dict):
    '''
    Maps a subset of feature indices to the corresponding subset of columns.
    mapping_dict contains feature indices as keys and their corresponding columns as values.
    '''

    S_cols_list = [mapping_dict[i] for i in S]
    S_cols = sorted([item for sublist in S_cols_list for item in sublist])
    return S_cols


def get_ordering(shap_vals):
    return np.argsort(np.abs(shap_vals))[::-1]

def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row

def calc_fwer(top_K):
    most_common_row = mode_rows(top_K)
    fwer = 1 - np.mean(np.all(np.array(top_K)==most_common_row,axis=1))
    return np.round(fwer, 2)