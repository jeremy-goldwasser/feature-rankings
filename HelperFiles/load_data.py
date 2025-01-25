import pandas as pd
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import load_breast_cancer


def make_mapping_dict(X_orig, X_binarized):
    mapping_dict = {}
    for i, col in enumerate(X_orig.columns):
        bin_cols = []
        for j, bin_col in enumerate(X_binarized.columns):
            if bin_col.startswith(col):
                bin_cols.append(j)
        mapping_dict[i] = bin_cols
    return mapping_dict


def make_bank(data_path):
    # import sage
    # df = sage.datasets.bank()
    df = pd.read_csv(join(data_path, "bank", "bank.csv"))
    y = df["Success"]
    X_orig = df.drop(columns=['Success'])
    n = df.shape[0]

    # X_train_raw = np.load(join(data_path, "bank", "X_train.npy"))
    X_binarized = pd.get_dummies(X_orig, dtype=float)
    mapping_dict = make_mapping_dict(X_orig, X_binarized)

    np.random.seed(1)
    X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
    n_train = round(n*0.75)
    train_idx = np.random.choice(n, n_train, replace=False)
    X_train, y_train = X_norm.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy()
    test_idx = np.setdiff1d(np.arange(n),train_idx)
    X_test, y_test = X_norm.iloc[test_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    return X_train, y_train, X_test, y_test, mapping_dict


def make_credit(data_path):
    df = pd.read_csv(join(data_path, "credit", "credit.csv"), index_col=0)

    # Property, other installment, housing, job, status of checking act, credit history, purpose, savings, employment since, marital status, old debtors
    n = df.shape[0]
    X_orig = df.drop(["Good Customer"], axis=1)
    y = df["Good Customer"]

    categorical_columns = [
        'Checking Status', 'Credit History', 'Purpose', #'Credit Amount', # It's listed but has 923 unique values
        'Savings Account/Bonds', 'Employment Since', 'Personal Status',
        'Debtors/Guarantors', 'Property Type', 'Other Installment Plans',
        'Housing Ownership', 'Job', #'Telephone', 'Foreign Worker' # These are just binary
    ]
    X_binarized = pd.get_dummies(X_orig, columns=categorical_columns, dtype=float)

    mapping_dict = make_mapping_dict(X_orig, X_binarized)

    np.random.seed(1)
    X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
    n_train = round(n*0.75)
    train_idx = np.random.choice(n, n_train, replace=False)
    X_train, y_train = X_norm.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy()
    test_idx = np.setdiff1d(np.arange(n),train_idx)
    X_test, y_test = X_norm.iloc[test_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    return X_train, y_train, X_test, y_test, mapping_dict


def make_brca(data_path):
    data = pd.read_csv(join(data_path, "brca", "brca_small.csv"))
    X = data.values[:, :-1][:,:20] # Just use first 20 genes
    Y = data.values[:, -1]
    Y = (Y==2).astype(int) # Formulate as binary classification problem
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=100, random_state=0)

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    mapping_dict = None

    return X_train, y_train, X_test, y_test, mapping_dict

def make_breast_cancer(data_path):
    breast_cancer = load_breast_cancer()
    X = breast_cancer['data']
    Y = breast_cancer['target']
    n = Y.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=int(n*.25), random_state=0)
    mapping_dict = None

    return X_train, y_train, X_test, y_test, mapping_dict

def make_census(data_path):
    # Adult census income dataset
    # import shap
    # X_orig, y_orig = shap.datasets.adult(display=True)
    # X_orig.to_csv("Data/census_X.csv")
    # np.save("Data/census_y.npy", y_orig)
    X_orig = pd.read_csv(join(data_path, "census", "census_X.csv"), index_col=0)
    y_display = np.load(join(data_path, "census", "census_y.npy"))
    X_binarized = pd.get_dummies(X_orig, dtype=float)
    mapping_dict = make_mapping_dict(X_orig, X_binarized)

    X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
    y_int = y_display.astype("int8")

    # Split into training and test sets
    np.random.seed(1)
    n, d = X_norm.shape
    n_train = round(n*0.75)
    train_idx = np.random.choice(n, size=n_train, replace=False)
    X_train_pd, y_train = X_norm.iloc[train_idx], y_int[train_idx]
    X_train = X_train_pd.to_numpy()

    test_idx = np.setdiff1d(list(range(n)), train_idx)
    X_test_pd, y_test = X_norm.iloc[test_idx], y_int[test_idx]
    X_test = X_test_pd.to_numpy()
    
    return X_train, y_train, X_test, y_test, mapping_dict


def make_data(data_path, dataset):
    if dataset=="bank":
        return make_bank(data_path)
    if dataset=="brca":
        return make_brca(data_path)
    if dataset=="credit":
        return make_credit(data_path)
    if dataset=="census":
        return make_census(data_path)
    if dataset=="breast_cancer":
        return make_breast_cancer(data_path)
    print("Dataset must be bank, brca, credit, or census.")
    return -1


def save_data(data_path, dataset, X_train, y_train, X_test, y_test, mapping_dict):
    np.save(join(data_path, dataset, "X_train.npy"), X_train)
    np.save(join(data_path, dataset, "X_test.npy"), X_test)
    np.save(join(data_path, dataset, "y_train.npy"), y_train)
    np.save(join(data_path, dataset, "y_test.npy"), y_test)
    with open(join(data_path, dataset, "mapping_dict"), "wb") as fp:
        pickle.dump(mapping_dict, fp)


def load_data(data_path, dataset):
    X_train = np.load(join(data_path, dataset, "X_train.npy"))
    X_test = np.load(join(data_path, dataset, "X_test.npy"))
    y_train = np.load(join(data_path, dataset, "y_train.npy"))
    y_test = np.load(join(data_path, dataset, "y_test.npy"))
    with open(join(data_path, dataset, "mapping_dict"), "rb") as fp:
        mapping_dict = pickle.load(fp) # May be none, in case of brca
    return X_train, y_train, X_test, y_test, mapping_dict