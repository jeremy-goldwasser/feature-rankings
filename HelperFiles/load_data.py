import pandas as pd
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split

def load_credit(path):
    # df = sage.datasets.credit()
    # df.to_csv("Data/credit.csv")
    df = pd.read_csv(join(path, "credit.csv"), index_col=0)

    # Property, other installment, housing, job, status of checking act, credit history, purpose, savings, employment since, marital status, old debtors
    n = df.shape[0]
    X_df = df.drop(["Good Customer"], axis=1)
    y = df["Good Customer"]

    categorical_columns = [
        'Checking Status', 'Credit History', 'Purpose', #'Credit Amount', # It's listed but has 923 unique values
        'Savings Account/Bonds', 'Employment Since', 'Personal Status',
        'Debtors/Guarantors', 'Property Type', 'Other Installment Plans',
        'Housing Ownership', 'Job', #'Telephone', 'Foreign Worker' # These are just binary
    ]
    X_binarized = pd.get_dummies(X_df, columns=categorical_columns)

    mapping_dict = {}
    for i, col in enumerate(X_df.columns):
        bin_cols = []
        for j, bin_col in enumerate(X_binarized.columns):
            if bin_col.startswith(col):
                bin_cols.append(j)
        mapping_dict[i] = bin_cols

    np.random.seed(1)
    X_norm = (X_binarized-X_binarized.min())/(X_binarized.max()-X_binarized.min())
    n_train = round(n*0.8)
    train_idx = np.random.choice(n, n_train, replace=False)
    X_train, y_train = X_norm.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy()
    test_idx = np.setdiff1d(np.arange(n),train_idx)
    X_test, y_test = X_norm.iloc[test_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    return X_train, y_train, X_test, y_test, mapping_dict


def load_brca(path):
    np.random.seed(1)
    data = pd.read_csv(join(path, "brca_small.csv"))
    X = data.values[:, :-1][:,:20]
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


def load_census(path):
    # Adult census income dataset
    # import shap
    # X_display, y_display = shap.datasets.adult(display=True)
    # X_display.to_csv("Data/census_X.csv")
    # np.save("Data/census_y.npy", y_display)
    X_display = pd.read_csv(join(path, "census_X.csv"), index_col=0)
    y_display = np.load(join(path, "census_y.npy"))
    X_binarized = pd.get_dummies(X_display)

    mapping_dict = {}
    for i, col in enumerate(X_display.columns):
        bin_cols = []
        for j, bin_col in enumerate(X_binarized.columns):
            if bin_col.startswith(col):
                bin_cols.append(j)
        mapping_dict[i] = bin_cols

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

def load_data(dataset, path):
    if dataset=="brca":
        return load_brca(path)
    if dataset=="credit":
        return load_credit(path)
    if dataset=="census":
        return load_census(path)
    print("Dataset must be brca, credit, or census.")
    return -1