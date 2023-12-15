from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
from helper import *

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

def train_neural_net(X_train, y_train, xloc=None, mapping_dict=None):
    # Convert the input and label data to PyTorch tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    # Compute the class weights
    class_counts = torch.bincount(labels)
    num_samples = len(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]

    # Create a sampler with balanced weights
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

    # Create a DataLoader with the sampler
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    # dataloader = DataLoader(dataset, batch_size=32)

    torch.manual_seed(0)

    # Create an instance
    net = TwoLayerNet(input_size=X_train.shape[1], hidden_size=50, output_size=2)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#.01

    # Iterate over the training data in batches
    num_epochs = 20

    # Train the network for the specified number of epochs
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    def neural_net(x):
        output = net(x)[0,1] if x.shape[0]==1 else net(x)[:,1]
        return output

    def model(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return neural_net(x).detach().numpy()

    if xloc is None:
        return model
    else:
        def compute_hessian(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            dim = x.shape[1]
            hessian = torch.autograd.functional.hessian(neural_net, x)
            hessian = hessian.reshape((dim,dim)).detach().numpy()
            return hessian
        
        feature_means = np.mean(X_train, axis=0)
        cov_mat = correct_cov(np.cov(X_train, rowvar=False))

        # Select point and compute its gradient and hessian
        xloc_torch = torch.tensor(xloc, dtype=torch.float32).requires_grad_(True)
        y_pred = net(xloc_torch)[0,1]
        y_pred.backward()
        gradient = xloc_torch.grad.detach().numpy().reshape((xloc.shape[1], 1))
        hessian = compute_hessian(xloc)

        # Obtain true SHAP values and verify their feasibility
        true_shap_vals = compute_shap_vals_quadratic(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict=mapping_dict)

        y_pred = y_pred.detach().numpy()
        def approx(input):
            return f_second_order_approx(y_pred,input,xloc,gradient,hessian)

        return model, approx, true_shap_vals



def train_logreg(X_train, y_train, xloc=None, mapping_dict=None):
    logreg = LogisticRegression().fit(X_train, y_train)
    # print("Class imbalance: {}".format(100*(max(np.mean(y_test), 1-np.mean(y_test)))))
    # print("Estimation accuracy: {}".format(np.mean((logreg.predict(X_test) > 0.5)==y_test)*100))

    def model(x):
        return logreg.predict_proba(x)[:,1]

    if xloc is None:
        return model
    else:
        feature_means = np.mean(X_train, axis=0)
        cov_mat = correct_cov(np.cov(X_train, rowvar=False))
        BETA = logreg.coef_.reshape(-1)
        y_pred = model(xloc)
        gradient = logreg_gradient(y_pred, BETA)
        hessian = logreg_hessian(y_pred, BETA)

        # Obtain true SHAP values and verify their feasibility
        true_shap_vals = compute_shap_vals_quadratic(xloc, gradient, hessian, feature_means, cov_mat, mapping_dict=mapping_dict)

        def approx(input):
            return f_second_order_approx(y_pred,input,xloc,gradient,hessian)
        
        return model, approx, true_shap_vals