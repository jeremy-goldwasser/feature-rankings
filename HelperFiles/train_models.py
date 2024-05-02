from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

def train_neural_net(X_train, y_train, lime=False):
    if lime:
        clf = MLPClassifier(random_state=1, hidden_layer_sizes=(50,)).fit(X_train, y_train)
        return clf.predict_proba
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
    return model


def train_logreg(X_train, y_train, lime=False):
    logreg = LogisticRegression(random_state=0).fit(X_train, y_train)
    if lime:
        return logreg.predict_proba
    def model(x):
        return logreg.predict_proba(x)[:,1]
    return model

def train_rf(X_train, y_train, lime=False):
    rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    if lime:
        return rf.predict_proba

    def model(x):
        return rf.predict_proba(x)[:,1]
    return model

def train_model(X_train, y_train, model, lime=False):
    # Already in mapped form
    if model=="nn":
        return train_neural_net(X_train, y_train, lime)
    elif model=="rf":
        return train_rf(X_train, y_train, lime)
    else:
        return train_logreg(X_train, y_train, lime)