import torch
import math
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from scipy.spatial.distance import pdist, squareform
import osqp
from scipy import sparse
import optuna

from models.dynamic_net import Vcnet, TR, Drnet
from utils.eval import curve


class Dataset_from_matrix(Dataset):
    """Dataset created from a tensor data_matrix."""

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def criterion(out, y, alpha=0.5, epsilon=1e-5):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()


def criterion_TR(out, trg, y, beta=1., epsilon=1e-5):
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()


def objective(trial):
    # Define the hyperparameter search space
    init_lr = trial.suggest_float('init_lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    beta = trial.suggest_float('beta', 0.1, 10.0)
    tr_init_lr = trial.suggest_float('tr_init_lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.99)

    # Add model architecture parameters to Optuna
    hidden_units1 = trial.suggest_int(
        'hidden_units1', 30, 100)  # Hidden layer 1
    hidden_units2 = trial.suggest_int(
        'hidden_units2', 30, 100)  # Hidden layer 2
    num_grid = 10
    degree = 2
    knots = [0.33, 0.66]
    n_epochs = 400
    verbose = 100
    # Load dataset
    data_matrix = pd.read_csv(os.path.join(
        'dataset/news', 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join('dataset/news', 't_grid.csv')).values

    idx = list(range(data_matrix.shape[0]))
    random.shuffle(idx)
    sample_size = 2500  # You can tune sample size if needed
    train_idx = idx[:sample_size]
    test_idx = idx[sample_size:sample_size + 500]

    train_matrix = torch.tensor(data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(data_matrix[test_idx, :], dtype=torch.float32)
    t_grid = torch.tensor(t_grid_all[:, test_idx], dtype=torch.float32)

    # Define the model based on the selected model_name
    model_name = trial.suggest_categorical(
        'model_name', ['Vcnet', 'Vcnet_TR', 'Drnet_tr'])

    if model_name == 'Vcnet' or model_name == 'Vcnet_TR':
        cfg_density = [(498, hidden_units1, 1, 'relu'),
                       (hidden_units1, hidden_units2, 1, 'relu')]
        cfg = [(hidden_units2, hidden_units2, 1, 'relu'),
               (hidden_units2, 1, 1, 'id')]
        model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    elif model_name == 'Drnet_tr':
        cfg_density = [(498, hidden_units1, 1, 'relu'),
                       (hidden_units1, hidden_units2, 1, 'relu')]
        cfg = [(hidden_units2, hidden_units2, 1, 'relu'),
               (hidden_units2, 1, 1, 'id')]
        isenhance = 1
        model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)

    # Move model to device and initialize weights

    model._initialize_weights()

    # Set optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)

    train_loader = get_iter(
        train_matrix, batch_size=len(train_matrix), shuffle=True)

    # Training loop
    for epoch in range(n_epochs):
        for inputs, y in train_loader:
            t = inputs[:, 0]
            x = inputs[:, 1:]

            optimizer.zero_grad()
            out = model.forward(t, x)
            loss = criterion(out, y, alpha=alpha)
            loss.backward()
            optimizer.step()

        if epoch % verbose == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluate model performance using MSE
    t_grid_hat, mse = curve(model, train_matrix, t_grid)
    mse = float(mse)

    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with News data')

    parser.add_argument('--data_dir', type=str,
                        default='dataset/news', help='dir of data matrix')
    parser.add_argument('--save_dir', type=str,
                        default='logs/news/eval', help='dir to save result')
    parser.add_argument('--n_epochs', type=int, default=800,
                        help='num of epochs to train')
    parser.add_argument('--verbose', type=int, default=100,
                        help='print train info freq')

    args = parser.parse_args()

    # Use Optuna for hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
