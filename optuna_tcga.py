
import torch
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
import optuna
from models.dynamic_net import Vcnet
from utils.eval import curve
import argparse

# Dataset class from matrix


class Dataset_from_matrix(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

# DataLoader helper function


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

# Loss function for Vcnet


def criterion(out, y, alpha=0.5, epsilon=1e-5):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()

# Objective function for Optuna optimization


def objective(trial):
    # Hyperparameter search space
    init_lr = trial.suggest_float('init_lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('wd', 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    hidden_units1 = trial.suggest_int('hidden_units1', 30, 500)
    hidden_units2 = trial.suggest_int('hidden_units2', 30, 500)
    num_epoch = 300
    verbose = 100
    # Load data
    data_matrix = pd.read_csv(os.path.join(
        args.data_dir, 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join(args.data_dir, 't_grid.csv')).values

    # Random sampling for train and test split
    idx = list(range(data_matrix.shape[0]))
    random.shuffle(idx)
    sample_size = trial.suggest_categorical(
        'sample_size', [1000])

    train_idx = idx[:sample_size]
    test_idx = idx[sample_size:sample_size + 1000]

    train_matrix = torch.tensor(data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(data_matrix[test_idx, :], dtype=torch.float32)
    t_grid = torch.tensor(t_grid_all[:, test_idx], dtype=torch.float32)

    train_loader = get_iter(
        train_matrix, batch_size=len(train_matrix), shuffle=True)
    test_loader = get_iter(
        test_matrix, batch_size=len(test_matrix), shuffle=False)

    # Model configuration for Vcnet
    cfg_density = [(4000, hidden_units1, 1, 'relu'),
                   (hidden_units1, hidden_units2, 1, 'relu')]
    cfg = [(hidden_units2, hidden_units2, 1, 'relu'),
           (hidden_units2, 1, 1, 'id')]
    model = Vcnet(cfg_density, 10, cfg, 2, [0.33, 0.66])

    model._initialize_weights()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=init_lr, momentum=0.8, weight_decay=wd, nesterov=True)

    # Training loop
    for epoch in range(num_epoch):
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

    # Model evaluation using MSE
    t_grid_hat, mse = curve(model, train_matrix, t_grid)
    mse = float(mse)
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate Vcnet model with Optuna')
    parser.add_argument('--data_dir', type=str,
                        default='dataset/tcga', help='Directory for the data')
    parser.add_argument('--n_epochs', type=int,
                        default=400, help='Number of epochs')
    parser.add_argument('--verbose', type=int, default=100,
                        help='Verbose print frequency')

    args = parser.parse_args()

    # Set random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use Optuna for hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print best trial results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
