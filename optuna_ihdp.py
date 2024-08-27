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


def adjust_learning_rate(optimizer, init_lr, epoch, lr_type='fixed', num_epoch=800):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()


def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()


def objective(trial):
    # Define the hyperparameter search space
    init_lr = trial.suggest_float('init_lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.99)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    beta = trial.suggest_float('beta', 0.1, 10.0)
    gamma = trial.suggest_float('gamma', 0.1, 10.0)
    lambda_ = trial.suggest_float('lambda_', 0.0, 1.0)
    num_epoch = 100  # We can fix this or tune as well

    # Other fixed configurations (you can also tune these if needed)
    cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]

    # Load your dataset
    data_matrix = pd.read_csv(os.path.join(
        'dataset/ihdp', 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join('dataset/ihdp', 't_grid.csv')).values

    idx = list(range(data_matrix.shape[0]))
    random.shuffle(idx)
    sample_size = 200
    train_idx = idx[:sample_size]
    test_idx = idx[sample_size:sample_size + 200]

    train_matrix = torch.tensor(data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(data_matrix[test_idx, :], dtype=torch.float32)
    t_grid = torch.tensor(t_grid_all[:, test_idx], dtype=torch.float32)

    # Initialize model and optimizer
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)

    # Training loop
    for epoch in range(num_epoch):
        train_loader = get_iter(
            train_matrix, batch_size=len(train_matrix), shuffle=True)

        for inputs, y in train_loader:
            t = inputs[:, 0]
            x = inputs[:, 1:]

            optimizer.zero_grad()
            out = model.forward(t, x)
            loss = criterion(out, y, alpha=alpha)
            loss.backward()
            optimizer.step()

    # Evaluate model performance using MSE
    t_grid_hat, mse = curve(model, train_matrix, t_grid)

    return mse


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
