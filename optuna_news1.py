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

# Import your model classes and any other necessary modules
# from your_module import Vcnet, Drnet, TR, get_iter, criterion, criterion_weighted, criterion_TR, weighted_TR, curve, independence_weights

# Assuming you have the necessary functions and classes defined as per your original code.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def objective(trial):
    # Define the hyperparameters to tune
    init_lr = trial.suggest_loguniform('init_lr', 1e-5, 1e-2)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-2)
    tr_init_lr = trial.suggest_loguniform('tr_init_lr', 1e-5, 1e-2)
    tr_wd = trial.suggest_loguniform('tr_wd', 1e-6, 1e-2)
    alpha = trial.suggest_uniform('alpha', 0.5, 1.5)
    beta = trial.suggest_uniform('beta', 0.5, 2.0)
    momentum = trial.suggest_uniform('momentum', 0.5, 0.99)

    num_epoch = args.n_epochs
    verbose = args.verbose

    # Load data
    data_matrix = pd.read_csv(os.path.join(
        args.data_dir, 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join(args.data_dir, 't_grid.csv')).values

    sample_size = trial.suggest_categorical(
        'sample_size', [1000])  # You can adjust sample sizes

    idx = list(range(data_matrix.shape[0]))
    random.shuffle(idx)

    train_idx = idx[:sample_size]
    test_idx = idx[sample_size:sample_size + 500]

    train_matrix = torch.tensor(
        data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(
        data_matrix[test_idx, :], dtype=torch.float32)
    t_grid = torch.tensor(t_grid_all[:, test_idx], dtype=torch.float32)

    train_loader = get_iter(
        train_matrix, batch_size=len(train_matrix), shuffle=True)
    test_loader = get_iter(
        test_matrix, batch_size=len(test_matrix), shuffle=False)

    # Define your model (e.g., Vcnet_TR)
    model_name = 'Vcnet_TR'
    cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model = model.to(device)
    model._initialize_weights()

    isTargetReg = True
    tr_knots = list(np.arange(0.05, 1, 0.05))
    tr_degree = 2
    TargetReg = TR(tr_degree, tr_knots)
    TargetReg._initialize_weights()

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=init_lr, momentum=momentum, weight_decay=wd)

    tr_optimizer = torch.optim.SGD(
        TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

    for epoch in range(num_epoch):
        for inputs, y in train_loader:
            t = inputs[:, 0]
            x = inputs[:, 1:]

            optimizer.zero_grad()
            out = model.forward(t, x)
            trg = TargetReg(t)
            loss = criterion(out, y, alpha=alpha) + \
                criterion_TR(out, trg, y, beta=beta)
            loss.backward()
            optimizer.step()

            tr_optimizer.zero_grad()
            out = model.forward(t, x)
            trg = TargetReg(t)
            tr_loss = criterion_TR(out, trg, y, beta=beta)
            tr_loss.backward()
            tr_optimizer.step()

        if epoch % verbose == 0:
            print(
                f'Epoch: {epoch}, Loss: {loss.item()}')

    # Evaluate the model
    if isTargetReg:
        t_grid_hat, mse = curve(
            model, test_matrix, t_grid, targetreg=TargetReg)
    else:
        t_grid_hat, mse = curve(model, test_matrix, t_grid)

    mse = float(mse)

    # Optuna tries to minimize the objective value
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data')

    parser.add_argument('--data_dir', type=str,
                        default='dataset/news', help='dir of data matrix')
    parser.add_argument('--save_dir', type=str,
                        default='logs/news/eval', help='dir to save result')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='num of epochs to train')
    parser.add_argument('--verbose', type=int, default=100,
                        help='print train info freq')

    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
