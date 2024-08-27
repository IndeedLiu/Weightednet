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

from models.dynamic_net import Vcnet, TR
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


def criterion_weighted(out, y, weight, alpha=0.5, epsilon=1e-5):
    return ((out[1].squeeze() - y.squeeze()) ** 2 * weight).mean() - alpha * torch.log(out[0] + epsilon).mean()


def weighted_TR(out, trg, y, weight, beta=1., epsilon=1e-5):
    return beta * ((y.squeeze() - trg.squeeze() * weight - out[1].squeeze()) ** 2).mean()


def independence_weights(A, X, lambda_=0.1, gamma=1.0):
    n = A.shape[0]
    A = np.asarray(A).reshape(-1, 1)
    Adist = squareform(pdist(A, 'euclidean'))
    Xdist = squareform(pdist(X, 'euclidean'))

    Q_energy_A = -Adist / n ** 2
    Q_energy_X = -Xdist / n ** 2
    aa_energy_A = np.sum(Adist, axis=1) / n ** 2
    aa_energy_X = np.sum(Xdist, axis=1) / n ** 2

    P = Q_energy_X * Q_energy_A / n ** 2

    Amat = sparse.vstack([np.eye(n), np.ones((1, n))])
    lvec = np.concatenate([np.zeros(n), [n]])
    uvec = np.concatenate([np.inf * np.ones(n), [n]])

    for _ in range(1, 50):
        p = sparse.csr_matrix(
            2 * (P + gamma * (Q_energy_A + Q_energy_X) + lambda_ * np.diag(np.ones(n)) / n ** 2))
        q = 2 * gamma * (aa_energy_A + aa_energy_X)
        m = osqp.OSQP()
        m.setup(P=p, q=q, A=Amat, l=lvec, u=uvec, max_iter=int(
            2e5), eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        results = m.solve()
        if not np.any(results.x > 1e5):
            break

    weights = np.maximum(results.x, 0)
    return weights


def objective(trial):
    # Define the hyperparameter search space
    init_lr = trial.suggest_float('init_lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    beta = trial.suggest_float('beta', 0.1, 10.0)
    tr_init_lr = trial.suggest_float('tr_init_lr', 1e-5, 1e-1, log=True)
    tr_wd = trial.suggest_float('tr_wd', 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.99)
    lambda_ = trial.suggest_float('lambda_', 0.0, 1.0)
    gamma = trial.suggest_float('gamma', 0.1, 10.0)

    cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    num_epoch = 400  # Set for faster experimentation
    verbose = 100

    # Load your dataset
    data_matrix = pd.read_csv(os.path.join(
        'dataset/news', 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join('dataset/news', 't_grid.csv')).values

    idx = list(range(data_matrix.shape[0]))
    random.shuffle(idx)
    sample_size = 2500  # Sample size tuning
    train_idx = idx[:sample_size]
    test_idx = idx[sample_size:sample_size + 500]

    train_matrix = torch.tensor(data_matrix[train_idx, :], dtype=torch.float32)
    test_matrix = torch.tensor(data_matrix[test_idx, :], dtype=torch.float32)
    t_grid = torch.tensor(t_grid_all[:, test_idx], dtype=torch.float32)

    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)

    weights_info = independence_weights(train_matrix[:, 0].numpy(
    ), train_matrix[:, 1:-1].numpy(), lambda_=lambda_, gamma=gamma)
    weights = torch.tensor(weights_info, dtype=torch.float32)

    train_loader = get_iter(
        train_matrix, batch_size=len(train_matrix), shuffle=True)

    # Initialize TargetReg if necessary
    isTargetReg = True
    tr_knots = list(np.arange(0.05, 1, 0.05))
    tr_degree = 2
    TargetReg = TR(tr_degree, tr_knots)
    TargetReg._initialize_weights()

    tr_optimizer = torch.optim.SGD(
        TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

    for epoch in range(num_epoch):
        for inputs, y in train_loader:
            t = inputs[:, 0]
            x = inputs[:, 1:]

            optimizer.zero_grad()
            out = model.forward(t, x)
            trg = TargetReg(t)
            loss = criterion_weighted(
                out, y, weights, alpha=alpha) + weighted_TR(out, trg, y, weights, beta=beta)
            loss.backward()
            optimizer.step()

            tr_optimizer.zero_grad()
            tr_loss = weighted_TR(out, trg, y, weights, beta=beta)
            tr_loss.backward()
            tr_optimizer.step()

        if epoch % verbose == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    t_grid_hat, mse = curve(model, train_matrix, t_grid)
    return float(mse)


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
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
