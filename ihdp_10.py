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

from models.dynamic_net import Vcnet, TR, Drnet
from utils.eval import curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()


def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()


def criterion_weighted(out, y, weight):
    return ((out[1].squeeze() - y.squeeze()) ** 2 * weight).mean()


def weighted_TR(out, trg, y, weight, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze() * weight - out[1].squeeze()) ** 2).mean()


def independence_weights(A, X, lambda_=0, decorrelate_moments=False, preserve_means=False, dimension_adj=True):
    n = A.shape[0]
    p = X.shape[1]
    gamma = 1
    # dif
    A = np.asarray(A).reshape(-1, 1)
    Adist = squareform(pdist(A, 'euclidean'))
    Xdist = squareform(pdist(X, 'euclidean'))

    # terms for energy-dist(Wtd A, A)
    Q_energy_A = -Adist / n ** 2
    aa_energy_A = np.sum(Adist, axis=1) / n ** 2

    # terms for energy-dist(Wtd X, X)
    Q_energy_X = -Xdist / n ** 2
    aa_energy_X = np.sum(Xdist, axis=1) / n ** 2

    mean_Adist = np.mean(Adist)
    mean_Xdist = np.mean(Xdist)

    Xmeans = np.mean(Xdist, axis=1)
    Xgrand_mean = np.mean(Xmeans)
    XA = Xdist + Xgrand_mean - np.add.outer(Xmeans, Xmeans)

    Ameans = np.mean(Adist, axis=1)
    Agrand_mean = np.mean(Ameans)
    AA = Adist + Agrand_mean - np.add.outer(Ameans, Ameans)

    # quadratic term for weighted total distance covariance
    P = XA * AA / n ** 2

    if preserve_means:
        if decorrelate_moments:
            Constr_mat = (A - np.mean(A)) * (X - np.mean(X, axis=0))
            Amat = sparse.vstack([np.eye(n), np.ones((1, n)), X.T,
                                  A.reshape(1, -1), Constr_mat.T])
            lvec = np.concatenate([np.zeros(n), [n], np.mean(
                X, axis=0), [np.mean(A)], np.zeros(X.shape[1])])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.mean(X, axis=0), [np.mean(A)], np.zeros(X.shape[1])])
        else:
            Amat = sparse.vstack(
                [np.eye(n), np.ones((1, n)), X.T, A.reshape(1, -1)])
            lvec = np.concatenate(
                [np.zeros(n), [n], np.mean(X, axis=0), [np.mean(A)]])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.mean(X, axis=0), [np.mean(A)]])
    else:
        if decorrelate_moments:
            Constr_mat = (A - np.mean(A)) * (X - np.mean(X, axis=0))
            Amat = sparse.vstack([np.eye(n), np.ones((1, n)), Constr_mat.T])
            lvec = np.concatenate([np.zeros(n), [n], np.zeros(X.shape[1])])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.zeros(X.shape[1])])
        else:
            Amat = sparse.vstack([np.eye(n), np.ones((1, n))])
            lvec = np.concatenate([np.zeros(n), [n]])
            uvec = np.concatenate([np.inf * np.ones(n), [n]])

    if dimension_adj:
        Q_energy_A_adj = 1 / np.sqrt(p)
        Q_energy_X_adj = 1
        sum_adj = Q_energy_A_adj + Q_energy_X_adj
        Q_energy_A_adj /= sum_adj
        Q_energy_X_adj /= sum_adj
    else:
        Q_energy_A_adj = Q_energy_X_adj = 1 / 2

    for na in range(1, 50):
        p = sparse.csr_matrix(2 * (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X *
                                                Q_energy_X_adj) + lambda_ * np.diag(np.ones(n)) / n ** 2))
        A = Amat

        l = lvec
        u = uvec
        q = 2 * gamma * (aa_energy_A * Q_energy_A_adj +
                         aa_energy_X * Q_energy_X_adj)
        m = osqp.OSQP()
        m.setup(P=p, q=q, A=A, l=l, u=u, max_iter=int(2e5),
                eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        results = m.solve()
        if not np.any(results.x > 1e5):
            break

    weights = results.x

    weights[weights < 0] = 0

    QM_unpen = P + gamma * (Q_energy_A * Q_energy_A_adj +
                            Q_energy_X * Q_energy_X_adj)

    quadpart_unpen = weights.T @ QM_unpen @ weights
    quadpart_unweighted = np.sum(QM_unpen)

    quadpart = quadpart_unpen + np.sum(weights ** 2) * lambda_ / n ** 2

    qvec = 2 * gamma * (aa_energy_A * Q_energy_A_adj +
                        aa_energy_X * Q_energy_X_adj)
    linpart = weights @ qvec
    linpart_unweighted = np.sum(qvec)

    objective_history = quadpart + linpart + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)

    D_w = quadpart_unpen + linpart + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)
    D_unweighted = quadpart_unweighted + linpart_unweighted + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)

    qvec_full = 2 * (aa_energy_A * Q_energy_A_adj +
                     aa_energy_X * Q_energy_X_adj)

    quadpart_energy_A = weights.T @ Q_energy_A @ weights * Q_energy_A_adj
    quadpart_energy_X = weights.T @ Q_energy_X @ weights * Q_energy_X_adj
    quadpart_energy = quadpart_energy_A * \
        Q_energy_A_adj + quadpart_energy_X * Q_energy_X_adj

    distcov_history = weights.T @ P @ weights
    unweighted_dist_cov = np.sum(P)

    linpart_energy = weights @ qvec_full
    linpart_energy_A = 2 * weights @ aa_energy_A * Q_energy_A_adj
    linpart_energy_X = 2 * weights @ aa_energy_X * Q_energy_X_adj

    energy_history = quadpart_energy + linpart_energy - \
        mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj
    energy_A = quadpart_energy_A + linpart_energy_A - mean_Adist * Q_energy_A_adj
    energy_X = quadpart_energy_X + linpart_energy_X - mean_Xdist * Q_energy_X_adj

    ess = (np.sum(weights)) ** 2 / np.sum(weights ** 2)

    ret_obj = {
        'weights': weights,
        'A': A,
        'opt': results,
        'objective': objective_history,
        'D_unweighted': D_unweighted,
        'D_w': D_w,
        'distcov_unweighted': unweighted_dist_cov,
        'distcov_weighted': distcov_history,
        'energy_A': energy_A,
        'energy_X': energy_X,
        'ess': ess
    }

    return ret_obj


def compute_irmse(mse_iter_list):
    """
    计算并累积每一行的 MSE 列向量，然后生成 IRMSE。
    每次迭代的 MSE 列向量会逐次累加，最后取平均、开根号，再计算总平均得到 IRMSE。

    mse_iter_list: 存放 MSE 列向量的列表，每个向量是一个张量，表示每次迭代的MSE。

    返回: IRMSE
    """
    # 确保 mse_iter_list 非空
    if len(mse_iter_list) == 0:
        raise ValueError("mse_iter_list is empty")

    # 初始化累计MSE为0，假设每个张量形状相同，以第一个张量的形状作为基准
    cumulative_mse = torch.zeros_like(mse_iter_list[0])

    # 遍历所有MSE列向量，逐个累加
    for mse in mse_iter_list:
        cumulative_mse += mse

    # 对所有迭代后的MSE列向量取平均
    mean_mse = cumulative_mse / len(mse_iter_list)

    # 对每行MSE取平方根，生成RMSE列向量
    rmse_vector = torch.sqrt(mean_mse)

    # 取RMSE列向量的平均值，得到IRMSE
    irmse = rmse_vector.mean().item()

    return irmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data')

    parser.add_argument('--data_dir', type=str,
                        default='dataset/ihdp', help='dir of data matrix')
    parser.add_argument('--save_dir', type=str,
                        default='logs/ihdp/eval', help='dir to save result')
    parser.add_argument('--n_epochs', type=int, default=800,
                        help='num of epochs to train')
    parser.add_argument('--verbose', type=int, default=100,
                        help='print train info freq')

    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    init_lr = 0.0006
    wd = 7.e-05
    momentum = 0.9
    alpha = 0.8
    beta = 2
    gamma = 8
    lambda_ = 0.4

    num_epoch = args.n_epochs
    verbose = args.verbose

    data_matrix = pd.read_csv(os.path.join(
        args.data_dir, 'data_matrix.csv')).values
    t_grid_all = pd.read_csv(os.path.join(args.data_dir, 't_grid.csv')).values

    sample_sizes = range(200, 501, 300)
    num_iterations = 100
    mse_vcnet = []
    mse_vcnet_tr = []
    mse_drnet_tr = []
    mse_weightednet = []
    mse_weightednet_tr = []

    for sample_size in sample_sizes:
        mse_vcnet_iter = []
        mse_vcnet_tr_iter = []
        mse_drnet_tr_iter = []
        mse_weightednet_iter = []
        mse_weightednet_tr_iter = []

        for _ in range(num_iterations):
            test_matrix = torch.tensor(
                data_matrix[500:, :], dtype=torch.float32)

            # 在前6000行的数据中生成随机数
            idx = list(range(500))
            random.shuffle(idx)

            # 选择train_idx
            train_idx = idx[:sample_size]

            # 设置train_matrix
            train_matrix = torch.tensor(
                data_matrix[train_idx, :], dtype=torch.float32)

            # 设置对应的t_grid
            t_grid = torch.tensor(t_grid_all[:, 500:], dtype=torch.float32)

            # 初始化两个loader
            train_loader = get_iter(
                train_matrix, batch_size=len(train_matrix), shuffle=True)
            test_loader = get_iter(
                test_matrix, batch_size=len(test_matrix), shuffle=False)

            models = ['Vcnet', 'Vcnet_TR', 'weightednet',
                      'weightednet_TR', 'Drnet_tr']
            for model_name in models:
                if model_name == 'Vcnet' or model_name == 'Vcnet_TR':
                    cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                    num_grid = 10
                    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                    degree = 2
                    knots = [0.33, 0.66]
                    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                    model = model.to(device)
                    model._initialize_weights()

                elif model_name == 'Drnet_tr':
                    cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                    num_grid = 10
                    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                    isenhance = 1
                    model = Drnet(cfg_density, num_grid,
                                  cfg, isenhance=isenhance)
                    model = model.to(device)
                    model._initialize_weights()

                elif model_name == 'weightednet' or model_name == 'weightednet_TR':
                    cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                    num_grid = 10
                    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                    degree = 2
                    knots = [0.33, 0.66]
                    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                    model = model.to(device)
                    model._initialize_weights()

                    weights_info = independence_weights(
                        train_matrix[:, 0].numpy(), train_matrix[:, 1:-16].numpy())
                    weights = torch.tensor(
                        weights_info['weights'], dtype=torch.float32)

                isTargetReg = model_name in [
                    'Vcnet_TR', 'Drnet_tr', 'weightednet_TR']

                if isTargetReg:
                    tr_knots = list(np.arange(0.05, 1, 0.05))
                    tr_degree = 2
                    TargetReg = TR(tr_degree, tr_knots)
                    TargetReg._initialize_weights()

                optimizer = torch.optim.SGD(
                    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)

                if isTargetReg:
                    tr_init_lr = 0.001
                    tr_optimizer = torch.optim.SGD(
                        TargetReg.parameters(), lr=tr_init_lr, weight_decay=1e-3)

                for epoch in range(num_epoch):
                    for inputs, y in train_loader:
                        t = inputs[:, 0]
                        x = inputs[:, 1:]

                        if isTargetReg:
                            optimizer.zero_grad()
                            out = model.forward(t, x)
                            trg = TargetReg(t)
                            if model_name == 'weightednet_TR':
                                loss = criterion_weighted(
                                    out, y, weights) + weighted_TR(out, trg, y, weights, beta=beta)
                            else:
                                loss = criterion(
                                    out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                            loss.backward()
                            optimizer.step()

                            tr_optimizer.zero_grad()
                            out = model.forward(t, x)
                            trg = TargetReg(t)
                            if model_name == 'weightednet_TR':
                                tr_loss = weighted_TR(
                                    out, trg, y, weights, beta=beta)
                            else:
                                tr_loss = criterion_TR(out, trg, y, beta=beta)
                            tr_loss.backward()
                            tr_optimizer.step()
                        elif model_name == 'weightednet':
                            optimizer.zero_grad()
                            out = model.forward(t, x)
                            loss = criterion_weighted(out, y, weights)
                            loss.backward()
                            optimizer.step()
                        else:
                            optimizer.zero_grad()
                            out = model.forward(t, x)
                            loss = criterion(out, y, alpha=alpha)
                            loss.backward()
                            optimizer.step()

                    if epoch % verbose == 0:
                        print(
                            f'Model: {model_name}, Epoch: {epoch}, Loss: {loss.item()}')

                if isTargetReg:
                    t_grid_hat, mse = curve(
                        model, test_matrix, t_grid, targetreg=TargetReg)
                else:
                    t_grid_hat, mse = curve(model, test_matrix, t_grid)

                # Keep MSE as a column vector, no conversion to float
                if model_name == 'Vcnet':
                    mse_vcnet_iter.append(mse)
                elif model_name == 'Vcnet_TR':
                    mse_vcnet_tr_iter.append(mse)
                elif model_name == 'Drnet_tr':
                    mse_drnet_tr_iter.append(mse)
                elif model_name == 'weightednet':
                    mse_weightednet_iter.append(mse)
                elif model_name == 'weightednet_TR':
                    mse_weightednet_tr_iter.append(mse)

        # Compute IRMSE for each model using the updated mse_iter lists
        irmse_vcnet = compute_irmse(mse_vcnet_iter)
        irmse_vcnet_tr = compute_irmse(mse_vcnet_tr_iter)
        irmse_drnet_tr = compute_irmse(mse_drnet_tr_iter)
        irmse_weightednet = compute_irmse(mse_weightednet_iter)
        irmse_weightednet_tr = compute_irmse(mse_weightednet_tr_iter)

        # Append the final IRMSEs for each sample size
        mse_vcnet.append(irmse_vcnet)
        mse_vcnet_tr.append(irmse_vcnet_tr)
        mse_drnet_tr.append(irmse_drnet_tr)
        mse_weightednet.append(irmse_weightednet)
        mse_weightednet_tr.append(irmse_weightednet_tr)

    # 打印 MSE 结果
    print("VCNet MSE:", mse_vcnet)
    print("VCNet_TR MSE:", mse_vcnet_tr)
    print("Drnet_tr MSE:", mse_drnet_tr)
    print("WeightedNet MSE:", mse_weightednet)
    print("WeightedNet_TR MSE:", mse_weightednet_tr)
