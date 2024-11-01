import torch
import numpy as np
import json
from scipy.stats import gaussian_kde
from data.data import get_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def curve(model, test_matrix, t_grid, targetreg=None, model_name=None, weight=None):
    model = model.to(device)
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(
        test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                t = torch.full((x.shape[0],), t_grid[0, _], dtype=t.dtype)
                break

            out = model.forward(t, x)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out

        # Calculate MSE without averaging, skipping NaN values
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2)
        # Skip NaN values, output as column vector
        mse = mse[~torch.isnan(mse)]
        return t_grid_hat, mse.unsqueeze(1)  # Return mse as a column vector

    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break

            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()

            # Check if model_name is 'weightednet_TR' and adjust the out calculation
            if model_name == 'weightednet_TR':
                # Adjusted for weightednet_TR
                out = out[1].data.squeeze() + tr_out*weight/torch.tensor(gaussian_kde(t)(
                    t), dtype=torch.float32)
            else:
                out = out[1].data.squeeze() + tr_out / (g + 1e-6)

            out = out.mean()
            t_grid_hat[1, _] = out

        # Calculate MSE without averaging, skipping NaN values
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2)
        # Skip NaN values, output as column vector
        mse = mse[~torch.isnan(mse)]
        return t_grid_hat, mse.unsqueeze(1)  # Return mse as a column vector
