import torch
import numpy as np
import json
from data.data import get_iter

def curve(model, test_matrix, t_grid, targetreg=None, model_name=None):
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

        # Calculate MSE, skipping NaN values
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2)
        mse = mse[~torch.isnan(mse)].mean().data  # Skip NaN values in MSE
        return t_grid_hat, mse

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
                out = out[1].data.squeeze() + tr_out  # Adjusted for weightednet_TR
            else:
                out = out[1].data.squeeze() + tr_out / (g + 1e-6)

            out = out.mean()
            t_grid_hat[1, _] = out

        # Calculate MSE, skipping NaN values
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2)
        mse = mse[~torch.isnan(mse)].mean().data  # Skip NaN values in MSE
        return t_grid_hat, mse

