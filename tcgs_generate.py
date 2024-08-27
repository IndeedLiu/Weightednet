import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate news data')
    parser.add_argument('--data_path', type=str,
                        default='tcga.csv', help='data path')
    parser.add_argument('--save_dir', type=str,
                        default='dataset/tcga', help='dir to save generated data')

    args = parser.parse_args()
    save_path = args.save_dir

    # load data from CSV
    data = pd.read_csv(args.data_path)

    # Remove first column (original treatment) and third column (original outcome)
    data = data.drop(columns=[data.columns[0], data.columns[2]])

    # Use second column as treatment t, and the remaining columns as covariates x
    t = data.iloc[:, 0].values
    x = data.iloc[:, 1:].values  # Remaining columns are covariates

    num_data = x.shape[0]
    num_feature = x.shape[1]

    # Define random vectors (or other appropriate transformations)
    np.random.seed(5)
    v1 = np.random.randn(num_feature)
    v1 = v1 / np.sqrt(np.sum(v1**2))
    v2 = np.random.randn(num_feature)
    v2 = v2 / np.sqrt(np.sum(v2**2))
    v3 = np.random.randn(num_feature)
    v3 = v3 / np.sqrt(np.sum(v3**2))

    # Function to generate y based on t and x, with y bounded between 0 and 1
    def generate_y(t, x):
        # You can customize this function further, here is an example:
        interaction = np.sum(v1 * x) * t + np.sin(np.sum(v2 * x) * t)
        # Sigmoid function to bound y between 0 and 1
        y = 1 / (1 + np.exp(-interaction))
        return y

    def data_matrix():
        data_matrix = torch.zeros(num_data, num_feature + 2)
        for i in range(num_data):
            covariates = x[i, :]
            treatment = torch.tensor([t[i]])
            outcome = torch.tensor([generate_y(t[i], covariates)])

            data_matrix[i, 0] = treatment
            data_matrix[i, num_feature + 1] = outcome
            data_matrix[i, 1:num_feature + 1] = torch.tensor(covariates)

        return data_matrix

    def t_grid():
        t_grid = torch.zeros(2, num_data)
        t_grid[0, :] = torch.tensor(t)

        for i in tqdm(range(num_data)):
            psi = 0
            treatment = t[i]
            for j in range(num_data):
                covariates = x[j, :]
                psi += generate_y(treatment, covariates)
            psi /= num_data
            t_grid[1, i] = psi

        return t_grid

    # Generate the data matrices
    dm = data_matrix()
    tg = t_grid()

    # Save data_matrix to CSV
    data_matrix_df = pd.DataFrame(dm.numpy())
    data_matrix_df.columns = [
        't'] + [f'x_{i}' for i in range(1, num_feature + 1)] + ['y']
    data_matrix_df.to_csv(os.path.join(
        save_path, 'data_matrix.csv'), index=False)

    # Save t_grid to CSV
    t_grid_df = pd.DataFrame([tg[0].numpy(), tg[1].numpy()])
    t_grid_df.columns = range(1, t_grid_df.shape[1] + 1)
    t_grid_df = pd.concat([pd.DataFrame(
        [t_grid_df.columns], columns=t_grid_df.columns), t_grid_df], ignore_index=True)
    t_grid_df.to_csv(os.path.join(save_path, 't_grid.csv'),
                     header=False, index=False)

    print(f"Data matrix saved to {os.path.join(save_path, 'data_matrix.csv')}")
    print(f"t_grid saved to {os.path.join(save_path, 't_grid.csv')}")
