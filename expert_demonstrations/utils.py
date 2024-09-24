import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pca import pca

def MAPEloss(output, target):
    """
    Calculates Mean Absolute Percentage Error (MAPE) loss.

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: MAPE loss value.
    """

    try:
        output, target = output.detach().numpy(), target.detach().numpy()
    except:
        pass
    output, target = np.array(output), np.array(target)

    # Avoid division by zero (consider epsilon for numerical stability)
    epsilon = 1e-8  # Adjust epsilon as needed
    absolute_diff = np.abs(target - output)
    # Handle potential zero true values with epsilon for MAPE calculation
    division_term = np.where(np.abs(target) > epsilon, np.abs(target), epsilon)
    percentage_error = np.divide(absolute_diff, division_term)  # Convert to percentage

    return np.mean(percentage_error)


def MAEloss(output, target):
    """
    Calculates Mean Absolute Error (MAE) loss.

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: MAE loss value.
    """
    try:
        output, target = output.detach().numpy(), target.detach().numpy()
    except:
        pass
    output, target = np.array(output), np.array(target)
    absolute_diff = np.abs(target - output)
    return np.mean(absolute_diff)


# Credits @https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
# Input X is numpy array for shape (samples, features)
def biplot(X, save_fig_path=False):
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    model = pca()
    out = model.fit_transform(X_df)
    model.plot()
    if save_fig_path:
        plt.savefig(f"{save_fig_path}_pca.pdf")
    try:
        ax = model.biplot(n_feat=23, legend=True, verbose=1)
        if save_fig_path:
            plt.savefig(f"{save_fig_path}_biplot.pdf")
        else:
            plt.show()
    except:
        print(
            f"Error in biplot. Most likely due to too many features. Skipping biplot."
        )
    plt.clf()


class nonLinearAE(torch.nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()

        scaling_factor = 2  # always use scaling of 2
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim, latent_dim * scaling_factor),
            torch.nn.Tanh(),
            torch.nn.Linear(latent_dim * scaling_factor, latent_dim * scaling_factor),
            torch.nn.Tanh(),
            torch.nn.Linear(latent_dim * scaling_factor, latent_dim),
        ).double()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * scaling_factor),
            torch.nn.Tanh(),
            torch.nn.Linear(latent_dim * scaling_factor, latent_dim * scaling_factor),
            torch.nn.Tanh(),
            torch.nn.Linear(latent_dim * scaling_factor, dim),
        ).double()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# loss function for AE that encourages latent space to be in [-1, 1]
def loss_function_AE_standardized(x, y, latent=None):
    loss = 0.0
    reconstruction_loss = torch.nn.MSELoss()(x, y)
    loss += reconstruction_loss
    if latent is not None:
        # loss function that encourages latent space to be in [-1, 1]
        # this loss function can possible improved by settings zero gradient in [-0.9, 0.9] and have a smoother function outside. For our experiments, though, the below function was sufficient.
        latent = torch.where((latent < -0.8) | (latent > 0.8), latent, 0.0)
        latent_loss = torch.mean(torch.exp((latent / 1.2) ** 10) - 1.0)
        loss += latent_loss
    return loss
