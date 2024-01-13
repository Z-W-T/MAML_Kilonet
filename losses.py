import torch
import matplotlib.pyplot as plt
import os


mse_fn = torch.nn.MSELoss()
per_element_mse_fn = torch.nn.MSELoss(reduction="none")


def batch_mse_fn(x1, x2):
    """
        Computes MSE between two batches of signals while preserving the batch
        dimension (per batch element MSE).
        Args:
           x1 (torch.Tensor): Shape (batch_size, *).
           x2 (torch.Tensor): Shape (batch_size, *).
        Returns:
           MSE tensor of shape (batch_size,).
    """
    per_element_mse = per_element_mse_fn(x1, x2)

    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def mse2psnr(mse):
    """
        Computes PSNR from MSE, assuming the MSE was calculated between signals
        lying in [0, 1].
        Args:
        mse (torch.Tensor or float):
    """

    return -10.0 * torch.log10(mse)


def psnr_fn(x1, x2):
    """
        Computes PSNR between signals x1 and x2. Note that the values of x1 and
        x2 are assumed to lie in [0, 1].
        Args:
            x1 (torch.Tensor): Shape (*).
            x2 (torch.Tensor): Shape (*).
    """
    return mse2psnr(mse_fn(x1, x2))


def plot_curve(x_vals, y_vals,
              x_label, y_label,
              x2_vals=None, y2_vals=None,
              x3_vals=None, y3_vals=None,
              x4_vals=None, y4_vals=None,
              legend=None, path=None, filename='training_losses.png'):
    width = 1.5
    plt.cla()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals, linewidth=width, linestyle='dashed', color='blue', alpha=0.7)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linewidth=width, linestyle='dashed', color='red', alpha=0.7)

    if x3_vals and y3_vals:
        plt.plot(x3_vals, y3_vals, linewidth=width, color='cornflowerblue',  alpha=0.7)

    if x4_vals and y4_vals:
        plt.plot(x4_vals, y4_vals, linewidth=width, color='salmon', alpha=0.7)

    if legend:
        plt.legend(legend)

    plt.grid()

    # save plt
    if path is None:
        plt.savefig(filename,dpi=800)
    else:
        plt.savefig(os.path.join(path, filename),dpi=800)