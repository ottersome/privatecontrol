import os
from math import ceil, sqrt
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from paretto_plots import paretto_frontier

def plot_given(
    func_a: np.ndarray,
    func_b: np.ndarray,
    func_a_label: str,
    func_b_label: str,
    x_label: str,
    y_label: str,
    sub_plot_tit: str,
    title: str,
    fig_savedest: str
):
    assert (
        len(func_a.shape) == 2 and len(func_b.shape) == 2
    ), "We expect both functions to be 2d"
    assert (
        func_a.shape[1] == func_b.shape[1]
    ), f"Expecting both functions to have same amount of features, instead we get {func_a.shape} and {func_b.shape}"
    assert (
        func_a.shape[0] == func_b.shape[0]
    ), "Expecting both functions to have the same amount of time steps"

    num_features = func_a.shape[1]
    num_cols = ceil(sqrt(func_a.shape[1]))
    num_rows = ceil(num_features / num_cols)

    fig, axes  = plt.subplots(num_rows, num_cols, figsize=(4*num_rows,4*num_cols))
    axes = axes.flatten()

    for nf in range(num_features):
        axes[nf].plot(func_a[:, nf], c = "b", label = func_a_label)
        axes[nf].plot(func_b[:, nf], c = "orange", label = func_b_label)
        axes[nf].set_title(sub_plot_tit.format(nf))
        axes[nf].set_xlabel(x_label)
        axes[nf].set_ylabel(y_label)

    remaining = num_cols*num_rows - num_features

    # Remove the remaining axes
    for ax_idx in range(num_features, len(axes)):
        fig.delaxes(axes[ax_idx])

    plt.title(title)
    plt.savefig(fig_savedest)
    plt.close()


def plot_comp(
    features_og: np.ndarray | torch.Tensor,
    features_san: np.ndarray | torch.Tensor,
    sanitized_idxs: Sequence[int],
    save_name: str,
):
    """
    Will plot all features in a single plot. Including private and public. 
    Hardcoded-ly expects 16 features and is very inflexible about this.
    Warning: Sort of hard-coded in the sense that a lot of assumptions are made
    args:
        - features_og: Original features
        - features_san: Sanitized features
        - sanitized_idxs: Which features are sanitized, conversely when, not to plot a sanitized features 
            (because its not part of the ds)
        - fig_name: Name of the figure
    """
    basedir = os.path.dirname(save_name)
    os.makedirs(basedir, exist_ok=True)
    plt.figure(figsize=(32, 16))
    plt.tight_layout()
    san_counter = 0
    for i in range(features_og.shape[-1]):
        plt.subplot(4, 4, i + 1)
        plt.plot(
            features_og[:, i],
            label="Original",
        )
        if i in sanitized_idxs:
            plt.plot(features_san[:, san_counter], label="Sanitized", alpha=0.7)
            san_counter += 1
        plt.legend()
        plt.title(f"Feature {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
    plt.savefig(f"{save_name}.png")
    plt.close()

def plot_signal_reconstructions(original, altered_signal, save_name: str, ids=None) -> None:
    """
    Plots the original signal and the reconstructed signal
    Args:
        original: Original signal
        altered_signal: Reconstructed/Altered signal
    Returns:
        None
    """
    basedir = os.path.dirname(save_name)
    os.makedirs(basedir, exist_ok=True)
    num_signals = original.shape[1] if ids is None else len(ids)
    cols = int(np.ceil(np.sqrt(num_signals)))
    rows = int(np.ceil(num_signals / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    # Convert axes to array and flatten
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes[:num_signals]):
        idx = i if ids is None else ids[i]
        sns.lineplot(
            x=np.arange(original.shape[0]),
            y=original[:, idx],
            ax=ax,
            label="Truth",
            color="orange",
        )
        sns.lineplot(
            x=np.arange(altered_signal.shape[0]),
            y=altered_signal[:, idx],
            ax=ax,
            label="Reconstruction",
            color="blue",
        )
        ax.set_title(f"Reconstruction Vs Truth of $f_{{{idx}}}$")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

# Warning: Very general use so I will use dictionary to take a varying amount of uvps
def plot_uvps(
    uvp_coeffs: list[np.ndarray],
    utilities: list[np.ndarray],
    privacies: list[np.ndarray],
    labels: list[str],
    save_dest: Optional[str],
):
    """
    All arguments that are list are meant to match in index with other arguments
    Args:
        - uvp_coeff: list of uvp coefficients (indep var that determines utilities and privacies)
        - utilities: list of utilities
        - privacies: list of privacies
        - labels: list of labels
        - save_dest: path to save plot
    Returns:
        None
    """
    assert len(utilities) == len(privacies) == len(uvp_coeffs), \
        "We expect equal length across `utilities`, `privacies`, and `uvp_coeff` parameters"
    amount_dots = len(utilities)
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("paper", font_scale=1.5)
    
    # Create figure with appropriate size for paper
    _, ax = plt.subplots(figsize=(8, 6))  # Standard single-column figure size

    potential_colors_dot_colors = sns.color_palette("husl", amount_dots)
    
    texts = []
    for i in range(amount_dots):
        # Create the main scatter plot
        ith_privacies = privacies[i].squeeze()
        ith_utilities = utilities[i].squeeze()
        uvp_points = uvp_coeffs[i].squeeze()


        assert len(ith_utilities.shape) == len(ith_privacies.shape) == 1 and len(uvp_points.shape) == 1,\
            f"Can only take 1d {i}th_utilities, 1d {i}th_privacies 1d uvp_points."\
            "But got {ith_utilities.shape} and {ith_privacies.shape} and {uvp_points.shape}"
        assert ith_utilities.shape[0] == ith_privacies.shape[0] == uvp_points.shape[0], \
            f"Mismatch in {i}th_utilities vs {i}th_privacies vs {i}th_uvp_points function shape: {ith_utilities.shape} and {ith_privacies.shape} and {uvp_points.shape}"
        num_data_points = ith_utilities.shape[0]

        left_hull_x, left_hull_y = paretto_frontier(ith_privacies, ith_utilities)

        ax.scatter(ith_privacies, ith_utilities, 
                            color=potential_colors_dot_colors[i],
                            s=80,  # Marker size
                            alpha=0.7,  # Slight transparency
                            label=labels[i])


        for j in range(num_data_points):
            uvp = uvp_coeffs[i][j]
            texts.append(
                ax.annotate(
                    f"UVP: {uvp:.3f}",
                    (ith_privacies[i], ith_utilities[i]),
                    fontsize=8,
                )
            )

        # Plot Pareto frontier
        ax.plot(left_hull_x, left_hull_y, 
                color=potential_colors_dot_colors[i],  # Professional green color
                linestyle='--',
                alpha=0.8,
                linewidth=2,
                label='Pareto frontier')
    
    # Customize the plot
    ax.set_title("Privacy-Utility Trade-off Analysis", 
                 pad=20, 
                 fontsize=14, 
                 fontweight='bold')
    ax.set_xlabel("Privacy Score (MSE)", labelpad=10)
    ax.set_ylabel("Utility Score (Negative MSE)", labelpad=10)
    
    # Add grid with proper styling
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(frameon=True, 
             facecolor='white', 
             edgecolor='none',
             loc='best')
    plt.savefig(save_dest)

