import os
from math import ceil, sqrt
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

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
    plt.savefig(save_name)
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
    uvp_coeffs: list[list[str]], 
    utilities: list[list[float]],
    privacies: list[list[float]],
    labels: list[str],
    save_dest: Optional[str],
):
    """
    All arguments that are list are meant to match in index with other arguments
    Args:
        - uvp_coeff: list of uvp coefficients already formatted for printing into the plot
        - utilities: list of utilities
        - privacies: list of privacies
        - labels: list of labels
        - save_dest: path to save plot
    Returns:
        None
    """
    assert len(utilities) == len(privacies) == len(uvp_coeffs), \
        "We expect equal length across `utilities`, `privacies`, and `uvp_coeff` parameters"\
        f"Instead we got utilties: {len(utilities)}, privacies: {len(privacies)}, uvp: {len(uvp_coeffs)} "
    amount_curves = len(utilities)
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("paper", font_scale=1.5)
    potential_colors_dot_colors = sns.color_palette("husl", amount_curves)
    
    # Create figure with appropriate size for paper
    _, ax = plt.subplots(figsize=(8, 6))  # Standard single-column figure size

    texts = []
    for i in range(amount_curves):
        # Create the main scatter plot
        ith_privacies = privacies[i]
        ith_utilities = utilities[i]
        uvp_points = uvp_coeffs[i]

        assert len(ith_utilities) == len(ith_privacies) == len(uvp_points), \
            f"Mismatch in {i}th_utilities vs {i}th_privacies vs {i}th_uvp_points function shape: {len(ith_utilities)} and {len(ith_privacies)} and {len(uvp_points)}"
        num_data_points = len(ith_utilities)

        left_hull_x, left_hull_y = paretto_frontier(np.array(ith_privacies), np.array(ith_utilities))

        ax.scatter(ith_privacies, ith_utilities, 
                            color=potential_colors_dot_colors[i],
                            s=80,  # Marker size
                            alpha=0.7,  # Slight transparency
                            label=labels[i])


        for j in range(num_data_points):
            uvp = uvp_coeffs[i][j]
            texts.append(
                ax.annotate(
                    uvp,
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


def paretto_frontier(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    assert len(x.shape) == 1, "x must be a 1d array"
    assert len(y.shape) == 1, "y must be a 1d array"
    assert x.shape == y.shape, "x and y must be the same shape"
    assert x.shape[0] >= 1, "x must have more than one element"

    # Sort the y direction
    xy = np.vstack((x, y)).transpose()
    sorted_idxs = (-xy[:, 1]).argsort() # Sorting by privacy (vertical axis)
    xy = xy[sorted_idxs]

    y_down = np.array([0, -1])
    down_angle_rad = np.arctan2(y_down[1], y_down[0])

    not_fixed = False # For sake of entering the loop
    cp = np.copy(xy)
    while not not_fixed:
        not_fixed = True; # Sake of initialization
        ccp = np.copy(cp)
        del_idxs = []
        pidx = 0
        while pidx < (ccp.shape[0]-2):
            diff12 = ccp[pidx+1] - ccp[pidx]
            point12_angle = np.arctan2(diff12[1], diff12[0]) - down_angle_rad
            diff23 = ccp[pidx+2] - ccp[pidx+1]
            point23_angle = np.arctan2(diff23[1], diff23[0]) - down_angle_rad
            if point23_angle < point12_angle:
                not_fixed = False
                del_idxs.append(pidx+1)
                pidx += 2
            else:
                pidx += 1

        # Round of deletions
        cp = np.delete(cp, del_idxs, axis=0)

    # Remove  Excess points
    x_min_idx = np.argmin(cp[:, 0])
    y_min_idx = np.argmin(cp[:, 1])
    x_min = cp[x_min_idx, 0]
    y_min = cp[y_min_idx, 1]

    excess_up = np.where(cp[:, 1] > cp[x_min_idx, 1])[0]
    excess_right = np.where(cp[:, 0] > cp[y_min_idx, 0])[0]

    # Remove excess points
    cp = np.delete(cp, excess_up, axis=0)
    cp = np.delete(cp, excess_right, axis=0)

    return cp[:, 0], cp[:,1]
