import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def plot_comp(
    features_og: np.ndarray | torch.Tensor,
    features_san: np.ndarray | torch.Tensor,
    sanitized_idxs: Sequence[int],
    save_name: str,
):
    """
    Will plot all features in a single plot. 
    Warning: Sort of hard-cody in the sense that a lot of assumptions are made
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
