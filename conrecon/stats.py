import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, spearmanr

def compute_correlations(original, sanitized):
    """Compute Pearson and Spearman correlation coefficients."""
    pearson_corrs = []
    spearman_corrs = []
    for i in range(original.shape[-1]):
        pearson_corr, _ = pearsonr(original[:, i].flatten(), sanitized[:, i].flatten())
        spearman_corr, _ = spearmanr(
            original[:, i].flatten(), sanitized[:, i].flatten()
        )
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
    return np.array(pearson_corrs), np.array(spearman_corrs)


def singleCol_compute_correlations(og_or_sanitized: np.ndarray, sensitive: np.ndarray):
    """
    For now we assume sensitive is a single column
    """
    assert (
        len(sensitive.shape) == 1
    ), f"Your sensitive tensor is of shape {sensitive.shape}, it should be a 1-d vector."

    pearson_corrs = []
    spearman_corrs = []
    for i in range(og_or_sanitized.shape[-1]):
        pearson_corr, _ = pearsonr(og_or_sanitized[:, i].flatten(), sensitive)
        spearman_corr, _ = spearmanr(og_or_sanitized[:, i].flatten(), sensitive)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
    return np.array(pearson_corrs), np.array(spearman_corrs)

def plot_feats_synch(
    df: pd.DataFrame, title: str, save_path: str = "./figures/defacto_data_synched/"
):
    """
    Plot Features at different time scales
    """
    num_features = df.shape[1] 
    os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), dpi=300)
    plt.tight_layout()

    possible_markers = ["o", "x", "^", "s", "D", ">", "<", "p", "*", "h", "H", "+", "d"]
    for f in range(num_features):
        # Once we have this we will create a smaller data frame consisting only of index and the feature
        spec_df = df.iloc[:, f]
        # Drop all rows that are empty
        spec_df = spec_df.dropna()

        ## Start: DEBUG
        # Drop all negative values
        spec_df = spec_df[spec_df > 0]

        ## Start: End 

        plt.plot(
            spec_df.index.astype(float),
            spec_df.values.astype(float),
            label=f"Feature {f+1}",
            marker=possible_markers[f % len(possible_markers)],
            alpha=0.5,
        )

    # Save figure
    fig.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title(title)
    fig.savefig(os.path.join(save_path, title))
    plt.close(fig)
