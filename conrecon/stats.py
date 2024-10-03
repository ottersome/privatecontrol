import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb


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
