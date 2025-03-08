"""
File to help interpret the results from the paretto simulations
"""
import argparse
import os

from adjustText import adjust_text
import numpy as np
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "--data_dir",
        default="./results/uvp_data_2025-03-06_17-09-22",
        type=str, 
        help="Where our data is stored",
    )
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/privacy_vs_utility_posthoc.png",
        type=str,
        help="Where to save the outputs",
    )

    return ap.parse_args()


def paretto_frontier(
    x: np.ndarray, y: np.ndarray, uvp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    assert len(x.shape) == 1, "x must be a 1d array"
    assert len(y.shape) == 1, "y must be a 1d array"
    assert x.shape == y.shape, "x and y must be the same shape"
    assert x.shape[0] >= 1, "x must have more than one element"

    # Sort the y direction
    xy = np.vstack((x, y)).transpose()
    xyu = np.hstack((xy, uvp.reshape(-1,1)))
    sorted_idxs = (-xyu[:, 1]).argsort()
    xyu = xyu[sorted_idxs]

    y_down = np.array([0, -1])
    down_angle_rad = np.arctan2(y_down[1], y_down[0])

    not_fixed = False # For sake of entering the loop
    cp = np.copy(xyu)
    while not not_fixed:
        not_fixed = True; # Sake of initialization
        ccp = np.copy(cp)
        del_idxs = []
        pidx = 0
        while pidx < (ccp.shape[0]-2):
            diff12 = ccp[pidx+1][:2] - ccp[pidx][:2]
            point12_angle = np.arctan2(diff12[1], diff12[0]) - down_angle_rad
            diff23 = ccp[pidx+2][:2] - ccp[pidx+1][:2]
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


 

def main(args: argparse.Namespace):
    # Set the style globally for the entire plot
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    
    # Import data
    data_dir = args.data_dir
    privacies = np.load(os.path.join(data_dir, "privacies.npy"))
    utilities = np.load(os.path.join(data_dir, "utilities.npy"))
    uvps = np.load(os.path.join(data_dir, "uvp.npy"))

    # Calculate Pareto frontier
    left_hull_x, left_hull_y = paretto_frontier(privacies, utilities, uvps)

    # Create figure with appropriate size for paper
    fig, ax = plt.subplots(figsize=(8, 6))  # Standard single-column figure size
    
    # Create the main scatter plot
    scatter = ax.scatter(privacies, utilities, 
                        color='#2E86C1',  # Professional blue color
                        s=80,  # Marker size
                        alpha=0.7,  # Slight transparency
                        label='Data points')

    # Add annotations for UVP values
    texts = []
    for i, uvp in enumerate(uvps):
        texts.append(
            ax.annotate(
                f"UVP: {uvp:.3f}",
                (privacies[i], utilities[i]),
                fontsize=8,
                # arrowprops=dict(
                #     arrowstyle='->',
                #     color='gray',
                #     alpha=0.6
                # )
            )
        )

    # Plot Pareto frontier
    ax.plot(left_hull_x, left_hull_y, 
            color='#27AE60',  # Professional green color
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

    # Adjust text annotations to prevent overlap
    adjust_text(
        texts,
        expand_points=(1.5, 1.5),
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
        force_points=(0.5, 0.5),
        force_text=(0.5, 0.5),
        ax=ax
    )

    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI for print quality
    plt.savefig(
        args.saveplot_dest,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        format='png'  # PDF format for vector graphics
    )
    
    # Show the plot
    # plt.show()
    plt.close()

if __name__ == "__main__":
    args = argsies()
    main(args)
