"""
File to help interpret the results from the paretto simulations
"""
import argparse
import os

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
    # Import data
    data_dir = args.data_dir
    privacies = np.load(os.path.join(data_dir, "privacies.npy"))
    utilities = np.load(os.path.join(data_dir, "utilities.npy"))
    uvps = np.load(os.path.join(data_dir, "uvp.npy"))
    # privacies = np.log(privacies)
    # utilities = -np.log(-utilities)
    # For now lets normalize the utilities

    left_hull_x, left_hull_y = paretto_frontier(privacies, utilities, uvps)

    print(f"Privacies are {privacies}")
    print(f"Utilities are {utilities}")

    # Plot the pareto frontier
    plt.figure(figsize=(16, 12))  # Standard figure size for paper columns
    plt.tight_layout()
    
    # Set the style for academic publications
    # plt.style.use("seaborn-whitegrid")
    sns.set_style("whitegrid")
    sns.set_context("paper")
    
    # Create the main scatter plot with improved aesthetics
    _ = plt.scatter(privacies, utilities, 
                    color='#2E86C1',  # Professional blue color
                    s=100,  # Marker size
                    alpha=0.7)  # Slight transparency

    # # Setup logarithmic scale axis
    # plt.xscale("log", base=10)
    # plt.yscale("log", base=10)   

    
    # Add annotations with improved positioning and style
    for i, uvp in enumerate(uvps):
        plt.annotate(f"UVP: {uvp:.4f}", 
                    (privacies[i], utilities[i]),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    

    plt.plot(left_hull_x, left_hull_y, 'g-', label='Pareto frontier curve')
    
    # Customize the plot
    plt.title("Privacy-Utility Trade-off Analysis", pad=20)
    plt.xlabel("Privacy Score", labelpad=10)
    plt.ylabel("Utility Score", labelpad=10)
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save with high DPI for print quality
    print(f"Saving to {args.saveplot_dest}")
    plt.savefig(args.saveplot_dest, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.show()
    plt.close()

if __name__ == "__main__":
    args = argsies()
    main(args)
