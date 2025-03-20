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
