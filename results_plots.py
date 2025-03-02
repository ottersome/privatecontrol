from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import debugpy
import argparse

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--debug", action="store_true", help="Whether or not to use debugpy for trainig"
    )
    return ap.parse_args()


def load_data(
    original_path, sanitized_path, guesses_path, latent_z_path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load original and sanitized npy data."""
    original = np.load(original_path)
    sanitized = np.load(sanitized_path)
    guesses = np.load(guesses_path)
    latent_z = np.load(latent_z_path)

    return original, sanitized, guesses, latent_z

def compute_correlations(original, sanitized):
    """Compute Pearson and Spearman correlation coefficients."""
    pearson_corrs = []
    spearman_corrs = []
    for i in range(original.shape[-1]):
        pearson_corr, _ = pearsonr(original[:, i].flatten(), sanitized[:, i].flatten())
        spearman_corr, _ = spearmanr(original[:, i].flatten(), sanitized[:, i].flatten())
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
    return np.array(pearson_corrs), np.array(spearman_corrs)

def meep_compute_correlations(og_or_sanitized: np.ndarray, sensitive: np.ndarray):
    """
    For now we assume sensitive is a single column
    """
    assert len(sensitive.shape) == 1, f"Your sensitive tensor is of shape {sensitive.shape}, it should be a 1-d vector."

    pearson_corrs = []
    spearman_corrs = []
    for i in range(og_or_sanitized.shape[-1]):
        pearson_corr, _ = pearsonr(og_or_sanitized[:, i].flatten(), sensitive)
        spearman_corr, _ = spearmanr(og_or_sanitized[:, i].flatten(), sensitive)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
    return np.array(pearson_corrs), np.array(spearman_corrs)


def plot_correlation_distributions(pearson_corrs, spearman_corrs, save_dist: str, title: str):
    """Plot distributions of Pearson and Spearman correlations."""
    plt.figure(figsize=(12, 5))
    sns.histplot(pearson_corrs, kde=True, label='Pearson', color='blue', bins=30)
    sns.histplot(spearman_corrs, kde=True, label='Spearman', color='orange', bins=30)
    plt.xlabel("Correlation Coefficient")
    plt.xlim(-1, 1)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig(save_dist, dpi=300)
    plt.show()

def perform_pca(original, sanitized, n_components=2):
    """Perform PCA on original and sanitized data."""
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original)
    sanitized_scaled = scaler.transform(sanitized)
    
    pca = PCA(n_components=n_components)
    original_pca = pca.fit_transform(original_scaled)
    sanitized_pca = pca.transform(sanitized_scaled)
    
    return original_pca, sanitized_pca

def plot_pca(original_pca, sanitized_pca):
    """Plot PCA comparison before and after sanitization."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.5, label='Original', color='blue')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Before Sanitization")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(sanitized_pca[:, 0], sanitized_pca[:, 1], alpha=0.5, label='Sanitized', color='red')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA After Sanitization")
    plt.legend()
    
    plt.show()

def plot_all(features: np.ndarray, idxs: List[int], fig_name: str):
    """
    Simply plot all 16 features in a 4x4 grid
    """
    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    print(f" feature_shape: {len(features)}, idxs_len: {len(idxs)}")
    for i in range(features.shape[-1]):
        current_idx = idxs[i]
        plt.subplot(4, 4, current_idx+1)
        plt.plot(features[:, i])
        plt.title(f"Feature {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
    plt.savefig(f"figures/{fig_name}.png")
    plt.close()


def main(args: argparse.Namespace):

    original_path = "./results/val_x_[5].npy"  # Original data
    sanitized_path = "./results/sanitized_x_[5].npy"  # Change to actual path
    guesses_path = "./results/adv_guesses_y_[5].npy"  # Change to actual path
    latent_z_path = "./results/latent_z_[5].npy"  # Change to actual path
    sequence_length = 32

    original, sanitized, guesses, latent_z = load_data(
        original_path, sanitized_path, guesses_path,  latent_z_path
    )
    original = original[sequence_length:,:]
    print(f"original data size {original.shape}")
    print(f"sanitizewd data size {sanitized.shape}")
    print(f"guesses data size {guesses.shape}")
    print(f"latent_z data size {latent_z.shape}")

    priv_col = 5
    public_feats = list(set(range(original.shape[-1])) - {priv_col})
    private_feats= [priv_col]

    # Just for good measure plot_all the figures
    plot_all(original, list(range(16)), "Original")

    none_idxs = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plot_all(sanitized, none_idxs, "Sanitized")
    

    # Plot the sanitized
    plot_signal_reconstructions(
        original[:, public_feats],
        sanitized,
        "figures/sanitized_signal_reconstructions.png",
        ids=[6, 7],
    )

    # Plot the adversary guesses
    plot_signal_reconstructions(
        original[:, private_feats], guesses, "figures/adversary_guesses.png"
    )

    # Compute all correlations
    compute_all_correlations(original, sanitized, latent_z, public_feats, private_feats)

    
    # Perform and plot PCA
    original_pca, sanitized_pca = perform_pca(original, sanitized, n_components=2)
    # plot_pca(original_pca, sanitized_pca)


def compute_all_correlations(
    original: np.ndarray,
    sanitized_feats: np.ndarray,
    latent: np.ndarray,
    public_feats: List[int],
    private_feats: List[int],
):
    """
    Simply for organizational purposes
    """
    # Compute correlations for real aspects
    pearson_corrs, spearman_corrs = meep_compute_correlations(
        original[:, public_feats], original[:, private_feats].flatten()
    )
    print(
        f"Correlations between original and sensitive: \n\tPearson: {pearson_corrs}\n\tSpearman:{spearman_corrs}"
    )
    plot_correlation_distributions(
        pearson_corrs,
        spearman_corrs,
        "figures/correlation_distributions_original.png",
        "Distribution of Pearson and Spearman Correlations: Original vs Sensitive",
    )
    pearson_corrs, spearman_corrs = meep_compute_correlations(
        latent, original[:, private_feats].flatten()
    )
    print(
        f"Correlations between latent_z and sensitive: \n\tPearson: {pearson_corrs}\n\tSpearman:{spearman_corrs}",
    )
    plot_correlation_distributions(
        pearson_corrs,
        spearman_corrs,
        "figures/correlation_distributions_latent.png",
        "Distribution of Pearson and Spearman Correlations: Original vs Sensitive: Original vs Latent",
    )


    # Compute Correlations for Sanitized
    pearson_corrs, spearman_corrs = meep_compute_correlations(
        sanitized_feats, original[:, private_feats].flatten()
    )
    print(
        f"Correlations between sanitized and sensitive: \n\tPearson: {pearson_corrs}\n\tSpearman:{spearman_corrs}",
    )
    plot_correlation_distributions(
        pearson_corrs,
        spearman_corrs,
        "figures/correlation_distributions_sanitized.png",
        "Distribution of Pearson and Spearman Correlations: Original vs Sensitive: Original vs Sanitized",
    )



def plot_signal_reconstructions(original, sanitized, save_name: str, ids=None):
    """Plot signal reconstructions using Seaborn with a clean look."""
    print(f"original data size {original.shape}")
    print(f"sanitized data size {sanitized.shape}")
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
        sns.lineplot(x=np.arange(original.shape[0]), y=original[:, idx], ax=ax, label="Truth", color="orange")
        sns.lineplot(x=np.arange(sanitized.shape[0]), y=sanitized[:, idx], ax=ax, label="Reconstruction", color="blue")
        ax.set_title(f"Reconstruction Vs Truth of $f_{{{idx}}}$")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    
if __name__ == "__main__":

    args = argsies()

    if args.debug:
        print("Waiting for debugger to attach...")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()
        print("Debugger attached.")

    main(args)

