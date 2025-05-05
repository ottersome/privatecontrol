import argparse
import logging
from math import ceil
import pickle
from typing import List, Optional, Sequence
import os

import debugpy
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from tqdm import tqdm

import wandb
from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import PCATemporalAdversary
from conrecon.figures.utils import set_professional_style
from conrecon.performance_test_functions import (
    pca_test_entire_file,
    test_pca_M_decorrelation,
)
from conrecon.stats import singleCol_compute_correlations
from conrecon.utils.common import (
    calculate_correlation,
    create_logger,
    inspect_array,
    set_seeds,
)
from conrecon.utils.datatyping import PCABenchmarkResults
from main_ts_adv import main_data_prep


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
    )

    # WARN: When you work with this code, keep in mind this hardcoded displacement for the private column
    ap.add_argument(
        "--cols_to_hide",
        default=[5 - 2],
        type=List,
        help="What columns to hide. For now too heavy assupmtion of singleton list",
    )
    ap.add_argument(
        "--correlation_threshold",
        default=0.1,
        type=float,
        help="Past which point we will no longer consider more influence from public features",
    )
    ap.add_argument("--episode_length", default=32, type=int)
    ap.add_argument(
        "--splits",
        default={"train_split": 0.8, "val_split": 0.2, "test_split": 0.0},
        type=list,
        nargs="+",
    )
    ap.add_argument(
        "--oversample_coefficient",
        default=1.6,
        type=float,
        help="The threshold for retaining principal components",
    )
    ap.add_argument("--wandb_on", "-w", action="store_true")
    ap.add_argument("--batch_size", "-b", default=64, type=int)
    ap.add_argument("--epochs", "-e", type=int, default=100)
    ap.add_argument(
        "--lr", type=float, default=0.001, help="Learning Rate (default: 0.001)"
    )
    ap.add_argument(
        "--debug", "-d", action="store_true", help="Wheter to active debugpy mode."
    )
    ap.add_argument("--device", default="cuda", type=str, help="What device to use")
    ap.add_argument("--seed", default=0, type=int, help="What device to use")

    ap.add_argument(
        "--test_file_name",
        type=str,
        default="run_5.csv",
        help="the one file that will be picked from the rest to test against.",
    )

    return ap.parse_args()


def baseline_pca_decorr_adversary_by_pc(
    principal_components: np.ndarray,
    pub_pc_projected: np.ndarray,
    train_all: np.ndarray,
    prv_features_idxs: Sequence[int],
    batch_size: int,
    num_pcs_to_remove: int,
    epochs: int,
    lr: float,
    device: torch.device,
    wandb_on: bool,
) -> tuple[nn.Module, nn.Module, torch.Tensor, np.ndarray]:

    # Prepping some data
    pub_features_idxs = list(set(range(train_all.shape[-1])) - set(prv_features_idxs))
    train_all_flat = train_all.reshape(-1, train_all.shape[-1])
    train_all_centered = train_all_flat - np.mean(train_all_flat, axis=0)
    train_pub = train_all[:, :, pub_features_idxs]
    train_prv = train_all[:, :, prv_features_idxs]
    train_prv_flat = train_prv.reshape(-1, train_prv.shape[-1])
    # train_pub_flat = train_pub.reshape(-1, train_pub.shape[-1])
    # train_pub_centered = train_pub_flat - np.mean(train_pub_flat, axis=0)
    sequence_length = train_all.shape[1]
    num_features = train_all.shape[-1]
    num_pub_features = len(pub_features_idxs)
    num_priv_features = len(prv_features_idxs)
    train_pub_tensor = torch.tensor(train_pub).to(device).to(torch.float32)
    train_prv_tensor = torch.tensor(train_prv).to(device).to(torch.float32)

    ########################################
    # Check on correlations
    ########################################
    # TODO: Use the function in utils/common.py to calculate the correlation instead of repeating this
    all_pc_corr_scores = []
    for i in range(principal_components.shape[0]):
        pc_i_timeseries = pub_pc_projected[:, i]
        corr_i = np.corrcoef(pc_i_timeseries, train_prv_flat.squeeze())[0, 1]
        all_pc_corr_scores.append(corr_i)
        # ensure corr_i is not nan
        assert not np.isnan(corr_i)

    # For debugging mostly
    all_pc_corr_scores = np.array(all_pc_corr_scores)

    # Argsort to get the most correlated components
    most_correlated_comps_idxs = np.argsort(np.abs(all_pc_corr_scores))[::-1][
        :num_pcs_to_remove
    ]
    logger.debug(
        f"Method 1 most correlated component: {most_correlated_comps_idxs if num_pcs_to_remove > 0 else None}"
    )
    remaining_idxs = np.setdiff1d(np.arange(num_features), most_correlated_comps_idxs)
    retained_components = principal_components[remaining_idxs]

    # Project the data onto the retained components
    sanitized_projections_for_training = train_all_centered.dot(retained_components.T)
    sanitized_feature_nums = retained_components.shape[0]
    sanitized_projections_for_training = sanitized_projections_for_training.reshape(
        -1, sequence_length, sanitized_feature_nums
    )
    sanitized_projections_for_training = (
        torch.from_numpy(sanitized_projections_for_training)
        .to(torch.float32)
        .to(device)
    )
    num_batches = ceil(sanitized_projections_for_training.shape[0] / batch_size)

    # We need to restructure the data.
    logger.info("Restructuring the data")

    criterion = nn.MSELoss()
    reconstructor = PCATemporalAdversary(
        num_principal_components=retained_components.shape[0],
        num_features_to_recon=num_pub_features,
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    adversary = PCATemporalAdversary(
        num_principal_components=retained_components.shape[0],
        num_features_to_recon=num_priv_features,
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    opt_reconstructor = torch.optim.Adam(reconstructor.parameters(), lr=lr)  # type: ignore
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr)  # type: ignore

    ############################################################
    # Training Based on the Sanitized PCA Components
    ############################################################
    reconstruction_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_inp = sanitized_projections_for_training[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            prv_feats = train_prv_tensor[
                batch_no * batch_size : (batch_no + 1) * batch_size,
                :,
            ]
            pub_feats = train_pub_tensor[
                batch_no * batch_size : (batch_no + 1) * batch_size,
                :,
            ]

            reconstructor_guess = reconstructor(batch_inp).squeeze()
            adversary_guess = adversary(batch_inp).squeeze()

            # Calculate the loss
            recon_loss = criterion(reconstructor_guess, pub_feats[:, -1].squeeze())
            adv_loss = criterion(adversary_guess, prv_feats[:, -1].squeeze())
            adversary.zero_grad()
            reconstructor.zero_grad()
            recon_loss.backward()
            adv_loss.backward()
            opt_reconstructor.step()
            opt_adversary.step()

            reconstruction_losses.append(recon_loss.item())
            adv_losses.append(adv_loss.item())

            if wandb_on:
                wandb.log(
                    {
                        "pca_recon_train_loss": recon_loss.item(),
                        "pca_adv_train_loss": adv_loss.item(),
                    }
                )

    # Plot reconstruction losses
    plt.figure(figsize=(16, 10))
    plt.plot(reconstruction_losses)

    # Find minimum value and its position
    min_value = np.min(reconstruction_losses)
    min_index = int(np.argmin(reconstruction_losses))

    # Add horizontal line at minimum
    plt.axhline(y=min_value, color="r", linestyle="--", label=f"Min: {min_value:.4f}")

    # Add a marker at the minimum point
    plt.plot(min_index, min_value, "ro", markersize=8)

    x_tick = min_index + len(reconstruction_losses) * 0.02
    # Add text annotation for the minimum value
    plt.text(x_tick, min_value, f"Min: {min_value:.4f}", verticalalignment="bottom")

    plt.title("Reconstruction Loss")
    plt.legend()
    os.makedirs("./figures/method_2_decorr_adv/", exist_ok=True)
    plt.savefig(
        f"./figures/method_2_decorr_adv/pca_recon_losses_numRem-{num_pcs_to_remove:02d}.png"
    )
    plt.close()
    retained_components = (
        torch.from_numpy(retained_components).to(device).to(torch.float32)
    )

    return reconstructor, adversary, retained_components, all_pc_corr_scores


def baseline_pca_decorr_adversary_w_threshold(
    principal_components: np.ndarray,
    pub_pc_projected: np.ndarray,
    train_prv: np.ndarray,
    train_pub: np.ndarray,
    batch_size: int,
    correlation_threshold: float,
    epochs: int,
    lr: float,
    device: torch.device,
    wandb_on: bool,
) -> tuple[nn.Module, torch.Tensor, np.ndarray]:

    # Prepping some data
    train_prv_flat = train_prv.reshape(-1, train_prv.shape[-1])
    train_pub_flat = train_pub.reshape(-1, train_pub.shape[-1])
    train_pub_centered = train_pub_flat - np.mean(train_pub_flat, axis=0)
    sequence_length = train_pub.shape[1]
    num_features = train_pub.shape[-1]
    num_priv_components = train_prv.shape[-1]
    num_pub_components = train_pub.shape[-1]
    train_pub_tensor = torch.tensor(train_pub).to(device).to(torch.float32)
    train_prv_tensor = torch.tensor(train_prv).to(device).to(torch.float32)

    ########################################
    # Check on correlations
    ########################################
    retained_components = []
    all_pc_corr_scores = []
    for i in range(principal_components.shape[0]):
        pc_i_timeseries = pub_pc_projected[:, i]
        corr_i = np.corrcoef(pc_i_timeseries, train_prv_flat.squeeze())[0, 1]
        all_pc_corr_scores.append(corr_i)
        # ensure corr_i is not nan
        assert not np.isnan(corr_i)
        if abs(corr_i) <= correlation_threshold:
            retained_components.append(principal_components[i])

    # For debugging mostly
    all_pc_corr_scores = np.array(all_pc_corr_scores)

    retained_components = np.array(retained_components)
    # Project the data onto the retained components
    sanitized_projections_for_training = train_pub_centered.dot(retained_components.T)
    sanitized_feature_nums = retained_components.shape[0]
    sanitized_projections_for_training = sanitized_projections_for_training.reshape(
        -1, sequence_length, sanitized_feature_nums
    )
    sanitized_projections_for_training = (
        torch.from_numpy(sanitized_projections_for_training)
        .to(torch.float32)
        .to(device)
    )
    num_batches = ceil(sanitized_projections_for_training.shape[0] / batch_size)

    # We need to restructure the data.
    logger.info("Restructuring the data")

    criterion = nn.MSELoss()
    reconstructor = PCATemporalAdversary(
        num_principal_components=retained_components.shape[0],
        num_features_to_recon=num_pub_components,
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    adversary = PCATemporalAdversary(
        num_principal_components=retained_components.shape[0],
        num_features_to_recon=num_priv_components,
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    opt_reconstructor = torch.optim.Adam(reconstructor.parameters(), lr=lr)  # type: ignore
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr)  # type: ignore

    ############################################################
    # Training Based on the Sanitized PCA Components
    ############################################################
    reconstruction_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_inp = sanitized_projections_for_training[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            prv_feats = train_prv_tensor[
                batch_no * batch_size : (batch_no + 1) * batch_size,
                :,
            ]
            pub_feats = train_pub_tensor[
                batch_no * batch_size : (batch_no + 1) * batch_size,
                :,
            ]

            reconstructor_guess = reconstructor(batch_inp).squeeze()
            adversary_guess = adversary(batch_inp).squeeze()

            # Calculate the loss
            recon_loss = criterion(reconstructor_guess, pub_feats[:, -1])
            adv_loss = criterion(adversary_guess, prv_feats[:, -1])
            adversary.zero_grad()
            reconstructor.zero_grad()
            recon_loss.backward()
            adv_loss.backward()
            opt_reconstructor.step()
            opt_adversary.step()

            reconstruction_losses.append(recon_loss.item())
            adv_losses.append(adv_loss.item())

            if wandb_on:
                wandb.log(
                    {
                        "pca_recon_train_loss": recon_loss.item(),
                        "pca_adv_train_loss": adv_loss.item(),
                    }
                )

    # Plot reconstruction losses
    plt.figure(figsize=(16, 10))
    plt.plot(reconstruction_losses)
    plt.title("Reconstruction Loss")
    plt.savefig(f"./figures/pca_recon_losses.png")
    plt.close()

    return adversary, torch.from_numpy(retained_components), all_pc_corr_scores


def pca_preprocessing(
    train_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Will preprocess the data for PCA
    args:
        - train_all: All data (public + private)
    returns:
        - pca_components: Principal Components
        - pca_projected_ds: Projected Public data onto the principal components
        - train_mean: Mean of the training data
    """

    pca = PCA()
    train_all_flat = train_all.reshape(-1, train_all.shape[-1])
    train_mean = np.mean(train_all_flat, axis=0)
    train_all_centered = train_all_flat - train_mean

    ########################################
    # PCA Fitting
    ########################################
    pca_transform = pca.fit(train_all_centered)
    # Shape is (num_components, num_features), fyi for indexing purposes
    principal_components = pca_transform.components_
    # Takes in (num_componets, num_features) and returns (num_components, num_samples)
    pca_projected_ds = train_all_centered.dot(principal_components.T)

    return principal_components, pca_projected_ds, train_mean


def pca_decomposition_w_heatmap(
    pca_components: np.ndarray,
    pca_projected_ds: np.ndarray,
    train_pub: np.ndarray,
    train_prv: np.ndarray,
    pub_features_idxs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    This will create a matrixM_{up} that will help us convert public components into private components.
    returns: M_UI
    """
    # inspect_array("clean_test_file", train_pub)
    train_prv_flat = train_prv.reshape(-1, train_prv.shape[-1])
    train_pub_flat = train_pub.reshape(-1, train_pub.shape[-1])
    C = pca_projected_ds

    # The projectionmatrix from public to latent space
    np.savetxt(
        f"./csvs/pca_components.csv",
        pca_components,
        delimiter=",",
        header=",".join(map(str, range(pca_components.shape[-1]))),
    )
    M_ul = pca_components.T  # (Num latent components, components size / num features)

    # inspect_array("pca_components", pca_components)
    # inspect_array("M_ul",M_ul)

    # ----- Step 2: Solve for M_CU via linear regression -----
    # We assume X_U \approx C @ M_CU
    # M_CU has shape (K, N-M)
    # Using np.linalg.lstsq to solve: M_CU = argmin ||C @ M - X_U||^2
    inspect_array("C", C)
    # inspect_array("train_prv_flat", train_prv_flat)
    inspect_array("train_pub_flat", train_pub_flat)
    M_li, _, _, s = np.linalg.lstsq(C, train_prv_flat, rcond=None)
    M_lu, _, _, s = np.linalg.lstsq(C, train_pub_flat, rcond=None)

    # ----- (Optional) Directly check (C^T C) invertibility -----
    # If C^T C is singular or near-singular, direct inversion is problematic
    CtC = C.T @ C
    # Attempt inversion (for demonstration; might fail if singular)
    try:
        CtC_inv = np.linalg.inv(CtC)
        print("C^T C was invertible.")
    except np.linalg.LinAlgError:
        print("C^T C is singular or not well-conditioned.")

    # ----- Step 3: Compute the final M_PU = M_CP @ M_CU -----
    M_UI = (
        M_ul @ M_li
    )  # shape: (M, N-M) i.e. (num_public_components, num_private_components)
    M_UU = (
        M_ul @ M_lu
    )  # shape: (M, N-M) i.e. (num_public_components, num_private_components)
    np.savetxt(f"./csvs/M_UI.csv", M_UI, delimiter=",")
    np.savetxt(f"./csvs/M_UU.csv", M_UU, delimiter=",")
    # # inspect_array("M_UI", M_UI)
    # # inspect_array("M_UU", M_UU)
    # inspect_array("M_ul", M_ul)
    # inspect_array("M_li", M_li)
    # inspect_array("M_lu", M_lu)
    return M_UI, M_UU


def plotNSave_heatmap_and_correlation(
    M_UI: np.ndarray,
    all_pc_corr_scores: np.ndarray,
    principal_components: np.ndarray,
    pathOut_heatNCor: str,
    pathOut_PCsMat: str,
    # Optional
    public_feature_names: Optional[list[str]] = None,
    private_feature_label: str = "Private Feature",
    heatmap_cmap: str = 'viridis', # Colormap choice
):
    ########################################
    # Plot the HeatMap
    ########################################
    set_professional_style()
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))

    paths = [pathOut_PCsMat, pathOut_heatNCor]
    for path in paths:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)

    ########################################
    # Plot 1: HeatMap and Correlation Scores
    ########################################
    # Adjust figsize for a potentially better slide aspect ratio
    fig1, axs = plt.subplots(2, 1, figsize=(12, 9)) # Slightly less wide, taller

    # --- Upper subplot: Heatmap ---
    # Use imshow for potentially better aspect control if needed, matshow is fine too
    influence_values = (M_UI.T - np.min(M_UI.T)) / (np.max(M_UI.T) - np.min(M_UI.T))
    im = axs[0].matshow(influence_values, cmap=heatmap_cmap, aspect='auto') # Use chosen cmap
    axs[0].set_title(r"$M_{UI}$: Public-to-Private Feature Influence Mapping", fontweight='bold') # Use r"" for latex
    axs[0].set_xlabel("Public Features")
    axs[0].set_ylabel(private_feature_label)
    # Hide Y ticks if they don't represent meaningful discrete steps (often the case for heatmaps)
    axs[0].set_yticks([])
    axs[0].tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False) # Ensure x-ticks are at bottom

    # Add a vertical colorbar (often fits better)
    cbar1 = fig1.colorbar(im, ax=axs[0], orientation='vertical', pad=0.03, shrink=0.8)
    cbar1.set_label('Influence Value') # Add label to colorbar

    # --- Lower subplot: Bar plot ---
    num_features = len(all_pc_corr_scores)
    feature_indices = np.arange(num_features)
    # Use the style's default color cycle by not specifying 'color'
    axs[1].bar(feature_indices, all_pc_corr_scores)
    axs[1].set_title("Correlation Scores", fontweight='bold')
    axs[1].set_xlabel("Public Features")
    axs[1].set_ylabel("Correlation Score")

    # Set x-ticks: Use names if provided, otherwise indices
    if public_feature_names and len(public_feature_names) == num_features:
        axs[1].set_xticks(feature_indices)
        axs[1].set_xticklabels(public_feature_names, rotation=45, ha='right') # Rotate if names are long
    else:
        # Show fewer ticks if there are too many features to avoid clutter
        tick_step = max(1, num_features // 10) # Show ~10 ticks max
        axs[1].set_xticks(np.arange(0, num_features, tick_step))

    # Customize grid (style sets defaults, but we can override)
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.5) # Horizontal grid lines only
    axs[1].grid(False, axis='x') # Turn off vertical grid lines for bar chart

    # Improve layout
    fig1.tight_layout(pad=2.0) # Add padding between plots and edges

    # Save the figure
    plt.savefig(pathOut_heatNCor, bbox_inches='tight')
    print(f"Saved combined heatmap and correlation plot to: {pathOut_heatNCor}")
    plt.close(fig1) # Close the figure to free memory

    ########################################
    # Plot 2: Principal Components Heatmap
    ########################################
    fig2, ax = plt.subplots(figsize=(12, 4)) # Adjust size

    # Assuming principal_components rows are components, columns are original features
    # If we want PC vs Public Features, we might not need transpose depending on shape
    # Original code used principal_components.T - assuming Private Features vs Public Features?
    # Let's assume shape is (n_components, n_public_features) and we want to show this.
    # If the intent was (n_public_features, n_private_features) like M_PU, transpose it.
    # Sticking to original code's transpose:
    pc_data = principal_components.T
    num_public_features_pc = pc_data.shape[1]
    num_private_features_pc = pc_data.shape[0]

    im_pc = ax.matshow(pc_data, cmap=heatmap_cmap, aspect='auto')
    ax.set_title("Principal Components Matrix", fontweight='bold')
    ax.set_xlabel("Public Features")
    ax.set_ylabel(private_feature_label) # Reuse label, assuming it corresponds

    # Ticks for PC heatmap
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    if public_feature_names and len(public_feature_names) == num_public_features_pc:
        ax.set_xticks(np.arange(num_public_features_pc))
        ax.set_xticklabels(public_feature_names, rotation=45, ha='right')
    else:
        tick_step_pc = max(1, num_public_features_pc // 10)
        ax.set_xticks(np.arange(0, num_public_features_pc, tick_step_pc))
        # Add labels for indices if no names provided
        ax.set_xticklabels(np.arange(0, num_public_features_pc, tick_step_pc))


    # Y-ticks for PC heatmap: Show if the number of private features/components is small
    if num_private_features_pc <= 10: # Show ticks if 10 or fewer
         ax.set_yticks(np.arange(num_private_features_pc))
         # You could add labels here if they represent specific components, e.g., ["PC1", "PC2", ...]
         # ax.set_yticklabels([f"Component {i+1}" for i in range(num_private_features_pc)])
    else:
         ax.set_yticks([]) # Hide if too many


    # Add vertical colorbar
    cbar2 = fig2.colorbar(im_pc, ax=ax, orientation='vertical', pad=0.03, shrink=0.8)
    cbar2.set_label('Component Value')

    fig2.tight_layout(pad=1.5)
    plt.savefig(pathOut_PCsMat, bbox_inches='tight')
    print(f"Saved Principal Components heatmap to: {pathOut_PCsMat}")
    plt.close(fig2) # Close the figure


def method_2_latent_spc_deco(
    num_pcs_to_remove: int,
    pca_components: np.ndarray,
    pub_pc_projected: np.ndarray,
    test_file: np.ndarray,
    all_train_seqs: np.ndarray,
    prv_features_idxs: Sequence[int],
    train_mean: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[float, float]:
    os.makedirs("./figures/method2/", exist_ok=True)
    # Method 1. Latent Space Decorrelatio
    pca_reconstructor, pca_model_adversary, retained_components, all_pc_corr_scores = (
        baseline_pca_decorr_adversary_by_pc(
            pca_components,
            pub_pc_projected,
            all_train_seqs,
            prv_features_idxs,
            args.batch_size,
            num_pcs_to_remove,
            args.epochs,
            args.lr,
            args.device,
            args.wandb_on,
        )
    )
    # Really quick use the salient heatmap to recover private features
    # private_guess = test_pub.squeeze().dot(M_PU)

    _, utility, privacy = pca_test_entire_file(
        test_file,
        args.cols_to_hide,
        pca_reconstructor,
        pca_model_adversary,
        retained_components,
        train_mean,
        args.episode_length,
        None,
        logger,
        args.batch_size,
        wandb_on=args.wandb_on,
    )
    return utility, privacy


def method_1_MUI_decorrelation(
    num_features_to_remove: int,
    tot_num_features: int,
    test_pub: np.ndarray,
    test_file_rel_pubs: np.ndarray,
    test_prv: np.ndarray,
    M_UI: np.ndarray,
    M_UU: np.ndarray,
) -> tuple[int, float, float]:
    """
    Will grab the already pretrained/conditioned M_ui and slowly test the removal of some columns until we get zero utility. Out of it
    Args:
        - tot_num_features: Total number of features in the data
        - num_features_to_remove: Number of features to remove
        - test_pub: Public data
        - test_file_rel_pubs: Public data pruned with the deemd most salient feature
        - test_prv: Private data
        - pub_features_idxs: Public features indices
        - priv_features_idxs: Private features indices
        - M_UI: M_UI matrix
        - M_UU: M_UU matrix
    returns:
        - most_salient_feat_idx: The most salient feature index
        - utility: Utility of the model after removing the most salient feature
        - privacy: Privacy of the model after removing the most salient feature
    """
    assert (
        M_UI.shape[1] == 1
    ), f"Currently only working with a single private component. Instead we got {M_UI.shape}"
    num_features = tot_num_features
    num_features_kept = num_features - num_features_to_remove
    most_correlated_feats_idxs = np.argsort(np.abs(M_UI.squeeze()))[::-1]
    logger.debug(
        f"M2: Most correlated feats when remoivng {num_features_to_remove} columns are {most_correlated_feats_idxs}"
    )
    most_salient_feat_idx = most_correlated_feats_idxs[0]

    # test_file_pruned = test_file_flattend[:,max_corr_idxs]

    recovered_private_guess = test_file_rel_pubs.dot(M_UI)
    recovered_public_guess = test_file_rel_pubs.dot(M_UU)

    inspect_array("recovered_private_guess", recovered_private_guess)
    test_pca_M_decorrelation(
        recovered_private_guess,
        recovered_public_guess,
        test_pub,
        test_prv,
        num_features_kept,
    )

    # Simply calculate the differences and report them back
    utility = -1 * np.mean((recovered_public_guess - test_pub) ** 2)
    privacy = np.mean((recovered_private_guess - test_prv) ** 2)

    return most_salient_feat_idx, utility, privacy

    # TODO: Make this work
    # plot_heatmap_and_correlation(
    #     M_UI_pruned,
    #     all_pc_corr_scores,
    #     pca_components,
    #     private_guess,
    #     test_prv,
    # )


def main(args: argparse.Namespace):

    if args.debug:
        print("Waiting for debugger to attach...")
        debugpy.listen(("0.0.0.0", 42022))
        debugpy.wait_for_client()
        print("Debugger attached.")

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")
    ########################################
    # Load Data
    ########################################
    # columns, runs_dict, test_file = load_defacto_data(args.defacto_data_raw_path)
    # train_seqs, val_seqs, test_file = split_defacto_runs(
    #     runs_dict,
    #     args.splits["train_split"],
    #     args.splits["val_split"],
    #     args.episode_length,
    #     args.oversample_coefficient,
    #     True,  # Scale
    # )
    # all_train_seqs = np.concatenate([seqs for _, seqs in train_seqs.items()], axis=0)
    #
    # non_0_idxs = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # test_file = test_file[:, non_0_idxs]
    # all_train_seqs = all_train_seqs[:, :, non_0_idxs]
    #
    # num_pub_features = len(pub_features_idxs)
    # print(f"Public features are {pub_features_idxs}")
    # print(f"Private features are {prv_features_idxs}")
    all_train_seqs, all_valid_seqs, test_file = main_data_prep(
        cols_to_hide=args.cols_to_hide,
        defacto_data_raw_path=args.defacto_data_raw_path,
        device=device,
        episode_length=args.episode_length,
        oversample_coefficient=args.oversample_coefficient,
        splits=args.splits,
        test_file_name=args.test_file_name,
    )
    all_train_seqs = all_train_seqs.cpu().numpy()
    num_features = all_train_seqs.shape[-1]
    num_pca_components = num_features
    prv_features_idxs: list[int] = args.cols_to_hide
    pub_features_idxs = list(set(range(num_features)) - set(prv_features_idxs))
    num_pub_features = len(pub_features_idxs)
    train_pub = all_train_seqs[:, :, pub_features_idxs]
    train_prv = all_train_seqs[:, :, prv_features_idxs]


    corrs = calculate_correlation(
        train_prv.reshape((-1, train_prv.shape[-1])),
        train_pub.reshape((-1, train_pub.shape[-1])),
    )
    logger.info(f"Correlations are {corrs}")
    argsort_idxs = np.argsort(np.abs(corrs))[::-1]
    logger.info(f"With the descending correlation indices: {argsort_idxs}")
    # Maybe we can plot them as well.
    # DO a barplot of the correlations
    plt.bar(range(corrs.shape[-1]), corrs.squeeze())
    plt.xticks(np.arange(0, corrs.shape[-1], 1))
    plt.savefig("figures/initial_correlations.png")
    plt.close()

    test_pub = test_file[:, pub_features_idxs]
    test_prv = test_file[:, prv_features_idxs]

    ########################################
    # PCA Preprocessing
    ########################################
    logger.info("Starting the PCA preprocessing")
    pca_components, pca_projected_ds, train_mean = pca_preprocessing(
        all_train_seqs
    )

    ########################################
    # Get M_UI for the MUI method
    ########################################
    M_UI, _ = pca_decomposition_w_heatmap(
        pca_components,
        pca_projected_ds,
        train_pub,
        train_prv,
        pub_features_idxs,
    )
    logger.info(f"M_UI components are {M_UI.shape}")

    ########################################
    # Figure Initialization
    #   We will initalize most figure stuff
    #   here so we can pass them as parameters
    #   to functions and keeep them separate
    ########################################
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    axs.set_title("Original Data")
    axs.set_xlabel("Privacy")
    axs.set_ylabel("Utility")

    ########################################
    # Training the PCA based baseline
    ########################################
    # TOREM: Move this lower down for when we are done with it.
    logger.info("Starting the PCA decorrelation and training")
    assert (
        num_features == num_pca_components
    ), "The following loop assumes these are the same amount"

    cur_train_pub = train_pub.copy()
    cur_test_pub = test_pub.copy()
    # Mostly for debugging
    cur_train_prv = train_prv.copy()
    cur_test_prv = test_prv.copy()
    #
    next_id_to_rm: Optional[int] = None
    m1_utilities, m1_privacies = [], []
    m2_utilities, m2_privacies = [], []
    m1_m2_uvps = []

    for num_comps_to_remove in range(0, num_pub_features):
        # Method 1. Latent Space Decorrelatio
        logger.info(
            f" Before method 1 pca_components shape is : {pca_components.shape}"
        )

        # Prep data for Method 2
        if next_id_to_rm is not None:
            inv_set = list(set(range(cur_train_pub.shape[-1])) - set([next_id_to_rm]))
            logger.debug(
                f"M2: inv_set looks like {inv_set}. Where {next_id_to_rm} was removed from {list(range(cur_train_pub.shape[-1]))}"
            )
            cur_train_pub = cur_train_pub[:, :, inv_set]
            cur_test_pub = cur_test_pub[:, inv_set]

        new_pca_components, _, _ = pca_preprocessing(
            cur_train_pub,
        )
        new_projected_test_ds = cur_test_pub.dot(new_pca_components)

        # inspect_array("new_pca_components", new_pca_components)
        new_M_UI, new_M_UU = pca_decomposition_w_heatmap(
            new_pca_components,
            new_projected_test_ds,
            test_pub,
            test_prv,
            pub_features_idxs,
        )

        next_id_to_rm, m1_utility, m1_privacy = method_1_MUI_decorrelation(
            num_comps_to_remove,
            num_features,
            test_pub,
            cur_test_pub,
            test_prv,
            new_M_UI,
            new_M_UU,
        )

        m1_utilities.append(m1_utility)
        m1_privacies.append(m1_privacy)
        m1_m2_uvps.append(num_comps_to_remove)

        # Method 2
        m2_utility, m2_privacy = method_2_latent_spc_deco(
            num_comps_to_remove,
            pca_components,
            pca_projected_ds,
            test_file,
            all_train_seqs,
            prv_features_idxs,
            train_mean,
            args,
            logger,
        )

        m2_utilities.append(m2_utility)
        m2_privacies.append(m2_privacy)

        # Some Visualization
        pear_scores, spear_scores = singleCol_compute_correlations(cur_test_pub, cur_test_prv.squeeze())
        plotNSave_heatmap_and_correlation(
            new_M_UI,
            np.abs(pear_scores),
            pca_components,
            f"./figures/heatmap_debugging/pca_heatmap_removedComps-{num_comps_to_remove:02d}.png",
            f"./figures/heatmap_debugging/pcs_debug_removedComps-{num_comps_to_remove:02d}.png",
        )

        # print(f"new_M_UI shape: {new_M_UI.shape} and content: {new_M_UI}")
        # # DEBUG: Let me just plouyt M_UI here to ensure some level of consistency.
        # plt.bar(range(new_M_UI.shape[0]), np.abs(new_M_UI.squeeze()))
        # plt.title(f"M_UI at at comp_remvd {num_comps_to_remove}. (id_rem: {next_id_to_rm})")
        # plt.savefig(f"figures/new_MUI/initial_correlations_{num_comps_to_remove:02d}.png")
        # plt.close()
        # TODO: make sure m2_removed_pub_columns is updated before we leave this loop
        logger.debug(
            f"--------------------END OF ITERATION FOR `num_comps_to_remove`={num_comps_to_remove}--------------------"
        )

    results = PCABenchmarkResults(
        m1_utilities=m1_utilities,
        m2_utilities=m2_utilities,
        m1_privacies=m1_privacies,
        m2_privacies=m2_privacies,
        m1_m2_num_removed_components=m1_m2_uvps,
        num_iterations=num_pub_features,
    )
    with open(f"results/results_benchmarks.pkl", "wb") as f:
        pickle.dump(results, f)

    # retained_components = retained_components.to(device).to(torch.float32)
    # logger.info("PCA training and decorrelation complete. Now testing...")
    # pca_test_entire_file(
    #     test_file,
    #     args.cols_to_hide,
    #     pca_model_adversary,
    #     retained_components,
    #     args.episode_length,
    #     None,
    #     logger,
    #     args.batch_size,
    #     wandb_on=args.wandb_on
    # )

    #  TODO: remove
    # method_2_plots(M_PU, all_pc_corr_scores, pca_components, private_guess, test_prv)

    # TODO: Make this work

    logger.info("Done with the PCA test")


if __name__ == "__main__":
    logger = create_logger("pca_benchmarks")
    args = argsies()
    main(args)
