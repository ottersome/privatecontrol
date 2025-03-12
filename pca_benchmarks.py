import argparse
import logging
from math import ceil
from typing import List, OrderedDict, Sequence

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
from conrecon.performance_test_functions import pca_test_entire_file
from conrecon.utils.common import create_logger, set_seeds


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
    )
    ap.add_argument("--cols_to_hide", default=[5], type=List, help="What columns to hide. For now too heavy assupmtion of singleton list")
    ap.add_argument("--correlation_threshold", default=0.1 , type=float, help="Past which point we will no longer consider more influence from public features")
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
    ap.add_argument("--lr", type=float, default=0.001, help="Learning Rate (default: 0.001)")
    ap.add_argument("--debug", "-d", action="store_true", help="Wheter to active debugpy mode.")
    ap.add_argument("--device", default="cuda", type=str, help="What device to use")
    ap.add_argument("--seed", default=0, type=int, help="What device to use")

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
    all_pc_corr_scores = []
    for i in range(principal_components.shape[0]):
        pc_i_timeseries = pub_pc_projected[:,i]
        corr_i = np.corrcoef(pc_i_timeseries, train_prv_flat.squeeze())[0,1]
        all_pc_corr_scores.append(corr_i)
        # ensure corr_i is not nan
        assert not np.isnan(corr_i)

    # For debugging mostly
    all_pc_corr_scores = np.array(all_pc_corr_scores)

    # Argsort to get the most correlated components
    most_correlate_comps_idxs = np.argsort(np.abs(all_pc_corr_scores))[::-1][:num_pcs_to_remove]
    remaining_idxs = np.setdiff1d(np.arange(num_features), most_correlate_comps_idxs)
    retained_components = principal_components[remaining_idxs]

    # Project the data onto the retained components
    sanitized_projections_for_training = train_all_centered.dot(retained_components.T)
    sanitized_feature_nums = retained_components.shape[0]
    sanitized_projections_for_training = sanitized_projections_for_training.reshape(-1, sequence_length, sanitized_feature_nums)
    sanitized_projections_for_training = torch.from_numpy(sanitized_projections_for_training).to(torch.float32).to(device)
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
    opt_reconstructor = torch.optim.Adam(reconstructor.parameters(), lr=lr) # type: ignore
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr) # type: ignore

    ############################################################
    # Training Based on the Sanitized PCA Components
    ############################################################
    reconstruction_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_inp = sanitized_projections_for_training[batch_no * batch_size : (batch_no + 1) * batch_size]
            prv_feats = train_prv_tensor[batch_no * batch_size : (batch_no + 1) * batch_size, :, ]
            pub_feats = train_pub_tensor[batch_no * batch_size : (batch_no + 1) * batch_size, :, ]

            reconstructor_guess = reconstructor(batch_inp).squeeze()
            adversary_guess = adversary(batch_inp).squeeze()

            # Calculate the loss
            recon_loss = criterion(reconstructor_guess, pub_feats[:,-1].squeeze())
            adv_loss = criterion(adversary_guess, prv_feats[:,-1].squeeze())
            adversary.zero_grad()
            reconstructor.zero_grad()
            recon_loss.backward()
            adv_loss.backward()
            opt_reconstructor.step()
            opt_adversary.step()

            reconstruction_losses.append(recon_loss.item())
            adv_losses.append(adv_loss.item())

            if wandb_on:
                wandb.log({
                    "pca_recon_train_loss": recon_loss.item(),
                    "pca_adv_train_loss": adv_loss.item(),
                })

    # Plot reconstruction losses
    plt.figure(figsize=(16,10))
    plt.plot(reconstruction_losses)
    plt.title("Reconstruction Loss")
    plt.savefig(f"./figures/pca_recon_losses.png")
    plt.close()

    retained_components = torch.from_numpy(retained_components).to(device).to(torch.float32)

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
        pc_i_timeseries = pub_pc_projected[:,i]
        corr_i = np.corrcoef(pc_i_timeseries, train_prv_flat.squeeze())[0,1]
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
    sanitized_projections_for_training = sanitized_projections_for_training.reshape(-1, sequence_length, sanitized_feature_nums)
    sanitized_projections_for_training = torch.from_numpy(sanitized_projections_for_training).to(torch.float32).to(device)
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
    opt_reconstructor = torch.optim.Adam(reconstructor.parameters(), lr=lr) # type: ignore
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr) # type: ignore

    ############################################################
    # Training Based on the Sanitized PCA Components
    ############################################################
    reconstruction_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_inp = sanitized_projections_for_training[batch_no * batch_size : (batch_no + 1) * batch_size]
            prv_feats = train_prv_tensor[batch_no * batch_size : (batch_no + 1) * batch_size, :, ]
            pub_feats = train_pub_tensor[batch_no * batch_size : (batch_no + 1) * batch_size, :, ]

            reconstructor_guess = reconstructor(batch_inp).squeeze()
            adversary_guess = adversary(batch_inp).squeeze()

            # Calculate the loss
            recon_loss = criterion(reconstructor_guess, pub_feats[:,-1])
            adv_loss = criterion(adversary_guess, prv_feats[:,-1])
            adversary.zero_grad()
            reconstructor.zero_grad()
            recon_loss.backward()
            adv_loss.backward()
            opt_reconstructor.step()
            opt_adversary.step()

            reconstruction_losses.append(recon_loss.item())
            adv_losses.append(adv_loss.item())

            if wandb_on:
                wandb.log({
                    "pca_recon_train_loss": recon_loss.item(),
                    "pca_adv_train_loss": adv_loss.item(),
                })

    # Plot reconstruction losses
    plt.figure(figsize=(16,10))
    plt.plot(reconstruction_losses)
    plt.title("Reconstruction Loss")
    plt.savefig(f"./figures/pca_recon_losses.png")
    plt.close()

    return adversary, torch.from_numpy(retained_components), all_pc_corr_scores

def pca_preprocessing(
    train_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Will preprocess the data for PCA
    args:
        - train_all: All data (public + private)
    returns:
        - pca_components: Principal Components
        - pca_projected_ds: Projected Public data onto the principal components
    """ 

    pca = PCA()
    train_all_flat = train_all.reshape(-1, train_all.shape[-1])
    train_all_centered = train_all_flat - np.mean(train_all_flat, axis=0)

    ########################################
    # PCA Fitting
    ########################################
    pca_transform = pca.fit(train_all_centered)
    # Shape is (num_components, num_features), fyi for indexing purposes
    principal_components = pca_transform.components_
    # Takes in (num_componets, num_features) and returns (num_components, num_samples)
    pca_projected_ds = train_all_centered.dot(principal_components.T)

    return principal_components, pca_projected_ds

def pca_decomposition_w_heatmap(
    pca_components: np.ndarray,
    pca_projected_ds: np.ndarray,
    train_prv: np.ndarray,
) -> np.ndarray:
    """
    This will create a matrix M_{up} that will help us convert public components into private components.
    returns: M_UI
    """
    train_prv_flat = train_prv.reshape(-1, train_prv.shape[-1])
    C = pca_projected_ds

    # The projection matrix from public to latent space
    M_ul = pca_components.T # (Num latent components, components size / num features)

    # ----- Step 2: Solve for M_CU via linear regression -----
    # We assume X_U \approx C @ M_CU
    # M_CU has shape (K, N-M)
    # Using np.linalg.lstsq to solve: M_CU = argmin ||C @ M - X_U||^2
    M_li, residuals, rank, s = np.linalg.lstsq(C, train_prv_flat, rcond=None)

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
    M_UI = M_ul @ M_li  # shape: (M, N-M) i.e. (num_public_components, num_private_components)

    return  M_UI


def plot_heatmap_and_correlation(private_guess, test_prv, M_PU, all_pc_corr_scores):
    # Now we plot the results
    plt.figure(figsize=(16,10))
    plt.plot(private_guess.squeeze(), label="Reconstruction")
    plt.plot(test_prv.squeeze(), label="Truth")
    plt.title("PCA Reconstruction vs Truth")
    plt.legend()
    plt.savefig("./figures/mcu_reconstruction.png")
    plt.close()
    ########################################
    # Plot the HeatMap
    ########################################
    fig, axs = plt.subplots(2, 1, figsize=(16,8))
    
    # Upper subplot: Heatmap
    im = axs[0].matshow(M_PU.T, cmap='viridis')
    axs[0].set_title("$M_{PU}$: Public to Private Features HeatMap")
    axs[0].set_xlabel("Public Features")
    axs[0].set_ylabel("Private Features")
    axs[0].set_yticks([])
    # axs[0].set_xticks(np.arange(0, len(M_PU.T), 1) + 1)
    plt.colorbar(im, ax=axs[0])
    
    # Lower subplot: Bar plot
    axs[1].bar(range(len(all_pc_corr_scores)), all_pc_corr_scores)
    axs[1].set_title("Correlation Scores")
    axs[1].set_xlabel("Public Features")
    axs[1].set_ylabel("Correlation Score")
    axs[1].set_xticks(np.arange(0, len(all_pc_corr_scores), 1))
    
    plt.tight_layout()
    plt.savefig("./figures/pca_heatmap.png")
    plt.close()

def method_1_latent_spc_deco(
    num_pca_components: int,
    axs: Axes,
    pca_components: np.ndarray,
    pub_pc_projected: np.ndarray,
    test_file: np.ndarray,
    all_train_seqs: np.ndarray,
    prv_features_idxs: Sequence[int],
    args: argparse.Namespace,
    logger: logging.Logger,
):
    # Method 1. Latent Space Decorrelatio
    pca_reconstructor, pca_model_adversary, retained_components, all_pc_corr_scores = baseline_pca_decorr_adversary_by_pc(
        pca_components,
        pub_pc_projected,
        all_train_seqs,
        prv_features_idxs,
        args.batch_size,
        num_pca_components,
        args.epochs,
        args.lr,
        args.device,
        args.wandb_on,
    )
    print(num_pca_components)
    print(retained_components.shape)
    # Really quick use the salient heatmap to recover private features
    # private_guess = test_pub.squeeze().dot(M_PU)

    pca_test_entire_file(
        axs,
        test_file,
        args.cols_to_hide,
        pca_reconstructor,
        pca_model_adversary,
        retained_components,
        args.episode_length,
        None,
        logger,
        args.batch_size,
        wandb_on=args.wandb_on
    )

def method_2_MUI_decorrelation(
    num_features_to_keep: int,
    test_file: np.ndarray,
    test_prv: np.ndarray,
    M_UI: np.ndarray,
):
    """
    Will grab the already pretrained/conditioned M_ui and slowly test the removal of some columns until we get zero utility. Out of it
    """
    assert (
        M_UI.shape[1] == 1
    ), f"Currently only working with a single private component. Instead we got {M_UI.shape}"
    max_corr_idxs = np.argsort(np.abs(M_UI.squeeze()))[:num_features_to_keep]

    M_UI_pruned = M_UI[max_corr_idxs, :]
    test_file_flattend = test_file.reshape(-1, test_file.shape[-1])
    test_file_pruned = test_file_flattend[:,max_corr_idxs]
    test_prv_flattend = test_prv.reshape(-1, test_prv.shape[-1])

    recovered_private_guess = test_file_pruned.dot(M_UI_pruned)

    # DEBUG: FOr now we plot it like this
    plt.figure(figsize=(16,10))
    plt.plot(recovered_private_guess.squeeze(), label="Reconstruction")
    plt.plot(test_prv_flattend, label="Truth")
    plt.title("PCA Reconstruction vs Truth")
    plt.legend()
    plt.savefig(f"./figures/pca/heatmap_mui_reconstruction_{num_features_to_keep}.png")
    plt.close()

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
    columns, runs_dict, debug_file = load_defacto_data(args.defacto_data_raw_path)
    train_seqs, val_seqs, test_file = split_defacto_runs(
        runs_dict,
        args.splits["train_split"],
        args.splits["val_split"],
        args.episode_length,
        args.oversample_coefficient,
        True, # Scale
    )
    all_train_seqs = np.concatenate([ seqs for _, seqs in train_seqs.items()], axis=0)
    num_features = all_train_seqs.shape[-1]
    num_pca_components = num_features
    prv_features_idxs = args.cols_to_hide
    pub_features_idxs = list(set(range(num_features)) - set(prv_features_idxs))
    print(f"Public features are {pub_features_idxs}")
    print(f"Private features are {prv_features_idxs}")
    train_pub = all_train_seqs[:,:,pub_features_idxs]
    train_prv = all_train_seqs[:,:,prv_features_idxs]

    test_pub = test_file[:,pub_features_idxs]
    test_prv = test_file[:,prv_features_idxs]

    ########################################
    # PCA Preprocessing
    ########################################
    logger.info("Starting the PCA preprocessing")
    pca_components, pca_projected_ds = pca_preprocessing(
        all_train_seqs,
    )

    ########################################
    # Get M_UI for the MUI method
    ########################################
    M_UI = pca_decomposition_w_heatmap(
        pca_components,
        pca_projected_ds,
        train_prv,
    )

    ########################################
    # Figure Initialization
    #   We will initalize most figure stuff
    #   here so we can pass them as parameters
    #   to functions and keeep them separate
    ########################################
    fig, axs = plt.subplots(1,1,figsize=(16,8))
    axs.set_title("Original Data")
    axs.set_xlabel("Privacy")
    axs.set_ylabel("Utility")

    ########################################
    # Training the PCA based baseline
    ########################################
    # TOREM: Move this lower down for when we are done with it. 
    logger.info("Starting the PCA decorrelation and training")
    for num_comp in range(num_pca_components): 
        # Method 1. Latent Space Decorrelatio
        method_1_latent_spc_deco(
            num_comp,
            axs,
            pca_components,
            pca_projected_ds,
            test_file,
            all_train_seqs,
            prv_features_idxs,
            args,
            logger,
        )
        method_2_MUI_decorrelation(
            num_comp,
            test_file,
            test_prv,
            M_UI,
        )

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
    logger.info("Done with the PCA test")

    
if __name__ == "__main__":
    logger = create_logger("pca_benchmarks")
    args = argsies()
    main(args)

