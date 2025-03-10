import argparse
from math import ceil
from typing import List, OrderedDict

import debugpy
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
from conrecon.utils import create_logger, set_seeds


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

def baseline_pca_decorrelation(
    ds_train: OrderedDict[str, np.ndarray],
    ds_val: OrderedDict[str, np.ndarray],
    prv_features_idxs: list[int],
    batch_size: int,
    correlation_threshold: float,
    epochs: int,
    lr: float,
    device: torch.device,
    wandb_on: bool,
) -> tuple[nn.Module, torch.Tensor, np.ndarray]:
    pca = PCA()

    ########################################
    # Bunch of data processing.
    ########################################
    first_key = list(ds_train.keys())[0]
    num_features = ds_train[first_key].shape[-1]
    pub_features_idxs = list(set(range(num_features)) - set(prv_features_idxs))
    sequence_length = ds_train[first_key].shape[1]

    # We need to concatenate the features as usual 
    all_train_seqs_np = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs_np = np.concatenate([ seqs for _, seqs in ds_val.items()], axis=0)
    train_pub_np = all_train_seqs_np[:,:,pub_features_idxs]
    train_prv_np = all_train_seqs_np[:,:,prv_features_idxs].squeeze()
    # Shuffle, Batch, Torch Coversion, Feature Separation
    
    all_train_seqs_tensor = torch.from_numpy(all_train_seqs_np).to(torch.float32).to(device)

    train_pub_flat = train_pub_np.reshape(-1, train_pub_np.shape[-1])
    train_pub_centered = train_pub_flat - np.mean(train_pub_flat, axis=0)
    train_prv_flat = train_prv_np.reshape(-1)

    ########################################
    # PCA Fitting
    ########################################
    pca_transform = pca.fit(train_pub_flat)
    # Shape is (num_components, num_features), fyi for indexing purposes
    principal_components = pca_transform.components_
    # Takes in (num_componets, num_features) and returns (num_components, num_samples)
    pub_pc_projected = train_pub_centered.dot(principal_components.T)

    # Plot principal components as matrix for debugging and save them in figures/pca_components
    fig, axs = plt.subplots(1, 1, figsize=(16,4))
    im = axs.matshow(principal_components.T, cmap='viridis')
    axs.set_title("$M_{PU}$: Public to Private Features HeatMap")
    axs.set_xlabel("Public Features")
    axs.set_ylabel("Private Features")
    axs.set_yticks([])
    axs.set_xticks(np.arange(0, len(principal_components.T), 1) + 1)
    plt.colorbar(im)
    plt.savefig("./figures/pca_components.png")
    plt.close()


    ########################################
    # Check on correlations
    ########################################
    retained_components = []
    all_pc_corr_scores = []
    inspect_array("Before", train_prv_flat)
    for i in range(principal_components.shape[0]):
        pc_i_timeseries = pub_pc_projected[:,i]
        corr_i = np.corrcoef(pc_i_timeseries, train_prv_flat)[0,1]
        all_pc_corr_scores.append(corr_i)
        # ensure corr_i is not nan
        assert not np.isnan(corr_i)
        if abs(corr_i) <= correlation_threshold:

            retained_components.append(principal_components[i])

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
    adversary = PCATemporalAdversary(
        num_principal_components=retained_components.shape[0],
        num_features_to_recon=num_features,
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr) # type: ignore

    ############################################################
    # Training Based on the Sanitized PCA Components
    ############################################################
    reconstruction_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_pub = sanitized_projections_for_training[batch_no * batch_size : (batch_no + 1) * batch_size]
            all_feats = all_train_seqs_tensor[batch_no * batch_size : (batch_no + 1) * batch_size]

            adversary_guess = adversary(batch_pub).squeeze()

            # Calculate the loss
            loss = criterion(adversary_guess, all_feats[:,-1])
            adversary.zero_grad()
            loss.backward()
            opt_adversary.step()
            reconstruction_losses.append(loss.item())

            if wandb_on:
                wandb.log({
                    "adv_train_loss": loss.item(),
                })

    # Plot reconstruction losses
    plt.figure(figsize=(16,10))
    plt.plot(reconstruction_losses)
    plt.title("Reconstruction Loss")
    plt.savefig(f"./figures/pca_recon_losses.png")
    plt.close()

    return adversary, torch.from_numpy(retained_components), all_pc_corr_scores

def pca_decomposition_w_heatmap(
    ds_train: OrderedDict[str, np.ndarray],
    test_file: np.ndarray,
    prv_features_idxs: List[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This will create a matrix M_{up} that will help us convert public components into private components.
    returns: private_guess, test_prv, M_PU
    """
    pca = PCA()

    first_key = list(ds_train.keys())[0]
    num_features = ds_train[first_key].shape[-1]
    pub_features_idxs = list(set(range(num_features)) - set(prv_features_idxs))
    sequence_length = ds_train[first_key].shape[1]

    # We need to concatenate the features as usual 
    all_train_seqs = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    
    train_pub = all_train_seqs[:,:,pub_features_idxs]
    train_prv = all_train_seqs[:,:,prv_features_idxs]
    train_pub_flat = train_pub.reshape(-1, train_pub.shape[-1])
    train_prv_flat = train_prv.reshape(-1, train_prv.shape[-1])
    train_pub_centered = train_pub_flat - np.mean(train_pub_flat, axis=0)
    train_pub_mean = np.mean(train_pub_flat, axis=0)

    #  Now we can start fitting the pca 
    pca_transform = pca.fit(train_pub_centered)
    # Shape is (num_components, num_features), fyi for indexing purposes
    principal_components = pca_transform.components_
    # Takes in (num_componets, num_features) and returns (num_components, num_samples)
    C = train_pub_centered.dot(principal_components.T)

    # The projection matrix from public to latent space
    M_CP = pca.components_.T  # shape: (M, K)

    # ----- Step 2: Solve for M_CU via linear regression -----
    # We assume X_U \approx C @ M_CU
    # M_CU has shape (K, N-M)
    # Using np.linalg.lstsq to solve: M_CU = argmin ||C @ M - X_U||^2
    M_CU, residuals, rank, s = np.linalg.lstsq(C, train_prv_flat, rcond=None)

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
    M_PU = M_CP @ M_CU  # shape: (M, N-M)

    print("M_CP shape:", M_CP.shape)
    print("M_CU shape:", M_CU.shape)
    print("M_PU shape:", M_PU.shape)

    ########################################
    # Run Evaluation on test set
    ########################################
    # Now for validation
    test_pub = test_file[:,pub_features_idxs]
    test_prv = test_file[:,prv_features_idxs]
    test_pub_centered = test_pub # - train_pub_mean
    private_guess = test_pub_centered.dot(M_PU)

    return private_guess, test_prv, M_PU

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

    ########################################
    # PCA Decomposition with heatmap
    ########################################
    logger.info("Creating the decomposition with heatmap")
    private_guess, test_prv, M_PU = pca_decomposition_w_heatmap(
        train_seqs,
        test_file,
        args.cols_to_hide,
    )
    ########################################
    # Training the PCA based baseline
    ########################################
    # TOREM: Move this lower down for when we are done with it. 
    logger.info("Starting the PCA decorrelation and training")
    pca_model_adversary, retained_components, all_pc_corr_scores = baseline_pca_decorrelation(
        train_seqs,
        val_seqs,
        args.cols_to_hide,
        args.batch_size,
        args.correlation_threshold,
        args.epochs,
        args.lr,
        device,
        args.wandb_on
    )
    plot_heatmap_and_correlation(private_guess, test_prv, M_PU, all_pc_corr_scores)

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
    logger = create_logger("main_vae")
    args = argsies()
    main(args)

