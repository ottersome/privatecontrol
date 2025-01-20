from typing import Sequence, Dict
from math import ceil
import logging

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from conrecon.data.dataset_generation import batch_generation_randUni, collect_n_sequential_batches, spot_backhistory

def triv_test_entire_file(
    validation_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_adversary: nn.Module,
    sequence_length: int,
    padding_value: int,
    batch_size: int = 16,
    wandb_on: bool = False,
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    amnt_columns = validation_file.shape[-1]
    pub_features_idxs = list(set(range(amnt_columns)) - set(idxs_colsToGuess))

    model_adversary.eval()
    device = next(model_adversary.parameters()).device
    val_x = torch.from_numpy(validation_file).to(torch.float32).to(device)

    # Generate the reconstruction
    public_columns = list(set(range(val_x.shape[-1])) - set(idxs_colsToGuess))
    private_columns = list(idxs_colsToGuess)
    num_columns = len(public_columns) + len(private_columns)
    num_batches = ceil(len(val_x) / batch_size)

    batch_guesses = []
    batch_reconstructions = []
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        start_idx = batch_no * batch_size
        end_idx = min((batch_no + 1) * batch_size, val_x.shape[0])
        backhistory = collect_n_sequential_batches(val_x.cpu().numpy(), start_idx, end_idx, sequence_length, padding_value)
        backhistory = torch.from_numpy(backhistory).to(torch.float32).to(device)
        backhistory_pub = backhistory[:,:,pub_features_idxs]

        # TODO: Incorporate Adversary Guess
        adversary_guess = model_adversary(backhistory_pub)
        batch_guesses.append(adversary_guess)

    seq_guesses = torch.cat(batch_guesses, dim=0)

    ########################################
    # Chart for Adversary
    ########################################
    adv_to_show = seq_guesses[:, :]
    adv_truth = validation_file[:, private_columns]
    
    fig = plt.figure(figsize=(16,10))
    plt.plot(adv_to_show.squeeze().detach().cpu().numpy(), label="Adversary")
    plt.title("Adversary")
    plt.legend()
    plt.plot(adv_truth.squeeze(), label="Truth")
    plt.title("Truth")
    plt.legend()

    plt.savefig(f"figures/triv_adversary.png")
    plt.close()

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})

    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}

def pca_test_entire_file(
    test_file: np.ndarray,
    prv_features_idxs: Sequence[int],
    model_adversary: nn.Module,
    principal_components: np.ndarray,
    sequence_length: int,
    padding_value: int,
    logger: logging.Logger,
    batch_size: int = 16,
    wandb_on: bool = False,
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    amnt_columns = test_file.shape[-1]

    model_adversary.eval()
    device = next(model_adversary.parameters()).device
    val_x = torch.from_numpy(test_file).to(torch.float32).to(device)
    pub_features_idxs = list(set(range(amnt_columns)) - set(prv_features_idxs))

    num_batches = ceil(len(val_x) / batch_size)

    batch_guesses = []
    batch_reconstructions = []
    logger.info("About to start the test")
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        logger.info(f"Batch {batch_no} out of {num_batches}")
        start_idx = batch_no * batch_size
        end_idx = min((batch_no + 1) * batch_size, val_x.shape[0])
        backhistory = collect_n_sequential_batches(val_x.cpu().numpy(), start_idx, end_idx, sequence_length, padding_value)[:,:,pub_features_idxs]
        projected_backhistory = backhistory.dot(principal_components.T)
        projected_backhistory = torch.from_numpy(projected_backhistory).to(torch.float32).to(device)

        # TODO: Incorporate Adversary Guess
        with torch.no_grad():
            adversary_guess = model_adversary(projected_backhistory)
        batch_guesses.append(adversary_guess)

    seq_guesses = torch.cat(batch_guesses, dim=0)
    ########################################
    # Chart for Reconstruction
    ########################################
    logger.info("PCA Reconstruction Graph")
    recon_to_show = seq_guesses[:, :]
    truth_to_compare = test_file[:, prv_features_idxs]
    fig,axs = plt.subplots(4,2,figsize=(32,20))
    for i in range(recon_to_show.shape[1]):
        mod = i % 4
        idx = i // 4
        axs[mod,idx].plot(recon_to_show[:,i].squeeze().detach().cpu().numpy(), label="Reconstruction")
        axs[mod,idx].set_title("Reconstruction Vs Truth")
        axs[mod,idx].legend()
        axs[mod,idx].plot(truth_to_compare[:,i].squeeze(), label="Truth")
        axs[mod,idx].legend()
        if wandb_on:
            wandb.log({f"Reconstruction (Col {i})": recon_to_show[:,i].squeeze().detach().cpu().numpy()})
            wandb.log({f"Truth (Col {i})": truth_to_compare[:,i].squeeze().detach().cpu().numpy()})
    plt.savefig(f"figures/pca_reconstruction.png")
    logger.info(f"Saved the reconstruction figure to {f'figures/pca_reconstruction.png'}")
    plt.close()

    ########################################
    # Chart for Adversary
    ########################################
    adv_to_show = seq_guesses[:, :]
    adv_truth = test_file[:, prv_features_idxs]
    
    logger.info("Creating PCA Adversary Graph")
    fig = plt.figure(figsize=(16,10))
    plt.plot(adv_to_show.squeeze().detach().cpu().numpy(), label="Adversary")
    plt.title("Adversary")
    plt.legend()
    plt.plot(adv_truth.squeeze(), label="Truth")
    plt.title("Truth")
    plt.legend()

    plt.savefig(f"figures/pca_adversary.png")
    plt.close()
    logger.info(f"Saved the adversary figure to {f'figures/pca_adversary.png'}")

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})

    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}

def test_entire_file(
    test_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
    sequence_length: int,
    padding_value: int,
    batch_size: int = 16,
    wandb_on: bool = False,
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }

    model_vae.eval()
    model_adversary.eval()
    device = next(model_vae.parameters()).device

    model_device = next(model_vae.parameters()).device
    val_x = torch.from_numpy(test_file).to(torch.float32).to(model_device)

    # Generate the reconstruction
    public_columns = list(set(range(val_x.shape[-1])) - set(idxs_colsToGuess))
    private_columns = list(idxs_colsToGuess)
    num_columns = len(public_columns) + len(private_columns)
    num_batches = ceil(len(val_x) / batch_size)

    batch_guesses = []
    batch_reconstructions = []
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        start_idx = batch_no * batch_size
        end_idx = min((batch_no + 1) * batch_size, val_x.shape[0])
        backhistory = collect_n_sequential_batches(val_x.cpu().numpy(), start_idx, end_idx, sequence_length, padding_value)
        backhistory = torch.from_numpy(backhistory).to(torch.float32).to(device)
        latent_z, sanitized_data, kl_divergence = model_vae(backhistory)

        # TODO: Incorporate Adversary Guess
        adversary_guess = model_adversary(latent_z)
        batch_guesses.append(adversary_guess)
        batch_reconstructions.append(sanitized_data)

    seq_guesses = torch.cat(batch_guesses, dim=0)
    seq_reconstructions = torch.cat(batch_reconstructions, dim=0)

    # Lets now save the figure
    some_8_idxs = np.random.randint(0, seq_reconstructions.shape[1], 8)

    ########################################
    # Chart For Reconstruction
    ########################################
    recon_to_show = seq_reconstructions[:, some_8_idxs]
    truth_to_compare = test_file[:, some_8_idxs]
    fig,axs = plt.subplots(4,2,figsize=(32,20))
    for i in range(recon_to_show.shape[1]):
        mod = i % 4
        idx = i // 4
        axs[mod,idx].plot(recon_to_show[:,i].squeeze().detach().cpu().numpy(), label="Reconstruction")
        axs[mod,idx].set_title("Reconstruction Vs Truth")
        axs[mod,idx].legend()
        axs[mod,idx].plot(truth_to_compare[:,i].squeeze(), label="Truth")
        axs[mod,idx].legend()
        if wandb_on:
            wandb.log({f"Reconstruction (Col {i})": recon_to_show[:,i].squeeze().detach().cpu().numpy()})
            wandb.log({f"Truth (Col {i})": truth_to_compare[:,i].squeeze().detach().cpu().numpy()})
    plt.savefig(f"figures/vae_reconstruction.png")
    plt.close()

    ########################################
    # Chart for Adversary
    ########################################
    adv_to_show = seq_guesses[:, :]
    adv_truth = test_file[:, private_columns]
    
    fig = plt.figure(figsize=(16,10))
    plt.plot(adv_to_show.squeeze().detach().cpu().numpy(), label="Adversary")
    plt.title("Adversary")
    plt.legend()
    plt.plot(adv_truth.squeeze(), label="Truth")
    plt.title("Truth")
    plt.legend()

    plt.savefig(f"figures/vae_adversary.png")
    plt.close()

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"reconstruction": recon_to_show.squeeze().detach().cpu().numpy()})
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})

    model_vae.train()
    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}
