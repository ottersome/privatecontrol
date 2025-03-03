import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple
import logging
from math import ceil

import numpy as np

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import (
    Adversary,
    TrivialTemporalAdversary,
    PCATemporalAdversary,
)
from conrecon.dplearning.vae import SequenceToScalarVAE, SequenceToScalarVAE
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger, set_seeds
from conrecon.validation_functions import calculate_validation_metrics
from conrecon.performance_test_functions import (
    vae_test_file,
    triv_test_entire_file,
    pca_test_entire_file,
)


def train_vae_and_adversary(
    batch_size: int,
    priv_columns: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model_vae: SequenceToScalarVAE,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
    wandb_on: bool,
    logger: logging.Logger,
    priv_utility_tradeoff_coeff: float,
) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Training Loop
        ds_train: np.ndarray (num_batches, batch_size, features),
    """

    pub_columns = list(set(range(len(data_columns))) - set(priv_columns))
    device = next(model_vae.parameters()).device

    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=learning_rate)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)  # type: ignore

    ##  A bit of extra processing of data. (Specific to this version of training)
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    batch_amnt = all_train_seqs.shape[0] // batch_size
    # all_train_seqs = all_train_seqs.reshape(batch_amnt, batch_size, all_train_seqs.shape[-2], all_train_seqs.shape[-1])
    # all_valid_seqs = all_valid_seqs.reshape(batch_amnt, batch_size, all_valid_seqs.shape[-2], all_valid_seqs.shape[-1])
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:, :, pub_columns]
    train_prv = all_train_seqs[:, :, priv_columns]

    num_batches = ceil(all_train_seqs.shape[0] / batch_size)
    sequence_len = all_train_seqs.shape[1]

    ########################################
    # Get Batches
    ########################################
    logger.info(
        f"Working with {num_batches} num_batches, each with size:  {batch_size}, and sequence/episode length {sequence_len}"
    )
    recon_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_all = all_train_seqs[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            # logger.info(
            #     f"Batch {batch_no} out of {num_batches} with shape {batch_pub.shape}"
            # )

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, _ = model_vae(batch_all)

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z).flatten()

            # Check on performance
            batch_y_flat = (
                batch_prv[:, -1, :].view(-1, batch_prv.shape[-1]).flatten()
            )  # Grab only last in sequeence
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            model_adversary.zero_grad()
            adv_train_loss.backward()
            opt_adversary.step()

            ########################################
            # 2. Calculate the Recon Loss
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all)
            pub_prediction = batch_pub[:, -1, :]

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_prv[:, -1, :].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data, pub_prediction)
            adver_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = (
                pub_recon_loss - priv_utility_tradeoff_coeff * adver_loss + kl_dig_hypr * kl_divergence.mean()
            )

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adver_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            opt_vae.step()

            if wandb_on:
                wandb.log(
                    {
                        "adv_train_loss": adv_train_loss.item(),
                        "pub_recon_loss": pub_recon_loss.item(),
                        "final_loss_scalar": final_loss_scalar.item(),
                    }
                )

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                model_vae.eval()
                model_adversary.eval()
                with torch.no_grad():
                    validation_metrics = calculate_validation_metrics(
                        all_valid_seqs,
                        pub_columns,
                        priv_columns,
                        model_vae,
                        model_adversary,
                    )
                model_vae.train()
                model_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)

    return model_vae, model_adversary, recon_losses, adv_losses

def train_adversary(
    model_vae: nn.Module,
    model_adversary: nn.Module,
    opt_adversary: torch.optim.Optimizer, # type: ignore
    epoch_sample_percent: float,
    global_samples: torch.Tensor,
    num_cols: int,
    prv_cols: List[int],
    batch_size: int,
) -> nn.Module:
    pub_cols = list(set(range(num_cols)) - set(prv_cols))

    num_subsamples = ceil(epoch_sample_percent * global_samples.shape[0])
    # TODO: test this out
    device = next(model_vae.parameters()).device
    random_indices = torch.randint(0, global_samples.shape[0], (num_subsamples,), dtype=torch.long).to(device)
    subsamples = torch.index_select(global_samples, 0, random_indices)

    train_pub = subsamples[:, :, pub_cols]
    train_prv = subsamples[:, :, prv_cols]

    num_batches = ceil(global_samples.shape[0] / batch_size)

    for batch_no in range(num_batches):
        ########################################
        # Data Preparation
        ########################################
        batch_all = subsamples[
            batch_no * batch_size : (batch_no + 1) * batch_size
        ]
        batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
        batch_privTrue_flat = (
            batch_prv[:, -1, :].view(-1, batch_prv.shape[-1]).flatten()
        )  # Grab only last in sequeence
        # WARNING: Check on that -1 seems sus.
        with torch.no_grad():
             latent_z, sanitized_data, _ = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence
        adversary_guess_flat = model_adversary(latent_z).flatten()
        # WARNING: Check on that -1 seems sus.
        batch_privTrue_flat = batch_prv[:, -1, :].view(-1, batch_prv.shape[-1])

        ########################################
        # Gradient Calculation
        ########################################
        adv_train_loss = F.mse_loss(adversary_guess_flat, batch_privTrue_flat)
        model_adversary.zero_grad()
        adv_train_loss.backward()
        opt_adversary.step()

    return model_adversary


def train_vae_and_adversary_bi_level(
    batch_size: int,
    priv_columns: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    inner_epochs: int,  # These are for the adversary
    epoch_sample_percent: float,
    model_vae: nn.Module,
    model_adversary: nn.Module,
    learning_rate: float,
    kl_dig_hypr: float,
    wandb_on: bool,
    logger: logging.Logger,
    priv_utility_tradeoff_coeff: float,
) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Training Loop
        ds_train: np.ndarray (num_batches, batch_size, features),
    """

    total_num_features = len(data_columns)
    pub_columns = list(set(range(total_num_features)) - set(priv_columns))
    device = next(model_vae.parameters()).device

    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=learning_rate)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)  # type: ignore

    ##  A bit of extra processing of data. (Specific to this version of training)
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    batch_amnt = all_train_seqs.shape[0] // batch_size
    # all_train_seqs = all_train_seqs.reshape(batch_amnt, batch_size, all_train_seqs.shape[-2], all_train_seqs.shape[-1])
    # all_valid_seqs = all_valid_seqs.reshape(batch_amnt, batch_size, all_valid_seqs.shape[-2], all_valid_seqs.shape[-1])
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:, :, pub_columns]
    train_prv = all_train_seqs[:, :, priv_columns]

    num_batches = ceil(all_train_seqs.shape[0] / batch_size)
    sequence_len = all_train_seqs.shape[1]

    ########################################
    # Get Batches
    ########################################
    logger.info(
        f"Working with {num_batches} num_batches, each with size:  {batch_size}, and sequence/episode length {sequence_len}"
    )
    recon_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):

            # Now Get the new VAE generations
            batch_all = all_train_seqs[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            model_adversary = train_adversary(
                model_vae,
                model_adversary,
                opt_adversary,
                inner_epochs,
                epoch_sample_percent,
                batch_all,
                total_num_features,
                priv_columns,
                batch_size,
            )

            # Now we train the autoencoder to try to fool the adversary
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_prv[:, -1, :].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data, batch_pub[:, -1, :]) # Guesss the set of features of sequence
            adver_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = (
                pub_recon_loss - priv_utility_tradeoff_coeff * adver_loss + kl_dig_hypr * kl_divergence.mean()
            )

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adver_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            opt_vae.step()

            if wandb_on:
                wandb.log(
                    {
                        "adv_train_loss": adv_train_loss.item(),
                        "pub_recon_loss": pub_recon_loss.item(),
                        "final_loss_scalar": final_loss_scalar.item(),
                    }
                )

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                model_vae.eval()
                model_adversary.eval()
                with torch.no_grad():
                    validation_metrics = calculate_validation_metrics(
                        all_valid_seqs,
                        pub_columns,
                        priv_columns,
                        model_vae,
                        model_adversary,
                    )
                model_vae.train()
                model_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)

    return model_vae, model_adversary, recon_losses, adv_losses
