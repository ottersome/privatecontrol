import os
import time
import torch
import debugpy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple
import logging
from math import ceil, exp

import numpy as np
import matplotlib.pyplot as plt

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import (
    Adversary,
    TrivialTemporalAdversary,
    PCATemporalAdversary,
)
from conrecon.dplearning.vae import SequenceToOneVectorSimple, SequenceToOneVectorGeneral
from conrecon.plotting import TrainLayout
from conrecon.utils.common import create_logger, deprecated
from conrecon.validation_functions import calculate_validation_metrics
from conrecon.performance_test_functions import (
    advVae_test_file,
    triv_test_entire_file,
    pca_test_entire_file,
)
from conrecon.validation_utils import validate_vae_recon

logger = create_logger("training_utils")

def train_vae_and_adversary(
    batch_size: int,
    priv_columns: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model_vae: SequenceToOneVectorSimple,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
    wandb_on: bool,
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
            with torch.no_grad():
                adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_prv[:, -1, :].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data, pub_prediction)
            adver_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = (
                # pub_recon_loss - (1/priv_utility_tradeoff_coeff) * adver_loss + kl_dig_hypr * kl_divergence.mean()
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
    epochs: int,
    epoch_sample_percent: float,
    global_samples: torch.Tensor,
    prv_cols: List[int],
    batch_size: int,
) -> nn.Module:

    model_vae.eval()
    num_subsamples = min(ceil(epoch_sample_percent * global_samples.shape[0]), global_samples.shape[0])
    # TODO: test this out
    device = next(model_vae.parameters()).device
    num_batches = ceil(global_samples.shape[0] / batch_size)

    debug_losses = []
    for e in range(epochs):
        random_indices = torch.randperm(global_samples.shape[0])[:num_subsamples].to(device)
        subsamples = torch.index_select(global_samples, 0, random_indices)
        y_priv = subsamples[:, -1, prv_cols]

        for batch_no in range(num_batches):
            ########################################
            # Data Preparation
            ########################################
            batch_all = subsamples[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            batch_prv = y_priv[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_privTrue_flat = ( batch_prv.view(-1, batch_prv.shape[-1]).flatten())  
            logger.debug(f"batch_privTrue_flat.is_nan? = {torch.isnan(batch_privTrue_flat).any()}")
            # WARNING: Check on that -1 seems sus.
            with torch.no_grad():
                latent_z, sanitized_data, _ = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence
            logger.debug(f"latent_z.is_nan? = {torch.isnan(latent_z).any()}")
            adversary_guess_flat = model_adversary(latent_z).flatten()
            logger.debug(f"adversary_guess_flat.is_nan? = {torch.isnan(adversary_guess_flat).any()}")
            # WARNING: Check on that -1 seems sus.

            ########################################
            # Gradient Calculation
            ########################################
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_privTrue_flat)
            logger.debug(f"adversary_guess_flat.is_nan? = {torch.isnan(adversary_guess_flat).any()}")
            logger.debug(f"adv_train_loss.is_nan? = {torch.isnan(adv_train_loss).any()}")
            if torch.isnan(adv_train_loss).any():
                debugpy.breakpoint()
            logger.debug(f"---------------------")
            debug_losses.append(adv_train_loss.item()) 
            model_adversary.zero_grad()
            adv_train_loss.backward()
            opt_adversary.step()

    # Temporal Figure making for debug
    plt.plot(debug_losses)
    logger.debug(f"Plotting the adversary losses")
    logger.debug(f"Adversary Losses are {debug_losses}")
    plt.savefig(f"./figures/method_vae/inner_adv_losses.png")
    plt.close()

    model_vae.train()
    return model_adversary

def simple_vae_reconstruction_training(
    batch_size: int,
    priv_columns: list[int], # Mostly so we know what to ignore.
    all_train_seqs: torch.Tensor,
    all_validation_seqs: torch.Tensor,
    epochs: int,
    model_vae: nn.Module,
    optimizer_vae: torch.optim.Optimizer,# type: ignore
    kl_dig_hypr: float, 
    wandb_on: bool, 
    validate_every_n_batches: Optional[int] = None,
) -> tuple[nn.Module, list[float]]:
    """
    The point of this funciton is to be able to test the VAE model independent of the adversary.
    So that we can focus on getting the best performance out of reconstruction
    """
    # Some Helper Variables
    pub_columns = list(set(range(all_train_seqs.shape[-1])) - set(priv_columns))

    num_seqs_as_samples = all_train_seqs.shape[0]
    recon_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        batch_steps = np.arange(0, num_seqs_as_samples, batch_size)
        recon_losses = []
        for _batch_offset in batch_steps:
            ########################################
            # Data Collection
            ########################################
            batch_end = min(_batch_offset + batch_size, num_seqs_as_samples)
            batch_all = all_train_seqs[_batch_offset: batch_end, :, :]
            batch_pub = all_train_seqs[_batch_offset: batch_end, :, pub_columns]

            ########################################
            # Actual Training
            ########################################
            # Now we train the autoencoder to try to fool the adversary
            _, sanitized_data, kl_divergence = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence

            # Check on performance
            pub_recon_loss = F.mse_loss(sanitized_data, batch_pub[:, -1, :], reduction="none").mean(-1) # Guesss the set of features of sequence
            # pub_recon_norm = pub_recon_loss / (pub_recon_loss.detach() + 1e-8) TEST: Just trying normaliztion
            final_loss_scalar = (
                pub_recon_loss + kl_dig_hypr * kl_divergence # Just good ol reconstruciton to see if it works
            ).mean()

            optimizer_vae.zero_grad()
            final_loss_scalar.backward()
            optimizer_vae.step()

            if wandb_on: 
                wandb.log({
                    "pub_recon_loss": pub_recon_loss
                })
            # Validalidation Steps
            we_should_do_validation_iteration = isinstance(validate_every_n_batches, int) and validate_every_n_batches > 0
            if we_should_do_validation_iteration: 
                # TODO: We need to finish this
                validation_score = validate_vae_recon(
                    model_vae = model_vae,
                    all_valid_seqs = all_validation_seqs,
                    validation_batch_size = batch_size,
                    pub_columns =pub_columns
                )
                if wandb_on:
                    wandb.log({
                        "validation_score" : validation_score
                    })
            else: 
                raise RuntimeError("Expected a positive integer for the time interval for validation.")

            recon_losses.append(pub_recon_loss.mean().item())

    return model_vae, recon_losses

def train_vae_and_adversary_bi_level(
    batch_size: int,
    prv_columns: list[int], # Mostly so we know what to ignore.
    all_train_seqs: torch.Tensor,
    all_validation_seqs: torch.Tensor,
    epochs: int,
    model_vae: nn.Module,
    model_adversary: nn.Module,
    adv_train_subepochs: int,
    adv_epoch_subsample_percent: float,
    optimizer_vae: torch.optim.Optimizer, # type: ignore
    optimizer_adv: torch.optim.Optimizer, # type:ignore
    kl_dig_hypr: float, 
    wandb_on: bool, 
    priv_utility_tradeoff_coeff: float,
    validate_every_n_batches: Optional[int] = None,
) -> tuple[nn.Module, nn.Module, list[float]]:
    """
    The point of this funciton is to be able to test the VAE model independent of the adversary.
    So that we can focus on getting the best performance out of reconstruction
    """
    # Some Helper Variables
    pub_columns = list(set(range(all_train_seqs.shape[-1])) - set(prv_columns))

    num_seqs_as_samples = all_train_seqs.shape[0]
    recon_losses = []
    for e in tqdm(range(epochs), desc="Training Vae-Adv Epochs"):
        batch_steps = np.arange(0, num_seqs_as_samples, batch_size)
        recon_losses = []
        for _batch_offset in batch_steps:
            ########################################
            # Data Collection
            ########################################
            batch_end = min(_batch_offset + batch_size, num_seqs_as_samples)
            batch_all = all_train_seqs[_batch_offset: batch_end, :, :]
            batch_pub = all_train_seqs[_batch_offset: batch_end, :, pub_columns]
            batch_prv = all_train_seqs[_batch_offset: batch_end, :, prv_columns]

            ########################################
            # Adversarial Training must take place first.
            ########################################
            model_aversary = train_adversary(
                model_vae=model_vae, 
                model_adversary=model_adversary,
                opt_adversary=optimizer_adv,
                epochs=adv_train_subepochs,
                epoch_sample_percent=adv_epoch_subsample_percent,
                global_samples=all_train_seqs,
                prv_cols=prv_columns,
                batch_size = batch_size # Share batch_size
            )

            ########################################
            # Actual Training
            ########################################
            # Now we train the autoencoder to try to fool the adversary
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence

            # Sneaky adversary comes here and reads the latent data
            model_adversary.freeze()
            adversary_guess = model_adversary(latent_z)

            adv_train_loss = F.mse_loss(adversary_guess, batch_prv[:, -1, :])
            # Check on performance
            pub_recon_loss = F.mse_loss(sanitized_data, batch_pub[:, -1, :], reduction="none").mean(-1) # Guesss the set of features of sequence
            # pub_recon_norm = pub_recon_loss / (pub_recon_loss.detach() + 1e-8) TEST: Just trying normaliztion
            # DEBUG: These two lines below and the change at final_scalar are highly experimetnal
            gamma_priv = priv_utility_tradeoff_coeff
            gamma_pub = 0.5
            final_loss_scalar = (
                gamma_pub * pub_recon_loss - gamma_priv * adv_train_loss + kl_dig_hypr * kl_divergence # Just good ol reconstruciton to see if it works
                # pub_recon_loss + kl_dig_hypr * kl_divergence # Just good ol reconstruciton to see if it works
            ).mean()

            # Check on performance
            # torch.nn.utils.clip_grad_norm_(model_vae.parameters(), max_norm=1.0)

            optimizer_vae.zero_grad()
            final_loss_scalar.backward()
            optimizer_vae.step()
            model_adversary.unfreeze()

            if wandb_on: 
                wandb.log({
                    "pub_recon_loss": pub_recon_loss
                })
            # Validalidation Steps
            we_should_do_validation_iteration = isinstance(validate_every_n_batches, int) and validate_every_n_batches > 0
            if we_should_do_validation_iteration: 
                # TODO: We need to finish this
                validation_score = validate_vae_recon(
                    model_vae = model_vae,
                    all_valid_seqs = all_validation_seqs,
                    validation_batch_size = batch_size,
                    pub_columns =pub_columns
                )
                if wandb_on:
                    wandb.log({
                        "validation_score" : validation_score
                    })
            else: 
                raise RuntimeError("Expected a positive integer for the time interval for validation.")

            recon_losses.append(pub_recon_loss.mean().item())

    return model_vae, model_adversary, recon_losses

@deprecated("Not entirely sure it works",date="2025-05-04 10:18")
def train_vae_and_adversary_bi_level_deprecated(
    batch_size: int,
    priv_columns: List[int],
    all_train_seqs: torch.Tensor,
    all_valid_seqs: torch.Tensor,
    ds_test:  Optional[np.ndarray], # TOREM: After debugging
    epochs: int,
    inner_epochs: int,  # These are for the adversary
    epoch_sample_percent: float,
    model_vae: nn.Module,
    model_adversary: nn.Module,
    opt_vae: torch.optim.Optimizer,# type: ignore
    opt_adversary: torch.optim.Optimizer, # type: ignore
    kl_dig_hypr: float,
    wandb_on: bool,
    priv_utility_tradeoff_coeff: float,
) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Training Loop
        ds_train: np.ndarray (num_batches, batch_size, features),
    """

    total_num_features = all_train_seqs.shape[-1]
    pub_columns = list(set(range(total_num_features)) - set(priv_columns))

    train_pub = all_train_seqs[:, :, pub_columns]
    print(f"priv columns are {priv_columns}")
    print(f"train_prv shape is {all_train_seqs.shape}")
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
            time_batch_start = time.time()
            batch_all = all_train_seqs[
                batch_no * batch_size : (batch_no + 1) * batch_size
            ]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            time_batch_collect = time.time()
            logger.debug(f"Batch {batch_no} data collection took {time_batch_collect - time_batch_start} seconds")
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
                all_train_seqs,
                total_num_features,
                priv_columns,
                batch_size,
            )
            time_train_adversary = time.time()
            logger.debug(f"Training the adversary took {time_train_adversary - time_batch_collect} seconds")
            if ds_test is not None:
                # DEBUG: Check on adversaries performance
                os.makedirs("./figures/progress/", exist_ok=True)
                metrics = advVae_test_file(
                    ds_test,
                    priv_columns,
                    model_vae,
                    model_adversary,
                    sequence_len,
                    -1,
                    batch_size,
                    wandb_on,
                    recon_savefig_loc=f"./figures/progress/vae_reconstruction_{e+1}.png",
                    adv_savefig_loc=f"./figures/progress/vae_adversary_{e+1}.png",
                )
            time_test_vae = time.time()
            logger.debug(f"Testing the VAE took {time_test_vae - time_train_adversary} seconds")

            # Now we train the autoencoder to try to fool the adversary
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all[:,:-1,:]) # Do not leak the last element of sequence

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_prv[:, -1, :].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data, batch_pub[:, -1, :], reduction="none").mean(-1) # Guesss the set of features of sequence
            adver_loss = F.mse_loss(adversary_guess_flat, batch_y_flat, reduction="none").squeeze()

            # TEST: Just trying normaliztion
            # pub_recon_norm = pub_recon_loss / (pub_recon_loss.detach() + 1e-8)
            # adver_loss_norm = adver_loss / (adver_loss.detach() + 1e-8)

            final_loss_scalar = (
                # pub_recon_loss - priv_utility_tradeoff_coeff * adver_loss + kl_dig_hypr * kl_divergence
                # pub_recon_loss - priv_utility_tradeoff_coeff * adver_loss + kl_dig_hypr * kl_divergence
                pub_recon_loss + kl_dig_hypr * kl_divergence # Just good ol reconstruciton to see if it works
            ).mean()

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adver_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            # TEST: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model_adversary.parameters(), max_norm=1.0)
            opt_vae.step()
            time_train_recon = time.time()
            logger.debug(f"Training the VAE took {time_train_recon - time_test_vae} seconds")

            if wandb_on:
                wandb.log(
                    {
                        "adv_train_loss": adver_loss.mean().item(),
                        "pub_recon_loss": pub_recon_loss.mean().item(),
                        "final_loss_scalar": final_loss_scalar.item(),
                    }
                )

            # if batch_no % 16 == 0:
            #     # TODO: Finish the validation implementaion with correlation
            #     # - Log the validation metrics here
            #     model_vae.eval()
            #     model_adversary.eval()
            #     with torch.no_grad():
            #         validation_metrics = calculate_validation_metrics(
            #             all_valid_seqs,
            #             pub_columns,
            #             priv_columns,
            #             model_vae,
            #             model_adversary,
            #         )
            #     model_vae.train()
            #     model_adversary.train()
            #
            #     # Report to wandb
            #     if wandb_on:
            #         wandb.log(validation_metrics)

    return model_vae, model_adversary, recon_losses, adv_losses
