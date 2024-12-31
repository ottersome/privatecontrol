import argparse
import os
from math import ceil
from typing import Dict, List, OrderedDict, Sequence, Tuple

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
from torch import nn
from torch.nn import functional as F
import pandas as pd

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.data.dataset_generation import batch_generation_randUni, collect_n_sequential_batches, spot_backhistory
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE, SequenceToScalarVAE
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger

traceback.install()

console = Console()


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=8, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
    )
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--rnn_num_layers", default=2, type=int)
    ap.add_argument("--rnn_hidden_size", default=15, type=int)
    ap.add_argument(
        "--cols_to_hide",
        default=[4],
        help="Which are the columsn we want no information of",
    )  # Remember 0-index (so 5th)
    ap.add_argument("--vae_latent_size", default=32, type=int)
    ap.add_argument("--episode_length", default=32, type=int)
    ap.add_argument("--episode_gap", default=3, type=int)
    ap.add_argument("--vae_hidden_size", default=32, type=int)
    ap.add_argument(
        "--splits",
        default={"train_split": 0.8, "val_split": 0.2, "test_split": 0.0},
        type=list,
        nargs="+",
    )
    ap.add_argument("--kl_dig_hypr", default=0.1, type=float)

    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.01, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)
    ap.add_argument("--adversary_hidden_size", default=32, type=int)
    ap.add_argument("--padding_value", default=-1, type=int)

    ap.add_argument("--vae_ds_cache", default=".cache/pykalpkg_vaeds.csv", type=str)
    ap.add_argument("--ds_cache", default=".cache/pykalpkg_ds.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/pykalman_transformer/",
        help="Where to save the outputs",
    )

    ap.add_argument(
        "--eval_interval", default=100, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--eval_size", default=4, help="How many systems to generate", type=int
    )

    ap.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether or not to use debugpy for trainig",
    )
    ap.add_argument(
        "--debug_port",
        default=42020,
        type=int,
        help="Port to attach debugpy to listen to.",
    )

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check


def train_w_metrics(
    model: nn.Module,
    dataset: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    saveplot_dest: str,
    train_percent: float,
    device: torch.device,
):
    tlosses = []
    vlosses = []
    train_size = int(len(dataset[0]) * train_percent)
    batch_count = int(train_size // batch_size)
    tdataset = (dataset[0][:train_size, :], dataset[1][:train_size, :])
    vdataset = (dataset[0][train_size:, :], dataset[1][train_size:, :])

    # layout, progress = make_layout(0, tlosses, vlosses)
    layout = TrainLayout(epochs, batch_count, tlosses, vlosses)
    batch_num = 0
    with Live(layout.layout, console=console, refresh_per_second=10) as live:
        for epoch, batch_no, tloss, vloss in train(
            model, tdataset, vdataset, epochs, batch_size, saveplot_dest, device
        ):
            batch_num += 1
            logger.debug(f"Batch number: {batch_num}")
            tlosses.append(tloss)
            vlosses.append(vloss)
            layout.update(epoch, batch_no, tloss, vloss)

    # TODO: Recover this
    # Plot the losses after the episode finishes
    # t_diff_in_order = np.max(tlosses) - np.min(tlosses) > 1e1
    # v_diff_in_order = np.max(vlosses) - np.min(vlosses) > 1e1
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(tlosses)
    # # if t_diff_in_order:
    # # axs[0].set_yscale("log")
    # axs[0].set_title("Training Loss")
    # axs[1].plot(vlosses)
    # # if v_diff_in_order:
    # #     axs[1].set_yscale("log")
    # axs[1].set_title("Validation Loss")
    # plt.show()
    #


def indiscriminate_supervision(ds: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Cannot think of better name for now.
    It will:
        - Take a Dict[str, np.ndarry].
        - Discard keys
        - Mix around the data that is correlated
    Yes, these are some assumptinos. But its only to get it running
    Returns:
        - np.ndarray: A new data set of shape (len(ds), context_columns))
    """
    final_ds = []
    for k, v in ds.items():
        final_ds.append(v)
    # Shuffle it around
    final_ds = np.concatenate(final_ds)
    # TODO: Check that the shuffling is being done right
    np.random.shuffle(final_ds)
    return final_ds

def validation_data_organization(
    ds: Dict[str, np.ndarray], snapshot_length: int = 12, num_episodes: int = 3
) -> List[np.ndarray]:
    """
    Will take random snapshots of samples from the validation data.
    """
    episodes = []
    for i in range(num_episodes):
        random_bucket_key = np.random.choice(list(ds.keys()))
        random_bucket = ds[random_bucket_key]
        bucket_length = len(random_bucket)
        random_position = np.random.randint(bucket_length - snapshot_length)
        episodes.append(
            random_bucket[random_position : random_position + snapshot_length]
        )

    return episodes

def validate_entire_file(
    validation_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
    sequence_length: int,
    debug_file: pd.DataFrame,
    padding_value: int,
    save_path: str = "./figures/new_data_vae/plot_vaerecon_eval_{}.png",
    batch_size: int = 16,
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
    val_x = torch.from_numpy(validation_file).to(torch.float32).to(model_device)

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
        if batch_no == num_batches - 1:
            print("Meep")
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
    some_4_idxs = np.random.randint(0, seq_reconstructions.shape[1], 4)

    ########################################
    # Chart For Reconstruction
    ########################################
    recon_to_show = seq_reconstructions[:, some_4_idxs]
    truth_to_compare = validation_file[:, some_4_idxs]
    fig,axs = plt.subplots(4,2,figsize=(16,10))
    for i in range(recon_to_show.shape[1]):
        axs[i,0].plot(recon_to_show[:,i].squeeze().detach().cpu().numpy(), label="Reconstruction")
        axs[i,0].set_title("Reconstruction")
        axs[i,0].legend()
        axs[i,1].plot(truth_to_compare[:,i].squeeze(), label="Truth")
        axs[i,1].set_title("Truth")
        axs[i,1].legend()
    plt.savefig(f"reconstruction.png")
    plt.close()

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

    plt.savefig(f"adversary.png")
    plt.close()

    model_vae.train()
    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}

# TODO: Fix this. It is not working
# def validation_iteration(
#     validation_episodes: torch.Tensor,
#     idxs_colsToGuess: Sequence[int],
#     model_vae: nn.Module,
#     model_adversary: nn.Module,
#     seq_length: int,
#     debug_file: pd.DataFrame,
#     save_path: str = "./figures/new_data_vae/plot_vaerecon_eval_{}.png",
# ) -> Dict[str, float]:
#     """
#     Will run a validation iteration for a model
#     Args:
#         - validation_episodes: Validation data (num_episodes, epsode_length, num_features)
#         - model: Model to run the validation on
#         - col_to_predict: (0-index) which column we would like to predict
#     """
#     metrics = {
#         "recon_loss": [],
#         "adv_loss": [],
#     }
#
#     model_vae.eval()
#     model_adversary.eval()
#
#     if validation_episodes.shape[0] > 3:
#         raise ValueError(
#             "You may be using too many samples. Please reduce the number of samples"
#         )
#
#     cols_as_features = list(
#         set((range(validation_episodes.shape[2]))) - set(idxs_colsToGuess)
#     )
#     model_device = next(model_vae.parameters()).device
#     # val_x = (
#     #     validation_episodes[:, :, cols_as_features].to(torch.float32).to(model_device)
#     # )
#     # Divide the episode into sequences of length seq_length
#     val_x = validation_episodes.view(-1, seq_length, validation_episodes.shape[-1]) # TODO: Test
#
#     # fig, axs = plt.subplots(val_x.shape[0], 2, figsize=(16, 10), dpi=200)
#     # plt.subplots_adjust(
#     #     left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.5
#     # )
#     # plt.tight_layout()
#     # plt.title("Validation Data")
#     #
#     latent_z, sanitized_data, kl_divergence = model_vae(val_x)
#
#     # TODO: Incorporate Adversary Guess
#     # adversary_guess_flat = model_adversary(latent_z)
#     # adversary_guess = adversary_guess_flat.view(val_x.shape[0], -1)
#     # non_sanitized_data = val_x
#
#
#
#
#     # Old stuff for reference
#     # for e in range(val_x.shape[0]):
#     #
#     #     # These two vectors are of shape (1, sequence_length, num_features)
#     #     # We want features to be in the same plot, and differerent sequences in differnt plots
#     #     axs[e, 0].plot(
#     #         non_sanitized_data[e, :, 6].squeeze().detach().cpu().numpy(),
#     #         label=f"True $f_{e}$",
#     #     )
#     #     axs[e, 0].set_title(f"Non-Sanitized episode {e}")
#     #     axs[e, 1].plot(
#     #         sanitized_data[e, :, 6].squeeze().detach().cpu().numpy(),
#     #         label=f"True $f_{e}$",
#     #     )
#     #     axs[e, 1].set_title(f"Sanitized episode  {e}")
#     #
#     #     recon_loss = F.mse_loss(
#     #         sanitized_data[e, :, :], non_sanitized_data[e, :, :], reduction="mean"
#     #     )
#     #     logger.debug(f"The mean loss for this episode was: {recon_loss.item()}")
#     #     # adv_loss = F.mse_loss(model(sanitized_data), episode_y)
#     #     metrics["recon_loss"].append(recon_loss.mean().item())
#     #     # metrics["adv_loss"].append(adv_loss.item())
#
#     # lets now save the figure
#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
#     plt.close()
#
#     model_vae.train()
#     model_adversary.train()
#
#     return {k: np.mean(v).item() for k, v in metrics.items()}


def compare_reconstruction():
    """
    Will take the original set of features and
    """
    raise NotImplementedError


# TODO: Later change the name of the function
# def train_v0(
#     batch_size: int,
#     columns_to_hide: List[int],
#     data_columns: List[str],
#     device: torch.device,
#     ds_train: Dict[str, np.ndarray],
#     ds_val: Dict[str, np.ndarray],
#     episode_gap: int,
#     episode_length: int,
#     epochs: int,
#     model_vae_adversary: SequentialVAE,
#     learning_rate: float,
#     kl_dig_hypr: float,
#     # Some Extra Params
#     saveplot_dest: str,
# ):
#     feature_columns = list(set(range(len(data_columns))) - set(columns_to_hide))
#     opt_vae = torch.optim.Adam(model_vae_adversary.parameters(), lr=learning_rate)  # type: ignore
#
#     device = next(model_vae_adversary.parameters()).device
#     # assert (
#     #     device == next(model_adversary.parameters()).device
#     # ), "DThe device of the VAE and the adversary should be the same"
#
#     all_train_data = timeseries_ds_formation(ds_train, episode_length, episode_gap)
#     train_x = all_train_data[:, :, feature_columns]
#     train_y = all_train_data[:, :, columns_to_hide]
#     sequence_length = train_x.shape[1]
#     num_priv_cols = train_y.shape[2]
#
#     # Similarly for the validation
#     # all_val_data = indiscriminate_supervision(ds_val)
#     # val_x = all_val_data[:, columns_to_share]
#     # val_y = all_val_data[:, columns_to_hide]
#     # Validation data will not be shuffled since we vant to visualize the results in time series
#
#     # TODO: Original Data for Validataion
#     validation_episodes_list: List[np.ndarray] = validation_data_organization(
#         ds_train, snapshot_length=episode_length, num_episodes=3
#     )
#     validation_episodes_tensor = torch.from_numpy(
#         np.stack(validation_episodes_list)
#     ).to(device)
#
#     num_batches = train_x.shape[0] // batch_size
#     logger.info(f"Working with {train_x.shape[0]} samples")
#     recon_losses = []
#     adv_losses = []
#     d_sanitized_dist = []
#     for e in range(epochs):
#         logger.info(f"Epoch {e} of {epochs}")
#         # losses_for_dist = []
#         adversary_losses_batch = []
#         for b in range(num_batches):
#             # Now Get the new VAE generations
#             model_vae_adversary.zero_grad()
#             batch_x = (
#                 torch.from_numpy(train_x[b * batch_size : (b + 1) * batch_size])
#                 .to(torch.float32)
#                 .to(device)
#             )
#             batch_y = (
#                 torch.from_numpy(train_y[b * batch_size : (b + 1) * batch_size])
#                 .to(torch.float32)
#                 .to(device)
#             )
#             recon_data = batch_x.clone().requires_grad_(False)
#             if batch_x.shape[0] != batch_size:
#                 continue
#             logger.info(f"Batch {b} of {num_batches} with shape {batch_x.shape}")
#
#             # Get the adversary to guess the sensitive column
#             sanitized_data, adversary_guess_flat, kl_divergence = model_vae_adversary(
#                 batch_x
#             )
#
#             # logger.info(f"batch_x sum is {batch_x.sum()}")
#             # plt.hist(batch_x.flatten().detach().cpu().numpy(), bins=100)
#             # plt.title(f"Training Sanitized Episode Input Dist {e}")
#             # plt.show()
#
#             # This should be it
#             recon_loss = F.mse_loss(sanitized_data, recon_data, reduction="none")
#             batch_y_flat = batch_y.view(-1, batch_y.shape[-1])
#             adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
#             # losses_for_dist.append(recon_loss.view(-1).tolist())
#             loss = (
#                 recon_loss.mean(-1)
#                 + torch.log(1 / (1.0e-10 + adv_loss))
#                 - kl_dig_hypr * kl_divergence
#             ).mean()
#
#             # model_adversary.zero_grad()
#             # logger.info(f"Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")
#             loss.backward()
#             recon_losses.append(recon_loss.mean().item())
#             adv_losses.append(adv_loss.mean().item())
#
#             opt_vae.step()
#             # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")
#
#             if b % 16 == 0:
#                 save_path = (
#                     f"./figures/new_data_vae/plot_vaerecon_eval_{e:02d}_{b:02d}.png"
#                 )
#                 metrics = validation_iteration(
#                     validation_episodes_tensor,
#                     columns_to_hide,
#                     model_vae_adversary,
#                     save_path,
#                 )
#                 logger.info(f"Validation Metrics are {metrics}")
#
#                 # Plot the histogram of losses
#                 # plt.hist(losses_for_dist, bins=100)
#                 # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")
#
#                 # meepo = sanitized_data.view(-1).tolist()
#                 # plt.hist(meepo, bins=100)
#                 # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")
#                 # plt.close()
#
#     fig, axs = plt.subplots(1, 2, figsize=(16,10))
#     axs[0].plot(recon_losses)
#     axs[0].set_title("Reconstruction Loss")
#     axs[1].plot(adv_losses)
#     axs[1].set_title("Adversary Loss")
#
#     plt.savefig(f"./figures/new_data_vae/recon-adv_losses.png")
#
#     plt.close()
#
#     return model_vae_adversary


def train_v1(
    batch_size: int,
    columns_to_hide: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    episode_gap: int,
    sequence_length: int,
    epochs: int,
    model_vae: SequenceToScalarVAE,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
    # Some Extra Params
    saveplot_dest: str,
    debug_file: pd.DataFrame, # TOREM: I don't think we need this
    padding_value: int,
):
    feature_columns = list(set(range(len(data_columns))) - set(columns_to_hide))
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=learning_rate)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)  # type: ignore

    device = next(model_vae.parameters()).device
    # assert (
    #     device == next(model_adversary.parameters()).device
    # ), "DThe device of the VAE and the adversary should be the same"

    # all_train_data = timeseries_ds_formation(ds_train, sequence_length, episode_gap)
    all_train_data, validation_sequence = batch_generation_randUni(ds_train, sequence_length, -1, 1.2)
    # Returns (num_rolluts, sequence_length, num_columns)
    train_pub = all_train_data[:, :, feature_columns]
    train_priv = all_train_data[:, :, columns_to_hide]
    sequence_length = train_pub.shape[1]
    num_priv_cols = train_priv.shape[2]

    # TOREM: Possibly remove this since this is old behavior.
    # validation_episodes_list: List[np.ndarray] = validation_data_organization(
    #     ds_train, snapshot_length=sequence_length, num_episodes=3
    # )
    # validation_episodes_tensor = torch.from_numpy(
    #     np.stack(validation_episodes_list)
    # ).to(device)

    ########################################
    # Get Batches
    ########################################

    num_batches = train_pub.shape[0] // batch_size
    logger.info(f"Working with {train_pub.shape[0]} samples")
    recon_losses = []
    adv_losses = []
    d_sanitized_dist = []
    for e in range(epochs):
        logger.info(f"Epoch {e} of {epochs}")
        # losses_for_dist = []
        adversary_losses_batch = []
        for b in range(num_batches):
            # Now Get the new VAE generations
            batch_pub = (
                torch.from_numpy(train_pub[b * batch_size : (b + 1) * batch_size])
                .to(torch.float32)
                .to(device)
            )
            batch_all = (
                torch.from_numpy(all_train_data[b * batch_size : (b + 1) * batch_size])
                .to(torch.float32)
                .to(device)
            )
            batch_priv = (
                torch.from_numpy(train_priv[b * batch_size : (b + 1) * batch_size, -1, :])
                .to(torch.float32)
                .to(device)
            )
            recon_data = batch_pub.clone().requires_grad_(False)
            if batch_pub.shape[0] != batch_size:
                continue
            logger.info(f"Batch {b} of {num_batches} with shape {batch_pub.shape}")

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all)

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)

            # Check on performance
            batch_y_flat = batch_priv.view(-1, batch_priv.shape[-1])
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            model_adversary.zero_grad()
            adv_loss.backward()
            opt_adversary.step()

            ########################################
            # 2. Calculate the Recon Loss
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all)
            pub_prediction = batch_pub[:,-1,:]

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_priv.view(-1, batch_priv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data[:, feature_columns], pub_prediction)
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = pub_recon_loss - 4.0 * adv_loss + kl_dig_hypr * kl_divergence.mean()

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adv_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            opt_vae.step()
            # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if b % 16 == 0:
                save_path = (
                    f"./figures/new_data_vae/plot_vaerecon_eval_{e:02d}_{b:02d}.png"
                )
                metrics = validate_entire_file(
                    validation_sequence,
                    columns_to_hide,
                    model_vae,
                    model_adversary,
                    sequence_length,
                    debug_file,
                    padding_value,
                    save_path,
                )
                logger.info(f"Validation Metrics are {metrics}")

                # Plot the histogram of losses
                # plt.hist(losses_for_dist, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")

                # meepo = sanitized_data.view(-1).tolist()
                # plt.hist(meepo, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")
                # plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].plot(recon_losses)
    axs[0].set_title("Reconstruction Loss")
    axs[1].plot(adv_losses)
    axs[1].set_title("Adversary Loss")

    plt.savefig(f"./figures/new_data_vae/recon-adv_losses.png")

    plt.close()

    return model_adversary


# TODO: We need to implement federated learning in this particular part of the expression
def federated():
    # We also need a federated aspect to all this. And its getting close to being time to implementing this
    raise NotImplementedError


# This ought to be ran iterarively witht the encoder so that this also learns to better extract from the new encoder version
# TOREM: (Maybe) Consider removing this if you end up moving it elsewhere
def train_adversary_iteration(
    adversary: nn.Module,
    data: torch.Tensor,
    epochs: int = 1,
    adv_batch_size: int = 64,
    col_idx_to_predict: int = 4,
):
    """
    Will train an adversary to guess the next value
    """
    adversary.train()
    criterion = nn.MSELoss()
    # FIX: Remove the hyperparameters
    optimizer = torch.optim.Adam(adversary.parameters(), lr=0.01)  # type: ignore

    # Once this is just regression
    for e in range(epochs):
        for b in range(data.shape[0] // adv_batch_size):
            # Get the batch
            adversary.zero_grad()
            batch_x = data[b * adv_batch_size : (b + 1) * adv_batch_size]
            batch_y = data[b * adv_batch_size : (b + 1) * adv_batch_size]
            # Get the prediction
            preds = adversary(batch_x)

            # Calculate the loss
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    adversary.eval()

    return adversary


def set_all_seeds(seed: int):
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)
    logger = create_logger("main_training")

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", args.debug_port))
        debugpy.wait_for_client()

    # TODO: Make it  so that generate_dataset checks if params are the same
    columns, runs_dict, debug_file = load_defacto_data(args.defacto_data_raw_path)
    num_columns = len(columns)
    num_private_cols = len(args.cols_to_hide)

    # Separate them into their splits (and also interpolate)
    train_runs, val_runs, test_runs = split_defacto_runs(
        runs_dict,
        **args.splits,
    )

    logger.info(f"Using device is {device}")

    # Get Informaiton for the VAE
    # vae_input = len(columns) - len(args.cols_to_hide)  # For when we want to send only the public ones
    vae_input_size = len(columns) # I think sending all of them is better
    # pub_dimensions 
    # TODO: Get the model going
    model_vae = SequenceToScalarVAE(
        # inout_size for model is output_dim for data
        input_size=vae_input_size,
        latent_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
        num_features_to_guess=1,
        rnn_num_layers=args.rnn_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
    ).to(device)

    model_adversary = Adversary(
        input_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)

    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    logger.info("Starting the VAE Training")
    vae = train_v1(
        args.batch_size,
        args.cols_to_hide,
        columns,
        device,
        train_runs,
        val_runs,
        args.episode_gap,
        args.episode_length,
        args.epochs,
        model_vae,
        model_adversary,
        args.kl_dig_hypr,
        args.lr,
        args.saveplot_dest,
        debug_file,
        args.padding_value,
    )

    # 🚩 development so far🚩
    exit()

    # todo: we might want to do a test run here
    # if len(test_runs) > 0:
    #     test_runs = train_new(test_runs)

    # trainvae_wprivacy(
    #     training_data,
    #     (hidden, outputs),
    #     args.epochs,
    #     args.saveplot_dest,
    #     [true,false,True]


# )
#
if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
