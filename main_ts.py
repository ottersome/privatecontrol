import argparse
import os
from math import ceil
from typing import Dict, List, Sequence, Tuple

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
from torch import nn
from torch.nn import functional as F

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.vae import SeqAdversarialVAE
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger

traceback.install()

console = Console()

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=10, help="How many epochs to train for", type=int
    )
    ap.add_argument("--defacto_data_raw_path", default="./data/", type=str, help="Where to load the data from")
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--rnn_num_layers", default=2, type=int)
    ap.add_argument("--cols_to_hide", default=[4], help="Which are the columsn we want no information of") # Remember 0-index (so 5th)
    ap.add_argument("--vae_latent_size", default=32, type=int)
    ap.add_argument("--episode_length", default=32, type=int)
    ap.add_argument("--episode_gap", default=3, type=int)
    ap.add_argument("--vae_hidden_size", default=32, type=int)
    ap.add_argument("--splits", default= { "train_split": 0.8, "val_split" : 0.2, "test_split" : 0.0 } , type=list, nargs="+")

    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.001, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)
    ap.add_argument("--adversary_hidden_size", default=32, type=int)

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

    ap.add_argument("--debug", "-d", action="store_true", help="Whether or not to use debugpy for trainig")
    ap.add_argument("--debug_port", default=42019, type=int, help="Port to attach debugpy to listen to.")

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

def timeseries_ds_formation(ds: Dict[str, np.ndarray], episode_length: int, gap: int):
    final_ds = []
    # Calculate the distances
    hop_distance = episode_length - gap

    rollouts = []
    for k, v in ds.items():
        for h in range(ceil(v.shape[0] // hop_distance)):
            offset = hop_distance * h
            logger.info(f"({k}: len({v.shape[0]})): Adding att offset {offset} to {offset+episode_length}")
            rollout = v[offset:offset+episode_length, :]
            if rollout.shape[0] != episode_length:
                continue
            rollouts.append(rollout)

    # Shuffle it around
    rollouts = np.stack(rollouts, axis=0)
    # TODO: Check that the shuffling is being done right
    np.random.shuffle(rollouts)
    return rollouts


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
        episodes.append(random_bucket[random_position : random_position + snapshot_length])

    return episodes

# TODO: Fix this. It is not working
def validation_iteration(
    validation_episodes: torch.Tensor,
    idxs_colsToGuess: Sequence[int],
    model: nn.Module,
    save_path: str = "./figures/new_data_vae/plot_vaerecon_eval_{}.png",
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_episodes: Validation data (num_episodes, epsode_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    model.eval()
    if validation_episodes.shape[0] > 3:
        raise ValueError("You may be using too many samples. Please reduce the number of samples")

    cols_as_features = list(set((range(validation_episodes.shape[2]))) - set(idxs_colsToGuess))
    model_device = next(model.parameters()).device
    val_x = validation_episodes[:, :, cols_as_features].to(torch.float32).to(model_device)
    val_y = validation_episodes[:, :, idxs_colsToGuess].to(torch.float32).to(model_device)

    fig,axs = plt.subplots(val_x.shape[0],2,figsize=(16, 10), dpi=300)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.5)
    plt.tight_layout()
    plt.title("Validation Data")
    features_to_check = [5]

    sanitized_data, guessed_features, _ = model(val_x)
    non_sanitized_data = val_x

    # Old stuff for reference
    for e in range(val_x.shape[0]):

        # These two vectors are of shape (1, sequence_length, num_features)
        # We want features to be in the same plot, and differerent sequences in differnt plots
        all_data = torch.cat([non_sanitized_data[e, :, :], sanitized_data[e, :, :]])
        min = torch.min(all_data)
        max = torch.max(all_data)
        axs[e, 0].plot(
            non_sanitized_data[e, :, 6].squeeze().detach().cpu().numpy(),
            label=f"True $f_{e}$",
        )
        axs[e, 0].set_title(f"Non-Sanitized episode {e}")
        axs[e, 1].plot(
            sanitized_data[e, :, 6].squeeze().detach().cpu().numpy(),
            label=f"True $f_{e}$",
        )
        axs[e, 1].set_title(f"Sanitized episode  {e}")
        # Now set Y_scale
        # axs[e,0].set_ylim(min.item(), max.item())
        # axs[e,1].set_ylim(min.item(), max.item())

        recon_loss = F.mse_loss(sanitized_data[e,:,:], non_sanitized_data[e,:,:], reduction='mean')
        logger.debug(f"The mean loss for this episode was: {recon_loss.item()}")
        # adv_loss = F.mse_loss(model(sanitized_data), episode_y)
        metrics["recon_loss"].append(recon_loss.mean().item())
        # metrics["adv_loss"].append(adv_loss.item())

    # lets now save the figure
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    model.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}


def compare_reconstruction():
    """
    Will take the original set of features and
    """
    raise NotImplementedError

# TODO: Later change the name of the function
def train_v0(
    batch_size: int,
    columns_to_hide: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    episode_gap: int,
    episode_length: int,
    epochs: int,
    model_vae_adversary: SeqAdversarialVAE,
    learning_rate: float,
    # Some Extra Params
    saveplot_dest: str,
):
    feature_columns = list(set(range(len(data_columns))) - set(columns_to_hide))
    opt_vae = torch.optim.Adam(model_vae_adversary.parameters(), lr=learning_rate) # type: ignore

    device = next(model_vae_adversary.parameters()).device
    # assert (
    #     device == next(model_adversary.parameters()).device
    # ), "DThe device of the VAE and the adversary should be the same"

    all_train_data = timeseries_ds_formation(ds_train, episode_length, episode_gap)
    train_x = all_train_data[:, :, feature_columns]
    train_y = all_train_data[:, :, columns_to_hide]
    sequence_length = train_x.shape[1]
    num_priv_cols = train_y.shape[2]

    # Similarly for the validation
    # all_val_data = indiscriminate_supervision(ds_val)
    # val_x = all_val_data[:, columns_to_share]
    # val_y = all_val_data[:, columns_to_hide]
    # Validation data will not be shuffled since we vant to visualize the results in time series

    # TODO: Original Data for Validataion
    validation_episodes_list: List[np.ndarray] = validation_data_organization(
        ds_train, snapshot_length=episode_length, num_episodes=3
    )
    validation_episodes_tensor =  torch.from_numpy(np.stack(validation_episodes_list)).to(device)

    num_batches = train_x.shape[0] // batch_size
    logger.info(f"Working with {train_x.shape[0]} samples")
    recon_losses = []
    d_sanitized_dist = []
    for e in range(epochs):
        logger.info(f"Epoch {e} of {epochs}")
        # losses_for_dist = []
        adversary_losses_batch = []
        for b in range(num_batches):
            # Now Get the new VAE generations
            model_vae_adversary.zero_grad()
            batch_x = torch.from_numpy(train_x[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            batch_y = torch.from_numpy(train_y[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            recon_data = batch_x.clone().requires_grad_(False)
            if batch_x.shape[0] != batch_size:
                continue
            logger.info(f"Batch {b} of {num_batches} with shape {batch_x.shape}")

            # Get the adversary to guess the sensitive column
            sanitized_data, adversary_guess_flat, kl_divergence = model_vae_adversary(batch_x)

            # logger.info(f"batch_x sum is {batch_x.sum()}")
            # plt.hist(batch_x.flatten().detach().cpu().numpy(), bins=100)
            # plt.title(f"Training Sanitized Episode Input Dist {e}")
            # plt.show()

            # This should be it
            recon_loss = F.mse_loss(sanitized_data, recon_data, reduction="none")
            batch_y_flat = batch_y.view(-1,batch_y.shape[-1])
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            # losses_for_dist.append(recon_loss.view(-1).tolist())
            loss = (recon_loss.mean(-1) - kl_divergence).mean()

            # model_adversary.zero_grad()
            # logger.info(f"Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")
            loss.backward()
            recon_losses.append(recon_loss.mean().item())

            opt_vae.step()
            # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if b % 16 == 0:
                save_path = f"./figures/new_data_vae/plot_vaerecon_eval_{e:02d}_{b:02d}.png"
                metrics = validation_iteration(validation_episodes_tensor, columns_to_hide, model_vae_adversary, save_path)
                logger.info(f"Validation Metrics are {metrics}")

                # Plot the histogram of losses
                # plt.hist(losses_for_dist, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")

                # meepo = sanitized_data.view(-1).tolist()
                # plt.hist(meepo, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")
                # plt.close()

    # Plot some old stuff
    # Try to pllot some stuff just for the sake of debugging
    # New plot
    plt.plot(recon_losses)
    plt.title("Reconstruction Loss")
    # Save the plot
    plt.show()

    return model_vae_adversary

def train_v1(
    batch_size: int,
    columns_to_hide: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    episode_gap: int,
    episode_length: int,
    epochs: int,
    model_vae: SeqAdversarialVAE,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
    # Some Extra Params
    saveplot_dest: str,
):
    feature_columns = list(set(range(len(data_columns))) - set(columns_to_hide))
    opt_vae = torch.optim.Adam(model_vae_adversary.parameters(), lr=learning_rate) # type: ignore

    device = next(model_vae_adversary.parameters()).device
    # assert (
    #     device == next(model_adversary.parameters()).device
    # ), "DThe device of the VAE and the adversary should be the same"

    all_train_data = timeseries_ds_formation(ds_train, episode_length, episode_gap)
    train_x = all_train_data[:, :, feature_columns]
    train_y = all_train_data[:, :, columns_to_hide]
    sequence_length = train_x.shape[1]
    num_priv_cols = train_y.shape[2]

    # Similarly for the validation
    # all_val_data = indiscriminate_supervision(ds_val)
    # val_x = all_val_data[:, columns_to_share]
    # val_y = all_val_data[:, columns_to_hide]
    # Validation data will not be shuffled since we vant to visualize the results in time series

    # TODO: Original Data for Validataion
    validation_episodes_list: List[np.ndarray] = validation_data_organization(
        ds_train, snapshot_length=episode_length, num_episodes=3
    )
    validation_episodes_tensor =  torch.from_numpy(np.stack(validation_episodes_list)).to(device)

    num_batches = train_x.shape[0] // batch_size
    logger.info(f"Working with {train_x.shape[0]} samples")
    recon_losses = []
    adv_losses = []
    d_sanitized_dist = []
    for e in range(epochs):
        logger.info(f"Epoch {e} of {epochs}")
        # losses_for_dist = []
        adversary_losses_batch = []
        for b in range(num_batches):
            # Now Get the new VAE generations
            model_vae_adversary.zero_grad()
            batch_x = torch.from_numpy(train_x[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            batch_y = torch.from_numpy(train_y[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            recon_data = batch_x.clone().requires_grad_(False)
            if batch_x.shape[0] != batch_size:
                continue
            logger.info(f"Batch {b} of {num_batches} with shape {batch_x.shape}")

            # Get the adversary to guess the sensitive column
            sanitized_data, adversary_guess_flat, kl_divergence = model_vae_adversary(batch_x)

            # logger.info(f"batch_x sum is {batch_x.sum()}")
            # plt.hist(batch_x.flatten().detach().cpu().numpy(), bins=100)
            # plt.title(f"Training Sanitized Episode Input Dist {e}")
            # plt.show()

            # This should be it
            recon_loss = F.mse_loss(sanitized_data, recon_data, reduction="none")
            batch_y_flat = batch_y.view(-1,batch_y.shape[-1])
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            # losses_for_dist.append(recon_loss.view(-1).tolist())
            loss = (recon_loss.mean(-1) + torch.log(1/(1.0e-10 + adv_loss)) - kl_dig_hypr * kl_divergence).mean()

            # model_adversary.zero_grad()
            # logger.info(f"Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")
            loss.backward()
            recon_losses.append(recon_loss.mean().item())
            adv_losses.append(adv_loss.mean().item())

            opt_vae.step()
            # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if b % 16 == 0:
                save_path = f"./figures/new_data_vae/plot_vaerecon_eval_{e:02d}_{b:02d}.png"
                metrics = validation_iteration(validation_episodes_tensor, columns_to_hide, model_vae_adversary, save_path)
                logger.info(f"Validation Metrics are {metrics}")

                # Plot the histogram of losses
                # plt.hist(losses_for_dist, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")

                # meepo = sanitized_data.view(-1).tolist()
                # plt.hist(meepo, bins=100)
                # plt.savefig(f"./figures/new_data_vae/plot_vaerecon_losses_{e:02d}_{b:02d}.png")
                # plt.close()

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(recon_losses)
    axs[0].set_title("Reconstruction Loss")
    axs[1].plot(adv_losses)
    axs[1].set_title("Adversary Loss")

    plt.savefig(f"./figures/new_data_vae/recon-adv_losses.png")

    plt.close()

    return model_vae_adversary

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
    optimizer = torch.optim.Adam(adversary.parameters(), lr=0.01) # type: ignore

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
    columns, runs_dict = load_defacto_data(args.defacto_data_raw_path)

    # Separate them into their splits (and also interpolate) 
    train_runs, val_runs, test_runs = split_defacto_runs(
        runs_dict,
        **args.splits,
    )

    logger.info(f"Using device is {device}")

    # Get Informaiton for the VAE
    vae_dimension = len(columns) - len(args.cols_to_hide)
    # TODO: Get the model going
    model_vae = SeqAdversarialVAE(
        # inout_size for model is output_dim for data
        input_size=vae_dimension,
        latent_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
        num_features_to_guess=1,
        rnn_num_layers=args.rnn_num_layers,
    ).to(device)

    model_discriminator = Adversary(
        input_size=vae_dimension,
        hidden_size=args.vae_hidden_size,
        num_classes=1,
    )

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
        args.lr,
        args.kl_dig_hypr,
        args.saveplot_dest,
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
