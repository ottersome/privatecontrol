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
from tqdm import tqdm
import pandas as pd
import wandb

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.data.dataset_generation import batch_generation_randUni, collect_n_sequential_batches, spot_backhistory
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE, SequenceToScalarVAE
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger

traceback.install()

console = Console()

wandb_on = False

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
    ap.add_argument(
        "--shuffle",
        action="store_false",
        help="Whether or not to shuffle the data",
    )
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="Whether or not to use wandb for logging",
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

def test_entire_file(
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
    truth_to_compare = validation_file[:, some_8_idxs]
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

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})
        wandb.log({"truth": adv_truth.squeeze().detach().cpu().numpy()})

    model_vae.train()
    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}

def compare_reconstruction():
    """
    Will take the original set of features and
    """
    raise NotImplementedError

def plot_training_losses(recon_losses: List, adv_losses: List):
    logger.info("Plotting the training losses")
    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].plot(recon_losses)
    axs[0].set_title("Reconstruction Loss")
    axs[1].plot(adv_losses)
    axs[1].set_title("Adversary Loss")

    plt.savefig(f"./figures/new_data_vae/recon-adv_losses.png")
    plt.close()

def train_v1(
    batch_size: int,
    prv_columns: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model_vae: SequenceToScalarVAE,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Training Loop
        ds_train: np.ndarray (num_batches, batch_size, features),
    """

    pub_columns = list(set(range(len(data_columns))) - set(prv_columns))
    device = next(model_vae.parameters()).device

    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=learning_rate)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)  # type: ignore

    ##  A bit of extra processing of data. (Specific to this version of training)
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([ seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    batch_amnt  = all_train_seqs.shape[0] // batch_size
    # all_train_seqs = all_train_seqs.reshape(batch_amnt, batch_size, all_train_seqs.shape[-2], all_train_seqs.shape[-1])
    # all_valid_seqs = all_valid_seqs.reshape(batch_amnt, batch_size, all_valid_seqs.shape[-2], all_valid_seqs.shape[-1])
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:,:,pub_columns]
    train_prv = all_train_seqs[:,:,prv_columns]

    num_batches     = ceil(all_train_seqs.shape[0] / batch_size)
    sequence_len    = all_train_seqs.shape[1]

    ########################################
    # Get Batches
    ########################################
    logger.info(f"Working with {num_batches} num_batches, each with size:  {batch_size}, and sequence/episode length {sequence_len}")
    recon_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        logger.info(f"Epoch {e} of {epochs}")
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_all = all_train_seqs[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            logger.info(f"Batch {batch_no} out of {num_batches} with shape {batch_pub.shape}")

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, _ = model_vae(batch_all)

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)

            # Check on performance
            batch_y_flat = batch_prv[:,-1,:].view(-1, batch_prv.shape[-1]) # Grab only last in sequeence
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            model_adversary.zero_grad()
            adv_train_loss.backward()
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
            batch_y_flat = batch_prv[:,-1,:].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data[:, pub_columns], pub_prediction)
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = pub_recon_loss - 4.0 * adv_loss + kl_dig_hypr * kl_divergence.mean()

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adv_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            opt_vae.step()
            # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if wandb_on:
                wandb.log({
                    "adv_train_loss": adv_train_loss.item(),
                    "pub_recon_loss": pub_recon_loss.item(),
                    "final_loss_scalar": final_loss_scalar.item(),
                })

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                model_vae.eval()
                model_adversary.eval()
                with torch.no_grad():
                    validation_metrics = (
                        calculate_validation_metrics(
                            all_valid_seqs,
                            pub_columns,
                            prv_columns,
                            model_vae,
                            model_adversary,
                        )
                    )
                model_vae.train()
                model_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)

    return model_vae, model_adversary, recon_losses, adv_losses


# TODO: We need to implement federated learning in this particular part of the expression
def federated():
    # We also need a federated aspect to all this. And its getting close to being time to implementing this
    raise NotImplementedError


def calculate_validation_metrics(
    all_features: torch.Tensor,
    pub_features_idxs: List[int],
    prv_features_idxs: List[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
) -> Dict[str, float]:
    """
    We use correlation here as our delta-epsilon metric.
    """
    prv_features = all_features[:, :, prv_features_idxs]
    pub_features = all_features[:, :, pub_features_idxs]

    # Run data through models.
    latent_z, sanitized_data, kl_divergence = model_vae(all_features)
    recon_pub = sanitized_data[:, pub_features_idxs]
    recon_priv = model_adversary(latent_z)

    # Lets just do MSE for now
    pub_mse = torch.mean((pub_features[:, -1, :] - recon_pub) ** 2)
    prv_mse = torch.mean((prv_features[:, -1, :] - recon_priv) ** 2)

    # TODO: Do torch equivalent so we can get the correlation coefficient of the sequences predcicted
    # corr_pub = torch.corrcoef(pub_features[:, -1, :].flatten(), recon_pub.flatten())[0, 1]
    # corr_prv = torch.corrcoef(prv_features[:, -1, :].flatten(), recon_priv.flatten())[0, 1]
    validation_metrics = {
        "pub_mse": pub_mse.item(),
        "prv_mse": prv_mse.item(),
    }

    return validation_metrics



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

def trivial_correlation_baseline(test_runs: OrderedDict[str, np.ndarray], pretrained_adversary: nn.Module):
    """
    Will try to remove a column and simply try to predeict it out of the other ones. 
    """
    pass


def kosambi_karhunen_loeve_baseline(test_runs: OrderedDict[str, np.ndarray], pretrained_adversary: nn.Module):
    """
    Kosambi Karhunen Loeve baseline
    """
    pass

def main():
    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)
    logger = create_logger("main_training")
    global wandb_on

    if args.wandb:
        wandb_on = True
        wandb.init(project="private_control", config=args)

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", args.debug_port))
        debugpy.wait_for_client()

    # TODO: Make it  so that generate_dataset checks if params are the same
    columns, runs_dict, debug_file = load_defacto_data(args.defacto_data_raw_path)
    num_columns = len(columns)
    num_private_cols = len(args.cols_to_hide)

    # Separate them into their splits (and also interpolate)
    train_batches, val_batches, test_file = split_defacto_runs(
        runs_dict,
        args.splits["train_split"],
        args.splits["val_split"],
        args.episode_length,
        True, # Scale
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

    ########################################
    # Training
    ########################################
    logger.info("Starting the VAE Training")
    model_vae, model_adversary, recon_losses, adv_losses = train_v1(
        args.batch_size,
        args.cols_to_hide,
        columns,
        device,
        train_batches,
        val_batches,
        args.epochs,
        model_vae,
        model_adversary,
        args.lr,
        args.kl_dig_hypr,
    )

    ########################################
    # Evaluation
    ########################################
    save_path = (
        f"./figures/new_data_vae/plot_vaerecon_eval_{e:02d}_{b:02d}.png"
    )

    plot_training_losses(recon_losses, adv_losses)
    # TODO: Move this to a test 
    metrics = test_entire_file(
        test_file,
        args.cols_to_hide,
        model_vae,
        model_adversary,
        args.episode_length,
        debug_file,
        args.padding_value,
        save_path,
    )
    logger.info(f"Validation Metrics are {metrics}")

    ########################################
    # Benchmarks
    ########################################
    trivial_correlation_baseline(test_runs, model_adversary) 

    kd_transform_baseline(test_runs, model_adversary)

    # 🚩 development so far🚩
    exit()

if __name__ == "__main__":
    logger = create_logger("main_vae")


    main()
