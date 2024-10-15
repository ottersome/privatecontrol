import argparse
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import random
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
from sktime.libs.pykalman import KalmanFilter
from torch import nn, tensor
from torch.nn import functional as F
from tqdm import tqdm
import pdb
from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.vae import FlexibleVAE
from conrecon.kalman.mo_core import Filter
from conrecon.plotting import TrainLayout, plot_functions, plot_functions_2by1
from conrecon.ss_generation import hand_design_matrices
from conrecon.utils import create_logger
from conrecon.models.models import SimpleRegressionModel

traceback.install()

console = Console()

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=3, help="How many epochs to train for", type=int
    )
    ap.add_argument("--defacto_data_raw_path", default="./data/", type=str, help="Where to load the data from")
    ap.add_argument("--batch_size", default=32)
    ap.add_argument("--cols_to_hide", default=[4], help="Which are the columsn we want no information of") # Remember 0-index (so 5th)
    ap.add_argument("--vae_latent_size", default=10, type=int)
    ap.add_argument("--vae_hidden_size", default=128, type=int)
    ap.add_argument("--splits", default= { "train_split": 0.8, "val_split" : 0.2, "test_split" : 0.0 } , type=list, nargs="+")

    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.001, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)

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

def validation_iteration(val_x: np.ndarray, val_y: np.ndarray, model: nn.Module, batch_size: int = 64) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - val_x: Validation data
        - val_y: Validation data
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    model_device = next(model.parameters()).device
    batchehs = len(val_x) // batch_size
    for b in range(batchehs):
        batch_x = torch.Tensor(val_x[b * batch_size : (b + 1) * batch_size]).to(model_device)
        batch_y = torch.Tensor(val_y[b * batch_size : (b + 1) * batch_size]).to(model_device)
        if len(batch_x.shape) != batch_size:
            continue
        sanitized_data = model(torch.from_numpy(batch_x))
        recon_loss = F.mse_loss(sanitized_data, batch_x)
        adv_loss = F.mse_loss(model(sanitized_data), batch_y)
        metrics["recon_loss"].append(recon_loss.item())
        metrics["adv_loss"].append(adv_loss.item())

    return {k: np.mean(v).item() for k, v in metrics.items()}


# TODO: Later change the name of the function
def train_v0(
    batch_size: int,
    columns_to_hide: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model_adversary: SimpleRegressionModel,
    model_vae: FlexibleVAE,
    # Some Extra Params
    saveplot_dest: str,
):
    columns_to_share = list(set(range(len(data_columns))) - set(columns_to_hide))
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=0.001) # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=0.001) # type: ignore

    device = next(model_vae.parameters()).device
    assert (
        device == next(model_adversary.parameters()).device
    ), "DThe device of the VAE and the adversary should be the same"

    all_train_data = indiscriminate_supervision(ds_train)
    train_x = all_train_data[:, columns_to_share]
    train_y = all_train_data[:, columns_to_hide]

    # Similarly for the validation
    all_val_data = indiscriminate_supervision(ds_val)
    val_x = all_val_data[:, columns_to_share]
    val_y = all_val_data[:, columns_to_hide]

    batches = train_x.shape[0] // batch_size
    recon_losses = []
    for e in range(epochs):
        logger.info(f"Epoch {e} of {epochs}")
        for b in range(batches):
            # Now Get the new VAE generations
            batch_x = torch.from_numpy(train_x[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            batch_y = torch.from_numpy(train_y[b * batch_size : (b + 1) * batch_size]).to(torch.float32).to(device)
            if batch_x.shape[0] != batch_size:
                continue
            logger.info(f"Batch {b} of {batches} with shape {batch_x.shape}")

            sanitized_data = model_vae(batch_x)
            adversary_guess = model_adversary(sanitized_data)

            # This should be it 
            recon_loss = F.mse_loss(sanitized_data, batch_x)
            adv_loss = F.mse_loss(adversary_guess, batch_y)
            loss = recon_loss - adv_loss
            loss.backward()

            model_vae.zero_grad()
            model_adversary.zero_grad()
            loss.backward()

            # TODO: Check. This feels on the ilegal side lol
            opt_vae.step()
            opt_adversary.step()

            logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if b % 4 == 0:
                metrics = validation_iteration(val_x, val_y, model_vae)
                logger.info(f"Validation Metrics are {metrics}")
            
    return model_vae, model_adversary



def train_v1():
    # TODO: This one will consider better the correlation between these things.
    raise NotImplementedError

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
    optimizer = torch.optim.Adam(adversary.parameters(), lr=0.001) # type: ignore

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




def main():
    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(int(time.time()))
    logger = create_logger("main_training")

    # TODO: Make it  so that generate_dataset checks if params are the same
    columns, runs_dict = load_defacto_data(args.defacto_data_raw_path)

    # Separate them into their splits
    train_runs, val_runs, test_runs = split_defacto_runs(
        runs_dict,
        **args.splits,
    )

    logger.info(f"Using device is {device}")

    # Get Informaiton for the VAE
    vae_dimension = len(columns) - len(args.cols_to_hide)
    # TODO: Get the model going
    model_vae = FlexibleVAE(
        # inout_size for model is output_dim for data
        input_size=vae_dimension,
        latent_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
    ).to(device)
    model_adversary = SimpleRegressionModel(
        input_size=vae_dimension, latent_size=args.vae_latent_size, output_size=1
    ).to(device)

    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    logger.info("Starting the VAE Training")
    vae = train_v0(
        args.batch_size,
        args.cols_to_hide,
        columns,
        device,
        train_runs,
        val_runs,
        args.epochs,
        model_adversary,
        model_vae,
        args.saveplot_dest,
    )

    # 🚩 Development so far🚩
    exit()

    # TODO: We might want to do a test run here 
    # if len(test_runs) > 0:
    #     test_runs = train_new(test_runs)

    # trainVAE_wprivacy(
    #     training_data,
    #     (hidden, outputs),
    #     args.epochs,
    #     args.saveplot_dest,
    #     [True,False,True]
    # )
    #
if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
