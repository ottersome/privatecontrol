import argparse
import os
import time
from typing import List, Tuple, Dict

import numpy as np
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
    pdb.set_trace()
    final_ds = np.random.shuffle(final_ds)
    return np.array(final_ds)


# TODO: Later change the name of the function
def train_v0(
    batch_size: int,
    columns_to_hide: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model: nn.Module,
    # Some Extra Params
    saveplot_dest: str,
):
    # Dataset comes preloaded as a DF
    # Organize some of the datasets
    # CHECK: this actually works
    pdb.set_trace()
    columns_to_share = list(set(range(len(data_columns))) - set(columns_to_hide))

    all_train_data = indiscriminate_supervision(ds_train)
    train_x = all_train_data[:, columns_to_share]
    train_y = all_train_data[:, columns_to_hide]

    # Similarly for the validation
    all_val_data = indiscriminate_supervision(ds_val)
    val_x = all_val_data[:, columns_to_share]
    val_y = all_val_data[:, columns_to_hide]

    # Now onwards with the model

def train_v1():
    # TODO: This one will consider better the correlation between these things.
    raise NotImplementedError

# TODO: We need to implement federated learning in this particular part of the expression
def federated():
    # We also need a federated aspect to all this. And its getting close to being time to implementing this
    raise NotImplementedError

class RecoveryNet(nn.Module):

    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()
        self.mean = torch.zeros(input_size, time_steps)
        self.variance = torch.zeros(input_size, time_steps)
        self.rnn = torch.nn.GRU(input_size, state_size, batch_first=True)
        # Final output layer
        self.output_layer = torch.nn.Linear(state_size, num_outputs)
        self.count = 0
        self.batch_norm = torch.nn.BatchNorm1d(num_features=input_size)

    def forward(self, x):
        # Normalize x
        # self.update(x)
        # normed_x = self.batch_norm(x)
        # norm_x = (x - self.mean) / (self.variance + 1e-8).sqrt()
        transposed_x = x.transpose(1, 2)
        logger.debug(f"Tranposed x looks like {transposed_x}")
        rnnout, hidden = self.rnn(transposed_x)
        logger.debug(f"RNN output looks like: {rnnout}")
        return self.output_layer(F.relu(rnnout)), hidden

    def update(self, x):
        self.count += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        if self.count == 1:
            self.mean = batch_mean
        else:
            old_mean = self.mean
            self.mean = (old_mean * (self.count - 1) + batch_mean) / self.count
            delta = batch_mean - old_mean
            self.variance = (self.variance * (self.count - 1) + batch_var) / self.count

            # self.variance = (
            #     self.variance * (self.count - 1)
            #     + (x - old_mean - delta).pow(2).sum(dim=0)
            # ) / self.count




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

    # Get Informaiton for the VAE 
    vae_dimension =  len(columns) - len(args.cols_to_hide)
    # TODO: Get the model going
    model = FlexibleVAE(
        # inout_size for model is output_dim for data
        input_size=vae_dimension,
        latent_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
    ).to(device)


    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # type: ignore

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    vae = train_new(
        args.batch_size,
        args.cols_to_hide,
        columns,
        device,
        train_runs,
        val_runs,
        args.epochs,
        model,
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
