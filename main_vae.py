import argparse
import json
import os
import pdb
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from rich import inspect, traceback
from rich.console import Console
from rich.live import Live
from sktime.libs.pykalman import KalmanFilter
from torch import nn, tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm

from conrecon.ss_generation import SSParam, hand_design_matrices
from conrecon.dplearning.vae import VAE, FlexibleVAE, RecurrentVAE
from conrecon.models.transformers import TorchsTransformer, TransformerBlock
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger, set_seeds
from conrecon.plotting import plot_functions
from conrecon.data.dataset_generation import TrainingMetaData, generate_dataset
from conrecon.kalman.mo_core import Filter

traceback.install()

console = Console()

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=3, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--num_layers", default=1, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--eval_interval", default=100, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "-n", "--num_samples", default=4, type=int, help="How many Samples to Evaluate"
    )
    ap.add_argument(
        "--eval_size", default=4, help="How many systems to generate", type=int
    )
    # Control stuff here
    ap.add_argument(
        "-t", "--time_steps", default=12, help="How many systems to generate", type=int
    )
    ap.add_argument(
        "-s", "--state_dim", default=3, help="Dimensionality of the state.", type=int
    )
    ap.add_argument(
        "-i", "--input_dim", default=3, help="Dimensionality of the input.", type=int
    )
    ap.add_argument(
        "-o", "--output_dim", default=2, help="Dimensionality of the output.", type=int
    )
    ap.add_argument("--ds_cache", default=".cache/pykalpkg_ds.csv", type=str)
    ap.add_argument("--vae_ds_cache", default=".cache/pykalpkg_vaeds.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/pykalman_transformer/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=10000, type=int)
    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.001, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check



#💫 Main function of interest
def trainVAE_wprivacy(
    training_metadata: TrainingMetaData,
    learning_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    plot_dest: str,
    batch_size = 64,
    tt_split: float = 0.8,
    vae_latent_size: int = 10,
    vae_hidden_size: int = 128,
):
    ### Learning Objects
    vae = FlexibleVAE(
        input_size=training_metadata.input_dim,
        latent_size=vae_latent_size,
        hidden_size=vae_hidden_size,
    )
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001) # type: ignore

    ### Data Management
    states, outputs = learning_data # x, y
    device = states.device
    logger.info(f"Using device is {device}")
    tstates, toutputs = tensor(states), tensor(outputs) # t(x), t(y)
    trn_states, val_states = torch.split(
        tstates,
        [int(len(states) * tt_split), len(states) - int(len(states) * tt_split)],
    )
    trn_outputs, val_outputs = torch.split(
        toutputs,
        [int(len(outputs) * tt_split), len(outputs) - int(len(outputs) * tt_split)],
    )
    t_val_states, t_val_outputs = tensor(val_states), tensor(val_outputs)

    ### Filter Data
    A, C, = training_metadata.params.A, training_metadata.params.C
    B = training_metadata.params.B
    kf = KalmanFilter(transition_matrices=A, observation_matrices=C)
    t_data, val_data = learning_data

    # My Filter
    torch_filter = Filter(
        transition_matrix=torch.from_numpy(A).to(torch.float32),
        observation_matrix=torch.from_numpy(C).to(torch.float32),
        input_matrix=torch.from_numpy(B).to(torch.float32),
        batch_size=batch_size,
    )
    # Native Filter
    kf = KalmanFilter(
        transition_matrices=training_metadata.params[0],
        observation_matrices=training_metadata.params[2],
    )

    ### Train the VAE
    criterion = nn.MSELoss()  # TODO: Change this to something else
    loss_list = []
    eval_data = []
    batch_count = int(len(t_data) / batch_size)
    train_layout = TrainLayout(epochs, batch_count, loss_list, eval_data)
    with Live(train_layout.layout, console=console, refresh_per_second=10) as live:
        for e in range(epochs):
            epoch_loss = 0
            for b in range(batch_count):
                ## BatchdWise Data
                cur_state = t_data[b * batch_size : (b + 1) * batch_size]
                t_cur_state = torch.from_numpy(cur_state).to(device)
                cur_output = trn_outputs[b * batch_size : (b + 1) * batch_size].to(
                    device
                )
                ## Change Data to HideStuff
                masked_output = vae(cur_output)
                state_estimates_w_vae = []
                state_estimates_wo_vae = []

                logger.debug(f"Tell em about the shape of the output {cur_output.shape} as well as its type {type(cur_output)}")
                logger.debug(f"Shape of masked_output is {masked_output.shape} as well as its type {type(masked_output)}")

                # Go Through Batch
                for i in range(cur_output.shape[0]):
                    logger.debug(f"Going through the batch {i}")
                    # First without VAE
                    (filtered_mean, filtered_covariance) = kf.filter(
                        cur_output[i, :, :].squeeze().detach().cpu().numpy()
                    )
                    (smoothed_mean, smoothed_covariance) = kf.smooth(
                        cur_output[i, :, :].squeeze().detach().cpu().numpy()
                    )
                    state_estimates_wo_vae.append(smoothed_mean)

                # Then for VAE
                state_estimates_wo_vae = torch.tensor(state_estimates_wo_vae).to(device)
                state_estimates_w_vae = torch_filter(cur_output)
                logger.debug(f"Done with the batch. Will add more stuff in a minute")

                ## Try to do reconstruction of state
                # state_estimates_wo_vae = torch.from_numpy(
                #     np.array(state_estimates_wo_vae)
                # ).to(device)
                # state_estimates_w_vae = torch.from_numpy(
                #     np.array(state_estimates_w_vae)
                # ).to(device)
                # diff = torch.abs(state_estimates_wo_vae - state_estimates_w_vae)
                # Show the differences in a plot
                # Now we will give our objective. We try to hide how close it is from the true source
                # Say we want to hide the last stage. 

                # CHECK: This will likely need a torch based implementation to 
                # have the computation graph involved
                logger.debug(f"Before loss estimation")
                similarities = F.mse_loss(state_estimates_w_vae[:,:,:-1], t_cur_state[:,:,:-1])
                diff = - F.mse_loss(state_estimates_wo_vae[:,:,-1], t_cur_state[:,:,-1])
                final_loss = similarities + diff
                fl_mean = final_loss.mean()
                loss_list.append(fl_mean.item())
                cur_loss = final_loss.mean().item()
                
                logger.debug(f"Before optimizing.")
                optimizer.zero_grad()
                fl_mean.backward()
                optimizer.step()

                logger.debug(f"Loss at epoch {e} batch {b} is {fl_mean.item()}")

                ## TODO: We need some sort of knob here to play with utility vs privacy
                train_layout.update(e, b, cur_loss, None)

            # Normal Reporting
            if (e + 1) % 1 == 0:
                print("Epoch: {}, Loss: {:.5f}".format(e + 1, epoch_loss))

    # We test with MSE for reconstruction for now
def main():
    args = argsies()
    # Get our A, B, C Matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = hand_design_matrices()
    np.random.seed(int(time.time()))

    logger.info(f"Generating dataset")

    # TODO: Make it  so that generate_dataset checks if params are the same
    hidden, outputs, training_data = generate_dataset(
        params,
        args.ds_cache,
        args.state_dim,
        args.input_dim,
        args.output_dim,
        args.time_steps,
        args.ds_size,
    )

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    # 🚩 Development so far🚩

    trainVAE_wprivacy(
        training_data,
        (hidden, outputs),
        args.epochs,
        args.saveplot_dest,
    )

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
