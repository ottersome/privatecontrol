"""
Script for training a model to learn the A,B,C Parameters behind a simulation. 
This is the pre-step to having it try to predict the previous step. 
"""

import argparse
import json
import logging
import os
from typing import Generator, Iterable, List, Tuple

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from conrecon.automated_generation import generate_state_space_system
from conrecon.models import SimpleModel
from conrecon.plotting import TrainLayout
from conrecon.utils.common import create_logger
from rich import inspect
from rich.console import Console
from rich.live import Live
from torch.nn import functional as F
from tqdm import tqdm

console = Console()


def plot_functions(outputs: np.ndarray, estimated_outputs: np.ndarray, save_path: str):
    assert len(outputs.shape) == 2, "Can only plot up to 2 outputs"
    assert len(estimated_outputs.shape) == 2, "Can only plot up to 2 outputs"
    num_outputs = outputs.shape[0]
    assert num_outputs <= 4, "Can only plot up to 4 outputs"
    fig, ax = plt.subplots(num_outputs, 2, figsize=(num_outputs * 6, 6))
    plt.tight_layout()

    for i in range(num_outputs):
        # Plot the outputs
        ax[i, 0].plot(estimated_outputs[i, :], label="Estimated")
        ax[i, 0].plot(outputs[i, :], label="True")
        ax[i, 0].set_xlabel("Time")
        ax[i, 0].set_ylabel("Output")
        ax[i, 0].set_title(f"Output {i+1}")
        ax[i, 0].legend()

        # Plot the error
        ax[i, 1].plot(np.abs(estimated_outputs[i, :] - outputs[i, :]), color="red")
        ax[i, 1].set_xlabel("Time")
        ax[i, 1].set_ylabel("Error")
        ax[i, 1].set_title(f"Output {i+1}")

    # Rather than showing them save them to a file
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument("-n", "--num_batches", default=64, type=int)
    ap.add_argument("-b", "--batch_size", default=128, type=int)
    ap.add_argument("--eval_size", default=4, help="How many systems to generate")
    ap.add_argument(
        "-e", "--epochs", default=10, help="How many epochs to train for", type=int
    )
    # Control stuff here
    ap.add_argument(
        "-t", "--time_steps", default=12, help="How many systems to generate"
    )
    ap.add_argument(
        "-s", "--state_size", default=6, help="Dimensionality of the state."
    )
    ap.add_argument("-i", "--input_dim", default=3, help="Dimensionality of the input.")
    ap.add_argument(
        "-g", "--num_of_gens", default=6, help="How many systems to generate"
    )
    ap.add_argument(
        "-o", "--num_outputs", default=1, help="Dimensionality of the output."
    )
    ap.add_argument("--ds_cache", default=".cache/ds.csv", type=str)
    ap.add_argument("--train_percent", default=0.8, type=float)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=10000, type=int)
    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.01, type=float)

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check


def get_sim(
    Amat: np.ndarray,
    Bmat: np.ndarray,
    Cmat: np.ndarray,
    init_cond: np.ndarray,
    time_length: int,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the matrices, get states and outputs
    """

    sys = ct.ss(Amat, Bmat, Cmat, 0)
    t = np.linspace(0, 10, time_length)
    u = np.zeros((input_dim, len(t)))
    # u[:, int(len(t) / 4)] = 1
    # TODO: think about this initial condition
    # Lets run the simulation and see what happens
    timepts = np.linspace(0, 10, time_length)
    response = ct.input_output_response(sys, timepts, u, X0=init_cond)  # type: ignore
    outputs = response.outputs
    states = response.states
    # inspect(outputs.shape)
    # inspect(states.shape)
    return states, outputs


def gen_n_sims(
    state_size: int,
    input_dim: int,
    num_outputs: int,
    time_steps: int,
    batch_size: int,
):

    np.random.seed(0)

    A, B, C, _ = generate_state_space_system(
        state_size,
        input_dim,
        num_outputs,
        state_size,
        batch_size,
    )

    # Generate the simulations
    hidden_truths = np.zeros(
        (
            batch_size,
            state_size,
            time_steps,
        )
    )
    system_outputs = np.zeros(
        (
            batch_size,
            num_outputs,
            time_steps,
        )
    )

    # Setup a bar
    for i in tqdm(range(batch_size)):
        # CHECK: Might be A[1]
        init_cond = np.random.uniform(0, 1, A.shape[0])
        results = get_sim(A, B, C, init_cond, time_steps, input_dim)
        hidden_truths[i, :, :] = results[0]
        system_outputs[i, :, :] = results[1]

    return hidden_truths, system_outputs


def generate_dataset(
    cache_path: str,
    state_dim: int,
    num_outputs: int,
    input_dim: int,
    time_steps: int,
    batch_size: int,
    ds_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # This will generate batch_size at a time and save as a dataset
    columns = [f"h{i}" for i in range(state_dim)]
    columns += [f"y{i}" for i in range(num_outputs)]

    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from {cache_path}")
        final_dataset = pd.read_csv(cache_path)
        # Read the json file to get the metadata
        with open(cache_path.replace(".csv", ".json"), "r") as f:
            metadata = json.load(f)
            state_dim = metadata["state_size"]
            output_dim = metadata["num_outputs"]
            ds_size = metadata["ds_size"]
            time_steps = metadata["time_steps"]

        hiddens = (
            final_dataset.iloc[:, :state_dim]
            .values.reshape((ds_size, time_steps, state_dim))
            .astype(np.float32)
        )
        outputs = (
            final_dataset.iloc[:, state_dim:]
            .values.reshape((ds_size, time_steps, output_dim))
            .astype(np.float32)
        )
        return hiddens, outputs

    logger.info(f"Generating dataset to {cache_path}")
    # TODO: Batch this out int batch_size for long enough ds_size
    hiddens, outputs = gen_n_sims(
        state_dim,
        input_dim,
        num_outputs,
        time_steps,
        ds_size,
    )
    hiddens_transposed = hiddens.transpose(0, 2, 1)
    outputs_transposed = outputs.transpose(0, 2, 1)
    hiddens_final = hiddens_transposed.reshape((ds_size * time_steps, state_dim))
    outputs_final = outputs_transposed.reshape((ds_size * time_steps, num_outputs))
    # Form hiddens and outputs into a dataframe
    hiddens = pd.DataFrame(
        hiddens_final,
        columns=columns[:state_dim],
    )
    outputs = pd.DataFrame(outputs_final, columns=columns[state_dim:])
    final_dataset = pd.concat([hiddens, outputs], axis=1)
    final_dataset.to_csv(cache_path, index=False)
    # Save metadata to json file
    json_file = cache_path.replace(".csv", ".json")
    metadata = {
        "state_size": state_dim,
        "num_outputs": num_outputs,
        "input_dim": input_dim,
        "time_steps": time_steps,
        "ds_size": ds_size,
    }
    with open(json_file, "w") as f:
        json.dump(metadata, f)

    return hiddens.values, outputs.values


def train(
    model: nn.Module,
    tdataset: Tuple[np.ndarray, np.ndarray],
    vdataset: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    fig_savedest: str,
    device: torch.device,
) -> Generator[Tuple[int, int, float, float], None, None]:
    """
    Train the RNN
    """
    model.train()
    trainx = torch.from_numpy(tdataset[0]).to(device)
    trainy = torch.from_numpy(tdataset[1]).to(device)
    valx = torch.from_numpy(vdataset[0]).to(device)
    valy = torch.from_numpy(vdataset[1]).to(device)
    len_trainsamp = trainx.shape[0]
    logger.debug(
        f"Finally got all the divisions. We have the following sizes:\n"
        f"trainx: {trainx.shape} trainy: {trainy.shape} valx: {valx.shape} valy: {valy.shape}"
    )

    num_batches = len_trainsamp // batch_size
    logger.debug(f"Number of batches: {num_batches}")
    evalevery_n_batches = int(num_batches / 10)

    for e in range(epochs):
        logger.debug(f"Inside the epoch {e}")
        for i in range(num_batches):
            logger.debug(f"Getting the batch stsuff")
            x = trainx[i * batch_size : (i + 1) * batch_size]
            y = trainy[i * batch_size : (i + 1) * batch_size]
            # CHECK: Perhaps if last batch is too small
            # Let me inspect hidden outputs
            logger.debug(f"About to present {x.shape}-sized input to model ")
            inf = model(x)
            logger.debug(f"Model came up with an inference of size {inf.shape}")
            loss = criterion(inf, y)
            loss.backward()
            logger.debug(f"Passed the backward pass")

            grad_min = float("inf")
            grad_max = float("-inf")
            for param in model.parameters():
                if param.grad is not None:
                    grad_min = min(grad_min, param.grad.min().item())
                    grad_max = max(grad_max, param.grad.max().item())
            # print(f"Gradient Min: {grad_min}")
            # print(f"Gradient Max: {grad_max}")

            # Get gradients
            optimizer.step()
            tloss = loss.mean().detach().cpu().item()
            vloss = None

            ## Evaluation Loop
            if e % evalevery_n_batches == 0:
                model.eval()
                # TODO: Retrieve validation data

                # outputs, hidden_states = model(hidden_truths, system_outputs)# For RNN
                inf = model(valx)  # For RN

                loss = criterion(inf, valy)
                vloss = loss.detach().cpu().item()
                vloss = loss.mean().item()

            yield e, i, tloss, vloss

        # TODO: Recover this
        logger.debug(
            f"At this point we see  shape of {valx.shape} for valx\n"
            f"and shape of {valy.shape} for valy"
        )
        y_val = model(valx[:3, :, :].squeeze()).squeeze().cpu().detach().numpy()
        logger.debug(f"Shape of y_val: {y_val.shape}")
        plot_functions(
            valy[:3, :, :].squeeze().cpu().detach().numpy(),
            y_val,
            save_path=f"{fig_savedest}/plot_{e:02d}.png",
        )


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


if __name__ == "__main__":

    args = argsies()
    # Get out matrices
    logger = create_logger("__main__")
    logger.info(f"Loading dataset")

    inputs, outputs = generate_dataset(
        args.ds_cache,
        args.state_size,
        args.num_outputs,
        args.input_dim,
        args.time_steps,
        args.batch_size,
        args.ds_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    # model = RecoveryNet(
    #     args.state_size, args.state_size, args.num_outputs, args.time_steps

    model = SimpleModel(
        args.state_size, args.state_size, args.num_outputs, args.time_steps
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ct.use_fbs_defaults()  # Use settings to match FBS

    # Start training
    train_w_metrics(
        model,
        (inputs, outputs),
        args.epochs,
        args.batch_size,
        args.saveplot_dest,
        args.train_percent,
        device,
    )
