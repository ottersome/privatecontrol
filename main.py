import argparse
import logging
import os
import shutil
from typing import Generator, Iterable, List, Tuple

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich import inspect
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from torch.nn import functional as F

from conrecon.automated_generation import generate_state_space_systems
from conrecon.models import RecoveryNet, RecoveryTrans, create_logger

console = Console()


class Plot:
    def __init__(self, width, height):
        self.height = height
        self.metric = []
        self.logger = create_logger("plot")

    def add_measurement(self, data):
        self.metric.append(data)

    def render(self):

        # Get terminal width
        term_width = shutil.get_terminal_size().columns
        width = min(len(self.metric), term_width)

        lines = [[" "] * width for _ in range(self.height)]
        if len(self.metric) <= 1:
            return "\n".join(["".join(l) for l in lines])
        max_value = max(self.metric)
        min_value = min(self.metric)

        # for j, value in enumerate(self.metric):
        for j in range(width):
            metric_idx = int(j * (len(self.metric) - 1) / (width - 1))
            value = self.metric[metric_idx]
            normed_thresh = int(value * (self.height - 0) / (max_value - min_value))
            for i in range(self.height):
                ai = self.height - i - 1
                if i <= normed_thresh:
                    lines[ai][j] = "█"

        string_dump = "\n".join(["".join(l) for l in lines])
        return string_dump

    def __rich__(self):
        return Text(self.render())


def plot_functions(outputs: np.ndarray, estimated_outputs: np.ndarray, save_path: str):
    num_outputs = outputs.shape[0]
    assert num_outputs <= 4, "Can only plot up to 4 outputs"
    fig, ax = plt.subplots(num_outputs, 2, figsize=(num_outputs * 6, 6))
    plt.tight_layout()

    for i in range(num_outputs):
        # Plot the outputs
        ax[i, 0].plot(estimated_outputs[i, :, :].squeeze(), label="Estimated")
        ax[i, 0].plot(outputs[i, :, :].squeeze(), label="True")
        ax[i, 0].set_xlabel("Time")
        ax[i, 0].set_ylabel("Output")
        ax[i, 0].set_title(f"Output {i+1}")
        ax[i, 0].legend()

        # Plot the error
        ax[i, 1].plot(
            np.abs(estimated_outputs[i, :, :] - outputs[i, :, :]).squeeze(), color="red"
        )
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
    ap.add_argument("-b", "--batch_size", default=32, type=int)
    ap.add_argument("-e", "--eval_size", default=4, help="How many systems to generate")
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
    ap.add_argument(
        "--save_destination",
        default="./figures/",
        help="Where to save the outputs",
    )
    ap.add_argument("--no-autoindent")

    args = ap.parse_args()

    if not os.path.exists(args.save_destination):
        os.makedirs(args.save_destination)
    return args
    # Sanity check


def get_n_sims(
    Amat: np.ndarray,
    Bmat: np.ndarray,
    Cmat: np.ndarray,
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
    init_cond = np.random.uniform(0, 1, sys.nstates)
    # Lets run the simulation and see what happens
    timepts = np.linspace(0, 10, time_length)
    response = ct.input_output_response(sys, timepts, u, X0=init_cond)  # type: ignore
    outputs = response.outputs
    states = response.states
    # inspect(outputs.shape)
    # inspect(states.shape)
    return states, outputs


def batch_wise_datagen(
    state_size: int, input_dim: int, num_outputs: int, time_steps: int, batch_size: int
):
    # Let me start with an RNN
    Amats, Bmats, Cmats, Dmats = generate_state_space_systems(
        state_size,
        input_dim,
        num_outputs,
        state_size,
        batch_size,
    )

    # Generate the simulations
    hidden_truths = torch.zeros(
        batch_size,
        state_size,
        time_steps,
        device=device,
    )
    system_outputs = torch.zeros(
        batch_size,
        num_outputs,
        time_steps,
        device=device,
    )

    for i in range(batch_size):
        results = get_n_sims(Amats[i], Bmats[i], Cmats[i], time_steps, input_dim)
        hidden_truths[i, :, :] = torch.from_numpy(results[0])
        system_outputs[i, :, :] = torch.from_numpy(results[1])

    return hidden_truths, system_outputs


def train(
    model: nn.Module, args: argparse.Namespace, device: torch.device
) -> Generator[Tuple[int, float, float], None, None]:
    """
    Train the RNN
    """
    model.train()
    for i in range(args.num_batches):

        hidden_truths, system_outputs = batch_wise_datagen(
            args.state_size,
            args.input_dim,
            args.num_outputs,
            args.time_steps,
            args.batch_size,
        )
        model.zero_grad()
        hidden_truths = hidden_truths.to(device)
        system_outputs = system_outputs.to(device)
        # inspect(hidden_truths)

        # Let me inspect hidden outputs
        logger.debug(
            f"Systems hidden max and min: {hidden_truths.max()} and {hidden_truths.min()}"
        )
        logger.debug(f"Hidden Truths ({hidden_truths.shape}): {hidden_truths}")
        nan_idx = torch.where(torch.isnan(hidden_truths))
        logger.debug(f"Hidden Truths Nan idx: {nan_idx}")
        outputs, hidden_states = model(hidden_truths)
        nan_idx = torch.where(torch.isnan(outputs))
        logger.debug(f"Outputs ({outputs.shape}): {outputs}")
        logger.debug(f"Output Nan Nan idx: {nan_idx}")
        # CHECK: Loss i performed in the right dimensions
        # Log pretty the shapes using rich
        transposed_system_outputs = system_outputs.transpose(1, 2)
        logger.debug(f"Shape of system_outputs: {system_outputs.shape}")

        loss = criterion(outputs, transposed_system_outputs)
        loss.backward()

        grad_min = float("inf")
        grad_max = float("-inf")
        for param in model.parameters():
            if param.grad is not None:
                grad_min = min(grad_min, param.grad.min().item())
                grad_max = max(grad_max, param.grad.max().item())
        print(f"Gradient Min: {grad_min}")
        print(f"Gradient Max: {grad_max}")

        # Get gradients
        logger.debug(f"Gradients: min and max {grad_min} and {grad_max}")
        optimizer.step()
        tloss = loss.mean().detach().cpu().item()
        logger.debug(f"Loss: {tloss}")
        # tlosses.append(loss.item())
        # mean_train_loss = np.mean(tlosses).item()
        # tlosses.append(mean_train_loss)

        ## Evaluation Loop
        model.eval()
        hidden_truths, system_outputs = batch_wise_datagen(
            args.state_size,
            args.input_dim,
            args.num_outputs,
            args.time_steps,
            args.batch_size,
        )

        outputs, hidden_states = model(hidden_truths)

        transposed_system_outputs = system_outputs.transpose(1, 2)
        loss = criterion(outputs, transposed_system_outputs)
        vloss = loss.detach().cpu().item()
        mean_eval_loss = loss.mean().item()

        # DEBUG:
        plot_functions(
            system_outputs.cpu().detach().numpy()[:3, :, :],
            outputs.cpu().detach().numpy()[:3, :, :],
            save_path=f"{args.save_destination}/plot_{i}.png",
        )

        yield i, tloss, vloss


class MyLayout:
    def __init__(self, epoch: int, tlosses: List[float], vlosses: List[float]):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("[cyan]Training..", total=args.num_batches)
        self.layout = Layout()
        self.table = Table(title=f"Epoch {epoch}")
        self.table.add_column("Metric")
        self.table.add_column("Value")
        if len(tlosses) > 1:
            self.table.add_row("Training Loss", f"{tlosses[-1]:.3f}")
            self.table.add_row("Validation Loss", f"{vlosses[-1]:.3f}")

        self.plot = Plot(width=60, height=20)
        # self.plot.add_series(tlosses, label="Loss", color="red")

        self.layout.split(
            Layout(self.progress, name="progress", size=3),
            # Layout(Panel(plot, title="Loss Curve"), name="plot"),
            Layout(self.table, name="table", size=10),
            Layout(self.plot, name="plot", size=20),
        )

    def update(self, epoch: int, tloss: float, vloss: float):
        self.progress.update(
            self.task, advance=1, description=f"Epoch {epoch+1}/{args.num_batches}"
        )
        # Clear the table from rows
        new_table = Table(title=f"Epoch {epoch}")
        new_table.add_column("Metric")
        new_table.add_column("Value")
        new_table.add_row("Training Loss", f"{tloss:.3f}")
        new_table.add_row("Validation Loss", f"{vloss:.3f}")
        self.plot.add_measurement(tloss)
        self.layout["table"].update(new_table)


def train_w_metrics(model: nn.Module, args: argparse.Namespace, device: torch.device):

    tlosses = []
    vlosses = []
    # layout, progress = make_layout(0, tlosses, vlosses)
    layout = MyLayout(0, tlosses, vlosses)
    with Live(layout.layout, console=console, refresh_per_second=10) as live:
        for epoch, tloss, vloss in train(model, args, device):
            tlosses.append(tloss)
            vlosses.append(vloss)
            layout.update(epoch, tloss, vloss)

    # Plot the losses after the episode finishes
    t_diff_in_order = np.max(tlosses) - np.min(tlosses) > 1e1
    v_diff_in_order = np.max(vlosses) - np.min(vlosses) > 1e1
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(tlosses)
    if t_diff_in_order:
        axs[0].set_yscale("log")
    axs[0].set_title("Training Loss")
    axs[1].plot(vlosses)
    if v_diff_in_order:
        axs[1].set_yscale("log")
    axs[1].set_title("Validation Loss")
    plt.show()

    plt.show()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    # model = RecoveryNet(
    #     args.state_size, args.state_size, args.num_outputs, args.time_steps
    model = torch.nn.Transformer(
        d_model=args.state_size,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ct.use_fbs_defaults()  # Use settings to match FBS

    train_w_metrics(model, args, device)
