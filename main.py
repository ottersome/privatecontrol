import argparse
import datetime
import os
from typing import Generator, Iterable, Tuple

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from rich import inspect
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from torch.nn import functional as F

from conrecon.automated_generation import generate_state_space_systems

console = Console()


class Plot:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.series = []

    def add_series(self, data, label=None, color="white"):
        self.series.append((data, label, color))

    def render(self):
        max_value = max([max(s[0]) for s in self.series] + [1])
        min_value = min([min(s[0]) for s in self.series] + [0])

        lines = []
        for y in range(self.height):
            value = max_value - y * (max_value - min_value) / (self.height - 1)
            line = []
            for x in range(self.width):
                index = int(x / self.width * (len(self.series[0][0]) - 1))
                char = " "
                for series, label, color in self.series:
                    if series[index] >= value:
                        char = "█"
                        break
                line.append(char)
            lines.append("".join(line))
        return "\n".join(lines)

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
    outputs = response.outputs.T
    states = response.states.T
    return states, outputs


def single_train_batch(
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
        time_steps,
        state_size,
        device=device,
    )
    system_outputs = torch.zeros(
        batch_size,
        time_steps,
        num_outputs,
        device=device,
    )

    for i in range(batch_size):
        results = get_n_sims(Amats[i], Bmats[i], Cmats[i], time_steps, input_dim)
        hidden_truths[i, :, :] = torch.from_numpy(results[0])
        system_outputs[i, :, :] = torch.from_numpy(results[1])

    return hidden_truths, system_outputs


def train(
    model: nn.Module, args: argparse.Namespace
) -> Generator[Tuple[int, float, float], None, None]:
    """
    Train the RNN
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(args.num_batches):
        tlosses = []
        model.train()
        for j in range(args.batch_size):

            hidden_truths, system_outputs = single_train_batch(
                args.state_size,
                args.input_dim,
                args.num_outputs,
                args.time_steps,
                args.batch_size,
            )
            # Train
            model.train()
            model.zero_grad()
            hidden_truths = hidden_truths.to(device)
            system_outputs = system_outputs.to(device)
            # inspect(hidden_truths)
            outputs, hidden_states = model(hidden_truths)
            # CHECK: Loss is being performed in the right dimensions
            # Log pretty the shapes using rich
            loss = criterion(outputs, system_outputs)
            tlosses.append(loss.mean().item())
            loss.backward()
            optimizer.step()

        ## Evaluate the loss on a single round
        mean_train_loss = np.mean(tlosses).item()
        tlosses.append(mean_train_loss)
        model.eval()
        hidden_truths, system_outputs = single_train_batch(
            args.state_size,
            args.input_dim,
            args.num_outputs,
            args.time_steps,
            args.batch_size,
        )

        outputs, hidden_states = model(hidden_truths)

        loss = criterion(outputs, system_outputs)
        mean_eval_loss = loss.item()

        # DEBUG:
        plot_functions(
            system_outputs.cpu().detach().numpy()[:3, :, :],
            outputs.cpu().detach().numpy()[:3, :, :],
            save_path=f"{args.save_destination}/plot_{i}.png",
        )

        yield i, mean_train_loss, mean_eval_loss


def train_w_metrics(model: nn.Module, args: argparse.Namespace):

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )
    task = progress.add_task("[cyan]Training..", total=args.num_batches)

    tlosses = []
    vlosses = []
    with Live(console=console, refresh_per_second=10) as live:
        for epoch, tloss, vloss in train(model, args):
            progress.update(
                task, advance=1, description=f"Epoch {epoch+1}/{args.num_batches}"
            )
            table = Table(title=f"Epoch {epoch}")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Training Loss", f"{tloss:.3f}")
            table.add_row("Validation Loss", f"{vloss:.3f}")

            tlosses.append(tloss)
            vlosses.append(vloss)

            plot = Plot(width=60, height=20)
            plot.add_series(tlosses, label="Loss", color="red")

            layout = Layout()
            layout.split(
                Layout(progress, name="progress"),
                Layout(Panel(plot, title="Loss Curve"), name="plot"),
                Layout(table, name="table"),
            )

            live.update(table)


class RecoveryNet(nn.Module):
    def __init__(self, state_size, num_outputs):
        super().__init__()
        self.rnn = torch.nn.RNN(state_size, state_size, batch_first=True)
        # Final output layer
        self.output_layer = torch.nn.Linear(state_size, num_outputs)

    def forward(self, x):
        # x is a tensor of shape (batch_size, time_steps, state_size)
        # We want to get a tensor of shape (batch_size, time_steps, num_outputs)
        x, hidden = self.rnn(x)
        return self.output_layer(x), hidden


if __name__ == "__main__":

    args = argsies()
    # Get out matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecoveryNet(args.state_size, args.num_outputs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ct.use_fbs_defaults()  # Use settings to match FBS
    outtie = tqdm.tqdm(total=args.num_batches)

    train_w_metrics(model, args)
