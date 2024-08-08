# All the imports
import shutil
from typing import List, Union
import math

from matplotlib import pyplot as plt
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from .utils import create_logger


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


class TrainLayout:
    def __init__(
        self,
        epochs: int,
        num_batches: int,
        tlosses: List[float],
        vlosses: List[float],
    ):
        self.latest_vloss = 0
        # Progress bars
        self.epoch_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        self.batch_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        self.epoch_task = self.epoch_progress.add_task("[green]Epochs..", total=epochs)
        self.batch_task = self.batch_progress.add_task(
            "[cyan]Batches..", total=num_batches
        )

        self.layout = Layout()
        self.epochs = epochs
        self.num_batches = num_batches
        self.table = Table(title=f"Epoch {0}/epochs")
        self.table.add_column("Metric")
        self.table.add_column("Value")
        if len(tlosses) > 1:
            self.table.add_row("Training Loss", f"{tlosses[-1]:.3f}")
            self.table.add_row("Validation Loss", f"{vlosses[-1]:.3f}")

        self.tplot = Plot(width=60, height=20)
        self.vplot = Plot(width=60, height=20)

        self.layout.split(
            Layout(self.epoch_progress, name="Epoch progress", size=3),
            Layout(self.batch_progress, name="Batch progress", size=3),
            # Layout(Panel(plot, title="Loss Curve"), name="plot"),
            Layout(self.table, name="table", size=10),
            Layout(self.tplot, name="Training Plot", size=20),
            Layout(self.vplot, name="Validation Plot", size=20),
        )
        self.cur_batch = 0

    def update(
        self, epoch: int, batch_no: int, tloss: float, vloss: Union[float, None]
    ):
        assert tloss is not None, "Loss should not be None"
        if self.cur_batch + 1 > self.num_batches:
            self.cur_batch = 0
            self.batch_progress.reset(self.batch_task)
            self.epoch_progress.update(
                self.epoch_task, advance=1, description=f"Epoch {epoch+1}/{self.epochs}"
            )

        self.batch_progress.update(
            self.batch_task,
            advance=1,
            description=f"Batch {batch_no+1}/{self.num_batches}",
        )
        self.cur_batch += 1
        # Clear the table from rows
        new_table = Table(
            title=f"Epoch {epoch}/{self.epochs}, Batch {batch_no+1}/{self.num_batches}"
        )
        new_table.add_column("Metric")
        new_table.add_column("Value")
        new_table.add_row("Training Loss", f"{tloss:.3f}")
        new_table.add_row("Validation Loss", f"{self.latest_vloss:.3f}")
        if vloss is not None:
            self.latest_vloss = vloss
            self.vplot.add_measurement(vloss)
        self.tplot.add_measurement(tloss)
        self.layout["table"].update(new_table)

def plot_functions(
    functions: np.ndarray,
    save_path: str,
    function_labels: List[str],
    first_n_states: int = 3,
    ):
    """
    Plots a set of functions in a grid of subplots
    Args:
        - functions: A numpy array of shape (num_independent_functions, num_functions, seq_length, dim)
        - save_path: The path to save the figure to
        - function_labels: A list of strings of length num_functions
        - first_n_states: The number of states to show in the plot
    Returns:
        None
    """
    assert len(functions.shape) <= 4, "`plot_functions` Can only deal with up to 4D tensors"
    while len(functions.shape) < 4:
        functions = np.expand_dims(functions, axis=0)
    assert len(function_labels) == functions.shape[1], "`plot_functions` Needs as many labels as functions"

    num_independent_functions = functions.shape[0]
    num_functions = functions.shape[1]
    _ = functions.shape[2] # Seq_length
    dim = functions.shape[3]
    dim_to_show = min(dim, first_n_states)

    # Do a grid configuration for independent functions
    sqrt = math.ceil(np.sqrt(num_independent_functions))
    _, ax = plt.subplots(
        sqrt, sqrt, figsize=(sqrt * 6, sqrt * 6)
    )
    ax = np.atleast_2d(ax)  # type: ignore
    plt.tight_layout()

    colormap = plt.get_cmap("tab10")
    lines_styles = ["-", "--", "-.", ":"]
    for n in range(num_independent_functions):
        i,j = n // sqrt, n % sqrt
        for s in range(dim_to_show):
            for f in range(num_functions):
                ax[i, j].plot(
                    functions[n, f, :, s],
                    label=function_labels[f] + f"-$S_{s}$",
                    linestyle=lines_styles[f],
                    color=colormap(s),
                )
                ax[i, j].set_xlabel("Time")
                ax[i, j].set_ylabel("Output")
                ax[i, j].set_title(f"Output {n+1}")
                ax[i, j].legend()
    # Save the figure
    plt.savefig(save_path)
    plt.close()
