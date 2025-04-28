# All the imports
import math
import shutil
from typing import List, Union, Optional

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib import pyplot as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from .utils.common import create_logger


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

def plot_functions_2by1(
    functions: npt.NDArray[np.float64],
    save_path: str,
    function_labels: List[str],
    indep_func_label = "Sample",
    dim_to_show: Optional[List] = None,
    dim_labels: Optional[List] = None,
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
    if dim_to_show is None:
        dim_to_show = list(range(functions.shape[3]))


    num_independent_functions = functions.shape[0]
    assert num_independent_functions == 2, "`plot_functions` Can only deal with 2 independent functions"
    num_functions = functions.shape[1]
    _ = functions.shape[2] # Seq_length
    num_dim = functions.shape[3]
    numdim_toshow = len(dim_to_show)

    sns.set_theme(style="whitegrid")

    # Do a grid configuration for independent functions
    num_rows = 2
    num_cols = 1
    _, ax = plt.subplots(
        num_rows, num_cols, figsize=( num_cols * 14,num_rows *  4)
    )
    # ax = np.atleast_2d(ax)  # type: ignore

    plt.tight_layout(pad=5)

    colormap = sns.color_palette("husl", num_dim)
    lines_styles = ["-",  "--", "-.", ":"]
    for n in range(num_independent_functions):
        i,j = n // num_cols, n % num_cols
        for s in dim_to_show:
            for f in range(num_functions):
                ax[n].plot(
                    functions[n, f, :, s],
                    label=function_labels[f] + f"-$S_{s}$",
                    linestyle=lines_styles[f % len(lines_styles)],
                    color=colormap[s],
                )
                ax[n].set_xlabel("Time")
                ax[n].set_ylabel("Magnitude")
                ax[n].set_title(indep_func_label+f" {n+1}")
                ax[n].legend()
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def plot_functions(
    functions: npt.NDArray[np.float64],
    save_path: str,
    function_labels: List[str],
    first_n_states: int = 4,
    dims_to_show: Optional[List] = None,
    indep_func_label = "Sample",
    dim_labels: Optional[List] = None,
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

    num_independent_functions = functions.shape[0]
    num_functions = functions.shape[1]
    _ = functions.shape[2] # Seq_length
    num_dims = functions.shape[3]

    assert len(functions.shape) <= 4, "`plot_functions` Can only deal with up to 4D tensors"
    while len(functions.shape) < 4:
        functions = np.expand_dims(functions, axis=0)
    assert len(function_labels) == functions.shape[1], "`plot_functions` Needs as many labels as functions"
    if dims_to_show is None:
        dims_to_show = list(range(num_dims))
    if dim_labels is None:
        dim_labels = list(range(num_dims))
    assert len(dims_to_show) == len(dim_labels), "`dims_to_show` and `dim_labels` must be of the same length"

    sns.set_theme(style="whitegrid")

    # Do a grid configuration for independent functions
    sqrt = math.ceil(np.sqrt(num_independent_functions)) # type: ignore
    num_rows = math.ceil(num_independent_functions/sqrt)
    fig, ax = plt.subplots(
        num_rows, sqrt, figsize=( sqrt * 8,num_rows * 8)
    )
    ax = np.atleast_2d(ax)  # type: ignore
    plt.tight_layout()

    colormap = sns.color_palette("husl", num_dims)
    lines_styles = ["-",  "-", "-", ":"]
    markers = ["o", "*", "x", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    for n in range(num_independent_functions):
        i,j = n // sqrt, n % sqrt
        for s in dims_to_show:
            for f in range(num_functions):
                label = function_labels[f] + " (" + dim_labels[s]+")" if n == 0 else None
                ax[i, j].plot(
                    functions[n, f, :, s],
                    label=label,
                    linestyle=lines_styles[f % len(lines_styles)],
                    marker=markers[f % len(markers)],
                    color=colormap[s],
                )
                ax[i, j].set_xlabel("Time")
                # ax[i, j].set_ylabel(indep_func_label)
                ax[i, j].set_ylabel("Magnitude")
                ax[i, j].set_title(indep_func_label+f" {n+1}")
                # ax[i, j].legend()
    # Save the figure
    fig.legend(loc='center left', bbox_to_anchor=(1,0.5), title='Legend')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
