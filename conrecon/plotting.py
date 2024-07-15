# All the imports
import shutil
from typing import List

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
        self, epoch: int, tlosses: List[float], vlosses: List[float], num_batches
    ):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("[cyan]Training..", total=num_batches)
        self.layout = Layout()
        self.num_batches = num_batches
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
            self.task, advance=1, description=f"Epoch {epoch+1}/{self.num_batches}"
        )
        # Clear the table from rows
        new_table = Table(title=f"Epoch {epoch}")
        new_table.add_column("Metric")
        new_table.add_column("Value")
        new_table.add_row("Training Loss", f"{tloss:.3f}")
        new_table.add_row("Validation Loss", f"{vloss:.3f}")
        self.plot.add_measurement(tloss)
        self.layout["table"].update(new_table)
