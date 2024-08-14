"""
Will compare the output of pykalman from library andy own torch implementation
Hopefully this will help us understand the differences
"""

import argparse
import os
from datetime import datetime
from typing import Tuple

import torch
import control as ct
import matplotlib.pyplot as plt
import numpy as np
from rich import inspect
from rich.console import Console
from sktime.libs.pykalman import KalmanFilter

from conrecon.kalman.mo_core import Filter
from conrecon.utils import create_logger

console = Console()

name_file = os.path.basename(__file__)
logger = create_logger(name_file)


def plot_states(
    estimated_states: np.ndarray,
    states: np.ndarray,
    save_path: str,
    first_n_states: int = 7,
):
    assert (
        len(states.shape) == 3
    ), f"Can only plot up to 3 outputs. Received shape {states.shape}"
    assert (
        len(estimated_states.shape) == 3
    ), f"Can only plot up to 3 outputs. Received shape {estimated_states.shape}"
    num_outputs = states.shape[0]
    num_elements = states.shape[2]
    assert num_outputs <= 4, "Can only plot up to 4 outputs"
    _, ax = plt.subplots(num_outputs, 2, figsize=(num_outputs * 12, 6))
    ax = np.atleast_2d(ax)  # type: ignore
    plt.tight_layout()

    inspect(states.shape, title="Shape of the states")
    inspect(estimated_states.shape, title="Shape of the estimated states")
    states_shown = min(first_n_states, num_elements)
    color_map = plt.get_cmap("tab10")
    print(f"Showing {num_outputs} outputs")
    for i in range(num_outputs):
        for j in range(states_shown):
            # Plot the outputs
            ax[i, 0].plot(
                estimated_states[i, :, j],
                label=f"Estimated S_{j}",
                color=color_map(j),
                linestyle="--",
            )
            ax[i, 0].plot(
                states[i, :, j],
                label=f"True S_{j}",
                color=color_map(j),
            )
            ax[i, 0].set_xlabel("Time")
            ax[i, 0].set_ylabel("State")
            ax[i, 0].set_title(f"Output")
            ax[i, 0].legend()

            # Plot the error
            ax[i, 1].plot(
                np.abs(estimated_states[i, :, j] - states[i, :, j]), color="red"
            )
            ax[i, 1].set_xlabel("Time")
            ax[i, 1].set_ylabel("Error")
            ax[i, 1].set_title(f"Error")

    # Rather than showing them save them to a file
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-n", "--num_samples", default=1, type=int, help="How many Samples to Evaluate"
    )
    ap.add_argument(
        "--eval_size", default=4, help="How many systems to generate", type=int
    )
    # Control stuff here
    ap.add_argument(
        "-t", "--time_steps", default=12, help="How many systems to generate", type=int
    )
    ap.add_argument(
        "-s", "--state_size", default=3, help="Dimensionality of the state.", type=int
    )
    ap.add_argument(
        "-i", "--input_dim", default=3, help="Dimensionality of the input.", type=int
    )
    ap.add_argument(
        "-o", "--output_dim", default=2, help="Dimensionality of the output.", type=int
    )
    ap.add_argument("-r", "--random", action="store_true", default=False)
    ap.add_argument("--ds_cache", default=".cache/ds_kf_classical.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=100, type=int)
    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.01, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check


if __name__ == "__main__":
    args = argsies()
    # Get our A, B, C Matrices

    seed = 0
    if args.random:
        from time import time

        seed = int(time())
    np.random.seed(seed)

    random_state = np.random.RandomState(0)
    A = np.array([[1, 0.1, 0], [0, 0.2, 0.3], [0, 0, 0.8]])
    B = np.zeros((A.shape[0], A.shape[0])) # Place holder for no
    C = np.eye(3)[:2, :] + random_state.randn(2, 3) * 0.1
    state_size = A.shape[0]
    input_size = B.shape[1]

    # Generate the simulations
    hidden_truths = np.zeros(
        (
            1,
            args.time_steps,
            args.state_size,
        )
    )
    output_observations = np.zeros(
        (
            1,
            args.time_steps,
            args.output_dim,
        )
    )

    preds = []

    init_cond = [5, -5, 0]
    kf = KalmanFilter(
        transition_matrices=A, observation_matrices=C, initial_state_mean=init_cond
    )
    our_filter = Filter(
        transition_matrix=A,
        input_matrix=B,
        observation_matrix=C,
    )

    states, obs = kf.sample(args.time_steps, initial_state=init_cond)

    hidden_truths[0, :, :] = states
    output_observations[0, :, :] = obs

    inspect(A, title="A Matrix")
    inspect(C, title="C Matrix")

    ## ML-Approach

    ## Native Approach
    # inspect(obs.shape, title="Shape of the output")
    # Now we try to recover with the KF
    (filtered_mean, filtered_covariance) = kf.filter(obs)
    (smoothed_mean, smoothed_covariance) = kf.smooth(obs)
    # inspect(smoothed_mean.shape, title="Shape of the smoothed mean")
    preds.append(smoothed_mean)

    preds = np.array(preds)
    # Now we plot the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_states(
        preds[0, :, :][np.newaxis, :],
        hidden_truths[0, :, :][np.newaxis, :],
        save_path=f"{args.saveplot_dest}/plot_{timestamp}.png",
        first_n_states=args.first_n_states,
    )
