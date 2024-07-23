import argparse
import os
import time
from datetime import datetime
from typing import Generator, Iterable, List, Tuple

import control as ct
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter
from rich import inspect
from rich.console import Console

from conrecon.automated_generation import generate_state_space_system

console = Console()


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
    fig, ax = plt.subplots(num_outputs, 2, figsize=(num_outputs * 12, 6))
    ax = np.atleast_2d(ax)  # Now ax is to be 2D
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
        "-s", "--state_size", default=3, help="Dimensionality of the state.", type=int
    )
    ap.add_argument(
        "-i", "--input_dim", default=3, help="Dimensionality of the input.", type=int
    )
    ap.add_argument(
        "-o", "--output_dim", default=1, help="Dimensionality of the output.", type=int
    )
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


def get_sim(
    Amat: np.ndarray,
    Bmat: np.ndarray,
    Cmat: np.ndarray,
    init_cond: np.ndarray,
    time_length: int,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the matrices, get states and outputs
    """

    sys = ct.ss(Amat, Bmat, Cmat, 0)
    # CHECK: How do this limits affect the subsquent pykalman resultsG
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
    return states, outputs, init_cond


if __name__ == "__main__":
    args = argsies()
    # Get our A, B, C Matrices

    np.random.seed(int(time.time()))

    # A, B, C, D = generate_state_space_system(
    #     args.input_dim,
    #     args.output_dim,
    #     args.state_size,
    #     seed=0,
    # )

    random_state = np.random.RandomState(0)
    # A = [[1, 0.1], [0, 1]]
    # C = np.eye(2) + random_state.randn(2, 2) * 0.1
    A = [
        [1, 0.1, 0],
        [0, 1, 0],
        [0, 0.3, 1],
    ]
    C = np.eye(3)[:1, :] + random_state.randn(1, 3) * 0.1

    # Generate the simulations
    hidden_truths = np.zeros(
        (
            args.num_samples,
            args.time_steps,
            args.state_size,
        )
    )
    output_observations = np.zeros(
        (
            args.num_samples,
            args.time_steps,
            args.output_dim,
        )
    )

    preds = []
    for i in range(args.num_samples):
        # CHECK: Might be A[1]
        # init_cond = np.random.uniform(0, 1, A.shape[0])
        # init_cond = np.random.normal(0, 1, A.shape[0])
        init_cond = [5, -5, 1]
        kf = KalmanFilter(
            transition_matrices=A, observation_matrices=C, initial_state_mean=init_cond
        )
        states, obs = kf.sample(args.time_steps, initial_state=init_cond)

        hidden_truths[i, :, :] = states
        output_observations[i, :, :] = obs

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
