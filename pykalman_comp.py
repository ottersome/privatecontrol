"""
Will compare the output of pykalman from library andy own torch implementation
Hopefully this will help us understand the differences
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from rich import traceback
from rich.console import Console
from sktime.libs.pykalman import KalmanFilter

from conrecon.kalman.mo_core import Filter
from conrecon.utils import create_logger


traceback.install()

console = Console()

name_file = os.path.basename(__file__)
logger = create_logger(name_file)


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-n", "--num_samples", default=4, type=int, help="How many Samples to Evaluate"
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
        default="./figures/comparison/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=100, type=int)
    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed = 0
    if args.random:
        from time import time

        seed = int(time())
    np.random.seed(seed)
    num_samples = args.num_samples

    random_state = np.random.RandomState(0)
    A = np.array([[1, 0.1, 0], [0, 0.2, 0.3], [0, 0, 0.8]])
    B = np.zeros((A.shape[0], A.shape[0]))  # Place holder for now
    C = np.eye(3)[:2, :] + random_state.randn(2, 3) * 0.1

    state_size = A.shape[0]
    input_size = B.shape[1]
    init_cond = [5, -5, 0]

    # Generate Torch Model for Later State Estimation
    torch_filter = Filter(
        transition_matrix=torch.from_numpy(A),
        input_matrix=torch.from_numpy(B),
        observation_matrix=torch.from_numpy(C),
        initial_state_mean=torch.Tensor(init_cond),
        batch_size=num_samples,
    )
    kf = KalmanFilter(
        transition_matrices=A, observation_matrices=C, initial_state_mean=init_cond
    )

    ### Gather Info
    states_samples = []
    for b in range(num_samples):
        states, obs = kf.sample(args.time_steps, initial_state=init_cond)
        states_samples.append(states)
    states_samples_t = torch.Tensor(states_samples)
    logger.info(
        f"Samples are of shape  {states_samples_t.shape} and of type {type(states_samples_t)}"
    )

    ### Estimation
    ## Torch Estimation
    our_filter_estimation: torch.Tensor = torch_filter(states_samples_t) # Star
    logger.info(
        f"Our filter is of shape {our_filter_estimation.shape} and type {type(our_filter_estimation)}"
    )
    ## Native Filter Estimation
    filtered_means = []
    for i in range(len(states_samples)):
        (filtered_mean, filtered_covariance) = kf.filter(states_samples[i])
        filtered_means.append(filtered_means)

    native_estimation = np.array(filtered_means)
    my_estimation = our_filter_estimation.numpy()
