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
from conrecon.plotting import plot_functions

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
    obs_size = C.shape[0]
    state_covariance = np.eye(state_size)
    observation_covariance = np.eye(obs_size)
    init_cond = [5, -5, 0]

    # Generate Torch Model for Later State Estimation
    torch_filter = Filter(
        transition_matrix=torch.from_numpy(A).to(torch.float32),
        observation_matrix=torch.from_numpy(C).to(torch.float32),
        input_matrix=torch.from_numpy(B).to(torch.float32),
        initial_state_mean=torch.Tensor(init_cond).to(torch.float32),
        batch_size=num_samples,
        process_noise_covariance=torch.from_numpy(state_covariance).to(torch.float32),
        measurement_noise_covariance=torch.from_numpy(observation_covariance).to(torch.float32)
    )
    kf = KalmanFilter(
        transition_matrices=A, observation_matrices=C, initial_state_mean=init_cond
    )

    ### Gather Info
    states_samples = []
    obs_samples = []
    for b in range(num_samples):
        states, obs = kf.sample(args.time_steps, initial_state=init_cond)
        states_samples.append(states)
        obs_samples.append(obs)
    states_samples_t = torch.Tensor(np.array(states_samples))
    obs_samples_t = torch.Tensor(obs_samples)

    ### Estimation
    ## Torch Estimation
    our_filter_estimation: torch.Tensor = torch_filter(obs_samples_t) # Star
    ## Native Filter Estimation
    filtered_means = []
    for i in range(states_samples_t.shape[0]):
        (filtered_mean, filtered_covariance) = kf.filter(obs_samples[i])
        filtered_means.append(filtered_mean)

    # Concatenate elements of list to numpy arrayj
    native_estimation = np.stack(filtered_means)
    my_estimation = our_filter_estimation.numpy()

    # Now we compare the recoveries
    diff = np.abs(native_estimation - my_estimation)
    # Now we plot the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    funcs_to_plot = np.stack([our_filter_estimation.numpy(), native_estimation]).transpose(1,0,2,3)
    plot_functions(
        funcs_to_plot,
        save_path=f"{args.saveplot_dest}/plot_{timestamp}.png",
        function_labels=["Torch Version","Native"],
        first_n_states=3,
    )
