import argparse
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from rich.live import Live
from conrecon.plotting import TrainLayout
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from conrecon.automated_generation import generate_state_space_system
from conrecon.models.transformers import TransformerBlock, TorchsTransformer
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from conrecon.utils import create_logger
from pykalman import KalmanFilter
from rich import inspect
from rich.console import Console
from torch import nn, tensor
from tqdm import tqdm
from dataclasses import dataclass

console = Console()

# Create an alias for a 4-tuple call SParam
SSParam = Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]
# Create a data class that stores info to be stored as npy
@dataclass
class TrainingData:
    params: SSParam
    state_size: int
    output_dim: int
    time_steps: int
    ds_size: int
        
logger = create_logger("__main__")


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
    ap.add_argument("-e", "--epochs", default=10, help="How many epochs to train for", type=int)
    ap.add_argument("--eval_interval", default=1, help="How many epochs to train for", type=int)
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
    ap.add_argument("--ds_cache", default=".cache/pykalpkg_ds.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=10000, type=int)
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


# TOREM: THis is an artifact from long ago
# def get_sim(
#     Amat: np.ndarray,
#     Bmat: np.ndarray,
#     Cmat: np.ndarray,
#     init_cond: np.ndarray,
#     time_length: int,
#     input_dim: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Given the matrices, get states and outputs
#     """
#
#     sys = ct.ss(Amat, Bmat, Cmat, 0)
#     # CHECK: How do this limits affect the subsquent pykalman resultsG
#     t = np.linspace(0, 10, time_length)
#     u = np.zeros((input_dim, len(t)))
#     # u[:, int(len(t) / 4)] = 1
#     # TODO: think about this initial condition
#     # Lets run the simulation and see what happens
#     timepts = np.linspace(0, 10, time_length)
#     response = ct.input_output_response(sys, timepts, u, X0=init_cond)  # type: ignore
#     outputs = response.outputs
#     states = response.states
#     # inspect(outputs.shape)
#     # inspect(states.shape)
#     return states, outputs, init_cond


def design_matrices() -> SSParam:
    random_state = np.random.RandomState(0)
    A = [
        [1, 0.1, 0],
        [0, 1, 0],
        [0, 0.3, 1],
    ]
    C = np.eye(3)[:1, :] + random_state.randn(1, 3) * 0.1
    A = np.array(A)
    C = np.array(C)
    return A, None, C, None


def train(
    epochs: int,
    eval_interval: int,
    data: Tuple[torch.Tensor, torch.Tensor],
    training_data: TrainingData,
    tt_split : float = 0.8,
    d_model: int = 128,
    attn_heads: int = 8,
    ff_hidden: int = 256,
    dropout: float = 0.1,
    batch_size: int = 16,
):
    # Dta Management 
    states, outputs = data
    device = states.device
    logger.info(f"Device is {device}")
    tstates, toutputs = tensor(states), tensor(outputs)
    trn_states, val_states = torch.split(
        tstates,
        [int(len(states) * tt_split), len(states) - int(len(states) * tt_split)],
    )
    trn_outputs, val_outputs = torch.split(
        toutputs,
        [int(len(outputs) * tt_split), len(outputs) - int(len(outputs) * tt_split)],
    )
    t_val_states, t_val_outputs = tensor(val_states), tensor(val_outputs)

    # Ensure data is correct
    A, _, C, _ = training_data.params
    assert isinstance(
        A, np.ndarray
    ), f"Current simulation requires A to be a matrix. A is type {type(A)}"
    assert isinstance(
        C, np.ndarray
    ), f"Current simulation requires C to be a matrix. C is type {type(C)}"
    # Setup Training Tools
    logger.info(f"Setting training fundamentals")
    # model = TransformerBlock(
    #     d_model, training_data.state_size, attn_heads, ff_hidden, dropout=dropout
    # ).to(device)
    model = TorchsTransformer(
        d_model, training_data.state_size, training_data.output_dim, attn_heads, ff_hidden, dropout=dropout
    ).to(device)
    # Show amount of parameters
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(
        f"Model is of type {type(model)} with device {next(model.parameters()).device}"
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_list = []
    kf = KalmanFilter(
        transition_matrices=A, observation_matrices=C
    )

    loss_list = []
    eval_data = []
    batch_count = int(len(trn_states) / batch_size)
    train_layout = TrainLayout(epochs, batch_count, loss_list, eval_data)

    # Generate the simulations
    # We will use rich for reporting this??
    # f = Live(train_layout.layout, console=console, refresh_per_second=10)
    logger.info(f"Beginning Training iwht {epochs} epochs and batch size of {batch_size} resulting in {batch_count} batches")

    with Live(train_layout.layout, console=console, refresh_per_second=10) as live:
        for e in range(epochs):
            epoch_loss = 0
            for b in range(batch_count):

                cur_state = trn_states[b * batch_size : (b + 1) * batch_size].to(device)
                cur_output = trn_outputs[b * batch_size : (b + 1) * batch_size].to(device)

                # Estimate the state
                est_state = model(cur_state)

                loss = criterion(est_state, cur_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                epoch_loss += loss.item()
                cur_loss = loss.item()

                train_layout.update(e, b, cur_loss, None)

            # Normal Reporting
            if (e + 1) % 1 == 0:
                print("Epoch: {}, Loss: {:.5f}".format(e + 1, epoch_loss))
            # Eval Reporting
            e_report = None
            if e % eval_interval == 0:
                model.eval()
                preds = model(t_val_states)
                # preds = preds.view(-1).data.numpy().reshape(hidden)
                loss = criterion(preds, t_val_outputs)
                eval_data.append(preds)
                e_report = loss.item()

                # TODO:: Plot for testing
                # plot_states(
                #     preds[0, :, :][np.newaxis, :],
                #     hidden_truths[0, :, :][np.newaxis, :],
                #     save_path=f"{args.saveplot_dest}/plot_{timestamp}.png",
                #     first_n_states=args.first_n_states,
                # )
            # train_layout.update(e, 0, loss.item(), None)

    f.stop()
    return model, loss_list, eval_data


def gen_n_sims(
    params: SSParam,
    state_size: int,
    output_dim: int,
    time_steps: int,
    dataset_size: int,
) -> Tuple[np.ndarray, np.ndarray]:

    np.random.seed(0)
    A, _, C, _ = params
    assert isinstance(
        A, np.ndarray
    ), f"Current simulation requires A to be a matrix. A is type {type(A)}"
    assert isinstance(
        C, np.ndarray
    ), f"Current simulation requires C to be a matrix. C is type {type(C)}"

    # Generate the simulations
    hidden_truths = np.zeros(
        (
            dataset_size,
            time_steps,
            state_size,
        )
    )
    system_outputs = np.zeros(
        (
            dataset_size,
            time_steps,
            output_dim,
        )
    )

    # Setup a bar
    kf = KalmanFilter(transition_matrices=A, observation_matrices=C)
    for i in tqdm(range(dataset_size)):
        # CHECK: Might be A[1]
        # Sample Kalman Filter
        init_cond = np.random.uniform(-5, 5, A.shape[0])
        state, obs = kf.sample(time_steps, initial_state=init_cond)
        # results = get_sim(A, B, C, init_cond, time_steps, input_dim)
        hidden_truths[i, :, :] = state
        system_outputs[i, :, :] = obs

    return hidden_truths, system_outputs


def generate_dataset(
    params: SSParam,
    cache_path: str,
    state_dim: int,
    output_dim: int,
    time_steps: int,
    ds_size: int,
) -> Tuple[np.ndarray, np.ndarray, TrainingData]:
    # This will generate batch_size at a time and save as a dataset
    columns = [f"h{i}" for i in range(state_dim)]
    columns += [f"y{i}" for i in range(output_dim)]

    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from {cache_path}")
        final_dataset = pd.read_csv(cache_path)
        # Read the json file to get the metadata
        with open(cache_path.replace(".csv", ".json"), "r") as f:
            metadata = json.load(f)
            state_dim = metadata["state_size"]
            output_dim = metadata["output_dim"]
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
        train_data = TrainingData(params, state_dim, output_dim, time_steps, ds_size)
        return hiddens, outputs, train_data

    logger.info(f"Generating dataset to {cache_path}")
    # TODO: Batch this out int batch_size for long enough ds_size
    hiddens, outputs = gen_n_sims(
        params,
        state_dim,
        output_dim,
        time_steps,
        ds_size,
    )
    logger.info(f"Hiddens are of shape {hiddens.shape}")
    logger.info(f"Outputs are of shape {outputs.shape}")

    # CHECK: the dimmensions are correntk
    # hiddens_transposed = hiddens.transpose(0, 2, 1)
    # outputs_transposed = outputs.transpose(0, 2, 1)

    hiddens_final = hiddens.reshape((ds_size * time_steps, state_dim))
    outputs_final = outputs.reshape((ds_size * time_steps, output_dim))
    # Form hiddens and outputs into a dataframe
    hiddens = pd.DataFrame( hiddens_final, columns=columns[:state_dim],) # type: ignore
    outputs = pd.DataFrame(outputs_final, columns=columns[state_dim:]) # type: ignore
    final_dataset = pd.concat([hiddens, outputs], axis=1)
    final_dataset.to_csv(cache_path, index=False)
    # Save metadata to json file
    # json_file = cache_path.replace(".csv", ".json")
    # metadata = {
    #     "state_size": state_dim,
    #     "output_dim": output_dim,
    #     "input_dim": input_dim,
    #     "time_steps": time_steps,
    #     "ds_size": ds_size,
    # }
    # with open(json_file, "w") as f:
    #     json.dump(metadata, f)
    train_data = TrainingData(params, state_dim, output_dim, time_steps, ds_size)
    pkl_file = cache_path.replace(".csv", ".pkl")
    pickle.dump(train_data, open(pkl_file, "wb"))

    return hiddens.values, outputs.values, train_data

def main():
    args = argsies()
    # Get our A, B, C Matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = design_matrices()
    np.random.seed(int(time.time()))

    logger.info(f"Generating dataset")
    # TODO: Make it  so that generate_dataset checks if params are the same
    hidden, outputs, training_data = generate_dataset(
        params,
        args.ds_cache,
        args.state_size,
        args.output_dim,
        args.time_steps,
        args.ds_size,
    )

    # Merge metadata into args
    print(f"Training with args {args}")

    t_hidden, t_outputs = (
        torch.from_numpy(hidden).to(torch.float32).to(device),
        torch.from_numpy(outputs).to(torch.float32).to(device),
    )
    logger.info(f"Training with type of hiddens {type(t_hidden)} and outputs {type(t_outputs)}")
    results = train(
        args.epochs,
        args.eval_interval,
        data=(t_hidden, t_outputs),
        training_data=training_data,
    )

    ## ML-Approach

    preds = np.array(preds)
    # Now we plot the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_states(
        preds[0, :, :][np.newaxis, :],
        hidden_truths[0, :, :][np.newaxis, :],
        save_path=f"{args.saveplot_dest}/plot_{timestamp}.png",
        first_n_states=args.first_n_states,
    )


if __name__ == "__main__":
    main()
