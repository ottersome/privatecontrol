from ..utils import create_logger
from ..ss_generation import SSParam
from typing import Dict, Tuple
import os
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sktime.libs.pykalman import KalmanFilter
from tqdm import tqdm

logger = create_logger("dataset_generation")

# Create a data class that stores info to be stored as npy
@dataclass
class TrainingMetaData:
    params: SSParam
    state_size: int
    output_dim: int
    input_dim: int
    time_steps: int
    ds_size: int



def generate_dataset(
    params: SSParam,
    cache_path: str,
    state_dim: int,
    input_dim: int,
    output_dim: int,
    time_steps: int,
    ds_size: int,
) -> Tuple[np.ndarray, np.ndarray, TrainingMetaData]:
    # This will generate batch_size at a time and save as a dataset
    columns = [f"h{i}" for i in range(state_dim)]
    columns += [f"y{i}" for i in range(output_dim)]

    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from {cache_path}")
        final_dataset = pd.read_csv(cache_path)
        # Read the json file to get the metadata
        with open(cache_path.replace(".csv", ".pkl"), "rb") as f:
            train_data = pickle.load(f)
            state_dim = train_data.state_size
            output_dim = train_data.output_dim
            ds_size = train_data.ds_size
            time_steps = train_data.time_steps
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
        train_data = TrainingMetaData(
            params, state_dim, output_dim,input_dim,time_steps, ds_size
        )
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
    hiddens = pd.DataFrame(
        hiddens_final,
        columns=columns[:state_dim], # type: ignore
    )  # type: ignore
    outputs = pd.DataFrame(outputs_final, columns=columns[state_dim:])  # type: ignore
    final_dataset = pd.concat([hiddens, outputs], axis=1)
    final_dataset.to_csv(cache_path, index=False)

    train_data = TrainingMetaData(
        params, state_dim, output_dim, input_dim, time_steps, ds_size
    )
    pkl_file = cache_path.replace(".csv", ".pkl")
    pickle.dump(train_data, open(pkl_file, "wb"))

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

    return hiddens, outputs, train_data


#CHECK: Potentially removable
def gen_n_sims(
    params: SSParam,
    state_size: int,
    output_dim: int,
    time_steps: int,
    dataset_size: int,
) -> Tuple[np.ndarray, np.ndarray]:

    np.random.seed(0)
    A,C = params.A, params.C
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


def batch_generation_randUni(
    ds: Dict[str,np.ndarray],
    sequence_len: int,
    padding_value: int = -1,
    oversample_coefficient = 1.2, 
    leave_final_file: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Will sample uniform at random across the time windows.
    Will add batching when necessary
    """
    last_key = list(ds.keys())[-1] # debug
    rollouts = []
    for k,v in ds.items():
        if leave_final_file and k == last_key:
            continue # ignore last file
        run_size = v.shape[0]
        num_sample_sequences = int(oversample_coefficient * (run_size // sequence_len))
        
        random_spots = np.random.randint(0, run_size, num_sample_sequences)
        for rs in random_spots:
            history = spot_backhistory(rs, sequence_len, v, padding_value)
            rollouts.append(history)

    rollouts = np.stack(rollouts, axis=0)
    np.random.shuffle(rollouts)

    # DEBUG: 
    validation_episode = np.array([])
    if leave_final_file:
        validation_episode = ds[last_key]

    return rollouts, validation_episode

def spot_backhistory(spot: int, sequence_len: int, run: np.ndarray, padding_value: int):
    """
    Args:
        spot: where the sampling will start from.
        sequence_len:  includes `spot` as part of the sequence length
    """
    sequence_so_far = run[: spot + 1, :]  # Inclusive
    sequence_so_far = sequence_so_far[-sequence_len:, :]
    if (spot - sequence_len + 1) < 0:
        # Then we have to add padding:
        padding_amnt = sequence_len - sequence_so_far.shape[0]
        padding = np.full((padding_amnt, sequence_so_far.shape[1]), padding_value)
        final_sequence = np.concat(
            [
                padding,
                sequence_so_far,
            ],
            axis=0,
        )
    else:
        final_sequence = sequence_so_far

    return final_sequence

def spot_shuanghistory():
    pass
                        

def timeseries_ds_formation(ds: Dict[str, np.ndarray], episode_length: int, gap: int):
    final_ds = []
    # Calculate the distances
    hop_distance = episode_length - gap

    rollouts = []
    for k, v in ds.items():
        for h in range(ceil(v.shape[0] // hop_distance)):
            offset = hop_distance * h
            logger.info(
                f"({k}: len({v.shape[0]})): Adding att offset {offset} to {offset+episode_length}"
            )
            rollout = v[offset : offset + episode_length, :]
            if rollout.shape[0] != episode_length:
                continue
            rollouts.append(rollout)

    # Shuffle it around
    rollouts = np.stack(rollouts, axis=0)
    # TODO: Check that the shuffling is being done right
    np.random.shuffle(rollouts)
    return rollouts
