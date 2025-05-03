import random
import pandas as pd
from typing import List, DefaultDict, OrderedDict, Set
import pdb
import itertools
import os
import numpy as np
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re

from conrecon.data.dataset_generation import spot_backhistory

def load_runs(path: str) -> OrderedDict[str, pd.DataFrame]:
    """
    Load the data from the defacto dataset
    """

    # Create a list of files starting with `run_` inside of the path
    files = [f for f in os.listdir(path) if f.startswith("run_")]

    # Organize them by number after the run_
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0].split("run_")[1]))
    obtained_runs = OrderedDict({ f_name : pd.DataFrame() for f_name in sorted_files })

    for f in sorted_files:
        print(f"Loading run: run {f}")
        df = pd.read_csv(os.path.join(path,f), index_col=0, header=0)
        obtained_runs[f] = df

    # Let me see how it looks
    return obtained_runs


def df_from_run(df: pd.DataFrame, features_per_run: int) -> pd.DataFrame:
    """
    Will take a dataframe and go one by one all the features and create a new timeline more easily to view
    """
    # Drop only rows that re completely empty. Still retain those that have nans
    df = df.dropna(axis=0, how="all")  # CHECK:
    INDEX_NAME = "Time"

    print(f"Logging dataframe with head {df.head()}")

    #  We need to ensure we get N/A values
    all_features = DefaultDict(lambda: {f"f_{i}": None for i in range(features_per_run)})
    for r in range(df.shape[0]):
        for c in range(features_per_run):
            time = df.iloc[r, 2 * c]
            val = df.iloc[r, 2 * c + 1]
            if time == ' ':
                continue
            all_features[time][f"f_{c}"] = val
    # Create a new dataframe with the new features
    df = pd.DataFrame.from_dict(all_features, orient="index")
    df.index.name = INDEX_NAME

    print(
        f"We ended up with the following stats with this dataframe:\n"
        f"Initial time is {df.index[0]} and final time is {df.index[-1]}\n"
        f"With a total of {df.shape[0]} rows and {df.shape[1]} columns\n"
    )
    # Drop rows that have each columns empty (AND ACROSS COLUMNS)
    df = df.dropna(axis=0, how="all", subset=df.columns.tolist())
    df_sorted = df.sort_values(by=INDEX_NAME,  ascending=True, key=lambda col: col.astype(float))

    # Write new_df to csv
    return df_sorted

def split_run(run: np.ndarray, split_percentage: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Will split a run into train, validation and test
    Returns: in Order: Train, Validation, Test
    """
    assert sum(split_percentage) == 1, "Split Percentages should sum to 1"
    assert len(split_percentage) == 3, "There should be three split_percentages"

    return (
        run[: int(len(run) * (split_percentage[0]))],
        run[
            int(len(run) * (split_percentage[0])) 
            : int(len(run) * (split_percentage[0] + split_percentage[1]))
        ],
        run[int(len(run) * (split_percentage[0] + split_percentage[1])) :],
    )

def randomly_pick_sequences_and_split(run: np.ndarray, split_percentage: List[float], seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Add capacity for oversampling
    """
    Will split a run into train, validation and test
    These two responsibilities are tightly coupled in my opinion. 
        You pick sequences without concerning yourself without how their split will be.
    Returns: in Order: Train, Validation, Tests
    """
    assert sum(split_percentage) == 1, "Split Percentages should sum to 1"
    assert len(split_percentage) == 3, "There should be three split_percentages"

    run_length = run.shape[0]
    num_seqs = run_length // seq_len

    # This will now selet them at random to have a better approach to it.
    seqs_offsets_idxs = np.arange(0,run_length, seq_len)

    # Check if last offset is big enough otherwise we drop
    if (run_length-1)-seqs_offsets_idxs[-1] != seq_len:
        seqs_offsets_idxs = seqs_offsets_idxs[:-1]

    # Now we shuffle the offsets
    np.random.shuffle(seqs_offsets_idxs)

    train_seqs_idxs = seqs_offsets_idxs[:int(num_seqs * split_percentage[0])]
    val_seqs_idxs = seqs_offsets_idxs[int(num_seqs * split_percentage[0]) : int(num_seqs * split_percentage[0] + num_seqs * split_percentage[1])]
    # Ignore test seqs for now...

    # Now we pick the actual seqs into a 3d array
    train_seqs   = [run[idx:idx+seq_len] for idx in train_seqs_idxs]
    val_seqs     = [run[idx:idx+seq_len] for idx in val_seqs_idxs]

    train_seqs = np.stack(train_seqs)
    val_seqs   = np.stack(val_seqs)

    return train_seqs, val_seqs, np.array([])

def space_divisor(space_size: int, min_window_size: int, num_chunks: int) -> List[int]:
    """
    Mostly used for creating validation data.
    Given a (sub)space size it will try to divide it into `num_chunks` chunks of randomly different size.

    Variables (3)
    ---------
    - space_size (int): How long the run is
    - min_window_size (int): The expected size of the input sequence for the downstream RNN
    - num_chunks (int): Number of discrete chunks scattered across the run that will be used for validation.

    Returns (1)
    ---------
    - chunks (List[int]): calculated chunk sizes.
    """
    assert space_size / min_window_size >= num_chunks, f"Space of size {space_size} cannot be divided into {num_chunks} chunks whilst keeping a minimum window size of {min_window_size}"

    chunks_sizes = [min_window_size] * num_chunks 
    space_remaining = space_size - min_window_size * num_chunks 

    budget = space_remaining

    for i in range(num_chunks-1):
        to_add = random.randint(0,budget)
        chunks_sizes[i] = chunks_sizes[i] + to_add
        budget -= to_add

    chunks_sizes[-1] += budget
    random.shuffle(chunks_sizes)
    return chunks_sizes


def cast_chunks(total_space: int, chunks_sizes: list, min_width: int) -> List[Tuple[int, int]]:
    """
    Given `chunk_sizes` it will cast/spread them across the larger `space`.
    It will ensure proper (able to fit `min_width` spacing) spacing between chunks.

    Variables (3)
    ---------
    - total_space (int): Size of super space
    - chunk_sizes list[int]: Sizes of chunks to cast/spread around `total_space`.
    - min_width (int): Generally represents the sequence length given to dowsntream RNNS.
      More specific to this function. Simply ensures that gaps can fit desired sequence lengths.

    Returns (1)
    ---------
    - positions (list[tuple[int,int]]): List of starting and ending positions of chunks
    """
    
    assert sum(chunks_sizes) + (len(chunks_sizes) - 1) * min_width <= total_space, "Not enough space to ensure min_width gaps."
    
    positions = []
    remaining_space = total_space - sum(chunks_sizes)
    num_gaps = len(chunks_sizes) - 1
    gap_sizes = [min_width] * num_gaps  # Start with minimum gaps
    
    remaining_space -= sum(gap_sizes)
    
    for i in range(num_gaps):
        max_addable = remaining_space if remaining_space > 0 else 0
        add_size = random.randint(0, max_addable)
        gap_sizes[i] += add_size
        remaining_space -= add_size
    
    current_pos = 0 # FIX: Not super happy that we force the first chunk to start from 0
    for chunk_size, gap in zip(chunks_sizes, gap_sizes + [0]):
        positions.append((current_pos, current_pos + chunk_size))
        current_pos += chunk_size + gap
    
    return positions

def randomly_pick_sequences_split_and_oversample(
    run: np.ndarray,
    split_percentage: List[float],
    seq_len: int,
    oversample_coefficient: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Add capacity for oversampling
    """
    Will split a run into train, validation and test
    These two responsibilities are tightly coupled in my opinion.
        You pick sequences without concerning yourself without how their split will be.
    Returns: in Order: Train, Validation, Tests
    """
    assert sum(split_percentage) == 1, "Split Percentages should sum to 1"
    assert (
        len(split_percentage) == 2
    ), "There should be two split_percentages"  # Test will remain our last file10
    run_length = run.shape[0]
    quotient = run_length // seq_len

    # Data preprocessing
    amounts = int(quotient * oversample_coefficient)
    train_amount = int(amounts * split_percentage[0])
    validation_amount = int(amounts * split_percentage[1])
    validation_window_length = int(run_length * split_percentage[1])

    # We shall simply chose how to divide the validation length such that our tiniest still holds atleast seq_len 
    num_chunks = 5 # TODO: fix the hardcode 

    # Divide the space for it to be sampled afterwards
    val_chunks = space_divisor(validation_window_length, seq_len, num_chunks)
    val_chunks_positions = cast_chunks(run_length, val_chunks, seq_len)

    # Now we sample a chunk at random, and sample a `backhistory` from it
    validation_sequences = []
    for val_selection in range(validation_amount):
        # Select chunk at random 
        chunk_idx = np.random.randint(0, len(val_chunks_positions))
        chunk_start, chunk_end = val_chunks_positions[chunk_idx]
        # Select spot at random 
        spot_idx = np.random.randint(chunk_start+seq_len, chunk_end+1)
        # spot = spot_idx - run_length + 1 # WERID ?
        run_itself = spot_backhistory(spot_idx, seq_len, run, padding_value=-1)
        validation_sequences.append(run_itself)


    ##  Now get chunk idxs out of chunks
    # Convert chunks endpoints to sets and remove from the larger set
    val_chunks_idxs: Set[int] = set()
    for chunk_start, chunk_end in val_chunks_positions:
        val_chunks_idxs |= set(range(chunk_start, chunk_end))
    all_set = set(range(run_length))
    train_idxs = all_set - val_chunks_idxs
    # Now we remove those that 
    useable_train_idxs = []
    for idx in train_idxs:
        if idx-seq_len in train_idxs:
            useable_train_idxs.append(idx)

    # Once that is done we start selecting just like before
    train_sequences = []
    for train_selection in range(train_amount):
        # Select chunk at random 
        spot = np.random.choice(useable_train_idxs, 1)[0]
        run_itself = spot_backhistory(spot, seq_len, run, padding_value=-1)
        train_sequences.append(run_itself)

    ### 🏇🏇 ###

    validation_sequences = np.stack(validation_sequences)
    train_sequences = np.stack(train_sequences)

    # NICE Tool for debigging. Might be useful later so I leave it here.
    # visualize_windows(validation_sequences, train_sequences)

    return train_sequences, validation_sequences, np.array([])


def visualize_windows(
    validation_windows: np.ndarray,
    training_windows: np.ndarray,
    output_path: str = "./figures/debug_windows.png",
):
    """
    Visualize validation and training windows with different colors.

    Args:
        validation_windows: numpy array of validation window sequences
        training_windows: numpy array of training window sequences
        output_path: path to save the output figure
    """
    plt.figure(figsize=(10, 6))

    # Plot validation windows in red
    for seq_no in range(validation_windows.shape[0]):
        times = validation_windows[seq_no, :, 0]
        random_y = np.full_like(times, random.random())
        plt.plot(times, random_y, color="r", alpha=0.5, label="Validation")

    # Plot training windows in blue
    for seq_no in range(training_windows.shape[0]):
        times = training_windows[seq_no, :, 0]
        random_y = np.full_like(times, random.random())
        plt.plot(times, random_y, color="b", alpha=0.5, label="Training")

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()


def split_defacto_runs(
    run_dict: OrderedDict[str, np.ndarray], 
    train_split: float,
    val_split: float,
    seq_length: int, 
    oversample_coefficient: float,
    scale: bool,
) -> Tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray], np.ndarray]:
    """
    Will basically split the output of load_defacto_data into train, validation and test
    Every run will be split into train, validation and test
    Test split will be an entirely different file
    """

    # Ensure test_split + val_split = 1
    if abs((train_split + val_split) - 1) > 0.001:
        raise RuntimeError(f"train_split ({train_split}) + val_split ({val_split} !~ 1)")

    train_ds = OrderedDict()
    val_ds = OrderedDict()

    split_percentages = [train_split, val_split]

    # Order (perhaps not necessary run_dict)
    sorted_run_dict = dict(
        sorted(
            run_dict.items(),
            key=lambda x: int(re.search("(\d+)\.csv$", x[0]).group(1))
        )
    )

    # Test DS will be returned as entire file as we might want the time order untouched
    _, test_file = sorted_run_dict.popitem()

    all_train = [] # So normalization is simpler
    for run_name in sorted_run_dict.keys():
        run = sorted_run_dict[run_name]
        all_train.append(run)
        train_ds[run_name], val_ds[run_name], _  = randomly_pick_sequences_split_and_oversample(run, split_percentages, seq_length, oversample_coefficient)
    all_train.append(test_file) # So we can scale the features

    all_train = np.concatenate(all_train)

    if scale: 
        scaler = MinMaxScaler()
        scaler.fit(all_train)
        for run_name in sorted_run_dict:
            train_ds[run_name] = scaler.transform(train_ds[run_name].reshape(-1, train_ds[run_name].shape[2])).reshape(-1, seq_length, train_ds[run_name].shape[2])
            if len(val_ds[run_name]) > 0 :
                val_ds[run_name] = scaler.transform(val_ds[run_name].reshape(-1, val_ds[run_name].shape[2])).reshape(-1, seq_length, val_ds[run_name].shape[2])
        # TODO: Set the test file
        test_file = scaler.transform(test_file)

    return train_ds, val_ds, test_file

def load_defacto_data(path: str) -> Tuple[OrderedDict[str, np.ndarray], list[str]]:
    """
    Load the data from multiple files referencing different runs.

    Returns:
        - obtained_runs: A dictionary with the runs as keys and the data as values. 
          Keys are sorted by the run number.
        - columns_so_far: The columns that are shared across all runs
    """

    # Create a list of files starting with `run_` inside of the path
    columns_so_far = []
    files: list[str] = []
    # These files do *NOT* include interpolation. Instead (I Think they have empty or zero elements)
    for ff in os.listdir(path):
        if ff.startswith("run_"):
            if len(columns_so_far)  == 0:
                # columns_so_far = pd.read_csv(os.path.join(path, ff), index_col=0, header=0).columns.values
                columns_so_far: List[str] = pd.read_csv(os.path.join(path, ff), header=0).columns.values.tolist()
                # Impute them if need be 
            else:
                if set(columns_so_far) != set(pd.read_csv(os.path.join(path, ff), header=0).columns):
                    raise ValueError("Columns are not the same for all runs")
            files.append(ff)

    obtained_files = "\n".join([f"\t- {file_name}" for file_name in files])
    print(f"All files obtained in load_defacto_data are :\n{obtained_files}")

    # Organize them by number after the run_
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0].split("run_")[1]))
    obtained_runs: OrderedDict[str, np.ndarray] = OrderedDict({ f_name : np.ndarray([]) for f_name in sorted_files })

    # Debug: Leave one out and see where things are going.

    # Now we start forming a 
    for f in sorted_files:
        print(f"Loading run: run {f}")
        df = pd.read_csv(os.path.join(path,f), header=0)
        df.interpolate(inplace=True)
        # Even after interpolation we might still get initial problems witht the row:
        df.dropna(inplace=True)
        obtained_runs[f] = df.values # NOTE: Confirm this doing what we expect it to 

    # Let me see how it looks
    return obtained_runs, columns_so_far


def new_format(path: str, features_per_run: int = 15):
    """
    This will only be for converting the csv to a new more amenable format
    args:
        - path: The path to the csv file
        - features_per_run: The number of features per run.
    """
    # Remove first column
    data_so_far = pd.read_csv(path, skiprows=1, index_col=0, header=1)

    num_runs = data_so_far.shape[1] / (features_per_run*2)
    print(f"Number of runs is {num_runs}")
    num_runs = int(num_runs)
    og_path = "/".join(path.split("/")[:-1])

    # On each run we will try to align
    for i in range(num_runs):
        # Get the data for this run
        run_data = data_so_far.iloc[
            :, i * (features_per_run * 2) : (i + 1) * (features_per_run * 2)
        ]
        # Get the time steps for this run
        print(f"About to enter the {i}th run. With a dataframe with {run_data.shape[1]} columns and {run_data.shape[0]} rows")
        new_df = df_from_run(run_data, features_per_run)
        new_df.to_csv(f"{og_path}/run_{i}.csv")

    # NOTE: This is basically all I need it for so I wont bother with nicer returns and the like
