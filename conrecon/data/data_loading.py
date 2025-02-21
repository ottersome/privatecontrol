import pandas as pd
from typing import List, DefaultDict, OrderedDict
import pdb
import itertools
import os
import numpy as np
from typing import Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re

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


def split_defacto_runs(
    run_dict: OrderedDict[str, np.ndarray], 
    train_split: float,
    val_split: float,
    seq_length: int, 
    scale: bool = True,
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

    split_percentages = [train_split, val_split, 0]

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
        train_ds[run_name], val_ds[run_name], _  = randomly_pick_sequences_and_split(run, split_percentages, seq_length)
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

def load_defacto_data(path: str) -> Tuple[List[str], OrderedDict[str, np.ndarray], pd.DataFrame]:
    """
    Load the data from the already-split files
    Also does interpolation

    Returns:
        - columns_so_far: The columns that are shred across all runs
        - obtained_runs: A dictionary with the runs as keys and the data as values
    """

    # Create a list of files starting with `run_` inside of the path
    columns_so_far = []
    files = []
    # These files do *NOT* include interpolation. Instead (I Think they have empty or zero elements)
    for ff in os.listdir(path):
        if ff.startswith("run_"):
            if len(columns_so_far)  == 0:
                # columns_so_far = pd.read_csv(os.path.join(path, ff), index_col=0, header=0).columns.values
                columns_so_far = pd.read_csv(os.path.join(path, ff), header=0).columns.values
                # Impute them if need be 
            else:
                if set(columns_so_far) != set(pd.read_csv(os.path.join(path, ff), header=0).columns):
                    raise ValueError("Columns are not the same for all runs")
            files.append(ff)

    # Organize them by number after the run_
    test_file = files[-1]
    test_file = pd.read_csv(os.path.join(path,test_file), header=0)
    files = files[:-1]
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0].split("run_")[1]))
    obtained_runs = OrderedDict({ f_name : np.ndarray([]) for f_name in sorted_files })

    # Debug: Leave one out and see where things are going.

    # Now we start forming a 
    for f in sorted_files:
        print(f"Loading run: run {f}")
        df = pd.read_csv(os.path.join(path,f), header=0)
        df.interpolate(inplace=True)
        # Even after interpolation we might still get initial problems witht the row:
        df.dropna(inplace=True)
        obtained_runs[f] = df.values # NOTE: Confirm this doing what we expect it to 

    assert isinstance(debug_file, pd.DataFrame)
    # Let me see how it looks
    return columns_so_far, obtained_runs, debug_file


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
