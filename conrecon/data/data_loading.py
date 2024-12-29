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


def split_defacto_runs(
    run_dict: OrderedDict[str, np.ndarray], 
    train_split: float,
    val_split: float,
    test_split: float,
) -> Tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]]:
    """
    Will basically split the output of load_defacto_data into train, validation and test
    Every run will be split into train, validation and test
    """
    d_img_path = './imgs/'

    train_ds = OrderedDict()
    val_ds = OrderedDict()
    test_ds = OrderedDict()
    split_percentages = [train_split, val_split, test_split]
    # d_slices = [slice(0,128), slice(3,4)] # Sequence Length and features
    # d_og_imgs = []
    for run_name in run_dict.keys():
        run = run_dict[run_name]
        # Split the run into train, validation and test
        train_ds[run_name], val_ds[run_name], test_ds[run_name] = split_run(run, split_percentages)

        # d_og_imgs.append(train_ds[run_name][d_slices[0],d_slices[1]])

    all_train = np.concat(list(train_ds.values()))
    scaler = MinMaxScaler()
    scaler.fit(all_train)

    # d_sc_imgs = []
    # TODO: Clean this bit of code if you can 
    # Then normalized it. 
    for run_name in run_dict:
        train_ds[run_name] = scaler.transform(train_ds[run_name])
        if len(val_ds[run_name]) > 0 :
            val_ds[run_name] = scaler.transform(val_ds[run_name])
        if len(test_ds[run_name]) > 0 :
            test_ds[run_name] = scaler.transform(test_ds[run_name])

        # d_sc_imgs.append(train_ds[run_name][d_slices[0], d_slices[1]])

    # for i in range(len(d_sc_imgs)):
    #     # Compare the paragraphs there. 
    #     fig, axs = plt.subplots(1,2,figsize=(16,10))
    #     plt.tight_layout()
    #     # axis = np.concatenate(d_sc_imgs[i].shape[0]*[np.arange(d_og_imgs[i].shape[1])])
    #     axs[0].plot(d_og_imgs[i],  label="Original Images")
    #     axs[1].plot(d_sc_imgs[i],  label="New ones")
    #     plt.show()

    return train_ds, val_ds, test_ds

def load_defacto_data(path: str) -> Tuple[List[str], OrderedDict[str, np.ndarray], pd.DataFrame]:
    """
    Load the data from the defacto dataset

    Returns:
        - columns_so_far: The columns that are shred across all runs
        - obtained_runs: A dictionary with the runs as keys and the data as values
    """

    # Create a list of files starting with `run_` inside of the path
    columns_so_far = []
    files = []
    for ff in os.listdir(path):
        if ff.startswith("run_"):
            if len(columns_so_far)  == 0:
                columns_so_far = pd.read_csv(os.path.join(path, ff), index_col=0, header=0).columns.values
                # Impute them if need be 
            else:
                if set(columns_so_far) != set(pd.read_csv(os.path.join(path, ff), index_col=0, header=0).columns):
                    raise ValueError("Columns are not the same for all runs")
            files.append(ff)

    # Organize them by number after the run_
    debug_file = files[-1]
    debug_file = pd.read_csv(os.path.join(path,debug_file), index_col=0,header=0)
    files = files[:-1]
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0].split("run_")[1]))
    obtained_runs = OrderedDict({ f_name : np.ndarray([]) for f_name in sorted_files })

    # Debug: Leave one out and see where things are going.

    # Now we start forming a 
    for f in sorted_files:
        print(f"Loading run: run {f}")
        df = pd.read_csv(os.path.join(path,f), index_col=0, header=0)
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
