import pandas as pd
from typing import List, DefaultDict
import pdb
import itertools

# def load_defacto_data(path: str) -> List:
#     """
#     Load the data from the defacto dataset
#     """
#     data_so_far = pd.read_csv(path, skiprows=1)
#
#     # Let me see how it looks
#     print(f"Data looks like {data_so_far.head()}")
#
#
#     return data_so_far.values


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



def new_format(path: str, features_per_run: int = 15):
    """
    This will only be for conveting the csv to a new more amenable format
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
