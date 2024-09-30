import pandas as pd
from typing import List
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

def new_format(path: str, features_per_run:int = 15) -> List:
    """
    This will only be for conveting the csv to a new more amenable format
    args:
        - path: The path to the csv file
        - features_per_run: The number of features per run. 
    """
    # Remove first column
    data_so_far = pd.read_csv(path, skiprows=1, index_col=0, header=1)

    num_runs = data_so_far.shape[1] / features_per_run
    print(f"Number of runs is {num_runs}")
    num_runs = int(num_runs)

    # On each run we will try to align 
    for i in range(num_runs):
        # Get the data for this run
        run_data = data_so_far.iloc[:, i * features_per_run : (i + 1) * features_per_run]
        # Get the time steps for this run
        indexers = [ 2*i for i in range(features_per_run)]
        pdb.set_trace()
        run_time_steps = data_so_far.iloc[:, indexers ]
        flattened_datapoints = run_data.values
        set_datapoints = list(set(itertools.chain.from_iterable(flattened_datapoints)))

        # Create new df with set_datapoints as indexers
        new_df = pd.DataFrame(index=set_datapoints)
        for i in range(features_per_run):
            new_df[i] = run_data.iloc[:, i]

        flattened_datapoints = run_data.values
        # Create a  set of time steps for this run : 

        # Lets see what we have
        print(f"Run {i} has shape {run_data.shape} and time steps {run_time_steps.shape}")
        print(f"Run {i} has data {run_data.head()} and time steps {run_time_steps.head()}")
        print(f"Run {i} has data {run_data.tail()} and time steps {run_time_steps.tail()}")


