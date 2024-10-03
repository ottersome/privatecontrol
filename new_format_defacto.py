"""
Will simply convert the old defactor format to the new format easier to process 
"""

## 1: For Creating a new format
from conrecon.data.data_loading import new_format

## 2: For showing statistcs on the data
from conrecon.stats import plot_feats_synch
from conrecon.data.data_loading import load_runs

if __name__ == "__main__":
    ## 1: For Creating a new format
    # new_format("./data/defacto.csv")

    ## 2: For showing statistcs on the data
    run_dictionary = load_runs("./data/")
    print(f"Running stats with a dictoinary containing {run_dictionary.keys()} runs")
    for run_idx in range(len(run_dictionary)):
        # This will plot things for us
        plot_feats_synch(
            run_dictionary[f"run_{run_idx}.csv"], f"run_{run_idx}_features.png"
        )
