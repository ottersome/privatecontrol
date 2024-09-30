"""
Will simply convert the old defactor format to the new format easier to process 
"""
from conrecon.data.data_loading import new_format

if __name__ == "__main__":
    new_format("./data/defacto.csv")
