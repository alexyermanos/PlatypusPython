import os 
import pandas as pd

def prepare_sequence(path_to_file:str):
    """
    Prepare sequence strings of a given file, for langauge model by adding spaces

    parameters
    ---------

    path_to_file: `str`
    Path to file with sequences to be turned to embeddings


    return
    ------
    List of sequences with correct format
    """

    data = pd.read_csv(path_to_file)
    sequences = data["sequence"]
    print(sequences)
    sequences = sequences.apply(add_space)
    return sequences

def add_space(row):
    if not isinstance(row, float):
        row = " ".join(row)
    return row